// Console command: NanoGS.TestMLPForwardGPU
//
// GPU-side MLP forward test driver.
//   - Loads a UGaussianSplatAsset at the given package path.
//   - Builds FGaussianSplatGPUResources (which uploads position/scale/rot/SH/
//     MLP weights to the GPU). FGaussianSplatGPUResources already contains
//     exactly the fields DispatchMLPForward reads — no dummy class needed.
//   - Dispatches FMLPForwardCS with a user-specified camera position.
//   - Reads back *five* buffers via FRHIGPUBufferReadback: the float2
//     PhiOpacity output plus the per-splat inputs Position / Scale / Rotation
//     (float) and SH (fp16, converted to fp32 on CPU).
//   - Computes torch.mean-style flat min/max/mean for each of the five buffers
//     (each treated as a 1-D float array) — the "raw UE-space" statistics.
//   - Additionally, replicates the shader's Phase 0 input assembly on CPU
//     (PLY-space axis permute + per-splat L2-normalize for SH/scale, viewdir
//     from PLY-space camera-delta normalize, quaternion axis permute+sign
//     flip) and computes flat min/max/mean over the resulting vectors — the
//     "MLP-direct input" statistics, i.e. the values the network actually
//     receives.
//   - Logs both sets of stats and writes them into the CSV header.
//   - CSV per-row contents are limited to the first 10 splats (phi / opacity).
//
// Usage (UE5 console or cmd):
//   NanoGS.TestMLPForwardGPU <asset_path> <cam_x> <cam_y> <cam_z> [output.csv]
//
// Example:
//   NanoGS.TestMLPForwardGPU /Game/Splats/MyScene 0 0 200
//
// Notes:
//   - Camera position is in world-space UE units (cm).
//   - Asset's LocalToWorld is taken as identity here (asset-local == world),
//     so WorldCameraPosition is used directly inside DispatchMLPForward.
//     If you want to simulate an actor offset, pre-bake it into cam_x/y/z.
//   - If the asset has no MLP weights configured, the shader runs its
//     placeholder path (phi=1.0, opacity=unpacked * OpacityScale).

#include "CoreMinimal.h"
#include "Engine/Engine.h"
#include "GaussianSplatAsset.h"
#include "GaussianSplatSceneProxy.h"
#include "GaussianSplatRenderer.h"
#include "HAL/PlatformProcess.h"
#include "Math/Float16.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "RHICommandList.h"
#include "RHIGPUReadback.h"
#include "RenderingThread.h"
#include "UObject/UObjectGlobals.h"

namespace
{

// ---- helpers ----

// torch.mean-style flat reduction: treat Data as a 1-D float array of length N.
static void ComputeStatsFlat(const float* Data, int64 N, float& OutMin, float& OutMax, float& OutMean)
{
	if (!Data || N <= 0)
	{
		OutMin = OutMax = OutMean = 0.0f;
		return;
	}
	float Mn = Data[0];
	float Mx = Data[0];
	double Sum = 0.0;
	for (int64 i = 0; i < N; i++)
	{
		const float V = Data[i];
		Mn = FMath::Min(Mn, V);
		Mx = FMath::Max(Mx, V);
		Sum += (double)V;
	}
	OutMin  = Mn;
	OutMax  = Mx;
	OutMean = (float)(Sum / (double)N);
}

// ---- console command entry ----

static void ExecTestMLPForwardGPU(const TArray<FString>& Args)
{
	if (Args.Num() < 4)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("Usage: NanoGS.TestMLPForwardGPU <asset_path> <cam_x> <cam_y> <cam_z> [output.csv]"));
		UE_LOG(LogTemp, Warning,
			TEXT("  asset_path: package path, e.g. /Game/Splats/MyAsset"));
		UE_LOG(LogTemp, Warning,
			TEXT("  camera: world-space UE units (cm)"));
		return;
	}

	const FString AssetPath = Args[0];
	const float CamX = FCString::Atof(*Args[1]);
	const float CamY = FCString::Atof(*Args[2]);
	const float CamZ = FCString::Atof(*Args[3]);
	const FString OutputPath = Args.Num() >= 5
		? Args[4]
		: (FPaths::ProjectSavedDir() / TEXT("MLPForwardGPU_Output.csv"));

	// ---------- 1) Load asset ----------
	UGaussianSplatAsset* Asset = LoadObject<UGaussianSplatAsset>(nullptr, *AssetPath);
	if (!Asset || !Asset->IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLPGPU] Cannot load GaussianSplatAsset: %s"), *AssetPath);
		return;
	}
	const int32 SplatCount = Asset->GetSplatCount();
	if (SplatCount <= 0)
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLPGPU] Asset has no splats: %s"), *AssetPath);
		return;
	}
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU] Asset: %s  SplatCount=%d  Camera=(%.2f, %.2f, %.2f)"),
		*AssetPath, SplatCount, CamX, CamY, CamZ);

	// ---------- 2) Build GPU resources ----------
	// FGaussianSplatGPUResources::Initialize() calls InitResource(FRHICommandListImmediate::Get())
	// internally, which asserts IsInRenderingThread() (RHICommandList.h:5524). In the normal
	// rendering pipeline this function is driven from FGaussianSplatSceneProxy on the render
	// thread; here we're on the game thread (console command), so we must wrap the call in
	// an ENQUEUE_RENDER_COMMAND. FlushRenderingCommands() after it blocks the game thread
	// until the init (and its InitRHI) finish, which also prevents GC from collecting the
	// Asset UObject while the render thread dereferences it.
	FGaussianSplatGPUResources* GPUResources = new FGaussianSplatGPUResources();
	ENQUEUE_RENDER_COMMAND(MLPForwardGPUTestInitResources)(
		[GPUResources, Asset](FRHICommandListImmediate& RHICmdList)
		{
			GPUResources->Initialize(Asset);
		});
	FlushRenderingCommands();

	if (!GPUResources->IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLPGPU] GPU resources invalid after Initialize."));
		ENQUEUE_RENDER_COMMAND(ReleaseGPURes)(
			[GPUResources](FRHICommandListImmediate& RHICmdList)
			{
				GPUResources->ReleaseResource();
				delete GPUResources;
			});
		FlushRenderingCommands();
		return;
	}
	if (!GPUResources->bHasMLPWeights)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("[TestMLPGPU] Asset has no MLP weights (MLPWeightsDirectory empty or file missing). ")
			TEXT("Running placeholder path: phi=1.0, opacity=alpha*OpacityScale."));
	}

	// ---------- 3) Allocate readbacks + dispatch on render thread ----------
	// We read back 5 buffers so we can log flat (torch.mean-style) statistics for
	// the per-splat inputs that feed the MLP in addition to the MLP output:
	//   - PhiOpacity : SplatCount * 2 floats              (MLP output)
	//   - Position   : SplatCount * 3 floats              (ByteAddressBuffer, AoS)
	//   - Scale      : SplatCount * 3 floats              (ByteAddressBuffer, AoS)
	//   - Rotation   : SplatCount * 4 floats              (ByteAddressBuffer, AoS)
	//   - SH         : SplatCount * SHCoeffCount * 3 fp16 (ByteAddressBuffer, AoS)
	const int32  SHBands       = GPUResources->GetSHBands();
	const int32  SHCoeffCount  = (SHBands + 1) * (SHBands + 1);
	const uint32 PhiBytes      = (uint32)SplatCount * 2u * (uint32)sizeof(float);
	const uint32 PosBytes      = (uint32)SplatCount * 3u * (uint32)sizeof(float);
	const uint32 ScaleBytes    = (uint32)SplatCount * 3u * (uint32)sizeof(float);
	const uint32 RotBytes      = (uint32)SplatCount * 4u * (uint32)sizeof(float);
	const uint32 SHBytes       = (SHBands > 0)
		? (uint32)SplatCount * (uint32)SHCoeffCount * 3u * (uint32)sizeof(uint16)
		: 0u;

	FRHIGPUBufferReadback* PhiReadback   = new FRHIGPUBufferReadback(TEXT("MLPPhiOpacityReadback"));
	FRHIGPUBufferReadback* PosReadback   = new FRHIGPUBufferReadback(TEXT("MLPPositionReadback"));
	FRHIGPUBufferReadback* ScaleReadback = new FRHIGPUBufferReadback(TEXT("MLPScaleReadback"));
	FRHIGPUBufferReadback* RotReadback   = new FRHIGPUBufferReadback(TEXT("MLPRotationReadback"));
	FRHIGPUBufferReadback* SHReadback    = (SHBytes > 0)
		? new FRHIGPUBufferReadback(TEXT("MLPSHReadback"))
		: nullptr;

	// Helper to release *all* readbacks on the render thread in one lambda.
	auto ReleaseAll = [GPUResources, PhiReadback, PosReadback, ScaleReadback, RotReadback, SHReadback]()
	{
		ENQUEUE_RENDER_COMMAND(ReleaseMLPTestGPURes)(
			[GPUResources, PhiReadback, PosReadback, ScaleReadback, RotReadback, SHReadback]
			(FRHICommandListImmediate& RHICmdList)
			{
				GPUResources->ReleaseResource();
				delete GPUResources;
				delete PhiReadback;
				delete PosReadback;
				delete ScaleReadback;
				delete RotReadback;
				if (SHReadback) { delete SHReadback; }
			});
		FlushRenderingCommands();
	};

	// Identity LocalToWorld: asset-local == world for this test. Inside
	// DispatchMLPForward, LocalCamera = Identity.InverseTransformPosition(WorldCam) = WorldCam.
	const FMatrix LocalToWorld = FMatrix::Identity;
	const FVector3f WorldCameraPosition(CamX, CamY, CamZ);
	const float OpacityScale = 1.0f;
	const bool  bEnableMLPWeights = true;

	ENQUEUE_RENDER_COMMAND(MLPForwardGPUTestDispatch)(
		[GPUResources, PhiReadback, PosReadback, ScaleReadback, RotReadback, SHReadback,
		 PhiBytes, PosBytes, ScaleBytes, RotBytes, SHBytes,
		 LocalToWorld, WorldCameraPosition, OpacityScale, bEnableMLPWeights, SplatCount]
		(FRHICommandListImmediate& RHICmdList)
		{
			FRHIBufferCreateDesc PhiDesc = FRHIBufferCreateDesc::Create(
				TEXT("TestPhiOpacityBuffer"),
				PhiBytes,
				2 * sizeof(float),
				BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
				.SetInitialState(ERHIAccess::UAVCompute);
			FBufferRHIRef PhiOpacityBuffer = RHICmdList.CreateBuffer(PhiDesc);

			// 离线测试始终走全量推理 — 不依赖 cluster 可见性位图, 便于和 CPU
			// 参考实现逐 splat 逐输出比对.
			FGaussianSplatRenderer::DispatchMLPForward(
				RHICmdList, GPUResources,
				LocalToWorld,
				WorldCameraPosition,
				SplatCount,
				OpacityScale,
				PhiOpacityBuffer,
				bEnableMLPWeights,
				/*bUseCulling=*/false);

			// --- PhiOpacity: SRVCompute -> CopySrc, then copy ---
			RHICmdList.Transition(FRHITransitionInfo(
				PhiOpacityBuffer, ERHIAccess::SRVCompute, ERHIAccess::CopySrc));
			PhiReadback->EnqueueCopy(RHICmdList, PhiOpacityBuffer, PhiBytes);

			// --- Input buffers (Position/Scale/Rotation/SH): SRVMask -> CopySrc, then copy ---
			// All four live on GPUResources and are bound as ByteAddressBuffer SRVs
			// normally; they're in SRVMask state. Transition them together then copy.
			TArray<FRHITransitionInfo, TInlineAllocator<4>> Transitions;
			Transitions.Add(FRHITransitionInfo(GPUResources->PositionBuffer, ERHIAccess::SRVMask, ERHIAccess::CopySrc));
			Transitions.Add(FRHITransitionInfo(GPUResources->ScaleBuffer,    ERHIAccess::SRVMask, ERHIAccess::CopySrc));
			Transitions.Add(FRHITransitionInfo(GPUResources->RotationBuffer, ERHIAccess::SRVMask, ERHIAccess::CopySrc));
			if (SHReadback)
			{
				Transitions.Add(FRHITransitionInfo(GPUResources->SHBuffer, ERHIAccess::SRVMask, ERHIAccess::CopySrc));
			}
			RHICmdList.Transition(Transitions);

			PosReadback  ->EnqueueCopy(RHICmdList, GPUResources->PositionBuffer, PosBytes);
			ScaleReadback->EnqueueCopy(RHICmdList, GPUResources->ScaleBuffer,    ScaleBytes);
			RotReadback  ->EnqueueCopy(RHICmdList, GPUResources->RotationBuffer, RotBytes);
			if (SHReadback)
			{
				SHReadback->EnqueueCopy(RHICmdList, GPUResources->SHBuffer, SHBytes);
			}
		});

	// ---------- 4) Wait for GPU ----------
	FlushRenderingCommands();
	// PhiReadback is written last-ish, but all EnqueueCopy calls share the same
	// command stream. Polling any one fence is sufficient in practice; we poll
	// the phi fence for simplicity.
	const double WaitT0 = FPlatformTime::Seconds();
	while (!PhiReadback->IsReady())
	{
		FPlatformProcess::Sleep(0.001f);
		if (FPlatformTime::Seconds() - WaitT0 > 5.0)
		{
			UE_LOG(LogTemp, Error, TEXT("[TestMLPGPU] Readback timed out after 5s."));
			ReleaseAll();
			return;
		}
	}

	// ---------- 5) Read back on render thread, copy into game-thread arrays ----------
	// FRHIGPUBufferReadback::Lock() internally calls RHILockStagingBuffer() ->
	// FRHICommandListExecutor::GetImmediateCommandList(), which asserts
	// IsInRenderingThread() (RHICommandList.h:5524). It *must* run on the render
	// thread, so we enqueue the Lock/copy/Unlock as a render command and block
	// the game thread with FlushRenderingCommands() until it finishes.
	// Allocation sizes fit int32: worst case ~SplatCount * SHCoeffCount * 3, which
	// for a 10M-splat scene with 16 coeffs is 480M floats (< INT32_MAX).
	TArray<float> Phi;         Phi        .SetNumUninitialized(SplatCount);
	TArray<float> Opacity;     Opacity    .SetNumUninitialized(SplatCount);
	TArray<float> PosFloats;   PosFloats  .SetNumUninitialized(SplatCount * 3);
	TArray<float> ScaleFloats; ScaleFloats.SetNumUninitialized(SplatCount * 3);
	TArray<float> RotFloats;   RotFloats  .SetNumUninitialized(SplatCount * 4);
	const int64 SHFloatCount = (SHBytes > 0) ? (int64)SplatCount * SHCoeffCount * 3 : 0;
	TArray<float> SHFloats;    SHFloats   .SetNumUninitialized((int32)SHFloatCount);

	bool bLockOK = false;
	ENQUEUE_RENDER_COMMAND(MLPForwardGPUTestReadback)(
		[PhiReadback, PosReadback, ScaleReadback, RotReadback, SHReadback,
		 PhiBytes, PosBytes, ScaleBytes, RotBytes, SHBytes,
		 SplatCount, SHFloatCount,
		 PhiData     = Phi        .GetData(),
		 OpData      = Opacity    .GetData(),
		 PosData     = PosFloats  .GetData(),
		 ScaleData   = ScaleFloats.GetData(),
		 RotData     = RotFloats  .GetData(),
		 SHData      = SHFloats   .GetData(),
		 bLockOKPtr  = &bLockOK]
		(FRHICommandListImmediate& RHICmdList)
		{
			const float* PhiMapped = static_cast<const float*>(PhiReadback->Lock(PhiBytes));
			if (!PhiMapped) { return; }
			for (int32 i = 0; i < SplatCount; i++)
			{
				PhiData[i] = PhiMapped[i * 2 + 0];
				OpData[i]  = PhiMapped[i * 2 + 1];
			}
			PhiReadback->Unlock();

			const float* PosMapped = static_cast<const float*>(PosReadback->Lock(PosBytes));
			if (PosMapped) { FMemory::Memcpy(PosData, PosMapped, PosBytes); PosReadback->Unlock(); }

			const float* ScMapped = static_cast<const float*>(ScaleReadback->Lock(ScaleBytes));
			if (ScMapped) { FMemory::Memcpy(ScaleData, ScMapped, ScaleBytes); ScaleReadback->Unlock(); }

			const float* RotMapped = static_cast<const float*>(RotReadback->Lock(RotBytes));
			if (RotMapped) { FMemory::Memcpy(RotData, RotMapped, RotBytes); RotReadback->Unlock(); }

			if (SHReadback && SHFloatCount > 0)
			{
				const uint16* SHMapped = static_cast<const uint16*>(SHReadback->Lock(SHBytes));
				if (SHMapped)
				{
					// fp16 -> fp32 on CPU (FFloat16::operator float()).
					for (int64 i = 0; i < SHFloatCount; i++)
					{
						FFloat16 F; F.Encoded = SHMapped[i];
						SHData[i] = (float)F;
					}
					SHReadback->Unlock();
				}
			}

			*bLockOKPtr = true;
		});
	// Capture-by-reference of stack locals (bLockOK, TArray data pointers) is only
	// safe because FlushRenderingCommands() blocks until the command above
	// completes, before this stack frame returns.
	FlushRenderingCommands();

	if (!bLockOK)
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLPGPU] Readback->Lock() returned null on render thread."));
		ReleaseAll();
		return;
	}

	// ---------- 6) Flat (torch.mean-style) stats on raw UE-space buffers ----------
	float PhiMin, PhiMax, PhiMean;
	float OpMin,  OpMax,  OpMean;
	float PosMin, PosMax, PosMean;
	float ScMin,  ScMax,  ScMean;
	float RotMin, RotMax, RotMean;
	float SHMin = 0.f, SHMax = 0.f, SHMean = 0.f;
	ComputeStatsFlat(Phi        .GetData(), SplatCount,              PhiMin, PhiMax, PhiMean);
	ComputeStatsFlat(Opacity    .GetData(), SplatCount,              OpMin,  OpMax,  OpMean);
	ComputeStatsFlat(PosFloats  .GetData(), (int64)SplatCount * 3,   PosMin, PosMax, PosMean);
	ComputeStatsFlat(ScaleFloats.GetData(), (int64)SplatCount * 3,   ScMin,  ScMax,  ScMean);
	ComputeStatsFlat(RotFloats  .GetData(), (int64)SplatCount * 4,   RotMin, RotMax, RotMean);
	if (SHFloatCount > 0)
	{
		ComputeStatsFlat(SHFloats.GetData(), SHFloatCount, SHMin, SHMax, SHMean);
	}

	// ---------- 6b) CPU replica of MLPForwardCS Phase 0 (MLP-direct input vectors) ----------
	// The shader transforms per-splat UE-space inputs into the *exact* vectors that
	// feed MLP layer 1. We replicate that here so the logged stats reflect what the
	// network actually sees (PLY-space axes, normalized where the shader normalizes):
	//
	//   viewdir_ply   = normalize( (pos_ue.y, -pos_ue.z, pos_ue.x) - CameraPosition_PLY )
	//   scale_n       = L2_normalize( (sc_ue.y,  sc_ue.z,  sc_ue.x) )
	//   rot_ply       = ( -rot_ue.y,  rot_ue.z, -rot_ue.x,  rot_ue.w )  (already unit quat)
	//   sh_n          = L2_normalize( sh_fp16_flat )                    (per-splat)
	//
	// Camera: reuse the same UE-local -> PLY-space mapping as DispatchMLPForward C++
	// side. With identity LocalToWorld, LocalCamera == WorldCameraPosition.
	const FVector LocalCameraVec = LocalToWorld.InverseTransformPosition(FVector(WorldCameraPosition));
	const FVector3f PLYCamera(
		(float)LocalCameraVec.Y,
		(float)-LocalCameraVec.Z,
		(float)LocalCameraVec.X);

	TArray<float> ViewDir;   ViewDir .SetNumUninitialized(SplatCount * 3);
	TArray<float> ScaleN;    ScaleN  .SetNumUninitialized(SplatCount * 3);
	TArray<float> RotPLY;    RotPLY  .SetNumUninitialized(SplatCount * 4);
	TArray<float> SHN;       SHN     .SetNumUninitialized((int32)SHFloatCount);
	const int32 SHPerSplat = SHCoeffCount * 3;

	for (int32 i = 0; i < SplatCount; i++)
	{
		// --- viewdir (PLY space, normalized) ---
		const float px = PosFloats[i * 3 + 0];
		const float py = PosFloats[i * 3 + 1];
		const float pz = PosFloats[i * 3 + 2];
		float vx = py       - PLYCamera.X;   // pos_ply.x = pos_ue.y
		float vy = -pz      - PLYCamera.Y;   // pos_ply.y = -pos_ue.z
		float vz = px       - PLYCamera.Z;   // pos_ply.z = pos_ue.x
		const float vLen  = FMath::Sqrt(vx * vx + vy * vy + vz * vz);
		const float vInv  = (vLen > 1e-6f) ? (1.0f / vLen) : 0.0f;
		ViewDir[i * 3 + 0] = vx * vInv;
		ViewDir[i * 3 + 1] = vy * vInv;
		ViewDir[i * 3 + 2] = vz * vInv;

		// --- scale (PLY axis permute + L2-normalize) ---
		const float sxu = ScaleFloats[i * 3 + 0];
		const float syu = ScaleFloats[i * 3 + 1];
		const float szu = ScaleFloats[i * 3 + 2];
		const float sx = syu;   // sc_ply.x = sc_ue.y
		const float sy = szu;   // sc_ply.y = sc_ue.z
		const float sz = sxu;   // sc_ply.z = sc_ue.x
		const float sSq  = sx * sx + sy * sy + sz * sz;
		const float sInv = (sSq > 1e-12f) ? (1.0f / FMath::Sqrt(sSq)) : 0.0f;
		ScaleN[i * 3 + 0] = sx * sInv;
		ScaleN[i * 3 + 1] = sy * sInv;
		ScaleN[i * 3 + 2] = sz * sInv;

		// --- rotation (PLY quat: permute + sign flip on X/Z; already unit magnitude) ---
		const float rxu = RotFloats[i * 4 + 0];
		const float ryu = RotFloats[i * 4 + 1];
		const float rzu = RotFloats[i * 4 + 2];
		const float rwu = RotFloats[i * 4 + 3];
		RotPLY[i * 4 + 0] = -ryu;   // rot_ply.x = -rot_ue.y
		RotPLY[i * 4 + 1] =  rzu;   // rot_ply.y =  rot_ue.z
		RotPLY[i * 4 + 2] = -rxu;   // rot_ply.z = -rot_ue.x
		RotPLY[i * 4 + 3] =  rwu;   // rot_ply.w =  rot_ue.w

		// --- SH per-splat L2-normalize over all SHPerSplat floats ---
		if (SHPerSplat > 0)
		{
			const int32 off = i * SHPerSplat;
			double normSq = 0.0;
			for (int32 k = 0; k < SHPerSplat; k++)
			{
				const double v = SHFloats[off + k];
				normSq += v * v;
			}
			const float invN = (normSq > 1e-12) ? (float)(1.0 / FMath::Sqrt(normSq)) : 0.0f;
			for (int32 k = 0; k < SHPerSplat; k++)
			{
				SHN[off + k] = SHFloats[off + k] * invN;
			}
		}
	}

	float VdMin,  VdMax,  VdMean;
	float ScnMin, ScnMax, ScnMean;
	float RpMin,  RpMax,  RpMean;
	float ShnMin = 0.f, ShnMax = 0.f, ShnMean = 0.f;
	ComputeStatsFlat(ViewDir.GetData(), (int64)SplatCount * 3, VdMin,  VdMax,  VdMean);
	ComputeStatsFlat(ScaleN .GetData(), (int64)SplatCount * 3, ScnMin, ScnMax, ScnMean);
	ComputeStatsFlat(RotPLY .GetData(), (int64)SplatCount * 4, RpMin,  RpMax,  RpMean);
	if (SHFloatCount > 0)
	{
		ComputeStatsFlat(SHN.GetData(), SHFloatCount, ShnMin, ShnMax, ShnMean);
	}

	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU] ================ MLP forward stats ================"));
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   Splats   : %d"), SplatCount);
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   MLPWeights: %s  InputDim=%u"),
		GPUResources->bHasMLPWeights ? TEXT("YES") : TEXT("NO (placeholder)"),
		GPUResources->MLPInputDim);
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   SHBands=%d  SHCoeffCount=%d"), SHBands, SHCoeffCount);
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   PLYCamera=(%.6f,%.6f,%.6f)  (from WorldCam via UE-local->PLY)"),
		PLYCamera.X, PLYCamera.Y, PLYCamera.Z);
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU] ---- output ----"));
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   phi           min=%.6f  max=%.6f  mean=%.6f"), PhiMin, PhiMax, PhiMean);
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   opacity       min=%.6f  max=%.6f  mean=%.6f"), OpMin,  OpMax,  OpMean);
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU] ---- raw UE-space buffers ----"));
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   position      min=%.6f  max=%.6f  mean=%.6f  (N=%lld)"),
		PosMin, PosMax, PosMean, (long long)((int64)SplatCount * 3));
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   scale         min=%.6f  max=%.6f  mean=%.6f  (N=%lld)"),
		ScMin, ScMax, ScMean, (long long)((int64)SplatCount * 3));
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   rotation      min=%.6f  max=%.6f  mean=%.6f  (N=%lld)"),
		RotMin, RotMax, RotMean, (long long)((int64)SplatCount * 4));
	if (SHFloatCount > 0)
	{
		UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   sh            min=%.6f  max=%.6f  mean=%.6f  (N=%lld)"),
			SHMin, SHMax, SHMean, (long long)SHFloatCount);
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   sh            (no SH data)"));
	}
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU] ---- MLP-direct inputs (CPU replica of Phase 0) ----"));
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   viewdir_ply   min=%.6f  max=%.6f  mean=%.6f  (N=%lld, normalized)"),
		VdMin, VdMax, VdMean, (long long)((int64)SplatCount * 3));
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   scale_n       min=%.6f  max=%.6f  mean=%.6f  (N=%lld, L2-normalized)"),
		ScnMin, ScnMax, ScnMean, (long long)((int64)SplatCount * 3));
	UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   rot_ply       min=%.6f  max=%.6f  mean=%.6f  (N=%lld, unit quat permuted)"),
		RpMin, RpMax, RpMean, (long long)((int64)SplatCount * 4));
	if (SHFloatCount > 0)
	{
		UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   sh_n          min=%.6f  max=%.6f  mean=%.6f  (N=%lld, per-splat L2-normalized)"),
			ShnMin, ShnMax, ShnMean, (long long)SHFloatCount);
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU]   sh_n          (no SH data)"));
	}

	// ---------- 7) Write CSV (first 10 rows only) ----------
	const int32 NumRows = FMath::Min(10, SplatCount);
	FString CSV;
	CSV.Reserve(4096);
	CSV += FString::Printf(TEXT("# asset=%s  splats=%d  camera=(%.6f,%.6f,%.6f)  mlp_weights=%d  input_dim=%u  sh_bands=%d  sh_coeff_count=%d\n"),
		*AssetPath, SplatCount, CamX, CamY, CamZ,
		GPUResources->bHasMLPWeights ? 1 : 0, GPUResources->MLPInputDim,
		SHBands, SHCoeffCount);
	CSV += FString::Printf(TEXT("# ply_camera=(%.6f,%.6f,%.6f)  (WorldCam transformed to PLY space)\n"),
		PLYCamera.X, PLYCamera.Y, PLYCamera.Z);
	CSV += TEXT("# ---- output ----\n");
	CSV += FString::Printf(TEXT("# phi           min=%.6f  max=%.6f  mean=%.6f\n"), PhiMin, PhiMax, PhiMean);
	CSV += FString::Printf(TEXT("# opacity       min=%.6f  max=%.6f  mean=%.6f\n"), OpMin,  OpMax,  OpMean);
	CSV += TEXT("# ---- raw UE-space buffers (torch.mean-style flat) ----\n");
	CSV += FString::Printf(TEXT("# position      min=%.6f  max=%.6f  mean=%.6f  (N=%lld)\n"),
		PosMin, PosMax, PosMean, (long long)((int64)SplatCount * 3));
	CSV += FString::Printf(TEXT("# scale         min=%.6f  max=%.6f  mean=%.6f  (N=%lld)\n"),
		ScMin, ScMax, ScMean, (long long)((int64)SplatCount * 3));
	CSV += FString::Printf(TEXT("# rotation      min=%.6f  max=%.6f  mean=%.6f  (N=%lld)\n"),
		RotMin, RotMax, RotMean, (long long)((int64)SplatCount * 4));
	if (SHFloatCount > 0)
	{
		CSV += FString::Printf(TEXT("# sh            min=%.6f  max=%.6f  mean=%.6f  (N=%lld, fp16->fp32)\n"),
			SHMin, SHMax, SHMean, (long long)SHFloatCount);
	}
	else
	{
		CSV += TEXT("# sh            (no SH data)\n");
	}
	CSV += TEXT("# ---- MLP-direct inputs (CPU replica of Phase 0, same values the MLP sees) ----\n");
	CSV += FString::Printf(TEXT("# viewdir_ply   min=%.6f  max=%.6f  mean=%.6f  (N=%lld, normalized)\n"),
		VdMin, VdMax, VdMean, (long long)((int64)SplatCount * 3));
	CSV += FString::Printf(TEXT("# scale_n       min=%.6f  max=%.6f  mean=%.6f  (N=%lld, L2-normalized)\n"),
		ScnMin, ScnMax, ScnMean, (long long)((int64)SplatCount * 3));
	CSV += FString::Printf(TEXT("# rot_ply       min=%.6f  max=%.6f  mean=%.6f  (N=%lld, unit quat permuted)\n"),
		RpMin, RpMax, RpMean, (long long)((int64)SplatCount * 4));
	if (SHFloatCount > 0)
	{
		CSV += FString::Printf(TEXT("# sh_n          min=%.6f  max=%.6f  mean=%.6f  (N=%lld, per-splat L2-normalized)\n"),
			ShnMin, ShnMax, ShnMean, (long long)SHFloatCount);
	}
	else
	{
		CSV += TEXT("# sh_n          (no SH data)\n");
	}
	CSV += FString::Printf(TEXT("# showing first %d of %d splats\n"), NumRows, SplatCount);
	CSV += TEXT("index,phi,opacity\n");
	for (int32 i = 0; i < NumRows; i++)
	{
		CSV += FString::Printf(TEXT("%d,%.6f,%.6f\n"), i, Phi[i], Opacity[i]);
	}

	if (FFileHelper::SaveStringToFile(CSV, *OutputPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
	{
		UE_LOG(LogTemp, Log, TEXT("[TestMLPGPU] CSV -> %s  (%d rows)"), *OutputPath, NumRows);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLPGPU] Cannot write CSV: %s"), *OutputPath);
	}

	// ---------- 8) Cleanup ----------
	// All readbacks go through the render command so their RHI refs
	// (FGPUFenceRHIRef, FStagingBufferRHIRef) release on the render thread.
	ReleaseAll();
}

// Auto-register: available as soon as the NanoGS module loads.
FAutoConsoleCommand GCmdTestMLPForwardGPU(
	TEXT("NanoGS.TestMLPForwardGPU"),
	TEXT("Dispatch MLPForward CS on GPU with a specified camera pos against an imported ")
	TEXT("GaussianSplat asset, then dump phi/opacity values + min/max/mean stats to CSV.\n")
	TEXT("Usage: NanoGS.TestMLPForwardGPU <asset_path> <cam_x> <cam_y> <cam_z> [output.csv]\n")
	TEXT("Example: NanoGS.TestMLPForwardGPU /Game/Splats/MyScene 0 0 200"),
	FConsoleCommandWithArgsDelegate::CreateStatic(&ExecTestMLPForwardGPU));

} // anonymous namespace
