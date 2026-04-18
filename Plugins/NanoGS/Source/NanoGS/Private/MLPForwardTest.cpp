// Console command: NanoGS.TestMLPForward
//
// Runs OpaictyPhiNN MLP forward on CPU using exported debug data.
// Math is identical to MLPForwardCS.usf — same weight layout, same layer sizes,
// same activations — so output differences pinpoint GPU-side issues.
//
// Usage (UE5 console or cmd):
//   NanoGS.TestMLPForward <mlp_input.bin> <mlp_weights.bin> [output.csv]
//
// If output.csv is omitted, writes to mlp_output_ue5.csv next to mlp_input.bin.

#include "CoreMinimal.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HAL/PlatformFileManager.h"

namespace
{

// ---- binary header of mlp_input.bin (Python side) ----
#pragma pack(push, 1)
struct FMLPInputHeader
{
	uint32 Magic;      // 0x4D4C5044 ("MLPD")
	uint32 N;          // num gaussians
	uint32 InputDim;   // total feature dimension
	uint32 SHDim;      // flattened SH dim = 3*(degree+1)^2
	uint32 ViewdirDim; // 3
	uint32 ScaleDim;   // 3
	uint32 RotDim;     // 4
};
#pragma pack(pop)
static_assert(sizeof(FMLPInputHeader) == 28, "packed 7 x uint32");

// ---- CPU MLP forward (mirrors MLPForwardCS.usf exactly) ----
//
// Network: Input(D) -> 256+ReLU -> 128+ReLU -> 64 -> phi(1+ReLU), opacity(1+Sigmoid)
// Weight layout: same row-major order as export_mlp_weights.py / shader offsets.
//
static void RunForwardCPU(
	const float* Feat,   // [N x D] assembled input (post-normalize)
	const float* W,      // flat weight array from mlp_weights.bin
	int32 N, int32 D,
	TArray<float>& OutPhi,
	TArray<float>& OutOp)
{
	constexpr int32 H1 = 256, H2 = 128, H3 = 64;

	// weight offsets (same as shader)
	const int32 oW1 = 0;
	const int32 oB1 = H1 * D;
	const int32 oW2 = oB1 + H1;
	const int32 oB2 = oW2 + H2 * H1;
	const int32 oW3 = oB2 + H2;
	const int32 oB3 = oW3 + H3 * H2;
	const int32 oWp = oB3 + H3;
	const int32 oBp = oWp + H3;
	const int32 oWo = oBp + 1;
	const int32 oBo = oWo + H3;

	OutPhi.SetNumUninitialized(N);
	OutOp.SetNumUninitialized(N);

	for (int32 i = 0; i < N; ++i)
	{
		const float* x = Feat + (int64)i * D;

		// Layer 1: Linear(D -> 256) + ReLU
		float h1[H1];
		for (int32 j = 0; j < H1; ++j)
		{
			float s = W[oB1 + j];
			const int32 base = oW1 + j * D;
			for (int32 k = 0; k < D; ++k)
				s += W[base + k] * x[k];
			h1[j] = FMath::Max(s, 0.f);
		}

		// Layer 2: Linear(256 -> 128) + ReLU
		float h2[H2];
		for (int32 j = 0; j < H2; ++j)
		{
			float s = W[oB2 + j];
			const int32 base = oW2 + j * H1;
			for (int32 k = 0; k < H1; ++k)
				s += W[base + k] * h1[k];
			h2[j] = FMath::Max(s, 0.f);
		}

		// Layer 3: Linear(128 -> 64), no activation
		float h3[H3];
		for (int32 j = 0; j < H3; ++j)
		{
			float s = W[oB3 + j];
			const int32 base = oW3 + j * H2;
			for (int32 k = 0; k < H2; ++k)
				s += W[base + k] * h2[k];
			h3[j] = s;
		}

		// Phi head: Linear(64 -> 1) + ReLU
		float phi = W[oBp];
		for (int32 k = 0; k < H3; ++k)
			phi += W[oWp + k] * h3[k];
		OutPhi[i] = FMath::Max(phi, 0.f);

		// Opacity head: Linear(64 -> 1) + Sigmoid
		float op = W[oBo];
		for (int32 k = 0; k < H3; ++k)
			op += W[oWo + k] * h3[k];
		OutOp[i] = 1.f / (1.f + FMath::Exp(-op));
	}
}

// ---- console command entry ----

static void ExecTestMLPForward(const TArray<FString>& Args)
{
	if (Args.Num() < 2)
	{
		UE_LOG(LogTemp, Warning,
			TEXT("Usage: NanoGS.TestMLPForward <mlp_input.bin> <mlp_weights.bin> [output.csv]"));
		return;
	}

	const FString& InputPath   = Args[0];
	const FString& WeightsPath = Args[1];
	const FString OutputPath   = Args.Num() >= 3
		? Args[2]
		: FPaths::GetPath(InputPath) / TEXT("mlp_output_ue5.csv");

	// ======== load mlp_input.bin ========
	TArray<uint8> InBuf;
	if (!FFileHelper::LoadFileToArray(InBuf, *InputPath))
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLP] Cannot load %s"), *InputPath);
		return;
	}
	if (InBuf.Num() < (int32)sizeof(FMLPInputHeader))
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLP] File too small: %s"), *InputPath);
		return;
	}

	const FMLPInputHeader& Hdr = *reinterpret_cast<const FMLPInputHeader*>(InBuf.GetData());
	if (Hdr.Magic != 0x4D4C5044)
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLP] Bad magic 0x%08X (expected 0x4D4C5044)"), Hdr.Magic);
		return;
	}

	const int32 N  = (int32)Hdr.N;
	const int32 ID = (int32)Hdr.InputDim;
	const int32 SD = (int32)Hdr.SHDim;

	// Data layout after header + camera(12 bytes):
	//   shs_raw [N*SD]  viewdirs [N*3]  scales [N*3]  rotations [N*4]  feat_concat [N*ID]
	const int32 HdrBytes   = (int32)sizeof(FMLPInputHeader) + 3 * (int32)sizeof(float); // 40
	const int64 SkipFloats = (int64)N * SD + (int64)N * 3 + (int64)N * 3 + (int64)N * 4;
	const int64 FeatOff    = HdrBytes + SkipFloats * sizeof(float);
	const int64 FeatBytes  = (int64)N * ID * sizeof(float);

	if (InBuf.Num() < FeatOff + FeatBytes)
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLP] mlp_input.bin truncated (need %lld, have %d)"),
			FeatOff + FeatBytes, InBuf.Num());
		return;
	}
	const float* FeatData = reinterpret_cast<const float*>(InBuf.GetData() + FeatOff);

	// ======== load mlp_weights.bin ========
	TArray<uint8> WBuf;
	if (!FFileHelper::LoadFileToArray(WBuf, *WeightsPath))
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLP] Cannot load %s"), *WeightsPath);
		return;
	}
	if (WBuf.Num() < 8)
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLP] mlp_weights.bin too small"));
		return;
	}

	const uint32 WInputDim    = *reinterpret_cast<const uint32*>(WBuf.GetData());
	const uint32 WTotalFloats = *reinterpret_cast<const uint32*>(WBuf.GetData() + 4);

	if ((int32)WInputDim != ID)
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLP] input_dim mismatch: input.bin=%d  weights.bin=%u"),
			ID, WInputDim);
		return;
	}
	if (WBuf.Num() < (int64)(8 + WTotalFloats * sizeof(float)))
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLP] mlp_weights.bin truncated"));
		return;
	}
	const float* WData = reinterpret_cast<const float*>(WBuf.GetData() + 8);

	// ======== CPU forward ========
	UE_LOG(LogTemp, Log, TEXT("[TestMLP] Running CPU forward:  N=%d  InputDim=%d ..."), N, ID);

	double T0 = FPlatformTime::Seconds();
	TArray<float> Phi, Opacity;
	RunForwardCPU(FeatData, WData, N, ID, Phi, Opacity);
	double Elapsed = FPlatformTime::Seconds() - T0;

	UE_LOG(LogTemp, Log, TEXT("[TestMLP] Forward done in %.3f s"), Elapsed);

	// ======== write CSV ========
	FString CSV;
	CSV.Reserve(N * 32 + 32);
	CSV += TEXT("index,phi,opacity\n");
	for (int32 i = 0; i < N; ++i)
	{
		CSV += FString::Printf(TEXT("%d,%.6f,%.6f\n"), i, Phi[i], Opacity[i]);
	}

	if (FFileHelper::SaveStringToFile(CSV, *OutputPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
	{
		UE_LOG(LogTemp, Log, TEXT("[TestMLP] Output -> %s  (%d rows)"), *OutputPath, N);
	}
	else
	{
		UE_LOG(LogTemp, Error, TEXT("[TestMLP] Cannot write %s"), *OutputPath);
	}
}

// Auto-register: available as soon as the NanoGS module loads.
FAutoConsoleCommand GCmdTestMLPForward(
	TEXT("NanoGS.TestMLPForward"),
	TEXT("CPU MLP forward pass for validation.\n")
	TEXT("Reads feat_concat from mlp_input.bin, weights from mlp_weights.bin, writes CSV.\n")
	TEXT("Usage: NanoGS.TestMLPForward <mlp_input.bin> <mlp_weights.bin> [output.csv]"),
	FConsoleCommandWithArgsDelegate::CreateStatic(&ExecTestMLPForward));

} // anonymous namespace
