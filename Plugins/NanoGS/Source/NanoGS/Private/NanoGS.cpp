// Copyright Epic Games, Inc. All Rights Reserved.

#include "NanoGS.h"
#include "GaussianSplatViewExtension.h"
#include "GaussianSplatRenderer.h"
#include "GaussianSplatSceneProxy.h"
#include "GaussianSplatRenderData.h"
#include "GaussianGlobalAccumulator.h"
#include "Interfaces/IPluginManager.h"
#include "Misc/Paths.h"
#include "ShaderCore.h"
#include "SceneViewExtension.h"
#include "Misc/CoreDelegates.h"
#include "Engine/Engine.h"
#if PLATFORM_WINDOWS
#include "CudaModule.h"
#endif
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "SceneView.h"
#include "ScreenPass.h"

#define LOCTEXT_NAMESPACE "FNanoGSModule"

// Compute pass parameter struct: no render target slots (compute dispatches must be outside Vulkan render pass)
BEGIN_SHADER_PARAMETER_STRUCT(FGaussianComputePhaseParameters, )
END_SHADER_PARAMETER_STRUCT()

// Pass 2 parameter struct: declares IntermediateTexture as an RDG-tracked shader resource
// so that RDG inserts the proper RTV→SRV barrier between Pass 1 (write) and Pass 2 (read).
BEGIN_SHADER_PARAMETER_STRUCT(FGaussianCompositePassParameters, )
	SHADER_PARAMETER_RDG_TEXTURE(Texture2D, IntermediateTexture)
	RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()

// OIT合成pass参数: 声明OIT累积纹理用于RDG屏障追踪
BEGIN_SHADER_PARAMETER_STRUCT(FGaussianOITCompositePassParameters, )
	SHADER_PARAMETER_RDG_TEXTURE(Texture2D, OITColorWeightTexture)
	SHADER_PARAMETER_RDG_TEXTURE(Texture2D, OITLogTransmitTexture)
	RENDER_TARGET_BINDING_SLOTS()
END_SHADER_PARAMETER_STRUCT()

// OIT渲染模式CVar (定义在GaussianSplatRenderer.cpp)
extern TAutoConsoleVariable<int32> CVarUseOITRendering;

//----------------------------------------------------------------------
// Console Variables for Gaussian Splatting
//----------------------------------------------------------------------

/** Show cluster debug visualization (Nanite-style coloring) */
TAutoConsoleVariable<int32> CVarShowClusterBounds(
	TEXT("gs.ShowClusterBounds"),
	0,
	TEXT("Debug visualization for Gaussian Splat clusters (Nanite-style).\n")
	TEXT("When enabled, shows cluster colors on black background (like Nanite debug view).\n")
	TEXT(" 0: Off (default)\n")
	TEXT(" 1: Show cluster colors (each cluster gets a unique random color)"),
	ECVF_RenderThreadSafe);

/** Maximum number of splats the global accumulator will allocate working buffers for.
 *  Caps VRAM usage for ViewData/sort/histogram buffers. If total visible splats exceed
 *  this budget (after Nanite LOD compaction), excess splats are simply not rendered.
 *  When budget is active, closer assets get priority (farther assets culled first).
 *  Default: 0 (unlimited). Example: 3M budget uses ~195 MB working buffers. */
TAutoConsoleVariable<int32> CVarMaxRenderBudget(
	TEXT("gs.MaxRenderBudget"),
	0,
	TEXT("Maximum number of splats to render per frame (render budget).\n")
	TEXT("Caps global accumulator buffer allocation and GPU-side visible count.\n")
	TEXT("When budget is exceeded, farther assets are culled first (closer assets have priority).\n")
	TEXT("Default: 0 (unlimited). Set to a positive value (e.g. 3000000) to limit splat count."),
	ECVF_RenderThreadSafe);

/** Debug: Force a specific LOD level for debugging LOD hierarchy */
TAutoConsoleVariable<int32> CVarDebugForceLODLevel(
	TEXT("gs.DebugForceLODLevel"),
	-1,
	TEXT("Force rendering of a specific LOD level for debugging (only affects Nanite-enabled assets).\n")
	TEXT("This ignores normal LOD selection and forces all clusters to use specified level.\n")
	TEXT(" -1: Auto - normal LOD selection based on distance/error (default)\n")
	TEXT("  0: Force leaf clusters only - render original splats (finest detail)\n")
	TEXT("  1+: Force specific LOD level (1 = first parent level, 2 = second, etc.)\n")
	TEXT("Note: Higher levels have fewer, coarser splats. Max level depends on asset size.\n")
	TEXT("Use with gs.ShowClusterBounds 2 to visualize which LOD level is being rendered."),
	ECVF_RenderThreadSafe);

/** Debug switch: run CUDA rasterizer bridge once per frame on the closest visible proxy. */
TAutoConsoleVariable<int32> CVarUseCudaRasterizerBridge(
	TEXT("gs.UseCudaRasterizerBridge"),
	0,
	TEXT("Enable UECudaRasterizerBridge::forward in post-opaque render pass.\n")
	TEXT(" 0: Off (default)\n")
	TEXT(" 1: On"),
	ECVF_RenderThreadSafe);

/** Debug display for the CUDA rasterizer output. Reads CudaOutColorBuffer and
 *  writes it as an overlay on top of SceneColor. Requires gs.UseCudaRasterizerBridge=1. */
TAutoConsoleVariable<int32> CVarShowCudaRasterizerDebug(
	TEXT("gs.ShowCudaRasterizerDebug"),
	0,
	TEXT("Display CUDA rasterizer output (sibr::UECudaRasterizerBridge::forward result) on SceneColor.\n")
	TEXT(" 0: Off (default)\n")
	TEXT(" 1: Small overlay in bottom-right quadrant (~1/4 screen)\n")
	TEXT(" 2: Full-screen replace"),
	ECVF_RenderThreadSafe);

TAutoConsoleVariable<float> CVarCudaRasterizerDebugExposure(
	TEXT("gs.CudaRasterizerDebugExposure"),
	1.0f,
	TEXT("Linear exposure multiplier applied to the CUDA rasterizer debug overlay."),
	ECVF_RenderThreadSafe);

// Export for other modules
int32 GGaussianSplatShowClusterBounds = 0;

// Helper to get the renderer module
static IRendererModule& GetRendererModuleRef()
{
	return FModuleManager::GetModuleChecked<IRendererModule>("Renderer");
}

void FNanoGSModule::StartupModule()
{
	// CUDA must be loaded by the game thread before any render-thread code touches it.
#if PLATFORM_WINDOWS
	FCUDAModule& CudaModule = FModuleManager::LoadModuleChecked<FCUDAModule>("CUDA");
	if (!CudaModule.IsAvailable())
	{
		UE_LOG(LogTemp, Warning, TEXT("NanoGS: CUDA module loaded but CUDA driver API is unavailable."));
	}
#endif

	// Pre-load sibr_cudaueinterop_rwdi.dll from the plugin's ThirdParty bin folder.
	//
	// The DLL is marked PublicDelayLoadDLLs in NanoGS.Build.cs, but it depends on a
	// large set of sibling SIBR DLLs (sibr_basic_rwdi.dll, sibr_system_rwdi.dll,
	// sibr_graphics_rwdi.dll, boost_filesystem*, glew32, embree3, tbb, opencv_*, ...)
	// that are NOT copied to $(TargetOutputDir). When the delay-load helper first
	// touches a symbol from the bridge DLL, the OS loader fails to find those
	// transitive deps and the helper raises a fatal exception inside delayhlp.cpp.
	//
	// To fix this we push the SIBR bin folder onto the DLL search path *before*
	// LoadLibrary, then load the bridge DLL explicitly. The OS resolves all
	// transitive imports from the same directory while the search path is active.
	// We keep SibrBridgeDllHandle alive for the lifetime of the module so the
	// delay-load helper just hits an already-loaded module on first symbol use.
	{
		const TSharedPtr<IPlugin> NanoGSPlugin = IPluginManager::Get().FindPlugin(TEXT("NanoGS"));
#if PLATFORM_WINDOWS
		if (NanoGSPlugin.IsValid())
		{
			const FString SibrBinDir = FPaths::ConvertRelativePathToFull(
				FPaths::Combine(NanoGSPlugin->GetBaseDir(),
					TEXT("Source/ThirdParty/SibrCudaUEInterop/bin/Win64")));

			const FString SibrBridgeDllPath = FPaths::Combine(SibrBinDir, TEXT("sibr_cudaueinterop_rwdi.dll"));

			if (FPaths::FileExists(SibrBridgeDllPath))
			{
				FPlatformProcess::PushDllDirectory(*SibrBinDir);
				SibrBridgeDllHandle = FPlatformProcess::GetDllHandle(*SibrBridgeDllPath);
				FPlatformProcess::PopDllDirectory(*SibrBinDir);

				if (SibrBridgeDllHandle == nullptr)
				{
					UE_LOG(LogTemp, Error,
						TEXT("NanoGS: Failed to pre-load sibr_cudaueinterop_rwdi.dll from '%s'. ")
						TEXT("CUDA rasterizer bridge will not be available — calls into UECudaRasterizerBridge ")
						TEXT("will trigger a delay-load fault. Verify that all SIBR sibling DLLs ")
						TEXT("(sibr_basic_rwdi.dll, sibr_system_rwdi.dll, etc.) are present in that folder."),
						*SibrBinDir);
				}
				else
				{
					UE_LOG(LogTemp, Log, TEXT("NanoGS: Pre-loaded SIBR CUDA bridge DLL from '%s'."), *SibrBridgeDllPath);
				}
			}
			else
			{
				UE_LOG(LogTemp, Error,
					TEXT("NanoGS: sibr_cudaueinterop_rwdi.dll not found at '%s'. CUDA rasterizer bridge disabled."),
					*SibrBridgeDllPath);
			}
		}
#endif
	}

	// Register the shader directory so we can use our custom shaders
	FString PluginShaderDir = FPaths::Combine(IPluginManager::Get().FindPlugin(TEXT("NanoGS"))->GetBaseDir(), TEXT("Shaders"));
	AddShaderSourceDirectoryMapping(TEXT("/Plugin/NanoGS"), PluginShaderDir);

	// Allocate the global accumulator (buffers are created lazily on first render)
	GlobalAccumulator = MakeUnique<FGaussianGlobalAccumulator>();

	// Register post-opaque render delegate for rendering
	PostOpaqueRenderDelegateHandle = GetRendererModuleRef().RegisterPostOpaqueRenderDelegate(
		FPostOpaqueRenderDelegate::CreateRaw(this, &FNanoGSModule::OnPostOpaqueRender_RenderThread));

	// Defer view extension creation until GEngine is valid
	// (StartupModule runs before GEngine is initialized, causing an ensure failure)
	PostEngineInitDelegateHandle = FCoreDelegates::OnPostEngineInit.AddRaw(this, &FNanoGSModule::OnPostEngineInit);

	UE_LOG(LogTemp, Log, TEXT("GaussianSplatting module started. Shader directory: %s"), *PluginShaderDir);
}

void FNanoGSModule::OnPostEngineInit()
{
	// Skip view extension creation during cooking/commandlet — GEngine is null in those contexts
	if (IsRunningCommandlet() || !GEngine)
	{
		UE_LOG(LogTemp, Log, TEXT("GaussianSplatting: Skipping ViewExtension creation (commandlet/cook mode)"));
		return;
	}

	// Now GEngine is valid — safe to create the view extension
	ViewExtension = FSceneViewExtensions::NewExtension<FGaussianSplatViewExtension>();

	if (!ViewExtension.IsValid())
	{
		UE_LOG(LogTemp, Error, TEXT("GaussianSplatting: Failed to create ViewExtension!"));
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("GaussianSplatting: ViewExtension created successfully (deferred init)"));
	}
}

// Shared state between compute and raster RDG passes (must be at namespace scope for GraphBuilder.Alloc)
struct FGaussianComputePassResult
{
	bool bHasValidProxies = false;
	bool bUseCompactionPath = false;
	uint32 CappedTotalSplatCount = 0;
};

void FNanoGSModule::OnPostOpaqueRender_RenderThread(FPostOpaqueRenderParameters& Parameters)
{
	FGaussianSplatViewExtension* Ext = FGaussianSplatViewExtension::Get();
	if (!Ext || !Parameters.GraphBuilder || !Parameters.View)
	{
		return;
	}

	FRDGBuilder& GraphBuilder = *Parameters.GraphBuilder;
	const FSceneView* SceneView = reinterpret_cast<const FSceneView*>(Parameters.View);

	TArray<FGaussianSplatSceneProxy*> Proxies;
	Ext->GetRegisteredProxies(Proxies);

	if (Proxies.Num() == 0)
	{
		return;
	}

	// Sort proxies back-to-front for correct depth ordering
	if (Proxies.Num() > 1)
	{
		FVector CameraPosition = SceneView->ViewMatrices.GetViewOrigin();
		Proxies.Sort([CameraPosition](const FGaussianSplatSceneProxy& A, const FGaussianSplatSceneProxy& B)
		{
			float DistA = FVector::DistSquared(CameraPosition, A.GetBounds().Origin);
			float DistB = FVector::DistSquared(CameraPosition, B.GetBounds().Origin);
			return DistA > DistB;
		});
	}

	FRDGTexture* ColorTexture = Parameters.ColorTexture;
	FRDGTexture* DepthTexture = Parameters.DepthTexture;
	FRDGTexture* VelocityTexture = Parameters.VelocityTexture;
	if (!ColorTexture)
	{
		return;
	}
	const FIntPoint RenderExtent = ColorTexture->Desc.Extent;

	int32 DebugMode = CVarShowClusterBounds.GetValueOnRenderThread();
	const bool bUseCudaRasterizerBridge = (CVarUseCudaRasterizerBridge.GetValueOnRenderThread() != 0);
	const bool bUseOIT = (CVarUseOITRendering.GetValueOnRenderThread() != 0);

	// Create intermediate render target for sRGB-space alpha blending.
	// Gaussian splatting trains in sRGB space, so blending must happen in sRGB space
	// to produce correct colors. After compositing, we convert sRGB→linear for SceneColor.
	FRDGTextureDesc IntermediateDesc = FRDGTextureDesc::Create2D(
		ColorTexture->Desc.Extent,
		PF_FloatRGBA,  // Need alpha channel for accumulation tracking
		FClearValueBinding(FLinearColor::Transparent),
		TexCreate_RenderTargetable | TexCreate_ShaderResource);
	FRDGTexture* IntermediateTexture = GraphBuilder.CreateTexture(IntermediateDesc, TEXT("GaussianSplatIntermediateRT"));

	// OIT累积纹理 (仅OIT模式使用)
	FRDGTexture* OITColorWeightTexture = nullptr;
	FRDGTexture* OITLogTransmitTexture = nullptr;
	if (bUseOIT)
	{
		// OIT ColorWeight: RGB = Σ(c×α×w), A = Σ(α×w)
		FRDGTextureDesc OITCWDesc = FRDGTextureDesc::Create2D(
			RenderExtent, PF_FloatRGBA,
			FClearValueBinding(FLinearColor::Transparent),
			TexCreate_RenderTargetable | TexCreate_ShaderResource);
		OITColorWeightTexture = GraphBuilder.CreateTexture(OITCWDesc, TEXT("OIT_ColorWeight"));

		// OIT LogTransmit: R = Σlog(1-α)
		FRDGTextureDesc OITLTDesc = FRDGTextureDesc::Create2D(
			RenderExtent, PF_R16F,
			FClearValueBinding(FLinearColor::Transparent),
			TexCreate_RenderTargetable | TexCreate_ShaderResource);
		OITLogTransmitTexture = GraphBuilder.CreateTexture(OITLTDesc, TEXT("OIT_LogTransmit"));
	}

	// Pass 1: 渲染splats到中间RT (原始) 或 OIT累积纹理
	FRenderTargetParameters* Pass1Parameters = GraphBuilder.AllocParameters<FRenderTargetParameters>();
	if (bUseOIT)
	{
		// OIT: 两个累积纹理 + 速度纹理
		Pass1Parameters->RenderTargets[0] = FRenderTargetBinding(OITColorWeightTexture, ERenderTargetLoadAction::EClear);
		Pass1Parameters->RenderTargets[1] = FRenderTargetBinding(OITLogTransmitTexture, ERenderTargetLoadAction::EClear);
		if (VelocityTexture)
		{
			Pass1Parameters->RenderTargets[2] = FRenderTargetBinding(VelocityTexture, ERenderTargetLoadAction::ELoad);
		}
		// OIT: 只读深度测试 (被场景不透明几何遮挡) + 可写stencil (TSR标记)
		// 不写深度: 无序渲染时深度写入会导致splat间错误遮挡
		if (DepthTexture)
		{
			Pass1Parameters->RenderTargets.DepthStencil = FDepthStencilBinding(
				DepthTexture,
				ERenderTargetLoadAction::ELoad,
				ERenderTargetLoadAction::ELoad,
				FExclusiveDepthStencil::DepthRead_StencilWrite
			);
		}
	}
	else
	{
		// 原始路径: 中间sRGB RT + 速度 + 深度
		Pass1Parameters->RenderTargets[0] = FRenderTargetBinding(IntermediateTexture, ERenderTargetLoadAction::EClear);
		if (VelocityTexture)
		{
			Pass1Parameters->RenderTargets[1] = FRenderTargetBinding(VelocityTexture, ERenderTargetLoadAction::ELoad);
		}
		if (DepthTexture)
		{
			Pass1Parameters->RenderTargets.DepthStencil = FDepthStencilBinding(
				DepthTexture,
				ERenderTargetLoadAction::ELoad,
				ERenderTargetLoadAction::ELoad,
				FExclusiveDepthStencil::DepthWrite_StencilWrite
			);
		}
	}

	if (!GlobalAccumulator.IsValid())
	{
		return;
	}

	//------------------------------------------------------------------
	// GLOBAL ACCUMULATOR PATH: Phase 1 (per-proxy CalcViewData) +
	// Phase 2 (single CalcDistances + RadixSort) + single DrawSplats
	//------------------------------------------------------------------

		// Build the list of visible proxies and compute total splat count (CPU-side)
		struct FProxyRenderInfo
		{
			FGaussianSplatSceneProxy* Proxy;
			FMatrix LocalToWorld;
			uint32 GlobalBaseOffset;
			bool bUseLODRendering;
			float DistanceToCamera;  // For budget priority sorting (closer = higher priority)
		};
		TArray<FProxyRenderInfo> VisibleProxies;
		uint32 TotalSplatCount = 0;
		bool bAllNanite = true;  // True if every visible proxy supports compaction

		FVector CameraLocation = SceneView->ViewLocation;

		for (FGaussianSplatSceneProxy* Proxy : Proxies)
		{
			if (!Proxy) continue;
			if (&Proxy->GetScene() != SceneView->Family->Scene) continue;
			if (!Proxy->IsShown(SceneView)) continue;

			const FBoxSphereBounds& Bounds = Proxy->GetBounds();
			if (!SceneView->ViewFrustum.IntersectBox(Bounds.Origin, Bounds.BoxExtent)) continue;

			FGaussianSplatGPUResources* GPUResources = Proxy->GetGPUResources();
			if (!GPUResources || !GPUResources->IsValid()) continue;

			FProxyRenderInfo Info;
			Info.Proxy = Proxy;
			Info.LocalToWorld = Proxy->GetLocalToWorld();
			Info.GlobalBaseOffset = 0;  // Will be computed after sorting
			Info.bUseLODRendering = GPUResources->bEnableNanite && GPUResources->bHasLODSplats;
			Info.DistanceToCamera = FVector::Dist(Bounds.Origin, CameraLocation);
			VisibleProxies.Add(Info);

			// All proxies must support Nanite compaction for the fast global path
			if (!GPUResources->bEnableNanite || !GPUResources->bHasClusterData || !GPUResources->bSupportsCompaction)
			{
				bAllNanite = false;
			}
		}

		// Sort by distance: closer proxies first (get budget priority when MaxRenderBudget is active)
		VisibleProxies.Sort([](const FProxyRenderInfo& A, const FProxyRenderInfo& B)
		{
			return A.DistanceToCamera < B.DistanceToCamera;
		});

		// Compute GlobalBaseOffset and TotalSplatCount after sorting
		for (FProxyRenderInfo& Info : VisibleProxies)
		{
			Info.GlobalBaseOffset = TotalSplatCount;
			TotalSplatCount += (uint32)Info.Proxy->GetSplatCount();
		}

		// Safety: global accumulator only supports up to MAX_PROXY_COUNT proxies
		if ((uint32)VisibleProxies.Num() > FGaussianGlobalAccumulator::MAX_PROXY_COUNT)
		{
			bAllNanite = false;
		}

		if (TotalSplatCount == 0)
		{
			return;
		}

		// Check camera-static skip: if nothing has changed, skip Phase 1+2 and reuse cached sort
		// Use ProjectionNoAAMatrix to ignore TSR/TAA per-frame jitter that changes every frame
		FMatrix CurrentVP = SceneView->ViewMatrices.GetViewMatrix() * SceneView->ViewMatrices.GetProjectionNoAAMatrix();
		int32 CurrentDebugMode = DebugMode;
		int32 CurrentDebugForceLODLevel = CVarDebugForceLODLevel.GetValueOnRenderThread();

		bool bCanSkip = GlobalAccumulator->bHasCachedSortData &&
			GlobalAccumulator->CachedTotalSplatCount == TotalSplatCount &&
			GlobalAccumulator->CachedViewProjectionMatrix.Equals(CurrentVP, 0.0f);

		if (bCanSkip)
		{
			for (const FProxyRenderInfo& Info : VisibleProxies)
			{
				FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
				float ProxyErrorThreshold = FMath::Max(0.1f, Info.Proxy->GetLODErrorThreshold());
				if (!GPUResources->bHasCachedSortData ||
					!GPUResources->CachedViewProjectionMatrix.Equals(CurrentVP, 0.0f) ||
					!GPUResources->CachedLocalToWorld.Equals(Info.LocalToWorld, 0.0f) ||
					GPUResources->CachedOpacityScale != Info.Proxy->GetOpacityScale() ||
					GPUResources->CachedSplatScale != Info.Proxy->GetSplatScale() ||
					GPUResources->CachedErrorThreshold != ProxyErrorThreshold ||
					GPUResources->CachedDebugMode != CurrentDebugMode ||
					GPUResources->CachedDebugForceLODLevel != CurrentDebugForceLODLevel)
				{
					bCanSkip = false;
					break;
				}
			}
		}

		// Grab index buffer from the first proxy (all proxies use identical quad geometry)
		FBufferRHIRef SharedIndexBuffer;
		if (VisibleProxies.Num() > 0)
		{
			FGaussianSplatGPUResources* FirstRes = VisibleProxies[0].Proxy->GetGPUResources();
			SharedIndexBuffer = FirstRes ? FirstRes->IndexBuffer : FBufferRHIRef();
		}

		FGaussianGlobalAccumulator* RawAccumulator = GlobalAccumulator.Get();

		// Read render budget for global accumulator buffer cap.
		// Disable budget when forcing LOD level — the debug command needs to show
		// all assets regardless of splat count (user expects to see quality vs performance).
		int32 BudgetVal = CVarMaxRenderBudget.GetValueOnRenderThread();
		uint32 MaxRenderBudget = (BudgetVal > 0) ? (uint32)BudgetVal : 0;
		if (CurrentDebugForceLODLevel >= 0)
		{
			MaxRenderBudget = 0;  // Unlimited — debug mode overrides budget
		}

		// Shared state between compute and raster passes (allocated from RDG arena)
		FGaussianComputePassResult* ComputeResult = GraphBuilder.AllocObject<FGaussianComputePassResult>();

		// ---- COMPUTE PASS: All compute shader dispatches (must be outside Vulkan render pass) ----
		FGaussianComputePhaseParameters* ComputePassParams = GraphBuilder.AllocParameters<FGaussianComputePhaseParameters>();
		GraphBuilder.AddPass(
			RDG_EVENT_NAME("GaussianSplat_ComputePhases"),
			ComputePassParams,
			ERDGPassFlags::Compute | ERDGPassFlags::NeverCull,
			[SceneView, VisibleProxies, TotalSplatCount, bCanSkip, bAllNanite, RawAccumulator,
			 SharedIndexBuffer, CurrentVP, CurrentDebugMode,
			 CurrentDebugForceLODLevel, MaxRenderBudget, RenderExtent, bUseCudaRasterizerBridge, bUseOIT, ComputeResult](FRHICommandListImmediate& RHICmdList)
			{
				if (!SceneView) return;
				SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatCompute_Global);

				// SAFETY CHECK: Re-validate all proxies before rendering.
				// Proxies may have been destroyed between when we built VisibleProxies
				// and when this lambda executes (RDG deferred execution).
				// We need to rebuild the list with only valid proxies.
				TArray<FProxyRenderInfo> ValidProxies;
				ValidProxies.Reserve(VisibleProxies.Num());
				uint32 NewTotalSplatCount = 0;

				for (const auto& Info : VisibleProxies)
				{
					// Check if proxy is still valid (not destroyed or pending destruction)
					if (Info.Proxy && Info.Proxy->IsValidForRendering())
					{
						FProxyRenderInfo ValidInfo = Info;
						ValidInfo.GlobalBaseOffset = NewTotalSplatCount;
						NewTotalSplatCount += (uint32)Info.Proxy->GetSplatCount();
						ValidProxies.Add(ValidInfo);
					}
				}

				// If no valid proxies remain, skip rendering entirely
				if (ValidProxies.Num() == 0 || NewTotalSplatCount == 0)
				{
					return;
				}

				ComputeResult->bHasValidProxies = true;

				// Invalidate cache skip if the proxy list changed (some proxies were destroyed)
				// This ensures we don't use stale cached data when the scene has changed
				bool bCanSkipAdjusted = bCanSkip;
				if (ValidProxies.Num() != VisibleProxies.Num() || NewTotalSplatCount != TotalSplatCount)
				{
					bCanSkipAdjusted = false;
					// Also invalidate the global accumulator cache since proxy set changed
					RawAccumulator->bHasCachedSortData = false;
				}

				// OIT模式切换时使缓存失效 (OIT与原始路径的ViewData/排序数据不兼容)
				if (bUseOIT != RawAccumulator->CachedUseOIT)
				{
					bCanSkipAdjusted = false;
					RawAccumulator->bHasCachedSortData = false;
				}

				// Initialize color textures (deferred init) - only for valid proxies
				for (const auto& Info : ValidProxies)
				{
					Info.Proxy->TryInitializeColorTexture(RHICmdList);
				}

				if (bUseCudaRasterizerBridge && ValidProxies.Num() > 0)
				{
					FGaussianSplatGPUResources* FirstGPUResources = ValidProxies[0].Proxy->GetGPUResources();
					FGaussianSplatRenderData* SharedRenderData = FirstGPUResources ? FirstGPUResources->GetSharedRenderData() : nullptr;
					if (SharedRenderData)
					{
						SharedRenderData->ForwardWithCudaRasterizer(RHICmdList, *SceneView, RenderExtent.X, RenderExtent.Y);
					}
				}

				// Ensure global buffers are large enough for all splats
				RawAccumulator->ResizeIfNeeded(RHICmdList, NewTotalSplatCount);

				// Re-check if all valid proxies support Nanite compaction
				bool bAllValidNanite = true;
				for (const auto& Info : ValidProxies)
				{
					FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
					if (!GPUResources || !GPUResources->bEnableNanite || !GPUResources->bHasClusterData || !GPUResources->bSupportsCompaction)
					{
						bAllValidNanite = false;
						break;
					}
				}

				// Cap to MAX_PROXY_COUNT
				if ((uint32)ValidProxies.Num() > FGaussianGlobalAccumulator::MAX_PROXY_COUNT)
				{
					bAllValidNanite = false;
				}

				if (bAllValidNanite)
				{
					//==================================================
					// GLOBAL + COMPACTION PATH
					// All proxies are Nanite-enabled: GPU compaction
					// reduces working set from TotalSplatCount → TotalVisible
					// (~140x reduction at LOD5 for a 719K-splat tile).
					//==================================================

					// Ensure fixed-size prefix-sum buffers exist (allocated once)
					RawAccumulator->EnsureCompactionBuffersAllocated(RHICmdList);

					if (!bCanSkipAdjusted)
					{
						// --------------------------------------------------
						// Phase 0: Per-proxy culling + compaction + indirect args
						// Early-out: skip proxies once cumulative splat count
						// exceeds MaxRenderBudget (CPU-side estimate using total
						// splat count as conservative upper bound for visible count).
						// Proxies are sorted by distance, so closer ones get priority.
						// --------------------------------------------------
						int32 NumProcessedProxies = 0;
						uint32 CumulativeSplatCount = 0;

						for (const auto& Info : ValidProxies)
						{
							// Budget early-out: if cumulative total already exceeds budget,
							// skip culling/compaction for remaining (farther) proxies
							if (MaxRenderBudget > 0 && CumulativeSplatCount >= MaxRenderBudget)
							{
								break;
							}

							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							if (!GPUResources) continue;  // Extra safety check
							int32 SplatCount = Info.Proxy->GetSplatCount();
							int32 OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;

							// Cluster culling → fills ClusterVisibilityBitmap
							FGaussianSplatRenderer::DispatchClusterCulling(
								RHICmdList, *SceneView, GPUResources,
								Info.LocalToWorld, Info.Proxy->GetLODErrorThreshold(), Info.bUseLODRendering);

							// Compact → fills CompactedSplatIndices + VisibleSplatCountBuffer
							FGaussianSplatRenderer::DispatchCompactSplats(
								RHICmdList, GPUResources,
								SplatCount, OriginalSplatCount, Info.bUseLODRendering);

							// PrepareIndirectArgs → fills IndirectDispatchArgsBuffer for CalcViewData
							FGaussianSplatRenderer::DispatchPrepareIndirectArgs(RHICmdList, GPUResources);

							CumulativeSplatCount += (uint32)SplatCount;
							NumProcessedProxies++;
						}

						// --------------------------------------------------
						// Phase 1: Gather visible counts + GPU prefix sum
						// Only gather from proxies that were actually processed
						// --------------------------------------------------
						for (int32 i = 0; i < NumProcessedProxies; i++)
						{
							FGaussianSplatGPUResources* GPUResources = ValidProxies[i].Proxy->GetGPUResources();
							if (!GPUResources) continue;  // Extra safety check
							FGaussianSplatRenderer::DispatchGatherVisibleCount(
								RHICmdList, GPUResources, RawAccumulator, i);
						}

						// Single 1-thread dispatch: computes prefix sums + writes all indirect args
						FGaussianSplatRenderer::DispatchPrefixSumVisibleCounts(
							RHICmdList, RawAccumulator, NumProcessedProxies, MaxRenderBudget);

						// --------------------------------------------------
						// Phase 2: Per-proxy CalcViewData → global buffer
						// (indirect dispatch, only visible splats per proxy)
						// Only process proxies that went through culling/compaction
						// --------------------------------------------------
						for (int32 i = 0; i < NumProcessedProxies; i++)
						{
							const auto& Info = ValidProxies[i];
							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							if (!GPUResources) continue;
							int32 SplatCount = Info.Proxy->GetSplatCount();
							int32 OriginalSplatCount = SplatCount - GPUResources->LODSplatCount;

							if (bUseOIT)
							{
								// OIT: MLP推理 → CalcViewDataOIT (紧凑模式)
								FRHIBufferCreateDesc PhiDesc = FRHIBufferCreateDesc::Create(
									TEXT("PhiOpacityBuffer"),
									SplatCount * 2 * sizeof(float),
									2 * sizeof(float),
									BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
									.SetInitialState(ERHIAccess::UAVCompute);
								FBufferRHIRef PhiOpacityBuffer = RHICmdList.CreateBuffer(PhiDesc);

								FGaussianSplatRenderer::DispatchMLPForward(
									RHICmdList, GPUResources,
									FVector3f(SceneView->ViewLocation),
									SplatCount,
									Info.Proxy->GetOpacityScale(),
									PhiOpacityBuffer,
									Info.Proxy->GetEnableMLPWeights());

								FShaderResourceViewRHIRef PhiSRV = RHICmdList.CreateShaderResourceView(
									PhiOpacityBuffer, FRHIViewDesc::CreateBufferSRV()
										.SetType(FRHIViewDesc::EBufferType::Structured)
										.SetStride(2 * sizeof(float)));

								FGaussianSplatRenderer::DispatchCalcViewDataOITCompactedGlobal(
									RHICmdList, *SceneView, GPUResources,
									Info.LocalToWorld,
									SplatCount, OriginalSplatCount,
									Info.Proxy->GetSHOrder(),
									Info.Proxy->GetOpacityScale(),
									Info.Proxy->GetSplatScale(),
									i, RawAccumulator, MaxRenderBudget,
									PhiSRV);
							}
							else
							{
								FGaussianSplatRenderer::DispatchCalcViewDataCompactedGlobal(
									RHICmdList, *SceneView, GPUResources,
									Info.LocalToWorld,
									SplatCount, OriginalSplatCount,
									Info.Proxy->GetSHOrder(),
									Info.Proxy->GetOpacityScale(),
									Info.Proxy->GetSplatScale(),
									i, RawAccumulator, MaxRenderBudget);
							}
						}

						// --------------------------------------------------
						// Phase 3: CalcDistances + RadixSort (原始路径)
						// OIT模式跳过排序 (加权平均混合与顺序无关)
						// --------------------------------------------------
						if (!bUseOIT)
						{
							FGaussianSplatRenderer::DispatchCalcDistancesGlobalIndirect(RHICmdList, RawAccumulator);
							FGaussianSplatRenderer::DispatchRadixSortGlobalIndirect(RHICmdList, RawAccumulator);
						}

						// Update caches — only for processed proxies
						RawAccumulator->bHasCachedSortData = true;
						RawAccumulator->CachedTotalSplatCount = NewTotalSplatCount;
						RawAccumulator->CachedViewProjectionMatrix = CurrentVP;
						RawAccumulator->CachedUseOIT = bUseOIT;

						for (int32 i = 0; i < ValidProxies.Num(); i++)
						{
							FGaussianSplatGPUResources* GPUResources = ValidProxies[i].Proxy->GetGPUResources();
							if (!GPUResources) continue;

							if (i < NumProcessedProxies)
							{
								const auto& Info = ValidProxies[i];
								GPUResources->CachedViewProjectionMatrix = CurrentVP;
								GPUResources->CachedLocalToWorld = Info.LocalToWorld;
								GPUResources->CachedOpacityScale = Info.Proxy->GetOpacityScale();
								GPUResources->CachedSplatScale = Info.Proxy->GetSplatScale();
								GPUResources->CachedErrorThreshold = FMath::Max(0.1f, Info.Proxy->GetLODErrorThreshold());
								GPUResources->CachedDebugMode = CurrentDebugMode;
								GPUResources->CachedDebugForceLODLevel = CurrentDebugForceLODLevel;
								GPUResources->bHasCachedSortData = true;
							}
							else
							{
								// Invalidate cache for budget-skipped proxies so they
								// don't block the camera-static skip check
								GPUResources->bHasCachedSortData = false;
							}
						}
					}

					// Store result for raster pass — draw call happens in raster pass below
					ComputeResult->bUseCompactionPath = true;
				}
				else
				{
					//==================================================
					// NON-COMPACTION GLOBAL PATH (fallback)
					// Not all proxies are Nanite-enabled.
					// Sorts all NewTotalSplatCount splats (no compaction benefit).
					// Still provides correct cross-tile alpha blending.
					//==================================================

					// Cap splat count to render budget (CPU-side enforcement)
					uint32 CappedTotalSplatCount = NewTotalSplatCount;
					if (MaxRenderBudget > 0 && CappedTotalSplatCount > MaxRenderBudget)
					{
						CappedTotalSplatCount = MaxRenderBudget;
					}

					if (!bCanSkipAdjusted)
					{
						// --------------------------------------------------
						// Phase 1: Per-proxy ClusterCulling + CalcViewData
						// --------------------------------------------------
						for (const auto& Info : ValidProxies)
						{
							// Skip proxies that would write beyond the render budget
							if (MaxRenderBudget > 0 && Info.GlobalBaseOffset >= MaxRenderBudget)
							{
								break;  // All subsequent proxies also exceed the budget
							}

							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							if (!GPUResources) continue;  // Extra safety check

							// Cluster culling for Nanite-enabled proxies
							if (GPUResources->bEnableNanite && GPUResources->bHasClusterData)
							{
								FGaussianSplatRenderer::DispatchClusterCulling(
									RHICmdList, *SceneView, GPUResources,
									Info.LocalToWorld, Info.Proxy->GetLODErrorThreshold(), Info.bUseLODRendering);
							}

							if (bUseOIT)
							{
								// OIT: MLP推理 → CalcViewDataOIT
								int32 ProxySplatCount = Info.Proxy->GetSplatCount();
								FRHIBufferCreateDesc PhiDesc = FRHIBufferCreateDesc::Create(
									TEXT("PhiOpacityBuffer"),
									ProxySplatCount * 2 * sizeof(float),
									2 * sizeof(float),
									BUF_UnorderedAccess | BUF_ShaderResource | BUF_StructuredBuffer)
									.SetInitialState(ERHIAccess::UAVCompute);
								FBufferRHIRef PhiOpacityBuffer = RHICmdList.CreateBuffer(PhiDesc);

								FGaussianSplatRenderer::DispatchMLPForward(
									RHICmdList, GPUResources,
									FVector3f(SceneView->ViewLocation),
									ProxySplatCount,
									Info.Proxy->GetOpacityScale(),
									PhiOpacityBuffer,
									Info.Proxy->GetEnableMLPWeights());

								FShaderResourceViewRHIRef PhiSRV = RHICmdList.CreateShaderResourceView(
									PhiOpacityBuffer, FRHIViewDesc::CreateBufferSRV()
										.SetType(FRHIViewDesc::EBufferType::Structured)
										.SetStride(2 * sizeof(float)));

								FGaussianSplatRenderer::DispatchCalcViewDataOIT(
									RHICmdList, *SceneView, GPUResources,
									Info.LocalToWorld,
									ProxySplatCount,
									Info.Proxy->GetSHOrder(),
									Info.Proxy->GetOpacityScale(),
									Info.Proxy->GetSplatScale(),
									Info.bUseLODRendering,
									Info.GlobalBaseOffset,
									RawAccumulator,
									PhiSRV);
							}
							else
							{
								// 原始路径: CalcViewData → GlobalViewDataBuffer
								FGaussianSplatRenderer::DispatchCalcViewDataGlobal(
									RHICmdList, *SceneView, GPUResources,
									Info.LocalToWorld,
									Info.Proxy->GetSplatCount(),
									Info.Proxy->GetSHOrder(),
									Info.Proxy->GetOpacityScale(),
									Info.Proxy->GetSplatScale(),
									Info.bUseLODRendering,
									Info.GlobalBaseOffset,
									RawAccumulator);
							}
						}

						// --------------------------------------------------
						// Phase 2: CalcDistances + RadixSort (原始路径)
						// OIT模式跳过排序
						// --------------------------------------------------
						if (!bUseOIT)
						{
							FGaussianSplatRenderer::DispatchCalcDistancesGlobal(RHICmdList, RawAccumulator, (int32)CappedTotalSplatCount);
							FGaussianSplatRenderer::DispatchRadixSortGlobal(RHICmdList, RawAccumulator, (int32)CappedTotalSplatCount);
						}

						// Update caches
						RawAccumulator->bHasCachedSortData = true;
						RawAccumulator->CachedTotalSplatCount = NewTotalSplatCount;
						RawAccumulator->CachedViewProjectionMatrix = CurrentVP;
						RawAccumulator->CachedUseOIT = bUseOIT;

						for (const auto& Info : ValidProxies)
						{
							FGaussianSplatGPUResources* GPUResources = Info.Proxy->GetGPUResources();
							if (!GPUResources) continue;  // Extra safety check
							GPUResources->CachedViewProjectionMatrix = CurrentVP;
							GPUResources->CachedLocalToWorld = Info.LocalToWorld;
							GPUResources->CachedOpacityScale = Info.Proxy->GetOpacityScale();
							GPUResources->CachedSplatScale = Info.Proxy->GetSplatScale();

							GPUResources->CachedErrorThreshold = FMath::Max(0.1f, Info.Proxy->GetLODErrorThreshold());
							GPUResources->CachedDebugMode = CurrentDebugMode;
							GPUResources->CachedDebugForceLODLevel = CurrentDebugForceLODLevel;
							GPUResources->bHasCachedSortData = true;
						}
					}

					// Store result for raster pass
					ComputeResult->CappedTotalSplatCount = CappedTotalSplatCount;
				}
			}
		);

		// ---- RASTER PASS: Draw calls only (inside Vulkan render pass) ----
		GraphBuilder.AddPass(
			RDG_EVENT_NAME("GaussianSplat_RenderToIntermediate"),
			Pass1Parameters,
			ERDGPassFlags::Raster,
			[SceneView, RawAccumulator, SharedIndexBuffer, DebugMode, bUseOIT, ComputeResult](FRHICommandListImmediate& RHICmdList)
			{
				if (!SceneView || !ComputeResult->bHasValidProxies) return;
				SCOPED_DRAW_EVENT(RHICmdList, GaussianSplatRendering_Global);

				if (bUseOIT)
				{
					// OIT: 使用加法混合绘制到累积纹理 (无需排序)
					if (ComputeResult->bUseCompactionPath)
					{
						FGaussianSplatRenderer::DrawSplatsOITGlobalIndirect(
							RHICmdList, *SceneView, RawAccumulator, SharedIndexBuffer, DebugMode);
					}
					else
					{
						FGaussianSplatRenderer::DrawSplatsOIT(
							RHICmdList, *SceneView, RawAccumulator, SharedIndexBuffer,
							(int32)ComputeResult->CappedTotalSplatCount, DebugMode);
					}
				}
				else
				{
					// 原始路径: 排序后的alpha混合
					if (ComputeResult->bUseCompactionPath)
					{
						FGaussianSplatRenderer::DrawSplatsGlobalIndirect(
							RHICmdList, *SceneView, RawAccumulator, SharedIndexBuffer, DebugMode);
					}
					else
					{
						FGaussianSplatRenderer::DrawSplatsGlobal(
							RHICmdList, *SceneView, RawAccumulator,
							SharedIndexBuffer, (int32)ComputeResult->CappedTotalSplatCount, DebugMode);
					}
				}
			}
		);

		// Pass 2: 合成到SceneColor
		ERenderTargetLoadAction CompositeColorLoadAction = (DebugMode > 0) ? ERenderTargetLoadAction::EClear : ERenderTargetLoadAction::ELoad;

		if (bUseOIT)
		{
			// OIT合成: 解算加权平均并写入SceneColor
			FGaussianOITCompositePassParameters* OITPass2Params = GraphBuilder.AllocParameters<FGaussianOITCompositePassParameters>();
			OITPass2Params->OITColorWeightTexture = OITColorWeightTexture;
			OITPass2Params->OITLogTransmitTexture = OITLogTransmitTexture;
			OITPass2Params->RenderTargets[0] = FRenderTargetBinding(ColorTexture, CompositeColorLoadAction);

			GraphBuilder.AddPass(
				RDG_EVENT_NAME("GaussianSplat_OITComposite"),
				OITPass2Params,
				ERDGPassFlags::Raster,
				[SceneView, OITColorWeightTexture, OITLogTransmitTexture](FRHICommandListImmediate& RHICmdList)
				{
					if (!SceneView) return;
					FRHITexture* CWRHI = OITColorWeightTexture->GetRHI();
					FRHITexture* LTRHI = OITLogTransmitTexture->GetRHI();
					if (!CWRHI || !LTRHI) return;
					FGaussianSplatRenderer::CompositeOITToSceneColor(RHICmdList, *SceneView, CWRHI, LTRHI);
				}
			);
		}
		else
		{
		// 原始sRGB合成 + CUDA debug overlay
		// RDG屏障追踪: IntermediateTexture从RTV→SRV
		FGaussianCompositePassParameters* Pass2Parameters = GraphBuilder.AllocParameters<FGaussianCompositePassParameters>();
		Pass2Parameters->IntermediateTexture = IntermediateTexture;
		Pass2Parameters->RenderTargets[0] = FRenderTargetBinding(ColorTexture, CompositeColorLoadAction);

		// CUDA rasterizer debug overlay settings, evaluated once at build time and
		// captured by Pass2's execution lambda. Only active when the bridge itself
		// is enabled — otherwise CudaOutColorBufferSRV will never be populated.
		const int32 CudaDebugMode = CVarShowCudaRasterizerDebug.GetValueOnRenderThread();
		const float CudaDebugExposure = CVarCudaRasterizerDebugExposure.GetValueOnRenderThread();
		const bool bCudaDebugEnabled = bUseCudaRasterizerBridge && CudaDebugMode > 0 && Proxies.Num() > 0;

		GraphBuilder.AddPass(
			RDG_EVENT_NAME("GaussianSplat_CompositeToSceneColor"),
			Pass2Parameters,
			ERDGPassFlags::Raster,
			[SceneView, IntermediateTexture, Proxies, RenderExtent, CudaDebugMode, CudaDebugExposure, bCudaDebugEnabled]
			(FRHICommandListImmediate& RHICmdList)
			{
				if (!SceneView) return;

				FRHITexture* IntermediateRHI = IntermediateTexture->GetRHI();
				if (!IntermediateRHI) return;

				FGaussianSplatRenderer::CompositeToSceneColor(
					RHICmdList, *SceneView, IntermediateRHI);

				if (!bCudaDebugEnabled)
				{
					return;
				}

				// Find the first valid proxy that has a populated CUDA output SRV.
				// Pass1 only runs forward() on ValidProxies[0], so normally only
				// the closest proxy has a live SRV — scan defensively in case the
				// proxy set changed between Pass1 and Pass2 execution.
				FGaussianSplatRenderData* DebugRenderData = nullptr;
				for (FGaussianSplatSceneProxy* Proxy : Proxies)
				{
					if (!Proxy || !Proxy->IsValidForRendering()) continue;
					FGaussianSplatGPUResources* GPUResources = Proxy->GetGPUResources();
					if (!GPUResources) continue;
					FGaussianSplatRenderData* SharedRenderData = GPUResources->GetSharedRenderData();
					if (SharedRenderData && SharedRenderData->CudaOutColorBufferSRV.IsValid()
						&& SharedRenderData->CudaOutColorBufferWidth > 0
						&& SharedRenderData->CudaOutColorBufferHeight > 0)
					{
						DebugRenderData = SharedRenderData;
						break;
					}
				}
				if (!DebugRenderData) return;

				// DestRect in viewport-space pixels (absolute). The PS uses these to
				// decide per-pixel whether to sample the CUDA buffer or discard.
				FVector4f DestRect;
				if (CudaDebugMode >= 2)
				{
					// Full-screen replace
					DestRect = FVector4f(0.0f, 0.0f,
						static_cast<float>(RenderExtent.X),
						static_cast<float>(RenderExtent.Y));
				}
				else
				{
					// Bottom-right quadrant, half width/half height
					const float HalfW = static_cast<float>(RenderExtent.X) * 0.5f;
					const float HalfH = static_cast<float>(RenderExtent.Y) * 0.5f;
					DestRect = FVector4f(
						static_cast<float>(RenderExtent.X) - HalfW,
						static_cast<float>(RenderExtent.Y) - HalfH,
						static_cast<float>(RenderExtent.X),
						static_cast<float>(RenderExtent.Y));
				}

				FGaussianSplatRenderer::BlitCudaOutColorDebug(
					RHICmdList,
					*SceneView,
					DebugRenderData->CudaOutColorBufferSRV,
					FUintVector2(DebugRenderData->CudaOutColorBufferWidth, DebugRenderData->CudaOutColorBufferHeight),
					DestRect,
					CudaDebugExposure);
			}
		);
		} // !bUseOIT
}

void FNanoGSModule::ShutdownModule()
{
	// Remove deferred init delegate
	if (PostEngineInitDelegateHandle.IsValid())
	{
		FCoreDelegates::OnPostEngineInit.Remove(PostEngineInitDelegateHandle);
		PostEngineInitDelegateHandle.Reset();
	}

	// Unregister post-opaque render delegate
	if (PostOpaqueRenderDelegateHandle.IsValid())
	{
		GetRendererModuleRef().RemovePostOpaqueRenderDelegate(PostOpaqueRenderDelegateHandle);
		PostOpaqueRenderDelegateHandle.Reset();
	}

	// Release global accumulator GPU buffers from the render thread
	if (GlobalAccumulator.IsValid())
	{
		FGaussianGlobalAccumulator* RawAccumulator = GlobalAccumulator.Release();
		ENQUEUE_RENDER_COMMAND(ReleaseGlobalAccumulator)(
			[RawAccumulator](FRHICommandListImmediate& RHICmdList)
			{
				RawAccumulator->Release();
				delete RawAccumulator;
			});
	}

	// Clear the view extension
	ViewExtension.Reset();

	// Free the pre-loaded SIBR bridge DLL handle. The render-thread bridge
	// instance is owned by FGaussianSplatRenderData, which is destroyed before
	// the module shuts down via the asset/proxy lifetime, so unloading here
	// is safe.
	if (SibrBridgeDllHandle != nullptr)
	{
		FPlatformProcess::FreeDllHandle(SibrBridgeDllHandle);
		SibrBridgeDllHandle = nullptr;
	}
}

FNanoGSModule& FNanoGSModule::Get()
{
	return FModuleManager::LoadModuleChecked<FNanoGSModule>("NanoGS");
}

bool FNanoGSModule::IsAvailable()
{
	return FModuleManager::Get().IsModuleLoaded("NanoGS");
}

#undef LOCTEXT_NAMESPACE

IMPLEMENT_MODULE(FNanoGSModule, NanoGS)
