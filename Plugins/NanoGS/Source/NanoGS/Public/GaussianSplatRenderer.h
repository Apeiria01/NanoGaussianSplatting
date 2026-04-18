// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "RenderGraphResources.h"
#include "RenderGraphBuilder.h"
#include "SceneView.h"

class FGaussianSplatSceneProxy;
class FGaussianSplatGPUResources;
struct FGaussianGlobalAccumulator;

/**
 * Handles the rendering of Gaussian Splats
 * Orchestrates compute passes for view calculation, sorting, and final rendering
 */
class NANOGS_API FGaussianSplatRenderer
{
public:
	FGaussianSplatRenderer();
	~FGaussianSplatRenderer();

	/**
	 * Dispatch the view data calculation compute shader
	 * @param bUseLODRendering If true, skip splats covered by parent LOD clusters
	 */
	static void DispatchCalcViewData(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale,
		bool bUseLODRendering = false
	);

	/**
	 * Dispatch the distance calculation compute shader
	 */
	static void DispatchCalcDistances(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources,
		int32 SplatCount
	);

	/**
	 * Dispatch radix sort for back-to-front ordering
	 */
	static void DispatchRadixSort(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources,
		int32 SplatCount
	);

	/**
	 * Dispatch radix sort with indirect dispatch (GPU-driven sort count)
	 * Uses SortIndirectArgsBuffer for CountCS/ScatterCS dispatch dimensions
	 * and SortParamsBuffer for Count/NumTiles read by shaders.
	 * Only sorts the visible splats after compaction.
	 */
	static void DispatchRadixSortIndirect(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources
	);

	/**
	 * Draw the Gaussian splats
	 */
	static void DrawSplats(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		int32 SplatCount
	);

	/**
	 * Dispatch cluster culling compute shader (Nanite-style optimization)
	 * Tests cluster bounding spheres against view frustum
	 * @param ErrorThreshold Screen-space error threshold in pixels for LOD selection
	 * @param bUseLODRendering If true, track unique LOD clusters for later rendering
	 * @return Number of visible clusters (for statistics)
	 */
	static int32 DispatchClusterCulling(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		float ErrorThreshold,
		bool bUseLODRendering = false
	);

	// NOTE: DispatchCalcLODViewDataGPUDriven and DispatchUpdateDrawArgs have been removed
	// in the unified approach. LOD splats are now processed by DispatchCalcViewData.

	//----------------------------------------------------------------------
	// Splat Compaction (GPU-driven work reduction)
	//----------------------------------------------------------------------

	/**
	 * Dispatch the splat compaction compute shader
	 * Builds a compact list of visible splat indices using atomics
	 */
	static void DispatchCompactSplats(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources,
		int32 TotalSplatCount,
		int32 OriginalSplatCount,
		bool bUseLODRendering
	);

	/**
	 * Dispatch the prepare indirect args compute shader
	 * Prepares indirect dispatch and draw arguments from visible splat count
	 */
	static void DispatchPrepareIndirectArgs(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources
	);

	/**
	 * Dispatch CalcViewData with compaction (indirect dispatch)
	 * Only processes visible splats from compacted list
	 */
	static void DispatchCalcViewDataCompacted(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 OriginalSplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale
	);

	/**
	 * Dispatch CalcDistances with indirect dispatch
	 * Only processes visible splats
	 */
	static void DispatchCalcDistancesIndirect(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources
	);

	//----------------------------------------------------------------------
	// Global Accumulator dispatch (one-draw-call path)
	//----------------------------------------------------------------------

	/**
	 * Dispatch CalcViewData writing into GlobalAccumulator->GlobalViewDataBuffer
	 * at GlobalBaseOffset, instead of the per-proxy ViewDataBuffer.
	 */
	static void DispatchCalcViewDataGlobal(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale,
		bool bUseLODRendering,
		uint32 GlobalBaseOffset,
		FGaussianGlobalAccumulator* GlobalAccumulator
	);

	/**
	 * Dispatch CalcDistances over the full global ViewDataBuffer.
	 * Must be called after all Phase-1 CalcViewData dispatches.
	 */
	static void DispatchCalcDistancesGlobal(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		int32 TotalSplatCount
	);

	/**
	 * Dispatch radix sort over the full global distance/key buffers.
	 */
	static void DispatchRadixSortGlobal(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		int32 TotalSplatCount
	);

	/**
	 * Draw all splats using global sorted keys and ViewData.
	 * Borrows the IndexBuffer from the first valid proxy.
	 */
	static void DrawSplatsGlobal(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		FBufferRHIRef IndexBuffer,
		int32 TotalSplatCount,
		int32 DebugMode
	);

	//----------------------------------------------------------------------
	// Global Accumulator + Nanite Compaction dispatch (one-draw-call path)
	// Phase sequence: GatherVisibleCount × N → PrefixSumVisibleCounts ×1 →
	//   CalcViewDataCompactedGlobal × N → CalcDistancesGlobalIndirect ×1 →
	//   RadixSortGlobalIndirect ×1 → DrawSplatsGlobalIndirect ×1
	//----------------------------------------------------------------------

	/**
	 * Copy GPUResources->VisibleSplatCountBuffer[0] into
	 * GlobalAccumulator->GlobalVisibleCountArrayBuffer[ProxyIndex].
	 * Dispatch: (1,1,1) per proxy, after DispatchCompactSplats.
	 */
	static void DispatchGatherVisibleCount(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		int32 ProxyIndex
	);

	/**
	 * Compute exclusive prefix sums over GlobalVisibleCountArray and write
	 * all indirect dispatch/draw args for Phase-3 passes.
	 * Dispatch: (1,1,1) once, after all GatherVisibleCount dispatches.
	 */
	static void DispatchPrefixSumVisibleCounts(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		int32 ProxyCount,
		uint32 MaxRenderBudget
	);

	/**
	 * CalcViewData for proxy ProxyIndex, writing into GlobalViewDataBuffer
	 * at offset GlobalBaseOffsetsBuffer[ProxyIndex].
	 * Uses IndirectDispatchArgsBuffer from GPUResources (set by PrepareIndirectArgs).
	 */
	static void DispatchCalcViewDataCompactedGlobal(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 OriginalSplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale,
		int32 ProxyIndex,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		uint32 MaxRenderBudget
	);

	/**
	 * CalcDistances over the global ViewDataBuffer using indirect dispatch
	 * (count from GlobalCalcDistIndirectArgsBuffer written by PrefixSumCS).
	 */
	static void DispatchCalcDistancesGlobalIndirect(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator
	);

	/**
	 * Radix sort over the global distance/key buffers using indirect dispatch
	 * (count/numTiles from GlobalSortParamsBuffer written by PrefixSumCS).
	 */
	static void DispatchRadixSortGlobalIndirect(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator
	);

	/**
	 * Draw all visible splats using GlobalDrawIndirectArgsBuffer
	 * (instance count written by PrefixSumCS, not a CPU constant).
	 */
	static void DrawSplatsGlobalIndirect(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		FBufferRHIRef IndexBuffer,
		int32 DebugMode
	);

	//----------------------------------------------------------------------
	// OIT渲染路径 (Mobile-GS加权平均OIT)
	//----------------------------------------------------------------------

	/**
	 * 分发MLP前向推理CS
	 * 为每个splat计算phi和opacity，写入PhiOpacityBuffer
	 *
	 * bUseCulling: 启用 pre-MLP cluster 可见性裁剪.
	 *   - false (默认): 对所有 SplatCount 个点做完整 MLP 推理 (原行为).
	 *   - true:  使用 ClusterCullingCS 已经产出的 ClusterVisibilityBitmap /
	 *            LODClusterSelectedBitmap / SelectedClusterBuffer 来门控每个 splat
	 *            的 MLP 工作; 不可见的 splat 直接写入 phi=0/opacity=0, 并省掉整个
	 *            线程组的 Layer1~3 协作矩阵乘 (当整组都不可见时).
	 *            需要 GPUResources->bHasClusterData 为 true, 否则 shader 内部会
	 *            fallback 成 "全量推理" (UseClusterCulling=0).
	 */
	static void DispatchMLPForward(
		FRHICommandListImmediate& RHICmdList,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		const FVector3f& WorldCameraPosition,
		int32 SplatCount,
		float OpacityScale,
		FBufferRHIRef PhiOpacityBuffer,
		bool bEnableMLPWeights = true,
		bool bUseCulling = false
	);

	/**
	 * OIT变体的CalcViewData: 额外读取PhiOpacityBuffer
	 * 计算OIT权重并写入ViewData.OITWeight
	 */
	static void DispatchCalcViewDataOIT(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale,
		bool bUseLODRendering,
		uint32 GlobalBaseOffset,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		FShaderResourceViewRHIRef PhiOpacityBufferSRV
	);

	/**
	 * OIT变体的CalcViewData (紧凑模式)
	 */
	static void DispatchCalcViewDataOITCompactedGlobal(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianSplatGPUResources* GPUResources,
		const FMatrix& LocalToWorld,
		int32 SplatCount,
		int32 OriginalSplatCount,
		int32 SHOrder,
		float OpacityScale,
		float SplatScale,
		int32 ProxyIndex,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		uint32 MaxRenderBudget,
		FShaderResourceViewRHIRef PhiOpacityBufferSRV
	);

	/**
	 * OIT变体的DrawSplats: 使用加法混合输出加权累积量
	 * 写入 RT0(ColorWeight) + RT1(LogTransmit) + RT2(Velocity)
	 */
	static void DrawSplatsOIT(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		FBufferRHIRef IndexBuffer,
		int32 TotalSplatCount,
		int32 DebugMode
	);

	/**
	 * OIT变体的DrawSplats (Indirect模式)
	 */
	static void DrawSplatsOITGlobalIndirect(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		FBufferRHIRef IndexBuffer,
		int32 DebugMode
	);

	/**
	 * OIT合成: 解算加权平均并合成到SceneColor
	 */
	static void CompositeOITToSceneColor(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FTextureRHIRef OITColorWeightTexture,
		FTextureRHIRef OITLogTransmitTexture
	);

	/**
	 * Composite the intermediate sRGB-blended splat texture onto SceneColor.
	 * Converts from sRGB to linear color space during compositing.
	 * This ensures gaussian splat alpha blending happens in sRGB space
	 * (matching 3DGS training) while still integrating with UE's linear pipeline.
	 */
	static void CompositeToSceneColor(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FTextureRHIRef IntermediateTexture
	);

	/**
	 * Debug overlay that samples the CHW float3 CUDA rasterizer output from
	 * the given raw ByteAddressBuffer SRV and writes it into a sub-rect of the
	 * currently-bound render target (SceneColor). Pixels outside DestRect are
	 * left untouched.
	 *
	 * Must be called inside an RDG pass lambda that has SceneColor bound.
	 *
	 * @param CudaColorSRV       SRV over the raw FBuffer that CUDA wrote into
	 * @param CudaExtent         (Width, Height) of the CUDA render
	 * @param DestRect           (MinX, MinY, MaxX, MaxY) in viewport-space pixels
	 * @param Exposure           Linear multiplier applied to the sampled color
	 */
	static void BlitCudaOutColorDebug(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FShaderResourceViewRHIRef CudaColorSRV,
		FUintVector2 CudaExtent,
		FVector4f DestRect,
		float Exposure
	);

private:
	/** Calculate next power of 2 */
	static uint32 NextPowerOfTwo(uint32 Value);

	/**
	 * Extract frustum planes from view-projection matrix
	 * Planes are in world space, normalized with normal pointing inward
	 * Order: Left, Right, Bottom, Top, Near, Far
	 */
	static void ExtractFrustumPlanes(const FMatrix& ViewProjection, FVector4f OutPlanes[6]);
};
