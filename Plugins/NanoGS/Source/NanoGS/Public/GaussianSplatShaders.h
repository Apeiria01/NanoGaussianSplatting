// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GlobalShader.h"
#include "ShaderParameterStruct.h"
#include "RenderGraphResources.h"
#include "RenderGraphBuilder.h"
#include "GaussianDataTypes.h"
#include "GaussianClusterTypes.h"

/**
 * Compute shader for calculating view-dependent data for each Gaussian splat
 * This runs per-frame to transform splats to screen space and evaluate spherical harmonics
 */
class FGaussianSplatCalcViewDataCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatCalcViewDataCS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatCalcViewDataCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(ByteAddressBuffer, PositionBuffer)      // 12 bytes/splat (float3)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, RotationBuffer)      // 16 bytes/splat (float4 quaternion)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, ScaleBuffer)         // 12 bytes/splat (float3)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, ColorOpacityBuffer)  // 4 bytes/splat (packed RGBA8)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, SHBuffer)           // SH data (currently unused)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<FGaussianSplatViewData>, ViewDataBuffer)
		// Cluster visibility integration (UNIFIED APPROACH)
		// SplatClusterIndexBuffer maps ALL splats to their cluster (original->leaf, LOD->parent)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SplatClusterIndexBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, LODClusterSelectedBitmap)  // For LOD splat visibility
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SelectedClusterBuffer)
		SHADER_PARAMETER(uint32, UseClusterCulling)
		SHADER_PARAMETER(uint32, UseLODRendering)  // 1 = enable LOD rendering (unified approach)
		SHADER_PARAMETER(uint32, OriginalSplatCount)  // Count of original splats (LOD splats start after)
		// Compaction mode: process only visible splats from CompactedSplatIndices
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, CompactedSplatIndices)
		SHADER_PARAMETER(uint32, UseCompaction)  // 1 = use compacted indices
		SHADER_PARAMETER(uint32, VisibleSplatCount)  // Number of visible splats (when using compaction)
		// Transform matrices
		SHADER_PARAMETER(FMatrix44f, LocalToWorld)
		SHADER_PARAMETER(FMatrix44f, WorldToPLY)  // Transforms world direction to PLY/SH space for SH evaluation
		SHADER_PARAMETER(FMatrix44f, WorldToClip)
		SHADER_PARAMETER(FMatrix44f, WorldToView)
		SHADER_PARAMETER(FVector3f, PreViewTranslation)  // For velocity calculation (TranslatedWorld = World + PreViewTranslation)
		SHADER_PARAMETER(FVector3f, CameraPosition)
		SHADER_PARAMETER(FVector2f, ScreenSize)
		SHADER_PARAMETER(FVector2f, FocalLength)
		SHADER_PARAMETER(uint32, SplatCount)
		SHADER_PARAMETER(uint32, SHOrder)
		SHADER_PARAMETER(uint32, NumSHCoeffs)      // Coefficients to evaluate (based on SHOrder: 4, 9, or 16)
		SHADER_PARAMETER(uint32, SHBufferCoeffs)   // Coefficients actually stored per splat in SHBuffer (based on asset SHBands)
		SHADER_PARAMETER(uint32, UseSHRendering)   // 1 = use view-dependent SH evaluation
		SHADER_PARAMETER(float, OpacityScale)
		SHADER_PARAMETER(float, SplatScale)
		SHADER_PARAMETER(uint32, GlobalBaseOffset)  // Offset into global ViewDataBuffer (non-compaction global path)
		// Global compaction path: GPU prefix-sum offsets
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, GlobalBaseOffsetsBuffer)  // prefix sums per proxy
		SHADER_PARAMETER(uint32, ProxyIndex)               // which proxy index this dispatch belongs to
		SHADER_PARAMETER(uint32, UseGlobalCompactionPath)  // 1 = read base from GlobalBaseOffsetsBuffer[ProxyIndex]
		SHADER_PARAMETER(uint32, MaxRenderBudget)          // Budget cap: skip writes at indices >= this value (0 = no cap)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 256);
	}
};

// NOTE: FGaussianSplatCalcLODViewDataGPUDrivenCS and FUpdateDrawArgsCS have been removed
// in the unified approach. LOD splats are now processed by CalcViewData using the same
// buffers as original splats. See CalcViewData.usf for unified LOD handling.

//----------------------------------------------------------------------
// Splat Compaction Shaders (for GPU-driven work reduction)
//----------------------------------------------------------------------

/**
 * Compute shader that builds a compact list of visible splat indices
 * This enables subsequent passes to only process visible splats
 */
class FCompactSplatsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FCompactSplatsCS);
	SHADER_USE_PARAMETER_STRUCT(FCompactSplatsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// Input: Cluster visibility data
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SplatClusterIndexBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, LODClusterSelectedBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SelectedClusterBuffer)
		// Output: Compact list of visible splat indices
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, CompactedSplatIndices)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, VisibleSplatCount)
		// Parameters
		SHADER_PARAMETER(uint32, TotalSplatCount)
		SHADER_PARAMETER(uint32, OriginalSplatCount)
		SHADER_PARAMETER(uint32, UseLODRendering)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 256);
	}
};

/**
 * Compute shader that prepares indirect dispatch and draw arguments
 * from the visible splat count
 */
class FPrepareIndirectArgsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FPrepareIndirectArgsCS);
	SHADER_USE_PARAMETER_STRUCT(FPrepareIndirectArgsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// Input: Visible splat count
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, VisibleSplatCount)
		// Output: Indirect dispatch args (uint3: numGroupsX, numGroupsY, numGroupsZ)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, IndirectDispatchArgs)
		// Output: Indirect draw args (5 uints for DrawIndexedIndirect)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, IndirectDrawArgs)
		// Output: Sort indirect dispatch args (uint3: NumTiles, 1, 1)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SortIndirectArgs)
		// Output: Sort parameters (uint2: SortCount, NumTiles)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SortParamsBuffer)
		// Thread group size for compute shaders
		SHADER_PARAMETER(uint32, ComputeThreadGroupSize)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * Compute shader for calculating sort distances (depth) for each splat
 */
class FGaussianSplatCalcDistancesCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatCalcDistancesCS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatCalcDistancesCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<FGaussianSplatViewData>, ViewDataBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DistanceBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, KeyBuffer)
		SHADER_PARAMETER(uint32, SplatCount)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 256);
	}
};

/**
 * Vertex shader for rendering Gaussian splats as quads
 */
class FGaussianSplatVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatVS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatVS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<FGaussianSplatViewData>, ViewDataBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SortKeysBuffer)
		SHADER_PARAMETER(uint32, SplatCount)
		SHADER_PARAMETER(uint32, DebugMode)  // 0=off, 1=cluster colors (Nanite-style debug)
		SHADER_PARAMETER(uint32, EnableNanite)  // 1=Nanite enabled, 0=disabled (hidden in debug mode)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * Pixel shader for rendering Gaussian splats with gaussian falloff
 */
class FGaussianSplatPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatPS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// Velocity calculation: transform translated world position to previous clip space
		// NOTE: Uses UN-JITTERED matrix to ensure stable velocity when camera is static
		SHADER_PARAMETER(FMatrix44f, PrevTranslatedWorldToClip)
		// PreViewTranslation for converting TranslatedWorld back to World
		SHADER_PARAMETER(FVector3f, PreViewTranslation)
		// Previous frame's PreViewTranslation for correct velocity calculation
		SHADER_PARAMETER(FVector3f, PrevPreViewTranslation)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

//----------------------------------------------------------------------
// OIT (Order-Independent Transparency) 渲染路径着色器
// 基于Mobile-GS的加权平均OIT混合方案
//----------------------------------------------------------------------

/**
 * MLP前向推理占位符CS
 * 为每个高斯图元计算视角相关的phi和opacity
 * 当前输出默认值，后续可替换为实际MLP推理
 */
class FMLPForwardCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FMLPForwardCS);
	SHADER_USE_PARAMETER_STRUCT(FMLPForwardCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(ByteAddressBuffer, PositionBuffer)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, ScaleBuffer)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, RotationBuffer)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, ColorOpacityBuffer)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, SHBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<float>, MLPWeightsBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<float2>, PhiOpacityBuffer) // 输出: [phi, opacity]
		// Cluster 裁剪 SRV: 与 FGaussianSplatCalcViewDataOITCS 完全同款语义.
		// UseMLPCulling=0 或 UseClusterCulling=0 时这些 SRV 不会被采样, 可传
		// 任何有效 StructuredBuffer<uint> (比如 PositionBuffer 的 uint 视图).
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SplatClusterIndexBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, LODClusterSelectedBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SelectedClusterBuffer)
		SHADER_PARAMETER(FVector3f, CameraPosition)
		SHADER_PARAMETER(uint32, SplatCount)
		SHADER_PARAMETER(float, OpacityScale)
		SHADER_PARAMETER(uint32, MLPInputDim)   // MLP第一层输入维度 (0=使用占位符)
		SHADER_PARAMETER(uint32, SHCoeffCount)  // 每通道SH系数数 = (bands+1)^2
		// Pre-MLP 裁剪控制
		SHADER_PARAMETER(uint32, UseMLPCulling)      // 0=全量推理, 1=按 cluster 可见性跳过
		SHADER_PARAMETER(uint32, UseClusterCulling)  // 0=proxy 没 cluster 数据 (强制关 MLP 裁剪)
		SHADER_PARAMETER(uint32, UseLODRendering)    // 0=不渲染 LOD splat
		SHADER_PARAMETER(uint32, OriginalSplatCount) // 原始 splat 数 (LOD splat 起始索引)
	END_SHADER_PARAMETER_STRUCT()

	static constexpr uint32 BatchSize = 16;

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 256);
		OutEnvironment.SetDefine(TEXT("MLP_BATCH_SIZE"), BatchSize);
	}
};

/**
 * OIT变体的CalcViewData CS
 * 在原版基础上额外读取PhiOpacityBuffer，计算OIT权重
 */
class FGaussianSplatCalcViewDataOITCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatCalcViewDataOITCS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatCalcViewDataOITCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		// 与CalcViewDataCS相同的参数
		SHADER_PARAMETER_SRV(ByteAddressBuffer, PositionBuffer)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, RotationBuffer)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, ScaleBuffer)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, ColorOpacityBuffer)
		SHADER_PARAMETER_SRV(ByteAddressBuffer, SHBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<FGaussianSplatViewData>, ViewDataBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SplatClusterIndexBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, LODClusterSelectedBitmap)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SelectedClusterBuffer)
		SHADER_PARAMETER(uint32, UseClusterCulling)
		SHADER_PARAMETER(uint32, UseLODRendering)
		SHADER_PARAMETER(uint32, OriginalSplatCount)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, CompactedSplatIndices)
		SHADER_PARAMETER(uint32, UseCompaction)
		SHADER_PARAMETER(uint32, VisibleSplatCount)
		SHADER_PARAMETER(FMatrix44f, LocalToWorld)
		SHADER_PARAMETER(FMatrix44f, WorldToPLY)
		SHADER_PARAMETER(FMatrix44f, WorldToClip)
		SHADER_PARAMETER(FMatrix44f, WorldToView)
		SHADER_PARAMETER(FVector3f, PreViewTranslation)
		SHADER_PARAMETER(FVector3f, CameraPosition)
		SHADER_PARAMETER(FVector2f, ScreenSize)
		SHADER_PARAMETER(FVector2f, FocalLength)
		SHADER_PARAMETER(uint32, SplatCount)
		SHADER_PARAMETER(uint32, SHOrder)
		SHADER_PARAMETER(uint32, NumSHCoeffs)
		SHADER_PARAMETER(uint32, SHBufferCoeffs)
		SHADER_PARAMETER(uint32, UseSHRendering)
		SHADER_PARAMETER(float, OpacityScale)
		SHADER_PARAMETER(float, SplatScale)
		SHADER_PARAMETER(uint32, GlobalBaseOffset)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, GlobalBaseOffsetsBuffer)
		SHADER_PARAMETER(uint32, ProxyIndex)
		SHADER_PARAMETER(uint32, UseGlobalCompactionPath)
		SHADER_PARAMETER(uint32, MaxRenderBudget)
		// OIT新增: MLP输出
		SHADER_PARAMETER_SRV(StructuredBuffer<float2>, PhiOpacityBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 256);
	}
};

/**
 * OIT变体VS: 额外传递OITWeight到PS
 */
class FGaussianSplatOITVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatOITVS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatOITVS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<FGaussianSplatViewData>, ViewDataBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SortKeysBuffer) // OIT: 恒等映射
		SHADER_PARAMETER(uint32, SplatCount)
		SHADER_PARAMETER(uint32, DebugMode)
		SHADER_PARAMETER(uint32, EnableNanite)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * OIT变体PS: 输出加权累积量 (ColorWeight + LogTransmit)
 */
class FGaussianSplatOITPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatOITPS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatOITPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER(FMatrix44f, PrevTranslatedWorldToClip)
		SHADER_PARAMETER(FVector3f, PreViewTranslation)
		SHADER_PARAMETER(FVector3f, PrevPreViewTranslation)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * OIT合成PS: 读取OIT累积纹理，执行加权平均解算
 */
class FGaussianSplatCompositeOITPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatCompositeOITPS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatCompositeOITPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_TEXTURE(Texture2D, OITColorWeightTexture)
		SHADER_PARAMETER_TEXTURE(Texture2D, OITLogTransmitTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, OITSampler)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * OIT合成VS: 复用全屏三角形 (与原版CompositeVS一致)
 */
class FGaussianSplatCompositeOITVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatCompositeOITVS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatCompositeOITVS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * Vertex shader for the sRGB→linear composite pass (full-screen triangle)
 */
class FGaussianSplatCompositeVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatCompositeVS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatCompositeVS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * Pixel shader for the sRGB→linear composite pass
 * Reads the intermediate sRGB-blended splat accumulation texture,
 * converts to linear, and outputs for premultiplied alpha blending onto SceneColor.
 */
class FGaussianSplatCompositePS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGaussianSplatCompositePS);
	SHADER_USE_PARAMETER_STRUCT(FGaussianSplatCompositePS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_TEXTURE(Texture2D, IntermediateTexture)
		SHADER_PARAMETER_SAMPLER(SamplerState, IntermediateSampler)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * Vertex shader for the CUDA rasterizer debug display (full-screen triangle)
 * Used by FCudaRasterizerDebugPS to blit sibr::UECudaRasterizerBridge::forward
 * output as a sub-rect overlay on top of SceneColor.
 */
class FCudaRasterizerDebugVS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FCudaRasterizerDebugVS);
	SHADER_USE_PARAMETER_STRUCT(FCudaRasterizerDebugVS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * Pixel shader that reads the CHW float3 CUDA rasterizer output from a
 * ByteAddressBuffer and writes it into a sub-rect of the current render target.
 * Pixels outside DestRect are discarded so SceneColor underneath is preserved.
 */
class FCudaRasterizerDebugPS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FCudaRasterizerDebugPS);
	SHADER_USE_PARAMETER_STRUCT(FCudaRasterizerDebugPS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(ByteAddressBuffer, CudaColorBuffer)
		SHADER_PARAMETER(FUintVector2, CudaExtent)
		SHADER_PARAMETER(FVector4f, DestRect)       // (MinX, MinY, MaxX, MaxY) in viewport pixels
		SHADER_PARAMETER(float, Exposure)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * Radix sort - CountCS: per-tile histogram of 256 digit bins
 */
class FRadixSortCountCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FRadixSortCountCS);
	SHADER_USE_PARAMETER_STRUCT(FRadixSortCountCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, HistogramBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SrcKeys)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SortParams)
		SHADER_PARAMETER(uint32, RadixShift)
		SHADER_PARAMETER(uint32, Count)
		SHADER_PARAMETER(uint32, NumTiles)
		SHADER_PARAMETER(uint32, UseIndirectSort)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("COUNT_CS"), 1);
	}
};

/**
 * Radix sort - PrefixSumCS: exclusive prefix sum per digit across tiles
 */
class FRadixSortPrefixSumCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FRadixSortPrefixSumCS);
	SHADER_USE_PARAMETER_STRUCT(FRadixSortPrefixSumCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, HistogramBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DigitOffsetBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SortParams)
		SHADER_PARAMETER(uint32, NumTiles)
		SHADER_PARAMETER(uint32, UseIndirectSort)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("PREFIX_SUM_CS"), 1);
	}
};

/**
 * Radix sort - DigitPrefixSumCS: exclusive prefix sum across 256 digit totals
 */
class FRadixSortDigitPrefixSumCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FRadixSortDigitPrefixSumCS);
	SHADER_USE_PARAMETER_STRUCT(FRadixSortDigitPrefixSumCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DigitOffsetBuffer)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("DIGIT_PREFIX_SUM_CS"), 1);
	}
};

/**
 * Radix sort - ScatterCS: write keys+values to sorted positions
 */
class FRadixSortScatterCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FRadixSortScatterCS);
	SHADER_USE_PARAMETER_STRUCT(FRadixSortScatterCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SrcKeys)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SrcVals)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DstKeys)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DstVals)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, HistogramBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, DigitOffsetBuffer)
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, SortParams)
		SHADER_PARAMETER(uint32, RadixShift)
		SHADER_PARAMETER(uint32, Count)
		SHADER_PARAMETER(uint32, NumTiles)
		SHADER_PARAMETER(uint32, UseIndirectSort)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("SCATTER_CS"), 1);
	}
};

//----------------------------------------------------------------------
// Cluster Culling Shaders (Nanite-style optimization)
//----------------------------------------------------------------------

/**
 * Compute shader to reset the visible cluster counter, indirect draw args, and visibility bitmap
 */
class FClusterCullingResetCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FClusterCullingResetCS);
	SHADER_USE_PARAMETER_STRUCT(FClusterCullingResetCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, VisibleClusterCountBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, IndirectDrawArgsBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SelectedClusterBuffer)
		// LOD cluster tracking buffers
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterCountBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterSelectedBitmap)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODSplatTotalBuffer)
		// GPU-driven LOD rendering
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODSplatOutputCountBuffer)
		SHADER_PARAMETER(uint32, ClusterVisibilityBitmapSize)
		SHADER_PARAMETER(uint32, LeafClusterCount)
		// Debug: must match MainCS parameters (shared .usf file)
		SHADER_PARAMETER(int32, DebugForceLODLevel)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}
};

/**
 * Compute shader for frustum culling and LOD selection of clusters
 * Tests each cluster's bounding sphere against the view frustum
 * Calculates screen-space error for LOD selection
 * Outputs list of visible cluster indices with LOD flags
 */
class FClusterCullingCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FClusterCullingCS);
	SHADER_USE_PARAMETER_STRUCT(FClusterCullingCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<FGaussianGPUCluster>, ClusterBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, VisibleClusterBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, VisibleClusterCountBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, IndirectDrawArgsBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, ClusterVisibilityBitmap)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, SelectedClusterBuffer)
		// LOD cluster tracking buffers
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterCountBuffer)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODClusterSelectedBitmap)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, LODSplatTotalBuffer)
		SHADER_PARAMETER(FMatrix44f, LocalToWorld)
		SHADER_PARAMETER(FMatrix44f, WorldToClip)
		SHADER_PARAMETER(uint32, ClusterCount)
		SHADER_PARAMETER(uint32, LeafClusterCount)
		SHADER_PARAMETER_ARRAY(FVector4f, FrustumPlanes, [6])
		// LOD selection parameters
		SHADER_PARAMETER(FVector3f, CameraPosition)
		SHADER_PARAMETER(float, ScreenHeight)
		SHADER_PARAMETER(float, ErrorThreshold)
		SHADER_PARAMETER(float, LODBias)
		SHADER_PARAMETER(uint32, UseLODRendering)
		// Debug: Force specific LOD level (-1 = auto, 0 = leaf, 1+ = specific level)
		SHADER_PARAMETER(int32, DebugForceLODLevel)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("THREADGROUP_SIZE"), 64);
	}
};

//----------------------------------------------------------------------
// Global Accumulator + Compaction prefix-sum shaders
//----------------------------------------------------------------------

/**
 * GatherCS: 1 thread per proxy.
 * Copies GPUResources->VisibleSplatCountBuffer[0] into GlobalVisibleCountArray[ProxyIndex].
 */
class FGatherVisibleCountsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FGatherVisibleCountsCS);
	SHADER_USE_PARAMETER_STRUCT(FGatherVisibleCountsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>, PerProxyVisibleCount)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalVisibleCountArray)
		SHADER_PARAMETER(uint32, ProxyIndex)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("GATHER_CS"), 1);
	}
};

/**
 * PrefixSumCS: 1 thread total.
 * Computes prefix sums over GlobalVisibleCountArray and writes all indirect
 * dispatch/draw args for the global Phase-3 passes (CalcDistances, RadixSort, DrawSplats).
 */
class FPrefixSumVisibleCountsCS : public FGlobalShader
{
	DECLARE_GLOBAL_SHADER(FPrefixSumVisibleCountsCS);
	SHADER_USE_PARAMETER_STRUCT(FPrefixSumVisibleCountsCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_SRV(StructuredBuffer<uint>,   GlobalVisibleCountArray)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalBaseOffsets)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalCalcDistIndirectArgs)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalSortIndirectArgs)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalSortParams)
		SHADER_PARAMETER_UAV(RWStructuredBuffer<uint>, GlobalDrawIndirectArgs)
		SHADER_PARAMETER(uint32, ProxyCount)
		SHADER_PARAMETER(uint32, MaxRenderBudget)  // Budget cap: clamp total visible to this value (0 = no cap)
	END_SHADER_PARAMETER_STRUCT()

	static bool ShouldCompilePermutation(const FGlobalShaderPermutationParameters& Parameters)
	{
		return IsFeatureLevelSupported(Parameters.Platform, ERHIFeatureLevel::ES3_1);
	}

	static void ModifyCompilationEnvironment(const FGlobalShaderPermutationParameters& Parameters, FShaderCompilerEnvironment& OutEnvironment)
	{
		FGlobalShader::ModifyCompilationEnvironment(Parameters, OutEnvironment);
		OutEnvironment.SetDefine(TEXT("VISIBLE_PREFIX_SUM_CS"), 1);
	}
};
