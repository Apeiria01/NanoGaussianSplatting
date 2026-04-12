// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GaussianDataTypes.h"
#include "GaussianClusterTypes.h"
#include "RHI.h"
#include "RHIResources.h"
#if PLATFORM_WINDOWS
#include "UECudaRasterizerBridge.hpp"
#endif

class UGaussianSplatAsset;
class FSceneView;

/**
 * Shared render data for a Gaussian Splat asset.
 * Holds CPU-side cached data and shared GPU buffers that are created once
 * per asset and shared across all FGaussianSplatGPUResources instances
 * referencing the same asset.
 */
class FGaussianSplatRenderData
{
public:
	FGaussianSplatRenderData();
	~FGaussianSplatRenderData();

	/** Initialize CPU-side data from asset. Only runs once (guarded by bIsInitialized). */
	void Initialize(UGaussianSplatAsset* Asset);

	/** Create shared GPU buffers. Thread-safe, only runs once. Must be called on render thread. */
	void CreateGPUBuffers(FRHICommandListBase& RHICmdList);

	/** Release shared GPU buffers. */
	void ReleaseGPUBuffers();

	/** Render with CUDA rasterizer bridge using shared D3D12 buffers. Must run on render thread. */
	bool ForwardWithCudaRasterizer(FRHICommandListBase& RHICmdList, const FSceneView& SceneView, int32 Width, int32 Height);

	/** Whether CPU-side data has been initialized */
	bool IsInitialized() const { return bIsInitialized; }

	/** Whether GPU buffers have been created */
	bool AreGPUBuffersCreated() const { return bGPUBuffersCreated; }

	/** Get asset name for logging */
	const FString& GetAssetName() const { return AssetName; }

public:
	// ---- Shared GPU buffers (created once, shared across all proxies) ----

	/** SoA position buffer (float3 per splat) */
	FBufferRHIRef PositionBuffer;
	FShaderResourceViewRHIRef PositionBufferSRV;

	/** SoA rotation buffer (float4 quaternion per splat) */
	FBufferRHIRef RotationBuffer;
	FShaderResourceViewRHIRef RotationBufferSRV;

	/** SoA scale buffer (float3 per splat) */
	FBufferRHIRef ScaleBuffer;
	FShaderResourceViewRHIRef ScaleBufferSRV;

	/** SoA color/opacity buffer (packed RGBA8 per splat) */
	FBufferRHIRef ColorOpacityBuffer;
	FShaderResourceViewRHIRef ColorOpacityBufferSRV;

	/** SoA color buffer for CUDA rasterizer (float3 per splat) */
	FBufferRHIRef ColorPrecompBuffer;
	FShaderResourceViewRHIRef ColorPrecompBufferSRV;

	/** SoA opacity buffer for CUDA rasterizer (float per splat) */
	FBufferRHIRef OpacityFloatBuffer;
	FShaderResourceViewRHIRef OpacityFloatBufferSRV;

	/** Spherical harmonics buffer */
	FBufferRHIRef SHBuffer;
	FShaderResourceViewRHIRef SHBufferSRV;

	/** Chunk info buffer */
	FBufferRHIRef ChunkBuffer;
	FShaderResourceViewRHIRef ChunkBufferSRV;

	/** Index buffer for quad rendering */
	FBufferRHIRef IndexBuffer;

	/** Cluster data buffer (static, loaded from asset) */
	FBufferRHIRef ClusterBuffer;
	FShaderResourceViewRHIRef ClusterBufferSRV;

	/** Splat-to-cluster index buffer (static, loaded from asset) */
	FBufferRHIRef SplatClusterIndexBuffer;
	FShaderResourceViewRHIRef SplatClusterIndexBufferSRV;

	// ---- Metadata ----

	int32 SplatCount = 0;
	int32 SHBands = 0;
	int32 ClusterCount = 0;
	int32 LeafClusterCount = 0;
	int32 LODSplatCount = 0;
	bool bEnableNanite = false;
	bool bHasClusterData = false;
	bool bHasLODSplats = false;
	EGaussianPositionFormat PositionFormat = EGaussianPositionFormat::Float32;

private:
	// ---- CPU-side cached data (freed after GPU upload) ----
#if PLATFORM_WINDOWS
	sibr::UECudaRasterizerBridge* CudaRasterizerBridge;
#endif
	TArray<float> PositionDataSOA;
	TArray<float> RotationDataSOA;
	TArray<float> ScaleDataSOA;
	TArray<uint32> ColorOpacityDataSOA;
	TArray<float> ColorPrecompDataSOA;
	TArray<float> OpacityDataSOA;
	TArray<uint8> SHData;
	TArray<FGaussianChunkInfo> CachedChunkData;
	TArray<FGaussianGPUCluster> CachedClusterData;
	TArray<uint32> CachedSplatClusterIndices;

public:
	/** Output color buffer written by the CUDA rasterizer (CHW float layout:
	 *  3 planes of Width*Height floats each). Exposed publicly so the debug
	 *  overlay pass in NanoGS.cpp can read it via CudaOutColorBufferSRV. */
	FBufferRHIRef CudaOutColorBuffer;
	FShaderResourceViewRHIRef CudaOutColorBufferSRV;
	uint32 CudaOutColorBufferWidth = 0;
	uint32 CudaOutColorBufferHeight = 0;

private:
	FBufferRHIRef CudaBackgroundBuffer;
	uint32 CudaOutColorBufferBytes = 0;

	bool bIsInitialized = false;
	bool bGPUBuffersCreated = false;
	FString AssetName;
	FCriticalSection InitLock;
	FCriticalSection GPUInitLock;
};
