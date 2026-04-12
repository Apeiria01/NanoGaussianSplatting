// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatRenderData.h"
#include "GaussianSplatAsset.h"
#include "RHICommandList.h"
#if PLATFORM_WINDOWS
#include "CUDAModule.h"
#endif
#include "SceneView.h"

#if PLATFORM_WINDOWS
// ID3D12DynamicRHI.h transitively pulls in AgilitySDK's <d3d12.h> and <d3dx12.h>,
// which provide the real ID3D12Device / ID3D12Resource definitions. The NanoGS
// Build.cs adds D3D12RHI + DX12 third-party deps so the modern AgilitySDK headers
// resolve before the older Windows SDK ones.
#include "ID3D12DynamicRHI.h"
#endif


FGaussianSplatRenderData::FGaussianSplatRenderData()
{
	
#if PLATFORM_WINDOWS
	CudaRasterizerBridge = new sibr::UECudaRasterizerBridge();
#endif
}

FGaussianSplatRenderData::~FGaussianSplatRenderData()
{
	ReleaseGPUBuffers();
#if PLATFORM_WINDOWS
	delete CudaRasterizerBridge;
#endif
}

void FGaussianSplatRenderData::Initialize(UGaussianSplatAsset* Asset)
{
	FScopeLock Lock(&InitLock);

	if (bIsInitialized)
	{
		return;
	}

	if (!Asset || !Asset->IsValid())
	{
		return;
	}

	AssetName = Asset->GetName();
	SplatCount = Asset->GetSplatCount();
	PositionFormat = Asset->PositionFormat;

	// --- Build SoA point data buffers ---
	{
		TArray<uint8> RawPositionData;
		TArray<uint8> RawOtherData;
		TArray<uint8> RawColorTextureData;
		Asset->GetPositionData(RawPositionData);
		Asset->GetOtherData(RawOtherData);
		Asset->GetColorTextureData(RawColorTextureData);

		const int32 ColorTexWidth = Asset->ColorTextureWidth;
		const int32 ColorTexHeight = Asset->ColorTextureHeight;
		const FFloat16Color* ColorPixels = nullptr;
		const bool bHasColor = (RawColorTextureData.Num() > 0 && ColorTexWidth > 0 && ColorTexHeight > 0);
		if (bHasColor)
		{
			ColorPixels = reinterpret_cast<const FFloat16Color*>(RawColorTextureData.GetData());
		}

		PositionDataSOA.SetNumUninitialized(SplatCount * 3);
		RotationDataSOA.SetNumUninitialized(SplatCount * 4);
		ScaleDataSOA.SetNumUninitialized(SplatCount * 3);
		ColorOpacityDataSOA.SetNumUninitialized(SplatCount);
		ColorPrecompDataSOA.SetNumUninitialized(SplatCount * 3);
		OpacityDataSOA.SetNumUninitialized(SplatCount);
		const float* PosFloats = reinterpret_cast<const float*>(RawPositionData.GetData());
		const float* OtherFloats = reinterpret_cast<const float*>(RawOtherData.GetData());

		for (int32 i = 0; i < SplatCount; i++)
		{
			PositionDataSOA[i * 3 + 0] = PosFloats[i * 3 + 0];
			PositionDataSOA[i * 3 + 1] = PosFloats[i * 3 + 1];
			PositionDataSOA[i * 3 + 2] = PosFloats[i * 3 + 2];

			const float* OtherBase = OtherFloats + i * 7;
			RotationDataSOA[i * 4 + 0] = OtherBase[0];
			RotationDataSOA[i * 4 + 1] = OtherBase[1];
			RotationDataSOA[i * 4 + 2] = OtherBase[2];
			RotationDataSOA[i * 4 + 3] = OtherBase[3];

			ScaleDataSOA[i * 3 + 0] = OtherBase[4];
			ScaleDataSOA[i * 3 + 1] = OtherBase[5];
			ScaleDataSOA[i * 3 + 2] = OtherBase[6];

			float ColorR = 1.0f, ColorG = 1.0f, ColorB = 1.0f, Opacity = 1.0f;
			if (bHasColor)
			{
				int32 TexX, TexY;
				GaussianSplattingUtils::SplatIndexToTextureCoord(i, ColorTexWidth, TexX, TexY);
				if (TexY < ColorTexHeight)
				{
					const FFloat16Color& Pixel = ColorPixels[TexY * ColorTexWidth + TexX];
					ColorR = FMath::Clamp(Pixel.R.GetFloat(), 0.0f, 1.0f);
					ColorG = FMath::Clamp(Pixel.G.GetFloat(), 0.0f, 1.0f);
					ColorB = FMath::Clamp(Pixel.B.GetFloat(), 0.0f, 1.0f);
					Opacity = FMath::Clamp(Pixel.A.GetFloat(), 0.0f, 1.0f);
				}
			}

			const uint8 R = static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(ColorR * 255.0f), 0, 255));
			const uint8 G = static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(ColorG * 255.0f), 0, 255));
			const uint8 B = static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(ColorB * 255.0f), 0, 255));
			const uint8 A = static_cast<uint8>(FMath::Clamp(FMath::RoundToInt(Opacity * 255.0f), 0, 255));
			ColorOpacityDataSOA[i] = static_cast<uint32>(R)
				| (static_cast<uint32>(G) << 8)
				| (static_cast<uint32>(B) << 16)
				| (static_cast<uint32>(A) << 24);

			ColorPrecompDataSOA[i * 3 + 0] = ColorR;
			ColorPrecompDataSOA[i * 3 + 1] = ColorG;
			ColorPrecompDataSOA[i * 3 + 2] = ColorB;
			OpacityDataSOA[i] = Opacity;
		}
	}

	// Load chunk data
	CachedChunkData = Asset->ChunkData;

	// Load SH data
	if (Asset->SHBands > 0)
	{
		Asset->GetSHData(SHData);
		SHBands = Asset->SHBands;
	}

	// Load Nanite cluster data
	bEnableNanite = Asset->IsNaniteEnabled();

	if (bEnableNanite && Asset->HasClusterHierarchy())
	{
		const FGaussianClusterHierarchy& Hierarchy = Asset->GetClusterHierarchy();
		Hierarchy.ToGPUClusters(CachedClusterData);
		ClusterCount = CachedClusterData.Num();
		LeafClusterCount = Hierarchy.NumLeafClusters;
		bHasClusterData = true;

		const uint32 OriginalSplatCount = Hierarchy.TotalSplatCount;
		LODSplatCount = Hierarchy.TotalLODSplatCount;
		bHasLODSplats = (LODSplatCount > 0);

		// Build splat-to-cluster index mapping
		CachedSplatClusterIndices.SetNumZeroed(SplatCount);

		for (int32 ClusterIdx = 0; ClusterIdx < Hierarchy.Clusters.Num(); ++ClusterIdx)
		{
			const FGaussianCluster& Cluster = Hierarchy.Clusters[ClusterIdx];
			if (Cluster.IsLeaf())
			{
				for (uint32 i = 0; i < Cluster.SplatCount; ++i)
				{
					uint32 SplatIdx = Cluster.SplatStartIndex + i;
					if (SplatIdx < OriginalSplatCount)
					{
						CachedSplatClusterIndices[SplatIdx] = ClusterIdx;
					}
				}
			}
		}

		for (int32 ClusterIdx = 0; ClusterIdx < Hierarchy.Clusters.Num(); ++ClusterIdx)
		{
			const FGaussianCluster& Cluster = Hierarchy.Clusters[ClusterIdx];
			if (!Cluster.IsLeaf() && Cluster.LODSplatCount > 0)
			{
				for (uint32 i = 0; i < Cluster.LODSplatCount; ++i)
				{
					uint32 SplatIdx = Cluster.LODSplatStartIndex + i;
					if (SplatIdx < static_cast<uint32>(SplatCount))
					{
						CachedSplatClusterIndices[SplatIdx] = ClusterIdx;
					}
				}
			}
		}
	}
	else
	{
		bHasClusterData = false;
		ClusterCount = 0;
		LeafClusterCount = 0;
		bHasLODSplats = false;
		LODSplatCount = 0;
	}

	bIsInitialized = true;

	UE_LOG(LogTemp, Verbose, TEXT("GaussianSplatRenderData: Created shared CPU data for asset '%s' (%d splats)"),
		*AssetName, SplatCount);
}

void FGaussianSplatRenderData::CreateGPUBuffers(FRHICommandListBase& RHICmdList)
{
	FScopeLock Lock(&GPUInitLock);

	if (bGPUBuffersCreated)
	{
		return;
	}

	int32 SharedBufferCount = 0;

	auto CreateRawByteAddressBuffer = [&RHICmdList, &SharedBufferCount](
		const TCHAR* BufferName,
		const void* SourceData,
		uint32 BufferSize,
		FBufferRHIRef& OutBuffer,
		FShaderResourceViewRHIRef& OutSRV)
	{
		if (BufferSize == 0 || SourceData == nullptr)
		{
			return;
		}

		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			BufferName,
			BufferSize,
			0,
			BUF_Static | BUF_ShaderResource | BUF_ByteAddressBuffer | BUF_Shared)
			.SetInitialState(ERHIAccess::SRVMask);
		OutBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(OutBuffer, 0, BufferSize, RLM_WriteOnly);
		FMemory::Memcpy(Data, SourceData, BufferSize);
		RHICmdList.UnlockBuffer(OutBuffer);

		OutSRV = RHICmdList.CreateShaderResourceView(
			OutBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Raw));
		SharedBufferCount++;
	};

	CreateRawByteAddressBuffer(
		TEXT("GaussianPositionBuffer"),
		PositionDataSOA.GetData(),
		static_cast<uint32>(PositionDataSOA.Num() * sizeof(float)),
		PositionBuffer,
		PositionBufferSRV);

	CreateRawByteAddressBuffer(
		TEXT("GaussianRotationBuffer"),
		RotationDataSOA.GetData(),
		static_cast<uint32>(RotationDataSOA.Num() * sizeof(float)),
		RotationBuffer,
		RotationBufferSRV);

	CreateRawByteAddressBuffer(
		TEXT("GaussianScaleBuffer"),
		ScaleDataSOA.GetData(),
		static_cast<uint32>(ScaleDataSOA.Num() * sizeof(float)),
		ScaleBuffer,
		ScaleBufferSRV);

	CreateRawByteAddressBuffer(
		TEXT("GaussianColorOpacityBuffer"),
		ColorOpacityDataSOA.GetData(),
		static_cast<uint32>(ColorOpacityDataSOA.Num() * sizeof(uint32)),
		ColorOpacityBuffer,
		ColorOpacityBufferSRV);

	CreateRawByteAddressBuffer(
		TEXT("GaussianColorPrecompBuffer"),
		ColorPrecompDataSOA.GetData(),
		static_cast<uint32>(ColorPrecompDataSOA.Num() * sizeof(float)),
		ColorPrecompBuffer,
		ColorPrecompBufferSRV);

	CreateRawByteAddressBuffer(
		TEXT("GaussianOpacityFloatBuffer"),
		OpacityDataSOA.GetData(),
		static_cast<uint32>(OpacityDataSOA.Num() * sizeof(float)),
		OpacityFloatBuffer,
		OpacityFloatBufferSRV);

	// --- SH buffer (always create at least a dummy for shader binding) ---
	{
		uint32 SHDataSize = SHData.Num();
		if (SHDataSize == 0)
		{
			SHDataSize = 16;
		}

		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSHBuffer"),
			SHDataSize,
			0,
			BUF_Static | BUF_ShaderResource | BUF_ByteAddressBuffer)
			.SetInitialState(ERHIAccess::SRVMask);
		SHBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(SHBuffer, 0, SHDataSize, RLM_WriteOnly);
		if (SHData.Num() > 0)
		{
			FMemory::Memcpy(Data, SHData.GetData(), SHData.Num());
		}
		else
		{
			FMemory::Memzero(Data, SHDataSize);
		}
		RHICmdList.UnlockBuffer(SHBuffer);

		SHBufferSRV = RHICmdList.CreateShaderResourceView(
			SHBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Raw));
		SharedBufferCount++;
	}

	// --- Chunk buffer (always create at least a dummy for shader binding) ---
	{
		uint32 ChunkCount = CachedChunkData.Num();
		if (ChunkCount == 0)
		{
			ChunkCount = 1;
		}

		const uint32 ChunkSize = ChunkCount * sizeof(FGaussianChunkInfo);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianChunkBuffer"),
			ChunkSize,
			sizeof(FGaussianChunkInfo),
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::SRVMask);
		ChunkBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(ChunkBuffer, 0, ChunkSize, RLM_WriteOnly);
		if (CachedChunkData.Num() > 0)
		{
			FMemory::Memcpy(Data, CachedChunkData.GetData(), ChunkSize);
		}
		else
		{
			FMemory::Memzero(Data, ChunkSize);
		}
		RHICmdList.UnlockBuffer(ChunkBuffer);

		ChunkBufferSRV = RHICmdList.CreateShaderResourceView(
			ChunkBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(FGaussianChunkInfo)));
		SharedBufferCount++;
	}

	// --- Index buffer (6 indices per quad) ---
	{
		TArray<uint16> Indices = { 0, 1, 2, 1, 3, 2 };

		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSplatIndexBuffer"),
			Indices.Num() * sizeof(uint16),
			sizeof(uint16),
			BUF_Static | BUF_IndexBuffer)
			.SetInitialState(ERHIAccess::VertexOrIndexBuffer);
		IndexBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(IndexBuffer, 0, Indices.Num() * sizeof(uint16), RLM_WriteOnly);
		FMemory::Memcpy(Data, Indices.GetData(), Indices.Num() * sizeof(uint16));
		RHICmdList.UnlockBuffer(IndexBuffer);
		SharedBufferCount++;
	}

	// --- Cluster buffer (static, from asset) ---
	if (bHasClusterData && CachedClusterData.Num() > 0)
	{
		const uint32 BufferSize = CachedClusterData.Num() * sizeof(FGaussianGPUCluster);
		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianClusterBuffer"),
			BufferSize,
			sizeof(FGaussianGPUCluster),
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::SRVMask);
		ClusterBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(ClusterBuffer, 0, BufferSize, RLM_WriteOnly);
		FMemory::Memcpy(Data, CachedClusterData.GetData(), BufferSize);
		RHICmdList.UnlockBuffer(ClusterBuffer);

		ClusterBufferSRV = RHICmdList.CreateShaderResourceView(
			ClusterBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(FGaussianGPUCluster)));
		SharedBufferCount++;
	}

	// --- Splat-to-cluster index buffer (static, from asset; or dummy for non-Nanite) ---
	{
		const uint32 BufferSize = CachedSplatClusterIndices.Num() > 0
			? CachedSplatClusterIndices.Num() * sizeof(uint32)
			: sizeof(uint32);  // dummy 1-element buffer for shader binding

		FRHIBufferCreateDesc Desc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianSplatClusterIndexBuffer"),
			BufferSize,
			sizeof(uint32),
			BUF_Static | BUF_ShaderResource | BUF_StructuredBuffer)
			.SetInitialState(ERHIAccess::SRVMask);
		SplatClusterIndexBuffer = RHICmdList.CreateBuffer(Desc);

		void* Data = RHICmdList.LockBuffer(SplatClusterIndexBuffer, 0, BufferSize, RLM_WriteOnly);
		if (CachedSplatClusterIndices.Num() > 0)
		{
			FMemory::Memcpy(Data, CachedSplatClusterIndices.GetData(), BufferSize);
		}
		else
		{
			FMemory::Memzero(Data, BufferSize);
		}
		RHICmdList.UnlockBuffer(SplatClusterIndexBuffer);

		SplatClusterIndexBufferSRV = RHICmdList.CreateShaderResourceView(
			SplatClusterIndexBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Structured)
				.SetStride(sizeof(uint32)));
		SharedBufferCount++;
	}

	// Free CPU-side cached data after GPU upload
	PositionDataSOA.Empty();
	RotationDataSOA.Empty();
	ScaleDataSOA.Empty();
	ColorOpacityDataSOA.Empty();
	ColorPrecompDataSOA.Empty();
	OpacityDataSOA.Empty();
	SHData.Empty();
	CachedChunkData.Empty();
	CachedClusterData.Empty();
	CachedSplatClusterIndices.Empty();

	bGPUBuffersCreated = true;

	UE_LOG(LogTemp, Verbose, TEXT("GaussianSplatRenderData: Created %d shared GPU buffers for asset '%s'"),
		SharedBufferCount, *AssetName);
}

void FGaussianSplatRenderData::ReleaseGPUBuffers()
{
	PositionBuffer.SafeRelease();
	PositionBufferSRV.SafeRelease();
	RotationBuffer.SafeRelease();
	RotationBufferSRV.SafeRelease();
	ScaleBuffer.SafeRelease();
	ScaleBufferSRV.SafeRelease();
	ColorOpacityBuffer.SafeRelease();
	ColorOpacityBufferSRV.SafeRelease();
	ColorPrecompBuffer.SafeRelease();
	ColorPrecompBufferSRV.SafeRelease();
	OpacityFloatBuffer.SafeRelease();
	OpacityFloatBufferSRV.SafeRelease();
	SHBuffer.SafeRelease();
	SHBufferSRV.SafeRelease();
	ChunkBuffer.SafeRelease();
	ChunkBufferSRV.SafeRelease();
	IndexBuffer.SafeRelease();
	ClusterBuffer.SafeRelease();
	ClusterBufferSRV.SafeRelease();
	SplatClusterIndexBuffer.SafeRelease();
	SplatClusterIndexBufferSRV.SafeRelease();
#if PLATFORM_WINDOWS
	CudaBackgroundBuffer.SafeRelease();
	CudaOutColorBufferSRV.SafeRelease();
	CudaOutColorBuffer.SafeRelease();
	CudaOutColorBufferBytes = 0;
	CudaOutColorBufferWidth = 0;
	CudaOutColorBufferHeight = 0;
	if (CudaRasterizerBridge)
	{
		CudaRasterizerBridge->clearExternalMappings();
	}
#endif
	bGPUBuffersCreated = false;
}

bool FGaussianSplatRenderData::ForwardWithCudaRasterizer(FRHICommandListBase& RHICmdList, const FSceneView& SceneView, int32 Width, int32 Height)
{
#if !PLATFORM_WINDOWS
	return false;
#else
	if (!CudaRasterizerBridge || Width <= 0 || Height <= 0 || SplatCount <= 0)
	{
		return false;
	}

	if (!PositionBuffer.IsValid() || !ColorPrecompBuffer.IsValid() || !OpacityFloatBuffer.IsValid() || !ScaleBuffer.IsValid() || !RotationBuffer.IsValid())
	{
		return false;
	}

	FCUDAModule* CudaModule = FModuleManager::GetModulePtr<FCUDAModule>("CUDA");
	if (!CudaModule || !CudaModule->IsAvailable())
	{
		return false;
	}

	if (!IsRHID3D12())
	{
		return false;
	}

	ID3D12DynamicRHI* D3D12RHI = GetID3D12DynamicRHI();
	if (!D3D12RHI)
	{
		return false;
	}

	const int32 DeviceIndex = static_cast<int32>(CudaModule->GetCudaDeviceIndex());
	CUcontext CudaContext = CudaModule->GetCudaContextForDevice(DeviceIndex);
	if (!CudaContext)
	{
		return false;
	}

	if (FCUDAModule::CUDA().cuCtxPushCurrent(CudaContext) != CUDA_SUCCESS)
	{
		return false;
	}

	auto PopCudaContext = []()
	{
		CUcontext PrevContext = nullptr;
		FCUDAModule::CUDA().cuCtxPopCurrent(&PrevContext);
	};

	if (!CudaBackgroundBuffer.IsValid())
	{
		const uint32 BackgroundBytes = sizeof(float) * 3;
		FRHIBufferCreateDesc BackgroundDesc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianCudaBridgeBackground"),
			BackgroundBytes,
			0,
			BUF_Static | BUF_ShaderResource | BUF_ByteAddressBuffer | BUF_Shared)
			.SetInitialState(ERHIAccess::SRVMask);
		CudaBackgroundBuffer = RHICmdList.CreateBuffer(BackgroundDesc);

		const float BackgroundData[3] = { 0.0f, 0.0f, 0.0f };
		void* Locked = RHICmdList.LockBuffer(CudaBackgroundBuffer, 0, BackgroundBytes, RLM_WriteOnly);
		FMemory::Memcpy(Locked, BackgroundData, BackgroundBytes);
		RHICmdList.UnlockBuffer(CudaBackgroundBuffer);
	}

	const uint64 OutColorBytes64 = static_cast<uint64>(Width) * static_cast<uint64>(Height) * static_cast<uint64>(3 * sizeof(float));
	if (OutColorBytes64 > static_cast<uint64>(MAX_uint32))
	{
		PopCudaContext();
		return false;
	}
	const uint32 OutColorBytes = static_cast<uint32>(OutColorBytes64);

	if (!CudaOutColorBuffer.IsValid() || CudaOutColorBufferBytes != OutColorBytes)
	{
		CudaRasterizerBridge->clearExternalMappings();
		CudaOutColorBufferSRV.SafeRelease();
		CudaOutColorBuffer.SafeRelease();

		FRHIBufferCreateDesc OutColorDesc = FRHIBufferCreateDesc::Create(
			TEXT("GaussianCudaBridgeOutColor"),
			OutColorBytes,
			0,
			BUF_Static | BUF_ShaderResource | BUF_ByteAddressBuffer | BUF_Shared)
			.SetInitialState(ERHIAccess::SRVMask);
		CudaOutColorBuffer = RHICmdList.CreateBuffer(OutColorDesc);
		CudaOutColorBufferBytes = OutColorBytes;
		CudaOutColorBufferWidth  = static_cast<uint32>(Width);
		CudaOutColorBufferHeight = static_cast<uint32>(Height);

		// SRV that the debug display pass (FCudaRasterizerDebugPS) samples.
		CudaOutColorBufferSRV = RHICmdList.CreateShaderResourceView(
			CudaOutColorBuffer, FRHIViewDesc::CreateBufferSRV()
				.SetType(FRHIViewDesc::EBufferType::Raw));
	}

	ID3D12Resource* BackgroundResource = D3D12RHI->RHIGetResource(CudaBackgroundBuffer.GetReference());
	ID3D12Resource* MeansResource = D3D12RHI->RHIGetResource(PositionBuffer.GetReference());
	ID3D12Resource* ColorsResource = D3D12RHI->RHIGetResource(ColorPrecompBuffer.GetReference());
	ID3D12Resource* OpacityResource = D3D12RHI->RHIGetResource(OpacityFloatBuffer.GetReference());
	ID3D12Resource* ScaleResource = D3D12RHI->RHIGetResource(ScaleBuffer.GetReference());
	ID3D12Resource* RotationResource = D3D12RHI->RHIGetResource(RotationBuffer.GetReference());
	ID3D12Resource* OutColorResource = D3D12RHI->RHIGetResource(CudaOutColorBuffer.GetReference());

	uint32 ResourceDeviceIndex = D3D12RHI->RHIGetResourceDeviceIndex(PositionBuffer.GetReference());
	ID3D12Device* NativeD3D12Device = D3D12RHI->RHIGetDevice(ResourceDeviceIndex);

	if (!BackgroundResource || !MeansResource || !ColorsResource || !OpacityResource || !ScaleResource || !RotationResource || !OutColorResource)
	{
		PopCudaContext();
		return false;
	}
	if (!NativeD3D12Device)
	{
		PopCudaContext();
		return false;
	}

	sibr::UECudaRasterizerBridge::ForwardParams ForwardParams;
	ForwardParams.pointCount = SplatCount;
	ForwardParams.width = Width;
	ForwardParams.height = Height;
	ForwardParams.shDegree = 0;
	ForwardParams.shCoeffCount = 16;
	ForwardParams.scaleModifier = 1.0f;
	ForwardParams.prefiltered = false;
	ForwardParams.antialiasing = false;
	ForwardParams.shResidualGain = 1.0f;
	ForwardParams.shResidualOnly = false;

	const FMatrix ViewMatrix = SceneView.ViewMatrices.GetViewMatrix();
	const FMatrix ProjMatrix = SceneView.ViewMatrices.GetProjectionNoAAMatrix();
	const FVector ViewOrigin = SceneView.ViewMatrices.GetViewOrigin();

	sibr::UECudaRasterizerBridge::CameraParams CameraParams;
	for (int32 Row = 0; Row < 4; ++Row)
	{
		for (int32 Col = 0; Col < 4; ++Col)
		{
			CameraParams.view[Row * 4 + Col] = static_cast<float>(ViewMatrix.M[Row][Col]);
			CameraParams.proj[Row * 4 + Col] = static_cast<float>(ProjMatrix.M[Row][Col]);
		}
	}
	CameraParams.camPos[0] = static_cast<float>(ViewOrigin.X);
	CameraParams.camPos[1] = static_cast<float>(ViewOrigin.Y);
	CameraParams.camPos[2] = static_cast<float>(ViewOrigin.Z);
	CameraParams.tanFovX = (ProjMatrix.M[0][0] != 0.0f) ? FMath::Abs(static_cast<float>(1.0 / ProjMatrix.M[0][0])) : 1.0f;
	CameraParams.tanFovY = (ProjMatrix.M[1][1] != 0.0f) ? FMath::Abs(static_cast<float>(1.0 / ProjMatrix.M[1][1])) : 1.0f;

	sibr::UECudaRasterizerBridge::ExternalBufferDesc BackgroundDesc;
	BackgroundDesc.resource = BackgroundResource;
	BackgroundDesc.bytes = sizeof(float) * 3;

	sibr::UECudaRasterizerBridge::ExternalBufferDesc MeansDesc;
	MeansDesc.resource = MeansResource;
	MeansDesc.bytes = static_cast<size_t>(SplatCount) * sizeof(float) * 3;

	sibr::UECudaRasterizerBridge::ExternalBufferDesc ColorsDesc;
	ColorsDesc.resource = ColorsResource;
	ColorsDesc.bytes = static_cast<size_t>(SplatCount) * sizeof(float) * 3;

	sibr::UECudaRasterizerBridge::ExternalBufferDesc OpacityDesc;
	OpacityDesc.resource = OpacityResource;
	OpacityDesc.bytes = static_cast<size_t>(SplatCount) * sizeof(float);

	sibr::UECudaRasterizerBridge::ExternalBufferDesc ScaleDesc;
	ScaleDesc.resource = ScaleResource;
	ScaleDesc.bytes = static_cast<size_t>(SplatCount) * sizeof(float) * 3;

	sibr::UECudaRasterizerBridge::ExternalBufferDesc RotationDesc;
	RotationDesc.resource = RotationResource;
	RotationDesc.bytes = static_cast<size_t>(SplatCount) * sizeof(float) * 4;

	sibr::UECudaRasterizerBridge::ExternalBufferDesc OutColorDesc;
	OutColorDesc.resource = OutColorResource;
	OutColorDesc.bytes = OutColorBytes;

	const int ForwardResult = CudaRasterizerBridge->forward(
		NativeD3D12Device,
		ForwardParams,
		CameraParams,
		BackgroundDesc,
		MeansDesc,
		ColorsDesc,
		OpacityDesc,
		ScaleDesc,
		RotationDesc,
		OutColorDesc);

	PopCudaContext();
	return ForwardResult >= 0;
#endif
}
