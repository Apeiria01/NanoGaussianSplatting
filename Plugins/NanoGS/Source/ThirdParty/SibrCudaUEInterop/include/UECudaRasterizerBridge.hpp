#pragma once

#include <array>
#include <cstddef>
#include <cstdint>



struct ID3D12Device;
struct ID3D12Resource;

#if defined(_WIN32)
#  ifdef SIBR_STATIC_DEFINE
#    define SIBR_CUDAUEINTEROP_EXPORT
#  else
#    ifdef sibr_cudaueinterop_EXPORTS
#      define SIBR_CUDAUEINTEROP_EXPORT __declspec(dllexport)
#    else
#      define SIBR_CUDAUEINTEROP_EXPORT __declspec(dllimport)
#    endif
#  endif
#else
#  define SIBR_CUDAUEINTEROP_EXPORT
#endif

namespace sibr
{
	class SIBR_CUDAUEINTEROP_EXPORT UECudaRasterizerBridge
	{
	public:
		struct ExternalBufferDesc
		{
			// Raw UE-side ID3D12Resource pointer and its valid byte size.
			// The resource must support shared-handle import for CUDA interop.
			ID3D12Resource* resource = nullptr;
			size_t bytes = 0;
		};

		struct CameraParams
		{
			float view[16] = {};
			float proj[16] = {};
			float camPos[3] = {};
			float tanFovX = 1.0f;
			float tanFovY = 1.0f;
		};

		struct ForwardParams
		{
			int pointCount = 0;
			int width = 1;
			int height = 1;
			int shDegree = 0;
			int shCoeffCount = 16;
			float scaleModifier = 1.0f;
			bool prefiltered = false;
			bool antialiasing = false;
			float shResidualGain = 1.0f;
			bool shResidualOnly = false;
		};

		UECudaRasterizerBridge();
		~UECudaRasterizerBridge();

		UECudaRasterizerBridge(const UECudaRasterizerBridge&) = delete;
		UECudaRasterizerBridge& operator=(const UECudaRasterizerBridge&) = delete;

		int forward(
			ID3D12Device* d3dDevice,
			const ForwardParams& params,
			const CameraParams& camera,
			const ExternalBufferDesc& background,
			const ExternalBufferDesc& means3D,
			const ExternalBufferDesc& colorsPrecomp,
			const ExternalBufferDesc& opacities,
			const ExternalBufferDesc& scales,
			const ExternalBufferDesc& rotations,
			const ExternalBufferDesc& outColor);
		// outColor is written as CHW float layout (3 x H x W), matching CudaRasterizer output.

		void clearExternalMappings();

	private:
		enum class BufferSlot : uint8_t
		{
			Background = 0,
			Means3D = 1,
			ColorsPrecomp = 2,
			Opacities = 3,
			Scales = 4,
			Rotations = 5,
			OutColor = 6,
			Count = 7
		};

		struct SharedExternalBuffer
		{
			ID3D12Resource* resource = nullptr;
			size_t bytes = 0;
			void* externalMemory = nullptr;
			void* mappedPtr = nullptr;
		};

		void uploadCamera(const CameraParams& camera);
		void validateForwardInputSizes(const ForwardParams& params,
			const ExternalBufferDesc& background,
			const ExternalBufferDesc& means3D,
			const ExternalBufferDesc& colorsPrecomp,
			const ExternalBufferDesc& opacities,
			const ExternalBufferDesc& scales,
			const ExternalBufferDesc& rotations,
			const ExternalBufferDesc& outColor) const;

		void* importOrReuseExternalBuffer(ID3D12Device* d3dDevice, BufferSlot slot, const ExternalBufferDesc& desc);
		void releaseExternalBuffer(SharedExternalBuffer& state);
		void releaseTemporaryBuffers();
		char* ensureTemporaryBuffer(void*& ptr, size_t& capacityBytes, size_t requiredBytes);

		std::array<SharedExternalBuffer, static_cast<size_t>(BufferSlot::Count)> externalBuffers_{};

		float* viewCuda_ = nullptr;
		float* projCuda_ = nullptr;
		float* camPosCuda_ = nullptr;

		void* geomPtr_ = nullptr;
		void* binningPtr_ = nullptr;
		void* imgPtr_ = nullptr;
		size_t geomCapacityBytes_ = 0;
		size_t binningCapacityBytes_ = 0;
		size_t imgCapacityBytes_ = 0;
	};
}
