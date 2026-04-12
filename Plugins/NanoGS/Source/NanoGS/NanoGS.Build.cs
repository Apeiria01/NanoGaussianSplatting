// Copyright Epic Games, Inc. All Rights Reserved.

using System;
using System.IO;
using UnrealBuildTool;

public class NanoGS : ModuleRules
{
	public NanoGS(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = ModuleRules.PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicIncludePaths.AddRange(
			new string[] {
			}
		);



		PrivateIncludePaths.AddRange(
			new string[] {
				// Access Renderer private headers for FViewInfo::ViewRect (screen percentage support)
				System.IO.Path.Combine(EngineDirectory, "Source/Runtime/Renderer/Private"),
				System.IO.Path.Combine(EngineDirectory, "Source/Runtime/Renderer/Internal"),
			}
		);

		PublicDependencyModuleNames.AddRange(
			new string[]
			{
				"Core",
				"CoreUObject",
				"Engine",
				"RenderCore",
				"RHI",
				"Renderer",
				"Projects",
            }
		);

        

        // D3D12 native interop for CUDA-DX12 sharing in GaussianSplatRenderData.cpp.
        // We need both the D3D12RHI module (for ID3D12DynamicRHI) AND the underlying
        // DX12 third-party dependency, which transitively pulls AgilitySDK and adds
        // BOTH "AgilitySDK/<ver>/Include" (for the modern d3d12.h) and ".../Include/d3dx12"
        // to PublicSystemIncludePaths. Adding only the d3dx12 directory is not enough,
        // because d3dx12.h includes <d3d12.h> and would otherwise resolve to the older
        // Windows SDK header that lacks D3D12_UAV_DIMENSION_TEXTURE2DMS, D3D_FORMAT_LAYOUT, etc.
        if (Target.Platform.IsInGroup(UnrealPlatformGroup.Windows))
        {
            PrivateDependencyModuleNames.Add("D3D12RHI");
            AddEngineThirdPartyPrivateStaticDependencies(Target, "DX12");
            string PluginDir = Path.GetFullPath(Path.Combine(ModuleDirectory, "..", ".."));
            string ThirdPartyDir = Path.Combine(PluginDir, "Source", "ThirdParty", "SibrCudaUEInterop");
            string IncludeDir = Path.Combine(ThirdPartyDir, "include");
            string LibDir = Path.Combine(ThirdPartyDir, "lib", "Win64");
            string BinDir = Path.Combine(ThirdPartyDir, "bin", "Win64");
			PublicDependencyModuleNames.Add("CUDA");

            PublicIncludePaths.Add(IncludeDir);
            PublicAdditionalLibraries.Add(Path.Combine(LibDir, "sibr_cudaueinterop_rwdi.lib"));

            PublicDelayLoadDLLs.Add("sibr_cudaueinterop_rwdi.dll");
            RuntimeDependencies.Add("$(TargetOutputDir)/sibr_cudaueinterop_rwdi.dll", Path.Combine(BinDir, "sibr_cudaueinterop_rwdi.dll"));

            PublicDelayLoadDLLs.Add("cudart64_12.dll");
            RuntimeDependencies.Add("$(TargetOutputDir)/cudart64_12.dll", @"D:\CUDA_12.6\bin\cudart64_12.dll");
        }

        PrivateDependencyModuleNames.AddRange(
			new string[]
			{
				"Slate",
				"SlateCore"
			}
		);

		DynamicallyLoadedModuleNames.AddRange(
			new string[]
			{
			}
		);
	}
}
