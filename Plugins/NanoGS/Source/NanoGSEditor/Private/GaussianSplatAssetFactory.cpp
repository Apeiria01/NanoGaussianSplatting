// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatAssetFactory.h"
#include "GaussianSplatAsset.h"
#include "PLYFileReader.h"
#include "EditorFramework/AssetImportData.h"
#include "Misc/FeedbackContext.h"
#include "Misc/ScopedSlowTask.h"
#include "UObject/UObjectIterator.h"
#include "Widgets/SBoxPanel.h"
#include "Widgets/Text/STextBlock.h"
#include "Widgets/Input/SButton.h"
#include "Widgets/Input/SCheckBox.h"
#include "Widgets/Layout/SUniformGridPanel.h"
#include "Interfaces/IMainFrameModule.h"
#include "Modules/ModuleManager.h"

// Static session state
UGaussianSplatAssetFactory::EPlyImporterChoice UGaussianSplatAssetFactory::SessionChoice =
	UGaussianSplatAssetFactory::EPlyImporterChoice::Ask;
bool UGaussianSplatAssetFactory::bRememberSessionChoice = false;

UGaussianSplatAssetFactory::UGaussianSplatAssetFactory()
{
	bCreateNew = false;
	bEditorImport = true;
	bText = false;

	SupportedClass = UGaussianSplatAsset::StaticClass();

	Formats.Add(TEXT("ply;PLY Gaussian Splatting File"));

	// Make sure UE always picks this factory first when both NanoGS and the
	// Niagara-based plugin register .ply. We then dispatch from FactoryCreateFile.
	ImportPriority = DefaultImportPriority + 100;
}

namespace NanoGSPlyImporterDispatch
{
	/** Locate the Niagara plugin's PLY factory class without a hard module dependency. */
	static UClass* FindNiagaraPlyFactoryClass()
	{
		// Class name from GaussianSplattingPointCloudAssetFactory.h in the other plugin.
		static const TCHAR* CandidateNames[] = {
			TEXT("GaussianSplattingPointCloudAssetFactory"),
		};

		for (const TCHAR* Name : CandidateNames)
		{
			if (UClass* Found = FindFirstObject<UClass>(Name, EFindFirstObjectOptions::ExactClass))
			{
				return Found;
			}
		}
		return nullptr;
	}

	/** Modal dialog: ask which PLY importer to use. Returns chosen value, writes remember flag. */
	static UGaussianSplatAssetFactory::EPlyImporterChoice ShowPickerDialog(
		const FString& Filename,
		bool bNiagaraAvailable,
		bool& bOutRemember)
	{
		using EChoice = UGaussianSplatAssetFactory::EPlyImporterChoice;

		EChoice Result = EChoice::NanoGS;
		bOutRemember = false;

		TSharedPtr<SWindow> Window;
		TSharedPtr<SCheckBox> RememberCheck;

		Window = SNew(SWindow)
			.Title(FText::FromString(TEXT("Import PLY — choose importer")))
			.SizingRule(ESizingRule::Autosized)
			.SupportsMaximize(false)
			.SupportsMinimize(false);

		auto MakeButton = [&](const FString& Label, const FString& Tooltip, EChoice Choice, bool bEnabled)
		{
			return SNew(SButton)
				.HAlign(HAlign_Center)
				.IsEnabled(bEnabled)
				.ToolTipText(FText::FromString(Tooltip))
				.OnClicked_Lambda([&Result, &Window, &RememberCheck, &bOutRemember, Choice]()
				{
					Result = Choice;
					if (RememberCheck.IsValid())
					{
						bOutRemember = RememberCheck->IsChecked();
					}
					if (Window.IsValid())
					{
						Window->RequestDestroyWindow();
					}
					return FReply::Handled();
				})
				[
					SNew(STextBlock).Text(FText::FromString(Label))
				];
		};

		Window->SetContent(
			SNew(SVerticalBox)
			+ SVerticalBox::Slot().AutoHeight().Padding(12)
			[
				SNew(STextBlock)
				.Text(FText::FromString(FString::Printf(
					TEXT("Two plugins can import this PLY:\n\n  %s\n\nWhich importer should handle it?"),
					*Filename)))
			]
			+ SVerticalBox::Slot().AutoHeight().Padding(12, 0, 12, 8)
			[
				SNew(SUniformGridPanel).SlotPadding(FMargin(6))
				+ SUniformGridPanel::Slot(0, 0)
				[
					MakeButton(
						TEXT("NanoGS (cluster + RDG)"),
						TEXT("Use the NanoGS plugin: clustered Nanite-style renderer."),
						EChoice::NanoGS, true)
				]
				+ SUniformGridPanel::Slot(1, 0)
				[
					MakeButton(
						TEXT("Niagara version"),
						bNiagaraAvailable
							? TEXT("Use the GaussianSplattingForUnrealEngine plugin (Niagara renderer).")
							: TEXT("The Niagara-based plugin is not loaded — option unavailable."),
						EChoice::Niagara, bNiagaraAvailable)
				]
			]
			+ SVerticalBox::Slot().AutoHeight().Padding(12, 0, 12, 12)
			[
				SAssignNew(RememberCheck, SCheckBox)
				[
					SNew(STextBlock).Text(FText::FromString(
						TEXT("Remember my choice for the rest of this editor session")))
				]
			]
		);

		// Parent to the main editor window if available so the dialog is properly modal.
		TSharedPtr<SWindow> ParentWindow;
		if (FModuleManager::Get().IsModuleLoaded("MainFrame"))
		{
			IMainFrameModule& MainFrame = FModuleManager::LoadModuleChecked<IMainFrameModule>("MainFrame");
			ParentWindow = MainFrame.GetParentWindow();
		}
		FSlateApplication::Get().AddModalWindow(Window.ToSharedRef(), ParentWindow);

		return Result;
	}
}

bool UGaussianSplatAssetFactory::FactoryCanImport(const FString& Filename)
{
	const FString Extension = FPaths::GetExtension(Filename);
	return Extension.Equals(TEXT("ply"), ESearchCase::IgnoreCase) && FPLYFileReader::IsValidPLYFile(Filename);
}

UObject* UGaussianSplatAssetFactory::FactoryCreateFile(
	UClass* InClass,
	UObject* InParent,
	FName InName,
	EObjectFlags Flags,
	const FString& Filename,
	const TCHAR* Parms,
	FFeedbackContext* Warn,
	bool& bOutOperationCanceled)
{
	bOutOperationCanceled = false;

	// --- Importer dispatch ---------------------------------------------------
	UClass* NiagaraFactoryClass = NanoGSPlyImporterDispatch::FindNiagaraPlyFactoryClass();
	const bool bNiagaraAvailable = (NiagaraFactoryClass != nullptr);

	EPlyImporterChoice Choice = SessionChoice;
	if (!bRememberSessionChoice || Choice == EPlyImporterChoice::Ask)
	{
		if (bNiagaraAvailable)
		{
			bool bRemember = false;
			Choice = NanoGSPlyImporterDispatch::ShowPickerDialog(Filename, true, bRemember);
			if (bRemember)
			{
				SessionChoice = Choice;
				bRememberSessionChoice = true;
			}
		}
		else
		{
			// Niagara plugin not present — just use NanoGS silently.
			Choice = EPlyImporterChoice::NanoGS;
		}
	}

	if (Choice == EPlyImporterChoice::Niagara && bNiagaraAvailable)
	{
		// Forward to the Niagara plugin's factory without a hard module dependency.
		UFactory* Forwarded = NewObject<UFactory>(GetTransientPackage(), NiagaraFactoryClass);
		if (Forwarded)
		{
			Forwarded->AddToRoot();
			UObject* Result = Forwarded->FactoryCreateFile(
				InClass, InParent, InName, Flags, Filename, Parms, Warn, bOutOperationCanceled);
			Forwarded->RemoveFromRoot();
			return Result;
		}
	}

	// --- Default: NanoGS path ------------------------------------------------
	UGaussianSplatAsset* NewAsset = ImportPLYFile(Filename, InParent, InName, Flags, nullptr);

	if (!NewAsset)
	{
		if (Warn)
		{
			Warn->Logf(ELogVerbosity::Error, TEXT("Failed to import Gaussian Splat from: %s"), *Filename);
		}
	}

	return NewAsset;
}

FText UGaussianSplatAssetFactory::GetDisplayName() const
{
	return FText::FromString(TEXT("Gaussian Splat Asset"));
}

bool UGaussianSplatAssetFactory::CanReimport(UObject* Obj, TArray<FString>& OutFilenames)
{
	UGaussianSplatAsset* Asset = Cast<UGaussianSplatAsset>(Obj);
	if (Asset && !Asset->SourceFilePath.IsEmpty())
	{
		OutFilenames.Add(Asset->SourceFilePath);
		return true;
	}
	return false;
}

void UGaussianSplatAssetFactory::SetReimportPaths(UObject* Obj, const TArray<FString>& NewReimportPaths)
{
	UGaussianSplatAsset* Asset = Cast<UGaussianSplatAsset>(Obj);
	if (Asset && NewReimportPaths.Num() > 0)
	{
		Asset->SourceFilePath = NewReimportPaths[0];
	}
}

EReimportResult::Type UGaussianSplatAssetFactory::Reimport(UObject* Obj)
{
	UGaussianSplatAsset* Asset = Cast<UGaussianSplatAsset>(Obj);
	if (!Asset)
	{
		return EReimportResult::Failed;
	}

	if (Asset->SourceFilePath.IsEmpty())
	{
		UE_LOG(LogTemp, Error, TEXT("Cannot reimport: source file path is empty"));
		return EReimportResult::Failed;
	}

	if (!FPaths::FileExists(Asset->SourceFilePath))
	{
		UE_LOG(LogTemp, Error, TEXT("Cannot reimport: source file not found: %s"), *Asset->SourceFilePath);
		return EReimportResult::Failed;
	}

	// Preserve Nanite setting before reimport
	const bool bWasNaniteEnabled = Asset->IsNaniteEnabled();

	// Use the original quality level
	QualityLevel = Asset->ImportQuality;

	UGaussianSplatAsset* ReimportedAsset = ImportPLYFile(
		Asset->SourceFilePath,
		Asset->GetOuter(),
		Asset->GetFName(),
		Asset->GetFlags(),
		Asset
	);

	if (ReimportedAsset)
	{
		// If Nanite was enabled before reimport, rebuild the cluster hierarchy
		if (bWasNaniteEnabled)
		{
			UE_LOG(LogTemp, Log, TEXT("Reimport: Rebuilding Nanite cluster hierarchy (was enabled before reimport)"));
			if (!ReimportedAsset->BuildNaniteClusterHierarchy())
			{
				UE_LOG(LogTemp, Warning, TEXT("Reimport: Failed to rebuild Nanite cluster hierarchy"));
			}
		}
		return EReimportResult::Succeeded;
	}

	return EReimportResult::Failed;
}

UGaussianSplatAsset* UGaussianSplatAssetFactory::ImportPLYFile(
	const FString& FilePath,
	UObject* InParent,
	FName InName,
	EObjectFlags Flags,
	UGaussianSplatAsset* ExistingAsset)
{
	FScopedSlowTask SlowTask(100.0f, FText::FromString(TEXT("Importing Gaussian Splat...")));
	SlowTask.MakeDialog(true);

	// Read PLY file
	SlowTask.EnterProgressFrame(30.0f, FText::FromString(TEXT("Reading PLY file...")));

	TArray<FGaussianSplatData> SplatData;
	FString ErrorMessage;
	int32 DetectedSHBands = 0;

	if (!FPLYFileReader::ReadPLYFile(FilePath, SplatData, ErrorMessage, &DetectedSHBands))
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to read PLY file: %s"), *ErrorMessage);
		return nullptr;
	}

	UE_LOG(LogTemp, Log, TEXT("Read %d splats from PLY file (SH bands: %d)"), SplatData.Num(), DetectedSHBands);

	// Create or reuse asset
	SlowTask.EnterProgressFrame(10.0f, FText::FromString(TEXT("Creating asset...")));

	UGaussianSplatAsset* Asset = ExistingAsset;
	if (!Asset)
	{
		Asset = NewObject<UGaussianSplatAsset>(InParent, UGaussianSplatAsset::StaticClass(), InName, Flags);
	}

	if (!Asset)
	{
		UE_LOG(LogTemp, Error, TEXT("Failed to create Gaussian Splat asset"));
		return nullptr;
	}

	// Store source file path
	Asset->SourceFilePath = FilePath;

	// Set the detected SH band count BEFORE initializing (CompressSH uses this)
	Asset->SHBands = DetectedSHBands;

	// Initialize asset from splat data (NO cluster building - user enables Nanite via Asset Actions)
	SlowTask.EnterProgressFrame(55.0f, FText::FromString(TEXT("Compressing splat data...")));

	Asset->InitializeFromSplatData(SplatData, QualityLevel);

	// NO cluster hierarchy by default - user enables Nanite via Asset Actions > Nanite
	Asset->ClusterHierarchy.Reset();

	// Mark package dirty
	Asset->MarkPackageDirty();

	UE_LOG(LogTemp, Log, TEXT("Successfully imported Gaussian Splat asset: %d splats, %lld bytes (Nanite disabled by default)"),
		Asset->GetSplatCount(), Asset->GetMemoryUsage());

	return Asset;
}
