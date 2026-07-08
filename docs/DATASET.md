# Dataset Inventory

## Primary Dataset

### StyleGAN2 Generated Faces

- **Location**: `D:\face_standardization_project\archive\generated_yellow-stylegan2\`
- **Count**: 10,000 images
- **Size**: 11.6 GB
- **Format**: PNG
- **Naming**: `0.png` through `9999.png`
- **Source**: StyleGAN2 generation (yellow-stylegan2 variant)
- **Status**: ✅ Present on 5060
- **Backup**: Also archived in `archive.zip` (11.6 GB, identical content)

### Screening Datasets (Pre-filtered)

- **p95**: `results/screening_p95/` — 10,008 files, 11.6 GB
- **p97.5**: `results/screening_p975/` — 10,008 files, 11.6 GB
- **Note**: These may be redundant copies of filtered subsets from the main archive

## Test Data

### Single Test Inputs

- **Location**: `single_test_inputs/`
- **Files**: `IMG_6033.jpeg`, `IMG_6033.jpg`
- **Purpose**: Quick inference testing

### Stress Test Inputs

- **Location**: `stress_test_inputs/`
- **Files**: `IMG_6033.jpeg`
- **Purpose**: Batch processing stress testing

## DECA Test Samples (Third-Party)

- **Location**: `DECA/TestSamples/`
- **Contents**: AFLW2000 subset, example images, expression test images
- **Purpose**: DECA library testing (do not use for research)

## Screening Results

- **Location**: `results/screening_p95/`, `results/screening_p975/`
- **Purpose**: Quality-filtered dataset subsets

## Unknown / Missing

- [待确认] Original non-generated face dataset — was it part of 2060?
- [待確認] Training/validation/test split definitions
- [待確認] Any additional augmentation data

## Dataset Usage Notes

- Dataset is excluded from Git (.gitignore: `archive/`)
- Archive backup: `D:\face_standardization_project\archive.zip` (12GB)
- Only on 5060 — not synced to Mac or GitHub
- If dataset needs to move, copy archive.zip or the archive/ directory
