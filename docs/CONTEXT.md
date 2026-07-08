# Project Context

## Project
- **Name**: face_standardization_project
- **Domain**: Face standardization research using 3D face reconstruction
- **GitHub**: `git@github.com:tsanghwww/face_standardization_project.git`

## Architecture Overview

```
Input Images → DECA (3D reconstruction) → Phase2 (standardization) → Standardized Faces
                    ↓                              ↓
              FLAME params                    Quality Scores
              (.mat files)                   (XGBoost)
```

### Components

1. **DECA** (third-party): 3D face reconstruction, extracts FLAME parameters (shape, expression, pose, texture, lighting)
2. **Phase2** (core research): Condition generator that learns to standardize expression and pose parameters
3. **Tools**: ArcFace embedding extraction, DECA output auditing, dataset screening, CUDA rasterizer compilation

### Key Files

| File | Role |
|---|---|
| `DECA/decalib/deca.py` | DECA model wrapper |
| `phase2/model.py` | ConditionGenerator (MLP-based) |
| `phase2/train_condition_generator.py` | Training loop with quality-aware loss |
| `phase2/infer_standardize_params.py` | Inference: apply standardization to DECA outputs |
| `phase2/train_xgboost_quality.py` | Train XGBoost quality scorer |

## Environment

| Role | Machine | GPU |
|---|---|---|
| Control / Code Editing | MacBook Pro | — |
| Training / Inference | Windows 11 Laptop (win-lenovo) | RTX 5060 Laptop (8GB) |

- Python 3.12.13, PyTorch 2.11.0+cu128
- SSH from Mac to 5060 using key `~/.ssh/id_rsa_windows`
- Project on 5060 at `D:\face_standardization_project`

## Current Status

- ✅ Code synced across Mac, 5060, and GitHub
- ✅ DECA model weights and FLAME assets present
- ✅ Phase2 training checkpoints recovered (stage1/2/3)
- ✅ Dataset (StyleGAN2 generated faces × 10000) in `archive/` on 5060
- ⚠️ Original RTX 2060 workstation lost — project recovered from external backups
- ⚠️ PIL initialized 2026-07-08

## Research Direction

- Face standardization via DECA parameter space manipulation
- Quality-aware conditioning (XGBoost quality scores guide generation)
- [待确认] Current research phase and target
- [待确认] Target conference/journal
