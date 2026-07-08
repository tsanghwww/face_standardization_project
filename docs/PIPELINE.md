# Pipeline

## Overview

```
Raw Images → DECA Reconstruction → FLAME Params (.mat) → Phase2 Standardization → Output
                  ↓                                               ↓
           ArcFace Embeddings                            XGBoost Quality Scores
```

## Stage 1: DECA Reconstruction

**Script**: `DECA/decalib/deca.py` (via `run_deca_batch_params.py` or `audit_deca.py`)

**Input**: Face images (PNG/JPG)
**Output**: `.mat` files containing FLAME parameters (shape, expression, pose, texture, lighting, camera)

**Environment**: 5060 with CUDA

## Stage 2: ArcFace Embedding Extraction

**Script**: `tools/extract_arcface_embeddings.py`

**Input**: Face images
**Output**: ArcFace identity embeddings (used for identity verification)

**Status**: [待确认] — Embeddings may have been lost from 2060, need regeneration

## Stage 3: Phase2 Training

**Script**: `phase2/train_condition_generator.py`

**Input**: DECA `.mat` outputs + quality scores
**Output**: Condition generator checkpoint (`best_model.pt`)

**Multi-stage**: Stage 1 → Stage 2 → Stage 3 (curriculum learning or progressive training)

## Stage 4: Phase2 Inference

**Script**: `phase2/infer_standardize_params.py`

**Input**: DECA `.mat` outputs + trained checkpoint
**Output**: Standardized FLAME parameters

## Stage 5: Visualization

**Script**: `phase2/make_visualizations.py`, `phase2/render_single_comparison.py`

**Input**: Original vs standardized DECA outputs
**Output**: Comparison images, evaluation metrics

## Supporting Tools

| Tool | Purpose |
|---|---|
| `tools/screen_percentile.py` | Screen dataset by percentile thresholds |
| `tools/screening_deca_params.py` | Screen based on DECA parameter quality |
| `tools/annotate_arcface_manifest.py` | Annotate data manifest with ArcFace info |
| `tools/merge_arcface_retry.py` | Merge retry results for ArcFace extraction |
| `tools/audit_deca_outputs.py` | Audit DECA output quality |
| `tools/compile_deca_rasterizer.py` | Compile CUDA rasterizer |

## Quality Assessment

**XGBoost Training**: `phase2/train_xgboost_quality.py`
**Baseline**: `phase2/baseline_hard_zero.py` (hard-zero benchmark)
**Comparison**: `phase2/compare_standardization_runs.py`
