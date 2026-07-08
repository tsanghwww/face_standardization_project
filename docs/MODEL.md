# Model Architecture & Checkpoints

## DECA (Third-Party)

- **Source**: DECA: Detailed Expression Capture and Animation
- **Architecture**: ResNet-50 encoder → FLAME decoder
- **Outputs**: 100 shape params, 50 expression params, 50 texture params, 3 pose, 27 lighting, 3 camera
- **Pretrained Weight**: `DECA/data/deca_model.tar`
- **FLAME Model**: `DECA/data/generic_model.pkl` + supporting files

## Phase2: Condition Generator

### Architecture

- **Type**: MLP-based condition generator
- **Input dim**: [待确认] (DECA parameter space subset)
- **Hidden dim**: [待确认]
- **Output**: Standardized DECA parameters
- **Defined in**: `phase2/model.py` (76 lines)

### Training

- **Script**: `phase2/train_condition_generator.py`
- **Loss**: Quality-weighted L1/L2 [待确认]
- **Optimizer**: [待确认]

### Checkpoints

| Stage | Path (5060) | Size | Status |
|---|---|---|---|
| Stage 1 | `results/phase2_real_train_stage1_recovered/best_model.pt` | 439 KB | ✅ Recovered |
| Stage 2 | `results/phase2_real_train_stage2_recovered/best_model.pt` | [待确认] | ✅ Recovered |
| Stage 3 | `results/phase2_real_train_stage3_recovered/best_model.pt` | [待确认] | ✅ Recovered |

### Checkpoint Contents

Each checkpoint contains:
- `model_state`: ConditionGenerator state dict
- `input_dim`: Input dimension
- `hidden_dim`: Hidden dimension
- Normalizer parameters (numpy arrays, stored as `normalizer.npz`)
- Training metadata (`train_summary.json`, `train_history.csv`)

### Inference Outputs

| Stage | Path | Size | Files |
|---|---|---|---|
| Stage 1 | `results/phase2_real_infer_stage1_recovered/` | 48 MB | 10,002 |
| Stage 2 | `results/phase2_real_infer_stage2_recovered/` | 48 MB | 10,002 |
| Stage 3 | `results/phase2_real_infer_stage3_recovered/` | 48 MB | 10,002 |
| Hard Zero | `results/phase2_hard_zero_recovered/` | 38 MB | 10,002 |

### Baseline & Comparison

| Output | Path | Size |
|---|---|---|
| Hard Zero | `results/phase2_hard_zero_recovered/` | 38 MB |
| Comparison | `results/phase2_recovered_compare/` | <1 MB |
| Full Comparison | `results/phase2_recovered_compare_full/` | <1 MB |

### Dataset Screening

| Screen | Path | Size | Files |
|---|---|---|---|
| p95 | `results/screening_p95/` | 11.6 GB | 10,008 |
| p97.5 | `results/screening_p975/` | 11.6 GB | 10,008 |

## XGBoost Quality Scorer

- **Script**: `phase2/train_xgboost_quality.py`
- **Purpose**: Predict output quality score for each sample
- **Used in**: Training loss weighting during Phase2
- **Checkpoint**: [待确认]

## Inference

- **Script**: `phase2/infer_standardize_params.py`
- **Input**: DECA `.mat` output directory + Phase2 checkpoint
- **Output**: Standardized parameters
- **Options**: `--reject-threshold`, `--weak-threshold`, `--device`
