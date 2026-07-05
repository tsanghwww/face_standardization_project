# Phase2 DECA Standardization

This package implements the first practical version of `phase2_deca_standardization_plan.tex`.

It keeps DECA frozen and learns a confidence-aware condition generator:

```text
DECA params + quality metrics
-> target expression / target pose
-> alpha_expression / alpha_head_pose / alpha_jaw_pose
-> standardized expression / standardized pose
-> confidence / reject score
```

## Inputs

Run DECA first with `--saveMat True --saveKpt True`. The expected layout is:

```text
results_dir/
  image_id/
    image_id.mat
    image_id_kpt2d.txt
    image_id_kpt3d.txt
```

ArcFace manifest is optional. If provided, `detector_score`, `arcface_status`, and `use_for_train` are used as quality features.

## Runtime

Minimal dependencies:

```bash
pip install -r phase2/requirements_phase2.txt
```

On your Windows machine, prefer the existing DECA virtual environment:

```powershell
cd H:\face_standardization_project\repo
H:\face_standardization_project\repo\DECA\.venv\Scripts\python.exe -m phase2.train_condition_generator --help
```

## 1. Build Quality Manifest

```bash
python -m phase2.build_manifest \
  --deca-results-dir DECA/results/my_run \
  --arcface-manifest results/arcface/arcface_manifest.csv \
  --out-csv results/phase2/manifest.csv \
  --out-json results/phase2/manifest_summary.json
```

This creates high/medium/low quality labels using structured proxy metrics. It is the non-XGBoost fallback for phase zero.

## 2. Train XGBoost Quality Filter

```bash
python -m phase2.train_xgboost_quality \
  --manifest results/phase2/manifest.csv \
  --screening-report results/screening/screening_report.json \
  --out-dir results/phase0_xgboost_quality
```

This trains a binary XGBoost quality filter from Phase1 `Pass` / `Warn` labels and applies it to every Phase2 sample. The default features intentionally exclude `arcface_train_flag`, because that flag is derived from the same screening label and would leak the target.

Outputs:

```text
xgb_quality_model.json
xgb_quality_manifest.csv
xgb_quality_summary.json
xgb_feature_importance.csv
xgb_training_log.json
```

The XGBoost manifest adds:

```text
xgb_quality_score
xgb_quality_label
xgb_sample_weight
xgb_use_for_strong_train
xgb_use_for_weak_train
```

## 3. Train Condition Generator

```bash
python -m phase2.train_condition_generator \
  --deca-results-dir DECA/results/my_run \
  --arcface-manifest results/arcface/arcface_manifest.csv \
  --out-dir results/phase2/model_stage1 \
  --epochs 40 \
  --batch-size 64 \
  --stage 1 \
  --device auto
```

Outputs:

```text
best_model.pt
normalizer.npz
train_history.csv
train_summary.json
```

Use `--stage 2` or `--stage 3` for stronger pose augmentation after stage 1 is stable.

## 4. Run Phase2 Inference

```bash
python -m phase2.infer_standardize_params \
  --deca-results-dir DECA/results/my_run \
  --arcface-manifest results/arcface/arcface_manifest.csv \
  --checkpoint results/phase2/model_stage1/best_model.pt \
  --out-dir results/phase2/inference_stage1
```

Outputs:

```text
phase2_inference_manifest.csv
phase2_inference_summary.json
params/*_phase2.npz
```

Each `.npz` contains original and standardized `expression` / `pose`, target parameters, alpha values, confidence, reject score, and quality score.

## 5. Baseline A

```bash
python -m phase2.baseline_hard_zero \
  --deca-results-dir DECA/results/my_run \
  --out-dir results/phase2/baseline_hard_zero
```

This creates the hard-zero baseline:

```text
expression_standardized = 0
pose_standardized = 0
```

## Current Scope

Implemented:

- Phase zero quality manifest with heuristic fallback.
- XGBoost phase-zero quality classifier and pre-screening manifest.
- Stage-one frozen-DECA training.
- DECA latent-space augmentation for expression, head pose, jaw pose, light, and camera.
- Partial canonicalization with three scalar alphas.
- Confidence and reject score.
- Smoothness regularization.
- Hard-zero baseline for ablation.

Not yet implemented:

- Differentiable DECA decoder validation.
- ArcFace image-space identity loss after rendering standardized outputs.
- Image-level corruption during DECA encoding.
