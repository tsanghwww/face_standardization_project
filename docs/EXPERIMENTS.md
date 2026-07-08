# Experiments

## Active Experiments

*None yet — [待确认] next research phase*

## Completed Experiments

### EXP-001: Phase2 Training Stage 1

- **Date**: [待确认] (pre-PIL, recovered from backup)
- **Status**: Completed
- **Hypothesis**: [待确认]
- **Configuration**: [待确认]
- **Results**: best_model.pt recovered at `results/phase2_real_train_stage1_recovered/`
- **Metrics**: See `train_summary.json` and `train_history.csv` in checkpoint directory

### EXP-002: Phase2 Training Stage 2

- **Date**: [待确认] (pre-PIL, recovered from backup)
- **Status**: Completed
- **Configuration**: [待确认]
- **Results**: best_model.pt recovered at `results/phase2_real_train_stage2_recovered/`

### EXP-003: Phase2 Training Stage 3

- **Date**: [待确认] (pre-PIL, recovered from backup)
- **Status**: Completed
- **Configuration**: [待确认]
- **Results**: best_model.pt recovered at `results/phase2_real_train_stage3_recovered/`

### EXP-004: Hard Zero Baseline

- **Date**: [待确认] (pre-PIL)
- **Status**: Completed
- **Hypothesis**: [待确认]
- **Results**: `results/phase2_hard_zero_recovered/`

### EXP-005: Dataset Screening (p95 / p975)

- **Date**: [待确认] (pre-PIL)
- **Status**: Completed
- **Description**: Screened dataset using percentile-based thresholds
- **Results**: `results/screening_p95/`, `results/screening_p975/`

### EXP-006: Phase2 Inference (Recovered Checkpoints)

- **Date**: 2026-07-08
- **Status**: Verified working
- **Description**: Verified that recovered stage1/2/3 checkpoints load correctly and inference CLI functions
- **Results**: All checkpoints load, `infer_standardize_params` CLI operational

## Experiment Template

```markdown
## EXP-NNN: Title

- **Date**: YYYY-MM-DD
- **Status**: Planned / Running / Completed / Failed / Inconclusive
- **Hypothesis**: ...
- **Configuration**:
  - Model: ...
  - Hyperparameters: {lr: ..., batch_size: ..., epochs: ...}
  - Dataset: ...
  - Random seed: ...
- **Results**:
  - Metric 1: ...
  - Metric 2: ...
- **Checkpoint**: results/{name}/best_model.pt
- **Conclusion**: ...
- **Next Steps**: ...
```
