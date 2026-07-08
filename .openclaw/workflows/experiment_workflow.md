# Experiment Workflow

## Before Starting

1. Define hypothesis clearly
2. Document in docs/EXPERIMENTS.md with:
   - Experiment ID
   - Date
   - Hypothesis
   - Expected outcome
   - Configuration (hyperparameters, dataset split, random seed)
3. Create a git branch if experiment is multi-day
4. Ensure all checkpoints are accessible

## Running

1. Always log the exact command line
2. Record random seed explicitly (do not rely on defaults)
3. Monitor GPU memory usage (8GB limit on 5060)
4. Save checkpoints to `results/{experiment_name}/`
5. Log training metrics to CSV

## After Completion

### Recording Results

Update docs/EXPERIMENTS.md with:

```markdown
## EXP-{NNN}: {Title}

- **Date**: YYYY-MM-DD
- **Status**: Completed / Failed / Inconclusive
- **Hypothesis**: ...
- **Configuration**:
  - Model: ...
  - Hyperparameters: ...
  - Dataset: ...
  - Random seed: ...
- **Results**:
  - Metric 1: ...
  - Metric 2: ...
- **Checkpoint**: results/{name}/best_model.pt
- **Conclusion**: ...
- **Next Steps**: ...
```

### Cleanup

- Delete intermediate checkpoints if not needed
- Keep best_model.pt and final checkpoints
- Add visualization outputs to phase2_visualizations/ if publishable
- Commit code changes (not results data)

## Experiment Naming

Format: `{stage}_{descriptor}_{date}`

Examples:
- `stage1_baseline_20260701`
- `stage2_xgb_weighted_20260705`
- `stage3_final_20260708`

## Quality Gates

Before considering an experiment complete:

- [ ] Training completed without crashes
- [ ] Metrics logged and saved
- [ ] Best checkpoint identified and saved
- [ ] Results recorded in EXPERIMENTS.md
- [ ] Any code changes committed to git
- [ ] Failure modes documented (if experiment failed)
