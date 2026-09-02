# Downstream Preparation Tasks

Branch:

```text
feature/condition-dataset-and-eval-skeleton
```

## Background

The current project is still focused on finishing Phase2.

Phase2 is not fully closed yet. The BUG-003 landmark-coordinate issue has been fixed in code, and a corrected manifest exists on `win-lenovo`, but Phase2 inference / checkpoint validity / possible retraining still need to be handled separately.

Your work should not block or modify Phase2. The goal is to prepare the downstream structure so that once Phase2 produces reliable standardized 3D parameters, the next model-training stage can start quickly.

## What Not To Do

Please do not modify Phase2 core files:

```text
phase2/model.py
phase2/train_condition_generator.py
phase2/infer_standardize_params.py
phase2/features.py
phase2/dataset.py
```

Please do not commit large data or generated outputs:

```text
archive/
results/
DECA/results/
*.pt
*.pth
*.ckpt
*.npy
*.npz
*.mat
```

Please do not start diffusion / ControlNet training yet. This task is only for interface design, dataset skeletons, evaluation skeletons, and experiment planning.

## Task 1: Condition Dataset Schema

Create:

```text
docs/CONDITION_DATASET_SCHEMA.md
```

Purpose:

Define the unified sample format for the downstream image-generation model.

The schema should assume that each sample may eventually include:

```json
{
  "image_id": "0001",
  "source_image": "...",
  "deca_mat": "...",
  "phase2_npz": "...",
  "depth_map": "...",
  "normal_map": "...",
  "landmark_map": "...",
  "arcface_embedding": "...",
  "gaze_pitch": 0.0,
  "gaze_yaw": 0.0,
  "quality_score": 0.0,
  "phase2_confidence": 0.0,
  "phase2_reject_score": 0.0,
  "split": "train"
}
```

The document should explain:

- Required fields
- Optional fields
- Where each field comes from
- Whether the field depends on Phase2
- How missing values should be represented
- Which fields are for training, evaluation, or debugging

## Task 2: Condition Design

Create:

```text
docs/CONDITION_DESIGN.md
```

Purpose:

Define what downstream diffusion / ControlNet-style models may use as conditioning inputs.

Cover these condition types:

```text
landmark map
depth map
normal map
face mask
DECA parameter vector
Phase2 standardized pose/expression vector
Phase2 alpha/confidence/reject signals
ArcFace identity embedding
L2CS gaze vector
```

For each condition, describe:

- Source artifact
- Shape / expected format
- Whether it is image-like or vector-like
- Whether it should be used by ControlNet, cross-attention, adapter, or evaluation only
- Risks and limitations

Important distinction:

```text
head pose != eye gaze
```

DECA pose and L2CS gaze should be treated as separate signals.

## Task 3: Condition Dataset Builder Skeleton

Create:

```text
scripts/build_condition_dataset.py
```

Purpose:

Build downstream `jsonl` manifests from existing project manifests.

This script does not need to generate final data yet. It should provide a robust skeleton with path checks and clear TODOs.

Expected CLI:

```bash
python scripts/build_condition_dataset.py \
  --phase1-manifest results/phase1_parity/phase1_master_manifest.csv \
  --phase2-manifest results/phase2_infer_stage3_bug003_fixed/phase2_inference_manifest.csv \
  --out-dir datasets/condition_dataset \
  --split-dir datasets/condition_dataset/splits \
  --dry-run
```

Expected outputs:

```text
datasets/condition_dataset/train.jsonl
datasets/condition_dataset/val.jsonl
datasets/condition_dataset/test.jsonl
datasets/condition_dataset/dataset_summary.json
```

The script should support:

- Reading Phase1 master manifest
- Optionally reading a Phase2 inference manifest
- Joining by `image_id`
- Checking source image path, DECA path, Phase2 path, ArcFace path
- Writing JSONL rows
- Dry-run mode
- Summary counts

It can include TODO placeholders for condition maps that do not exist yet.

## Task 4: Evaluation Skeletons

Create:

```text
eval/evaluate_identity_preservation.py
eval/evaluate_pose_standardization.py
eval/evaluate_gaze_behavior.py
```

Purpose:

Prepare evaluation entry points before the final model exists.

These scripts should initially support:

- CLI argument parsing
- Manifest loading
- Output directory creation
- Missing-file checks
- Empty / placeholder metric outputs
- Clear TODO sections for actual metric implementation

### Identity Evaluation

File:

```text
eval/evaluate_identity_preservation.py
```

Future purpose:

Compare identity consistency between source images and generated / rendered outputs using ArcFace.

Initial output:

```text
identity_metrics.csv
identity_summary.json
```

### Pose Evaluation

File:

```text
eval/evaluate_pose_standardization.py
```

Future purpose:

Evaluate whether generated or rendered outputs move closer to canonical head pose.

Initial output:

```text
pose_metrics.csv
pose_summary.json
```

### Gaze Evaluation

File:

```text
eval/evaluate_gaze_behavior.py
```

Future purpose:

Evaluate gaze behavior after standardization.

Important:

This should not claim true gaze disentanglement yet. For now it should only measure gaze-related behavior.

Initial output:

```text
gaze_metrics.csv
gaze_summary.json
```

## Task 5: Ablation Plan

Create:

```text
docs/ABLATION_PLAN.md
```

Purpose:

Define future experiment groups for the downstream model.

Suggested groups:

| Group | Description |
|---|---|
| A | Hard-zero parameter baseline |
| B | Phase2-only parameter standardization |
| C | Image U-Net baseline |
| D | ControlNet with landmarks only |
| E | ControlNet with depth only |
| F | ControlNet with normals only |
| G | ControlNet with DECA parameter vector |
| H | ControlNet with Phase2 standardized params |
| I | Full model with identity condition |
| J | Full model with identity + gaze condition |

For each group, define:

- Input
- Output
- Required artifacts
- Metrics
- What hypothesis it tests
- Whether it depends on Phase2 being complete

## Task 6: Gaze Disentanglement Design

Create:

```text
docs/GAZE_DISENTANGLEMENT_DESIGN.md
```

Purpose:

Prepare the conceptual design for gaze adjustment after Phase2.

The document should answer:

- What is the difference between head pose and eye gaze?
- What does L2CS provide?
- Is the goal to preserve gaze, neutralize gaze, or control gaze?
- What data is needed for each goal?
- What can be done with pseudo-labels?
- What cannot be claimed without ground-truth gaze labels?
- What metrics should be used?

Important wording:

Do not claim that the project has achieved gaze disentanglement yet. The current project has gaze extraction and future gaze-evaluation plans, but not a completed disentanglement model.

## Suggested Directory Structure

Please add only lightweight source/docs:

```text
docs/
  CONDITION_DATASET_SCHEMA.md
  CONDITION_DESIGN.md
  ABLATION_PLAN.md
  GAZE_DISENTANGLEMENT_DESIGN.md

scripts/
  build_condition_dataset.py

eval/
  evaluate_identity_preservation.py
  evaluate_pose_standardization.py
  evaluate_gaze_behavior.py
```

## Development Rules

Before committing:

```bash
git status
git diff
```

Commit message suggestion:

```bash
git add docs scripts eval
git commit -m "Add downstream condition dataset and evaluation skeleton"
git push
```

Then open a PR against `main`.

PR title:

```text
Add downstream condition dataset and evaluation skeleton
```

PR description:

```text
This PR prepares downstream work after Phase2 by adding:
- condition dataset schema
- 3D condition design notes
- ablation plan
- gaze disentanglement design
- condition dataset builder skeleton
- identity / pose / gaze evaluation skeletons

No model training is included.
No Phase2 core logic is modified.
No large data or generated outputs are committed.
```

## Success Criteria

This task is complete when:

- The downstream dataset schema is clear.
- The condition design explains how DECA / Phase2 / ArcFace / L2CS will feed future models.
- The dataset builder skeleton can read manifests and write JSONL placeholders.
- Evaluation scripts can run in dry-run / placeholder mode and produce structured empty outputs.
- The ablation plan clearly separates Phase2, image baseline, ControlNet, identity, and gaze variants.
- No Phase2 core code or large artifacts are modified.
