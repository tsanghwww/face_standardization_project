# DATA REQUIREMENT

Analysis date: 2026-08-07

## Existing Data in Current Mac Workspace

| Data | Path | Status | Use |
|---|---|---|---|
| DECA sample images | `DECA/TestSamples/` | Present | DECA smoke tests only |
| Single test image | `single_test_inputs/` | Present | Quick inference testing |
| Stress test image | `stress_test_inputs/` | Present | Small stress test only |
| DECA sample outputs | `DECA/TestSamples/examples/results*` | Present | Runtime validation and visual sanity |
| DECA/FLAME support assets | `DECA/data/` | Partly present locally | DECA runtime support |
| Phase2 visualization PNGs | `phase2_visualizations/` | Present | Historical summary visualization |
| Historical docs | `docs/*.md`, `phase2_deca_standardization_plan.tex` | Present | Recovery context |

## Existing Data Verified on 5060

| Data | Documented Path | Status from Docs | Needed For |
|---|---|---|---|
| 10K StyleGAN2 generated faces | `D:\face_standardization_project\archive\generated_yellow-stylegan2\` | Verified present, 10,000 files | Main training/evaluation |
| Phase1 master manifest | `results/phase1_parity/phase1_master_manifest.csv` | Verified present | Canonical join across source, DECA, gaze, identity |
| Phase1 summary | `results/phase1_parity/phase1_master_summary.json` | Verified present | Audit |
| DECA 10K outputs | `DECA\results\archive_phase2_params` | Verified present, 30,000 files | Phase2 training/inference |
| L2CS 10K gaze outputs | `results/gaze_10k_l2cs_rebuilt` | Verified present, 10,006 files | Gaze metrics |
| ArcFace outputs | `results\arcface_p95_rebuilt` and retry outputs | Verified present | Identity metrics and training flags |
| p95 screening outputs | `results\screening_p95` | Verified present | Canonical quality branch |
| p97.5 screening outputs | `results\screening_p975` | Verified present | Benchmark branch |
| Hard-zero baseline outputs | `results\phase2_hard_zero_recovered` | Verified present, 10,002 files | Baseline |
| Phase2 stage 1/2/3 checkpoints | `results\phase2_real_train_stage{1,2,3}_recovered` | Verified present | Learned standardization |
| Phase2 stage 1/2/3 inference | `results\phase2_real_infer_stage{1,2,3}_recovered` | Verified present, 10,002 files each | Comparison/evaluation |

## Missing Data for Training the Current Phase2 Model Locally

- Full DECA `.mat` outputs for the 10K dataset.
- ArcFace manifest with detector scores and training flags.
- XGBoost quality manifest, if training with `--quality-source xgb` or `blend`.
- Recovered Phase2 checkpoints and normalizers.
- Phase1 master manifest to ensure consistent ID matching.
- Train/validation/test split file.

## Missing Data for Diffusion / ControlNet-Style Training

- Paired or pseudo-paired source-to-standardized image data.
- Target frontal/standard face images or a self-supervised target construction protocol.
- Condition maps or tensors derived from DECA/FLAME, such as depth, normals, landmarks, UV maps, pose maps, or parameter embeddings.
- Identity embeddings linked to both source and generated/target images.
- Gaze labels or validated pseudo-labels for both source and target.
- A dataset class defining image, condition, identity, gaze, and target fields.
- Held-out real-world evaluation data, if the paper claims arbitrary in-the-wild generalization.

## Missing Data for Gaze Disentanglement

- Ground-truth gaze labels, or a validated pseudo-label protocol with known limitations.
- Same-identity multi-gaze samples, synthetic gaze-controlled samples, or another design that separates eye gaze from head pose.
- Generated/standardized output images for gaze re-estimation.
- Evaluation labels for whether gaze should be preserved, neutralized, or controlled.

## Data That Needs Regeneration or Revalidation

- Corrected Phase2 quality manifest after BUG-003 landmark denormalization fix.
- XGBoost quality model and manifest using corrected landmark features.
- Phase2 train/inference outputs if old checkpoints were trained before BUG-003.
- Standardized render outputs for baseline and learned Phase2.
- Identity metrics on rendered standardized outputs.
- Gaze metrics on rendered/generated standardized outputs.
- Compact artifact inventory with counts, hashes, and checkpoint/config hashes.

## Data Integrity Checks Needed Next

1. Verify 5060 still has the documented canonical artifacts.
2. Copy or expose only the minimum required manifests/checkpoints to the Mac workspace for analysis.
3. Confirm every result row is keyed by `image_id`, not inferred from filename patterns.
4. Confirm source hashes match between archive, screening copies, and manifests.
5. Confirm eye-invalid IDs remain represented by flags instead of silent deletion.
6. Confirm p95 and p97.5 branches are not mixed in training/evaluation.
7. Confirm whether checkpoints were trained before or after BUG-003.

## UNKNOWN

- Whether a real non-generated dataset was ever used.
- Whether any paired frontal targets exist outside the repository.
- Whether any human annotations exist for visual quality, identity failure, gaze, or collapse.
- Whether the historical 2060 outputs can be recovered beyond the 5060 rebuild.
