# 5060 REMOTE AUDIT

Audit date: 2026-08-07

Machine: `win-lenovo`

Project path: `D:\face_standardization_project`

Method: SSH read-only inspection from Mac workspace.

## Connectivity

`win-lenovo` was reachable by SSH. The project directory exists and was last modified on 2026-07-15 22:31:16.

## Key Directory Inventory

| Path | Present | Files | Size |
|---|---:|---:|---:|
| `archive` | YES | 10,000 | 11.279 GB |
| `archive\generated_yellow-stylegan2` | YES | 10,000 | 11.279 GB |
| `results` | YES | 90,158 | 24.553 GB |
| `results\phase1_parity` | YES | 6 | 0.003 GB |
| `results\gaze_10k_l2cs_rebuilt` | YES | 10,006 | 0.002 GB |
| `results\arcface_p95_rebuilt` | YES | 19,987 | 1.796 GB |
| `results\phase2_hard_zero_recovered` | YES | 10,002 | 0.037 GB |
| `results\phase2_real_train_stage1_recovered` | YES | 4 | <0.001 GB |
| `results\phase2_real_train_stage2_recovered` | YES | 4 | <0.001 GB |
| `results\phase2_real_train_stage3_recovered` | YES | 4 | <0.001 GB |
| `results\phase2_real_infer_stage1_recovered` | YES | 10,002 | 0.047 GB |
| `results\phase2_real_infer_stage2_recovered` | YES | 10,002 | 0.047 GB |
| `results\phase2_real_infer_stage3_recovered` | YES | 10,002 | 0.047 GB |
| `results\screening_p95` | YES | 10,008 | 11.282 GB |
| `results\screening_p975` | YES | 10,008 | 11.282 GB |

## Phase 1 Parity Files

`results\phase1_parity` contains:

- `phase1_master_manifest.csv`：3,397,894 bytes
- `phase1_master_summary.json`：398 bytes
- `finalizer.log`：13,452 bytes
- `finalizer_state.json`
- `processes.json`
- `arcface_retry_ids.txt`

`phase1_master_summary.json` reports:

| Metric | Value |
|---|---:|
| total_images | 10,000 |
| unique_image_ids | 10,000 |
| eye_invalid | 10 |
| p95 Pass / Warn / Missing | 9,500 / 500 / 0 |
| p97.5 Pass / Warn / Missing | 9,750 / 250 / 0 |
| DECA success | 10,000 |
| L2CS success | 10,000 |
| ArcFace success | 9,990 |
| strict_train | 9,482 |
| full_train | 9,499 |
| hashes_recorded | true |

Manifest columns observed:

`image_id,image_path,image_sha256,eye_valid,p95_label,p95_D2,p975_label,p975_D2,deca_status,deca_mat_path,l2cs_status,pitch,yaw,gaze_x,gaze_y,gaze_z,arcface_status,arcface_embedding_path,arcface_detector_score,arcface_stage,use_for_train_strict,use_for_train_full`

## DECA Assets and Outputs

`DECA\results\archive_phase2_params` exists with 30,000 files and size 1.245 GB.

Key DECA assets exist:

| File | Size |
|---|---:|
| `DECA\data\deca_model.tar` | 434,142,943 bytes |
| `DECA\data\generic_model.pkl` | 53,023,716 bytes |
| `DECA\data\FLAME_albedo_from_BFM.npz` | 1,258,291,694 bytes |
| `DECA\data\landmark_embedding.npy` | 31,292 bytes |

## Phase2 Training Checkpoints

Each recovered training directory contains:

- `best_model.pt`
- `normalizer.npz`
- `train_history.csv`
- `train_summary.json`

| Stage | Samples | Train | Val | Input Dim | best_val_loss | Checkpoint Size |
|---|---:|---:|---:|---:|---:|---:|
| Stage 1 | 10,000 | 8,500 | 1,500 | 99 | 0.17634443565209706 | 438,709 bytes |
| Stage 2 | 10,000 | 8,500 | 1,500 | 99 | 0.17655711909135183 | 438,709 bytes |
| Stage 3 | 10,000 | 8,500 | 1,500 | 99 | 0.17615941083431244 | 438,709 bytes |

## Phase2 Inference Status

All recovered inference directories contain 10,000 inferred parameter outputs plus summary and manifest files.

| Run | Count | Standardize | Weak Standardize | Reject |
|---|---:|---:|---:|---:|
| stage1_recovered | 10,000 | 46 | 0 | 9,954 |
| stage2_recovered | 10,000 | 34 | 2 | 9,964 |
| stage3_recovered | 10,000 | 8 | 1 | 9,991 |
| hard_zero | 10,000 | N/A | N/A | N/A |

This is the most important caution from the audit: the recovered Phase2 inference outputs exist and cover 10K samples, but default decision thresholds classify almost all samples as reject. A follow-up BUG-003 artifact check shows these outputs are stale pre-fix artifacts generated from a manifest with `landmark_score=0`; see `BUG003_ARTIFACT_CHECK.md`.

## Phase2 Comparison Table

`results\phase2_recovered_compare_full\standardization_comparison_table.csv` exists.

Observed means:

| Run | Exp Std Mean | Exp Ratio Mean | Head Std Mean | Head Ratio Mean | Jaw Std Mean | Jaw Ratio Mean |
|---|---:|---:|---:|---:|---:|---:|
| hard_zero | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 | 0.00000000 |
| stage1_recovered | 0.00944876 | 0.03481680 | 0.00952366 | 0.02876837 | 0.01401672 | 0.31234879 |
| stage2_recovered | 0.01085485 | 0.03961066 | 0.01427955 | 0.04220080 | 0.01434579 | 0.32197526 |
| stage3_recovered | 0.01028380 | 0.03762060 | 0.01685440 | 0.04947001 | 0.01185411 | 0.26746548 |

These comparison outputs are parameter-norm summaries, not image-level identity/gaze/render evaluations.

## Screening Summary

`tools\screening_p975_p95_summary.json` reports:

| Metric | Value |
|---|---:|
| total | 10,000 |
| quality_min | 0.30629251028973653 |
| quality_max | 0.4526018476498248 |
| quality_mean | 0.3815655506644198 |
| quality_std | 0.016590682058944893 |
| p2.5 cutoff | 0.3472079935659363 |
| p5 cutoff | 0.35368989800609124 |
| p97.5 PASS/WARN | 9,750 / 250 |
| p95 PASS/WARN | 9,500 / 500 |

## Remote Git Status

The 5060 worktree is dirty. Observed changes include:

Modified:

- `.gitignore`
- `docs/LOG.md`
- `docs/NEXT.md`
- `docs/PIPELINE.md`
- `docs/RECOVERY.md`
- `docs/STATUS.md`
- `tools/extract_arcface_embeddings.py`

Untracked:

- `configs/`
- `docs/PHASE1_PARITY.md`
- `tools/benchmark_screening_thresholds.py`
- `tools/build_phase1_master_manifest.py`
- `tools/finalize_phase1_parity.py`
- `tools/run_l2cs_batch.py`
- `tools/run_l2cs_scheduled.ps1`
- `tools/run_phase1_finalizer_scheduled.ps1`
- `tools/schedule_phase1_parity.ps1`
- `tools/screening_p975_p95_summary.json`

## Updated Interpretation

The 5060 machine does contain the major recovered assets described in project documentation. The earlier Mac-local caution remains valid only for the Mac workspace, not for the full project state.

Current reliable statement:

- Phase 1 artifact recovery/parity is materially complete on 5060.
- Phase2 parameter-space training and inference artifacts exist on 5060.
- Phase2 recovered inference is not yet scientifically sufficient because almost all samples are rejected from stale BUG-003-era quality scores, and evaluation is parameter-level rather than rendered/image-level.
- Diffusion/control/gaze-disentanglement stages are still not implemented in the inspected codebase.
