# RECOVERY EXPERIMENT PLAN

Analysis date: 2026-08-07

This plan is intentionally conservative. It recovers and validates existing work before attempting final latent diffusion training.

## Experiment 0: Environment Validation

Purpose: Verify that the current execution environments can load DECA and Phase2 dependencies.

Hypothesis: The Mac workspace can perform code-level checks and small DECA smoke tests; full training likely belongs on 5060.

Required Data: DECA sample images only.

Required Code: `DECA/RUNNING_MODERN.md`, `DECA/demos/demo_reconstruct.py`, `phase2/*`, `tools/smoke_deca_rasterizer.py`.

Expected Result: CLI help commands run; sample DECA params can be produced or existing sample outputs can be read; Phase2 modules import.

Success Criteria: Environment report records Python/PyTorch/device, DECA asset availability, and pass/fail status for sample inference.

## Experiment 1: Artifact and Manifest Parity Check

Purpose: Confirm the actual presence and integrity of 5060 artifacts documented in `docs/STATUS.md` and `docs/PHASE1_PARITY.md`.

Hypothesis: The project can be restarted fastest by using the 5060 canonical `phase1_master_manifest.csv` rather than rebuilding from scratch.

Required Data: 5060 `archive/`, `results/phase1_parity/`, DECA outputs, ArcFace/L2CS outputs, Phase2 recovered outputs.

Required Code: `tools/build_phase1_master_manifest.py`, `tools/finalize_phase1_parity.py`, checksum utilities.

Expected Result: A current artifact inventory with counts, sizes, hashes, paths, and missing files.

Success Criteria: Every documented canonical artifact is either verified present or explicitly marked missing with next recovery action.

## Experiment 2: Data Pipeline Validation

Purpose: Validate the full path from one batch of source images to DECA, ArcFace, L2CS, and joined manifest rows.

Hypothesis: The pipeline still works after recovery and BUG-003 fix.

Required Data: A small fixed subset, for example 20-200 images from the 10K dataset.

Required Code: `phase2/run_deca_batch_params.py`, `tools/extract_arcface_embeddings.py`, `tools/run_l2cs_batch.py`, `phase2/build_manifest.py`.

Expected Result: Sample-level manifest rows with DECA params, keypoint metrics, ArcFace status, gaze vectors, and quality labels.

Success Criteria: At least 95% of the sample subset produces valid joined rows; any failure has a reproducible reason.

## Experiment 3: Baseline Reproduction

Purpose: Recreate or verify hard-zero baseline outputs against the same manifest used by learned Phase2.

Hypothesis: Hard-zero provides a stable baseline but may not preserve identity/render quality under difficult pose/expression.

Required Data: DECA `.mat` outputs for a representative subset or full 10K.

Required Code: `phase2/baseline_hard_zero.py`, `phase2/compare_standardization_runs.py`.

Expected Result: `hard_zero_manifest.csv`, standardized `.npz` files, residual norm summary.

Success Criteria: Baseline outputs cover the chosen evaluation IDs and can be compared against learned Phase2 outputs by `image_id`.

## Experiment 4: Learned Phase2 Checkpoint Revalidation

Purpose: Verify recovered Stage 1/2/3 Phase2 checkpoints and rerun inference on a controlled subset.

Hypothesis: Recovered checkpoints still load and produce consistent standardized parameter outputs.

Required Data: DECA `.mat` subset, ArcFace manifest, optional XGBoost quality manifest, recovered checkpoints.

Required Code: `phase2/infer_standardize_params.py`, `phase2/compare_standardization_runs.py`, `phase2/make_visualizations.py`.

Expected Result: Inference manifests, summaries, and comparison plots for Stage 1/2/3 vs hard-zero.

Success Criteria: Checkpoints load without manual patching; output counts match input counts; residual ratios and decision counts are reproducible.

## Experiment 5: Corrected Quality Manifest Rebuild

Purpose: Rebuild Phase2 quality features after the documented landmark coordinate fix.

Hypothesis: BUG-003 fix changes landmark-derived quality scores enough that old manifests/checkpoints may need retraining or at least revalidation.

Required Data: Full or subset DECA outputs and ArcFace manifest.

Required Code: `phase2/build_manifest.py`, `phase2/train_xgboost_quality.py`, `phase2/features.py`.

Expected Result: Updated heuristic and XGBoost manifests with quality distributions and validation metrics.

Success Criteria: Landmark scores are non-collapsed; high/medium/low distributions are plausible; no leakage feature is used for target prediction.

## Experiment 6: Rendered Output Validation

Purpose: Check whether parameter standardization produces visually valid render outputs.

Hypothesis: Learned partial standardization should reduce collapse or identity damage relative to hard-zero on non-canonical samples.

Required Data: Source images, original DECA outputs, hard-zero outputs, learned Phase2 outputs.

Required Code: DECA renderer, `phase2/render_single_comparison.py`, `phase2/visualize_single_comparison.py`.

Expected Result: Side-by-side render panels and a render failure manifest.

Success Criteria: Render success rate and visual sanity are measured on a representative subset before full-scale evaluation.

## Experiment 7: Identity and Gaze Metric Evaluation

Purpose: Quantify identity preservation and gaze/head-pose changes before any diffusion model work.

Hypothesis: DECA parameter-space standardization should reduce head-pose/expression norms while preserving ArcFace identity; gaze behavior is currently only measurable, not controlled.

Required Data: Rendered standardized images or generated outputs, source images, ArcFace embeddings, L2CS outputs.

Required Code: Existing ArcFace/L2CS tools plus new evaluation scripts.

Expected Result: Identity cosine table, gaze angle table, pose-to-canonical metrics, failure cases.

Success Criteria: Metrics are computed for baseline and learned Phase2 on identical IDs; gaze disentanglement claims remain marked unsupported unless controlled evidence exists.

## Experiment 8: Proposed Method Extension

Purpose: Decide and implement the next genuine research innovation after recovery.

Hypothesis: The current project can either become a strong DECA parameter-space standardization paper or proceed toward a diffusion-control paper, but these require different next steps.

Required Data: Validated Stage 1-7 artifacts.

Required Code: To be decided. For diffusion: dataset class, condition encoder, training loop, sampler, checkpoints. For parameter-space paper: stronger renderer/evaluation/ablation code.

Expected Result: Chosen method branch with minimal reproducible prototype.

Success Criteria: Prototype produces measurable improvement over hard-zero and existing Phase2 baseline on held-out samples.

## Experiment 9: Ablation Suite

Purpose: Test which components matter.

Hypothesis: Quality-aware weighting, partial alphas, latent-space augmentation, and reject gate each contribute measurable stability.

Required Data: Fixed evaluation split.

Required Code: Training configs or switches for hard-zero, no-augmentation, no-XGBoost, no-reject, stage 1/2/3.

Expected Result: Ablation table and qualitative figure grid.

Success Criteria: Every ablation row has identical data split, checkpoint/config hash, and metric outputs.

## Experiment 10: Final Evaluation and Paper Assets

Purpose: Produce paper-ready evidence.

Hypothesis: The final selected method improves standardization robustness while preserving identity.

Required Data: Held-out test set, possibly real-world data beyond StyleGAN2.

Required Code: Evaluation scripts, visualization scripts, table/figure generation.

Expected Result: Final metrics tables, figure panels, failure analysis, reproducibility appendix.

Success Criteria: Each core claim is supported by quantitative and qualitative evidence; unsupported claims are removed or marked future work.

## Do Not Run Yet

- Do not train a final latent diffusion model before Stage 1-7 artifacts are verified.
- Do not claim gaze disentanglement from L2CS extraction alone.
- Do not tune thresholds on the final test set.
- Do not rebuild the entire 10K pipeline until canonical artifacts are confirmed missing or stale.
