# STAGE GAP ANALYSIS

Analysis date: 2026-08-07

| Stage | Expected Output | Existing | Missing | Priority |
|---|---|---|---|---|
| Stage 0: Research Problem Definition | Research question, hypotheses, contribution list, target task scope | Project title, `phase2_deca_standardization_plan.tex`, `docs/PAPERS.md`, `docs/DECISIONS.md` | Precise claim boundary between parameter-space standardization, diffusion generation, and gaze disentanglement | High |
| Stage 1: Dataset Inventory and Preparation | Raw-image manifest, hashes, labels, splits, invalid flags | Documented 10K StyleGAN2 dataset on 5060; local smoke-test images; `configs/phase1_eye_invalid_ids.txt` | Full local dataset, split files, confirmation of 5060 availability, real-data benchmark | High |
| Stage 2: 3D Face Representation Extraction | DECA params, keypoints, optional renders/meshes, reconstruction QC | DECA code/assets; sample outputs; documented 10K DECA success on 5060 | Local 10K DECA outputs; full render/QC manifest in current workspace | High |
| Stage 3: Auxiliary Feature Extraction | ArcFace embeddings, L2CS gaze vectors, cleaning labels, joined manifest | ArcFace/L2CS tools; documented 10K L2CS and 9,990 ArcFace on 5060 | Local full outputs; verified master manifest file; gaze reliability validation | High |
| Stage 4: Quality Screening and Sample Weighting | Heuristic/XGBoost quality manifest, sample weights, label distribution | `phase2/build_manifest.py`, `phase2/train_xgboost_quality.py`, local visualization PNGs, documented p95/p97.5 | Local XGBoost model/manifest, leakage audit, held-out validation summary | High |
| Stage 5: Baseline Parameter Standardization | Hard-zero `.npz`, manifest, baseline metrics | `phase2/baseline_hard_zero.py`; documented 10K recovered baseline on 5060 | Local hard-zero outputs; rendered identity/gaze evaluation | Medium |
| Stage 6: Learned Parameter-Space Condition Generator | Stage 1/2/3 checkpoints, normalizer, training logs, inference manifests | Implemented model/training/inference; documented recovered checkpoints/inference on 5060 | Local checkpoints; exact configs; post-BUG-003 retrain confirmation; reproducibility run | High |
| Stage 7: Rendered Output Validation | Rendered standardized faces/meshes, comparison panels, render failure manifest | Sample render outputs; `phase2/render_single_comparison.py`, `phase2/visualize_single_comparison.py` | Full standardized render parity; identity metrics on rendered outputs; collapse-rate report | High |
| Stage 8: Image-Level Generative Model / 3D Control Injection | Latent diffusion/control model, condition encoder, generated frontal images | No implementation found | Dataset format, model code, training loop, sampler, checkpoints, generated outputs | High for full title, Low for immediate recovery |
| Stage 9: Gaze Disentanglement Evaluation | Gaze-control protocol, metrics, ablation against head pose and identity | L2CS extraction tooling and documented gaze outputs | Disentanglement model/loss, ground-truth or validated pseudo-label protocol, metric scripts | High for full title |
| Stage 10: Evaluation, Ablation, and Paper | Quantitative tables, qualitative figures, ablation matrix, manuscript | Plan-level ablation table in TeX; comparison scripts | Complete metrics, baselines, final figures, paper draft, target venue formatting | Medium |

## Highest-Priority Gaps

1. Confirm and mirror the 5060 canonical artifacts: raw dataset, Phase1 master manifest, DECA outputs, ArcFace/L2CS outputs, Phase2 checkpoints/inference.
2. Re-establish reproducible Stage 5/6 runs from manifest-backed data.
3. Build rendered output validation before any final-model or diffusion work.
4. Define whether the next paper claim is Phase2 parameter-space standardization or full diffusion-based frontalization/gaze disentanglement.
