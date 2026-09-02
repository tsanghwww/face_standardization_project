# PROJECT OVERVIEW

Project: 基于 3D 融合控制与隐空间扩散模型的人脸标准化重构及视线解耦研究

Analysis date: 2026-08-07

Scope: Read-only project restart analysis based on files currently present in `/Users/houwingtsang/Documents/face_standardization_project`. Historical status recorded in `docs/` is treated as evidence, but distinguished from artifacts physically present in this Mac workspace.

## 1. Main Modules

| Module | Purpose | Main Paths | Current Evidence |
|---|---|---|---|
| Project memory / recovery docs | Preserve recovered context, status, decisions, dataset inventory, next actions | `README.md`, `docs/*.md`, `.openclaw/` | Present |
| DECA 3D reconstruction runtime | Extract FLAME/DECA parameters, keypoints, meshes/renders from 2D face images | `DECA/`, `DECA/decalib/`, `DECA/demos/`, `DECA/RUNNING_MODERN.md` | Present; sample outputs exist |
| DECA/FLAME model assets | Runtime assets for DECA inference | `DECA/data/` | Present locally: `generic_model.pkl`, FLAME support arrays/images. README says large assets are normally not tracked, but this workspace contains several assets. |
| Dataset / source images | Research images for training/evaluation | Historical: `D:\face_standardization_project\archive\generated_yellow-stylegan2\`; local: `single_test_inputs/`, `stress_test_inputs/`, `DECA/TestSamples/` | Full 10K dataset not present locally; documented as present on 5060 |
| Phase 1 feature extraction | Build canonical data manifest from source images, DECA, gaze, identity, cleaning labels | `tools/run_l2cs_batch.py`, `tools/extract_arcface_embeddings.py`, `tools/finalize_phase1_parity.py`, `tools/build_phase1_master_manifest.py`, `docs/PHASE1_PARITY.md` | Code present; full artifacts documented on 5060, not present locally |
| Dataset screening / quality assessment | Filter or label samples by quality thresholds and DECA/ArcFace/L2CS signals | `tools/screen_percentile.py`, `tools/screening_deca_params.py`, `phase2/build_manifest.py`, `phase2/train_xgboost_quality.py` | Code present; visual summaries present locally; full manifests/checkpoints documented on 5060 |
| Phase2 parameter-space standardization | Learn a quality-aware condition generator that standardizes DECA expression and pose parameters | `phase2/model.py`, `phase2/train_condition_generator.py`, `phase2/infer_standardize_params.py`, `phase2/augmentation.py`, `phase2/baseline_hard_zero.py` | Code present; recovered training/inference artifacts documented on 5060, not present locally |
| Visualization / comparison | Compare hard-zero vs learned parameter standardization | `phase2/make_visualizations.py`, `phase2/compare_standardization_runs.py`, `phase2/render_single_comparison.py`, `phase2_visualizations/` | Code and several local PNG summaries present |
| Evaluation | Identity, gaze, landmark, pose/canonical, collapse/reject evaluation | Partly in `tools/`, `phase2/compare_standardization_runs.py`, `phase2_deca_standardization_plan.tex` | Mostly design/partial code; no full image-level benchmark artifact locally |
| Latent diffusion / Stable Diffusion / ControlNet | Image-level generative frontalization and 3D-controlled latent diffusion | No concrete source module found | Design-level only / missing |
| Gaze disentanglement model | Explicit decoupling of gaze from identity and head pose | L2CS extraction scripts and manifest fields only | Feature extraction exists; disentanglement model/training missing |
| Paper/report | Research narrative, literature, ablation plan | `phase2_deca_standardization_plan.tex`, `docs/PAPERS.md` | Early plan exists; paper-grade experiments missing |

## 2. Which Module Does the Current Code Correspond To?

The current original project code corresponds primarily to:

1. DECA modernization and parameter extraction.
2. Phase 1 feature reconstruction: DECA, L2CS gaze, ArcFace identity, cleaning labels, master manifest.
3. Phase2 DECA parameter-space standardization: an MLP condition generator over DECA parameters and quality metrics.
4. Quality-aware sample screening: heuristic manifest plus XGBoost quality classifier.
5. Comparison/visualization of parameter-space standardization outputs.

It does not currently contain a latent diffusion training codebase, a Stable Diffusion pipeline, a ControlNet implementation, a UNet/VAE training loop, or a complete image-generation evaluation stack.

## 3. Existing Modules

Existing in current workspace:

- DECA codebase and modern runtime notes.
- DECA sample test inputs and sample inference outputs.
- DECA/FLAME support assets under `DECA/data/`.
- Phase2 package with dataset, augmentation, MLP model, training, inference, baseline, comparison, visualization.
- Tools for L2CS gaze extraction, ArcFace embedding extraction, DECA auditing, screening, parity finalization, and master manifest construction.
- Project memory documents under `docs/`.
- Local visualization PNGs under `phase2_visualizations/`.

Documented as existing on 5060 but not physically present in this Mac workspace:

- 10,000 StyleGAN2 generated face images.
- `results/phase1_parity/phase1_master_manifest.csv`.
- Full L2CS 10K outputs.
- Full ArcFace 9,990 outputs.
- Phase2 stage 1/2/3 recovered checkpoints and inference outputs.
- p95/p97.5 screening output directories.
- Hard-zero baseline full 10K outputs.

## 4. Designed But Not Fully Implemented

- Differentiable DECA decoder validation.
- ArcFace image-space identity loss after rendering standardized outputs.
- Image-level corruption during DECA encoding.
- Full evaluation pipeline with ArcFace, LPIPS/FID-style quality, gaze consistency, landmark reprojection, reject precision, collapse rate, and human visual scoring.
- Paper-level ablation groups from `phase2_deca_standardization_plan.tex`.
- Dataset split definitions and reproducible experiment configuration registry.

## 5. Missing Modules

- Latent diffusion / Stable Diffusion / ControlNet code.
- Image-space generator for final high-quality standardized frontal faces.
- Explicit gaze disentanglement model or loss.
- Paired input-to-frontal training set or alternative self-supervised protocol for frontalization.
- Ground-truth frontal face labels.
- Identity labels beyond ArcFace embeddings/proxy success flags.
- Gaze labels beyond L2CS-estimated yaw/pitch/vector.
- Formal evaluation benchmark and final paper experiment suite.
- Local copies of full data/results needed to rerun or verify Phase2 from this Mac workspace.

## Current Interpretation

This project is currently best understood as a recovered DECA-parameter-space standardization project with Phase 1 feature parity documented and a Phase2 learned condition generator implemented. The larger research title mentions latent diffusion and 3D fusion control, but those parts are not yet represented by concrete implementation in the repository.
