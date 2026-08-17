# RESEARCH PIPELINE

Analysis date: 2026-08-07

This pipeline is organized by research lifecycle rather than code directory.

## Stage 0: Research Problem Definition

Research Objective: Define the scientific claim: can 3D face priors and learned conditioning standardize arbitrary-pose faces while preserving identity and disentangling gaze?

Why This Stage Is Needed: The current repository contains a strong Phase1/Phase2 engineering base, but the final paper claim still mixes DECA parameter-space standardization with latent diffusion image generation.

Input: Literature review, project title, `phase2_deca_standardization_plan.tex`, `docs/PAPERS.md`, prior experiment notes.

Output: A precise research question, contribution list, task definition, assumptions, and target venue constraints.

Required Data: None directly.

Required Code: None directly.

Metrics: Clarity of research question, falsifiable hypotheses, defined baselines and evaluation protocol.

Completion Standard: A one-page research brief specifying whether the paper is about parameter-space standardization, image-level frontalization, gaze disentanglement, or a staged combination.

Common Risks: Overclaiming latent diffusion before implementation exists; conflating head-pose normalization with gaze disentanglement.

## Stage 1: Dataset Inventory and Preparation

Research Objective: Establish the usable training/evaluation population and preserve sample identity across all derived artifacts.

Why This Stage Is Needed: The current Mac workspace does not contain the full 10K dataset; documents say it exists on 5060. Reproducibility depends on confirming paths, hashes, labels, and splits.

Input: Raw face images, historical eye-invalid IDs, cleaning labels, dataset backup.

Output: Canonical source manifest with `image_id`, source path, SHA256, cleaning label, split, and exclusion flags.

Required Data: 10,000 StyleGAN2 images documented at `D:\face_standardization_project\archive\generated_yellow-stylegan2\`; local smoke-test images only.

Required Code: `tools/build_phase1_master_manifest.py`, `tools/finalize_phase1_parity.py`, screening utilities.

Metrics: Source image count, unique ID count, hash coverage, split coverage, invalid/excluded sample accounting.

Completion Standard: Exactly 10,000 unique source IDs with hashes and non-destructive exclusion flags; train/val/test splits defined.

Common Risks: Local/remote artifact drift, duplicate screening copies, missing split definitions, treating generated-only data as sufficient for in-the-wild claims.

## Stage 2: 3D Face Representation Extraction

Research Objective: Convert each 2D image into structured 3D priors: shape, expression, pose, camera, lighting, detail, keypoints, and optional mesh/renders.

Why This Stage Is Needed: DECA/FLAME parameters are the current project's main controllable representation.

Input: Source images and DECA assets.

Output: Per-image `.mat` or `.npz`, `*_kpt2d.txt`, `*_kpt3d.txt`, optional OBJ/depth/normal/render outputs.

Required Data: Source images; `DECA/data/deca_model.tar` if available on training machine; FLAME assets.

Required Code: `DECA/decalib/deca.py`, `phase2/run_deca_batch_params.py`, `DECA/demos/demo_reconstruct.py`, `tools/audit_deca_outputs.py`.

Metrics: DECA success rate, keypoint validity, parameter completeness, render availability, reconstruction sanity checks.

Completion Standard: Full manifest links every image ID to DECA status and parameter path; failures have reasons.

Common Risks: Renderer incompatibility across Mac/Windows/CUDA, normalized vs pixel landmark coordinate mismatch, missing DECA weights, path reconstruction without manifest evidence.

## Stage 3: Auxiliary Feature Extraction

Research Objective: Extract non-DECA signals for identity, gaze, and quality control.

Why This Stage Is Needed: Identity preservation and gaze disentanglement require measurements independent of DECA parameter norms.

Input: Source RGB images and Phase1 manifest.

Output: ArcFace embeddings/success flags, L2CS gaze pitch/yaw/vector, cleaning labels, quality features.

Required Data: Eye-valid source images; L2CS checkpoint; InsightFace model package.

Required Code: `tools/extract_arcface_embeddings.py`, `tools/run_l2cs_batch.py`, `tools/annotate_arcface_manifest.py`, `tools/merge_arcface_retry.py`.

Metrics: ArcFace coverage, L2CS coverage, detection thresholds, retry provenance, gaze extraction success rate.

Completion Standard: Joined master manifest with DECA, gaze, identity, cleaning, and hash fields for all 10K IDs.

Common Risks: Rebuilt gaze values not byte-identical to lost historical outputs, ArcFace detection failure on eye-invalid samples, leakage if quality labels are reused as features.

## Stage 4: Quality Screening and Sample Weighting

Research Objective: Separate reliable samples from risky samples and generate training weights/quality gates.

Why This Stage Is Needed: The Phase2 model depends on confidence-aware standardization; low-quality inputs should be weakly standardized or rejected.

Input: DECA parameters, keypoint metrics, ArcFace data, cleaning labels.

Output: Heuristic Phase2 manifest; optional XGBoost quality manifest with score, label, sample weight, strong/weak train flags.

Required Data: Phase1 joined manifest and DECA outputs.

Required Code: `phase2/build_manifest.py`, `phase2/train_xgboost_quality.py`, `tools/screen_percentile.py`, `tools/screening_deca_params.py`.

Metrics: Quality class distribution, train/validation quality split, XGBoost validation metrics, leakage audit.

Completion Standard: A stable quality manifest exists and can be used by `phase2.train_condition_generator`.

Common Risks: Proxy labels may not match visual quality; leakage from screening labels; thresholds tuned without held-out validation.

## Stage 5: Baseline Parameter Standardization

Research Objective: Establish a simple lower-bound method: hard-zero expression and pose.

Why This Stage Is Needed: It is the minimum baseline for DECA parameter-space standardization and an ablation anchor.

Input: DECA `.mat` outputs.

Output: Standardized parameter `.npz` files and `hard_zero_manifest.csv`.

Required Data: DECA outputs.

Required Code: `phase2/baseline_hard_zero.py`.

Metrics: Expression/pose residual norms, render stability, identity consistency after rendering, failure rate.

Completion Standard: Baseline generated for the same IDs and split as learned Phase2 runs.

Common Risks: Hard-zero may look acceptable on overly canonical generated data but fail on real-world data; no image-space validation if not rendered.

## Stage 6: Learned DECA Parameter-Space Condition Generator

Research Objective: Learn quality-aware partial standardization of expression, head pose, and jaw pose while keeping identity-related parameters fixed.

Why This Stage Is Needed: This is the current implemented research contribution candidate.

Input: DECA params, quality features, optional ArcFace/XGBoost manifest.

Output: `best_model.pt`, `normalizer.npz`, `train_history.csv`, `train_summary.json`, inference `.npz`, inference manifest and summary.

Required Data: DECA outputs and quality manifests.

Required Code: `phase2/model.py`, `phase2/dataset.py`, `phase2/augmentation.py`, `phase2/train_condition_generator.py`, `phase2/infer_standardize_params.py`.

Metrics: Training/validation loss, expression/pose residual ratio, reject score behavior, confidence calibration, render sanity.

Completion Standard: Stage 1/2/3 checkpoints load and produce reproducible outputs; outputs are benchmarked against hard-zero.

Common Risks: Loss is proxy-based and may reduce parameter norm without improving image quality; no direct identity loss currently implemented; no final diffusion image output.

## Stage 7: Rendered Output Validation

Research Objective: Verify whether standardized parameters produce valid rendered faces/geometry.

Why This Stage Is Needed: Parameter-space improvement is not enough; outputs must render correctly and preserve face identity.

Input: Original DECA outputs, baseline and Phase2 standardized params, DECA decoder/renderer.

Output: Rendered standardized faces/meshes, comparison panels, render manifest, failure report.

Required Data: DECA assets, parameter outputs, source image links.

Required Code: `phase2/render_single_comparison.py`, `phase2/visualize_single_comparison.py`, DECA renderer.

Metrics: Render success rate, landmark error, visual collapse rate, identity similarity if ArcFace can run on renders.

Completion Standard: Full 10K or representative held-out render verification with manifest-backed paths.

Common Risks: Renderer environment mismatch, no automated visual quality metric, one-off screenshots mistaken for full validation.

## Stage 8: Image-Level Generative Model / 3D Control Injection

Research Objective: Use 3D priors as conditioning signals for high-quality frontal face generation.

Why This Stage Is Needed: This stage is required to satisfy the full project title involving latent diffusion and ControlNet-like conditional generation.

Input: Source images, standardized 3D conditions, identity embeddings, gaze features, target frontal/standard conditions.

Output: Generated standardized frontal face images and trained diffusion/control checkpoints.

Required Data: Paired or self-supervised training protocol, image-condition pairs, target conditions, evaluation split.

Required Code: Currently missing: latent diffusion dataset, VAE/UNet/ControlNet integration, conditioning encoder, training loop, sampler.

Metrics: Identity similarity, gaze/head-pose disentanglement, perceptual quality, FID/KID if appropriate, pose/gaze error, human preference.

Completion Standard: Reproducible training and inference producing image-level standardized faces on held-out data.

Common Risks: Insufficient data, identity drift, diffusion overfitting, unclear supervision for frontal ground truth, conflating gaze with head pose.

## Stage 9: Gaze Disentanglement Evaluation

Research Objective: Demonstrate that gaze direction can be controlled or neutralized independently from head pose and identity.

Why This Stage Is Needed: L2CS extraction alone is not disentanglement; the claim needs controlled evidence.

Input: Original images, generated/standardized outputs, gaze estimates, head pose estimates, identity embeddings.

Output: Gaze disentanglement benchmark and failure cases.

Required Data: Gaze labels or estimated gaze with validated reliability; ideally multi-gaze identities or synthetic controlled data.

Required Code: Gaze evaluation scripts beyond extraction; disentanglement losses or conditioning controls.

Metrics: Gaze angular error, head-pose error, identity cosine, gaze/pose correlation reduction, identity failure rate.

Completion Standard: Ablation shows gaze change/neutralization without identity degradation or pose leakage.

Common Risks: L2CS pseudo-label noise, no ground truth gaze, generated eyes may fool detector, missing paired identity/gaze variation.

## Stage 10: Evaluation, Ablation, and Paper

Research Objective: Convert the engineering system into a defensible research result.

Why This Stage Is Needed: Paper claims require baselines, ablations, metrics, reproducibility, and failure analysis.

Input: Outputs from Stages 5-9.

Output: Tables, figures, ablation report, final manuscript/report.

Required Data: Held-out evaluation split, baselines, generated outputs, human review samples if used.

Required Code: Evaluation scripts, table generation, visualization scripts, reproducibility scripts.

Metrics: Identity preservation, pose/frontalization, gaze disentanglement, image quality, robustness, runtime, ablation deltas.

Completion Standard: Every central claim has at least one metric table and one qualitative figure; code/results are traceable by manifest and checkpoint hash.

Common Risks: Missing baselines, cherry-picked images, undocumented recovered outputs, no real-data validation.
