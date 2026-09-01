# Phase2.1 Scope Decision and Next Step

Date: 2026-08-31

## 1. Decision

Phase2.1 will not continue expanding external difficult face datasets as a required condition for this undergraduate project.

The current project will treat dataset diversity as an explicit experimental limitation. Phase2.1 remains valid as an engineering framework for outcome-supervised standardization, but its current Gate result should not be described as deployment-ready or scientifically conclusive.

This decision is based on three constraints:

- The existing Kaggle/base training set is too clean and standardized.
- Additional large-pose, occlusion, blur, and low-landmark-quality data collection requires more data engineering than is realistic for the current project scope.
- Current Gate discrimination is weak, so simply collecting a small number of extra samples is unlikely to fully solve the reliability problem.

## 2. What Can Be Claimed

The project can claim the following:

- A manifest-centered DECA/ArcFace/L2CS/XGBoost evaluation pipeline was built and made reproducible.
- Phase2 v1 completed fixed-split training, ablation, rendering, rescue sensitivity analysis, Gate calibration, final figures, and artifact hashes.
- Learned Phase2 standardization produced small but measurable average improvements over hard-zero on identity cosine, head pose, and gaze diagnostics.
- Rescue preprocessing increases technical coverage but introduces large domain shift, so it is kept as an independent audit or manual-review path.
- Phase2.1 correctly reframes the problem from predicting standardized parameters to supervising actual outcomes: identity preservation, pose improvement, gaze behavior, and render failure.
- Phase2.1 includes guardrails against validation leakage, fixed-test threshold tuning, rescue leakage, missing-label zero filling, and label-dependent Gate decisions.

## 3. What Cannot Be Claimed

The project should not claim the following:

- The current Gate can safely decide which samples should be standardized in deployment.
- Phase2.1 has fully solved robustness under large pose, occlusion, blur, or low landmark quality.
- Rescue outputs can replace FAN-main outputs in the primary evaluation.
- Gaze disentanglement has been achieved. The current system measures gaze-related behavior but does not control gaze with ground-truth supervision.
- The downstream generator can rely on the current Gate as a reliable safety filter.

## 4. Thesis Limitation Statement

Recommended wording for the thesis:

> A limitation of this project is that the available training data is dominated by relatively clean and standardized face images. Although external hard samples from WIDER Face, COFW Color, and AFLW2000-3D were introduced for evaluation, the project does not contain enough diverse hard samples to train a robust outcome-supervised Gate. Therefore, Phase2.1 is reported as an engineering and methodological extension rather than a fully validated deployment mechanism. Future work should expand the hard-sample training pool and collect more positive examples of rendering failure, severe occlusion, large head pose, and low landmark quality.

## 5. Current Phase Position

Phase2 v1 is complete as a reproducible experiment.

Phase2.1 is complete as an engineering skeleton and protocol guardrail, but incomplete as a fully validated scientific model.

The project may proceed to the next stage only under this wording:

> Phase3/downstream work is exploratory and must consume Phase2 outputs with conservative filtering. It should not claim that Phase2.1 Gate has passed deployment-level safety validation.

## 6. Next Step

The next practical step is to prepare the downstream condition-dataset and evaluation interface, without starting full image-generation training yet.

This means building:

- A downstream condition dataset schema.
- A manifest builder that joins source image, DECA parameters, Phase2 standardized parameters, ArcFace identity data, gaze data, and quality metadata.
- Evaluation interfaces for identity preservation, pose standardization, and gaze/head-pose disentanglement.
- A future ablation plan for comparing hard-zero, Phase2 parameters, condition maps, and identity/gaze conditioning.

This step is suitable for the current project because it does not require collecting more external data immediately, and it turns the completed Phase2 outputs into a reusable interface for future work.

## 7. Immediate Task List

1. Create `docs/CONDITION_DATASET_SCHEMA.md`.
2. Create `docs/CONDITION_DESIGN.md`.
3. Create `scripts/build_condition_dataset.py` as a dry-run manifest builder.
4. Create evaluation entry points under `eval/`:
   - `eval/evaluate_identity_preservation.py`
   - `eval/evaluate_pose_standardization.py`
   - `eval/evaluate_gaze_behavior.py`
5. Create `docs/ABLATION_PLAN.md`.
6. Create `docs/GAZE_DISENTANGLEMENT_DESIGN.md`.
7. Keep Phase2.1 Gate as diagnostic until a future dataset expansion can support stronger calibration.

## 8. Go / No-Go Rule

Go:

- Start downstream interface and evaluation skeletons.
- Use current Phase2 outputs for exploratory manifests.
- Keep all safety claims conservative.
- After the Phase3.0 condition/coordinate gate passes, start a bounded frozen-backbone adapter pilot defined in `PHASE3_LATENT_DIFFUSION_TRAINING_PLAN.md`.

No-go:

- Do not train a full diffusion backbone from scratch or start a formal Phase3 run before the Phase3.0 gate passes.
- Do not merge rescue outputs into the primary training/evaluation path.
- Do not tune Gate thresholds on the 775 fixed test set.
- Do not describe current Phase2.1 Gate as deployment-qualified.
