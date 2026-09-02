# CURRENT STATUS

Analysis date: 2026-08-07

## Current Stage

Current effective stage: Stage 6, learned DECA parameter-space condition generator, with Stage 1-4 feature pipeline documented as complete on the Windows 5060 machine.

Important caveat: In the current Mac workspace, full data/results/checkpoints are not physically present. A 2026-08-07 SSH audit confirmed that the Windows 5060 machine does contain the major recovered data/results/checkpoints described in `docs/STATUS.md`; see `5060_REMOTE_AUDIT.md`.

## Completion Estimate

Estimated completion toward the full title goal, including latent diffusion and gaze disentanglement: 30-40%.

Estimated completion toward a narrower DECA-parameter-space standardization project: 65-75%, based on documented 5060 recovery status.

## Why

- The repository has robust DECA integration, Phase1 extraction tooling, ArcFace/L2CS tools, Phase2 training/inference code, baseline code, and recovery documentation.
- `docs/PHASE1_PARITY.md` documents 10K source coverage, 10K DECA, 10K L2CS, and 9,990 ArcFace success on 5060.
- `docs/STATUS.md` documents recovered Phase2 stage 1/2/3 checkpoints and inference outputs on 5060.
- Current local filesystem does not include the full `results/`, `archive/`, recovered checkpoints, or 10K manifests, but these were verified on 5060.
- The recovered Phase2 inference outputs cover 10K samples but classify almost all samples as `reject`: Stage1 9,954 reject, Stage2 9,964 reject, Stage3 9,991 reject. A follow-up check confirmed these are stale BUG-003-era artifacts generated before the landmark coordinate fix; see `BUG003_ARTIFACT_CHECK.md`.
- No concrete latent diffusion / Stable Diffusion / ControlNet training code was found.
- No explicit gaze disentanglement model or image-level generation pipeline was found.

## Completed

- ✅ Project documentation and recovery memory exist under `docs/`.
- ✅ DECA third-party runtime exists under `DECA/`.
- ✅ DECA modern runtime notes exist in `DECA/RUNNING_MODERN.md`.
- ✅ DECA sample inputs and sample output artifacts exist under `DECA/TestSamples/`.
- ✅ DECA/FLAME support files exist locally under `DECA/data/`.
- ✅ Phase2 package exists with training, inference, augmentation, baseline, comparison, and visualization scripts.
- ✅ Phase2 MLP `ConditionGenerator` is implemented.
- ✅ Hard-zero baseline generator is implemented.
- ✅ Quality manifest builder is implemented.
- ✅ XGBoost quality classifier script is implemented.
- ✅ ArcFace extraction tooling exists.
- ✅ L2CS gaze extraction tooling exists.
- ✅ Phase1 parity completion is documented for 5060: 10,000 images, 10,000 DECA, 10,000 L2CS, 9,990 ArcFace.
- ✅ Recovered Phase2 stage 1/2/3 checkpoints and inference outputs are documented for 5060.
- ✅ 2026-08-07 SSH audit verified the 5060 artifacts are physically present.
- ✅ Local Phase2 visualization PNG summaries exist in `phase2_visualizations/`.
- ✅ BUG-003 landmark coordinate denormalization is documented as fixed.

## Partially Completed

- 🟡 Dataset inventory: verified on 5060, but full 10K data is not present locally.
- 🟡 DECA outputs: verified 10K on 5060, but only sample DECA outputs are present locally.
- 🟡 Phase1 master manifest: verified as `results/phase1_parity/phase1_master_manifest.csv` on 5060, not present locally.
- 🟡 ArcFace identity: extraction and annotation code exists; full embeddings/manifests are verified on 5060, not local.
- 🟡 L2CS gaze: extraction code exists; full outputs are verified on 5060, not local.
- 🟡 Phase2 training: code and checkpoints exist on 5060; local checkpoint files are absent.
- 🟡 Phase2 inference: full 10K outputs exist on 5060, but they are stale pre-BUG-003-fix outputs and should not be treated as valid post-fix results.
- 🟡 Phase2 evaluation: parameter-norm comparison code exists; image-level identity/gaze/quality evaluation is incomplete.
- 🟡 Rendering verification: sample render outputs exist; full standardized render parity is not verified.
- 🟡 Paper plan: `phase2_deca_standardization_plan.tex` exists, but it describes later losses/ablation not fully implemented.

## Not Completed

- ❌ Latent diffusion training.
- ❌ Stable Diffusion / ControlNet condition injection.
- ❌ Image-level frontal face generation model.
- ❌ Explicit gaze disentanglement training.
- ❌ Differentiable DECA decoder validation.
- ❌ ArcFace image-space identity loss after standardized rendering.
- ❌ Image-level corruption during DECA encoding.
- ❌ Full evaluation benchmark with identity, gaze, perceptual quality, collapse rate, and human scoring.
- ❌ Paper-ready ablation suite.
- ❌ Defined publication target.

## Missing

- Full local 10K raw dataset.
- Full local Phase1 master manifest.
- Full local DECA 10K output directory.
- Full local ArcFace and L2CS output manifests.
- Full local recovered Phase2 checkpoints and inference outputs.
- Train/validation/test split definitions.
- Paired frontal target images.
- Ground-truth gaze labels.
- Identity labels beyond ArcFace proxy embeddings.
- Diffusion dataset format and model code.
- Evaluation benchmark scripts and generated result tables.

## UNKNOWN

- Whether `DECA/data/deca_model.tar` exists locally; README expects it, but current file listing did not show it under `DECA/data/`.
- Exact Phase2 checkpoint hyperparameters for recovered stage 1/2/3 beyond code defaults and documented summaries.
- Whether original non-generated face data existed on the lost 2060 workstation.
- Target venue and expected evaluation standard.
