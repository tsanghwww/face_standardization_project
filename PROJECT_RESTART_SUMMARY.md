# PROJECT RESTART SUMMARY

Analysis date: 2026-08-07

## 1. Where Is This Project Now?

The project is currently at a recovered DECA/Phase2 parameter-space standardization stage.

Concretely, the repository contains:

- A modernized DECA runtime.
- Tools for DECA, L2CS gaze, ArcFace identity, screening, and master manifest construction.
- A Phase2 MLP condition generator that predicts standardized DECA expression/pose targets, standardization strengths, confidence, and reject score.
- Baseline, training, inference, comparison, and visualization scripts for Phase2.
- Documentation stating that Phase1 parity and Phase2 recovered runs were completed on the Windows 5060 machine.

It is not yet at the latent diffusion / ControlNet / final image-generation stage. No implementation for Stable Diffusion-style training or explicit gaze disentanglement was found.

## 2. Biggest Problem

The biggest problem is context and artifact fragmentation.

The code and documentation are present locally, while the full data, canonical manifests, checkpoints, and 10K experiment outputs are on 5060 rather than in this Mac workspace. A 2026-08-07 SSH audit verified those major 5060 artifacts are physically present.

The newly confirmed technical problem is that recovered Phase2 inference exists but almost everything is rejected: Stage1 rejects 9,954/10,000, Stage2 rejects 9,964/10,000, and Stage3 rejects 9,991/10,000. A follow-up check confirmed this is the old BUG-003 landmark-coordinate artifact, not a valid post-fix result. The code fix exists and works, but no full corrected 10K Phase2 manifest/inference output was found on 5060.

The second biggest problem is scope drift: the project title promises 3D fusion control plus latent diffusion plus gaze disentanglement, while the implemented code currently supports DECA parameter-space standardization.

## 3. Next Most Important Work

The next most important work is not final model training.

Immediate sequence:

1. Regenerate the corrected Phase2 quality manifest after BUG-003.
2. Rerun Phase2 inference on the corrected manifest, first on a subset and then on 10K if the distribution is sane.
3. Revalidate hard-zero and learned Phase2 outputs on identical IDs.
4. Render standardized outputs and measure identity/gaze/pose behavior.
5. Decide whether the paper scope is parameter-space standardization or full diffusion-based frontalization.

## 4. Work Not To Do Now

- Do not train a final diffusion model yet.
- Do not add ControlNet or Stable Diffusion code before the dataset/manifest/evaluation base is stable.
- Do not claim gaze disentanglement from L2CS feature extraction alone.
- Do not rewrite Phase2 architecture before verifying recovered checkpoints and corrected quality features.
- Do not use only sample DECA outputs as evidence for full project progress.
- Do not mix p95 and p97.5 branches without explicit experiment names.

## 5. Distance To a Paper

For a DECA parameter-space standardization paper, the missing key experiments are:

- Reproducible baseline vs learned Phase2 comparison.
- Rendered-output validation.
- Identity preservation metrics with ArcFace.
- Gaze/head-pose metrics with L2CS/DECA.
- Ablations for hard-zero, partial alpha standardization, augmentation, XGBoost weighting, and reject gate.
- Failure-case analysis and qualitative panels.
- Real-data or harder-data validation if claiming arbitrary-pose/in-the-wild robustness.

For the full stated latent diffusion / gaze disentanglement paper, additional missing work is much larger:

- Diffusion/control model implementation.
- 3D condition representation for diffusion.
- Training protocol and paired/self-supervised target definition.
- Generated image outputs.
- Explicit gaze disentanglement objective or evaluation protocol.
- Full image-quality and identity-preservation benchmark.

## Restart Decision

Recommended restart direction:

First recover and validate the existing Phase1/Phase2 pipeline as the stable base. Only after Stage 7 rendered validation is trustworthy should the project branch into latent diffusion or a stronger paper contribution.
