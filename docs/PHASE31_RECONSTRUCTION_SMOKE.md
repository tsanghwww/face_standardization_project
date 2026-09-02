# Phase3.1 Source Reconstruction Engineering Smoke

## Material Passport

- Date: 2026-09-02
- Type: code experiment implementation and bounded execution
- Stage: Phase3.1a; not completion of Phase3.1 or gaze disentanglement
- Scope: source-conditioned reconstruction, frozen VAE/UNet, train-only subset
- Verification: CPU protocol tests and real GPU run artifacts; results reported separately

## Protocol

The prior 32 VAE audit IDs belong to validation and remain audit-only. Select
32 new IDs from the immutable train registry with seed 20260902. Eligibility
requires source image, DECA parameters, Phase2 output, and ArcFace embedding.
Record ineligible IDs explicitly. No image from validation, fixed test, or the
rescue path may be optimized on. Selection is an engineering subset, not a
representative statistical sample.

For this run the image target is the source image. Only **source** normal,
depth, landmark, and face-mask maps are loaded. Phase2 target maps are generated
for later use but never passed as conditions for source reconstruction loss.
No standardized RGB ground truth is fabricated.

RGB uses full-image Lanczos resize to 256. Normals use RGB bilinear resize;
16-bit depth is divided by 65535 before bilinear resize; face masks use nearest
neighbor. ArcFace vectors must be finite, nonzero, and 512-dimensional, and are
L2 normalized. Missing inputs cause failure, not zero filling or silent skips.

## Trainable Components

1. Face adapter: six channels (normal RGB, depth, landmark, face mask), four
   convolution scales, zero-initialized 1x1 outputs aligned with the existing
   Diffusers intrablock residual API.
2. Identity projection: normalized ArcFace vector to four identity tokens.
3. Independent identity cross-attention: native Diffusers
   `IPAdapterAttnProcessor2_0`, with separate identity key/value projections.
   Identity values start at zero; identity keys initialize from frozen text
   attention keys. This is an implementation adaptation, not a pretrained
   IP-Adapter/FaceID checkpoint or a claim of reproducing its results.

The VAE and all original UNet parameters stay frozen. New identity processors
are trainable, even though registered under the UNet object. Hashing and gradient
checks distinguish those new processors from the original backbone.

Eye adapter, gaze inputs/losses, head intervention, identity outcome loss,
quality conditioning, and LoRA are disabled. This is a deliberately smaller
subset of the overall Phase3 design.

## Optimization and Diagnostics

- Latents: deterministic VAE posterior mode, scaled by the VAE config factor.
- Loss: epsilon prediction MSE on source reconstruction only.
- Mixed precision: frozen UNet FP16, trainable weights FP32, autocast and scaler.
- Micro-batch 1; gradient accumulation 8; adapter LR 1e-4; gradient clipping 1.
- Non-reentrant gradient checkpointing retains paths through the frozen UNet.
- First gate: 2 optimizer steps. Next bound: resume to 64 total steps. This is
  shorter than the planned 500-step overfit stage and does not replace it.
- Fixed diagnostic: identical per-image noise at timestep 250, before/after,
  no-face, no-identity, and shuffled joint conditions. These are **training-set
  diagnostics**, not ablation retraining or held-out generalization results.
- Sampling: separate DDIM-from-noise previews with fixed seeds. A single-step
  x0 estimate is labeled separately and never called a generated reconstruction.

Checkpoints contain adapters, optimizer/scaler state, step, input/code/model
hashes, and original-backbone hash. Resume rejects incompatible fingerprints.
Data order, timestep, and noise depend on absolute micro-step seeds, not on
unrecorded process RNG. Logs and exact commands retain the resume boundary.

## Tests and Interpretation

`python -m tests.test_phase31_reconstruction` checks train/validation/fixed-test
isolation, rescue rejection, missing inputs, duplicate IDs, uint16 depth,
source-only loading, zero-init baseline parity, finite gradients, updates to all
three trainable groups, and unchanged original UNet weights.

A lower epsilon MSE does not establish identity preservation, successful
standardization, or gaze control. In particular, successful source reconstruction
does not prove the face branch responds correctly to novel target geometry.
Identity/head/gaze image metrics, 32/32 generation audit, condition-intervention
checks, and longer overfit training remain separate gates.

## Remote Locations

Root: `D:\face_standardization_project\results\phase31_train_smoke_20260902`

- `selection/`: train IDs, eligibility exclusions, manifest hashes
- `condition_cache/`: source/target geometry for the new train IDs
- `dataset/train.jsonl`: 32 train rows; val/test JSONL empty
- `alignment_audit.json`: saved DECA vs full-image re-encoding check
- `run/`: configurations, commands, training log, checkpoint, diagnostics, samples

Entry points: `phase3.prepare_reconstruction_selection`, existing
`scripts/build_phase3_condition_cache.py`, `scripts/build_condition_dataset.py`,
and `phase3.train_reconstruction_smoke`. Run artifacts retain exact paths and
commands. Model weights and generated outputs remain outside Git.
