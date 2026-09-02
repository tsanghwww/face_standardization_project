# Phase3.1 Frozen Backbone Preflight

Date: 2026-09-02

## Scope

This stage prepares a bounded 32-sample validation-only overfit experiment. It
does not train a diffusion adapter and does not establish gaze disentanglement
or generalization. The 775-sample fixed test remains evaluation-only.

The frozen backbone is
`stable-diffusion-v1-5/stable-diffusion-v1-5`. Its UNet, scheduler, tokenizer,
and text encoder are stored outside Git at `models/phase3/sd15_backbone`. The
pipeline VAE is not used; Phase3 keeps the separately audited
`models/phase3/sd-vae-ft-mse` VAE.

## Condition Dataset

The 32 IDs come from the Phase3.0B VAE validation audit and are supplied through
`--include-ids-file`. Source and Phase2-target depth, normal, landmark, face-mask,
and eye-mask maps are generated independently. The builder preserves the ID-file
order and records its SHA256.

The current gaze coordinate convention remains `pending_manual_audit`.
Consequently, `target_gaze_heatmap` and all head-local gaze fields must remain
null. This is intentional and prevents an unapproved coordinate transform from
becoming a training target.

## Frozen Preflight

`phase3.preflight_sd15_backbone` performs the following checks:

1. Load the tokenizer and frozen CLIP text encoder locally.
2. Encode the real empty prompt to `[1, 77, 768]`; zeros are not substituted.
3. Save the embedding as a safetensors artifact, then release the text encoder.
4. Load the frozen UNet in FP16 on CUDA.
5. Run one deterministic `[1, 4, 32, 32]` latent forward pass at timestep 500.
6. Require shape preservation, finite output, and no trainable UNet parameters.
7. Record peak allocated VRAM, wall time, environment, exact command, model-file
   hashes, source model ID, and immutable Hugging Face revision.

## Reproduction

```powershell
$py = D:\face_standardization_project\.venv\Scripts\python.exe

$py scripts\build_phase3_condition_cache.py `
  --phase1-manifest results\phase1_parity\phase1_master_manifest.csv `
  --phase2-manifest results\phase2_infer_sanity_bug003_fixed_arcface_ok\phase2_inference_manifest.csv `
  --project-root D:\face_standardization_project `
  --deca-root D:\face_standardization_project\DECA `
  --split-registry-dir results\phase30_20260901\splits `
  --gaze-policy preserve_eye_in_head `
  --out-dir results\phase31_overfit_20260902\condition_cache `
  --device cuda --rasterizer-type standard `
  --ids-file results\phase30_20260901\vae_roundtrip_audit\selection\vae_audit_ids.txt

$py scripts\build_condition_dataset.py `
  --phase1-manifest results\phase1_parity\phase1_master_manifest.csv `
  --phase2-manifest results\phase2_infer_sanity_bug003_fixed_arcface_ok\phase2_inference_manifest.csv `
  --condition-cache-manifest results\phase31_overfit_20260902\condition_cache\phase3_condition_cache.csv `
  --split-dir results\phase30_20260901\splits `
  --include-ids-file results\phase30_20260901\vae_roundtrip_audit\selection\vae_audit_ids.txt `
  --out-dir results\phase31_overfit_20260902\dataset

$py -m phase3.preflight_sd15_backbone `
  --backbone-path models\phase3\sd15_backbone `
  --revision <immutable-hugging-face-commit> `
  --device cuda --dtype fp16 `
  --out-dir results\phase31_overfit_20260902\backbone_preflight
```

## Entry Gate

Adapter work may start only after the real preflight passes. The first training
step is a 32-sample overfit diagnostic with the UNet and VAE frozen. It must not
use the fixed test for checkpoint selection. A gaze loss or gaze control channel
must remain disabled until the head-local coordinate convention is approved.
