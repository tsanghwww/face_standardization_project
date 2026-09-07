# Phase3.1c Geometry Causal-Response Audit

## Material Passport

- Date: 2026-09-07
- Stage: Phase3.1c, held-out geometry-response audit
- Optimization: none
- Allowed data: train diagnostics or validation model-selection data
- Forbidden data: fixed test and rescue outputs
- Claim boundary: geometry response only; no gaze-disentanglement claim

## Why This Audit Exists

The 64-step reconstruction adapter reduced denoising error and improved source-latent img2img identity relative to the frozen UNet. That evidence is not sufficient for 3D control: the old shuffled diagnostic changed geometry and identity together, and disabling the Face Adapter barely changed epsilon MSE.

This audit closes four engineering gaps:

1. Hold source identity, source latent, noise, schedule, and strength fixed while changing geometry only.
2. Re-estimate DECA pose, expression, and 68 landmarks from generated RGB and compare them with the Phase2 target.
3. Load validation through a dedicated evaluation dataset with separate training and evaluation fingerprints.
4. Bind target conditions to a concrete Phase2 checkpoint, inference config/command, per-sample NPZ, and rendered condition-map hashes. Directory names are not evidence.

## Intervention Matrix

| Arm | Geometry | Identity | Purpose |
| --- | --- | --- | --- |
| `source_geometry` | same sample source maps | same sample | source-conditioned reference |
| `target_geometry` | same sample Phase2 target maps | same sample | intended standardization intervention |
| `zero_geometry` | zeros | same sample | geometry-branch removal control |
| `shuffled_geometry` | next sample target maps | same sample | mismatched geometry control |
| `target_geometry_shuffled_identity` | same sample target maps | next sample | identity-specific diagnostic, excluded from geometry-only causal comparison |

The first four arms keep identity exactly equal. Every arm also reuses the same source VAE latent, CPU-generated noise, DDIM timestep schedule, and strength. The planned strengths are 0.25 and 0.50; there is no strength search on fixed test.

## Output Metrics

- Pose target error: SO(3) geodesic angle between the generated RGB's re-estimated DECA head rotation and the Phase2 target.
- Expression target error: RMSE between re-estimated 50D expression and Phase2 standardized expression.
- Landmark target error: mean 68-point error after independently centering and RMS-scale normalizing both landmark sets. This removes crop translation and scale; it is a shape diagnostic, not pixel-coordinate accuracy.
- Identity: the primary ArcFace cosine uses only pairs where both source and output contain exactly one detected face. Largest-box cosine on multi-face detections is retained as exploratory output, with no calibrated verification threshold.
- Validity: generation failure, FAN/DECA failure, ArcFace no-face, multi-face, and single-face-valid coverage, all using the complete selected denominator.
- Mechanism: per-scale mean absolute and RMS Face Adapter residual amplitudes.

For each strength, `target_geometry` is paired by `image_id` against source, zero, and shuffled geometry. Paired mean differences and bootstrap 95% confidence intervals are reported. A negative target-minus-comparator error favors target geometry. The script also reports the weaker descriptive check that target geometry has lower mean error than all three controls for pose, expression, and landmarks; it does not turn this check into a deployment gate.

## Canonical Phase2 Target

The historical Phase3 manifests point to `phase2_infer_sanity_bug003_fixed_arcface_ok`. The name alone neither validates nor invalidates those targets. The formally packaged Phase2 Full checkpoint is:

```text
results\phase2_ablation_20260825\full\best_model.pt
SHA256 bc520af061812d9a52e3793729a70e5a0a693d0b1179b60ab35e9adc3cb2a004
quality_source=blend
alpha_mode=learned
XGBoost OOF SHA256 85ed5ac8749c6ae681a4ef542863551c02041dbf9b733862948d6a8bef0eb846
```

Its checked-in inference manifest covers the fixed-test experiment, not the selected train/validation IDs. Therefore Phase3.1c must create a new selected-ID inference directory from that exact checkpoint. Old source-reconstruction and img2img results remain valid because they did not use target geometry, but they do not establish target-control provenance.

## 5060 Execution

Run from `D:\face_standardization_project` after syncing the code. First infer the canonical Phase2 target for all 1,440 validation IDs, then select 32 IDs stratified by actual source-to-target geometry change. This deliberately over-samples high-change cases for mechanism detection and is not a population-representative estimate.

```powershell
$root = "D:\face_standardization_project"
$py = "$root\.venv\Scripts\python.exe"
$p30 = "$root\results\phase30_20260901"
$p31c = "$root\results\phase31c_geometry_audit_20260907"
$allVal = "$p30\splits\validation_ids.txt"
$full = "$root\results\phase2_ablation_20260825\full"

# 1. Produce all validation targets from the canonical Phase2 Full checkpoint.
& $py -m phase2.infer_standardize_params `
  --deca-results-dir "$root\DECA\results\archive_phase2_params" `
  --checkpoint "$full\best_model.pt" `
  --arcface-manifest "$root\results\phase1_parity\phase1_master_manifest.csv" `
  --xgb-quality-manifest "$root\results\phase2_xgb_rebuilt_20260824\xgb_oof_manifest.csv" `
  --quality-source blend --alpha-mode learned --include-ids-file $allVal `
  --out-dir "$p31c\phase2_full_targets" --device cuda

# 2. Select 16 high-, 8 medium-, and 8 low-change validation examples.
& $py scripts\select_phase31c_geometry_audit_ids.py `
  --phase1-manifest "$root\results\phase1_parity\phase1_master_manifest.csv" `
  --phase2-manifest "$p31c\phase2_full_targets\phase2_inference_manifest.csv" `
  --validation-ids $allVal --fixed-test-ids "$p30\splits\fixed_test_ids.txt" `
  --project-root $root --count 32 --out-dir "$p31c\selection"
$ids = "$p31c\selection\geometry_audit_ids.txt"

# 3. Freshly render source/target maps. Do not use --resume for the canonical run.
& $py scripts\build_phase3_condition_cache.py `
  --phase1-manifest "$root\results\phase1_parity\phase1_master_manifest.csv" `
  --phase2-manifest "$p31c\phase2_full_targets\phase2_inference_manifest.csv" `
  --project-root $root --deca-root "$root\DECA" `
  --split-registry-dir "$p30\splits" --ids-file $ids `
  --out-dir "$p31c\condition_cache" --device cuda

# 4. Build an independent validation JSONL.
& $py scripts\build_condition_dataset.py `
  --phase1-manifest "$root\results\phase1_parity\phase1_master_manifest.csv" `
  --phase2-manifest "$p31c\phase2_full_targets\phase2_inference_manifest.csv" `
  --condition-cache-manifest "$p31c\condition_cache\phase3_condition_cache.csv" `
  --split-dir "$p30\splits" --include-ids-file $ids `
  --out-dir "$p31c\dataset"

# 5. Fail closed unless target provenance matches the canonical checkpoint and files.
& $py -m phase3.audit_phase2_target_provenance `
  --phase2-manifest "$p31c\phase2_full_targets\phase2_inference_manifest.csv" `
  --inference-config "$p31c\phase2_full_targets\inference_config.json" `
  --inference-command "$p31c\phase2_full_targets\inference_exact_command.txt" `
  --phase2-checkpoint "$full\best_model.pt" `
  --condition-cache-manifest "$p31c\condition_cache\phase3_condition_cache.csv" `
  --evaluation-manifest "$p31c\dataset\val.jsonl" --ids-file $ids `
  --out-dir "$p31c\target_provenance"

# 6. Generate paired geometry interventions with the existing 64-step adapter.
& $py -m phase3.sample_geometry_response `
  --evaluation-manifest "$p31c\dataset\val.jsonl" --evaluation-ids $ids `
  --evaluation-split validation --split-dir "$p30\splits" `
  --target-provenance "$p31c\target_provenance\target_provenance.json" `
  --checkpoint "$root\results\phase31_train_smoke_20260902\run\checkpoint.pt" `
  --backbone-path "$root\models\phase3\sd15_backbone" `
  --vae-path "$root\models\phase3\sd-vae-ft-mse" `
  --empty-prompt "$root\results\phase31_overfit_20260902\backbone_preflight\empty_prompt_embedding.safetensors" `
  --strengths 0.25 0.5 --sampling-steps 20 --device cuda `
  --out-dir "$p31c\samples"

# 7. Re-estimate output geometry and identity from RGB.
& $py -m phase3.evaluate_geometry_response `
  --run-dir "$p31c\samples" --deca-root "$root\DECA" --device cuda `
  --out-dir "$p31c\metrics"

# 8. CPU protocol tests.
& $py -m tests.test_phase31_geometry_audit
```

The exact local SD1.5 and empty-prompt paths should be taken from the successful img2img run if the example paths differ. The sampler verifies their contents against the training checkpoint by SHA-256 rather than accepting a path-name match.

The evaluator revalidates the evaluation manifest, ID list, split registry, MAT/NPZ files, identity embeddings, and every source/target condition map against the fingerprint frozen at sampling time. The adapter checkpoint must also prove that all recorded training inputs belong to the canonical train registry and have zero overlap with the selected validation IDs and fixed test.

The sampler refuses `retrospective_only` provenance. `--allow-retrospective-cache-binding` exists only to inspect legacy caches and can never authorize the canonical Phase3.1c sampling command.

## Decision After the Run

- If target geometry consistently lowers pose/expression/landmark target error against source, zero, and shuffled controls while retaining useful single-face and identity coverage, proceed to a bounded geometry-aware training iteration.
- If the Face Adapter residual is nonzero but target geometry does not move RGB toward the Phase2 target, the learned residual is not acting as useful control; revise supervision or architecture before adding training steps.
- If residual amplitudes are near zero and controls are indistinguishable, prioritize high-geometry-delta examples and condition-specific dropout/corruption rather than scaling to all 8,160 images.
- Gaze remains out of scope. Head-pose response is a prerequisite, not evidence of gaze disentanglement.
