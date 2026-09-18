# Phase3.1g High-Resolution Geometry-Delta Result

Date: 2026-09-18  
Status: geometry mechanism gate passed on optimization and independent
validation identities; fixed test remains sealed.

## Why Phase3.1f Was Insufficient

Phase3.1f moved a canonical target closer but produced only 0.084 degrees of
median projected response to a roughly 20-degree same-identity counterfactual.
It also degraded source-self geometry. The likely causes were the 32x32-first
condition encoder, additive control only in the UNet down path, a trainable
source reconstruction path, and ranking losses that could be satisfied by a
very small change.

Phase3.1g therefore freezes the complete source reconstruction adapter and adds
a separate 1.396M-parameter edit branch. It encodes the 256x256
`target_condition - source_condition` signal before producing a 32/16/8/4 UNet
residual pyramid. A hard zero gate makes a source-equals-target request exactly
identical to the frozen source path.

## Loss-Level Failure and Correction

The first Phase3.1g run used an unsigned output-pair geodesic separation loss.
It increased mean output separation to 2.522 degrees, but all 24 optimization
audit rows moved against the requested target axis:

| Variant | Success | Direction + ordering | Median projected change | Gate |
| --- | ---: | ---: | ---: | --- |
| Unsigned pair separation | 24/24 | 0/24 | -1.187 deg | Fail |
| Signed SO(3) projection | 24/24 | 23/24 | +3.239 deg | Pass |

The correction represents each negative-to-positive output change as a signed
SO(3) relative rotation vector and projects it onto the target rotation axis.
An opposite-direction change now increases the loss. This isolates the causal
effect of direction-aware supervision; the architecture, identities, step
budget, timesteps, and optimizer settings were otherwise unchanged.

## Bounded Training

- Optimization identities: 8 train IDs.
- Steps: 64, learning rate `1e-4`.
- Timesteps: uniformly restricted to 100-400.
- Counterfactual pair: same identity, latent, noise, and timestep; -10/+10
  degree target yaw.
- Geometry/ranking/signed-separation weights: `0.003 / 1.0 / 1.0`.
- Source no-op: bitwise exact.
- Frozen: source adapter, UNet, identity adapter, VAE, and DECA evaluator.
- Peak allocated memory: 6.516 GiB in paired preflight; 5,549 MiB during the
  optimizer run.
- Training wall time: 40.6 seconds on the RTX 5060 Laptop GPU.

## Independent Validation

The validation audit used 32 frozen Phase3.1c validation identities. The
checkpoint optimization IDs and validation IDs are disjoint. The fixed test
registry was not used. The predeclared gate was applied without threshold
search: complete denominator, at least 75% direction-and-ordering correctness,
and median projected change of at least 0.5 degrees.

| Split | Rows | Success | Direction correct | Ordering correct | Both correct | Median projected change | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 8-ID optimization audit | 24 | 24 | 23 | 23 | 23 (95.8%) | 3.239 deg | Pass |
| 32-ID validation audit | 96 | 96 | 88 | 89 | 88 (91.7%) | 2.586 deg | Pass |

Validation by timestep:

| Timestep | Both correct | Median projected change | Mean output separation | Mean ordering margin |
| ---: | ---: | ---: | ---: | ---: |
| 100 | 28/32 | 1.888 deg | 2.368 deg | 0.971 deg |
| 250 | 29/32 | 3.178 deg | 3.559 deg | 1.907 deg |
| 400 | 31/32 | 4.255 deg | 4.633 deg | 3.360 deg |

The increasing response with timestep is expected for a one-step latent audit,
but it also means control magnitude is not calibrated across noise levels.

## Scientific Decision

Phase3.1g resolves the immediate geometry-injection failure: the model now
learns target-dependent pose direction and ordering, and that behavior transfers
from 8 optimization identities to 32 isolated validation identities. This is a
mechanism result, not yet a complete standardization result.

The result does not establish RGB identity preservation, expression/landmark
target accuracy, calibrated 20-degree transfer, gaze disentanglement, or
fixed-test performance. In particular, the achieved validation motion is only
about 13% of the requested 20-degree pair at the median, and eight validation
rows still fail the joint direction-and-ordering criterion.

The next bounded stage should evaluate decoded RGB identity and absolute
geometry error on validation, add magnitude calibration across target offsets,
and then attach a head-local gaze branch with separate gaze labels and
counterfactual gaze controls. The 775 fixed test must remain sealed until those
validation checks and the gaze-coordinate protocol pass.

## Artifact Hashes

Paths are relative to `D:\face_standardization_project` on win-lenovo.

| Artifact | SHA256 |
| --- | --- |
| `results/phase31g_delta_adapter_8id_audit_20260918/summary.json` | `d6977f1e8229ce173f76d61ce60fd904ddbf6d85f79b3da344e560db521e6aaa` |
| `results/phase31g_signed_delta_8id_64step_20260918/config.json` | `0f67f83a633fa5fcdf27e2b0044e8d142b7029e3e3c3481ddee32e48afa67ffb` |
| `results/phase31g_signed_delta_8id_64step_20260918/preflight.json` | `f8c8937883e2c53f865de8eb3d3c24bc3c30e51774d3e2964d4c9c65824f7789` |
| `results/phase31g_signed_delta_8id_64step_20260918/summary.json` | `6954bcb09d365749e7e5137493e67f65d511c24a64538425ce2ff3dbfc8e293b` |
| `results/phase31g_signed_delta_8id_64step_20260918/checkpoint_step_0064.pt` | `66e296900158f24b1480359ae01a0e4b3126999bf9d215d549524d795da73fee` |
| `results/phase31g_signed_delta_8id_audit_v2_20260918/summary.json` | `a29362d5e5b234b0ec074935eda6098f1860864dd0f4ad43d6b5a14ea851800b` |
| `results/phase31g_validation_counterfactual_conditions_20260918/summary.json` | `da20d767b00134e894f4566e2749a3d7b2c2b97f80346654928655ebf15b1dfa` |
| `results/phase31g_signed_delta_validation_audit_v2_20260918/config.json` | `ade380f370d5e2fbe00d906eb09e1c30937bcf6ee814590584dbde6770aa9311` |
| `results/phase31g_signed_delta_validation_audit_v2_20260918/summary.json` | `0e18a571e5261d3a81c0a6311296e71932e7ffc583a393bac49dea3a2309bf6f` |
