# Phase3.1h External Targets and Quantitative Evaluation

## Execution result

On 2026-10-09, the RTX 5060 produced frozen Phase2 targets for 81 COFW
official-training candidates. The candidate pool remains separate from the
canonical train, validation and fixed-test registries.

| Item | Result |
| --- | --- |
| Quarantined COFW candidates | 128 |
| ArcFace conditional machine pass | 81 |
| Potential identity-overlap rejection | 41 |
| Similarity-band manual review | 6 |
| Protected identity embedding coverage | 2,129 / 2,215 (96.12%) |
| FAN-only DECA success | 81 / 81 |
| Valid Phase2 targets | 81 / 81 |
| Phase2 standardize / weak / reject | 72 / 0 / 9 |
| Source/model/split hashes unchanged | Verified |
| Training-eligible candidates | 0 |

The 72 standardize decisions are predictions of the existing proxy-quality
model, not observed evidence that generated RGB is safe or accurate. Missing
protected embeddings (86) require review before identity isolation can be
claimed. COFW has no gaze ground truth in this pipeline; this pool can improve
real-image reconstruction and occlusion coverage but cannot independently
establish gaze disentanglement.

Remote artifact root:
`D:\face_standardization_project\results\phase31h_external_phase2_targets_v2_20261009`

The final v2 run additionally hashes the source images and landmark text files
used for quality features. The first run is retained as an earlier artifact.

Frozen Phase2 checkpoint SHA256:
`bc520af061812d9a52e3793729a70e5a0a693d0b1179b60ab35e9adc3cb2a004`

Frozen XGBoost final-model SHA256:
`d87dd136d157467ccf7491189337711702ce10e8638b855287ff24631d1735dc`

The run uses real external XGBoost predictions, `quality_source=blend`, learned
alpha, reject threshold 0.60 and weak-confidence threshold 0.40. These are
existing inference settings, not newly calibrated safety thresholds.

## Current-stage metrics

The project is in Phase3.1 geometry-control and identity-reconstruction pilot
experiments, with Phase3.1h preparing real external training inputs.

| Dimension | Quantitative measure | Interpretation and current evidence |
| --- | --- | --- |
| Intake coverage | Success / all candidate IDs | DECA 81/81; Phase2 81/81. Separate from training eligibility. |
| Identity isolation | Max candidate-to-protected ArcFace cosine; protected embedding coverage | Diagnostic triage: cosine >=0.40 rejects, 0.30–0.40 reviews; these are not calibrated identity-verification thresholds. |
| Identity retention | Normalized ArcFace cosine; mean, median, P10; single-face coverage | Higher is better. Compare with the same-source reconstruction control at the same timestep and preprocessing. |
| Absolute head accuracy | SO(3) target angular error in degrees | Lower is better. Report error to requested target and source-minus-output improvement. |
| Expression accuracy | RMSE of 50 DECA expression coefficients | Lower is better; DECA model-space diagnostic, not human expression ground truth. |
| Landmark accuracy | Center/scale-normalized 68-point shape distance | Lower is better. Existing shape NME is not interocular-normalized benchmark NME; it discards translation and scale. |
| Condition response | Correct signed direction and own-target ordering | Existing diagnostic gate: >=75% both correct, median projected motion >=0.5 degrees, full denominator. |
| Image preservation | RGB MAE, PSNR, SSIM, LPIPS | MAE/LPIPS lower, PSNR/SSIM higher. Full-image pixel similarity must not penalize intended head edits without a matched control. |
| Reliability | Generation/FAN/ArcFace/DECA failures and coverage | Every split ID stays in the denominator; failed metrics stay null. |
| Cost | Peak allocated GPU MiB, wall time, seconds/image | Report startup and inference costs separately when available. |

Latest counterfactual validation RGB audit: 32 IDs, 2 yaw arms, timesteps
100/250/400; generation, ArcFace and FAN/DECA all cover 192/192 outputs.

| Timestep | Mean identity cosine, negative / positive | Mean absolute pose error, negative / positive | Mean pose improvement over source, negative / positive |
| --- | --- | --- | --- |
| 100 | 0.5856 / 0.5732 | 16.55 / 15.22 degrees | +0.706 / +0.750 degrees |
| 250 | 0.4390 / 0.4220 | 15.87 / 14.60 degrees | +1.381 / +1.375 degrees |
| 400 | 0.3376 / 0.3283 | 15.49 / 14.23 degrees | +1.759 / +1.743 degrees |

The geometry mechanism audit achieved 88/96 both-correct pairs (91.67%) and
median projected motion 2.588 degrees. The positive pose improvements have
paired bootstrap 95% intervals above zero. Absolute pose errors remain large,
and identity declines as timestep rises. Mechanism-gate passage therefore does
not establish overall reconstruction or standardization success. The VAE anchor
near 0.917 is historical reference; it is not a matched validation control for
this table. The latest evaluator uses cached source embeddings and generated
largest-face embeddings; a single-face-only companion table and matched frozen
source-output arm are still required for the primary identity conclusion.

## Whole-project metrics

| Stage / claim | Primary measurements | Required comparison |
| --- | --- | --- |
| Phase1 quality screening | AUROC, AUPRC, per-class precision/recall, feature coverage, OOF performance | Ground-truth labels versus predictions; clarify that Pass/Warn labels are quality proxies. |
| Phase2 standardization | Identity cosine, head target angular error, expression RMSE, render failure, decision-aware delivered coverage | original, hard-zero, full, no-alpha, no-augmentation, no-XGBoost on identical IDs. |
| Phase2.1 outcome Gate | AUROC, AUPRC, Brier, ECE, risk-coverage curve, unsafe accept rate, false reject rate | Independent gate-train and calibration splits; frozen application to evaluation data. |
| Phase3 image quality | Identity cosine, PSNR/SSIM/LPIPS on reconstruction; target error on edits; optional KID/FID | Frozen reconstruction and matched source-condition controls; comparable sampling and dataset sizes. |
| 3D head control | Target angular error, signed response, own-target ordering, expression and landmark errors | Same identity/noise/timestep, change only head geometry. |
| Gaze control | Camera-frame and head-local target gaze angular errors, gaze-only success/coverage | Same identity and head, change only gaze condition. |
| Head-to-gaze leakage | Head-local gaze change under a head-only intervention, in degrees | Hold requested eye-in-head gaze fixed while changing head pose. |
| Gaze-to-head leakage | Head rotation change under a gaze-only intervention, in degrees | Hold requested head pose fixed while changing eye gaze. |
| Locality | Non-eye LPIPS, eye-region SSIM/LPIPS and eye landmark consistency | Gaze-only edit versus matched reconstruction; explicitly define mask and mask coverage. |
| Robustness | Every metric stratified by quality, pose, occlusion, blur and source dataset | Report stratum counts, failure rates and confidence intervals. |

Definitions:

- Identity: `cos = dot(e_source, e_output) / (||e_source|| ||e_output||)`.
- Head error: `acos(clip((trace(R_target^T R_output)-1)/2,-1,1))`, converted to degrees.
- Gaze error: `acos(clip(dot(g_target,g_output),-1,1))` for unit gaze vectors.
- Head-local gaze: `g_head = R_head^T g_camera`, conditional on approved coordinate conventions.
- Coverage: accepted and successfully delivered outputs / all intended split IDs.
- Unsafe accepted risk: accepted unsafe / all accepted evaluable outcomes; separately report accepted outputs with unknown safety status.
- Unsafe false acceptance: accepted unsafe / all unsafe outcomes.
- Safe false rejection: rejected safe / all safe outcomes.

Report means, medians and low identity quantiles or high error quantiles, plus
95% confidence intervals. Repeated arms/timesteps must be resampled by image ID
(or identity/source family when grouped), not treated as independent images.
Report missing-outcome coverage alongside risk; unknown safety is not safe.

Historical frozen Phase2 Gate audit on the 775 fixed set had accepted-set
unsafe risk 44.27%, FAR 60.69% and FRR 42.82% among its available outcome
population. This is diagnostic evidence against deployment qualification.
Validation risk settings of 5/10/15% are operating-point research choices;
they are not achieved safety guarantees on external data.

## Promotion criteria and next work

Existing numerical gates are stage-specific diagnostics. There is currently no
validated project-wide ArcFace cutoff, gaze-error cutoff or safety/coverage
acceptance threshold. Such criteria must be fixed using development/calibration
data before the final evaluation. The already-inspected 775 set cannot be
presented as a completely untouched confirmatory test for new hypotheses;
future confirmatory claims require an additional untouched test set, or must
explicitly describe the 775 results as a frozen repeated audit.

1. Generate source/target normal, depth, landmark, face and eye masks for the 72 standardize candidates; preserve the 9 rejects in the ledger.
2. Review the 86 unavailable protected identity embeddings and suspicious image/source-family relationships; record unresolved cases explicitly.
3. Build a separate external training registry with approved lineage and a documented admission decision; retain canonical validation/fixed splits.
4. Run a bounded comparison of synthetic-only versus source-balanced synthetic+real training with equal update budgets and matched validation.
5. Add an explicit identity-preservation objective and matched source-output controls, starting with the 100–250 timestep range.
6. Approve gaze coordinates, implement gaze-only/head-only interventions and bidirectional leakage evaluation before making a gaze-disentanglement claim.

Repository protocol sources: `PHASE31G_HIGHRES_GEOMETRY_DELTA_PROTOCOL.md`,
`PHASE31C_GEOMETRY_CAUSAL_AUDIT.md`, `PHASE2_RESCUE_GATE_STATUS_20260826.md`,
and `PHASE3_LATENT_DIFFUSION_TRAINING_PLAN.md` in this docs directory.
