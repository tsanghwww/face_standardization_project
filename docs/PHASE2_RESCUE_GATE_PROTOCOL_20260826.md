# Phase2 Rescue and Gate Completion Protocol

## Material Passport

- Artifact type: experiment completion protocol
- Project: DECA parameter-space face standardization
- Date: 2026-08-26
- Scope: rescue sensitivity and gate calibration only
- Fixed test set: 775 samples
- Calibration source: the existing 1,440-sample Phase2 validation split
- Status: protocol specified; final figures and conclusions remain blocked until both work packages pass

## Why These Work Packages Are Separate

The 32 `fan_no_face` cases are upstream preprocessing failures. They are not false rejects from the Phase2 gate because the gate never receives their DECA features. Rescue sensitivity asks whether a fallback crop can safely recover pipeline coverage. Gate calibration asks whether samples that reach Phase2 should be accepted, weakly standardized, or rejected.

The primary FAN path must remain the primary result. Rescue results are reported as a separate policy and must never overwrite the immutable test manifest.

## Work Package 1: Rescue Sensitivity

### Available Inputs

- Main FAN/DECA path: 343/375 external successes and 32 WIDER failures.
- Rescue path: 375/375 successful MAT files.
- Rescue sidecars: 375 `_kpt2d.txt` and 375 `_kpt3d.txt` files.
- Main/rescue paired external cases: 343.
- Rescue-only cases: 32.

No DECA extraction rerun is required.

### Experiment 1A: Quantify Rescue Domain Shift on 343 Paired Cases

For each external sample with both main and rescue parameters, compare:

$$
d_{\mathrm{exp},i}=\frac{\lVert\beta^{\mathrm{rescue}}_{\mathrm{exp},i}-
\beta^{\mathrm{FAN}}_{\mathrm{exp},i}\rVert_2}{\sqrt{50}},
$$

$$
d_{\mathrm{pose},i}=\lVert\theta^{\mathrm{rescue}}_i-
\theta^{\mathrm{FAN}}_i\rVert_2,
$$

and analogous distances for camera and lighting. Rebuild the same ordered 10-dimensional XGBoost feature vector for rescue and compare main-versus-rescue XGBoost scores. Render the two original codedicts and measure main-versus-rescue ArcFace cosine, DECA pose difference, and L2CS gaze difference.

Report the mean, median, p90, p95, and paired-bootstrap 95% CI overall and for WIDER pose, WIDER occlusion, WIDER blur, COFW, and AFLW separately.

Purpose: estimate the coordinate/crop distribution shift introduced by whole-image warp. A 100% decode rate alone is not evidence that rescue is equivalent to FAN.

### Experiment 1B: Evaluate the 32 Rescue-Only Cases

Build a separate `rescue_failed_mats/` directory containing only the 32 failed `eval_id` values, with renamed MAT and keypoint sidecars. Do not modify `fixed_test_manifest_v2.csv`.

For all four trained models:

1. Run inference on the 32 rescue MAT files.
2. Rebuild the same ordered 10-dimensional feature vector and run the saved final XGBoost model in predict-only mode. Record `xgb_input_provenance=rescue`; do not reuse the 32 blank primary predictions.
3. Record `quality_source_effective` explicitly. Use `heuristic_fallback` only if a required rescue feature is genuinely unavailable; never manufacture an XGBoost score.
4. Render rescue-original, hard-zero, Full, Fixed-alpha=1, No augmentation, and No XGBoost.
5. Compute the same within-render-domain ArcFace, DECA pose, and L2CS gaze metrics used by the primary evaluation.
6. Record all results under a new rescue result root.

The 32 samples have no FAN reference. Compare them against matched successful WIDER controls from the same difficulty group using available blur, occlusion, and pose metadata. Report results as rescue-only observational evidence, not as paired equivalence to the main path.

### Policy-Level Reporting

Report two policies side by side:

$$
C_{\mathrm{main}}=\frac{743}{775}=95.87\%,
$$

$$
C_{\mathrm{fallback}}=
\frac{743+N_{\mathrm{valid\ rescue}}}{775}.
$$

`N_valid rescue` requires finite parameters, successful rendering, and passing the pre-registered rescue quality rule. It must not automatically equal 32 merely because MAT extraction succeeded.

Every fallback output must include `preprocess_source=whole_image_rescue`. Until Experiment 1A shows acceptable shift, the conservative operational action is `weak_standardize` or `reject`, not silent standardization.

### Required Rescue Artifacts

- `rescue_sensitivity_manifest.csv`
- `main_vs_rescue_paired_metrics.csv`
- `rescue_only_render_metrics.csv`
- `rescue_policy_summary.csv`
- `rescue_failures.csv`
- exact command, config, seed, and file hashes

### Rescue Completion Gate

- All 343 paired cases appear exactly once in the domain-shift table.
- All 32 failed FAN cases appear exactly once in the rescue-only table.
- Main and fallback denominators are both 775.
- XGBoost-unavailable rows contain no fabricated score.
- Primary and rescue outputs remain in separate directories and manifests.
- Conclusions distinguish extraction recovery from scientifically usable recovery.

## Work Package 2: Gate Error Analysis and Calibration

### Current Gate Limitation

The training target is currently:

$$
y_{\mathrm{reject}}=1-q_{\mathrm{proxy}},
$$

where `q_proxy` is the heuristic/XGBoost quality score. Therefore the gate predicts proxy input quality, not observed standardization failure.

On the current 743 available fixed-test cases, `reject_score` and `1-confidence` have correlation approximately 0.99993. The current two-threshold rule therefore behaves almost like a single threshold. A preliminary, explicitly non-confirmatory outcome audit produced AUROC approximately 0.656. Threshold movement alone is unlikely to create a reliable safety gate.

### Define Observable Outcomes Before Calibration

Do not call a decision false without an outcome definition. Use three outcome classes:

- `unsafe`: render/metric failure or material degradation relative to hard-zero.
- `safe_but_ineffective`: no material degradation, but insufficient pose/expression standardization.
- `safe_and_effective`: no material degradation and useful standardization.

For method $m$, define paired changes relative to hard-zero:

$$
\Delta_{\mathrm{id}}=s_m-s_{\mathrm{hardzero}},
$$

$$
\Delta_{\mathrm{pose}}=
\lVert\theta_m\rVert_2-\lVert\theta_{\mathrm{hardzero}}\rVert_2,
$$

$$
\Delta_{\mathrm{gaze}}=
g_m-g_{\mathrm{hardzero}}.
$$

The primary safety rule should mark a sample unsafe when any required renderer/metric fails, identity falls below a pre-registered non-inferiority margin, or pose/gaze degradation exceeds its margin. L2CS missingness is treated as unsafe in the conservative analysis and as missing in a secondary sensitivity analysis.

Before choosing margins, add re-encoded DECA expression norm to the evaluator. Estimate metric repeatability, then freeze identity, expression, pose, and gaze margins. Values such as cosine 0.02, pose norm 0.03, and gaze 10 degrees may be used for an exploratory table only; they must not become final margins merely because they improve the test result.

### Correct Calibration Split

Do not tune thresholds on the 775 fixed-test samples. They have already been inspected and must remain the final audit set.

1. Infer and render Original, Hard-zero, and Full for the existing 1,440 validation IDs.
2. Generate outcome labels from those validation renders.
3. Use deterministic five-fold cross-validation inside the 1,440 cases for calibrator selection.
4. Freeze the outcome definition, model, and thresholds.
5. Apply them once to the 775 fixed-test cases and report the result as a post-training audit.

### Calibration Models

Evaluate in this order:

1. `reject_score` only.
2. Mean risk score $r=(\mathrm{reject}+1-\mathrm{confidence})/2$.
3. A small post-hoc logistic calibrator using only operationally available features: reject score, confidence, blended/heuristic quality, XGBoost score/status, landmark metrics, ArcFace status/score, original pose norm, and predicted alphas.

Do not use `source_group` or dataset identity as model inputs. They may be used only for stratified reporting.

If the post-hoc calibrator remains weak, the next model change is to retrain the gate head on outcome-derived labels or OOF outcome predictions. Do not keep searching test-set thresholds.

### Decision Rule and Operating Points

Use one calibrated risk score $r$:

$$
\mathrm{decision}(r)=
\begin{cases}
\mathrm{standardize}, & r<t_{\mathrm{weak}},\\
\mathrm{weak\_standardize}, & t_{\mathrm{weak}}\le r<t_{\mathrm{reject}},\\
\mathrm{reject}, & r\ge t_{\mathrm{reject}}.
\end{cases}
$$

Choose thresholds from validation risk-coverage curves. Report at least conservative, balanced, and high-coverage operating points. The recommended primary point maximizes coverage subject to a pre-registered accepted-set unsafe rate; 5%, 10%, and 15% risk constraints should be shown rather than hiding the trade-off behind one threshold.

### False Accept and False Reject Definitions

For the binary safety audit:

$$
\mathrm{FAR}=\frac{N(\mathrm{unsafe\ and\ accepted})}{N(\mathrm{unsafe})},
$$

$$
\mathrm{FRR}=\frac{N(\mathrm{safe\ and\ rejected})}{N(\mathrm{safe})}.
$$

Also report selective risk, coverage, AUROC, AUPRC, Brier score, expected calibration error, and paired-bootstrap or Wilson 95% intervals. Upstream FAN failures remain a separate pipeline-failure category.

### Mandatory Stratification

Report the confusion matrix and risk/coverage separately for:

- XGBoost high, medium, and low;
- WIDER pose, occlusion, and blur;
- COFW occlusion;
- AFLW large pose;
- ArcFace success/failure;
- FAN main versus rescue provenance.

Small strata must include counts and confidence intervals. Do not compare percentages without denominators.

### Required Gate Artifacts

- `gate_outcome_definition.json`
- `gate_validation_outcomes.csv`
- `gate_threshold_search.csv`
- `gate_calibrator.json` or checkpoint
- `gate_fixed_test_predictions.csv`
- `gate_confusion_by_quality.csv`
- `gate_risk_coverage.csv`
- `gate_calibration_summary.json`
- exact command, config, split IDs, and hashes

### Gate Completion Gate

- Outcome labels are independent of the current gate decision.
- Thresholds and margins are frozen using validation data only.
- The 775-sample result is not used to choose the operating point.
- FAR, FRR, risk, and coverage are reported overall and by quality stratum.
- Missing metrics and 32 upstream failures are not silently discarded.
- Any final claim distinguishes proxy-label calibration from true human-perceived quality.

## Execution Order

1. Add rescue manifest preparation and main-versus-rescue comparison.
2. Run Experiment 1A on 343 paired external cases.
3. Run Experiment 1B on 32 rescue-only cases.
4. Extend evaluation with expression outcome and validation-set rendering.
5. Freeze outcome margins on validation data.
6. Fit and cross-validate the gate calibrator.
7. Apply the frozen gate once to the 775 fixed test set.
8. Only then generate final figures, hashes, and research conclusions.
