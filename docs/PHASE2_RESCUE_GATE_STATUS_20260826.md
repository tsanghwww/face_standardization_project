# Phase2 Rescue and Gate Execution Status

## Scope

- Execution date: 2026-08-26
- Fixed test denominator: 775
- Gate calibration split: frozen 1,440 validation IDs
- Final figures, artifact hash manifest, and research conclusions: intentionally pending

## Rescue Domain Shift: 343 Paired Samples

- Paired rows: 343/343.
- Parameter, XGBoost, and ArcFace metrics: 343/343.
- L2CS gaze: 341/343; two WIDER pose renders failed only in L2CS.
- Main-versus-rescue ArcFace cosine: mean 0.6651, median 0.6849.
- Main-versus-rescue gaze difference: mean 24.12 degrees, median 13.65 degrees, p95 80.90 degrees.
- Expression RMSE: mean 0.1203, median 0.0995, p95 0.2639.
- Head-pose L2: mean 0.1910, median 0.1288, p95 0.5733.
- Rescue-minus-main XGBoost score: mean +0.0250, median +0.00018, p95 +0.2197.

Artifacts: `results/phase2_rescue_sensitivity_20260826/`.

## Rescue-Only: 32 FAN Failures

- Rescue MAT preparation: 32/32.
- Four-model inference: 32/32 per model.
- Six-method rendering: 192/192, zero render failures.
- Full decisions before the frozen calibrator: 29 standardize, 3 reject.
- Full ArcFace cosine versus rescue-original render: mean 0.3157.
- Full gaze change versus rescue-original render: mean 78.65 degrees.
- Frozen-gate outcome audit: 16 unsafe, 16 safe-but-ineffective, 0 safe-and-effective.
- Frozen-gate decisions: 0 standardize, 3 weak-standardize, 29 reject.
- Valid rescue under the registered rule: 0/32.
- Policy coverage: primary 743/775; primary plus valid rescue 743/775.

Artifacts: `results/phase2_rescue_only_20260826/`.

## Validation Rendering and Gate Calibration

- Validation inputs: 1,440/1,440 MAT and 1,440/1,440 OOF XGBoost rows.
- Full inference: 1,440/1,440.
- Original, hard-zero, and Full renders: 4,320/4,320, zero render failures.
- Outcome counts: 167 unsafe, 1,270 safe-but-ineffective, 3 safe-and-effective.
- Frozen engineering margins: identity 0.02 cosine, pose 0.03 norm, expression 0.02 RMS norm, gaze 10 degrees.
- Current reject score: AUROC 0.5184, AUPRC 0.1364.
- Selected 18-feature logistic OOF: AUROC 0.6049, AUPRC 0.1590, Brier 0.1018, ECE 0.0127.
- Five-percent risk point: 1.46% coverage.
- Ten-percent risk point: 74.31% coverage; validation FAR 64.07%, FRR 24.35%.
- Fifteen-percent risk point: 100% coverage because validation prevalence is 11.60%.

Artifacts: `results/phase2_gate_calibration_20260826/calibrator/`.

## Frozen Fixed-Test Audit

- Thresholds and coefficients were frozen on validation before this audit.
- Available Phase2 samples: 743; upstream FAN/DECA failures: 32.
- Outcome counts among available samples: 318 unsafe, 386 safe-but-ineffective, 39 safe-and-effective.
- Decisions over all 775: 204 standardize, 232 weak-standardize, 339 reject.
- Accepted available samples: 436/743 (58.68%).
- Accepted-set unsafe rate: 44.27%.
- FAR: 60.69%; FRR: 42.82%.
- This frozen calibrator is not deployment-qualified. The fixed test was not used to retune it.

Artifacts: `results/phase2_gate_calibration_20260826/fixed_test_audit/`.

## Remaining Work Held by Request

- Generate final figures and tables.
- Generate the final artifact hash manifest.
- Write the research conclusions and limitations section.
- Do not tune the frozen gate on the 775 fixed test. A future gate improvement requires outcome-supervised training or a new independent calibration split.
