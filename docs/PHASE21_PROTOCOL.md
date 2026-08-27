# Phase2.1 Protocol (outcome supervision + frozen gate + rescue isolation)

Skeleton only — no formal training, no threshold tuning on the 775 fixed test,
no commit/push.

## Data flow

```
rendered_metrics.csv (original/hard_zero/full per eval_id)
phase2_inference_manifest.csv (image_id -> quality/alpha/standardized)
xgb_oof_manifest.csv (image_id -> quality features)
        |  build_outcome_supervision_manifest.py  (keyed by image_id, dedup)
        v
outcome_manifest.csv  (per image_id: render status, deltas, unsafe, features)
        |
        +-- train_outcome_surrogate.py  -> outcome_surrogate.pt (4 heads + frozen normalizer)
        |
        +-- make_gate_splits.py (seed=20260827, 70/30 stratified by unsafe+xgb_label)
        |        -> gate_train_ids.txt, hard_calibration_ids.txt
        +-- train_outcome_gate.py (fit logistic on gate_train) -> outcome_gate_model.json
        +-- calibrate_outcome_gate.py (threshold on hard_calibration only) -> outcome_gate_frozen.json
        +-- apply_phase21_gate.py (frozen gate on fixed test, once, no threshold search)
```

`unsafe` label comes from the rendered outcome deltas vs hard_zero (identity
non-inferiority, pose/expression/gaze degradation) with CLI margins; missing
metrics are `unsafe` + `missing_required_metric`, never filled with 0.

## Splits

- gate_train / hard_calibration are mutually exclusive, stratified by
  `(unsafe, xgb_quality_label)`, ratio `--train-ratio` (default 0.70),
  seed `20260827`.  Fixed-test IDs are excluded; overlap with both splits is
  asserted to be 0.
- All new split writers dedup by `image_id` and record seed + input hash + ID files.

## Loss interface (train_condition_generator.py)

- `--outcome-manifest`, `--outcome-loss-weight` optionally supervise confidence
  from known labels; rows without labels are masked rather than assigned 0.5.
- `--outcome-surrogate` loads a frozen four-head surrogate. Identity
  non-inferiority, pose improvement, gaze ceiling, and render-failure losses
  are controlled by separate `--outcome-*-weight` arguments, all defaulting to 0.
- Live standardized expression/pose norms and live alphas are fed into the
  frozen surrogate, so enabled head losses propagate to the condition generator.
- A surrogate trained on any condition-generator validation ID is rejected by
  default. `--allow-outcome-validation-overlap` exists for smoke diagnostics
  only and is forbidden for formal experiments.
- Off by default means Phase2 v1 behavior remains unchanged.

## Gate freeze flow

1. `train_outcome_gate.py` fits median imputation and logistic coefficients on
   gate_train only. Missing indicators are appended and all preprocessing is
   frozen in a model with `threshold: null`.
2. `calibrate_outcome_gate.py` selects + freezes a single threshold on
   hard_calibration only. Qualification requires the one-sided Wilson risk
   upper bound at `--risk-confidence` to meet `--target-risk`. If no threshold
   qualifies, it writes `threshold: null` and `deployment_qualified: false`.
3. `apply_phase21_gate.py` reads the frozen threshold and applies it once to the
   fixed test; it raises if the gate has no threshold (no test threshold search).
   Prediction features and optional outcome labels are separate inputs, so
   labels cannot alter decisions. Explicit upstream failures become manual review.

## Rescue policy

- rescue lives in a separate manifest + separate directory (`deca_params_rescue`).
- primary inference treats FAN/DECA failure as reject/manual_review and never
  auto-reads rescue mats.
- only `apply_phase21_gate.py --rescue-audit --rescue-manifest ...` may consult
  rescue, and it writes `phase21_rescue_audit.csv` without changing decisions.
- `test_phase21_protocol.py::test_rescue_isolation` asserts the primary manifest
  mat_path never references the rescue dir.

## Known limitations

- Outcome / gate labels are DECA-render diagnostic domain, not real ArcFace/L2CS
  measurements (surrogate heads and gate risk are flagged accordingly).
- The current 1,440 outcome rows come from the old condition validation split.
  They are valid for smoke only. Formal Phase2.1 requires candidate outcomes
  generated from the 8,160 condition-training IDs, grouped by image ID before
  splitting, so surrogate training cannot leak validation outcomes.
- The render-failure head has no positive examples in the current 1,440 rows and
  must stay at weight 0 until candidate rendering supplies real failures.
- Gate is logistic over 18 source-quality features plus 18 missing indicators;
  no hyperparameter or threshold search is performed on fixed test.

## Reviewed smoke result (2026-08-27)

- Outcome rows: 1,440; unsafe 167; missing required metrics 0.
- Corrected three-epoch surrogate: identity-delta MAE 0.0105, pose-improvement
  MAE 0.0547, gaze MAE 13.92 degrees. Render-failure remains disabled because
  the validation target has one class.
- Gate calibration: AUROC 0.5898, AUPRC 0.1821, Brier 0.1031, ECE 0.0177.
- No threshold satisfies a 10% accepted-risk target at 95% one-sided confidence.
  The best observed bound is 11.79% at 41.34% coverage (empirical risk 7.82%),
  therefore `threshold=null` and `deployment_qualified=false` are required.
- This smoke does not authorize fixed-test application or formal Phase2.1
  training. The next data task is candidate rendering on condition-training IDs.

## Smoke commands (CPU, no GPU training)

```
PY = D:\face_standardization_project\.venv\Scripts\python.exe
GC  = D:\face_standardization_project\results\phase2_gate_calibration_20260826

# 1. build the 1440-row validation outcome manifest
%PY% -m phase2.build_outcome_supervision_manifest \
  --metrics-csv %GC%\metrics\rendered_metrics.csv \
  --inference-manifest %GC%\inference\full\phase2_inference_manifest.csv \
  --xgb-oof-manifest %GC%\validation_xgb_oof.csv \
  --id-manifest %GC%\validation_manifest.csv \
  --split validation --out-dir results\phase21_smoke\outcome

# 2. gate splits (70/30, seed 20260827, exclude fixed test)
%PY% -m phase2.make_gate_splits \
  --outcome-manifest results\phase21_smoke\outcome\outcome_manifest.csv \
  --exclude-ids-file results\phase2_eval_fixed_20260824_v2\base_test_ids.txt \
  --seed 20260827 --train-ratio 0.7 --out-dir results\phase21_smoke\split

# 3. fit gate on gate_train
%PY% -m phase2.train_outcome_gate \
  --outcome-manifest results\phase21_smoke\outcome\outcome_manifest.csv \
  --gate-train-ids results\phase21_smoke\split\gate_train_ids.txt \
  --out-dir results\phase21_smoke\gate

# 4. freeze threshold on hard_calibration
%PY% -m phase2.calibrate_outcome_gate \
  --outcome-manifest results\phase21_smoke\outcome\outcome_manifest.csv \
  --hard-calibration-ids results\phase21_smoke\split\hard_calibration_ids.txt \
  --gate-model results\phase21_smoke\gate\outcome_gate_model.json \
  --target-risk 0.10 --risk-confidence 0.95 \
  --out-dir results\phase21_smoke\calibrate

# 5. surrogate (multi-head) smoke
%PY% -m phase2.train_outcome_surrogate \
  --outcome-manifest results\phase21_smoke\outcome\outcome_manifest.csv \
  --epochs 3 --batch-size 16 --device cpu \
  --identity-weight 1 --pose-weight 1 --gaze-weight 1 --render-failure-weight 0 \
  --out-dir results\phase21_smoke\surrogate

# 6. protocol tests
%PY% -m phase2.test_phase21_protocol
```
