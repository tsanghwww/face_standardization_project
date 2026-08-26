# Phase2 Execution Status

## Material Passport

- Artifact type: experiment execution status
- Project: DECA parameter-space face standardization
- Date: 2026-08-25
- Compute node: `win-lenovo`, NVIDIA GeForce RTX 5060 Laptop GPU (8,151 MiB)
- Fixed split: `results/phase2_eval_fixed_20260824_v2/fixed_test_manifest_v2.csv`
- Fixed seed: `20260824`
- Status: formal training, inference, six-way rendering, unified metrics, and paired ablation statistics complete

## Research Objective

Phase2 trains a quality-aware condition generator in DECA parameter space. It does not train DECA, FAN, ArcFace, L2CS, or an image generator. Given DECA parameters and quality features, the model predicts target expression/pose parameters, adaptive standardization strengths, confidence, and rejection score.

The standardized expression is representative of the three parameter groups:

$$
\boldsymbol{\beta}_{\mathrm{exp}}'=
(1-\alpha_{\mathrm{exp}})\boldsymbol{\beta}_{\mathrm{exp}}+
\alpha_{\mathrm{exp}}\hat{\boldsymbol{\beta}}_{\mathrm{exp}}^{*}.
$$

The same interpolation is applied separately to head pose and jaw pose.

## Fixed Evaluation Split

The immutable test split contains 775 unique samples:

| Source group | Count |
|---|---:|
| XGBoost high | 100 |
| XGBoost medium | 100 |
| XGBoost low | 100 |
| Base hard pose | 50 |
| Base low landmark | 50 |
| WIDER pose | 75 |
| WIDER occlusion | 75 |
| WIDER blur | 75 |
| COFW occlusion | 75 |
| AFLW large pose | 75 |

The 400 internal test IDs are stored in `base_test_ids.txt` and must be excluded from every learned component, including XGBoost and the Phase2 condition generator.

## External Preprocessing Results

### FAN/DECA Main Path

- External samples: 375
- Successful DECA extraction: 343
- FAN failures: 32
- All failures: `fan_no_face`
- All failures are WIDER samples
- WIDER failure rate: $32/225=14.2\%$
- COFW success: 75/75
- AFLW success: 75/75
- Silent fallback count: 0
- Wall time: 1,656 seconds
- Peak GPU memory: 6.9 GB

The 32 failures remain in the fixed manifest and in the primary end-to-end denominator. They are not silently removed or replaced.

### Rescue Path

The whole-image warp rescue path succeeded on all 375 external samples. Rescue outputs are stored separately and must not replace the primary FAN results. Rescue is a fallback-policy sensitivity analysis.

### External ArcFace

- Success: 292/375 (78%)
- Failure: 83/375
- WIDER: 212 success, 13 failure
- COFW: 41 success, 34 failure
- AFLW: 39 success, 36 failure

ArcFace failure is preserved as a training/inference quality feature (`arcface_status=0`). The render-domain identity metric is computed separately from directly embedded, already aligned DECA renders.

## Leakage-Free XGBoost Protocol

- Original base pool: 10,000
- Fixed internal test IDs excluded: 400
- XGBoost training pool: 9,600
- Five-fold OOF predictions: 9,600
- Fixed-test predictions covered: 743/775
- Missing predictions: 32 upstream FAN/DECA failures
- Covered fixed-test labels: high 157, medium 62, low 524
- Final model SHA256: `d87dd136d157467ccf7491189337711702ce10e8638b855287ff24631d1735dc`

Phase2 training must consume OOF XGBoost scores. Fixed-test inference must consume predictions from the final XGBoost model fitted only on the 9,600-sample training pool. Missing upstream features must never be replaced by fabricated XGBoost scores.

## Phase2 Input and Models

The condition generator input dimension is 99:

$$
50_{\mathrm{expression}}+6_{\mathrm{pose}}+3_{\mathrm{camera}}+
27_{\mathrm{light}}+13_{\mathrm{quality}}=99.
$$

Four models require training:

| Model | Quality source | Alpha | Latent augmentation |
|---|---|---|---|
| Full | heuristic/XGBoost blend | learned | enabled |
| Fixed-alpha=1 | heuristic/XGBoost blend | fixed one | enabled |
| No augmentation | heuristic/XGBoost blend | learned | disabled |
| No XGBoost | heuristic only | learned | enabled |

The hard-zero method is an untrained baseline. Original is a reference render, not a trained model.

All trained models use the same seed, stage 3, 40 epochs, batch size 64, learning rate $10^{-3}$, hidden dimension 256, train/validation IDs, and best-validation-loss checkpoint rule.

## Evaluation Protocol

Primary reporting uses all 775 samples as the end-to-end denominator. The 32 FAN failures are upstream pipeline failures for every method that requires DECA parameters.

Quality metrics are also reported conditionally on paired successful samples, with the actual sample count shown. Metrics include:

- render failure rate;
- ArcFace embedding availability and diagnostic cosine similarity to the original render;
- DECA head-pose norm and change from original;
- L2CS detection rate and gaze-angle change from the original render;
- paired bootstrap 95% confidence intervals;
- method, source group, source dataset, and XGBoost-tier stratification.

ArcFace and L2CS measurements on DECA shape-detail renders are diagnostic and are not treated as equivalent to measurements on photorealistic images.

For ArcFace, the identity-preservation reference is the same sample's `original` DECA render. Because DECA outputs are already aligned square renders, they are passed directly to the ArcFace recognition backbone; applying the face detector to these texture-poor renders produced no detections and an unusable metric. Thus:

$$
s_{m,i}=\frac{f(R_{\mathrm{original},i})^\top f(R_{m,i})}
{\lVert f(R_{\mathrm{original},i})\rVert_2\lVert f(R_{m,i})\rVert_2}.
$$

This is a within-render-domain diagnostic of identity-feature preservation. It must not be reported as real-photo face-verification accuracy.

## Formal Training Results

All four formal models completed 40 epochs on the same 8,160/1,440 train/validation split. Checkpoints were selected by minimum deterministic validation loss.

| Model | Best epoch | Best validation loss | Checkpoint SHA256 |
|---|---:|---:|---|
| Full | 37 | 0.147455 | `BC520AF061812D9A52E3793729A70E5A0A693D0B1179B60AB35E9ADC3CB2A004` |
| Fixed-alpha=1 | 40 | 0.139543 | `86F470891AA4748FBD2F92BA5A5860957D1EFA87A3B34381C69A4E889CF2595A` |
| No augmentation | 35 | 0.145133 | `3770E9C98F71D63A72058802F1032D27F5FB10F00A38910EC37F0C91AEA28F07` |
| No XGBoost | 39 | 0.160422 | `AB5051A4210005A38D40FAA8B8E549811DC18A0871DDC5427BC3AFE8CD5B4564` |

The lower validation loss of Fixed-alpha=1 does not by itself establish that fixed alpha is preferable. Final selection depends on the fixed-test trade-off among standardization, identity preservation, gaze/pose behavior, and rejection/failure rates.

## Fixed-Test Inference

FAN/DECA parameters and complete sidecars were available for 743/775 samples. All four models produced 743 finite NPZ outputs; the same 32 WIDER FAN failures remained unavailable upstream.

| Model | Standardize | Weak | Reject | Quality source |
|---|---:|---:|---:|---|
| Full | 628 | 3 | 112 | blend |
| Fixed-alpha=1 | 499 | 1 | 243 | blend |
| No augmentation | 699 | 0 | 44 | blend |
| No XGBoost | 743 | 0 | 0 | heuristic |

Observed alpha ranges were 0.544-0.841 for Full, exactly 1 for Fixed-alpha=1, 0.552-0.869 for No augmentation, and 0.707-0.820 for No XGBoost.

## Rendering Result

The unified renderer generated `original`, `hard_zero`, `full`, `no_alpha`, `no_augmentation`, and `no_xgboost` outputs with one shared DECA/FAN encode per source image.

- Expected render attempts per method: 775
- Successful renders per method: 743
- Upstream failures per method: 32
- Unexpected decode, parameter, or image-write failures: 0
- Total successful method renders: $743\times6=4,458$

## Unified Metric Result

The formal run produced all $775\times6=4,650$ expected metric rows and unique `(eval_id, method)` pairs. The 32 upstream FAN failures appear once per method. ArcFace and DECA succeeded on all 743 available renders for every method. L2CS additionally failed on 57 method-render pairs because its detector produced an empty crop; these failures remain in the method denominator.

| Method | Render failure | ArcFace cosine | DECA pose norm | Pose delta vs original | Gaze delta vs original |
|---|---:|---:|---:|---:|---:|
| Original | 4.13% | 1.0000 | 0.4032 | -- | -- |
| Hard-zero | 4.13% | 0.3652 | 0.2071 | 0.2090 | 52.96 deg |
| Full | 4.13% | 0.3688 | 0.2018 | 0.2116 | 51.14 deg |
| Fixed-alpha=1 | 4.13% | 0.3652 | 0.2072 | 0.2090 | 53.51 deg |
| No augmentation | 4.13% | 0.3675 | 0.1897 | 0.2165 | 47.26 deg |
| No XGBoost | 4.13% | 0.3651 | 0.1989 | 0.2139 | 51.47 deg |

All quality values above are conditional on metric availability; failure rates use all 775 samples. Gaze deltas use 729-739 paired successes depending on the method.

### Paired Bootstrap Findings

Compared with hard-zero, Full has:

- ArcFace cosine delta $+0.00359$ (95% CI $[+0.00146,+0.00574]$);
- DECA pose-norm delta $-0.00525$ (95% CI $[-0.00961,-0.00072]$);
- gaze-delta change $-1.96$ deg (95% CI $[-3.80,-0.18]$).

These improvements are statistically detectable under paired resampling but small in magnitude. Full also outperforms Fixed-alpha=1 in identity cosine, pose norm, and gaze preservation, supporting learned partial alpha. Removing XGBoost reduces identity cosine by 0.00367 relative to Full, while slightly reducing pose norm; its gaze difference is not statistically resolved.

No augmentation produces a 0.01210 lower pose norm and a 4.26 deg lower gaze delta than Full, with no resolved ArcFace difference. Therefore the current latent augmentation recipe is not supported by this fixed-test result and should be tuned or weakened rather than claimed as beneficial.

### Decision-Aware Delivery

Rendered metrics above include counterfactual outputs even when the model gate says `reject`. Operational delivery rates are:

| Model | Accepted outputs | Reject | Upstream failure | End-to-end accept rate |
|---|---:|---:|---:|---:|
| Full | 631 | 112 | 32 | 81.42% |
| Fixed-alpha=1 | 500 | 243 | 32 | 64.52% |
| No augmentation | 699 | 44 | 32 | 90.19% |
| No XGBoost | 743 | 0 | 32 | 95.87% |

The absence of any rejection in No XGBoost is not automatically an advantage: it shows that the heuristic-only gate is poorly selective under the current thresholds. Gate calibration and error inspection are still required.

## Completed Gates

- [x] Immutable 775-sample fixed test split
- [x] Unique image IDs and valid source paths
- [x] FAN failures exposed without silent fallback
- [x] Separate rescue outputs
- [x] External ArcFace extraction
- [x] XGBoost retrained after fixed-test exclusion
- [x] Five-fold OOF XGBoost predictions
- [x] Fixed-test XGBoost predict-only path
- [x] Fixed train/validation indices
- [x] Validation augmentation disabled
- [x] Normalizer restricted to the training subset
- [x] Deterministic validation loss
- [x] Fixed-test leakage test
- [x] Paired bootstrap implementation
- [x] Command/config provenance artifacts

## Remaining Gates

- [x] Normalize OOF manifest fields to `xgb_quality_score` and `xgb_quality_label`
- [x] Run four one-epoch preflight trainings
- [x] Verify identical split hashes and complete OOF coverage
- [x] Train four formal 40-epoch models
- [x] Run fixed-test inference for all four models
- [x] Render original, hard-zero, and four Phase2 variants
- [x] Compute identity, pose, gaze, failure, and stratified metrics
- [ ] Run rescue fallback sensitivity analysis
- [ ] Audit inventories, hashes, commands, and reproducibility
- [ ] Produce final Phase2 tables, plots, and research conclusions

## Completion Estimate

Two completion measures are tracked:

- Engineering and protocol readiness: approximately 97% complete.
- Final empirical evidence package: approximately 90% complete.

The remaining work is concentrated in rescue-policy sensitivity, gate calibration/error inspection, final figures, and the reproducibility audit. The core learned models, primary renders, raw metrics, paired comparisons, and decision-aware tables already exist.

## Preflight Result

The four one-epoch runs passed the gate:

| Model | One-epoch validation loss |
|---|---:|
| Full | 0.206069 |
| Fixed-alpha=1 | 0.195475 |
| No augmentation | 0.163839 |
| No XGBoost | 0.213009 |

- Train ID SHA256: `8198837E4463DC9AB9807416A75E8D974304BCD1773BCCCD2B5312D54AA75F73`
- Validation ID SHA256: `9877E1C43E648A714985D363C609E15112A3AD0CF9CBECDF6591D809F3CEF2F3`
- OOF rows: 9,600 unique IDs, zero missing scores
- Input dimension: 99 for all models
- Required preflight artifacts: complete
- Fixed-test overlap: zero

The first preflight attempt exposed a Windows UTF-8 BOM in an intermediate OOF CSV, which caused the XGBoost rows to be ignored. The OOF generator now writes the standard Phase2 fields directly without BOM. The XGBoost model was rebuilt deterministically and retained SHA256 `d87dd136d157467ccf7491189337711702ce10e8638b855287ff24631d1735dc`. The corrected Full and No-XGBoost losses differ, confirming that the blend input is active.

## Immediate Next Gate

Evaluate the 32 FAN failures under the separate rescue policy, inspect false accepts/rejects by quality tier, and generate the final figures and artifact inventory. Do not claim that latent augmentation improves robustness unless a revised augmentation ablation reverses the current result on the immutable test split.

The pre-registered execution details and completion criteria for the first two work packages are in `docs/PHASE2_RESCUE_GATE_PROTOCOL_20260826.md`.

## Key Artifacts

- Status document: `docs/PHASE2_EXECUTION_STATUS_20260825.md`
- Formal run root on 5060: `results/phase2_ablation_20260825/`
- Unified render manifest: `renders_all/render_manifest_all.csv`
- Raw metrics: `metrics_all/rendered_metrics.csv`
- Method summary: `metrics_all/metrics_by_method.csv`
- Method-by-group/dataset/XGBoost tables: `metrics_all/metrics_by_method_and_*.csv`
- Paired bootstrap comparisons: `metrics_all/paired_method_comparisons.csv`
- Decision-aware acceptance table: `metrics_all/decision_aware_summary.csv`
- Machine-readable summary: `metrics_all/ablation_summary.json`
