# Phase3.1c Geometry Causal Audit Result

## Material Passport

- Execution date: 2026-09-09
- Machine: RTX 5060 Laptop, 8 GB
- Code: `main@9152db890cc24c535a63333a25e8a8fa99c88b6b`
- Data role: 32 geometry-delta-stratified validation samples
- Optimization: none
- Fixed test/rescue/gaze: not used
- Verification status: `VERIFIED`

## Protocol Integrity

All eight execution steps exited with code 0. Target provenance was strictly verified for 32/32 samples with `condition_lineage_mode=generation_time`, no warnings, and no errors.

- Phase2 Full checkpoint: `bc520af061812d9a52e3793729a70e5a0a693d0b1179b60ab35e9adc3cb2a004`.
- XGBoost OOF: `85ed5ac8749c6ae681a4ef542863551c02041dbf9b733862948d6a8bef0eb846`.
- Phase2 inference: 1,440/1,440 validation outputs; 1,425 standardize, 1 weak, 14 reject.
- Selection: 16 high, 8 medium, 8 low; fixed-test overlap 0.
- Condition cache and dataset: 32/32 available; no missing fields; no resume.
- Sampling: 320/320 outputs, no generation failures; 1,811 MiB peak allocated GPU memory; 153.22 seconds.
- Evaluation: 320/320 rows; ArcFace and FAN/DECA completed; protocol tests passed.

The geometry intervention was nontrivial. Source-target six-channel condition L1 was nonzero for all 32 samples, with mean 0.09337, median 0.10155, and range 0.05613-0.12488. The selected source-to-target pose changes were also large:

| Tier | n | Mean pose delta | Range | Mean expression RMSE |
| --- | ---: | ---: | ---: | ---: |
| High | 16 | 25.43 deg | 22.95-28.91 deg | 0.3489 |
| Medium | 8 | 18.27 deg | 14.34-22.70 deg | 0.2905 |
| Low | 8 | 13.68 deg | 11.39-14.58 deg | 0.1865 |

This rules out an explanation based on nearly identical source and target conditions.

## Main Results

### Strength 0.25

| Geometry arm | Pose error | Expression RMSE | Landmark shape NME | Source cosine, single-face |
| --- | ---: | ---: | ---: | ---: |
| Source | 14.0109 deg | 0.33046 | 0.105270 | 0.52543 (23/32) |
| Target | 14.0097 deg | 0.33048 | 0.105262 | 0.52524 (23/32) |
| Zero | 14.0083 deg | 0.33107 | 0.105199 | 0.52445 (23/32) |
| Shuffled | 13.9949 deg | 0.33025 | 0.105163 | 0.52446 (23/32) |

Target geometry did not beat all controls. Compared with shuffled geometry, target geometry was slightly worse for pose and landmarks. The absolute differences were very small.

### Strength 0.50

| Geometry arm | Pose error | Expression RMSE | Landmark shape NME | Source cosine, single-face |
| --- | ---: | ---: | ---: | ---: |
| Source | 13.8873 deg | 0.32165 | 0.104622 | 0.37819 (22/32) |
| Target | 13.8690 deg | 0.32249 | 0.104547 | 0.37811 (22/32) |
| Zero | 13.8826 deg | 0.32198 | 0.104712 | 0.37685 (22/32) |
| Shuffled | 13.8868 deg | 0.32193 | 0.104627 | 0.37831 (22/32) |

For target minus source geometry:

- Pose: -0.01835 deg, paired bootstrap 95% CI [-0.03578, -0.00482].
- Expression: +0.000837, 95% CI [0.000154, 0.001708], which is in the wrong direction.
- Landmark: -0.0000747, 95% CI [-0.000176, 0.0000028].

The pose interval excludes zero, but a 0.018-degree change against approximately 13.9 degrees of absolute error is practically negligible. Target did not reliably beat zero or shuffled geometry, and expression became worse.

## Adapter Mechanism

Face Adapter residuals were nonzero, but zero-geometry inputs also produced substantial residuals:

| Scale | Target mean abs / RMS | Zero mean abs / RMS |
| --- | ---: | ---: |
| 0 | 0.00752 / 0.01053 | 0.00375 / 0.00436 |
| 1 | 0.02510 / 0.03835 | 0.01501 / 0.01833 |
| 2 | 0.05095 / 0.07871 | 0.04637 / 0.06181 |
| 3 | 0.04936 / 0.06652 | 0.05077 / 0.06758 |

Nonzero residual magnitude therefore does not establish condition use. The deep-scale zero-input response suggests that learned biases or nearly condition-invariant components contribute strongly. Source, target, and shuffled residual magnitudes are almost identical at the aggregate level.

Identity remained responsive: shuffling identity lowered single-face cosine from 0.3781 to 0.3448 at strength 0.50. This is consistent with the earlier finding that the identity branch dominates the learned behavior.

## Decision

Phase3.1c passes its engineering and provenance gates but fails the geometry-control scientific gate.

The supported conclusion is:

> The current 64-step adapter learned an identity-related denoising prior, but the Face Adapter does not provide useful or practically meaningful control of Phase2 target geometry.

The result does not support 3D standardization, geometry controllability, or gaze disentanglement. Additional sampling with the same checkpoint is unlikely to resolve the problem.

## Limitations

- The 32 samples deliberately over-sample large geometry changes and are not population representative.
- DECA is used both to construct target conditions and to evaluate generated geometry, so these are model-domain diagnostics rather than independent human measurements.
- Shuffled Phase2 targets are a weak negative control for standardization because the targets are intentionally near-canonical. In the Phase3.1d train selection, pairwise target-target pose distance averaged 0.87 degrees and expression RMSE averaged 0.0153. Failure to distinguish shuffled targets is therefore not sufficient by itself to reject standardization; the stronger failure is that target geometry did not meaningfully beat source or zero geometry.
- ArcFace primary identity statistics require exactly one detected face in both source and output; coverage was 23/32 at strength 0.25 and 22/32 at 0.50.
- No calibrated identity threshold or fixed-test evaluation was performed.

## Next Stage

Proceed to Phase3.1d only as a bounded train-only geometry-supervision experiment. Do not scale the current loss to all 8,160 samples and do not enable gaze training yet. The next experiment must make target geometry part of the optimization objective and demonstrate held-out causal response before formal training.
