# Phase3.1f Guarded Counterfactual Geometry Result

## Material Passport

- Date: 2026-09-18
- Machine: win-lenovo, RTX 5060 Laptop 8 GB
- Code: `main@647511b3cbd09df6bbbd51839cf05330bf913bcf`
- Data: the existing 32 high-geometry-delta train-only identities
- Fixed test and Phase3.1c validation: not used
- Rescue and gaze: disabled
- Decision: `STOP_BEFORE_VALIDATION`

This experiment tested whether the existing Face Adapter can learn target-specific
geometry after correcting the loss balance and adding explicit source preservation.
It was a bounded mechanism experiment, not formal Phase3 training.

## Protocol

The frozen candidate objective was:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{src}}+
0.003\left[
\mathcal{L}_{\mathrm{geom}}(\hat x_{\mathrm{target}},y^*)+
0.3\mathcal{L}_{\mathrm{geom}}(\hat x_{\mathrm{source}},y_{\mathrm{source}})
\right]+
1.0\mathcal{L}_{\mathrm{rank}}.
$$

The internal pose/expression/landmark weights remained
`(1.0, 75.044619, 148.788535)`. Geometry timesteps were restricted to
`[100, 400]`. Source and target arms shared source latent, noise, timestep, and
identity. The identity branch, UNet, VAE, and DECA were frozen.

For each identity, fresh counterfactual conditions were rendered at `-10` and
`+10` degrees of camera-frame yaw around its canonical Phase2 target. The two
targets were separated by approximately 20 degrees. The 96 optimizer steps gave
each of 32 identities exactly one canonical, negative-yaw, and positive-yaw
exposure. No validation or fixed-test ID entered selection, optimization, or
threshold selection.

The counterfactual audit fixed latent, noise, timestep, and identity while
changing only geometry. It reported:

1. whether output relative rotation projected positively onto the requested
   target-relative rotation axis;
2. whether each output was closer to its own target than to the crossed target;
3. the actual projected output change in degrees.

The train-only diagnostic gate was frozen before the post-training audit:

- complete 96/96 denominator;
- direction and ordering both correct for at least 75% of pairs;
- median projected output change at least 0.5 degrees.

## Execution Integrity

- Counterfactual conditions: 32 identities, 64 conditions, minimum target-pair
  separation `19.9999993` degrees.
- Complete-objective preflight: finite nonzero gradients at all four Face Adapter
  scales; peak allocated memory `6.510 GiB`, below the `7.2 GiB` limit.
- Two-step smoke: completed; peak allocated memory `5420.7 MiB`.
- Bounded training: 96/96 optimizer steps completed in `219.6 s`; peak allocated
  memory `5427.2 MiB`.
- Adapter update L2: `face.=3.5120`, `identity.=0`, `unet.=0`.
- Frozen UNet hash was identical before and after training.
- Pre- and post-training counterfactual audits both completed 96/96 with zero
  failures.

## Counterfactual Result

| Metric | Original reconstruction checkpoint | Step 96 | Change |
| --- | ---: | ---: | ---: |
| Direction correct | 69/96 | 83/96 | +14 |
| Own-target ordering correct | 62/96 | 79/96 | +17 |
| Direction and ordering both correct | 62/96 | 78/96 | +16 |
| Median projected output change | 0.00186 deg | 0.08440 deg | about 45x |
| Mean projected output change | 0.00174 deg | 0.09920 deg | about 57x |
| Mean output-pair separation | 0.01256 deg | 0.27968 deg | about 22x |

By timestep after training:

| Timestep | Both correct | Median projected change | Mean output separation |
| ---: | ---: | ---: | ---: |
| 100 | 29/32 | 0.05057 deg | 0.11832 deg |
| 250 | 22/32 | 0.08646 deg | 0.24121 deg |
| 400 | 27/32 | 0.16974 deg | 0.47951 deg |

The model learned a real directional response: sign and ordering improved, and
the response magnitude increased substantially. However, the median output
change remained only `0.0844` degrees for a 20-degree target separation. The
pre-registered 0.5-degree magnitude condition failed, so the complete gate did
not pass.

## Absolute Geometry And Source Guard

The same 32 train identities at the fixed diagnostic timestep `t=250` gave:

| Arm / metric | Original checkpoint | Step 96 |
| --- | ---: | ---: |
| Target arm to canonical target, pose | 25.443 deg | 18.402 deg |
| Source arm to canonical target, pose | 25.449 deg | 24.883 deg |
| Target arm to canonical target, expression RMSE | 0.31584 | 0.23005 |
| Source arm to canonical target, expression RMSE | 0.31564 | 0.31477 |
| Target arm to canonical target, landmark NME | 0.17057 | 0.12198 |
| Source arm to canonical target, landmark NME | 0.17060 | 0.16597 |
| Source arm to source pose | 1.362 deg | 1.713 deg |
| Source arm to source expression RMSE | 0.12079 | 0.12055 |
| Source arm to source landmark NME | 0.01671 | 0.01856 |

Target conditioning now clearly beats source and zero conditions for canonical
pose, expression, and landmarks. This is stronger evidence than the failed
Phase3.1d run, where source and target conditions were indistinguishable.

Nevertheless, source-self pose worsened by about `0.351` degrees and source-self
landmark NME worsened by about `0.00185`. The source absolute term reduced the
unconstrained failure mode but did not satisfy the required no-degradation
guard. Source epsilon MSE was optimized during every step, but a separate frozen
post-training reconstruction audit was not used to override this failed
source-geometry result.

## Scientific Conclusion

This run supports a narrow claim: the current Face Adapter and loss can learn a
small target-dependent pose direction on the 32 optimization identities. It
does not support successful 3D geometry control, robust standardization, or gaze
disentanglement.

The dominant behavior remains canonicalization. Step 96 substantially improves
the canonical target, while the response to a 20-degree counterfactual target is
less than one tenth of a degree at the median. Near-canonical shuffled conditions
remain close to the canonical target arm and are still not a useful primary
negative.

Because counterfactual amplitude and source preservation both failed their
train-level requirements, this checkpoint must not be evaluated on the frozen
validation split and must not enter Phase3 gaze training. The next engineering
change should target the geometry injection mechanism rather than lengthening
this run: preserve high-resolution spatial geometry, encode target-minus-source
residuals explicitly, or apply a direct pair-separation objective. Any such
change requires a new train-only smoke and counterfactual audit before validation.

## Artifact Hashes

All paths are relative to `D:\face_standardization_project` on win-lenovo.

| Artifact | SHA256 |
| --- | --- |
| `results/phase31f_counterfactual_conditions_20260918/counterfactual_manifest.jsonl` | `bbc6ad61745d55cd72f01517b68624ff8fc62602864f129f4aa810f6ff8f9932` |
| `results/phase31f_counterfactual_conditions_20260918/summary.json` | `8b984929e472ec5087b31f11601caad613f91cf08e034ce83f2cd7d63afc6cf5` |
| `results/phase31f_counterfactual_pretrain_20260918/metrics.jsonl` | `bd484db47a93f082681a4d0081160f7fc920ae9a8864d81453a6cba42fcea38b` |
| `results/phase31f_counterfactual_pretrain_20260918/summary.json` | `485fac90a5c37e527f4eb77c1f6bd5b489da23728d829e5ad704ac09b1005834` |
| `results/phase31f_full_objective_smoke_20260918/preflight.json` | `64783220055d7ac29899a57685ce7416d09c218ba02303f0ad472b413e4da981` |
| `results/phase31f_guarded_counterfactual_96step_20260918/config.json` | `84aa39225f58f2c4962bb4dd05f5faee6f5673d8045582f91ac82b8db1426d32` |
| `results/phase31f_guarded_counterfactual_96step_20260918/exact_command.txt` | `2f6fc3686536a915a92e49c526e5ef414612be4b29a7fc77db3bf725e9b9c9ff` |
| `results/phase31f_guarded_counterfactual_96step_20260918/preflight.json` | `6a9a3db691e4c6b77c2253f7f3e6822e44efd0e3e0e98f2d6428aaabe1eceacf` |
| `results/phase31f_guarded_counterfactual_96step_20260918/training_log.jsonl` | `e31a987c579a4eff92cdb62ea9c751334fbe199e19055ebea7c2dc6fe791c16a` |
| `results/phase31f_guarded_counterfactual_96step_20260918/checkpoint_step_0096.pt` | `303225e250d7fec4fbf50cd2834803715c68f26700845d73b943661d05ef7c07` |
| `results/phase31f_guarded_counterfactual_96step_20260918/diagnostics/step_0096.json` | `77f4c94b3b1800a0e6a28c7327bbc2695816910f8e0b09cff9bbf99cbf0a281f` |
| `results/phase31f_counterfactual_posttrain_20260918/metrics.jsonl` | `a2a367a1763599ce31aa535491acacf5b506a8a44239ac1587e2185c00223148` |
| `results/phase31f_counterfactual_posttrain_20260918/summary.json` | `c539206c2585631b0b86b04dcd3db40d4556e79fc875ee5604cf184d73d74e92` |
