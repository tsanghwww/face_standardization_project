# Phase3.1d Geometry Supervision Result

## Material Passport

- Execution date: 2026-09-09
- Machine: RTX 5060 Laptop, 8 GB
- Code: `main@989e51a`
- Data role: 32 high-geometry-delta train-only samples
- Fixed test and Phase3.1c validation: not used
- Rescue and gaze: disabled
- Decision: `STOP_BEFORE_VALIDATION`

## Data And Protocol

The selection contains 32 unique train IDs and has zero overlap with the Phase3.1c validation selection, the complete validation registry, and the 775 fixed-test IDs. Its mean source-to-target pose delta is 26.25 degrees and mean expression RMSE is 0.3675.

The differentiable path was:

`Face Adapter -> frozen UNet -> one-step x0 -> frozen VAE -> fixed warp -> frozen DECA`

All four Face Adapter scales received finite nonzero gradients. Preflight peak allocated GPU memory was 6.51 GiB, below the 7.2 GiB budget. The 64-step run completed with 5,427 MiB peak allocated memory. The identity branch and UNet remained unchanged.

The original `geometry_loss_plus_ranking` run is invalid and excluded from all decisions because:

1. Pose error in degrees was incorrectly normalized by `radians(45)` rather than 45 degrees, inflating the intended pose contribution by 57.30 times.
2. A shuffled Phase2 target was used as the negative even though Phase2 targets are near-canonical. Among these 32 samples, pairwise target-target pose distance averaged only 0.87 degrees and pairwise target-target expression RMSE averaged 0.0153.

The corrected run used the same sample's source geometry as the paired ranking negative, with identical latent, noise, timestep, and source identity.

## Corrected 64-Step Result

At the fixed step-64 train diagnostic:

| Arm | Pose error | Expression RMSE | Landmark NME |
| --- | ---: | ---: | ---: |
| Source geometry | 17.246809 deg | 0.2276925 | 0.1151200 |
| Target geometry | 17.251842 deg | 0.2286109 | 0.1151815 |
| Zero geometry | 18.681563 deg | 0.2334378 | 0.1239901 |
| Shuffled geometry, descriptive only | 17.289109 deg | 0.2288169 | 0.1152782 |

Target minus source was worse for all three outcomes:

- Pose: +0.005033 degrees.
- Expression: +0.0009184.
- Landmark: +0.0000615.

The final per-step ranking loss was 0.04908 against a margin of 0.05, indicating almost no target-versus-source separation. Zero geometry was worse, but source, target, and shuffled nonzero conditions remained nearly indistinguishable. The valid `geometry_loss_only` run showed the same pattern.

## Interpretation

Phase3.1d did not pass its train-level causal-response gate. The current objective primarily learned a condition-present or dataset-level canonicalization response, not a response to the requested target geometry. Since separation is absent even on the 32 optimization samples, validation and 256-step extension are not justified.

This run does not prove that the injection architecture can never learn geometry. The ranking contribution was also severely under-scaled: at step 64, `0.1 * ranking` was approximately 0.0049 while the geometry loss was approximately 34.6. A no-update gradient contribution audit is required to distinguish insufficient ranking signal from an ineffective injection path.

## Decision And Next Check

- Do not run Phase3.1c validation with this checkpoint.
- Do not run 256 steps or full-data training.
- Do not enable Eye Adapter or make gaze-disentanglement claims.
- Next, compute separate Face Adapter gradient norms for source reconstruction, target geometry, and ranking on the same fixed train batches without optimizer steps.
- Consider one further bounded run only if a frozen, train-only reweighting makes the ranking gradient contribution non-negligible. Otherwise stop Phase3.1d and redesign the geometry injection mechanism.

## Corrected Artifact Hashes

Run directory: `results/phase31d_geometry_supervision_20260909/run_geometry_loss_plus_source_ranking_corrected`

| Artifact | SHA256 |
| --- | --- |
| `config.json` | `a3e5b80aab1055e0fcb70ae104179bc5c12bfee5a839b196e32e5a6fdc0764c6` |
| `exact_command.txt` | `1e4c711c5e5cf237edd540da3e5a671d05ab61c4aaf5b9720121ca188a6101d9` |
| `preflight.json` | `8111eb973351699843787e81d4e5f092b066b5a344d356b8f959039bcf7a2816` |
| `training_log.jsonl` | `23e94704a9e76eb993062a0c78213a5980ec269c0b18a96447d21993f1e022bd` |
| `summary.json` | `7c907582ff8d920d4b8e599f398e65c3f28c6be63f2638f05797efa52bf7a21e` |
| `checkpoint_step_0064.pt` | `6903ba8b12b1167b6594dccadb7427422910aa5684d03cfff0f24f5c4b6e1e4f` |
| `diagnostics/step_0064.json` | `6f88dbc76ac46a2a338e66e416622c70bf04288aaa3dcff1c0a21a1d001aa412` |
| `diagnostics/step_0064_contact.png` | `ca9a6be0b5163145c79d2c73110c6381eba707d8ea602f13627a27139c638f59` |
