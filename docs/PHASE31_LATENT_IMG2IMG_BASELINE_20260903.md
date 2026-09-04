# Phase3.1 Latent Img2Img Baseline

## Material Passport

- Date: 2026-09-03
- Stage: Phase3.1 source-latent reconstruction diagnostic
- Data: the same 32 train-only IDs used by the 64-step reconstruction smoke
- Models: frozen SD1.5 UNet versus the existing 64-step Face/Identity Adapter checkpoint
- Result scope: sampling, artifact integrity, and paired ArcFace/pixel audit completed

## Protocol

The source image is encoded with the frozen VAE posterior mode and scaling factor. For each image, a CPU-generated fixed noise tensor is reused across every strength and both model variants. DDIM uses 20 inference steps, `eta=0`, the cached empty prompt, and no classifier-free guidance.

Strengths are 0, 0.25, 0.5, 0.75, and 1.0, corresponding to 0, 5, 10, 15, and 20 denoising steps. Their first scheduler timesteps are null, 201, 451, 701, and 951. Strength 0 is the explicit VAE-only anchor and makes no UNet call. Strength 1 begins from the source latent noised at timestep 951; it is not the earlier pure-noise DDIM baseline.

The frozen arm calls the original UNet directly. The trained arm loads the exact adapter state from `results/phase31_train_smoke_20260902/run/checkpoint.pt`. It validates dataset/model/split/code fingerprints and the frozen-backbone hash before sampling. The run performs no optimization and uses no target geometry or gaze intervention.

## Execution Result

All 320 expected images were generated: 32 IDs × 5 strengths × 2 variants. Generation failures: 0. Wall time: 185.84 seconds. Peak allocated CUDA memory: 1811.04 MiB. The sample manifest SHA256 is `ba7e51f5abd66c904df0a58ecdb456c200e9eb399421d0179c9d209825c5f4b7`.

Outputs include full-resolution references, per-variant/per-strength images, five contact sheets, the exact command, a configuration with hashes and schedules, a JSONL sample ledger, and a summary. Every row stores the fixed-noise hash, initial noised-latent hash, output hash, status, and failure reason.

## Identity and Pixel Audit

The audit completed all 320 rows using buffalo_l on CPU, `det_thresh=0.1`, and the largest detected face while retaining no-face and multi-face flags. The source-to-VAE anchor mean ArcFace cosine was 0.916970 on 32/32 images.

| Strength | Frozen source cosine (n) | Trained source cosine (n) | Frozen no-face | Trained no-face | Frozen source MAE | Trained source MAE |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.916970 (32) | 0.916970 (32) | 0 | 0 | 0.01659 | 0.01659 |
| 0.25 | 0.318213 (32) | 0.548950 (32) | 0 | 0 | 0.05449 | 0.03759 |
| 0.50 | 0.046180 (31) | 0.434439 (32) | 1 | 0 | 0.12709 | 0.05575 |
| 0.75 | 0.027641 (28) | 0.289903 (32) | 4 | 0 | 0.22935 | 0.09020 |
| 1.00 | 0.017551 (15) | 0.212600 (28) | 17 | 4 | 0.32627 | 0.17286 |

On the shared detectable subset, trained-minus-frozen source-cosine mean differences were +0.2307 at strength 0.25, +0.3883 at 0.50, +0.2630 at 0.75, and about +0.18 at 1.00. Different unpaired counts at higher strengths are why the paired differences and each arm's overall means are reported separately.

The adapter consistently improves identity retention and reduces pixel drift relative to the frozen UNet for every positive strength. Among positive strengths, 0.25 is the strongest current reconstruction operating point: 32/32 faces detected, mean source cosine 0.548950, median 0.560216, p10 0.454111, and mean RGB MAE 0.03759. It is still substantially below the VAE anchor and therefore does not pass the Phase3.1 identity-quality gate. Strength 0 only reproduces the VAE anchor and cannot be interpreted as learned reconstruction or controllability. These are train-smoke diagnostics, not held-out generalization or target-condition control evidence.

Audit artifacts are `identity_audit/metrics.csv` and `identity_audit/summary.json`. The summary records evaluator hashes, complete denominators, per-arm statistics, paired comparisons, and reference metrics.

## Verification

`python -m tests.test_phase31_img2img` passed schedule boundaries, the strength-zero bypass, source-plus-noise initialization, an oracle DDIM trajectory, deterministic replay, paired-denominator accounting, null metrics, and strict checkpoint key/shape/finite checks.

`python -m tests.test_phase31_reconstruction` also passed the existing split isolation, rescue rejection, source-only input, uint16 depth, zero-init parity, adapter-gradient/update, frozen-backbone, and branch-off checks.

## Entry Points

- Sampling: `python -m phase3.sample_latent_img2img`
- Paired audit: `python -m phase3.evaluate_latent_img2img`
- Protocol tests: `python -m tests.test_phase31_img2img`
- Local run: `results/phase31_img2img_20260903`