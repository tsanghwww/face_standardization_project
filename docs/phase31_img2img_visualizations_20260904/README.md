# Phase3.1 Img2Img Paper Visualizations

Exported on 2026-09-04 from the train-only 32-image Phase3.1 reconstruction smoke run.

## Files

- `figure_01` through `figure_05`: identical four-ID qualitative comparisons at strengths 0, 0.25, 0.5, 0.75, and 1.0. Columns are source, VAE anchor, frozen UNet, and the 64-step adapter.
- `figure_06_identity_and_coverage`: source ArcFace identity retention (mean and P10) and face-detection coverage. The horizontal reference is the mean source-to-VAE cosine.
- `figure_07_pixel_drift`: RGB MAE and RMSE relative to the source image.
- `figure_data.csv`: exact aggregate values used for Figures 6 and 7.

Figures 6 and 7 are supplied as 300 DPI PNG and vector PDF. The qualitative sheets remain lossless PNG so that labels and image evidence are preserved exactly.

## Interpretation Boundary

These are training-subset diagnostics from 32 smoke-test IDs. They support comparison of source reconstruction behavior under fixed noise, but do not establish held-out generalization, 3D control, standardization quality, or gaze disentanglement. Strength 0 is a VAE-only anchor and makes no UNet call. Strength 1 starts from a heavily noised source latent rather than pure noise.

The complete protocol and numeric audit are documented in `../PHASE31_LATENT_IMG2IMG_BASELINE_20260903.md`. Raw run outputs and model checkpoints remain excluded from Git.
