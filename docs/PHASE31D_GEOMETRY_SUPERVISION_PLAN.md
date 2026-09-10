# Phase3.1d Geometry-Supervised Overfit Plan

## Objective

Determine whether the existing frozen SD1.5/VAE adapter architecture can learn target geometry when optimization explicitly supervises geometry, rather than only reconstructing source RGB.

This is a bounded mechanism experiment, not formal Phase3 training.

## Data

- Generate canonical Phase2 Full targets for the 8,160 train IDs using the same checkpoint and OOF XGBoost provenance as Phase3.1c.
- Select 32 train-only samples from the highest geometry-delta tier, while retaining pose and expression variation.
- Require zero overlap with the 32 Phase3.1c validation IDs and all 775 fixed-test IDs.
- Build fresh generation-time source/target condition caches; rescue and gaze remain disabled.

The existing 32 validation samples remain frozen as the only model-selection audit. They must not receive optimizer updates.

## Model Strategy

Warm-start the source-reconstruction checkpoint, not the failed geometry-supervision checkpoint. Keep the VAE, SD1.5 UNet, identity estimator, and geometry estimator frozen. Train the Face Adapter first; retain the learned identity branch but freeze it during the first geometry-isolation run.

Run two bounded variants with identical data, seeds, and step budgets:

1. `geometry_loss_only`: train Face Adapter with explicit geometry outcome supervision.
2. `geometry_loss_plus_ranking`: add a paired target-versus-source geometry ranking loss.

Do not add more architecture branches until these variants show whether the present injection points can transmit a usable signal.

Do not use another sample's Phase2 target as the primary negative. Phase2 targets are deliberately near-canonical: in the 32-sample train selection, pairwise target-target pose distance has mean 0.87 degrees and pairwise expression RMSE has mean 0.0153. A shuffled target is therefore often semantically equivalent for the standardization objective.

## Training Paths

Each optimizer step contains two paths.

### Source Reconstruction Path

Feed source geometry and source identity. Retain epsilon prediction loss:

$$
\mathcal{L}_{\mathrm{src}} = \lVert \epsilon - \epsilon_\theta(z_t,t,c_{\mathrm{src}},e_{\mathrm{id}}) \rVert_2^2.
$$

This anchors reconstruction and prevents unconstrained target editing.

### Target Geometry Path

Feed target geometry with the same source identity and noisy source latent. Convert the predicted epsilon to a one-step clean-latent estimate:

$$
\hat z_0 = \frac{z_t-\sqrt{1-\bar\alpha_t}\,\hat\epsilon_\theta}{\sqrt{\bar\alpha_t}}.
$$

Decode through the frozen VAE and pass the RGB through a frozen differentiable DECA estimator. The training path must not contain FAN or any other nondifferentiable detector. It must use the fixed whole-image warp audited in Phase3.1c, followed by the estimator's differentiable encoder.

Before either bounded run, execute a gradient preflight for
`Face Adapter -> UNet -> x0 estimate -> VAE decode -> resize -> DECA encoder`.
The preflight must confirm finite gradients at every Face Adapter output scale and
peak allocated GPU memory below 7.2 GiB. A detached estimator output does not count
as a successful preflight.

Apply target geometry supervision only at a fixed low/mid-noise timestep interval
chosen on the 32 train samples. At high noise, the one-step clean estimate may not
be face-like enough for DECA to provide a meaningful training signal. The source
epsilon loss may continue to sample the original timestep distribution. Record and
freeze the geometry timestep interval before validation.

Supervise:

$$
\mathcal{L}_{\mathrm{geom}} =
\lambda_p d_{\mathrm{SO(3)}}(\hat p,p^*)+
\lambda_e \operatorname{RMSE}(\hat e,e^*)+
\lambda_l \operatorname{NME}(\hat \ell,\ell^*).
$$

Failed or nonfinite DECA estimates must remain explicit; no FAN rescue is allowed.

For the ranking variant, pair the same source latent/noise/identity with target and source geometry:

$$
\mathcal{L}_{\mathrm{rank}}=
\max\left(0,m+D(\hat x_{\mathrm{target}},y^*)-D(\hat x_{\mathrm{source}},y^*)\right),
$$

where $D$ is the unit-consistent normalized sum of pose, expression, and landmark target errors. Pose error is measured in degrees and normalized by a degree-valued constant. The margin and normalization constants must be selected on train-only diagnostics and frozen before validation.

An optional control-sensitivity audit may render synthetic counterfactual conditions with at least 10 degrees of target-pose separation. These counterfactuals are diagnostics only and must preserve source identity and use valid DECA parameter ranges. They are preferable to shuffled near-canonical targets when testing continuous geometric control.

Total loss with an absolute source-condition guard:

$$
\mathcal{L}=\mathcal{L}_{\mathrm{src}}+
\lambda_g\left[
\mathcal{L}_{\mathrm{geom}}(\hat x_{\mathrm{target}},y^*)+
\rho_s\mathcal{L}_{\mathrm{geom}}(\hat x_{\mathrm{source}},y_{\mathrm{source}})
\right]+
\lambda_r\mathcal{L}_{\mathrm{rank}}.
$$

The source and target arms share source latent, noise, timestep, and identity.
The source absolute term prevents the ranking margin from being satisfied only
by degrading the negative arm. The bounded Phase3.1f candidate is
`lambda_g=0.003`, `rho_s=0.3`, and `lambda_r=1.0`, selected by a train-only
first-order audit; it remains unverified until the bounded update is run.

## Anti-Collapse Controls

- Report zero-input residuals throughout training. If they remain comparable to target-input residuals, run a separate no-bias Face Adapter diagnostic; do not silently change the main architecture.
- Log residual-to-UNet-activation RMS ratios at all four injection scales.
- Use identity-condition dropout only as a train-only diagnostic; never change identity in the target-geometry causal arm.
- Include source, target, zero, and shuffled geometry in fixed-noise diagnostics every 16 optimizer steps, but treat shuffled as descriptive rather than a required negative unless its target parameters are demonstrably separated.
- Preserve exact checkpoint, split, code, estimator, and input hashes.

## Budget

1. Full paired-objective preflight and two-optimizer-step smoke test with backward/finite/VRAM checks.
2. One 96-step train-only run: each of 32 identities receives canonical, negative-yaw, and positive-yaw supervision exactly once.
3. Re-run the frozen-noise counterfactual direction/order audit and the canonical source/target diagnostics.
4. If target-versus-control separation, source preservation, or counterfactual tracking is absent, stop before validation.
5. Only the passing checkpoint may enter the already frozen 32-sample validation audit. No 256-step extension is authorized by this revision.

No 8,160-sample training is authorized by this plan.

## Promotion Criteria

At strength 0.25 on the frozen Phase3.1c validation set:

- Target geometry must beat source and zero geometry for pose and at least one of expression/landmark.
- Pose improvement must be at least 1 degree or 10% relative to the source-geometry error, not merely have a bootstrap interval excluding zero.
- The paired bootstrap interval for the primary pose contrast must remain below zero.
- If a counterfactual control audit is run, output pose must change in the requested direction and preserve the ordering of targets separated by at least 10 degrees.
- Single-face ArcFace cosine must not fall by more than 0.03 relative to source geometry.
- Single-face-valid coverage must not fall by more than 5 percentage points.
- Generation and DECA failures remain in the 32-sample denominator.

These are engineering promotion thresholds for the bounded experiment, not universal perceptual or identity-verification standards.

## Gaze Boundary

Head-pose control is a prerequisite for gaze disentanglement, not its substitute. Eye Adapter, head-local gaze loss, and gaze claims remain disabled until geometry control passes and the gaze coordinate convention is independently approved.
