# Condition Design

## Purpose

This document defines candidate conditioning inputs for downstream diffusion or
ControlNet-style models. It is a design interface for future training, not a
training implementation.

The design separates spatial geometry, vector geometry, identity, reliability,
and gaze. A critical boundary is:

```text
head pose != eye gaze
```

DECA pose describes rotation of the head. L2CS gaze estimates where the eyes
look. They may correlate in real images, but they must be represented and
evaluated as separate signals.

## Condition Inventory

| Condition | Source artifact | Expected format | Kind | Suggested model use | Risks and limitations |
|---|---|---|---|---|---|
| Landmark map | Landmark detector or rasterized DECA landmarks | Single or multi-channel image aligned to source resolution or training crop | Image-like | ControlNet spatial branch | Sparse landmarks may miss texture, eyelid state, and depth; detector failures can create brittle supervision |
| Depth map | DECA reconstruction or renderer output | 1-channel float/uint image, normalized per crop or by fixed depth range | Image-like | ControlNet spatial branch | DECA depth is model-derived, not ground truth; bad fits propagate geometry errors |
| Normal map | DECA renderer output | 3-channel normal image in camera or canonical coordinates | Image-like | ControlNet spatial branch | Normal conventions must be fixed; lighting and normal direction mistakes can mislead training |
| Face mask | Face parser, segmentation model, or projected DECA mesh mask | 1-channel binary or soft mask | Image-like | ControlNet spatial branch or loss mask | Mask errors can remove hair, ears, jaw edges, glasses, or occluders |
| DECA parameter vector | DECA `.mat` or extracted parameter manifest | Vector containing pose, shape, expression, camera, texture, or lighting subsets | Vector-like | Adapter, MLP projection, cross-attention token, or evaluation stratifier | Raw DECA parameters are not standardized targets; head pose is not eye gaze |
| Phase2 standardized pose/expression vector | Phase2 inference `.npz` or manifest columns | Vector of standardized pose/expression parameters | Vector-like | Adapter, MLP projection, cross-attention token, or target-control input | Depends on Phase2 validity; should not be trusted until checkpoint/inference quality is closed |
| Phase2 alpha/confidence/reject signals | Phase2 inference manifest | Scalar or small vector, e.g. confidence and reject score | Vector-like | Sampling weight, filtering signal, adapter side input, or evaluation-only stratifier | Reliability scores can encode dataset bias; low confidence should not silently disappear |
| ArcFace identity embedding | ArcFace embedding file or manifest path | Usually L2-normalized identity vector | Vector-like | Cross-attention token, identity adapter, identity loss, or evaluation | Can over-constrain generation; embedding quality drops on blur, occlusion, extreme pose, or failed alignment |
| L2CS gaze vector | L2CS gaze manifest or merged Phase1 manifest | Two scalars, commonly pitch and yaw | Vector-like | Future gaze adapter or evaluation-only signal | Pseudo-labels are noisy; gaze cannot be treated as DECA head pose; no ground-truth claim without labels |

## Recommended Routing

| Signal family | Preferred routing | Notes |
|---|---|---|
| Landmark/depth/normal/mask | ControlNet-like spatial branch | Keep resolution and coordinate conventions stable across splits |
| DECA vector | Adapter or cross-attention token | Start with ablations before mixing with all spatial maps |
| Phase2 vector | Adapter or cross-attention token | Use only after Phase2 outputs pass quality checks |
| Phase2 confidence/reject | Filtering, sampling weight, or evaluation stratifier | Avoid using it as a hidden shortcut without reporting stratified metrics |
| ArcFace identity | Identity adapter, cross-attention token, or identity loss | Evaluate identity preservation separately from visual realism |
| L2CS gaze | Evaluation first; future adapter only when goals are clear | Do not claim gaze disentanglement from pseudo-label consistency alone |

## Head Pose And Gaze Boundary

Head pose is the orientation of the whole head in the camera frame. It is often
represented by yaw, pitch, and roll from DECA or a pose estimator.

Eye gaze is the direction the eyes are looking. L2CS estimates gaze pitch and
yaw from image evidence. A face can have frontal head pose and side gaze, or a
turned head with eyes looking toward the camera. Therefore:

- DECA pose can support pose standardization.
- L2CS gaze can support gaze-behavior evaluation.
- DECA pose must not be used as a substitute for L2CS gaze.
- A future model may preserve, neutralize, or control gaze, but those are
  separate objectives.

## Current Scope

The current downstream preparation work may define manifests, skeleton scripts,
and evaluation placeholders. It does not implement diffusion training,
ControlNet training, identity-conditioned generation, or a completed gaze
disentanglement model.
