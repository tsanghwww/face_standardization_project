# Ablation Plan

## Purpose

This document defines future downstream experiment groups. These groups are
for planning only; no diffusion, ControlNet, or identity-conditioned model
training is started by this task.

## Experiment Groups

| Group | Description | Input | Output | Required artifacts | Metrics | Hypothesis | Depends on Phase2 completion |
|---|---|---|---|---|---|---|---|
| A | Hard-zero parameter baseline | Source image plus zeroed pose/expression control | Rendered or generated standardized face | Source image, DECA metadata for evaluation | Pose error, identity similarity, quality filters | A trivial control exposes how much improvement comes from learned conditioning | No |
| B | Phase2-only parameter standardization | Source image plus Phase2 standardized pose/expression vector | Rendered or generated standardized face | Source image, Phase2 `.npz`, Phase2 confidence/reject scores | Pose error, reject-rate stratification, identity similarity | Phase2 parameters alone should move outputs toward canonical pose/expression | Yes |
| C | Image U-Net baseline | Source image only | Generated standardized face | Source image, train/val/test manifest | Identity similarity, pose error, visual quality | An image-only model provides a lower bound for condition usefulness | No |
| D | ControlNet with landmarks only | Source image plus landmark map | Generated standardized face | Source image, landmark map | Pose error, landmark consistency, identity similarity | Sparse geometry improves alignment over image-only conditioning | No |
| E | ControlNet with depth only | Source image plus depth map | Generated standardized face | Source image, DECA depth map | Pose error, depth consistency, identity similarity | Dense depth provides stronger 3D shape control than landmarks alone | No |
| F | ControlNet with normals only | Source image plus normal map | Generated standardized face | Source image, DECA normal map | Pose error, normal consistency, identity similarity | Surface orientation improves geometry and lighting robustness | No |
| G | ControlNet with DECA parameter vector | Source image plus DECA vector | Generated standardized face | Source image, DECA `.mat` or parameter manifest | Pose error, expression consistency, identity similarity | Raw 3DMM parameters add compact geometry control beyond image-only input | No |
| H | ControlNet with Phase2 standardized params | Source image plus Phase2 standardized vector | Generated standardized face | Source image, Phase2 `.npz`, confidence/reject scores | Pose error, confidence-stratified pose error, identity similarity | Standardized Phase2 parameters outperform raw DECA for canonicalization | Yes |
| I | Full model with identity condition | Source image plus geometry conditions plus ArcFace embedding | Generated standardized face | Source image, chosen geometry condition, ArcFace embedding | ArcFace similarity, pose error, visual quality | Explicit identity conditioning preserves identity better than geometry-only models | Depends on chosen geometry; yes if Phase2 is included |
| J | Full model with identity + gaze condition | Source image plus geometry, ArcFace, and L2CS gaze vector | Generated standardized face with specified gaze behavior | Source image, chosen geometry condition, ArcFace embedding, L2CS gaze labels | Gaze-behavior delta, ArcFace similarity, pose error | Gaze signal improves preserve/neutralize/control behavior without reducing identity | Depends on chosen geometry; yes if Phase2 is included |

## Metric Families

| Metric family | Intended use | Notes |
|---|---|---|
| Identity preservation | Compare source and generated/rendered identity using ArcFace similarity | Identity should be reported separately from pose and gaze |
| Pose standardization | Measure distance from canonical head pose before and after generation | Head pose metrics should not be reused as gaze metrics |
| Gaze behavior | Measure gaze-related consistency or control using L2CS or a validated estimator | This does not by itself prove true gaze disentanglement |
| Artifact quality | Track missing files, failed renders, reject scores, and visual-quality filters | Useful for debugging upstream condition quality |

## Phase2 Dependency

Groups B and H directly depend on Phase2 being complete because their primary
input is the Phase2 standardized vector. Groups I and J depend on Phase2 only
when their selected geometry branch includes Phase2 outputs. Groups A, C, D, E,
F, and G can be prepared before Phase2 is closed.

## Reporting Rules

Each future run should report:

- group id and condition set
- training/evaluation split counts
- missing artifact counts
- Phase2 dependency status
- identity, pose, and gaze metrics where applicable
- whether results are based on ground truth, model-derived labels, or
  pseudo-labels
