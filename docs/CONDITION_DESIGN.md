# Downstream Condition Design

Date: 2026-09-01

## Scope

This document defines candidate conditioning signals for future downstream image-generation experiments. The current project has completed Phase2 v1 and built a guarded Phase2.1 framework, but the Phase2.1 Gate is not deployment-qualified.

## Condition Types

| Condition | Format | Recommended Use | Source | Risk |
| --- | --- | --- | --- | --- |
| Landmark map | image-like | ControlNet / spatial adapter | FAN/DECA landmarks | Fails under severe pose or missed detection |
| Depth map | image-like | ControlNet | DECA render | Carries DECA reconstruction bias |
| Normal map | image-like | ControlNet | DECA render | Sensitive to bad geometry |
| Face mask | image-like | loss mask / conditioning | parser or renderer | Occlusion and hair boundaries may be wrong |
| DECA parameter vector | vector-like | MLP adapter / cross-attention token | DECA mat | Does not directly encode image realism |
| Phase2 standardized vector | vector-like | MLP adapter / cross-attention token | Phase2 inference | Must be filtered conservatively |
| Phase2 alpha/confidence/reject | vector-like | diagnostic or adapter feature | Phase2 inference | Current Gate is diagnostic, not safety-certified |
| ArcFace identity embedding | vector-like | identity condition or evaluation | ArcFace extractor | Detector failures and privacy sensitivity |
| Camera-frame gaze vector | vector-like | measurement and coordinate conversion | L2CS | Entangles head rotation and eye-in-head gaze |
| Head-local gaze vector | vector-like | independent gaze condition | L2CS + validated head rotation | Pseudo-label and coordinate-convention errors |
| Target head rotation | vector/matrix | head-pose condition | Phase2/experiment target | Must remain separate from gaze target |

## Head Pose Versus Eye Gaze

Head pose and eye gaze are different variables.

DECA pose mainly describes global head orientation and jaw/neck-related pose terms in the reconstructed face model. L2CS gaze estimates where the eyes are looking. A face can have frontal head pose but side gaze, or side head pose with gaze toward the camera.

Phase3 therefore uses the decomposition $\mathbf{g}_{head}=R_h^\top\mathbf{g}_{cam}$, where $R_h$ maps head coordinates to camera coordinates. The condition interface keeps target head rotation and target eye-in-head gaze as separate factors.

Therefore:

- Pose standardization should be measured with DECA/L2CS head-pose diagnostics.
- Head-only interventions must preserve eye-in-head gaze.
- Gaze-only interventions must preserve head pose.
- Bidirectional leakage and identity preservation must be measured before claiming successful disentanglement.
- L2CS/DECA pseudo-label experiments may establish an engineering protocol, while stronger scientific claims require independent paired or ground-truth validation.

## Recommended First Downstream Setup

Start with an interface-only experiment:

- Input: source image plus DECA/Phase2 metadata manifest.
- Output: dry-run JSONL dataset and placeholder evaluation outputs.
- Training: none.
- Purpose: verify that Phase2 outputs can be consumed by a future image-generation model without changing Phase2 core code.

The first actual model-training attempt should begin only after deciding the filtering policy for unsafe, missing, and rescue-only samples.

## Filtering Policy

Use the following default policy:

- Include FAN-main successful samples.
- Exclude upstream DECA/FAN failures from primary training.
- Keep rescue rows in a separate audit manifest.
- Record Gate decisions, but do not trust the current Gate as the sole safety filter.
- Prefer conservative manual review for low-quality or missing-metric rows.
