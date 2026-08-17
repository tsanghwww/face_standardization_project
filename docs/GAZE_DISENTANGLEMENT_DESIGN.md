# Gaze Disentanglement Design

## Purpose

This document prepares the conceptual design for future gaze adjustment after
Phase2. The project currently has gaze extraction and planned gaze-behavior
evaluation. It does not yet have a completed gaze disentanglement model.

## Head Pose Vs Eye Gaze

Head pose describes the rotation and translation of the whole head relative to
the camera. It is commonly represented as yaw, pitch, and roll. DECA pose
parameters are head-pose and face-geometry signals.

Eye gaze describes where the eyes are looking. A person can rotate their head
left while looking right, or keep a frontal head pose while looking down. Gaze
therefore must be modeled separately from DECA pose.

## What L2CS Provides

L2CS provides image-based gaze estimates, typically gaze pitch and gaze yaw. In
this project those values should be treated as gaze pseudo-labels unless a
ground-truth gaze dataset is introduced. They are useful for filtering,
stratifying, evaluation, and future conditioning experiments, but they are not
proof that a model has learned causal gaze disentanglement.

## Possible Gaze Goals

| Goal | Meaning | Required data | Suitable first use | Claim boundary |
|---|---|---|---|---|
| Preserve gaze | Keep generated gaze close to source gaze while standardizing head pose | Source image, source L2CS gaze, generated-output gaze estimate | Evaluation and identity-safe generation checks | Can claim measured gaze consistency only under the estimator used |
| Neutralize gaze | Move gaze toward a canonical forward-looking target | Source gaze pseudo-labels, canonical target definition, generated-output gaze estimate | Controlled experiments after pose standardization is stable | Cannot claim true neutralization without validated labels or human/benchmark review |
| Control gaze | Generate a requested target gaze independent of source head pose | Target gaze labels, enough training diversity, generated-output gaze estimate | Later model variant or synthetic-data experiment | Cannot claim disentanglement without evidence that head pose and identity are preserved while gaze changes |

## What Pseudo-labels Can Do

L2CS pseudo-labels can support:

- dataset stratification by gaze direction
- filtering extreme or unreliable gaze cases
- placeholder gaze-behavior metrics
- weak conditioning experiments
- regression checks that generated outputs do not drift unexpectedly

Pseudo-labels should be stored with provenance. Reports should say which
estimator produced them and whether they were used for filtering, conditioning,
or evaluation.

## What Cannot Be Claimed Without Ground Truth

Without ground-truth gaze labels or a validated benchmark, the project should
not claim:

- true gaze disentanglement
- causal separation of eye gaze from head pose
- accurate gaze control in physical degrees
- human-perceived eye-contact correction
- fairness or robustness across glasses, occlusion, lighting, ethnicity, or
  extreme pose

It is acceptable to claim that a skeleton measures gaze-related behavior using
a stated estimator.

## Metrics

| Metric | Use | Limitation |
|---|---|---|
| Source/generated gaze delta | Preserve-gaze experiments | Depends on estimator consistency; small delta is not proof of identity or pose quality |
| Distance to canonical gaze | Neutralize-gaze experiments | Requires a precise canonical target definition |
| Distance to requested target gaze | Control-gaze experiments | Requires target labels and generated-output gaze estimation |
| Pose/gaze correlation check | Detect leakage between head pose and eye gaze | Correlation does not prove causality |
| Identity similarity under gaze changes | Ensure gaze edits do not erase identity | ArcFace can be affected by gaze, blur, and crop quality |
| Human or benchmark validation | Sanity-check visual gaze perception | More expensive and not available in the current skeleton |

## Evaluation Boundary

Gaze evaluation should be reported separately from pose evaluation. Head-pose
improvement cannot be used as a substitute for gaze improvement. The initial
`evaluate_gaze_behavior.py` entry point should produce placeholder metrics and
make this limitation explicit until real generated outputs and validated gaze
labels are available.
