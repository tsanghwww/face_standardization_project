# Gaze Disentanglement Design

Date: 2026-08-31

## Question

The project currently measures gaze-related behavior, but it has not achieved gaze disentanglement.

Gaze disentanglement would require separating at least three factors:

- Identity.
- Head pose.
- Eye gaze.

The current Phase2 system mainly standardizes DECA expression and pose parameters. It does not directly control the eyeball/gaze state with ground-truth gaze supervision.

## What L2CS Provides

L2CS provides pseudo-label estimates for gaze pitch and yaw. These are useful for diagnostics and weak supervision, but they are not equivalent to controlled ground-truth gaze labels.

Use L2CS for:

- Measuring whether outputs show large gaze changes.
- Stratifying samples with abnormal gaze behavior.
- Building future weak labels.

Do not use L2CS alone to claim:

- True gaze disentanglement.
- Accurate gaze control.
- Ground-truth eye-direction preservation.

## Possible Goals

| Goal | Meaning | Feasible Now |
| --- | --- | --- |
| Preserve gaze | Keep gaze close to the source | Partially measurable |
| Neutralize gaze | Move gaze toward frontal/canonical | Not validated |
| Control gaze | Generate a chosen gaze direction | Not feasible without stronger labels |

## Recommended Thesis Wording

Use:

> The project evaluates gaze-related behavior using L2CS pseudo-labels, but does not claim completed gaze disentanglement.

Avoid:

> The model disentangles gaze from head pose.

## Future Data Needs

To claim stronger gaze control, future work needs:

- Ground-truth gaze datasets or controlled gaze annotations.
- Separate supervision for head pose and eye gaze.
- Evaluation across same-head-pose/different-gaze and different-head-pose/same-gaze pairs.
- Human or high-confidence perceptual checks for eye direction.

## Metrics

- Mean absolute gaze pitch/yaw change.
- Gaze angular error if ground truth is available.
- Identity cosine before and after gaze control.
- Pose delta separated from gaze delta.
- Failure and coverage rate by gaze bin.
