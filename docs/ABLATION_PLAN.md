# Ablation Plan


## 1. Purpose

This document defines future ablation experiments
for evaluating the contribution of different conditions.


## 2. Experiment Groups


| Group | Condition |
|-|-|
| A | Hard-zero baseline |
| B | Phase2 only |
| C | Image generation baseline |
| D | Landmark condition |
| E | Depth condition |
| F | Normal condition |
| G | DECA parameter condition |
| H | Phase2 parameter condition |
| I | Identity condition |
| J | Identity + gaze condition |


## 3. Evaluation


Each experiment should evaluate:


### Identity

Metric:

- ArcFace similarity


Purpose:

Measure whether identity is preserved.


### Pose

Metric:

- pose error


Purpose:

Measure whether generated faces approach canonical pose.


### Gaze

Metric:

- gaze consistency


Purpose:

Measure gaze behavior preservation or control.


## 4. Hypothesis


Expected observations:

- Geometry conditions improve pose control.
- Identity condition improves identity preservation.
- Gaze condition improves gaze controllability.


## 5. Phase2 Dependency


Experiments using standardized parameters:

- B
- H
- I
- J


depend on Phase2 outputs.