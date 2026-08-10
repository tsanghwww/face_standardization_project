# Condition Design


## 1. Overview

The downstream generation model requires multiple conditioning
signals to control identity, geometry, pose and gaze.


The condition design separates:

- spatial conditions
- vector conditions
- identity conditions
- gaze conditions


---

# 2. Spatial Conditions


| Condition | Source | Type | Purpose |
|-|-|-|-|
| Landmark map | Face landmark detector | Image | Preserve facial structure |
| Depth map | DECA | Image | Control 3D geometry |
| Normal map | DECA | Image | Control surface orientation |
| Face mask | Segmentation model | Image | Preserve face region |


Spatial conditions can be used by ControlNet-like modules.


---

# 3. Vector Conditions


| Condition | Source | Type | Purpose |
|-|-|-|-|
| DECA parameters | DECA | Vector | 3D face representation |
| Phase2 parameters | Phase2 | Vector | Standardized face representation |
| Phase2 confidence | Phase2 | Scalar | Reliability estimation |


Vector conditions can be injected through:

- adapter modules
- cross attention
- feature modulation


---

# 4. Identity Condition


Identity preservation is important because
face standardization should change pose/expression
without changing identity.


Possible identity condition:

- ArcFace embedding


Usage:

Guide generation to preserve the same person.


---

# 5. Gaze Condition


Gaze is treated separately from head pose.


Head pose:

- describes the rotation of the whole face


Eye gaze:

- describes eye looking direction


DECA mainly provides head geometry.

L2CS provides gaze estimation.


Therefore:

head pose and gaze should not be merged into one condition.


---

# 6. Condition Combination Strategy


A possible future design:


Image condition:

- landmark
- depth
- normal


Vector condition:

- Phase2 parameter
- identity embedding
- gaze vector


These conditions can be combined through
ControlNet and cross-attention mechanisms.


---

# 7. Current Limitation


Current implementation provides:

- DECA extraction
- Phase2 outputs
- gaze extraction


It does not yet implement:

- diffusion training
- ControlNet training
- gaze disentanglement model
