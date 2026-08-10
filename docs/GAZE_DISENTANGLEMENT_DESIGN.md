# Gaze Disentanglement Design


## 1. Motivation


Face standardization requires controlling
head pose and eye gaze independently.


However:

head rotation and eye gaze are different factors.


---

# 2. Head Pose vs Eye Gaze


## Head Pose

Describes:

- face rotation
- yaw
- pitch
- roll


Possible source:

- DECA parameters


## Eye Gaze

Describes:

- eye looking direction


Possible source:

- L2CS gaze estimation


They should be modeled separately.


---

# 3. Current Capability


Current system supports:

- gaze extraction using L2CS


Current system does NOT provide:

- gaze disentanglement model
- gaze generation control
- gaze ground truth supervision


---

# 4. Future Design


Possible future pipeline:


Input:

- source image
- identity condition
- target pose
- target gaze


Model:

- diffusion model
- gaze condition module


Output:

- standardized face
- controlled gaze


---

# 5. Required Data


Possible requirements:

- gaze estimation labels
- multi-view face data
- synthetic gaze labels
- L2CS pseudo labels


---

# 6. Evaluation


Possible metrics:

- gaze direction error
- gaze consistency
- identity preservation


Gaze evaluation should be performed separately
from head pose evaluation.