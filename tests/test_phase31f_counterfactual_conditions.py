"""CPU protocol checks for train-only Phase3.1f pose counterfactuals."""

import numpy as np

from scripts.build_phase31f_counterfactual_conditions import (
    compose_camera_yaw, pose_distance_degrees,
)


def main() -> None:
    base = np.zeros(6, dtype=np.float32)
    negative = compose_camera_yaw(base, -10.0)
    positive = compose_camera_yaw(base, 10.0)
    assert abs(pose_distance_degrees(negative, positive) - 20.0) < 1e-4
    assert negative[1] < 0 < positive[1]

    nonzero = np.array([0.1, -0.05, 0.03, 0.2, 0.0, 0.0], dtype=np.float32)
    left = compose_camera_yaw(nonzero, -10.0)
    right = compose_camera_yaw(nonzero, 10.0)
    assert pose_distance_degrees(left, right) >= 19.999
    assert np.array_equal(left[3:], nonzero[3:])
    assert np.array_equal(right[3:], nonzero[3:])
    print("Phase3.1f counterfactual pose composition and separation passed")


if __name__ == "__main__":
    main()
