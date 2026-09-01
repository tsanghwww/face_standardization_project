"""Geometry helpers for separating camera-frame gaze from head-local gaze.

The rotation matrix convention is explicit: ``R_head_to_camera`` maps vectors
from the head coordinate system into the camera coordinate system.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def normalize_vector(values: Sequence[float]) -> Vector3:
    if len(values) != 3:
        raise ValueError("gaze vector must contain exactly three values")
    vector = tuple(float(value) for value in values)
    norm = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("gaze vector must be finite and non-zero")
    return tuple(value / norm for value in vector)  # type: ignore[return-value]


def l2cs_angles_to_camera_vector(pitch: float, yaw: float) -> Vector3:
    """Convert L2CS pitch/yaw radians using the project's historical convention."""
    return normalize_vector(
        [
            -math.sin(yaw) * math.cos(pitch),
            -math.sin(pitch),
            -math.cos(yaw) * math.cos(pitch),
        ]
    )


def axis_angle_to_matrix(axis_angle: Sequence[float]) -> Matrix3:
    """Convert a 3D axis-angle vector to a rotation matrix with Rodrigues' formula."""
    if len(axis_angle) != 3:
        raise ValueError("axis-angle vector must contain exactly three values")
    vector = tuple(float(value) for value in axis_angle)
    angle = math.sqrt(sum(value * value for value in vector))
    if not math.isfinite(angle):
        raise ValueError("axis-angle vector must be finite")
    if angle < 1e-12:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    axis = tuple(value / angle for value in vector)
    x, y, z = axis
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus_cosine = 1.0 - cosine
    return (
        (cosine + x * x * one_minus_cosine, x * y * one_minus_cosine - z * sine, x * z * one_minus_cosine + y * sine),
        (y * x * one_minus_cosine + z * sine, cosine + y * y * one_minus_cosine, y * z * one_minus_cosine - x * sine),
        (z * x * one_minus_cosine - y * sine, z * y * one_minus_cosine + x * sine, cosine + z * z * one_minus_cosine),
    )


def matrix_vector_product(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> Vector3:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise ValueError("rotation matrix must be 3x3")
    normalized = normalize_vector(vector)
    return normalize_vector([sum(float(matrix[row][column]) * normalized[column] for column in range(3)) for row in range(3)])


def camera_to_head_gaze(gaze_camera: Sequence[float], rotation_head_to_camera: Sequence[Sequence[float]]) -> Vector3:
    transpose = tuple(tuple(float(rotation_head_to_camera[column][row]) for column in range(3)) for row in range(3))
    return matrix_vector_product(transpose, gaze_camera)


def head_to_camera_gaze(gaze_head: Sequence[float], rotation_head_to_camera: Sequence[Sequence[float]]) -> Vector3:
    return matrix_vector_product(rotation_head_to_camera, gaze_head)


def angular_error_deg(first: Sequence[float], second: Sequence[float]) -> float:
    first_normalized = normalize_vector(first)
    second_normalized = normalize_vector(second)
    cosine = max(-1.0, min(1.0, sum(a * b for a, b in zip(first_normalized, second_normalized))))
    return math.degrees(math.acos(cosine))
