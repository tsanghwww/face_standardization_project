"""Pure numerical helpers for counterfactual pose-control auditing."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation


def relative_rotvec_deg(left_pose, right_pose) -> np.ndarray:
    """Relative rotation taking ``left_pose`` to ``right_pose``, as degrees."""
    left = Rotation.from_rotvec(np.asarray(left_pose, dtype=np.float64).reshape(-1)[:3])
    right = Rotation.from_rotvec(np.asarray(right_pose, dtype=np.float64).reshape(-1)[:3])
    return np.degrees((right * left.inv()).as_rotvec())


def pose_distance_deg(left_pose, right_pose) -> float:
    return float(np.linalg.norm(relative_rotvec_deg(left_pose, right_pose)))


def counterfactual_tracking_metrics(
    output_negative,
    output_positive,
    target_negative,
    target_positive,
) -> dict:
    """Coordinate-invariant direction and own-target ordering metrics."""
    target_delta = relative_rotvec_deg(target_negative, target_positive)
    target_separation = float(np.linalg.norm(target_delta))
    if not math.isfinite(target_separation) or target_separation < 10.0:
        raise ValueError("Counterfactual target separation must be at least 10 degrees")
    axis = target_delta / target_separation
    output_delta = relative_rotvec_deg(output_negative, output_positive)
    projected = float(np.dot(output_delta, axis))
    output_separation = float(np.linalg.norm(output_delta))
    own_error = pose_distance_deg(output_negative, target_negative) + pose_distance_deg(output_positive, target_positive)
    crossed_error = pose_distance_deg(output_negative, target_positive) + pose_distance_deg(output_positive, target_negative)
    return {
        "target_separation_deg": target_separation,
        "output_separation_deg": output_separation,
        "projected_output_change_deg": projected,
        "direction_correct": projected > 0.0,
        "own_target_error_sum_deg": own_error,
        "crossed_target_error_sum_deg": crossed_error,
        "ordering_margin_deg": crossed_error - own_error,
        "ordering_correct": own_error < crossed_error,
    }


def summarize_counterfactual_rows(rows: list[dict], expected_ids: list[str], timesteps: list[int]) -> dict:
    expected = {(image_id, timestep) for image_id in expected_ids for timestep in timesteps}
    observed = {(row["image_id"], row["timestep"]) for row in rows}
    if len(observed) != len(rows) or observed != expected:
        raise ValueError("Counterfactual audit rows are duplicate, missing, or unexpected")

    def group(values: list[dict]) -> dict:
        valid = [row for row in values if row["status"] == "success"]
        projections = np.asarray([row["projected_output_change_deg"] for row in valid], dtype=np.float64)
        separations = np.asarray([row["output_separation_deg"] for row in valid], dtype=np.float64)
        margins = np.asarray([row["ordering_margin_deg"] for row in valid], dtype=np.float64)
        return {
            "n_total": len(values),
            "n_success": len(valid),
            "n_failed": len(values) - len(valid),
            "direction_correct": sum(bool(row["direction_correct"]) for row in valid),
            "ordering_correct": sum(bool(row["ordering_correct"]) for row in valid),
            "both_correct": sum(bool(row["direction_correct"] and row["ordering_correct"]) for row in valid),
            "projected_output_change_deg_mean": float(projections.mean()) if len(valid) else None,
            "projected_output_change_deg_median": float(np.median(projections)) if len(valid) else None,
            "output_separation_deg_mean": float(separations.mean()) if len(valid) else None,
            "ordering_margin_deg_mean": float(margins.mean()) if len(valid) else None,
        }

    overall = group(rows)
    by_timestep = {str(timestep): group([row for row in rows if row["timestep"] == timestep]) for timestep in timesteps}
    overall["candidate_gate"] = {
        "definition": "diagnostic only; fixed before post-training audit",
        "requires_complete_denominator": overall["n_success"] == overall["n_total"],
        "requires_both_correct_at_least_75_percent": overall["both_correct"] >= math.ceil(0.75 * overall["n_total"]),
        "requires_median_projected_change_at_least_0_5_deg": (
            overall["projected_output_change_deg_median"] is not None
            and overall["projected_output_change_deg_median"] >= 0.5
        ),
    }
    overall["candidate_gate"]["passed"] = all(
        value for key, value in overall["candidate_gate"].items() if key.startswith("requires_")
    )
    return {"overall": overall, "by_timestep": by_timestep}
