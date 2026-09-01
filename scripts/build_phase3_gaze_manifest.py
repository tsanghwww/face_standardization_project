#!/usr/bin/env python3
"""Build an auditable DECA/L2CS gaze-coordinate manifest for Phase3.

The default status is deliberately ``candidate_unvalidated``. A project
reviewer must approve a named convention after visual inspection before these
head-local gaze values can be used as training labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from eval.gaze_geometry import (
    angular_error_deg,
    axis_angle_to_matrix,
    camera_to_head_gaze,
    head_to_camera_gaze,
    normalize_vector,
)


FIELDS = [
    "image_id", "split", "status", "failure_reason", "deca_mat", "pose_x", "pose_y", "pose_z",
    "gaze_camera_x", "gaze_camera_y", "gaze_camera_z", "gaze_head_direct_x", "gaze_head_direct_y",
    "gaze_head_direct_z", "gaze_head_inverse_x", "gaze_head_inverse_y", "gaze_head_inverse_z",
    "direct_roundtrip_error_deg", "inverse_roundtrip_error_deg", "rotation_6d_0", "rotation_6d_1",
    "rotation_6d_2", "rotation_6d_3", "rotation_6d_4", "rotation_6d_5", "coordinate_convention",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_split_dir(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    split_map: dict[str, str] = {}
    for split in ("train", "validation", "fixed_test_base", "fixed_test_external"):
        ids_path = path / f"{split}_ids.txt"
        if ids_path.exists():
            for image_id in ids_path.read_text(encoding="utf-8-sig").splitlines():
                if image_id.strip():
                    split_map[image_id.strip()] = split
    return split_map


def resolve(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def vector(row: dict[str, str], names: tuple[str, str, str]) -> tuple[float, float, float]:
    values = tuple(float(row[name]) for name in names)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite_gaze")
    return normalize_vector(values)


def transpose(matrix: tuple[tuple[float, float, float], ...]) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def compute_candidates(
    pose: np.ndarray, gaze_camera: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float], float, float, tuple[float, ...]]:
    rotation = axis_angle_to_matrix(pose[:3])
    direct = camera_to_head_gaze(gaze_camera, rotation)
    inverse_rotation = transpose(rotation)
    inverse = camera_to_head_gaze(gaze_camera, inverse_rotation)
    direct_error = angular_error_deg(head_to_camera_gaze(direct, rotation), gaze_camera)
    inverse_error = angular_error_deg(head_to_camera_gaze(inverse, inverse_rotation), gaze_camera)
    rotation_6d = tuple(rotation[row][column] for column in range(2) for row in range(3))
    return direct, inverse, direct_error, inverse_error, rotation_6d


def build_row(base: dict[str, str], root: Path, split_map: dict[str, str]) -> dict[str, str]:
    image_id = base.get("image_id", "").strip()
    out = {field: "" for field in FIELDS}
    out.update({"image_id": image_id, "split": split_map.get(image_id, "unassigned"), "status": "failed"})
    try:
        from scipy.io import loadmat

        mat_path = resolve(base["deca_mat_path"], root)
        data = loadmat(mat_path)
        pose = np.asarray(data["pose"], dtype=np.float64).reshape(-1)
        if pose.size < 3 or not np.isfinite(pose[:3]).all():
            raise ValueError("invalid_deca_pose")
        gaze_camera = vector(base, ("gaze_x", "gaze_y", "gaze_z"))
        direct, inverse, direct_error, inverse_error, rotation_6d = compute_candidates(pose, gaze_camera)
        values = {
            "deca_mat": str(mat_path),
            **{f"pose_{axis}": f"{pose[index]:.9g}" for index, axis in enumerate("xyz")},
            **{f"gaze_camera_{axis}": f"{gaze_camera[index]:.9g}" for index, axis in enumerate("xyz")},
            **{f"gaze_head_direct_{axis}": f"{direct[index]:.9g}" for index, axis in enumerate("xyz")},
            **{f"gaze_head_inverse_{axis}": f"{inverse[index]:.9g}" for index, axis in enumerate("xyz")},
            "direct_roundtrip_error_deg": f"{direct_error:.9g}",
            "inverse_roundtrip_error_deg": f"{inverse_error:.9g}",
            "coordinate_convention": "candidate_unvalidated",
            "status": "candidate_unvalidated",
        }
        values.update({f"rotation_6d_{index}": f"{value:.9g}" for index, value in enumerate(rotation_6d)})
        out.update(values)
    except Exception as exc:  # failures stay in the denominator
        out["failure_reason"] = f"{type(exc).__name__}:{exc}"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-manifest", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--split-registry-dir", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = read_csv(args.phase1_manifest)
    if args.limit:
        rows = rows[: args.limit]
    split_map = read_split_dir(args.split_registry_dir)
    output = [build_row(row, args.project_root, split_map) for row in rows]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "phase3_gaze_coordinate_candidates.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output)
    successful = [row for row in output if row["status"] == "candidate_unvalidated"]
    summary = {
        "n_total": len(output),
        "n_candidates": len(successful),
        "n_failed": len(output) - len(successful),
        "coordinate_status": "candidate_unvalidated",
        "training_use_permitted": False,
        "required_next_check": "manual direction audit on pose-stratified samples",
        "manifest": str(manifest),
    }
    (args.out_dir / "gaze_coordinate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
