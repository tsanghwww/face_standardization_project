"""Feature extraction and quality heuristics for Phase2 standardization."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DECA_DIMS = {
    "shape": 100,
    "expression": 50,
    "pose": 6,
    "camera": 3,
    "light": 27,
    "detail": 128,
}


@dataclass(frozen=True)
class Phase2Sample:
    image_id: str
    mat_path: Path
    params: dict[str, np.ndarray]
    metrics: dict[str, float]


def find_deca_mat_files(results_dir: Path) -> list[Path]:
    files = list(results_dir.glob("*/*.mat")) + list(results_dir.glob("*/*.npz"))
    return sorted(files, key=lambda p: (p.parent.name, p.name))


def _load_mat(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        return {k: data[k] for k in data.files}
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover - depends on runtime
        raise RuntimeError("Reading .mat files requires scipy. Install scipy or use .npz inputs.") from exc
    return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}


def _vector(data: dict[str, Any], keys: tuple[str, ...], dim: int) -> tuple[np.ndarray, bool]:
    for key in keys:
        if key in data:
            arr = np.asarray(data[key], dtype=np.float32).reshape(-1)
            if arr.size >= dim:
                return arr[:dim].copy(), True
            padded = np.zeros(dim, dtype=np.float32)
            padded[: arr.size] = arr
            return padded, True
    return np.zeros(dim, dtype=np.float32), False


def load_deca_params(mat_path: Path) -> tuple[dict[str, np.ndarray], dict[str, bool]]:
    data = _load_mat(mat_path)
    params = {
        "shape": _vector(data, ("shape",), DECA_DIMS["shape"]),
        "expression": _vector(data, ("expression", "exp"), DECA_DIMS["expression"]),
        "pose": _vector(data, ("pose",), DECA_DIMS["pose"]),
        "camera": _vector(data, ("camera", "cam"), DECA_DIMS["camera"]),
        "light": _vector(data, ("light", "illumination"), DECA_DIMS["light"]),
        "detail": _vector(data, ("detail",), DECA_DIMS["detail"]),
    }
    vectors = {key: item[0] for key, item in params.items()}
    present = {key: item[1] for key, item in params.items()}
    return vectors, present


def read_kpt_quality(mat_path: Path, image_size: float = 224.0) -> dict[str, float]:
    kpt_path = mat_path.with_name(f"{mat_path.stem}_kpt2d.txt")
    if not kpt_path.exists():
        return {
            "landmark_available": 0.0,
            "landmark_out_ratio": 1.0,
            "landmark_bbox_area": 0.0,
            "landmark_center_dist": image_size,
            "landmark_score": 0.0,
        }
    try:
        kpt = np.loadtxt(kpt_path, dtype=np.float32)
    except Exception:
        return {
            "landmark_available": 0.0,
            "landmark_out_ratio": 1.0,
            "landmark_bbox_area": 0.0,
            "landmark_center_dist": image_size,
            "landmark_score": 0.0,
        }
    if kpt.ndim != 2 or kpt.shape[0] < 5 or kpt.shape[1] < 2 or not np.isfinite(kpt).all():
        return {
            "landmark_available": 0.0,
            "landmark_out_ratio": 1.0,
            "landmark_bbox_area": 0.0,
            "landmark_center_dist": image_size,
            "landmark_score": 0.0,
        }
    xy = kpt[:, :2]
    # DECA outputs landmarks in orthographic projection space (roughly [-1, 1]).
    # Convert to pixel coordinates for quality heuristics.
    # (This mirrors the commented-out denormalization in DECA decalib/deca.py L182.)
    xy = xy * (image_size / 2.0) + (image_size / 2.0)
    out = (xy[:, 0] < 0) | (xy[:, 0] > image_size) | (xy[:, 1] < 0) | (xy[:, 1] > image_size)
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    wh = np.maximum(maxs - mins, 0)
    area = float((wh[0] * wh[1]) / (image_size * image_size))
    center = (mins + maxs) * 0.5
    center_dist = float(np.linalg.norm(center - np.array([image_size * 0.5, image_size * 0.5])) / image_size)
    out_ratio = float(out.mean())
    area_score = max(0.0, min(1.0, (area - 0.05) / 0.25))
    center_score = max(0.0, 1.0 - center_dist / 0.45)
    out_score = max(0.0, 1.0 - out_ratio / 0.30)
    landmark_score = float(0.45 * area_score + 0.35 * center_score + 0.20 * out_score)
    return {
        "landmark_available": 1.0,
        "landmark_out_ratio": out_ratio,
        "landmark_bbox_area": area,
        "landmark_center_dist": center_dist,
        "landmark_score": landmark_score,
    }


def read_arcface_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["image_id"]: row for row in csv.DictReader(f) if row.get("image_id")}


def _float(row: dict[str, str], key: str, default: float) -> float:
    try:
        value = float(row.get(key, ""))
    except ValueError:
        return default
    return value if math.isfinite(value) else default


def compute_metrics(
    params: dict[str, np.ndarray],
    mat_path: Path,
    present: dict[str, bool],
    arcface_row: dict[str, str] | None = None,
) -> dict[str, float]:
    exp = params["expression"]
    pose = params["pose"]
    cam = params["camera"]
    light = params["light"]
    kpt = read_kpt_quality(mat_path)
    exp_norm = float(np.linalg.norm(exp) / math.sqrt(max(exp.size, 1)))
    head_pose_norm = float(np.linalg.norm(pose[:3]))
    jaw_pose_norm = float(np.linalg.norm(pose[3:]))
    camera_scale = float(cam[0]) if cam.size else 0.0
    light_norm = float(np.linalg.norm(light) / math.sqrt(max(light.size, 1)))

    arcface_status = 0.0
    arcface_score = 0.0
    arcface_train_flag = 0.0
    if arcface_row:
        arcface_status = 1.0 if arcface_row.get("arcface_status") == "success" else 0.0
        arcface_score = _float(arcface_row, "detector_score", 0.0)
        arcface_train_flag = 1.0 if arcface_row.get("use_for_train", "").lower() == "true" else 0.0

    pose_score = max(0.0, 1.0 - head_pose_norm / 0.9)
    exp_score = max(0.0, 1.0 - exp_norm / 0.22)
    jaw_score = max(0.0, 1.0 - jaw_pose_norm / 0.55)
    arc_score = 0.75 + 0.25 * arcface_score if arcface_status else 0.65
    completeness = float(sum(present.values()) / len(present))
    quality = (
        0.32 * kpt["landmark_score"]
        + 0.22 * pose_score
        + 0.16 * exp_score
        + 0.10 * jaw_score
        + 0.12 * arc_score
        + 0.08 * completeness
    )
    quality = float(max(0.0, min(1.0, quality)))
    if quality >= 0.72:
        quality_class = 2.0
    elif quality >= 0.45:
        quality_class = 1.0
    else:
        quality_class = 0.0

    return {
        "exp_norm": exp_norm,
        "head_pose_norm": head_pose_norm,
        "jaw_pose_norm": jaw_pose_norm,
        "camera_scale": camera_scale,
        "light_norm": light_norm,
        "arcface_status": arcface_status,
        "arcface_score": arcface_score,
        "arcface_train_flag": arcface_train_flag,
        "quality_score": quality,
        "quality_class": quality_class,
        **kpt,
    }


def feature_vector(params: dict[str, np.ndarray], metrics: dict[str, float]) -> np.ndarray:
    metric_keys = [
        "exp_norm",
        "head_pose_norm",
        "jaw_pose_norm",
        "camera_scale",
        "light_norm",
        "landmark_score",
        "landmark_out_ratio",
        "landmark_bbox_area",
        "landmark_center_dist",
        "arcface_status",
        "arcface_score",
        "arcface_train_flag",
        "quality_score",
    ]
    parts = [
        params["expression"],
        params["pose"],
        params["camera"],
        params["light"],
        np.asarray([metrics.get(k, 0.0) for k in metric_keys], dtype=np.float32),
    ]
    return np.concatenate(parts).astype(np.float32)


def sample_from_mat(mat_path: Path, arcface_row: dict[str, str] | None = None) -> Phase2Sample:
    params, present = load_deca_params(mat_path)
    metrics = compute_metrics(params, mat_path, present, arcface_row)
    return Phase2Sample(image_id=mat_path.stem, mat_path=mat_path, params=params, metrics=metrics)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
