"""PyTorch dataset for outcome supervision.

Reads an outcome manifest CSV (built by build_outcome_supervision_manifest.py)
and yields (features, targets) for the multi-head surrogate.  Missing outcome
values are represented by a per-head *mask*, never silently filled with 0:
a head's target mask is 0.0 when that sample's outcome is missing, so the loss
can skip it and coverage can be reported.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Input features for the surrogate: source quality + alphas + standardized params.
FEATURE_COLUMNS = [
    "quality_score",
    "xgb_quality_score",
    "landmark_score",
    "landmark_out_ratio",
    "arcface_status",
    "arcface_score",
    "alpha_expression",
    "alpha_head_pose",
    "alpha_jaw_pose",
    "standardized_exp_norm",
    "standardized_head_pose_norm",
    "standardized_jaw_pose_norm",
]

# Outcome targets (regression heads) + classification head.
REGRESSION_TARGETS = {
    "identity": "identity_delta_vs_hard_zero",
    "pose": "pose_improvement_vs_original",
    "gaze": "l2cs_gaze_delta_vs_original_deg",
}
RENDER_FAILURE_TARGET = "render_full_status"  # "success" -> 0, else 1


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if np.isfinite(f) else None


class OutcomeDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ):
        self.rows = rows
        self.features = np.zeros((len(rows), len(FEATURE_COLUMNS)), dtype=np.float32)
        self.identity = np.zeros(len(rows), dtype=np.float32)
        self.pose = np.zeros(len(rows), dtype=np.float32)
        self.gaze = np.zeros(len(rows), dtype=np.float32)
        self.render_failure = np.zeros(len(rows), dtype=np.float32)
        self.mask_identity = np.zeros(len(rows), dtype=np.float32)
        self.mask_pose = np.zeros(len(rows), dtype=np.float32)
        self.mask_gaze = np.zeros(len(rows), dtype=np.float32)
        self.mask_render = np.zeros(len(rows), dtype=np.float32)
        self.image_ids = [r.get("image_id", "") for r in rows]

        for i, r in enumerate(rows):
            for j, col in enumerate(FEATURE_COLUMNS):
                v = _number(r.get(col, ""))
                self.features[i, j] = v if v is not None else 0.0
            for head, col in REGRESSION_TARGETS.items():
                v = _number(r.get(col, ""))
                arr = getattr(self, head)
                mask = getattr(self, f"mask_{head}")
                if v is not None:
                    arr[i] = v
                    mask[i] = 1.0
            status = str(r.get(RENDER_FAILURE_TARGET, ""))
            if status == "":
                self.mask_render[i] = 0.0
            else:
                self.mask_render[i] = 1.0
                self.render_failure[i] = 0.0 if status == "success" else 1.0
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("feature_mean and feature_std must be provided together")
        if feature_mean is not None:
            self.features = ((self.features - feature_mean) / feature_std).astype(np.float32)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image_id": self.image_ids[index],
            "features": torch.from_numpy(self.features[index]),
            "identity": torch.tensor(self.identity[index]),
            "pose": torch.tensor(self.pose[index]),
            "gaze": torch.tensor(self.gaze[index]),
            "render_failure": torch.tensor(self.render_failure[index]),
            "mask_identity": torch.tensor(self.mask_identity[index]),
            "mask_pose": torch.tensor(self.mask_pose[index]),
            "mask_gaze": torch.tensor(self.mask_gaze[index]),
            "mask_render": torch.tensor(self.mask_render[index]),
        }


def read_outcome_rows(path: Path) -> list[dict]:
    import csv

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def fit_feature_normalizer(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Fit surrogate feature statistics on the training rows only."""
    raw = OutcomeDataset(rows).features.astype(np.float64)
    mean = raw.mean(axis=0)
    std = raw.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)
