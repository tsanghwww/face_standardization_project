"""Dataset helpers for Phase2 training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .augmentation import AugmentConfig, augment_params, expression_stats
from .features import Phase2Sample, feature_vector


class Phase2Dataset(Dataset):
    def __init__(
        self,
        samples: list[Phase2Sample],
        augment: bool,
        seed: int = 2026,
        stage: int = 1,
    ):
        self.samples = samples
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        expressions = np.vstack([s.params["expression"] for s in samples]).astype(np.float32)
        self.stats = expression_stats(expressions)
        self.augment_cfg = AugmentConfig(stage=stage)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        params = {k: v.copy() for k, v in sample.params.items()}
        metrics = dict(sample.metrics)
        sample_weight = float(metrics.get("sample_weight", 1.0))
        if self.augment and len(self.samples) > 1:
            other = self.samples[int(self.rng.integers(0, len(self.samples)))]
            params = augment_params(params, self.stats, self.rng, self.augment_cfg, other.params["expression"])
            exp_norm = float(np.linalg.norm(params["expression"]) / np.sqrt(params["expression"].size))
            head_norm = float(np.linalg.norm(params["pose"][:3]))
            jaw_norm = float(np.linalg.norm(params["pose"][3:]))
            metrics["exp_norm"] = exp_norm
            metrics["head_pose_norm"] = head_norm
            metrics["jaw_pose_norm"] = jaw_norm
            metrics["quality_score"] = float(
                max(0.0, min(1.0, metrics.get("quality_score", 0.5) - 0.10 * head_norm - 0.06 * jaw_norm))
            )
        features = feature_vector(params, metrics)
        return {
            "image_id": sample.image_id,
            "mat_path": str(sample.mat_path),
            "features": torch.from_numpy(features),
            "expression": torch.from_numpy(params["expression"]),
            "pose": torch.from_numpy(params["pose"]),
            "quality": torch.tensor([metrics.get("quality_score", 0.5)], dtype=torch.float32),
            "reject_target": torch.tensor([1.0 - metrics.get("quality_score", 0.5)], dtype=torch.float32),
            "sample_weight": torch.tensor([sample_weight], dtype=torch.float32),
        }


def save_normalizer(path: Path, feature_mean: np.ndarray, feature_std: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, feature_mean=feature_mean.astype(np.float32), feature_std=feature_std.astype(np.float32))


def load_normalizer(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["feature_mean"].astype(np.float32), data["feature_std"].astype(np.float32)
