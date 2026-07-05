"""Latent-space augmentation for DECA parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    stage: int = 1
    expression_strength: float = 0.5
    expression_mixup_prob: float = 0.35
    light_strength: float = 0.05
    camera_strength: float = 0.015


POSE_LIMITS_DEG = {
    1: (15.0, 10.0, 8.0),
    2: (30.0, 20.0, 15.0),
    3: (45.0, 25.0, 20.0),
}


def expression_stats(expressions: np.ndarray) -> dict[str, np.ndarray]:
    if expressions.ndim != 2:
        raise ValueError("expressions must be [N, D]")
    mean = expressions.mean(axis=0).astype(np.float32)
    std = expressions.std(axis=0).astype(np.float32) + 1e-6
    cov = np.cov(expressions.T).astype(np.float32)
    cov = cov + np.eye(cov.shape[0], dtype=np.float32) * 1e-5
    return {"mean": mean, "std": std, "cov": cov}


def augment_params(
    params: dict[str, np.ndarray],
    stats: dict[str, np.ndarray],
    rng: np.random.Generator,
    cfg: AugmentConfig,
    mix_expression: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    out = {k: v.copy() for k, v in params.items()}
    mean = stats["mean"]
    std = stats["std"]
    cov = stats["cov"]

    if mix_expression is not None and rng.random() < cfg.expression_mixup_prob:
        lam = float(rng.uniform(0.3, 0.7))
        out["expression"] = lam * out["expression"] + (1.0 - lam) * mix_expression
    else:
        noise = rng.multivariate_normal(np.zeros_like(mean), cov).astype(np.float32)
        out["expression"] = out["expression"] + cfg.expression_strength * noise
    out["expression"] = np.clip(out["expression"], mean - 2.5 * std, mean + 2.5 * std)

    yaw, pitch, roll = POSE_LIMITS_DEG.get(cfg.stage, POSE_LIMITS_DEG[1])
    head_noise = np.deg2rad(
        np.asarray(
            [
                rng.uniform(-pitch, pitch),
                rng.uniform(-yaw, yaw),
                rng.uniform(-roll, roll),
            ],
            dtype=np.float32,
        )
    )
    out["pose"][:3] = out["pose"][:3] + head_noise

    jaw_noise = np.deg2rad(
        np.asarray(
            [
                rng.uniform(0.0, 20.0),
                rng.uniform(-5.0, 5.0),
                rng.uniform(-5.0, 5.0),
            ],
            dtype=np.float32,
        )
    )
    out["pose"][3:] = out["pose"][3:] + jaw_noise

    out["light"] = out["light"] + rng.normal(0.0, cfg.light_strength, size=out["light"].shape).astype(np.float32)
    out["camera"] = out["camera"] + rng.normal(0.0, cfg.camera_strength, size=out["camera"].shape).astype(np.float32)
    return out

