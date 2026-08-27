"""Fit the Phase2.1 outcome gate on gate_train ONLY.

The gate is a logistic regression over source quality features + alphas; the
label is the rendered ``unsafe`` outcome.  Thresholds are NOT selected here --
they are frozen later on hard_calibration (calibrate_outcome_gate.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

GATE_FEATURES = [
    "reject_score", "confidence", "quality_score", "heuristic_quality_score",
    "xgb_quality_score", "xgb_status", "landmark_score", "landmark_out_ratio",
    "landmark_bbox_area", "landmark_center_dist", "arcface_status", "arcface_score",
    "original_exp_norm", "original_head_pose_norm", "original_jaw_pose_norm",
    "alpha_expression", "alpha_head_pose", "alpha_jaw_pose",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcome-manifest", required=True, type=Path)
    p.add_argument("--gate-train-ids", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--l2", type=float, default=1.0)
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def num(row, key):
    s = str(row.get(key, "")).strip()
    if s == "":
        return float("nan")
    try:
        value = float(s)
        return value if np.isfinite(value) else float("nan")
    except ValueError:
        return float("nan")


def gate_feature_matrix(rows: list[dict], impute_values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute base features and append explicit missing indicators."""
    raw = np.asarray([[num(row, key) for key in GATE_FEATURES] for row in rows], dtype=np.float64)
    missing = ~np.isfinite(raw)
    if impute_values is None:
        impute_values = np.nanmedian(raw, axis=0)
        impute_values = np.where(np.isfinite(impute_values), impute_values, 0.0)
    filled = np.where(missing, impute_values, raw)
    return np.column_stack([filled, missing.astype(np.float64)]), np.asarray(impute_values, dtype=np.float64)


def sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(v, -35, 35)))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float) -> dict:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.r_[0.0, np.full(z.shape[1], l2)]
    for _ in range(100):
        pred = sigmoid(design @ beta)
        grad = design.T @ ((pred - y)) / len(y) + penalty * beta / len(y)
        curv = np.maximum(pred * (1 - pred), 1e-8)
        hess = (design.T * curv) @ design / len(y) + np.diag(penalty / len(y))
        step = np.linalg.solve(hess + np.eye(len(beta)) * 1e-8, grad)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {"mean": mean, "scale": scale, "intercept": float(beta[0]), "coefficient": beta[1:]}


def predict_logistic(model: dict, x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    return sigmoid(z @ model["coefficient"] + model["intercept"])


def main() -> None:
    args = parse_args()
    outcomes = {r["image_id"]: r for r in read_csv(args.outcome_manifest)}
    train_ids = [ln.strip() for ln in args.gate_train_ids.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [outcomes[i] for i in train_ids if i in outcomes]
    if len(rows) != len(train_ids):
        raise SystemExit(f"Gate outcome coverage {len(rows)}/{len(train_ids)}")
    x, impute_values = gate_feature_matrix(rows)
    y = np.asarray([1.0 if r["unsafe"] == "1" else 0.0 for r in rows], dtype=np.float64)
    if np.unique(y).size < 2:
        raise SystemExit(f"gate_train contains only one class: {np.unique(y)}")
    model = fit_logistic(x, y, args.l2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gate_model = {
        "base_features": GATE_FEATURES,
        "features": [*GATE_FEATURES, *(f"{name}__missing" for name in GATE_FEATURES)],
        "impute_values": impute_values.tolist(),
        "mean": model["mean"].tolist(),
        "scale": model["scale"].tolist(),
        "coefficient": model["coefficient"].tolist(),
        "intercept": model["intercept"],
        "l2": args.l2,
        "n_train": len(rows),
        "fit": "logistic, Newton-Raphson",
        "threshold": None,  # frozen on hard_calibration only
        "label_scope": "rendered unsafe outcome (diagnostic DECA-render domain)",
    }
    (args.out_dir / "outcome_gate_model.json").write_text(json.dumps(gate_model, indent=2), encoding="utf-8")
    print(json.dumps({"n_train": len(rows), "unsafe": int(y.sum()), "features": x.shape[1]}, indent=2))


if __name__ == "__main__":
    main()
