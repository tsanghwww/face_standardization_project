"""Train and apply an XGBoost quality filter for Phase2 samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_FEATURE_COLUMNS = [
    "quality_score",
    "exp_norm",
    "head_pose_norm",
    "jaw_pose_norm",
    "landmark_score",
    "landmark_out_ratio",
    "landmark_bbox_area",
    "landmark_center_dist",
    "arcface_status",
    "arcface_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--screening-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--feature-columns", nargs="*", default=DEFAULT_FEATURE_COLUMNS)
    parser.add_argument("--num-boost-round", type=int, default=220)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--high-threshold", type=float, default=0.55)
    parser.add_argument("--medium-threshold", type=float, default=0.45)
    parser.add_argument("--high-weight", type=float, default=1.0)
    parser.add_argument("--medium-weight", type=float, default=0.45)
    parser.add_argument("--low-weight", type=float, default=0.08)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_screening_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    labels: dict[str, str] = {}
    for row in rows:
        if "image_id" in row and "label" in row:
            labels[str(row["image_id"])] = str(row["label"])
    return labels


def as_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except ValueError:
        return 0.0
    return value if math.isfinite(value) else 0.0


def label_to_binary(label: str) -> int | None:
    normalized = label.strip().lower()
    if normalized == "pass":
        return 1
    if normalized in {"warn", "fail", "failed", "reject"}:
        return 0
    return None


def stratified_split(y: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_parts = []
    val_parts = []
    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        val_count = max(1, int(round(idx.size * val_ratio))) if idx.size > 1 else 0
        val_parts.append(idx[:val_count])
        train_parts.append(idx[val_count:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    weights = np.ones_like(y, dtype=np.float32)
    counts = {int(cls): int((y == cls).sum()) for cls in np.unique(y)}
    total = float(y.size)
    for cls, count in counts.items():
        weights[y == cls] = total / max(1.0, len(counts) * count)
    return weights


def binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1, dtype=np.float64)
    pos_rank_sum = float(ranks[pos].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = scores >= threshold
    pos = y_true == 1
    neg = y_true == 0
    tp = float(np.logical_and(pred, pos).sum())
    tn = float(np.logical_and(~pred, neg).sum())
    fp = float(np.logical_and(pred, neg).sum())
    fn = float(np.logical_and(~pred, pos).sum())
    acc = (tp + tn) / max(1.0, y_true.size)
    tpr = tp / max(1.0, tp + fn)
    tnr = tn / max(1.0, tn + fp)
    precision = tp / max(1.0, tp + fp)
    return {
        "accuracy": acc,
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "precision": precision,
        "recall": tpr,
        "specificity": tnr,
        "auc": binary_auc(y_true, scores),
    }


def quality_label(score: float, high_threshold: float, medium_threshold: float) -> str:
    if score >= high_threshold:
        return "high"
    if score >= medium_threshold:
        return "medium"
    return "low"


def row_weight(label: str, high_weight: float, medium_weight: float, low_weight: float) -> float:
    if label == "high":
        return high_weight
    if label == "medium":
        return medium_weight
    return low_weight


def write_feature_importance(path: Path, booster: Any, feature_columns: list[str]) -> None:
    gains = booster.get_score(importance_type="gain")
    covers = booster.get_score(importance_type="cover")
    freqs = booster.get_score(importance_type="weight")
    rows = []
    for name in feature_columns:
        rows.append(
            {
                "feature": name,
                "gain": float(gains.get(name, 0.0)),
                "cover": float(covers.get(name, 0.0)),
                "frequency": float(freqs.get(name, 0.0)),
            }
        )
    rows.sort(key=lambda r: r["gain"], reverse=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "gain", "cover", "frequency"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - runtime dependency check
        raise SystemExit("xgboost is not installed. Run: python -m pip install xgboost") from exc

    manifest_rows = read_csv_rows(args.manifest)
    screening_labels = read_screening_labels(args.screening_report)
    if not manifest_rows:
        raise SystemExit(f"No rows found in {args.manifest}")

    rows: list[dict[str, str]] = []
    labels: list[int] = []
    for row in manifest_rows:
        image_id = str(row.get("image_id", ""))
        screening_label = screening_labels.get(image_id, "")
        binary = label_to_binary(screening_label)
        if binary is None:
            continue
        enriched = dict(row)
        enriched["screening_label"] = screening_label
        rows.append(enriched)
        labels.append(binary)
    if not rows:
        raise SystemExit("No manifest rows could be matched to screening labels.")

    missing_columns = [column for column in args.feature_columns if column not in rows[0]]
    if missing_columns:
        raise SystemExit(f"Missing feature columns in manifest: {missing_columns}")

    x = np.asarray([[as_float(row, column) for column in args.feature_columns] for row in rows], dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    train_idx, val_idx = stratified_split(y, args.val_ratio, args.seed)
    weights = class_balanced_weights(y)

    dtrain = xgb.DMatrix(x[train_idx], label=y[train_idx], weight=weights[train_idx], feature_names=args.feature_columns)
    dval = xgb.DMatrix(x[val_idx], label=y[val_idx], weight=weights[val_idx], feature_names=args.feature_columns)
    dall = xgb.DMatrix(x, feature_names=args.feature_columns)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": 0.045,
        "max_depth": 3,
        "min_child_weight": 4.0,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "lambda": 2.0,
        "alpha": 0.2,
        "tree_method": "hist",
        "seed": args.seed,
    }
    evals_result: dict[str, Any] = {}
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=args.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=25,
    )

    train_scores = booster.predict(dtrain, iteration_range=(0, booster.best_iteration + 1))
    val_scores = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    all_scores = booster.predict(dall, iteration_range=(0, booster.best_iteration + 1))
    train_metrics = metrics_at_threshold(y[train_idx], train_scores)
    val_metrics = metrics_at_threshold(y[val_idx], val_scores)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "xgb_quality_model.json"
    booster.set_attr(
        feature_columns=json.dumps(args.feature_columns),
        high_threshold=str(args.high_threshold),
        medium_threshold=str(args.medium_threshold),
    )
    booster.save_model(model_path)
    write_feature_importance(args.out_dir / "xgb_feature_importance.csv", booster, args.feature_columns)

    counts: dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    out_manifest = args.out_dir / "xgb_quality_manifest.csv"
    output_fields = list(rows[0].keys()) + [
        "xgb_quality_score",
        "xgb_quality_label",
        "xgb_sample_weight",
        "xgb_use_for_strong_train",
        "xgb_use_for_weak_train",
    ]
    with out_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row, score in zip(rows, all_scores):
            q = float(score)
            label = quality_label(q, args.high_threshold, args.medium_threshold)
            counts[label] += 1
            out = dict(row)
            out["xgb_quality_score"] = f"{q:.8f}"
            out["xgb_quality_label"] = label
            out["xgb_sample_weight"] = f"{row_weight(label, args.high_weight, args.medium_weight, args.low_weight):.6f}"
            out["xgb_use_for_strong_train"] = "true" if label == "high" else "false"
            out["xgb_use_for_weak_train"] = "true" if label in {"high", "medium"} else "false"
            writer.writerow(out)

    with (args.out_dir / "xgb_training_log.json").open("w", encoding="utf-8") as f:
        json.dump(evals_result, f, indent=2)

    summary = {
        "manifest": str(args.manifest),
        "screening_report": str(args.screening_report),
        "model": str(model_path),
        "xgb_manifest": str(out_manifest),
        "feature_importance": str(args.out_dir / "xgb_feature_importance.csv"),
        "feature_columns": args.feature_columns,
        "rows": len(rows),
        "label_counts": {
            "Pass": int((y == 1).sum()),
            "Warn_or_low": int((y == 0).sum()),
        },
        "train_rows": int(train_idx.size),
        "val_rows": int(val_idx.size),
        "best_iteration": int(booster.best_iteration),
        "best_score": float(booster.best_score),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "thresholds": {
            "high": args.high_threshold,
            "medium": args.medium_threshold,
        },
        "xgb_quality_counts": counts,
        "weights": {
            "high": args.high_weight,
            "medium": args.medium_weight,
            "low": args.low_weight,
        },
    }
    (args.out_dir / "xgb_quality_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
