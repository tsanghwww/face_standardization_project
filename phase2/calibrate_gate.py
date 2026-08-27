"""Fit and freeze a validation-only post-hoc Phase2 safety gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


MARGINS = {
    "arcface_cosine_noninferiority": 0.02,
    "deca_pose_norm_degradation": 0.03,
    "deca_expression_rms_degradation": 0.02,
    "l2cs_gaze_degradation_deg": 10.0,
}
FEATURES = [
    "reject_score", "confidence", "quality_score", "heuristic_quality_score",
    "xgb_quality_score", "xgb_status", "landmark_score", "landmark_out_ratio",
    "landmark_bbox_area", "landmark_center_dist", "arcface_status", "arcface_score",
    "original_exp_norm", "original_head_pose_norm", "original_jaw_pose_norm",
    "alpha_expression", "alpha_head_pose", "alpha_jaw_pose",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--xgb-oof-manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row.get(key, ""))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def ece(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = y.size
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & (score < hi if hi < 1.0 else score <= hi)
        if mask.any():
            value += mask.sum() / total * abs(float(y[mask].mean()) - float(score[mask].mean()))
    return value


def score_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positives = int(y.sum())
    negatives = len(y) - positives
    auroc = (float(ranks[y == 1].sum()) - positives * (positives + 1) / 2.0) / (positives * negatives)
    desc = np.argsort(-score, kind="mergesort")
    sorted_y = y[desc]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    auprc = float(precision[sorted_y == 1].mean())
    return {
        "auroc": auroc,
        "auprc": auprc,
        "brier": float(np.mean((score - y) ** 2)),
        "ece_10bin": ece(y, score),
    }


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for label in (0, 1):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            buckets[position % n_splits].append(int(index))
    return [np.asarray(sorted(bucket), dtype=np.int64) for bucket in buckets]


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iterations: int = 100) -> dict[str, np.ndarray | float]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(design.shape[1], dtype=np.float64)
    weights = np.ones(len(y), dtype=np.float64)
    penalty = np.r_[0.0, np.full(z.shape[1], l2)]
    for _ in range(iterations):
        prediction = sigmoid(design @ beta)
        grad = design.T @ ((prediction - y) * weights) / len(y) + penalty * beta / len(y)
        curvature = np.maximum(prediction * (1.0 - prediction) * weights, 1e-8)
        hessian = (design.T * curvature) @ design / len(y) + np.diag(penalty / len(y))
        step = np.linalg.solve(hessian + np.eye(len(beta)) * 1e-8, grad)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            break
    return {"mean": mean, "scale": scale, "intercept": float(beta[0]), "coefficient": beta[1:]}


def predict_logistic(model: dict[str, np.ndarray | float], x: np.ndarray) -> np.ndarray:
    z = (x - model["mean"]) / model["scale"]
    return sigmoid(z @ model["coefficient"] + model["intercept"])


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return center - radius, center + radius


def classification_summary(y: np.ndarray, accepted: np.ndarray) -> dict[str, float | int]:
    unsafe = y == 1
    safe = ~unsafe
    false_accept = int((unsafe & accepted).sum())
    false_reject = int((safe & ~accepted).sum())
    accepted_count = int(accepted.sum())
    unsafe_count = int(unsafe.sum())
    safe_count = int(safe.sum())
    far_ci = wilson(false_accept, unsafe_count)
    frr_ci = wilson(false_reject, safe_count)
    risk_ci = wilson(false_accept, accepted_count)
    return {
        "n": len(y), "unsafe": unsafe_count, "safe": safe_count,
        "accepted": accepted_count, "rejected": int((~accepted).sum()),
        "false_accept": false_accept, "false_reject": false_reject,
        "coverage": accepted_count / len(y) if len(y) else float("nan"),
        "selective_risk": false_accept / accepted_count if accepted_count else float("nan"),
        "selective_risk_ci95_lo": risk_ci[0], "selective_risk_ci95_hi": risk_ci[1],
        "far": false_accept / unsafe_count if unsafe_count else float("nan"),
        "far_ci95_lo": far_ci[0], "far_ci95_hi": far_ci[1],
        "frr": false_reject / safe_count if safe_count else float("nan"),
        "frr_ci95_lo": frr_ci[0], "frr_ci95_hi": frr_ci[1],
    }


def operating_point(y: np.ndarray, risk: np.ndarray, max_risk: float) -> dict[str, float | int]:
    candidates = np.unique(np.r_[0.0, np.nextafter(risk, np.inf), 1.000001])
    best: dict[str, float | int] | None = None
    for threshold in candidates:
        accepted = risk < threshold
        n = int(accepted.sum())
        selective_risk = float(y[accepted].mean()) if n else 0.0
        if selective_risk <= max_risk + 1e-12:
            candidate = {
                "max_accepted_unsafe_rate": max_risk,
                "threshold": float(threshold),
                "accepted": n,
                "coverage": n / len(y),
                "accepted_unsafe": int(y[accepted].sum()),
                "selective_risk": selective_risk,
            }
            if best is None or candidate["coverage"] > best["coverage"]:
                best = candidate
    assert best is not None
    return best


def main() -> None:
    args = parse_args()
    metrics = read_csv(args.metrics_csv)
    by_eval_method = {(row["eval_id"], row["method"]): row for row in metrics}
    inference = {row["image_id"]: row for row in read_csv(args.inference_manifest)}
    xgb = {row["image_id"]: row for row in read_csv(args.xgb_oof_manifest)}
    eval_ids = sorted(inference)
    outcomes: list[dict] = []
    matrix: list[list[float]] = []

    for eval_id in eval_ids:
        original = by_eval_method.get((eval_id, "original"), {})
        hard = by_eval_method.get((eval_id, "hard_zero"), {})
        full = by_eval_method.get((eval_id, "full"), {})
        inf = inference[eval_id]
        xr = xgb.get(eval_id, {})
        required = {
            "hard_arcface": number(hard, "arcface_cosine"),
            "full_arcface": number(full, "arcface_cosine"),
            "hard_pose": number(hard, "deca_head_pose_norm"),
            "full_pose": number(full, "deca_head_pose_norm"),
            "hard_expression": number(hard, "deca_expression_norm"),
            "full_expression": number(full, "deca_expression_norm"),
            "hard_gaze": number(hard, "l2cs_gaze_delta_vs_original_deg"),
            "full_gaze": number(full, "l2cs_gaze_delta_vs_original_deg"),
            "original_pose": number(original, "deca_head_pose_norm"),
            "original_expression": number(original, "deca_expression_norm"),
        }
        missing = [key for key, value in required.items() if value is None]
        id_delta = None if missing else required["full_arcface"] - required["hard_arcface"]
        pose_delta = None if missing else required["full_pose"] - required["hard_pose"]
        expression_delta = None if missing else required["full_expression"] - required["hard_expression"]
        gaze_delta = None if missing else required["full_gaze"] - required["hard_gaze"]
        unsafe_reasons: list[str] = []
        if missing:
            unsafe_reasons.append("missing_required_metric:" + ",".join(missing))
        else:
            if id_delta < -MARGINS["arcface_cosine_noninferiority"]:
                unsafe_reasons.append("identity_noninferiority_failed")
            if pose_delta > MARGINS["deca_pose_norm_degradation"]:
                unsafe_reasons.append("pose_degradation")
            if expression_delta > MARGINS["deca_expression_rms_degradation"]:
                unsafe_reasons.append("expression_degradation")
            if gaze_delta > MARGINS["l2cs_gaze_degradation_deg"]:
                unsafe_reasons.append("gaze_degradation")
        unsafe = bool(unsafe_reasons)
        effective = False
        if not unsafe:
            effective = (
                required["full_pose"] <= 0.5 * required["original_pose"] + MARGINS["deca_pose_norm_degradation"]
                and required["full_expression"] <= 0.5 * required["original_expression"] + MARGINS["deca_expression_rms_degradation"]
            )
        outcome = "unsafe" if unsafe else ("safe_and_effective" if effective else "safe_but_ineffective")

        feature_row = {
            "reject_score": number(inf, "reject_score") or 0.0,
            "confidence": number(inf, "confidence") or 0.0,
            "quality_score": number(inf, "quality_score") or 0.0,
            "heuristic_quality_score": number(inf, "heuristic_quality_score") or 0.0,
            "xgb_quality_score": number(inf, "xgb_quality_score") or 0.0,
            "xgb_status": 1.0 if number(inf, "xgb_quality_score") is not None else 0.0,
            "landmark_score": number(xr, "landmark_score") or 0.0,
            "landmark_out_ratio": number(xr, "landmark_out_ratio") or 0.0,
            "landmark_bbox_area": number(xr, "landmark_bbox_area") or 0.0,
            "landmark_center_dist": number(xr, "landmark_center_dist") or 0.0,
            "arcface_status": number(xr, "arcface_status") or 0.0,
            "arcface_score": number(xr, "arcface_score") or 0.0,
            "original_exp_norm": number(inf, "original_exp_norm") or 0.0,
            "original_head_pose_norm": number(inf, "original_head_pose_norm") or 0.0,
            "original_jaw_pose_norm": number(inf, "original_jaw_pose_norm") or 0.0,
            "alpha_expression": number(inf, "alpha_expression") or 0.0,
            "alpha_head_pose": number(inf, "alpha_head_pose") or 0.0,
            "alpha_jaw_pose": number(inf, "alpha_jaw_pose") or 0.0,
        }
        matrix.append([feature_row[name] for name in FEATURES])
        outcomes.append({
            "eval_id": eval_id, "outcome": outcome, "unsafe": int(unsafe),
            "unsafe_reason": ";".join(unsafe_reasons), "xgb_quality_label": xr.get("xgb_quality_label", ""),
            "identity_delta_vs_hard_zero": id_delta, "pose_delta_vs_hard_zero": pose_delta,
            "expression_delta_vs_hard_zero": expression_delta, "gaze_delta_vs_hard_zero_deg": gaze_delta,
            **feature_row,
        })

    x = np.asarray(matrix, dtype=np.float64)
    y = np.asarray([row["unsafe"] for row in outcomes], dtype=np.int64)
    if np.unique(y).size != 2:
        raise SystemExit(f"Outcome labels contain only one class: {np.unique(y)}")
    folds = stratified_folds(y, args.folds, args.seed)
    logistic_oof = np.empty(len(y), dtype=np.float64)
    all_indices = np.arange(len(y))
    for held_out in folds:
        train = np.setdiff1d(all_indices, held_out, assume_unique=True)
        fold_model = fit_logistic(x[train], y[train])
        logistic_oof[held_out] = predict_logistic(fold_model, x[held_out])
    scores = {
        "reject_score": np.asarray([row["reject_score"] for row in outcomes]),
        "mean_reject_one_minus_confidence": np.asarray([
            (row["reject_score"] + 1.0 - row["confidence"]) / 2.0 for row in outcomes
        ]),
        "logistic": logistic_oof,
    }
    model_rows = []
    for name, score in scores.items():
        model_rows.append({"model": name, **score_metrics(y, score)})
    selected = max(model_rows, key=lambda row: (row["auprc"], row["auroc"], -row["brier"]))["model"]
    selected_risk = scores[selected]
    for index, row in enumerate(outcomes):
        for name, score in scores.items():
            row[f"risk_{name}"] = float(score[index])

    fitted = fit_logistic(x, y)
    ops = [operating_point(y, selected_risk, target) for target in (0.05, 0.10, 0.15)]
    weak_threshold = float(ops[0]["threshold"])
    reject_threshold = float(ops[1]["threshold"])
    for index, row in enumerate(outcomes):
        risk = float(selected_risk[index])
        row["gate_decision"] = "standardize" if risk < weak_threshold else ("weak_standardize" if risk < reject_threshold else "reject")

    risk_curve = []
    for threshold in np.unique(np.r_[0.0, np.nextafter(selected_risk, np.inf), 1.000001]):
        accepted = selected_risk < threshold
        risk_curve.append({"threshold": float(threshold), **classification_summary(y, accepted)})

    accepted = selected_risk < reject_threshold
    quality_rows = []
    labels = np.asarray([row["xgb_quality_label"] or "unknown" for row in outcomes])
    for label in ["overall", *sorted(set(labels))]:
        mask = np.ones(len(y), dtype=bool) if label == "overall" else labels == label
        quality_rows.append({"xgb_quality_label": label, **classification_summary(y[mask], accepted[mask])})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "gate_validation_outcomes.csv", outcomes)
    write_csv(args.out_dir / "gate_model_comparison.csv", model_rows)
    write_csv(args.out_dir / "gate_threshold_search.csv", ops)
    write_csv(args.out_dir / "gate_risk_coverage.csv", risk_curve)
    write_csv(args.out_dir / "gate_confusion_by_quality.csv", quality_rows)
    definition = {
        "frozen_on": "phase2_validation_1440",
        "fixed_test_used": False,
        "margins": MARGINS,
        "missing_required_metric_policy": "unsafe",
        "effectiveness_rule": "full pose/expression <= 0.5 * original + corresponding margin",
        "scope": "diagnostic DECA render domain; not human perceptual ground truth",
    }
    (args.out_dir / "gate_outcome_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    calibrator = {
        "selected_model": selected,
        "features": FEATURES,
        "logistic": {
            "standardizer_mean": fitted["mean"].tolist(), "standardizer_scale": fitted["scale"].tolist(),
            "coefficient": fitted["coefficient"].tolist(), "intercept": fitted["intercept"],
            "fit": "Newton-Raphson logistic with L2=1.0; no class weighting",
        },
        "operating_points": ops,
        "seed": args.seed, "folds": args.folds,
        "fixed_test_used_for_selection": False,
    }
    (args.out_dir / "gate_calibrator.json").write_text(json.dumps(calibrator, indent=2), encoding="utf-8")
    summary = {
        "n": len(y), "unsafe": int(y.sum()), "safe": int((1 - y).sum()),
        "outcome_counts": {name: sum(row["outcome"] == name for row in outcomes) for name in ("unsafe", "safe_but_ineffective", "safe_and_effective")},
        "model_comparison": model_rows, "selected_model": selected, "operating_points": ops,
        "primary_validation_confusion": quality_rows[0],
        "test_data_used": False,
    }
    (args.out_dir / "gate_calibration_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
