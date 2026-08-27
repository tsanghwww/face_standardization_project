"""Apply a validation-frozen gate once to the immutable fixed test set."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .calibrate_gate import FEATURES, MARGINS, classification_summary, number, predict_logistic, write_csv
from .features import read_arcface_rows, sample_from_mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--inference-manifest", required=True, type=Path)
    parser.add_argument("--xgb-manifest", required=True, type=Path)
    parser.add_argument("--arcface-manifest", required=True, type=Path)
    parser.add_argument("--calibrator-json", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def outcome_for(metrics: dict[tuple[str, str], dict[str, str]], eval_id: str) -> tuple[str, str]:
    original = metrics.get((eval_id, "original"), {})
    hard = metrics.get((eval_id, "hard_zero"), {})
    full = metrics.get((eval_id, "full"), {})
    values = {
        "hard_arcface": number(hard, "arcface_cosine"), "full_arcface": number(full, "arcface_cosine"),
        "hard_pose": number(hard, "deca_head_pose_norm"), "full_pose": number(full, "deca_head_pose_norm"),
        "hard_expression": number(hard, "deca_expression_norm"), "full_expression": number(full, "deca_expression_norm"),
        "hard_gaze": number(hard, "l2cs_gaze_delta_vs_original_deg"), "full_gaze": number(full, "l2cs_gaze_delta_vs_original_deg"),
        "original_pose": number(original, "deca_head_pose_norm"), "original_expression": number(original, "deca_expression_norm"),
    }
    missing = [key for key, value in values.items() if value is None]
    reasons: list[str] = []
    if missing:
        reasons.append("missing_required_metric:" + ",".join(missing))
    else:
        if values["full_arcface"] - values["hard_arcface"] < -MARGINS["arcface_cosine_noninferiority"]:
            reasons.append("identity_noninferiority_failed")
        if values["full_pose"] - values["hard_pose"] > MARGINS["deca_pose_norm_degradation"]:
            reasons.append("pose_degradation")
        if values["full_expression"] - values["hard_expression"] > MARGINS["deca_expression_rms_degradation"]:
            reasons.append("expression_degradation")
        if values["full_gaze"] - values["hard_gaze"] > MARGINS["l2cs_gaze_degradation_deg"]:
            reasons.append("gaze_degradation")
    if reasons:
        return "unsafe", ";".join(reasons)
    effective = (
        values["full_pose"] <= 0.5 * values["original_pose"] + MARGINS["deca_pose_norm_degradation"]
        and values["full_expression"] <= 0.5 * values["original_expression"] + MARGINS["deca_expression_rms_degradation"]
    )
    return ("safe_and_effective" if effective else "safe_but_ineffective"), ""


def main() -> None:
    args = parse_args()
    frozen = json.loads(args.calibrator_json.read_text(encoding="utf-8"))
    if frozen["selected_model"] != "logistic":
        raise SystemExit(f"This audit expects frozen logistic model, got {frozen['selected_model']}")
    logistic = frozen["logistic"]
    model = {
        "mean": np.asarray(logistic["standardizer_mean"], dtype=np.float64),
        "scale": np.asarray(logistic["standardizer_scale"], dtype=np.float64),
        "coefficient": np.asarray(logistic["coefficient"], dtype=np.float64),
        "intercept": float(logistic["intercept"]),
    }
    thresholds = {float(row["max_accepted_unsafe_rate"]): float(row["threshold"]) for row in frozen["operating_points"]}
    weak_threshold, reject_threshold = thresholds[0.05], thresholds[0.10]
    tests = read_csv(args.test_manifest)
    metrics = {(row["eval_id"], row["method"]): row for row in read_csv(args.metrics_csv)}
    inference = {row["image_id"]: row for row in read_csv(args.inference_manifest)}
    xgb = {row["image_id"]: row for row in read_csv(args.xgb_manifest)}
    arcface = read_arcface_rows(args.arcface_manifest)
    rows: list[dict] = []
    scored_features: list[list[float]] = []
    scored_rows: list[dict] = []
    for test in tests:
        image_id = test["image_id"]
        inf = inference.get(image_id)
        if inf is None:
            rows.append({
                "eval_id": test["eval_id"], "image_id": image_id, "source_group": test.get("source_group", ""),
                "xgb_quality_label": (xgb.get(image_id) or {}).get("xgb_quality_label", ""),
                "pipeline_status": "upstream_deca_failure", "risk": "", "gate_decision": "reject",
                "outcome": "upstream_failure", "unsafe_reason": "upstream_deca_failure",
            })
            continue
        sample = sample_from_mat(Path(inf["mat_path"]), arcface.get(image_id))
        xr = xgb.get(image_id, {})
        feature = {
            "reject_score": number(inf, "reject_score") or 0.0, "confidence": number(inf, "confidence") or 0.0,
            "quality_score": number(inf, "quality_score") or 0.0,
            "heuristic_quality_score": number(inf, "heuristic_quality_score") or 0.0,
            "xgb_quality_score": number(inf, "xgb_quality_score") or 0.0,
            "xgb_status": 1.0 if number(inf, "xgb_quality_score") is not None else 0.0,
            "landmark_score": sample.metrics["landmark_score"], "landmark_out_ratio": sample.metrics["landmark_out_ratio"],
            "landmark_bbox_area": sample.metrics["landmark_bbox_area"], "landmark_center_dist": sample.metrics["landmark_center_dist"],
            "arcface_status": sample.metrics["arcface_status"], "arcface_score": sample.metrics["arcface_score"],
            "original_exp_norm": number(inf, "original_exp_norm") or 0.0,
            "original_head_pose_norm": number(inf, "original_head_pose_norm") or 0.0,
            "original_jaw_pose_norm": number(inf, "original_jaw_pose_norm") or 0.0,
            "alpha_expression": number(inf, "alpha_expression") or 0.0,
            "alpha_head_pose": number(inf, "alpha_head_pose") or 0.0,
            "alpha_jaw_pose": number(inf, "alpha_jaw_pose") or 0.0,
        }
        outcome, reason = outcome_for(metrics, test["eval_id"])
        row = {
            "eval_id": test["eval_id"], "image_id": image_id, "source_group": test.get("source_group", ""),
            "xgb_quality_label": xr.get("xgb_quality_label", test.get("xgb_quality_label", "")),
            "pipeline_status": "available", "outcome": outcome, "unsafe_reason": reason, **feature,
        }
        scored_features.append([feature[name] for name in FEATURES])
        scored_rows.append(row)
        rows.append(row)
    risks = predict_logistic(model, np.asarray(scored_features, dtype=np.float64))
    for row, risk in zip(scored_rows, risks):
        row["risk"] = float(risk)
        row["gate_decision"] = "standardize" if risk < weak_threshold else ("weak_standardize" if risk < reject_threshold else "reject")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "gate_fixed_test_predictions.csv", rows)
    available = [row for row in rows if row["pipeline_status"] == "available"]
    y = np.asarray([row["outcome"] == "unsafe" for row in available], dtype=np.int64)
    accepted = np.asarray([row["gate_decision"] != "reject" for row in available])
    strata_rows = []
    for field in ("xgb_quality_label", "source_group"):
        labels = np.asarray([row[field] or "unknown" for row in available])
        for label in ["overall", *sorted(set(labels))]:
            mask = np.ones(len(y), dtype=bool) if label == "overall" else labels == label
            strata_rows.append({"stratify_by": field, "stratum": label, **classification_summary(y[mask], accepted[mask])})
    write_csv(args.out_dir / "gate_fixed_test_confusion.csv", strata_rows)
    summary = {
        "fixed_test_n": len(rows), "available": len(available),
        "upstream_failure": sum(row["pipeline_status"] != "available" for row in rows),
        "outcome_counts": {name: sum(row["outcome"] == name for row in rows) for name in ("unsafe", "safe_but_ineffective", "safe_and_effective", "upstream_failure")},
        "decision_counts": {name: sum(row["gate_decision"] == name for row in rows) for name in ("standardize", "weak_standardize", "reject")},
        "available_confusion": strata_rows[0],
        "calibrator_source": str(args.calibrator_json),
        "thresholds_frozen": {"weak": weak_threshold, "reject": reject_threshold},
        "test_used_for_selection": False,
    }
    (args.out_dir / "gate_fixed_test_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
