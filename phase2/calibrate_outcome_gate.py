"""Select and freeze the gate threshold on hard_calibration ONLY.

Outputs AUROC/AUPRC/Brier/ECE, risk-coverage, FAR/FRR, accepted-unsafe-rate,
and high/medium/low + source stratified confusion.  The chosen threshold is
written into a frozen gate JSON; fixed test is never touched here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .train_outcome_gate import gate_feature_matrix, predict_logistic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcome-manifest", required=True, type=Path)
    p.add_argument("--hard-calibration-ids", required=True, type=Path)
    p.add_argument("--gate-model", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--target-risk", type=float, default=0.10, help="max accepted unsafe rate (CLI)")
    p.add_argument("--risk-confidence", type=float, default=0.95)
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def auroc(y, score):
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    s = 0
    while s < len(order):
        e = s + 1
        while e < len(order) and score[order[e]] == score[order[s]]:
            e += 1
        ranks[order[s:e]] = (s + 1 + e) / 2.0
        s = e
    pos = int(y.sum())
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * (len(y) - pos)))


def auprc(y, score):
    order = np.argsort(-score, kind="mergesort")
    sy = y[order]
    prec = np.cumsum(sy) / np.arange(1, len(y) + 1)
    return float(prec[sy == 1].mean())


def ece(y, score, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    total = y.size
    v = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if m.any():
            v += m.sum() / total * abs(float(y[m].mean()) - float(score[m].mean()))
    return v


def confusion(y, accepted):
    unsafe = y == 1
    safe = ~unsafe
    fa = int((unsafe & accepted).sum())
    fr = int((safe & ~accepted).sum())
    return {
        "n": len(y), "accepted": int(accepted.sum()),
        "false_accept": fa, "false_reject": fr,
        "coverage": int(accepted.sum()) / len(y),
        "far": fa / unsafe.sum() if unsafe.sum() else float("nan"),
        "frr": fr / safe.sum() if safe.sum() else float("nan"),
        "accepted_unsafe_rate": fa / accepted.sum() if accepted.sum() else float("nan"),
    }


def wilson_upper(successes: int, total: int, confidence: float) -> float:
    if total == 0:
        return float("nan")
    # One-sided normal quantiles used by the supported protocol settings.
    z_by_confidence = {0.90: 1.2815515655, 0.95: 1.6448536270, 0.975: 1.9599639845, 0.99: 2.3263478740}
    if confidence not in z_by_confidence:
        raise ValueError(f"Unsupported --risk-confidence {confidence}; choose {sorted(z_by_confidence)}")
    z = z_by_confidence[confidence]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total))
    return (center + radius) / denominator


def main() -> None:
    args = parse_args()
    outcomes = {r["image_id"]: r for r in read_csv(args.outcome_manifest)}
    cal_ids = [ln.strip() for ln in args.hard_calibration_ids.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = [outcomes[i] for i in cal_ids if i in outcomes]
    gate = json.loads(args.gate_model.read_text(encoding="utf-8"))
    model = {"mean": np.asarray(gate["mean"]), "scale": np.asarray(gate["scale"]), "intercept": gate["intercept"], "coefficient": np.asarray(gate["coefficient"])}
    if len(rows) != len(cal_ids):
        raise SystemExit(f"Calibration outcome coverage {len(rows)}/{len(cal_ids)}")
    x, _ = gate_feature_matrix(rows, np.asarray(gate["impute_values"], dtype=np.float64))
    y = np.asarray([1.0 if r["unsafe"] == "1" else 0.0 for r in rows], dtype=np.float64)
    risk = predict_logistic(model, x)

    # Maximize coverage subject to a one-sided confidence bound on accepted risk.
    threshold = None
    for t in sorted(np.unique(np.r_[0.0, np.nextafter(risk, np.inf), 1.000001]), reverse=True):
        accepted = risk < t
        false_accepts = int(y[accepted].sum())
        risk_upper = wilson_upper(false_accepts, int(accepted.sum()), args.risk_confidence)
        if accepted.sum() and risk_upper <= args.target_risk + 1e-12:
            threshold = float(t)
            break
    accepted = risk < threshold if threshold is not None else np.zeros(len(risk), dtype=bool)
    false_accepts = int(y[accepted].sum())
    accepted_risk_upper = (
        wilson_upper(false_accepts, int(accepted.sum()), args.risk_confidence) if accepted.any() else None
    )

    labels = np.asarray([r.get("xgb_quality_label") or "unknown" for r in rows])
    stratified = []
    for lab in ["overall", *sorted(set(labels))]:
        m = np.ones(len(y), bool) if lab == "overall" else labels == lab
        stratified.append({"xgb_quality_label": lab, **confusion(y[m], accepted[m])})

    metrics = {
        "auroc": auroc(y, risk), "auprc": auprc(y, risk),
        "brier": float(np.mean((risk - y) ** 2)), "ece_10bin": ece(y, risk),
        "threshold": threshold, "target_risk": args.target_risk,
        "overall": confusion(y, accepted), "stratified_by_xgb_quality_label": stratified,
        "n_calibration": len(y), "unsafe": int(y.sum()),
        "accepted_unsafe_rate_upper": accepted_risk_upper,
        "risk_confidence": args.risk_confidence,
        "deployment_qualified": threshold is not None,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frozen = dict(gate)
    frozen["threshold"] = threshold
    frozen["target_risk"] = args.target_risk
    frozen["risk_confidence"] = args.risk_confidence
    frozen["deployment_qualified"] = threshold is not None
    frozen["frozen_on"] = f"hard_calibration_{len(cal_ids)}"
    frozen["fixed_test_used"] = False
    (args.out_dir / "outcome_gate_frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (args.out_dir / "outcome_gate_calibration_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.out_dir / "outcome_gate_calibration_stratified.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["xgb_quality_label", "n", "accepted", "false_accept", "false_reject", "coverage", "far", "frr", "accepted_unsafe_rate"])
        w.writeheader()
        w.writerows(stratified)
    risk_curve_rows = []
    with (args.out_dir / "outcome_gate_risk_coverage.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["threshold", "accepted", "coverage", "accepted_unsafe_rate", "accepted_unsafe_rate_upper"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for threshold_value in sorted(np.unique(np.nextafter(risk, np.inf))):
            mask = risk < threshold_value
            if not mask.any():
                continue
            false_accept_count = int(y[mask].sum())
            curve_row = {
                "threshold": threshold_value,
                "accepted": int(mask.sum()),
                "coverage": float(mask.mean()),
                "accepted_unsafe_rate": float(y[mask].mean()),
                "accepted_unsafe_rate_upper": wilson_upper(false_accept_count, int(mask.sum()), args.risk_confidence),
            }
            risk_curve_rows.append(curve_row)
            writer.writerow(curve_row)
    if risk_curve_rows:
        metrics["best_observed_risk_bound"] = min(
            risk_curve_rows, key=lambda row: (row["accepted_unsafe_rate_upper"], -row["coverage"])
        )
        (args.out_dir / "outcome_gate_calibration_summary.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
