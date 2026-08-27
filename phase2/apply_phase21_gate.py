"""Apply the FROZEN Phase2.1 gate to the fixed test exactly once.

Reads a frozen gate (features + threshold from hard_calibration), applies it to
a fixed-test outcome manifest, and writes decisions.  No threshold search is
possible here: the threshold comes only from the frozen gate JSON.

Rescue policy: primary inference FAN/DECA failures are NOT auto-filled from
rescue mats.  A --rescue-audit flag is the only path that may consult the
rescue manifest, and it only writes an audit table (never the primary mat_path).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .train_outcome_gate import gate_feature_matrix, predict_logistic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcome-manifest", dest="prediction_manifest", required=True, type=Path, help="fixed-test prediction feature manifest")
    p.add_argument("--evaluation-outcome-manifest", type=Path, help="optional labels used only after decisions")
    p.add_argument("--frozen-gate", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--rescue-audit", action="store_true", help="optional rescue audit (never fills primary mat_path)")
    p.add_argument("--rescue-manifest", type=Path)
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    gate = json.loads(args.frozen_gate.read_text(encoding="utf-8"))
    if gate.get("threshold") is None:
        raise SystemExit("frozen gate has no threshold")
    threshold = float(gate["threshold"])
    model = {"mean": np.asarray(gate["mean"]), "scale": np.asarray(gate["scale"]), "intercept": gate["intercept"], "coefficient": np.asarray(gate["coefficient"])}
    rows = read_csv(args.prediction_manifest)
    impute_values = np.asarray(gate["impute_values"], dtype=np.float64)
    matrix, _ = gate_feature_matrix(rows, impute_values)
    risks = predict_logistic(model, matrix)

    out_rows = []
    for r, risk_value in zip(rows, risks):
        risk = float(risk_value)
        pipeline_status = r.get("pipeline_status", "available") or "available"
        upstream_failure = pipeline_status not in {"available", "success"}
        decision = "manual_review" if upstream_failure else ("reject" if risk >= threshold else "accept")
        out_rows.append({
            "image_id": r["image_id"], "split": r.get("split", ""),
            "xgb_quality_label": r.get("xgb_quality_label", ""),
            "unsafe": r.get("unsafe", ""), "unsafe_reason": r.get("unsafe_reason", ""),
            "gate_risk": f"{risk:.6f}", "gate_decision": decision,
            "threshold_used": f"{threshold:.6f}",
            "pipeline_status": pipeline_status,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["image_id", "split", "xgb_quality_label", "unsafe", "unsafe_reason", "gate_risk", "gate_decision", "threshold_used", "pipeline_status"]
    with (args.out_dir / "phase21_gate_decisions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    evaluation = None
    if args.evaluation_outcome_manifest:
        labels = {row["image_id"]: row for row in read_csv(args.evaluation_outcome_manifest)}
        evaluable = [(row, labels.get(row["image_id"])) for row in out_rows]
        evaluable = [(row, label) for row, label in evaluable if label and label.get("unsafe", "") in {"0", "1"}]
        accepted = np.asarray([row["gate_decision"] == "accept" for row, _ in evaluable], dtype=bool)
        unsafe = np.asarray([label["unsafe"] == "1" for _, label in evaluable], dtype=bool)
        evaluation = {
            "n": len(evaluable),
            "false_accept": int((accepted & unsafe).sum()),
            "false_reject": int((~accepted & ~unsafe).sum()),
            "coverage": float(accepted.mean()) if len(accepted) else None,
            "accepted_unsafe_rate": float(unsafe[accepted].mean()) if accepted.any() else None,
            "far": float(accepted[unsafe].mean()) if unsafe.any() else None,
            "frr": float((~accepted[~unsafe]).mean()) if (~unsafe).any() else None,
        }

    rescue_rows = []
    if args.rescue_audit:
        if args.rescue_manifest is None:
            raise SystemExit("--rescue-audit requires --rescue-manifest")
        for row in read_csv(args.rescue_manifest):
            rescue_rows.append({
                "eval_id": row.get("eval_id", row.get("image_id", "")),
                "image_id": row.get("image_id", ""),
                "rescue_status": row.get("pipeline_status", row.get("status", "")),
                "rescue_mat_path": row.get("mat_path", ""),
                "policy_decision": "manual_review_only",
            })
        with (args.out_dir / "phase21_rescue_audit.csv").open("w", encoding="utf-8", newline="") as f:
            rescue_fields = ["eval_id", "image_id", "rescue_status", "rescue_mat_path", "policy_decision"]
            writer = csv.DictWriter(f, fieldnames=rescue_fields)
            writer.writeheader()
            writer.writerows(rescue_rows)

    summary = {
        "n": len(out_rows),
        "threshold": threshold,
        "accept": sum(r["gate_decision"] == "accept" for r in out_rows),
        "reject": sum(r["gate_decision"] == "reject" for r in out_rows),
        "manual_review": sum(r["gate_decision"] == "manual_review" for r in out_rows),
        "rescue_audit_requested": args.rescue_audit,
        "rescue_audit_rows": len(rescue_rows),
        "threshold_search_performed": False,
        "evaluation": evaluation,
        "scope_note": "gate risk is a diagnostic surrogate, not real ArcFace/L2CS",
    }
    (args.out_dir / "phase21_gate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
