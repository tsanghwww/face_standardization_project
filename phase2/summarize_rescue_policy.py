"""Write primary-versus-rescue policy coverage without altering the fixed manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-gate-predictions", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--total", type=int, default=775)
    parser.add_argument("--primary-available", type=int, default=743)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.rescue_gate_predictions.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 32 or len({row["eval_id"] for row in rows}) != 32:
        raise SystemExit(f"Expected 32 unique rescue-only rows, got {len(rows)}")
    technically_recovered = sum(row["pipeline_status"] == "available" for row in rows)
    accepted = sum(row["gate_decision"] != "reject" for row in rows)
    valid = sum(
        row["pipeline_status"] == "available"
        and row["gate_decision"] != "reject"
        and row["outcome"] == "safe_and_effective"
        for row in rows
    )
    policy_rows = [
        {
            "policy": "primary_fan_only", "denominator": args.total,
            "scientifically_usable": args.primary_available, "coverage": args.primary_available / args.total,
            "technical_rescue_recovered": 0, "gate_accepted_rescue": 0, "valid_rescue": 0,
        },
        {
            "policy": "primary_plus_rescue", "denominator": args.total,
            "scientifically_usable": args.primary_available + valid,
            "coverage": (args.primary_available + valid) / args.total,
            "technical_rescue_recovered": technically_recovered,
            "gate_accepted_rescue": accepted, "valid_rescue": valid,
        },
    ]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / "rescue_policy_summary.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(policy_rows[0]))
        writer.writeheader()
        writer.writerows(policy_rows)
    summary = {
        "denominator": args.total, "primary_available": args.primary_available,
        "technical_rescue_recovered": technically_recovered, "gate_accepted_rescue": accepted,
        "valid_rescue": valid, "fallback_coverage": (args.primary_available + valid) / args.total,
        "validity_rule": "available AND gate accepted AND outcome=safe_and_effective",
        "primary_manifest_modified": False,
    }
    (args.out_dir / "rescue_policy_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
