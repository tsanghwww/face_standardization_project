"""Compare hard-zero and Phase2 standardization manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


NORM_COLUMNS = [
    ("exp", "original_exp_norm", "standardized_exp_norm"),
    ("head_pose", "original_head_pose_norm", "standardized_head_pose_norm"),
    ("jaw_pose", "original_jaw_pose_norm", "standardized_jaw_pose_norm"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", nargs=2, metavar=("NAME", "MANIFEST"), required=True)
    parser.add_argument("--xgb-manifest", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, ""))
    except ValueError:
        return 0.0
    return value if math.isfinite(value) else 0.0


def summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p90": float("nan"), "max": float("nan")}
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def summarize_run(rows: list[dict[str, str]], xgb_labels: dict[str, str], include_by_xgb: bool = True) -> dict[str, object]:
    decision_counts: dict[str, int] = {}
    xgb_counts: dict[str, int] = {}
    norm_summary: dict[str, object] = {}
    for row in rows:
        decision = row.get("decision", "hard_zero") or "hard_zero"
        decision_counts[decision] = decision_counts.get(decision, 0) + 1
        label = xgb_labels.get(row.get("image_id", ""), "unknown")
        xgb_counts[label] = xgb_counts.get(label, 0) + 1

    for prefix, original_key, standardized_key in NORM_COLUMNS:
        original = [as_float(row, original_key) for row in rows]
        standardized = [as_float(row, standardized_key) for row in rows]
        ratios = [s / max(o, 1e-8) for o, s in zip(original, standardized)]
        norm_summary[prefix] = {
            "original": summarize_values(original),
            "standardized": summarize_values(standardized),
            "ratio": summarize_values(ratios),
        }

    by_xgb: dict[str, object] = {}
    if include_by_xgb:
        for label in ["high", "medium", "low", "unknown"]:
            subset = [row for row in rows if xgb_labels.get(row.get("image_id", ""), "unknown") == label]
            if not subset:
                continue
            by_xgb[label] = summarize_run(subset, xgb_labels, include_by_xgb=False)

    return {
        "count": len(rows),
        "decision_counts": decision_counts,
        "xgb_counts": xgb_counts,
        "norms": norm_summary,
        "by_xgb": by_xgb,
    }


def flatten_summary(name: str, summary: dict[str, object]) -> dict[str, str]:
    norms = summary["norms"]  # type: ignore[index]
    row = {
        "run": name,
        "count": str(summary["count"]),
        "decision_counts": json.dumps(summary["decision_counts"], sort_keys=True),
        "xgb_counts": json.dumps(summary["xgb_counts"], sort_keys=True),
    }
    for prefix in ["exp", "head_pose", "jaw_pose"]:
        item = norms[prefix]  # type: ignore[index]
        row[f"{prefix}_orig_mean"] = f"{item['original']['mean']:.8f}"
        row[f"{prefix}_std_mean"] = f"{item['standardized']['mean']:.8f}"
        row[f"{prefix}_ratio_mean"] = f"{item['ratio']['mean']:.8f}"
        row[f"{prefix}_ratio_p90"] = f"{item['ratio']['p90']:.8f}"
    return row


def main() -> None:
    args = parse_args()
    xgb_labels: dict[str, str] = {}
    if args.xgb_manifest:
        for row in read_csv(args.xgb_manifest):
            xgb_labels[row["image_id"]] = row.get("xgb_quality_label", "unknown")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    flat_rows = []
    for name, manifest in args.run:
        rows = read_csv(Path(manifest))
        summary = summarize_run(rows, xgb_labels)
        summaries[name] = summary
        flat_rows.append(flatten_summary(name, summary))

    summary_path = args.out_dir / "standardization_comparison_summary.json"
    table_path = args.out_dir / "standardization_comparison_table.csv"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with table_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({"summary": str(summary_path), "table": str(table_path), "runs": list(summaries)}, indent=2))


if __name__ == "__main__":
    main()
