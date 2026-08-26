"""Create paired Phase2 ablation comparisons and decision-aware summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


METRICS = {
    "arcface_cosine": "higher_is_more_identity_preserving",
    "deca_head_pose_norm": "lower_is_more_canonical",
    "deca_pose_delta_vs_original": "larger_is_more_change",
    "l2cs_gaze_delta_vs_original_deg": "lower_is_more_gaze_preserving",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--inference", action="append", default=[], metavar="NAME=CSV")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finite(value: str) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_named_paths(values: list[str]) -> dict[str, Path]:
    result = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Expected NAME=CSV, got {value!r}")
        name, path = value.split("=", 1)
        result[name.strip()] = Path(path)
    return result


def paired_delta(
    rows_by_method: dict[str, dict[str, dict[str, str]]],
    reference: str,
    method: str,
    metric: str,
    seed: int,
    n_boot: int,
) -> dict:
    ref_rows = rows_by_method[reference]
    method_rows = rows_by_method[method]
    ids = sorted(set(ref_rows) & set(method_rows))
    differences = []
    for eval_id in ids:
        ref_value = finite(ref_rows[eval_id].get(metric, ""))
        method_value = finite(method_rows[eval_id].get(metric, ""))
        if ref_value is not None and method_value is not None:
            differences.append(method_value - ref_value)
    values = np.asarray(differences, dtype=np.float64)
    if values.size:
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(0, values.size, size=(n_boot, values.size))
        boot_means = values[sample_indices].mean(axis=1)
        lo, hi = np.percentile(boot_means, [2.5, 97.5])
        mean = float(values.mean())
        median = float(np.median(values))
    else:
        mean = median = lo = hi = float("nan")
    return {
        "reference": reference,
        "method": method,
        "metric": metric,
        "interpretation": METRICS[metric],
        "n_pairs": int(values.size),
        "mean_delta_method_minus_reference": mean,
        "median_delta_method_minus_reference": median,
        "ci95_lo": float(lo),
        "ci95_hi": float(hi),
        "ci_excludes_zero": bool(values.size and (lo > 0 or hi < 0)),
    }


def main() -> None:
    args = parse_args()
    metric_rows = read_csv(args.metrics_csv)
    test_rows = read_csv(args.test_manifest)
    total = len(test_rows)
    eval_id_by_image_id = {row.get("image_id", row["eval_id"]): row["eval_id"] for row in test_rows}

    rows_by_method: dict[str, dict[str, dict[str, str]]] = {}
    for row in metric_rows:
        rows_by_method.setdefault(row["method"], {})[row["eval_id"]] = row

    comparisons = []
    for method in ("full", "no_alpha", "no_augmentation", "no_xgboost"):
        for metric in METRICS:
            comparisons.append(
                paired_delta(rows_by_method, "hard_zero", method, metric, args.seed, args.bootstrap_samples)
            )
    for method in ("no_alpha", "no_augmentation", "no_xgboost"):
        for metric in METRICS:
            comparisons.append(
                paired_delta(rows_by_method, "full", method, metric, args.seed, args.bootstrap_samples)
            )

    decision_rows = []
    for name, path in parse_named_paths(args.inference).items():
        inference_rows = read_csv(path)
        decisions = Counter(row.get("decision", "unknown") or "unknown" for row in inference_rows)
        represented = {eval_id_by_image_id.get(row.get("image_id", ""), "") for row in inference_rows}
        represented.discard("")
        upstream_failure = total - len(represented)
        accepted = decisions["standardize"] + decisions["weak_standardize"]
        decision_rows.append(
            {
                "method": name,
                "n_total": total,
                "n_inference": len(inference_rows),
                "standardize": decisions["standardize"],
                "weak_standardize": decisions["weak_standardize"],
                "reject": decisions["reject"],
                "upstream_failure": upstream_failure,
                "accepted_outputs": accepted,
                "conditional_accept_rate": accepted / len(inference_rows) if inference_rows else float("nan"),
                "end_to_end_accept_rate": accepted / total if total else float("nan"),
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comparison_fields = list(comparisons[0])
    decision_fields = list(decision_rows[0]) if decision_rows else []
    write_csv(args.out_dir / "paired_method_comparisons.csv", comparison_fields, comparisons)
    if decision_rows:
        write_csv(args.out_dir / "decision_aware_summary.csv", decision_fields, decision_rows)
    summary = {
        "n_metric_rows": len(metric_rows),
        "n_unique_metric_pairs": len({(row["eval_id"], row["method"]) for row in metric_rows}),
        "n_test_samples": total,
        "comparisons": comparisons,
        "decisions": decision_rows,
    }
    (args.out_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"comparisons": len(comparisons), "decisions": decision_rows}, indent=2))


if __name__ == "__main__":
    main()
