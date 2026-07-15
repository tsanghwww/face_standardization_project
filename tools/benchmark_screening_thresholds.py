#!/usr/bin/env python3
"""Benchmark p95 vs p97.5 screening thresholds for Phase2 data cleaning.

The benchmark compares two existing screening runs over the same image IDs.
It focuses on the extra samples removed by the stricter threshold and checks
whether those samples are actually worse under independent Phase2 features and
inference behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


BAD_LABELS = {"Warn", "WARN", "Fail", "FAIL", "Reject", "REJECT"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p95-report", required=True, type=Path, help="screening_p95/screening_report.json")
    parser.add_argument("--p975-report", required=True, type=Path, help="screening_p975/screening_report.json")
    parser.add_argument("--phase2-manifest", required=True, type=Path, help="phase2_real_manifest/manifest.csv")
    parser.add_argument(
        "--inference-manifest",
        action="append",
        nargs=2,
        metavar=("NAME", "CSV"),
        help="Optional Phase2 inference manifest, e.g. stage3 path/to/phase2_inference_manifest.csv",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=80, help="Review sample size per diagnostic group.")
    return parser.parse_args()


def read_json_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        rows = payload.get("rows", payload.get("report", []))
    else:
        rows = payload
    if not isinstance(rows, list):
        raise SystemExit(f"{path} does not contain a row list")
    return [dict(row) for row in rows]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("image_id", "")).strip()


def is_bad(row: dict[str, Any]) -> bool:
    return str(row.get("label", "")).strip() in BAD_LABELS


def as_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0, "mean": float("nan"), "min": float("nan"), "max": float("nan")}
    xs = sorted(values)
    return {
        "n": len(xs),
        "mean": float(mean(xs)),
        "min": float(xs[0]),
        "p50": float(xs[len(xs) // 2]),
        "p90": float(xs[min(len(xs) - 1, int(len(xs) * 0.90))]),
        "max": float(xs[-1]),
    }


def group_stats(ids: set[str], rows: dict[str, dict[str, Any]], columns: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for col in columns:
        out[col] = summarize([as_float(rows[i], col) for i in ids if i in rows])
    return out


def decision_counts(ids: set[str], rows: dict[str, dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for image_id in ids:
        if image_id in rows:
            counts[str(rows[image_id].get("decision", "missing") or "missing")] += 1
    return dict(counts)


def ratio_stats(ids: set[str], rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    specs = {
        "exp_ratio": ("original_exp_norm", "standardized_exp_norm"),
        "head_pose_ratio": ("original_head_pose_norm", "standardized_head_pose_norm"),
        "jaw_pose_ratio": ("original_jaw_pose_norm", "standardized_jaw_pose_norm"),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (orig_key, std_key) in specs.items():
        vals = []
        for image_id in ids:
            row = rows.get(image_id)
            if not row:
                continue
            orig = as_float(row, orig_key)
            std = as_float(row, std_key)
            vals.append(std / max(orig, 1e-8))
        out[name] = summarize(vals)
    return out


def sorted_review_rows(
    group_name: str,
    ids: set[str],
    phase2_rows: dict[str, dict[str, Any]],
    p95_rows: dict[str, dict[str, Any]],
    p975_rows: dict[str, dict[str, Any]],
    limit: int,
) -> list[dict[str, str]]:
    ranked = sorted(ids, key=lambda i: as_float(phase2_rows.get(i, {}), "quality_score"), reverse=False)
    output = []
    for image_id in ranked[:limit]:
        p2 = phase2_rows.get(image_id, {})
        r95 = p95_rows.get(image_id, {})
        r975 = p975_rows.get(image_id, {})
        output.append(
            {
                "group": group_name,
                "image_id": image_id,
                "p95_label": str(r95.get("label", "")),
                "p975_label": str(r975.get("label", "")),
                "p95_D2": str(r95.get("D2", "")),
                "p975_D2": str(r975.get("D2", "")),
                "quality_score": str(p2.get("quality_score", "")),
                "exp_norm": str(p2.get("exp_norm", "")),
                "head_pose_norm": str(p2.get("head_pose_norm", "")),
                "jaw_pose_norm": str(p2.get("jaw_pose_norm", "")),
                "landmark_score": str(p2.get("landmark_score", "")),
                "landmark_out_ratio": str(p2.get("landmark_out_ratio", "")),
            }
        )
    return output


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    p95_rows = {row_id(row): row for row in read_json_rows(args.p95_report) if row_id(row)}
    p975_rows = {row_id(row): row for row in read_json_rows(args.p975_report) if row_id(row)}
    phase2_rows = {row_id(row): row for row in read_csv_rows(args.phase2_manifest) if row_id(row)}
    common_ids = set(p95_rows) & set(p975_rows) & set(phase2_rows)

    p95_warn = {i for i in common_ids if is_bad(p95_rows[i])}
    p975_warn = {i for i in common_ids if is_bad(p975_rows[i])}
    both_warn = p95_warn & p975_warn
    p95_only_warn = p95_warn - p975_warn
    both_pass = common_ids - p95_warn - p975_warn

    feature_columns = [
        "quality_score",
        "exp_norm",
        "head_pose_norm",
        "jaw_pose_norm",
        "landmark_score",
        "landmark_out_ratio",
        "landmark_bbox_area",
        "landmark_center_dist",
    ]
    groups = {
        "both_pass": both_pass,
        "p95_only_warn": p95_only_warn,
        "both_warn": both_warn,
    }
    summary: dict[str, Any] = {
        "inputs": {
            "p95_report": str(args.p95_report),
            "p975_report": str(args.p975_report),
            "phase2_manifest": str(args.phase2_manifest),
        },
        "counts": {
            "common": len(common_ids),
            "p95_warn": len(p95_warn),
            "p975_warn": len(p975_warn),
            "p95_only_warn": len(p95_only_warn),
            "both_warn": len(both_warn),
            "both_pass": len(both_pass),
        },
        "phase2_feature_stats": {
            name: group_stats(ids, phase2_rows, feature_columns) for name, ids in groups.items()
        },
        "screening_score_stats": {
            name: {
                "p95_D2": summarize([as_float(p95_rows[i], "D2") for i in ids if i in p95_rows]),
                "p975_D2": summarize([as_float(p975_rows[i], "D2") for i in ids if i in p975_rows]),
            }
            for name, ids in groups.items()
        },
        "inference": {},
    }

    if args.inference_manifest:
        for name, path_text in args.inference_manifest:
            rows = {row_id(row): row for row in read_csv_rows(Path(path_text)) if row_id(row)}
            summary["inference"][name] = {
                group_name: {
                    "decision_counts": decision_counts(ids, rows),
                    "ratio_stats": ratio_stats(ids, rows),
                    "confidence": summarize([as_float(rows[i], "confidence") for i in ids if i in rows]),
                    "reject_score": summarize([as_float(rows[i], "reject_score") for i in ids if i in rows]),
                }
                for group_name, ids in groups.items()
            }

    q_pass = summary["phase2_feature_stats"]["both_pass"]["quality_score"]["mean"]
    q_extra = summary["phase2_feature_stats"]["p95_only_warn"]["quality_score"]["mean"]
    q_warn = summary["phase2_feature_stats"]["both_warn"]["quality_score"]["mean"]
    recommendation = "inconclusive"
    if math.isfinite(q_extra) and math.isfinite(q_pass) and math.isfinite(q_warn):
        if q_extra < q_pass and q_extra > q_warn:
            recommendation = "p95 is a useful stricter cleanup candidate; validate p95_only_warn visually."
        elif q_extra >= q_pass:
            recommendation = "p97.5 is likely preferable; p95 removes samples that look similar to pass samples."
        elif q_extra <= q_warn:
            recommendation = "p95 is likely preferable; extra removed samples look as bad as the core warn set."
    summary["recommendation_rule"] = recommendation

    (args.out_dir / "screening_threshold_benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    review_rows: list[dict[str, str]] = []
    for name, ids in groups.items():
        review_rows.extend(sorted_review_rows(name, ids, phase2_rows, p95_rows, p975_rows, args.sample_size))
    with (args.out_dir / "screening_threshold_review_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "group",
            "image_id",
            "p95_label",
            "p975_label",
            "p95_D2",
            "p975_D2",
            "quality_score",
            "exp_norm",
            "head_pose_norm",
            "jaw_pose_norm",
            "landmark_score",
            "landmark_out_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(review_rows)

    print(json.dumps({"summary": str(args.out_dir / "screening_threshold_benchmark_summary.json"), "review": str(args.out_dir / "screening_threshold_review_manifest.csv"), "recommendation_rule": recommendation}, indent=2))


if __name__ == "__main__":
    main()
