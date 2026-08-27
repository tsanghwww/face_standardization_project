"""Prepare the 32 FAN failures for isolated rescue inference and evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import xgboost as xgb

from .features import sample_from_mat
from .rebuild_xgboost import FEATURE_COLUMNS, quality_label


PROJECT = Path(r"D:\face_standardization_project")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-manifest", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv")
    parser.add_argument("--failure-manifest", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "deca_params" / "deca_failures.csv")
    parser.add_argument("--rescue-dir", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "deca_params_rescue")
    parser.add_argument("--arcface-manifest", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "arcface_fixed_test_manifest.csv")
    parser.add_argument("--xgb-model", type=Path, default=PROJECT / "results" / "phase2_xgb_rebuilt_20260824" / "xgb_final_model.json")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results" / "phase2_rescue_only_20260826")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    fixed_rows = read_csv(args.fixed_manifest)
    fixed_by_eval = {row["eval_id"]: row for row in fixed_rows}
    failure_ids = [row["eval_id"] for row in read_csv(args.failure_manifest)]
    arcface_rows = {row["image_id"]: row for row in read_csv(args.arcface_manifest)}
    if len(failure_ids) != 32 or len(set(failure_ids)) != 32:
        raise SystemExit(f"Expected 32 unique FAN failures, got {len(failure_ids)}/{len(set(failure_ids))}")

    booster = xgb.Booster()
    booster.load_model(args.xgb_model)
    mats_dir = args.out_dir / "mats"
    mats_dir.mkdir(parents=True, exist_ok=True)
    rescue_rows = []
    xgb_rows = []
    failures = []
    for eval_id in failure_ids:
        source = fixed_by_eval[eval_id]
        image_id = source["image_id"]
        source_dir = args.rescue_dir / eval_id
        source_mat = source_dir / f"{image_id}.mat"
        target_dir = mats_dir / image_id
        target_dir.mkdir(parents=True, exist_ok=True)
        target_mat = target_dir / f"{image_id}.mat"
        try:
            shutil.copy2(source_mat, target_mat)
            for suffix in ("_kpt2d.txt", "_kpt3d.txt"):
                shutil.copy2(source_dir / f"{image_id}{suffix}", target_dir / f"{image_id}{suffix}")
            sample = sample_from_mat(target_mat, arcface_rows.get(image_id))
            features = np.asarray([[float(sample.metrics[name]) for name in FEATURE_COLUMNS]], dtype=np.float32)
            score = float(booster.predict(xgb.DMatrix(features, feature_names=FEATURE_COLUMNS))[0])
            rescue_row = dict(source)
            rescue_row["mat_path"] = str(target_mat)
            rescue_row["xgb_quality_score"] = f"{score:.8f}"
            rescue_row["xgb_quality_label"] = quality_label(score)
            rescue_row["preprocess_source"] = "whole_image_rescue"
            rescue_rows.append(rescue_row)
            xgb_rows.append({
                "image_id": image_id,
                "eval_id": eval_id,
                "source_group": source["source_group"],
                "xgb_quality_score": f"{score:.8f}",
                "xgb_quality_label": quality_label(score),
                "feature_coverage": "full",
                "failure_reason": "",
                "xgb_input_provenance": "rescue",
            })
        except Exception as exc:  # noqa: BLE001
            failures.append({"eval_id": eval_id, "image_id": image_id, "failure_reason": f"{type(exc).__name__}:{exc}"})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_fields = list(fixed_rows[0]) + ["preprocess_source"]
    with (args.out_dir / "rescue_only_test_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rescue_rows)
    with (args.out_dir / "rescue_xgb_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["image_id", "eval_id", "source_group", "xgb_quality_score", "xgb_quality_label", "feature_coverage", "failure_reason", "xgb_input_provenance"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(xgb_rows)
    with (args.out_dir / "rescue_preparation_failures.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["eval_id", "image_id", "failure_reason"])
        writer.writeheader()
        writer.writerows(failures)
    (args.out_dir / "rescue_only_ids.txt").write_text("\n".join(row["image_id"] for row in rescue_rows) + "\n", encoding="utf-8")
    summary = {
        "expected": 32,
        "prepared": len(rescue_rows),
        "failed": len(failures),
        "mats_dir": str(mats_dir),
        "manifest": str(args.out_dir / "rescue_only_test_manifest.csv"),
        "xgb_predictions": str(args.out_dir / "rescue_xgb_predictions.csv"),
        "xgb_input_provenance": "rescue",
    }
    (args.out_dir / "rescue_preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
