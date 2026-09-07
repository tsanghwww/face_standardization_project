#!/usr/bin/env python3
"""Select validation IDs stratified by actual Phase2 source-to-target change."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from phase3.reconstruction_data import file_hash, read_ids


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve(path: str, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def pose_delta_deg(source, target) -> float:
    left = Rotation.from_rotvec(np.asarray(source, dtype=np.float64).reshape(-1)[:3])
    right = Rotation.from_rotvec(np.asarray(target, dtype=np.float64).reshape(-1)[:3])
    return float(np.degrees((right * left.inv()).magnitude()))


def rank01(values: list[float]) -> np.ndarray:
    order = np.argsort(np.asarray(values), kind="stable")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.linspace(0.0, 1.0, len(values)) if len(values) > 1 else 0.5
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-manifest", required=True, type=Path)
    parser.add_argument("--phase2-manifest", required=True, type=Path)
    parser.add_argument("--validation-ids", required=True, type=Path)
    parser.add_argument("--fixed-test-ids", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--count", default=32, type=int)
    args = parser.parse_args()
    validation, fixed = read_ids(args.validation_ids), read_ids(args.fixed_test_ids)
    if validation & fixed:
        raise ValueError("Validation overlaps fixed test")
    phase1_rows, phase2_rows = read_csv(args.phase1_manifest), read_csv(args.phase2_manifest)
    phase1 = {row["image_id"]: row for row in phase1_rows}
    phase2 = {row["image_id"]: row for row in phase2_rows}
    if len(phase1) != len(phase1_rows) or len(phase2) != len(phase2_rows):
        raise ValueError("Duplicate manifest IDs")
    records, failures = [], []
    for image_id in sorted(validation):
        try:
            base, target_row = phase1[image_id], phase2[image_id]
            mat_path = resolve(base["deca_mat_path"], args.project_root)
            npz_path = resolve(target_row["out_npz"], args.project_root)
            mat = loadmat(mat_path)
            source_pose = np.asarray(mat["pose"], dtype=np.float32).reshape(-1)
            source_expression = np.asarray(mat["expression"], dtype=np.float32).reshape(-1)
            with np.load(npz_path, allow_pickle=False) as values:
                target_pose = np.asarray(values["pose_standardized"], dtype=np.float32).reshape(-1)
                target_expression = np.asarray(values["expression_standardized"], dtype=np.float32).reshape(-1)
            if source_pose.shape != (6,) or target_pose.shape != (6,) or source_expression.shape != (50,) or target_expression.shape != (50,):
                raise ValueError("invalid_parameter_dimensions")
            record = {
                "image_id": image_id,
                "pose_delta_deg": pose_delta_deg(source_pose, target_pose),
                "expression_rmse": float(np.sqrt(np.mean(np.square(source_expression - target_expression)))),
                "phase2_npz": str(npz_path),
            }
            if not all(np.isfinite(record[key]) for key in ("pose_delta_deg", "expression_rmse")):
                raise ValueError("nonfinite_delta")
            records.append(record)
        except Exception as error:
            failures.append({"image_id": image_id, "reason": f"{type(error).__name__}:{error}"})
    if args.count < 3 or len(records) < args.count:
        raise ValueError("Insufficient valid validation targets")
    pose_rank = rank01([row["pose_delta_deg"] for row in records])
    expression_rank = rank01([row["expression_rmse"] for row in records])
    for index, row in enumerate(records):
        row["combined_rank"] = float((pose_rank[index] + expression_rank[index]) / 2)
        row["tier"] = "low" if row["combined_rank"] < 1 / 3 else "high" if row["combined_rank"] >= 2 / 3 else "medium"
    quotas = {"high": args.count // 2, "medium": args.count // 4}
    quotas["low"] = args.count - quotas["high"] - quotas["medium"]
    selected = []
    for tier in ("high", "medium", "low"):
        candidates = [row for row in records if row["tier"] == tier]
        if tier == "high":
            candidates.sort(key=lambda row: (-row["combined_rank"], row["image_id"]))
        elif tier == "low":
            candidates.sort(key=lambda row: (row["combined_rank"], row["image_id"]))
        else:
            candidates.sort(key=lambda row: (abs(row["combined_rank"] - 0.5), row["image_id"]))
        if len(candidates) < quotas[tier]:
            raise ValueError(f"Insufficient {tier} geometry-delta candidates")
        selected.extend(candidates[:quotas[tier]])
    if len({row["image_id"] for row in selected}) != args.count or {row["image_id"] for row in selected} & fixed:
        raise RuntimeError("Selection uniqueness or fixed-test isolation failed")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    ids_path = args.out_dir / "geometry_audit_ids.txt"
    ids_path.write_text("".join(f"{row['image_id']}\n" for row in selected), encoding="utf-8")
    with (args.out_dir / "geometry_audit_selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "scope": "validation model-selection diagnostic; geometry-delta stratified and not population representative",
        "count": len(selected), "eligible": len(records), "failures": failures, "quotas": quotas,
        "tier_counts": {tier: sum(row["tier"] == tier for row in selected) for tier in quotas},
        "ids_sha256": file_hash(ids_path), "phase1_sha256": file_hash(args.phase1_manifest),
        "phase2_sha256": file_hash(args.phase2_manifest), "validation_ids_sha256": file_hash(args.validation_ids),
        "fixed_test_ids_sha256": file_hash(args.fixed_test_ids), "fixed_test_overlap": 0,
    }
    (args.out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
