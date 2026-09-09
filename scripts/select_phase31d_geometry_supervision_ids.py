#!/usr/bin/env python3
"""Select 32 train-only high geometry-delta IDs for Phase3.1d geometry supervision.

The selection is restricted to the train registry, sits in the highest combined
(pose + expression) geometry-delta tier, and is stratified across pose and
expression within that tier to retain variation. It must have zero overlap with
the frozen Phase3.1c validation 32 and the 775 fixed-test IDs.
"""

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


def quartile(rank: float) -> int:
    return min(int(rank * 4), 3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-manifest", required=True, type=Path)
    parser.add_argument("--phase2-manifest", required=True, type=Path)
    parser.add_argument("--train-ids", required=True, type=Path)
    parser.add_argument("--validation-ids", required=True, type=Path)
    parser.add_argument("--fixed-test-ids", required=True, type=Path)
    parser.add_argument("--phase31c-validation-ids", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--count", default=32, type=int)
    args = parser.parse_args()

    train = read_ids(args.train_ids)
    validation = read_ids(args.validation_ids)
    fixed = read_ids(args.fixed_test_ids)
    if train & validation or train & fixed or validation & fixed:
        raise ValueError("Split registries overlap")
    frozen_val = {line.strip() for line in args.phase31c_validation_ids.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
    if not frozen_val or len(frozen_val) != 32:
        raise ValueError("Phase3.1c validation selection must contain exactly 32 IDs")
    if frozen_val - validation:
        raise ValueError("Phase3.1c validation IDs are not in the validation registry")
    if frozen_val & fixed or frozen_val & train:
        raise ValueError("Phase3.1c validation IDs overlap train or fixed test")

    phase1_rows = read_csv(args.phase1_manifest)
    phase2_rows = read_csv(args.phase2_manifest)
    phase1 = {row["image_id"]: row for row in phase1_rows}
    phase2 = {row["image_id"]: row for row in phase2_rows}
    if len(phase1) != len(phase1_rows) or len(phase2) != len(phase2_rows):
        raise ValueError("Duplicate manifest IDs")

    records: list[dict] = []
    failures: list[dict] = []
    for image_id in sorted(train):
        try:
            base = phase1[image_id]
            target_row = phase2[image_id]
            embedding_path = base.get("arcface_embedding_path", "")
            if not embedding_path or not resolve(embedding_path, args.project_root).is_file():
                failures.append({"image_id": image_id, "reason": "missing_arcface_embedding"})
                continue
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
        except Exception as error:  # noqa: BLE001 - explicit per-sample failure, never silent fill
            failures.append({"image_id": image_id, "reason": f"{type(error).__name__}:{error}"})

    if len(records) < args.count:
        raise ValueError("Insufficient valid train targets")

    pose_rank = rank01([row["pose_delta_deg"] for row in records])
    expression_rank = rank01([row["expression_rmse"] for row in records])
    for index, row in enumerate(records):
        row["pose_rank"] = float(pose_rank[index])
        row["expression_rank"] = float(expression_rank[index])
        row["combined_rank"] = float((pose_rank[index] + expression_rank[index]) / 2)
        row["tier"] = "low" if row["combined_rank"] < 1 / 3 else "high" if row["combined_rank"] >= 2 / 3 else "medium"

    high = [row for row in records if row["tier"] == "high"]
    if len(high) < args.count:
        raise ValueError("Insufficient high-tier train candidates")

    # 2D stratification within the high tier: 4 pose-rank bins x 4 expression-rank bins.
    bins: dict[tuple[int, int], list[dict]] = {}
    for row in high:
        bins.setdefault((quartile(row["pose_rank"]), quartile(row["expression_rank"])), []).append(row)
    for key in bins:
        bins[key].sort(key=lambda r: (-r["combined_rank"], r["image_id"]))

    selected: list[dict] = []
    cells = sorted(bins.keys())
    per_cell = 2
    for _round in range(per_cell):
        for key in cells:
            if len(selected) < args.count and bins[key]:
                selected.append(bins[key].pop(0))
    if len(selected) < args.count:
        remaining = [row for key in cells for row in bins[key]]
        remaining.sort(key=lambda r: (-r["combined_rank"], r["image_id"]))
        for row in remaining:
            if len(selected) >= args.count:
                break
            if all(s["image_id"] != row["image_id"] for s in selected):
                selected.append(row)
    if len(selected) != args.count:
        raise RuntimeError("Could not select exactly count high-tier samples")

    selected_ids = {row["image_id"] for row in selected}
    if len(selected_ids) != args.count or selected_ids & fixed or selected_ids & frozen_val or selected_ids & validation or selected_ids - train:
        raise RuntimeError("Selection uniqueness or split/fixed-test isolation failed")

    args.out_dir.mkdir(parents=True, exist_ok=False)
    ids_path = args.out_dir / "geometry_supervision_ids.txt"
    ids_path.write_text("".join(f"{row['image_id']}\n" for row in selected), encoding="utf-8")
    fieldnames = ["image_id", "pose_delta_deg", "expression_rmse", "pose_rank", "expression_rank", "combined_rank", "tier", "phase2_npz"]
    with (args.out_dir / "geometry_supervision_selection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row[k] for k in fieldnames})

    split_dir = args.train_ids.parent
    split_hashes = {n: file_hash(split_dir / n) for n in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")}
    summary = {
        "scope": "Phase3.1d train-only geometry supervision; highest geometry-delta tier with pose+expression stratification; not population representative",
        "count": len(selected),
        "eligible": len(records),
        "high_tier_count": len(high),
        "failures": failures,
        "tier_counts": {tier: sum(1 for row in selected if row["tier"] == tier) for tier in ("high", "medium", "low")},
        "bin_counts": {str(key): sum(1 for row in selected if (quartile(row["pose_rank"]), quartile(row["expression_rank"])) == key) for key in cells},
        "ids_sha256": file_hash(ids_path),
        "phase1_sha256": file_hash(args.phase1_manifest),
        "phase2_sha256": file_hash(args.phase2_manifest),
        "train_ids_sha256": file_hash(args.train_ids),
        "validation_ids_sha256": file_hash(args.validation_ids),
        "fixed_test_ids_sha256": file_hash(args.fixed_test_ids),
        "phase31c_validation_ids_sha256": file_hash(args.phase31c_validation_ids),
        "split_hashes": split_hashes,
        "validation_overlap": len(selected_ids & validation),
        "fixed_test_overlap": len(selected_ids & fixed),
        "phase31c_validation_overlap": len(selected_ids & frozen_val),
        "selected_pose_delta_deg": {"mean": float(np.mean([row["pose_delta_deg"] for row in selected])), "min": min(row["pose_delta_deg"] for row in selected), "max": max(row["pose_delta_deg"] for row in selected)},
        "selected_expression_rmse": {"mean": float(np.mean([row["expression_rmse"] for row in selected])), "min": min(row["expression_rmse"] for row in selected), "max": max(row["expression_rmse"] for row in selected)},
    }
    (args.out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
