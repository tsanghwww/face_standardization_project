#!/usr/bin/env python3
"""Select a deterministic pose-stratified validation subset for coordinate audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def select_pose_stratified(rows: list[dict], count: int) -> tuple[list[dict], dict[str, list[str]]]:
    """Deterministic pose-stratified selection (reused by the VAE audit)."""
    for row in rows:
        pose = [float(row[f"pose_{axis}"]) for axis in "xyz"]
        row["_pose_norm"] = math.sqrt(sum(value * value for value in pose))
    rankings = [
        ("pose_norm_high", lambda row: float(row["_pose_norm"]), True),
        ("pose_x_low", lambda row: float(row["pose_x"]), False),
        ("pose_x_high", lambda row: float(row["pose_x"]), True),
        ("pose_y_low", lambda row: float(row["pose_y"]), False),
        ("pose_y_high", lambda row: float(row["pose_y"]), True),
        ("pose_z_low", lambda row: float(row["pose_z"]), False),
        ("pose_z_high", lambda row: float(row["pose_z"]), True),
    ]
    selected: list[dict] = []
    reasons: dict[str, list[str]] = {}
    depth = 0
    while len(selected) < count:
        made_progress = False
        for reason, key, reverse in rankings:
            ranked = sorted(rows, key=key, reverse=reverse)
            if depth >= len(ranked):
                continue
            row = ranked[depth]
            image_id = row["image_id"]
            reasons.setdefault(image_id, []).append(reason)
            if all(existing["image_id"] != image_id for existing in selected):
                selected.append(row)
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress and depth >= len(rows):
            break
        depth += 1
    return selected, reasons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaze-manifest", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    with args.gaze_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == args.split and row["status"] == "candidate_unvalidated"]
    if len(rows) < args.count:
        raise SystemExit(f"Only {len(rows)} eligible rows for requested count={args.count}")

    selected, reasons = select_pose_stratified(rows, args.count)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ids = [row["image_id"] for row in selected]
    (args.out_dir / "coordinate_audit_ids.txt").write_text("".join(f"{value}\n" for value in ids), encoding="utf-8")
    summary = {
        "split": args.split,
        "count": len(ids),
        "selection": [
            {
                "image_id": row["image_id"],
                "pose": [float(row[f"pose_{axis}"]) for axis in "xyz"],
                "pose_norm": float(row["_pose_norm"]),
                "reasons": reasons[row["image_id"]],
            }
            for row in selected
        ],
    }
    (args.out_dir / "coordinate_audit_selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
