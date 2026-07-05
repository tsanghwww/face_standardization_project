"""Build a Phase2 training manifest from DECA .mat outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .features import find_deca_mat_files, read_arcface_rows, sample_from_mat, write_json


FIELDS = [
    "image_id",
    "mat_path",
    "quality_score",
    "quality_label",
    "exp_norm",
    "head_pose_norm",
    "jaw_pose_norm",
    "landmark_score",
    "landmark_out_ratio",
    "landmark_bbox_area",
    "landmark_center_dist",
    "arcface_status",
    "arcface_score",
    "arcface_train_flag",
    "use_for_train",
]


def quality_label(value: float) -> str:
    if value >= 0.72:
        return "high"
    if value >= 0.45:
        return "medium"
    return "low"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deca-results-dir", required=True, type=Path)
    parser.add_argument("--arcface-manifest", type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args()

    arcface_rows = read_arcface_rows(args.arcface_manifest)
    mat_files = find_deca_mat_files(args.deca_results_dir)
    if not mat_files:
        raise SystemExit(f"No .mat files found under {args.deca_results_dir}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    counts = {"high": 0, "medium": 0, "low": 0}
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for mat_path in mat_files:
            sample = sample_from_mat(mat_path, arcface_rows.get(mat_path.stem))
            q = sample.metrics["quality_score"]
            label = quality_label(q)
            counts[label] += 1
            writer.writerow(
                {
                    "image_id": sample.image_id,
                    "mat_path": str(sample.mat_path),
                    "quality_score": f"{q:.6f}",
                    "quality_label": label,
                    "exp_norm": f"{sample.metrics['exp_norm']:.6f}",
                    "head_pose_norm": f"{sample.metrics['head_pose_norm']:.6f}",
                    "jaw_pose_norm": f"{sample.metrics['jaw_pose_norm']:.6f}",
                    "landmark_score": f"{sample.metrics['landmark_score']:.6f}",
                    "landmark_out_ratio": f"{sample.metrics['landmark_out_ratio']:.6f}",
                    "landmark_bbox_area": f"{sample.metrics['landmark_bbox_area']:.6f}",
                    "landmark_center_dist": f"{sample.metrics['landmark_center_dist']:.6f}",
                    "arcface_status": f"{sample.metrics['arcface_status']:.0f}",
                    "arcface_score": f"{sample.metrics['arcface_score']:.6f}",
                    "arcface_train_flag": f"{sample.metrics['arcface_train_flag']:.0f}",
                    "use_for_train": "true" if label in {"high", "medium"} else "false",
                }
            )

    write_json(
        args.out_json,
        {
            "deca_results_dir": str(args.deca_results_dir),
            "arcface_manifest": str(args.arcface_manifest) if args.arcface_manifest else None,
            "mat_count": len(mat_files),
            "quality_counts": counts,
            "out_csv": str(args.out_csv),
        },
    )
    print(f"Wrote {args.out_csv} with {len(mat_files)} rows")
    print(counts)


if __name__ == "__main__":
    main()

