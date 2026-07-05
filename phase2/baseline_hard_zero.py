"""Create the Phase2 baseline: expression=0 and pose=0 for every DECA .mat."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .features import find_deca_mat_files, sample_from_mat


FIELDS = [
    "image_id",
    "mat_path",
    "out_npz",
    "quality_score",
    "original_exp_norm",
    "original_head_pose_norm",
    "original_jaw_pose_norm",
    "standardized_exp_norm",
    "standardized_head_pose_norm",
    "standardized_jaw_pose_norm",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deca-results-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    mats = find_deca_mat_files(args.deca_results_dir)
    if not mats:
        raise SystemExit(f"No .mat files found under {args.deca_results_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    params_dir = args.out_dir / "params"
    params_dir.mkdir(exist_ok=True)
    rows = []
    for mat_path in mats:
        sample = sample_from_mat(mat_path)
        std_exp = np.zeros_like(sample.params["expression"], dtype=np.float32)
        std_pose = np.zeros_like(sample.params["pose"], dtype=np.float32)
        out_npz = params_dir / f"{sample.image_id}_hard_zero.npz"
        np.savez(
            out_npz,
            image_id=sample.image_id,
            source_mat=str(sample.mat_path),
            expression_original=sample.params["expression"],
            pose_original=sample.params["pose"],
            expression_standardized=std_exp,
            pose_standardized=std_pose,
            alpha_expression=np.float32(1.0),
            alpha_head_pose=np.float32(1.0),
            alpha_jaw_pose=np.float32(1.0),
            confidence=np.float32(1.0),
            reject_score=np.float32(0.0),
        )
        rows.append(
            {
                "image_id": sample.image_id,
                "mat_path": str(sample.mat_path),
                "out_npz": str(out_npz),
                "quality_score": f"{sample.metrics['quality_score']:.6f}",
                "original_exp_norm": f"{sample.metrics['exp_norm']:.6f}",
                "original_head_pose_norm": f"{sample.metrics['head_pose_norm']:.6f}",
                "original_jaw_pose_norm": f"{sample.metrics['jaw_pose_norm']:.6f}",
                "standardized_exp_norm": "0.000000",
                "standardized_head_pose_norm": "0.000000",
                "standardized_jaw_pose_norm": "0.000000",
            }
        )
    manifest = args.out_dir / "hard_zero_manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {"count": len(rows), "manifest": str(manifest), "params_dir": str(params_dir)}
    (args.out_dir / "hard_zero_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

