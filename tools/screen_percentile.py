"""Quality-score screening at p95 / p97.5 percentiles.

Computes per-sample quality_score from DECA .mat + kpt files,
then flags the bottom 5% (p95) / bottom 2.5% (p97.5) as WARN.
"""

from __future__ import annotations

import argparse
import csv
import math
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_mat(path: Path) -> dict[str, Any]:
    from scipy.io import loadmat
    return {k: v for k, v in loadmat(path).items() if not k.startswith("__")}


def load_kpt(path: Path) -> np.ndarray | None:
    try:
        arr = np.loadtxt(path, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim != 2 or arr.shape[0] < 5 or arr.shape[1] < 2:
        return None
    if not np.isfinite(arr).all():
        return None
    return arr


def quality_from_kpt(kpt: np.ndarray, image_size: float = 224.0) -> dict[str, float]:
    xy = kpt[:, :2]
    out = (xy[:, 0] < 0) | (xy[:, 0] > image_size) | (xy[:, 1] < 0) | (xy[:, 1] > image_size)
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    wh = np.maximum(maxs - mins, 0)
    area = float((wh[0] * wh[1]) / (image_size * image_size))
    center = (mins + maxs) * 0.5
    center_dist = float(np.linalg.norm(center - np.array([112.0, 112.0])) / image_size)
    out_ratio = float(out.mean())
    area_score = max(0.0, min(1.0, (area - 0.05) / 0.25))
    center_score = max(0.0, 1.0 - center_dist / 0.45)
    out_score = max(0.0, 1.0 - out_ratio / 0.30)
    landmark_score = float(0.45 * area_score + 0.35 * center_score + 0.20 * out_score)
    return {
        "landmark_out_ratio": out_ratio,
        "landmark_bbox_area": area,
        "landmark_center_dist": center_dist,
        "landmark_score": landmark_score,
    }


def compute_quality(mat_path: Path) -> dict[str, float]:
    data = load_mat(mat_path)
    expression = np.asarray(data.get("expression", np.zeros(50)), dtype=np.float32).flatten()[:50]
    pose = np.asarray(data.get("pose", np.zeros(6)), dtype=np.float32).flatten()[:6]

    kpt_path = mat_path.with_name(f"{mat_path.stem}_kpt2d.txt")
    kpt = load_kpt(kpt_path)
    if kpt is not None:
        kpt_q = quality_from_kpt(kpt)
    else:
        kpt_q = {"landmark_score": 0.0, "landmark_out_ratio": 1.0, "landmark_bbox_area": 0.0, "landmark_center_dist": 1.0}

    exp_norm = float(np.linalg.norm(expression) / math.sqrt(max(expression.size, 1)))
    head_pose_norm = float(np.linalg.norm(pose[:3]))
    jaw_pose_norm = float(np.linalg.norm(pose[3:]))

    pose_score = max(0.0, 1.0 - head_pose_norm / 0.9)
    exp_score = max(0.0, 1.0 - exp_norm / 0.22)
    jaw_score = max(0.0, 1.0 - jaw_pose_norm / 0.55)

    has_shape = "shape" in data
    has_tex = "tex" in data or "detail" in data
    has_verts = "verts" in data
    completeness = (int(has_shape) + int(has_tex) + int(has_verts) + 1 + 1) / 5.0  # +expression +pose

    quality = (
        0.32 * kpt_q["landmark_score"]
        + 0.22 * pose_score
        + 0.16 * exp_score
        + 0.10 * jaw_score
        + 0.12 * 0.65  # no arcface -> default arc_score
        + 0.08 * completeness
    )
    quality = float(max(0.0, min(1.0, quality)))

    return {
        "quality_score": quality,
        "exp_norm": exp_norm,
        "head_pose_norm": head_pose_norm,
        "jaw_pose_norm": jaw_pose_norm,
        "pose_score": pose_score,
        "exp_score": exp_score,
        "jaw_score": jaw_score,
        "landmark_score": kpt_q["landmark_score"],
        "landmark_out_ratio": kpt_q["landmark_out_ratio"],
        "has_mat_issue": 0.0 if mat_path.stat().st_size > 50000 else 1.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--out-csv", default="screening.csv", type=Path)
    parser.add_argument("--out-summary", default="screening_summary.json", type=Path)
    args = parser.parse_args()

    mat_files = sorted(args.results_dir.glob("*/*.mat"))
    print(f"Found {len(mat_files)} .mat files")

    rows = []
    for p in mat_files:
        image_id = p.stem
        try:
            q = compute_quality(p)
        except Exception as e:
            q = {"quality_score": 0.0, "has_mat_issue": 1.0, "error": str(e)}
        q["image_id"] = image_id
        q["mat_path"] = str(p)
        rows.append(q)

    # sort by quality_score ascending (worst first)
    rows.sort(key=lambda r: r["quality_score"])

    scores = [r["quality_score"] for r in rows]
    p2_5 = np.percentile(scores, 2.5)   # p97.5 cutoff: bottom 2.5%
    p5 = np.percentile(scores, 5)        # p95 cutoff: bottom 5%

    print(f"Quality range: [{scores[0]:.4f}, {scores[-1]:.4f}]")
    print(f"p2.5 cutoff: {p2_5:.4f}  p5 cutoff: {p5:.4f}")

    # classify
    for r in rows:
        s = r["quality_score"]
        if s <= p2_5:
            r["label_p975"] = "WARN"
        else:
            r["label_p975"] = "PASS"

        if s <= p5:
            r["label_p95"] = "WARN"
        else:
            r["label_p95"] = "PASS"

    fieldnames = ["image_id", "quality_score", "label_p975", "label_p95", "head_pose_norm",
                  "exp_norm", "jaw_pose_norm", "landmark_score", "landmark_out_ratio",
                  "pose_score", "exp_score", "jaw_score", "has_mat_issue", "mat_path"]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in sorted(rows, key=lambda x: int(x["image_id"])):
            writer.writerow(r)

    cnt_p975 = {"PASS": 0, "WARN": 0}
    cnt_p95 = {"PASS": 0, "WARN": 0}
    for r in rows:
        cnt_p975[r["label_p975"]] += 1
        cnt_p95[r["label_p95"]] += 1

    summary = {
        "total": len(rows),
        "quality_min": float(scores[0]),
        "quality_max": float(scores[-1]),
        "quality_mean": float(np.mean(scores)),
        "quality_std": float(np.std(scores)),
        "p2_5_cutoff": float(p2_5),
        "p5_cutoff": float(p5),
        "p975": cnt_p975,
        "p95": cnt_p95,
    }

    with args.out_summary.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"p97.5: PASS={cnt_p975['PASS']} WARN={cnt_p975['WARN']}")
    print(f"p95:   PASS={cnt_p95['PASS']} WARN={cnt_p95['WARN']}")
    print(f"CSV: {args.out_csv}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
