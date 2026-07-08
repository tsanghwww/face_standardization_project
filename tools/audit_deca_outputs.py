#!/usr/bin/env python3
"""Audit DECA batch outputs and classify each sample as PASS/WARN/FAIL.

This script is read-only for DECA outputs. It checks file completeness, file
sizes, image statistics, keypoint validity, mesh vertices, and MAT readability.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy.io import loadmat


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class Thresholds:
    vis_min_bytes: int = 20 * 1024
    depth_min_bytes: int = 2 * 1024
    normals_min_bytes: int = 10 * 1024
    texture_min_bytes: int = 10 * 1024
    mat_min_bytes: int = 50 * 1024
    obj_min_bytes: int = 100 * 1024
    detail_obj_min_bytes: int = 1 * 1024 * 1024
    depth_warn_std: float = 5.0
    depth_fail_std: float = 2.0
    vis_warn_std: float = 10.0
    vis_fail_std: float = 3.0
    image_dark_mean: float = 15.0
    image_bright_mean: float = 240.0
    normals_warn_std: float = 5.0
    texture_warn_std: float = 5.0
    landmark_min_points: int = 68
    landmark_warn_out_of_range_ratio: float = 0.30
    landmark_warn_bbox_px: float = 40.0
    landmark_warn_center_distance_px: float = 80.0
    landmark_warn_min_area_ratio: float = 0.05
    landmark_warn_max_area_ratio: float = 0.85
    obj_warn_vertices: int = 1000
    detail_obj_warn_vertices: int = 10000
    mesh_warn_abs_coord: float = 100.0


def iter_images(input_dir: Path, recursive: bool) -> list[Path]:
    if input_dir.is_file():
        if input_dir.suffix.lower() in IMAGE_EXTS:
            return [input_dir]
        raise ValueError(f"--input-dir points to a non-image file: {input_dir}")
    pattern = "**/*" if recursive else "*"
    return sorted(
        p for p in input_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def read_input_list(path: Path) -> list[Path]:
    images = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line)
            if p.suffix.lower() in IMAGE_EXTS:
                images.append(p)
    return images


def file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return None


def image_stats(path: Path) -> tuple[float, float, tuple[int, int]] | None:
    try:
        with Image.open(path) as im:
            arr = np.asarray(im.convert("L"), dtype=np.float32)
            return float(arr.mean()), float(arr.std()), im.size
    except Exception:
        return None


def check_size(path: Path, min_bytes: int, label: str, reasons: list[str], fail_missing: bool = True) -> None:
    size = file_size(path)
    if size is None:
        reasons.append(("FAIL:" if fail_missing else "WARN:") + f"missing_{label}")
    elif size == 0:
        reasons.append(f"FAIL:empty_{label}")
    elif size < min_bytes:
        reasons.append(f"WARN:{label}_too_small:{size}")


def check_image(path: Path, label: str, reasons: list[str], warn_std: float, fail_std: float | None = None) -> None:
    stats = image_stats(path)
    if stats is None:
        reasons.append(f"FAIL:cannot_read_{label}")
        return
    mean, std, size = stats
    if size[0] <= 0 or size[1] <= 0:
        reasons.append(f"FAIL:bad_{label}_size")
    if mean < Thresholds.image_dark_mean:
        reasons.append(f"WARN:{label}_too_dark:{mean:.2f}")
    if mean > Thresholds.image_bright_mean:
        reasons.append(f"WARN:{label}_too_bright:{mean:.2f}")
    if fail_std is not None and std < fail_std:
        reasons.append(f"FAIL:{label}_std_too_low:{std:.2f}")
    elif std < warn_std:
        reasons.append(f"WARN:{label}_std_low:{std:.2f}")


def load_kpt(path: Path, label: str, reasons: list[str]) -> np.ndarray | None:
    try:
        arr = np.loadtxt(path)
    except Exception:
        reasons.append(f"FAIL:cannot_read_{label}")
        return None
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        reasons.append(f"FAIL:bad_{label}_shape:{arr.shape}")
        return None
    if arr.shape[0] < Thresholds.landmark_min_points:
        reasons.append(f"WARN:{label}_too_few_points:{arr.shape[0]}")
    if not np.isfinite(arr).all():
        reasons.append(f"FAIL:{label}_has_nan_or_inf")
        return None
    return arr


def check_kpts(kpt2d_path: Path, kpt3d_path: Path, image_size: int, reasons: list[str], t: Thresholds, kpt_normalized: bool = False) -> None:
    k2 = load_kpt(kpt2d_path, "kpt2d", reasons)
    _ = load_kpt(kpt3d_path, "kpt3d", reasons)
    if k2 is None:
        return

    xy = k2[:, :2]

    if kpt_normalized:
        # DECA-style normalized coords [-1, 1] -> scale to [0, image_size]
        # Normalized coords typically center at 0, so shift by 0.5 and scale
        xy_scaled = (xy + 1.0) * (image_size / 2.0)
        img_center = np.array([image_size * 0.5, image_size * 0.5])
    else:
        xy_scaled = xy
        img_center = np.array([image_size * 0.5, image_size * 0.5])

    out = (
        (xy_scaled[:, 0] < 0) | (xy_scaled[:, 0] > image_size) |
        (xy_scaled[:, 1] < 0) | (xy_scaled[:, 1] > image_size)
    )
    out_ratio = float(out.mean())
    if out_ratio > t.landmark_warn_out_of_range_ratio:
        reasons.append(f"WARN:kpt2d_out_of_range_ratio:{out_ratio:.3f}")

    mins = xy_scaled.min(axis=0)
    maxs = xy_scaled.max(axis=0)
    bbox_wh = maxs - mins
    bbox_w, bbox_h = float(bbox_wh[0]), float(bbox_wh[1])
    if bbox_w < t.landmark_warn_bbox_px or bbox_h < t.landmark_warn_bbox_px:
        reasons.append(f"WARN:kpt2d_bbox_too_small:{bbox_w:.1f}x{bbox_h:.1f}")

    center = (mins + maxs) * 0.5
    center_dist = float(np.linalg.norm(center - img_center))
    if center_dist > t.landmark_warn_center_distance_px:
        reasons.append(f"WARN:kpt2d_center_far:{center_dist:.1f}")

    area_ratio = float((bbox_w * bbox_h) / (image_size * image_size))
    if area_ratio < t.landmark_warn_min_area_ratio:
        reasons.append(f"WARN:kpt2d_bbox_area_low:{area_ratio:.3f}")
    if area_ratio > t.landmark_warn_max_area_ratio:
        reasons.append(f"WARN:kpt2d_bbox_area_high:{area_ratio:.3f}")


def mesh_vertex_stats(path: Path) -> tuple[int, float, bool] | None:
    vertices = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("v "):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    except Exception:
        return None
    if not vertices:
        return 0, math.inf, False
    arr = np.asarray(vertices, dtype=np.float64)
    finite = bool(np.isfinite(arr).all())
    max_abs = float(np.nanmax(np.abs(arr))) if arr.size else math.inf
    return int(arr.shape[0]), max_abs, finite


def check_mesh(path: Path, label: str, min_vertices: int, reasons: list[str], t: Thresholds) -> None:
    stats = mesh_vertex_stats(path)
    if stats is None:
        reasons.append(f"FAIL:cannot_read_{label}")
        return
    count, max_abs, finite = stats
    if count == 0:
        reasons.append(f"FAIL:{label}_no_vertices")
        return
    if count < min_vertices:
        reasons.append(f"WARN:{label}_few_vertices:{count}")
    if not finite:
        reasons.append(f"FAIL:{label}_has_nan_or_inf")
    if max_abs > t.mesh_warn_abs_coord:
        reasons.append(f"WARN:{label}_coord_range_large:{max_abs:.2f}")


def check_mat(path: Path, reasons: list[str]) -> None:
    try:
        data = loadmat(path)
    except Exception:
        reasons.append("FAIL:cannot_read_mat")
        return
    keys = {k for k in data if not k.startswith("__")}
    for key in ["verts", "trans_verts", "landmarks2d", "landmarks3d"]:
        if key not in keys:
            reasons.append(f"WARN:mat_missing_{key}")


def classify(reasons: Iterable[str]) -> str:
    reasons = list(reasons)
    if any(r.startswith("FAIL:") for r in reasons):
        return "FAIL"
    if any(r.startswith("WARN:") for r in reasons):
        return "WARN"
    return "PASS"


def expected_paths(results_dir: Path, stem: str) -> dict[str, Path]:
    sample_dir = results_dir / stem
    return {
        "vis": results_dir / f"{stem}_vis.jpg",
        "depth": sample_dir / f"{stem}_depth.jpg",
        "obj": sample_dir / f"{stem}.obj",
        "detail_obj": sample_dir / f"{stem}_detail.obj",
        "mtl": sample_dir / f"{stem}.mtl",
        "texture": sample_dir / f"{stem}.png",
        "normals": sample_dir / f"{stem}_normals.png",
        "kpt2d": sample_dir / f"{stem}_kpt2d.txt",
        "kpt3d": sample_dir / f"{stem}_kpt3d.txt",
        "mat": sample_dir / f"{stem}.mat",
    }


def audit_one(input_path: Path, results_dir: Path, profile: str, image_size: int, t: Thresholds, kpt_normalized: bool = False) -> dict[str, str]:
    stem = input_path.stem
    paths = expected_paths(results_dir, stem)
    reasons: list[str] = []

    if profile in {"full", "light"}:
        check_size(paths["vis"], t.vis_min_bytes, "vis", reasons)
        check_size(paths["depth"], t.depth_min_bytes, "depth", reasons)
        if paths["vis"].exists():
            check_image(paths["vis"], "vis", reasons, t.vis_warn_std, t.vis_fail_std)
        if paths["depth"].exists():
            check_image(paths["depth"], "depth", reasons, t.depth_warn_std, t.depth_fail_std)

    if profile == "full":
        check_size(paths["obj"], t.obj_min_bytes, "obj", reasons)
        check_size(paths["detail_obj"], t.detail_obj_min_bytes, "detail_obj", reasons)
        check_size(paths["texture"], t.texture_min_bytes, "texture", reasons, fail_missing=False)
        check_size(paths["normals"], t.normals_min_bytes, "normals", reasons, fail_missing=False)
        if paths["obj"].exists():
            check_mesh(paths["obj"], "obj", t.obj_warn_vertices, reasons, t)
        if paths["detail_obj"].exists():
            check_mesh(paths["detail_obj"], "detail_obj", t.detail_obj_warn_vertices, reasons, t)
        if paths["texture"].exists():
            check_image(paths["texture"], "texture", reasons, t.texture_warn_std)
        if paths["normals"].exists():
            check_image(paths["normals"], "normals", reasons, t.normals_warn_std)

    check_size(paths["kpt2d"], 1, "kpt2d", reasons)
    check_size(paths["kpt3d"], 1, "kpt3d", reasons)
    check_size(paths["mat"], t.mat_min_bytes, "mat", reasons)
    if paths["kpt2d"].exists() and paths["kpt3d"].exists():
        check_kpts(paths["kpt2d"], paths["kpt3d"], image_size, reasons, t, kpt_normalized=kpt_normalized)
    if paths["mat"].exists():
        check_mat(paths["mat"], reasons)

    status = classify(reasons)
    return {
        "image_id": stem,
        "status": status,
        "reason": ";".join(reasons),
        "input_path": str(input_path),
        "vis_path": str(paths["vis"]),
        "depth_path": str(paths["depth"]),
        "obj_path": str(paths["obj"]),
        "detail_obj_path": str(paths["detail_obj"]),
        "mat_path": str(paths["mat"]),
        "kpt2d_path": str(paths["kpt2d"]),
        "kpt3d_path": str(paths["kpt3d"]),
    }


def write_markdown_summary(rows: list[dict[str, str]], path: Path) -> None:
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    top_reasons: dict[str, int] = {}
    for row in rows:
        for reason in filter(None, row["reason"].split(";")):
            key = reason.split(":", 2)[1] if ":" in reason else reason
            top_reasons[key] = top_reasons.get(key, 0) + 1

    with path.open("w", encoding="utf-8") as f:
        f.write("# DECA Output Audit Summary\n\n")
        f.write(f"- Total: {len(rows)}\n")
        f.write(f"- PASS: {counts.get('PASS', 0)}\n")
        f.write(f"- WARN: {counts.get('WARN', 0)}\n")
        f.write(f"- FAIL: {counts.get('FAIL', 0)}\n\n")
        f.write("## Top Reasons\n\n")
        for reason, count in sorted(top_reasons.items(), key=lambda x: x[1], reverse=True)[:30]:
            f.write(f"- {reason}: {count}\n")
        f.write("\n## Review Priority\n\n")
        f.write("1. Inspect every FAIL sample.\n")
        f.write("2. Inspect every WARN sample, or at least the most frequent WARN reason groups.\n")
        f.write("3. Randomly inspect a small subset of PASS samples.\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DECA batch outputs into PASS/WARN/FAIL.")
    parser.add_argument("--input-dir", type=Path, help="Directory or single image containing input face images.")
    parser.add_argument("--input-list", type=Path, help="Optional text file with one input image path per line.")
    parser.add_argument("--results-dir", required=True, type=Path, help="DECA output directory passed via -s.")
    parser.add_argument("--out-csv", default=Path("audit_results.csv"), type=Path)
    parser.add_argument("--out-md", default=Path("audit_summary.md"), type=Path)
    parser.add_argument("--profile", choices=["full", "light", "no_render"], default="full",
                        help="full: obj/depth/vis/mat/kpt; light: depth/vis/mat/kpt; no_render: mat/kpt.")
    parser.add_argument("--image-size", default=224, type=int, help="DECA crop/render size for kpt sanity checks.")
    parser.add_argument("--kpt-normalized", action="store_true",
                        help="Keypoints are normalized coord [-1,1] (DECA default). Scales to --image-size for checks.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input-dir for images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input_list:
        images = read_input_list(args.input_list)
    elif args.input_dir:
        images = iter_images(args.input_dir, args.recursive)
    else:
        raise SystemExit("Please provide --input-dir or --input-list.")
    thresholds = Thresholds()
    rows = [audit_one(p, args.results_dir, args.profile, args.image_size, thresholds, kpt_normalized=args.kpt_normalized) for p in images]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id", "status", "reason", "input_path", "vis_path", "depth_path",
        "obj_path", "detail_obj_path", "mat_path", "kpt2d_path", "kpt3d_path",
    ]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_markdown_summary(rows, args.out_md)

    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"Audited {len(rows)} images")
    print(f"PASS={counts.get('PASS', 0)} WARN={counts.get('WARN', 0)} FAIL={counts.get('FAIL', 0)}")
    print(f"CSV: {args.out_csv}")
    print(f"Summary: {args.out_md}")


if __name__ == "__main__":
    main()
