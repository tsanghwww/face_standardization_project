#!/usr/bin/env python3
"""Screen DECA parameter outputs into Pass/Warn groups.

This script reads DECA `.mat` files, builds a high-dimensional parameter vector
per image, computes a Mahalanobis-distance score, and writes screening outputs:

- screening_stats.json
- screening_report.json
- fail_report.json
- review_manifest.csv
- _progress.txt
- optional pass_images/ and warn_images/ copies

Example on Windows:

python tools\\screening_deca_params.py ^
  --params-dir H:\\face_standardization_project\\results\\screening_params_cuda ^
  --images-dir H:\\face_standardization_project\\datasets\\my_faces ^
  --out-dir H:\\face_standardization_project\\results\\screening_v4_p97_5 ^
  --warn-percentile 97.5 ^
  --copy-images

The default parameter keys target DECA-style outputs. If your MAT files use
different key names, run with `--param-keys exp pose cam light` or inspect one
MAT file with `--inspect-one`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")
DEFAULT_PARAM_KEYS = ("exp", "pose", "cam", "light")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_progress(path: Path, msg: str) -> None:
    line = f"{utc_now()} {msg}"
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def scalarize_mat_value(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    arr = np.squeeze(arr)
    if arr.dtype == object:
        parts = [scalarize_mat_value(x) for x in arr.ravel()]
        return np.concatenate([p.ravel() for p in parts if p.size])
    return arr.astype(np.float64).ravel()


def mat_public_keys(data: dict[str, Any]) -> list[str]:
    return sorted(k for k in data.keys() if not k.startswith("__"))


def find_mat_files(params_dir: Path) -> list[Path]:
    return sorted(params_dir.rglob("*.mat"))


def image_id_from_mat(path: Path) -> str:
    return path.stem


def image_path_for_id(images_dir: Path | None, image_id: str) -> Path | None:
    if images_dir is None:
        return None
    for ext in IMAGE_EXTS:
        p = images_dir / f"{image_id}{ext}"
        if p.exists():
            return p
    return None


def load_param_vector(path: Path, param_keys: tuple[str, ...]) -> tuple[np.ndarray | None, list[str], dict[str, int]]:
    reasons: list[str] = []
    dims: dict[str, int] = {}
    try:
        data = loadmat(path)
    except Exception as exc:
        return None, [f"cannot_read_mat:{type(exc).__name__}:{exc}"], dims

    chunks: list[np.ndarray] = []
    keys = set(mat_public_keys(data))
    for key in param_keys:
        if key not in keys:
            reasons.append(f"missing_key:{key}")
            continue
        try:
            vec = scalarize_mat_value(data[key])
        except Exception as exc:
            reasons.append(f"bad_key:{key}:{type(exc).__name__}:{exc}")
            continue
        if vec.size == 0:
            reasons.append(f"empty_key:{key}")
            continue
        if not np.isfinite(vec).all():
            reasons.append(f"nonfinite_key:{key}")
            continue
        dims[key] = int(vec.size)
        chunks.append(vec)

    if not chunks:
        return None, reasons or ["no_usable_param_keys"], dims

    vec = np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(vec).all():
        return None, reasons + ["nonfinite_vector"], dims
    return vec, reasons, dims


def robust_inverse_covariance(x: np.ndarray, ridge: float) -> np.ndarray:
    cov = np.cov(x, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]], dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    diag = np.diag(cov)
    scale = float(np.nanmedian(diag[diag > 0])) if np.any(diag > 0) else 1.0
    cov = cov + np.eye(cov.shape[0]) * ridge * scale
    return np.linalg.pinv(cov)


def mahalanobis_scores(x: np.ndarray, center: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    return np.array([mahalanobis(row, center, inv_cov) for row in x], dtype=np.float64)


def copy_group_images(rows: list[dict[str, Any]], images_dir: Path | None, out_dir: Path) -> None:
    if images_dir is None:
        return
    for group in ("Pass", "Warn"):
        (out_dir / f"{group.lower()}_images").mkdir(parents=True, exist_ok=True)
    for row in rows:
        image_path = image_path_for_id(images_dir, str(row["image_id"]))
        if image_path is None:
            row["image_exists"] = False
            continue
        row["image_exists"] = True
        dest = out_dir / f"{str(row['label']).lower()}_images" / image_path.name
        if not dest.exists():
            shutil.copy2(image_path, dest)


def build_review_manifest(
    ids: list[str],
    labels: np.ndarray,
    scores: np.ndarray,
    p_values: np.ndarray,
    fail_reasons: dict[str, list[str]],
    warn_percentile: float,
    seed: int,
) -> list[dict[str, str]]:
    rng = np.random.default_rng(seed)
    review: list[dict[str, str]] = []

    for image_id in list(fail_reasons.keys())[:50]:
        review.append({"image_id": image_id, "layer": "fail", "label": "Fail", "D2": "", "p_value": ""})

    order = np.argsort(scores)[::-1]
    for idx in order[:20]:
        review.append({
            "image_id": ids[idx], "layer": "top20", "label": str(labels[idx]),
            "D2": f"{float(scores[idx]):.4f}", "p_value": f"{float(p_values[idx]):.6g}",
        })

    p_main = np.percentile(scores, warn_percentile)
    p_upper = np.percentile(scores, min(99.9, warn_percentile + 2.5))
    boundary = np.array([i for i, d in enumerate(scores) if p_main <= d <= p_upper], dtype=int)
    if boundary.size:
        for idx in rng.choice(boundary, size=min(250, boundary.size), replace=False):
            review.append({
                "image_id": ids[int(idx)], "layer": "boundary", "label": str(labels[int(idx)]),
                "D2": f"{float(scores[int(idx)]):.4f}", "p_value": f"{float(p_values[int(idx)]):.6g}",
            })

    p50, p70 = np.percentile(scores, [50, 70])
    center = np.array([i for i, d in enumerate(scores) if p50 <= d <= p70], dtype=int)
    if center.size:
        for idx in rng.choice(center, size=min(30, center.size), replace=False):
            review.append({
                "image_id": ids[int(idx)], "layer": "center", "label": str(labels[int(idx)]),
                "D2": f"{float(scores[int(idx)]):.4f}", "p_value": f"{float(p_values[int(idx)]):.6g}",
            })

    if len(ids):
        random_idx = rng.choice(len(ids), size=min(50, len(ids)), replace=False)
        for idx in random_idx:
            review.append({
                "image_id": ids[int(idx)], "layer": "random", "label": str(labels[int(idx)]),
                "D2": f"{float(scores[int(idx)]):.4f}", "p_value": f"{float(p_values[int(idx)]):.6g}",
            })

    layer_prio = {"fail": 0, "top20": 1, "boundary": 2, "random": 3, "center": 4}
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in sorted(review, key=lambda x: layer_prio.get(x["layer"], 99)):
        if row["image_id"] in seen:
            continue
        unique.append(row)
        seen.add(row["image_id"])
    return unique


def inspect_one(path: Path) -> None:
    data = loadmat(path)
    print(f"MAT: {path}")
    for key in mat_public_keys(data):
        arr = np.asarray(data[key])
        print(f"{key}: shape={arr.shape} dtype={arr.dtype}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen DECA parameters with Mahalanobis distance.")
    parser.add_argument("--params-dir", required=True, type=Path, help="Directory containing DECA .mat parameter files.")
    parser.add_argument("--images-dir", type=Path, help="Optional directory with source images named <image_id>.<ext>.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Output directory.")
    parser.add_argument("--warn-percentile", default=97.5, type=float, help="Percentile above which samples are Warn.")
    parser.add_argument("--extreme-quantile", default=0.999, type=float, help="Chi-square quantile for robust extreme removal.")
    parser.add_argument("--param-keys", nargs="+", default=list(DEFAULT_PARAM_KEYS),
                        help="MAT keys to concatenate into the screening vector.")
    parser.add_argument("--ridge", default=1e-6, type=float, help="Ridge factor added before covariance inversion.")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for review manifest sampling.")
    parser.add_argument("--copy-images", action="store_true", help="Copy available images into pass_images/warn_images.")
    parser.add_argument("--inspect-one", type=Path, help="Inspect a single .mat file and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inspect_one:
        inspect_one(args.inspect_one)
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    progress = args.out_dir / "_progress.txt"
    if progress.exists():
        progress.unlink()

    param_keys = tuple(args.param_keys)
    write_progress(progress, "=== DECA parameter screening started ===")
    write_progress(progress, f"params_dir={args.params_dir}")
    write_progress(progress, f"images_dir={args.images_dir}")
    write_progress(progress, f"out_dir={args.out_dir}")
    write_progress(progress, f"param_keys={param_keys}")
    write_progress(progress, f"warn_percentile={args.warn_percentile}")

    mats = find_mat_files(args.params_dir)
    if not mats:
        raise SystemExit(f"No .mat files found under {args.params_dir}")

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    fail_reasons: dict[str, list[str]] = {}
    key_dim_counter: Counter[str] = Counter()
    expected_dim: int | None = None

    for i, mat_path in enumerate(mats, start=1):
        image_id = image_id_from_mat(mat_path)
        vec, reasons, dims = load_param_vector(mat_path, param_keys)
        for key, dim in dims.items():
            key_dim_counter[f"{key}:{dim}"] += 1
        if vec is None:
            fail_reasons[image_id] = reasons
            continue
        if expected_dim is None:
            expected_dim = int(vec.size)
        elif int(vec.size) != expected_dim:
            fail_reasons[image_id] = [f"dim_mismatch:{vec.size}!={expected_dim}"] + reasons
            continue
        ids.append(image_id)
        vectors.append(vec)
        if i % 1000 == 0:
            write_progress(progress, f"loaded {i}/{len(mats)} mats; ok={len(ids)} fail={len(fail_reasons)}")

    if not vectors:
        raise SystemExit("No usable parameter vectors were loaded.")

    x = np.vstack(vectors)
    n_total = len(mats)
    n_fail = len(fail_reasons)
    n_nonfail, dim = x.shape
    write_progress(progress, f"Process 1 done: {n_nonfail} non-fail, {n_fail} fail / {n_total}")
    write_progress(progress, f"Screen matrix: {x.shape}")

    median = np.median(x, axis=0)
    inv_cov_r = robust_inverse_covariance(x, args.ridge)
    robust_scores = mahalanobis_scores(x, median, inv_cov_r)

    extreme_thresh = chi2.ppf(args.extreme_quantile, df=dim)
    # scipy returns Mahalanobis distance, while chi-square threshold is for squared distance.
    clean_mask = robust_scores ** 2 <= extreme_thresh
    n_extreme = int(np.sum(~clean_mask))
    x_clean = x[clean_mask] if np.any(clean_mask) else x
    write_progress(progress, f"Extreme outliers removed: {n_extreme}")

    mu = np.mean(x_clean, axis=0)
    inv_cov = robust_inverse_covariance(x_clean, args.ridge)
    scores = mahalanobis_scores(x, mu, inv_cov)
    score_sq = scores ** 2
    p_values = 1 - chi2.cdf(score_sq, df=dim)

    warn_thresh = float(np.percentile(scores, args.warn_percentile))
    labels = np.where(scores > warn_thresh, "Warn", "Pass")
    n_pass = int(np.sum(labels == "Pass"))
    n_warn = int(np.sum(labels == "Warn"))
    write_progress(progress, f"P{args.warn_percentile:g} threshold: {warn_thresh:.4f}")
    write_progress(progress, f"Pass: {n_pass} ({100*n_pass/max(1,len(labels)):.2f}%)")
    write_progress(progress, f"Warn: {n_warn} ({100*n_warn/max(1,len(labels)):.2f}%)")
    write_progress(progress, f"D P50={np.percentile(scores,50):.4f} P95={np.percentile(scores,95):.4f} P99={np.percentile(scores,99):.4f}")

    report: list[dict[str, Any]] = []
    for i, image_id in enumerate(ids):
        row = {
            "image_id": image_id,
            "label": str(labels[i]),
            "D2": round(float(scores[i]), 4),
            "D2_squared": round(float(score_sq[i]), 4),
            "p_value": round(float(p_values[i]), 8),
        }
        report.append(row)
    report.sort(key=lambda r: r["D2"], reverse=True)

    if args.copy_images:
        write_progress(progress, "Copying source images into pass_images/warn_images...")
        copy_group_images(report, args.images_dir, args.out_dir)

    stats = {
        "source_script": "tools/screening_deca_params.py",
        "generated_at": utc_now(),
        "params_dir": str(args.params_dir),
        "images_dir": str(args.images_dir) if args.images_dir else None,
        "n_total": n_total,
        "n_fail": n_fail,
        "n_nonfail": n_nonfail,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_extreme_removed": n_extreme,
        "screen_dim": dim,
        "param_keys": list(param_keys),
        "key_dims_seen": dict(key_dim_counter),
        "warn_percentile": args.warn_percentile,
        "warn_threshold": round(warn_thresh, 6),
        "extreme_quantile": args.extreme_quantile,
        "extreme_threshold_chi2_for_squared_distance": round(float(extreme_thresh), 6),
        "score_note": "D2 is the Mahalanobis distance; D2_squared is used for chi-square p_value.",
    }

    fail_list = [{"image_id": image_id, "reason": ";".join(reasons)} for image_id, reasons in fail_reasons.items()]

    review = build_review_manifest(ids, labels, scores, p_values, fail_reasons, args.warn_percentile, args.seed)

    with (args.out_dir / "screening_stats.json").open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    with (args.out_dir / "screening_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with (args.out_dir / "fail_report.json").open("w", encoding="utf-8") as f:
        json.dump(fail_list, f, indent=2, ensure_ascii=False)
    with (args.out_dir / "review_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "layer", "label", "D2", "p_value"])
        writer.writeheader()
        writer.writerows(review)

    write_progress(progress, "Saving outputs...")
    write_progress(progress, f"screening_stats.json")
    write_progress(progress, f"screening_report.json ({len(report)} rows)")
    write_progress(progress, f"fail_report.json ({len(fail_list)} rows)")
    write_progress(progress, f"review_manifest.csv ({len(review)} samples): {dict(Counter(r['layer'] for r in review))}")
    write_progress(progress, f"=== DONE === All outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
