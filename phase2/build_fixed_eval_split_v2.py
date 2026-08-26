"""Build the immutable Phase2 fixed test manifest v2.

v2 = v1 (results/phase2_eval_fixed_20260824/fixed_test_manifest.csv, 625 rows,
kept verbatim) + 75 COFW occlusion samples + 75 AFLW2000-3D large-pose samples.

COFW is an occlusion dataset and AFLW2000-3D is a large-pose dataset by
construction, so no per-sample attribute gating is performed: 75 samples are
drawn from each with a fixed seed (deterministic, reproducible).  Occlusion
ratio (COFW) and yaw (AFLW) are still recorded as factual provenance columns.

External images are materialized under <out-dir>/images so the v2 manifest stays
immutable even if the external dataset folders change.  COFW images are read
from the HDF5 mat in (C, W, H) layout -> numpy (H, W, C), then cropped around
the annotated face box (same margin rule as v1 WIDER crops).

Outputs in <out-dir> (default results/phase2_eval_fixed_20260824_v2):
  fixed_test_manifest_v2.csv   - 775 rows (v1 FIELDS + cofw_occlusion_ratio,
                                 aflw_yaw_deg)
  base_test_ids.txt            - sorted stylegan2_base image_ids (400)
  images/                      - cofw_occlusion_XXX.jpg, aflw_large_pose_XXX.jpg
  deca_params/                 - placeholder for later DECA extraction
  split_summary.json           - provenance + counts
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

from .build_fixed_eval_split import FIELDS as V1_FIELDS

V2_FIELDS = V1_FIELDS + ["cofw_occlusion_ratio", "aflw_yaw_deg"]

PROJECT = Path(r"D:\face_standardization_project")
SEED = 20260824


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1-manifest",
        type=Path,
        default=PROJECT / "results" / "phase2_eval_fixed_20260824" / "fixed_test_manifest.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2",
    )
    parser.add_argument(
        "--cofw-mat",
        type=Path,
        default=PROJECT / "datasets" / "external" / "COFW_Color" / "COFW_color" / "COFW_test_color.mat",
    )
    parser.add_argument("--cofw-count", type=int, default=75)
    parser.add_argument(
        "--aflw-crop-dir",
        type=Path,
        default=PROJECT / "datasets" / "external" / "AFLW2000-3D" / "test.data" / "AFLW2000-3D_crop",
    )
    parser.add_argument(
        "--aflw-list",
        type=Path,
        default=PROJECT / "datasets" / "external" / "AFLW2000-3D" / "test.data" / "AFLW2000-3D_crop.list",
    )
    parser.add_argument(
        "--aflw-pose",
        type=Path,
        default=PROJECT / "datasets" / "external" / "AFLW2000-3D" / "annotations" / "AFLW2000-3D.pose.npy",
    )
    parser.add_argument("--aflw-count", type=int, default=75)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def read_v1_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_cofw_image(dataset: h5py.Dataset) -> np.ndarray:
    """Decode a COFW HDF5 image dataset (stored (C, W, H) or (W, H)) to (H, W, C) BGR."""
    arr = dataset[:]
    if arr.ndim == 3:
        image = np.transpose(arr, (2, 1, 0))          # (C, W, H) -> (H, W, C) RGB
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = arr.T                                     # (W, H) -> (H, W) gray
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def crop_face(image: np.ndarray, box: tuple[float, float, float, float], margin_ratio: float = 0.20) -> np.ndarray | None:
    """Crop around an [x, y, w, h] face box plus margin (mirrors v1 WIDER crops)."""
    x, y, w, h = box
    height, width = image.shape[:2]
    margin = int(round(max(w, h) * margin_ratio))
    x1, y1 = max(0, int(x) - margin), max(0, int(y) - margin)
    x2, y2 = min(width, int(x + w) + margin), min(height, int(y + h) + margin)
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def select_cofw(args: argparse.Namespace, out_images: Path) -> list[dict[str, str]]:
    rng = random.Random(args.seed)
    with h5py.File(args.cofw_mat, "r") as f:
        phis = f["phisT"][:]          # (87, 507) = 29*x, 29*y, 29*occlusion
        bboxes = f["bboxesT"][:]      # (4, 507) = x, y, w, h
        n_total = phis.shape[1]
        indices = sorted(rng.sample(range(n_total), args.cofw_count))
        rows: list[dict[str, str]] = []
        for rank, index in enumerate(indices):
            eval_id = f"cofw_occlusion_{rank:03d}"
            ref = f["IsT"][0, index]
            image = read_cofw_image(f[ref])
            x, y, w, h = (float(v) for v in bboxes[:, index])
            crop = crop_face(image, (x, y, w, h))
            if crop is None:
                raise SystemExit(f"COFW index {index}: empty crop")
            out_path = out_images / f"{eval_id}.jpg"
            if not cv2.imwrite(str(out_path), crop):
                raise SystemExit(f"COFW index {index}: failed to write {out_path}")
            n_occ = int((phis[58:87, index] == 1).sum())
            rows.append(
                {
                    "eval_id": eval_id,
                    "split": "test",
                    "source_group": "cofw_occlusion",
                    "source_dataset": "COFW_Color",
                    "image_id": f"cofw_test_{index + 1}",
                    "image_path": str(out_path),
                    "source_image_path": str(args.cofw_mat),
                    "mat_path": str(args.out_dir / "deca_params" / eval_id / f"{eval_id}.mat"),
                    "xgb_quality_label": "external_unlabeled",
                    "xgb_quality_score": "",
                    "head_pose_norm": "",
                    "landmark_score": "",
                    "landmark_out_ratio": "",
                    "wider_blur": "",
                    "wider_occlusion": "",
                    "wider_pose": "",
                    "cofw_occlusion_ratio": f"{n_occ / 29.0:.6f}",
                    "aflw_yaw_deg": "",
                }
            )
    return rows


def select_aflw(args: argparse.Namespace, out_images: Path) -> list[dict[str, str]]:
    pose = np.load(args.aflw_pose)          # (2000,) yaw in degrees
    lines = [ln.strip() for ln in args.aflw_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) != len(pose):
        raise SystemExit(f"AFLW list lines ({len(lines)}) != pose entries ({len(pose)})")
    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(pose)), args.aflw_count))
    rows: list[dict[str, str]] = []
    for rank, index in enumerate(indices):
        eval_id = f"aflw_large_pose_{rank:03d}"
        name = Path(lines[index]).name
        src = args.aflw_crop_dir / name
        if not src.exists():
            raise SystemExit(f"AFLW crop missing: {src}")
        out_path = out_images / f"{eval_id}.jpg"
        shutil.copyfile(src, out_path)
        rows.append(
            {
                "eval_id": eval_id,
                "split": "test",
                "source_group": "aflw_large_pose",
                "source_dataset": "AFLW2000-3D",
                "image_id": src.stem,
                "image_path": str(out_path),
                "source_image_path": str(src),
                "mat_path": str(args.out_dir / "deca_params" / eval_id / f"{eval_id}.mat"),
                "xgb_quality_label": "external_unlabeled",
                "xgb_quality_score": "",
                "head_pose_norm": "",
                "landmark_score": "",
                "landmark_out_ratio": "",
                "wider_blur": "",
                "wider_occlusion": "",
                "wider_pose": "",
                "cofw_occlusion_ratio": "",
                "aflw_yaw_deg": f"{float(pose[index]):.3f}",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_images = args.out_dir / "images"
    # Clear any stale artifacts from a previous build of this same dir.
    for stale in out_images.glob("*.jpg"):
        stale.unlink()
    out_images.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "deca_params").mkdir(parents=True, exist_ok=True)

    v1_rows = read_v1_rows(args.v1_manifest)
    if len(v1_rows) != 625:
        raise SystemExit(f"v1 manifest has {len(v1_rows)} rows, expected 625")

    cofw_rows = select_cofw(args, out_images)
    aflw_rows = select_aflw(args, out_images)
    all_rows = v1_rows + cofw_rows + aflw_rows

    eval_ids = [row["eval_id"] for row in all_rows]
    if len(set(eval_ids)) != len(eval_ids):
        raise SystemExit("Duplicate eval_id in v2 manifest")
    image_ids = [row["image_id"] for row in all_rows]
    if len(set(image_ids)) != len(image_ids):
        raise SystemExit("Duplicate image_id in v2 manifest")
    for row in all_rows:
        if not Path(row["image_path"]).exists():
            raise SystemExit(f"Missing image for {row['eval_id']}: {row['image_path']}")

    manifest_path = args.out_dir / "fixed_test_manifest_v2.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=V2_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    base_ids = sorted(row["image_id"] for row in all_rows if row["source_dataset"] == "stylegan2_base")
    (args.out_dir / "base_test_ids.txt").write_text("\n".join(base_ids) + "\n", encoding="utf-8")

    counts: dict[str, int] = {}
    for row in all_rows:
        counts[row["source_group"]] = counts.get(row["source_group"], 0) + 1
    summary = {
        "seed": args.seed,
        "v2_count": len(all_rows),
        "counts": counts,
        "base_test_ids_count": len(base_ids),
        "cofw": {
            "count": len(cofw_rows),
            "source": str(args.cofw_mat),
            "sample_indices": [int(r["image_id"].split("_")[-1]) - 1 for r in cofw_rows],
            "occlusion_ratio_range": (
                min(float(r["cofw_occlusion_ratio"]) for r in cofw_rows),
                max(float(r["cofw_occlusion_ratio"]) for r in cofw_rows),
            ),
        },
        "aflw": {
            "count": len(aflw_rows),
            "source": str(args.aflw_crop_dir),
            "yaw_range_deg": (
                min(float(r["aflw_yaw_deg"]) for r in aflw_rows),
                max(float(r["aflw_yaw_deg"]) for r in aflw_rows),
            ),
        },
        "manifest": str(manifest_path),
        "base_test_ids": str(args.out_dir / "base_test_ids.txt"),
        "external_image_dir": str(out_images),
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
