"""Build a deterministic Phase2 test manifest with in-domain and WIDER hard cases.

The script never silently drops a requested stratum.  It writes every requested
row to one manifest and records unavailable sources (such as a corrupt archive)
in ``split_summary.json``.  WIDER images are cropped by the annotated bounding
box so that DECA, ArcFace, and L2CS operate on the same face.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2

from .features import find_deca_mat_files, read_xgb_rows, sample_from_mat


FIELDS = [
    "eval_id",
    "split",
    "source_group",
    "source_dataset",
    "image_id",
    "image_path",
    "source_image_path",
    "mat_path",
    "xgb_quality_label",
    "xgb_quality_score",
    "head_pose_norm",
    "landmark_score",
    "landmark_out_ratio",
    "wider_blur",
    "wider_occlusion",
    "wider_pose",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-deca-results", required=True, type=Path)
    parser.add_argument("--base-images-dir", required=True, type=Path)
    parser.add_argument("--xgb-quality-manifest", required=True, type=Path)
    parser.add_argument("--wider-images-dir", required=True, type=Path)
    parser.add_argument("--wider-annotations", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--per-xgb-class", type=int, default=100)
    parser.add_argument("--base-hard-pose", type=int, default=50)
    parser.add_argument("--base-low-landmark", type=int, default=50)
    parser.add_argument("--wider-per-condition", type=int, default=75)
    parser.add_argument("--min-wider-box", type=int, default=64)
    return parser.parse_args()


def choose(rows: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    if len(rows) < count:
        raise SystemExit(f"Requested {count} examples but only {len(rows)} are available")
    return rng.sample(rows, count)


def parse_wider(path: Path) -> list[tuple[str, list[tuple[int, int, int, int, int, int, int, int, int, int]]]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    result = []
    index = 0
    while index < len(lines):
        image_name = lines[index]
        index += 1
        if not image_name:
            continue
        face_count = int(lines[index])
        index += 1
        boxes = []
        for _ in range(face_count):
            values = [int(float(value)) for value in lines[index].split()]
            index += 1
            if len(values) >= 10:
                boxes.append(tuple(values[:10]))
        result.append((image_name, boxes))
    return result


def crop_wider(
    item: tuple[str, tuple[int, int, int, int, int, int, int, int, int, int]],
    images_dir: Path,
    output_path: Path,
) -> bool:
    image_name, values = item
    x, y, w, h = values[:4]
    image = cv2.imread(str(images_dir / image_name))
    if image is None:
        return False
    height, width = image.shape[:2]
    margin = int(round(max(w, h) * 0.20))
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(width, x + w + margin), min(height, y + h + margin)
    if x2 <= x1 or y2 <= y1:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_path), image[y1:y2, x1:x2]))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = args.out_dir / "images"
    xgb = read_xgb_rows(args.xgb_quality_manifest)

    base_rows: list[dict[str, str]] = []
    for mat_path in find_deca_mat_files(args.base_deca_results):
        image_id = mat_path.stem
        xgb_row = xgb.get(image_id)
        if not xgb_row:
            continue
        sample = sample_from_mat(mat_path)
        base_rows.append(
            {
                "eval_id": f"base_{image_id}",
                "split": "test",
                "source_group": "",
                "source_dataset": "stylegan2_base",
                "image_id": image_id,
                "image_path": str(args.base_images_dir / f"{image_id}.png"),
                "source_image_path": str(args.base_images_dir / f"{image_id}.png"),
                "mat_path": str(mat_path),
                "xgb_quality_label": xgb_row.get("xgb_quality_label", ""),
                "xgb_quality_score": xgb_row.get("xgb_quality_score", ""),
                "head_pose_norm": f"{sample.metrics['head_pose_norm']:.6f}",
                "landmark_score": f"{sample.metrics['landmark_score']:.6f}",
                "landmark_out_ratio": f"{sample.metrics['landmark_out_ratio']:.6f}",
                "wider_blur": "",
                "wider_occlusion": "",
                "wider_pose": "",
            }
        )

    selected: list[dict[str, str]] = []
    selected_ids: set[str] = set()
    for label in ("high", "medium", "low"):
        picked = choose([row for row in base_rows if row["xgb_quality_label"] == label], args.per_xgb_class, rng)
        for row in picked:
            row = dict(row)
            row["source_group"] = f"xgb_{label}"
            selected.append(row)
            selected_ids.add(row["eval_id"])

    for group, key, count, reverse in (
        ("base_hard_pose", "head_pose_norm", args.base_hard_pose, True),
        ("base_low_landmark", "landmark_score", args.base_low_landmark, False),
    ):
        candidates = [row for row in base_rows if row["eval_id"] not in selected_ids]
        candidates.sort(key=lambda row: float(row[key]), reverse=reverse)
        picked = candidates[:count]
        if len(picked) < count:
            raise SystemExit(f"Insufficient base examples for {group}")
        for row in picked:
            row = dict(row)
            row["source_group"] = group
            selected.append(row)
            selected_ids.add(row["eval_id"])

    wider = parse_wider(args.wider_annotations)
    wider_candidates: dict[str, list[tuple[str, tuple[int, int, int, int, int, int, int, int, int, int]]]] = {
        "wider_pose": [],
        "wider_occlusion": [],
        "wider_blur": [],
    }
    for image_name, boxes in wider:
        for box in boxes:
            x, y, w, h, blur, _expression, _illumination, invalid, occlusion, pose = box
            if invalid or w < args.min_wider_box or h < args.min_wider_box:
                continue
            if pose > 0:
                wider_candidates["wider_pose"].append((image_name, box))
            if occlusion > 0:
                wider_candidates["wider_occlusion"].append((image_name, box))
            if blur > 0:
                wider_candidates["wider_blur"].append((image_name, box))

    used_wider: set[tuple[str, tuple[int, ...]]] = set()
    for group, candidates in wider_candidates.items():
        rng.shuffle(candidates)
        added = 0
        for image_name, box in candidates:
            key = (image_name, box)
            if key in used_wider:
                continue
            eval_id = f"{group}_{added:03d}"
            crop_path = crops_dir / f"{eval_id}.jpg"
            if not crop_wider((image_name, box), args.wider_images_dir, crop_path):
                continue
            x, y, w, h, blur, _expression, _illumination, _invalid, occlusion, pose = box
            selected.append(
                {
                    "eval_id": eval_id,
                    "split": "test",
                    "source_group": group,
                    "source_dataset": "WIDER_FACE_val",
                    "image_id": eval_id,
                    "image_path": str(crop_path),
                    "source_image_path": str(args.wider_images_dir / image_name),
                    "mat_path": str(args.out_dir / "deca_params" / eval_id / f"{eval_id}.mat"),
                    "xgb_quality_label": "external_unlabeled",
                    "xgb_quality_score": "",
                    "head_pose_norm": "",
                    "landmark_score": "",
                    "landmark_out_ratio": "",
                    "wider_blur": str(blur),
                    "wider_occlusion": str(occlusion),
                    "wider_pose": str(pose),
                }
            )
            used_wider.add(key)
            added += 1
            if added == args.wider_per_condition:
                break
        if added != args.wider_per_condition:
            raise SystemExit(f"Only cropped {added}/{args.wider_per_condition} examples for {group}")

    manifest_path = args.out_dir / "fixed_test_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(selected)
    base_test_ids = sorted(row["image_id"] for row in selected if row["source_dataset"] == "stylegan2_base")
    (args.out_dir / "base_test_ids.txt").write_text("\n".join(base_test_ids) + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for row in selected:
        counts[row["source_group"]] = counts.get(row["source_group"], 0) + 1
    summary = {
        "seed": args.seed,
        "count": len(selected),
        "counts": counts,
        "manifest": str(manifest_path),
        "external_crop_dir": str(crops_dir),
        "base_test_ids": str(args.out_dir / "base_test_ids.txt"),
        "cofw_status": "not_included: COFW_color.zip failed integrity/decompression on 2026-08-24",
    }
    (args.out_dir / "split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
