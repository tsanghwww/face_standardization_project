#!/usr/bin/env python3
"""Draw direct/inverse gaze-convention candidates on Phase3 target normals."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.io import loadmat

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from eval.gaze_geometry import axis_angle_to_matrix, camera_to_head_gaze, head_to_camera_gaze, normalize_vector


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def transpose(matrix):
    return tuple(tuple(matrix[column][row] for column in range(3)) for row in range(3))


def arrow_image(normal_path: str, eye_mask_path: str, gaze: tuple[float, float, float], color: tuple[int, int, int]) -> Image.Image:
    image = cv2.imread(normal_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(eye_mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        raise ValueError("audit_input_missing")
    moments = cv2.moments(mask)
    if moments["m00"] <= 0:
        raise ValueError("empty_eye_mask")
    center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    scale = image.shape[0] * 0.22
    endpoint = (int(center[0] + gaze[0] * scale), int(center[1] + gaze[1] * scale))
    cv2.arrowedLine(image, center, endpoint, color, 4, cv2.LINE_AA, tipLength=0.25)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-manifest", required=True, type=Path)
    parser.add_argument("--phase2-manifest", required=True, type=Path)
    parser.add_argument("--condition-cache", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    ids = [line.strip() for line in args.ids_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    base = {row["image_id"]: row for row in read_csv(args.phase1_manifest)}
    phase2 = {row["image_id"]: row for row in read_csv(args.phase2_manifest)}
    cache = {row["image_id"]: row for row in read_csv(args.condition_cache)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sheets: list[Image.Image] = []
    audit_rows: list[dict[str, object]] = []

    for image_id in ids:
        b, p, c = base[image_id], phase2[image_id], cache[image_id]
        mat_path = Path(b["deca_mat_path"])
        if not mat_path.is_absolute():
            mat_path = args.project_root / mat_path
        source_pose = np.asarray(loadmat(mat_path)["pose"], dtype=np.float64).reshape(-1)[:3]
        npz_path = Path(p["out_npz"])
        if not npz_path.is_absolute():
            npz_path = args.project_root / npz_path
        with np.load(npz_path, allow_pickle=False) as data:
            target_pose = np.asarray(data["pose_standardized"], dtype=np.float64).reshape(-1)[:3]
        gaze_camera = normalize_vector([float(b["gaze_x"]), float(b["gaze_y"]), float(b["gaze_z"])])
        source_rotation = axis_angle_to_matrix(source_pose)
        target_rotation = axis_angle_to_matrix(target_pose)
        direct_head = camera_to_head_gaze(gaze_camera, source_rotation)
        direct_target = head_to_camera_gaze(direct_head, target_rotation)
        inverse_head = camera_to_head_gaze(gaze_camera, transpose(source_rotation))
        inverse_target = head_to_camera_gaze(inverse_head, transpose(target_rotation))
        direct_image = arrow_image(c["target_normal"], c["target_eye_mask"], direct_target, (40, 220, 40))
        inverse_image = arrow_image(c["target_normal"], c["target_eye_mask"], inverse_target, (40, 80, 230))
        tile = Image.new("RGB", (direct_image.width * 2, direct_image.height + 34), "white")
        tile.paste(direct_image, (0, 34))
        tile.paste(inverse_image, (direct_image.width, 34))
        draw = ImageDraw.Draw(tile)
        draw.text((8, 8), f"{image_id}  direct (green)", fill="black")
        draw.text((direct_image.width + 8, 8), "inverse (red)", fill="black")
        tile_path = args.out_dir / f"{image_id}_coordinate_candidates.png"
        tile.save(tile_path)
        sheets.append(tile)
        audit_rows.append({
            "image_id": image_id,
            "source_pose_norm": float(np.linalg.norm(source_pose)),
            "direct_target_gaze": list(direct_target),
            "inverse_target_gaze": list(inverse_target),
            "candidate_angle_deg": math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(direct_target, inverse_target)))))),
            "panel": str(tile_path),
        })

    if sheets:
        columns = 2
        rows = math.ceil(len(sheets) / columns)
        sheet = Image.new("RGB", (sheets[0].width * columns, sheets[0].height * rows), "white")
        for index, tile in enumerate(sheets):
            sheet.paste(tile, ((index % columns) * tile.width, (index // columns) * tile.height))
        sheet.save(args.out_dir / "coordinate_audit_contact_sheet.png")
    (args.out_dir / "coordinate_audit_values.json").write_text(json.dumps(audit_rows, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(audit_rows), "status": "manual_review_required", "out_dir": str(args.out_dir)}, indent=2))


if __name__ == "__main__":
    main()
