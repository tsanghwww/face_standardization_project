#!/usr/bin/env python3
"""Render Phase3 source/target 3D conditions from DECA and Phase2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.io import loadmat

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from eval.gaze_geometry import axis_angle_to_matrix, camera_to_head_gaze, head_to_camera_gaze, normalize_vector


PARAM_DIMS = {"shape": 100, "expression": 50, "pose": 6, "camera": 3, "light": 27, "detail": 128}
FIELDS = [
    "image_id", "split", "status", "failure_reason", "source_normal", "source_depth", "source_landmark",
    "source_face_mask", "source_eye_mask", "target_normal", "target_depth", "target_landmark",
    "target_face_mask", "target_eye_mask", "target_gaze_heatmap", "gaze_policy", "coordinate_status",
    "gaze_head_x", "gaze_head_y", "gaze_head_z", "target_gaze_head_x", "target_gaze_head_y", "target_gaze_head_z",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def split_map(path: Path | None) -> dict[str, str]:
    result: dict[str, str] = {}
    if path is None:
        return result
    for split in ("train", "validation", "fixed_test_base", "fixed_test_external"):
        candidate = path / f"{split}_ids.txt"
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    result[line.strip()] = split
    return result


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_source_params(path: Path) -> dict[str, np.ndarray]:
    data = loadmat(path)
    params: dict[str, np.ndarray] = {}
    for key, dim in PARAM_DIMS.items():
        array = np.asarray(data[key], dtype=np.float32).reshape(-1)
        if array.size != dim or not np.isfinite(array).all():
            raise ValueError(f"invalid_{key}:{array.shape}")
        params[key] = array
    return params


def apply_phase2(source: dict[str, np.ndarray], npz_path: Path) -> dict[str, np.ndarray]:
    target = {key: value.copy() for key, value in source.items()}
    with np.load(npz_path, allow_pickle=False) as data:
        target["expression"] = np.asarray(data["expression_standardized"], dtype=np.float32).reshape(-1)
        target["pose"] = np.asarray(data["pose_standardized"], dtype=np.float32).reshape(-1)
    if target["expression"].size != 50 or target["pose"].size != 6:
        raise ValueError("invalid_phase2_dimensions")
    return target


def codedict(params: dict[str, np.ndarray], device: str, image_size: int) -> dict[str, torch.Tensor]:
    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(value).to(device).float()[None, :]

    return {
        "shape": tensor(params["shape"]),
        "exp": tensor(params["expression"]),
        "pose": tensor(params["pose"]),
        "cam": tensor(params["camera"]),
        "light": tensor(params["light"]).reshape(1, 9, 3),
        "detail": tensor(params["detail"]),
        "images": torch.zeros((1, 3, image_size, image_size), dtype=torch.float32, device=device),
    }


def uint8_image(tensor: torch.Tensor, signed: bool = False) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    array = np.moveaxis(array, 0, -1) if array.ndim == 3 else array
    if signed:
        array = (array + 1.0) * 0.5
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def write_conditions(deca, params: dict[str, np.ndarray], directory: Path, device: str, image_size: int) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        op = deca.decode(codedict(params, device, image_size), rendering=True, return_vis=False, use_detail=False)
        depth = deca.render.render_depth(op["trans_verts"].clone())[0, 0]
    alpha = uint8_image(op["alpha_images"][0, 0])
    normal = uint8_image(op["normal_images"][0], signed=True)
    normal[alpha == 0] = 0
    normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    depth_u16 = np.clip(depth.detach().cpu().numpy(), 0.0, 1.0)
    depth_u16 = np.round(depth_u16 * 65535.0).astype(np.uint16)

    landmarks = op["landmarks2d"][0, :, :2].detach().cpu().numpy()
    points = np.round((landmarks + 1.0) * (image_size / 2.0)).astype(np.int32)
    landmark_map = np.zeros((image_size, image_size), dtype=np.uint8)
    for x, y in points:
        if 0 <= x < image_size and 0 <= y < image_size:
            cv2.circle(landmark_map, (int(x), int(y)), 2, 255, -1, lineType=cv2.LINE_AA)
    landmark_map = cv2.GaussianBlur(landmark_map, (0, 0), 1.5)
    eye_mask = np.zeros_like(landmark_map)
    eye_points = points[36:48]
    if eye_points.shape == (12, 2):
        hull = cv2.convexHull(eye_points)
        cv2.fillConvexPoly(eye_mask, hull, 255)
        eye_mask = cv2.dilate(eye_mask, np.ones((9, 9), np.uint8), iterations=1)

    paths = {
        "normal": directory / "normal.png",
        "depth": directory / "depth_u16.png",
        "landmark": directory / "landmark.png",
        "face_mask": directory / "face_mask.png",
        "eye_mask": directory / "eye_mask.png",
    }
    images = {"normal": normal, "depth": depth_u16, "landmark": landmark_map, "face_mask": alpha, "eye_mask": eye_mask}
    for key, path in paths.items():
        if not cv2.imwrite(str(path), images[key]):
            raise RuntimeError(f"write_failed:{path}")
    return {key: str(path) for key, path in paths.items()}


def approved_convention(path: Path | None) -> tuple[str, str]:
    if path is None:
        return "pending_manual_audit", ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "approved" or payload.get("convention") != "direct_head_to_camera":
        raise SystemExit("Coordinate approval must explicitly approve convention=direct_head_to_camera")
    return "approved", "direct_head_to_camera"


def write_gaze_heatmap(
    path: Path,
    target_eye_mask_path: str,
    source_pose: np.ndarray,
    target_pose: np.ndarray,
    gaze_camera: tuple[float, float, float],
    policy: str,
    image_size: int,
) -> None:
    source_rotation = axis_angle_to_matrix(source_pose[:3])
    target_rotation = axis_angle_to_matrix(target_pose[:3])
    gaze_head = camera_to_head_gaze(gaze_camera, source_rotation)
    target_gaze = (0.0, 0.0, -1.0) if policy == "canonical_camera_gaze" else head_to_camera_gaze(gaze_head, target_rotation)
    target_gaze = normalize_vector(target_gaze)
    eye_mask = cv2.imread(target_eye_mask_path, cv2.IMREAD_GRAYSCALE)
    moments = cv2.moments(eye_mask)
    center = (image_size // 2, image_size // 2)
    if moments["m00"] > 0:
        center = (int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"]))
    scale = image_size * 0.18
    endpoint = (int(center[0] + target_gaze[0] * scale), int(center[1] + target_gaze[1] * scale))
    heatmap = np.zeros((image_size, image_size), dtype=np.uint8)
    cv2.arrowedLine(heatmap, center, endpoint, 255, 3, line_type=cv2.LINE_AA, tipLength=0.25)
    heatmap = cv2.GaussianBlur(heatmap, (0, 0), 2.0)
    if not cv2.imwrite(str(path), heatmap):
        raise RuntimeError(f"write_failed:{path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-manifest", required=True, type=Path)
    parser.add_argument("--phase2-manifest", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--deca-root", required=True, type=Path)
    parser.add_argument("--split-registry-dir", type=Path)
    parser.add_argument("--coordinate-approval", type=Path)
    parser.add_argument("--gaze-policy", choices=["preserve_eye_in_head", "canonical_camera_gaze"], default="preserve_eye_in_head")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--rasterizer-type", default="standard")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.deca_root))
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    coordinate_status, _ = approved_convention(args.coordinate_approval)
    rows = read_csv(args.phase1_manifest)
    if args.ids_file:
        selected = {line.strip() for line in args.ids_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()}
        rows = [row for row in rows if row.get("image_id", "") in selected]
        missing_ids = selected - {row.get("image_id", "") for row in rows}
        if missing_ids:
            raise SystemExit(f"Selected IDs absent from Phase1 manifest: {sorted(missing_ids)[:10]}")
    if args.limit:
        rows = rows[: args.limit]
    phase2 = {row["image_id"]: row for row in read_csv(args.phase2_manifest)}
    splits = split_map(args.split_registry_dir)
    image_size = int(deca_cfg.dataset.image_size)
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.use_tex = False
    deca = DECA(config=deca_cfg, device=args.device, render_enabled=True)

    manifest = args.out_dir / "phase3_condition_cache.csv"
    completed: dict[str, dict[str, str]] = {}
    if args.resume and manifest.exists():
        completed = {row["image_id"]: row for row in read_csv(manifest)}
    output = list(completed.values())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for base in rows:
        image_id = base["image_id"]
        if image_id in completed:
            continue
        row = {field: "" for field in FIELDS}
        row.update({"image_id": image_id, "split": splits.get(image_id, "unassigned"), "status": "failed", "gaze_policy": args.gaze_policy, "coordinate_status": coordinate_status})
        try:
            phase = phase2.get(image_id)
            if not phase or not phase.get("out_npz"):
                raise ValueError("missing_phase2_output")
            source = load_source_params(resolve(base["deca_mat_path"], args.project_root))
            target = apply_phase2(source, resolve(phase["out_npz"], args.project_root))
            source_paths = write_conditions(deca, source, args.out_dir / "maps" / image_id / "source", args.device, image_size)
            target_paths = write_conditions(deca, target, args.out_dir / "maps" / image_id / "target", args.device, image_size)
            for key, value in source_paths.items():
                row[f"source_{key}"] = value
            for key, value in target_paths.items():
                row[f"target_{key}"] = value
            if coordinate_status == "approved":
                gaze_camera = normalize_vector([float(base["gaze_x"]), float(base["gaze_y"]), float(base["gaze_z"])])
                source_rotation = axis_angle_to_matrix(source["pose"][:3])
                target_rotation = axis_angle_to_matrix(target["pose"][:3])
                gaze_head = camera_to_head_gaze(gaze_camera, source_rotation)
                target_gaze_head = (
                    camera_to_head_gaze((0.0, 0.0, -1.0), target_rotation)
                    if args.gaze_policy == "canonical_camera_gaze"
                    else gaze_head
                )
                row.update({f"gaze_head_{axis}": f"{gaze_head[index]:.9g}" for index, axis in enumerate("xyz")})
                row.update({f"target_gaze_head_{axis}": f"{target_gaze_head[index]:.9g}" for index, axis in enumerate("xyz")})
                gaze_path = args.out_dir / "maps" / image_id / "target" / "gaze_heatmap.png"
                write_gaze_heatmap(gaze_path, target_paths["eye_mask"], source["pose"], target["pose"], gaze_camera, args.gaze_policy, image_size)
                row["target_gaze_heatmap"] = str(gaze_path)
            row["status"] = "geometry_ready_gaze_pending" if coordinate_status != "approved" else "ready"
        except Exception as exc:  # failures remain explicit
            row["failure_reason"] = f"{type(exc).__name__}:{exc}"
        output.append(row)
        with manifest.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(output)

    summary = {
        "n_total": len(output),
        "n_ready": sum(row["status"] == "ready" for row in output),
        "n_geometry_ready_gaze_pending": sum(row["status"] == "geometry_ready_gaze_pending" for row in output),
        "n_failed": sum(row["status"] == "failed" for row in output),
        "coordinate_status": coordinate_status,
        "gaze_policy": args.gaze_policy,
        "head_local_gaze_written": coordinate_status == "approved",
        "manifest": str(manifest),
    }
    (args.out_dir / "condition_cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
