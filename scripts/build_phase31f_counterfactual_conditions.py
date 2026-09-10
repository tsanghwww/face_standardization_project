#!/usr/bin/env python3
"""Build same-identity Phase3 counterfactual pose conditions for train-only use."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from phase3.reconstruction_data import file_hash, read_ids
from scripts.build_phase3_condition_cache import (
    apply_phase2, load_source_params, write_conditions,
)


def compose_camera_yaw(pose: np.ndarray, yaw_degrees: float) -> np.ndarray:
    """Left-compose a camera-frame Y-axis rotation onto DECA global pose."""
    value = np.asarray(pose, dtype=np.float32).copy()
    base = Rotation.from_rotvec(value[:3].astype(np.float64))
    delta = Rotation.from_rotvec(np.array([0.0, math.radians(yaw_degrees), 0.0]))
    value[:3] = (delta * base).as_rotvec().astype(np.float32)
    return value


def pose_distance_degrees(left: np.ndarray, right: np.ndarray) -> float:
    a = Rotation.from_rotvec(np.asarray(left, dtype=np.float64).reshape(-1)[:3])
    b = Rotation.from_rotvec(np.asarray(right, dtype=np.float64).reshape(-1)[:3])
    return float(np.degrees((b * a.inv()).magnitude()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--deca-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--yaw-offset-deg", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.out_dir.exists():
        raise ValueError("Counterfactual output directory must be new")
    if not math.isfinite(args.yaw_offset_deg) or args.yaw_offset_deg < 5.0:
        raise ValueError("yaw-offset-deg must be finite and at least 5 degrees")

    ids = [line.strip() for line in args.ids_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Counterfactual IDs must be nonempty and unique")
    train = read_ids(args.split_dir / "train_ids.txt")
    validation = read_ids(args.split_dir / "validation_ids.txt")
    fixed = read_ids(args.split_dir / "fixed_test_ids.txt")
    if set(ids) - train or set(ids) & validation or set(ids) & fixed:
        raise ValueError("Counterfactual conditions are restricted to train IDs")

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["image_id"]): row for row in rows}
    if len(by_id) != len(rows) or set(ids) - set(by_id):
        raise ValueError("Manifest contains duplicate IDs or misses selected IDs")

    sys.path.insert(0, str(args.deca_root))
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    deca_cfg.rasterizer_type = "standard"
    deca_cfg.model.use_tex = False
    deca = DECA(config=deca_cfg, device=args.device, render_enabled=True)
    image_size = int(deca_cfg.dataset.image_size)

    args.out_dir.mkdir(parents=True)
    output = []
    for image_id in ids:
        row = by_id[image_id]
        source = load_source_params(Path(row["deca_mat"]))
        canonical = apply_phase2(source, Path(row["phase2_npz"]))
        variants = {}
        for name, offset in (("negative_yaw", -args.yaw_offset_deg), ("positive_yaw", args.yaw_offset_deg)):
            target = {key: value.copy() for key, value in canonical.items()}
            target["pose"] = compose_camera_yaw(canonical["pose"], offset)
            directory = args.out_dir / "maps" / image_id / name
            paths = write_conditions(deca, target, directory, args.device, image_size)
            npz_path = directory / "target_params.npz"
            np.savez_compressed(
                npz_path,
                pose_standardized=target["pose"],
                expression_standardized=target["expression"],
            )
            variants[name] = {
                "offset_degrees": offset,
                "phase2_npz": str(npz_path),
                "target_pose": target["pose"].tolist(),
                **{f"target_{key}_map": value for key, value in paths.items() if key != "eye_mask"},
                "target_eye_mask": paths["eye_mask"],
                "artifact_hashes": {
                    "phase2_npz": file_hash(npz_path),
                    **{key: file_hash(Path(value)) for key, value in paths.items()},
                },
            }
        separation = pose_distance_degrees(
            np.asarray(variants["negative_yaw"]["target_pose"]),
            np.asarray(variants["positive_yaw"]["target_pose"]),
        )
        if separation + 1e-4 < 2.0 * args.yaw_offset_deg:
            raise RuntimeError(f"Counterfactual separation too small: {image_id}:{separation}")
        output.append({
            "image_id": image_id,
            "identity_source_image_id": image_id,
            "source_deca_mat": row["deca_mat"],
            "canonical_phase2_npz": row["phase2_npz"],
            "pair_pose_separation_deg": separation,
            "composition": "left_multiply_camera_y_axis",
            "variants": variants,
        })

    manifest = args.out_dir / "counterfactual_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in output), encoding="utf-8")
    summary = {
        "status": "completed",
        "n_identities": len(output),
        "n_conditions": 2 * len(output),
        "yaw_offset_degrees": args.yaw_offset_deg,
        "minimum_pair_pose_separation_degrees": min(row["pair_pose_separation_deg"] for row in output),
        "train_only": True,
        "gaze_enabled": False,
        "manifest_sha256": file_hash(manifest),
        "ids_sha256": file_hash(args.ids_file),
        "source_manifest_sha256": file_hash(args.manifest),
        "split_hashes": {
            name: file_hash(args.split_dir / name)
            for name in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")
        },
        "scope": "same-identity synthetic pose controls; diagnostic/training conditions, not real multiview ground truth",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "exact_command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
