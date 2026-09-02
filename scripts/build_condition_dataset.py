#!/usr/bin/env python3
"""Build a downstream condition-dataset JSONL manifest.

This is an interface skeleton. It checks paths, joins available manifests by
image_id, and writes explicit missing-field records. It does not generate depth,
normal, landmark, or mask images yet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]


def read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None:
        return []
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def first_present(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value:
            return value
    return ""


def parse_float(value: str) -> float | None:
    if value in ("", "None", "null", None):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_path(value: str) -> tuple[str, bool]:
    if not value:
        return "", False
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT / path
    return str(path), path.exists()


def load_split_map(split_dir: Path | None, split_file: Path | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if split_dir is not None:
        if not split_dir.exists():
            raise SystemExit(f"Split directory not found: {split_dir}")
        split_files = {
            "train": ("train_ids.txt",),
            "val": ("val_ids.txt", "validation_ids.txt"),
            "test": ("test_ids.txt", "fixed_test_base_ids.txt"),
        }
        for split, names in split_files.items():
            for name in names:
                path = split_dir / name
                if not path.exists():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    image_id = line.strip()
                    if image_id:
                        out[image_id] = split

    if split_file is not None:
        if not split_file.exists():
            raise SystemExit(f"Split file not found: {split_file}")
        payload = json.loads(split_file.read_text(encoding="utf-8"))
        for split in ("train", "val", "test"):
            for image_id in payload.get(split, []):
                out[str(image_id)] = split
    return out


def load_include_ids(path: Path | None) -> tuple[list[str], str]:
    if path is None:
        return [], ""
    if not path.exists():
        raise SystemExit(f"Include IDs file not found: {path}")
    content = path.read_bytes()
    image_ids = [line.strip() for line in content.decode("utf-8-sig").splitlines() if line.strip()]
    if len(image_ids) != len(set(image_ids)):
        raise SystemExit(f"Include IDs file contains duplicates: {path}")
    return image_ids, hashlib.sha256(content).hexdigest()


def build_row(
    base: dict[str, str],
    phase2: dict[str, str] | None,
    condition_cache: dict[str, str] | None,
    split_map: dict[str, str],
    default_split: str,
    gaze_policy: str,
) -> dict[str, Any]:
    image_id = first_present(base, ["image_id", "eval_id", "id", "file_id"])
    source_raw = first_present(base, ["source_image", "image_path", "path", "file_path", "img_path"])
    deca_raw = first_present(base, ["deca_mat", "deca_mat_path", "mat_path", "deca_path"])
    source_path, source_exists = resolve_path(source_raw)
    deca_path, deca_exists = resolve_path(deca_raw)

    phase2 = phase2 or {}
    condition_cache = condition_cache or {}
    phase2_raw = first_present(phase2, ["phase2_npz", "out_npz", "npz_path", "standardized_npz", "output_npz"])
    phase2_path, phase2_exists = resolve_path(phase2_raw)

    missing: list[str] = []
    if not source_exists:
        missing.append("source_image")
    if not deca_exists:
        missing.append("deca_mat")
    if phase2 and not phase2_exists:
        missing.append("phase2_npz")
    if not phase2:
        missing.append("phase2_row")

    if not source_exists:
        status = "missing_source"
    elif not deca_exists:
        status = "missing_deca"
    elif "phase2_row" in missing or "phase2_npz" in missing:
        status = "phase2_missing"
    else:
        status = "available"

    cache_columns = {
        "source_depth_map": "source_depth",
        "source_normal_map": "source_normal",
        "source_landmark_map": "source_landmark",
        "source_face_mask": "source_face_mask",
        "source_eye_mask": "source_eye_mask",
        "target_depth_map": "target_depth",
        "target_normal_map": "target_normal",
        "target_landmark_map": "target_landmark",
        "target_face_mask": "target_face_mask",
        "target_eye_mask": "target_eye_mask",
        "target_gaze_heatmap": "target_gaze_heatmap",
    }
    coordinate_status = first_present(condition_cache, ["coordinate_status"]) or "pending_head_rotation"
    modality_fields: dict[str, str] = {}
    for name, column in cache_columns.items():
        raw = first_present(condition_cache, [column])
        if name == "target_gaze_heatmap" and coordinate_status != "approved":
            raw = ""
        resolved, exists = resolve_path(raw)
        modality_fields[name] = resolved if exists else ""
        if raw and not exists:
            missing.append(name)
    modalities_todo = [name for name, value in modality_fields.items() if not value]
    target_aliases = {
        "depth_map": modality_fields["target_depth_map"],
        "normal_map": modality_fields["target_normal_map"],
        "landmark_map": modality_fields["target_landmark_map"],
        "face_mask": modality_fields["target_face_mask"],
        "eye_mask": modality_fields["target_eye_mask"],
        "gaze_heatmap": modality_fields["target_gaze_heatmap"],
    }
    arcface_raw = first_present(base, ["arcface_embedding", "arcface_embedding_path", "embedding_path"])
    arcface_path, arcface_exists = resolve_path(arcface_raw)
    if arcface_raw and not arcface_exists:
        missing.append("arcface_embedding")
    gaze_values = {
        name: parse_float(first_present(condition_cache, [name])) if coordinate_status == "approved" else None
        for name in (
            "gaze_head_x", "gaze_head_y", "gaze_head_z",
            "target_gaze_head_x", "target_gaze_head_y", "target_gaze_head_z",
        )
    }

    return {
        "image_id": image_id,
        "split": split_map.get(image_id, first_present(base, ["split"]) or default_split),
        "source_image": source_path if source_exists else source_raw,
        "source_image_exists": source_exists,
        "deca_mat": deca_path if deca_exists else deca_raw,
        "deca_mat_exists": deca_exists,
        "phase2_npz": phase2_path if phase2_exists else phase2_raw,
        "phase2_npz_exists": phase2_exists,
        **{name: value or None for name, value in modality_fields.items()},
        **{name: value or None for name, value in target_aliases.items()},
        "modalities_todo": modalities_todo,
        "arcface_embedding": arcface_path if arcface_exists else arcface_raw,
        "arcface_embedding_exists": arcface_exists,
        "condition_cache_status": first_present(condition_cache, ["status"]),
        "gaze_pitch": parse_float(first_present(base, ["gaze_pitch", "pitch"])),
        "gaze_yaw": parse_float(first_present(base, ["gaze_yaw", "yaw"])),
        "gaze_camera_x": parse_float(first_present(base, ["gaze_camera_x", "gaze_x"])),
        "gaze_camera_y": parse_float(first_present(base, ["gaze_camera_y", "gaze_y"])),
        "gaze_camera_z": parse_float(first_present(base, ["gaze_camera_z", "gaze_z"])),
        **gaze_values,
        "gaze_policy": gaze_policy,
        "gaze_coordinate_status": coordinate_status,
        "alpha_expression": parse_float(first_present(phase2, ["alpha_expression"])),
        "alpha_head_pose": parse_float(first_present(phase2, ["alpha_head_pose"])),
        "alpha_jaw_pose": parse_float(first_present(phase2, ["alpha_jaw_pose"])),
        "standardized_exp_norm": parse_float(first_present(phase2, ["standardized_exp_norm"])),
        "standardized_head_pose_norm": parse_float(first_present(phase2, ["standardized_head_pose_norm"])),
        "standardized_jaw_pose_norm": parse_float(first_present(phase2, ["standardized_jaw_pose_norm"])),
        "quality_score": parse_float(first_present(phase2, ["xgb_quality_score", "quality_score"])),
        "quality_label": first_present(phase2, ["xgb_quality_label", "quality_label"]),
        "phase2_confidence": parse_float(first_present(phase2, ["confidence", "phase2_confidence"])),
        "phase2_reject_score": parse_float(first_present(phase2, ["reject_score", "phase2_reject_score"])),
        "phase2_gate_decision": first_present(phase2, ["decision", "gate_decision", "phase2_gate_decision"]),
        "rescue_source": False,
        "status": status,
        "missing_fields": missing,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1-manifest", required=True, type=Path)
    parser.add_argument("--phase2-manifest", type=Path)
    parser.add_argument("--condition-cache-manifest", type=Path)
    parser.add_argument("--include-ids-file", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--split-dir", type=Path)
    split_group.add_argument("--split-file", type=Path, help="JSON object containing train/val/test ID lists")
    parser.add_argument("--default-split", default="train", choices=["train", "val", "test"])
    parser.add_argument(
        "--gaze-policy",
        default="preserve_eye_in_head",
        choices=["preserve_eye_in_head", "canonical_camera_gaze", "controlled_head_local"],
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_rows = read_csv(args.phase1_manifest)
    include_ids, include_ids_sha256 = load_include_ids(args.include_ids_file)
    if include_ids:
        base_by_id = {
            first_present(row, ["image_id", "eval_id", "id", "file_id"]): row
            for row in base_rows
        }
        missing_ids = set(include_ids) - set(base_by_id)
        if missing_ids:
            raise SystemExit(f"Include IDs absent from Phase1 manifest: {sorted(missing_ids)[:10]}")
        base_rows = [base_by_id[image_id] for image_id in include_ids]
    phase2_rows = read_csv(args.phase2_manifest)
    condition_rows = read_csv(args.condition_cache_manifest)
    phase2_by_id = {first_present(row, ["image_id", "eval_id", "id"]): row for row in phase2_rows}
    condition_by_id = {first_present(row, ["image_id", "eval_id", "id"]): row for row in condition_rows}
    split_map = load_split_map(args.split_dir, args.split_file)

    out_rows = [
        build_row(
            row,
            phase2_by_id.get(first_present(row, ["image_id", "eval_id", "id", "file_id"])),
            condition_by_id.get(first_present(row, ["image_id", "eval_id", "id", "file_id"])),
            split_map,
            args.default_split,
            args.gaze_policy,
        )
        for row in base_rows
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_rows = [row for row in out_rows if row["split"] == split]
        write_jsonl(args.out_dir / f"{split}.jsonl", split_rows)

    summary = {
        "dry_run": args.dry_run,
        "n_rows": len(out_rows),
        "n_total": len(out_rows),
        "status_counts": {status: sum(row["status"] == status for row in out_rows) for status in sorted({row["status"] for row in out_rows})},
        "split_counts": {split: sum(row["split"] == split for row in out_rows) for split in ("train", "val", "test")},
        "missing_field_counts": {
            field: sum(field in row["missing_fields"] for row in out_rows)
            for field in sorted({field for row in out_rows for field in row["missing_fields"]})
        },
        "phase2_manifest_used": str(args.phase2_manifest) if args.phase2_manifest else "",
        "gaze_policy": args.gaze_policy,
        "condition_cache_manifest_used": str(args.condition_cache_manifest) if args.condition_cache_manifest else "",
        "include_ids_file": str(args.include_ids_file) if args.include_ids_file else "",
        "include_ids_count": len(include_ids) if args.include_ids_file else None,
        "include_ids_sha256": include_ids_sha256,
        "scope_note": "condition maps remain null unless a generated cache is supplied",
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
