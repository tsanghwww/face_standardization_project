#!/usr/bin/env python3
"""Build a downstream condition-dataset JSONL manifest.

This is an interface skeleton. It checks paths, joins available manifests by
image_id, and writes explicit missing-field records. It does not generate depth,
normal, landmark, or mask images yet.
"""

from __future__ import annotations

import argparse
import csv
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


def load_split_map(split_dir: Path | None) -> dict[str, str]:
    if split_dir is None or not split_dir.exists():
        return {}
    out: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = split_dir / f"{split}_ids.txt"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            image_id = line.strip()
            if image_id:
                out[image_id] = split
    return out


def build_row(
    base: dict[str, str],
    phase2: dict[str, str] | None,
    split_map: dict[str, str],
    default_split: str,
) -> dict[str, Any]:
    image_id = first_present(base, ["image_id", "eval_id", "id", "file_id"])
    source_raw = first_present(base, ["source_image", "image_path", "path", "file_path", "img_path"])
    deca_raw = first_present(base, ["deca_mat", "mat_path", "deca_path"])
    source_path, source_exists = resolve_path(source_raw)
    deca_path, deca_exists = resolve_path(deca_raw)

    phase2 = phase2 or {}
    phase2_raw = first_present(phase2, ["phase2_npz", "npz_path", "standardized_npz", "output_npz"])
    phase2_path, phase2_exists = resolve_path(phase2_raw)

    missing: list[str] = []
    if not source_exists:
        missing.append("source_image")
    if not deca_exists:
        missing.append("deca_mat")
    if phase2_raw and not phase2_exists:
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

    return {
        "image_id": image_id,
        "split": split_map.get(image_id, first_present(base, ["split"]) or default_split),
        "source_image": source_path if source_exists else source_raw,
        "deca_mat": deca_path if deca_exists else deca_raw,
        "phase2_npz": phase2_path if phase2_exists else phase2_raw,
        "depth_map": "",
        "normal_map": "",
        "landmark_map": "",
        "face_mask": "",
        "arcface_embedding": first_present(base, ["arcface_embedding", "embedding_path"]),
        "gaze_pitch": parse_float(first_present(base, ["gaze_pitch", "pitch"])),
        "gaze_yaw": parse_float(first_present(base, ["gaze_yaw", "yaw"])),
        "quality_score": parse_float(first_present(phase2, ["xgb_quality_score", "quality_score"])),
        "quality_label": first_present(phase2, ["xgb_quality_label", "quality_label"]),
        "phase2_confidence": parse_float(first_present(phase2, ["confidence", "phase2_confidence"])),
        "phase2_reject_score": parse_float(first_present(phase2, ["reject_score", "phase2_reject_score"])),
        "phase2_gate_decision": first_present(phase2, ["gate_decision", "phase2_gate_decision"]),
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
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--split-dir", type=Path)
    parser.add_argument("--default-split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_rows = read_csv(args.phase1_manifest)
    phase2_rows = read_csv(args.phase2_manifest)
    phase2_by_id = {first_present(row, ["image_id", "eval_id", "id"]): row for row in phase2_rows}
    split_map = load_split_map(args.split_dir)

    out_rows = [build_row(row, phase2_by_id.get(first_present(row, ["image_id", "eval_id", "id", "file_id"])), split_map, args.default_split) for row in base_rows]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_rows = [row for row in out_rows if row["split"] == split]
        write_jsonl(args.out_dir / f"{split}.jsonl", split_rows)

    summary = {
        "dry_run": args.dry_run,
        "n_rows": len(out_rows),
        "status_counts": {status: sum(row["status"] == status for row in out_rows) for status in sorted({row["status"] for row in out_rows})},
        "split_counts": {split: sum(row["split"] == split for row in out_rows) for split in ("train", "val", "test")},
        "phase2_manifest_used": str(args.phase2_manifest) if args.phase2_manifest else "",
        "scope_note": "interface skeleton; condition maps are placeholders",
    }
    (args.out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
