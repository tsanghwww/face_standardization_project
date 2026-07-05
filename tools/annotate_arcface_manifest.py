"""Annotate ArcFace manifest with strict and retry-recovered training flags."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path


EXTRA_FIELDS = [
    "arcface_stage",
    "det_thresh",
    "use_for_train_strict",
    "use_for_train_full",
]


SIGNATURE = "ARCFACE-IDENTITY-EXTRACTION-ZENG-HAORONG-2026-06-12"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", required=True, type=Path)
    parser.add_argument("--retry-dir", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.main_dir / "arcface_manifest.csv"
    rows = read_rows(manifest_path)
    retry_rows = read_rows(args.retry_dir / "arcface_manifest.csv")
    retry_ids = {row["image_id"] for row in retry_rows if row.get("arcface_status") == "success"}

    base_fields = [name for name in rows[0].keys() if name not in EXTRA_FIELDS]
    fieldnames = base_fields + EXTRA_FIELDS

    for row in rows:
        is_success = row.get("arcface_status") == "success"
        is_pass = row.get("deca_clean_label") == "Pass"
        is_retry = row["image_id"] in retry_ids
        row["arcface_stage"] = "retry_recovered" if is_retry else "main"
        row["det_thresh"] = "0.05" if is_retry else "0.1"
        row["use_for_train_strict"] = "true" if is_success and is_pass and not is_retry else "false"
        row["use_for_train_full"] = "true" if is_success and is_pass else "false"
        row["use_for_train"] = row["use_for_train_full"]

    write_rows(manifest_path, rows, fieldnames)

    strict_count = sum(1 for row in rows if row["use_for_train_strict"] == "true")
    full_count = sum(1 for row in rows if row["use_for_train_full"] == "true")
    retry_pass = sum(
        1
        for row in rows
        if row["arcface_stage"] == "retry_recovered" and row["deca_clean_label"] == "Pass"
    )
    retry_warn = sum(
        1
        for row in rows
        if row["arcface_stage"] == "retry_recovered" and row["deca_clean_label"] == "Warn"
    )
    summary = {
        "signature": SIGNATURE,
        "author": "ZENG HAORONG",
        "date": str(date.today()),
        "total_images_seen": len(rows),
        "success": sum(1 for row in rows if row.get("arcface_status") == "success"),
        "fail": sum(1 for row in rows if row.get("arcface_status") != "success"),
        "pass_success": sum(
            1
            for row in rows
            if row.get("arcface_status") == "success" and row.get("deca_clean_label") == "Pass"
        ),
        "warn_success": sum(
            1
            for row in rows
            if row.get("arcface_status") == "success" and row.get("deca_clean_label") == "Warn"
        ),
        "main_success": sum(1 for row in rows if row["arcface_stage"] == "main"),
        "retry_recovered_success": len(retry_ids),
        "retry_recovered_pass": retry_pass,
        "retry_recovered_warn": retry_warn,
        "use_for_train_strict": strict_count,
        "use_for_train_full": full_count,
        "strict_definition": "deca_clean_label == Pass AND arcface_status == success AND arcface_stage == main",
        "full_definition": "deca_clean_label == Pass AND arcface_status == success",
    }
    (args.main_dir / "arcface_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    md = f"""# ArcFace Identity Embedding Run

Signature: `{SIGNATURE}`

Author / rights marker: `ZENG HAORONG`

Date: `{date.today()}`

## Input Scope

- Image input directory: `H:\\face_standardization_project\\datasets\\my_faces`
- Current image count: `9990`
- Excluded eye-invalid IDs are not present in the input directory.
- ArcFace was run on original RGB face images only, not on DECA normal/depth/render outputs.

## Model And Runtime

- Model package: InsightFace `buffalo_l`
- Recognition model: `w600k_r50.onnx`
- Main detector setting: `det_size=640`, `det_thresh=0.1`
- Retry detector setting: `det_size=640`, `det_thresh=0.05`
- ONNXRuntime provider used: `CPUExecutionProvider`

The machine has an RTX 2060, but the installed ONNXRuntime package did not expose `CUDAExecutionProvider`.
The run therefore used CPU inference to avoid changing the existing CUDA/PyTorch environment.

## Why There Are Two Training Flags

The retry pass is not treated as the strict experimental default. Lowering the detector threshold can recover
low-confidence but valid faces, but it can also increase false-positive risk if used without controls.

For that reason the manifest contains two explicit training flags:

- `use_for_train_strict`: `Pass` samples detected in the main run only (`det_thresh=0.1`)
- `use_for_train_full`: all `Pass` samples with successful ArcFace embedding, including retry-recovered samples

## Final Counts

- Total images: `{len(rows)}`
- ArcFace success: `{summary["success"]}`
- ArcFace fail: `{summary["fail"]}`
- Main-run success: `{summary["main_success"]}`
- Retry-recovered success: `{summary["retry_recovered_success"]}`
- Retry-recovered Pass: `{retry_pass}`
- Retry-recovered Warn: `{retry_warn}`
- Strict train count: `{strict_count}`
- Full train count: `{full_count}`

## Recommended Usage

Use `use_for_train_strict == true` for the main experiment.
Use `use_for_train_full == true` only for a sensitivity / completeness run after visually checking retry-recovered aligned crops.

The `arcface_stage`, `det_thresh`, and `detector_score` fields must be preserved in downstream manifests.
"""
    (args.main_dir / "ARCFACE_IDENTITY_EXTRACTION_README.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
