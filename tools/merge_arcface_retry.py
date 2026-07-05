"""Merge a retry ArcFace run back into the main ArcFace output directory."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


FIELD = "image_id"


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

    main_manifest = args.main_dir / "arcface_manifest.csv"
    retry_manifest = args.retry_dir / "arcface_manifest.csv"
    main_rows = read_rows(main_manifest)
    retry_rows = read_rows(retry_manifest)
    fieldnames = list(main_rows[0].keys())

    retry_success = {
        row[FIELD]: row for row in retry_rows if row.get("arcface_status") == "success"
    }

    for image_id in retry_success:
        for subdir, suffix in [("embeddings", ".npy"), ("aligned_faces", ".jpg")]:
            src = args.retry_dir / subdir / f"{image_id}{suffix}"
            dst = args.main_dir / subdir / f"{image_id}{suffix}"
            if src.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    merged_rows: list[dict[str, str]] = []
    for row in main_rows:
        replacement = retry_success.get(row[FIELD])
        if replacement:
            merged_rows.append(replacement)
        else:
            merged_rows.append(row)

    write_rows(main_manifest, merged_rows, fieldnames)

    failures = [row for row in merged_rows if row.get("arcface_status") != "success"]
    write_rows(args.main_dir / "arcface_failures.csv", failures, fieldnames)

    counts = {
        "total_images_seen": len(merged_rows),
        "success": sum(1 for row in merged_rows if row.get("arcface_status") == "success"),
        "fail": sum(1 for row in merged_rows if row.get("arcface_status") != "success"),
        "pass_success": sum(
            1
            for row in merged_rows
            if row.get("arcface_status") == "success" and row.get("deca_clean_label") == "Pass"
        ),
        "warn_success": sum(
            1
            for row in merged_rows
            if row.get("arcface_status") == "success" and row.get("deca_clean_label") == "Warn"
        ),
        "pass_fail": sum(
            1
            for row in merged_rows
            if row.get("arcface_status") != "success" and row.get("deca_clean_label") == "Pass"
        ),
        "warn_fail": sum(
            1
            for row in merged_rows
            if row.get("arcface_status") != "success" and row.get("deca_clean_label") == "Warn"
        ),
        "final_use_for_train": sum(1 for row in merged_rows if row.get("use_for_train") == "true"),
        "merged_retry_success": len(retry_success),
    }
    (args.main_dir / "arcface_summary.json").write_text(
        json.dumps(counts, indent=2), encoding="utf-8"
    )
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
