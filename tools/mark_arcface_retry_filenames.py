"""Rename retry-recovered ArcFace artifacts so provenance is visible in filenames."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


RETRY_SUFFIX = "_retry_det005"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def copy_retry_artifact(src_text: str, fallback: Path, dst: Path) -> None:
    src = Path(src_text) if src_text else fallback
    if not src.exists():
        src = fallback
    if not src.exists():
        raise FileNotFoundError(f"Missing retry artifact source: {src_text} / {fallback}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--main-dir", required=True, type=Path)
    args = parser.parse_args()

    main_dir = args.main_dir
    manifest_path = main_dir / "arcface_manifest.csv"
    rows = read_rows(manifest_path)
    fieldnames = list(rows[0].keys())

    retry_count = 0
    for row in rows:
        image_id = row["image_id"]
        if row.get("arcface_stage") != "retry_recovered":
            row["embedding_path"] = str(main_dir / "embeddings" / f"{image_id}.npy")
            row["aligned_face_path"] = str(main_dir / "aligned_faces" / f"{image_id}.jpg")
            continue

        retry_count += 1
        retry_stem = f"{image_id}{RETRY_SUFFIX}"
        new_embedding = main_dir / "embeddings" / f"{retry_stem}.npy"
        new_aligned = main_dir / "aligned_faces" / f"{retry_stem}.jpg"
        old_main_embedding = main_dir / "embeddings" / f"{image_id}.npy"
        old_main_aligned = main_dir / "aligned_faces" / f"{image_id}.jpg"

        copy_retry_artifact(row.get("embedding_path", ""), old_main_embedding, new_embedding)
        copy_retry_artifact(row.get("aligned_face_path", ""), old_main_aligned, new_aligned)

        if old_main_embedding.exists() and old_main_embedding.resolve() != new_embedding.resolve():
            old_main_embedding.unlink()
        if old_main_aligned.exists() and old_main_aligned.resolve() != new_aligned.resolve():
            old_main_aligned.unlink()

        row["embedding_path"] = str(new_embedding)
        row["aligned_face_path"] = str(new_aligned)

    write_rows(manifest_path, rows, fieldnames)

    summary_path = main_dir / "arcface_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["retry_filename_suffix"] = RETRY_SUFFIX
    summary["retry_files_renamed"] = retry_count
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme_path = main_dir / "ARCFACE_IDENTITY_EXTRACTION_README.md"
    readme = readme_path.read_text(encoding="utf-8")
    addition = f"""

## Retry File Naming

Retry-recovered artifacts are explicitly marked in their filenames with `{RETRY_SUFFIX}`.

Examples:

- `embeddings\\707{RETRY_SUFFIX}.npy`
- `aligned_faces\\707{RETRY_SUFFIX}.jpg`

Main-run artifacts keep the plain `image_id` filename, for example `embeddings\\0.npy`.
Downstream code should use `arcface_manifest.csv` paths rather than reconstructing paths from image IDs.
"""
    if "## Retry File Naming" not in readme:
        readme_path.write_text(readme.rstrip() + addition, encoding="utf-8")

    print(json.dumps({"retry_files_renamed": retry_count, "suffix": RETRY_SUFFIX}, indent=2))


if __name__ == "__main__":
    main()
