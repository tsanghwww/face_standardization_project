"""Join cleaning, DECA, L2CS, and ArcFace state into one Phase 1 manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--eye-invalid-ids", required=True, type=Path)
    parser.add_argument("--p95-report", required=True, type=Path)
    parser.add_argument("--p975-report", required=True, type=Path)
    parser.add_argument("--deca-results-dir", required=True, type=Path)
    parser.add_argument("--l2cs-summary", required=True, type=Path)
    parser.add_argument("--arcface-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--hash-images", action="store_true")
    return parser.parse_args()


def read_ids(path: Path) -> set[int]:
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values.extend(re.split(r"[\s,]+", line.split("#", 1)[0].strip()))
    return {int(value) for value in values if value}


def read_json_rows(path: Path) -> dict[int, dict[str, object]]:
    return {int(row["image_id"]): row for row in json.loads(path.read_text(encoding="utf-8"))}


def read_csv_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["image_id"]): row for row in csv.DictReader(handle)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    invalid_ids = read_ids(args.eye_invalid_ids)
    p95 = read_json_rows(args.p95_report)
    p975 = read_json_rows(args.p975_report)
    l2cs = read_csv_rows(args.l2cs_summary)
    arcface = read_csv_rows(args.arcface_manifest)
    mats = {int(path.stem): path for path in args.deca_results_dir.rglob("*.mat") if path.stem.isdigit()}
    images = sorted(args.images_dir.glob("*.png"), key=lambda path: int(path.stem))

    rows: list[dict[str, object]] = []
    for index, image in enumerate(images, start=1):
        current_id = int(image.stem)
        r95 = p95.get(current_id, {})
        r975 = p975.get(current_id, {})
        gaze = l2cs.get(current_id, {})
        identity = arcface.get(current_id, {})
        eye_valid = current_id not in invalid_ids
        arc_success = identity.get("arcface_status") == "success"
        p95_pass = r95.get("label") == "Pass"
        rows.append(
            {
                "image_id": current_id,
                "image_path": str(image),
                "image_sha256": sha256(image) if args.hash_images else "",
                "eye_valid": str(eye_valid).lower(),
                "p95_label": r95.get("label", "Missing"),
                "p95_D2": r95.get("D2", ""),
                "p975_label": r975.get("label", "Missing"),
                "p975_D2": r975.get("D2", ""),
                "deca_status": "success" if current_id in mats else "missing",
                "deca_mat_path": str(mats.get(current_id, "")),
                "l2cs_status": gaze.get("status", "missing"),
                "pitch": gaze.get("pitch", ""),
                "yaw": gaze.get("yaw", ""),
                "gaze_x": gaze.get("gaze_x", ""),
                "gaze_y": gaze.get("gaze_y", ""),
                "gaze_z": gaze.get("gaze_z", ""),
                "arcface_status": identity.get("arcface_status", "missing"),
                "arcface_embedding_path": identity.get("embedding_path", ""),
                "arcface_detector_score": identity.get("detector_score", ""),
                "arcface_stage": identity.get("arcface_stage", "main" if arc_success else ""),
                "use_for_train_strict": identity.get(
                    "use_for_train_strict", str(eye_valid and p95_pass and arc_success).lower()
                ),
                "use_for_train_full": identity.get(
                    "use_for_train_full", str(eye_valid and p95_pass and arc_success).lower()
                ),
            }
        )
        if index % 500 == 0:
            print(f"[{index}/{len(images)}]", flush=True)

    manifest_path = args.output_dir / "phase1_master_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_images": len(rows),
        "unique_image_ids": len({row["image_id"] for row in rows}),
        "eye_invalid": sum(row["eye_valid"] == "false" for row in rows),
        "p95_counts": {label: sum(row["p95_label"] == label for row in rows) for label in ["Pass", "Warn", "Missing"]},
        "p975_counts": {label: sum(row["p975_label"] == label for row in rows) for label in ["Pass", "Warn", "Missing"]},
        "deca_success": sum(row["deca_status"] == "success" for row in rows),
        "l2cs_success": sum(row["l2cs_status"] == "success" for row in rows),
        "arcface_success": sum(row["arcface_status"] == "success" for row in rows),
        "strict_train": sum(row["use_for_train_strict"] == "true" for row in rows),
        "full_train": sum(row["use_for_train_full"] == "true" for row in rows),
        "hashes_recorded": args.hash_images,
    }
    (args.output_dir / "phase1_master_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
