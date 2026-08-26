"""Merge base and external ArcFace rows for the immutable Phase2 test split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT = Path(r"D:\face_standardization_project")
FIELDS = ["image_id", "eval_id", "source_group", "arcface_status", "detector_score", "failure_reason"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-manifest",
        type=Path,
        default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv",
    )
    parser.add_argument(
        "--base-arcface",
        type=Path,
        default=PROJECT / "results" / "arcface_p95_rebuilt" / "arcface_manifest.csv",
    )
    parser.add_argument(
        "--external-arcface",
        type=Path,
        default=PROJECT / "results" / "phase2_arcface_external_20260824" / "arcface_external_manifest.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "arcface_fixed_test_manifest.csv",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    test_rows = read_rows(args.test_manifest)
    base = {row["image_id"]: row for row in read_rows(args.base_arcface) if row.get("image_id")}
    external = {row["image_id"]: row for row in read_rows(args.external_arcface) if row.get("image_id")}

    output_rows: list[dict[str, str]] = []
    for test in test_rows:
        image_id = test["image_id"]
        source = base.get(image_id) if test.get("source_dataset") == "stylegan2_base" else external.get(image_id)
        source = source or {}
        output_rows.append(
            {
                "image_id": image_id,
                "eval_id": test["eval_id"],
                "source_group": test["source_group"],
                "arcface_status": source.get("arcface_status", "missing"),
                "detector_score": source.get("detector_score", ""),
                "failure_reason": source.get("failure_reason", "arcface_row_missing" if not source else ""),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    success = sum(row["arcface_status"] == "success" for row in output_rows)
    print(f"rows={len(output_rows)} success={success} failure={len(output_rows) - success} output={args.output}")


if __name__ == "__main__":
    main()
