#!/usr/bin/env python3
"""Placeholder identity-preservation evaluator for downstream outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--generated-dir", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = read_jsonl(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.out_dir / "identity_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "source_image", "generated_image", "arcface_cosine", "status", "note"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "image_id": row.get("image_id", ""),
                "source_image": row.get("source_image", ""),
                "generated_image": "",
                "arcface_cosine": "",
                "status": "not_computed",
                "note": "TODO: run ArcFace on generated output and compare with source",
            })

    summary = {
        "n_rows": len(rows),
        "metric": "ArcFace cosine",
        "status": "placeholder",
        "scope_note": "identity evaluator entry point only; no ArcFace inference executed",
    }
    (args.out_dir / "identity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
