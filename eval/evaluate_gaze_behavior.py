#!/usr/bin/env python3
"""Placeholder gaze-behavior evaluator for downstream outputs."""

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

    metrics_path = args.out_dir / "gaze_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "source_gaze_pitch", "source_gaze_yaw", "output_gaze_pitch", "output_gaze_yaw", "gaze_delta_deg", "status", "note"])
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "image_id": row.get("image_id", ""),
                "source_gaze_pitch": row.get("gaze_pitch", ""),
                "source_gaze_yaw": row.get("gaze_yaw", ""),
                "output_gaze_pitch": "",
                "output_gaze_yaw": "",
                "gaze_delta_deg": "",
                "status": "not_computed",
                "note": "TODO: run L2CS on generated output; report gaze behavior, not gaze disentanglement",
            })

    summary = {
        "n_rows": len(rows),
        "metric": "L2CS gaze delta",
        "status": "placeholder",
        "scope_note": "gaze behavior only; no claim of gaze disentanglement",
    }
    (args.out_dir / "gaze_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
