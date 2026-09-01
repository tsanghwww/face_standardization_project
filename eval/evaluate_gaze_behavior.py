#!/usr/bin/env python3
"""Gaze-disentanglement evaluation interface for downstream outputs.

This stage converts available L2CS angles to camera-frame vectors. Output-image
inference and head-local gaze require future L2CS and head-rotation measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .gaze_geometry import l2cs_angles_to_camera_vector
except ImportError:  # Direct script execution.
    from gaze_geometry import l2cs_angles_to_camera_vector


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_float(value) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


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
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "source_gaze_pitch",
                "source_gaze_yaw",
                "source_gaze_camera_x",
                "source_gaze_camera_y",
                "source_gaze_camera_z",
                "source_gaze_head_x",
                "source_gaze_head_y",
                "source_gaze_head_z",
                "output_gaze_camera_x",
                "output_gaze_camera_y",
                "output_gaze_camera_z",
                "output_gaze_head_x",
                "output_gaze_head_y",
                "output_gaze_head_z",
                "camera_gaze_delta_deg",
                "head_local_gaze_error_deg",
                "head_to_gaze_leakage_deg",
                "gaze_to_head_leakage_deg",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        for row in rows:
            pitch = parse_float(row.get("gaze_pitch"))
            yaw = parse_float(row.get("gaze_yaw"))
            camera = l2cs_angles_to_camera_vector(pitch, yaw) if pitch is not None and yaw is not None else None
            writer.writerow({
                "image_id": row.get("image_id", ""),
                "source_gaze_pitch": pitch if pitch is not None else "",
                "source_gaze_yaw": yaw if yaw is not None else "",
                "source_gaze_camera_x": camera[0] if camera is not None else "",
                "source_gaze_camera_y": camera[1] if camera is not None else "",
                "source_gaze_camera_z": camera[2] if camera is not None else "",
                "source_gaze_head_x": "",
                "source_gaze_head_y": "",
                "source_gaze_head_z": "",
                "output_gaze_camera_x": "",
                "output_gaze_camera_y": "",
                "output_gaze_camera_z": "",
                "output_gaze_head_x": "",
                "output_gaze_head_y": "",
                "output_gaze_head_z": "",
                "camera_gaze_delta_deg": "",
                "head_local_gaze_error_deg": "",
                "head_to_gaze_leakage_deg": "",
                "gaze_to_head_leakage_deg": "",
                "status": "source_geometry_ready" if camera is not None else "missing_source_gaze",
                "note": "TODO: estimate output gaze and source/output head rotation, then compute head-local gaze and intervention leakage",
            })

    summary = {
        "n_rows": len(rows),
        "metric": "camera-frame gaze, head-local gaze preservation, and bidirectional intervention leakage",
        "status": "source geometry only; output disentanglement metrics not computed",
        "scope_note": "the protocol targets gaze/head-pose disentanglement; this skeleton is not evidence that disentanglement has been achieved",
    }
    (args.out_dir / "gaze_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
