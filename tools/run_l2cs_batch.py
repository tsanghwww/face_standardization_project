"""Run resumable L2CS-Net gaze extraction with the historical Phase 1 schema."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
from l2cs import Pipeline


CSV_FIELDS = ["image_id", "pitch", "yaw", "gaze_x", "gaze_y", "gaze_z", "status"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arch", default="ResNet50")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", default=0.5, type=float)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    return parser.parse_args()


def image_id(path: Path) -> int:
    return int(path.stem)


def gaze_vector(pitch: float, yaw: float) -> tuple[float, float, float]:
    return (
        -math.sin(yaw) * math.cos(pitch),
        -math.sin(pitch),
        -math.cos(yaw) * math.cos(pitch),
    )


def load_completed(progress_path: Path) -> set[int]:
    if not progress_path.exists():
        return set()
    data = json.loads(progress_path.read_text(encoding="utf-8"))
    return {int(value) for value in data.get("completed", [])}


def write_progress(path: Path, completed: set[int], failed: dict[int, str]) -> None:
    path.write_text(
        json.dumps(
            {"completed": sorted(completed), "failed": {str(k): v for k, v in sorted(failed.items())}},
            indent=2,
        ),
        encoding="utf-8",
    )


def largest_face_index(bboxes: np.ndarray) -> int:
    widths = np.maximum(0.0, bboxes[:, 2] - bboxes[:, 0])
    heights = np.maximum(0.0, bboxes[:, 3] - bboxes[:, 1])
    return int(np.argmax(widths * heights))


def main() -> None:
    args = parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / ".gaze_progress.json"
    summary_path = args.output_dir / "l2cs_gaze_summary_10k.csv"
    config_path = args.output_dir / "l2cs_gaze_config.json"

    paths = sorted(args.images_dir.glob("*.png"), key=image_id)
    if args.limit:
        paths = paths[: args.limit]
    completed = load_completed(progress_path) if args.resume else set()
    failed: dict[int, str] = {}

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    pipeline = Pipeline(
        weights=args.weights,
        arch=args.arch,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )
    config_path.write_text(
        json.dumps(
            {
                "images_dir": str(args.images_dir),
                "weights": str(args.weights),
                "arch": args.arch,
                "device": str(device),
                "confidence_threshold": args.confidence_threshold,
                "image_count": len(paths),
                "resume": args.resume,
                "started_unix": started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows: dict[int, dict[str, str]] = {}
    if args.resume and summary_path.exists():
        with summary_path.open("r", encoding="utf-8", newline="") as handle:
            rows = {int(row["image_id"]): row for row in csv.DictReader(handle)}

    for index, path in enumerate(paths, start=1):
        current_id = image_id(path)
        if current_id in completed:
            continue
        payload: dict[str, object] = {"image_id": current_id, "status": "fail"}
        row = {field: "" for field in CSV_FIELDS}
        row.update({"image_id": str(current_id), "status": "fail"})
        try:
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError("image_read_failed")
            result = pipeline.step(frame)
            bboxes = np.asarray(result.bboxes)
            if bboxes.size == 0:
                raise ValueError("no_face_detected")
            selected = largest_face_index(bboxes)
            pitch = float(np.asarray(result.pitch).reshape(-1)[selected])
            yaw = float(np.asarray(result.yaw).reshape(-1)[selected])
            vector = gaze_vector(pitch, yaw)
            payload.update(
                {
                    "pitch": round(pitch, 6),
                    "yaw": round(yaw, 6),
                    "gaze_vector": [round(value, 6) for value in vector],
                    "status": "success",
                }
            )
            row.update(
                {
                    "pitch": f"{pitch:.6f}",
                    "yaw": f"{yaw:.6f}",
                    "gaze_x": f"{vector[0]:.6f}",
                    "gaze_y": f"{vector[1]:.6f}",
                    "gaze_z": f"{vector[2]:.6f}",
                    "status": "success",
                }
            )
            completed.add(current_id)
        except Exception as exc:  # pragma: no cover - runtime/model failures are recorded
            reason = f"{type(exc).__name__}:{exc}"
            payload["failure_reason"] = reason
            failed[current_id] = reason

        (args.output_dir / f"{current_id}_gaze.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        rows[current_id] = row

        if index == 1 or index % 100 == 0:
            with summary_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows[key] for key in sorted(rows))
            write_progress(progress_path, completed, failed)
            elapsed = time.time() - started
            print(
                f"[{index}/{len(paths)}] success={len(completed)} fail={len(failed)} "
                f"rate={index / max(elapsed, 1e-6):.2f}/s",
                flush=True,
            )

    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows[key] for key in sorted(rows))
    write_progress(progress_path, completed, failed)
    print(json.dumps({"total": len(paths), "success": len(completed), "fail": len(failed)}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        output_dir: Path | None = None
        if "--output-dir" in sys.argv:
            index = sys.argv.index("--output-dir")
            if index + 1 < len(sys.argv):
                output_dir = Path(sys.argv[index + 1])
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "fatal_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
