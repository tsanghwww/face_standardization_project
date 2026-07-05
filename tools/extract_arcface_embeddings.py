"""Extract ArcFace identity embeddings for the face standardization dataset.

This script is designed for the Windows project layout:
H:\\face_standardization_project\\datasets\\my_faces
H:\\face_standardization_project\\results\\screening_v3_p95\\screening_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


FIELDNAMES = [
    "image_id",
    "image_path",
    "embedding_path",
    "aligned_face_path",
    "arcface_status",
    "embedding_dim",
    "embedding_norm",
    "detector_score",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "landmark_left_eye_x",
    "landmark_left_eye_y",
    "landmark_right_eye_x",
    "landmark_right_eye_y",
    "landmark_nose_x",
    "landmark_nose_y",
    "landmark_mouth_left_x",
    "landmark_mouth_left_y",
    "landmark_mouth_right_x",
    "landmark_mouth_right_y",
    "deca_clean_label",
    "eye_valid",
    "use_for_train",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--screening-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-name", default="buffalo_l")
    parser.add_argument("--det-size", default=640, type=int)
    parser.add_argument("--det-thresh", default=None, type=float)
    parser.add_argument("--ctx-id", default=-1, type=int, help="-1 for CPU, 0 for first GPU provider if available.")
    parser.add_argument("--save-aligned", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    return parser.parse_args()


def read_screening_labels(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    labels: dict[int, str] = {}
    for row in rows:
        labels[int(row["image_id"])] = str(row["label"])
    return labels


def image_id_from_path(path: Path) -> int:
    return int(path.stem)


def largest_face(faces):
    def area(face) -> float:
        x1, y1, x2, y2 = face.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))

    return max(faces, key=area)


def safe_float(value) -> str:
    if value is None:
        return ""
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.8g}"


def crop_face(image: np.ndarray, bbox: np.ndarray, margin: float = 0.2) -> np.ndarray | None:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    x1 = max(0, int(round(x1 - bw * margin)))
    y1 = max(0, int(round(y1 - bh * margin)))
    x2 = min(w, int(round(x2 + bw * margin)))
    y2 = min(h, int(round(y2 + bh * margin)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def empty_row(image_id: int, image_path: Path, label: str, reason: str) -> dict[str, str]:
    row = {key: "" for key in FIELDNAMES}
    row.update(
        {
            "image_id": str(image_id),
            "image_path": str(image_path),
            "arcface_status": "fail",
            "deca_clean_label": label,
            "eye_valid": "true",
            "use_for_train": "false",
            "failure_reason": reason,
        }
    )
    return row


def main() -> None:
    args = parse_args()
    started = time.time()

    images_dir: Path = args.images_dir
    output_dir: Path = args.output_dir
    embeddings_dir = output_dir / "embeddings"
    aligned_dir = output_dir / "aligned_faces"
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    if args.save_aligned:
        aligned_dir.mkdir(parents=True, exist_ok=True)

    labels = read_screening_labels(args.screening_report)
    image_paths = sorted(images_dir.glob("*.png"), key=image_id_from_path)
    if args.limit:
        image_paths = image_paths[: args.limit]

    manifest_path = output_dir / "arcface_manifest.csv"
    failures_path = output_dir / "arcface_failures.csv"
    summary_path = output_dir / "arcface_summary.json"
    config_path = output_dir / "arcface_config.json"

    done_ids: set[int] = set()
    if args.resume and manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("arcface_status") == "success":
                    done_ids.add(int(row["image_id"]))

    config = {
        "images_dir": str(images_dir),
        "screening_report": str(args.screening_report),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "det_size": args.det_size,
        "det_thresh": args.det_thresh,
        "ctx_id": args.ctx_id,
        "save_aligned": args.save_aligned,
        "resume": args.resume,
        "limit": args.limit,
        "started_unix": started,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Preparing InsightFace model={args.model_name} ctx_id={args.ctx_id} det_size={args.det_size}")
    app = FaceAnalysis(name=args.model_name)
    app.prepare(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size))
    if args.det_thresh is not None and "detection" in app.models:
        app.models["detection"].det_thresh = args.det_thresh
        print(f"Set detection threshold to {args.det_thresh}")

    write_header = not manifest_path.exists() or not args.resume
    mode = "a" if args.resume and manifest_path.exists() else "w"
    counts = {
        "total_images_seen": len(image_paths),
        "skipped_resume": 0,
        "success": 0,
        "fail": 0,
        "pass_success": 0,
        "warn_success": 0,
        "pass_fail": 0,
        "warn_fail": 0,
    }

    with manifest_path.open(mode, encoding="utf-8", newline="") as mf, failures_path.open(
        "w", encoding="utf-8", newline=""
    ) as ff:
        writer = csv.DictWriter(mf, fieldnames=FIELDNAMES)
        failure_writer = csv.DictWriter(ff, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        failure_writer.writeheader()

        for index, image_path in enumerate(image_paths, start=1):
            image_id = image_id_from_path(image_path)
            label = labels.get(image_id, "Unknown")
            if image_id in done_ids:
                counts["skipped_resume"] += 1
                continue

            row: dict[str, str]
            image = cv2.imread(str(image_path))
            if image is None:
                row = empty_row(image_id, image_path, label, "image_read_failed")
            else:
                try:
                    faces = app.get(image)
                except Exception as exc:  # pragma: no cover - defensive for remote runtime
                    row = empty_row(image_id, image_path, label, f"inference_error:{type(exc).__name__}:{exc}")
                else:
                    if not faces:
                        row = empty_row(image_id, image_path, label, "no_face_detected")
                    else:
                        face = largest_face(faces)
                        embedding = np.asarray(face.normed_embedding, dtype=np.float32)
                        emb_norm = float(np.linalg.norm(embedding))
                        embedding_path = embeddings_dir / f"{image_id}.npy"
                        np.save(embedding_path, embedding)

                        aligned_path = ""
                        if args.save_aligned:
                            crop = crop_face(image, face.bbox)
                            if crop is not None:
                                aligned_file = aligned_dir / f"{image_id}.jpg"
                                cv2.imwrite(str(aligned_file), crop)
                                aligned_path = str(aligned_file)

                        bbox = [safe_float(v) for v in face.bbox]
                        kps = np.asarray(face.kps, dtype=float) if getattr(face, "kps", None) is not None else np.zeros((5, 2))
                        use_for_train = label == "Pass"
                        row = {
                            "image_id": str(image_id),
                            "image_path": str(image_path),
                            "embedding_path": str(embedding_path),
                            "aligned_face_path": aligned_path,
                            "arcface_status": "success",
                            "embedding_dim": str(int(embedding.shape[0])),
                            "embedding_norm": safe_float(emb_norm),
                            "detector_score": safe_float(getattr(face, "det_score", None)),
                            "bbox_x1": bbox[0],
                            "bbox_y1": bbox[1],
                            "bbox_x2": bbox[2],
                            "bbox_y2": bbox[3],
                            "landmark_left_eye_x": safe_float(kps[0, 0]),
                            "landmark_left_eye_y": safe_float(kps[0, 1]),
                            "landmark_right_eye_x": safe_float(kps[1, 0]),
                            "landmark_right_eye_y": safe_float(kps[1, 1]),
                            "landmark_nose_x": safe_float(kps[2, 0]),
                            "landmark_nose_y": safe_float(kps[2, 1]),
                            "landmark_mouth_left_x": safe_float(kps[3, 0]),
                            "landmark_mouth_left_y": safe_float(kps[3, 1]),
                            "landmark_mouth_right_x": safe_float(kps[4, 0]),
                            "landmark_mouth_right_y": safe_float(kps[4, 1]),
                            "deca_clean_label": label,
                            "eye_valid": "true",
                            "use_for_train": "true" if use_for_train else "false",
                            "failure_reason": "",
                        }

            writer.writerow(row)
            if row["arcface_status"] == "success":
                counts["success"] += 1
                if label == "Pass":
                    counts["pass_success"] += 1
                elif label == "Warn":
                    counts["warn_success"] += 1
            else:
                counts["fail"] += 1
                if label == "Pass":
                    counts["pass_fail"] += 1
                elif label == "Warn":
                    counts["warn_fail"] += 1
                failure_writer.writerow(row)

            if index == 1 or index % 100 == 0:
                elapsed = time.time() - started
                rate = index / elapsed if elapsed > 0 else 0
                print(
                    f"[{index}/{len(image_paths)}] success={counts['success']} fail={counts['fail']} "
                    f"pass_success={counts['pass_success']} rate={rate:.2f}/s",
                    flush=True,
                )

    counts["elapsed_seconds"] = round(time.time() - started, 3)
    counts["final_use_for_train"] = counts["pass_success"]
    summary_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
