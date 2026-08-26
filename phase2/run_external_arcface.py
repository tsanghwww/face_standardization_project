"""Run ArcFace on the 375 external fixed-test samples for XGBoost feature coverage.

Produces an arcface manifest keyed by image_id with the two fields the XGBoost
model needs -- ``arcface_status`` (success/fail) and ``detector_score`` -- plus
failure_reason.  This is a separate manifest from the base ArcFace manifest so
the two can be merged by image_id during XGBoost prediction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

PROJECT = Path(r"D:\face_standardization_project")
EXTERNAL_GROUPS = {"wider_pose", "wider_occlusion", "wider_blur", "cofw_occlusion", "aflw_large_pose"}

FIELDS = ["image_id", "eval_id", "image_path", "arcface_status", "detector_score", "failure_reason"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results" / "phase2_arcface_external_20260824")
    parser.add_argument("--ctx-id", type=int, default=-1, help="-1 CPU, 0 first GPU provider")
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--det-thresh", type=float, default=0.1, help="Match base ArcFace protocol (0.1).")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["source_group"] in EXTERNAL_GROUPS]
    if args.limit:
        rows = rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=args.ctx_id, det_size=(args.det_size, args.det_size))
    if "detection" in app.models:
        app.models["detection"].det_thresh = args.det_thresh

    out_rows: list[dict[str, str]] = []
    success = fail = 0
    for r in rows:
        image_path = r["image_path"]
        row = {k: "" for k in FIELDS}
        row["image_id"] = r["image_id"]
        row["eval_id"] = r["eval_id"]
        row["image_path"] = image_path
        image = cv2.imread(image_path)
        if image is None:
            row["arcface_status"] = "fail"
            row["failure_reason"] = "image_read_failed"
            fail += 1
        else:
            faces = app.get(image)
            if not faces:
                row["arcface_status"] = "fail"
                row["failure_reason"] = "no_face_detected"
                fail += 1
            else:
                face = max(faces, key=lambda x: float((x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1])))
                row["arcface_status"] = "success"
                det = getattr(face, "det_score", None)
                row["detector_score"] = f"{float(det):.8f}" if det is not None and np.isfinite(det) else ""
                success += 1
        out_rows.append(row)

    with (args.out_dir / "arcface_external_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)
    summary = {"total": len(rows), "success": success, "fail": fail, "ctx_id": args.ctx_id}
    (args.out_dir / "arcface_external_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
