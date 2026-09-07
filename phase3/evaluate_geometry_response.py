"""Re-estimate identity, pose, expression, and landmarks from geometry-audit RGB outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import torch

from phase3.audit_vae_roundtrip import load_arcface, load_deca, load_fan
from phase3.geometry_audit_data import GeometryAuditDataset, evaluation_fingerprint
from phase3.reconstruction_data import file_hash
from scripts.build_phase3_condition_cache import apply_phase2, codedict, load_source_params


GEOMETRY_METRICS = ("pose_target_error_deg", "expression_target_rmse", "landmark_target_shape_nme")
IDENTITY_METRICS = ("source_arcface_cosine_single_face", "source_arcface_cosine_largest_face")
METRICS = (*GEOMETRY_METRICS, *IDENTITY_METRICS)
GEOMETRY_COMPARATORS = ("source_geometry", "zero_geometry", "shuffled_geometry")


def geodesic_angle_deg(a, b) -> float:
    from scipy.spatial.transform import Rotation
    relative = Rotation.from_rotvec(np.asarray(a, dtype=np.float64)) * Rotation.from_rotvec(np.asarray(b, dtype=np.float64)).inv()
    return float(np.degrees(relative.magnitude()))


def expression_rmse(a, b) -> float:
    left, right = np.asarray(a, dtype=np.float64).reshape(-1), np.asarray(b, dtype=np.float64).reshape(-1)
    if left.shape != (50,) or right.shape != (50,):
        raise ValueError("Expression vectors must have 50 dimensions")
    return float(np.sqrt(np.mean(np.square(left - right))))


def normalize_landmarks(value) -> np.ndarray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] != 68 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError(f"Invalid landmarks: {points.shape}")
    points = points[:, :2] - points[:, :2].mean(axis=0, keepdims=True)
    scale = float(np.sqrt(np.mean(np.square(points).sum(axis=1))))
    if scale < 1e-8:
        raise ValueError("Degenerate landmarks")
    return points / scale


def landmark_shape_nme(a, b) -> float:
    return float(np.linalg.norm(normalize_landmarks(a) - normalize_landmarks(b), axis=1).mean())


def stats(values) -> dict:
    valid = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    return {
        "count": int(valid.size),
        "mean": float(valid.mean()) if valid.size else None,
        "median": float(np.median(valid)) if valid.size else None,
        "p10": float(np.percentile(valid, 10)) if valid.size else None,
        "p90": float(np.percentile(valid, 90)) if valid.size else None,
    }


def bootstrap_mean_ci(values, seed: int = 20260907, repetitions: int = 10000) -> dict:
    valid = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    if not valid.size:
        return {"count": 0, "mean": None, "ci95": [None, None]}
    rng = np.random.default_rng(seed)
    estimates = valid[rng.integers(0, valid.size, size=(repetitions, valid.size))].mean(axis=1)
    return {"count": int(valid.size), "mean": float(valid.mean()),
            "ci95": [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]}


def detect_arcface(app, rgb: np.ndarray):
    faces = app.get(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not faces:
        return None, 0, "no_face_detected"
    face = max(faces, key=lambda item: float((item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])))
    embedding = np.asarray(face.normed_embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(embedding))
    if embedding.shape != (512,) or not np.isfinite(embedding).all() or norm < 1e-8:
        return None, len(faces), "invalid_embedding"
    return embedding / norm, len(faces), ""


def estimate_deca(deca, fan, device: str, rgb: np.ndarray) -> dict:
    from phase2.run_fixed_external_deca import crop_to_tensor
    bbox, bbox_type = fan.run(rgb)
    if len(bbox) == 1:
        raise ValueError("fan_no_face")
    tensor = torch.from_numpy(crop_to_tensor(rgb, bbox, bbox_type)).unsqueeze(0).to(device)
    with torch.no_grad():
        code = deca.encode(tensor)
        output = deca.decode(code, rendering=False, return_vis=False)
    return {
        "pose": code["pose"][0].detach().cpu().numpy(),
        "expression": code["exp"][0].detach().cpu().numpy(),
        "landmarks": output["landmarks2d"][0].detach().cpu().numpy(),
    }


def target_values(deca, device: str, deca_mat: Path, phase2_npz: Path) -> dict:
    source = load_source_params(deca_mat)
    target = apply_phase2(source, phase2_npz)
    with torch.no_grad():
        output = deca.decode(codedict(target, device, 224), rendering=False, return_vis=False)
    return {"pose": target["pose"], "expression": target["expression"],
            "landmarks": output["landmarks2d"][0].detach().cpu().numpy()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--deca-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arcface-det-thresh", default=0.1, type=float)
    parser.add_argument("--bootstrap-repetitions", default=10000, type=int)
    args = parser.parse_args()
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    generation = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    if generation.get("config_sha256") != file_hash(args.run_dir / "config.json"):
        raise ValueError("Sampling config hash mismatch")
    if generation["samples_sha256"] != file_hash(args.run_dir / "samples.jsonl"):
        raise ValueError("Sample manifest hash mismatch")
    if generation.get("reference_hashes_sha256") != file_hash(args.run_dir / "reference_hashes.json"):
        raise ValueError("Reference hash manifest mismatch")
    reference_hashes = json.loads((args.run_dir / "reference_hashes.json").read_text(encoding="utf-8"))
    for name, digest in reference_hashes.items():
        if file_hash(args.run_dir / "references" / name) != digest:
            raise ValueError(f"Reference image hash mismatch: {name}")
    if file_hash(Path(config["target_provenance"])) != config["target_provenance_sha256"]:
        raise ValueError("Target provenance report changed after sampling")
    samples = [json.loads(line) for line in (args.run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines() if line]
    ids, arms, schedules = config["image_ids"], config["arms"], config["schedules"]
    dataset = GeometryAuditDataset(Path(config["evaluation_manifest"]), Path(config["split_dir"]),
                                   Path(config["evaluation_ids"]), config["evaluation_split"])
    current_fingerprint = evaluation_fingerprint(dataset, Path(config["evaluation_manifest"]),
                                                 Path(config["evaluation_ids"]), Path(config["split_dir"]),
                                                 config["evaluation_split"])
    if current_fingerprint != config["evaluation_fingerprint"]:
        raise ValueError("Evaluation inputs changed after sampling")
    lookup = {(row["image_id"], row["arm"], row["key"]): row for row in samples}
    expected = {(image_id, arm, spec["key"]) for image_id in ids for arm in arms for spec in schedules}
    if len(lookup) != len(samples) or set(lookup) != expected:
        raise ValueError("Duplicate, missing, or unexpected generation records")
    eval_manifest = Path(config["evaluation_manifest"])
    raw_manifest_rows = [json.loads(line) for line in eval_manifest.read_text(encoding="utf-8").splitlines() if line]
    manifest_rows = {str(row["image_id"]): row for row in raw_manifest_rows}
    if len(manifest_rows) != len(raw_manifest_rows):
        raise ValueError("Duplicate evaluation manifest IDs")
    if set(ids) - set(manifest_rows):
        raise ValueError("Run IDs absent from evaluation manifest")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    (args.out_dir / "exact_command.txt").write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding="utf-8")
    arcface, fan = load_arcface(args.arcface_det_thresh), load_fan()
    deca = load_deca(args.deca_root, args.device)

    rows = []
    for image_id in ids:
        base = manifest_rows[image_id]
        target = target_values(deca, args.device, Path(base["deca_mat"]), Path(base["phase2_npz"]))
        with Image.open(args.run_dir / "references" / f"{image_id}_source.png") as image:
            source_rgb = np.asarray(image.convert("RGB"))
        source_embedding, source_faces, source_arc_error = detect_arcface(arcface, source_rgb)
        try:
            source_deca = estimate_deca(deca, fan, args.device, source_rgb)
            source_target = {
                "pose": geodesic_angle_deg(source_deca["pose"][:3], target["pose"][:3]),
                "expression": expression_rmse(source_deca["expression"], target["expression"]),
                "landmark": landmark_shape_nme(source_deca["landmarks"], target["landmarks"]),
            }
            source_deca_status = "success"
        except Exception as error:
            source_target = {"pose": None, "expression": None, "landmark": None}
            source_deca_status = f"{type(error).__name__}: {error}"
        for spec in schedules:
            for arm in arms:
                sample = lookup[image_id, arm, spec["key"]]
                row = {
                    "image_id": image_id, "arm": arm, "key": spec["key"], "strength": spec["strength"],
                    "generation_status": sample["status"], "source_faces": source_faces, "generated_faces": None,
                    "source_arcface_cosine_single_face": None, "source_arcface_cosine_largest_face": None,
                    "pose_target_error_deg": None, "expression_target_rmse": None,
                    "landmark_target_shape_nme": None, "source_pose_target_error_deg": source_target["pose"],
                    "source_expression_target_rmse": source_target["expression"],
                    "source_landmark_target_shape_nme": source_target["landmark"],
                    "deca_status": "", "arcface_status": "", "failure_reason": "",
                }
                try:
                    if sample["status"] != "generated":
                        raise ValueError(sample.get("failure_reason") or "generation_failed")
                    output_path = args.run_dir / sample["output"]
                    if file_hash(output_path) != sample["sha256"]:
                        raise ValueError("generated_image_hash_mismatch")
                    with Image.open(output_path) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    embedding, count, error = detect_arcface(arcface, rgb)
                    row["generated_faces"] = count
                    row["arcface_status"] = error or "success"
                    if source_embedding is not None and embedding is not None:
                        cosine = float(np.dot(source_embedding, embedding))
                        row["source_arcface_cosine_largest_face"] = cosine
                        if source_faces == 1 and count == 1:
                            row["source_arcface_cosine_single_face"] = cosine
                    try:
                        estimate = estimate_deca(deca, fan, args.device, rgb)
                        row.update(
                            pose_target_error_deg=geodesic_angle_deg(estimate["pose"][:3], target["pose"][:3]),
                            expression_target_rmse=expression_rmse(estimate["expression"], target["expression"]),
                            landmark_target_shape_nme=landmark_shape_nme(estimate["landmarks"], target["landmarks"]),
                            deca_status="success",
                        )
                    except Exception as error:
                        row["deca_status"] = f"{type(error).__name__}: {error}"
                except Exception as error:
                    row["failure_reason"] = f"{type(error).__name__}: {error}"
                if source_arc_error and not row["arcface_status"]:
                    row["arcface_status"] = f"source:{source_arc_error}"
                if source_deca_status != "success" and not row["deca_status"]:
                    row["deca_status"] = f"source:{source_deca_status}"
                rows.append(row)
        print(json.dumps({"image_id": image_id, "rows": len(rows)}), flush=True)

    with (args.out_dir / "geometry_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups, contrasts = [], []
    for spec in schedules:
        subset = [row for row in rows if row["key"] == spec["key"]]
        for arm in arms:
            group = [row for row in subset if row["arm"] == arm]
            groups.append({
                "arm": arm, **spec, "n_total": len(ids), "metrics": {metric: stats([row[metric] for row in group]) for metric in METRICS},
                "no_face": sum(row["generated_faces"] == 0 for row in group),
                "multiple_faces": sum(row["generated_faces"] is not None and row["generated_faces"] > 1 for row in group),
                "single_face_valid": sum(row["source_faces"] == 1 and row["generated_faces"] == 1
                                         and row["source_arcface_cosine_single_face"] is not None for row in group),
                "source_multiple_faces": sum(row["source_faces"] is not None and row["source_faces"] > 1 for row in group),
                "deca_success": sum(row["deca_status"] == "success" for row in group),
            })
        keyed = {(row["image_id"], row["arm"]): row for row in subset}
        for comparator in GEOMETRY_COMPARATORS:
            metrics = {}
            for metric in GEOMETRY_METRICS:
                values = []
                for image_id in ids:
                    target_value = keyed[image_id, "target_geometry"][metric]
                    comparator_value = keyed[image_id, comparator][metric]
                    values.append(target_value - comparator_value if target_value is not None and comparator_value is not None else None)
                metrics[metric] = bootstrap_mean_ci(values, repetitions=args.bootstrap_repetitions)
            contrasts.append({"strength": spec["strength"], "target_geometry_minus": comparator, "metrics": metrics})
    directional = {}
    for spec in schedules:
        relevant = [item for item in contrasts if item["strength"] == spec["strength"]]
        directional[str(spec["strength"])] = {
            metric: all(item["metrics"][metric]["mean"] is not None and item["metrics"][metric]["mean"] < 0 for item in relevant)
            for metric in GEOMETRY_METRICS
        }
    summary = {
        "n_ids": len(ids), "expected_rows": len(expected), "audited_rows": len(rows), "groups": groups,
        "paired_contrasts": contrasts, "directional_response": directional,
        "metric_definitions": {
            "pose_target_error_deg": "SO(3) geodesic angle between re-estimated output and Phase2 target head rotvec",
            "expression_target_rmse": "RMSE between re-estimated 50D DECA expression and Phase2 standardized expression",
            "landmark_target_shape_nme": "mean point error after independent centroid/RMS-scale normalization of 68 DECA landmarks",
        },
        "denominator_policy": "all selected IDs in every arm; missing evaluator values remain empty and failures are counted",
        "geometry_claim_rule": "descriptive only: target geometry should reduce all three target errors versus source/zero/shuffled geometry; CIs are reported, not tuned",
        "arcface_model": "buffalo_l", "arcface_det_thresh": args.arcface_det_thresh,
        "face_selection": "largest_bbox; no-face and multi-face retained; single-face-valid reported separately",
        "deca_preprocess": "FAN crop with no rescue fallback", "gaze_evaluated": False,
        "config_sha256": file_hash(args.run_dir / "config.json"), "samples_sha256": file_hash(args.run_dir / "samples.jsonl"),
        "evaluator_sha256": file_hash(Path(__file__)),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"n_ids": len(ids), "directional_response": directional}, indent=2))


if __name__ == "__main__":
    main()
