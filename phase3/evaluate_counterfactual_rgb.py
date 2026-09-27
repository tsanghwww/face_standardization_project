"""Evaluate identity and absolute geometry for saved Phase3 counterfactual RGB outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from phase3.audit_vae_roundtrip import load_arcface, load_deca, load_fan
from phase3.evaluate_geometry_response import (
    bootstrap_mean_ci,
    detect_arcface,
    estimate_deca,
    expression_rmse,
    geodesic_angle_deg,
    landmark_shape_nme,
    stats,
    target_values,
)
from phase3.geometry_audit_data import GeometryAuditDataset
from phase3.reconstruction_data import file_hash, read_ids


VARIANTS = ("negative_yaw", "positive_yaw")
ABSOLUTE_METRICS = ("pose_target_error_deg", "expression_target_rmse", "landmark_target_shape_nme")


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--deca-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arcface-det-thresh", type=float, default=0.1)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError("Evaluation output directory must be new")

    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    generation = json.loads((args.run_dir / "summary.json").read_text(encoding="utf-8"))
    if not config.get("save_images"):
        raise ValueError("Counterfactual run did not save RGB outputs")
    if generation.get("config_sha256") != file_hash(args.run_dir / "config.json"):
        raise ValueError("Generation config hash mismatch")
    if generation.get("metrics_sha256") != file_hash(args.run_dir / "metrics.jsonl"):
        raise ValueError("Generation metrics hash mismatch")
    if config.get("split") != "validation":
        raise ValueError("This evaluator is restricted to validation; fixed test remains sealed")

    dataset = GeometryAuditDataset(
        Path(config["manifest"]), Path(config["split_dir"]), Path(config["ids_file"]), "validation"
    )
    ids = [item["image_id"] for item in dataset]
    if ids != list(config["image_ids"]):
        raise ValueError("Evaluation manifest order changed after generation")
    fixed = read_ids(Path(config["split_dir"]) / "fixed_test_ids.txt")
    if set(ids) & fixed:
        raise ValueError("Fixed-test leakage into RGB evaluation")

    generated = load_rows(args.run_dir / "metrics.jsonl")
    timesteps = [int(value) for value in config["timesteps"]]
    lookup = {(str(row["image_id"]), int(row["timestep"])): row for row in generated}
    expected = {(image_id, timestep) for image_id in ids for timestep in timesteps}
    if len(lookup) != len(generated) or set(lookup) != expected:
        raise ValueError("Generation ledger is incomplete or duplicated")
    counter_rows = load_rows(Path(config["counterfactual_manifest"]))
    counters = {str(row["image_id"]): row for row in counter_rows}
    if len(counters) != len(counter_rows) or set(ids) - set(counters):
        raise ValueError("Counterfactual target manifest is incomplete")

    args.out_dir.mkdir(parents=True)
    (args.out_dir / "exact_command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]), encoding="utf-8"
    )
    arcface, fan = load_arcface(args.arcface_det_thresh), load_fan()
    deca = load_deca(args.deca_root, args.device)

    rows = []
    for item in dataset:
        image_id = item["image_id"]
        reference_path = args.run_dir / "references" / f"{image_id}_source.png"
        with Image.open(reference_path) as image:
            source_rgb = np.asarray(image.convert("RGB"))
        source_embedding = item["identity"].numpy().astype(np.float32)
        source_embedding /= np.linalg.norm(source_embedding)
        try:
            source_estimate = estimate_deca(deca, fan, args.device, source_rgb)
            source_status = "success"
        except Exception as error:
            source_estimate = None
            source_status = f"{type(error).__name__}: {error}"

        targets = {}
        for variant in VARIANTS:
            variant_row = counters[image_id]["variants"][variant]
            phase2_path = Path(variant_row["phase2_npz"])
            if file_hash(phase2_path) != variant_row["artifact_hashes"]["phase2_npz"]:
                raise ValueError(f"Counterfactual target changed: {image_id}:{variant}")
            targets[variant] = target_values(deca, args.device, Path(item["deca_mat"]), phase2_path)

        for timestep in timesteps:
            generation_row = lookup[image_id, timestep]
            for variant in VARIANTS:
                target = targets[variant]
                row = {
                    "image_id": image_id,
                    "timestep": timestep,
                    "variant": variant,
                    "generation_status": generation_row.get("status"),
                    "arcface_status": "",
                    "generated_faces": None,
                    "identity_cosine": None,
                    "deca_status": "",
                    "source_deca_status": source_status,
                    "pose_target_error_deg": None,
                    "expression_target_rmse": None,
                    "landmark_target_shape_nme": None,
                    "source_pose_target_error_deg": None,
                    "source_expression_target_rmse": None,
                    "source_landmark_target_shape_nme": None,
                    "failure_reason": "",
                }
                if source_estimate is not None:
                    row.update(
                        source_pose_target_error_deg=geodesic_angle_deg(source_estimate["pose"][:3], target["pose"][:3]),
                        source_expression_target_rmse=expression_rmse(source_estimate["expression"], target["expression"]),
                        source_landmark_target_shape_nme=landmark_shape_nme(source_estimate["landmarks"], target["landmarks"]),
                    )
                try:
                    if generation_row.get("status") != "success":
                        raise ValueError(generation_row.get("failure_reason") or "generation_failed")
                    relative = generation_row.get(f"{variant}_output")
                    digest = generation_row.get(f"{variant}_sha256")
                    if not relative or not digest:
                        raise ValueError("missing_saved_output")
                    output_path = args.run_dir / relative
                    if file_hash(output_path) != digest:
                        raise ValueError("generated_image_hash_mismatch")
                    with Image.open(output_path) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    embedding, face_count, arc_error = detect_arcface(arcface, rgb)
                    row["generated_faces"] = face_count
                    row["arcface_status"] = arc_error or "success"
                    if embedding is not None:
                        row["identity_cosine"] = float(np.dot(source_embedding, embedding))
                    try:
                        estimate = estimate_deca(deca, fan, args.device, rgb)
                        row.update(
                            deca_status="success",
                            pose_target_error_deg=geodesic_angle_deg(estimate["pose"][:3], target["pose"][:3]),
                            expression_target_rmse=expression_rmse(estimate["expression"], target["expression"]),
                            landmark_target_shape_nme=landmark_shape_nme(estimate["landmarks"], target["landmarks"]),
                        )
                    except Exception as error:
                        row["deca_status"] = f"{type(error).__name__}: {error}"
                except Exception as error:
                    row["failure_reason"] = f"{type(error).__name__}: {error}"
                rows.append(row)
        print(json.dumps({"image_id": image_id, "rows": len(rows)}), flush=True)

    csv_path = args.out_dir / "counterfactual_rgb_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    groups = []
    for timestep in timesteps:
        for variant in VARIANTS:
            subset = [row for row in rows if row["timestep"] == timestep and row["variant"] == variant]
            improvements = {}
            for metric in ABSOLUTE_METRICS:
                source_metric = "source_" + metric
                values = [
                    row[source_metric] - row[metric]
                    if row[source_metric] is not None and row[metric] is not None else None
                    for row in subset
                ]
                improvements[metric] = bootstrap_mean_ci(values, repetitions=args.bootstrap_repetitions)
            groups.append({
                "timestep": timestep,
                "variant": variant,
                "n_total": len(ids),
                "generation_success": sum(row["generation_status"] == "success" for row in subset),
                "arcface_success": sum(row["identity_cosine"] is not None for row in subset),
                "deca_success": sum(row["deca_status"] == "success" for row in subset),
                "identity_cosine": stats([row["identity_cosine"] for row in subset]),
                "absolute_target_error": {metric: stats([row[metric] for row in subset]) for metric in ABSOLUTE_METRICS},
                "source_minus_output_improvement": improvements,
            })
    report = {
        "status": "completed",
        "scope": "validation-only RGB identity and absolute geometry audit; no fixed-test tuning and no gaze claim",
        "n_ids": len(ids),
        "n_expected_outputs": len(ids) * len(timesteps) * len(VARIANTS),
        "n_rows": len(rows),
        "fixed_test_overlap": len(set(ids) & fixed),
        "groups": groups,
        "denominator_policy": "every selected validation ID at every timestep and variant; failures retained",
        "metrics_sha256": file_hash(csv_path),
        "generation_config_sha256": file_hash(args.run_dir / "config.json"),
        "generation_metrics_sha256": file_hash(args.run_dir / "metrics.jsonl"),
        "gaze_evaluated": False,
    }
    write_json(args.out_dir / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
