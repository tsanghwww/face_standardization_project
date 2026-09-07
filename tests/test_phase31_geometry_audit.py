"""CPU protocol tests for Phase3.1c geometry response and target provenance."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
import torch

from phase3.audit_phase2_target_provenance import audit
from phase3.evaluate_geometry_response import expression_rmse, geodesic_angle_deg, landmark_shape_nme
from phase3.geometry_audit_data import (
    GeometryAuditDataset, evaluation_fingerprint, intervention, verify_target_provenance, verify_training_isolation,
)
from phase3.reconstruction_data import file_hash
from scripts.select_phase31c_geometry_audit_ids import pose_delta_deg, rank01


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_inputs(root: Path):
    root.mkdir(parents=True)
    split = root / "splits"
    split.mkdir()
    (split / "train_ids.txt").write_text("train0\n", encoding="utf-8")
    (split / "validation_ids.txt").write_text("val0\nval1\n", encoding="utf-8")
    (split / "fixed_test_ids.txt").write_text("fixed0\n", encoding="utf-8")
    rows = []
    for index, image_id in enumerate(("val0", "val1")):
        directory = root / image_id
        directory.mkdir()
        source = directory / "source.png"
        Image.new("RGB", (16, 16), (40 + index, 80, 120)).save(source)
        normal_paths, depth_paths, landmark_paths, mask_paths = {}, {}, {}, {}
        for prefix in ("source", "target"):
            normal_paths[prefix] = directory / f"{prefix}_normal.png"
            depth_paths[prefix] = directory / f"{prefix}_depth.png"
            landmark_paths[prefix] = directory / f"{prefix}_landmark.png"
            mask_paths[prefix] = directory / f"{prefix}_mask.png"
            Image.new("RGB", (16, 16), (127 + index, 128, 129)).save(normal_paths[prefix])
            Image.fromarray(np.full((16, 16), 20000 + 1000 * index, dtype=np.uint16)).save(depth_paths[prefix])
            Image.new("L", (16, 16), 60 + index).save(landmark_paths[prefix])
            Image.new("L", (16, 16), 255).save(mask_paths[prefix])
        embedding = directory / "embedding.npy"
        np.save(embedding, np.arange(1, 513, dtype=np.float32) + index)
        deca = directory / "deca.mat"
        deca.write_bytes(b"mat" + bytes([index]))
        npz = directory / "phase2.npz"
        np.savez(npz, image_id=image_id, source_mat=str(deca), expression_standardized=np.zeros(50, np.float32),
                 pose_standardized=np.zeros(6, np.float32))
        rows.append({
            "image_id": image_id, "split": "val", "source_image": str(source), "deca_mat": str(deca),
            "phase2_npz": str(npz), "arcface_embedding": str(embedding), "rescue_source": False,
            "condition_cache_status": "geometry_ready_gaze_pending",
            **{f"source_{name}_map" if name not in ("face_mask",) else "source_face_mask": str(path)
               for name, path in (("normal", normal_paths["source"]), ("depth", depth_paths["source"]),
                                  ("landmark", landmark_paths["source"]), ("face_mask", mask_paths["source"]))},
            **{f"target_{name}_map" if name not in ("face_mask",) else "target_face_mask": str(path)
               for name, path in (("normal", normal_paths["target"]), ("depth", depth_paths["target"]),
                                  ("landmark", landmark_paths["target"]), ("face_mask", mask_paths["target"]))},
        })
    manifest = root / "validation.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    ids = root / "ids.txt"
    ids.write_text("val0\nval1\n", encoding="utf-8")
    return split, manifest, ids, rows


def test_dataset_and_interventions(root: Path):
    split, manifest, ids, _ = make_inputs(root)
    dataset = GeometryAuditDataset(manifest, split, ids, "validation", size=16)
    items = [dataset[index] for index in range(len(dataset))]
    assert len(items) == 2 and items[0]["source_condition"].shape == (6, 16, 16)
    assert items[0]["target_condition"].shape == (6, 16, 16)
    source_identity = items[0]["identity"]
    for arm in ("source_geometry", "target_geometry", "zero_geometry", "shuffled_geometry"):
        _, identity, _, identity_id = intervention(items, 0, arm)
        assert torch.equal(identity, source_identity) and identity_id == "val0"
    _, shuffled_identity, _, identity_id = intervention(items, 0, "target_geometry_shuffled_identity")
    assert not torch.equal(shuffled_identity, source_identity) and identity_id == "val1"
    fingerprint = evaluation_fingerprint(dataset, manifest, ids, split, "validation")
    assert fingerprint["scope"] == "evaluation_only" and "training_fingerprint" not in fingerprint
    provenance_path = root / "strict_target_provenance.json"
    provenance_path.write_text(json.dumps({
        "status": "verified", "evaluation_manifest_sha256": fingerprint["manifest_sha256"],
        "ids_sha256": fingerprint["ids_sha256"], "condition_lineage_mode": "generation_time",
    }), encoding="utf-8")
    assert verify_target_provenance(provenance_path, fingerprint)["status"] == "verified"
    provenance_path.write_text(json.dumps({"status": "retrospective_only"}), encoding="utf-8")
    try:
        verify_target_provenance(provenance_path, fingerprint)
    except ValueError:
        pass
    else:
        raise AssertionError("Retrospective provenance accepted for sampling")
    Image.new("RGB", (16, 16), (1, 2, 3)).save(Path(dataset.rows[0]["target_normal_map"]))
    changed = GeometryAuditDataset(manifest, split, ids, "validation", size=16)
    assert evaluation_fingerprint(changed, manifest, ids, split, "validation") != fingerprint
    training = {
        "split_hashes": {name: file_hash(split / name) for name in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")},
        "inputs": [{"image_id": "train0", "field": "source_image", "sha256": "x"}],
    }
    assert verify_training_isolation(training, {"val0", "val1"}, split) == {"train0"}
    training["inputs"][0]["image_id"] = "val0"
    try:
        verify_training_isolation(training, {"val0", "val1"}, split)
    except ValueError:
        pass
    else:
        raise AssertionError("Validation-trained checkpoint accepted")
    bad_ids = root / "fixed_ids.txt"
    bad_ids.write_text("fixed0\n", encoding="utf-8")
    try:
        GeometryAuditDataset(manifest, split, bad_ids, "validation", size=16)
    except ValueError:
        pass
    else:
        raise AssertionError("Fixed-test ID accepted")
    print("[1-3] validation entry, fixed-test seal, fixed-identity geometry arms, checkpoint isolation OK")


def test_geometry_metrics():
    assert geodesic_angle_deg(np.zeros(3), np.zeros(3)) == 0
    assert abs(geodesic_angle_deg(np.array([0, 0, np.pi / 2]), np.zeros(3)) - 90) < 1e-6
    assert expression_rmse(np.zeros(50), np.ones(50)) == 1
    points = np.stack((np.linspace(-1, 1, 68), np.linspace(1, -1, 68)), axis=1)
    assert landmark_shape_nme(points, points * 3 + 8) < 1e-12
    assert abs(pose_delta_deg(np.array([0, 0, np.pi / 2]), np.zeros(3)) - 90) < 1e-6
    assert np.allclose(rank01([30, 10, 20]), [1, 0, 0.5])
    print("[4] pose/expression/landmark metrics and geometry-delta ranking OK")


def test_provenance(root: Path):
    split, manifest, ids, rows = make_inputs(root)
    checkpoint = root / "best_model.pt"
    torch.save({"config": {"quality_source": "blend", "alpha_mode": "learned"}}, checkpoint)
    xgb = root / "xgb.csv"
    xgb.write_text("image_id,xgb_quality_score\nval0,0.5\n", encoding="utf-8")
    inference_config = root / "inference_config.json"
    inference_config.write_text(json.dumps({"checkpoint": str(checkpoint), "quality_source": "blend", "alpha_mode": "learned",
                                            "xgb_quality_manifest": str(xgb), "include_ids_file": str(ids)}), encoding="utf-8")
    inference_command = root / "inference_exact_command.txt"
    inference_command.write_text(f"python -m phase2.infer_standardize_params --checkpoint {checkpoint}\n", encoding="utf-8")
    phase2_manifest = root / "phase2.csv"
    write_csv(phase2_manifest, [{"image_id": row["image_id"], "out_npz": row["phase2_npz"],
                                "mat_path": row["deca_mat"], "quality_source_effective": "blend",
                                "quality_source_requested": "blend", "xgb_quality_score": "0.5"} for row in rows])
    cache_manifest = root / "phase3_condition_cache.csv"
    write_csv(cache_manifest, [{
        "image_id": row["image_id"], "target_normal": row["target_normal_map"],
        "target_depth": row["target_depth_map"], "target_landmark": row["target_landmark_map"],
        "target_face_mask": row["target_face_mask"],
    } for row in rows])
    args = argparse.Namespace(
        ids_file=ids, phase2_manifest=phase2_manifest, condition_cache_manifest=cache_manifest,
        evaluation_manifest=manifest, inference_config=inference_config, inference_command=inference_command,
        phase2_checkpoint=checkpoint, expected_checkpoint_sha256=file_hash(checkpoint),
        expected_quality_source="blend", expected_alpha_mode="learned", allow_retrospective_cache_binding=True,
        expected_xgb_sha256=file_hash(xgb),
    )
    report, lineage = audit(args)
    assert report["status"] == "retrospective_only" and report["n_verified"] == 2
    assert report["condition_lineage_mode"] == "absent" and len(lineage) == 2
    args.allow_retrospective_cache_binding = False
    no_lineage, _ = audit(args)
    assert no_lineage["status"] == "failed" and any("lineage_absent" in item for item in no_lineage["errors"])
    lineage_path = root / "condition_cache_lineage.jsonl"
    with lineage_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({
                "image_id": row["image_id"], "binding_mode": "generation_time",
                "source_mat_sha256": file_hash(Path(row["deca_mat"])),
                "phase2_npz_sha256": file_hash(Path(row["phase2_npz"])),
                "target_map_sha256": {
                    "target_normal": file_hash(Path(row["target_normal_map"])),
                    "target_depth": file_hash(Path(row["target_depth_map"])),
                    "target_landmark": file_hash(Path(row["target_landmark_map"])),
                    "target_face_mask": file_hash(Path(row["target_face_mask"])),
                },
            }) + "\n")
    provenance = {
        "phase2_manifest_sha256": file_hash(phase2_manifest),
        "condition_manifest_sha256": file_hash(cache_manifest),
        "lineage_sha256": file_hash(lineage_path),
        "code_sha256": file_hash(Path(__file__).parents[1] / "scripts" / "build_phase3_condition_cache.py"),
    }
    (root / "condition_cache_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    strict, _ = audit(args)
    assert strict["status"] == "verified" and strict["condition_lineage_mode"] == "generation_time"
    args.expected_checkpoint_sha256 = "0" * 64
    failed, _ = audit(args)
    assert failed["status"] == "failed" and any("checkpoint_sha256_mismatch" in item for item in failed["errors"])
    print("[5-6] content-bound Phase2 target provenance and checkpoint mismatch failure OK")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as temporary:
        test_dataset_and_interventions(Path(temporary) / "dataset")
    test_geometry_metrics()
    with tempfile.TemporaryDirectory() as temporary:
        test_provenance(Path(temporary) / "provenance")
    print("ALL PHASE3.1C GEOMETRY AUDIT TESTS PASSED")
