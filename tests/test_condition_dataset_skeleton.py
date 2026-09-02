"""Cross-platform smoke test for the downstream condition-dataset skeleton."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def run(command: list[str]) -> None:
    subprocess.run(command, check=True, cwd=PROJECT, capture_output=True, text=True)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_fixture(root: Path) -> None:
    for name in (
        "source.png", "deca.mat", "phase2.npz", "embedding.npy",
        "source_depth.png", "source_normal.png", "source_landmark.png",
        "source_face_mask.png", "source_eye_mask.png", "target_depth.png",
        "target_normal.png", "target_landmark.png", "target_face_mask.png",
        "target_eye_mask.png", "target_gaze.png",
    ):
        (root / name).write_bytes(b"fixture")

    with (root / "phase1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "image_path", "deca_mat_path", "pitch", "yaw", "gaze_x", "gaze_y", "gaze_z", "arcface_embedding_path"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "image_id": "complete",
                    "image_path": str(root / "source.png"),
                    "deca_mat_path": str(root / "deca.mat"),
                    "pitch": "0.1",
                    "yaw": "-0.2",
                    "gaze_x": "0.197677",
                    "gaze_y": "-0.099833",
                    "gaze_z": "-0.975170",
                    "arcface_embedding_path": str(root / "embedding.npy"),
                },
                {
                    "image_id": "empty_phase2",
                    "image_path": str(root / "missing.png"),
                    "deca_mat_path": str(root / "missing.mat"),
                },
                {"image_id": "no_phase2_row", "image_path": "", "deca_mat_path": ""},
            ]
        )

    with (root / "phase2.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_id",
                "out_npz",
                "alpha_expression",
                "alpha_head_pose",
                "alpha_jaw_pose",
                "standardized_exp_norm",
                "standardized_head_pose_norm",
                "standardized_jaw_pose_norm",
                "quality_score",
                "xgb_quality_label",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "image_id": "complete",
                    "out_npz": str(root / "phase2.npz"),
                    "alpha_expression": "0.5",
                    "alpha_head_pose": "0.6",
                    "alpha_jaw_pose": "0.4",
                    "standardized_exp_norm": "0.1",
                    "standardized_head_pose_norm": "0.1",
                    "standardized_jaw_pose_norm": "0.05",
                    "quality_score": "0.7",
                    "xgb_quality_label": "high",
                },
                {"image_id": "empty_phase2", "out_npz": ""},
            ]
        )

    with (root / "condition.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "image_id", "status", "coordinate_status", "source_depth", "source_normal",
            "source_landmark", "source_face_mask", "source_eye_mask", "target_depth",
            "target_normal", "target_landmark", "target_face_mask", "target_eye_mask",
            "target_gaze_heatmap", "gaze_head_x", "gaze_head_y", "gaze_head_z",
            "target_gaze_head_x", "target_gaze_head_y", "target_gaze_head_z",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({
            "image_id": "complete", "status": "geometry_ready_gaze_pending",
            "coordinate_status": "pending_manual_audit",
            "source_depth": str(root / "source_depth.png"),
            "source_normal": str(root / "source_normal.png"),
            "source_landmark": str(root / "source_landmark.png"),
            "source_face_mask": str(root / "source_face_mask.png"),
            "source_eye_mask": str(root / "source_eye_mask.png"),
            "target_depth": str(root / "target_depth.png"),
            "target_normal": str(root / "target_normal.png"),
            "target_landmark": str(root / "target_landmark.png"),
            "target_face_mask": str(root / "target_face_mask.png"),
            "target_eye_mask": str(root / "target_eye_mask.png"),
            "target_gaze_heatmap": str(root / "target_gaze.png"),
            "gaze_head_x": "0.1", "gaze_head_y": "0.2", "gaze_head_z": "-0.97",
            "target_gaze_head_x": "0.1", "target_gaze_head_y": "0.2", "target_gaze_head_z": "-0.97",
        })

    (root / "split.json").write_text(
        json.dumps({"train": ["complete"], "val": ["empty_phase2"], "test": ["no_phase2_row"]}),
        encoding="utf-8",
    )


def test_condition_dataset_and_evaluator_skeletons() -> None:
    with tempfile.TemporaryDirectory(prefix="condition-skeleton-") as temporary:
        root = Path(temporary)
        write_fixture(root)
        output = root / "dataset"

        run(
            [
                PYTHON,
                str(PROJECT / "scripts" / "build_condition_dataset.py"),
                "--phase1-manifest",
                str(root / "phase1.csv"),
                "--phase2-manifest",
                str(root / "phase2.csv"),
                "--condition-cache-manifest",
                str(root / "condition.csv"),
                "--split-file",
                str(root / "split.json"),
                "--out-dir",
                str(output),
            ]
        )

        complete = read_jsonl(output / "train.jsonl")[0]
        assert complete["missing_fields"] == []
        assert complete["source_image_exists"] is True
        assert complete["deca_mat_exists"] is True
        assert complete["phase2_npz_exists"] is True
        assert complete["arcface_embedding_exists"] is True
        assert complete["alpha_expression"] == 0.5
        assert complete["gaze_camera_x"] == 0.197677
        assert complete["gaze_head_x"] is None
        assert complete["gaze_policy"] == "preserve_eye_in_head"
        assert complete["gaze_coordinate_status"] == "pending_manual_audit"
        assert complete["condition_cache_status"] == "geometry_ready_gaze_pending"
        assert complete["source_depth_map"] == str(root / "source_depth.png")
        assert complete["source_normal_map"] == str(root / "source_normal.png")
        assert complete["target_depth_map"] == str(root / "target_depth.png")
        assert complete["target_normal_map"] == str(root / "target_normal.png")
        assert complete["depth_map"] == complete["target_depth_map"]
        assert complete["normal_map"] == complete["target_normal_map"]
        assert complete["target_gaze_heatmap"] is None
        assert complete["gaze_heatmap"] is None
        assert complete["gaze_head_x"] is None
        assert complete["target_gaze_head_x"] is None
        assert complete["modalities_todo"] == ["target_gaze_heatmap"]

        empty_phase2 = read_jsonl(output / "val.jsonl")[0]
        assert empty_phase2["status"] == "missing_source"
        assert "phase2_npz" in empty_phase2["missing_fields"]
        assert empty_phase2["alpha_expression"] is None

        no_phase2_row = read_jsonl(output / "test.jsonl")[0]
        assert "phase2_row" in no_phase2_row["missing_fields"]
        assert no_phase2_row["phase2_npz_exists"] is False

        summary = json.loads((output / "dataset_summary.json").read_text(encoding="utf-8"))
        assert summary["n_total"] == 3
        assert summary["split_counts"] == {"train": 1, "val": 1, "test": 1}
        assert summary["missing_field_counts"]["source_image"] == 2
        assert summary["missing_field_counts"]["phase2_npz"] == 1
        assert summary["gaze_policy"] == "preserve_eye_in_head"

        include_ids = root / "include_ids.txt"
        include_ids.write_text("complete\n", encoding="utf-8")
        filtered_output = root / "dataset_filtered"
        run(
            [
                PYTHON,
                str(PROJECT / "scripts" / "build_condition_dataset.py"),
                "--phase1-manifest",
                str(root / "phase1.csv"),
                "--phase2-manifest",
                str(root / "phase2.csv"),
                "--condition-cache-manifest",
                str(root / "condition.csv"),
                "--split-file",
                str(root / "split.json"),
                "--include-ids-file",
                str(include_ids),
                "--out-dir",
                str(filtered_output),
            ]
        )
        filtered_summary = json.loads((filtered_output / "dataset_summary.json").read_text(encoding="utf-8"))
        assert filtered_summary["n_total"] == 1
        assert filtered_summary["include_ids_count"] == 1
        assert len(filtered_summary["include_ids_sha256"]) == 64
        assert [row["image_id"] for row in read_jsonl(filtered_output / "train.jsonl")] == ["complete"]

        unknown_ids = root / "unknown_ids.txt"
        unknown_ids.write_text("absent\n", encoding="utf-8")
        failed = subprocess.run(
            [
                PYTHON,
                str(PROJECT / "scripts" / "build_condition_dataset.py"),
                "--phase1-manifest",
                str(root / "phase1.csv"),
                "--include-ids-file",
                str(unknown_ids),
                "--out-dir",
                str(root / "dataset_unknown"),
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        assert failed.returncode != 0
        assert "absent from Phase1 manifest" in (failed.stdout + failed.stderr)

        evaluators = [
            ("evaluate_identity_preservation.py", "identity_metrics.csv", "identity_summary.json", "placeholder"),
            ("evaluate_pose_standardization.py", "pose_metrics.csv", "pose_summary.json", "placeholder"),
            (
                "evaluate_gaze_behavior.py",
                "gaze_metrics.csv",
                "gaze_summary.json",
                "source geometry only; output disentanglement metrics not computed",
            ),
        ]
        for script, metrics_name, summary_name, expected_status in evaluators:
            evaluator_output = root / script.removesuffix(".py")
            run(
                [
                    PYTHON,
                    str(PROJECT / "eval" / script),
                    "--manifest",
                    str(output / "train.jsonl"),
                    "--out-dir",
                    str(evaluator_output),
                ]
            )
            assert (evaluator_output / metrics_name).exists()
            evaluator_summary = json.loads((evaluator_output / summary_name).read_text(encoding="utf-8"))
            assert evaluator_summary["status"] == expected_status

        gaze_summary = json.loads(
            (root / "evaluate_gaze_behavior" / "gaze_summary.json").read_text(encoding="utf-8")
        )
        assert "targets gaze/head-pose disentanglement" in gaze_summary["scope_note"]
        assert "not evidence" in gaze_summary["scope_note"]


def test_gaze_coordinate_disentanglement_geometry() -> None:
    sys.path.insert(0, str(PROJECT / "eval"))
    from gaze_geometry import (  # pylint: disable=import-outside-toplevel
        angular_error_deg,
        axis_angle_to_matrix,
        camera_to_head_gaze,
        head_to_camera_gaze,
        l2cs_angles_to_camera_vector,
    )

    gaze_head = l2cs_angles_to_camera_vector(0.1, -0.2)
    first_rotation = axis_angle_to_matrix([0.0, 0.35, 0.0])
    second_rotation = axis_angle_to_matrix([-0.2, 0.0, 0.0])

    first_camera = head_to_camera_gaze(gaze_head, first_rotation)
    second_camera = head_to_camera_gaze(gaze_head, second_rotation)
    assert angular_error_deg(first_camera, second_camera) > 1.0
    assert angular_error_deg(camera_to_head_gaze(first_camera, first_rotation), gaze_head) < 1e-6
    assert angular_error_deg(camera_to_head_gaze(second_camera, second_rotation), gaze_head) < 1e-6


def main() -> None:
    test_condition_dataset_and_evaluator_skeletons()
    test_gaze_coordinate_disentanglement_geometry()
    print("CONDITION DATASET SKELETON SMOKE PASSED")


if __name__ == "__main__":
    main()
