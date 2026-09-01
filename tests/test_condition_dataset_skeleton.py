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
    for name in ("source.png", "deca.mat", "phase2.npz"):
        (root / name).write_bytes(b"fixture")

    with (root / "phase1.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "image_path", "deca_mat_path", "pitch", "yaw", "arcface_embedding_path"],
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
        assert complete["alpha_expression"] == 0.5
        assert complete["depth_map"] is None
        assert complete["modalities_todo"] == ["depth_map", "normal_map", "landmark_map", "face_mask"]

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

        evaluators = [
            ("evaluate_identity_preservation.py", "identity_metrics.csv", "identity_summary.json"),
            ("evaluate_pose_standardization.py", "pose_metrics.csv", "pose_summary.json"),
            ("evaluate_gaze_behavior.py", "gaze_metrics.csv", "gaze_summary.json"),
        ]
        for script, metrics_name, summary_name in evaluators:
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
            assert evaluator_summary["status"] == "placeholder"

        gaze_summary = json.loads(
            (root / "evaluate_gaze_behavior" / "gaze_summary.json").read_text(encoding="utf-8")
        )
        assert "no claim of gaze disentanglement" in gaze_summary["scope_note"]


def main() -> None:
    test_condition_dataset_and_evaluator_skeletons()
    print("CONDITION DATASET SKELETON SMOKE PASSED")


if __name__ == "__main__":
    main()
