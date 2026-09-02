#!/usr/bin/env python3
"""CPU-only protocol checks for the Phase3.0 evidence gate."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from scripts.build_phase3_gaze_manifest import compute_candidates
from scripts.build_condition_dataset import build_row, load_split_map


PROJECT = Path(__file__).resolve().parents[1]


def run(*arguments: str) -> None:
    subprocess.run([sys.executable, *arguments], cwd=PROJECT, check=True)


def test_split_registry_and_candidate_gaze() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        train = root / "train_ids.txt"
        val = root / "val_ids.txt"
        test = root / "test_ids.txt"
        train.write_text("a\nb\n", encoding="utf-8")
        val.write_text("c\n", encoding="utf-8")
        test.write_text("d\n", encoding="utf-8")
        external = root / "external.csv"
        external.write_text(
            "eval_id,source_dataset\nbase_duplicate,stylegan2_base\next_0,WIDER_FACE_val\n",
            encoding="utf-8",
        )
        registry = root / "registry"
        run(
            "scripts/prepare_phase3_splits.py",
            "--train-ids", str(train),
            "--val-ids", str(val),
            "--base-test-ids", str(test),
            "--external-manifest", str(external),
            "--external-filter-column", "source_dataset",
            "--external-filter-values", "WIDER_FACE_val",
            "--expected-counts", "2,1,1,1",
            "--out-dir", str(registry),
        )
        summary = json.loads((registry / "phase3_split_registry.json").read_text(encoding="utf-8"))
        assert summary["counts"] == {
            "train": 2, "validation": 1, "fixed_test_base": 1, "fixed_test_external": 1, "fixed_test": 2
        }
        assert not any(summary["overlaps"].values())
        resolved_splits = load_split_map(registry, None)
        assert resolved_splits == {"a": "train", "b": "train", "c": "val", "d": "test"}

        direct, inverse, direct_error, inverse_error, rotation_6d = compute_candidates(
            np.array([0.1, -0.2, 0.05], dtype=np.float64), (0.1, -0.2, -0.97)
        )
        assert len(direct) == len(inverse) == 3
        assert len(rotation_6d) == 6
        assert direct_error < 1e-5 and inverse_error < 1e-5


def test_split_leakage_fails_closed() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        for name, payload in (("train", "same\n"), ("val", "same\n"), ("test", "different\n")):
            (root / f"{name}.txt").write_text(payload, encoding="utf-8")
        (root / "external.csv").write_text("eval_id\next\n", encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable, "scripts/prepare_phase3_splits.py",
                "--train-ids", str(root / "train.txt"), "--val-ids", str(root / "val.txt"),
                "--base-test-ids", str(root / "test.txt"), "--external-manifest", str(root / "external.csv"),
                "--out-dir", str(root / "out"),
            ],
            cwd=PROJECT,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Split leakage" in result.stderr + result.stdout


def test_unapproved_gaze_is_not_exposed_to_training() -> None:
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        paths = {name: root / name for name in ("source.png", "deca.mat", "phase2.npz", "source_normal.png", "normal.png", "gaze.png")}
        for path in paths.values():
            path.touch()
        base = {"image_id": "sample", "image_path": str(paths["source.png"]), "deca_mat_path": str(paths["deca.mat"])}
        phase2 = {"image_id": "sample", "out_npz": str(paths["phase2.npz"])}
        cache = {
            "source_normal": str(paths["source_normal.png"]),
            "target_normal": str(paths["normal.png"]),
            "target_gaze_heatmap": str(paths["gaze.png"]),
            "coordinate_status": "pending_manual_audit",
            "gaze_head_x": "0.1", "gaze_head_y": "0.2", "gaze_head_z": "-0.97",
            "target_gaze_head_x": "0.1", "target_gaze_head_y": "0.2", "target_gaze_head_z": "-0.97",
        }
        pending = build_row(base, phase2, cache, {}, "train", "preserve_eye_in_head")
        assert pending["source_normal_map"] == str(paths["source_normal.png"])
        assert pending["target_normal_map"] == str(paths["normal.png"])
        assert pending["normal_map"] == str(paths["normal.png"])
        assert pending["gaze_heatmap"] is None
        assert pending["gaze_head_x"] is None

        cache["coordinate_status"] = "approved"
        approved = build_row(base, phase2, cache, {}, "train", "preserve_eye_in_head")
        assert approved["gaze_heatmap"] == str(paths["gaze.png"])
        assert approved["gaze_head_x"] == 0.1


def main() -> None:
    test_split_registry_and_candidate_gaze()
    test_split_leakage_fails_closed()
    test_unapproved_gaze_is_not_exposed_to_training()
    print("PHASE3.0 PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    main()
