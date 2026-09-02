#!/usr/bin/env python3
"""CPU-only protocol tests for the Phase3.0B VAE round-trip audit (no network, no
real VAE weights, no ArcFace/DECA/L2CS model loading)."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from phase3.audit_vae_roundtrip import (
    _write_artifact_hashes,
    _tree_sha256,
    build_row,
    encode_decode,
    freeze_model,
    load_arcface,
    load_completed,
    l2cs_pair,
    select_audit_ids,
    summarize,
    verify_resume_config,
)

PROJECT = Path(__file__).resolve().parents[1]
PY = sys.executable
TMP = PROJECT / ".runtime_tmp" / "phase3_vae_protocol_test"


def _reset() -> None:
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)


def _write_gaze(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "split", "status", "pose_x", "pose_y", "pose_z"])
        w.writeheader()
        w.writerows(rows)


class FakeLatentDist:
    def __init__(self, value):
        self._v = value

    def mode(self):
        return self._v

    def sample(self):
        raise AssertionError("posterior.sample() must NOT be called; use mode()")


class FakeVAE:
    def __init__(self, mode_value, scaling=0.18215):
        self._mode = mode_value
        self.config = SimpleNamespace(scaling_factor=scaling)
        self.decoded_z = None

    def encode(self, x):
        return SimpleNamespace(latent_dist=FakeLatentDist(self._mode))

    def decode(self, z):
        self.decoded_z = z
        return SimpleNamespace(sample=z.clone())


class FakeGazeResult:
    bboxes = np.array([])
    pitch = np.array([])
    yaw = np.array([])


class FakeGaze:
    def step(self, image):
        return FakeGazeResult()


def _mk(vae="success", cosine="", a_src="", a_rec="", head="", gaze="", l_src="", l_rec="", runtime="0.1"):
    """Build a row with the 24-arg build_row signature, defaults for unused fields."""
    return build_row(
        "x", "validation", "", "", "", vae, "", "", "", "", "not_available",
        cosine, a_src, a_rec, "", "",
        head, "success", "success",
        gaze, l_src, l_rec,
        float(runtime), 10.0,
    )


def test_selection_fail_closed_on_fixed() -> None:
    _reset()
    gaze = TMP / "gaze.csv"
    _write_gaze(gaze, [
        {"image_id": "leak", "split": "validation", "status": "candidate_unvalidated", "pose_x": "9", "pose_y": "0", "pose_z": "0"},
        {"image_id": "v1", "split": "validation", "status": "candidate_unvalidated", "pose_x": "0.1", "pose_y": "0", "pose_z": "0"},
    ])
    (TMP / "val.txt").write_text("leak\nv1\n", encoding="utf-8")
    (TMP / "fixed.txt").write_text("leak\n", encoding="utf-8")
    try:
        select_audit_ids(gaze, TMP / "val.txt", TMP / "fixed.txt", 1, TMP / "sel")
    except ValueError:
        pass
    else:
        raise AssertionError("select_audit_ids did not raise ValueError for fixed-test leak")
    print("[1] selection fail-closed on fixed-test ID (ValueError, not assert) OK")


def test_selection_unique_deterministic() -> None:
    _reset()
    gaze = TMP / "gaze.csv"
    _write_gaze(gaze, [{"image_id": f"v{i}", "split": "validation", "status": "candidate_unvalidated",
                        "pose_x": f"{i * 0.1}", "pose_y": f"{i * 0.02}", "pose_z": f"{i * 0.03}"} for i in range(10)])
    (TMP / "val.txt").write_text("\n".join(f"v{i}" for i in range(10)) + "\n", encoding="utf-8")
    (TMP / "fixed.txt").write_text("f1\n", encoding="utf-8")
    ids_a = select_audit_ids(gaze, TMP / "val.txt", TMP / "fixed.txt", 5, TMP / "sel_a")
    ids_b = select_audit_ids(gaze, TMP / "val.txt", TMP / "fixed.txt", 5, TMP / "sel_b")
    assert len(ids_a) == 5 and len(set(ids_a)) == 5
    assert ids_a == ids_b
    assert hashlib.sha256((TMP / "sel_a" / "vae_audit_ids.txt").read_bytes()).hexdigest() == hashlib.sha256((TMP / "sel_b" / "vae_audit_ids.txt").read_bytes()).hexdigest()
    print("[2] selection unique + deterministic re-run hash OK")


def test_selection_requires_exact_count() -> None:
    _reset()
    gaze = TMP / "gaze.csv"
    _write_gaze(gaze, [{"image_id": "v0", "split": "validation", "status": "candidate_unvalidated",
                        "pose_x": "0", "pose_y": "0", "pose_z": "0"}])
    (TMP / "val.txt").write_text("v0\n", encoding="utf-8")
    (TMP / "fixed.txt").write_text("f0\n", encoding="utf-8")
    try:
        select_audit_ids(gaze, TMP / "val.txt", TMP / "fixed.txt", 2, TMP / "sel")
    except ValueError:
        pass
    else:
        raise AssertionError("selection silently returned fewer rows than requested")
    print("[2b] selection fails closed when eligible candidates are insufficient OK")


def test_posterior_mode_not_sample() -> None:
    mode = torch.ones(1, 4, 8, 8)
    vae = FakeVAE(mode, scaling=0.18215)
    recon = encode_decode(vae, torch.zeros(1, 3, 8, 8))
    assert torch.allclose(vae.decoded_z, mode)
    assert recon.shape == (1, 4, 8, 8)
    print("[3][4] posterior mode() + scaling_factor on encode/decode OK")


def test_missing_preserved_empty_not_zero() -> None:
    row = _mk(vae="fail")
    assert row["psnr_rgb"] == "" and row["ssim_rgb"] == "" and row["arcface_cosine"] == ""
    assert row["head_pose_delta_deg"] == "" and row["gaze_camera_delta_deg"] == ""
    assert row["gaze_head_delta_deg"] == "" and row["gaze_coordinate_status"] == "candidate_unvalidated_diagnostic_only"
    assert row["arcface_source_det_score"] == "" and row["arcface_recon_det_score"] == ""
    print("[5][6] missing preserved as empty (not 0); head-local gaze empty when unapproved OK")


def test_freeze() -> None:
    model = torch.nn.Linear(4, 2)
    freeze_model(model)
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
    print("[7] model frozen (eval + requires_grad=False) OK")


def test_resume_no_overwrite_within_selection() -> None:
    _reset()
    metrics = TMP / "m.csv"
    with metrics.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "vae_status"])
        w.writeheader()
        w.writerows([{"image_id": "a", "vae_status": "success"}, {"image_id": "b", "vae_status": "success"}, {"image_id": "c", "vae_status": "fail"}])
    done = load_completed(metrics, ["a", "c"])
    assert done == {"a": {"image_id": "a", "vae_status": "success"}}, "resume must keep only in-selection success samples"
    print("[8] resume keeps only success within selection OK")


def test_summary_full_denominator() -> None:
    rows = [
        _mk(vae="success", cosine="0.8", a_src="success", a_rec="success", head="1.0", gaze="2.0", l_src="success", l_rec="success", runtime="0.1"),
        _mk(vae="success", cosine="", a_src="fail", a_rec="fail", head="", gaze="", l_src="fail", l_rec="fail", runtime="0.2"),
        _mk(vae="fail", runtime="0.3"),
    ]
    s = summarize(rows, {"n_total": 3, "current_run_wall_seconds": 5.0, "gpu_peak_mb": 100.0, "deca_preprocess": "fan", "arcface_det_thresh": 0.1})
    assert s["n_total"] == 3 and s["rows"] == 3
    assert s["vae"]["coverage"] == 2 / 3
    assert s["arcface"]["pair_coverage"] == 1 / 3 and s["arcface"]["source_coverage"] == 1 / 3
    assert s["arcface_det_thresh"] == 0.1 and s["deca_preprocess"] == "fan"
    assert s["head_local_gaze_not_evaluated"] is True
    print("[9] summary denominator = full selection + new fields OK")


def test_dry_run_no_vae_load() -> None:
    _reset()
    gaze = TMP / "gaze.csv"
    _write_gaze(gaze, [{"image_id": f"v{i}", "split": "validation", "status": "candidate_unvalidated", "pose_x": "0.1", "pose_y": "0", "pose_z": "0"} for i in range(5)])
    (TMP / "val.txt").write_text("\n".join(f"v{i}" for i in range(5)) + "\n", encoding="utf-8")
    (TMP / "fixed.txt").write_text("f1\n", encoding="utf-8")
    base = TMP / "base.csv"
    with base.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "image_path"])
        w.writeheader()
        w.writerows([{"image_id": f"v{i}", "image_path": ""} for i in range(5)])
    r = subprocess.run(
        [PY, "-m", "phase3.audit_vae_roundtrip", "--base-manifest", str(base), "--validation-ids", str(TMP / "val.txt"),
         "--fixed-test-ids", str(TMP / "fixed.txt"), "--gaze-candidates", str(gaze), "--selection-count", "5",
         "--out-dir", str(TMP / "out"), "--dry-run"],
        cwd=PROJECT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    assert r.returncode == 0
    assert (TMP / "out" / "config.json").exists()
    assert (TMP / "out" / "selection" / "vae_audit_ids.txt").exists()
    print("[10] dry-run completes without loading/downloading VAE OK")


def test_l2cs_no_face_no_crash() -> None:
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    gd, s1, s2 = l2cs_pair(FakeGaze(), rgb, rgb)
    assert gd == "" and s1 == "no_face_detected" and s2 == "no_face_detected"
    print("[11] L2CS no-face returns empty metric + explicit status (no TypeError) OK")


def test_arcface_default_thresh() -> None:
    sig = inspect.signature(load_arcface)
    assert sig.parameters["det_thresh"].default == 0.1
    print("[12] ArcFace default det_thresh == 0.1 OK")


def test_resume_config_mismatch_rejected() -> None:
    prev = {"selection_hash": "aaa", "base_manifest_sha256": "base", "vae_resolved": "x",
            "vae_snapshot_sha256": "weights-a", "resolution": 256, "dtype": "fp16",
            "scaling_factor": 0.18215, "arcface_mode": "existing", "deca_mode": "existing",
            "l2cs_mode": "existing", "arcface_det_thresh": 0.1, "deca_preprocess": "fan"}
    cur = dict(prev)
    cur["vae_snapshot_sha256"] = "weights-b"
    try:
        verify_resume_config(prev, cur)
    except RuntimeError:
        pass
    else:
        raise AssertionError("verify_resume_config did not reject mismatched config")
    print("[13] resume rejects an in-place VAE snapshot change OK")


def test_tree_hash_detects_content_change() -> None:
    _reset()
    model = TMP / "model"
    model.mkdir()
    weight = model / "weights.bin"
    weight.write_bytes(b"a")
    before = _tree_sha256(model)
    weight.write_bytes(b"b")
    assert before != _tree_sha256(model)
    print("[13b] VAE tree hash detects in-place weight replacement OK")


def test_artifact_hash_after_artifacts() -> None:
    _reset()
    out = TMP / "hashout"
    out.mkdir(parents=True)
    (out / "config.json").write_text('{"a":1}', encoding="utf-8")
    (out / "vae_roundtrip_metrics.csv").write_text("image_id\n1\n", encoding="utf-8")
    _write_artifact_hashes(out, {}, None)
    content = (out / "artifact_hashes.sha256").read_text(encoding="utf-8")
    assert "config.json" in content and "vae_roundtrip_metrics.csv" in content
    assert "artifact_hashes.sha256" not in content
    print("[14] artifact hash covers final artifacts and excludes itself OK")


def main() -> None:
    test_selection_fail_closed_on_fixed()
    test_selection_unique_deterministic()
    test_selection_requires_exact_count()
    test_posterior_mode_not_sample()
    test_missing_preserved_empty_not_zero()
    test_freeze()
    test_resume_no_overwrite_within_selection()
    test_summary_full_denominator()
    test_dry_run_no_vae_load()
    test_l2cs_no_face_no_crash()
    test_arcface_default_thresh()
    test_resume_config_mismatch_rejected()
    test_tree_hash_detects_content_change()
    test_artifact_hash_after_artifacts()
    print("PHASE3 VAE AUDIT PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    main()
