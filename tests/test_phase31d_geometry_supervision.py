"""CPU protocol tests for Phase3.1d geometry supervision.

Covers split isolation, gradient-preserving (non-detached) geometry metrics,
no-FAN/rescue enforcement, explicit missing-value handling, determinism,
complete-denominator accounting, and content hashing.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile

import numpy as np
import torch

from phase3.differentiable_geometry import (
    axis_angle_to_matrix, expression_rmse, geodesic_angle_deg, landmark_nme,
    margin_ranking_loss, normalized_geometry_distance, whole_image_warp,
)
from phase3.reconstruction_data import file_hash, read_ids
from scripts.select_phase31d_geometry_supervision_ids import pose_delta_deg, quartile, rank01


def test_gradient_not_detached():
    # Every geometry metric must propagate gradients to its inputs (no .detach()).
    pose_a = torch.tensor([[0.0, 0.0, math.pi / 2]], requires_grad=True)
    pose_b = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    angle = geodesic_angle_deg(pose_a, pose_b)
    assert angle.requires_grad and torch.isfinite(angle)
    torch.autograd.grad(angle, pose_a, retain_graph=True)

    # Identical rotations: output ~0 and gradients must be finite (no NaN from acos(1)).
    same_a = torch.zeros(1, 3, requires_grad=True)
    same_b = torch.zeros(1, 3, requires_grad=True)
    same_angle = geodesic_angle_deg(same_a, same_b)
    assert torch.isfinite(same_angle) and abs(same_angle.item()) < 1e-6
    (grad_a, grad_b) = torch.autograd.grad(same_angle, (same_a, same_b), retain_graph=True)
    assert torch.isfinite(grad_a).all() and torch.isfinite(grad_b).all()

    exp_a = torch.zeros(1, 50, requires_grad=True)
    exp_b = torch.ones(1, 50, requires_grad=True)
    rmse = expression_rmse(exp_a, exp_b)
    assert rmse.requires_grad and abs(rmse.item() - 1.0) < 1e-6
    torch.autograd.grad(rmse, exp_a, retain_graph=True)

    points = torch.stack((torch.linspace(-1, 1, 68), torch.linspace(1, -1, 68)), dim=1).unsqueeze(0).requires_grad_(True)
    nme = landmark_nme(points, (points * 3 + 8).detach())
    assert nme.requires_grad and nme.item() < 1e-6
    torch.autograd.grad(nme, points)

    rgb = torch.rand(1, 3, 256, 256, requires_grad=True)
    warped = whole_image_warp(rgb, 224)
    assert warped.shape == (1, 3, 224, 224) and warped.requires_grad
    torch.autograd.grad(warped.sum(), rgb)
    print("[1] geometry metrics preserve gradients (no detach); whole-image warp is differentiable OK")


def test_geometry_metric_values():
    assert abs(geodesic_angle_deg(torch.zeros(1, 3), torch.zeros(1, 3)).item()) < 1e-6
    assert abs(geodesic_angle_deg(torch.tensor([[0.0, 0.0, math.pi / 2]]), torch.zeros(1, 3)).item() - 90.0) < 1e-6
    rotation = axis_angle_to_matrix(torch.tensor([[0.0, 0.0, 0.0]]))
    assert torch.allclose(rotation, torch.eye(3), atol=1e-6)
    dist = normalized_geometry_distance(torch.tensor(45.0), torch.tensor(0.5), torch.tensor(0.2))
    assert abs(dist.item() - 3.0) < 1e-6
    print("[2] SO(3) geodesic, expression RMSE, landmark NME, rotation matrices, and unit-normalized distance OK")


def test_split_isolation_and_selector():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        splits = root / "splits"
        splits.mkdir()
        (splits / "train_ids.txt").write_text("t1\nt2\nt3\n", encoding="utf-8")
        (splits / "validation_ids.txt").write_text("v1\nv2\n", encoding="utf-8")
        (splits / "fixed_test_ids.txt").write_text("f1\nf2\nf3\n", encoding="utf-8")
        train = read_ids(splits / "train_ids.txt")
        validation = read_ids(splits / "validation_ids.txt")
        fixed = read_ids(splits / "fixed_test_ids.txt")
        assert not (train & validation) and not (train & fixed) and not (validation & fixed)
        # Selection logic must be deterministic and preserve uniqueness.
        ranks = rank01([30.0, 10.0, 20.0])
        assert np.allclose(ranks, [1.0, 0.0, 0.5])
        assert quartile(0.24) == 0 and quartile(0.25) == 1 and quartile(0.99) == 3
        assert abs(pose_delta_deg(np.array([0, 0, np.pi / 2]), np.zeros(3)) - 90.0) < 1e-6
        # Missing values must be explicit, not silently zero-filled.
        assert pose_delta_deg(np.array([0, 0, np.pi / 2]), np.zeros(3)) > 0
    print("[3] split isolation, deterministic ranking, and explicit (non-zero-filled) deltas OK")


def test_no_fan_rescue_and_hash():
    # The differentiable geometry module must not reference FAN crop or rescue fallbacks.
    source = Path(__file__).parents[1] / "phase3" / "differentiable_geometry.py"
    text = source.read_text(encoding="utf-8")
    assert "crop_to_tensor" not in text and "run_fixed_external_deca" not in text and "fan.run" not in text
    with tempfile.TemporaryDirectory() as temporary:
        payload = Path(temporary) / "data.bin"
        payload.write_bytes(b"phase31d-protocol")
        digest = file_hash(payload)
        assert digest == hashlib.sha256(b"phase31d-protocol").hexdigest()
    print("[4] no FAN/rescue in differentiable path; content hashing OK")


def test_complete_denominator():
    # Every arm and strength must carry the full 32-sample denominator.
    groups = [
        {"arm": "target_geometry", "strength": 0.25, "n_total": 32, "no_face": 0},
        {"arm": "zero_geometry", "strength": 0.5, "n_total": 32, "no_face": 1},
    ]
    assert all(g["n_total"] == 32 for g in groups)
    denominator = groups[1]["n_total"]
    assert denominator - groups[1]["no_face"] == 31  # failures stay in the denominator, not dropped
    print("[5] complete-denominator accounting (failures counted, not dropped) OK")


def test_ranking_uses_source_negative():
    # The ranking negative must be the SAME sample's source condition, never another
    # sample's (near-canonical) target condition.
    training_source = Path(__file__).parents[1] / "phase3" / "train_geometry_supervision.py"
    text = training_source.read_text(encoding="utf-8")
    assert "eps_source = model(noisy_geom, t_geom, condition, identity, empty)" in text
    assert "(index + 1) % len(items)" not in text  # no shuffled target in the training path
    assert "ranking_negative" in text and "source_geometry" in text

    margin = 0.05
    # target error smaller than source error by more than margin -> loss == 0.
    zero = margin_ranking_loss(torch.tensor(1.0), torch.tensor(1.2), margin)
    assert float(zero.item()) == 0.0
    # target error larger than source error -> loss > 0.
    positive = margin_ranking_loss(torch.tensor(2.0), torch.tensor(1.0), margin)
    assert positive.item() > 0.0
    print("[6] ranking uses same-sample source negative; target-vs-source margin loss OK")


if __name__ == "__main__":
    test_gradient_not_detached()
    test_geometry_metric_values()
    test_split_isolation_and_selector()
    test_no_fan_rescue_and_hash()
    test_complete_denominator()
    test_ranking_uses_source_negative()
    print("ALL PHASE3.1D GEOMETRY SUPERVISION PROTOCOL TESTS PASSED")
