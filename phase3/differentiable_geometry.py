"""Differentiable DECA geometry estimation and losses for Phase3.1d supervision.

The training path is Face Adapter -> frozen UNet -> one-step x0 -> frozen VAE
decode -> fixed whole-image warp -> frozen DECA encoder. It contains no FAN, no
rescue, and no nondifferentiable detector. The DECA parameters are frozen
(requires_grad=False), but the forward graph must propagate gradients to the
input so the Face Adapter can be optimized. Missing or nonfinite estimates are
never zero-filled.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


def load_deca_frozen(deca_root: Path, device: str):
    """Load DECA for differentiable geometry estimation (no renderer, frozen params)."""
    import sys

    sys.path.insert(0, str(deca_root))
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = "standard"
    deca_cfg.model.extract_tex = False
    deca = DECA(config=deca_cfg, device=device, render_enabled=False)
    deca.eval()
    deca.requires_grad_(False)
    return deca


def whole_image_warp(rgb01: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Fixed differentiable whole-image warp (bilinear resize). No detector/FAN.

    Input `rgb01` is a (B, 3, H, W) tensor in [0, 1]. The resize is the fixed,
    detector-free warp used by the Phase3.1d differentiable training path.
    """
    if rgb01.ndim != 4 or rgb01.shape[1] != 3:
        raise ValueError(f"Expected (B,3,H,W) RGB in [0,1], got {tuple(rgb01.shape)}")
    if not torch.isfinite(rgb01).all():
        raise ValueError("Nonfinite RGB passed to whole-image warp")
    return F.interpolate(rgb01, size=(size, size), mode="bilinear", align_corners=False)


def estimate_geometry(deca, rgb: torch.Tensor):
    """Differentiable DECA geometry estimate from a decoded RGB tensor.

    `rgb` is (B, 3, H, W) in [-1, 1] (diffusers convention). Returns
    pose (B,6), expression (B,50), landmarks (B,68,2). Raises on nonfinite.
    """
    rgb01 = (rgb.clamp(-1.0, 1.0) + 1.0) * 0.5
    warped = whole_image_warp(rgb01, int(deca.image_size))
    codedict = deca.encode(warped, use_detail=False)
    op = deca.decode(codedict, rendering=False, return_vis=False)
    pose = codedict["pose"]
    expression = codedict["exp"]
    landmarks = op["landmarks2d"][:, :, :2]
    if not (torch.isfinite(pose).all() and torch.isfinite(expression).all() and torch.isfinite(landmarks).all()):
        raise ValueError("Nonfinite differentiable DECA estimate")
    return pose, expression, landmarks


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation matrix from axis-angle vectors (..., 3) -> (..., 3, 3)."""
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zeros = torch.zeros_like(x)
    k = torch.stack([
        torch.stack([zeros, -z, y], dim=-1),
        torch.stack([z, zeros, -x], dim=-1),
        torch.stack([-y, x, zeros], dim=-1),
    ], dim=-2)
    eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)
    sin, cos = torch.sin(angle), torch.cos(angle)
    rotation = eye + sin.unsqueeze(-1) * k + (1.0 - cos).unsqueeze(-1) * (k @ k)
    return rotation


def geodesic_angle_deg(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SO(3) geodesic angle (degrees) between axis-angle vectors (..., 3).

    Uses the stable atan2 form on the relative rotation matrix rather than
    acos(1), so that identical rotations give a finite (non-NaN) gradient.
    """
    relative = axis_angle_to_matrix(a).transpose(-1, -2) @ axis_angle_to_matrix(b)
    trace = torch.diagonal(relative, dim1=-2, dim2=-1).sum(-1)
    cos_angle = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
    skew = relative - relative.transpose(-1, -2)
    sin_angle = torch.norm(skew, dim=(-2, -1)) / (2.0 * math.sqrt(2.0))
    angle = torch.atan2(sin_angle, cos_angle)
    return torch.rad2deg(angle)


def expression_rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """RMSE over the last dimension (50D expression)."""
    if a.shape[-1] != 50 or b.shape[-1] != 50:
        raise ValueError("Expression vectors must have 50 dimensions")
    return torch.sqrt(torch.mean(torch.square(a - b), dim=-1))


def normalize_landmarks(points: torch.Tensor) -> torch.Tensor:
    """Center and RMS-scale normalize landmarks (..., 68, 2)."""
    if points.shape[-2] != 68 or points.shape[-1] < 2:
        raise ValueError(f"Invalid landmarks shape {tuple(points.shape)}")
    pts = points[..., :2]
    centered = pts - pts.mean(dim=-2, keepdim=True)
    scale = torch.sqrt(torch.mean(torch.sum(torch.square(centered), dim=-1), dim=-1, keepdim=True))
    scale = scale + (scale < 1e-8).to(scale.dtype)
    return centered / scale.unsqueeze(-1)


def landmark_nme(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean normalized landmark error after independent centering + RMS scaling."""
    return torch.mean(torch.norm(normalize_landmarks(a) - normalize_landmarks(b), dim=-1), dim=-1)


def one_step_x0(scheduler, noisy: torch.Tensor, timestep, epsilon_theta: torch.Tensor) -> torch.Tensor:
    """One-step clean-latent estimate from DDPM forward noising."""
    alpha = scheduler.alphas_cumprod.to(noisy.device)[timestep]
    while alpha.ndim < noisy.ndim:
        alpha = alpha.unsqueeze(-1)
    return (noisy - torch.sqrt(1.0 - alpha) * epsilon_theta) / torch.sqrt(alpha)


def load_target_geometry(deca, deca_mat: Path, phase2_npz: Path, device: str) -> dict:
    """Frozen Phase2 target geometry: pose (6,), expression (50,), landmarks (68,2)."""
    from scripts.build_phase3_condition_cache import apply_phase2, codedict, load_source_params

    source = load_source_params(deca_mat)
    target = apply_phase2(source, phase2_npz)
    with torch.no_grad():
        op = deca.decode(codedict(target, device, int(deca.image_size)), rendering=False, return_vis=False)
    pose = torch.from_numpy(np.asarray(target["pose"], dtype=np.float32)).to(device)
    expression = torch.from_numpy(np.asarray(target["expression"], dtype=np.float32)).to(device)
    landmarks = op["landmarks2d"][0, :, :2].clone()
    if pose.shape != (6,) or expression.shape != (50,) or landmarks.shape != (68, 2):
        raise ValueError(f"Invalid target geometry: pose {tuple(pose.shape)} exp {tuple(expression.shape)} lm {tuple(landmarks.shape)}")
    if not (torch.isfinite(pose).all() and torch.isfinite(expression).all() and torch.isfinite(landmarks).all()):
        raise ValueError("Nonfinite target geometry")
    return {"pose": pose, "expression": expression, "landmarks": landmarks}


def geometry_loss(
    pose_pred: torch.Tensor,
    exp_pred: torch.Tensor,
    lm_pred: torch.Tensor,
    pose_tgt: torch.Tensor,
    exp_tgt: torch.Tensor,
    lm_tgt: torch.Tensor,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Weighted pose SO(3) + expression RMSE + landmark NME loss.

    Returns a dict with the three scalar terms and the weighted total. Missing
    values are surfaced by raising, never zero-filled.
    """
    pose_error = geodesic_angle_deg(pose_pred[:, :3], pose_tgt[:, :3])
    exp_error = expression_rmse(exp_pred, exp_tgt)
    lm_error = landmark_nme(lm_pred, lm_tgt)
    if not (torch.isfinite(pose_error).all() and torch.isfinite(exp_error).all() and torch.isfinite(lm_error).all()):
        raise ValueError("Nonfinite geometry loss term")
    w_p, w_e, w_l = weights
    terms = {
        "pose_so3_deg": pose_error.mean(),
        "expression_rmse": exp_error.mean(),
        "landmark_nme": lm_error.mean(),
    }
    total = w_p * terms["pose_so3_deg"] + w_e * terms["expression_rmse"] + w_l * terms["landmark_nme"]
    return {"terms": terms, "total": total}


def normalized_geometry_distance(pose_err: torch.Tensor, exp_err: torch.Tensor, lm_err: torch.Tensor) -> torch.Tensor:
    """Normalized sum of the three target errors for the ranking loss.

    pose_err is already in degrees (from geodesic_angle_deg), so the pose term
    is normalized by 45 degrees, not by radians(45).
    """
    return pose_err / 45.0 + exp_err / 0.5 + lm_err / 0.2


def margin_ranking_loss(
    target_distance: torch.Tensor,
    negative_distance: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Require the target arm to beat the paired negative by ``margin``."""
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Ranking margin must be finite and nonnegative")
    if not (torch.isfinite(target_distance).all() and torch.isfinite(negative_distance).all()):
        raise ValueError("Nonfinite geometry distance passed to ranking loss")
    return torch.clamp(margin + target_distance - negative_distance, min=0.0)
