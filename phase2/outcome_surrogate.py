"""Multi-head outcome surrogate model for Phase2.1.

Predicts rendered *diagnostic* outcomes (identity/pose/gaze/render-failure) from
source quality features + alphas + standardized params.  These are NOT real
ArcFace/L2CS measurements -- they are surrogates of the DECA-render diagnostic
metrics, used to make gate decisions without re-rendering.

Heads:
  identity        -> 1 scalar (arcface_cosine / identity_delta)
  pose            -> 1 scalar (pose_improvement_vs_original)
  gaze            -> 1 scalar (l2cs_gaze_delta_vs_original_deg)
  render_failure  -> 1 scalar (logit for P(render_full_status != success))
"""

from __future__ import annotations

import torch
from torch import nn


class OutcomeSurrogate(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head_identity = nn.Linear(hidden_dim, 1)
        self.head_pose = nn.Linear(hidden_dim, 1)
        self.head_gaze = nn.Linear(hidden_dim, 1)
        self.head_render_failure = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(features)
        return {
            "identity": self.head_identity(h).squeeze(-1),
            "pose": self.head_pose(h).squeeze(-1),
            "gaze": self.head_gaze(h).squeeze(-1),
            "render_failure": self.head_render_failure(h).squeeze(-1),
        }
