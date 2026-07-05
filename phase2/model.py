"""Phase2 condition generator model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class Phase2Output:
    target_expression: torch.Tensor
    target_head_pose: torch.Tensor
    target_jaw_pose: torch.Tensor
    alpha_expression: torch.Tensor
    alpha_head_pose: torch.Tensor
    alpha_jaw_pose: torch.Tensor
    confidence: torch.Tensor
    reject_score: torch.Tensor
    standardized_expression: torch.Tensor
    standardized_pose: torch.Tensor


class ConditionGenerator(nn.Module):
    """Predict canonical targets, standardization strengths, and quality gates."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.10):
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
            nn.Linear(hidden_dim, 50 + 3 + 3 + 3 + 1 + 1),
        )

    def forward(self, features: torch.Tensor, expression: torch.Tensor, pose: torch.Tensor) -> Phase2Output:
        raw = self.backbone(features)
        idx = 0
        target_exp = raw[:, idx : idx + 50]
        idx += 50
        target_head = raw[:, idx : idx + 3]
        idx += 3
        target_jaw = raw[:, idx : idx + 3]
        idx += 3
        alphas = torch.sigmoid(raw[:, idx : idx + 3])
        idx += 3
        confidence = torch.sigmoid(raw[:, idx : idx + 1])
        idx += 1
        reject = torch.sigmoid(raw[:, idx : idx + 1])

        alpha_exp = alphas[:, 0:1]
        alpha_head = alphas[:, 1:2]
        alpha_jaw = alphas[:, 2:3]
        std_exp = (1.0 - alpha_exp) * expression + alpha_exp * target_exp
        std_head = (1.0 - alpha_head) * pose[:, :3] + alpha_head * target_head
        std_jaw = (1.0 - alpha_jaw) * pose[:, 3:] + alpha_jaw * target_jaw
        std_pose = torch.cat([std_head, std_jaw], dim=1)
        return Phase2Output(
            target_expression=target_exp,
            target_head_pose=target_head,
            target_jaw_pose=target_jaw,
            alpha_expression=alpha_exp,
            alpha_head_pose=alpha_head,
            alpha_jaw_pose=alpha_jaw,
            confidence=confidence,
            reject_score=reject,
            standardized_expression=std_exp,
            standardized_pose=std_pose,
        )

