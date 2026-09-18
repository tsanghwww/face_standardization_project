"""High-resolution target-minus-source geometry editing for Phase3.1g."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DownBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__()
        groups = min(8, output_channels)
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, output_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.block(value)


class HighResolutionGeometryEditAdapter(nn.Module):
    """Encode a 256-scale geometry delta before constructing a UNet pyramid.

    Unlike the original FaceControlAdapter, this module does not first collapse
    the condition to the 32x32 latent grid. A per-sample edit gate guarantees an
    exact zero residual when target and source conditions are identical.
    """

    def __init__(self, output_channels=(320, 640, 1280, 1280), input_channels: int = 6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        widths = (32, 48, 64, 96, 128, 160)
        blocks = []
        previous = 32
        for width in widths:
            blocks.append(DownBlock(previous, width))
            previous = width
        self.blocks = nn.ModuleList(blocks)
        pyramid_widths = (64, 96, 128, 160)
        self.outputs = nn.ModuleList()
        for width, output in zip(pyramid_widths, output_channels):
            projection = nn.Conv2d(width, output, 1)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.outputs.append(projection)

    def forward(self, source_condition: torch.Tensor, target_condition: torch.Tensor,
                latent_size: tuple[int, int]) -> list[torch.Tensor]:
        if source_condition.shape != target_condition.shape or source_condition.ndim != 4:
            raise ValueError("Source and target geometry conditions must have matching BCHW shapes")
        delta = target_condition - source_condition
        gate = (delta.detach().abs().amax(dim=(1, 2, 3), keepdim=True) > 1e-8).to(delta.dtype)
        high_resolution = (int(latent_size[0]) * 8, int(latent_size[1]) * 8)
        feature = F.interpolate(delta, size=high_resolution, mode="bilinear", align_corners=False)
        feature = self.stem(feature)
        pyramid = []
        for index, block in enumerate(self.blocks):
            feature = block(feature)
            if index >= 2:
                pyramid.append(feature)
        if len(pyramid) != 4:
            raise RuntimeError("Geometry edit pyramid must contain four scales")
        residuals = [projection(value) * gate for projection, value in zip(self.outputs, pyramid)]
        expected_sizes = [
            (int(latent_size[0]) // (2 ** index), int(latent_size[1]) // (2 ** index))
            for index in range(4)
        ]
        if [tuple(value.shape[-2:]) for value in residuals] != expected_sizes:
            raise RuntimeError("Geometry edit pyramid does not align with UNet scales")
        return residuals


class GeometryResidualControl(nn.Module):
    """Frozen source reconstruction plus a trainable target-minus-source edit."""

    architecture = "highres_geometry_delta_v1"

    def __init__(self, base_model):
        super().__init__()
        self.base = base_model.requires_grad_(False).eval()
        self.edit = HighResolutionGeometryEditAdapter(base_model.unet.config.block_out_channels)

    def forward(self, noisy, timestep, source_condition, target_condition, identity, empty_prompt):
        edit_residuals = self.edit(source_condition, target_condition, noisy.shape[-2:])
        source_residuals = self.base.face(source_condition, noisy.shape[-2:])
        if len(source_residuals) != len(edit_residuals):
            raise ValueError("Source and edit residual pyramids differ in length")
        residuals = []
        for source, edit in zip(source_residuals, edit_residuals):
            if source.shape != edit.shape:
                raise ValueError(f"Source/edit residual shape mismatch: {source.shape} != {edit.shape}")
            residuals.append(source + edit)
        tokens = self.base.identity(identity)
        from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0

        for processor in self.base.unet.attn_processors.values():
            if isinstance(processor, IPAdapterAttnProcessor2_0):
                processor.scale = [1.0]
        return self.base.unet(
            noisy,
            timestep,
            encoder_hidden_states=(empty_prompt.expand(noisy.shape[0], -1, -1), [tokens]),
            down_intrablock_additional_residuals=[value.to(noisy.dtype) for value in residuals],
        ).sample

    def source_forward(self, noisy, timestep, source_condition, identity, empty_prompt):
        """Exact frozen no-edit path used as the paired source reference."""
        return self.base(noisy, timestep, source_condition, identity, empty_prompt)

    def edit_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu() for name, value in self.edit.state_dict().items()}


def minimum_pair_separation_loss(
    output_separation: torch.Tensor,
    target_separation: torch.Tensor,
    minimum_transfer_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize a counterfactual pair that moves less than a target fraction.

    This prevents a ranking hinge from succeeding on an arbitrarily small
    directionally correct change. The target separation is supervision only.
    """
    if not 0 < minimum_transfer_ratio <= 1:
        raise ValueError("minimum_transfer_ratio must be in (0, 1]")
    if not (torch.isfinite(output_separation).all() and torch.isfinite(target_separation).all()):
        raise ValueError("Pair separations must be finite")
    required = target_separation.detach() * minimum_transfer_ratio
    loss = torch.relu(required - output_separation) / required.clamp_min(1e-6)
    return loss, required
