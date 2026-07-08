import os
import sys

ROOT = r"D:\face_standardization_project"

sys.path.insert(0, os.path.join(ROOT, "tools"))
import compile_deca_rasterizer  # noqa: F401

import torch

from decalib.utils.renderer import StandardRasterizer

device = torch.device("cuda")
rasterizer = StandardRasterizer(32).to(device)
vertices = torch.tensor(
    [[[-0.6, -0.6, 0.5], [0.6, -0.6, 0.5], [0.0, 0.6, 0.5]]],
    device=device,
)
faces = torch.tensor([[[0, 1, 2]]], device=device, dtype=torch.long)
attributes = torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]], device=device)

out = rasterizer(vertices, faces, attributes)
print("rasterizer smoke ok", out.shape, out.device, float(out[:, -1].sum().item()))
