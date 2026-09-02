"""Source reconstruction adapters; no gaze inputs and no intervention supervision."""

import hashlib

import torch
from torch import nn
from torch.nn import functional as F
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0


class FaceControlAdapter(nn.Module):
    def __init__(self, channels=(320, 640, 1280, 1280)):
        super().__init__()
        widths = (32, 64, 96, 128)
        self.blocks = nn.ModuleList()
        self.outputs = nn.ModuleList()
        previous = 6
        for width, output in zip(widths, channels):
            self.blocks.append(nn.Sequential(nn.Conv2d(previous, width, 3, padding=1),
                                            nn.GroupNorm(8, width), nn.SiLU(),
                                            nn.Conv2d(width, width, 3, padding=1), nn.SiLU()))
            projection = nn.Conv2d(width, output, 1)
            nn.init.zeros_(projection.weight)
            nn.init.zeros_(projection.bias)
            self.outputs.append(projection)
            previous = width

    def forward(self, condition, latent_size):
        feature = F.interpolate(condition, size=latent_size, mode='bilinear', align_corners=False)
        residuals = []
        for index, (block, projection) in enumerate(zip(self.blocks, self.outputs)):
            if index:
                feature = F.avg_pool2d(feature, 2)
            feature = block(feature)
            residuals.append(projection(feature))
        return residuals


class ReconstructionAdapter(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet.requires_grad_(False).eval()
        self.face = FaceControlAdapter(unet.config.block_out_channels)
        dimension = unet.config.cross_attention_dim
        self.identity = nn.Sequential(nn.Linear(512, 4 * dimension), nn.Unflatten(1, (4, dimension)),
                                      nn.LayerNorm(dimension))
        processors = {}
        for name, processor in unet.attn_processors.items():
            if name.endswith('attn1.processor'):
                processors[name] = processor
                continue
            attention = unet.get_submodule(name.removesuffix('.processor'))
            ip = IPAdapterAttnProcessor2_0(attention.to_q.out_features, dimension, num_tokens=(4,))
            with torch.no_grad():
                ip.to_k_ip[0].weight.copy_(attention.to_k.weight.float())
                ip.to_v_ip[0].weight.zero_()
            processors[name] = ip
        unet.set_attn_processor(processors)

    def forward(self, noisy, timestep, condition, identity, empty_prompt, face_enabled=True, identity_enabled=True):
        residuals = self.face(condition, noisy.shape[-2:])
        if not face_enabled:
            residuals = [r * 0 for r in residuals]
        tokens = self.identity(identity)
        # Skip the identity branch by scale, not by feeding a zero embedding.
        for processor in self.unet.attn_processors.values():
            if isinstance(processor, IPAdapterAttnProcessor2_0):
                processor.scale = [1.0 if identity_enabled else 0.0]
        return self.unet(noisy, timestep,
                         encoder_hidden_states=(empty_prompt.expand(noisy.shape[0], -1, -1), [tokens]),
                         down_intrablock_additional_residuals=[r.to(noisy.dtype) for r in residuals]).sample

    def adapter_state(self):
        return {name: value.detach().cpu() for name, value in self.state_dict().items()
                if not name.startswith('unet.') or '.processor.' in name}

    def frozen_hash(self):
        digest = hashlib.sha256()
        for name, parameter in self.unet.named_parameters():
            if '.processor.' not in name:
                if parameter.requires_grad or parameter.grad is not None:
                    raise RuntimeError(f'Backbone is not frozen: {name}')
                digest.update(name.encode())
                digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()
