"""CPU protocol checks for source-only train-split reconstruction."""

import json
from functools import partial
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from diffusers import UNet2DConditionModel
from torch.utils.checkpoint import checkpoint

from phase3.reconstruction_adapter import ReconstructionAdapter
from phase3.reconstruction_data import ReconstructionDataset, registry


def test_model():
    torch.manual_seed(11)
    torch.set_num_threads(2)
    unet = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1,
                               block_out_channels=(32,64,64,64), cross_attention_dim=32,
                               attention_head_dim=8, norm_num_groups=8,
                               down_block_types=('CrossAttnDownBlock2D',)*3+('DownBlock2D',),
                               up_block_types=('UpBlock2D',)+('CrossAttnUpBlock2D',)*3)
    noisy, cond, identity, empty = torch.randn(1,4,8,8), torch.rand(1,6,64,64), torch.rand(1,512), torch.randn(1,77,32)
    timestep = torch.tensor([250])
    with torch.no_grad():
        reference = unet(noisy, timestep, encoder_hidden_states=empty).sample
    model = ReconstructionAdapter(unet)
    unet.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
    frozen = model.frozen_hash()
    output = model(noisy, timestep, cond, identity, empty)
    assert torch.allclose(reference, output, atol=2e-6), (reference-output).abs().max()
    assert all(torch.count_nonzero(r) == 0 for r in model.face(cond, (8,8)))
    before = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    for _ in range(3):
        optimizer.zero_grad()
        output = model(noisy, timestep, cond, identity, empty)
        output.square().mean().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    for prefix in ('face.', 'identity.', 'unet.'):
        assert any(not torch.equal(before[n], p) for n,p in model.named_parameters() if n in before and n.startswith(prefix)), prefix
    assert model.frozen_hash() == frozen
    with torch.no_grad():
        restored = model(noisy, timestep, cond, identity, empty, face_enabled=False, identity_enabled=False)
        assert torch.allclose(reference, restored, atol=2e-6)
    assert all(not name.startswith('unet.') or '.processor.' in name for name in model.adapter_state())
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'adapter.pt'
        torch.save(model.adapter_state(), path)
        saved = torch.load(path, weights_only=True)
        parameter = next(p for p in model.parameters() if p.requires_grad)
        with torch.no_grad():
            parameter.add_(1)
        status = model.load_state_dict(saved, strict=False)
        assert not status.unexpected_keys
        assert all(torch.equal(value, model.state_dict()[name]) for name,value in saved.items())
        assert model.frozen_hash() == frozen
    print('zero-init parity, adapter gradients/updates, frozen backbone, branch-off baseline: OK')


def test_dataset():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, value in [('train_ids.txt','a\n'), ('validation_ids.txt','b\n'), ('fixed_test_ids.txt','c\n')]:
            (root/name).write_text(value)
        Image.new('RGB',(32,32),(100,120,140)).save(root/'image.png')
        Image.new('L',(32,32),255).save(root/'mask.png')
        Image.fromarray(np.full((32,32),32768,dtype=np.uint16)).save(root/'depth.png')
        np.save(root/'id.npy', np.ones(512,dtype=np.float32))
        row = {'image_id':'a', 'split':'train', 'rescue_source':False,
               'condition_cache_status':'geometry_ready_gaze_pending',
               'source_image':str(root/'image.png'), 'source_normal_map':str(root/'image.png'),
               'source_depth_map':str(root/'depth.png'), 'source_landmark_map':str(root/'mask.png'),
               'source_face_mask':str(root/'mask.png'), 'arcface_embedding':str(root/'id.npy'),
               'target_depth_map':'MUST_NOT_BE_READ', 'gaze_head_x':None}
        path = root/'train.jsonl'
        path.write_text(json.dumps(row)+'\n')
        data = ReconstructionDataset(path, root)[0]
        assert data['condition'].shape == (6,256,256)
        assert torch.allclose(data['condition'][3].mean(), torch.tensor(32768/65535), atol=1e-5)
        assert torch.allclose(data['identity'].norm(),torch.tensor(1.0))
        for change in ({'image_id':'b'}, {'image_id':'c'}, {'split':'val'}, {'rescue_source':True}, {'source_depth_map':None}):
            path.write_text(json.dumps({**row, **change})+'\n')
            try:
                ReconstructionDataset(path,root)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Unsafe input accepted: {change}')
        path.write_text((json.dumps(row)+'\n')*2)
        try:
            ReconstructionDataset(path,root)
        except ValueError:
            pass
        else:
            raise AssertionError('Duplicate IDs accepted')
        (root/'train_ids.txt').write_text('a\nb\n')
        try:
            registry(root)
        except ValueError:
            pass
        else:
            raise AssertionError('Overlapping registry accepted')
    print('split isolation, rescue rejection, missing/duplicate rejection, uint16 depth, source-only loading: OK')


if __name__ == '__main__':
    test_dataset()
    test_model()
    print('PHASE3.1 RECONSTRUCTION PROTOCOL TESTS PASSED')
