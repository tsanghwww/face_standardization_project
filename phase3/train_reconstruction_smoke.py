"""Bounded train-only diffusion reconstruction diagnostic, not an intervention trainer."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, UNet2DConditionModel
from safetensors.torch import load_file

from phase3.reconstruction_adapter import ReconstructionAdapter
from phase3.reconstruction_data import ReconstructionDataset, file_hash


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def rgb(tensor):
    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(array.permute(1, 2, 0).numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--backbone-path', type=Path, required=True)
    parser.add_argument('--vae-path', type=Path, required=True)
    parser.add_argument('--empty-prompt', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=64, help='Total optimizer steps, including resumed steps')
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20260902)
    parser.add_argument('--sample-count', type=int, default=4)
    parser.add_argument('--sampling-steps', type=int, default=20)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if args.steps < 1 or args.accumulation < 1 or args.sample_count < 0:
        raise ValueError('Invalid training budget')
    if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
        raise ValueError('Output directory not empty; use a new run directory or explicit --resume')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    use_cuda = device.type == 'cuda'
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    dtype = torch.float16 if use_cuda else torch.float32
    amp = lambda: torch.autocast('cuda', dtype=torch.float16) if use_cuda else nullcontext()
    dataset = ReconstructionDataset(args.manifest, args.split_dir)
    items = [dataset[i] for i in range(len(dataset))]
    fingerprint = {
        'manifest': file_hash(args.manifest), 'inputs': dataset.hashes,
        'split_hashes': {n: file_hash(args.split_dir / n) for n in ('train_ids.txt', 'validation_ids.txt', 'fixed_test_ids.txt')},
        'model_files': {str(p): file_hash(p) for p in (
            args.backbone_path / 'unet/diffusion_pytorch_model.fp16.safetensors',
            args.backbone_path / 'unet/config.json',
            args.backbone_path / 'scheduler/scheduler_config.json',
            args.vae_path / 'diffusion_pytorch_model.safetensors', args.vae_path / 'config.json', args.empty_prompt)},
        'lr': args.lr, 'accumulation': args.accumulation, 'seed': args.seed,
        'mode': 'source_reconstruction_only', 'gaze_loss': False, 'size': 256,
        'code_hashes': {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name('reconstruction_adapter.py'),
                                                    Path(__file__).with_name('reconstruction_data.py'))},
    }
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'fingerprint': fingerprint, 'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'crop_policy': 'full_source_resize_256; crop alignment requires audit',
                   'supervision': 'source RGB noise-prediction only; no target-pose or gaze loss'})

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True, torch_dtype=torch.float32).to(device).eval()
    vae.requires_grad_(False)
    latents, anchors = [], []
    sf = float(vae.config.scaling_factor)
    with torch.no_grad():
        for item in items:
            latent = vae.encode(item['image'][None].to(device)).latent_dist.mode() * sf
            if not torch.isfinite(latent).all():
                raise RuntimeError(f'Nonfinite VAE latent: {item["image_id"]}')
            latents.append(latent.cpu())
            anchors.append(vae.decode(latent / sf).sample.cpu()[0])
    del vae
    if use_cuda:
        torch.cuda.empty_cache()
    unet = UNet2DConditionModel.from_pretrained(args.backbone_path / 'unet', variant='fp16',
                                               torch_dtype=dtype, local_files_only=True)
    model = ReconstructionAdapter(unet).to(device)
    unet.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
    frozen_before = model.frozen_hash()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda, init_scale=128)
    scheduler = DDPMScheduler.from_pretrained(args.backbone_path / 'scheduler', local_files_only=True)
    if scheduler.config.prediction_type != 'epsilon':
        raise ValueError('This reconstruction smoke requires an epsilon-prediction scheduler')
    empty = load_file(str(args.empty_prompt))['empty_prompt_embedding'].to(device, dtype=dtype)
    if tuple(empty.shape) != (1, 77, 768) or not torch.isfinite(empty).all():
        raise ValueError('Invalid cached empty-prompt embedding')
    begin = 0
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True)
        if saved['fingerprint'] != fingerprint or saved['frozen_hash'] != frozen_before:
            raise ValueError('Resume data/model/config fingerprint mismatch')
        model.load_state_dict(saved['adapter'], strict=False)
        optimizer.load_state_dict(saved['optimizer'])
        scaler.load_state_dict(saved['scaler'])
        begin = saved['step']
    if begin >= args.steps:
        raise ValueError('Resume checkpoint already reached requested total steps')
    save_json(args.out_dir / f'config_from_step_{begin}.json', config)
    (args.out_dir / f'exact_command_from_step_{begin}.txt').write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding='utf-8')

    def predict(index, noisy, timestep, face=True, identity=True, shuffled=False):
        item = items[(index + 1) % len(items)] if shuffled else items[index]
        with amp():
            return model(noisy, timestep, item['condition'][None].to(device), item['identity'][None].to(device),
                         empty, face_enabled=face, identity_enabled=identity)

    def diagnostic(face=True, identity=True, shuffled=False):
        rows, estimates = [], []
        with torch.no_grad():
            for index, latent in enumerate(latents):
                generator = torch.Generator().manual_seed(args.seed + 100000 + index)
                noise = torch.randn(latent.shape, generator=generator).to(device, dtype=dtype)
                timestep = torch.tensor([250], device=device)
                noisy = scheduler.add_noise(latent.to(device, dtype=dtype), noise, timestep)
                output = predict(index, noisy, timestep, face, identity, shuffled)
                if not torch.isfinite(output).all():
                    raise RuntimeError('Nonfinite diagnostic output')
                loss = F.mse_loss(output.float(), noise.float()).item()
                rows.append({'image_id': items[index]['image_id'], 'epsilon_mse': loss})
                a = scheduler.alphas_cumprod[250].to(device)
                estimates.append(((noisy.float() - (1-a).sqrt() * output.float()) / a.sqrt()).cpu())
        return {'mean_epsilon_mse': sum(r['epsilon_mse'] for r in rows)/len(rows), 'rows': rows}, estimates

    baseline, before_latents = diagnostic(face=False, identity=False)
    save_json(args.out_dir / 'baseline_diagnostic.json', baseline)
    initial_state = {n: v.clone() for n, v in model.adapter_state().items()}
    log_path = args.out_dir / 'training_log.jsonl'
    with log_path.open('a' if args.resume else 'w', encoding='utf-8', buffering=1) as log:
        for step in range(begin, args.steps):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(args.accumulation):
                serial = step * args.accumulation + micro
                order = torch.randperm(len(items), generator=torch.Generator().manual_seed(args.seed + serial // len(items)))
                index = int(order[serial % len(items)])
                latent = latents[index].to(device, dtype=dtype)
                generator = torch.Generator().manual_seed(args.seed + serial + 1)
                noise = torch.randn(latent.shape, generator=generator).to(device, dtype=dtype)
                timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=generator).to(device)
                noisy = scheduler.add_noise(latent, noise, timestep)
                prediction = predict(index, noisy, timestep)
                loss = F.mse_loss(prediction.float(), noise.float())
                if not torch.isfinite(loss):
                    raise RuntimeError(f'Nonfinite training loss at step {step}')
                scaler.scale(loss / args.accumulation).backward()
                losses.append(loss.item())
            scaler.unscale_(optimizer)
            norms = {}
            for group, prefix in [('face', 'face.'), ('identity_projection', 'identity.'), ('identity_attention', 'unet.')]:
                grads = [p.grad.detach().float().norm().square() for n, p in model.named_parameters()
                         if n.startswith(prefix) and p.requires_grad and p.grad is not None]
                norms[group] = float(torch.stack(grads).sum().sqrt()) if grads else 0.0
            total_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            row = {'step': step + 1, 'loss': sum(losses)/len(losses), 'gradient_norms': norms,
                   'total_gradient_norm': float(total_norm), 'amp_scale': scaler.get_scale(),
                   'elapsed_seconds': time.perf_counter() - start}
            log.write(json.dumps(row) + '\n')
            if (step + 1) % 8 == 0 or step == begin:
                print(json.dumps(row), flush=True)
            if (step + 1) % 16 == 0 or step + 1 == args.steps:
                saved = {'step': step+1, 'adapter': model.adapter_state(), 'optimizer': optimizer.state_dict(),
                         'scaler': scaler.state_dict(), 'fingerprint': fingerprint, 'frozen_hash': frozen_before}
                temporary = args.out_dir / 'checkpoint.tmp'
                torch.save(saved, temporary)
                temporary.replace(args.out_dir / 'checkpoint.pt')

    after, after_latents = diagnostic()
    no_face, _ = diagnostic(face=False)
    no_identity, _ = diagnostic(identity=False)
    shuffled, _ = diagnostic(shuffled=True)
    frozen_after = model.frozen_hash()
    if frozen_after != frozen_before:
        raise RuntimeError('Frozen UNet weights changed')
    state = model.adapter_state()
    changes = {prefix: sum(float((v - initial_state[n]).square().sum()) for n, v in state.items() if n.startswith(prefix)) ** 0.5
               for prefix in ('face.', 'identity.', 'unet.')}
    if any(v <= 0 for v in changes.values()):
        raise RuntimeError(f'An adapter group did not update: {changes}')

    generated = []
    ddim = DDIMScheduler.from_config(scheduler.config)
    count = min(args.sample_count, len(items))
    with torch.no_grad():
        for index in range(count):
            ddim.set_timesteps(args.sampling_steps, device=device)
            sample = torch.randn((1,4,32,32), generator=torch.Generator().manual_seed(args.seed + 200000 + index)).to(device, dtype=dtype)
            for timestep in ddim.timesteps:
                output = predict(index, ddim.scale_model_input(sample, timestep), timestep)
                sample = ddim.step(output, timestep, sample, eta=0).prev_sample
            if not torch.isfinite(sample).all():
                raise RuntimeError('Nonfinite DDIM sample')
            generated.append(sample.float().cpu())
    model.to('cpu')
    del optimizer
    if use_cuda:
        torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    image_dir = args.out_dir / 'samples'
    image_dir.mkdir(exist_ok=True)
    sheet = Image.new('RGB', (256*5, (256+24)*count), 'white') if count else None
    with torch.no_grad():
        for index in range(count):
            decoded = [vae.decode(latent.to(device)/sf).sample[0].cpu() for latent in
                       (before_latents[index], after_latents[index], generated[index])]
            images = [rgb(items[index]['image']), rgb(anchors[index]), *(rgb(x) for x in decoded)]
            for column, (label, image) in enumerate(zip(('source', 'VAE', 'base x0 t250', 'trained x0 t250', 'DDIM from noise'), images)):
                sheet.paste(image, (column*256, index*280+24))
                ImageDraw.Draw(sheet).text((column*256+4, index*280+4), f'{items[index]["image_id"]} {label}', fill='black')
            images[-1].save(image_dir / f'{items[index]["image_id"]}_ddim.png')
    if sheet:
        sheet.save(args.out_dir / 'contact_sheet.png')
    summary = {
        'status': 'engineering_run_completed', 'optimization_split': 'train', 'n_train': len(items),
        'optimizer_steps': args.steps, 'accumulation': args.accumulation,
        'trainable_parameters': sum(p.numel() for p in parameters),
        'baseline': baseline, 'trained': after, 'no_face': no_face, 'no_identity': no_identity, 'shuffled_conditions': shuffled,
        'adapter_update_l2': changes, 'frozen_unet_hash_before': frozen_before, 'frozen_unet_hash_after': frozen_after,
        'checkpoint_sha256': file_hash(args.out_dir / 'checkpoint.pt'),
        'ddim_samples': count, 'full_32_generation_complete': count == len(items),
        'gpu_peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None,
        'wall_seconds': time.perf_counter()-start,
        'gaze_enabled': False, 'identity_outcome_loss_enabled': False,
        'scope': 'training-set denoising diagnostic, not held-out quality, standardization, or gaze disentanglement evidence',
    }
    save_json(args.out_dir / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in ('baseline', 'trained', 'no_face', 'no_identity', 'shuffled_conditions')}, indent=2))


if __name__ == '__main__':
    main()
