"""Paired source-latent DDIM diagnostic on the existing train-only smoke IDs."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image, ImageDraw
from safetensors.torch import load_file
import torch

from phase3.reconstruction_adapter import ReconstructionAdapter
from phase3.reconstruction_data import ReconstructionDataset, file_hash


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rgb(value):
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite decoded image')
    array = ((value.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(array.permute(1, 2, 0).numpy())


def img2img_timesteps(scheduler, steps, strength, device='cpu'):
    """Diffusers strength slicing; zero explicitly means VAE-only round trip."""
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('strength must be finite and in [0, 1]')
    if not 1 <= steps <= scheduler.config.num_train_timesteps:
        raise ValueError('Invalid inference step budget')
    if scheduler.order != 1:
        raise ValueError('This sampler requires a first-order DDIM scheduler')
    count = int(steps * strength)
    if strength > 0 and count == 0:
        raise ValueError('Positive strength rounds to zero steps; use strength=0 explicitly')
    scheduler.set_timesteps(steps, device=device)
    return scheduler.timesteps[steps - count:]


@torch.no_grad()
def sample_latent(source, noise, scheduler, steps, strength, predict):
    timesteps = img2img_timesteps(scheduler, steps, strength, source.device)
    if not len(timesteps):
        return source.clone(), None
    sample = scheduler.add_noise(source, noise, timesteps[:1])
    initial_hash = tensor_hash(sample)
    for timestep in timesteps:
        output = predict(scheduler.scale_model_input(sample, timestep), timestep)
        sample = scheduler.step(output, timestep, sample, eta=0).prev_sample
        if not torch.isfinite(sample).all():
            raise ValueError('Nonfinite DDIM latent')
    return sample, initial_hash


def load_adapter_exact(model, state):
    expected = model.adapter_state()
    if set(state) != set(expected):
        raise ValueError('Checkpoint adapter keys mismatch')
    for name, value in state.items():
        if value.shape != expected[name].shape or not torch.isfinite(value).all():
            raise ValueError(f'Invalid adapter tensor: {name}')
    model.load_state_dict(state, strict=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'split-dir', 'backbone-path', 'vae-path', 'empty-prompt', 'out-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Optional trained adapter comparison')
    parser.add_argument('--strengths', nargs='+', type=float, default=[0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--sampling-steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20260902)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    started = time.perf_counter()
    if len(set(args.strengths)) != len(args.strengths):
        raise ValueError('Duplicate strengths')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError('Output directory must be empty')
    scheduler = DDIMScheduler.from_pretrained(args.backbone_path / 'scheduler', local_files_only=True)
    if scheduler.config.prediction_type != 'epsilon':
        raise ValueError('Requires an epsilon scheduler')
    schedules = []
    for index, strength in enumerate(args.strengths):
        ts = img2img_timesteps(scheduler, args.sampling_steps, strength)
        schedules.append({'key': f's{index:02d}', 'strength': strength,
                          'denoising_steps': len(ts), 'timesteps': ts.tolist(),
                          'start_timestep': int(ts[0]) if len(ts) else None})
    dataset = ReconstructionDataset(args.manifest, args.split_dir)
    items = [dataset[i] for i in range(len(dataset))]
    files = [args.backbone_path / 'unet/diffusion_pytorch_model.fp16.safetensors',
             args.backbone_path / 'unet/config.json', args.backbone_path / 'scheduler/scheduler_config.json',
             args.vae_path / 'diffusion_pytorch_model.safetensors', args.vae_path / 'config.json', args.empty_prompt]
    fingerprint = {'manifest': file_hash(args.manifest), 'inputs': dataset.hashes,
                   'split_hashes': {n: file_hash(args.split_dir / n) for n in
                                    ('train_ids.txt', 'validation_ids.txt', 'fixed_test_ids.txt')},
                   'model_files': {str(p): file_hash(p) for p in files}}
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True) if args.checkpoint else None
    if saved:
        for key, value in fingerprint.items():
            if saved['fingerprint'][key] != value:
                raise ValueError(f'Checkpoint fingerprint mismatch: {key}')
        for name in ('reconstruction_adapter.py', 'reconstruction_data.py'):
            if saved['fingerprint']['code_hashes'][name] != file_hash(Path(__file__).with_name(name)):
                raise ValueError(f'Checkpoint implementation mismatch: {name}')
    device = torch.device(args.device)
    use_cuda = device.type == 'cuda'
    dtype = torch.float16 if use_cuda else torch.float32
    amp = lambda: torch.autocast('cuda', dtype=torch.float16) if use_cuda else nullcontext()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / 'references').mkdir()
    variants = ['frozen', 'trained'] if saved else ['frozen']
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'fingerprint': fingerprint, 'schedules': schedules, 'variants': variants,
                   'image_ids': [i['image_id'] for i in items], 'torch': torch.__version__,
                   'diffusers': diffusers.__version__, 'cuda': torch.version.cuda,
                   'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'code_hashes': {n: file_hash(Path(__file__).with_name(n)) for n in
                                   ('sample_latent_img2img.py', 'reconstruction_adapter.py', 'reconstruction_data.py')},
                   'checkpoint_sha256': file_hash(args.checkpoint) if saved else None,
                   'checkpoint_step': saved['step'] if saved else None,
                   'latent_policy': 'FP32 VAE posterior mode * config scaling_factor; cast to UNet dtype for positive strength',
                   'noise_policy': 'CPU FP32 seed + 200000 + manifest index; reused across strengths and variants',
                   'prompt': 'cached empty prompt', 'cfg_enabled': False, 'eta': 0,
                   'strength_zero': 'FP32 VAE-only anchor; no UNet calls',
                   'strength_one': 'noised source at highest inference timestep; NOT pure-noise initialization',
                   'scope': 'train-only source reconstruction diagnostic; no optimization or target/gaze intervention'})
    save_json(args.out_dir / 'config.json', config)
    (args.out_dir / 'exact_command.txt').write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding='utf-8')
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    sf = float(vae.config.scaling_factor)
    latents = []
    with torch.no_grad():
        for item in items:
            latent = vae.encode(item['image'][None].to(device)).latent_dist.mode() * sf
            if not torch.isfinite(latent).all():
                raise ValueError(f'Nonfinite source latent: {item["image_id"]}')
            latents.append(latent.cpu())
            rgb(item['image']).save(args.out_dir / 'references' / f'{item["image_id"]}_source.png')
            rgb(vae.decode(latent / sf).sample[0]).save(args.out_dir / 'references' / f'{item["image_id"]}_vae.png')
    vae.to('cpu')
    if use_cuda:
        torch.cuda.empty_cache()
    unet = UNet2DConditionModel.from_pretrained(args.backbone_path / 'unet', variant='fp16',
                                               torch_dtype=dtype, local_files_only=True).to(device).eval().requires_grad_(False)
    empty = load_file(str(args.empty_prompt))['empty_prompt_embedding'].to(device, dtype=dtype)
    if tuple(empty.shape) != (1, 77, 768) or not torch.isfinite(empty).all():
        raise ValueError('Invalid cached empty prompt')
    # The frozen arm calls the unmodified UNet directly. Install adapters only afterward.
    model = None
    pending, records = [], []
    for variant in variants:
        if variant == 'trained':
            model = ReconstructionAdapter(unet).to(device).eval()
            if model.frozen_hash() != saved['frozen_hash']:
                raise ValueError('Checkpoint frozen UNet hash mismatch')
            load_adapter_exact(model, saved['adapter'])
            model.requires_grad_(False)
        for spec in schedules:
            directory = args.out_dir / variant / spec['key']
            directory.mkdir(parents=True)
            for index, item in enumerate(items):
                seed = args.seed + 200000 + index
                noise = torch.randn(latents[index].shape, generator=torch.Generator().manual_seed(seed))
                path = directory / f'{item["image_id"]}_img2img.png'
                row = {'image_id': item['image_id'], 'variant': variant, **spec,
                       'noise_seed': seed, 'noise_sha256': tensor_hash(noise),
                       'output': str(path.relative_to(args.out_dir)), 'status': 'pending', 'failure_reason': ''}
                try:
                    if spec['strength'] == 0:
                        with Image.open(args.out_dir / 'references' / f'{item["image_id"]}_vae.png') as anchor:
                            anchor.save(path)
                        row.update(status='generated', initial_latent_sha256=None, sha256=file_hash(path))
                    else:
                        def predict(value, timestep):
                            with amp():
                                if variant == 'frozen':
                                    return unet(value, timestep, encoder_hidden_states=empty).sample
                                return model(value, timestep, item['condition'][None].to(device),
                                             item['identity'][None].to(device), empty)
                        sample, initial_hash = sample_latent(latents[index].to(device, dtype=dtype),
                                                             noise.to(device, dtype=dtype), scheduler,
                                                             args.sampling_steps, spec['strength'], predict)
                        row['initial_latent_sha256'] = initial_hash
                        pending.append((sample.float().cpu(), path, row))
                except Exception as error:
                    row.update(status='generation_failed', failure_reason=f'{type(error).__name__}: {error}')
                records.append(row)
            print(json.dumps({'variant': variant, 'strength': spec['strength'], 'sampled': len(items),
                              'elapsed_seconds': time.perf_counter() - started}), flush=True)
    if model is not None:
        if model.frozen_hash() != saved['frozen_hash']:
            raise RuntimeError('Frozen UNet changed during sampling')
        model.to('cpu')
    else:
        unet.to('cpu')
    if use_cuda:
        torch.cuda.empty_cache()
    vae.to(device)
    with torch.no_grad():
        for latent, path, row in pending:
            try:
                rgb(vae.decode(latent.to(device) / sf).sample[0]).save(path)
                row.update(status='generated', sha256=file_hash(path))
            except Exception as error:
                row.update(status='generation_failed', failure_reason=f'{type(error).__name__}: {error}')
    with (args.out_dir / 'samples.jsonl').open('w', encoding='utf-8') as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + '\n')
    # Four deterministic IDs, all arms/strengths; full-resolution outputs remain separate.
    for spec in schedules:
        count = min(4, len(items))
        sheet = Image.new('RGB', (256 * (2 + len(variants)), 280 * count), 'white')
        for index, item in enumerate(items[:count]):
            columns = [('source', args.out_dir / 'references' / f'{item["image_id"]}_source.png'),
                       ('VAE anchor', args.out_dir / 'references' / f'{item["image_id"]}_vae.png')]
            columns += [(v, args.out_dir / v / spec['key'] / f'{item["image_id"]}_img2img.png') for v in variants]
            for col, (label, path) in enumerate(columns):
                if path.exists():
                    with Image.open(path) as image:
                        sheet.paste(image, (256 * col, 280 * index + 24))
                ImageDraw.Draw(sheet).text((256 * col + 3, 280 * index + 3),
                                           f'{item["image_id"]} {label} s={spec["strength"]}', fill='black')
        sheet.save(args.out_dir / f'contact_{spec["key"]}.png')
    failed = sum(r['status'] != 'generated' for r in records)
    summary = {'status': 'completed' if not failed else 'completed_with_failures', 'n_ids': len(items),
               'expected_outputs': len(items) * len(variants) * len(schedules),
               'generated': len(records) - failed, 'generation_failed': failed,
               'wall_seconds': time.perf_counter() - started,
               'gpu_peak_allocated_mb': torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None,
               'samples_sha256': file_hash(args.out_dir / 'samples.jsonl'), 'scope': config['scope']}
    save_json(args.out_dir / 'summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
