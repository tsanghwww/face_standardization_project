"""Read-only Face Adapter gradient attribution; never constructs an optimizer.

Recompute each arm separately to bound VRAM. Ranking derivatives are assembled
exactly from the paired target/source distance gradients when the hinge is active.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from functools import partial
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from safetensors.torch import load_file

from phase3.train_geometry_supervision import freeze_identity_branch, model_hashes, verify_warm_start, save_json
from phase3.reconstruction_adapter import ReconstructionAdapter
from phase3.reconstruction_data import file_hash
from phase3.geometry_audit_data import GeometryAuditDataset
from phase3.sample_latent_img2img import load_adapter_exact
from phase3.differentiable_geometry import (
    estimate_geometry, geometry_loss, load_deca_frozen, load_source_geometry,
    load_target_geometry, one_step_x0,
)


def state_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def cosine(a, b):
    den = float(a.norm() * b.norm())
    return float(torch.dot(a, b) / den) if den > 1e-20 else None


def joint_directional_audit(vectors, geometry_weight, ranking_weight, source_geometry_ratio=1.0):
    """Evaluate one local SGD direction without updating model parameters."""
    direction = -(
        vectors["src"]
        + geometry_weight * (
            vectors["geometry"] + source_geometry_ratio * vectors["source_geometry"]
        )
        + ranking_weight * vectors["ranking"]
    )
    direction = direction / max(float(direction.norm()), 1e-20)
    derivatives = {
        "target_distance_dd": float(torch.dot(vectors["d_target"], direction)),
        "target_source_gap_dd": float(torch.dot(vectors["d_target"] - vectors["d_source"], direction)),
        "source_mse_dd": float(torch.dot(vectors["src"], direction)),
        "source_self_distance_dd": float(torch.dot(vectors["d_source_self"], direction)),
    }
    derivatives["all_four_improve"] = all(value < 0 for key, value in derivatives.items() if key.endswith("_dd"))
    return {
        "geometry_weight": geometry_weight,
        "ranking_weight": ranking_weight,
        "source_geometry_ratio": source_geometry_ratio,
        **derivatives,
    }


def attribution(vectors, weights, masks):
    weighted = {k: vectors[k] * weights[k] for k in weights}
    total = sum(weighted.values())
    result = {"raw_norms": {k: float(v.norm()) for k, v in vectors.items()},
              "weighted_norms": {k: float(v.norm()) for k, v in weighted.items()},
              "cosines": {f"{a}__{b}": cosine(weighted[a], weighted[b])
                          for a, b in (("src", "geometry"), ("src", "ranking"), ("geometry", "ranking"))},
              "total_norm": float(total.norm()), "clip_factor_at_1": min(1., 1. / max(float(total.norm()), 1e-20)),
              "per_scale": {str(k): {n: float(v[m].norm()) for n, v in weighted.items()} for k, m in masks.items()}}
    gt, gs = vectors["d_target"], vectors["d_source"]
    result["distance_gradient_cosine"] = cosine(gt, gs)
    result["ranking_cancellation_ratio"] = float(vectors["ranking"].norm()) / max(float(gt.norm() + gs.norm()), 1e-20)
    # First-order directional derivatives for a hypothetical -total gradient;
    # these are NOT actual optimizer updates or predicted AdamW outcomes.
    result["reweighting"] = []
    for factor in (0., 1., 10., 100., 1000.):
        direction = -(
            weighted["src"]
            + weighted["geometry"]
            + weighted.get("source_geometry", torch.zeros_like(weighted["src"]))
            + factor * weighted["ranking"]
        )
        direction = direction / max(float(direction.norm()), 1e-20)
        result["reweighting"].append({"ranking_multiplier": factor,
            "target_distance_directional_derivative": float(torch.dot(gt, direction)),
            "source_distance_directional_derivative": float(torch.dot(gs, direction)),
            "gap_directional_derivative": float(torch.dot(gt - gs, direction)),
            "src_mse_directional_derivative": float(torch.dot(vectors["src"], direction))})
    result["balanced_reweighting"] = []
    for wg in (.001, .003, .01):
        for wr in (.1, .3, 1.):
            if "source_geometry" in vectors and "d_source_self" in vectors:
                for source_ratio in (.1, .3, 1.):
                    result["balanced_reweighting"].append(
                        joint_directional_audit(vectors, wg, wr, source_ratio)
                    )
            else:
                direction = -(vectors['src'] + wg*vectors['geometry'] + wr*vectors['ranking'])
                direction /= max(float(direction.norm()), 1e-20)
                result['balanced_reweighting'].append({'geometry_weight': wg, 'ranking_weight': wr,
                    'target_distance_dd': float(torch.dot(gt, direction)),
                    'target_source_gap_dd': float(torch.dot(gt-gs, direction)),
                    'source_mse_dd': float(torch.dot(vectors['src'], direction))})
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--only-anchor', action='store_true')
    p.add_argument('--all-timesteps', action='store_true')
    p.add_argument('--timesteps', nargs='+', type=int, default=(100, 250, 400))
    args = p.parse_args()
    if args.out_dir.exists():
        raise ValueError('Audit output must be new')
    if not args.timesteps or any(t < 1 or t > 999 for t in args.timesteps):
        raise ValueError('Audit timesteps must be in [1, 999]')
    cfg = json.loads((args.run_dir / 'config.json').read_text())
    ckpt = args.checkpoint or args.run_dir / 'checkpoint_step_0064.pt'
    dataset = GeometryAuditDataset(Path(cfg['manifest']), Path(cfg['split_dir']), Path(cfg['ids_file']), 'train')
    items = [dataset[i] for i in range(len(dataset))]
    saved = torch.load(ckpt, map_location='cpu', weights_only=True)
    verify_warm_start(saved, model_hashes(Path(cfg['backbone_path']), Path(cfg['vae_path']), Path(cfg['empty_prompt'])), Path(__file__).parent)
    for name, digest in cfg['code_hashes'].items():
        if file_hash(Path(__file__).with_name(name)) != digest:
            raise ValueError('Original training code changed: ' + name)
    for name, digest in saved['fingerprint']['split_hashes'].items():
        if file_hash(Path(cfg['split_dir']) / name) != digest:
            raise ValueError('Split changed')
    if args.checkpoint is None:
        if saved['fingerprint']['manifest'] != file_hash(Path(cfg['manifest'])) or saved['fingerprint']['inputs'] != dataset.input_hashes:
            raise ValueError('Training input contents changed')
    torch.manual_seed(cfg['seed'])
    device, dtype = 'cuda', torch.float16
    vae = AutoencoderKL.from_pretrained(cfg['vae_path'], local_files_only=True, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    unet = UNet2DConditionModel.from_pretrained(Path(cfg['backbone_path'])/'unet', variant='fp16', local_files_only=True, torch_dtype=dtype).to(device).eval()
    model = ReconstructionAdapter(unet).to(device).eval()
    load_adapter_exact(model, saved['adapter'])
    freeze_identity_branch(model)
    if model.frozen_hash() != saved['frozen_hash']:
        raise ValueError('Frozen UNet mismatch')
    unet.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
    deca = load_deca_frozen(Path(cfg['deca_root']), device)
    empty = load_file(cfg['empty_prompt'])['empty_prompt_embedding'].to(device, dtype=dtype)
    scheduler = DDPMScheduler.from_pretrained(Path(cfg['backbone_path'])/'scheduler', local_files_only=True)
    if scheduler.config.prediction_type != 'epsilon':
        raise ValueError('epsilon scheduler required')
    named = [(n, v) for n, v in model.named_parameters() if v.requires_grad]
    if any(not n.startswith('face.') for n, _ in named):
        raise ValueError('Only face gradients allowed')
    params = [v for _, v in named]
    masks = {i: torch.cat([torch.full((v.numel(),), n.startswith((f'face.blocks.{i}.', f'face.outputs.{i}.')), dtype=torch.bool) for n, v in named]) for i in range(4)}
    def gradient(loss, retain=False):
        values = torch.autograd.grad(loss * 128., params, retain_graph=retain, allow_unused=False)
        vector = torch.cat([g.detach().float().cpu().flatten() / 128. for g in values])
        if not torch.isfinite(vector).all():
            raise ValueError('Nonfinite gradient')
        return vector
    before = {n: state_hash(m) for n, m in [('adapter_and_unet', model), ('vae', vae), ('deca', deca)]}
    checkpoint_before = file_hash(ckpt)
    args.out_dir.mkdir(parents=True)
    save_json(args.out_dir/'config.json', {'run_config': cfg, 'checkpoint': str(ckpt), 'checkpoint_sha256': checkpoint_before,
        'audit_code_sha256': file_hash(Path(__file__)), 'inputs': dataset.input_hashes,
        'protocol': ('all selected train IDs at explicit timesteps' if args.all_timesteps else
                     '32 train IDs at t250; every fourth ID at explicit auxiliary timesteps') +
                    '; fixed same noise per ID across arms/timesteps; no optimizer',
        'gradient_scale': 128, 'only_anchor': args.only_anchor})
    (args.out_dir/'exact_command.txt').write_text(' '.join(sys.argv))
    rows = []
    sf = float(vae.config.scaling_factor)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for i, item in enumerate(items):
        if args.only_anchor and i % 4:
            continue
        target = load_target_geometry(deca, Path(item['deca_mat']), Path(item['phase2_npz']), device)
        source_target = load_source_geometry(deca, Path(item['deca_mat']), device)
        with torch.no_grad():
            latent = (vae.encode(item['image'][None].to(device)).latent_dist.mode() * sf).to(dtype)
        noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(cfg['seed'] + 300000 + i)).to(device, dtype=dtype)
        ts = list(args.timesteps) if args.all_timesteps else ([250] if args.only_anchor or i % 4 else list(args.timesteps))
        for timestep in ts:
            t = torch.tensor([timestep], device=device)
            noisy = scheduler.add_noise(latent, noise, t)
            def predict(prefix):
                with torch.autocast('cuda', dtype=dtype):
                    return model(noisy, t, item[prefix+'_condition'][None].to(device), item['identity'][None].to(device), empty)
            loss = F.mse_loss(predict('source').float(), noise.float())
            values = {'src': float(loss.detach())}
            vectors = {'src': gradient(loss)}
            del loss
            for arm in ('target', 'source'):
                eps = predict(arm)
                decoded = vae.decode(one_step_x0(scheduler, noisy, t, eps).float()/sf).sample
                saturation = float((decoded.detach().abs() >= 1).float().mean())
                pose, exp, lm = estimate_geometry(deca, decoded)
                geo = geometry_loss(pose, exp, lm, target['pose'][None], target['expression'][None], target['landmarks'][None], tuple(cfg['geometry_weights']))
                terms = list(geo['terms'].values())
                keep_for_source_self = arm == 'source'
                grads = [gradient(term, retain=keep_for_source_self or j < 2) for j, term in enumerate(terms)]
                vectors['d_'+arm] = grads[0]/45. + grads[1]/.5 + grads[2]/.2
                values['d_'+arm] = float(terms[0].detach())/45. + float(terms[1].detach())/.5 + float(terms[2].detach())/.2
                values[arm+'_terms'] = {k: float(v.detach()) for k, v in geo['terms'].items()}
                values[arm+'_rgb_saturation'] = saturation
                if arm == 'target':
                    vectors['geometry'] = sum(g*w for g, w in zip(grads, cfg['geometry_weights']))
                    for name, g in zip(('pose', 'expression', 'landmark'), grads):
                        vectors[name] = g
                else:
                    source_geo = geometry_loss(
                        pose, exp, lm,
                        source_target['pose'][None], source_target['expression'][None], source_target['landmarks'][None],
                        tuple(cfg['geometry_weights']),
                    )
                    source_terms = list(source_geo['terms'].values())
                    source_grads = [gradient(term, retain=j < 2) for j, term in enumerate(source_terms)]
                    vectors['d_source_self'] = source_grads[0]/45. + source_grads[1]/.5 + source_grads[2]/.2
                    vectors['source_geometry'] = sum(g*w for g, w in zip(source_grads, cfg['geometry_weights']))
                    values['source_self_distance'] = (
                        float(source_terms[0].detach())/45.
                        + float(source_terms[1].detach())/.5
                        + float(source_terms[2].detach())/.2
                    )
                    values['source_self_terms'] = {k: float(v.detach()) for k, v in source_geo['terms'].items()}
                del eps, decoded, pose, exp, lm, geo, terms, grads
            hinge = cfg['ranking_margin'] + values['d_target'] - values['d_source']
            vectors['ranking'] = (vectors['d_target'] - vectors['d_source']) * float(hinge > 0)
            row = {'image_id': item['image_id'], 'timestep': timestep, 'losses': values, 'hinge_active': hinge > 0,
                   'ranking_loss': max(0., hinge), **attribution(vectors, {
                       'src': 1.,
                       'geometry': cfg['geometry_loss_weight'],
                       'source_geometry': cfg['geometry_loss_weight'],
                       'ranking': cfg['ranking_loss_weight'],
                   }, masks)}
            rows.append(row)
            with (args.out_dir/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps({'completed': len(rows), 'id': item['image_id'], 't': timestep, 'weighted_norms': row['weighted_norms']}), flush=True)
            del vectors
            if torch.cuda.max_memory_allocated()/1024**3 >= 7.2:
                raise RuntimeError('Audit exceeded 7.2 GiB budget')
    after = {n: state_hash(m) for n, m in [('adapter_and_unet', model), ('vae', vae), ('deca', deca)]}
    unchanged = before == after and file_hash(ckpt) == checkpoint_before
    if not unchanged or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError('State changed or parameter .grad populated')
    save_json(args.out_dir/'summary.json', {'status': 'completed', 'n_rows': len(rows), 'optimizer_steps': 0,
        'state_hashes_before': before, 'state_hashes_after': after, 'checkpoint_unchanged': unchanged,
        'wall_seconds': time.perf_counter()-started, 'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated()/1024**2,
        'scope': 'train-only one-step local gradient geometry, not validation or observed reweighted training'})


if __name__ == '__main__':
    main()
