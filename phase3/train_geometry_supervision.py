"""Phase3.1d bounded train-only geometry supervision for the Face Adapter.

The optimizer path is:
    Face Adapter -> frozen UNet -> one-step x0 -> frozen VAE decode
    -> fixed whole-image warp -> frozen DECA encoder -> geometry loss.

No FAN, rescue, or nondifferentiable detector is present. The identity branch is
frozen (source identity is always used for the geometry arms). Missing or
nonfinite DECA estimates raise; they are never zero-filled.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
from safetensors.torch import load_file

from phase3.reconstruction_adapter import ReconstructionAdapter
from phase3.reconstruction_data import file_hash
from phase3.geometry_audit_data import GeometryAuditDataset, intervention
from phase3.sample_latent_img2img import load_adapter_exact
from phase3.differentiable_geometry import (
    estimate_geometry, geodesic_angle_deg, expression_rmse, geometry_loss,
    landmark_nme, load_deca_frozen, load_target_geometry, margin_ranking_loss,
    normalized_geometry_distance, one_step_x0,
)

GB = 1024**3
GPU_BUDGET_GIB = 7.2
ARMS = ("source_geometry", "target_geometry", "zero_geometry", "shuffled_geometry")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def rgb(tensor: torch.Tensor) -> Image.Image:
    if not torch.isfinite(tensor).all():
        raise ValueError("Nonfinite decoded image")
    array = ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(array.permute(1, 2, 0).numpy())


def freeze_identity_branch(model: ReconstructionAdapter) -> None:
    """Freeze the identity projection and identity-attention processors."""
    model.identity.requires_grad_(False)
    for processor in model.unet.attn_processors.values():
        if isinstance(processor, IPAdapterAttnProcessor2_0):
            for parameter in processor.parameters():
                parameter.requires_grad_(False)


def model_hashes(backbone: Path, vae: Path, empty_prompt: Path) -> dict[str, str]:
    files = (
        backbone / "unet/diffusion_pytorch_model.fp16.safetensors",
        backbone / "unet/config.json",
        backbone / "scheduler/scheduler_config.json",
        vae / "diffusion_pytorch_model.safetensors",
        vae / "config.json",
        empty_prompt,
    )
    return {str(path): file_hash(path) for path in files}


def verify_warm_start(saved: dict, current_hashes: dict[str, str], root: Path) -> None:
    expected = list(saved["fingerprint"]["model_files"].values())
    actual = list(current_hashes.values())
    if sorted(expected) != sorted(actual):
        raise ValueError("Backbone/VAE/prompt contents differ from the warm-start checkpoint")
    for name in ("reconstruction_adapter.py", "reconstruction_data.py"):
        if saved["fingerprint"]["code_hashes"][name] != file_hash(root / name):
            raise ValueError(f"Checkpoint implementation mismatch: {name}")


class ResidualHooks:
    """Capture Face Adapter residuals and UNet down-block activations."""

    def __init__(self, model: ReconstructionAdapter):
        self.model = model
        self.residuals: list[torch.Tensor] = []
        self.activations: list[torch.Tensor] = []
        self._hooks = []
        for index, projection in enumerate(model.face.outputs):
            self._hooks.append(projection.register_forward_hook(self._make_residual_hook(index)))
        for index, block in enumerate(model.unet.down_blocks):
            self._hooks.append(block.register_forward_hook(self._make_activation_hook(index)))

    def _make_residual_hook(self, index):
        def hook(module, inp, out):
            if out.requires_grad:
                out.retain_grad()
            while len(self.residuals) <= index:
                self.residuals.append(None)
            self.residuals[index] = out
        return hook

    def _make_activation_hook(self, index):
        def hook(module, inp, out):
            while len(self.activations) <= index:
                self.activations.append(None)
            self.activations[index] = out[0] if isinstance(out, tuple) else out
        return hook

    def remove(self):
        for hook in self._hooks:
            hook.remove()


def residual_scales(residuals: list[torch.Tensor]) -> list[dict]:
    return [{"scale": i, "mean_abs": float(r.detach().float().abs().mean()), "rms": float(r.detach().float().square().mean().sqrt())}
            for i, r in enumerate(residuals) if r is not None]


def residual_activation_rms(residuals: list[torch.Tensor], activations: list[torch.Tensor]) -> list[dict]:
    rows = []
    for index, (residual, activation) in enumerate(zip(residuals, activations)):
        if residual is None or activation is None:
            rows.append({"scale": index, "ratio": None})
            continue
        r_rms = float(residual.detach().float().square().mean().sqrt())
        a_rms = float(activation.detach().float().square().mean().sqrt())
        rows.append({"scale": index, "residual_rms": r_rms, "activation_rms": a_rms,
                     "ratio": r_rms / a_rms if a_rms > 1e-12 else None})
    return rows


def run_preflight(model, vae, deca, empty, scheduler, device, dtype, amp, items, targets,
                  timestep: int = 300) -> dict:
    """Gradient preflight through the full differentiable geometry path."""
    torch.cuda.reset_peak_memory_stats()
    item = items[0]
    target = targets[0]
    with torch.no_grad():
        latent = vae.encode(item["image"][None].to(device)).latent_dist.mode() * float(vae.config.scaling_factor)
    generator = torch.Generator().manual_seed(20260909 + 1)
    noise = torch.randn(latent.shape, generator=generator).to(device, dtype=dtype)
    t = torch.tensor([timestep], device=device)
    noisy = scheduler.add_noise(latent.to(device, dtype=dtype), noise, t)
    hooks = ResidualHooks(model)
    try:
        with amp():
            eps = model(noisy, t, item["source_condition"][None].to(device), item["identity"][None].to(device), empty)
        x0 = one_step_x0(scheduler, noisy, t, eps)
        rgb_tensor = vae.decode(x0.float() / float(vae.config.scaling_factor)).sample
        pose, exp, lm = estimate_geometry(deca, rgb_tensor)
        loss = geometry_loss(pose, exp, lm, target["pose"][None], target["expression"][None], target["landmarks"][None])["total"]
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite preflight geometry loss")
        loss.backward()
    finally:
        hooks.remove()
    per_scale = []
    for index, residual in enumerate(hooks.residuals):
        if residual is None or residual.grad is None:
            per_scale.append({"scale": index, "grad_finite": False, "grad_norm": None, "nonzero": False})
            continue
        grad_norm = float(residual.grad.detach().float().norm())
        per_scale.append({"scale": index, "grad_finite": bool(torch.isfinite(residual.grad).all()),
                          "grad_norm": grad_norm, "nonzero": grad_norm > 1e-12})
    if not all(row["grad_finite"] and row["nonzero"] for row in per_scale):
        raise RuntimeError(f"Geometry gradient did not reach every Face Adapter scale: {per_scale}")
    peak_gib = torch.cuda.max_memory_allocated() / GB
    report = {
        "status": "passed",
        "loss": float(loss.detach().cpu()),
        "gradient_per_scale": per_scale,
        "gpu_peak_allocated_gib": peak_gib,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "budget_gib": GPU_BUDGET_GIB,
    }
    if peak_gib >= GPU_BUDGET_GIB:
        raise RuntimeError(f"Preflight GPU peak {peak_gib:.3f} GiB exceeds budget {GPU_BUDGET_GIB} GiB")
    return report


def diagnose_timestep_interval(model, vae, deca, empty, scheduler, device, dtype, amp, items, targets,
                               grid: list[int] | None = None) -> dict:
    """Train-only diagnostic that freezes the low/mid-noise geometry timestep interval and weights."""
    grid = grid or [100, 200, 300, 400, 500, 600, 700, 800]
    sf = float(vae.config.scaling_factor)
    results = {}
    with torch.no_grad():
        for t_value in grid:
            lms, poses, exps = [], [], []
            for index, item in enumerate(items):
                latent = vae.encode(item["image"][None].to(device)).latent_dist.mode() * sf
                generator = torch.Generator().manual_seed(20260909 + 700000 + index)
                noise = torch.randn(latent.shape, generator=generator).to(device, dtype=dtype)
                t = torch.tensor([t_value], device=device)
                noisy = scheduler.add_noise(latent.to(device, dtype=dtype), noise, t)
                with amp():
                    eps = model(noisy, t, item["source_condition"][None].to(device), item["identity"][None].to(device), empty)
                x0 = one_step_x0(scheduler, noisy, t, eps)
                rgb_tensor = vae.decode(x0.float() / sf).sample
                pose, exp, lm = estimate_geometry(deca, rgb_tensor)
                target = targets[index]
                lms.append(float(landmark_nme(lm, target["landmarks"][None]).item()))
                poses.append(float(geodesic_angle_deg(pose[:, :3], target["pose"][None][:, :3]).item()))
                exps.append(float(expression_rmse(exp, target["expression"][None]).item()))
            results[str(t_value)] = {
                "landmark_nme_median": float(np.median(lms)), "landmark_nme_mean": float(np.mean(lms)),
                "pose_deg_mean": float(np.mean(poses)), "expression_rmse_mean": float(np.mean(exps)),
                "all_finite": bool(np.isfinite(lms).all() and np.isfinite(poses).all() and np.isfinite(exps).all()),
            }
    finite_ts = [t for t in grid if results[str(t)]["all_finite"]]
    if not finite_ts:
        raise RuntimeError("No timestep produced finite differentiable DECA estimates")
    min_nme = min(results[str(t)]["landmark_nme_median"] for t in finite_ts)
    threshold = min(0.30, 1.6 * min_nme)
    valid = [t for t in finite_ts if results[str(t)]["landmark_nme_median"] <= threshold]
    if not valid:
        raise RuntimeError("No timestep passed the landmark-NME meaningfulness threshold")
    t_low = min(valid)
    t_high = max(valid)
    low = results[str(t_low)]
    pose_scale = max(low["pose_deg_mean"], 1e-3)
    exp_scale = max(low["expression_rmse_mean"], 1e-3)
    lm_scale = max(low["landmark_nme_mean"], 1e-3)
    weights = (1.0, pose_scale / exp_scale, pose_scale / lm_scale)
    return {
        "timestep_low": t_low, "timestep_high": t_high,
        "geometry_weights": [round(float(w), 6) for w in weights],
        "geometry_loss_weight": 1.0, "ranking_loss_weight": 0.1, "ranking_margin": 0.05,
        "threshold": threshold, "min_nme": min_nme,
        "per_timestep": results,
        "normalization_note": "pose/expression/landmark weights normalized to the t_low source-geometry diagnostic",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "split-dir", "ids-file", "backbone-path", "vae-path", "empty-prompt", "checkpoint", "deca-root", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--variant", choices=("geometry_loss_only", "geometry_loss_plus_ranking"), required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--geometry-timestep-low", type=int, default=100)
    parser.add_argument("--geometry-timestep-high", type=int, default=400)
    parser.add_argument("--geometry-weights", nargs="+", type=float, default=(1.0, 1.0, 1.0))
    parser.add_argument("--geometry-loss-weight", type=float, default=1.0)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.1)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--diagnose-interval", action="store_true")
    parser.add_argument("--diagnostic-timestep", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.steps < 1 or args.accumulation < 1:
        raise ValueError("Invalid budget")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    if len(args.geometry_weights) != 3 or any(not math.isfinite(w) or w <= 0 for w in args.geometry_weights):
        raise ValueError("geometry-weights must be three positive finite values")
    if not (0 < args.geometry_timestep_low <= args.geometry_timestep_high < 1000):
        raise ValueError("Invalid geometry timestep interval")
    args.out_dir.mkdir(parents=True)
    started = time.perf_counter()

    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32
    amp = lambda: torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    dataset = GeometryAuditDataset(args.manifest, args.split_dir, args.ids_file, "train")
    items = [dataset[i] for i in range(len(dataset))]
    if len(items) < 2:
        raise ValueError("At least two train samples are required")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    ranking_negative = "source_geometry" if args.variant == "geometry_loss_plus_ranking" else "not_applicable"
    config.update({
        "variant": args.variant, "arms": ARMS, "image_ids": [i["image_id"] for i in items],
        "split": "train", "identity_branch_frozen": True, "ranking_negative": ranking_negative,
        "code_hashes": {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("reconstruction_adapter.py"),
                                                       Path(__file__).with_name("reconstruction_data.py"),
                                                       Path(__file__).with_name("differentiable_geometry.py"),
                                                       Path(__file__).with_name("geometry_audit_data.py"))},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "supervision": "source epsilon MSE + target geometry (pose SO3 / expression RMSE / landmark NME); no FAN/rescue/gaze",
    })
    save_json(args.out_dir / "config.json", config)
    (args.out_dir / "exact_command.txt").write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding="utf-8")

    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    sf = float(vae.config.scaling_factor)
    latents = []
    with torch.no_grad():
        for item in items:
            latent = vae.encode(item["image"][None].to(device)).latent_dist.mode() * sf
            if not torch.isfinite(latent).all():
                raise ValueError(f"Nonfinite source latent: {item['image_id']}")
            latents.append(latent.cpu())
    unet = UNet2DConditionModel.from_pretrained(args.backbone_path / "unet", variant="fp16",
                                                torch_dtype=dtype, local_files_only=True).to(device).eval().requires_grad_(False)
    model = ReconstructionAdapter(unet).to(device)
    unet.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
    frozen_before = model.frozen_hash()
    hashes = model_hashes(args.backbone_path, args.vae_path, args.empty_prompt)
    fingerprint = {
        "manifest": file_hash(args.manifest),
        "inputs": dataset.input_hashes,
        "split_hashes": {n: file_hash(args.split_dir / n) for n in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")},
        "model_files": hashes,
        "lr": args.lr, "accumulation": args.accumulation, "seed": args.seed,
        "mode": f"geometry_supervision_{args.variant}", "gaze_loss": False, "size": 256,
        "code_hashes": {p.name: file_hash(p) for p in (Path(__file__), Path(__file__).with_name("reconstruction_adapter.py"), Path(__file__).with_name("reconstruction_data.py"))},
    }
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    verify_warm_start(saved, hashes, Path(__file__).parent)
    if model.frozen_hash() != saved["frozen_hash"]:
        raise ValueError("Warm-start frozen UNet hash mismatch")
    load_adapter_exact(model, saved["adapter"])
    freeze_identity_branch(model)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable or any(not n.startswith("face.") for n, p in model.named_parameters() if p.requires_grad):
        raise ValueError("Expected only the Face Adapter to be trainable after freezing identity")
    model.requires_grad_(False)
    for p in trainable:
        p.requires_grad_(True)

    empty = load_file(str(args.empty_prompt))["empty_prompt_embedding"].to(device, dtype=dtype)
    if tuple(empty.shape) != (1, 77, 768) or not torch.isfinite(empty).all():
        raise ValueError("Invalid cached empty prompt")
    scheduler = DDPMScheduler.from_pretrained(args.backbone_path / "scheduler", local_files_only=True)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("Requires an epsilon scheduler")

    deca = load_deca_frozen(args.deca_root, device)
    targets = [load_target_geometry(deca, Path(item["deca_mat"]), Path(item["phase2_npz"]), device) for item in items]

    preflight = run_preflight(model, vae, deca, empty, scheduler, device, dtype, amp, items, targets)
    save_json(args.out_dir / "preflight.json", preflight)
    print(json.dumps({"preflight": preflight}, indent=2))
    if args.preflight_only:
        return

    if args.diagnose_interval:
        schedule = diagnose_timestep_interval(model, vae, deca, empty, scheduler, device, dtype, amp, items, targets)
        save_json(args.out_dir / "geometry_schedule.json", schedule)
        print(json.dumps(schedule, indent=2))
        return

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda, init_scale=128)
    geometry_weights = tuple(args.geometry_weights)
    frozen_unet_hash = frozen_before
    initial_state = {n: v.clone() for n, v in model.adapter_state().items()}

    def geometry_from_eps(noisy, timestep, eps):
        x0 = one_step_x0(scheduler, noisy, timestep, eps)
        rgb_tensor = vae.decode(x0.float() / sf).sample
        return estimate_geometry(deca, rgb_tensor), x0

    def geometry_error_vs_target(pose, exp, lm, target):
        return {
            "pose": geodesic_angle_deg(pose[:, :3], target["pose"][None][:, :3]).mean(),
            "expression": expression_rmse(exp, target["expression"][None]).mean(),
            "landmark": landmark_nme(lm, target["landmarks"][None]).mean(),
        }

    log_path = args.out_dir / "training_log.jsonl"
    diagnostics_dir = args.out_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    loss_weight_geom = args.geometry_loss_weight
    loss_weight_rank = args.ranking_loss_weight
    margin = args.ranking_margin

    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            total_terms = {}
            for micro in range(args.accumulation):
                serial = step * args.accumulation + micro
                order = torch.randperm(len(items), generator=torch.Generator().manual_seed(args.seed + serial // len(items)))
                index = int(order[serial % len(items)])
                item = items[index]
                latent = latents[index].to(device, dtype=dtype)
                condition = item["source_condition"][None].to(device)
                target_condition = item["target_condition"][None].to(device)
                identity = item["identity"][None].to(device)
                target = targets[index]

                # Source reconstruction path (full timestep range, epsilon MSE).
                gen = torch.Generator().manual_seed(args.seed + serial + 1)
                noise = torch.randn(latent.shape, generator=gen).to(device, dtype=dtype)
                t_src = torch.randint(0, scheduler.config.num_train_timesteps, (1,), generator=gen).to(device)
                noisy_src = scheduler.add_noise(latent, noise, t_src)
                with amp():
                    eps_src = model(noisy_src, t_src, condition, identity, empty)
                loss_src = F.mse_loss(eps_src.float(), noise.float())

                # Target geometry path (fixed low/mid-noise interval).
                t_geom = torch.randint(args.geometry_timestep_low, args.geometry_timestep_high + 1, (1,), generator=torch.Generator().manual_seed(args.seed + serial + 2)).to(device)
                gen_g = torch.Generator().manual_seed(args.seed + serial + 3)
                noise_g = torch.randn(latent.shape, generator=gen_g).to(device, dtype=dtype)
                noisy_geom = scheduler.add_noise(latent, noise_g, t_geom)
                with amp():
                    eps_geom = model(noisy_geom, t_geom, target_condition, identity, empty)
                (pose_p, exp_p, lm_p), _ = geometry_from_eps(noisy_geom, t_geom, eps_geom)
                geom = geometry_loss(pose_p, exp_p, lm_p, target["pose"][None], target["expression"][None], target["landmarks"][None], geometry_weights)
                if not (torch.isfinite(loss_src) and torch.isfinite(geom["total"])):
                    raise RuntimeError(f"Nonfinite training loss at step {step}")
                terms = {"src_epsilon_mse": float(loss_src.detach().cpu()), "geometry_total": float(geom["total"].detach().cpu()),
                         **{f"geometry_{k}": float(v.detach().cpu()) for k, v in geom["terms"].items()}}
                scaler.scale(loss_src / args.accumulation).backward()

                if args.variant == "geometry_loss_plus_ranking":
                    # The negative arm is the SAME sample's source condition, sharing the
                    # identical source latent / noise / timestep / source identity as the
                    # target arm. Shuffled near-canonical targets are descriptive only and
                    # are never used as a training negative.
                    with amp():
                        eps_source = model(noisy_geom, t_geom, condition, identity, empty)
                    (pose_s, exp_s, lm_s), _ = geometry_from_eps(noisy_geom, t_geom, eps_source)
                    d_target = normalized_geometry_distance(geom["terms"]["pose_so3_deg"], geom["terms"]["expression_rmse"], geom["terms"]["landmark_nme"])
                    d_source = normalized_geometry_distance(
                        geodesic_angle_deg(pose_s[:, :3], target["pose"][None][:, :3]).mean(),
                        expression_rmse(exp_s, target["expression"][None]).mean(),
                        landmark_nme(lm_s, target["landmarks"][None]).mean())
                    loss_rank = margin_ranking_loss(d_target, d_source, margin)
                    if not torch.isfinite(loss_rank):
                        raise RuntimeError(f"Nonfinite ranking loss at step {step}")
                    terms["ranking"] = float(loss_rank.detach().cpu())
                    combined = loss_weight_geom * geom["total"] + loss_weight_rank * loss_rank
                    scaler.scale(combined / args.accumulation).backward()
                else:
                    scaler.scale(loss_weight_geom * geom["total"] / args.accumulation).backward()

                for key, value in terms.items():
                    total_terms[key] = total_terms.get(key, 0.0) + value
            scaler.unscale_(optimizer)
            grad_norms = {n: float(p.grad.detach().float().norm()) for n, p in model.named_parameters()
                          if p.requires_grad and p.grad is not None}
            total_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            row = {"step": step + 1, "loss": {k: v / args.accumulation for k, v in total_terms.items()},
                   "gradient_norm_max": float(max(grad_norms.values())) if grad_norms else 0.0,
                   "total_gradient_norm": float(total_norm), "amp_scale": scaler.get_scale(),
                   "elapsed_seconds": time.perf_counter() - started}
            log.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
            if (step + 1) % 16 == 0 or step + 1 == args.steps:
                diagnostic = run_diagnostic(model, vae, deca, empty, scheduler, device, dtype, amp, items, targets,
                                            latents, sf, args.diagnostic_timestep, args.seed, frozen_unet_hash)
                save_json(diagnostics_dir / f"step_{(step + 1):04d}.json", diagnostic)
                with torch.no_grad():
                    make_diagnostic_sheet(model, vae, empty, scheduler, device, dtype, amp, items, latents, sf,
                                          args.diagnostic_timestep, args.seed, diagnostics_dir / f"step_{(step + 1):04d}_contact.png")
                save_json(args.out_dir / f"checkpoint_step_{(step + 1):04d}.pt_state_note.json",
                          {"note": "adapter state saved via atomic checkpoint", "step": step + 1})
                saved_state = {"step": step + 1, "adapter": model.adapter_state(), "optimizer": optimizer.state_dict(),
                               "scaler": scaler.state_dict(), "frozen_hash": frozen_unet_hash, "fingerprint": fingerprint}
                tmp = args.out_dir / "checkpoint.tmp"
                torch.save(saved_state, tmp)
                tmp.replace(args.out_dir / f"checkpoint_step_{(step + 1):04d}.pt")

    if model.frozen_hash() != frozen_unet_hash:
        raise RuntimeError("Frozen UNet changed during training")
    changes = {prefix: sum(float((v - initial_state[n]).square().sum()) for n, v in model.adapter_state().items() if n.startswith(prefix)) ** 0.5
               for prefix in ("face.", "identity.", "unet.")}
    summary = {
        "status": "completed", "variant": args.variant, "optimizer_steps": args.steps,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "identity_frozen": True, "adapter_update_l2": changes,
        "frozen_unet_hash_before": frozen_before, "frozen_unet_hash_after": model.frozen_hash(),
        "geometry_timestep_low": args.geometry_timestep_low, "geometry_timestep_high": args.geometry_timestep_high,
        "geometry_weights": list(geometry_weights), "geometry_loss_weight": loss_weight_geom,
        "ranking_loss_weight": loss_weight_rank, "ranking_margin": margin, "ranking_negative": ranking_negative,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None,
        "wall_seconds": time.perf_counter() - started, "scope": config["supervision"],
    }
    save_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def run_diagnostic(model, vae, deca, empty, scheduler, device, dtype, amp, items, targets, latents, sf, timestep, seed, frozen_hash):
    rows = []
    hooks = ResidualHooks(model)
    with torch.no_grad():
        for index, item in enumerate(items):
            latent = latents[index].to(device, dtype=dtype)
            t = torch.tensor([timestep], device=device)
            generator = torch.Generator().manual_seed(seed + 300000 + index)
            noise = torch.randn(latent.shape, generator=generator).to(device, dtype=dtype)
            noisy = scheduler.add_noise(latent, noise, t)
            for arm in ARMS:
                condition, identity, geometry_id, identity_id = intervention(items, index, arm)
                with amp():
                    eps = model(noisy, t, condition[None].to(device), identity[None].to(device), empty)
                x0 = one_step_x0(scheduler, noisy, t, eps)
                rgb_tensor = vae.decode(x0.float() / sf).sample
                pose, exp, lm = estimate_geometry(deca, rgb_tensor)
                target = targets[index]
                errors = {
                    "pose_deg": float(geodesic_angle_deg(pose[:, :3], target["pose"][None][:, :3]).item()),
                    "expression_rmse": float(expression_rmse(exp, target["expression"][None]).item()),
                    "landmark_nme": float(landmark_nme(lm, target["landmarks"][None]).item()),
                }
                rows.append({"image_id": item["image_id"], "arm": arm, **errors,
                             "residual_scales": residual_scales(hooks.residuals),
                             "residual_activation_rms": residual_activation_rms(hooks.residuals, hooks.activations)})
        zero_condition = torch.zeros_like(items[0]["source_condition"][None]).to(device)
        zero_residuals = model.face(zero_condition, latents[0].shape[-2:])
        zero_rows = [{"scale": i, "mean_abs": float(r.detach().float().abs().mean()), "rms": float(r.detach().float().square().mean().sqrt())}
                     for i, r in enumerate(zero_residuals)]
    hooks.remove()
    aggregates = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r["arm"] == arm]
        aggregates[arm] = {
            "pose_deg_mean": float(np.mean([r["pose_deg"] for r in arm_rows])),
            "expression_rmse_mean": float(np.mean([r["expression_rmse"] for r in arm_rows])),
            "landmark_nme_mean": float(np.mean([r["landmark_nme"] for r in arm_rows])),
        }
    return {"timestep": timestep, "n_samples": len(items), "arms": aggregates,
            "per_sample": rows, "zero_input_residuals": zero_rows,
            "frozen_unet_hash": frozen_hash}


def make_diagnostic_sheet(model, vae, empty, scheduler, device, dtype, amp, items, latents, sf, timestep, seed, path):
    count = min(4, len(items))
    sheet = Image.new("RGB", (256 * len(ARMS), 280 * count), "white")
    t = torch.tensor([timestep], device=device)
    with torch.no_grad():
        for index in range(count):
            latent = latents[index].to(device, dtype=dtype)
            generator = torch.Generator().manual_seed(seed + 300000 + index)
            noise = torch.randn(latent.shape, generator=generator).to(device, dtype=dtype)
            noisy = scheduler.add_noise(latent, noise, t)
            for arm_index, arm in enumerate(ARMS):
                condition, identity, _, _ = intervention(items, index, arm)
                with amp():
                    eps = model(noisy, t, condition[None].to(device), identity[None].to(device), empty)
                x0 = one_step_x0(scheduler, noisy, t, eps)
                image = rgb(vae.decode(x0.float() / sf).sample[0])
                sheet.paste(image, (arm_index * 256, index * 280 + 24))
                ImageDraw.Draw(sheet).text((arm_index * 256 + 3, index * 280 + 3), f"{items[index]['image_id']} {arm}", fill="black")
    sheet.save(path)


if __name__ == "__main__":
    main()
