"""Train-only Phase3.1g high-resolution geometry-delta mechanism experiment."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import load_file

from phase3.differentiable_geometry import (
    estimate_geometry, expression_rmse, geodesic_angle_deg, geometry_loss,
    landmark_nme, load_deca_frozen, load_target_geometry, margin_ranking_loss,
    normalized_geometry_distance, one_step_x0,
)
from phase3.geometry_audit_data import GeometryAuditDataset, load_condition
from phase3.geometry_residual_adapter import GeometryResidualControl, minimum_pair_separation_loss
from phase3.reconstruction_adapter import ReconstructionAdapter
from phase3.reconstruction_data import file_hash
from phase3.sample_latent_img2img import load_adapter_exact
from phase3.train_geometry_supervision import model_hashes, verify_warm_start


GB = 1024 ** 3
GPU_BUDGET_GIB = 7.2
VARIANTS = ("negative_yaw", "positive_yaw")


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def load_counterfactuals(path: Path, items: list[dict], dataset_size: int, deca, device: str) -> list[dict]:
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {str(row["image_id"]): row for row in raw}
    expected = {item["image_id"] for item in items}
    if len(raw) != len(by_id) or expected - set(by_id):
        raise ValueError("Counterfactual manifest must contain every selected train ID exactly once")
    output = []
    for item in items:
        row = by_id[item["image_id"]]
        if row.get("identity_source_image_id") != item["image_id"]:
            raise ValueError(f"Counterfactual identity mismatch: {item['image_id']}")
        if float(row.get("pair_pose_separation_deg", 0)) < 10:
            raise ValueError(f"Counterfactual pair is below 10 degrees: {item['image_id']}")
        variants = {}
        for name in VARIANTS:
            value = row["variants"][name]
            condition_row = {
                "image_id": item["image_id"],
                "target_normal_map": value["target_normal_map"],
                "target_depth_map": value["target_depth_map"],
                "target_landmark_map": value["target_landmark_map"],
                "target_face_mask": value["target_face_mask_map"],
            }
            variants[name] = {
                "condition": load_condition(condition_row, "target", dataset_size),
                "target": load_target_geometry(deca, Path(item["deca_mat"]), Path(value["phase2_npz"]), device),
            }
        output.append({"image_id": item["image_id"], "variants": variants})
    return output


def geometry_distance(pose, expression, landmarks, target) -> torch.Tensor:
    return normalized_geometry_distance(
        geodesic_angle_deg(pose[:, :3], target["pose"][None][:, :3]).mean(),
        expression_rmse(expression, target["expression"][None]).mean(),
        landmark_nme(landmarks, target["landmarks"][None]).mean(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest", "split-dir", "ids-file", "counterfactual-manifest", "backbone-path",
        "vae-path", "empty-prompt", "checkpoint", "deca-root", "out-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--max-identities", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--geometry-timestep-low", type=int, default=100)
    parser.add_argument("--geometry-timestep-high", type=int, default=400)
    parser.add_argument("--geometry-weights", nargs=3, type=float, default=(1.0, 75.044619, 148.788535))
    parser.add_argument("--geometry-loss-weight", type=float, default=0.003)
    parser.add_argument("--ranking-loss-weight", type=float, default=1.0)
    parser.add_argument("--ranking-margin", type=float, default=0.05)
    parser.add_argument("--pair-separation-weight", type=float, default=1.0)
    parser.add_argument("--minimum-transfer-ratio", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Output directory must be empty")
    if args.steps < 1 or args.max_identities < 1:
        raise ValueError("steps and max-identities must be positive")
    if not (100 <= args.geometry_timestep_low <= args.geometry_timestep_high <= 400):
        raise ValueError("Geometry timesteps must remain in [100, 400]")
    weights = (*args.geometry_weights, args.geometry_loss_weight, args.ranking_loss_weight,
               args.pair_separation_weight, args.minimum_transfer_ratio)
    if any(not math.isfinite(value) or value <= 0 for value in weights):
        raise ValueError("All loss and transfer weights must be finite and positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dataset = GeometryAuditDataset(args.manifest, args.split_dir, args.ids_file, "train")
    items = [dataset[index] for index in range(min(len(dataset), args.max_identities))]
    if len(items) < 2:
        raise ValueError("At least two train identities are required")
    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32
    amp = lambda: torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    hashes = model_hashes(args.backbone_path, args.vae_path, args.empty_prompt)
    source_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    verify_warm_start(source_checkpoint, hashes, Path(__file__).parent)
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, local_files_only=True, torch_dtype=torch.float32
    ).to(device).eval().requires_grad_(False)
    scaling_factor = float(vae.config.scaling_factor)
    unet = UNet2DConditionModel.from_pretrained(
        args.backbone_path / "unet", variant="fp16", local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    base = ReconstructionAdapter(unet).to(device).eval()
    load_adapter_exact(base, source_checkpoint["adapter"])
    if base.frozen_hash() != source_checkpoint["frozen_hash"]:
        raise ValueError("Source reconstruction frozen UNet mismatch")
    base.requires_grad_(False)
    unet.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
    model = GeometryResidualControl(base).to(device)
    model.base.eval()
    model.edit.train()
    trainable = list(model.edit.parameters())
    if not trainable or any(parameter.requires_grad for parameter in model.base.parameters()):
        raise RuntimeError("Only the high-resolution geometry edit branch may be trainable")

    empty = load_file(str(args.empty_prompt))["empty_prompt_embedding"].to(device, dtype=dtype)
    scheduler = DDPMScheduler.from_pretrained(args.backbone_path / "scheduler", local_files_only=True)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("Requires an epsilon scheduler")
    deca = load_deca_frozen(args.deca_root, str(device))
    pairs = load_counterfactuals(args.counterfactual_manifest, items, dataset.size, deca, str(device))

    latents = []
    with torch.no_grad():
        for item in items:
            latent = vae.encode(item["image"][None].to(device)).latent_dist.mode() * scaling_factor
            if not torch.isfinite(latent).all():
                raise ValueError(f"Nonfinite source latent: {item['image_id']}")
            latents.append(latent.cpu())

    fingerprint = {
        "architecture": model.architecture,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_checkpoint": file_hash(args.checkpoint),
        "manifest": file_hash(args.manifest),
        "ids_file": file_hash(args.ids_file),
        "counterfactual_manifest": file_hash(args.counterfactual_manifest),
        "split_hashes": {name: file_hash(args.split_dir / name) for name in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")},
        "model_files": hashes,
        "input_hashes": dataset.input_hashes,
        "image_ids": [item["image_id"] for item in items],
        "code_hashes": {path.name: file_hash(path) for path in (
            Path(__file__), Path(__file__).with_name("geometry_residual_adapter.py"),
            Path(__file__).with_name("reconstruction_adapter.py"),
            Path(__file__).with_name("reconstruction_data.py"),
            Path(__file__).with_name("differentiable_geometry.py"),
        )},
    }
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(fingerprint=fingerprint, source_no_op="hard frozen path",
                  pair_supervision="same identity/latent/noise/timestep; negative and positive yaw",
                  validation_used=False, fixed_test_used=False, gaze_enabled=False)
    save_json(args.out_dir / "config.json", config)
    (args.out_dir / "training_ids.txt").write_text(
        "\n".join(item["image_id"] for item in items) + "\n", encoding="utf-8"
    )
    (args.out_dir / "exact_command.txt").write_text(
        subprocess.list2cmdline([sys.executable, *sys.argv]), encoding="utf-8"
    )

    def decode_geometry(noisy, timestep, epsilon):
        latent = one_step_x0(scheduler, noisy, timestep, epsilon)
        image = vae.decode(latent.float() / scaling_factor).sample
        return estimate_geometry(deca, image)

    def paired_objective(index: int, serial: int):
        item, pair = items[index], pairs[index]
        latent = latents[index].to(device, dtype=dtype)
        source_condition = item["source_condition"][None].to(device)
        identity = item["identity"][None].to(device)
        generator = torch.Generator().manual_seed(args.seed + 1000 + serial)
        noise = torch.randn(latent.shape, generator=generator).to(device, dtype=dtype)
        timestep = torch.randint(
            args.geometry_timestep_low, args.geometry_timestep_high + 1, (1,), generator=generator
        ).to(device)
        noisy = scheduler.add_noise(latent, noise, timestep)
        with torch.no_grad(), amp():
            source_epsilon = model.source_forward(noisy, timestep, source_condition, identity, empty)
        with torch.no_grad():
            source_pose, source_exp, source_lm = decode_geometry(noisy, timestep, source_epsilon)

        predictions, absolute_losses, distances, rankings = {}, [], [], []
        for name in VARIANTS:
            variant = pair["variants"][name]
            target_condition = variant["condition"][None].to(device)
            with amp():
                epsilon = model(noisy, timestep, source_condition, target_condition, identity, empty)
            pose, expression, landmarks = decode_geometry(noisy, timestep, epsilon)
            target = variant["target"]
            absolute = geometry_loss(
                pose, expression, landmarks,
                target["pose"][None], target["expression"][None], target["landmarks"][None],
                tuple(args.geometry_weights),
            )
            target_distance = geometry_distance(pose, expression, landmarks, target)
            source_distance = geometry_distance(source_pose, source_exp, source_lm, target)
            ranking = margin_ranking_loss(target_distance, source_distance, args.ranking_margin)
            predictions[name] = (pose, expression, landmarks)
            absolute_losses.append(absolute["total"])
            distances.append(target_distance)
            rankings.append(ranking)

        output_separation = geodesic_angle_deg(
            predictions["negative_yaw"][0][:, :3], predictions["positive_yaw"][0][:, :3]
        ).mean()
        target_separation = geodesic_angle_deg(
            pair["variants"]["negative_yaw"]["target"]["pose"][None][:, :3],
            pair["variants"]["positive_yaw"]["target"]["pose"][None][:, :3],
        ).mean()
        separation_loss, required_separation = minimum_pair_separation_loss(
            output_separation, target_separation, args.minimum_transfer_ratio
        )
        absolute_mean = torch.stack(absolute_losses).mean()
        ranking_mean = torch.stack(rankings).mean()
        total = (
            args.geometry_loss_weight * absolute_mean
            + args.ranking_loss_weight * ranking_mean
            + args.pair_separation_weight * separation_loss
        )
        values = {
            "image_id": item["image_id"], "timestep": int(timestep.item()),
            "total": float(total.detach()), "absolute_geometry": float(absolute_mean.detach()),
            "ranking": float(ranking_mean.detach()), "pair_separation": float(separation_loss.detach()),
            "output_separation_deg": float(output_separation.detach()),
            "required_separation_deg": float(required_separation.detach()),
            "target_separation_deg": float(target_separation.detach()),
            "negative_distance": float(distances[0].detach()), "positive_distance": float(distances[1].detach()),
        }
        return total, values

    # Full paired preflight and hard no-op equivalence.
    source = items[0]["source_condition"][None].to(device)
    latent = latents[0].to(device, dtype=dtype)
    timestep = torch.tensor([250], device=device)
    noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(args.seed)).to(device, dtype=dtype)
    noisy = scheduler.add_noise(latent, noise, timestep)
    identity = items[0]["identity"][None].to(device)
    with torch.no_grad(), amp():
        base_epsilon = model.source_forward(noisy, timestep, source, identity, empty)
        no_op_epsilon = model(noisy, timestep, source, source, identity, empty)
    if not torch.equal(base_epsilon, no_op_epsilon):
        raise RuntimeError("Source no-op path is not exactly equal to the frozen reconstruction adapter")
    hooks, hook_values = [], []
    def capture_projection(module, inputs, output):
        output.retain_grad()
        hook_values.append(output)

    for projection in model.edit.outputs:
        hooks.append(projection.register_forward_hook(capture_projection))
    try:
        preflight_loss, preflight_values = paired_objective(0, 0)
        preflight_loss.backward()
    finally:
        for hook in hooks:
            hook.remove()
    gradient_scales = [float(value.grad.detach().float().norm()) if value.grad is not None else None for value in hook_values]
    if len(gradient_scales) != 8 or any(value is None or not math.isfinite(value) or value <= 0 for value in gradient_scales):
        raise RuntimeError(f"Invalid paired edit gradients: {gradient_scales}")
    peak_gib = torch.cuda.max_memory_allocated() / GB if use_cuda else 0.0
    preflight = {"status": "passed", "source_no_op_exact": True, "loss": preflight_values,
                 "projection_gradient_norms": gradient_scales, "gpu_peak_allocated_gib": peak_gib,
                 "budget_gib": GPU_BUDGET_GIB}
    if use_cuda and peak_gib >= GPU_BUDGET_GIB:
        raise RuntimeError(f"Preflight GPU peak {peak_gib:.3f} GiB exceeds budget")
    save_json(args.out_dir / "preflight.json", preflight)
    model.edit.zero_grad(set_to_none=True)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda, init_scale=128)
    started = time.perf_counter()
    log_path = args.out_dir / "training_log.jsonl"
    with log_path.open("w", encoding="utf-8", buffering=1) as handle:
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(len(items), generator=torch.Generator().manual_seed(args.seed + step // len(items)))
            index = int(order[step % len(items)])
            loss, row = paired_objective(index, step + 1)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite loss at step {step + 1}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            row.update(step=step + 1, gradient_norm=float(gradient_norm), amp_scale=scaler.get_scale(),
                       elapsed_seconds=time.perf_counter() - started)
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)

    checkpoint_path = args.out_dir / f"checkpoint_step_{args.steps:04d}.pt"
    torch.save({
        "step": args.steps, "architecture": model.architecture,
        "base_adapter": base.adapter_state(), "edit_adapter": model.edit_state(),
        "frozen_hash": base.frozen_hash(), "fingerprint": fingerprint,
    }, checkpoint_path)
    summary = {
        "status": "completed", "architecture": model.architecture, "optimizer_steps": args.steps,
        "n_identities": len(items), "trainable_parameters": sum(value.numel() for value in trainable),
        "source_no_op_exact": True, "identity_frozen": True, "unet_frozen": True,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2 if use_cuda else None,
        "wall_seconds": time.perf_counter() - started, "checkpoint_sha256": file_hash(checkpoint_path),
        "validation_used": False, "fixed_test_used": False, "gaze_enabled": False,
    }
    save_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
