"""Paired geometry-only interventions for the Phase3.1 adapter checkpoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
import time

import diffusers
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from PIL import Image, ImageDraw
from safetensors.torch import load_file
import torch

from phase3.geometry_audit_data import (
    GeometryAuditDataset, evaluation_fingerprint, intervention, verify_target_provenance, verify_training_isolation,
)
from phase3.reconstruction_adapter import ReconstructionAdapter
from phase3.reconstruction_data import file_hash
from phase3.sample_latent_img2img import load_adapter_exact, rgb, sample_latent, tensor_hash


ARMS = (
    "source_geometry",
    "target_geometry",
    "zero_geometry",
    "shuffled_geometry",
    "target_geometry_shuffled_identity",
)


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _model_hashes(backbone: Path, vae: Path, empty_prompt: Path) -> dict[str, str]:
    files = (
        backbone / "unet/diffusion_pytorch_model.fp16.safetensors",
        backbone / "unet/config.json",
        backbone / "scheduler/scheduler_config.json",
        vae / "diffusion_pytorch_model.safetensors",
        vae / "config.json",
        empty_prompt,
    )
    return {str(path): file_hash(path) for path in files}


def verify_checkpoint(saved: dict, current_hashes: dict[str, str]) -> None:
    expected = list(saved["fingerprint"]["model_files"].values())
    actual = list(current_hashes.values())
    if sorted(expected) != sorted(actual):
        raise ValueError("Backbone/VAE/prompt contents differ from the training checkpoint")
    root = Path(__file__).parent
    for name in ("reconstruction_adapter.py", "reconstruction_data.py"):
        if saved["fingerprint"]["code_hashes"][name] != file_hash(root / name):
            raise ValueError(f"Checkpoint implementation mismatch: {name}")


def residual_stats(model: ReconstructionAdapter, condition: torch.Tensor, latent_size: tuple[int, int]) -> list[dict]:
    with torch.no_grad():
        values = model.face(condition, latent_size)
    return [
        {
            "scale": index,
            "shape": list(value.shape),
            "mean_abs": float(value.detach().float().abs().mean()),
            "rms": float(value.detach().float().square().mean().sqrt()),
        }
        for index, value in enumerate(values)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-manifest", "evaluation-ids", "split-dir", "target-provenance", "checkpoint",
                 "backbone-path", "vae-path", "empty-prompt", "out-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evaluation-split", choices=("train", "validation"), default="validation")
    parser.add_argument("--strengths", nargs="+", type=float, default=(0.25, 0.5))
    parser.add_argument("--sampling-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    started = time.perf_counter()
    if len(args.strengths) != len(set(args.strengths)) or any(not 0 < value <= 1 for value in args.strengths):
        raise ValueError("Strengths must be unique and in (0,1]")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Output directory must be empty")

    dataset = GeometryAuditDataset(args.evaluation_manifest, args.split_dir, args.evaluation_ids, args.evaluation_split)
    if len(dataset) < 2:
        raise ValueError("At least two IDs are required for deterministic shuffling")
    items = [dataset[index] for index in range(len(dataset))]
    condition_deltas = [float((item["source_condition"] - item["target_condition"]).abs().mean()) for item in items]
    if not any(value > 1e-8 for value in condition_deltas):
        raise ValueError("All source/target geometry conditions are identical; intervention is uninformative")
    model_hashes = _model_hashes(args.backbone_path, args.vae_path, args.empty_prompt)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    verify_checkpoint(saved, model_hashes)
    training_fingerprint = saved["fingerprint"]
    training_ids = verify_training_isolation(training_fingerprint, {item["image_id"] for item in items}, args.split_dir)
    eval_fingerprint = evaluation_fingerprint(dataset, args.evaluation_manifest, args.evaluation_ids, args.split_dir, args.evaluation_split)
    target_provenance = verify_target_provenance(args.target_provenance, eval_fingerprint)

    scheduler = DDIMScheduler.from_pretrained(args.backbone_path / "scheduler", local_files_only=True)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("Requires an epsilon scheduler")
    schedules = []
    from phase3.sample_latent_img2img import img2img_timesteps
    for index, strength in enumerate(args.strengths):
        timesteps = img2img_timesteps(scheduler, args.sampling_steps, strength)
        schedules.append({"key": f"s{index:02d}", "strength": strength, "denoising_steps": len(timesteps),
                          "timesteps": timesteps.tolist(), "start_timestep": int(timesteps[0])})

    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32
    amp = lambda: torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "references").mkdir()
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update({
        "arms": ARMS,
        "schedules": schedules,
        "image_ids": [item["image_id"] for item in items],
        "training_fingerprint": training_fingerprint,
        "checkpoint_training_ids": sorted(training_ids),
        "evaluation_fingerprint": eval_fingerprint,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "target_provenance_sha256": file_hash(args.target_provenance),
        "target_provenance_checkpoint_sha256": target_provenance["checkpoint_sha256"],
        "model_files": model_hashes,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "code_hashes": {path.name: file_hash(path) for path in (Path(__file__), Path(__file__).with_name("geometry_audit_data.py"))},
        "torch": torch.__version__, "diffusers": diffusers.__version__, "cuda": torch.version.cuda,
        "pairing": "source latent, CPU noise seed, timestep schedule, and source identity fixed across geometry-only arms",
        "source_target_condition_l1": condition_deltas,
        "nonzero_target_interventions": sum(value > 1e-8 for value in condition_deltas),
        "scope": "Phase3.1c held-out geometry causal-response audit; no optimization, threshold tuning, or gaze claim",
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
            rgb(item["image"]).save(args.out_dir / "references" / f"{item['image_id']}_source.png")
            rgb(vae.decode(latent / sf).sample[0]).save(args.out_dir / "references" / f"{item['image_id']}_vae.png")
    reference_hashes = {
        path.name: file_hash(path) for path in sorted((args.out_dir / "references").glob("*.png"))
    }
    save_json(args.out_dir / "reference_hashes.json", reference_hashes)
    vae.to("cpu")
    if use_cuda:
        torch.cuda.empty_cache()

    unet = UNet2DConditionModel.from_pretrained(args.backbone_path / "unet", variant="fp16", torch_dtype=dtype,
                                               local_files_only=True).to(device).eval().requires_grad_(False)
    model = ReconstructionAdapter(unet).to(device).eval()
    if model.frozen_hash() != saved["frozen_hash"]:
        raise ValueError("Checkpoint frozen UNet hash mismatch")
    load_adapter_exact(model, saved["adapter"])
    model.requires_grad_(False)
    empty = load_file(str(args.empty_prompt))["empty_prompt_embedding"].to(device, dtype=dtype)
    if tuple(empty.shape) != (1, 77, 768) or not torch.isfinite(empty).all():
        raise ValueError("Invalid cached empty prompt")

    records, pending, residual_rows = [], [], []
    for spec in schedules:
        for index, item in enumerate(items):
            seed = args.seed + 200000 + index
            noise = torch.randn(latents[index].shape, generator=torch.Generator().manual_seed(seed))
            source_target_l1 = condition_deltas[index]
            for arm in ARMS:
                condition, identity, geometry_id, identity_id = intervention(items, index, arm)
                directory = args.out_dir / arm / spec["key"]
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{item['image_id']}_img2img.png"
                row = {
                    "image_id": item["image_id"], "arm": arm, **spec, "geometry_id": geometry_id,
                    "identity_id": identity_id, "noise_seed": seed, "noise_sha256": tensor_hash(noise),
                    "source_target_condition_l1": source_target_l1, "output": str(path.relative_to(args.out_dir)),
                    "status": "pending", "failure_reason": "",
                }
                try:
                    condition_device = condition[None].to(device)
                    identity_device = identity[None].to(device)
                    residual_rows.append({"image_id": item["image_id"], "arm": arm, "key": spec["key"],
                                          "geometry_id": geometry_id, "scales": residual_stats(model, condition_device, latents[index].shape[-2:])})
                    def predict(value, timestep):
                        with amp():
                            return model(value, timestep, condition_device, identity_device, empty)
                    sample, initial_hash = sample_latent(latents[index].to(device, dtype=dtype), noise.to(device, dtype=dtype),
                                                         scheduler, args.sampling_steps, spec["strength"], predict)
                    row["initial_latent_sha256"] = initial_hash
                    pending.append((sample.float().cpu(), path, row))
                except Exception as error:
                    row.update(status="generation_failed", failure_reason=f"{type(error).__name__}: {error}")
                records.append(row)
        print(json.dumps({"strength": spec["strength"], "records": len(records), "elapsed_seconds": time.perf_counter() - started}), flush=True)

    if model.frozen_hash() != saved["frozen_hash"]:
        raise RuntimeError("Frozen UNet changed during audit")
    model.to("cpu")
    if use_cuda:
        torch.cuda.empty_cache()
    vae.to(device)
    with torch.no_grad():
        for latent, path, row in pending:
            try:
                rgb(vae.decode(latent.to(device) / sf).sample[0]).save(path)
                row.update(status="generated", sha256=file_hash(path))
            except Exception as error:
                row.update(status="generation_failed", failure_reason=f"{type(error).__name__}: {error}")

    with (args.out_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    with (args.out_dir / "face_residuals.jsonl").open("w", encoding="utf-8") as handle:
        for row in residual_rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    for spec in schedules:
        count = min(4, len(items))
        sheet = Image.new("RGB", (256 * (2 + len(ARMS)), 280 * count), "white")
        for index, item in enumerate(items[:count]):
            columns = [("source", args.out_dir / "references" / f"{item['image_id']}_source.png"),
                       ("VAE", args.out_dir / "references" / f"{item['image_id']}_vae.png")]
            columns.extend((arm, args.out_dir / arm / spec["key"] / f"{item['image_id']}_img2img.png") for arm in ARMS)
            for column, (label, path) in enumerate(columns):
                if path.exists():
                    with Image.open(path) as image:
                        sheet.paste(image, (column * 256, index * 280 + 24))
                ImageDraw.Draw(sheet).text((column * 256 + 3, index * 280 + 3), f"{item['image_id']} {label}", fill="black")
        sheet.save(args.out_dir / f"contact_{spec['key']}.png")
    failed = sum(row["status"] != "generated" for row in records)
    summary = {
        "status": "completed" if not failed else "completed_with_failures", "n_ids": len(items),
        "expected_outputs": len(items) * len(ARMS) * len(schedules), "generated": len(records) - failed,
        "generation_failed": failed, "wall_seconds": time.perf_counter() - started,
        "gpu_peak_allocated_mb": torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None,
        "samples_sha256": file_hash(args.out_dir / "samples.jsonl"),
        "residuals_sha256": file_hash(args.out_dir / "face_residuals.jsonl"), "scope": config["scope"],
        "config_sha256": file_hash(args.out_dir / "config.json"),
        "reference_hashes_sha256": file_hash(args.out_dir / "reference_hashes.json"),
    }
    save_json(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
