"""Audit same-identity counterfactual pose tracking without optimization.

For each selected train identity, the negative-yaw and positive-yaw arms share
the source latent, noise, timestep, and identity embedding. Only the geometry
condition changes. Output pose is re-estimated through the detector-free,
differentiable DECA path used by Phase3.1d training.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from functools import partial
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from safetensors.torch import load_file

from phase3.differentiable_geometry import estimate_geometry, load_deca_frozen, one_step_x0
from phase3.counterfactual_geometry import counterfactual_tracking_metrics, summarize_counterfactual_rows
from phase3.geometry_audit_data import GeometryAuditDataset, load_condition
from phase3.reconstruction_adapter import ReconstructionAdapter
from phase3.reconstruction_data import file_hash, read_ids
from phase3.sample_latent_img2img import load_adapter_exact
from phase3.train_geometry_supervision import freeze_identity_branch, model_hashes, verify_warm_start


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "manifest", "ids-file", "split-dir", "counterfactual-manifest", "checkpoint",
        "backbone-path", "vae-path", "empty-prompt", "deca-root", "out-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--timesteps", nargs="+", type=int, default=(100, 250, 400))
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.out_dir.exists():
        raise ValueError("Audit output directory must be new")
    if not args.timesteps or len(args.timesteps) != len(set(args.timesteps)):
        raise ValueError("Audit timesteps must be nonempty and unique")
    if any(timestep < 100 or timestep > 400 for timestep in args.timesteps):
        raise ValueError("Counterfactual audit is restricted to timesteps 100-400")

    dataset = GeometryAuditDataset(args.manifest, args.split_dir, args.ids_file, "train")
    items = [dataset[index] for index in range(len(dataset))]
    ids = [item["image_id"] for item in items]
    train = read_ids(args.split_dir / "train_ids.txt")
    validation = read_ids(args.split_dir / "validation_ids.txt")
    fixed = read_ids(args.split_dir / "fixed_test_ids.txt")
    if set(ids) - train or set(ids) & validation or set(ids) & fixed:
        raise ValueError("Counterfactual audit is restricted to train IDs")

    raw = [json.loads(line) for line in args.counterfactual_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    counterfactuals = {str(row["image_id"]): row for row in raw}
    if len(counterfactuals) != len(raw) or set(counterfactuals) != set(ids):
        raise ValueError("Counterfactual manifest must contain every selected ID exactly once")
    for image_id, row in counterfactuals.items():
        if row.get("identity_source_image_id") != image_id or float(row.get("pair_pose_separation_deg", 0)) < 10:
            raise ValueError(f"Invalid same-identity counterfactual pair: {image_id}")

    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    dtype = torch.float16 if use_cuda else torch.float32
    amp = lambda: torch.autocast("cuda", dtype=torch.float16) if use_cuda else nullcontext()
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    hashes = model_hashes(args.backbone_path, args.vae_path, args.empty_prompt)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    verify_warm_start(saved, hashes, Path(__file__).parent)
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
    sf = float(vae.config.scaling_factor)
    unet = UNet2DConditionModel.from_pretrained(
        args.backbone_path / "unet", variant="fp16", local_files_only=True, torch_dtype=dtype
    ).to(device).eval().requires_grad_(False)
    model = ReconstructionAdapter(unet).to(device).eval()
    load_adapter_exact(model, saved["adapter"])
    freeze_identity_branch(model)
    model.requires_grad_(False)
    if model.frozen_hash() != saved["frozen_hash"]:
        raise ValueError("Frozen UNet hash mismatch")
    unet.enable_gradient_checkpointing(gradient_checkpointing_func=partial(checkpoint, use_reentrant=False))
    scheduler = DDPMScheduler.from_pretrained(args.backbone_path / "scheduler", local_files_only=True)
    if scheduler.config.prediction_type != "epsilon":
        raise ValueError("Requires an epsilon scheduler")
    empty = load_file(str(args.empty_prompt))["empty_prompt_embedding"].to(device, dtype=dtype)
    deca = load_deca_frozen(args.deca_root, str(device))

    args.out_dir.mkdir(parents=True)
    config = {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "split": "train",
        "image_ids": ids,
        "pairing": "same source latent, CPU noise, timestep, and identity; geometry condition only",
        "checkpoint_sha256": file_hash(args.checkpoint),
        "counterfactual_manifest_sha256": file_hash(args.counterfactual_manifest),
        "split_hashes": {name: file_hash(args.split_dir / name) for name in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")},
        "model_files": hashes,
        "code_sha256": file_hash(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scope": "train-only one-step counterfactual geometry tracking; no optimizer, validation, fixed test, or gaze claim",
    }
    save_json(args.out_dir / "config.json", config)
    (args.out_dir / "exact_command.txt").write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding="utf-8")

    rows = []
    started = time.perf_counter()
    for index, item in enumerate(items):
        pair = counterfactuals[item["image_id"]]
        with torch.no_grad():
            latent = (vae.encode(item["image"][None].to(device)).latent_dist.mode() * sf).to(dtype)
        identity = item["identity"][None].to(device)
        variants = {}
        for name in ("negative_yaw", "positive_yaw"):
            row = pair["variants"][name]
            condition_row = {
                "image_id": item["image_id"],
                "target_normal_map": row["target_normal_map"],
                "target_depth_map": row["target_depth_map"],
                "target_landmark_map": row["target_landmark_map"],
                "target_face_mask": row["target_face_mask_map"],
            }
            variants[name] = {
                "condition": load_condition(condition_row, "target", dataset.size)[None].to(device),
                "pose": np.asarray(row["target_pose"], dtype=np.float64),
            }
        noise = torch.randn(latent.shape, generator=torch.Generator().manual_seed(args.seed + index)).to(device, dtype=dtype)
        for timestep in args.timesteps:
            result = {"image_id": item["image_id"], "timestep": timestep, "status": "failed", "failure_reason": ""}
            try:
                t = torch.tensor([timestep], device=device)
                noisy = scheduler.add_noise(latent, noise, t)
                outputs = {}
                for name in ("negative_yaw", "positive_yaw"):
                    with torch.no_grad(), amp():
                        eps = model(noisy, t, variants[name]["condition"], identity, empty)
                    with torch.no_grad():
                        decoded = vae.decode(one_step_x0(scheduler, noisy, t, eps).float() / sf).sample
                        pose, _, _ = estimate_geometry(deca, decoded)
                    outputs[name] = pose[0].detach().float().cpu().numpy()
                result.update(
                    status="success",
                    output_negative_pose=outputs["negative_yaw"].tolist(),
                    output_positive_pose=outputs["positive_yaw"].tolist(),
                    **counterfactual_tracking_metrics(
                        outputs["negative_yaw"], outputs["positive_yaw"],
                        variants["negative_yaw"]["pose"], variants["positive_yaw"]["pose"],
                    ),
                )
            except Exception as error:
                result["failure_reason"] = f"{type(error).__name__}: {error}"
            rows.append(result)
            print(json.dumps({"completed": len(rows), "image_id": item["image_id"], "timestep": timestep, "status": result["status"]}), flush=True)

    with (args.out_dir / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    report = summarize_counterfactual_rows(rows, ids, list(args.timesteps))
    report.update({
        "status": "completed",
        "metrics_sha256": file_hash(args.out_dir / "metrics.jsonl"),
        "wall_seconds": time.perf_counter() - started,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if use_cuda else None,
        "denominator_policy": "all selected train IDs at every timestep; failures retained",
        "gaze_evaluated": False,
    })
    save_json(args.out_dir / "summary.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
