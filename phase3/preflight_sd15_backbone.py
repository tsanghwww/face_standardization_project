#!/usr/bin/env python3
"""Audit a frozen SD1.5 text encoder and UNet before adapter training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shlex
import sys
import time
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file
from transformers import CLIPTextModel, CLIPTokenizer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and ".cache" not in path.parts)


def tree_hash(root: Path) -> tuple[str, list[dict[str, object]]]:
    records = []
    aggregate = hashlib.sha256()
    for path in model_files(root):
        relative = path.relative_to(root).as_posix()
        file_hash = sha256(path)
        records.append({"path": relative, "size": path.stat().st_size, "sha256": file_hash})
        aggregate.update(f"{relative}\0{file_hash}\n".encode("utf-8"))
    return aggregate.hexdigest(), records


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone-path", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    required = ["unet", "scheduler", "tokenizer", "text_encoder"]
    missing = [name for name in required if not (args.backbone_path / name).is_dir()]
    if missing:
        raise SystemExit(f"Backbone subdirectories missing: {missing}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)

    tokenizer = CLIPTokenizer.from_pretrained(args.backbone_path / "tokenizer", local_files_only=True)
    text_encoder = CLIPTextModel.from_pretrained(
        args.backbone_path / "text_encoder", torch_dtype=dtype, variant="fp16", local_files_only=True
    ).to(args.device).eval()
    text_encoder.requires_grad_(False)
    tokens = tokenizer(
        [""], padding="max_length", max_length=tokenizer.model_max_length,
        truncation=True, return_tensors="pt",
    ).input_ids.to(args.device)
    with torch.no_grad():
        empty_embedding = text_encoder(tokens)[0]
    if empty_embedding.shape != (1, tokenizer.model_max_length, 768) or not torch.isfinite(empty_embedding).all():
        raise RuntimeError(f"Invalid empty-prompt embedding: {tuple(empty_embedding.shape)}")
    embedding_path = args.out_dir / "empty_prompt_embedding.safetensors"
    save_file({"empty_prompt_embedding": empty_embedding.float().cpu().contiguous()}, str(embedding_path))
    del text_encoder, tokens
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    unet = UNet2DConditionModel.from_pretrained(
        args.backbone_path / "unet", torch_dtype=dtype, variant="fp16", local_files_only=True
    ).to(args.device).eval()
    unet.requires_grad_(False)
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    latent = torch.randn((1, 4, 32, 32), generator=generator, dtype=torch.float32).to(args.device, dtype=dtype)
    timestep = torch.tensor([500], device=args.device, dtype=torch.long)
    with torch.no_grad():
        output = unet(latent, timestep, encoder_hidden_states=empty_embedding.to(args.device, dtype=dtype)).sample
    finite = bool(torch.isfinite(output).all().item())
    if output.shape != latent.shape or not finite:
        raise RuntimeError(f"Invalid UNet output: shape={tuple(output.shape)} finite={finite}")

    backbone_hash, files = tree_hash(args.backbone_path)
    peak_mb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2) if args.device.startswith("cuda") else 0.0
    summary = {
        "status": "passed",
        "model_id": args.model_id,
        "revision": args.revision,
        "backbone_tree_sha256": backbone_hash,
        "empty_prompt_shape": list(empty_embedding.shape),
        "empty_prompt_sha256": sha256(embedding_path),
        "unet_input_shape": list(latent.shape),
        "unet_output_shape": list(output.shape),
        "unet_output_finite": finite,
        "unet_frozen": not any(parameter.requires_grad for parameter in unet.parameters()),
        "dtype": args.dtype,
        "device": args.device,
        "seed": args.seed,
        "gpu_peak_allocated_mb": peak_mb,
        "wall_seconds": time.perf_counter() - started,
        "scope": "frozen backbone interface preflight; no adapter training and no gaze-disentanglement claim",
    }
    environment = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
        **{name: package_version(name) for name in ("diffusers", "transformers", "accelerate", "tokenizers", "safetensors", "huggingface-hub")},
    }
    (args.out_dir / "preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    (args.out_dir / "backbone_artifact_hashes.json").write_text(json.dumps(files, indent=2), encoding="utf-8")
    (args.out_dir / "exact_command.txt").write_text(" ".join(shlex.quote(item) for item in sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
