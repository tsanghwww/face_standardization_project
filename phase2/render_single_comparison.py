"""Render hard-zero vs Phase2 standardized DECA parameters for one image."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.io import savemat

from .features import feature_vector, sample_from_mat
from .model import ConditionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--deca-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--iscrop", default=True, type=lambda x: x.lower() in {"true", "1"})
    parser.add_argument("--detector", default="fan")
    parser.add_argument("--rasterizer-type", default="standard")
    parser.add_argument("--reject-threshold", type=float, default=0.60)
    parser.add_argument("--weak-threshold", type=float, default=0.40)
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_phase2(checkpoint: Path, device: torch.device) -> tuple[ConditionGenerator, torch.Tensor, torch.Tensor]:
    ckpt = torch.load(checkpoint, map_location=device)
    model = ConditionGenerator(input_dim=int(ckpt["input_dim"]), hidden_dim=int(ckpt["hidden_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = torch.from_numpy(np.asarray(ckpt["feature_mean"], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(ckpt["feature_std"], dtype=np.float32)).to(device)
    return model, mean, std


def clone_codedict(codedict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in codedict.items():
        out[key] = value.clone() if isinstance(value, torch.Tensor) else value
    return out


def tensor_image(tensor: torch.Tensor, util_module) -> np.ndarray:
    return util_module.tensor2image(tensor[0])


def save_rgb(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def font(size: int) -> ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_contact_sheet(
    out_path: Path,
    image_paths: list[tuple[str, Path]],
    metrics: dict[str, float | str],
) -> None:
    tile_w, tile_h = 320, 390
    width = tile_w * len(image_paths)
    height = tile_h + 220
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = font(28)
    label_font = font(20)
    small_font = font(17)
    title = "Hard-Zero vs stage2_xgb_weighted"
    bbox = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (bbox[2] - bbox[0])) / 2, 18), title, fill=(30, 30, 30), font=title_font)

    for idx, (label, path) in enumerate(image_paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((285, 285))
        x = idx * tile_w + (tile_w - img.width) // 2
        y = 70
        canvas.paste(img, (x, y))
        draw.text((idx * tile_w + 22, 370), label, fill=(30, 30, 30), font=label_font)

    lines = [
        f"decision={metrics['decision']}   confidence={metrics['confidence']:.4f}   reject={metrics['reject_score']:.4f}",
        f"exp_norm: original={metrics['original_exp_norm']:.4f}, hard_zero=0.0000, phase2={metrics['phase2_exp_norm']:.4f}",
        f"head_pose_norm: original={metrics['original_head_pose_norm']:.4f}, hard_zero=0.0000, phase2={metrics['phase2_head_pose_norm']:.4f}",
        f"jaw_pose_norm: original={metrics['original_jaw_pose_norm']:.4f}, hard_zero=0.0000, phase2={metrics['phase2_jaw_pose_norm']:.4f}",
        f"alpha_exp={metrics['alpha_expression']:.4f}, alpha_head={metrics['alpha_head_pose']:.4f}, alpha_jaw={metrics['alpha_jaw_pose']:.4f}",
    ]
    y = tile_h + 15
    for line in lines:
        draw.text((28, y), line, fill=(45, 45, 45), font=small_font)
        y += 34
    canvas.save(out_path)


def decision(reject: float, confidence: float, reject_threshold: float, weak_threshold: float) -> str:
    if reject >= reject_threshold and confidence < weak_threshold:
        return "reject"
    if confidence < weak_threshold:
        return "weak_standardize"
    return "standardize"


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.deca_root))
    from decalib.deca import DECA
    from decalib.datasets import datasets
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg

    device_name = resolve_device(args.device)
    device = torch.device(device_name)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    testdata = datasets.TestData(str(args.image), iscrop=args.iscrop, face_detector=args.detector)
    if len(testdata) != 1:
        raise SystemExit(f"Expected one image, found {len(testdata)}")

    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device_name, render_enabled=True)
    model, mean, std = load_phase2(args.checkpoint, device)

    item = testdata[0]
    name = item["imagename"]
    images = item["image"].to(device)[None, ...]
    with torch.no_grad():
        codedict = deca.encode(images)
        original_op, original_vis = deca.decode(codedict, rendering=True, return_vis=True)

    sample_dir = args.out_dir / "deca_sample" / name
    sample_dir.mkdir(parents=True, exist_ok=True)
    mat_path = sample_dir / f"{name}.mat"
    np.savetxt(sample_dir / f"{name}_kpt2d.txt", original_op["landmarks2d"][0].detach().cpu().numpy())
    np.savetxt(sample_dir / f"{name}_kpt3d.txt", original_op["landmarks3d"][0].detach().cpu().numpy())
    matdict = util.dict_tensor2npy(original_op)
    matdict.update(
        {
            "shape": codedict["shape"].detach().cpu().numpy(),
            "expression": codedict["exp"].detach().cpu().numpy(),
            "pose": codedict["pose"].detach().cpu().numpy(),
            "camera": codedict["cam"].detach().cpu().numpy(),
            "light": codedict["light"].detach().cpu().numpy(),
            "detail": codedict["detail"].detach().cpu().numpy(),
        }
    )
    savemat(mat_path, matdict)

    sample = sample_from_mat(mat_path)
    feat = feature_vector(sample.params, sample.metrics)
    with torch.no_grad():
        x = ((torch.from_numpy(feat).to(device).float()[None, :] - mean) / std)
        exp = torch.from_numpy(sample.params["expression"]).to(device).float()[None, :]
        pose = torch.from_numpy(sample.params["pose"]).to(device).float()[None, :]
        out = model(x, exp, pose)
        phase2_exp = out.standardized_expression.detach().cpu().numpy()[0].astype(np.float32)
        phase2_pose = out.standardized_pose.detach().cpu().numpy()[0].astype(np.float32)
        confidence = float(out.confidence.item())
        reject_score = float(out.reject_score.item())
        alpha_expression = float(out.alpha_expression.item())
        alpha_head_pose = float(out.alpha_head_pose.item())
        alpha_jaw_pose = float(out.alpha_jaw_pose.item())

    hard_zero = clone_codedict(codedict)
    hard_zero["exp"] = torch.zeros_like(hard_zero["exp"])
    hard_zero["pose"] = torch.zeros_like(hard_zero["pose"])

    phase2 = clone_codedict(codedict)
    phase2["exp"] = torch.from_numpy(phase2_exp).to(device).float()[None, :]
    phase2["pose"] = torch.from_numpy(phase2_pose).to(device).float()[None, :]

    with torch.no_grad():
        hard_op, hard_vis = deca.decode(hard_zero, rendering=True, return_vis=True)
        phase2_op, phase2_vis = deca.decode(phase2, rendering=True, return_vis=True)

    paths = {
        "input": args.out_dir / "input_crop.jpg",
        "original_shape": args.out_dir / "original_shape_detail.jpg",
        "hard_zero_shape": args.out_dir / "hard_zero_shape_detail.jpg",
        "phase2_shape": args.out_dir / "stage2_xgb_weighted_shape_detail.jpg",
        "contact_sheet": args.out_dir / "hard_zero_vs_stage2_xgb_weighted_contact.png",
    }
    save_rgb(paths["input"], tensor_image(original_vis["inputs"], util))
    save_rgb(paths["original_shape"], tensor_image(original_vis["shape_detail_images"], util))
    save_rgb(paths["hard_zero_shape"], tensor_image(hard_vis["shape_detail_images"], util))
    save_rgb(paths["phase2_shape"], tensor_image(phase2_vis["shape_detail_images"], util))

    phase2_decision = decision(reject_score, confidence, args.reject_threshold, args.weak_threshold)
    metrics: dict[str, float | str] = {
        "decision": phase2_decision,
        "confidence": confidence,
        "reject_score": reject_score,
        "alpha_expression": alpha_expression,
        "alpha_head_pose": alpha_head_pose,
        "alpha_jaw_pose": alpha_jaw_pose,
        "original_exp_norm": sample.metrics["exp_norm"],
        "original_head_pose_norm": sample.metrics["head_pose_norm"],
        "original_jaw_pose_norm": sample.metrics["jaw_pose_norm"],
        "phase2_exp_norm": float(np.linalg.norm(phase2_exp) / np.sqrt(phase2_exp.size)),
        "phase2_head_pose_norm": float(np.linalg.norm(phase2_pose[:3])),
        "phase2_jaw_pose_norm": float(np.linalg.norm(phase2_pose[3:])),
    }
    make_contact_sheet(
        paths["contact_sheet"],
        [
            ("Input crop", paths["input"]),
            ("Original DECA", paths["original_shape"]),
            ("Hard-zero", paths["hard_zero_shape"]),
            ("Stage2 XGB weighted", paths["phase2_shape"]),
        ],
        metrics,
    )

    with (args.out_dir / "single_comparison_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"paths": {k: str(v) for k, v in paths.items()}, "metrics": metrics}, f, indent=2)
    with (args.out_dir / "single_comparison_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)
    print(json.dumps({"paths": {k: str(v) for k, v in paths.items()}, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
