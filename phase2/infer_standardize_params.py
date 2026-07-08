"""Run Phase2 standardization on DECA .mat outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .features import feature_vector, find_deca_mat_files, read_arcface_rows, sample_from_mat
from .model import ConditionGenerator


FIELDS = [
    "image_id",
    "mat_path",
    "out_npz",
    "quality_score",
    "alpha_expression",
    "alpha_head_pose",
    "alpha_jaw_pose",
    "confidence",
    "reject_score",
    "decision",
    "failure_reason",
    "original_exp_norm",
    "original_head_pose_norm",
    "original_jaw_pose_norm",
    "standardized_exp_norm",
    "standardized_head_pose_norm",
    "standardized_jaw_pose_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deca-results-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--arcface-manifest", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--reject-threshold", type=float, default=0.60)
    parser.add_argument("--weak-threshold", type=float, default=0.40)
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def decision(reject: float, confidence: float, reject_threshold: float, weak_threshold: float) -> tuple[str, str]:
    if reject >= reject_threshold and confidence < weak_threshold:
        return "reject", "high_reject_low_confidence"
    if confidence < weak_threshold:
        return "weak_standardize", "low_confidence"
    return "standardize", ""


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    # Phase2 checkpoints are produced locally by train_condition_generator and
    # include numpy arrays for the feature normalizer. PyTorch 2.6+ defaults to
    # weights_only=True, which rejects those trusted metadata objects.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = ConditionGenerator(input_dim=int(ckpt["input_dim"]), hidden_dim=int(ckpt["hidden_dim"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    mean = torch.from_numpy(np.asarray(ckpt["feature_mean"], dtype=np.float32)).to(device)
    std = torch.from_numpy(np.asarray(ckpt["feature_std"], dtype=np.float32)).to(device)

    arcface = read_arcface_rows(args.arcface_manifest)
    mat_files = find_deca_mat_files(args.deca_results_dir)
    if not mat_files:
        raise SystemExit(f"No .mat files found under {args.deca_results_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    params_dir = args.out_dir / "params"
    params_dir.mkdir(exist_ok=True)
    rows = []
    with torch.no_grad():
        for mat_path in mat_files:
            sample = sample_from_mat(mat_path, arcface.get(mat_path.stem))
            feat = feature_vector(sample.params, sample.metrics)
            x = ((torch.from_numpy(feat).to(device).float()[None, :] - mean) / std)
            exp = torch.from_numpy(sample.params["expression"]).to(device).float()[None, :]
            pose = torch.from_numpy(sample.params["pose"]).to(device).float()[None, :]
            out = model(x, exp, pose)
            std_exp = out.standardized_expression[0].cpu().numpy().astype(np.float32)
            std_pose = out.standardized_pose[0].cpu().numpy().astype(np.float32)
            target_exp = out.target_expression[0].cpu().numpy().astype(np.float32)
            target_pose = np.concatenate(
                [
                    out.target_head_pose[0].cpu().numpy().astype(np.float32),
                    out.target_jaw_pose[0].cpu().numpy().astype(np.float32),
                ]
            )
            alphas = [
                float(out.alpha_expression.item()),
                float(out.alpha_head_pose.item()),
                float(out.alpha_jaw_pose.item()),
            ]
            confidence = float(out.confidence.item())
            reject = float(out.reject_score.item())
            dec, reason = decision(reject, confidence, args.reject_threshold, args.weak_threshold)
            out_npz = params_dir / f"{sample.image_id}_phase2.npz"
            np.savez(
                out_npz,
                image_id=sample.image_id,
                source_mat=str(sample.mat_path),
                expression_original=sample.params["expression"],
                pose_original=sample.params["pose"],
                expression_standardized=std_exp,
                pose_standardized=std_pose,
                target_expression=target_exp,
                target_pose=target_pose,
                alpha_expression=np.float32(alphas[0]),
                alpha_head_pose=np.float32(alphas[1]),
                alpha_jaw_pose=np.float32(alphas[2]),
                confidence=np.float32(confidence),
                reject_score=np.float32(reject),
                quality_score=np.float32(sample.metrics["quality_score"]),
            )
            rows.append(
                {
                    "image_id": sample.image_id,
                    "mat_path": str(sample.mat_path),
                    "out_npz": str(out_npz),
                    "quality_score": f"{sample.metrics['quality_score']:.6f}",
                    "alpha_expression": f"{alphas[0]:.6f}",
                    "alpha_head_pose": f"{alphas[1]:.6f}",
                    "alpha_jaw_pose": f"{alphas[2]:.6f}",
                    "confidence": f"{confidence:.6f}",
                    "reject_score": f"{reject:.6f}",
                    "decision": dec,
                    "failure_reason": reason,
                    "original_exp_norm": f"{sample.metrics['exp_norm']:.6f}",
                    "original_head_pose_norm": f"{sample.metrics['head_pose_norm']:.6f}",
                    "original_jaw_pose_norm": f"{sample.metrics['jaw_pose_norm']:.6f}",
                    "standardized_exp_norm": f"{np.linalg.norm(std_exp) / np.sqrt(std_exp.size):.6f}",
                    "standardized_head_pose_norm": f"{np.linalg.norm(std_pose[:3]):.6f}",
                    "standardized_jaw_pose_norm": f"{np.linalg.norm(std_pose[3:]):.6f}",
                }
            )

    csv_path = args.out_dir / "phase2_inference_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "count": len(rows),
        "standardize": sum(1 for r in rows if r["decision"] == "standardize"),
        "weak_standardize": sum(1 for r in rows if r["decision"] == "weak_standardize"),
        "reject": sum(1 for r in rows if r["decision"] == "reject"),
        "manifest": str(csv_path),
        "params_dir": str(params_dir),
    }
    (args.out_dir / "phase2_inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
