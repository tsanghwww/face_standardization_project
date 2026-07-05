"""Train the Phase2 condition generator on DECA parameter outputs."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from .dataset import Phase2Dataset, save_normalizer
from .features import Phase2Sample, find_deca_mat_files, read_arcface_rows, sample_from_mat
from .model import ConditionGenerator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deca-results-dir", required=True, type=Path)
    parser.add_argument("--arcface-manifest", type=Path)
    parser.add_argument("--xgb-quality-manifest", type=Path)
    parser.add_argument("--quality-source", default="heuristic", choices=["heuristic", "xgb", "blend"])
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--no-augment", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_xgb_rows(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["image_id"]: row for row in csv.DictReader(f) if row.get("image_id")}


def _float(row: dict[str, str], key: str, default: float) -> float:
    try:
        value = float(row.get(key, ""))
    except ValueError:
        return default
    return value if np.isfinite(value) else default


def apply_xgb_quality(sample: Phase2Sample, xgb_row: dict[str, str] | None, quality_source: str) -> Phase2Sample:
    if not xgb_row:
        return sample
    metrics = dict(sample.metrics)
    xgb_quality = _float(xgb_row, "xgb_quality_score", metrics["quality_score"])
    if quality_source == "xgb":
        metrics["quality_score"] = xgb_quality
    elif quality_source == "blend":
        metrics["quality_score"] = 0.5 * metrics["quality_score"] + 0.5 * xgb_quality
    metrics["xgb_quality_score"] = xgb_quality
    metrics["sample_weight"] = _float(xgb_row, "xgb_sample_weight", 1.0)
    label = xgb_row.get("xgb_quality_label", "")
    metrics["xgb_quality_class"] = {"low": 0.0, "medium": 1.0, "high": 2.0}.get(label, -1.0)
    return Phase2Sample(image_id=sample.image_id, mat_path=sample.mat_path, params=sample.params, metrics=metrics)


def load_samples(results_dir: Path, arcface_manifest: Path | None, xgb_manifest: Path | None, quality_source: str):
    arcface = read_arcface_rows(arcface_manifest)
    xgb_rows = read_xgb_rows(xgb_manifest)
    mats = find_deca_mat_files(results_dir)
    if not mats:
        raise SystemExit(f"No .mat files found under {results_dir}")
    return [apply_xgb_quality(sample_from_mat(path, arcface.get(path.stem)), xgb_rows.get(path.stem), quality_source) for path in mats]


def normalizer_from_dataset(dataset: Phase2Dataset) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for item in dataset:
        rows.append(item["features"].numpy())
    arr = np.vstack(rows).astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-6
    return mean, std


def normalize_batch(batch: dict, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "features": ((batch["features"].to(device) - mean) / std).float(),
        "expression": batch["expression"].to(device).float(),
        "pose": batch["pose"].to(device).float(),
        "quality": batch["quality"].to(device).float(),
        "reject_target": batch["reject_target"].to(device).float(),
        "sample_weight": batch["sample_weight"].to(device).float(),
    }


def alpha_target(quality: torch.Tensor) -> torch.Tensor:
    exp = 0.20 + 0.75 * quality
    head = 0.25 + 0.70 * quality
    jaw = 0.15 + 0.70 * quality
    return torch.cat([exp, head, jaw], dim=1).clamp(0.0, 1.0)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def compute_loss(model: ConditionGenerator, batch: dict[str, torch.Tensor], mean: torch.Tensor, std: torch.Tensor, device: torch.device):
    b = normalize_batch(batch, mean, std, device)
    out = model(b["features"], b["expression"], b["pose"])
    q = b["quality"]
    w = b["sample_weight"]

    cond = (
        out.standardized_expression.abs().mean(dim=1, keepdim=True)
        + out.standardized_pose.abs().mean(dim=1, keepdim=True)
    )
    cond_loss = weighted_mean(q * cond, w)

    target_reg = (
        out.target_expression.abs().mean()
        + out.target_head_pose.abs().mean()
        + out.target_jaw_pose.abs().mean()
    )
    alphas = torch.cat([out.alpha_expression, out.alpha_head_pose, out.alpha_jaw_pose], dim=1)
    alpha_loss = weighted_mean((alphas - alpha_target(q)).pow(2).mean(dim=1, keepdim=True), w)
    confidence_loss = weighted_mean((out.confidence - q).pow(2), w)
    reject_loss = weighted_mean(
        nn.functional.binary_cross_entropy(out.reject_score, b["reject_target"].clamp(0.0, 1.0), reduction="none"), w
    )

    noise = torch.randn_like(b["features"]) * 0.015
    out_noisy = model(b["features"] + noise, b["expression"], b["pose"])
    smooth_per_sample = (
        (out.standardized_expression - out_noisy.standardized_expression).pow(2).mean(dim=1, keepdim=True)
        + (out.standardized_pose - out_noisy.standardized_pose).pow(2).mean(dim=1, keepdim=True)
        + (alphas - torch.cat([out_noisy.alpha_expression, out_noisy.alpha_head_pose, out_noisy.alpha_jaw_pose], dim=1))
        .pow(2)
        .mean(dim=1, keepdim=True)
    )
    smooth_loss = weighted_mean(smooth_per_sample, w)

    total = cond_loss + 0.03 * target_reg + 0.45 * alpha_loss + 0.35 * confidence_loss + 0.25 * reject_loss + 0.10 * smooth_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "cond": float(cond_loss.detach().cpu()),
        "target_reg": float(target_reg.detach().cpu()),
        "alpha": float(alpha_loss.detach().cpu()),
        "confidence": float(confidence_loss.detach().cpu()),
        "reject": float(reject_loss.detach().cpu()),
        "smooth": float(smooth_loss.detach().cpu()),
    }
    return total, metrics


def evaluate(model, loader, mean, std, device):
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            _, metrics = compute_loss(model, batch, mean, std, device)
            bs = int(batch["features"].shape[0])
            count += bs
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * bs
    return {key: value / max(count, 1) for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = load_samples(args.deca_results_dir, args.arcface_manifest, args.xgb_quality_manifest, args.quality_source)
    base_dataset = Phase2Dataset(samples, augment=False, seed=args.seed, stage=args.stage)
    mean_np, std_np = normalizer_from_dataset(base_dataset)
    dataset = Phase2Dataset(samples, augment=not args.no_augment, seed=args.seed, stage=args.stage)

    val_len = max(1, int(len(dataset) * args.val_ratio)) if len(dataset) > 3 else 0
    train_len = len(dataset) - val_len
    if val_len:
        train_ds, val_ds = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False) if val_ds is not None else None
    device = resolve_device(args.device)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    input_dim = int(mean_np.shape[0])
    model = ConditionGenerator(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_normalizer(args.out_dir / "normalizer.npz", mean_np, std_np)
    history_path = args.out_dir / "train_history.csv"
    best_val = float("inf")
    history_fields = ["epoch", "split", "loss", "cond", "target_reg", "alpha", "confidence", "reject", "smooth"]
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals: dict[str, float] = {}
            seen = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = compute_loss(model, batch, mean, std, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                bs = int(batch["features"].shape[0])
                seen += bs
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value * bs
            train_metrics = {key: value / max(seen, 1) for key, value in totals.items()}
            writer.writerow({"epoch": epoch, "split": "train", **train_metrics})

            val_metrics = evaluate(model, val_loader, mean, std, device) if val_loader is not None else train_metrics
            writer.writerow({"epoch": epoch, "split": "val", **val_metrics})
            f.flush()
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "input_dim": input_dim,
                        "hidden_dim": args.hidden_dim,
                        "feature_mean": mean_np,
                        "feature_std": std_np,
                        "config": vars(args),
                    },
                    args.out_dir / "best_model.pt",
                )
            print(
                f"epoch={epoch:03d} train_loss={train_metrics['loss']:.5f} "
                f"val_loss={val_metrics['loss']:.5f} best={best_val:.5f}"
            )

    summary = {
        "samples": len(samples),
        "train_samples": train_len,
        "val_samples": val_len,
        "input_dim": input_dim,
        "best_val_loss": best_val,
        "checkpoint": str(args.out_dir / "best_model.pt"),
        "normalizer": str(args.out_dir / "normalizer.npz"),
        "history": str(history_path),
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
