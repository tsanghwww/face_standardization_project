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
from torch.utils.data import DataLoader

from .dataset import Phase2Dataset, save_normalizer
from .features import (
    Phase2Sample,
    apply_xgb_quality,
    find_deca_mat_files,
    read_arcface_rows,
    read_xgb_rows,
    sample_from_mat,
)
from .model import ConditionGenerator
from .outcome_dataset import FEATURE_COLUMNS as OUTCOME_FEATURE_COLUMNS
from .outcome_surrogate import OutcomeSurrogate


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
    parser.add_argument("--alpha-mode", default="learned", choices=["learned", "fixed_one"])
    parser.add_argument("--exclude-ids-file", type=Path)
    # Phase2.1 outcome supervision (default OFF -> Phase2 v1 behavior unchanged).
    parser.add_argument("--outcome-manifest", type=Path, default=None)
    parser.add_argument("--outcome-loss-weight", type=float, default=0.0)
    parser.add_argument("--no-outcome-supervision", action="store_true")
    parser.add_argument("--outcome-surrogate", type=Path)
    parser.add_argument("--outcome-identity-weight", type=float, default=0.0)
    parser.add_argument("--outcome-pose-weight", type=float, default=0.0)
    parser.add_argument("--outcome-gaze-weight", type=float, default=0.0)
    parser.add_argument("--outcome-render-failure-weight", type=float, default=0.0)
    parser.add_argument("--outcome-identity-floor", type=float, default=-0.02)
    parser.add_argument("--outcome-pose-improvement-floor", type=float, default=0.0)
    parser.add_argument("--outcome-gaze-ceiling-deg", type=float, default=10.0)
    parser.add_argument(
        "--allow-outcome-validation-overlap",
        "--allow-outcome-surrogate-val-overlap",
        dest="allow_outcome_validation_overlap",
        action="store_true",
        help="Smoke diagnostics only; forbidden for formal experiments.",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def read_id_file(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def write_command_artifacts(out_dir: Path, args: argparse.Namespace) -> None:
    """Persist config.json and exact_command.txt for experiment provenance."""
    import sys as _sys

    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    cmd = " ".join([_sys.executable, "-m", "phase2.train_condition_generator", *_sys.argv[1:]])
    (out_dir / "exact_command.txt").write_text(cmd + "\n", encoding="utf-8")


def make_split(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministic train/val index split (identical across ablation groups)."""
    val_len = max(1, int(n * val_ratio)) if n > 3 else 0
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    return sorted(perm[:val_len]), sorted(perm[val_len:])


def load_samples(
    results_dir: Path,
    arcface_manifest: Path | None,
    xgb_manifest: Path | None,
    quality_source: str,
    excluded_ids: set[str],
):
    arcface = read_arcface_rows(arcface_manifest)
    xgb_rows = read_xgb_rows(xgb_manifest)
    mats = find_deca_mat_files(results_dir)
    if not mats:
        raise SystemExit(f"No .mat files found under {results_dir}")
    return [
        apply_xgb_quality(sample_from_mat(path, arcface.get(path.stem)), xgb_rows.get(path.stem), quality_source)
        for path in mats
        if path.stem not in excluded_ids
    ]


def normalizer_from_dataset(dataset: Phase2Dataset) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for item in dataset:
        rows.append(item["features"].numpy())
    arr = np.vstack(rows).astype(np.float32)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0) + 1e-6
    return mean, std


def normalize_batch(batch: dict, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    outcome_context = batch.get("outcome_context")
    if outcome_context is None:
        outcome_context = torch.zeros((batch["features"].shape[0], 6), dtype=torch.float32)
    return {
        "features": ((batch["features"].to(device) - mean) / std).float(),
        "expression": batch["expression"].to(device).float(),
        "pose": batch["pose"].to(device).float(),
        "quality": batch["quality"].to(device).float(),
        "reject_target": batch["reject_target"].to(device).float(),
        "sample_weight": batch["sample_weight"].to(device).float(),
        "outcome_context": outcome_context.to(device).float(),
    }


def alpha_target(quality: torch.Tensor) -> torch.Tensor:
    exp = 0.20 + 0.75 * quality
    head = 0.25 + 0.70 * quality
    jaw = 0.15 + 0.70 * quality
    return torch.cat([exp, head, jaw], dim=1).clamp(0.0, 1.0)


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def build_outcome_surrogate_features(batch: dict[str, torch.Tensor], output) -> torch.Tensor:
    """Build the exact feature order used by OutcomeDataset, with live output norms."""
    exp_norm = torch.linalg.vector_norm(output.standardized_expression, dim=1, keepdim=True) / np.sqrt(
        output.standardized_expression.shape[1]
    )
    head_norm = torch.linalg.vector_norm(output.standardized_pose[:, :3], dim=1, keepdim=True)
    jaw_norm = torch.linalg.vector_norm(output.standardized_pose[:, 3:], dim=1, keepdim=True)
    return torch.cat(
        [
            batch["outcome_context"],
            output.alpha_expression,
            output.alpha_head_pose,
            output.alpha_jaw_pose,
            exp_norm,
            head_norm,
            jaw_norm,
        ],
        dim=1,
    )


def compute_loss(
    model: ConditionGenerator,
    batch: dict[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    training: bool = True,
    outcome_lookup: dict | None = None,
    outcome_weight: float = 0.0,
    outcome_supervision: dict | None = None,
):
    b = normalize_batch(batch, mean, std, device)
    out = model(b["features"], b["expression"], b["pose"], alpha_mode="fixed_one" if model.alpha_mode == "fixed_one" else "learned")
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
    alpha_loss = (
        weighted_mean((alphas - alpha_target(q)).pow(2).mean(dim=1, keepdim=True), w)
        if model.alpha_mode == "learned"
        else torch.zeros((), device=device)
    )
    confidence_loss = weighted_mean((out.confidence - q).pow(2), w)
    reject_loss = weighted_mean(
        nn.functional.binary_cross_entropy(out.reject_score, b["reject_target"].clamp(0.0, 1.0), reduction="none"), w
    )

    total = cond_loss + 0.03 * target_reg + 0.45 * alpha_loss + 0.35 * confidence_loss + 0.25 * reject_loss
    if training:
        # Smoothness regularization uses random feature noise; training only so
        # that validation stays deterministic and reproducible.
        noise = torch.randn_like(b["features"]) * 0.015
        out_noisy = model(
            b["features"] + noise,
            b["expression"],
            b["pose"],
            alpha_mode="fixed_one" if model.alpha_mode == "fixed_one" else "learned",
        )
        smooth_per_sample = (
            (out.standardized_expression - out_noisy.standardized_expression).pow(2).mean(dim=1, keepdim=True)
            + (out.standardized_pose - out_noisy.standardized_pose).pow(2).mean(dim=1, keepdim=True)
            + (alphas - torch.cat([out_noisy.alpha_expression, out_noisy.alpha_head_pose, out_noisy.alpha_jaw_pose], dim=1))
            .pow(2)
            .mean(dim=1, keepdim=True)
        )
        smooth_loss = weighted_mean(smooth_per_sample, w)
        total = total + 0.10 * smooth_loss
    else:
        smooth_loss = torch.zeros((), device=device)

    # Phase2.1 outcome supervision (off by default): push the model's quality
    # confidence toward (1 - unsafe) for samples that have a rendered outcome.
    # Exact loss form is a placeholder; the weight is a CLI-only knob.
    outcome_loss = torch.zeros((), device=device)
    if outcome_lookup is not None and outcome_weight > 0.0:
        ids = list(batch.get("image_id", ()))
        valid = [i for i, image_id in enumerate(ids) if outcome_lookup.get(image_id, {}).get("unsafe", "") in {"0", "1"}]
        if valid:
            unsafe = torch.tensor(
                [float(outcome_lookup[ids[i]]["unsafe"]) for i in valid], device=device
            ).unsqueeze(1)
            target_conf = 1.0 - unsafe
            outcome_loss = weighted_mean((out.confidence[valid] - target_conf).pow(2), w[valid])

    total = total + outcome_weight * outcome_loss

    surrogate_losses = {
        "outcome_identity": torch.zeros((), device=device),
        "outcome_pose": torch.zeros((), device=device),
        "outcome_gaze": torch.zeros((), device=device),
        "outcome_render_failure": torch.zeros((), device=device),
    }
    if outcome_supervision is not None:
        live_features = build_outcome_surrogate_features(b, out)
        normalized = (live_features - outcome_supervision["mean"]) / outcome_supervision["std"]
        predicted = outcome_supervision["model"](normalized)
        surrogate_losses["outcome_identity"] = weighted_mean(
            torch.relu(outcome_supervision["identity_floor"] - predicted["identity"].unsqueeze(1)).pow(2), w
        )
        surrogate_losses["outcome_pose"] = weighted_mean(
            torch.relu(outcome_supervision["pose_floor"] - predicted["pose"].unsqueeze(1)).pow(2), w
        )
        surrogate_losses["outcome_gaze"] = weighted_mean(
            torch.relu(predicted["gaze"].unsqueeze(1) - outcome_supervision["gaze_ceiling"]).pow(2), w
        )
        surrogate_losses["outcome_render_failure"] = weighted_mean(
            nn.functional.binary_cross_entropy_with_logits(
                predicted["render_failure"].unsqueeze(1),
                torch.zeros_like(out.confidence),
                reduction="none",
            ),
            w,
        )
        for name, loss_value in surrogate_losses.items():
            total = total + outcome_supervision["weights"][name] * loss_value

    metrics = {
        "loss": float(total.detach().cpu()),
        "cond": float(cond_loss.detach().cpu()),
        "target_reg": float(target_reg.detach().cpu()),
        "alpha": float(alpha_loss.detach().cpu()),
        "confidence": float(confidence_loss.detach().cpu()),
        "reject": float(reject_loss.detach().cpu()),
        "smooth": float(smooth_loss.detach().cpu()),
        "outcome": float(outcome_loss.detach().cpu()),
        **{name: float(value.detach().cpu()) for name, value in surrogate_losses.items()},
    }
    return total, metrics


def evaluate(model, loader, mean, std, device, outcome_lookup=None, outcome_weight=0.0, outcome_supervision=None):
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            _, metrics = compute_loss(
                model, batch, mean, std, device, training=False,
                outcome_lookup=outcome_lookup, outcome_weight=outcome_weight,
                outcome_supervision=outcome_supervision,
            )
            bs = int(batch["features"].shape[0])
            count += bs
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value * bs
    return {key: value / max(count, 1) for key, value in totals.items()}


def load_outcome_supervision(path: Path, device: torch.device, args: argparse.Namespace, val_ids: list[str]) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("feature_columns") != OUTCOME_FEATURE_COLUMNS:
        raise SystemExit("Outcome surrogate feature schema does not match the current trainer")
    source_ids = set(checkpoint.get("train_image_ids", ())) | set(checkpoint.get("val_image_ids", ()))
    overlap = source_ids & set(val_ids)
    if overlap and not args.allow_outcome_validation_overlap:
        raise SystemExit(
            f"Outcome surrogate source overlaps condition-generator validation by {len(overlap)} IDs; "
            "build surrogate outcomes from the condition-training partition"
        )
    surrogate = OutcomeSurrogate(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
    ).to(device)
    surrogate.load_state_dict(checkpoint["model_state"])
    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    weights = {
        "outcome_identity": args.outcome_identity_weight,
        "outcome_pose": args.outcome_pose_weight,
        "outcome_gaze": args.outcome_gaze_weight,
        "outcome_render_failure": args.outcome_render_failure_weight,
    }
    return {
        "model": surrogate,
        "mean": torch.as_tensor(checkpoint["feature_mean"], device=device),
        "std": torch.as_tensor(checkpoint["feature_std"], device=device),
        "weights": weights,
        "identity_floor": args.outcome_identity_floor,
        "pose_floor": args.outcome_pose_improvement_floor,
        "gaze_ceiling": args.outcome_gaze_ceiling_deg,
        "source_overlap_with_val": len(overlap),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    excluded_ids = read_id_file(args.exclude_ids_file)
    samples = load_samples(
        args.deca_results_dir,
        args.arcface_manifest,
        args.xgb_quality_manifest,
        args.quality_source,
        excluded_ids,
    )
    n = len(samples)
    # Fixed, deterministic split indices (identical across ablation groups).
    val_indices, train_indices = make_split(n, args.val_ratio, args.seed)
    if not train_indices:
        raise SystemExit("Empty training split")

    # normalizer from the un-augmented TRAINING subset only (never validation).
    norm_dataset = Phase2Dataset([samples[i] for i in train_indices], augment=False, seed=args.seed, stage=args.stage)
    mean_np, std_np = normalizer_from_dataset(norm_dataset)

    train_dataset = Phase2Dataset([samples[i] for i in train_indices], augment=not args.no_augment, seed=args.seed, stage=args.stage)
    val_dataset = Phase2Dataset([samples[i] for i in val_indices], augment=False, seed=args.seed, stage=args.stage)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) if val_dataset else None
    device = resolve_device(args.device)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    input_dim = int(mean_np.shape[0])
    model = ConditionGenerator(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    model.alpha_mode = args.alpha_mode
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_command_artifacts(args.out_dir, args)
    train_ids = [samples[i].image_id for i in train_indices]
    val_ids = [samples[i].image_id for i in val_indices]
    (args.out_dir / "train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (args.out_dir / "val_ids.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")
    save_normalizer(args.out_dir / "normalizer.npz", mean_np, std_np)

    # Phase2.1 outcome supervision (off by default)
    outcome_weight = args.outcome_loss_weight
    if args.no_outcome_supervision:
        outcome_weight = 0.0
    outcome_lookup: dict | None = None
    if args.outcome_manifest and outcome_weight > 0.0:
        import csv as _csv

        with args.outcome_manifest.open("r", encoding="utf-8-sig", newline="") as _f:
            outcome_lookup = {row["image_id"]: row for row in _csv.DictReader(_f)}
        label_ids = set(outcome_lookup)
        label_val_overlap = label_ids & set(val_ids)
        if label_val_overlap and not args.allow_outcome_validation_overlap:
            raise SystemExit(
                f"Outcome labels overlap condition-generator validation by {len(label_val_overlap)} IDs; "
                "build supervision labels from the condition-training partition"
            )
        if not (label_ids & set(train_ids)):
            raise SystemExit("Outcome manifest has no labels for the condition-generator training partition")
    outcome_supervision = None
    surrogate_weights = [
        args.outcome_identity_weight,
        args.outcome_pose_weight,
        args.outcome_gaze_weight,
        args.outcome_render_failure_weight,
    ]
    if not args.no_outcome_supervision and any(weight > 0.0 for weight in surrogate_weights):
        if args.outcome_surrogate is None:
            raise SystemExit("Positive outcome head weight requires --outcome-surrogate")
        outcome_supervision = load_outcome_supervision(args.outcome_surrogate, device, args, val_ids)
    history_path = args.out_dir / "train_history.csv"
    best_val = float("inf")
    history_fields = [
        "epoch", "split", "loss", "cond", "target_reg", "alpha", "confidence", "reject", "smooth", "outcome",
        "outcome_identity", "outcome_pose", "outcome_gaze", "outcome_render_failure",
    ]
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            totals: dict[str, float] = {}
            seen = 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = compute_loss(
                    model, batch, mean, std, device, training=True,
                    outcome_lookup=outcome_lookup, outcome_weight=outcome_weight,
                    outcome_supervision=outcome_supervision,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                bs = int(batch["features"].shape[0])
                seen += bs
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.0) + value * bs
            train_metrics = {key: value / max(seen, 1) for key, value in totals.items()}
            writer.writerow({"epoch": epoch, "split": "train", **train_metrics})

            val_metrics = evaluate(
                model, val_loader, mean, std, device,
                outcome_lookup=outcome_lookup, outcome_weight=outcome_weight,
                outcome_supervision=outcome_supervision,
            ) if val_loader is not None else train_metrics
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
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "input_dim": input_dim,
        "best_val_loss": best_val,
        "checkpoint": str(args.out_dir / "best_model.pt"),
        "normalizer": str(args.out_dir / "normalizer.npz"),
        "history": str(history_path),
        "excluded_ids_file": str(args.exclude_ids_file) if args.exclude_ids_file else None,
        "excluded_ids": len(excluded_ids),
        "train_ids_file": str(args.out_dir / "train_ids.txt"),
        "val_ids_file": str(args.out_dir / "val_ids.txt"),
        "outcome_surrogate": str(args.outcome_surrogate) if args.outcome_surrogate else None,
        "outcome_supervision_active": outcome_supervision is not None,
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
