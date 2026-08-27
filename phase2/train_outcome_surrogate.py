"""Train the multi-head outcome surrogate (CPU-safe, small-batch smoke friendly).

Reports per-head loss, MAE (regression) / AUROC (classification), calibration
(ECE, classification head) and missing coverage.  Head loss weights are CLI
arguments only; defaults are 0.0 (head off).  Surrogate outputs are DECA-render
diagnostic surrogates, NOT real ArcFace/L2CS measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .outcome_dataset import FEATURE_COLUMNS, OutcomeDataset, fit_feature_normalizer, read_outcome_rows
from .outcome_surrogate import OutcomeSurrogate
from .train_condition_generator import make_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcome-manifest", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--device", default="cpu")
    p.add_argument("--identity-weight", type=float, default=0.0)
    p.add_argument("--pose-weight", type=float, default=0.0)
    p.add_argument("--gaze-weight", type=float, default=0.0)
    p.add_argument("--render-failure-weight", type=float, default=0.0)
    return p.parse_args()


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    if y.sum() == 0 or (1 - y).sum() == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and score[order[end]] == score[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos = int(y.sum())
    neg = len(y) - pos
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def ece(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = y.size
    val = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (score >= lo) & (score < hi if hi < 1 else score <= hi)
        if mask.any():
            val += mask.sum() / total * abs(float(y[mask].mean()) - float(score[mask].mean()))
    return val


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rows = read_outcome_rows(args.outcome_manifest)
    # dedup by image_id
    seen: set[str] = set()
    dedup = [r for r in rows if not (r["image_id"] in seen or seen.add(r["image_id"]))]
    val_idx, train_idx = make_split(len(dedup), args.val_ratio, args.seed)
    train_rows = [dedup[i] for i in train_idx]
    val_rows = [dedup[i] for i in val_idx]
    feature_mean, feature_std = fit_feature_normalizer(train_rows)
    train_ds = OutcomeDataset(train_rows, feature_mean, feature_std)
    val_ds = OutcomeDataset(val_rows, feature_mean, feature_std)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = OutcomeSurrogate(input_dim=train_ds.features.shape[1], hidden_dim=args.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    weights = {"identity": args.identity_weight, "pose": args.pose_weight, "gaze": args.gaze_weight, "render_failure": args.render_failure_weight}
    device = torch.device(args.device)
    model.to(device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            feats = batch["features"].to(device)
            out = model(feats)
            total = torch.zeros((), device=device)
            mse = torch.nn.functional.mse_loss
            if weights["identity"]:
                total = total + weights["identity"] * (mse(out["identity"], batch["identity"].to(device), reduction="none") * batch["mask_identity"].to(device)).sum() / batch["mask_identity"].sum().clamp_min(1)
            if weights["pose"]:
                total = total + weights["pose"] * (mse(out["pose"], batch["pose"].to(device), reduction="none") * batch["mask_pose"].to(device)).sum() / batch["mask_pose"].sum().clamp_min(1)
            if weights["gaze"]:
                total = total + weights["gaze"] * (mse(out["gaze"], batch["gaze"].to(device), reduction="none") * batch["mask_gaze"].to(device)).sum() / batch["mask_gaze"].sum().clamp_min(1)
            if weights["render_failure"]:
                total = total + weights["render_failure"] * (torch.nn.functional.binary_cross_entropy_with_logits(out["render_failure"], batch["render_failure"].to(device), reduction="none") * batch["mask_render"].to(device)).sum() / batch["mask_render"].sum().clamp_min(1)
            if total.item() != 0:
                opt.zero_grad(set_to_none=True)
                total.backward()
                opt.step()

    # evaluate on val
    model.eval()
    preds: dict[str, list[float]] = {k: [] for k in weights}
    trues: dict[str, list[float]] = {k: [] for k in weights}
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch["features"].to(device))
            for k in ("identity", "pose", "gaze"):
                preds[k].extend(out[k].cpu().numpy().tolist())
                trues[k].extend(batch[k].cpu().numpy().tolist())
            preds["render_failure"].extend(torch.sigmoid(out["render_failure"]).cpu().numpy().tolist())
            trues["render_failure"].extend(batch["render_failure"].cpu().numpy().tolist())
        mask_names = {"identity": "mask_identity", "pose": "mask_pose", "gaze": "mask_gaze", "render_failure": "mask_render"}
        masks = {k: val_ds.__getattribute__(mask_names[k]) for k in weights}

    metrics: dict[str, dict] = {}
    for k in ("identity", "pose", "gaze"):
        m = masks[k].astype(bool)
        y = np.asarray(trues[k])[m]
        p = np.asarray(preds[k])[m]
        metrics[k] = {"head": k, "type": "regression", "n": int(m.sum()), "coverage": float(m.mean()), "mae": float(np.mean(np.abs(p - y))) if y.size else float("nan")}
    m = masks["render_failure"].astype(bool)
    y = np.asarray(trues["render_failure"])[m]
    p = np.asarray(preds["render_failure"])[m]
    metrics["render_failure"] = {"head": "render_failure", "type": "classification", "n": int(m.sum()), "coverage": float(m.mean()), "auroc": auroc(y, p), "ece": ece(y, p) if y.sum() not in (0, y.size) else float("nan")}

    torch.save({
        "model_state": model.state_dict(),
        "input_dim": train_ds.features.shape[1],
        "hidden_dim": args.hidden_dim,
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "train_image_ids": [row["image_id"] for row in train_rows],
        "val_image_ids": [row["image_id"] for row in val_rows],
        "config": vars(args),
    }, args.out_dir / "outcome_surrogate.pt")
    summary = {"seed": args.seed, "epochs": args.epochs, "n_train": len(train_rows), "n_val": len(val_rows), "weights": weights, "heads": metrics, "scope_note": "diagnostic DECA-render surrogates; NOT real ArcFace/L2CS"}
    (args.out_dir / "outcome_surrogate_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.out_dir / "outcome_surrogate_heads.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["head", "type", "n", "coverage", "mae", "auroc", "ece"])
        w.writeheader()
        for k, d in metrics.items():
            w.writerow({"head": k, "type": d["type"], "n": d["n"], "coverage": round(d["coverage"], 6), "mae": d.get("mae", ""), "auroc": d.get("auroc", ""), "ece": d.get("ece", "")})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
