"""Export Phase3.1 img2img qualitative sheets and paper-ready metric figures."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt


CONTACT_NAMES = {
    "s00": "figure_01_qualitative_strength_000.png",
    "s01": "figure_02_qualitative_strength_025.png",
    "s02": "figure_03_qualitative_strength_050.png",
    "s03": "figure_04_qualitative_strength_075.png",
    "s04": "figure_05_qualitative_strength_100.png",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.out_dir}")
    audit_path = args.run_dir / "identity_audit" / "summary.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True)

    for key, destination in CONTACT_NAMES.items():
        source = args.run_dir / f"contact_{key}.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copyfile(source, args.out_dir / destination)

    rows = []
    for group in audit["groups"]:
        metrics = group["metrics"]
        rows.append({
            "strength": group["strength"],
            "variant": group["variant"],
            "n_total": group["n_total"],
            "arcface_count": metrics["source_cosine"]["count"],
            "arcface_mean": metrics["source_cosine"]["mean"],
            "arcface_median": metrics["source_cosine"]["median"],
            "arcface_p10": metrics["source_cosine"]["p10"],
            "face_detection_coverage": metrics["source_cosine"]["count"] / group["n_total"],
            "no_face": group["no_face"],
            "multiple_faces": group["multiple_faces"],
            "source_mae": metrics["source_mae"]["mean"],
            "source_rmse": metrics["source_rmse"]["mean"],
        })
    rows.sort(key=lambda row: (row["variant"], row["strength"]))
    with (args.out_dir / "figure_data.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    colors = {"frozen": "#0072B2", "trained": "#D55E00"}
    labels = {"frozen": "Frozen UNet", "trained": "64-step adapter"}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), constrained_layout=True)
    for variant in ("frozen", "trained"):
        selected = [row for row in rows if row["variant"] == variant]
        x = [row["strength"] for row in selected]
        axes[0].plot(x, [row["arcface_mean"] for row in selected], marker="o", linewidth=1.8,
                     color=colors[variant], label=f"{labels[variant]} mean")
        axes[0].plot(x, [row["arcface_p10"] for row in selected], marker=".", linewidth=1.1,
                     linestyle="--", color=colors[variant], alpha=0.75, label=f"{labels[variant]} P10")
        axes[1].plot(x, [100 * row["face_detection_coverage"] for row in selected], marker="o",
                     linewidth=1.8, color=colors[variant], label=labels[variant])
    axes[0].axhline(audit["source_vae_cosine"]["mean"], color="#555555", linewidth=1,
                    linestyle=":", label="VAE anchor mean")
    axes[0].set(title="Identity retention", xlabel="Img2img strength", ylabel="Source ArcFace cosine",
                xlim=(-0.02, 1.02), ylim=(-0.1, 1.02))
    axes[1].set(title="Face-detection coverage", xlabel="Img2img strength", ylabel="Coverage (%)",
                xlim=(-0.02, 1.02), ylim=(-3, 103))
    for axis in axes:
        axis.grid(True, linewidth=0.5, alpha=0.25)
        axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"figure_06_identity_and_coverage.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.05), constrained_layout=True)
    for variant in ("frozen", "trained"):
        selected = [row for row in rows if row["variant"] == variant]
        x = [row["strength"] for row in selected]
        axes[0].plot(x, [row["source_mae"] for row in selected], marker="o", linewidth=1.8,
                     color=colors[variant], label=labels[variant])
        axes[1].plot(x, [row["source_rmse"] for row in selected], marker="o", linewidth=1.8,
                     color=colors[variant], label=labels[variant])
    axes[0].set(title="Mean absolute pixel drift", xlabel="Img2img strength", ylabel="RGB MAE [0, 1]",
                xlim=(-0.02, 1.02))
    axes[1].set(title="Root-mean-square pixel drift", xlabel="Img2img strength", ylabel="RGB RMSE [0, 1]",
                xlim=(-0.02, 1.02))
    for axis in axes:
        axis.grid(True, linewidth=0.5, alpha=0.25)
        axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out_dir / f"figure_07_pixel_drift.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)

    readme = """# Phase3.1 Img2Img Paper Visualizations

Exported on 2026-09-04 from the train-only 32-image Phase3.1 reconstruction smoke run.

## Files

- `figure_01` through `figure_05`: identical four-ID qualitative comparisons at strengths 0, 0.25, 0.5, 0.75, and 1.0. Columns are source, VAE anchor, frozen UNet, and the 64-step adapter.
- `figure_06_identity_and_coverage`: source ArcFace identity retention (mean and P10) and face-detection coverage. The horizontal reference is the mean source-to-VAE cosine.
- `figure_07_pixel_drift`: RGB MAE and RMSE relative to the source image.
- `figure_data.csv`: exact aggregate values used for Figures 6 and 7.

Figures 6 and 7 are supplied as 300 DPI PNG and vector PDF. The qualitative sheets remain lossless PNG so that labels and image evidence are preserved exactly.

## Interpretation Boundary

These are training-subset diagnostics from 32 smoke-test IDs. They support comparison of source reconstruction behavior under fixed noise, but do not establish held-out generalization, 3D control, standardization quality, or gaze disentanglement. Strength 0 is a VAE-only anchor and makes no UNet call. Strength 1 starts from a heavily noised source latent rather than pure noise.

The complete protocol and numeric audit are documented in `../PHASE31_LATENT_IMG2IMG_BASELINE_20260903.md`. Raw run outputs and model checkpoints remain excluded from Git.
"""
    (args.out_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
