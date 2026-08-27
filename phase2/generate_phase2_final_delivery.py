"""Generate final Phase2 figures, metric summary, and critical-artifact hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_LABELS = {
    "hard_zero": "Hard-zero",
    "full": "Full",
    "no_alpha": "No alpha",
    "no_augmentation": "No augmentation",
    "no_xgboost": "No XGBoost",
}
COLORS = ["#176B87", "#D95F59", "#4C956C", "#E3A008", "#725AC1", "#667085"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text_lf(path: Path, text: str) -> None:
    """Write deterministic UTF-8 text without platform newline translation."""
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def style_axes(axis, title: str, ylabel: str = "") -> None:
    axis.set_title(title, fontsize=11, fontweight="bold", pad=9)
    axis.set_ylabel(ylabel)
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#D0D5DD", linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def save_figure(fig, path: Path) -> None:
    fig.patch.set_facecolor("white")
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def errorbar_panel(axis, frame: pd.DataFrame, metric: str, lo: str, hi: str, title: str, ylabel: str) -> None:
    values = frame[metric].to_numpy(dtype=float)
    lower = values - frame[lo].to_numpy(dtype=float)
    upper = frame[hi].to_numpy(dtype=float) - values
    x = np.arange(len(frame))
    axis.bar(x, values, color=COLORS[: len(frame)], width=0.68)
    axis.errorbar(x, values, yerr=np.vstack([lower, upper]), fmt="none", ecolor="#101828", capsize=3, linewidth=1)
    axis.set_xticks(x, frame["label"], rotation=22, ha="right")
    style_axes(axis, title, ylabel)


def build_figures(root: Path, out_dir: Path) -> dict[str, str]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ablation_root = root / "results/phase2_ablation_20260825"
    rescue_root = root / "results/phase2_rescue_only_20260826"
    sensitivity_root = root / "results/phase2_rescue_sensitivity_20260826"
    gate_root = root / "results/phase2_gate_calibration_20260826"

    methods = pd.read_csv(ablation_root / "metrics_all/metrics_by_method.csv")
    methods = methods[methods.method.isin(METHOD_LABELS)].copy()
    methods["order"] = methods.method.map({name: index for index, name in enumerate(METHOD_LABELS)})
    methods = methods.sort_values("order")
    methods["label"] = methods.method.map(METHOD_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    errorbar_panel(axes[0], methods, "arcface_cosine_mean", "arcface_cosine_ci95_lo", "arcface_cosine_ci95_hi", "Identity diagnostic", "ArcFace cosine")
    errorbar_panel(axes[1], methods, "deca_pose_norm_mean", "deca_pose_norm_ci95_lo", "deca_pose_norm_ci95_hi", "Canonical head pose", "DECA pose norm (lower)")
    errorbar_panel(axes[2], methods, "l2cs_gaze_delta_vs_original_deg_mean", "l2cs_gaze_delta_vs_original_deg_ci95_lo", "l2cs_gaze_delta_vs_original_deg_ci95_hi", "Gaze change", "L2CS angle (degrees, lower)")
    fig.suptitle("Fixed-test ablation on 743 available samples", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    p1 = figures / "fig1_fixed_test_ablation.png"
    save_figure(fig, p1)

    decisions = pd.read_csv(ablation_root / "metrics_all/decision_aware_summary.csv")
    decisions = decisions[decisions.method.isin(METHOD_LABELS)].copy()
    decisions["order"] = decisions.method.map({name: index for index, name in enumerate(METHOD_LABELS)})
    decisions = decisions.sort_values("order")
    decisions["label"] = decisions.method.map(METHOD_LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    x = np.arange(len(decisions))
    axes[0].bar(x, decisions.end_to_end_accept_rate * 100, color=COLORS[: len(decisions)])
    axes[0].axhline(743 / 775 * 100, color="#101828", linestyle="--", linewidth=1, label="Upstream ceiling")
    axes[0].set_xticks(x, decisions.label, rotation=22, ha="right")
    axes[0].set_ylim(0, 103)
    axes[0].legend(frameon=False, fontsize=8)
    style_axes(axes[0], "End-to-end accepted coverage", "Percent of 775")
    axes[1].bar(x, decisions.standardize, color="#4C956C", label="Standardize")
    axes[1].bar(x, decisions.weak_standardize, bottom=decisions.standardize, color="#E3A008", label="Weak")
    axes[1].bar(x, decisions.reject, bottom=decisions.standardize + decisions.weak_standardize, color="#D95F59", label="Reject")
    axes[1].set_xticks(x, decisions.label, rotation=22, ha="right")
    axes[1].legend(frameon=False, fontsize=8, ncol=3)
    style_axes(axes[1], "Decisions among 743 available samples", "Samples")
    fig.tight_layout()
    p2 = figures / "fig2_decision_coverage.png"
    save_figure(fig, p2)

    shift = pd.read_csv(sensitivity_root / "main_vs_rescue_summary.csv")
    groups = ["aflw_large_pose", "cofw_occlusion", "wider_blur", "wider_occlusion", "wider_pose"]
    labels = ["AFLW pose", "COFW occl.", "WIDER blur", "WIDER occl.", "WIDER pose"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for axis, metric, title, ylabel in [
        (axes[0], "arcface_cosine_main_vs_rescue", "Main vs rescue identity", "ArcFace cosine (higher)"),
        (axes[1], "l2cs_gaze_delta_main_vs_rescue_deg", "Main vs rescue gaze shift", "Angle (degrees, lower)"),
    ]:
        selected = shift[(shift.source_group.isin(groups)) & (shift.metric == metric)].set_index("source_group").loc[groups]
        values = selected["mean"].to_numpy(float)
        errors = np.vstack([values - selected.ci95_lo.to_numpy(float), selected.ci95_hi.to_numpy(float) - values])
        axis.bar(np.arange(len(groups)), values, color=COLORS[: len(groups)])
        axis.errorbar(np.arange(len(groups)), values, yerr=errors, fmt="none", ecolor="#101828", capsize=3)
        axis.set_xticks(np.arange(len(groups)), labels, rotation=20, ha="right")
        style_axes(axis, title, ylabel)
    fig.suptitle("Whole-image rescue introduces a measurable domain shift", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    p3 = figures / "fig3_rescue_domain_shift.png"
    save_figure(fig, p3)

    rescue = pd.read_csv(rescue_root / "metrics_with_expression/metrics_by_method.csv")
    rescue = rescue[rescue.method.isin(METHOD_LABELS)].copy()
    rescue["order"] = rescue.method.map({name: index for index, name in enumerate(METHOD_LABELS)})
    rescue = rescue.sort_values("order")
    rescue["label"] = rescue.method.map(METHOD_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    errorbar_panel(axes[0], rescue, "arcface_cosine_mean", "arcface_cosine_ci95_lo", "arcface_cosine_ci95_hi", "Identity diagnostic", "ArcFace cosine")
    errorbar_panel(axes[1], rescue, "deca_pose_norm_mean", "deca_pose_norm_ci95_lo", "deca_pose_norm_ci95_hi", "Canonical head pose", "DECA pose norm")
    errorbar_panel(axes[2], rescue, "l2cs_gaze_delta_vs_original_deg_mean", "l2cs_gaze_delta_vs_original_deg_ci95_lo", "l2cs_gaze_delta_vs_original_deg_ci95_hi", "Gaze change", "Degrees")
    fig.suptitle("Rescue-only sensitivity on 32 FAN failures", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    p4 = figures / "fig4_rescue_only_sensitivity.png"
    save_figure(fig, p4)

    comparison = pd.read_csv(gate_root / "calibrator/gate_model_comparison.csv")
    risk_curve = pd.read_csv(gate_root / "calibrator/gate_risk_coverage.csv")
    val_conf = pd.read_csv(gate_root / "calibrator/gate_confusion_by_quality.csv")
    fixed_conf = pd.read_csv(gate_root / "fixed_test_audit/gate_fixed_test_confusion.csv")
    val_overall = val_conf[val_conf.xgb_quality_label == "overall"].iloc[0]
    fixed_overall = fixed_conf[(fixed_conf.stratify_by == "xgb_quality_label") & (fixed_conf.stratum == "overall")].iloc[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    x = np.arange(len(comparison))
    width = 0.36
    axes[0].bar(x - width / 2, comparison.auroc, width, color="#176B87", label="AUROC")
    axes[0].bar(x + width / 2, comparison.auprc, width, color="#D95F59", label="AUPRC")
    axes[0].set_xticks(x, ["Reject", "Mean risk", "Logistic"], rotation=15, ha="right")
    axes[0].set_ylim(0, 0.72)
    axes[0].legend(frameon=False)
    style_axes(axes[0], "Validation discrimination", "Score")
    curve = risk_curve.sort_values("coverage")
    axes[1].plot(curve.coverage * 100, curve.selective_risk * 100, color="#176B87", linewidth=2)
    axes[1].axhline(10, color="#D95F59", linestyle="--", linewidth=1)
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(0, max(20, float(curve.selective_risk.max() * 110)))
    style_axes(axes[1], "Validation risk-coverage", "Accepted unsafe rate (%)")
    axes[1].set_xlabel("Coverage (%)")
    labels2 = ["Validation", "Fixed test"]
    selective = [val_overall.selective_risk * 100, fixed_overall.selective_risk * 100]
    coverage = [val_overall.coverage * 100, fixed_overall.coverage * 100]
    x2 = np.arange(2)
    axes[2].bar(x2 - width / 2, selective, width, color="#D95F59", label="Unsafe rate")
    axes[2].bar(x2 + width / 2, coverage, width, color="#4C956C", label="Coverage")
    axes[2].set_xticks(x2, labels2)
    axes[2].set_ylim(0, 82)
    axes[2].legend(frameon=False)
    style_axes(axes[2], "Frozen 10% operating point", "Percent")
    fig.suptitle("Gate calibration does not transfer to the difficult fixed test", fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    p5 = figures / "fig5_gate_calibration_and_transfer.png"
    save_figure(fig, p5)

    names = ["Base corpus", "XGB/Phase2 pool", "Train", "Validation", "Fixed test", "Available", "FAN failure", "Valid rescue"]
    counts = [10000, 9600, 8160, 1440, 775, 743, 32, 0]
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    y = np.arange(len(names))
    bars = axis.barh(y, counts, color=COLORS + ["#98A2B3", "#344054"])
    axis.set_yticks(y, names)
    axis.invert_yaxis()
    for bar, count in zip(bars, counts):
        axis.text(max(bar.get_width(), 0) + 100, bar.get_y() + bar.get_height() / 2, f"{count:,}", va="center", fontsize=9)
    axis.set_xlim(0, 10800)
    style_axes(axis, "Phase2 sample accounting", "Samples")
    axis.set_xlabel("Count; categories are protocol stages, not one funnel denominator")
    fig.tight_layout()
    p6 = figures / "fig6_phase2_sample_accounting.png"
    save_figure(fig, p6)

    return {path.stem: str(path) for path in (p1, p2, p3, p4, p5, p6)}


def collect_hash_targets(root: Path, out_dir: Path) -> list[tuple[Path, str]]:
    targets: list[tuple[Path, str]] = []

    def add(path: Path, role: str) -> None:
        if path.is_file():
            targets.append((path, role))

    for method in ("full", "no_alpha", "no_augmentation", "no_xgboost"):
        base = root / "results/phase2_ablation_20260825" / method
        for name, role in [
            ("best_model.pt", "checkpoint"), ("normalizer.npz", "normalizer"),
            ("train_summary.json", "training_summary"), ("train_history.csv", "training_history"),
            ("train_ids.txt", "split_ids"), ("val_ids.txt", "split_ids"),
            ("config.json", "training_config"), ("exact_command.txt", "exact_command"),
            ("phase2_inference_manifest.csv", "inference_manifest"),
        ]:
            add(base / name, f"{method}:{role}")
    critical = [
        ("results/phase2_xgb_rebuilt_20260824/xgb_final_model.json", "xgboost_model"),
        ("results/phase2_xgb_rebuilt_20260824/xgb_oof_manifest.csv", "xgboost_oof"),
        ("results/phase2_xgb_rebuilt_20260824/xgb_fixed_test_predictions.csv", "xgboost_fixed_predictions"),
        ("results/phase2_xgb_rebuilt_20260824/xgb_rebuild_summary.json", "xgboost_summary"),
        ("results/phase2_eval_fixed_20260824_v2/fixed_test_manifest_v2.csv", "fixed_test_manifest"),
        ("results/phase2_eval_fixed_20260824_v2/base_test_ids.txt", "test_exclusions"),
        ("results/phase2_eval_fixed_20260824_v2/arcface_fixed_test_manifest.csv", "arcface_manifest"),
        ("results/phase2_ablation_20260825/renders_all/render_manifest_all.csv", "render_manifest"),
        ("results/phase2_ablation_20260825/metrics_all/rendered_metrics.csv", "fixed_metrics"),
        ("results/phase2_ablation_20260825/metrics_all/paired_method_comparisons.csv", "paired_ablation"),
        ("results/phase2_ablation_20260825/metrics_all/decision_aware_summary.csv", "decision_summary"),
        ("results/phase2_rescue_sensitivity_20260826/main_vs_rescue_paired_metrics.csv", "rescue_paired_metrics"),
        ("results/phase2_rescue_sensitivity_20260826/main_vs_rescue_summary.csv", "rescue_shift_summary"),
        ("results/phase2_rescue_only_20260826/metrics_with_expression/rendered_metrics.csv", "rescue_only_metrics"),
        ("results/phase2_rescue_only_20260826/policy/rescue_policy_summary.csv", "rescue_policy"),
        ("results/phase2_gate_calibration_20260826/calibrator/gate_outcome_definition.json", "gate_definition"),
        ("results/phase2_gate_calibration_20260826/calibrator/gate_calibrator.json", "gate_calibrator"),
        ("results/phase2_gate_calibration_20260826/calibrator/gate_validation_outcomes.csv", "gate_validation_outcomes"),
        ("results/phase2_gate_calibration_20260826/calibrator/gate_risk_coverage.csv", "gate_risk_coverage"),
        ("results/phase2_gate_calibration_20260826/fixed_test_audit/gate_fixed_test_predictions.csv", "gate_fixed_predictions"),
        ("results/phase2_gate_calibration_20260826/fixed_test_audit/gate_fixed_test_confusion.csv", "gate_fixed_confusion"),
    ]
    for relative, role in critical:
        add(root / relative, role)
    for path in sorted((out_dir / "figures").glob("*.png")):
        add(path, "final_figure")
    add(out_dir / "final_metrics_summary.json", "final_summary")
    add(out_dir / "PHASE2_FINAL_REPORT_20260827.md", "final_report")
    add(out_dir / "exact_command.txt", "exact_command")
    deduped = {}
    for path, role in targets:
        deduped[path.resolve()] = role
    return [(path, deduped[path]) for path in sorted(deduped, key=str)]


def build_summary(root: Path) -> dict:
    ablation = pd.read_csv(root / "results/phase2_ablation_20260825/metrics_all/metrics_by_method.csv").set_index("method")
    paired = pd.read_csv(root / "results/phase2_ablation_20260825/metrics_all/paired_method_comparisons.csv")
    decisions = pd.read_csv(root / "results/phase2_ablation_20260825/metrics_all/decision_aware_summary.csv").set_index("method")
    shift = pd.read_csv(root / "results/phase2_rescue_sensitivity_20260826/main_vs_rescue_summary.csv")
    policy = pd.read_csv(root / "results/phase2_rescue_only_20260826/policy/rescue_policy_summary.csv").set_index("policy")
    gate = read_json(root / "results/phase2_gate_calibration_20260826/calibrator/gate_calibration_summary.json")
    fixed = read_json(root / "results/phase2_gate_calibration_20260826/fixed_test_audit/gate_fixed_test_summary.json")

    def shift_value(metric: str, field: str = "mean") -> float:
        return float(shift[(shift.source_group == "all") & (shift.metric == metric)].iloc[0][field])

    def paired_value(metric: str, field: str) -> float:
        row = paired[(paired.reference == "hard_zero") & (paired.method == "full") & (paired.metric == metric)].iloc[0]
        return float(row[field])

    return {
        "protocol": {
            "base_corpus": 10000, "phase2_pool": 9600, "train": 8160, "validation": 1440,
            "fixed_test": 775, "fixed_available": 743, "upstream_failure": 32,
            "four_model_training_complete": True, "test_leakage_overlap": 0,
        },
        "full_vs_hard_zero": {
            "arcface_cosine_full": float(ablation.loc["full", "arcface_cosine_mean"]),
            "arcface_cosine_hard_zero": float(ablation.loc["hard_zero", "arcface_cosine_mean"]),
            "paired_arcface_delta": paired_value("arcface_cosine", "mean_delta_method_minus_reference"),
            "paired_arcface_ci95": [paired_value("arcface_cosine", "ci95_lo"), paired_value("arcface_cosine", "ci95_hi")],
            "paired_pose_norm_delta": paired_value("deca_head_pose_norm", "mean_delta_method_minus_reference"),
            "paired_gaze_delta_deg": paired_value("l2cs_gaze_delta_vs_original_deg", "mean_delta_method_minus_reference"),
            "end_to_end_accept_rate": float(decisions.loc["full", "end_to_end_accept_rate"]),
        },
        "rescue": {
            "paired_cases": 343,
            "main_rescue_arcface_cosine_mean": shift_value("arcface_cosine_main_vs_rescue"),
            "main_rescue_gaze_delta_mean_deg": shift_value("l2cs_gaze_delta_main_vs_rescue_deg"),
            "main_rescue_gaze_delta_p95_deg": shift_value("l2cs_gaze_delta_main_vs_rescue_deg", "p95"),
            "technical_recovery": int(policy.loc["primary_plus_rescue", "technical_rescue_recovered"]),
            "valid_rescue": int(policy.loc["primary_plus_rescue", "valid_rescue"]),
            "fallback_coverage": float(policy.loc["primary_plus_rescue", "coverage"]),
        },
        "gate": {
            "validation_n": gate["n"], "validation_unsafe": gate["unsafe"],
            "selected_model": gate["selected_model"],
            "validation_models": gate["model_comparison"],
            "validation_operating_points": gate["operating_points"],
            "fixed_available": fixed["available"], "fixed_unsafe": fixed["outcome_counts"]["unsafe"],
            "fixed_selective_risk": fixed["available_confusion"]["selective_risk"],
            "fixed_coverage": fixed["available_confusion"]["coverage"],
            "fixed_far": fixed["available_confusion"]["far"],
            "fixed_frr": fixed["available_confusion"]["frr"],
            "deployment_qualified": False,
        },
        "scope_note": "ArcFace, DECA re-encoding, and L2CS values are diagnostic metrics on DECA shape-detail renders, not human perceptual ground truth.",
    }


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = build_figures(root, out_dir)
    summary = build_summary(root)
    summary["figures"] = figures
    summary_path = out_dir / "final_metrics_summary.json"
    write_text_lf(summary_path, json.dumps(summary, indent=2))
    command = " ".join([sys.executable, "-m", "phase2.generate_phase2_final_delivery", *sys.argv[1:]])
    write_text_lf(out_dir / "exact_command.txt", command + "\n")

    rows = []
    for path, role in collect_hash_targets(root, out_dir):
        rows.append({
            "sha256": sha256(path), "bytes": path.stat().st_size,
            "role": role, "path": str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        })
    manifest = out_dir / "artifact_sha256.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sha256", "bytes", "role", "path"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    verification = []
    for row in rows:
        path = Path(row["path"])
        if not path.is_absolute():
            path = root / path
        verification.append(sha256(path) == row["sha256"])
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:  # noqa: BLE001
        git_commit = "unavailable"
    hash_summary = {
        "algorithm": "SHA-256", "entries": len(rows), "verified": sum(verification),
        "all_verified": all(verification), "git_commit": git_commit,
        "scope": "Critical models, manifests, summaries, metrics, configs, split IDs, final report, exact command, and final figures.",
        "excluded": "Raw datasets, model dependencies, per-image renders, caches, and virtual environments.",
    }
    write_text_lf(out_dir / "artifact_hash_summary.json", json.dumps(hash_summary, indent=2))
    print(json.dumps({"figures": len(figures), **hash_summary}, indent=2))


if __name__ == "__main__":
    main()
