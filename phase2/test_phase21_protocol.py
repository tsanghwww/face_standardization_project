"""Phase2.1 protocol tests (CPU, synthetic smoke). Verifies:
  1. gate splits are reproducible and disjoint;
  2. train/calibration/fixed-test have zero image_id overlap;
  3. surrogate/gate normalizer comes from the training subset only;
  4. fixed-test application has no threshold search (frozen gate required);
  5. missing outcome values are preserved (not filled with 0);
  6. rescue never enters primary inference;
  7. disabling outcome supervision keeps Phase2 v1 behavior;
  8. outcome labels cannot alter Gate decisions and rescue stays separate;
  9. frozen surrogate losses propagate to the condition generator;
 10. surrogate sources cannot overlap condition-generator validation.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(r"D:\face_standardization_project")
PY = sys.executable
TMP = PROJECT / ".runtime_tmp" / "phase21_protocol_test"
SEED = 20260827


def _run(module: str, args: list[str]) -> None:
    subprocess.run([PY, "-m", f"phase2.{module}", *args], check=True, cwd=PROJECT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _write_synthetic() -> None:
    TMP.mkdir(parents=True, exist_ok=True)
    ids = [f"s{i:03d}" for i in range(40)]
    with (TMP / "rendered_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["eval_id", "method", "render_status", "arcface_cosine", "deca_head_pose_norm", "deca_expression_norm", "l2cs_gaze_delta_vs_original_deg"])
        w.writeheader()
        for i in ids:
            for method, status, cos, pose, exp, gaze in [
                ("original", "success", "0.9", "0.8", "0.5", "0.0"),
                ("hard_zero", "success", "0.75", "0.05", "0.05", "0.0"),
                ("full", "success", "0.95", "0.12", "0.15", "3.0"),
            ]:
                # s005: missing full arcface_cosine -> explicit empty, not 0
                row = {"eval_id": i, "method": method, "render_status": status, "arcface_cosine": cos, "deca_head_pose_norm": pose, "deca_expression_norm": exp, "l2cs_gaze_delta_vs_original_deg": gaze}
                if i == "s005" and method == "full":
                    row["arcface_cosine"] = ""
                w.writerow(row)
    with (TMP / "inference_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "quality_score", "heuristic_quality_score", "xgb_quality_score", "reject_score", "confidence", "alpha_expression", "alpha_head_pose", "alpha_jaw_pose", "original_exp_norm", "original_head_pose_norm", "original_jaw_pose_norm", "standardized_exp_norm", "standardized_head_pose_norm", "standardized_jaw_pose_norm"])
        w.writeheader()
        for k, i in enumerate(ids):
            w.writerow({"image_id": i, "quality_score": "0.6", "heuristic_quality_score": "0.6", "xgb_quality_score": f"{0.5 + 0.01 * k:.4f}", "reject_score": "0.2", "confidence": "0.8", "alpha_expression": "0.5", "alpha_head_pose": "0.6", "alpha_jaw_pose": "0.4", "original_exp_norm": "0.3", "original_head_pose_norm": "0.7", "original_jaw_pose_norm": "0.2", "standardized_exp_norm": "0.1", "standardized_head_pose_norm": "0.1", "standardized_jaw_pose_norm": "0.05"})
    with (TMP / "xgb_oof.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "xgb_quality_label", "landmark_score", "landmark_out_ratio", "landmark_bbox_area", "landmark_center_dist", "arcface_status", "arcface_score"])
        w.writeheader()
        for i in ids:
            w.writerow({"image_id": i, "xgb_quality_label": "high" if int(i[1:]) % 3 == 0 else "medium", "landmark_score": "0.9", "landmark_out_ratio": "0.0", "landmark_bbox_area": "0.3", "landmark_center_dist": "0.1", "arcface_status": "1", "arcface_score": "0.8"})
    with (TMP / "fixed_ids.txt").open("w", encoding="utf-8") as f:
        f.write("fixed_001\nfixed_002\n")


def test_build_preserves_missing() -> None:
    _write_synthetic()
    out = TMP / "outcome"
    _run("build_outcome_supervision_manifest", ["--metrics-csv", str(TMP / "rendered_metrics.csv"), "--inference-manifest", str(TMP / "inference_manifest.csv"), "--xgb-oof-manifest", str(TMP / "xgb_oof.csv"), "--out-dir", str(out), "--split", "validation"])
    rows = {r["image_id"]: r for r in csv.DictReader(open(out / "outcome_manifest.csv", encoding="utf-8-sig"))}
    assert rows["s005"]["arcface_cosine"] == "", "missing metric was filled with 0"
    assert rows["s005"]["identity_delta_vs_hard_zero"] == ""
    assert rows["s005"]["unsafe"] == "1" and "missing_required_metric" in rows["s005"]["unsafe_reason"]
    assert rows["s000"]["arcface_cosine"] != ""
    print("[5] missing outcome preserved as explicit empty (s005) OK")


def test_split_reproducible_disjoint_no_leak() -> None:
    out = TMP / "outcome"
    for run in ("a", "b"):
        _run("make_gate_splits", ["--outcome-manifest", str(out / "outcome_manifest.csv"), "--exclude-ids-file", str(TMP / "fixed_ids.txt"), "--seed", str(SEED), "--train-ratio", "0.7", "--out-dir", str(TMP / f"split_{run}")])
    train_a = (TMP / "split_a" / "gate_train_ids.txt").read_text().splitlines()
    train_b = (TMP / "split_b" / "gate_train_ids.txt").read_text().splitlines()
    cal_a = (TMP / "split_a" / "hard_calibration_ids.txt").read_text().splitlines()
    assert train_a == train_b, "gate_train split is not reproducible"
    assert not (set(train_a) & set(cal_a)), "gate_train and hard_calibration overlap"
    fixed = set((TMP / "fixed_ids.txt").read_text().splitlines())
    assert not (set(train_a) & fixed) and not (set(cal_a) & fixed), "fixed test leaked into gate splits"
    print(f"[1][2] split reproducible + disjoint + no fixed leak OK (train={len(train_a)}, cal={len(cal_a)})")


def test_normalizer_train_only() -> None:
    from phase2.outcome_dataset import OutcomeDataset, fit_feature_normalizer, read_outcome_rows
    from phase2.train_condition_generator import make_split

    rows = read_outcome_rows(TMP / "outcome" / "outcome_manifest.csv")
    val_idx, train_idx = make_split(len(rows), 0.2, SEED)
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    mean, std = fit_feature_normalizer(train_rows)
    train_ds = OutcomeDataset(train_rows, mean, std)
    val_ds = OutcomeDataset(val_rows, mean, std)
    raw_val = OutcomeDataset(val_rows).features
    assert np.allclose(train_ds.features.mean(0), 0.0, atol=1e-5), "train normalizer not centered"
    assert np.allclose(val_ds.features, (raw_val - mean) / std), "validation did not reuse raw train statistics"
    print("[3] normalizer derived from training subset only OK")


def test_apply_no_threshold_search() -> None:
    # apply must reject a gate without a frozen threshold (no search)
    gate_no_thr = TMP / "gate_no_threshold.json"
    gate_no_thr.write_text(json.dumps({"features": [], "mean": [], "scale": [], "coefficient": [], "intercept": 0.0, "threshold": None}))
    try:
        _run("apply_phase21_gate", ["--outcome-manifest", str(TMP / "outcome" / "outcome_manifest.csv"), "--frozen-gate", str(gate_no_thr), "--out-dir", str(TMP / "apply")])
        raise AssertionError("apply_phase21_gate did not reject a threshold-less gate")
    except subprocess.CalledProcessError:
        pass
    print("[4] fixed-test apply has no threshold search (frozen threshold required) OK")


def test_rescue_isolation() -> None:
    # primary external mat_path must never reference the rescue dir
    manifest = PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv"
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    ext = [r for r in rows if r["source_group"] in ("wider_pose", "wider_occlusion", "wider_blur", "cofw_occlusion", "aflw_large_pose")]
    assert ext and all("rescue" not in (r["mat_path"] or "") for r in ext), "primary mat_path references rescue dir"
    assert (PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "deca_params_rescue").is_dir()
    print("[6] rescue isolation OK (primary mat_path never points to rescue)")


def test_gate_decision_ignores_evaluation_label_and_rescue_is_separate() -> None:
    from phase2.train_outcome_gate import GATE_FEATURES

    outcome = TMP / "outcome" / "outcome_manifest.csv"
    rows = list(csv.DictReader(outcome.open(encoding="utf-8-sig")))
    prediction = TMP / "prediction_manifest.csv"
    for row in rows:
        row["unsafe"] = ""
        row["unsafe_reason"] = ""
    with prediction.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    gate = {
        "mean": [0.0] * (2 * len(GATE_FEATURES)), "scale": [1.0] * (2 * len(GATE_FEATURES)),
        "coefficient": [0.0] * (2 * len(GATE_FEATURES)), "intercept": -2.0,
        "impute_values": [0.0] * len(GATE_FEATURES), "threshold": 0.5,
    }
    gate_path = TMP / "frozen_gate.json"
    gate_path.write_text(json.dumps(gate))
    rescue = TMP / "rescue.csv"
    with rescue.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eval_id", "image_id", "status", "mat_path"])
        writer.writeheader(); writer.writerow({"eval_id": "r1", "image_id": "r1", "status": "success", "mat_path": "deca_params_rescue/r1.mat"})
    out_dir = TMP / "apply_frozen"
    _run("apply_phase21_gate", [
        "--outcome-manifest", str(prediction), "--evaluation-outcome-manifest", str(outcome),
        "--frozen-gate", str(gate_path), "--rescue-audit", "--rescue-manifest", str(rescue),
        "--out-dir", str(out_dir),
    ])
    decisions = list(csv.DictReader((out_dir / "phase21_gate_decisions.csv").open()))
    assert decisions and all(row["gate_decision"] == "accept" for row in decisions)
    audit = list(csv.DictReader((out_dir / "phase21_rescue_audit.csv").open()))
    assert len(audit) == 1 and audit[0]["policy_decision"] == "manual_review_only"
    assert all("rescue" not in row.get("image_id", "") for row in decisions)
    print("[8] gate decisions ignore outcome labels; rescue audit stays separate OK")


def test_surrogate_loss_changes_generator_gradient() -> None:
    from phase2.model import ConditionGenerator
    from phase2.outcome_surrogate import OutcomeSurrogate
    from phase2.train_condition_generator import compute_loss

    torch.manual_seed(11)
    base = ConditionGenerator(input_dim=99, hidden_dim=32); base.alpha_mode = "learned"
    supervised = ConditionGenerator(input_dim=99, hidden_dim=32); supervised.alpha_mode = "learned"
    supervised.load_state_dict(base.state_dict())
    surrogate = OutcomeSurrogate(input_dim=12, hidden_dim=16)
    surrogate.eval()
    for parameter in surrogate.parameters(): parameter.requires_grad_(False)
    batch = {
        "image_id": ("a", "b"), "features": torch.randn(2, 99),
        "expression": torch.randn(2, 50), "pose": torch.randn(2, 6),
        "quality": torch.rand(2, 1), "reject_target": torch.rand(2, 1),
        "sample_weight": torch.ones(2, 1), "outcome_context": torch.rand(2, 6),
    }
    mean = torch.zeros(99); std = torch.ones(99)
    loss_base, _ = compute_loss(base, batch, mean, std, torch.device("cpu"), training=False)
    loss_base.backward()
    supervision = {
        "model": surrogate, "mean": torch.zeros(12), "std": torch.ones(12),
        "weights": {"outcome_identity": 1.0, "outcome_pose": 1.0, "outcome_gaze": 1.0, "outcome_render_failure": 0.0},
        "identity_floor": 10.0, "pose_floor": 10.0, "gaze_ceiling": -10.0,
    }
    loss_supervised, metrics = compute_loss(
        supervised, batch, mean, std, torch.device("cpu"), training=False, outcome_supervision=supervision
    )
    loss_supervised.backward()
    grad_base = torch.cat([p.grad.flatten() for p in base.parameters() if p.grad is not None])
    grad_supervised = torch.cat([p.grad.flatten() for p in supervised.parameters() if p.grad is not None])
    assert not torch.allclose(grad_base, grad_supervised), "surrogate loss did not reach generator outputs"
    assert metrics["outcome_identity"] > 0 and metrics["outcome_pose"] > 0
    print("[9] frozen surrogate outcome loss changes condition-generator gradient OK")


def test_surrogate_validation_overlap_is_blocked() -> None:
    from phase2.outcome_dataset import FEATURE_COLUMNS
    from phase2.outcome_surrogate import OutcomeSurrogate
    from phase2.train_condition_generator import load_outcome_supervision

    model = OutcomeSurrogate(input_dim=len(FEATURE_COLUMNS), hidden_dim=8)
    checkpoint = TMP / "overlap_surrogate.pt"
    torch.save({
        "model_state": model.state_dict(), "input_dim": len(FEATURE_COLUMNS), "hidden_dim": 8,
        "feature_columns": FEATURE_COLUMNS, "feature_mean": np.zeros(len(FEATURE_COLUMNS), np.float32),
        "feature_std": np.ones(len(FEATURE_COLUMNS), np.float32), "train_image_ids": ["val_001"],
        "val_image_ids": [],
    }, checkpoint)
    args = SimpleNamespace(
        allow_outcome_validation_overlap=False,
        outcome_identity_weight=1.0, outcome_pose_weight=0.0,
        outcome_gaze_weight=0.0, outcome_render_failure_weight=0.0,
        outcome_identity_floor=-0.02, outcome_pose_improvement_floor=0.0,
        outcome_gaze_ceiling_deg=10.0,
    )
    try:
        load_outcome_supervision(checkpoint, torch.device("cpu"), args, ["val_001"])
        raise AssertionError("validation-overlapping surrogate was accepted")
    except SystemExit:
        pass
    print("[10] surrogate source overlap with generator validation is blocked OK")


def test_v1_compat_outcome_off() -> None:
    from phase2.train_condition_generator import compute_loss
    from phase2.model import ConditionGenerator

    torch.manual_seed(0)
    model = ConditionGenerator(input_dim=99, hidden_dim=32)
    model.alpha_mode = "learned"
    batch = {
        "image_id": ("a", "b"),
        "features": torch.randn(2, 99),
        "expression": torch.randn(2, 50),
        "pose": torch.randn(2, 6),
        "quality": torch.rand(2, 1),
        "reject_target": torch.rand(2, 1),
        "sample_weight": torch.ones(2, 1),
    }
    mean = torch.zeros(99); std = torch.ones(99)
    _, metrics_off = compute_loss(model, batch, mean, std, torch.device("cpu"), training=True, outcome_lookup=None, outcome_weight=0.0)
    assert metrics_off["outcome"] == 0.0, "outcome supervision leaked into v1 (weight 0)"
    print("[7] outcome supervision off -> Phase2 v1 loss unchanged (outcome=0) OK")


def main() -> None:
    test_build_preserves_missing()
    test_split_reproducible_disjoint_no_leak()
    test_normalizer_train_only()
    test_apply_no_threshold_search()
    test_rescue_isolation()
    test_gate_decision_ignores_evaluation_label_and_rescue_is_separate()
    test_surrogate_loss_changes_generator_gradient()
    test_surrogate_validation_overlap_is_blocked()
    test_v1_compat_outcome_off()
    print("ALL PHASE2.1 PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    main()
