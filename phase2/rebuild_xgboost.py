"""Rebuild the Phase2 XGBoost quality model WITHOUT fixed-test leakage.

Modes:
  train   - exclude base_test_ids.txt from the 10K feature manifest, generate a
            5-fold stratified OOF prediction (leakage-free quality score for
            Phase2 training), then fit a final model on all remaining ~9,600
            samples.  Saves feature_names, fold, OOF score, final model, and
            model SHA256.
  predict - predict the fixed test (base 400 + external 375) with the final
            model.  Base and external samples use the EXACT same 10-feature
            order, computed through phase2.features.sample_from_mat.  Samples
            whose DECA mat is missing (FAN/DECA failure) are marked
            ``upstream_deca_failure`` and get no score (never a fake value).

Feature source (train): results/phase2_xgb_quality_bug003_fixed_arcface_ok/xgb_quality_manifest.csv
Labels: screening_label (Pass -> 1, Warn/Fail/Reject -> 0).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import xgboost as xgb

PROJECT = Path(r"D:\face_standardization_project")
FEATURE_COLUMNS = [
    "quality_score", "exp_norm", "head_pose_norm", "jaw_pose_norm", "landmark_score",
    "landmark_out_ratio", "landmark_bbox_area", "landmark_center_dist", "arcface_status", "arcface_score",
]
EXTERNAL_GROUPS = {"wider_pose", "wider_occlusion", "wider_blur", "cofw_occlusion", "aflw_large_pose"}
HIGH_THRESHOLD = 0.55
MEDIUM_THRESHOLD = 0.45

XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": ["logloss", "auc"],
    "eta": 0.045,
    "max_depth": 3,
    "min_child_weight": 4.0,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "lambda": 2.0,
    "alpha": 0.2,
    "tree_method": "hist",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=["train", "predict"])
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results" / "phase2_xgb_rebuilt_20260824")
    parser.add_argument("--feature-manifest", type=Path, default=PROJECT / "results" / "phase2_xgb_quality_bug003_fixed_arcface_ok" / "xgb_quality_manifest.csv")
    parser.add_argument("--exclude-ids-file", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "base_test_ids.txt")
    parser.add_argument("--fixed-manifest", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv")
    parser.add_argument("--base-deca-dir", type=Path, default=PROJECT / "DECA" / "results" / "archive_phase2_params")
    parser.add_argument("--external-deca-dir", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "deca_params")
    parser.add_argument("--base-arcface", type=Path, default=PROJECT / "results" / "arcface_p95_rebuilt" / "arcface_manifest.csv")
    parser.add_argument("--external-arcface", type=Path, default=PROJECT / "results" / "phase2_arcface_external_20260824" / "arcface_external_manifest.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--num-boost-round", type=int, default=220)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def label_to_binary(label: str) -> int | None:
    norm = label.strip().lower()
    if norm == "pass":
        return 1
    if norm in {"warn", "fail", "failed", "reject"}:
        return 0
    return None


def quality_label(score: float) -> str:
    if score >= HIGH_THRESHOLD:
        return "high"
    if score >= MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def class_balanced_weights(y: np.ndarray) -> np.ndarray:
    w = np.ones_like(y, dtype=np.float32)
    counts = {int(c): int((y == c).sum()) for c in np.unique(y)}
    total = float(y.size)
    for c, cnt in counts.items():
        w[y == c] = total / max(1.0, len(counts) * cnt)
    return w


def stratified_kfold(y: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for cls in sorted(np.unique(y).tolist()):
        idx = np.where(y == cls)[0].tolist()
        rng.shuffle(idx)
        for i, j in enumerate(idx):
            folds[i % k].append(j)
    for f in folds:
        rng.shuffle(f)
    return [np.asarray(f, dtype=int) for f in folds]


def stratified_inner(y_sub: np.ndarray, val_ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    tr_parts, va_parts = [], []
    for cls in sorted(np.unique(y_sub).tolist()):
        idx = np.where(y_sub == cls)[0]
        rng.shuffle(idx)
        vc = max(1, int(round(idx.size * val_ratio))) if idx.size > 1 else 0
        va_parts.append(idx[:vc])
        tr_parts.append(idx[vc:])
    tr = np.concatenate(tr_parts)
    va = np.concatenate(va_parts)
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va


def _float(row, key):
    try:
        v = float(row.get(key, ""))
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def run_train(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.feature_manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    excluded = {ln.strip() for ln in args.exclude_ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()}

    kept = []
    dropped_test = 0
    for r in rows:
        if r["image_id"] in excluded:
            dropped_test += 1
            continue
        b = label_to_binary(r.get("screening_label", ""))
        if b is None:
            continue
        kept.append((r["image_id"], b, r))

    image_ids = [k[0] for k in kept]
    y = np.asarray([k[1] for k in kept], dtype=np.int32)
    X = np.asarray([[float(r.get(c, 0.0)) for c in FEATURE_COLUMNS] for _, _, r in kept], dtype=np.float32)
    n = X.shape[0]
    w = class_balanced_weights(y)
    folds = stratified_kfold(y, args.folds, args.seed)

    oof_scores = np.zeros(n, dtype=np.float64)
    fold_of = np.zeros(n, dtype=np.int32)
    best_iters = []
    for k in range(args.folds):
        val_fold = folds[k]
        tr_idx = np.concatenate([folds[i] for i in range(args.folds) if i != k])
        tr_tr, tr_va = stratified_inner(y[tr_idx], 0.15, args.seed + k)
        dtrain = xgb.DMatrix(X[tr_idx[tr_tr]], label=y[tr_idx[tr_tr]], weight=w[tr_idx[tr_tr]], feature_names=FEATURE_COLUMNS)
        dval = xgb.DMatrix(X[tr_idx[tr_va]], label=y[tr_idx[tr_va]], weight=w[tr_idx[tr_va]], feature_names=FEATURE_COLUMNS)
        params = dict(XGB_PARAMS, seed=args.seed)
        booster = xgb.train(params, dtrain, num_boost_round=args.num_boost_round, evals=[(dval, "val")], early_stopping_rounds=args.early_stopping_rounds, verbose_eval=False)
        dhold = xgb.DMatrix(X[val_fold], feature_names=FEATURE_COLUMNS)
        oof_scores[val_fold] = booster.predict(dhold, iteration_range=(0, booster.best_iteration + 1))
        fold_of[val_fold] = k
        best_iters.append(int(booster.best_iteration))

    with (args.out_dir / "xgb_oof_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_id",
                "fold",
                "oof_score",
                "oof_label",
                "xgb_quality_score",
                "xgb_quality_label",
                "quality_source",
            ]
            + FEATURE_COLUMNS,
        )
        writer.writeheader()
        for i, image_id in enumerate(image_ids):
            score = f"{oof_scores[i]:.8f}"
            label = quality_label(float(oof_scores[i]))
            writer.writerow(
                {
                    "image_id": image_id,
                    "fold": int(fold_of[i]),
                    "oof_score": score,
                    "oof_label": label,
                    "xgb_quality_score": score,
                    "xgb_quality_label": label,
                    "quality_source": "xgb_oof",
                    **{c: f"{X[i, j]:.6f}" for j, c in enumerate(FEATURE_COLUMNS)},
                }
            )

    final_rounds = int(round(float(np.mean(best_iters)) * 1.1))
    params = dict(XGB_PARAMS, seed=args.seed)
    dfull = xgb.DMatrix(X, label=y, weight=w, feature_names=FEATURE_COLUMNS)
    final = xgb.train(params, dfull, num_boost_round=final_rounds)
    final.set_attr(feature_columns=json.dumps(FEATURE_COLUMNS), high_threshold=str(HIGH_THRESHOLD), medium_threshold=str(MEDIUM_THRESHOLD))
    model_path = args.out_dir / "xgb_final_model.json"
    final.save_model(model_path)
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()

    summary = {
        "feature_columns": FEATURE_COLUMNS,
        "folds": args.folds,
        "seed": args.seed,
        "total_rows": len(rows),
        "excluded_fixed_test": dropped_test,
        "kept_rows": n,
        "label_counts": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "best_iterations": best_iters,
        "final_num_boost_round": final_rounds,
        "model": str(model_path),
        "model_sha256": sha,
        "oof_manifest": str(args.out_dir / "xgb_oof_manifest.csv"),
    }
    (args.out_dir / "xgb_rebuild_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_predict(args: argparse.Namespace) -> None:
    import sys

    sys.path.insert(0, str(PROJECT))
    from phase2.features import sample_from_mat

    model_path = args.out_dir / "xgb_final_model.json"
    if not model_path.exists():
        raise SystemExit(f"final model missing: {model_path}")
    booster = xgb.Booster()
    booster.load_model(model_path)

    base_arc = {r["image_id"]: r for r in csv.DictReader(open(args.base_arcface, encoding="utf-8", newline=""))}
    ext_arc = {r["image_id"]: r for r in csv.DictReader(open(args.external_arcface, encoding="utf-8", newline=""))} if Path(args.external_arcface).exists() else {}

    with args.fixed_manifest.open("r", encoding="utf-8", newline="") as f:
        fixed_rows = list(csv.DictReader(f))

    out_rows = []
    covered = missing = 0
    for r in fixed_rows:
        image_id = r["image_id"]
        is_base = r["source_dataset"] == "stylegan2_base"
        mat_path = (args.base_deca_dir / image_id / f"{image_id}.mat") if is_base else (args.external_deca_dir / r["eval_id"] / f"{image_id}.mat")
        arc_row = base_arc.get(image_id) if is_base else ext_arc.get(image_id)
        if not mat_path.exists():
            out_rows.append({"image_id": image_id, "eval_id": r["eval_id"], "source_group": r["source_group"], "xgb_quality_score": "", "xgb_quality_label": "", "feature_coverage": "missing", "failure_reason": "upstream_deca_failure"})
            missing += 1
            continue
        sample = sample_from_mat(mat_path, arc_row)
        feat = np.asarray([[float(sample.metrics[c]) for c in FEATURE_COLUMNS]], dtype=np.float32)
        score = float(booster.predict(xgb.DMatrix(feat, feature_names=FEATURE_COLUMNS))[0])
        out_rows.append({"image_id": image_id, "eval_id": r["eval_id"], "source_group": r["source_group"], "xgb_quality_score": f"{score:.8f}", "xgb_quality_label": quality_label(score), "feature_coverage": "full", "failure_reason": ""})
        covered += 1

    with (args.out_dir / "xgb_fixed_test_predictions.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "eval_id", "source_group", "xgb_quality_score", "xgb_quality_label", "feature_coverage", "failure_reason"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(json.dumps({"total": len(fixed_rows), "covered": covered, "missing": missing}, indent=2))


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    else:
        run_predict(args)
