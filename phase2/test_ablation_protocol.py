"""Protocol tests for the Phase2 ablation setup (no training run).

Proves:
  1. full/no_alpha/no_augmentation/no_xgboost share identical val IDs.
  2. validation dataset never augments.
  3. repeated validation on the same checkpoint is numerically identical.
  4. fixed test IDs never appear in XGBoost or Phase2 train/val.

Run: python -m phase2.test_ablation_protocol
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import Phase2Dataset
from .features import find_deca_mat_files, sample_from_mat
from .model import ConditionGenerator
from .train_condition_generator import compute_loss, evaluate, make_split

PROJECT = Path(r"D:\face_standardization_project")
BASE_DECA = PROJECT / "DECA" / "results" / "archive_phase2_params"
BASE_TEST_IDS = PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "base_test_ids.txt"
FIXED_MANIFEST = PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv"
OOF_MANIFEST = PROJECT / "results" / "phase2_xgb_rebuilt_20260824" / "xgb_oof_manifest.csv"
VAL_RATIO = 0.15
SEED = 20260824

GROUPS = ["full", "no_alpha", "no_augmentation", "no_xgboost"]


def _train_pool_ids() -> list[str]:
    excluded = {ln.strip() for ln in BASE_TEST_IDS.read_text(encoding="utf-8").splitlines() if ln.strip()}
    mats = find_deca_mat_files(BASE_DECA)
    return sorted(p.stem for p in mats if p.stem not in excluded)


def test_split_identical() -> None:
    ids = _train_pool_ids()
    n = len(ids)
    splits = [make_split(n, VAL_RATIO, SEED) for _ in GROUPS]
    first_val = [ids[i] for i in splits[0][0]]
    for g, (val_idx, _tr_idx) in zip(GROUPS, splits):
        val_ids = [ids[i] for i in val_idx]
        assert val_ids == first_val, f"{g} val IDs differ from full"
    print(f"[1] split identical across {GROUPS}: val_ids={len(first_val)}")


def test_validation_no_augment() -> None:
    ids = _train_pool_ids()
    sample_paths = [BASE_DECA / i / f"{i}.mat" for i in ids[:8]]
    samples = [sample_from_mat(p) for p in sample_paths]
    ds = Phase2Dataset(samples, augment=False, seed=SEED, stage=3)
    assert ds.augment is False
    for idx in range(len(samples)):
        expr = ds[idx]["expression"].numpy()
        assert np.allclose(expr, samples[idx].params["expression"], atol=1e-6), f"val sample {idx} was augmented"
    print(f"[2] validation no-augment: {len(samples)} samples verified")


def test_validation_deterministic() -> None:
    ids = _train_pool_ids()
    sample_paths = [BASE_DECA / i / f"{i}.mat" for i in ids[:16]]
    samples = [sample_from_mat(p) for p in sample_paths]
    val_ds = Phase2Dataset(samples, augment=False, seed=SEED, stage=3)
    # build normalizer from the same subset (as training does)
    feats = np.vstack([d["features"].numpy() for d in val_ds]).astype(np.float32)
    mean = torch.from_numpy(feats.mean(axis=0)).float()
    std = torch.from_numpy(feats.std(axis=0) + 1e-6).float()
    loader = DataLoader(val_ds, batch_size=8, shuffle=False)
    model = ConditionGenerator(input_dim=int(mean.shape[0]), hidden_dim=64)
    model.alpha_mode = "learned"
    r1 = evaluate(model, loader, mean, std, torch.device("cpu"))
    r2 = evaluate(model, loader, mean, std, torch.device("cpu"))
    for k in r1:
        assert np.isclose(r1[k], r2[k], atol=1e-6), f"metric {k} differs: {r1[k]} vs {r2[k]}"
    # also assert smooth==0 in val (no random noise term)
    assert r1["smooth"] == 0.0
    print(f"[3] validation deterministic: loss={r1['loss']:.6f} (two runs identical), smooth=0")


def test_no_leakage() -> None:
    import csv

    base_test = {ln.strip() for ln in BASE_TEST_IDS.read_text(encoding="utf-8").splitlines() if ln.strip()}
    with FIXED_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        fixed = list(csv.DictReader(f))
    external_ids = {r["image_id"] for r in fixed if r["source_dataset"] != "stylegan2_base"}
    all_fixed = {r["image_id"] for r in fixed}

    pool = set(_train_pool_ids())
    assert not (pool & base_test), "base test IDs leaked into Phase2 train pool"
    assert not (pool & external_ids), "external test IDs leaked into Phase2 train pool"

    oof_ids = {r["image_id"] for r in csv.DictReader(open(OOF_MANIFEST, encoding="utf-8", newline=""))}
    assert not (oof_ids & all_fixed), "fixed test IDs leaked into XGBoost OOF manifest"
    print(f"[4] no leakage: pool={len(pool)} oof={len(oof_ids)} fixed_test={len(all_fixed)} overlap=0")


def main() -> None:
    test_split_identical()
    test_validation_no_augment()
    test_validation_deterministic()
    test_no_leakage()
    print("ALL PROTOCOL TESTS PASSED")


if __name__ == "__main__":
    main()
