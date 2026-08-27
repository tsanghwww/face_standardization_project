"""Split validation outcomes into gate_train and hard_calibration.

Stratified by (unsafe, xgb_quality_label) with a parameterized train ratio
(default 0.70) and seed=20260827.  Fixed-test IDs are excluded (their overlap
with both splits is asserted to be 0).  Writes ID files + a reproducible
summary (seed, ratio, input hash).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outcome-manifest", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--exclude-ids-file", type=Path, required=True, help="fixed test image_id file")
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    rows = read_csv(args.outcome_manifest)
    seen: set[str] = set()
    dedup = [r for r in rows if not (r["image_id"] in seen or seen.add(r["image_id"]))]
    excluded = {ln.strip() for ln in args.exclude_ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
    pool = [r for r in dedup if r["image_id"] not in excluded]
    overlap = len({r["image_id"] for r in dedup} & excluded)
    if overlap:
        raise SystemExit(f"{overlap} fixed-test IDs appear in the validation outcome pool")

    rng = np.random.default_rng(args.seed)
    strata: dict[tuple[str, str], list[dict]] = {}
    for r in pool:
        key = (r["unsafe"], r.get("xgb_quality_label") or "unknown")
        strata.setdefault(key, []).append(r)

    train_rows: list[dict] = []
    cal_rows: list[dict] = []
    for key, items in strata.items():
        idx = np.arange(len(items))
        rng.shuffle(idx)
        n_train = int(round(len(items) * args.train_ratio))
        n_train = max(1, min(n_train, len(items) - 1)) if len(items) > 1 else len(items)
        for i in idx[:n_train]:
            train_rows.append(items[int(i)])
        for i in idx[n_train:]:
            cal_rows.append(items[int(i)])

    train_ids = sorted(r["image_id"] for r in train_rows)
    cal_ids = sorted(r["image_id"] for r in cal_rows)
    if set(train_ids) & set(cal_ids):
        raise SystemExit("gate_train and hard_calibration are not disjoint")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "gate_train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (args.out_dir / "hard_calibration_ids.txt").write_text("\n".join(cal_ids) + "\n", encoding="utf-8")
    summary = {
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "n_pool": len(pool),
        "n_gate_train": len(train_ids),
        "n_hard_calibration": len(cal_ids),
        "strata": {f"{k[0]}|{k[1]}": len(v) for k, v in strata.items()},
        "fixed_test_overlap": overlap,
        "input_hash": hashlib.sha256(args.outcome_manifest.read_bytes()).hexdigest()[:16],
    }
    (args.out_dir / "gate_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
