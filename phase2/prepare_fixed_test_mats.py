"""Consolidate the 775 fixed-test DECA .mat files into one image_id-keyed dir.

The fixed-test DECA params live in two places:
  * 400 base samples      -> DECA/results/archive_phase2_params/<image_id>/<image_id>.mat
  * 375 external samples  -> results/phase2_eval_fixed_20260824_v2/deca_params/<eval_id>/<image_id>.mat

infer_standardize_params.py globs a single --deca-results-dir and keys on the
mat stem == image_id, so both sets must be materialized under
<out-dir>/<image_id>/<image_id>.mat (hardlinked, no data duplication).  It also
writes <out-dir>/fixed_test_ids.txt (the 775 image_ids in manifest order) for
use with --include-ids-file.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

PROJECT = Path(r"D:\face_standardization_project")


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv",
    )
    parser.add_argument("--base-deca-dir", type=Path, default=PROJECT / "DECA" / "results" / "archive_phase2_params")
    parser.add_argument(
        "--external-deca-dir",
        type=Path,
        default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "deca_params",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results" / "phase2_ablation_20260824" / "fixed_test_mats")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    linked = 0
    ids: list[str] = []
    for row in rows:
        image_id = row["image_id"]
        ids.append(image_id)
        if row["source_dataset"] == "stylegan2_base":
            src = args.base_deca_dir / image_id / f"{image_id}.mat"
        else:
            src = args.external_deca_dir / row["eval_id"] / f"{image_id}.mat"
        dst_dir = args.out_dir / image_id
        dst = dst_dir / f"{image_id}.mat"
        if not src.exists():
            missing.append(f"{row['eval_id']} -> {src}")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        link_or_copy(src, dst)
        for suffix in ("_kpt2d.txt", "_kpt3d.txt"):
            sidecar = src.with_name(f"{src.stem}{suffix}")
            if not sidecar.exists():
                missing.append(f"{row['eval_id']} sidecar -> {sidecar}")
                continue
            link_or_copy(sidecar, dst.with_name(f"{image_id}{suffix}"))
        linked += 1

    (args.out_dir / "fixed_test_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"linked={linked} missing={len(missing)} total={len(rows)} out_dir={args.out_dir}")
    for m in missing[:20]:
        print(f"  MISSING {m}")


if __name__ == "__main__":
    main()
