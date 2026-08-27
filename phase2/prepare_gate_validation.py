"""Materialize the frozen Phase2 validation split for gate calibration."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


PROJECT = Path(r"D:\face_standardization_project")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-ids", type=Path, default=PROJECT / "results/phase2_ablation_20260825/full/val_ids.txt")
    parser.add_argument("--base-deca-dir", type=Path, default=PROJECT / "DECA/results/archive_phase2_params")
    parser.add_argument("--xgb-oof-manifest", type=Path, default=PROJECT / "results/phase2_xgb_rebuilt_20260824/xgb_oof_manifest.csv")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results/phase2_gate_calibration_20260826")
    return parser.parse_args()


def link_or_copy(source: Path, target: Path) -> None:
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    ids = [line.strip() for line in args.val_ids.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(ids) != 1440 or len(set(ids)) != 1440:
        raise SystemExit(f"Expected 1,440 unique validation IDs, got {len(ids)} rows / {len(set(ids))} unique")

    out_mats = args.out_dir / "validation_mats"
    out_mats.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    manifest: list[dict[str, str]] = []
    for image_id in ids:
        source = args.base_deca_dir / image_id / f"{image_id}.mat"
        target_dir = out_mats / image_id
        target = target_dir / f"{image_id}.mat"
        if not source.exists():
            missing.append(str(source))
            continue
        target_dir.mkdir(parents=True, exist_ok=True)
        link_or_copy(source, target)
        for suffix in ("_kpt2d.txt", "_kpt3d.txt"):
            sidecar = source.with_name(f"{image_id}{suffix}")
            if not sidecar.exists():
                missing.append(str(sidecar))
            else:
                link_or_copy(sidecar, target.with_name(f"{image_id}{suffix}"))
        manifest.append({
            "eval_id": image_id,
            "image_id": image_id,
            "source_group": "phase2_validation",
            "source_dataset": "stylegan2_base",
            "image_path": "",
            "mat_path": str(target),
            "preprocess_source": "fan_saved_mat",
        })
    if missing:
        raise SystemExit("Missing validation inputs:\n" + "\n".join(missing[:30]))

    oof_all = {row["image_id"]: row for row in read_csv(args.xgb_oof_manifest)}
    oof_rows = [oof_all[image_id] for image_id in ids if image_id in oof_all]
    if len(oof_rows) != 1440:
        absent = [image_id for image_id in ids if image_id not in oof_all]
        raise SystemExit(f"OOF coverage {len(oof_rows)}/1440; missing={absent[:20]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "validation_manifest.csv", manifest, list(manifest[0]))
    write_csv(args.out_dir / "validation_xgb_oof.csv", oof_rows, list(oof_rows[0]))
    (args.out_dir / "validation_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    summary = {
        "validation_count": len(manifest),
        "xgb_oof_count": len(oof_rows),
        "mat_count": len(list(out_mats.glob("*/*.mat"))),
        "split_source": str(args.val_ids),
        "xgb_source": str(args.xgb_oof_manifest),
        "test_data_used_for_calibration": False,
    }
    (args.out_dir / "validation_prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
