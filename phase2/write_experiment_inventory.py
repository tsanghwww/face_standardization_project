"""Write experiment_inventory.json for one Phase2 ablation experiment.

Captures provenance (seed, git commit, checkpoint/manifest SHA256) and outcome
counts (train/test samples, success/failure) so each experiment directory is
self-describing and auditable.  Run this after an experiment's train/infer/
render/evaluate steps have produced their artifacts.

Usage:
  python -m phase2.write_experiment_inventory --exp-dir results/phase2_ablation_20260824/full \
      --exp-name full --seed 20260824 --quality-source blend --alpha-mode learned \
      --augmentation true --command "python -m phase2.train_condition_generator ..."
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(r"D:\face_standardization_project")
FIXED_MANIFEST = PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv"
EXCLUDE_IDS = PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "base_test_ids.txt"
XGB_MANIFEST = PROJECT / "results" / "phase2_xgb_quality_bug003_fixed_arcface_ok" / "xgb_quality_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-dir", required=True, type=Path)
    parser.add_argument("--exp-name", required=True, type=str)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--quality-source", required=True, type=str)
    parser.add_argument("--alpha-mode", required=True, type=str)
    parser.add_argument("--augmentation", required=True, type=lambda x: str(x).lower() in {"true", "1", "yes"})
    parser.add_argument("--command", required=True, type=str)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fixed-manifest", type=Path, default=FIXED_MANIFEST)
    parser.add_argument("--exclude-ids-file", type=Path, default=EXCLUDE_IDS)
    parser.add_argument("--xgb-manifest", type=Path)
    parser.add_argument("--started-at", type=str, default="")
    return parser.parse_args()


def sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(PROJECT), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # noqa: BLE001 - git may be unavailable
        pass
    # fallback: read .git/HEAD directly
    try:
        head = (PROJECT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref:"):
            ref = head.split(":", 1)[1].strip()
            ref_path = PROJECT / ".git" / ref
            if ref_path.exists():
                return ref_path.read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        return None
    return None


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def main() -> None:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint or (exp_dir / "best_model.pt")
    started = args.started_at or datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    ended = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    train_summary = read_json(exp_dir / "train_summary.json")
    infer_summary = read_json(exp_dir / "phase2_inference_summary.json")
    render_summary = read_json(exp_dir / "render_summary.json")
    eval_summary = read_json(exp_dir / "rendered_metrics_summary.json")

    inventory = {
        "experiment_name": args.exp_name,
        "seed": args.seed,
        "git_commit": git_commit(),
        "checkpoint": str(checkpoint) if checkpoint.exists() else None,
        "checkpoint_sha256": sha256(checkpoint),
        "fixed_manifest": str(args.fixed_manifest),
        "fixed_manifest_sha256": sha256(args.fixed_manifest),
        "excluded_ids_file": str(args.exclude_ids_file),
        "excluded_ids_sha256": sha256(args.exclude_ids_file),
        "xgb_manifest": str(args.xgb_manifest) if args.xgb_manifest else None,
        "xgb_manifest_sha256": sha256(args.xgb_manifest),
        "quality_source": args.quality_source,
        "alpha_mode": args.alpha_mode,
        "augmentation": args.augmentation,
        "train_samples": train_summary.get("train_samples"),
        "val_samples": train_summary.get("val_samples"),
        "test_samples": infer_summary.get("count"),
        "render_success": render_summary.get("complete_triplets"),
        "eval_metric_rows": eval_summary.get("n_metric_rows"),
        "eval_failure_count": eval_summary.get("failure_count"),
        "started_at": started,
        "ended_at": ended,
        "command": args.command,
    }
    out = exp_dir / "experiment_inventory.json"
    out.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    print(json.dumps(inventory, indent=2))


if __name__ == "__main__":
    main()
