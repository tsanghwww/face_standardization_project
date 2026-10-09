"""Prepare frozen Phase2 targets for quarantined external training candidates."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import xgboost as xgb

from phase2.features import sample_from_mat
from phase2.rebuild_xgboost import FEATURE_COLUMNS, quality_label
from phase3.reconstruction_data import file_hash, read_ids


EXPECTED_CHECKPOINT = "bc520af061812d9a52e3793729a70e5a0a693d0b1179b60ab35e9adc3cb2a004"


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deca-audit", "checkpoint", "xgb-model", "split-dir", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError("Use a new output directory to preserve generation-time provenance")
    if file_hash(args.checkpoint) != EXPECTED_CHECKPOINT:
        raise ValueError("Checkpoint differs from the frozen Phase2 Full model")
    records = [json.loads(line) for line in args.deca_audit.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [str(row["image_id"]) for row in records]
    if not records or len(set(ids)) != len(ids):
        raise ValueError("Empty or duplicate external audit IDs")
    protected = read_ids(args.split_dir / "validation_ids.txt") | read_ids(args.split_dir / "fixed_test_ids.txt")
    if set(ids) & protected:
        raise ValueError("External candidates overlap protected split IDs")
    for row in records:
        if row["deca_status"] != "success" or row["fallback_used"] or row["training_eligible"]:
            raise ValueError("Expected successful quarantined FAN-only DECA candidates")
        if row["identity_decision"] != "machine_pass_conditional_on_protected_coverage":
            raise ValueError("Candidate did not pass identity screening")
        if file_hash(Path(row["source_image"])) != row["source_image_sha256"]:
            raise ValueError("Candidate source image changed")
    sources = {
        str(path): file_hash(path) for path in [args.checkpoint, args.xgb_model, args.deca_audit,
            *(args.split_dir / name for name in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")),
            *(Path(row["deca_mat"]) for row in records),
            *(Path(row["deca_mat"]).with_name(f"{row['image_id']}_kpt2d.txt") for row in records),
            *(Path(row["source_image"]) for row in records), *(Path(row["embedding_path"]) for row in records)]
    }
    booster = xgb.Booster()
    booster.load_model(args.xgb_model)
    if booster.feature_names != FEATURE_COLUMNS:
        raise ValueError("Frozen XGBoost feature order mismatch")
    args.out_dir.mkdir(parents=True)
    arc_rows, quality_rows = [], []
    for row in records:
        arc = {"image_id": row["image_id"], "arcface_status": row["arcface_status"],
               "detector_score": row["arcface_detector_score"]}
        sample = sample_from_mat(Path(row["deca_mat"]), arc)
        features = np.array([[sample.metrics[key] for key in FEATURE_COLUMNS]], dtype=np.float32)
        if not np.isfinite(features).all():
            raise ValueError("Nonfinite XGBoost features")
        score = float(booster.predict(xgb.DMatrix(features, feature_names=FEATURE_COLUMNS))[0])
        arc_rows.append(arc)
        quality_rows.append({"image_id": row["image_id"], "xgb_quality_score": score,
                             "xgb_quality_label": quality_label(score)})
    write_csv(args.out_dir / "arcface_manifest.csv", arc_rows)
    write_csv(args.out_dir / "xgb_predictions.csv", quality_rows)
    (args.out_dir / "external_candidate_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    mat_roots = {Path(row["deca_mat"]).parent.parent for row in records}
    if len(mat_roots) != 1:
        raise ValueError("DECA candidates must share one parameter root")
    command = [sys.executable, "-m", "phase2.infer_standardize_params", "--deca-results-dir", str(mat_roots.pop()),
        "--checkpoint", str(args.checkpoint), "--arcface-manifest", str(args.out_dir / "arcface_manifest.csv"),
        "--xgb-quality-manifest", str(args.out_dir / "xgb_predictions.csv"), "--quality-source", "blend",
        "--alpha-mode", "learned", "--include-ids-file", str(args.out_dir / "external_candidate_ids.txt"),
        "--out-dir", str(args.out_dir / "inference"), "--device", args.device,
        "--reject-threshold", "0.60", "--weak-threshold", "0.40"]
    (args.out_dir / "generation_sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
    subprocess.run(command, check=True)
    with (args.out_dir / "inference" / "phase2_inference_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        outputs = list(csv.DictReader(handle))
    if {row["image_id"] for row in outputs} != set(ids) or len(outputs) != len(ids):
        raise ValueError("Incomplete or duplicate Phase2 output ledger")
    for path, digest in sources.items():
        if file_hash(Path(path)) != digest:
            raise ValueError(f"Generation input changed: {path}")
    lookup = {row["image_id"]: row for row in records}
    enriched, counts = [], {}
    for row in outputs:
        if row["quality_source_effective"] != "blend":
            raise ValueError("Unexpected quality-source fallback")
        with np.load(row["out_npz"], allow_pickle=False) as payload:
            for key, size in (("expression_standardized", 50), ("pose_standardized", 6)):
                if payload[key].size != size or not np.isfinite(payload[key]).all():
                    raise ValueError("Invalid Phase2 target parameters")
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
        enriched.append({**lookup[row["image_id"]], "phase2_npz": row["out_npz"],
            "phase2_npz_sha256": file_hash(Path(row["out_npz"])), "phase2_decision": row["decision"],
            "condition_lineage_mode": "generation_time", "training_eligible": False,
            "eligibility_status": "pending_condition_maps_and_protected_identity_review"})
    manifest = args.out_dir / "external_target_manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in enriched), encoding="utf-8")
    report = {"status": "verified", "condition_lineage_mode": "generation_time", "n_targets": len(outputs),
        "decisions": counts, "training_eligible_count": 0, "checkpoint_sha256": EXPECTED_CHECKPOINT,
        "xgb_model_sha256": file_hash(args.xgb_model), "target_manifest_sha256": file_hash(manifest),
        "inference_config_sha256": file_hash(args.out_dir / "inference" / "inference_config.json"),
        "sources_unchanged": True, "command": command}
    (args.out_dir / "target_provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
