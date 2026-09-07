"""Verify Phase3 target lineage against a concrete Phase2 checkpoint and outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from phase3.reconstruction_data import file_hash


FINAL_PHASE2_FULL_SHA256 = "bc520af061812d9a52e3793729a70e5a0a693d0b1179b60ab35e9adc3cb2a004"
FINAL_XGB_OOF_SHA256 = "85ed5ac8749c6ae681a4ef542863551c02041dbf9b733862948d6a8bef0eb846"
TARGET_MAP_FIELDS = ("target_normal_map", "target_depth_map", "target_landmark_map", "target_face_mask")
CACHE_COLUMNS = {
    "target_normal_map": "target_normal", "target_depth_map": "target_depth",
    "target_landmark_map": "target_landmark", "target_face_mask": "target_face_mask",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def same_file_content(left: str, right: str) -> bool:
    return bool(left and right and Path(left).is_file() and Path(right).is_file() and file_hash(Path(left)) == file_hash(Path(right)))


def audit(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    ids = [line.strip() for line in args.ids_file.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Empty or duplicate audit IDs")
    phase2_rows = read_csv(args.phase2_manifest)
    cache_rows = read_csv(args.condition_cache_manifest)
    evaluation_rows = [json.loads(line) for line in args.evaluation_manifest.read_text(encoding="utf-8").splitlines() if line]
    phase2 = {row["image_id"]: row for row in phase2_rows}
    cache = {row["image_id"]: row for row in cache_rows}
    evaluation = {str(row["image_id"]): row for row in evaluation_rows}
    config = json.loads(args.inference_config.read_text(encoding="utf-8"))
    command = args.inference_command.read_text(encoding="utf-8")
    checkpoint_sha = file_hash(args.phase2_checkpoint)
    checkpoint = torch.load(args.phase2_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    errors, warnings, rows = [], [], []
    if len(phase2) != len(phase2_rows):
        errors.append("duplicate_phase2_manifest_ids")
    if len(cache) != len(cache_rows):
        errors.append("duplicate_condition_cache_ids")
    if len(evaluation) != len(evaluation_rows):
        errors.append("duplicate_evaluation_manifest_ids")
    if set(cache) != set(ids):
        errors.append("condition_cache_ids_do_not_exactly_match_audit_ids")
    if set(evaluation) != set(ids):
        errors.append("evaluation_manifest_ids_do_not_exactly_match_audit_ids")

    expected_sha = args.expected_checkpoint_sha256.lower()
    if checkpoint_sha != expected_sha:
        errors.append(f"checkpoint_sha256_mismatch:{checkpoint_sha}")
    if config.get("quality_source") != args.expected_quality_source:
        errors.append(f"inference_quality_source:{config.get('quality_source')}")
    configured_alpha = config.get("alpha_mode")
    effective_alpha = checkpoint_config.get("alpha_mode") if configured_alpha == "auto" else configured_alpha
    if effective_alpha != args.expected_alpha_mode:
        errors.append(f"inference_alpha_mode:{effective_alpha}")
    if checkpoint_config.get("quality_source") != args.expected_quality_source:
        errors.append(f"checkpoint_quality_source:{checkpoint_config.get('quality_source')}")
    if checkpoint_config.get("alpha_mode") != args.expected_alpha_mode:
        errors.append(f"checkpoint_alpha_mode:{checkpoint_config.get('alpha_mode')}")
    checkpoint_name = args.phase2_checkpoint.name.lower()
    if checkpoint_name not in command.lower() and str(config.get("checkpoint", "")).lower().endswith(checkpoint_name) is False:
        errors.append("checkpoint_not_bound_by_inference_config_or_command")
    configured_checkpoint = Path(str(config.get("checkpoint", "")))
    if not configured_checkpoint.is_file() or file_hash(configured_checkpoint) != checkpoint_sha:
        errors.append("configured_checkpoint_content_mismatch")
    artifact_dirs = {
        args.phase2_manifest.parent.resolve(), args.inference_config.parent.resolve(), args.inference_command.parent.resolve()
    }
    if len(artifact_dirs) != 1:
        errors.append("inference_manifest_config_command_not_colocated")
    inference_input_hashes = {}
    for key in ("arcface_manifest", "xgb_quality_manifest", "include_ids_file"):
        value = config.get(key)
        if value and Path(value).is_file():
            inference_input_hashes[key] = file_hash(Path(value))
            if key == "include_ids_file":
                configured_ids = {line.strip() for line in Path(value).read_text(encoding="utf-8-sig").splitlines() if line.strip()}
                if set(ids) - configured_ids:
                    errors.append("audit_ids_not_in_configured_include_ids")
        elif value:
            errors.append(f"configured_input_missing:{key}")
        elif key == "xgb_quality_manifest" and args.expected_quality_source in ("xgb", "blend"):
            errors.append("configured_xgb_manifest_missing")
    expected_xgb_sha = args.expected_xgb_sha256.lower()
    if args.expected_quality_source in ("xgb", "blend") and inference_input_hashes.get("xgb_quality_manifest") != expected_xgb_sha:
        errors.append("xgb_oof_sha256_mismatch")

    for image_id in ids:
        row_errors = []
        phase = phase2.get(image_id)
        data = evaluation.get(image_id)
        condition = cache.get(image_id)
        if phase is None or data is None or condition is None:
            row_errors.append("missing_manifest_row")
            rows.append({"image_id": image_id, "status": "failed", "errors": row_errors})
            continue
        phase_npz = phase.get("out_npz", "")
        eval_npz = data.get("phase2_npz", "")
        if phase.get("quality_source_requested") not in ("", args.expected_quality_source):
            row_errors.append(f"quality_source_requested:{phase.get('quality_source_requested')}")
        if phase.get("quality_source_effective") != args.expected_quality_source:
            row_errors.append(f"quality_source_effective:{phase.get('quality_source_effective')}")
        if args.expected_quality_source in ("xgb", "blend"):
            try:
                if not np.isfinite(float(phase.get("xgb_quality_score", ""))):
                    raise ValueError
            except (TypeError, ValueError):
                row_errors.append("missing_or_nonfinite_xgb_quality_score")
        if not same_file_content(phase_npz, eval_npz):
            row_errors.append("phase2_npz_binding_mismatch")
        try:
            with np.load(eval_npz, allow_pickle=False) as values:
                stored_id = str(np.asarray(values["image_id"]).reshape(-1)[0])
                expression = np.asarray(values["expression_standardized"], dtype=np.float32).reshape(-1)
                pose = np.asarray(values["pose_standardized"], dtype=np.float32).reshape(-1)
                source_mat = str(np.asarray(values["source_mat"]).reshape(-1)[0])
            if stored_id != image_id or expression.shape != (50,) or pose.shape != (6,):
                row_errors.append("invalid_phase2_npz_schema")
            if not np.isfinite(expression).all() or not np.isfinite(pose).all():
                row_errors.append("nonfinite_phase2_target")
            if not same_file_content(source_mat, phase.get("mat_path", "")) or not same_file_content(source_mat, data.get("deca_mat", "")):
                row_errors.append("source_deca_binding_mismatch")
        except Exception as error:
            row_errors.append(f"phase2_npz_read:{type(error).__name__}:{error}")
        map_hashes = {}
        for field in TARGET_MAP_FIELDS:
            eval_path = data.get(field)
            cache_path = condition.get(CACHE_COLUMNS[field])
            if not same_file_content(eval_path, cache_path):
                row_errors.append(f"condition_cache_binding_mismatch:{field}")
            elif eval_path:
                map_hashes[field] = file_hash(Path(eval_path))
        rows.append({
            "image_id": image_id, "status": "verified" if not row_errors else "failed", "errors": row_errors,
            "source_mat_sha256": file_hash(Path(data["deca_mat"])) if data.get("deca_mat") and Path(data["deca_mat"]).is_file() else None,
            "phase2_npz_sha256": file_hash(Path(eval_npz)) if eval_npz and Path(eval_npz).is_file() else None,
            "target_map_sha256": map_hashes,
        })
    errors.extend(f"sample:{row['image_id']}:{item}" for row in rows for item in row["errors"])
    lineage_path = args.condition_cache_manifest.parent / "condition_cache_lineage.jsonl"
    cache_provenance_path = args.condition_cache_manifest.parent / "condition_cache_provenance.json"
    lineage_mode = "absent"
    if lineage_path.is_file():
        lineage_rows = [json.loads(line) for line in lineage_path.read_text(encoding="utf-8").splitlines() if line]
        lineage_by_id = {row.get("image_id"): row for row in lineage_rows}
        modes = {row.get("binding_mode", "unknown") for row in lineage_rows if row.get("image_id") in ids}
        lineage_mode = "+".join(sorted(modes)) if modes else "no_selected_rows"
        for row in rows:
            lineage = lineage_by_id.get(row["image_id"])
            if not lineage:
                message = "missing_condition_lineage"
                row["errors"].append(message)
                row["status"] = "failed"
                errors.append(f"sample:{row['image_id']}:{message}")
                continue
            if lineage.get("phase2_npz_sha256") != row.get("phase2_npz_sha256"):
                message = "lineage_phase2_npz_mismatch"
                row["errors"].append(message)
                row["status"] = "failed"
                errors.append(f"sample:{row['image_id']}:{message}")
            if lineage.get("source_mat_sha256") != row.get("source_mat_sha256"):
                message = "lineage_source_mat_mismatch"
                row["errors"].append(message)
                row["status"] = "failed"
                errors.append(f"sample:{row['image_id']}:{message}")
            expected_maps = {
                "target_normal": row.get("target_map_sha256", {}).get("target_normal_map"),
                "target_depth": row.get("target_map_sha256", {}).get("target_depth_map"),
                "target_landmark": row.get("target_map_sha256", {}).get("target_landmark_map"),
                "target_face_mask": row.get("target_map_sha256", {}).get("target_face_mask"),
            }
            if lineage.get("target_map_sha256") != expected_maps:
                message = "lineage_target_map_mismatch"
                row["errors"].append(message)
                row["status"] = "failed"
                errors.append(f"sample:{row['image_id']}:{message}")
        if "generation_time" not in modes or any(mode != "generation_time" for mode in modes):
            if getattr(args, "allow_retrospective_cache_binding", False):
                warnings.append("condition_cache_contains_retrospective_binding")
            else:
                errors.append("condition_cache_not_fully_generation_time_bound")
    else:
        message = "condition_cache_lineage_absent:current files can be bound retrospectively but historical render execution is not proven"
        if getattr(args, "allow_retrospective_cache_binding", False):
            warnings.append(message)
        else:
            errors.append(message)
    if cache_provenance_path.is_file():
        cache_provenance = json.loads(cache_provenance_path.read_text(encoding="utf-8"))
        expected_provenance = {
            "phase2_manifest_sha256": file_hash(args.phase2_manifest),
            "condition_manifest_sha256": file_hash(args.condition_cache_manifest),
            "lineage_sha256": file_hash(lineage_path) if lineage_path.is_file() else None,
            "code_sha256": file_hash(Path(__file__).parents[1] / "scripts" / "build_phase3_condition_cache.py"),
        }
        for key, value in expected_provenance.items():
            if cache_provenance.get(key) != value:
                errors.append(f"condition_cache_provenance_mismatch:{key}")
    elif not getattr(args, "allow_retrospective_cache_binding", False):
        errors.append("condition_cache_provenance_absent")
    else:
        warnings.append("condition_cache_provenance_absent")
    strict_lineage = lineage_mode == "generation_time" and cache_provenance_path.is_file()
    status = "failed" if errors else "verified" if strict_lineage else "retrospective_only"
    report = {
        "status": status,
        "target_source_decision": "final_phase2_full" if checkpoint_sha == FINAL_PHASE2_FULL_SHA256 else "noncanonical_or_unknown",
        "checkpoint": str(args.phase2_checkpoint), "checkpoint_sha256": checkpoint_sha,
        "expected_checkpoint_sha256": expected_sha, "checkpoint_matches_expected": checkpoint_sha == expected_sha,
        "inference_config_sha256": file_hash(args.inference_config), "inference_command_sha256": file_hash(args.inference_command),
        "inference_input_hashes": inference_input_hashes,
        "phase2_manifest_sha256": file_hash(args.phase2_manifest),
        "condition_cache_manifest_sha256": file_hash(args.condition_cache_manifest),
        "evaluation_manifest_sha256": file_hash(args.evaluation_manifest), "ids_sha256": file_hash(args.ids_file),
        "expected_quality_source": args.expected_quality_source, "effective_alpha_mode": effective_alpha,
        "expected_xgb_sha256": expected_xgb_sha,
        "n_ids": len(ids), "n_verified": sum(row["status"] == "verified" for row in rows),
        "n_failed": sum(row["status"] != "verified" for row in rows),
        "condition_lineage_mode": lineage_mode, "errors": errors, "warnings": warnings,
        "interpretation": "Directory names are ignored. Verification uses checkpoint/config/command/manifest/NPZ/map contents.",
    }
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("phase2-manifest", "inference-config", "inference-command", "phase2-checkpoint",
                 "condition-cache-manifest", "evaluation-manifest", "ids-file", "out-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--expected-checkpoint-sha256", default=FINAL_PHASE2_FULL_SHA256)
    parser.add_argument("--expected-quality-source", choices=("heuristic", "xgb", "blend"), default="blend")
    parser.add_argument("--expected-alpha-mode", choices=("learned", "fixed_one"), default="learned")
    parser.add_argument("--expected-xgb-sha256", default=FINAL_XGB_OOF_SHA256)
    parser.add_argument("--allow-retrospective-cache-binding", action="store_true",
                        help="Permit legacy cache binding without proof that this invocation rendered the maps")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    report, rows = audit(args)
    (args.out_dir / "target_provenance.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    with (args.out_dir / "target_lineage.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    (args.out_dir / "exact_command.txt").write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
