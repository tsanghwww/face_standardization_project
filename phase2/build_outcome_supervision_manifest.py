"""Build a per-sample outcome-supervision manifest from rendered metrics.

Joins rendered_metrics.csv (per eval_id x method: original/hard_zero/full),
the Phase2 inference manifest (per image_id: quality/alpha/standardized), and
the XGBoost OOF manifest (per image_id: quality features) into one row per
image_id.  Missing metrics are preserved as empty strings -- never filled with
0 or dropped.  A derived binary ``unsafe`` label (from rendered outcome deltas
vs hard_zero, with CLI margins) is attached for the outcome gate.

Keyed by image_id (deduplicated).  Optional --id-manifest maps eval_id -> image_id
(for the fixed test where eval_id == "base_<id>" but image_id == "<id>").
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

OUTCOME_FIELDS = [
    "image_id", "split",
    "render_original_status", "render_hard_zero_status", "render_full_status",
    "arcface_cosine", "identity_delta_vs_hard_zero",
    "deca_head_pose_norm", "pose_improvement_vs_original", "pose_delta_vs_hard_zero",
    "expression_norm", "expression_delta_vs_hard_zero",
    "l2cs_gaze_delta_vs_original_deg", "gaze_delta_vs_hard_zero_deg",
    "quality_score", "xgb_quality_score", "xgb_quality_label",
    "landmark_score", "landmark_out_ratio", "landmark_bbox_area", "landmark_center_dist",
    "arcface_status", "arcface_score", "reject_score", "confidence", "heuristic_quality_score", "xgb_status",
    "original_exp_norm", "original_head_pose_norm", "original_jaw_pose_norm",
    "alpha_expression", "alpha_head_pose", "alpha_jaw_pose",
    "standardized_exp_norm", "standardized_head_pose_norm", "standardized_jaw_pose_norm",
    "unsafe", "unsafe_reason",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics-csv", required=True, type=Path)
    p.add_argument("--inference-manifest", required=True, type=Path)
    p.add_argument("--xgb-oof-manifest", required=True, type=Path)
    p.add_argument("--id-manifest", type=Path, help="CSV with image_id + eval_id columns (optional).")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--split", default="validation")
    p.add_argument("--seed", type=int, default=20260827)
    p.add_argument("--identity-margin", type=float, default=0.02)
    p.add_argument("--pose-margin", type=float, default=0.03)
    p.add_argument("--expression-margin", type=float, default=0.02)
    p.add_argument("--gaze-margin-deg", type=float, default=10.0)
    return p.parse_args()


def _num(row, key):
    s = str(row.get(key, "")).strip()
    if s == "":
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    args = parse_args()
    metrics = read_csv(args.metrics_csv)
    by_eval_method = {(r["eval_id"], r["method"]): r for r in metrics}
    inference = {r["image_id"]: r for r in read_csv(args.inference_manifest)}
    xgb = {r["image_id"]: r for r in read_csv(args.xgb_oof_manifest)}

    id_map: dict[str, str] = {}
    order: list[str] = []
    if args.id_manifest:
        for r in read_csv(args.id_manifest):
            image_id = r["image_id"]
            id_map[r["eval_id"]] = image_id
            if image_id not in order:
                order.append(image_id)
    else:
        order = sorted({r["eval_id"] for r in metrics})

    rows: list[dict] = []
    for image_id in order:
        eval_id = image_id if not args.id_manifest else next((e for e, i in id_map.items() if i == image_id), image_id)
        original = by_eval_method.get((eval_id, "original"), {})
        hard = by_eval_method.get((eval_id, "hard_zero"), {})
        full = by_eval_method.get((eval_id, "full"), {})
        inf = inference.get(image_id, {})
        xr = xgb.get(image_id, {})

        full_arc = _num(full, "arcface_cosine")
        hard_arc = _num(hard, "arcface_cosine")
        full_pose = _num(full, "deca_head_pose_norm")
        orig_pose = _num(original, "deca_head_pose_norm")
        hard_pose = _num(hard, "deca_head_pose_norm")
        full_exp = _num(full, "deca_expression_norm")
        hard_exp = _num(hard, "deca_expression_norm")
        full_gaze = _num(full, "l2cs_gaze_delta_vs_original_deg")
        hard_gaze = _num(hard, "l2cs_gaze_delta_vs_original_deg")

        id_delta = None if full_arc is None or hard_arc is None else full_arc - hard_arc
        pose_delta = None if full_pose is None or hard_pose is None else full_pose - hard_pose
        exp_delta = None if full_exp is None or hard_exp is None else full_exp - hard_exp
        gaze_delta = None if full_gaze is None or hard_gaze is None else full_gaze - hard_gaze
        pose_improvement = None if full_pose is None or orig_pose is None else orig_pose - full_pose

        required = [full_arc, hard_arc, full_pose, hard_pose, full_exp, hard_exp, full_gaze, hard_gaze, orig_pose]
        missing = any(v is None for v in required)
        unsafe_reasons: list[str] = []
        if missing:
            unsafe_reasons.append("missing_required_metric")
        else:
            if id_delta < -args.identity_margin:
                unsafe_reasons.append("identity_noninferiority_failed")
            if pose_delta > args.pose_margin:
                unsafe_reasons.append("pose_degradation")
            if exp_delta > args.expression_margin:
                unsafe_reasons.append("expression_degradation")
            if gaze_delta > args.gaze_margin_deg:
                unsafe_reasons.append("gaze_degradation")

        rows.append({
            "image_id": image_id,
            "split": args.split,
            "render_original_status": original.get("render_status", ""),
            "render_hard_zero_status": hard.get("render_status", ""),
            "render_full_status": full.get("render_status", ""),
            "arcface_cosine": "" if full_arc is None else f"{full_arc:.6f}",
            "identity_delta_vs_hard_zero": "" if id_delta is None else f"{id_delta:.6f}",
            "deca_head_pose_norm": "" if full_pose is None else f"{full_pose:.6f}",
            "pose_improvement_vs_original": "" if pose_improvement is None else f"{pose_improvement:.6f}",
            "pose_delta_vs_hard_zero": "" if pose_delta is None else f"{pose_delta:.6f}",
            "expression_norm": "" if full_exp is None else f"{full_exp:.6f}",
            "expression_delta_vs_hard_zero": "" if exp_delta is None else f"{exp_delta:.6f}",
            "l2cs_gaze_delta_vs_original_deg": "" if full_gaze is None else f"{full_gaze:.6f}",
            "gaze_delta_vs_hard_zero_deg": "" if gaze_delta is None else f"{gaze_delta:.6f}",
            "quality_score": inf.get("quality_score", ""),
            "xgb_quality_score": inf.get("xgb_quality_score", ""),
            "xgb_quality_label": xr.get("xgb_quality_label", inf.get("xgb_quality_label", "")),
            "landmark_score": xr.get("landmark_score", ""),
            "landmark_out_ratio": xr.get("landmark_out_ratio", ""),
            "landmark_bbox_area": xr.get("landmark_bbox_area", ""),
            "landmark_center_dist": xr.get("landmark_center_dist", ""),
            "arcface_status": xr.get("arcface_status", ""),
            "arcface_score": xr.get("arcface_score", ""),
            "reject_score": inf.get("reject_score", ""),
            "confidence": inf.get("confidence", ""),
            "heuristic_quality_score": inf.get("heuristic_quality_score", ""),
            "xgb_status": "1" if _num(inf, "xgb_quality_score") is not None else "0",
            "original_exp_norm": inf.get("original_exp_norm", ""),
            "original_head_pose_norm": inf.get("original_head_pose_norm", ""),
            "original_jaw_pose_norm": inf.get("original_jaw_pose_norm", ""),
            "alpha_expression": inf.get("alpha_expression", ""),
            "alpha_head_pose": inf.get("alpha_head_pose", ""),
            "alpha_jaw_pose": inf.get("alpha_jaw_pose", ""),
            "standardized_exp_norm": inf.get("standardized_exp_norm", ""),
            "standardized_head_pose_norm": inf.get("standardized_head_pose_norm", ""),
            "standardized_jaw_pose_norm": inf.get("standardized_jaw_pose_norm", ""),
            "unsafe": "1" if unsafe_reasons else "0",
            "unsafe_reason": ";".join(unsafe_reasons),
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "outcome_manifest.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUTCOME_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    ids_file = args.out_dir / "outcome_image_ids.txt"
    ids_file.write_text("\n".join(r["image_id"] for r in rows) + "\n", encoding="utf-8")
    summary = {
        "n": len(rows),
        "split": args.split,
        "seed": args.seed,
        "unsafe": sum(r["unsafe"] == "1" for r in rows),
        "safe": sum(r["unsafe"] == "0" for r in rows),
        "missing_required_metric": sum("missing_required_metric" in r["unsafe_reason"] for r in rows),
        "margins": {"identity": args.identity_margin, "pose": args.pose_margin, "expression": args.expression_margin, "gaze_deg": args.gaze_margin_deg},
        "manifest": str(out),
        "ids_file": str(ids_file),
        "input_hash": hashlib.sha256(args.metrics_csv.read_bytes()).hexdigest()[:16],
    }
    (args.out_dir / "outcome_manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
