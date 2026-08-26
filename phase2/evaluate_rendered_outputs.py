"""Measure identity, detected pose, gaze, and failures on batch-rendered Phase2 outputs.

This module computes per-sample metrics for selected render methods and
aggregates them with explicit failure accounting. Failure-rate denominators
are always the total number of samples
that should be evaluated (the fixed test manifest), never the number of
successfully rendered samples.

Scope note: ArcFace cosine, L2CS gaze, and DECA pose measured on DECA
shape-detail renders are *diagnostic rendering-quality* metrics.  They are NOT
a photorealistic-domain identity or gaze ground truth.  All such metrics are
flagged ``diagnostic`` in the outputs.

Outputs (in --out-dir):
  rendered_metrics.csv              per-sample, per-method raw metrics
  metrics_by_method.csv             aggregated by render method
  metrics_by_source_group.csv       aggregated by source_group
  metrics_by_source_dataset.csv     aggregated by source_dataset
  metrics_by_xgb_quality_label.csv  aggregated by xgb_quality_label
  metrics_by_method_and_group.csv   aggregated by method x source_group
  metrics_by_method_and_dataset.csv aggregated by method x source_dataset
  metrics_by_method_and_xgb.csv     aggregated by method x xgb quality label
  failure_analysis.csv              every failed (eval_id, method) with reason
  rendered_metrics_summary.json     nested summary + diagnostic scope note
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis

from .run_fixed_external_deca import load_image, whole_image_tensor

FIELDS = [
    "eval_id", "source_group", "source_dataset", "xgb_quality_label", "method",
    "render_status", "arcface_status", "arcface_reference", "arcface_cosine", "deca_status",
    "deca_head_pose_norm", "deca_pose_delta_vs_original",
    "l2cs_status", "l2cs_pitch", "l2cs_yaw",
    "l2cs_gaze_delta_vs_source_deg", "l2cs_gaze_delta_vs_original_deg",
    "diagnostic", "failure_reason",
]

DEFAULT_METHODS = ("original", "hard_zero", "phase2")

DIAGNOSTIC_NOTE = (
    "ArcFace cosine, L2CS gaze, and DECA pose metrics are measured on DECA "
    "shape-detail renders and are diagnostic rendering-quality metrics, not a "
    "photorealistic-domain identity or gaze ground truth."
)

STAT_COLUMNS = [
    "n_total", "n_rendered", "render_failure_rate",
    "arcface_success_rate", "arcface_failure_rate",
    "arcface_cosine_mean", "arcface_cosine_median", "arcface_cosine_std",
    "arcface_cosine_p10", "arcface_cosine_p90", "arcface_cosine_ci95_lo", "arcface_cosine_ci95_hi", "arcface_cosine_n",
    "deca_pose_norm_mean", "deca_pose_norm_median", "deca_pose_norm_p10", "deca_pose_norm_p90",
    "deca_pose_norm_ci95_lo", "deca_pose_norm_ci95_hi", "deca_pose_norm_n",
    "deca_pose_delta_vs_original_mean", "deca_pose_delta_vs_original_ci95_lo", "deca_pose_delta_vs_original_ci95_hi", "deca_pose_delta_vs_original_n",
    "l2cs_gaze_delta_vs_source_deg_mean", "l2cs_gaze_delta_vs_source_deg_ci95_lo", "l2cs_gaze_delta_vs_source_deg_ci95_hi", "l2cs_gaze_delta_vs_source_deg_n",
    "l2cs_gaze_delta_vs_original_deg_mean", "l2cs_gaze_delta_vs_original_deg_ci95_lo", "l2cs_gaze_delta_vs_original_deg_ci95_hi", "l2cs_gaze_delta_vs_original_deg_n",
    "l2cs_failure_rate", "deca_failure_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--render-manifest", required=True, type=Path)
    parser.add_argument("--xgb-manifest", type=Path)
    parser.add_argument("--deca-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--l2cs-weights", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _f(value) -> float | None:
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def face_embedding(app: FaceAnalysis, image_path: Path) -> tuple[np.ndarray | None, str, str]:
    """Return (embedding, status, detail). status is success/no_face_detected/
    multi_face/image_read_failed. Multi-face still returns the largest face's
    embedding (flagged), never silently dropping the observation."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "image_read_failed", ""
    faces = app.get(image)
    if not faces:
        return None, "no_face_detected", ""
    if len(faces) > 1:
        face = max(faces, key=lambda value: float((value.bbox[2] - value.bbox[0]) * (value.bbox[3] - value.bbox[1])))
        return np.asarray(face.normed_embedding, dtype=np.float32), "multi_face", f"multi_face_detected:{len(faces)}"
    return np.asarray(faces[0].normed_embedding, dtype=np.float32), "success", ""


def aligned_render_embedding(app: FaceAnalysis, image_path: Path) -> tuple[np.ndarray | None, str, str]:
    """Embed an already aligned square DECA render without face detection."""
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "image_read_failed", ""
    aligned = cv2.resize(image, (112, 112), interpolation=cv2.INTER_AREA)
    try:
        embedding = np.asarray(app.models["recognition"].get_feat(aligned), dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if not math.isfinite(norm) or norm <= 0:
            return None, "invalid_embedding", ""
        return embedding / norm, "success", ""
    except Exception as exc:  # noqa: BLE001
        return None, "embedding_failed", type(exc).__name__


def gaze_vector(pitch: float, yaw: float) -> np.ndarray:
    return np.asarray([-math.sin(yaw) * math.cos(pitch), -math.sin(pitch), -math.cos(yaw) * math.cos(pitch)])


def gaze_on(gaze, image_path: Path) -> tuple[float | None, float | None, str]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None, "image_read_failed"
    height, width = image.shape[:2]
    if max(height, width) > 640:
        scale = 640.0 / max(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    try:
        result = gaze.step(image)
    except Exception as exc:  # noqa: BLE001
        return None, None, type(exc).__name__
    if not np.asarray(result.bboxes).size:
        return None, None, "no_face_detected"
    pitch = float(np.asarray(result.pitch).reshape(-1)[0])
    yaw = float(np.asarray(result.yaw).reshape(-1)[0])
    return pitch, yaw, "success"


def status_key(method: str) -> str:
    return f"{method}_status"


def path_key(method: str) -> str:
    return f"{method}_render_path"


def _group_by_eval_id(eval_ids: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Collapse (eval_id, value) pairs into one value per unique eval_id."""
    uniq, inv = np.unique(eval_ids, return_inverse=True)
    out = np.empty(uniq.size, dtype=np.float64)
    for k in range(uniq.size):
        out[k] = np.nanmean(values[inv == k])
    return out


def paired_bootstrap_mean(eval_ids: np.ndarray, values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    """Bootstrap the mean resampling *eval_ids* (paired), preserving within-id structure."""
    if eval_ids.size == 0:
        return float("nan"), float("nan")
    per_id = _group_by_eval_id(eval_ids, values)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = per_id[rng.integers(0, per_id.size, per_id.size)]
        means[i] = float(np.nanmean(sample))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap_prop(eval_ids: np.ndarray, mask: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    """Bootstrap a proportion resampling *eval_ids* (paired)."""
    if eval_ids.size == 0:
        return float("nan"), float("nan")
    per_id = _group_by_eval_id(eval_ids, mask.astype(np.float64))
    rng = np.random.default_rng(seed)
    props = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        sample = per_id[rng.integers(0, per_id.size, per_id.size)]
        props[i] = float(np.nanmean(sample))
    lo, hi = np.percentile(props, [2.5, 97.5])
    return float(lo), float(hi)


def pct(values: np.ndarray, p: float) -> float:
    return float(np.nanpercentile(values, p)) if values.size else float("nan")


def metric_block(eval_ids: np.ndarray, values: np.ndarray, seed: int, n_boot: int) -> dict[str, float | int]:
    finite_mask = np.isfinite(values)
    e = eval_ids[finite_mask]
    v = values[finite_mask]
    lo, hi = paired_bootstrap_mean(e, v, seed, n_boot)
    return {
        "mean": float(np.nanmean(v)) if v.size else float("nan"),
        "median": float(np.nanmedian(v)) if v.size else float("nan"),
        "std": float(np.nanstd(v)) if v.size else float("nan"),
        "p10": pct(v, 10),
        "p90": pct(v, 90),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "n": int(v.size),
    }


def _collect(rows: list[dict], key: str) -> tuple[np.ndarray, np.ndarray]:
    eids: list[str] = []
    vals: list[float] = []
    for r in rows:
        if not r.get(key):
            continue
        v = _f(r[key])
        if v is not None:
            eids.append(r["eval_id"])
            vals.append(v)
    return np.asarray(eids), np.asarray(vals, dtype=np.float64)


def compute_stats(rows: list[dict], n_total: int, seed: int, n_boot: int) -> dict:
    if n_total <= 0:
        return {}
    n_rendered = sum(1 for r in rows if r["render_status"] == "success")
    all_eids = np.asarray([r["eval_id"] for r in rows])
    arc_ok = np.asarray([1.0 if r.get("arcface_cosine") else 0.0 for r in rows])
    deca_fail = np.asarray([1.0 if r.get("deca_status") != "success" else 0.0 for r in rows])
    l2cs_fail = np.asarray([1.0 if r.get("l2cs_status") != "success" else 0.0 for r in rows])

    cos_e, cos_v = _collect(rows, "arcface_cosine")
    pose_e, pose_v = _collect(rows, "deca_head_pose_norm")
    delta_e, delta_v = _collect(rows, "deca_pose_delta_vs_original")
    gsrc_e, gsrc_v = _collect(rows, "l2cs_gaze_delta_vs_source_deg")
    gorig_e, gorig_v = _collect(rows, "l2cs_gaze_delta_vs_original_deg")

    def rate(mask) -> float:
        return float(mask.mean()) if mask.size else float("nan")

    arc_lo, arc_hi = paired_bootstrap_prop(all_eids, arc_ok, seed, n_boot)
    return {
        "n_total": n_total,
        "n_rendered": n_rendered,
        "render_failure_rate": round((n_total - n_rendered) / n_total, 6),
        "arcface_success_rate": round(rate(arc_ok), 6),
        "arcface_failure_rate": round(1.0 - rate(arc_ok), 6),
        "arcface_cosine": metric_block(cos_e, cos_v, seed, n_boot),
        "arcface_cosine_ci95": [arc_lo, arc_hi],
        "deca_head_pose_norm": metric_block(pose_e, pose_v, seed, n_boot),
        "deca_pose_delta_vs_original": metric_block(delta_e, delta_v, seed, n_boot),
        "l2cs_gaze_delta_vs_source_deg": metric_block(gsrc_e, gsrc_v, seed, n_boot),
        "l2cs_gaze_delta_vs_original_deg": metric_block(gorig_e, gorig_v, seed, n_boot),
        "l2cs_failure_rate": round(rate(l2cs_fail), 6),
        "deca_failure_rate": round(rate(deca_fail), 6),
    }


def flatten_stats(stats: dict) -> dict:
    """Flatten the nested metric blocks into a flat row matching STAT_COLUMNS."""
    row: dict[str, float | int | str] = {}
    row["n_total"] = stats.get("n_total", "")
    row["n_rendered"] = stats.get("n_rendered", "")
    row["render_failure_rate"] = stats.get("render_failure_rate", "")
    row["arcface_success_rate"] = stats.get("arcface_success_rate", "")
    row["arcface_failure_rate"] = stats.get("arcface_failure_rate", "")
    c = stats.get("arcface_cosine", {})
    row["arcface_cosine_mean"] = c.get("mean", "")
    row["arcface_cosine_median"] = c.get("median", "")
    row["arcface_cosine_std"] = c.get("std", "")
    row["arcface_cosine_p10"] = c.get("p10", "")
    row["arcface_cosine_p90"] = c.get("p90", "")
    row["arcface_cosine_ci95_lo"] = c.get("ci95_lo", "")
    row["arcface_cosine_ci95_hi"] = c.get("ci95_hi", "")
    row["arcface_cosine_n"] = c.get("n", "")
    d = stats.get("deca_head_pose_norm", {})
    row["deca_pose_norm_mean"] = d.get("mean", "")
    row["deca_pose_norm_median"] = d.get("median", "")
    row["deca_pose_norm_p10"] = d.get("p10", "")
    row["deca_pose_norm_p90"] = d.get("p90", "")
    row["deca_pose_norm_ci95_lo"] = d.get("ci95_lo", "")
    row["deca_pose_norm_ci95_hi"] = d.get("ci95_hi", "")
    row["deca_pose_norm_n"] = d.get("n", "")
    dd = stats.get("deca_pose_delta_vs_original", {})
    row["deca_pose_delta_vs_original_mean"] = dd.get("mean", "")
    row["deca_pose_delta_vs_original_ci95_lo"] = dd.get("ci95_lo", "")
    row["deca_pose_delta_vs_original_ci95_hi"] = dd.get("ci95_hi", "")
    row["deca_pose_delta_vs_original_n"] = dd.get("n", "")
    gs = stats.get("l2cs_gaze_delta_vs_source_deg", {})
    row["l2cs_gaze_delta_vs_source_deg_mean"] = gs.get("mean", "")
    row["l2cs_gaze_delta_vs_source_deg_ci95_lo"] = gs.get("ci95_lo", "")
    row["l2cs_gaze_delta_vs_source_deg_ci95_hi"] = gs.get("ci95_hi", "")
    row["l2cs_gaze_delta_vs_source_deg_n"] = gs.get("n", "")
    go = stats.get("l2cs_gaze_delta_vs_original_deg", {})
    row["l2cs_gaze_delta_vs_original_deg_mean"] = go.get("mean", "")
    row["l2cs_gaze_delta_vs_original_deg_ci95_lo"] = go.get("ci95_lo", "")
    row["l2cs_gaze_delta_vs_original_deg_ci95_hi"] = go.get("ci95_hi", "")
    row["l2cs_gaze_delta_vs_original_deg_n"] = go.get("n", "")
    row["l2cs_failure_rate"] = stats.get("l2cs_failure_rate", "")
    row["deca_failure_rate"] = stats.get("deca_failure_rate", "")
    return row


def write_group_csv(path: Path, key_fields: list[str], groups: list[tuple[dict, dict]]) -> None:
    columns = key_fields + STAT_COLUMNS
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for key, stats in groups:
            flat = flatten_stats(stats)
            flat.update(key)
            writer.writerow(flat)


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.deca_root))
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg
    from l2cs import Pipeline

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    deca_cfg.model.use_tex = False
    deca = DECA(config=deca_cfg, device=device, render_enabled=False)
    arcface = FaceAnalysis(name="buffalo_l")
    arcface.prepare(ctx_id=-1, det_thresh=0.1, det_size=(640, 640))
    gaze = Pipeline(weights=args.l2cs_weights, arch="ResNet50", device=torch.device(device), confidence_threshold=0.5) if args.l2cs_weights else None

    test_rows = read_csv(args.test_manifest)
    render_rows = {row["eval_id"]: row for row in read_csv(args.render_manifest)}
    methods = tuple(method.strip() for method in args.methods.split(",") if method.strip())
    if not methods:
        raise SystemExit("--methods must contain at least one method")
    xgb_rows = {
        row["image_id"]: row
        for row in (read_csv(args.xgb_manifest) if args.xgb_manifest else [])
        if row.get("image_id")
    }
    if args.limit:
        test_rows = test_rows[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "rendered_metrics.csv"
    completed: set[tuple[str, str]] = set()
    rows: list[dict[str, str]] = []
    if args.resume and output_path.exists():
        rows = read_csv(output_path)
        # Source-photo gaze is not a stable reference for crowded external
        # images. Keep the legacy column empty and report render-vs-original.
        for row in rows:
            row["l2cs_gaze_delta_vs_source_deg"] = ""
        completed = {(row["eval_id"], row["method"]) for row in rows}

    # per-eval_id caches
    orig_emb_cache: dict[str, tuple[np.ndarray | None, str]] = {}
    orig_gaze_cache: dict[str, tuple[float | None, float | None]] = {}
    orig_pose_cache: dict[str, float | None] = {}

    def deca_pose_on(image_path: Path) -> tuple[float | None, str]:
        try:
            image = load_image(image_path)
            tensor = torch.from_numpy(whole_image_tensor(image)).to(device)[None, ...]
            with torch.no_grad():
                codedict = deca.encode(tensor)
            return float(torch.linalg.vector_norm(codedict["pose"][0, :3])), "success"
        except Exception as exc:  # noqa: BLE001
            return None, type(exc).__name__

    for eval_index, eval_id_row in enumerate(test_rows):
        eval_id = eval_id_row["eval_id"]
        test = eval_id_row
        render = render_rows.get(eval_id)

        for method in methods:
            key = (eval_id, method)
            if key in completed:
                continue
            row = {field: "" for field in FIELDS}
            row.update({
                "eval_id": eval_id,
                "source_group": test.get("source_group", ""),
                "source_dataset": test.get("source_dataset", ""),
                "xgb_quality_label": (xgb_rows.get(test.get("image_id", "")) or {}).get(
                    "xgb_quality_label", test.get("xgb_quality_label", "")
                ),
                "method": method,
                "diagnostic": "1",
            })
            if render is None:
                row["render_status"] = "not_rendered"
                row["failure_reason"] = "render_missing"
                rows.append(row)
                continue
            status = render.get(status_key(method), "")
            if status != "success":
                row["render_status"] = status or "fail"
                row["failure_reason"] = (render.get("failure_reason") or "render_failed") + f"::{method}"
                rows.append(row)
                continue
            row["render_status"] = "success"
            image_path = Path(render[path_key(method)])

            # ArcFace identity (diagnostic)
            embedding, arc_status, arc_detail = aligned_render_embedding(arcface, image_path)
            row["arcface_status"] = arc_status
            row["arcface_reference"] = "original_render"
            reasons = [arc_detail] if arc_detail else []
            if method == "original":
                orig_emb_cache[eval_id] = (embedding, arc_status)
            elif eval_id not in orig_emb_cache:
                orig_path = render.get(path_key("original"))
                if orig_path and render.get(status_key("original")) == "success":
                    orig_emb, orig_status, _ = aligned_render_embedding(arcface, Path(orig_path))
                    orig_emb_cache[eval_id] = (orig_emb, orig_status)
            orig_emb, orig_arc_status = orig_emb_cache.get(eval_id, (None, "unavailable"))
            if embedding is not None and orig_emb is not None:
                row["arcface_cosine"] = f"{float(np.dot(orig_emb, embedding)):.6f}"
            elif orig_emb is None:
                reasons.append(f"original_render_arcface={orig_arc_status}")
            if arc_status != "success":
                reasons.append(f"render_arcface={arc_status}")

            # DECA pose on render (diagnostic)
            pose_norm, pose_status = deca_pose_on(image_path)
            if pose_norm is not None:
                row["deca_status"] = "success"
                row["deca_head_pose_norm"] = f"{pose_norm:.6f}"
                if method == "original":
                    orig_pose_cache[eval_id] = pose_norm
            else:
                row["deca_status"] = "fail"
                reasons.append(f"deca:{pose_status}")

            # L2CS gaze (diagnostic)
            if gaze is not None:
                pitch, yaw, gstatus = gaze_on(gaze, image_path)
                if gstatus != "success":
                    row["l2cs_status"] = "fail"
                    reasons.append(f"l2cs:{gstatus}")
                else:
                    row["l2cs_status"] = "success"
                    row["l2cs_pitch"] = f"{pitch:.6f}"
                    row["l2cs_yaw"] = f"{yaw:.6f}"
                    if method == "original":
                        orig_gaze_cache[eval_id] = (pitch, yaw)
                    # vs original render
                    if method != "original":
                        orig_path = render.get(path_key("original"))
                        if orig_path and render.get(status_key("original")) == "success":
                            if eval_id not in orig_gaze_cache:
                                orig_gaze_cache[eval_id] = gaze_on(gaze, Path(orig_path))[:2]
                            op, oy = orig_gaze_cache[eval_id]
                            if op is not None and oy is not None:
                                ang2 = math.degrees(math.acos(float(np.clip(np.dot(gaze_vector(pitch, yaw), gaze_vector(op, oy)), -1.0, 1.0))))
                                row["l2cs_gaze_delta_vs_original_deg"] = f"{ang2:.6f}"
            else:
                row["l2cs_status"] = "fail"
                reasons.append("l2cs:weights_not_configured")

            # deca pose delta vs original render
            if method != "original" and row["deca_status"] == "success":
                orig_path = render.get(path_key("original"))
                if orig_path and render.get(status_key("original")) == "success":
                    if eval_id not in orig_pose_cache:
                        orig_pose_cache[eval_id] = deca_pose_on(Path(orig_path))[0]
                    orig_norm = orig_pose_cache[eval_id]
                    if orig_norm is not None:
                        row["deca_pose_delta_vs_original"] = f"{abs(float(row['deca_head_pose_norm']) - orig_norm):.6f}"
                    else:
                        reasons.append("deca_orig_delta:unavailable")

            if reasons:
                row["failure_reason"] = ";".join(reasons)
            rows.append(row)

        if (eval_index + 1) % 10 == 0:
            with output_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            print(f"evaluated {eval_index + 1}/{len(test_rows)} eval_ids", flush=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ---- aggregation ----
    n_eval = len(test_rows)
    by_method: dict[str, list[dict]] = {m: [] for m in methods}
    by_group: dict[str, list[dict]] = defaultdict(list)
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    by_xgb: dict[str, list[dict]] = defaultdict(list)
    by_method_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_method_dataset: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_method_xgb: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)
        by_group[row["source_group"] or "unknown"].append(row)
        by_dataset[row["source_dataset"] or "unknown"].append(row)
        by_xgb[row["xgb_quality_label"] or "unknown"].append(row)
        by_method_group[(row["method"], row["source_group"] or "unknown")].append(row)
        by_method_dataset[(row["method"], row["source_dataset"] or "unknown")].append(row)
        by_method_xgb[(row["method"], row["xgb_quality_label"] or "unknown")].append(row)

    stats_by_method = [(m, compute_stats(by_method[m], n_eval, args.seed, args.bootstrap_samples)) for m in methods]
    stats_by_group = sorted(
        ((g, compute_stats(rows_g, len(rows_g), args.seed, args.bootstrap_samples)) for g, rows_g in by_group.items()),
        key=lambda t: t[0],
    )
    stats_by_dataset = sorted(
        ((d, compute_stats(rows_d, len(rows_d), args.seed, args.bootstrap_samples)) for d, rows_d in by_dataset.items()),
        key=lambda t: t[0],
    )
    stats_by_xgb = sorted(
        ((x, compute_stats(rows_x, len(rows_x), args.seed, args.bootstrap_samples)) for x, rows_x in by_xgb.items()),
        key=lambda t: t[0],
    )
    stats_by_mg = sorted(
        (((m, g), compute_stats(rows_mg, len(rows_mg), args.seed, args.bootstrap_samples)) for (m, g), rows_mg in by_method_group.items()),
        key=lambda t: (t[0][0], t[0][1]),
    )
    stats_by_md = sorted(
        (((m, d), compute_stats(rows_md, len(rows_md), args.seed, args.bootstrap_samples)) for (m, d), rows_md in by_method_dataset.items()),
        key=lambda t: (t[0][0], t[0][1]),
    )
    stats_by_mx = sorted(
        (((m, x), compute_stats(rows_mx, len(rows_mx), args.seed, args.bootstrap_samples)) for (m, x), rows_mx in by_method_xgb.items()),
        key=lambda t: (t[0][0], t[0][1]),
    )

    write_group_csv(args.out_dir / "metrics_by_method.csv", ["method"], [({"method": m}, s) for m, s in stats_by_method])
    write_group_csv(args.out_dir / "metrics_by_source_group.csv", ["source_group"], [({"source_group": g}, s) for g, s in stats_by_group])
    write_group_csv(args.out_dir / "metrics_by_source_dataset.csv", ["source_dataset"], [({"source_dataset": d}, s) for d, s in stats_by_dataset])
    write_group_csv(args.out_dir / "metrics_by_xgb_quality_label.csv", ["xgb_quality_label"], [({"xgb_quality_label": x}, s) for x, s in stats_by_xgb])
    write_group_csv(args.out_dir / "metrics_by_method_and_group.csv", ["method", "source_group"], [({"method": m, "source_group": g}, s) for (m, g), s in stats_by_mg])
    write_group_csv(args.out_dir / "metrics_by_method_and_dataset.csv", ["method", "source_dataset"], [({"method": m, "source_dataset": d}, s) for (m, d), s in stats_by_md])
    write_group_csv(args.out_dir / "metrics_by_method_and_xgb_quality_label.csv", ["method", "xgb_quality_label"], [({"method": m, "xgb_quality_label": x}, s) for (m, x), s in stats_by_mx])

    # ---- failure analysis ----
    failures = [r for r in rows if r["render_status"] != "success" or r.get("failure_reason")]
    with (args.out_dir / "failure_analysis.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eval_id", "source_group", "method", "render_status", "arcface_status", "deca_status", "l2cs_status", "failure_reason"])
        writer.writeheader()
        for r in failures:
            writer.writerow({k: r.get(k, "") for k in ["eval_id", "source_group", "method", "render_status", "arcface_status", "deca_status", "l2cs_status", "failure_reason"]})

    summary = {
        "diagnostic_scope_note": DIAGNOSTIC_NOTE,
        "n_eval_samples": n_eval,
        "n_metric_rows": len(rows),
        "by_method": {m: s for m, s in stats_by_method},
        "by_source_group": {g: s for g, s in stats_by_group},
        "by_source_dataset": {d: s for d, s in stats_by_dataset},
        "by_xgb_quality_label": {x: s for x, s in stats_by_xgb},
        "by_method_and_group": {f"{m}__{g}": s for (m, g), s in stats_by_mg},
        "failure_count": len(failures),
    }
    (args.out_dir / "rendered_metrics_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"n_eval_samples": n_eval, "n_metric_rows": len(rows), "failure_count": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
