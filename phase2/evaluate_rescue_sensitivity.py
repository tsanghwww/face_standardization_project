"""Evaluate FAN-crop versus whole-image rescue domain shift on paired external samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import xgboost as xgb
from insightface.app import FaceAnalysis
from scipy.io import loadmat

from .evaluate_rendered_outputs import aligned_render_embedding, gaze_on, gaze_vector
from .features import sample_from_mat
from .rebuild_xgboost import FEATURE_COLUMNS


PROJECT = Path(r"D:\face_standardization_project")
EXTERNAL_GROUPS = {"wider_pose", "wider_occlusion", "wider_blur", "cofw_occlusion", "aflw_large_pose"}
NUMERIC_FIELDS = [
    "shape_rmse", "expression_rmse", "pose_l2", "head_pose_l2", "jaw_pose_l2",
    "camera_l2", "light_rmse", "detail_rmse", "landmark_score_main",
    "landmark_score_rescue", "xgb_score_main", "xgb_score_rescue",
    "xgb_score_delta_rescue_minus_main", "arcface_cosine_main_vs_rescue",
    "l2cs_gaze_delta_main_vs_rescue_deg",
]
FIELDS = [
    "eval_id", "image_id", "source_group", "source_dataset", "main_mat_path",
    "rescue_mat_path", "main_render_path", "rescue_render_path", "status",
    *NUMERIC_FIELDS, "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv")
    parser.add_argument("--main-dir", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "deca_params")
    parser.add_argument("--rescue-dir", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "deca_params_rescue")
    parser.add_argument("--external-arcface", type=Path, default=PROJECT / "results" / "phase2_arcface_external_20260824" / "arcface_external_manifest.csv")
    parser.add_argument("--xgb-model", type=Path, default=PROJECT / "results" / "phase2_xgb_rebuilt_20260824" / "xgb_final_model.json")
    parser.add_argument("--deca-root", type=Path, default=PROJECT / "DECA")
    parser.add_argument("--l2cs-weights", type=Path, default=PROJECT / "models" / "l2cs" / "L2CSNet_gaze360.pkl")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "results" / "phase2_rescue_sensitivity_20260826")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def as_vector(data: dict, key: str, size: int) -> np.ndarray:
    value = np.asarray(data[key], dtype=np.float32).reshape(-1)
    if value.size != size or not np.isfinite(value).all():
        raise ValueError(f"invalid_{key}:{value.shape}")
    return value


def load_params(path: Path) -> dict[str, np.ndarray]:
    data = loadmat(path)
    return {
        "shape": as_vector(data, "shape", 100),
        "expression": as_vector(data, "expression", 50),
        "pose": as_vector(data, "pose", 6),
        "camera": as_vector(data, "camera", 3),
        "light": as_vector(data, "light", 27),
        "detail": as_vector(data, "detail", 128),
    }


def codedict_from(params: dict[str, np.ndarray], device: str) -> dict[str, torch.Tensor]:
    tensor = lambda value: torch.from_numpy(value).to(device).float()[None, :]
    return {
        "shape": tensor(params["shape"]),
        "exp": tensor(params["expression"]),
        "pose": tensor(params["pose"]),
        "cam": tensor(params["camera"]),
        "light": tensor(params["light"]).reshape(1, 9, 3),
        "detail": tensor(params["detail"]),
        "images": torch.zeros((1, 3, 224, 224), dtype=torch.float32, device=device),
    }


def save_render(deca, params: dict[str, np.ndarray], path: Path, device: str, util) -> None:
    with torch.no_grad():
        _opdict, visual = deca.decode(codedict_from(params, device), rendering=True, return_vis=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), util.tensor2image(visual["shape_detail_images"][0])):
        raise RuntimeError("render_write_failed")


def rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def xgb_score(booster: xgb.Booster, mat_path: Path, arcface_row: dict[str, str] | None) -> tuple[float, float]:
    sample = sample_from_mat(mat_path, arcface_row)
    features = np.asarray([[float(sample.metrics[name]) for name in FEATURE_COLUMNS]], dtype=np.float32)
    score = float(booster.predict(xgb.DMatrix(features, feature_names=FEATURE_COLUMNS))[0])
    return score, float(sample.metrics["landmark_score"])


def bootstrap(values: np.ndarray, seed: int, n_boot: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    for index in range(n_boot):
        means[index] = values[rng.integers(0, values.size, values.size)].mean()
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def aggregate(rows: list[dict[str, str]], seed: int, n_boot: int) -> list[dict]:
    groups = {"all": list(rows)}
    for group in sorted({row["source_group"] for row in rows}):
        groups[group] = [row for row in rows if row["source_group"] == group]
    output = []
    for group, group_rows in groups.items():
        for metric in NUMERIC_FIELDS:
            values = np.asarray([float(row[metric]) for row in group_rows if row.get(metric, "")], dtype=np.float64)
            low, high = bootstrap(values, seed, n_boot)
            output.append({
                "source_group": group,
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()) if values.size else float("nan"),
                "median": float(np.median(values)) if values.size else float("nan"),
                "p90": float(np.percentile(values, 90)) if values.size else float("nan"),
                "p95": float(np.percentile(values, 95)) if values.size else float("nan"),
                "ci95_lo": low,
                "ci95_hi": high,
            })
    return output


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.deca_root.resolve()))
    from decalib.deca import DECA
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg
    from l2cs import Pipeline

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = False
    deca_cfg.rasterizer_type = "standard"
    deca = DECA(config=deca_cfg, device=device, render_enabled=True)
    arcface = FaceAnalysis(name="buffalo_l")
    arcface.prepare(ctx_id=-1, det_thresh=0.1, det_size=(640, 640))
    gaze = Pipeline(weights=args.l2cs_weights, arch="ResNet50", device=torch.device(device), confidence_threshold=0.5)
    booster = xgb.Booster()
    booster.load_model(args.xgb_model)
    arcface_rows = {row["image_id"]: row for row in read_csv(args.external_arcface)}

    candidates = []
    for row in read_csv(args.manifest):
        if row["source_group"] not in EXTERNAL_GROUPS:
            continue
        main_mat = args.main_dir / row["eval_id"] / f"{row['image_id']}.mat"
        rescue_mat = args.rescue_dir / row["eval_id"] / f"{row['image_id']}.mat"
        if main_mat.exists() and rescue_mat.exists():
            candidates.append((row, main_mat, rescue_mat))
    if args.limit:
        candidates = candidates[: args.limit]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "main_vs_rescue_paired_metrics.csv"
    rows = read_csv(output_path) if args.resume and output_path.exists() else []
    completed = {row["eval_id"] for row in rows}

    for index, (source, main_mat, rescue_mat) in enumerate(candidates, start=1):
        if source["eval_id"] in completed:
            continue
        row = {field: "" for field in FIELDS}
        row.update({
            "eval_id": source["eval_id"],
            "image_id": source["image_id"],
            "source_group": source["source_group"],
            "source_dataset": source["source_dataset"],
            "main_mat_path": str(main_mat),
            "rescue_mat_path": str(rescue_mat),
            "status": "fail",
        })
        try:
            main_params = load_params(main_mat)
            rescue_params = load_params(rescue_mat)
            row.update({
                "shape_rmse": f"{rmse(main_params['shape'], rescue_params['shape']):.8f}",
                "expression_rmse": f"{rmse(main_params['expression'], rescue_params['expression']):.8f}",
                "pose_l2": f"{float(np.linalg.norm(main_params['pose'] - rescue_params['pose'])):.8f}",
                "head_pose_l2": f"{float(np.linalg.norm(main_params['pose'][:3] - rescue_params['pose'][:3])):.8f}",
                "jaw_pose_l2": f"{float(np.linalg.norm(main_params['pose'][3:] - rescue_params['pose'][3:])):.8f}",
                "camera_l2": f"{float(np.linalg.norm(main_params['camera'] - rescue_params['camera'])):.8f}",
                "light_rmse": f"{rmse(main_params['light'], rescue_params['light']):.8f}",
                "detail_rmse": f"{rmse(main_params['detail'], rescue_params['detail']):.8f}",
            })
            main_score, main_landmark = xgb_score(booster, main_mat, arcface_rows.get(source["image_id"]))
            rescue_score, rescue_landmark = xgb_score(booster, rescue_mat, arcface_rows.get(source["image_id"]))
            row.update({
                "landmark_score_main": f"{main_landmark:.8f}",
                "landmark_score_rescue": f"{rescue_landmark:.8f}",
                "xgb_score_main": f"{main_score:.8f}",
                "xgb_score_rescue": f"{rescue_score:.8f}",
                "xgb_score_delta_rescue_minus_main": f"{rescue_score - main_score:.8f}",
            })

            render_dir = args.out_dir / "renders" / source["eval_id"]
            main_render = render_dir / "main_original.jpg"
            rescue_render = render_dir / "rescue_original.jpg"
            save_render(deca, main_params, main_render, device, util)
            save_render(deca, rescue_params, rescue_render, device, util)
            row["main_render_path"] = str(main_render)
            row["rescue_render_path"] = str(rescue_render)

            main_emb, main_status, _ = aligned_render_embedding(arcface, main_render)
            rescue_emb, rescue_status, _ = aligned_render_embedding(arcface, rescue_render)
            if main_emb is None or rescue_emb is None:
                raise RuntimeError(f"arcface:{main_status}/{rescue_status}")
            row["arcface_cosine_main_vs_rescue"] = f"{float(np.dot(main_emb, rescue_emb)):.8f}"

            main_pitch, main_yaw, main_gaze_status = gaze_on(gaze, main_render)
            rescue_pitch, rescue_yaw, rescue_gaze_status = gaze_on(gaze, rescue_render)
            if main_pitch is not None and rescue_pitch is not None:
                cosine = float(np.clip(np.dot(gaze_vector(main_pitch, main_yaw), gaze_vector(rescue_pitch, rescue_yaw)), -1.0, 1.0))
                row["l2cs_gaze_delta_main_vs_rescue_deg"] = f"{math.degrees(math.acos(cosine)):.8f}"
            else:
                raise RuntimeError(f"l2cs:{main_gaze_status}/{rescue_gaze_status}")
            row["status"] = "success"
        except Exception as exc:  # noqa: BLE001
            row["failure_reason"] = f"{type(exc).__name__}:{exc}"
        rows.append(row)
        if index % 10 == 0 or index == len(candidates):
            write_csv(output_path, rows)
            print(f"paired={index}/{len(candidates)}", flush=True)

    write_csv(output_path, rows)
    summary_rows = aggregate(rows, args.seed, args.bootstrap_samples)
    summary_path = args.out_dir / "main_vs_rescue_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    summary = {
        "expected_pairs": len(candidates),
        "rows": len(rows),
        "success": sum(row["status"] == "success" for row in rows),
        "failed": sum(row["status"] != "success" for row in rows),
        "paired_metrics": str(output_path),
        "summary": str(summary_path),
    }
    (args.out_dir / "rescue_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
