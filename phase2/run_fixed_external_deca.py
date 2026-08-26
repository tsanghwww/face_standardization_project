"""Run DECA parameter extraction for the 375 external samples of the fixed test v2.

Two explicit modes, never silently mixed:

  * fan    (main): FAN face detection -> tight crop -> DECA.  If FAN detects no
            face, the sample is recorded as a *failure* (fan_detected=false,
            fallback_used=false, failure_reason="fan_no_face") and NO .mat is
            produced -- there is no silent whole-image fallback.
  * rescue (comparison only): whole-image warp (iscrop=False) -> DECA.  Results
            are written to a separate directory and never substitute the main
            fan evaluation.

Per-sample status (fan_detected, fallback_used, failure_reason, landmark_score,
head_pose_norm, elapsed) is written to <output-dir>/deca_sample_status.csv and
the JSON checkpoint.  Keyed by unique eval_id (directory) and image_id (mat
file name), matching the downstream inference/render join on image_id.

Usage:
  python -m phase2.run_fixed_external_deca --mode fan --eval-ids <6 ids>
  python -m phase2.run_fixed_external_deca --mode rescue --output-dir ...rescue --eval-ids <6 ids>
  python -m phase2.run_fixed_external_deca --mode fan --resume --update-manifest   # full 375
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat
from skimage.io import imread
from skimage.transform import estimate_transform, warp

PROJECT = Path(r"D:\face_standardization_project")
EXTERNAL_GROUPS = {"wider_pose", "wider_occlusion", "wider_blur", "cofw_occlusion", "aflw_large_pose"}
REQUIRED_PARAMS = {"shape": 100, "expression": 50, "pose": 6, "camera": 3}

STATUS_FIELDS = [
    "eval_id", "image_id", "mode", "fan_detected", "fallback_used", "success",
    "failure_reason", "elapsed_s", "landmark_score", "head_pose_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / "fixed_test_manifest_v2.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--deca-root", type=Path, default=PROJECT / "DECA")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--mode", default="fan", choices=["fan", "rescue"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--eval-ids", type=str, default="")
    parser.add_argument("--update-manifest", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path: Path) -> np.ndarray:
    image = np.array(imread(str(path)))
    if image.ndim == 2:
        image = image[:, :, None].repeat(3, axis=2)
    if image.ndim == 3 and image.shape[2] > 3:
        image = image[:, :, :3]
    return image


def bbox2point(left: float, right: float, top: float, bottom: float, bbox_type: str) -> tuple[float, np.ndarray]:
    if bbox_type == "kpt68":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    else:
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
    return old_size, center


def crop_to_tensor(image: np.ndarray, bbox: list, bbox_type: str, scale: float = 1.25) -> np.ndarray:
    left, right, top, bottom = bbox[0], bbox[2], bbox[1], bbox[3]
    old_size, center = bbox2point(left, right, top, bottom, bbox_type)
    size = int(old_size * scale)
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2], [center[0] + size / 2, center[1] - size / 2]])
    dst_pts = np.array([[0, 0], [0, 223], [223, 0]])
    tform = estimate_transform("similarity", src_pts, dst_pts)
    img = image / 255.0
    dst = warp(img, tform.inverse, output_shape=(224, 224))
    return dst.transpose(2, 0, 1).astype(np.float32)


def whole_image_tensor(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape
    src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])
    dst_pts = np.array([[0, 0], [0, 223], [223, 0]])
    tform = estimate_transform("similarity", src_pts, dst_pts)
    img = image / 255.0
    dst = warp(img, tform.inverse, output_shape=(224, 224))
    return dst.transpose(2, 0, 1).astype(np.float32)


def extract_param_outputs(codedict: dict) -> dict[str, np.ndarray]:
    mapping = {"shape": "shape", "exp": "expression", "pose": "pose", "cam": "camera", "light": "light", "detail": "detail"}
    return {target: np.asarray(codedict[source].detach().cpu().numpy()) for source, target in mapping.items() if source in codedict}


def validate_mat(mat_path: Path) -> tuple[bool, str]:
    try:
        data = loadmat(str(mat_path))
    except Exception as exc:  # noqa: BLE001
        return False, f"reload_failed:{type(exc).__name__}:{exc}"
    for key, size in REQUIRED_PARAMS.items():
        if key not in data:
            return False, f"missing_param:{key}"
        arr = np.asarray(data[key])
        if arr.size != size:
            return False, f"bad_size:{key}={arr.shape}"
        if not np.isfinite(arr).all():
            return False, f"non_finite:{key}"
    for key, value in data.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.dtype.kind == "f" and not np.isfinite(arr).all():
            return False, f"non_finite:{key}"
    return True, "ok"


def landmark_score_from(mat_path: Path) -> float | None:
    from phase2.features import read_kpt_quality

    return read_kpt_quality(mat_path, image_size=224.0)["landmark_score"]


def main() -> None:
    args = parse_args()
    deca_root = args.deca_root.resolve()
    sys.path.insert(0, str(deca_root))
    import face_alignment

    from decalib.deca import DECA
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg

    if args.output_dir is None:
        args.output_dir = PROJECT / "results" / "phase2_eval_fixed_20260824_v2" / ("deca_params" if args.mode == "fan" else "deca_params_rescue")

    with args.manifest.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
        fieldnames = list(rows[0].keys()) if rows else []

    externals = [r for r in rows if r["source_group"] in EXTERNAL_GROUPS]
    if args.eval_ids:
        wanted = {e.strip() for e in args.eval_ids.split(",") if e.strip()}
        externals = [r for r in externals if r["eval_id"] in wanted]
    if args.limit:
        externals = externals[: args.limit]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)

    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = "standard"
    deca_cfg.model.extract_tex = False
    deca = DECA(config=deca_cfg, device=device, render_enabled=False)

    class _FAN:
        """FAN wrapper with torch.compile disabled (avoids a torch inductor
        nvcc-subprocess crash that can occur during compilation)."""

        def __init__(self) -> None:
            self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, compile=False)

        def run(self, image):
            out = self.model.get_landmarks(image)
            if out is None:
                return [0], "kpt68"
            kpt = out[0].squeeze()
            return [float(np.min(kpt[:, 0])), float(np.min(kpt[:, 1])), float(np.max(kpt[:, 0])), float(np.max(kpt[:, 1]))], "kpt68"

    fan = _FAN() if args.mode == "fan" else None

    status: dict[str, dict] = {}
    success = failed = skipped = 0
    times: list[float] = []
    t_start = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    for i, row in enumerate(externals):
        eval_id = row["eval_id"]
        image_id = row["image_id"]
        sample_dir = output_dir / eval_id
        mat_path = sample_dir / f"{image_id}.mat"
        if args.resume and mat_path.exists():
            ok, reason = validate_mat(mat_path)
            if ok:
                skipped += 1
                status[eval_id] = {"eval_id": eval_id, "image_id": image_id, "mode": args.mode, "fan_detected": "", "fallback_used": "false", "success": "true", "failure_reason": "", "elapsed_s": "0.0", "landmark_score": "", "head_pose_norm": ""}
                _checkpoint(args, output_dir, status, success, failed, skipped, len(externals))
                continue
            mat_path.unlink(missing_ok=True)

        t0 = time.perf_counter()
        entry = {"eval_id": eval_id, "image_id": image_id, "mode": args.mode, "fan_detected": "false", "fallback_used": "false", "success": "false", "failure_reason": "", "elapsed_s": "0.0", "landmark_score": "", "head_pose_norm": ""}
        try:
            image = load_image(Path(row["image_path"]))
            if args.mode == "fan":
                bbox, bbox_type = fan.run(image)
                if len(bbox) < 4:
                    raise RuntimeError("fan_no_face")
                entry["fan_detected"] = "true"
                tensor = crop_to_tensor(image, bbox, bbox_type)
            else:
                tensor = whole_image_tensor(image)
            images = torch.from_numpy(tensor).to(device)[None, ...]
            with torch.no_grad():
                codedict = deca.encode(images)
                opdict = deca.decode(codedict, rendering=False, return_vis=False)
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(sample_dir / f"{image_id}_kpt2d.txt", opdict["landmarks2d"][0].detach().cpu().numpy())
            np.savetxt(sample_dir / f"{image_id}_kpt3d.txt", opdict["landmarks3d"][0].detach().cpu().numpy())
            matdict = util.dict_tensor2npy(opdict)
            matdict.update(extract_param_outputs(codedict))
            savemat(mat_path, matdict)
            ok, reason = validate_mat(mat_path)
            if not ok:
                raise RuntimeError(f"validate_failed:{reason}")
            entry["success"] = "true"
            entry["head_pose_norm"] = f"{float(torch.linalg.vector_norm(codedict['pose'][0, :3])):.6f}"
            try:
                entry["landmark_score"] = f"{landmark_score_from(mat_path):.6f}"
            except Exception:  # noqa: BLE001
                entry["landmark_score"] = ""
        except Exception as exc:  # noqa: BLE001
            entry["failure_reason"] = f"{type(exc).__name__}:{exc}"
            if entry["failure_reason"].startswith("RuntimeError:fan_no_face"):
                entry["failure_reason"] = "fan_no_face"
            for leftover in sample_dir.glob(f"{image_id}*"):
                leftover.unlink()

        entry["elapsed_s"] = f"{time.perf_counter() - t0:.3f}"
        times.append(time.perf_counter() - t0)
        if entry["success"] == "true":
            success += 1
        else:
            failed += 1
        status[eval_id] = entry
        if (success + failed) % args.progress_every == 0 or i == len(externals) - 1:
            _checkpoint(args, output_dir, status, success, failed, skipped, len(externals))

    if args.update_manifest and args.mode == "fan":
        update_manifest_mat_path(args.manifest, output_dir, externals, fieldnames)

    wall = time.perf_counter() - t_start
    peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    summary = {
        "mode": args.mode,
        "total": len(externals),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "wall_seconds": round(wall, 2),
        "per_image_seconds": {
            "mean": round(float(np.mean(times)), 3) if times else 0.0,
            "median": round(float(np.median(times)), 3) if times else 0.0,
            "min": round(float(np.min(times)), 3) if times else 0.0,
            "max": round(float(np.max(times)), 3) if times else 0.0,
        },
        "gpu_peak_mb": round(peak_mb, 1),
        "device": device,
        "output_dir": str(output_dir),
    }
    (output_dir / "deca_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _checkpoint(args, output_dir, status, success, failed, skipped, total):
    (output_dir / "deca_progress.json").write_text(json.dumps({"total": total, "success": success, "failed": failed, "skipped": skipped, "completed": success + failed + skipped, "samples": status}, indent=2), encoding="utf-8")
    with (output_dir / "deca_sample_status.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(status.values())
    with (output_dir / "deca_failures.csv").open("w", encoding="utf-8", newline="") as f:
        f.write("eval_id,failure_reason\n")
        for eid, st in status.items():
            if st["success"] == "false":
                f.write(f"{eid},{st['failure_reason']}\n")


def update_manifest_mat_path(manifest_path: Path, output_dir: Path, externals: list[dict], fieldnames: list[str]) -> None:
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    target = {r["eval_id"]: str(output_dir / r["eval_id"] / f"{r['image_id']}.mat") for r in externals}
    for row in rows:
        if row["eval_id"] in target:
            row["mat_path"] = target[row["eval_id"]]
    tmp = manifest_path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(manifest_path)
    print(f"manifest mat_path updated for {len(target)} external rows -> {manifest_path}")


if __name__ == "__main__":
    main()
