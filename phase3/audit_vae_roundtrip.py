#!/usr/bin/env python3
"""Phase3.0B: frozen VAE round-trip audit (32 validation samples).

Measures the identity/head/gaze drift introduced by a frozen AutoencoderKL on
32 pose-stratified *validation* samples (never fixed test / external / rescue).

Core frozen-VAE protocol:
  x_preprocessed = Lanczos resize(source, 256)
  latent          = vae.encode(x).latent_dist.mode() * scaling_factor      # posterior mode(), NEVER sample()
  reconstruction  = vae.decode(latent / scaling_factor).sample
  reconstruction  = clamp(reconstruction, -1, 1) -> uint8

All metrics compare source_preprocessed vs reconstruction (so resize error is
not attributed to the VAE).  ArcFace/DECA/L2CS are re-predicted on BOTH images
with the same entry point/config.  DECA uses the SAME FAN crop as the Phase2
runner (phase2.run_fixed_external_deca.crop_to_tensor), no whole-image fallback.
Missing/failed evaluators stay empty with an explicit status (never 0-filled)
and count in the full 32-sample denominator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
COORDINATE_STATUS = "candidate_unvalidated_diagnostic_only"
DECA_PREPROCESS = "fan"  # FAN detect+crop via phase2.run_fixed_external_deca.crop_to_tensor

CSV_FIELDS = [
    "image_id", "split", "source_image", "source_preprocessed", "reconstruction",
    "vae_status", "failure_reason", "psnr_rgb", "ssim_rgb", "lpips", "lpips_status",
    "arcface_source_status", "arcface_recon_status", "arcface_cosine",
    "arcface_source_det_score", "arcface_recon_det_score",
    "deca_source_status", "deca_recon_status", "head_pose_delta_deg",
    "l2cs_source_status", "l2cs_recon_status", "gaze_camera_delta_deg",
    "gaze_head_delta_deg", "gaze_coordinate_status",
    "runtime_seconds", "gpu_peak_mb",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-manifest", required=True, type=Path)
    p.add_argument("--validation-ids", required=True, type=Path)
    p.add_argument("--fixed-test-ids", required=True, type=Path)
    p.add_argument("--gaze-candidates", required=True, type=Path)
    p.add_argument("--selection-count", type=int, default=32)
    p.add_argument("--selection-seed", type=int, default=20260901)
    p.add_argument("--vae-model", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--vae-path", type=Path, default=None)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    p.add_argument("--arcface-mode", default="off", choices=["off", "existing"])
    p.add_argument("--arcface-det-thresh", type=float, default=0.1)
    p.add_argument("--deca-mode", default="off", choices=["off", "existing"])
    p.add_argument("--l2cs-mode", default="off", choices=["off", "existing"])
    p.add_argument("--deca-root", type=Path, default=PROJECT / "DECA")
    p.add_argument("--l2cs-weights", type=Path, default=PROJECT / "models" / "l2cs" / "L2CSNet_gaze360.pkl")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------ selection
def select_audit_ids(gaze_csv: Path, validation_ids_file: Path, fixed_test_ids_file: Path, count: int, out_dir: Path) -> list[str]:
    sys.path.insert(0, str(PROJECT / "scripts"))
    from select_phase30_coordinate_audit_ids import select_pose_stratified

    with gaze_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "validation" and r["status"] == "candidate_unvalidated"]
    if count <= 0:
        raise ValueError(f"selection count must be positive, got {count}")
    if len(rows) < count:
        raise ValueError(f"only {len(rows)} eligible validation candidates for requested count={count}")
    selected, reasons = select_pose_stratified(rows, count)
    ids = [r["image_id"] for r in selected]
    if len(ids) != count:
        raise RuntimeError(f"selector returned {len(ids)} IDs for requested count={count}")

    validation = {ln.strip() for ln in validation_ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
    fixed = {ln.strip() for ln in fixed_test_ids_file.read_text(encoding="utf-8").splitlines() if ln.strip()}
    if len(set(ids)) != len(ids):
        raise ValueError("selected IDs are not unique")
    if not all(i in validation for i in ids):
        raise ValueError("selected ID not in validation set")
    if set(ids) & fixed:
        raise ValueError("selected ID leaks into fixed test")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vae_audit_ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    payload = {
        "count": len(ids),
        "split": "validation",
        "ids": ids,
        "selection": [
            {"image_id": r["image_id"], "pose": [float(r[f"pose_{a}"]) for a in "xyz"], "reasons": reasons[r["image_id"]]}
            for r in selected
        ],
        "fixed_test_overlap": len(set(ids) & fixed),
        "gaze_candidates_sha256": hashlib.sha256(gaze_csv.read_bytes()).hexdigest(),
    }
    (out_dir / "vae_audit_selection.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return ids


# ----------------------------------------------------------- image + VAE ops
def preprocess_source(source_path: Path, out_path: Path, resolution: int = 256) -> np.ndarray:
    from PIL import Image

    im = Image.open(source_path).convert("RGB")
    im = im.resize((resolution, resolution), Image.Resampling.LANCZOS)
    im.save(out_path)
    return np.asarray(im).astype(np.uint8)


def image_to_vae_input(rgb_uint8: np.ndarray, device: str, dtype: str):
    import torch

    x = rgb_uint8.astype(np.float32) / 127.5 - 1.0
    x = np.transpose(x, (2, 0, 1))
    t = torch.from_numpy(x).unsqueeze(0).to(device)
    if dtype == "fp16" and device != "cpu":
        t = t.half()
    return t


def vae_output_to_image(tensor) -> np.ndarray:
    import torch

    x = tensor.detach().float().clamp(-1.0, 1.0)[0].cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    return (x * 127.5 + 127.5).clip(0, 255).astype(np.uint8)


def encode_decode(vae, x):
    latent = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
    reconstruction = vae.decode(latent / vae.config.scaling_factor).sample
    return reconstruction


def freeze_model(model) -> None:
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def load_vae(vae_model: str, vae_path: Path | None, device: str, dtype: str):
    import torch
    from diffusers import AutoencoderKL

    torch_dtype = torch.float16 if dtype == "fp16" and device != "cpu" else torch.float32
    if vae_path is not None:
        vae = AutoencoderKL.from_pretrained(str(vae_path), torch_dtype=torch_dtype)
        resolved = str(vae_path)
    else:
        vae = AutoencoderKL.from_pretrained(vae_model, torch_dtype=torch_dtype)
        resolved = vae_model
    vae = vae.to(device)
    freeze_model(vae)
    return vae, float(vae.config.scaling_factor), resolved


# ------------------------------------------------------------------ metrics
def psnr_rgb(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else float(20.0 * math.log10(255.0) - 10.0 * math.log10(mse))


def ssim_rgb(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity as ssim

    return float(ssim(a, b, channel_axis=2, data_range=255))


def geodesic_angle_deg(rot_a, rot_b) -> float:
    from scipy.spatial.transform import Rotation

    ra = Rotation.from_rotvec(np.asarray(rot_a, dtype=np.float64))
    rb = Rotation.from_rotvec(np.asarray(rot_b, dtype=np.float64))
    return float((rb * ra.inv()).magnitude()) * 180.0 / math.pi


# --------------------------------------------------------------- evaluators
def load_arcface(det_thresh: float = 0.1):
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))
    if "detection" in app.models:
        app.models["detection"].det_thresh = det_thresh
    return app


def arcface_pair(app, src_img: np.ndarray, rec_img: np.ndarray):
    import cv2

    def detect(rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces:
            return None, None, "no_face_detected"
        face = max(faces, key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])))
        det = getattr(face, "det_score", None)
        det = float(det) if det is not None and np.isfinite(det) else None
        return np.asarray(face.normed_embedding, dtype=np.float32), det, "success"

    es, ds, ss = detect(src_img)
    er, dr, rs = detect(rec_img)
    cosine = float(np.dot(es, er)) if (es is not None and er is not None) else None
    return cosine, ss, rs, (f"{ds:.6f}" if ds is not None else ""), (f"{dr:.6f}" if dr is not None else "")


def load_fan():
    import face_alignment

    class FAN:
        def __init__(self):
            self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, compile=False)

        def run(self, image):
            out = self.model.get_landmarks(image)
            if out is None:
                return [0], "kpt68"
            kpt = out[0].squeeze()
            return [float(np.min(kpt[:, 0])), float(np.min(kpt[:, 1])), float(np.max(kpt[:, 0])), float(np.max(kpt[:, 1]))], "kpt68"

    return FAN()


def load_deca(deca_root: Path, device: str):
    sys.path.insert(0, str(deca_root))
    from decalib.deca import DECA
    from decalib.utils.config import cfg as deca_cfg

    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = "standard"
    deca_cfg.model.extract_tex = False
    deca = DECA(config=deca_cfg, device=device, render_enabled=False)
    freeze_model(deca)
    return deca


def load_l2cs(weights: Path, device: str):
    import torch
    from l2cs import Pipeline

    return Pipeline(weights=weights, arch="ResNet50", device=torch.device(device), confidence_threshold=0.5)


# ------------------------------------------------------------------ audit
def build_row(image_id: str, split: str, source_image: str, source_preprocessed: str, reconstruction: str,
              vae_status: str, failure_reason: str, psnr: str, ssim: str, lpips: str, lpips_status: str,
              arcface_cosine: str, arcface_src_status: str, arcface_rec_status: str,
              arcface_src_det: str, arcface_rec_det: str,
              head_delta: str, deca_src_status: str, deca_rec_status: str,
              gaze_delta: str, l2cs_src_status: str, l2cs_rec_status: str,
              runtime_s: float, gpu_mb: float) -> dict:
    return {
        "image_id": image_id, "split": split, "source_image": source_image,
        "source_preprocessed": source_preprocessed, "reconstruction": reconstruction,
        "vae_status": vae_status, "failure_reason": failure_reason,
        "psnr_rgb": psnr, "ssim_rgb": ssim, "lpips": lpips, "lpips_status": lpips_status,
        "arcface_source_status": arcface_src_status, "arcface_recon_status": arcface_rec_status, "arcface_cosine": arcface_cosine,
        "arcface_source_det_score": arcface_src_det, "arcface_recon_det_score": arcface_rec_det,
        "deca_source_status": deca_src_status, "deca_recon_status": deca_rec_status, "head_pose_delta_deg": head_delta,
        "l2cs_source_status": l2cs_src_status, "l2cs_recon_status": l2cs_rec_status, "gaze_camera_delta_deg": gaze_delta,
        "gaze_head_delta_deg": "",
        "gaze_coordinate_status": COORDINATE_STATUS,
        "runtime_seconds": f"{runtime_s:.3f}", "gpu_peak_mb": f"{gpu_mb:.1f}",
    }


def _num(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _stat(values):
    vals = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if vals.size == 0:
        return {"count": 0, "mean": None, "median": None, "p05": None, "p95": None}
    return {"count": int(vals.size), "mean": float(vals.mean()), "median": float(np.median(vals)),
            "p05": float(np.percentile(vals, 5)), "p95": float(np.percentile(vals, 95))}


def summarize(rows: list[dict], meta: dict) -> dict:
    n_total = meta["n_total"]

    def count(pred):
        return int(sum(1 for r in rows if pred(r)))

    def nonempty(v):
        return v not in ("", None)

    def rate(n):
        return n / n_total if n_total else None

    arc_src_ok = count(lambda r: r.get("arcface_source_status") == "success")
    arc_rec_ok = count(lambda r: r.get("arcface_recon_status") == "success")
    arc_pair_ok = count(lambda r: nonempty(r.get("arcface_cosine")))

    runtimes = [_num(r.get("runtime_seconds")) for r in rows]
    rt = np.asarray([v for v in runtimes if v is not None], dtype=np.float64)

    summary = {
        "n_total": n_total,
        "rows": len(rows),
        "vae": {"success": count(lambda r: r["vae_status"] == "success"),
                "failed": count(lambda r: r["vae_status"] == "fail"),
                "coverage": rate(count(lambda r: r["vae_status"] == "success"))},
        "arcface": {"success": arc_pair_ok, "failed": count(lambda r: r.get("arcface_source_status") not in ("", "success") or r.get("arcface_recon_status") not in ("", "success")),
                    "coverage": rate(arc_pair_ok),
                    "source_coverage": rate(arc_src_ok), "reconstruction_coverage": rate(arc_rec_ok), "pair_coverage": rate(arc_pair_ok)},
        "deca": {"success": count(lambda r: nonempty(r.get("head_pose_delta_deg"))),
                 "failed": count(lambda r: r.get("deca_source_status") not in ("", "success") or r.get("deca_recon_status") not in ("", "success")),
                 "coverage": rate(count(lambda r: nonempty(r.get("head_pose_delta_deg"))))},
        "l2cs": {"success": count(lambda r: nonempty(r.get("gaze_camera_delta_deg"))),
                 "failed": count(lambda r: r.get("l2cs_source_status") not in ("", "success") or r.get("l2cs_recon_status") not in ("", "success")),
                 "coverage": rate(count(lambda r: nonempty(r.get("gaze_camera_delta_deg"))))},
        "psnr_rgb": _stat([_num(r["psnr_rgb"]) for r in rows]),
        "ssim_rgb": _stat([_num(r["ssim_rgb"]) for r in rows]),
        "lpips": _stat([_num(r["lpips"]) for r in rows]),
        "arcface_cosine": _stat([_num(r["arcface_cosine"]) for r in rows]),
        "head_pose_delta_deg": _stat([_num(r["head_pose_delta_deg"]) for r in rows]),
        "gaze_camera_delta_deg": _stat([_num(r["gaze_camera_delta_deg"]) for r in rows]),
        "sample_runtime_seconds": _stat(runtimes),
        "current_run_wall_seconds": meta.get("current_run_wall_seconds"),
        "cumulative_sample_runtime_seconds": round(float(rt.sum()), 3) if rt.size else None,
        "deca_preprocess": meta.get("deca_preprocess", DECA_PREPROCESS),
        "arcface_det_thresh": meta.get("arcface_det_thresh"),
        "selection_hash": meta.get("selection_hash"),
        "base_manifest_sha256": meta.get("base_manifest_sha256"),
        "vae_resolved": meta.get("vae_resolved"),
        "vae_snapshot_sha256": meta.get("vae_snapshot_sha256"),
        "scaling_factor": meta.get("scaling_factor"),
        "head_local_gaze_not_evaluated": True,
        "gaze_coordinate_status": COORDINATE_STATUS,
        "fixed_test_overlap": meta.get("fixed_test_overlap", 0),
        "split": meta.get("split", "validation"),
        "gpu_peak_mb": meta.get("gpu_peak_mb"),
    }
    return summary


def _selection_hash(selection_dir: Path) -> str:
    return hashlib.sha256((selection_dir / "vae_audit_ids.txt").read_bytes()).hexdigest()


def _tree_sha256(root: Path | None) -> str | None:
    """Hash file names and contents so an in-place model replacement is detected."""
    if root is None or not root.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


RESUME_CHECK_FIELDS = (
    "selection_hash", "base_manifest_sha256", "vae_resolved", "vae_snapshot_sha256",
    "resolution", "dtype", "scaling_factor", "arcface_mode", "deca_mode",
    "l2cs_mode", "arcface_det_thresh", "deca_preprocess",
)


def verify_resume_config(prev: dict, cur: dict) -> None:
    """Fail closed when resume config/selection differs from the current run."""
    for k in RESUME_CHECK_FIELDS:
        if prev.get(k) != cur.get(k):
            raise RuntimeError(f"resume config mismatch: {k} = {cur.get(k)!r} vs previous {prev.get(k)!r}")


def main() -> None:
    args = parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "exact_command.txt").write_text(" ".join([sys.executable, "-m", "phase3.audit_vae_roundtrip", *sys.argv[1:]]) + "\n", encoding="utf-8")

    base = {r["image_id"]: r for r in csv.DictReader(open(args.base_manifest, encoding="utf-8-sig", newline=""))}
    ids = select_audit_ids(args.gaze_candidates, args.validation_ids, args.fixed_test_ids, args.selection_count, out / "selection")
    sel_hash = _selection_hash(out / "selection")

    config = {
        "vae_model": args.vae_model, "vae_path": str(args.vae_path) if args.vae_path else None,
        "resolution": args.resolution, "device": args.device, "dtype": args.dtype,
        "resize_filter": "lanczos", "selection_count": args.selection_count, "selection_seed": args.selection_seed,
        "selection_strategy": "deterministic_pose_extrema_no_rng",
        "selection_hash": sel_hash,
        "arcface_mode": args.arcface_mode, "arcface_det_thresh": args.arcface_det_thresh,
        "deca_mode": args.deca_mode, "l2cs_mode": args.l2cs_mode,
        "deca_preprocess": DECA_PREPROCESS,
        "coordinate_status": COORDINATE_STATUS,
        "base_manifest_sha256": hashlib.sha256(args.base_manifest.read_bytes()).hexdigest(),
        "vae_snapshot_sha256": _tree_sha256(args.vae_path),
    }
    if args.dry_run:
        (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
        print(json.dumps({"dry_run": True, "selected": len(ids), "ids": ids}, indent=2))
        return

    import torch

    device = args.device if torch.cuda.is_available() else "cpu"
    vae, scaling, resolved = load_vae(args.vae_model, args.vae_path, device, args.dtype)
    config["vae_resolved"] = resolved
    config["scaling_factor"] = scaling

    # resume: fail closed on config / selection mismatch
    if args.resume:
        prev_cfg_path = out / "config.json"
        if prev_cfg_path.exists():
            verify_resume_config(json.loads(prev_cfg_path.read_text(encoding="utf-8")), config)

    evaluators = {}
    if args.arcface_mode == "existing":
        evaluators["arcface"] = load_arcface(args.arcface_det_thresh)
    if args.deca_mode == "existing":
        evaluators["deca"] = load_deca(args.deca_root, device)
        evaluators["fan"] = load_fan()
    if args.l2cs_mode == "existing":
        evaluators["l2cs"] = load_l2cs(args.l2cs_weights, device)

    lpips_model = load_lpips(device)

    metrics_path = out / "vae_roundtrip_metrics.csv"
    done = load_completed(metrics_path, ids) if args.resume else {}
    rows = list(done.values())
    t_start = time.time()
    peak_mb = 0.0
    for image_id in ids:
        if image_id in done:
            continue
        row = base.get(image_id)
        if row is None:
            rows.append(build_row(image_id, "validation", "", "", "", "fail", "base_manifest_missing", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "", 0.0, peak_mb))
            continue
        source_image = row.get("image_path", "")
        src_pre = out / "source_preprocessed" / f"{image_id}.png"
        rec_path = out / "reconstructed" / f"{image_id}.png"
        src_pre.parent.mkdir(parents=True, exist_ok=True)
        rec_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        try:
            src_rgb = preprocess_source(PROJECT / source_image if not Path(source_image).is_absolute() else Path(source_image), src_pre, args.resolution)
            x = image_to_vae_input(src_rgb, device, args.dtype)
            with torch.no_grad():
                recon_t = encode_decode(vae, x)
            rec_rgb = vae_output_to_image(recon_t)
            from PIL import Image

            Image.fromarray(rec_rgb).save(rec_path)
            p = f"{psnr_rgb(src_rgb, rec_rgb):.4f}"
            s = f"{ssim_rgb(src_rgb, rec_rgb):.6f}"
            vae_status = "success"
            failure = ""
        except Exception as exc:  # noqa: BLE001
            p = s = ""
            vae_status = "fail"
            failure = f"{type(exc).__name__}:{exc}"

        lp, lp_status = "", "not_available"
        cos, a_src, a_rec, a_sdet, a_rdet = "", "", "", "", ""
        hd, d_src, d_rec = "", "", ""
        gd, l_src, l_rec = "", "", ""

        if vae_status == "success":
            if lpips_model is not None:
                try:
                    lp = f"{lpips_distance(lpips_model, src_rgb, rec_rgb):.6f}"
                    lp_status = "available"
                except Exception as exc:  # noqa: BLE001
                    lp, lp_status = "", f"error:{type(exc).__name__}"
            if args.arcface_mode == "existing":
                cos, a_src, a_rec, a_sdet, a_rdet = arcface_pair(evaluators["arcface"], src_rgb, rec_rgb)
                cos = "" if cos is None else f"{cos:.6f}"
            if args.deca_mode == "existing":
                hd, d_src, d_rec = deca_pair(evaluators["deca"], evaluators["fan"], device, src_rgb, rec_rgb)
            if args.l2cs_mode == "existing":
                gd, l_src, l_rec = l2cs_pair(evaluators["l2cs"], src_rgb, rec_rgb)

        if torch.cuda.is_available():
            peak_mb = max(peak_mb, torch.cuda.max_memory_allocated() / 1e6)
        rows.append(build_row(image_id, "validation", source_image, str(src_pre), str(rec_path), vae_status, failure,
                              p, s, lp, lp_status, cos, a_src, a_rec, a_sdet, a_rdet, hd, d_src, d_rec, gd, l_src, l_rec,
                              time.time() - t0, peak_mb))
        if (len(rows) - len(done)) % 4 == 0:
            _write_metrics(metrics_path, rows)

    _write_metrics(metrics_path, rows)
    if len(rows) != len(ids):
        raise RuntimeError(f"row count {len(rows)} != n_total {len(ids)}")
    meta = {
        "n_total": len(ids),
        "current_run_wall_seconds": round(time.time() - t_start, 2),
        "gpu_peak_mb": round(peak_mb, 1),
        "fixed_test_overlap": 0,
        "split": "validation",
        "deca_preprocess": DECA_PREPROCESS,
        "arcface_det_thresh": args.arcface_det_thresh,
        "selection_hash": config["selection_hash"],
        "base_manifest_sha256": config["base_manifest_sha256"],
        "vae_resolved": config["vae_resolved"],
        "vae_snapshot_sha256": config["vae_snapshot_sha256"],
        "scaling_factor": config["scaling_factor"],
    }
    summary = summarize(rows, meta)
    (out / "vae_roundtrip_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    failures = [r for r in rows if r["vae_status"] != "success"]
    with (out / "vae_roundtrip_failures.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "vae_status", "failure_reason"])
        w.writeheader()
        for r in failures:
            w.writerow({"image_id": r["image_id"], "vae_status": r["vae_status"], "failure_reason": r["failure_reason"]})
    _write_environment(out, config)
    _make_contact_sheet(out, rows)
    config["artifact_hashes_file"] = str(out / "artifact_hashes.sha256")
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _write_artifact_hashes(out, config, args.vae_path)
    print(json.dumps(summary, indent=2))


def _write_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def load_completed(metrics_path: Path, selection_ids: list[str]) -> dict[str, dict]:
    """Resume helper: return previous *successful* samples that are in the current selection."""
    if not metrics_path.exists():
        return {}
    sel = set(selection_ids)
    with metrics_path.open("r", encoding="utf-8", newline="") as f:
        return {r["image_id"]: r for r in csv.DictReader(f) if r.get("vae_status") == "success" and r["image_id"] in sel}


def deca_pair(deca, fan, device, src_rgb, rec_rgb):
    import torch

    from phase2.run_fixed_external_deca import crop_to_tensor

    def pose_of(rgb):
        bbox, bbox_type = fan.run(rgb)
        if len(bbox) < 4:
            raise RuntimeError("fan_no_face")
        tensor = crop_to_tensor(rgb, bbox, bbox_type)  # uint8 HxWx3 -> (3,224,224) float32
        t = torch.from_numpy(tensor.astype(np.float32)).to(device)[None, ...]
        with torch.no_grad():
            codedict = deca.encode(t)
        return codedict["pose"][0, :3].detach().cpu().numpy()

    try:
        a = pose_of(src_rgb)
        b = pose_of(rec_rgb)
        return f"{geodesic_angle_deg(a, b):.4f}", "success", "success"
    except Exception as exc:  # noqa: BLE001
        return "", f"{type(exc).__name__}", f"{type(exc).__name__}"


def l2cs_pair(gaze, src_rgb, rec_rgb):
    import cv2

    from phase2.evaluate_rendered_outputs import gaze_vector

    def pred(rgb):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        if max(h, w) > 640:
            s = 640.0 / max(h, w)
            bgr = cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        r = gaze.step(bgr)
        if not np.asarray(r.bboxes).size:
            return (None, None), "no_face_detected"
        return (float(np.asarray(r.pitch).reshape(-1)[0]), float(np.asarray(r.yaw).reshape(-1)[0])), "success"

    (p1, y1), s1 = pred(src_rgb)
    (p2, y2), s2 = pred(rec_rgb)
    if s1 == "success" and s2 == "success":
        v1, v2 = gaze_vector(p1, y1), gaze_vector(p2, y2)
        ang = math.degrees(math.acos(float(np.clip(np.dot(v1, v2), -1.0, 1.0))))
        return f"{ang:.4f}", "success", "success"
    return "", s1, s2


def _pkg_version(name: str) -> str | None:
    try:
        import importlib.metadata

        return importlib.metadata.version(name)
    except Exception:  # noqa: BLE001
        try:
            import importlib

            return importlib.import_module(name).__version__
        except Exception:  # noqa: BLE001
            return None


def load_lpips(device: str):
    try:
        import lpips

        model = lpips.LPIPS(net="alex").to(device)
        freeze_model(model)
        return model
    except Exception:  # noqa: BLE001
        return None


def lpips_distance(model, src_rgb: np.ndarray, rec_rgb: np.ndarray) -> float:
    import torch

    dev = next(model.parameters()).device

    def to_t(rgb):
        x = rgb.astype(np.float32) / 127.5 - 1.0
        return torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(dev)

    with torch.no_grad():
        d = model(to_t(src_rgb), to_t(rec_rgb))
    return float(d.item())


def _write_environment(out: Path, config: dict) -> None:
    import platform

    import torch

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": _pkg_version("torchvision"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6, 1) if torch.cuda.is_available() else None,
        "diffusers": _pkg_version("diffusers"),
        "safetensors": _pkg_version("safetensors"),
        "huggingface_hub": _pkg_version("huggingface_hub"),
        "transformers": _pkg_version("transformers"),
        "accelerate": _pkg_version("accelerate"),
        "lpips": _pkg_version("lpips"),
    }
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    config["environment"] = env


def _write_artifact_hashes(out: Path, config: dict, vae_path: Path | None) -> None:
    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    hashes: dict[str, str] = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "artifact_hashes.sha256":
            hashes[str(p.relative_to(out))] = sha(p)
    if vae_path is not None and vae_path.is_dir():
        for p in sorted(vae_path.rglob("*")):
            if p.is_file():
                hashes[f"vae_snapshot/{p.relative_to(vae_path)}"] = sha(p)
    lines = [f"{k}  {v}" for k, v in sorted(hashes.items())]
    (out / "artifact_hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_contact_sheet(out: Path, rows: list[dict]) -> None:
    from PIL import Image, ImageDraw

    contact_dir = out / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)
    cell = 96
    label_w = 150
    valid = []
    for r in rows:
        src = Path(r["source_preprocessed"]) if r.get("source_preprocessed") else None
        rec = Path(r["reconstruction"]) if r.get("reconstruction") else None
        if src and src.exists() and rec and rec.exists():
            valid.append((r, src, rec))
    if not valid:
        return
    sheet = Image.new("RGB", (label_w + 3 * cell, cell * len(valid)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (r, src, rec) in enumerate(valid):
        a = Image.open(src).convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        b = Image.open(rec).convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        diff = (np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)).mean(axis=2) * 10).clip(0, 255).astype(np.uint8)
        d = Image.fromarray(diff).convert("RGB")
        y = i * cell
        sheet.paste(a, (label_w, y))
        sheet.paste(b, (label_w + cell, y))
        sheet.paste(d, (label_w + 2 * cell, y))
        label = f"{r['image_id']} {r['vae_status']}\ncos={r.get('arcface_cosine') or 'NA'}\nhd={r.get('head_pose_delta_deg') or 'NA'}\ngd={r.get('gaze_camera_delta_deg') or 'NA'}"
        draw.text((6, y + 4), label, fill="black")
    sheet.save(contact_dir / "contact_sheet_all32.png")
    meta = {"samples": len(valid), "total": len(rows), "layout": "label|source|reconstruction|abs_diff", "deca_preprocess": DECA_PREPROCESS}
    (contact_dir / "contact_sheet_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
