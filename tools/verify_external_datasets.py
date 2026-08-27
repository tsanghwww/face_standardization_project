# -*- coding: utf-8 -*-
"""Extract & verify external evaluation datasets: AFLW2000-3D and 300W-LP.

Usage:
  python verify_external_datasets.py extract_aflw
  python verify_external_datasets.py extract_300wlp
  python verify_external_datasets.py verify_aflw
  python verify_external_datasets.py verify_300wlp

Extraction target layout:
  AFLW2000-3D/test.data/AFLW2000-3D_crop/*.jpg   (2,000 images, the AFLW2000-3D test set)
  AFLW2000-3D/test.data/AFLW_GT_crop/*.jpg       (21,080 images, informational)
  300W-LP/extracted/300W_LP/<SET>/*.jpg + *.mat  (122,450 images, 61,225 unique + flips)
"""
import json
import os
import random
import re
import sys
import zipfile

import numpy as np

ROOT = r"D:\face_standardization_project\datasets\external"
AFLW_DIR = os.path.join(ROOT, "AFLW2000-3D")
AFLW_ZIP = os.path.join(AFLW_DIR, "images", "test.data.zip")
WLP_DIR = os.path.join(ROOT, "300W-LP")
WLP_ZIP = os.path.join(WLP_DIR, "300W_LP.zip")
WLP_EXTRACT = os.path.join(WLP_DIR, "extracted")

EXPECT = {
    "aflw_images": 2000,          # AFLW2000-3D_crop images
    "aflw_gt_images": 21080,      # AFLW_GT_crop images (full AFLW crops)
    "wlp_jpg": 122450,            # all jpg (61,225 unique x2 with _Flip dirs)
    "wlp_unique": 61225,          # landmark mats in landmarks/ and image count /2
}


def log(msg):
    print(msg, flush=True)


def extract_zip(zip_path, dest, progress_every=2000):
    """Extract zip_path into dest; returns (ok, file_count, errors)."""
    errors = []
    n = 0
    try:
        z = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as e:
        return False, 0, [f"BadZipFile opening {zip_path}: {e}"]
    try:
        for info in z.infolist():
            if info.is_dir():
                continue
            out_path = os.path.join(dest, info.filename)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            try:
                with z.open(info) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
                n += 1
            except Exception as e:  # noqa: BLE001 - per-file failure, keep going
                errors.append(f"{info.filename}: {type(e).__name__}: {e}")
                if len(errors) > 50:
                    break
            if n % progress_every == 0:
                log(f"  extracted {n} files (errors so far: {len(errors)})")
    except zipfile.BadZipFile as e:
        return False, n, [f"BadZipFile during extraction: {e}"] + errors
    finally:
        z.close()
    return len(errors) == 0, n, errors


def cmd_extract_aflw():
    log(f"[AFLW] extracting {AFLW_ZIP} -> {AFLW_DIR}")
    os.makedirs(AFLW_DIR, exist_ok=True)
    ok, n, errors = extract_zip(AFLW_ZIP, AFLW_DIR, progress_every=2000)
    log(f"[AFLW] done: ok={ok} files={n} errors={len(errors)}")
    for e in errors[:20]:
        log(f"  ERR {e}")


def cmd_extract_300wlp():
    log(f"[300W-LP] extracting {WLP_ZIP} -> {WLP_EXTRACT}")
    os.makedirs(WLP_EXTRACT, exist_ok=True)
    ok, n, errors = extract_zip(WLP_ZIP, WLP_EXTRACT, progress_every=5000)
    log(f"[300W-LP] done: ok={ok} files={n} errors={len(errors)}")
    for e in errors[:20]:
        log(f"  ERR {e}")


# ---------------------------------------------------------------- AFLW2000-3D
def cmd_verify_aflw():
    report = {"dataset": "AFLW2000-3D", "checks": {}}
    crop_dir = os.path.join(AFLW_DIR, "test.data", "AFLW2000-3D_crop")
    gt_dir = os.path.join(AFLW_DIR, "test.data", "AFLW_GT_crop")
    ann_dir = os.path.join(AFLW_DIR, "annotations")

    imgs = sorted(f for f in os.listdir(crop_dir) if f.lower().endswith(".jpg"))
    gt_imgs = sorted(f for f in os.listdir(gt_dir) if f.lower().endswith(".jpg"))
    report["checks"]["crop_images"] = len(imgs)
    report["checks"]["gt_images"] = len(gt_imgs)
    report["checks"]["crop_images_ok"] = len(imgs) == EXPECT["aflw_images"]
    report["checks"]["gt_images_ok"] = len(gt_imgs) == EXPECT["aflw_gt_images"]

    list_path = os.path.join(AFLW_DIR, "test.data", "AFLW2000-3D_crop.list")
    if os.path.exists(list_path):
        with open(list_path, "r", encoding="utf-8") as f:
            list_lines = [ln.strip() for ln in f if ln.strip()]
        report["checks"]["list_lines"] = len(list_lines)
        report["checks"]["list_matches_images"] = (
            len(list_lines) == len(imgs)
            and all(os.path.basename(ln) in imgs for ln in list_lines)
        )
    else:
        report["checks"]["list_lines"] = None
        report["checks"]["list_matches_images"] = None

    # --- annotation .npy lengths (first dim must equal image count) ---
    npy_files = {
        "pose": "AFLW2000-3D.pose.npy",
        "pts68": "AFLW2000-3D.pts68.npy",
        "pts68_reannotated": "AFLW2000-3D-Reannotated.pts68.npy",
        "roi_box": "AFLW2000-3D_crop.roi_box.npy",
    }
    for key, fn in npy_files.items():
        p = os.path.join(ann_dir, fn)
        if not os.path.exists(p):
            report["checks"][key] = {"exists": False}
            continue
        try:
            arr = np.load(p)
            report["checks"][key] = {
                "exists": True,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "length": int(arr.shape[0]),
                "length_ok": int(arr.shape[0]) == EXPECT["aflw_images"],
            }
        except Exception as e:  # noqa: BLE001
            report["checks"][key] = {"exists": True, "load_error": str(e)}

    # sanity: pose value ranges should be degrees (roughly -100..100)
    pose_p = os.path.join(ann_dir, npy_files["pose"])
    if os.path.exists(pose_p):
        try:
            pose = np.load(pose_p)
            report["checks"]["pose_stats"] = {
                "min": float(np.nanmin(pose)),
                "max": float(np.nanmax(pose)),
                "mean": float(np.nanmean(pose)),
            }
        except Exception:  # noqa: BLE001
            pass

    report["pass"] = all(
        v is not False
        for k, v in report["checks"].items()
        if isinstance(v, dict) and ("length_ok" in v or "exists" in v and not v.get("exists"))
    ) and report["checks"]["crop_images_ok"] and report["checks"]["gt_images_ok"]
    # simpler: pass iff crop image count ok and every npy that exists has length_ok
    report["pass"] = report["checks"]["crop_images_ok"]
    for k in npy_files:
        c = report["checks"].get(k, {})
        if c.get("exists") and c.get("length_ok") is False:
            report["pass"] = False

    out = os.path.join(AFLW_DIR, "verify_aflw_report.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(json.dumps(report, indent=2))
    log(f"[AFLW] report written to {out}")
    return 0 if report["pass"] else 1


# ---------------------------------------------------------------- 300W-LP
def cmd_verify_300wlp(n_sample=100, seed=20260824):
    from PIL import Image

    try:
        from scipy.io import loadmat
    except ImportError:
        log("scipy not available in venv")
        return 1

    report = {"dataset": "300W-LP", "n_sample": n_sample, "seed": seed, "samples": []}
    base = os.path.join(WLP_EXTRACT, "300W_LP")
    if not os.path.isdir(base):
        log(f"[300W-LP] not extracted yet: {base} missing")
        return 1

    jpgs = []
    for dirpath, _dirs, files in os.walk(base):
        for fn in files:
            if fn.lower().endswith(".jpg"):
                jpgs.append(os.path.join(dirpath, fn))
    jpgs.sort()
    report["total_jpg"] = len(jpgs)
    report["total_jpg_ok"] = len(jpgs) == EXPECT["wlp_jpg"]

    # unique image count = half (each unique image appears in <SET> and <SET>_Flip)
    uniq_jpg = sum(1 for p in jpgs if re.search(r"(?:^|[\\/])(AFW|HELEN|IBUG|LFPW)[\\/]", p) is not None)
    report["unique_jpg_est"] = uniq_jpg

    landmarks_dir = os.path.join(base, "landmarks")
    landmarks_mats = 0
    for _dirpath, _dirs, files in os.walk(landmarks_dir):
        landmarks_mats += sum(1 for f in files if f.lower().endswith(".mat"))
    report["landmarks_mat_count"] = landmarks_mats
    report["landmarks_mat_count_ok"] = landmarks_mats == EXPECT["wlp_unique"]

    rng = random.Random(seed)
    sample = rng.sample(jpgs, min(n_sample, len(jpgs)))

    missing = []
    bad = []
    n_ok = 0
    for p in sample:
        d = os.path.dirname(p)
        stem = os.path.splitext(os.path.basename(p))[0]
        entry = {"image": os.path.relpath(p, base)}

        # 1) same-folder <basename>.mat: 2D landmarks (pt2d) + pose (Pose_Para)
        mat_path = os.path.join(d, stem + ".mat")
        entry["mat_found"] = os.path.exists(mat_path)
        if not os.path.exists(mat_path):
            missing.append(entry["image"])
            report["samples"].append(entry)
            continue
        entry["mat"] = os.path.relpath(mat_path, base)
        try:
            md = loadmat(mat_path)
            pt2d = md.get("pt2d")
            pose = md.get("Pose_Para")
            entry["pt2d_shape"] = None if pt2d is None else [int(x) for x in pt2d.shape]
            entry["pose_shape"] = None if pose is None else [int(x) for x in pose.shape]
            entry["pt2d_ok"] = pt2d is not None and pt2d.shape[0] == 2 and pt2d.shape[1] == 68
            entry["pose_ok"] = pose is not None and pose.size == 7
            if not (entry["pt2d_ok"] and entry["pose_ok"]):
                bad.append(entry["mat"])
        except Exception as e:  # noqa: BLE001
            entry["load_error"] = f"{type(e).__name__}: {e}"
            bad.append(entry["mat"])

        # 2) optional landmarks/<SET>/<stem>_pts.mat (exists for non-flip images):
        #    pts_2d / pts_3d 68-point landmarks
        pts_rel = os.path.relpath(d, base)          # e.g. "AFW" or "AFW_Flip"
        pts_path = os.path.join(landmarks_dir, pts_rel, stem + "_pts.mat")
        if os.path.exists(pts_path):
            entry["pts_mat"] = os.path.relpath(pts_path, base)
            try:
                mdp = loadmat(pts_path)
                p2 = mdp.get("pts_2d")
                p3 = mdp.get("pts_3d")
                entry["pts_2d_shape"] = None if p2 is None else [int(x) for x in p2.shape]
                entry["pts_3d_shape"] = None if p3 is None else [int(x) for x in p3.shape]
                entry["pts_mat_ok"] = (
                    p2 is not None and p2.shape == (68, 2)
                    and p3 is not None and p3.shape == (68, 2)
                )
                if not entry["pts_mat_ok"]:
                    bad.append(entry["pts_mat"])
            except Exception as e:  # noqa: BLE001
                entry["pts_mat_load_error"] = f"{type(e).__name__}: {e}"
                bad.append(entry["pts_mat"])
        else:
            entry["pts_mat"] = None

        # 3) image decodes
        try:
            with Image.open(p) as im:
                im.load()
                entry["image_ok"] = True
                entry["image_size"] = list(im.size)
        except Exception as e:  # noqa: BLE001
            entry["image_ok"] = False
            entry["image_error"] = f"{type(e).__name__}: {e}"
            bad.append(entry["image"])

        if entry.get("pt2d_ok") and entry.get("pose_ok") and entry.get("image_ok"):
            n_ok += 1
        report["samples"].append(entry)

    report["missing_mat"] = len(missing)
    report["bad_entries"] = len(bad)
    report["samples_ok"] = n_ok
    report["pass"] = (
        report["total_jpg_ok"]
        and report["landmarks_mat_count_ok"]
        and len(missing) == 0
        and len(bad) == 0
        and n_ok == len(sample)
    )

    out = os.path.join(WLP_DIR, "verify_300wlp_report.json")
    sample_out = os.path.join(WLP_DIR, "sample_100.txt")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(sample_out, "w", encoding="utf-8") as f:
        f.write("\n".join(s["image"] for s in report["samples"]) + "\n")
    log(json.dumps(report, indent=2))
    log(f"[300W-LP] report written to {out}")
    log(f"[300W-LP] sampled list written to {sample_out}")
    if missing:
        log("[300W-LP] first missing mats:")
        for m in missing[:10]:
            log(f"  {m}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "extract_aflw":
        sys.exit(cmd_extract_aflw())
    elif cmd == "extract_300wlp":
        sys.exit(cmd_extract_300wlp())
    elif cmd == "verify_aflw":
        sys.exit(cmd_verify_aflw())
    elif cmd == "verify_300wlp":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        sys.exit(cmd_verify_300wlp(n_sample=n))
    else:
        print(__doc__)
        sys.exit(0)
