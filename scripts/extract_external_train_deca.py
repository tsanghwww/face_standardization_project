"""Run FAN-only DECA extraction for identity-screened external train candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.io import loadmat, savemat

from phase2.run_fixed_external_deca import (
    crop_to_tensor,
    extract_param_outputs,
    landmark_score_from,
    load_image,
    validate_mat,
)
from phase3.reconstruction_data import file_hash


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_checkpoint(out_dir: Path, output_rows: list[dict], total: int) -> None:
    path = out_dir / "external_deca_progress.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8")
    progress = {
        "total": total,
        "completed": len(output_rows),
        "success": sum(row["deca_status"] == "success" for row in output_rows),
        "failed": sum(row["deca_status"] != "success" for row in output_rows),
        "progress_manifest_sha256": file_hash(path),
    }
    (out_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-audit", required=True, type=Path)
    parser.add_argument("--deca-root", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out_dir.exists() and not args.resume:
        raise ValueError("DECA quarantine output directory must be new unless --resume is set")
    selected = [row for row in rows(args.identity_audit) if row.get("identity_decision") == "machine_pass_conditional_on_protected_coverage"]
    if not selected:
        raise ValueError("No identity-screened candidates")
    ids = [str(row["image_id"]) for row in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate external candidate IDs")

    sys.path.insert(0, str(args.deca_root.resolve()))
    import face_alignment
    from decalib.deca import DECA
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg

    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = "standard"
    deca_cfg.model.extract_tex = False
    deca = DECA(config=deca_cfg, device=args.device, render_enabled=False)

    class FAN:
        def __init__(self) -> None:
            self.model = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False, compile=False
            )

        def run(self, image):
            output = self.model.get_landmarks(image)
            if output is None:
                return [0], "kpt68"
            points = output[0].squeeze()
            return [
                float(np.min(points[:, 0])), float(np.min(points[:, 1])),
                float(np.max(points[:, 0])), float(np.max(points[:, 1])),
            ], "kpt68"

    fan = FAN()
    args.out_dir.mkdir(parents=True, exist_ok=args.resume)
    mat_root = args.out_dir / "mats"
    mat_root.mkdir(exist_ok=args.resume)
    output_rows = []
    started = time.perf_counter()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for index, source in enumerate(selected):
        image_id = str(source["image_id"])
        record = {
            **source,
            "fan_detected": False,
            "fallback_used": False,
            "deca_status": "failed",
            "deca_failure_reason": "",
            "deca_mat": "",
            "landmark_score": None,
            "head_pose_norm": None,
            "elapsed_seconds": None,
            "training_eligible": False,
            "eligibility_status": "pending_deca",
        }
        tick = time.perf_counter()
        try:
            image_path = Path(source["source_image"])
            if file_hash(image_path) != source["source_image_sha256"]:
                raise ValueError("source_image_hash_mismatch")
            directory = mat_root / image_id
            mat_path = directory / f"{image_id}.mat"
            if args.resume and mat_path.is_file():
                valid, reason = validate_mat(mat_path)
                if valid:
                    data = loadmat(mat_path)
                    record.update(
                        fan_detected=True,
                        deca_status="success",
                        deca_mat=str(mat_path.resolve()),
                        landmark_score=float(landmark_score_from(mat_path)),
                        head_pose_norm=float(np.linalg.norm(np.asarray(data["pose"]).reshape(-1)[:3])),
                        elapsed_seconds=0.0,
                        eligibility_status="identity_and_deca_pass_pending_phase2_provenance",
                        resumed_from_valid_mat=True,
                    )
                    output_rows.append(record)
                    if (index + 1) % args.progress_every == 0 or index + 1 == len(selected):
                        write_checkpoint(args.out_dir, output_rows, len(selected))
                        print(json.dumps({"completed": index + 1, "total": len(selected), "image_id": image_id, "status": "resumed"}), flush=True)
                    continue
            image = load_image(image_path)
            bbox, bbox_type = fan.run(image)
            if len(bbox) < 4:
                raise RuntimeError("fan_no_face")
            record["fan_detected"] = True
            tensor = torch.from_numpy(crop_to_tensor(image, bbox, bbox_type)).to(args.device)[None]
            with torch.no_grad():
                codedict = deca.encode(tensor)
                opdict = deca.decode(codedict, rendering=False, return_vis=False)
            directory.mkdir(exist_ok=True)
            np.savetxt(directory / f"{image_id}_kpt2d.txt", opdict["landmarks2d"][0].detach().cpu().numpy())
            np.savetxt(directory / f"{image_id}_kpt3d.txt", opdict["landmarks3d"][0].detach().cpu().numpy())
            payload = util.dict_tensor2npy(opdict)
            payload.update(extract_param_outputs(codedict))
            savemat(mat_path, payload)
            valid, reason = validate_mat(mat_path)
            if not valid:
                raise RuntimeError(f"validate_failed:{reason}")
            record.update(
                deca_status="success",
                deca_mat=str(mat_path.resolve()),
                landmark_score=float(landmark_score_from(mat_path)),
                head_pose_norm=float(torch.linalg.vector_norm(codedict["pose"][0, :3]).item()),
                eligibility_status="identity_and_deca_pass_pending_phase2_provenance",
            )
        except Exception as error:
            reason = f"{type(error).__name__}:{error}"
            record["deca_failure_reason"] = "fan_no_face" if reason == "RuntimeError:fan_no_face" else reason
            record["eligibility_status"] = "reject_deca_failure"
        record["elapsed_seconds"] = time.perf_counter() - tick
        output_rows.append(record)
        if (index + 1) % args.progress_every == 0 or index + 1 == len(selected):
            write_checkpoint(args.out_dir, output_rows, len(selected))
            print(json.dumps({"completed": index + 1, "total": len(selected), "image_id": image_id, "status": record["deca_status"]}), flush=True)

    manifest_path = args.out_dir / "external_deca_audit.jsonl"
    manifest_path.write_text("".join(json.dumps(row) + "\n" for row in output_rows), encoding="utf-8")
    csv_path = args.out_dir / "external_deca_status.csv"
    status_fields = (
        "image_id", "fan_detected", "fallback_used", "deca_status", "deca_failure_reason",
        "deca_mat", "landmark_score", "head_pose_norm", "elapsed_seconds", "eligibility_status",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=status_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output_rows)
    success = [row for row in output_rows if row["deca_status"] == "success"]
    summary = {
        "status": "deca_audited_not_training_ready",
        "n_identity_machine_pass": len(selected),
        "deca_success": len(success),
        "deca_failure": len(selected) - len(success),
        "fallback_used": sum(bool(row["fallback_used"]) for row in output_rows),
        "training_eligible_count": 0,
        "next_gate": "run frozen final Phase2 inference and record generation-time checkpoint/config/output provenance",
        "identity_audit_sha256": file_hash(args.identity_audit),
        "deca_manifest_sha256": file_hash(manifest_path),
        "deca_model_sha256": file_hash(args.deca_root / "data" / "deca_model.tar"),
        "split_hashes": {name: file_hash(args.split_dir / name) for name in ("train_ids.txt", "validation_ids.txt", "fixed_test_ids.txt")},
        "exact_command": subprocess.list2cmdline([sys.executable, *sys.argv]),
        "wall_seconds": time.perf_counter() - started,
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2 if args.device == "cuda" else None,
        "denominator_policy": "all identity machine-pass candidates; FAN failures retained; no rescue",
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
