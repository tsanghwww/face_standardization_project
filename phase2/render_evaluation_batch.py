"""Render original, hard-zero, and Phase2 parameter outputs for a fixed test split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import face_alignment
import numpy as np
import torch

from .run_fixed_external_deca import crop_to_tensor, load_image


FIELDS = [
    "eval_id",
    "image_id",
    "source_group",
    "source_dataset",
    "source_image_path",
    "original_render_path",
    "hard_zero_render_path",
    "phase2_render_path",
    "original_status",
    "hard_zero_status",
    "phase2_status",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--phase2-manifest", required=True, type=Path)
    parser.add_argument("--deca-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--rasterizer-type", default="standard")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clone_codedict(codedict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in codedict.items()}


def save_tensor_image(path: Path, tensor: torch.Tensor, util_module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = util_module.tensor2image(tensor[0])
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not write {path}")


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(args.deca_root))
    from decalib.deca import DECA
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg

    phase_rows = {row["image_id"]: row for row in read_csv(args.phase2_manifest) if row.get("image_id")}
    test_rows = read_csv(args.test_manifest)
    if args.limit:
        test_rows = test_rows[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.out_dir / "render_manifest.csv"
    completed: dict[str, dict[str, str]] = {}
    if args.resume and output_manifest.exists():
        completed = {row["eval_id"]: row for row in read_csv(output_manifest)}

    device_name = resolve_device(args.device)
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device_name, render_enabled=True)

    class FAN:
        def __init__(self) -> None:
            self.model = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                flip_input=False,
                compile=False,
            )

        def crop(self, image_path: Path) -> torch.Tensor:
            image = load_image(image_path)
            landmarks = self.model.get_landmarks(image)
            if landmarks is None:
                raise RuntimeError("fan_no_face")
            points = landmarks[0].squeeze()
            bbox = [
                float(np.min(points[:, 0])),
                float(np.min(points[:, 1])),
                float(np.max(points[:, 0])),
                float(np.max(points[:, 1])),
            ]
            return torch.from_numpy(crop_to_tensor(image, bbox, "kpt68"))

    fan = FAN()
    rows = list(completed.values())

    for index, test_row in enumerate(test_rows, start=1):
        eval_id = test_row["eval_id"]
        if eval_id in completed:
            continue
        row = {field: "" for field in FIELDS}
        row.update({
            "eval_id": eval_id,
            "image_id": test_row.get("image_id", ""),
            "source_group": test_row.get("source_group", ""),
            "source_dataset": test_row.get("source_dataset", ""),
            "source_image_path": test_row.get("image_path", ""),
            "original_status": "fail",
            "hard_zero_status": "not_run",
            "phase2_status": "not_run",
        })
        phase_row = phase_rows.get(test_row.get("image_id", ""))
        reasons: list[str] = []
        # Encode the source once; original/hard_zero/phase2 all reuse codedict.
        try:
            images = fan.crop(Path(test_row["image_path"])).to(device_name)[None, ...]
            with torch.no_grad():
                codedict = deca.encode(images)
        except Exception as exc:  # Runtime failures are a result, not a crash of the batch.
            row["failure_reason"] = f"encode:{type(exc).__name__}:{exc}"
            rows.append(row)
            if index == 1 or index % 25 == 0:
                _write_manifest(output_manifest, FIELDS, rows)
            continue

        output_dir = args.out_dir / "images" / eval_id

        # original render
        try:
            with torch.no_grad():
                _op, original_vis = deca.decode(codedict, rendering=True, return_vis=True)
            original_path = output_dir / "original.jpg"
            save_tensor_image(original_path, original_vis["shape_detail_images"], util)
            row["original_render_path"] = str(original_path)
            row["original_status"] = "success"
        except Exception as exc:  # noqa: BLE001
            row["original_status"] = "fail"
            reasons.append(f"original:{type(exc).__name__}")

        # hard-zero render (independent of the Phase2 model)
        try:
            hard = clone_codedict(codedict)
            hard["exp"] = torch.zeros_like(hard["exp"])
            hard["pose"] = torch.zeros_like(hard["pose"])
            with torch.no_grad():
                _op, hard_vis = deca.decode(hard, rendering=True, return_vis=True)
            hard_path = output_dir / "hard_zero.jpg"
            save_tensor_image(hard_path, hard_vis["shape_detail_images"], util)
            row["hard_zero_render_path"] = str(hard_path)
            row["hard_zero_status"] = "success"
        except Exception as exc:  # noqa: BLE001
            row["hard_zero_status"] = "fail"
            reasons.append(f"hard_zero:{type(exc).__name__}")

        # phase2 render: only when the standardized params are available
        if phase_row and phase_row.get("out_npz"):
            try:
                params = np.load(phase_row["out_npz"], allow_pickle=False)
                phase = clone_codedict(codedict)
                phase["exp"] = torch.from_numpy(params["expression_standardized"]).to(device_name).float()[None, :]
                phase["pose"] = torch.from_numpy(params["pose_standardized"]).to(device_name).float()[None, :]
                with torch.no_grad():
                    _op, phase_vis = deca.decode(phase, rendering=True, return_vis=True)
                phase_path = output_dir / "phase2.jpg"
                save_tensor_image(phase_path, phase_vis["shape_detail_images"], util)
                row["phase2_render_path"] = str(phase_path)
                row["phase2_status"] = "success"
            except Exception as exc:  # noqa: BLE001
                row["phase2_status"] = "fail"
                reasons.append(f"phase2:{type(exc).__name__}")
        else:
            row["phase2_status"] = "fail"
            reasons.append("phase2_output_missing")

        if reasons:
            row["failure_reason"] = ";".join(reasons)
        rows.append(row)
        if index == 1 or index % 25 == 0:
            _write_manifest(output_manifest, FIELDS, rows)
            print(f"rendered={index}/{len(test_rows)}", flush=True)

    with output_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "count": len(rows),
        "complete_triplets": sum(1 for row in rows if row["phase2_status"] == "success"),
        "any_failure": sum(1 for row in rows if row["failure_reason"]),
        "manifest": str(output_manifest),
    }
    (args.out_dir / "render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
