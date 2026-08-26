"""Render common baselines and multiple Phase2 ablations with one FAN/DECA encode."""

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


BASE_FIELDS = [
    "eval_id",
    "image_id",
    "source_group",
    "source_dataset",
    "source_image_path",
    "original_render_path",
    "hard_zero_render_path",
    "original_status",
    "hard_zero_status",
    "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        metavar="NAME=MANIFEST",
        help="Repeat for each Phase2 model.",
    )
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
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_variants(values: list[str]) -> dict[str, Path]:
    variants: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"Invalid --variant {value!r}; expected NAME=MANIFEST")
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum():
            raise SystemExit(f"Invalid variant name: {name!r}")
        if name in variants:
            raise SystemExit(f"Duplicate variant name: {name}")
        variants[name] = Path(raw_path)
    return variants


def clone_codedict(codedict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in codedict.items()}


def save_tensor_image(path: Path, tensor: torch.Tensor, util_module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), util_module.tensor2image(tensor[0])):
        raise RuntimeError(f"Could not write {path}")


def compatible_fields() -> list[str]:
    return [
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


def write_compatible_manifests(out_dir: Path, rows: list[dict[str, str]], variants: list[str]) -> None:
    fields = compatible_fields()
    for name in variants:
        converted = []
        for source in rows:
            row = {field: source.get(field, "") for field in fields}
            row["phase2_render_path"] = source.get(f"{name}_render_path", "")
            row["phase2_status"] = source.get(f"{name}_status", "")
            converted.append(row)
        write_csv(out_dir / f"render_manifest_{name}.csv", fields, converted)


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variant)
    sys.path.insert(0, str(args.deca_root))
    from decalib.deca import DECA
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg

    phase_rows = {
        name: {row["image_id"]: row for row in read_csv(path) if row.get("image_id")}
        for name, path in variants.items()
    }
    test_rows = read_csv(args.test_manifest)
    if args.limit:
        test_rows = test_rows[: args.limit]

    variant_fields = [item for name in variants for item in (f"{name}_render_path", f"{name}_status")]
    fields = BASE_FIELDS + variant_fields
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_manifest = args.out_dir / "render_manifest_all.csv"
    completed: dict[str, dict[str, str]] = {}
    if args.resume and output_manifest.exists():
        completed = {row["eval_id"]: row for row in read_csv(output_manifest)}

    device = resolve_device(args.device)
    deca_cfg.rasterizer_type = args.rasterizer_type
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = True
    deca = DECA(config=deca_cfg, device=device, render_enabled=True)
    fan = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D,
        flip_input=False,
        compile=False,
    )
    rows = list(completed.values())

    for index, test in enumerate(test_rows, start=1):
        eval_id = test["eval_id"]
        if eval_id in completed:
            continue
        row = {field: "" for field in fields}
        row.update(
            {
                "eval_id": eval_id,
                "image_id": test.get("image_id", ""),
                "source_group": test.get("source_group", ""),
                "source_dataset": test.get("source_dataset", ""),
                "source_image_path": test.get("image_path", ""),
                "original_status": "fail",
                "hard_zero_status": "not_run",
            }
        )
        for name in variants:
            row[f"{name}_status"] = "not_run"
        reasons: list[str] = []

        try:
            image = load_image(Path(test["image_path"]))
            landmarks = fan.get_landmarks(image)
            if landmarks is None:
                raise RuntimeError("fan_no_face")
            points = landmarks[0].squeeze()
            bbox = [
                float(np.min(points[:, 0])),
                float(np.min(points[:, 1])),
                float(np.max(points[:, 0])),
                float(np.max(points[:, 1])),
            ]
            images = torch.from_numpy(crop_to_tensor(image, bbox, "kpt68")).to(device)[None, ...]
            with torch.no_grad():
                codedict = deca.encode(images)
        except Exception as exc:  # noqa: BLE001
            row["failure_reason"] = f"encode:{type(exc).__name__}:{exc}"
            rows.append(row)
            if index == 1 or index % 25 == 0:
                write_csv(output_manifest, fields, rows)
                write_compatible_manifests(args.out_dir, rows, list(variants))
            continue

        output_dir = args.out_dir / "images" / eval_id
        try:
            with torch.no_grad():
                _op, visual = deca.decode(codedict, rendering=True, return_vis=True)
            path = output_dir / "original.jpg"
            save_tensor_image(path, visual["shape_detail_images"], util)
            row["original_render_path"] = str(path)
            row["original_status"] = "success"
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"original:{type(exc).__name__}")

        try:
            hard = clone_codedict(codedict)
            hard["exp"] = torch.zeros_like(hard["exp"])
            hard["pose"] = torch.zeros_like(hard["pose"])
            with torch.no_grad():
                _op, visual = deca.decode(hard, rendering=True, return_vis=True)
            path = output_dir / "hard_zero.jpg"
            save_tensor_image(path, visual["shape_detail_images"], util)
            row["hard_zero_render_path"] = str(path)
            row["hard_zero_status"] = "success"
        except Exception as exc:  # noqa: BLE001
            row["hard_zero_status"] = "fail"
            reasons.append(f"hard_zero:{type(exc).__name__}")

        for name, manifest_rows in phase_rows.items():
            phase_row = manifest_rows.get(test.get("image_id", ""))
            if not phase_row or not phase_row.get("out_npz"):
                row[f"{name}_status"] = "fail"
                reasons.append(f"{name}:phase2_output_missing")
                continue
            try:
                params = np.load(phase_row["out_npz"], allow_pickle=False)
                phase = clone_codedict(codedict)
                phase["exp"] = torch.from_numpy(params["expression_standardized"]).to(device).float()[None, :]
                phase["pose"] = torch.from_numpy(params["pose_standardized"]).to(device).float()[None, :]
                with torch.no_grad():
                    _op, visual = deca.decode(phase, rendering=True, return_vis=True)
                path = output_dir / f"{name}.jpg"
                save_tensor_image(path, visual["shape_detail_images"], util)
                row[f"{name}_render_path"] = str(path)
                row[f"{name}_status"] = "success"
            except Exception as exc:  # noqa: BLE001
                row[f"{name}_status"] = "fail"
                reasons.append(f"{name}:{type(exc).__name__}")

        row["failure_reason"] = ";".join(reasons)
        rows.append(row)
        if index == 1 or index % 25 == 0:
            write_csv(output_manifest, fields, rows)
            write_compatible_manifests(args.out_dir, rows, list(variants))
            print(f"rendered={index}/{len(test_rows)}", flush=True)

    write_csv(output_manifest, fields, rows)
    write_compatible_manifests(args.out_dir, rows, list(variants))
    summary = {
        "count": len(rows),
        "success": {
            "original": sum(row["original_status"] == "success" for row in rows),
            "hard_zero": sum(row["hard_zero_status"] == "success" for row in rows),
            **{name: sum(row[f"{name}_status"] == "success" for row in rows) for name in variants},
        },
        "any_failure": sum(bool(row["failure_reason"]) for row in rows),
        "manifest": str(output_manifest),
    }
    (args.out_dir / "render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
