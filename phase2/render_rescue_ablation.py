"""Render rescue-only original, hard-zero, and Phase2 ablation outputs from saved MAT files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .evaluate_rescue_sensitivity import codedict_from, load_params


PROJECT = Path(r"D:\face_standardization_project")
BASE_FIELDS = [
    "eval_id", "image_id", "source_group", "source_dataset", "source_image_path",
    "preprocess_source", "original_render_path", "hard_zero_render_path",
    "original_status", "hard_zero_status", "failure_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--variant", action="append", required=True, metavar="NAME=MANIFEST")
    parser.add_argument("--deca-root", type=Path, default=PROJECT / "DECA")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--preprocess-source", default="whole_image_rescue")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_variants(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        name, path = value.split("=", 1)
        output[name] = Path(path)
    return output


def render(deca, params: dict[str, np.ndarray], path: Path, device: str, util) -> None:
    with torch.no_grad():
        _opdict, visual = deca.decode(codedict_from(params, device), rendering=True, return_vis=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), util.tensor2image(visual["shape_detail_images"][0])):
        raise RuntimeError("render_write_failed")


def main() -> None:
    args = parse_args()
    variants = parse_variants(args.variant)
    sys.path.insert(0, str(args.deca_root.resolve()))
    from decalib.deca import DECA
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg

    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    deca_cfg.model.use_tex = False
    deca_cfg.model.extract_tex = False
    deca_cfg.rasterizer_type = "standard"
    deca = DECA(config=deca_cfg, device=device, render_enabled=True)
    test_rows = read_csv(args.test_manifest)
    variant_rows = {
        name: {row["image_id"]: row for row in read_csv(path)}
        for name, path in variants.items()
    }
    fields = BASE_FIELDS + [field for name in variants for field in (f"{name}_render_path", f"{name}_status")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / "render_manifest_all.csv"
    rows = read_csv(output_path) if args.resume and output_path.exists() else []
    completed = {row["eval_id"] for row in rows}

    for index, source in enumerate(test_rows, start=1):
        if source["eval_id"] in completed:
            continue
        row = {field: "" for field in fields}
        row.update({
            "eval_id": source["eval_id"],
            "image_id": source["image_id"],
            "source_group": source["source_group"],
            "source_dataset": source["source_dataset"],
            "source_image_path": source["image_path"],
            "preprocess_source": args.preprocess_source,
            "original_status": "not_run",
            "hard_zero_status": "not_run",
        })
        for name in variants:
            row[f"{name}_status"] = "not_run"
        reasons = []
        try:
            base = load_params(Path(source["mat_path"]))
        except Exception as exc:  # noqa: BLE001
            row["failure_reason"] = f"mat:{type(exc).__name__}:{exc}"
            rows.append(row)
            continue
        image_dir = args.out_dir / "images" / source["eval_id"]
        try:
            path = image_dir / "original.jpg"
            render(deca, base, path, device, util)
            row["original_render_path"] = str(path)
            row["original_status"] = "success"
        except Exception as exc:  # noqa: BLE001
            row["original_status"] = "fail"
            reasons.append(f"original:{type(exc).__name__}:{exc}")
        try:
            hard = {key: value.copy() for key, value in base.items()}
            hard["expression"][:] = 0
            hard["pose"][:] = 0
            path = image_dir / "hard_zero.jpg"
            render(deca, hard, path, device, util)
            row["hard_zero_render_path"] = str(path)
            row["hard_zero_status"] = "success"
        except Exception as exc:  # noqa: BLE001
            row["hard_zero_status"] = "fail"
            reasons.append(f"hard_zero:{type(exc).__name__}:{exc}")
        for name in variants:
            inference = variant_rows[name].get(source["image_id"])
            try:
                if not inference or not inference.get("out_npz"):
                    raise RuntimeError("inference_missing")
                prediction = np.load(inference["out_npz"], allow_pickle=False)
                params = {key: value.copy() for key, value in base.items()}
                params["expression"] = np.asarray(prediction["expression_standardized"], dtype=np.float32)
                params["pose"] = np.asarray(prediction["pose_standardized"], dtype=np.float32)
                path = image_dir / f"{name}.jpg"
                render(deca, params, path, device, util)
                row[f"{name}_render_path"] = str(path)
                row[f"{name}_status"] = "success"
            except Exception as exc:  # noqa: BLE001
                row[f"{name}_status"] = "fail"
                reasons.append(f"{name}:{type(exc).__name__}:{exc}")
        row["failure_reason"] = ";".join(reasons)
        rows.append(row)
        if index % 10 == 0 or index == len(test_rows):
            write_csv(output_path, fields, rows)
            print(f"rendered={index}/{len(test_rows)}", flush=True)

    write_csv(output_path, fields, rows)
    summary = {
        "count": len(rows),
        "preprocess_source": args.preprocess_source,
        "success": {
            "original": sum(row["original_status"] == "success" for row in rows),
            "hard_zero": sum(row["hard_zero_status"] == "success" for row in rows),
            **{name: sum(row[f"{name}_status"] == "success" for row in rows) for name in variants},
        },
        "any_failure": sum(bool(row["failure_reason"]) for row in rows),
        "manifest": str(output_path),
    }
    (args.out_dir / "render_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
