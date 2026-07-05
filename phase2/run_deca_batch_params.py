"""Run DECA on a real image folder and save Phase2-ready parameter .mat files.

This is a resumable no-render runner for the Phase2 pipeline. It saves the
coarse geometry outputs plus DECA parameter vectors needed by Phase2:
shape, expression, pose, camera, light, and detail when available.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deca-root", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--iscrop", default=False, type=lambda x: str(x).lower() in {"true", "1", "yes"})
    parser.add_argument("--detector", default="fan")
    parser.add_argument("--sample-step", default=10, type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", default=0, type=int)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tensor_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def extract_param_outputs(codedict: dict) -> dict[str, np.ndarray]:
    mapping = {
        "shape": "shape",
        "exp": "expression",
        "pose": "pose",
        "cam": "camera",
        "light": "light",
        "detail": "detail",
    }
    params = {}
    for source, target in mapping.items():
        if source not in codedict:
            continue
        value = tensor_to_numpy(codedict[source])
        params[target] = np.asarray(value)
    return params


def main() -> None:
    args = parse_args()
    deca_root = args.deca_root.resolve()
    if not deca_root.exists():
        raise SystemExit(f"DECA root not found: {deca_root}")
    sys.path.insert(0, str(deca_root))

    from decalib.deca import DECA
    from decalib.datasets import datasets
    from decalib.utils import util
    from decalib.utils.config import cfg as deca_cfg

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    testdata = datasets.TestData(
        str(args.input_dir),
        iscrop=args.iscrop,
        face_detector=args.detector,
        sample_step=args.sample_step,
    )
    indices = list(range(len(testdata)))
    if args.limit:
        indices = indices[: args.limit]

    deca_cfg.model.use_tex = False
    deca_cfg.rasterizer_type = "standard"
    deca_cfg.model.extract_tex = False
    deca = DECA(config=deca_cfg, device=device, render_enabled=False)

    written = 0
    skipped = 0
    failed: list[tuple[str, str]] = []
    for i in tqdm(indices, desc="DECA phase2 params"):
        item = testdata[i]
        name = item["imagename"]
        sample_dir = args.output_dir / name
        mat_path = sample_dir / f"{name}.mat"
        if args.resume and mat_path.exists():
            skipped += 1
            continue
        try:
            images = item["image"].to(device)[None, ...]
            with torch.no_grad():
                codedict = deca.encode(images)
                opdict = deca.decode(codedict, rendering=False, return_vis=False)
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(sample_dir / f"{name}_kpt2d.txt", opdict["landmarks2d"][0].detach().cpu().numpy())
            np.savetxt(sample_dir / f"{name}_kpt3d.txt", opdict["landmarks3d"][0].detach().cpu().numpy())
            matdict = util.dict_tensor2npy(opdict)
            matdict.update(extract_param_outputs(codedict))
            savemat(mat_path, matdict)
            written += 1
        except Exception as exc:  # pragma: no cover - operational runner
            failed.append((name, f"{type(exc).__name__}:{exc}"))

    if failed:
        fail_path = args.output_dir / "deca_phase2_failures.csv"
        with fail_path.open("w", encoding="utf-8") as f:
            f.write("image_id,reason\n")
            for image_id, reason in failed:
                f.write(f"{image_id},{reason}\n")

    print(f"input_count={len(testdata)} selected_count={len(indices)} written={written} skipped={skipped} failed={len(failed)}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()

