#!/usr/bin/env python3
"""Summarize DECA parameter vectors saved in per-image .mat files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


PARAM_KEYS = ("shape", "expression", "pose", "camera")


def find_mat_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*/*.mat"), key=lambda p: p.parent.name)


def load_vector(mat_path: Path, key: str) -> np.ndarray:
    data = loadmat(mat_path)
    if key not in data:
        raise KeyError(f"{mat_path} missing {key}")
    arr = np.asarray(data[key], dtype=np.float64)
    return arr.reshape(-1)


def summarize(values: np.ndarray) -> dict[str, object]:
    return {
        "count": int(values.shape[0]),
        "dim": int(values.shape[1]),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0, ddof=0).tolist(),
    }


def write_param_csv(path: Path, summary: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "index", "mean", "std", "count"])
        for key in PARAM_KEYS:
            item = summary[key]
            for idx, (mean, std) in enumerate(zip(item["mean"], item["std"])):
                writer.writerow([key, idx, mean, std, item["count"]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DECA shape/expression/pose/camera parameters.")
    parser.add_argument("--results-dir", required=True, type=Path, help="Directory containing per-image DECA result folders.")
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--out-csv", required=True, type=Path)
    args = parser.parse_args()

    mat_files = find_mat_files(args.results_dir)
    if not mat_files:
        raise SystemExit(f"No .mat files found under {args.results_dir}")

    summary: dict[str, dict[str, object]] = {}
    missing: dict[str, list[str]] = {key: [] for key in PARAM_KEYS}
    for key in PARAM_KEYS:
        rows = []
        for mat_path in mat_files:
            try:
                rows.append(load_vector(mat_path, key))
            except KeyError:
                missing[key].append(str(mat_path))
        if missing[key]:
            raise SystemExit(f"{key} missing in {len(missing[key])} files; first: {missing[key][0]}")
        summary[key] = summarize(np.vstack(rows))

    payload = {
        "results_dir": str(args.results_dir),
        "mat_count": len(mat_files),
        "parameters": summary,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_param_csv(args.out_csv, summary)

    print(f"Summarized {len(mat_files)} .mat files")
    for key in PARAM_KEYS:
        print(f"{key}: dim={summary[key]['dim']} count={summary[key]['count']}")
    print(f"JSON: {args.out_json}")
    print(f"CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
