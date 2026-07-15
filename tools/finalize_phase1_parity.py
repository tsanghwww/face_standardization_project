"""Wait for Phase 1 rebuilds, retry ArcFace failures, and build the master manifest."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARCFACE_MAIN = RESULTS / "arcface_p95_rebuilt"
ARCFACE_RETRY = RESULTS / "arcface_p95_retry_rebuilt"
L2CS_DIR = RESULTS / "gaze_10k_l2cs_rebuilt"
PARITY_DIR = RESULTS / "phase1_parity"
TARGET_ARCFACE = 9990
TARGET_L2CS = 10000


def csv_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def wait_for_main_runs() -> None:
    while True:
        arcface_count = csv_count(ARCFACE_MAIN / "arcface_manifest.csv")
        l2cs_count = csv_count(L2CS_DIR / "l2cs_gaze_summary_10k.csv")
        print(f"WAIT arcface={arcface_count}/{TARGET_ARCFACE} l2cs={l2cs_count}/{TARGET_L2CS}", flush=True)
        if arcface_count == TARGET_ARCFACE and l2cs_count == TARGET_L2CS:
            return
        time.sleep(60)


def prepare_retry() -> int:
    manifest_path = ARCFACE_MAIN / "arcface_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    failures = [row for row in rows if row.get("arcface_status") != "success"]
    ARCFACE_RETRY.mkdir(parents=True, exist_ok=True)
    ids_path = PARITY_DIR / "arcface_retry_ids.txt"
    ids_path.write_text("\n".join(row["image_id"] for row in failures), encoding="utf-8")
    if failures:
        run(
            "tools/extract_arcface_embeddings.py",
            "--images-dir", "archive/generated_yellow-stylegan2",
            "--screening-report", "results/screening_p95/screening_report.json",
            "--output-dir", "results/arcface_p95_retry_rebuilt",
            "--model-name", "buffalo_l",
            "--det-size", "640",
            "--det-thresh", "0.05",
            "--ctx-id", "-1",
            "--save-aligned",
            "--include-ids", str(ids_path),
        )
        run(
            "tools/merge_arcface_retry.py",
            "--main-dir", "results/arcface_p95_rebuilt",
            "--retry-dir", "results/arcface_p95_retry_rebuilt",
        )
    else:
        with (ARCFACE_RETRY / "arcface_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
    return len(failures)


def main() -> None:
    PARITY_DIR.mkdir(parents=True, exist_ok=True)
    wait_for_main_runs()
    retry_count = prepare_retry()
    run(
        "tools/annotate_arcface_manifest.py",
        "--main-dir", "results/arcface_p95_rebuilt",
        "--retry-dir", "results/arcface_p95_retry_rebuilt",
    )
    run(
        "tools/build_phase1_master_manifest.py",
        "--images-dir", "archive/generated_yellow-stylegan2",
        "--eye-invalid-ids", "configs/phase1_eye_invalid_ids.txt",
        "--p95-report", "results/screening_p95/screening_report.json",
        "--p975-report", "results/screening_p975/screening_report.json",
        "--deca-results-dir", "DECA/results/archive_phase2_params",
        "--l2cs-summary", "results/gaze_10k_l2cs_rebuilt/l2cs_gaze_summary_10k.csv",
        "--arcface-manifest", "results/arcface_p95_rebuilt/arcface_manifest.csv",
        "--output-dir", "results/phase1_parity",
        "--hash-images",
    )
    state = {
        "status": "complete",
        "completed_unix": time.time(),
        "arcface_retry_requested": retry_count,
    }
    (PARITY_DIR / "finalizer_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        PARITY_DIR.mkdir(parents=True, exist_ok=True)
        (PARITY_DIR / "finalizer_state.json").write_text(
            json.dumps({"status": "failed", "error": f"{type(exc).__name__}:{exc}"}, indent=2),
            encoding="utf-8",
        )
        raise
