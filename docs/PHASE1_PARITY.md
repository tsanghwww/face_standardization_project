# Phase 1 Parity Record

## Completion

- Date: 2026-07-15
- Compute node: `win-lenovo`, RTX 5060 Laptop GPU
- Finalizer status: complete, exit code 0
- Canonical data artifact: `results/phase1_parity/phase1_master_manifest.csv`
- Summary artifact: `results/phase1_parity/phase1_master_summary.json`

## Coverage

| Component | Coverage | Notes |
|---|---:|---|
| Source images | 10,000 | 10,000 unique IDs; SHA256 recorded |
| Eye-valid set | 9,990 | 10 historical exclusions retained with `eye_valid=false` |
| p95 cleaning | 9,500 Pass / 500 Warn | Canonical cleaning branch |
| p97.5 cleaning | 9,750 Pass / 250 Warn | Benchmark branch |
| DECA parameters | 10,000 | All IDs linked to `.mat` paths |
| L2CS-Net | 10,000 | All successful |
| ArcFace | 9,990 | All eye-valid images successful |

## ArcFace Provenance

- InsightFace model package: `buffalo_l`
- Recognition model: `w600k_r50.onnx`
- Runtime provider: `CPUExecutionProvider`
- Main pass: `det_size=640`, `det_thresh=0.1`, 9,968 successful
- Retry pass: `det_size=640`, `det_thresh=0.05`, 22 recovered
- Retry composition: 17 p95 Pass and 5 p95 Warn
- Strict train set: 9,482 (`Pass`, success, main pass only)
- Full train set: 9,499 (`Pass`, success, retry included)

## L2CS Provenance

- Architecture: ResNet50
- Checkpoint: `L2CSNet_gaze360.pkl`
- Device: CUDA
- Face confidence threshold: 0.5
- Checkpoint SHA256: `8a7f3480d868dd48261e1d59f915b0ef0bb33ea12ea00938fb2168f212080665`
- Output schema: `image_id,pitch,yaw,gaze_x,gaze_y,gaze_z,status`

The L2CS output is marked `rebuilt`. It reproduces the historical schema and model family, but does not claim byte-identical predictions because the original workstation implementation and crop state are unavailable.

## Reproduction

- `tools/run_l2cs_batch.py`: resumable gaze extraction
- `tools/extract_arcface_embeddings.py`: main/retry identity extraction
- `tools/finalize_phase1_parity.py`: retry, merge, annotation, hashing, and final join
- `tools/build_phase1_master_manifest.py`: canonical 10K joined manifest
- `configs/phase1_eye_invalid_ids.txt`: historical non-destructive exclusions
