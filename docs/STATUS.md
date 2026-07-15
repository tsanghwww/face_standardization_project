# Project Status

- **Last Updated**: 2026-07-15
- **Current Phase**: Phase 1 feature parity complete on win-lenovo

## 5060 Workstation Snapshot

| Metric | Value |
|---|---|
| Total files | 130,117 |
| Total size | ~54.4 GB |
| Free disk space (D:) | ~258 GB |
| Git commits | 5 (bca9d81) |
| Git status | Clean (1 untracked screening summary) |

### Directory Breakdown

| Directory | Size | Files | Role |
|---|---|---|---|
| `results/` | 23.3 GB | 60,042 | Experiment outputs |
| `archive/` + `archive.zip` | 23.1 GB | 10,000 | Dataset + backup |
| `.venv/` | 5.1 GB | 29,889 | Python environment |
| `DECA/` | 3.0 GB | 30,091 | Third-party code + assets |
| `tools/` | 6 MB | 19 | Utility scripts |
| `phase2/` | <1 MB | 28 | Core research code |
| `.openclaw/` | <1 MB | 10 | PIL agent layer |
| `docs/` | <1 MB | 14 | Project memory |

### Results Detail

| Subdirectory | Size | Files | Type |
|---|---|---|---|
| `screening_p95/` | 11.6 GB | 10,008 | Filtered dataset (p95) |
| `screening_p975/` | 11.6 GB | 10,008 | Filtered dataset (p97.5) |
| `phase2_hard_zero_recovered/` | 38 MB | 10,002 | Hard-zero baseline outputs |
| `phase2_real_infer_stage{1,2,3}_recovered/` | 48 MB × 3 | 10,002 × 3 | Stage 1/2/3 inference outputs |
| `phase2_real_train_stage{1,2,3}_recovered/` | <1 MB each | 4 each | Training checkpoints |
| `phase2_recovered_compare{,_full}/` | <1 MB each | 2 each | Comparison results |

## Overall Progress

| Phase | Status | Notes |
|---|---|---|
| Environment Setup | ✅ Complete | Python 3.12, PyTorch 2.11, CUDA working |
| DECA Integration | ✅ Complete | Rasterizer compiles, smoke test passes |
| Phase2 Training Stage 1 | ✅ Complete | best_model.pt (439 KB) |
| Phase2 Training Stage 2 | ✅ Complete | best_model.pt recovered |
| Phase2 Training Stage 3 | ✅ Complete | best_model.pt recovered |
| Phase2 Inference | ✅ Complete | Stage 1/2/3 inference outputs present (10K each) |
| Screening (p95/p975) | ✅ Complete | Filtered subsets with ~10K samples each |
| Hard Zero Baseline | ✅ Complete | 10K baseline outputs |
| Git & Code Sync | ✅ Complete | 3-way sync (Mac/5060/GitHub) |
| PIL Setup | ✅ Complete | 24 files, 1300 lines, installed as skill |
| BUG-003 Fix | ✅ Complete | kpt coordinate denormalization, landmark_score restored |
| Data cleaning + class labels | ✅ Complete | p95=9,500 Pass / 500 Warn; p97.5=9,750 Pass / 250 Warn |
| L2CS-Net gaze extraction | ✅ Complete | 10,000/10,000 successful; Gaze360 weight SHA256 recorded |
| ArcFace identity extraction | ✅ Complete | 9,990/9,990 successful after 22 low-threshold retries |
| Phase 1 master manifest | ✅ Complete | 10,000 unique IDs with source SHA256, cleaning, DECA, gaze, and identity fields |
| DECA standardized re-rendering | ⚠️ Partial | Parameter inference exists; full normal/displacement/render parity set has not been verified on Lenovo |
| [待确认] Next Research Phase | ⏳ Pending | |

## Phase 1 Final Metrics

| Metric | Value |
|---|---:|
| Source / unique IDs | 10,000 / 10,000 |
| Eye-invalid exclusions | 10 |
| DECA success | 10,000 |
| L2CS success | 10,000 |
| ArcFace main success | 9,968 |
| ArcFace retry recovered | 22 |
| ArcFace final success | 9,990 |
| ArcFace strict train | 9,482 |
| ArcFace full train | 9,499 |
| Image hashes recorded | 10,000 |

## Blockers

- None for Phase 1 parity. Historical workstation remains unavailable, so regenerated L2CS results retain the explicit `rebuilt` provenance label.

## Risks

- RTX 5060 has only 8GB VRAM — may limit batch size and model scale
- Original 2060 data irretrievably lost — recovery may have gaps
- Screening outputs duplicate ~23GB of data (p95 + p975 each ~11.5GB)
- [待确认] Are there other datasets that haven't been restored?

## Environment Health

| Component | Status |
|---|---|
| 5060 SSH | ✅ Reachable (`win-lenovo`) |
| GitHub Push | ✅ SSH key configured |
| Python venv | ✅ D:\face_standardization_project\.venv (5.1 GB) |
| CUDA | ✅ RTX 5060, cu128 |
| DECA models | ✅ deca_model.tar + FLAME present |
| InsightFace | ✅ `buffalo_l`, ONNX Runtime CPU provider |
| L2CS-Net | ✅ Official package commit `4a0f978`, Gaze360 weight SHA256 `8a7f3480...80665` |
| D: free space | ✅ ~258 GB remaining |

## Latest Quality Metrics (200-sample spot check, after BUG-003 fix)

| Metric | Mean | Range |
|---|---|---|
| landmark_score | 0.943 | 0.904 – 0.967 |
| landmark_out_ratio | 0.000 | 0.000 |
| landmark_bbox_area | 0.311 | 0.274 – |
| landmark_center_dist | 0.071 | |
| quality_score | 0.683 | 0.632 – 0.744 |
| quality_class distribution | high=5, mid=195, low=0 | |
