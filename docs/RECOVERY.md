# Recovery Record

## INC-001: RTX 2060 Workstation Complete Loss

- **Date**: Before 2026-07-08
- **Severity**: Critical
- **Impact**: All project data on original 2060 workstation lost
- **Recovery Date**: 2026-07-08
- **Recovery Status**: ✅ Mostly Recovered

### Assets Recovered

| Asset | Source | Status |
|---|---|---|
| Code (phase2/, tools/, DECA/) | External backup → 5060 | ✅ Full |
| DECA model weights | External backup → 5060 | ✅ Full |
| FLAME assets | External backup → 5060 | ✅ Full |
| Phase2 checkpoints (stage1/2/3) | External backup → 5060 | ✅ Recovered |
| Dataset (StyleGAN2 10000 images) | External backup → 5060 | ✅ In archive/ |
| Git history | Mac remnants + new commits | ✅ Reconstructed |
| [待確認] Original face dataset | Unknown | ⚠️ Status unclear |
| Historical data cleaning / p95 labels | Rebuilt on Lenovo | ✅ p95 and p97.5 labels joined into master manifest |
| L2CS-Net gaze outputs | Rebuilt on Lenovo | ✅ 10,000/10,000 with explicit rebuilt provenance |
| ArcFace embeddings + strict/full labels | Rebuilt on Lenovo | ✅ 9,990/9,990 after retry; strict/full labels restored |
| [待確認] Experiment logs | Unknown | ⚠️ Status unclear |

### Historical Phase 1 Evidence

The pre-Lenovo workstation was verified on 2026-06-11 with these L2CS artifacts:

- `H:\face_standardization_project\results\gaze_10k_l2cs\`
- 10,000 `*_gaze.json` files
- `l2cs_gaze_summary_10k.csv` with `image_id,pitch,yaw,gaze_x,gaze_y,gaze_z,status`
- `gaze_10k_l2cs.zip`

Historical cleaning used `screening_v3_p95` as the main p95 result. ArcFace used original RGB images, InsightFace `buffalo_l`, a main pass plus low-threshold retry, and preserved `arcface_stage`, `det_thresh`, `detector_score`, `use_for_train_strict`, and `use_for_train_full`. These facts are sufficient to regenerate the missing Lenovo artifacts when direct migration is unavailable.

The Lenovo rebuild completed on 2026-07-15. Its canonical joined outputs are `results/phase1_parity/phase1_master_manifest.csv` and `phase1_master_summary.json`. Source images were retained; the 10 historical eye-invalid IDs are represented by `eye_valid=false` rather than physical deletion.

### Lessons Learned

1. **Git is essential**: Code survived because part was on Mac and GitHub
2. **External backups work**: 12GB archive.zip + archive/ folder saved the project
3. **Model weights are critical**: DECA pretrained model is not easily re-downloadable
4. **Need better backup discipline**: [待确认] No regular cloud backup was in place
5. **PIL could have helped**: If PIL existed before the loss, recovery inventory would have been easier

### Actions Taken

1. Restored project to 5060 (win-lenovo) from external backup
2. Installed Git, set up GitHub sync
3. Applied PyTorch 2.6+ compatibility fixes
4. Verified all code, models, and checkpoints
5. Created PIL for future resilience
