# Operation Log

## 2026-07-08 14:15 — Session Start & Project Recovery Assessment

- **Agent**: DeepSeek
- **Tasks**: 
  - Connected to 5060 workstation (win-lenovo)
  - Audited project directory structure
  - Found face_standardization_project on D:\
  - Verified Python 3.12.13, PyTorch 2.11.0+cu128, CUDA available
  - Verified DECA model weights present (deca_model.tar + FLAME)
  - Verified Phase2 checkpoints recovered (stage1/2/3)
  - Identified dataset in archive/ (10000 StyleGAN2 generated faces)
- **Files changed**: None
- **Outcome**: Project recovery confirmed complete, all core assets present

## 2026-07-08 14:26 — Git Setup & Code Sync

- **Agent**: DeepSeek
- **Tasks**:
  - Installed Git 2.51.0 on 5060
  - Generated SSH key on 5060, added to GitHub
  - Committed local 5060 changes (PyTorch 2.6+ fixes, CUDA rasterizer migration)
  - Synced code between Mac → GitHub → 5060
  - Updated .gitignore to exclude local scripts, temp files, archive
  - Verified DECA rasterizer compiles and smoke test passes
  - Verified Phase2 inference CLI works
- **Files changed**: .gitignore, DECA CUDA kernel, renderer.py, multiple tools
- **Outcome**: 3-way code sync established, all tests pass

## 2026-07-08 15:38 — PIL Design & Initialization

- **Agent**: DeepSeek
- **Tasks**:
  - Designed Project Intelligence Layer structure
  - Created .openclaw/ with rules, workflows, and skill scaffold
  - Created docs/ with CONTEXT, STATUS, TODO, NEXT, LOG, and memory files
  - Defined context loading and saving protocols
  - Documented safety rules and code boundaries
- **Files changed**: 24 new files under .openclaw/ and docs/
- **Outcome**: PIL v1.0 initialized, committed and synced to 5060

## 2026-07-08 21:32 — Skill Installation & 5060 Statistics

- **Agent**: DeepSeek
- **Tasks**:
  - Installed pil-context-sync skill to ~/.openclaw/skills/ (OpenClaw discovery path)
  - Scanned 5060 project directory: 130,117 files, 54.4 GB total
  - Broke down by directory: results (23.3GB), archive+dataset (23.1GB), venv (5.1GB), DECA (3.0GB)
  - Detailed results inventory: screening sets, inference outputs, training checkpoints
  - Verified git status clean on 5060
- **Files changed**: STATUS.md, LOG.md, MODEL.md, DATASET.md, NEXT.md, TODO.md
- **Key outcomes**:
  - Full project inventory documented with exact sizes and file counts
  - PIL skill operational
  - All experiment outputs accounted for (stage 1/2/3 training + inference + screening + baseline)
  - D: drive has ~258 GB free

## 2026-07-08 21:53 — BUG-003: kpt coordinate system fix

- **Agent**: DeepSeek
- **Tasks**:
  - Diagnosed landmark_score=0 bug: DECA kpt2d in [-1,1] normalized space vs features.py expecting 224px pixel space
  - Root cause: denormalization commented out in DECA decalib/deca.py L182
  - Fix: added `xy = xy * (image_size/2) + (image_size/2)` in features.py L108
  - Committed and synced to GitHub + 5060
  - Updated BUGS.md with bug lifecycle record
- **Files changed**: phase2/features.py (+4 lines), docs/BUGS.md
- **Key outcomes**: landmark_score should now work correctly for all 10K samples
