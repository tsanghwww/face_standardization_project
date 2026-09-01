# face_standardization_project

A face standardization research pipeline built around DECA, ArcFace, and quality-aware latent-space modeling for 3D face reconstruction, identity embedding extraction, dataset screening, and reproducible experiments.

## Contributors

- [@tsanghwww](https://github.com/tsanghwww)
- [@H-1-m](https://github.com/H-1-m)

## Overview

This repository collects the code used for a face standardization research workflow:

- `DECA/`: a modernized DECA runtime for 3D face reconstruction and parameter extraction.
- `tools/`: ArcFace embedding extraction, manifest annotation, retry merging, and lighting analysis utilities.
- `phase2/`: frozen-DECA condition generation for standardizing expression and pose parameters with quality-aware confidence scores.

The repository intentionally does not include private datasets, experiment outputs, virtual environments, or large model assets.

## Branch Policy

- `main` is the integration baseline for Phase2 development, experiment execution, reproducible evaluation, and release-ready documentation.
- `feature/*` branches are isolated contribution branches for team members. They are not experiment baselines and must not become the source of record for shared results.
- Feature work is reviewed and integrated into `main` before it is used by the main research pipeline.
- Commands, reports, and remote-machine runs should record the exact `main` commit used whenever possible.

## What Is Tracked

- Source code and scripts.
- Runtime configuration files.
- Reproducibility notes and package requirement files.
- Lightweight documentation.

## What Is Not Tracked

- Face datasets and private images.
- DECA / FLAME model assets and trained weights.
- Generated reconstruction outputs, logs, and visualizations.
- Local virtual environments.
- Project planning documents and business documents.

## Quick Start

For DECA runtime notes, see:

```text
DECA/RUNNING_MODERN.md
```

For Phase2 standardization training and inference:

```bash
pip install -r phase2/requirements_phase2.txt
python -m phase2.train_condition_generator --help
python -m phase2.infer_standardize_params --help
```

## Research Reports

- [2026 Summer Work Report](docs/SUMMER_WORK_REPORT_2026.md)
- [Phase2 Final Experimental Report](docs/phase2_final_20260827/PHASE2_FINAL_REPORT_20260827.md)
- [Phase2.1 Protocol](docs/PHASE21_PROTOCOL.md)
- [Phase2.1 Scope Decision and Next Step](docs/PHASE21_SCOPE_DECISION_AND_NEXT_STEP.md)
- [Phase3 Latent Diffusion Training Plan](docs/PHASE3_LATENT_DIFFUSION_TRAINING_PLAN.md)

## License Notice

This repository includes code adapted from DECA. DECA is provided for non-commercial scientific research purposes; see `DECA/LICENSE` for details.
