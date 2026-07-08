# Development Workflow

## Environment Overview

```
MacBook (Control) ──SSH──► Windows 11 Laptop "win-lenovo" (5060)
     │                           │
     │  Code editing             │  Training / Inference
     │  Git management           │  GPU compute
     │  PIL maintenance          │  Data storage
     │                           │
     └──── GitHub ───────────────┘
          Code sync hub
```

## Daily Workflow

### Start of Day

1. On Mac: `git pull origin main` to sync latest code
2. SSH to 5060: verify connection
3. On 5060: `git pull origin main` to sync
4. Read PIL context (see context_lifecycle.md)
5. Confirm TODO priorities

### Code Development

1. Edit code on Mac (phase2/, tools/)
2. Test syntax: `python -m py_compile <file>`
3. Push to GitHub: `git add -A && git commit -m "..." && git push`
4. SSH to 5060: `git pull origin main`
5. Run on 5060 (with GPU)

### Running Experiments on 5060

```bash
# From Mac:
ssh win-lenovo

# On 5060:
cd D:\face_standardization_project
git pull origin main
.venv\Scripts\activate
python -m phase2.train_condition_generator [args]
```

### Recording Results

1. Note experiment in docs/EXPERIMENTS.md
2. Update STATUS.md if progress changed
3. Commit PIL changes to Git

## Common Tasks

### Compile DECA Rasterizer (if needed)

```bash
# On 5060:
D:\face_standardization_project\.venv\Scripts\python.exe tools\compile_deca_rasterizer.py
```

### Verify Environment

```bash
# On 5060:
D:\face_standardization_project\.venv\Scripts\python.exe tools\smoke_deca_rasterizer.py
```

### Sync 5060 with GitHub

```bash
ssh win-lenovo "cd /d D:\face_standardization_project && git pull origin main"
```

## File Locations

| What | Mac Path | 5060 Path |
|---|---|---|
| Project | `~/Documents/face_standardization_project/` | `D:\face_standardization_project\` |
| Python venv | N/A | `D:\face_standardization_project\.venv\` |
| Datasets | N/A | `D:\face_standardization_project\archive\` |
| Model weights | `~/Documents/face_standardization_project/DECA/data/` | `D:\face_standardization_project\DECA\data\` |
| Results | N/A | `D:\face_standardization_project\results\` |
