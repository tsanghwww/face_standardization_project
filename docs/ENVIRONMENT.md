# Environment Configuration

## Control Node: MacBook Pro

| Item | Value |
|---|---|
| OS | macOS (Darwin, arm64) |
| Role | Code editing, Git management, SSH control, PIL maintenance |
| Project Path | `~/Documents/face_standardization_project/` |
| Python | Via system (no venv for project) |
| Git | Git via Homebrew |
| GitHub CLI | `gh` authenticated as `tsanghwww` |
| SSH Config | `~/.ssh/config` (hosts: win, win-pub, win-lenovo, aliyun-pune) |

## Compute Node: 5060 (win-lenovo)

| Item | Value |
|---|---|
| OS | Windows 11 |
| Hostname | win-lenovo (SSH alias) |
| IP | 192.168.1.121 |
| User | 47846 |
| SSH Key | `~/.ssh/id_rsa_windows` |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| VRAM | 8,151 MiB |
| CUDA | 13.3 (UMD), cu128 |
| Driver | 610.47 |

### Python Environment

| Item | Value |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| Venv Path | `D:\face_standardization_project\.venv\` |
| Activate | `.venv\Scripts\activate` |

### Key Python Packages

| Package | Version | Purpose |
|---|---|---|
| torch | 2.11.0+cu128 | Deep learning |
| [待确认] | | Other packages in venv |

## GitHub

| Item | Value |
|---|---|
| Repository | `tsanghwww/face_standardization_project` |
| Remote (SSH) | `git@github.com:tsanghwww/face_standardization_project.git` |
| 5060 SSH Key | `C:\Users\47846\.ssh\id_ed25519` |
| Branch | `main` |

## Network

| Connection | Details |
|---|---|
| Mac → 5060 (local) | SSH via 192.168.1.121 |
| 5060 → GitHub | SSH via id_ed25519 |
| Both → Internet | Required for package installs |

## Adding New Dependencies

1. Install on 5060: `.venv\Scripts\pip install <package>`
2. Update requirements file if tracked
3. Document here in ENVIRONMENT.md
4. Test import in both scripts
