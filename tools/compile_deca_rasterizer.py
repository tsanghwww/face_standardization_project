import os
import sys

ROOT = r"D:\face_standardization_project"
CUDA_HOME = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
VENV_SCRIPTS = os.path.join(ROOT, ".venv", "Scripts")
VS_ENV_FILE = os.path.join(ROOT, ".torch_extensions", "vs_env.txt")

if os.path.exists(VS_ENV_FILE):
    preferred_path = None
    with open(VS_ENV_FILE, "r", encoding="mbcs", errors="ignore") as handle:
        for line in handle:
            key, sep, value = line.rstrip("\r\n").partition("=")
            if sep and key:
                if key == "PATH":
                    preferred_path = value
                if key == "Path" and preferred_path is not None:
                    continue
                os.environ[key] = value
    if preferred_path is not None:
        os.environ["PATH"] = preferred_path
        os.environ["Path"] = preferred_path

os.environ["CUDA_HOME"] = CUDA_HOME
os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(ROOT, ".torch_extensions")
os.environ["TORCH_CUDA_ARCH_LIST"] = "12.0"
base_path = os.environ.get("Path") or os.environ.get("PATH") or ""
merged_path = os.pathsep.join(
    [
        VENV_SCRIPTS,
        os.path.join(CUDA_HOME, "bin"),
        base_path,
    ]
)
os.environ["PATH"] = merged_path
os.environ["Path"] = merged_path

sys.path.insert(0, os.path.join(ROOT, "DECA"))

import torch.utils.cpp_extension as cpp_extension

cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")

from decalib.utils.renderer import set_rasterizer

set_rasterizer("standard")
print("standard rasterizer compiled/imported ok")
