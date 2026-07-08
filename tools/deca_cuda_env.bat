@echo off
chcp 65001 > nul
set "VSLANG=1033"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "TORCH_EXTENSIONS_DIR=D:\face_standardization_project\.torch_extensions"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PATH=D:\face_standardization_project\.venv\Scripts;%CUDA_HOME%\bin;%PATH%"
