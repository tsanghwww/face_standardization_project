@echo off
cd /d D:\face_standardization_project
call tools\deca_cuda_env.bat
if not exist .torch_extensions mkdir .torch_extensions
set > .torch_extensions\vs_env.txt
.\.venv\Scripts\python.exe tools\compile_deca_rasterizer.py
