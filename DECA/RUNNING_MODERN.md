# DECA Modern Runtime

This project was originally written for Python 3.7, PyTorch 1.6, and CUDA 10.1.
The files below add a modern PyTorch path while keeping the original project intact.

## 1. Create Environment

Recommended on this Mac:

```bash
cd /Users/houwingtsang/Documents/face_standardization_project/DECA
conda env create -f environment_modern.yml
conda activate deca-modern
python -m pip install chumpy --no-build-isolation
```

## 2. Install Current PyTorch

On Apple Silicon macOS, upgrade to the newest wheels pip can resolve:

```bash
python -m pip install --upgrade torch torchvision torchaudio
```

On this machine that resolved to:

```text
torch 2.12.0
torchvision 0.27.0
torchaudio 2.11.0
```

On Linux with NVIDIA CUDA 13.0:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
```

On Linux CPU-only:

```bash
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

## 3. Smoke Test Imports

```bash
python - <<'PY'
import torch, torchvision, cv2, skimage, scipy, yaml
print('torch', torch.__version__)
print('torchvision', torchvision.__version__)
print('cuda', torch.cuda.is_available())
print('mps', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())
PY
```

## 4. Run Without Renderer

This works when CUDA/PyTorch3D rendering is unavailable. It runs image loading,
face crop, model inference, FLAME geometry, and saves keypoints/parameters.

```bash
python demos/demo_reconstruct.py \
  -i TestSamples/examples \
  -s TestSamples/examples/results_modern_no_render \
  --device cpu \
  --rendering False \
  --iscrop False \
  --saveKpt True \
  --saveMat True
```

Use `--iscrop False` when the input images are already reasonably face-centered;
it avoids the face-alignment detector download/initialization path. Use
`--device mps` only after the CPU run works. Some older tensor operations in
research code may still fall back poorly on MPS.

## 5. Full Rendered Outputs

For visualization images, depth, and textured OBJ files, this project still needs
a renderer.

Linux + NVIDIA GPU:

```bash
python demos/demo_reconstruct.py \
  -i TestSamples/examples \
  -s TestSamples/examples/results_modern_rendered \
  --device cuda \
  --rasterizer_type standard \
  --saveDepth True \
  --saveObj True
```

macOS:

```bash
MACOSX_DEPLOYMENT_TARGET=10.14 CC=clang CXX=clang++ \
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

Then try:

```bash
python demos/demo_reconstruct.py \
  -i TestSamples/examples \
  -s TestSamples/examples/results_modern_rendered \
  --device cpu \
  --rasterizer_type pytorch3d \
  --saveDepth True
```

PyTorch3D on macOS can require local compilation and may fail depending on Xcode,
Python, and PyTorch versions. The no-render command above is the reliable modern
Mac path.

## Notes

- Data files are already present under `data/`, including `generic_model.pkl`,
  `deca_model.tar`, and the albedo model.
- `fetch_data.sh` is not needed unless you want to re-download licensed assets.
- The original `requirements.txt` is kept for the legacy Python 3.7/CUDA setup.
