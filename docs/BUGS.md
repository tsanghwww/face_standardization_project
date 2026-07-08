# Bugs

## Active

*None currently*

## Resolved

### BUG-001: CUDA rasterizer compilation failure on PyTorch 2.6+

- **Discovered**: [待确认] (pre-PIL)
- **Symptom**: DECA rasterizer failed to compile with deprecated `.data()` and `.type()` API calls
- **Root Cause**: PyTorch 2.6+ removed deprecated tensor methods
- **Resolution**: Updated `.data()` → `.data_ptr()` and `.type()` → `.scalar_type()` in `standard_rasterize_cuda_kernel.cu`
- **Status**: ✅ Resolved — committed as part of 5060 compatibility fixes

### BUG-003: landmark_score=0 — keypoint coordinate system mismatch

- **Discovered**: 2026-07-08 21:53
- **Severity**: High
- **Symptom**: Default-threshold Phase2 inference rejects nearly all samples. `landmark_score=0`, `landmark_out_ratio≈0.7`, `landmark_bbox_area≈0`, `landmark_center_dist≈0.707` across all 10K samples.
- **Root Cause**: DECA outputs kpt2d in orthographic projection space ([-1, 1]), but `read_kpt_quality()` in `phase2/features.py` treats them as 224×224 pixel coordinates. The denormalization step in DECA's decalib/deca.py L182 (`landmarks2d*image_size/2 + image_size/2`) is commented out. 70% of keypoints have negative coordinates → flagged out-of-bounds → score collapses to zero.
- **Resolution**: Added denormalization `xy = xy * (image_size/2) + (image_size/2)` in `features.py` L108.
- **Status**: ✅ Fixed — commit `17e4a4e`

### BUG-002: torch.load fails on Phase2 checkpoints with weights_only=True

- **Discovered**: [待确认] (pre-PIL)
- **Symptom**: `torch.load(checkpoint)` raised error due to numpy arrays in checkpoint
- **Root Cause**: PyTorch 2.6+ defaults to `weights_only=True` for security, but Phase2 checkpoints contain trusted numpy normalizer arrays
- **Resolution**: Added `weights_only=False` to `torch.load` in `phase2/infer_standardize_params.py`
- **Status**: ✅ Resolved

## Bug Report Template

```markdown
### BUG-NNN: Title

- **Discovered**: YYYY-MM-DD
- **Severity**: Critical / High / Medium / Low
- **Symptom**: ...
- **Reproduction Steps**:
  1. ...
  2. ...
- **Expected Behavior**: ...
- **Actual Behavior**: ...
- **Root Cause**: ...
- **Resolution**: ...
- **Status**: Open / Investigating / Fixed / Won't Fix
```
