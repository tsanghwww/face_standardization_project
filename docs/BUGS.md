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
