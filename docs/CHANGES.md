# Recent Changes and Fixes

## Deformable Convolution Fix (2026-01-29)

### Issue
The deformable convolution implementation was causing `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED` due to non-contiguous tensors being passed to `F.grid_sample()`.

### Fix
Added `.contiguous()` calls after all `reshape()` operations that feed into `grid_sample()` in both `DeformableConv2d` and `ModulatedDeformableConv2d` classes.

**Files Modified:**
- `src/models/deform_conv.py`: Lines 180, 189, 197, 416, 424, 431, 453

### Changes Summary

1. **`DeformableConv2d._deform_conv()` method:**
   - Line 180: Added `.contiguous()` to `sample_coords.reshape()`
   - Line 189: Added `.contiguous()` to `x_expanded.reshape()`
   - Line 197: Added `.contiguous()` to `sample_coords_expanded.reshape()`

2. **`ModulatedDeformableConv2d._deform_conv()` method:**
   - Line 416: Added `.contiguous()` to `grid.reshape()`
   - Line 424: Added `.contiguous()` to `x_expanded.reshape()`
   - Line 431: Added `.contiguous()` to `grid_expanded.reshape()`
   - Fixed einsum operation to match the working implementation from `DeformableConv2d`

### Configuration Update

Changed default `num_rrdb_blocks` from 16 to 12 in `configs/lp_asrn.yaml` for improved stability and memory efficiency.

## OCR Model Change (2026-01-29)

### Issue
Parseq model had vocabulary compatibility issues causing CUDA errors during training.

### Fix
Switched to SimpleCRNN as the primary OCR model. SimpleCRNN:
- Uses a simple 36-character vocabulary (0-9, A-Z)
- CNN+BiLSTM architecture designed specifically for license plates
- No complex tokenization issues
- Faster training

**Files Modified:**
- `src/ocr/parseq_wrapper.py`: Modified `ParseqOCR` to use `SimpleCRNN` fallback
- `requirements.txt`: Removed Parseq-specific dependencies
- Documentation updated to reflect SimpleCRNN usage
