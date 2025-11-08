# Qwen-Image Specific Improvements

This document outlines the architecture-specific improvements made to optimize DyPE for Qwen-Image models.

## Key Improvements

### 1. **Intelligent Model Structure Detection**
- Added `_detect_qwen_model_structure()` function that automatically detects:
  - Transformer/diffusion_model location
  - Positional embedder path (`pos_embed` vs `pe_embedder`)
  - Patch size from model config
  - VAE scale factor
  - Base training resolution
- Eliminates hardcoded assumptions and adapts to different Qwen model variants

### 2. **Qwen-Specific Parameter Extraction**
- **Patch Size Detection**: Automatically extracts `patch_size` from model config (defaults to 2 for MMDiT)
- **VAE Scale Factor**: Detects actual VAE downsampling factor (typically 8x)
- **Base Resolution**: Attempts to detect from model config, falls back to 1024
- **Axes Dimensions**: Extracts from model or uses Qwen-Image defaults `[16, 56, 56]`

### 3. **Optimized Base Patches Calculation**
```python
# Old: Hardcoded calculation
self.base_patches = (self.base_resolution // 8) // 2

# New: Uses detected patch_size and base_resolution
self.base_patches = (self.base_resolution // vae_scale_factor) // patch_size
```
- More accurate for different Qwen model variants
- Adapts to actual model architecture

### 4. **Enhanced Positional Embedding Class**
- Added `base_resolution` and `patch_size` parameters to `QwenPosEmbed`
- Better device-aware dtype selection (handles MPS, NPU, CUDA)
- Improved comments explaining Qwen-specific behavior
- More robust handling of different tensor formats

### 5. **Improved Scheduler Compatibility**
- Better fallback for non-Flux schedulers (FlowMatch, etc.)
- Conservative scaling approach for unknown scheduler types
- More robust error handling with `AttributeError` instead of bare `except`

### 6. **Better Sequence Length Calculation**
```python
# Now uses detected vae_scale_factor and patch_size
latent_h, latent_w = height // vae_scale_factor, width // vae_scale_factor
padded_h = math.ceil(latent_h / patch_size) * patch_size
padded_w = math.ceil(latent_w / patch_size) * patch_size
image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)
```
- More accurate for Qwen's specific architecture
- Accounts for both VAE downsampling and patch-based downsampling

### 7. **Enhanced Timestep Handling**
- Better handling of different timestep formats (tensor, scalar, etc.)
- More robust normalization logic
- Improved error handling for edge cases

### 8. **Architecture-Aware Defaults**
- Qwen-Image specific defaults:
  - `axes_dim = [16, 56, 56]` (MMDiT standard)
  - `theta = 10000` (RoPE base frequency)
  - `patch_size = 2` (MMDiT patch size)
  - `vae_scale_factor = 8` (standard VAE downsampling)

## Benefits

1. **Better Compatibility**: Works with different Qwen-Image model variants
2. **More Accurate**: Uses actual model parameters instead of assumptions
3. **Robust**: Better error handling and fallbacks
4. **Optimized**: Qwen-specific optimizations for better performance
5. **Maintainable**: Clear structure detection makes debugging easier

## Testing Recommendations

When testing with your Qwen-Image model:

1. Check console output for detected parameters (add logging if needed)
2. Verify patch_size matches your model (typically 2 for MMDiT)
3. Verify base_resolution matches training resolution
4. Test with different resolutions to ensure proper extrapolation
5. Monitor for any warnings about fallback behavior

## Future Enhancements

Potential further improvements:

1. **MSRoPE Integration**: Qwen uses Multimodal Scalable RoPE - could add specific support
2. **Aspect Ratio Presets**: Qwen supports specific aspect ratios - could add presets
3. **Text Rendering Optimization**: Qwen excels at text - could add text-specific optimizations
4. **Multi-Image Support**: Qwen-Image-Edit supports multi-image - could extend for that
5. **Config File Support**: Allow users to override detected parameters via config

