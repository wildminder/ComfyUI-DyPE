import torch
import torch.nn as nn
import math
import types
from comfy.model_patcher import ModelPatcher
from comfy import model_sampling
from .rope import get_1d_rotary_pos_embed


class FluxPosEmbed(nn.Module):
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', dype: bool = True, dype_exponent: float = 2.0): # Add dype_exponent
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method
        self.dype = dype if method != 'base' else False
        self.dype_exponent = dype_exponent
        self.current_timestep = 1.0
        self.base_resolution = 1024
        self.base_patches = (self.base_resolution // 8) // 2

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb_parts = []
        pos = ids.float()
        freqs_dtype = torch.bfloat16

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'repeat_interleave_real': True, 'use_real': True, 'freqs_dtype': freqs_dtype}
            
            # Pass the exponent to the RoPE function
            dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_exponent': self.dype_exponent}

            if i > 0:
                max_pos = axis_pos.max().item()
                current_patches = int(max_pos + 1)

                if self.method == 'yarn' and current_patches > self.base_patches:
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, yarn=True, max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches, **dype_kwargs)
                elif self.method == 'ntk' and current_patches > self.base_patches:
                    base_ntk_scale = (current_patches / self.base_patches)
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs, ntk_factor=base_ntk_scale, **dype_kwargs)
                else:
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs)
            else:
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)

            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)

def apply_dype_to_flux(model: ModelPatcher, width: int, height: int, method: str, enable_dype: bool, dype_exponent: float, base_shift: float, max_shift: float) -> ModelPatcher:
    m = model.clone()
    
    if not hasattr(m.model.model_sampling, "_dype_patched"):
        model_sampler = m.model.model_sampling
        if isinstance(model_sampler, model_sampling.ModelSamplingFlux):
            patch_size = m.model.diffusion_model.patch_size
            latent_h, latent_w = height // 8, width // 8
            padded_h, padded_w = math.ceil(latent_h / patch_size) * patch_size, math.ceil(latent_w / patch_size) * patch_size
            image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)
            base_seq_len, max_seq_len = 256, 4096
            slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            intercept = base_shift - slope * base_seq_len
            dype_shift = image_seq_len * slope + intercept

            def patched_sigma_func(self, timestep):
                return model_sampling.flux_time_shift(dype_shift, 1.0, timestep)

            model_sampler.sigma = types.MethodType(patched_sigma_func, model_sampler)
            model_sampler._dype_patched = True

    try:
        orig_embedder = m.model.diffusion_model.pe_embedder
        theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX model.")

    new_pe_embedder = FluxPosEmbed(theta, axes_dim, method, dype=enable_dype, dype_exponent=dype_exponent)
    m.add_object_patch("diffusion_model.pe_embedder", new_pe_embedder)
    
    sigma_max = m.model.model_sampling.sigma_max.item()

    def dype_wrapper_function(model_function, args_dict):
        if enable_dype:
            timestep_tensor = args_dict.get("timestep")
            if timestep_tensor is not None and timestep_tensor.numel() > 0:
                current_sigma = timestep_tensor.item()
                if sigma_max > 0:
                    normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                    new_pe_embedder.set_timestep(normalized_timestep)
        
        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)
    
    return m


class QwenPosEmbed(nn.Module):
    """
    Qwen-Image specific positional embedding with DyPE support.
    Optimized for Qwen's MMDiT architecture and MSRoPE implementation.
    """
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', dype: bool = True, dype_exponent: float = 2.0, base_resolution: int = 1024, patch_size: int = 2, editing_strength: float = 0.6, editing_mode: str = "adaptive"):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method
        self.dype = dype if method != 'base' else False
        self.dype_exponent = dype_exponent
        self.current_timestep = 1.0
        self.base_resolution = base_resolution
        self.patch_size = patch_size
        self.editing_strength = editing_strength
        self.editing_mode = editing_mode
        self.is_editing_mode = False  # Will be set dynamically during inference
        # Qwen-Image uses 8x VAE downsampling, then patches are further divided by patch_size
        # For Qwen, base_patches calculation: (base_resolution // 8) // patch_size
        self.base_patches = (self.base_resolution // 8) // patch_size

    def set_timestep(self, timestep: float, is_editing: bool = False):
        """Set current timestep for DyPE. Timestep normalized to [0, 1] where 1 is pure noise."""
        self.current_timestep = timestep
        self.is_editing_mode = is_editing

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Qwen positional embeddings.
        Returns embeddings in the format expected by Qwen's attention mechanism.
        """
        n_axes = ids.shape[-1]
        emb_parts = []
        pos = ids.float()
        
        # Qwen models typically use bfloat16 for better performance
        # Check device type for optimal dtype selection
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.bfloat16

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            
            common_kwargs = {
                'dim': axis_dim, 
                'pos': axis_pos, 
                'theta': self.theta, 
                'repeat_interleave_real': True, 
                'use_real': True, 
                'freqs_dtype': freqs_dtype
            }
            
            # Calculate effective DyPE strength (reduce for editing mode)
            effective_dype = self.dype
            effective_exponent = self.dype_exponent
            
            # Pass the exponent to the RoPE function
            dype_kwargs = {
                'dype': effective_dype, 
                'current_timestep': self.current_timestep, 
                'dype_exponent': effective_exponent
            }

            if i > 0:  # Spatial dimensions (height, width)
                max_pos = axis_pos.max().item()
                current_patches = int(max_pos + 1)
                
                # Calculate scale factor for editing mode (after we know current_patches)
                scale_factor = 1.0
                effective_exponent = self.dype_exponent
                
                if self.is_editing_mode and self.editing_mode != "full":
                    # Calculate effective strength based on editing mode
                    if self.editing_mode == "adaptive":
                        # Adaptive: Full DyPE early (structure), gradually reduce later (details)
                        # timestep 1.0 = pure noise (early), 0.0 = clean (late)
                        # Use more DyPE early, less late
                        timestep_factor = 0.3 + (self.current_timestep * 0.7)  # 1.0 at start, 0.3 at end
                        effective_strength = self.editing_strength * timestep_factor
                    elif self.editing_mode == "timestep_aware":
                        # More aggressive: Full early, minimal late
                        timestep_factor = 0.2 + (self.current_timestep * 0.8)  # 1.0 at start, 0.2 at end
                        effective_strength = self.editing_strength * timestep_factor
                    elif self.editing_mode == "resolution_aware":
                        # Only reduce when editing at high resolutions
                        if current_patches > self.base_patches:
                            effective_strength = self.editing_strength
                        else:
                            effective_strength = 1.0  # Full strength at base resolution
                    elif self.editing_mode == "minimal":
                        # Minimal DyPE for editing (original approach)
                        effective_strength = self.editing_strength
                    else:  # "full" or unknown
                        effective_strength = 1.0
                    
                    # Apply effective strength
                    if effective_strength < 1.0:
                        effective_exponent = self.dype_exponent * effective_strength
                        dype_kwargs['dype_exponent'] = effective_exponent
                        
                        # Calculate scale factor for extrapolation reduction
                        if current_patches > self.base_patches:
                            # Scale down the extrapolation ratio for editing
                            # More conservative scaling for adaptive modes
                            if self.editing_mode in ["adaptive", "timestep_aware"]:
                                # Use timestep-aware scaling for extrapolation too
                                timestep_scale = 0.5 + (self.current_timestep * 0.5)  # 1.0 at start, 0.5 at end
                                scale_factor = 1.0 - (1.0 - effective_strength) * 0.3 * (1.0 - timestep_scale * 0.5)
                            else:
                                scale_factor = 1.0 - (1.0 - effective_strength) * 0.4
                        else:
                            scale_factor = 1.0
                else:
                    scale_factor = 1.0

                if self.method == 'yarn' and current_patches > self.base_patches:
                    # Apply scale factor for editing mode
                    if self.is_editing_mode and scale_factor < 1.0:
                        # Interpolate between base and target patches based on editing_strength
                        adjusted_patches = int(self.base_patches + (current_patches - self.base_patches) * scale_factor)
                        max_pe_len = torch.tensor(adjusted_patches, dtype=freqs_dtype, device=pos.device)
                    else:
                        max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_rotary_pos_embed(
                        **common_kwargs, 
                        yarn=True, 
                        max_pe_len=max_pe_len, 
                        ori_max_pe_len=self.base_patches, 
                        **dype_kwargs
                    )
                elif self.method == 'ntk' and current_patches > self.base_patches:
                    # Apply scale factor for editing mode
                    if self.is_editing_mode and scale_factor < 1.0:
                        base_ntk_scale = 1.0 + ((current_patches / self.base_patches) - 1.0) * scale_factor
                    else:
                        base_ntk_scale = (current_patches / self.base_patches)
                    cos, sin = get_1d_rotary_pos_embed(
                        **common_kwargs, 
                        ntk_factor=base_ntk_scale, 
                        **dype_kwargs
                    )
                else:
                    cos, sin = get_1d_rotary_pos_embed(**common_kwargs)
            else:  # Channel dimension (typically not extrapolated)
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)

            # Qwen's attention expects cos/sin format, convert to rotation matrix
            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)


def _detect_qwen_model_structure(model: ModelPatcher):
    """
    Detect Qwen-Image model structure and extract key parameters.
    Returns a dictionary with detected attributes.
    """
    structure = {
        'transformer': None,
        'transformer_path': None,
        'pos_embed': None,
        'pos_embed_path': None,
        'patch_size': 2,  # Default for MMDiT models
        'vae_scale_factor': 8,  # Default VAE downsampling
        'base_resolution': 1024,  # Qwen-Image base training resolution
    }
    
    # Try to find transformer
    if hasattr(model.model, "transformer"):
        structure['transformer'] = model.model.transformer
        structure['transformer_path'] = "transformer"
    elif hasattr(model.model, "diffusion_model"):
        structure['transformer'] = model.model.diffusion_model
        structure['transformer_path'] = "diffusion_model"
    else:
        return None
    
    transformer = structure['transformer']
    
    # Try to find positional embedder
    if hasattr(transformer, "pos_embed"):
        structure['pos_embed'] = transformer.pos_embed
        structure['pos_embed_path'] = f"{structure['transformer_path']}.pos_embed"
    elif hasattr(transformer, "pe_embedder"):
        structure['pos_embed'] = transformer.pe_embedder
        structure['pos_embed_path'] = f"{structure['transformer_path']}.pe_embedder"
    else:
        return None
    
    # Extract patch_size if available
    if hasattr(transformer, "patch_size"):
        structure['patch_size'] = transformer.patch_size
    elif hasattr(transformer, "config") and hasattr(transformer.config, "patch_size"):
        structure['patch_size'] = transformer.config.patch_size
    
    # Extract VAE scale factor if available
    if hasattr(model.model, "vae_scale_factor"):
        structure['vae_scale_factor'] = model.model.vae_scale_factor
    elif hasattr(model.model, "vae") and hasattr(model.model.vae, "scale_factor"):
        structure['vae_scale_factor'] = model.model.vae.scale_factor
    
    # Try to detect base resolution from config
    if hasattr(transformer, "config"):
        config = transformer.config
        if hasattr(config, "sample_size"):
            # sample_size is typically the latent size, multiply by 8 for image size
            structure['base_resolution'] = config.sample_size * 8
        elif hasattr(config, "base_resolution"):
            structure['base_resolution'] = config.base_resolution
    
    return structure


def apply_dype_to_qwen(model: ModelPatcher, width: int, height: int, method: str, enable_dype: bool, dype_exponent: float, base_shift: float, max_shift: float, editing_strength: float = 0.6, editing_mode: str = "adaptive") -> ModelPatcher:
    """
    Apply DyPE to a Qwen-Image model with architecture-specific optimizations.
    """
    m = model.clone()
    
    # Detect Qwen model structure
    structure = _detect_qwen_model_structure(m)
    if structure is None:
        raise ValueError("Could not detect Qwen-Image model structure. This node is only compatible with Qwen-Image models.")
    
    transformer = structure['transformer']
    patch_size = structure['patch_size']
    vae_scale_factor = structure['vae_scale_factor']
    base_resolution = structure['base_resolution']
    
    # Patch noise schedule if available (Qwen may use FlowMatch or similar schedulers)
    if not hasattr(m.model.model_sampling, "_dype_patched"):
        model_sampler = m.model.model_sampling
        
        # Check if it's a compatible sampler
        if hasattr(model_sampler, "sigma_max"):
            # Calculate sequence length based on Qwen's architecture
            latent_h, latent_w = height // vae_scale_factor, width // vae_scale_factor
            # Qwen uses patch_size for further downsampling
            padded_h = math.ceil(latent_h / patch_size) * patch_size
            padded_w = math.ceil(latent_w / patch_size) * patch_size
            image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)
            
            # Qwen-specific sequence length parameters
            base_seq_len, max_seq_len = 256, 4096
            slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
            intercept = base_shift - slope * base_seq_len
            dype_shift = image_seq_len * slope + intercept

            def patched_sigma_func(self, timestep):
                # Try to use flux_time_shift if available (Qwen may use similar schedulers)
                try:
                    return model_sampling.flux_time_shift(dype_shift, 1.0, timestep)
                except AttributeError:
                    # Fallback for other scheduler types (FlowMatch, etc.)
                    # Apply shift proportionally to timestep
                    if hasattr(self, "sigma"):
                        original_sigma = self.sigma.__func__(self, timestep) if hasattr(self.sigma, "__func__") else timestep
                        return original_sigma * (1.0 + dype_shift * 0.1)  # Conservative scaling
                    return timestep

            model_sampler.sigma = types.MethodType(patched_sigma_func, model_sampler)
            model_sampler._dype_patched = True

    # Find and extract positional embedder parameters
    orig_embedder = structure['pos_embed']
    
    # Extract theta and axes_dim from the original embedder
    if hasattr(orig_embedder, "theta") and hasattr(orig_embedder, "axes_dim"):
        theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    elif hasattr(orig_embedder, "theta") and hasattr(orig_embedder, "axes_dims_rope"):
        theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dims_rope
    elif hasattr(orig_embedder, "theta"):
        # If only theta is available, use Qwen-Image defaults
        theta = orig_embedder.theta
        # Qwen-Image typically uses (16, 56, 56) for axes_dims_rope
        axes_dim = [16, 56, 56]
    else:
        # Fallback to Qwen-Image defaults
        theta = 10000
        axes_dim = [16, 56, 56]  # Default for Qwen-Image MMDiT models

    # Create new positional embedder with Qwen-specific parameters
    new_pe_embedder = QwenPosEmbed(
        theta=theta, 
        axes_dim=axes_dim, 
        method=method, 
        dype=enable_dype,  # Note: parameter is 'dype' not 'enable_dype'
        dype_exponent=dype_exponent,
        base_resolution=base_resolution,
        patch_size=patch_size,
        editing_strength=editing_strength,
        editing_mode=editing_mode
    )
    
    # Patch the positional embedder using the detected path
    m.add_object_patch(structure['pos_embed_path'], new_pe_embedder)
    
    # Get sigma_max for timestep normalization
    sigma_max = 1.0
    if hasattr(m.model.model_sampling, "sigma_max"):
        sigma_max_val = m.model.model_sampling.sigma_max
        if hasattr(sigma_max_val, "item"):
            sigma_max = sigma_max_val.item()
        else:
            sigma_max = float(sigma_max_val)
        if sigma_max <= 0:
            sigma_max = 1.0

    # Capture editing_mode in closure
    def dype_wrapper_function(model_function, args_dict):
        """
        Wrapper function to update timestep for DyPE during inference.
        Optimized for Qwen's forward pass signature with editing mode detection.
        """
        # Detect editing mode by checking for image inputs in conditioning
        is_editing = False
        c = args_dict.get("c", {})
        input_x = args_dict.get("input")
        
        # Check for image editing indicators in conditioning
        # Qwen-Image editing typically includes image embeddings or image tokens in conditioning
        if isinstance(c, dict):
            # Check for common image editing keys in Qwen models
            editing_keys = ['image', 'image_embeds', 'image_tokens', 'concat_latent_image', 'concat_mask', 'concat_mask_image']
            for key in editing_keys:
                if key in c and c[key] is not None:
                    is_editing = True
                    break
            
            # Also check if input contains non-zero values (not pure noise/empty latent)
            # This is a heuristic: editing often starts with a partially denoised image
            if input_x is not None and hasattr(input_x, 'abs'):
                # If input has low variance, it might be an edited image rather than pure noise
                input_variance = input_x.abs().mean().item() if hasattr(input_x, 'abs') else 0.0
                # Pure noise typically has higher variance, edited images have lower
                # This is a rough heuristic - adjust threshold as needed
                if input_variance < 0.5:  # Threshold may need tuning
                    is_editing = True
        
        if enable_dype:
            timestep_tensor = args_dict.get("timestep")
            if timestep_tensor is not None:
                # Handle both tensor and scalar timestep values
                if hasattr(timestep_tensor, "numel") and timestep_tensor.numel() > 0:
                    current_sigma = timestep_tensor.item() if hasattr(timestep_tensor, "item") else float(timestep_tensor)
                else:
                    current_sigma = float(timestep_tensor) if not isinstance(timestep_tensor, (int, float)) else timestep_tensor
                
                if sigma_max > 0:
                    # Improved timestep normalization for editing
                    # Editing often uses different timestep ranges, so we normalize more carefully
                    normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                    
                    # For adaptive/timestep_aware modes, we want full DyPE early (structure) and less late (details)
                    # So we don't need to reduce the normalized timestep - the mode handles it
                    # Only apply conservative scaling for minimal mode
                    if is_editing and editing_mode == "minimal" and current_sigma < sigma_max * 0.3:
                        # For early timesteps in minimal mode, preserve more structure
                        normalized_timestep = normalized_timestep * 0.8
                    
                    new_pe_embedder.set_timestep(normalized_timestep, is_editing=is_editing)
        
        # Forward pass with original arguments
        timestep = args_dict.get("timestep")
        return model_function(input_x, timestep, **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)
    
    return m