import logging
import math
import types
import torch
import torch.nn.functional as F
import comfy

logger = logging.getLogger("ComfyUI-DyPE")
from comfy.model_patcher import ModelPatcher
from comfy import model_sampling

from .models.flux import PosEmbedFlux
from .models.nunchaku import PosEmbedNunchaku
from .models.qwen import PosEmbedQwen
from .models.zimage import PosEmbedZImage
from .models.anima import PosEmbedAnima

# Namespaced attribute for cache invalidation (stored on ModelPatcher, not raw model)
_DYPE_PARAMS_ATTR = "_comfyui_dype_params"


def _snap_to_multiple(value: int, multiple: int = 16) -> int:
    """Round value to the nearest multiple (minimum = multiple)."""
    snapped = max(multiple, round(value / multiple) * multiple)
    return snapped


def apply_dype_to_model(model: ModelPatcher, model_type: str, width: int, height: int, method: str, yarn_alt_scaling: bool, enable_dype: bool, dype_scale: float, dype_exponent: float, base_shift: float, max_shift: float, base_resolution: int = 1024, dype_start_sigma: float = 1.0) -> ModelPatcher:
    # Snap resolution to nearest multiple of 16 for latent space compatibility
    width = _snap_to_multiple(width, 16)
    height = _snap_to_multiple(height, 16)

    m = model.clone()

    is_nunchaku = False
    is_qwen = False
    is_z_image = False
    is_anima = False

    if model_type == "nunchaku":
        is_nunchaku = True
    elif model_type == "qwen":
        is_qwen = True
    elif model_type in ("z_image", "zimage"):
        is_z_image = True
    elif model_type == "anima":
        is_anima = True
    elif model_type == "flux":
        pass
    else: # auto
        if hasattr(m.model, "diffusion_model"):
            dm = m.model.diffusion_model
            model_class_name = dm.__class__.__name__
            if "QwenImage" in model_class_name:
                is_qwen = True
            elif "Anima" in model_class_name or "MiniTrainDIT" in model_class_name:
                is_anima = True
            elif hasattr(dm, "rope_embedder"):
                is_z_image = True
            elif hasattr(dm, "model") and hasattr(dm.model, "pos_embed"):
                is_nunchaku = True
            elif hasattr(dm, "pos_embedder") and hasattr(dm.pos_embedder, "dim_spatial_range"):
                is_anima = True
        else:
            raise ValueError("The provided model is not a compatible model.")

    detected_type = 'nunchaku' if is_nunchaku else 'qwen' if is_qwen else 'zimage' if is_z_image else 'anima' if is_anima else 'flux'
    logger.info(f"DyPE: Detected model type: {detected_type}")

    new_dype_params = (width, height, base_shift, max_shift, method, yarn_alt_scaling, base_resolution, dype_start_sigma, is_nunchaku, is_qwen, is_z_image, is_anima)

    should_patch_schedule = True
    if hasattr(m, _DYPE_PARAMS_ATTR):
        if getattr(m, _DYPE_PARAMS_ATTR) == new_dype_params:
            should_patch_schedule = False

    base_patch_h_tokens = None
    base_patch_w_tokens = None
    if is_z_image:
        axes_lens = getattr(m.model.diffusion_model, "axes_lens", None)
        if isinstance(axes_lens, (list, tuple)) and len(axes_lens) >= 3:
            base_patch_h_tokens = int(axes_lens[1])
            base_patch_w_tokens = int(axes_lens[2])

    patch_size = 2
    try:
        if is_nunchaku:
            patch_size = m.model.diffusion_model.model.config.patch_size
        elif is_anima:
            patch_size = m.model.diffusion_model.patch_spatial
        else:
            patch_size = m.model.diffusion_model.patch_size
    except (AttributeError, TypeError) as e:
        logger.warning(f"Could not read patch_size from model (defaulting to 2): {e}")

    if base_patch_h_tokens is not None and base_patch_w_tokens is not None:
        derived_base_patches = max(base_patch_h_tokens, base_patch_w_tokens)
        derived_base_seq_len = base_patch_h_tokens * base_patch_w_tokens
    else:
        derived_base_patches = (base_resolution // 8) // 2
        derived_base_seq_len = derived_base_patches * derived_base_patches

    if enable_dype and should_patch_schedule and not is_anima:
        try:
            if isinstance(m.model.model_sampling, model_sampling.ModelSamplingFlux) or is_qwen or is_z_image:
                latent_h, latent_w = height // 8, width // 8
                padded_h, padded_w = math.ceil(latent_h / patch_size) * patch_size, math.ceil(latent_w / patch_size) * patch_size
                image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)

                base_seq_len = derived_base_seq_len
                max_seq_len = derived_base_seq_len * 4

                effective_base_shift = base_shift
                effective_max_shift = max_shift

                if max_seq_len <= base_seq_len:
                    dype_shift = effective_base_shift
                else:
                    slope = (effective_max_shift - effective_base_shift) / (max_seq_len - base_seq_len)
                    intercept = effective_base_shift - slope * base_seq_len
                    dype_shift = image_seq_len * slope + intercept

                dype_shift = max(0.0, dype_shift)

                class DypeModelSamplingFlux(model_sampling.ModelSamplingFlux, model_sampling.CONST):
                    pass
                new_model_sampler = DypeModelSamplingFlux(m.model.model_config)
                new_model_sampler.set_parameters(shift=dype_shift)

                m.add_object_patch("model_sampling", new_model_sampler)
                setattr(m, _DYPE_PARAMS_ATTR, new_dype_params)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"DyPE noise schedule patching failed (model will use default schedule): {e}")

    elif not enable_dype and not is_anima:
        if hasattr(m, _DYPE_PARAMS_ATTR):
            class DefaultModelSamplingFlux(model_sampling.ModelSamplingFlux, model_sampling.CONST): pass
            default_sampler = DefaultModelSamplingFlux(m.model.model_config)
            m.add_object_patch("model_sampling", default_sampler)
            delattr(m, _DYPE_PARAMS_ATTR)

    try:
        if is_nunchaku:
            orig_embedder = m.model.diffusion_model.model.pos_embed
            target_patch_path = "diffusion_model.model.pos_embed"
        elif is_z_image:
            orig_embedder = m.model.diffusion_model.rope_embedder
            target_patch_path = "diffusion_model.rope_embedder"
        elif is_anima:
            orig_embedder = m.model.diffusion_model.pos_embedder
            target_patch_path = "diffusion_model.pos_embedder"
        else:
            orig_embedder = m.model.diffusion_model.pe_embedder
            target_patch_path = "diffusion_model.pe_embedder"

        if is_anima:
            theta_base = 10000.0
            dm = m.model.diffusion_model
            head_dim = dm.model_channels // dm.num_heads
            dim_h = head_dim // 6 * 2
            dim_t = head_dim - 2 * dim_h
            dim_w = dim_h
            axes_dim = [dim_t, dim_h, dim_w]
            # Read extrapolation ratios from the diffusion_model config (not from
            # orig_embedder, which may have been replaced by a previous DyPE call's
            # PosEmbedAnima instance that lacks t_ntk_factor/h_ntk_factor/w_ntk_factor).
            # The MiniTrainDIT stores rope_h/w/t_extrapolation_ratio as attributes.
            h_extrap = getattr(dm, 'rope_h_extrapolation_ratio', 1.0)
            w_extrap = getattr(dm, 'rope_w_extrapolation_ratio', 1.0)
            t_extrap = getattr(dm, 'rope_t_extrapolation_ratio', 1.0)
            # Compute NTK factors the same way VideoRopePosition3DEmb.__init__ does
            t_ntk = t_extrap ** (dim_t / (dim_t - 2))
            h_ntk = h_extrap ** (dim_h / (dim_h - 2))
            w_ntk = w_extrap ** (dim_w / (dim_w - 2))
            theta = [theta_base * t_ntk, theta_base * h_ntk, theta_base * w_ntk]
        else:
            theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX/Qwen model structure.")

    embedder_cls = PosEmbedFlux
    if is_nunchaku:
        embedder_cls = PosEmbedNunchaku
    elif is_qwen:
        embedder_cls = PosEmbedQwen
    elif is_z_image:
        embedder_cls = PosEmbedZImage
    elif is_anima:
        embedder_cls = PosEmbedAnima

    embedder_base_patches = derived_base_patches if is_z_image else None

    new_pe_embedder = embedder_cls(
        theta, axes_dim, method, yarn_alt_scaling, enable_dype,
        dype_scale, dype_exponent, base_resolution, dype_start_sigma, embedder_base_patches
    )
        
    m.add_object_patch(target_patch_path, new_pe_embedder)

    if is_z_image:
        base_hw_override = None
        if base_patch_h_tokens is not None and base_patch_w_tokens is not None:
            base_hw_override = (base_patch_h_tokens, base_patch_w_tokens)
        elif derived_base_patches is not None:
            base_hw_override = (derived_base_patches, derived_base_patches)

        if base_hw_override is not None:
            m.model.diffusion_model._dype_base_hw = base_hw_override

        # Compute isotropic scale hint for Z-Image RoPE.
        # This is set on the embedder before each forward pass via the wrapper.
        # We no longer override patchify_and_embed — the native Lumina model handles
        # position generation, and PosEmbedZImage applies DyPE scaling to whatever
        # positions it receives.
        raw_scale_y = float(base_resolution) / max(1.0, float(height))
        raw_scale_x = float(base_resolution) / max(1.0, float(width))
        iso_scale = min(raw_scale_y, raw_scale_x)
        zimage_freq_scale_factor = max(1.0, 1.0 / iso_scale)
        logger.debug(f"DyPE Z-Image: scale hint = {zimage_freq_scale_factor:.4f} (iso_scale={iso_scale:.4f})")

    sigma_max = m.model.model_sampling.sigma_max.item()
    
    def dype_wrapper_function(model_function, args_dict):
        timestep_tensor = args_dict.get("timestep")
        if timestep_tensor is not None and timestep_tensor.numel() > 0:
            current_sigma = timestep_tensor.flatten()[0].item()
            
            if sigma_max > 0:
                normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                new_pe_embedder.set_timestep(normalized_timestep)
        
        # Set Z-Image scale hint before each forward pass
        if is_z_image:
            new_pe_embedder.set_scale_hint(zimage_freq_scale_factor)

        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)

    return m
