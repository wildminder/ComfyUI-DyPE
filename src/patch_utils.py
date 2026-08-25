import logging
import math

logger = logging.getLogger("ComfyUI-DyPE")
from comfy import model_sampling
from comfy.model_patcher import ModelPatcher

from .models.anima import PosEmbedAnima
from .models.flux import PosEmbedFlux
from .models.nunchaku import PosEmbedNunchaku
from .models.qwen import PosEmbedQwen
from .models.sega_anima import SegAPosEmbedAnima
from .models.sega_flux import SegAPosEmbedFlux
from .models.sega_nunchaku import SegAPosEmbedNunchaku
from .models.sega_qwen import SegAPosEmbedQwen
from .models.sega_zimage import SegAPosEmbedZImage
from .models.zimage import PosEmbedZImage
from .sega import compute_axis_spectral_profiles, compute_dynamic_spread, compute_spectral_energy_profile

# Namespaced attribute for cache invalidation (stored on ModelPatcher, not raw model)
_DYPE_PARAMS_ATTR = "_comfyui_dype_params"


def _snap_to_multiple(value: int, multiple: int = 16) -> int:
    """Round value to the nearest multiple (minimum = multiple)."""
    snapped = max(multiple, round(value / multiple) * multiple)
    return snapped


def _dype_sega_reject_spa(orig_embedder) -> None:
    """Mirror of the SPA exclusivity guard (remediation decision 6).

    Rejects applying DyPE/SEGA when an SPA embedder is already present.  Uses a
    name-based check to avoid a circular import with ``src.spa``.
    """
    name = type(orig_embedder).__name__
    if name.startswith("PosEmbedSPA") or "SPA" in name:
        raise ValueError(
            "SPA and DyPE/SEGA are mutually exclusive in v1. Apply only one."
        )


# ---------------------------------------------------------------------------
# W4.4 (IMP-007) — shared model-geometry resolution
# ---------------------------------------------------------------------------

from dataclasses import dataclass  # noqa: E402  (kept beside its only users)


@dataclass(frozen=True)
class ModelGeometry:
    """Resolved per-model geometry shared by the DyPE and SEGA installers.

    Attributes mirror the values the two duplicated blocks used to compute
    inline (verbatim semantics — see tests/test_geometry_resolution.py).
    """

    patch_size: int                  # token size of one latent patch (default 2)
    base_patch_h_tokens: int | None  # Z-Image axes_lens[1]
    base_patch_w_tokens: int | None  # Z-Image axes_lens[2]
    derived_base_patches: int        # base grid side in patches
    derived_base_seq_len: int        # base sequence length (patches^2, or h*w)
    detected: str                    # from resolve_model_type


def resolve_model_geometry(m: ModelPatcher, model_type: str,
                           base_resolution: int = 1024) -> ModelGeometry:
    """Single source for detection + patch-size + base-grid resolution.

    Both :func:`apply_dype_to_model` and :func:`apply_sega_to_model` consumed
    byte-identical copies of this logic; they now call this once.
    """
    from .model_detect import resolve_model_type  # late-bound: test-seam friendly

    dm = m.model.diffusion_model
    detected = resolve_model_type(dm, model_type)
    logger.info(f"Detected model type: {detected}")

    base_patch_h_tokens = None
    base_patch_w_tokens = None
    if detected == "zimage":
        axes_lens = getattr(dm, "axes_lens", None)
        if isinstance(axes_lens, (list, tuple)) and len(axes_lens) >= 3:
            base_patch_h_tokens = int(axes_lens[1])
            base_patch_w_tokens = int(axes_lens[2])

    patch_size = 2
    try:
        if detected == "nunchaku":
            patch_size = dm.model.config.patch_size
        elif detected == "anima":
            patch_size = dm.patch_spatial
        else:
            patch_size = dm.patch_size
    except (AttributeError, TypeError) as e:
        logger.warning(f"Could not read patch_size from model (defaulting to 2): {e}")

    if base_patch_h_tokens is not None and base_patch_w_tokens is not None:
        derived_base_patches = max(base_patch_h_tokens, base_patch_w_tokens)
        derived_base_seq_len = base_patch_h_tokens * base_patch_w_tokens
    else:
        derived_base_patches = (base_resolution // 8) // 2
        derived_base_seq_len = derived_base_patches * derived_base_patches

    return ModelGeometry(
        patch_size=patch_size,
        base_patch_h_tokens=base_patch_h_tokens,
        base_patch_w_tokens=base_patch_w_tokens,
        derived_base_patches=derived_base_patches,
        derived_base_seq_len=derived_base_seq_len,
        detected=detected,
    )


def apply_dype_to_model(model: ModelPatcher, model_type: str, width: int, height: int, method: str, yarn_alt_scaling: bool, enable_dype: bool, dype_scale: float, dype_exponent: float, base_shift: float, max_shift: float, base_resolution: int = 1024, dype_start_sigma: float = 1.0) -> ModelPatcher:
    # Snap resolution to nearest multiple of 16 for latent space compatibility
    width = _snap_to_multiple(width, 16)
    height = _snap_to_multiple(height, 16)

    m = model.clone()

    # W4.4 (IMP-007): detection + geometry via the shared resolver — the two
    # duplicated inline blocks (here and in apply_sega_to_model) are gone.
    geo = resolve_model_geometry(m, model_type, base_resolution)
    detected_type = geo.detected
    logger.info(f"DyPE: Detected model type: {detected_type}")

    is_nunchaku = detected_type == "nunchaku"
    is_qwen = detected_type == "qwen"
    is_z_image = detected_type == "zimage"
    is_anima = detected_type == "anima"

    new_dype_params = (width, height, base_shift, max_shift, method, yarn_alt_scaling, base_resolution, dype_start_sigma, is_nunchaku, is_qwen, is_z_image, is_anima)

    should_patch_schedule = True
    if hasattr(m, _DYPE_PARAMS_ATTR):
        if getattr(m, _DYPE_PARAMS_ATTR) == new_dype_params:
            should_patch_schedule = False

    patch_size = geo.patch_size
    derived_base_patches = geo.derived_base_patches
    derived_base_seq_len = geo.derived_base_seq_len

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

        _dype_sega_reject_spa(orig_embedder)

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
        # W6.2a (IMP-003): the ``_dype_base_hw`` write was REMOVED — the attr
        # was write-only (no reader anywhere in src/); the embedders receive
        # their base grid via constructor args instead.  The shared
        # diffusion_model no longer carries DyPE-private state.

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


# ---------------------------------------------------------------------------
# SEGA (Spectral-Energy Guided Attention)
# ---------------------------------------------------------------------------

def apply_sega_to_model(
    model: ModelPatcher,
    model_type: str,
    width: int,
    height: int,
    method: str = "sega",
    mscale_alpha: float = 0.15,
    mscale_beta: float = 1.5,
    mscale_min: float = 1.0,
    spread_min: float = 0.0,
    spread_max: float = 1.0,
    spread_alpha: float = 1.5,
    base_mscale_formula: str = "power_res",
    base_mscale_coefficient: float | None = None,
    base_resolution: int = 1024,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> ModelPatcher:
    """Apply SEGA (Spectral-Energy Guided Attention) to a model.

    SEGA computes per-RoPE-dimension mscale from the latent's Fourier
    spectrum at each denoising step, providing content-aware attention
    sharpening for ultra-high-resolution generation.

    Unlike DyPE, SEGA needs access to the hidden states (latent) at each
    step to compute the spectral profiles.  This is handled via the
    unet function wrapper.
    """
    # Snap resolution to nearest multiple of 16
    width = _snap_to_multiple(width, 16)
    height = _snap_to_multiple(height, 16)

    m = model.clone()

    # --- W4.4 (IMP-007): detection + geometry via the shared resolver ---
    geo = resolve_model_geometry(m, model_type, base_resolution)
    detected_type = geo.detected
    logger.info(f"SEGA: Detected model type: {detected_type}")

    is_nunchaku = detected_type == "nunchaku"
    is_qwen = detected_type == "qwen"
    is_z_image = detected_type == "zimage"
    is_anima = detected_type == "anima"

    patch_size = geo.patch_size
    derived_base_patches = geo.derived_base_patches

    # --- Noise schedule patching (same as DyPE, except Anima) ---
    if not is_anima:
        try:
            if isinstance(m.model.model_sampling, model_sampling.ModelSamplingFlux) or is_qwen or is_z_image:
                latent_h, latent_w = height // 8, width // 8
                padded_h, padded_w = math.ceil(latent_h / patch_size) * patch_size, math.ceil(latent_w / patch_size) * patch_size
                image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)

                base_seq_len = derived_base_patches * derived_base_patches
                max_seq_len = derived_base_patches * derived_base_patches * 4

                effective_base_shift = base_shift
                effective_max_shift = max_shift

                if max_seq_len <= base_seq_len:
                    sega_shift = effective_base_shift
                else:
                    slope = (effective_max_shift - effective_base_shift) / (max_seq_len - base_seq_len)
                    intercept = effective_base_shift - slope * base_seq_len
                    sega_shift = image_seq_len * slope + intercept

                sega_shift = max(0.0, sega_shift)

                class SegaModelSamplingFlux(model_sampling.ModelSamplingFlux, model_sampling.CONST):
                    pass
                new_model_sampler = SegaModelSamplingFlux(m.model.model_config)
                new_model_sampler.set_parameters(shift=sega_shift)

                m.add_object_patch("model_sampling", new_model_sampler)
        except (AttributeError, TypeError, ValueError) as e:
            logger.warning(f"SEGA noise schedule patching failed (model will use default schedule): {e}")

    # --- Get theta and axes_dim from the original embedder ---
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

        _dype_sega_reject_spa(orig_embedder)

        if is_anima:
            theta_base = 10000.0
            dm = m.model.diffusion_model
            head_dim = dm.model_channels // dm.num_heads
            dim_h = head_dim // 6 * 2
            dim_t = head_dim - 2 * dim_h
            dim_w = dim_h
            axes_dim = [dim_t, dim_h, dim_w]
            h_extrap = getattr(dm, 'rope_h_extrapolation_ratio', 1.0)
            w_extrap = getattr(dm, 'rope_w_extrapolation_ratio', 1.0)
            t_extrap = getattr(dm, 'rope_t_extrapolation_ratio', 1.0)
            t_ntk = t_extrap ** (dim_t / (dim_t - 2))
            h_ntk = h_extrap ** (dim_h / (dim_h - 2))
            w_ntk = w_extrap ** (dim_w / (dim_w - 2))
            theta = [theta_base * t_ntk, theta_base * h_ntk, theta_base * w_ntk]
        else:
            theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX/Qwen model structure.")

    # --- Select SEGA embedder class ---
    sega_embedder_cls = SegAPosEmbedFlux
    if is_nunchaku:
        sega_embedder_cls = SegAPosEmbedNunchaku
    elif is_qwen:
        sega_embedder_cls = SegAPosEmbedQwen
    elif is_z_image:
        sega_embedder_cls = SegAPosEmbedZImage
    elif is_anima:
        sega_embedder_cls = SegAPosEmbedAnima

    embedder_base_patches = derived_base_patches if is_z_image else None

    # For Anima, use the native patch grid from the model config
    if is_anima:
        try:
            dm = m.model.diffusion_model
            max_img_h = getattr(dm, 'max_img_h', None)
            patch_spatial = getattr(dm, 'patch_spatial', 2)
            if max_img_h is not None:
                native_patches = max_img_h // patch_spatial
                embedder_base_patches = native_patches
        except Exception as e:  # degrade: optional Anima grid read
            logger.debug(f"Could not read Anima native patch grid: {e}")

    new_pe_embedder = sega_embedder_cls(
        theta, axes_dim, method=method,
        yarn_alt_scaling=False, dype=False,
        dype_scale=2.0, dype_exponent=2.0,
        base_resolution=base_resolution, dype_start_sigma=1.0,
        base_patch_grid=embedder_base_patches,
        # SEGA-specific parameters
        mscale_alpha=mscale_alpha,
        mscale_beta=mscale_beta,
        mscale_min=mscale_min,
        spread_min=spread_min,
        spread_max=spread_max,
        spread_alpha=spread_alpha,
        base_mscale_formula=base_mscale_formula,
        base_mscale_coefficient=base_mscale_coefficient,
        training_res_pixels=base_resolution,
    )

    m.add_object_patch(target_patch_path, new_pe_embedder)

    # --- Z-Image scale hint ---
    if is_z_image:
        # W6.2a (IMP-003): ``_dype_base_hw`` write removed (write-only attr).
        raw_scale_y = float(base_resolution) / max(1.0, float(height))
        raw_scale_x = float(base_resolution) / max(1.0, float(width))
        iso_scale = min(raw_scale_y, raw_scale_x)
        zimage_freq_scale_factor = max(1.0, 1.0 / iso_scale)
    else:
        zimage_freq_scale_factor = 1.0

    sigma_max = m.model.model_sampling.sigma_max.item()

    # --- SEGA wrapper: computes spectral profiles from latent at each step ---
    def sega_wrapper_function(model_function, args_dict):
        timestep_tensor = args_dict.get("timestep")
        if timestep_tensor is not None and timestep_tensor.numel() > 0:
            current_sigma = timestep_tensor.flatten()[0].item()
            if sigma_max > 0:
                normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                new_pe_embedder.set_timestep(normalized_timestep)

        # Set Z-Image scale hint
        if is_z_image:
            new_pe_embedder.set_scale_hint(zimage_freq_scale_factor)

        # --- Compute spectral profiles from input latent ---
        input_x = args_dict.get("input")
        if input_x is not None and input_x.dim() >= 4:
            try:
                # Handle both 4D (B,C,H,W) and 5D (B,C,T,H,W) latents
                if input_x.dim() == 5:
                    # Video model (e.g. Anima): use first frame for spectral analysis
                    B, C_lat, T_lat, H_lat, W_lat = input_x.shape
                    spatial = input_x[:, :, 0].float().permute(0, 2, 3, 1)  # (B, H, W, C)
                else:
                    B, C_lat, H_lat, W_lat = input_x.shape
                    spatial = input_x.float().permute(0, 2, 3, 1)  # (B, H, W, C)

                # Convert to patch grid dimensions
                H_patches = max(H_lat // patch_size, 1)
                W_patches = max(W_lat // patch_size, 1)

                n_bins_h = max(H_patches // 2, 8)
                n_bins_w = max(W_patches // 2, 8)

                energy_h, energy_w = compute_axis_spectral_profiles(
                    spatial, H_patches, W_patches, n_bins_h, n_bins_w
                )
                iso_profile = compute_spectral_energy_profile(
                    spatial, H_patches, W_patches, max(H_patches, W_patches) // 2
                )
                dynamic_spread = compute_dynamic_spread(
                    iso_profile,
                    spread_min=spread_min,
                    spread_max=spread_max,
                    alpha=spread_alpha,
                )

                new_pe_embedder.set_spectral_data(
                    energy_h, energy_w, dynamic_spread,
                    target_res_h=height, target_res_w=width,
                )
            except Exception as e:  # degrade: spectral enhancement skipped
                logger.debug(f"SEGA spectral computation skipped: {e}")

        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(sega_wrapper_function)

    return m
