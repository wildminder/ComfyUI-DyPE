"""
PixelRush ComfyUI node — cascade-based high-resolution generation.

Provides a ComfyUI node that applies the PixelRush algorithm to upscale
and refine images using partial DDIM inversion + patch-based denoising.
Works with any ComfyUI model (SDXL, SD1.5, FLUX, etc.).
"""

from __future__ import annotations

import logging

import torch
from comfy_api.latest import io

from .pixelrush import PixelRushConfig, pixelrush_cascade

logger = logging.getLogger("ComfyUI-DyPE")


def _scale_k_timestep(model, k_timestep):
    """Scale k_timestep to the model's native timestep range.

    EPS/SD models use 0-999 timesteps; FLUX/CONST-flow models use 0-1.
    Passing k_timestep=249 (paper default for EPS) to a FLUX model gives
    sigma>1 (invalid), making 1-sigma negative in eps_to_x0 -> pure noise.

    Returns the scaled k_timestep (0-1 range for flow models, unchanged for EPS).
    """
    try:
        ms = model.model.model_sampling
        sigma_max = ms.sigma_max
        timestep_at_max = ms.timestep(sigma_max)
        if timestep_at_max <= 1.0 + 1e-3:
            # 0-1 timestep range (FLUX/CONST flow): scale 0-999 -> 0-1
            scaled = k_timestep / 999.0
            logger.info(
                "PixelRush: model uses 0-1 timestep range, scaling k_timestep %d -> %.4f",
                k_timestep, scaled,
            )
            return scaled
    except Exception as e:  # degrade: keep raw k_timestep (user action may matter)
        logger.warning("PixelRush: could not detect timestep range (%s), using raw k_timestep", e)
    return k_timestep


def _detect_prediction_type(model_sampling):
    """Detect the model's prediction type from its model_sampling MRO.

    ComfyUI uses different prediction types:
      - ``CONST`` (flow matching, e.g. FLUX): model predicts velocity v = eps - x0
      - ``EPS``: model predicts epsilon directly
      - ``V_PREDICTION``: model predicts v-prediction
      - ``X0``: model predicts x0 directly

    The PixelRush DDIM equations assume epsilon prediction. For non-EPS models
    we must convert the raw model output to epsilon using the correct formula.
    """
    mro_names = [c.__name__ for c in type(model_sampling).__mro__]
    if "CONST" in mro_names or "IMG_TO_IMG_FLOW" in mro_names:
        return "const"
    if "V_PREDICTION" in mro_names:
        return "v_prediction"
    if "X0" in mro_names:
        return "x0"
    if "EPS" in mro_names:
        return "eps"
    return "eps"


def _make_model_output_to_eps(model_sampling, prediction_type):
    """Create a function converting raw model output to epsilon.

    Uses numerically-stable formulas (no division by sigma at sigma≈0):

    - EPS: eps = model_output (raw output IS epsilon)
    - CONST (flow): eps = x_t + model_output * (1 - sigma)
      (derivation: v = eps - x0, x0 = x_t - v*sigma → eps = v + x0 = x_t + v*(1-sigma))
    - V_PREDICTION: eps = x_t*sigma/(sigma^2+sd^2) + model_output*sd/(sigma^2+sd^2)^0.5
    - X0: eps = (x_t - model_output) / sigma (clamped; unstable at sigma≈0)
    """
    sd = getattr(model_sampling, "sigma_data", 1.0)

    def model_output_to_eps(model_output, x_t, sigma):
        sigma_r = sigma.reshape(sigma.shape + (1,) * (x_t.ndim - sigma.ndim))
        if prediction_type == "eps":
            return model_output
        elif prediction_type == "const":
            return x_t + model_output * (1.0 - sigma_r)
        elif prediction_type == "v_prediction":
            denom = sigma_r ** 2 + sd ** 2
            return x_t * sigma_r / denom + model_output * sd / denom.sqrt()
        elif prediction_type == "x0":
            sigma_safe = sigma_r.clamp_min(1e-6)
            return (x_t - model_output) / sigma_safe
        else:
            return model_output

    return model_output_to_eps


def _make_eps_to_x0(model_sampling, prediction_type):
    """Create a function converting epsilon to x0 (inverse of noise_scaling).

    - EPS: x0 = x_t - sigma * eps
    - CONST (flow): x0 = (x_t - sigma * eps) / (1 - sigma)
    - V_PREDICTION: x0 = x_t - sigma * eps
    - X0: x0 = model_output (already x0)
    """
    def eps_to_x0(x_t, eps, sigma):
        sigma_r = sigma.reshape(sigma.shape + (1,) * (x_t.ndim - sigma.ndim))
        if prediction_type == "eps":
            return x_t - sigma_r * eps
        elif prediction_type == "const":
            one_minus_sigma = (1.0 - sigma_r).clamp_min(1e-6)
            return (x_t - sigma_r * eps) / one_minus_sigma
        elif prediction_type == "v_prediction":
            return x_t - sigma_r * eps
        elif prediction_type == "x0":
            return eps  # already x0
        else:
            return x_t - sigma_r * eps

    return eps_to_x0


def _make_predict_eps(model, positive, negative, cfg_scale, latent_dimensions=2,
                      operate_in_vae_space=True):
    """Create a predict_eps adapter that runs the model with CFG.

    Uses ComfyUI's full conditioning pipeline:
      1. ``convert_cond`` — convert tuple conditioning to dict format
      2. ``process_conds`` — build model_conds (y, c_crossattn, etc.)
      3. ``get_area_and_mult`` — extract processed conditioning tensors
      4. ``diffusion_model`` — run the model (raw output)

    The raw model output is converted to epsilon using the model's prediction
    type (EPS, CONST/flow, V_PREDICTION, X0). This is critical: FLUX uses
    CONST (flow matching) where the raw output is velocity, not epsilon.

    For 3D latent models (latent_dimensions=3), the core PixelRush algorithm
    works in 4D spatial [B, C, H, W], but the model expects 5D [B, C, T, H, W].
    This adapter unsqueezes 4D patches to 5D before calling diffusion_model, and
    squeezes the 5D eps output back to 4D.

    When ``operate_in_vae_space`` is True, the adapter accepts a VAE-space latent
    (std ≈ 1) and returns a VAE-space epsilon. It converts to model space via
    ``process_latent_in`` before running the model and converts the epsilon back
    via ``process_latent_out``. This keeps the core algorithm in a space where
    the fixed-magnitude noise injection (std ≈ 0.95) is comparable to the signal
    (std ≈ 1), which is required for models whose ``process_latent_in`` scales
    the latent (e.g. SDXL ``scale_factor=0.13025``).

    Returns a callable: predict_eps(latent, timestep) -> eps [B, C, H, W]
    """
    import comfy.model_management
    import comfy.sampler_helpers
    import comfy.samplers
    import comfy.utils

    device = model.load_device if hasattr(model, 'load_device') else torch.device("cpu")
    is_3d = latent_dimensions == 3

    # Capture latent-space converters. When operating in VAE space, the adapter
    # converts VAE latents -> model space before the model call and converts the
    # resulting epsilon back to VAE space. When None (no process_latent_in/out
    # on the model, or legacy mode), the conversions are no-ops.
    process_latent_in = getattr(model.model, 'process_latent_in', None)
    if not operate_in_vae_space:
        # Legacy mode: execute() already applied process_latent_in; eps stays in
        # model space. Disable the conversion here (process_latent_out is only
        # needed in VAE-space mode, so it is not captured at all).
        process_latent_in = None

    # Ensure the model is loaded to GPU and pre_run is called
    # pre_run sets model.model.current_patcher = model (the ModelPatcher)
    # which is required by apply_hooks and prepare_state
    comfy.model_management.load_models_gpu([model])
    model.pre_run()

    # Detect prediction type and create conversion functions
    model_sampling = model.model.model_sampling
    prediction_type = _detect_prediction_type(model_sampling)
    model_output_to_eps = _make_model_output_to_eps(model_sampling, prediction_type)
    logger.info("PixelRush: detected model prediction type '%s'", prediction_type)

    # Cache for processed conditioning (built once, reused across calls)
    _processed = None

    def _get_processed(latent_5d):
        """Build processed conditioning using ComfyUI's canonical pipeline.

        convert_cond converts tuple format [(tensor, dict), ...] to dict format
        [dict, ...] which process_conds expects.

        ``latent_5d`` must be 5D for 3D latent models so that conditioning
        area/mask dimensions match what get_area_and_mult expects.
        """
        nonlocal _processed
        if _processed is not None:
            return _processed
        # Step 1: Convert tuple conditioning to dict format
        pos_converted = comfy.sampler_helpers.convert_cond(positive)
        neg_converted = comfy.sampler_helpers.convert_cond(negative)
        conds_dict = {"positive": pos_converted, "negative": neg_converted}
        # Step 2: Process conds (builds model_conds via encode_model_conds)
        noise = torch.zeros_like(latent_5d)
        _processed = comfy.samplers.process_conds(
            model.model, noise, conds_dict, device
        )
        return _processed

    def predict_eps(latent: torch.Tensor, timestep: int) -> torch.Tensor:
        # Move latent to model device for inference
        latent = latent.to(device)

        # For 3D latent models, unsqueeze 4D [B,C,H,W] to 5D [B,C,1,H,W]
        # The model's _forward expects 5D for temporal models.
        was_4d = latent.ndim == 4
        if is_3d and was_4d:
            latent = latent.unsqueeze(2)  # [B, C, 1, H, W]

        # When operating in VAE space, convert the input latent to model space
        # before running the model. process_latent_in applies the latent
        # format's scale_factor (e.g. SDXL 0.13025) and mean/std shifts.
        if process_latent_in is not None:
            latent = process_latent_in(latent)

        # Convert timestep (0-999) to sigma using model_sampling.sigma().
        # The timestep is a value in the model's internal timestep space,
        # NOT an index into the sigmas array (which has only ~20 entries).
        ts_tensor = torch.tensor([float(timestep)], device=device)
        sigma_val = model.model.model_sampling.sigma(ts_tensor).item()
        # Clamp to small minimum to avoid division-by-zero in epsilon extraction
        # (timestep=0 gives sigma=0, which would make eps = (x - x0) / 0 = NaN)
        sigma_val = max(sigma_val, 1e-6)
        B = latent.shape[0]
        sigma = torch.full((B,), sigma_val, device=latent.device, dtype=latent.dtype)

        # Get processed conditioning (cached after first call)
        processed = _get_processed(latent)

        def run_cond(prompt_type):
            cond_list = processed.get(prompt_type, [])
            if len(cond_list) == 0:
                return torch.zeros_like(latent)
            cond = cond_list[0]
            # Use get_area_and_mult to properly process COND objects
            # This calls model_conds[c].process_cond(batch_size, area) internally
            p = comfy.samplers.get_area_and_mult(cond, latent, sigma)
            if p is None:
                return torch.zeros_like(latent)
            # p.conditioning is a dict of COND objects (e.g. CONDCrossAttn)
            # apply_model expects raw tensors, not COND objects.
            # cond_cat extracts .cond from each COND object and concatenates.
            # With a single cond, concat([]) returns self.cond (the tensor).
            c = comfy.samplers.cond_cat([p.conditioning])
            # apply_model requires transformer_options
            # model is the ModelPatcher; apply_hooks returns the transformer_options dict
            c['transformer_options'] = model.apply_hooks(hooks=None)

            # Replicate _apply_model's logic but skip calculate_denoised to get
            # the raw model_output (velocity for CONST, epsilon for EPS, etc.)
            m = model.model
            ms = m.model_sampling
            xc = ms.calculate_input(sigma, p.input_x)
            if c.get('c_concat') is not None:
                xc = torch.cat([xc] + [comfy.model_management.cast_to_device(
                    c['c_concat'], xc.device, xc.dtype)], dim=1)
            dtype = m.get_dtype_inference()
            xc = xc.to(dtype)
            t = ms.timestep(sigma).float()
            device = xc.device
            context = c.get('c_crossattn')
            if context is not None:
                context = comfy.model_management.cast_to_device(context, device, dtype)
            extra_conds = {}
            for o in c:
                if o in ('c_crossattn', 'c_concat', 'transformer_options'):
                    continue
                extra = c[o]
                if hasattr(extra, 'dtype'):
                    extra = comfy.model_management.cast_to_device(extra, device, dtype)
                elif isinstance(extra, list):
                    ex = []
                    for ext in extra:
                        ex.append(comfy.model_management.cast_to_device(ext, device, dtype))
                    extra = ex
                extra_conds[o] = extra
            t = m.process_timestep(t, x=p.input_x, **extra_conds)
            to = c['transformer_options'].copy()
            to["prefetch_dynamic_vbars"] = (
                m.current_patcher is not None and m.current_patcher.is_dynamic()
            )
            model_output = m.diffusion_model(
                xc, t, context=context, control=c.get('control'),
                transformer_options=to, **extra_conds,
            )
            if len(model_output) > 1 and not torch.is_tensor(model_output):
                model_output, _ = comfy.utils.pack_latents(model_output)
            # Convert raw model output to epsilon using the prediction type.
            # For CONST/flow (FLUX), model_output is velocity v = eps - x0,
            # so eps = x_t + v*(1-sigma). This is stable at sigma≈0.
            eps = model_output_to_eps(model_output.float(), p.input_x, sigma)
            return eps

        eps_cond = run_cond("positive")
        eps_uncond = run_cond("negative")

        # CFG
        eps = eps_uncond + cfg_scale * (eps_cond - eps_uncond)

        # For 3D latent models, squeeze 5D eps back to 4D for the core algorithm
        if is_3d and eps.ndim == 5 and was_4d:
            eps = eps.squeeze(2)  # [B, C, H, W]

        # When operating in VAE space, the input latent was converted to model
        # space via process_latent_in before the model call, but the returned
        # epsilon is intentionally NOT converted back. The model predicts noise
        # with std ≈ 1 in model space; the VAE-space latent also has std ≈ 1.
        # Keeping the epsilon at std ≈ 1 (numerically comparable to the latent)
        # is what makes the fixed-magnitude noise injection (std ≈ 0.95) balanced
        # against the signal, instead of being scaled down by scale_factor and
        # drowned out. forward_step/reverse_step then operate directly in VAE
        # space (no latent conversion), so x (std ≈ 1) and eps (std ≈ 1) stay
        # comparable throughout.
        return eps

    return predict_eps


def _make_forward_step(model, operate_in_vae_space=True):
    """Create a forward_step adapter using the model's noise_scaling.

    forward_step(x_0, eps, sigma) -> x_K  (noises x_0 to timestep K)
    Uses the model's own noise schedule, which is correct for all
    prediction types (EPS, CONST/flow, V_PREDICTION, etc.).

    When ``operate_in_vae_space`` is True (default), ``x_0`` and ``eps`` both
    arrive in VAE space with std ≈ 1 (the epsilon is NOT scaled by
    process_latent_out — see _make_predict_eps). The DDIM forward is applied
    directly in VAE space, so the fixed-magnitude noise injection stays
    balanced against the signal. No latent<->model conversion happens here.
    """
    ms = model.model.model_sampling

    def forward_step(x_0, eps, sigma):
        return ms.noise_scaling(sigma, eps, x_0)

    return forward_step


def _make_reverse_step(model, operate_in_vae_space=True):
    """Create a reverse_step adapter using eps_to_x0.

    reverse_step(x_K, eps_injected, sigma) -> x_0_hat
    Converts injected epsilon back to x0 using the inverse of noise_scaling,
    which is correct for all prediction types.

    When ``operate_in_vae_space`` is True (default), ``x_K`` and ``eps_injected``
    both arrive in VAE space with std ≈ 1, so the DDIM reverse is applied
    directly in VAE space. No latent<->model conversion happens here.
    """
    ms = model.model.model_sampling
    prediction_type = _detect_prediction_type(ms)
    eps_to_x0 = _make_eps_to_x0(ms, prediction_type)

    def reverse_step(x_K, eps_injected, sigma):
        return eps_to_x0(x_K, eps_injected, sigma)

    return reverse_step


def _make_sigma_at(model):
    """Create a sigma_at adapter: timestep (0-999) -> sigma float."""
    ms = model.model.model_sampling

    def sigma_at(timestep):
        ts_tensor = torch.tensor([float(timestep)], device=model.load_device
                                 if hasattr(model, 'load_device') else torch.device("cpu"))
        sigma_val = ms.sigma(ts_tensor).item()
        return max(sigma_val, 1e-6)

    return sigma_at


def _make_alpha_bar_at(model):
    """Create an alpha_bar_at adapter from the model's sigma schedule.

    Converts timestep (0-999) to sigma via model_sampling.sigma(), then
    computes alpha_bar = 1 / (sigma^2 + 1).

    Returns a callable: alpha_bar_at(timestep) -> float
    """
    model_sampling = model.model.model_sampling

    def alpha_bar_at(timestep: int) -> float:
        # Convert timestep to sigma using the model's internal conversion
        ts_tensor = torch.tensor([float(timestep)], device=model.load_device
                                 if hasattr(model, 'load_device') else torch.device("cpu"))
        sigma = model_sampling.sigma(ts_tensor)
        alpha_bar = 1.0 / (sigma ** 2 + 1.0)
        return alpha_bar.item()

    return alpha_bar_at


def _make_vae_adapters(vae, device, model=None, operate_in_vae_space=True):
    """Create VAE decode/encode adapters.

    Returns (vae_decode, vae_encode) callables.
    All tensors are moved to ``device`` for GPU acceleration.
    Handles both 2D VAEs (latent_dim=2, 4D latents [B,C,H,W]) and
    3D/video VAEs (latent_dim=3, 5D latents [B,C,T,H,W]).

    Uses model.process_latent_out/in to convert between model latent
    format and VAE latent format — UNLESS ``operate_in_vae_space`` is True,
    in which case the latent already lives in VAE space and the adapters must
    NOT apply process_latent_out/in (that would re-scale an already-VAE-space
    latent and corrupt it). The adapters then only handle shape (3D unsqueeze)
    and the raw vae.decode/encode calls.

    For 3D latent models (Wan21, Krea2, Qwen, Anima), the model's
    ``process_latent_out``/``process_latent_in`` use 5D ``latents_mean``/
    ``latents_std`` with shape ``[1, C, 1, 1, 1]``.  Calling these on a 4D
    tensor causes a broadcasting misalignment that corrupts the batch
    (see plan 2026-08-10-freescale-krea2-5d-latent-fix.md).

    Therefore, for 3D latent models:
    - ``vae_decode`` accepts 5D latents and calls ``process_latent_out``
      directly on the 5D tensor.
    - ``vae_encode`` returns 5D latents (with singleton temporal dim) so
      they can be passed directly to the sampler.
    """
    latent_dim = getattr(vae, 'latent_dim', 2)
    process_latent_out = None
    process_latent_in = None
    if model is not None and hasattr(model, 'model'):
        if hasattr(model.model, 'process_latent_out'):
            process_latent_out = model.model.process_latent_out
        if hasattr(model.model, 'process_latent_in'):
            process_latent_in = model.model.process_latent_in
    # When operating in VAE space, the latent is already in VAE format; the
    # adapters must not re-apply the model<->VAE scaling.
    if operate_in_vae_space:
        process_latent_out = None
        process_latent_in = None

    def vae_decode(latent: torch.Tensor) -> torch.Tensor:
        if isinstance(latent, dict):
            latent = latent["samples"]
        latent = latent.to(device)
        # Convert from model latent format to VAE latent format.
        # For 3D latent models, process_latent_out expects 5D input.
        if process_latent_out is not None:
            if latent_dim == 3:
                # Ensure 5D for process_latent_out
                if latent.ndim == 4:
                    latent = latent.unsqueeze(2)  # [B, C, 1, H, W]
                latent = process_latent_out(latent)
            else:
                latent = process_latent_out(latent)
        # For 3D VAEs, ensure temporal dimension is present for vae.decode
        if latent_dim == 3 and latent.ndim == 4:
            latent = latent.unsqueeze(2)
        decoded = vae.decode(latent)
        if decoded.ndim == 5:
            decoded = decoded[:, 0]
        elif decoded.ndim == 3:
            decoded = decoded.unsqueeze(0)
        if decoded.dim() == 4 and decoded.shape[-1] == 3:
            decoded = decoded.movedim(-1, 1)
        elif decoded.dim() == 4 and decoded.shape[1] == 3:
            pass
        return decoded

    def vae_encode(image: torch.Tensor) -> torch.Tensor:
        image = image.to(device)
        if image.dim() == 4 and image.shape[1] == 3:
            image = image.movedim(1, -1)
        encoded = vae.encode(image)
        if isinstance(encoded, dict):
            encoded = encoded["samples"]
        # For 3D VAEs, take first temporal frame to get 4D
        if latent_dim == 3 and encoded.ndim == 5:
            encoded = encoded[:, :, 0]
        # Convert from VAE latent format to model latent format (legacy path).
        # In VAE-space mode (operate_in_vae_space=True) process_latent_in is
        # None, so no scaling is applied and the latent stays in VAE space.
        if process_latent_in is not None:
            if latent_dim == 3:
                if encoded.ndim == 4:
                    encoded = encoded.unsqueeze(2)  # [B, C, 1, H, W]
                encoded = process_latent_in(encoded)
            else:
                encoded = process_latent_in(encoded)
        # For 3D latent models, always return 5D [B, C, 1, H, W] (singleton
        # temporal dim) so the latent can be passed directly to the sampler /
        # core algorithm (which squeezes to 4D). This is independent of whether
        # process_latent_in scaling was applied.
        if latent_dim == 3 and encoded.ndim == 4:
            encoded = encoded.unsqueeze(2)  # [B, C, 1, H, W]
        return encoded

    return vae_decode, vae_encode


def _prepare_initial_latent(initial_latent, process_latent_in, latent_dimensions,
                             operate_in_vae_space):
    """Convert the initial latent to model format ONLY when not in VAE space.

    When ``operate_in_vae_space`` is True (default), the latent stays in VAE
    space (std ≈ 1) and the adapters convert to model space internally. This is
    required for models whose ``process_latent_in`` scales the latent down (e.g.
    SDXL ``scale_factor=0.13025``), otherwise the fixed-magnitude noise
    injection would dominate the signal.

    When ``operate_in_vae_space`` is False (legacy path), ``process_latent_in``
    is applied here. For 3D latent models, ``process_latent_in`` expects 5D input
    ``[B, C, T, H, W]`` — a 4D latent is unsqueezed first to avoid broadcasting
    misalignment with 5D ``latents_mean``/``latents_std``.

    Parameters
    ----------
    initial_latent : Tensor
        ``[B, C, H, W]`` (or ``[B, C, T, H, W]`` for 3D) latent at native res.
    process_latent_in : callable or None
        Model's ``process_latent_in`` (scales latent to model space), or None.
    latent_dimensions : int
        2 for 2D VAEs, 3 for 3D/video VAEs.
    operate_in_vae_space : bool
        If True, skip ``process_latent_in`` (latent already in VAE space).

    Returns
    -------
    Tensor
        The (possibly converted) initial latent.
    """
    if process_latent_in is not None and not operate_in_vae_space:
        if latent_dimensions == 3:
            if initial_latent.ndim == 4:
                initial_latent = initial_latent.unsqueeze(2)  # [B, C, 1, H, W]
        initial_latent = process_latent_in(initial_latent)
    return initial_latent


class PixelRushNode(io.ComfyNode):
    """
    PixelRush — cascade-based high-resolution generation.

    Generates high-resolution images by repeatedly upscaling and refining
    with partial DDIM inversion + patch-based denoising. Works with any
    ComfyUI model (SDXL, SD1.5, FLUX, etc.).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PixelRush",
            display_name="PixelRush",
            category="image/upscaling",
            description="Cascade-based high-resolution generation via partial DDIM inversion + patch denoising. Works with SDXL, SD1.5, and other models.",
            inputs=[
                io.Model.Input("model", tooltip="The diffusion model."),
                io.Vae.Input("vae", tooltip="VAE for decode/encode."),
                io.Conditioning.Input("positive", tooltip="Positive conditioning."),
                io.Conditioning.Input("negative", tooltip="Negative conditioning."),
                io.Latent.Input("latent_image", tooltip="Base latent at native resolution."),
                io.Float.Input(
                    "cfg", default=7.0, min=0.0, max=20.0, step=0.1,
                    tooltip="Classifier-free guidance scale.",
                ),
                io.Int.Input(
                    "num_cascade_stages", default=1, min=1, max=5, step=1,
                    tooltip="Number of 2x upscale stages. 1=2x, 2=4x, 3=8x.",
                ),
                io.Int.Input(
                    "k_timestep", default=249, min=1, max=999, step=1,
                    tooltip="Partial inversion timestep K. Must align with model schedule.",
                ),
                io.Float.Input(
                    "noise_lambda", default=0.95, min=0.0, max=1.0, step=0.01,
                    tooltip="Noise injection strength (slerp between predicted and random noise).",
                ),
                io.Float.Input(
                    "overlap", default=0.50, min=0.0, max=0.75, step=0.05,
                    tooltip="Patch overlap fraction. 0.5=50% overlap.",
                ),
                io.Float.Input(
                    "gaussian_sigma", default=8.0, min=1.0, max=20.0, step=0.5,
                    tooltip="Gaussian feathering sigma for patch blending.",
                ),
                io.Int.Input(
                    "gaussian_kernel_size", default=41, min=3, max=101, step=2,
                    tooltip="Gaussian blur kernel size (must be odd).",
                ),
                io.Int.Input(
                    "patch_h", default=0, min=0, max=512, step=8,
                    tooltip="Latent patch height. 0=auto (native resolution).",
                ),
                io.Int.Input(
                    "patch_w", default=0, min=0, max=512, step=8,
                    tooltip="Latent patch width. 0=auto (native resolution).",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="Refined Latent"),
            ],
        )

    @classmethod
    def execute(cls, model, vae, positive, negative, latent_image, cfg=7.0,
                num_cascade_stages=1, k_timestep=249, noise_lambda=0.95,
                overlap=0.50, gaussian_sigma=8.0, gaussian_kernel_size=41,
                patch_h=0, patch_w=0) -> io.NodeOutput:
        import comfy.utils

        # Get initial latent
        if isinstance(latent_image, dict):
            initial_latent = latent_image["samples"]
        else:
            initial_latent = latent_image

        device = model.load_device if hasattr(model, 'load_device') else torch.device("cpu")

        # Get model's latent format info
        model_latent_channels = getattr(model.model.latent_format, 'latent_channels', None)
        latent_dimensions = getattr(model.model.latent_format, 'latent_dimensions', 2)
        process_latent_in = getattr(model.model, 'process_latent_in', None)

        # Convert input latent to model's internal format if channels don't match.
        # EmptyLatentImage may produce fewer channels than the model expects
        # (e.g., 4 channels for a 16-channel Krea2/Wan21 model).
        # Use repeat_to_batch_size (like ComfyUI's fix_empty_latent_channels)
        # instead of zero-padding, which produces garbage.
        if model_latent_channels is not None and initial_latent.shape[1] != model_latent_channels:
            is_empty = torch.count_nonzero(initial_latent) == 0
            if is_empty:
                logger.info(
                    "PixelRush: empty input latent has %d channels, model expects %d — repeating channels",
                    initial_latent.shape[1], model_latent_channels,
                )
                initial_latent = comfy.utils.repeat_to_batch_size(
                    initial_latent, model_latent_channels, dim=1,
                )
            else:
                logger.warning(
                    "PixelRush: non-empty input latent has %d channels, model expects %d — "
                    "channel mismatch may produce unexpected results",
                    initial_latent.shape[1], model_latent_channels,
                )

        # Move initial latent to model device for GPU acceleration
        initial_latent = initial_latent.to(device)

        # PixelRush runs in VAE latent space by default (std ≈ 1). The adapters
        # convert to model space internally. Set to False only to use the legacy
        # model-space path.
        operate_in_vae_space = True

        # Convert initial latent to model format (process_latent_in) ONLY when
        # NOT operating in VAE space. When operate_in_vae_space is True, the
        # latent stays in VAE space (std ≈ 1) and the adapters convert to model
        # space internally. This is required for models whose process_latent_in
        # scales the latent down (e.g. SDXL scale_factor=0.13025), otherwise the
        # fixed-magnitude noise injection would dominate the signal.
        # In normal ComfyUI sampling, the guider calls process_latent_in before
        # apply_model. Since PixelRush calls apply_model directly via predict_eps,
        # we must convert here (legacy path). For 3D latent models,
        # process_latent_in expects 5D input [B, C, T, H, W] — passing 4D causes
        # broadcasting misalignment with 5D latents_mean/std
        # (see plan 2026-08-10-freescale-krea2-5d-latent-fix.md).
        initial_latent = _prepare_initial_latent(
            initial_latent, process_latent_in, latent_dimensions, operate_in_vae_space
        )

        # Auto-detect patch size from native resolution
        # Handle both 4D [B,C,H,W] and 5D [B,C,T,H,W]
        if initial_latent.ndim == 5:
            _, _, _, h, w = initial_latent.shape
        else:
            _, _, h, w = initial_latent.shape
        if patch_h == 0 or patch_w == 0:
            patch_h = h if patch_h == 0 else patch_h
            patch_w = w if patch_w == 0 else patch_w

        # Scale k_timestep to the model's native timestep range.
        # EPS/SD models use 0-999; FLUX/CONST-flow models use 0-1.
        # Passing k_timestep=249 (paper default for EPS) to a FLUX model
        # gives sigma>1 (invalid), making 1-sigma negative -> pure noise.
        k_timestep_scaled = _scale_k_timestep(model, k_timestep)

        cfg_obj = PixelRushConfig(
            patch_h=patch_h,
            patch_w=patch_w,
            overlap=overlap,
            k_timestep=k_timestep_scaled,
            noise_lambda=noise_lambda,
            gaussian_sigma=gaussian_sigma,
            gaussian_kernel_size=gaussian_kernel_size,
            operate_in_vae_space=operate_in_vae_space,
        )

        # Create adapters — predict_eps needs to know if model is 3D latent
        predict_eps = _make_predict_eps(
            model, positive, negative, cfg, latent_dimensions,
            operate_in_vae_space=cfg_obj.operate_in_vae_space,
        )
        alpha_bar_at = _make_alpha_bar_at(model)
        vae_decode, vae_encode = _make_vae_adapters(
            vae, device, model, operate_in_vae_space=cfg_obj.operate_in_vae_space,
        )
        # Model-agnostic forward/reverse steps (handle CONST/flow, V_PRED, EPS)
        forward_step = _make_forward_step(model, operate_in_vae_space=cfg_obj.operate_in_vae_space)
        reverse_step = _make_reverse_step(model, operate_in_vae_space=cfg_obj.operate_in_vae_space)
        sigma_at = _make_sigma_at(model)

        # For 3D latent models, squeeze temporal dim for the core algorithm
        # (which works in 4D spatial). predict_eps will unsqueeze back to 5D
        # before calling apply_model.
        if latent_dimensions == 3 and initial_latent.ndim == 5:
            initial_latent_4d = initial_latent.squeeze(2)  # [B, C, H, W]
        else:
            initial_latent_4d = initial_latent

        # Pre-compute total patches across all cascade stages for the progress bar.
        # Each stage doubles the latent spatial dimensions.
        from .pixelrush import patch_positions

        total_patches = 0
        stage_patch_counts = []
        cur_h, cur_w = patch_h, patch_w
        for stage in range(num_cascade_stages):
            # After VAE decode → 2x bicubic → VAE encode, latent is 2x in each dim
            cur_h = cur_h * 2
            cur_w = cur_w * 2
            n = len(list(patch_positions(
                full_h=cur_h, full_w=cur_w,
                patch_h=patch_h, patch_w=patch_w,
                overlap=overlap,
            )))
            stage_patch_counts.append(n)
            total_patches += n

        # Progress bar
        pbar = comfy.utils.ProgressBar(total_patches)

        def progress_callback(patch_idx, total_patches_in_stage, stage, num_stages):
            # Accumulate patches from previous stages + current patch
            prev_patches = sum(stage_patch_counts[:stage]) if stage > 0 else 0
            pbar.update_absolute(prev_patches + patch_idx)

        # Run PixelRush cascade (works in 4D spatial)
        result_latent_4d = pixelrush_cascade(
            initial_latent=initial_latent_4d,
            num_cascade_stages=num_cascade_stages,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            predict_eps=predict_eps,
            alpha_bar_at=alpha_bar_at,
            cfg=cfg_obj,
            progress_callback=progress_callback,
            forward_step=forward_step,
            reverse_step=reverse_step,
            sigma_at=sigma_at,
        )

        # Mark progress bar as complete
        pbar.update_absolute(total_patches)

        # For 3D latent models, unsqueeze back to 5D for the output.
        # The downstream VAEDecode node will call process_latent_out on this.
        if latent_dimensions == 3 and result_latent_4d.ndim == 4:
            result_latent = result_latent_4d.unsqueeze(2)  # [B, C, 1, H, W]
        else:
            result_latent = result_latent_4d

        return io.NodeOutput({"samples": result_latent})
