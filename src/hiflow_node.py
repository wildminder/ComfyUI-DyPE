"""
HiFlow ComfyUI node — trajectory-guided flow upscaling.

Runs the HiFlow cascade (plan 2026-09-03): a base-resolution rectified-flow
trajectory is recorded, then each upscale stage reuses it as a time-matched
virtual reference (initialization + direction + acceleration alignment).
Flow models only (FLUX, Qwen-Image, AuraFlow, ...); works in VAE latent
space with the x0 adapter owning the VAE<->model conversions.

Reference: HiFlow paper (arXiv:2504.06232, NeurIPS 2025).
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from comfy_api.latest import io

from .freescale import gaussian_blur_2d
from .hiflow import HiFlowConfig, hiflow_cascade
from .pixelrush_node import _detect_prediction_type

logger = logging.getLogger("ComfyUI-DyPE")


# Timestep/sigma conversions go through model_sampling.timestep/.sigma only —
# never hand-multiplied. FLUX (ModelSamplingFlux) uses timestep(sigma)==sigma;
# DiscreteFlow models use a x1000 multiplier; the adapter is agnostic either way.

_FLOW_PREDICTIONS = ("const", "img_to_img_flow", "cosmos_rflow")


def _require_flow_model(model) -> str:
    """Gate HiFlow to rectified-flow models (plan D1, D12).

    Raises ValueError with an actionable message for non-flow prediction
    types and for 3D-latent (video) models.
    Returns the detected flow prediction family name.
    """
    model_sampling = model.model.model_sampling
    mro_names = [c.__name__ for c in type(model_sampling).__mro__]

    detected = _detect_prediction_type(model_sampling)
    if detected == "const" and "IMG_TO_IMG_FLOW" in mro_names:
        detected = "img_to_img_flow"
    elif detected == "const" and "COSMOS_RFLOW" in mro_names:
        detected = "cosmos_rflow"

    if detected not in _FLOW_PREDICTIONS:
        raise ValueError(
            f"HiFlow needs a rectified-flow model (FLUX, Qwen-Image, "
            f"AuraFlow, Z-Image...); this model predicts "
            f"'{detected.upper()}'. For SD/SDXL-style models use the "
            f"PixelRush node instead."
        )

    latent_dimensions = getattr(
        model.model.latent_format, "latent_dimensions", 2)
    if latent_dimensions == 3:
        raise ValueError(
            "HiFlow does not support 3D-latent (video) models yet — the "
            "frequency alignment is 2D per-frame. For video models use "
            "PixelRush."
        )

    return detected


def _make_predict_x0(
    model,
    positive,
    negative,
    cfg_scale: float,
) -> Callable[[torch.Tensor, float], torch.Tensor]:
    """Create the x0 adapter: (x_vae, sigma) -> x0 in VAE space (plan D3).

    Conditioning is prepared once per latent SHAPE via ComfyUI's canonical
    pipeline (convert_cond -> process_conds — the HAP-calibration precedent).
    Each call runs comfy.samplers.sampling_function, which returns the
    DENOISED x0 (apply_model applies calculate_denoised) with full CFG,
    areas, control nets and hooks. The VAE<->model conversions bracket the
    model call and cancel per call (plan D2).
    """
    import comfy.model_management
    import comfy.sampler_helpers
    import comfy.samplers

    device = model.load_device if hasattr(model, "load_device") \
        else torch.device("cpu")
    inner_model = model.model
    process_latent_in = getattr(inner_model, "process_latent_in", None)
    process_latent_out = getattr(inner_model, "process_latent_out", None)

    comfy.model_management.load_models_gpu([model])
    model.pre_run()

    _conds_by_shape: dict[tuple, dict] = {}

    def _get_conds(shape: tuple) -> dict:
        key = tuple(shape)
        if key not in _conds_by_shape:
            conds = {
                "positive": comfy.sampler_helpers.convert_cond(positive),
                "negative": comfy.sampler_helpers.convert_cond(negative),
            }
            noise = torch.zeros(shape, device=device)
            _conds_by_shape[key] = comfy.samplers.process_conds(
                inner_model, noise, conds, device,
            )
        return _conds_by_shape[key]

    # CFG guard (Z-Image bugfix 2026-09-03): with a NEGATIVE that carries no
    # tokens (an empty CLIPTextEncode — NOT ConditioningZeroOut), CFG is
    # undefined: the "uncond" branch is a real encoding of the empty string,
    # and cond_scale amplifies a meaningless difference. Guidance-free
    # models (Z-Image, Chroma) always land here when the user leaves the
    # negative empty. Mirror ComfyUI's cfg=1 skip: run the conditional
    # branch only (sampling_function with cond_scale=1.0 sets uncond=None
    # and returns the conditional x0 exactly).
    _has_negative = False
    for _entry in negative or []:
        if isinstance(_entry, (tuple, list)) and len(_entry) == 2:
            _tensor, _opts = _entry
            _tokens = 0
            if torch.is_tensor(_tensor):
                _tokens = int(_tensor.numel())
            elif isinstance(_tensor, (list, tuple)):
                _tokens = sum(
                    int(t.numel()) if torch.is_tensor(t) else len(t)
                    for t in _tensor
                )
            if _tokens > 0:
                _has_negative = True
                break
    if not _has_negative:
        cfg_scale = 1.0
        logger.info(
            "HiFlow: negative conditioning carries no tokens — running "
            "the conditional branch only (CFG skipped, scale forced to 1.0)"
        )

    def predict_x0(x_vae: torch.Tensor, sigma: float) -> torch.Tensor:
        x = x_vae.to(device)
        if process_latent_in is not None:
            x = process_latent_in(x)

        conds = _get_conds(tuple(x_vae.shape))
        sigma_t = torch.tensor([float(sigma)], device=device)
        timestep = inner_model.model_sampling.timestep(sigma_t)

        x0 = comfy.samplers.sampling_function(
            inner_model, x, timestep,
            uncond=conds["negative"], cond=conds["positive"],
            cond_scale=cfg_scale,
            model_options=model.model_options,
        )
        if process_latent_out is not None:
            x0 = process_latent_out(x0)
        return x0.to(x_vae.dtype).to(x_vae.device)

    return predict_x0


# ---------------------------------------------------------------------------
# VAE adapters (2D only — the gate rejects 3D-latent models)
# ---------------------------------------------------------------------------

def _make_vae_adapters(vae, device):
    """Create (vae_decode, vae_encode) callables in VAE latent space.

    HiFlow runs entirely in VAE latent space (plan D2), so these adapters do
    NOT apply process_latent_out/in — that would double-convert an already
    VAE-space tensor (the PixelRush space contract). They only normalize the
    tensor layout around the raw vae.decode/encode calls.

    LAYOUT (Z-Image bugfix 2026-09-03): ComfyUI's VAE boundary is channels-
    LAST — ``VAE.decode`` returns [B, H, W, 3] and ``VAE.encode`` expects
    [B, H, W, 3] (it applies ``movedim(-1, 1)`` internally, sd.py:1359).
    The adapters speak that layout end-to-end: decode passes the decoded
    image through, bicubic/sharpen run on channels-last tensors, and encode
    hands channels-last straight back. Converting to channels-first here
    (the old PixelRush-style convention) made VAE.encode move the WIDTH
    axis into the channel slot — spatial dims corrupted, encoder conv
    crashed ("Kernel size can't be greater than actual input size").
    """

    def vae_decode(latent: torch.Tensor) -> torch.Tensor:
        """latent [B,C,h,w] -> image [B, H, W, 3] (ComfyUI decode layout)."""
        if isinstance(latent, dict):
            latent = latent["samples"]
        latent = latent.to(device)
        decoded = vae.decode(latent)
        if isinstance(decoded, dict):
            decoded = decoded["samples"]
        if decoded.ndim == 5:
            decoded = decoded[:, 0]
        elif decoded.ndim == 3:
            decoded = decoded.unsqueeze(0)
        return decoded  # [B, H, W, 3] channels-last, untouched

    def vae_encode(image: torch.Tensor) -> torch.Tensor:
        """image [B, H, W, 3] -> latent [B, C, h, w] (ComfyUI encode layout)."""
        image = image.to(device)
        encoded = vae.encode(image)
        if isinstance(encoded, dict):
            encoded = encoded["samples"]
        return encoded

    return vae_decode, vae_encode


def _sharpen(image: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Gaussian unsharp mask: (alpha + 1) * I - alpha * blur(I).

    The reference (utils.gaussian_blur_image_sharpening) sharpens the
    pixel-space upscaled image before re-encoding (pixel mode only). The
    image arrives channels-LAST ([B, H, W, 3] — the ComfyUI VAE boundary);
    gaussian_blur_2d needs channels-first, so convert around the blur.
    """
    channels_last = image.dim() == 4 and image.shape[-1] == 3
    if channels_last:
        image = image.movedim(-1, 1)
    blurred = gaussian_blur_2d(image, kernel_size=3, sigma=2.0)
    sharpened = (alpha + 1.0) * image - alpha * blurred
    if channels_last:
        sharpened = sharpened.movedim(1, -1)
    return sharpened


def _base_sigmas(model, steps: int) -> torch.Tensor:
    """The model's own descending sigma schedule for the base stage.

    Uses comfy.samplers.calculate_sigmas with the "simple" scheduler (index
    sampling of the model's sigmas — no spacing resampling), ending at 0.
    """
    import comfy.samplers

    ms = model.model.model_sampling
    sigmas = comfy.samplers.calculate_sigmas(ms, "simple", steps)
    return sigmas.float().cpu()


def _downscale_ratio(vae) -> int:
    ratio = getattr(vae, "downscale_ratio", 8)
    if isinstance(ratio, (tuple, list)):
        ratio = ratio[1]  # (callable, h_ratio, w_ratio) convention
    return int(ratio)


# ---------------------------------------------------------------------------
# HiFlow node (V3 schema)
# ---------------------------------------------------------------------------

class HiFlowNode(io.ComfyNode):
    """HiFlow — trajectory-guided high-resolution upscaling for flow models.

    Records the base-resolution rectified-flow trajectory, then guides each
    upscale stage with it as a time-matched virtual reference: initialization
    alignment (start at noise level tau from the reference prediction),
    direction alignment (low-frequency nudge toward the reference) and
    acceleration alignment (match the reference's velocity rhythm).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HiFlow",
            display_name="HiFlow",
            category="image/upscaling",
            description=(
                "Training-free high-resolution upscaling for rectified-flow "
                "models (FLUX, Qwen-Image, ...) via flow-aligned guidance. "
                "Works from a base latent; chain DyPE (ntk) before the "
                "loader for RoPE extrapolation at the target resolution."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The flow model."),
                io.Vae.Input("vae", tooltip="VAE (used by pixel upsampling)."),
                io.Conditioning.Input(
                    "positive", tooltip="Positive conditioning."),
                io.Conditioning.Input(
                    "negative", tooltip="Negative conditioning."),
                io.Latent.Input(
                    "latent_image",
                    tooltip="Base latent at the model's native resolution "
                            "(e.g. from EmptySD3LatentImage). The cascade "
                            "noises it to the first sigma itself — an empty "
                            "latent + noise_seed reproduces the reference "
                            "pipeline's from-noise start."),
                io.Int.Input(
                    "noise_seed", default=0, min=0, max=2**32 - 1, step=1,
                    tooltip="Seed for the base-stage noise and each stage's "
                            "initialization noise (one shared generator)."),
                io.Float.Input(
                    "denoise", default=1.0, min=0.05, max=1.0, step=0.05,
                    tooltip="Img2img strength for a CONTENT latent (KSampler "
                            "convention): 1.0 regenerates from pure noise; "
                            "lower keeps more of the input image (0.6 "
                            "enters at ~37% content). Ignored for an empty "
                            "latent — that always runs the full schedule."),
                io.Float.Input(
                    "cfg", default=3.5, min=0.0, max=20.0, step=0.1,
                    tooltip="Classifier-free guidance for the BASE stage "
                            "(FLUX-dev default 3.5). Guidance-free models "
                            "(Z-Image, Chroma) or empty negatives: leave at "
                            "1.0 — CFG is auto-skipped when the negative "
                            "carries no tokens."),
                io.Int.Input(
                    "steps", default=30, min=1, max=200, step=1,
                    tooltip="Base-stage sampling steps (paper: 30). The "
                            "per-step clean predictions form the reference "
                            "trajectory."),
                io.Float.Input(
                    "guidance", default=4.5, min=0.0, max=20.0, step=0.1,
                    tooltip="Classifier-free guidance for the guided upscale "
                            "stages (paper uses 4.5-6). Same auto-skip rule "
                            "as cfg."),
                io.Int.Input(
                    "steps_per_stage", default=16, min=1, max=50, step=1,
                    tooltip="Guided-stage sampling steps per cascade stage "
                            "(upper bound; repo default 16/10)."),
                io.Float.Input(
                    "tau", default=0.6, min=0.05, max=0.95, step=0.05,
                    tooltip="Stage-entry noise level (paper cascade: 0.6, "
                            "0.3, 0.3). Lower = stronger content preservation."),
                io.Float.Input(
                    "filter_ratio", default=0.2, min=0.05, max=0.95,
                    step=0.05,
                    tooltip="Normalized Butterworth low-pass cutoff D "
                            "(direction alignment). Paper 0.4, repo default "
                            "0.2."),
                io.Float.Input(
                    "alpha_scale", default=1.0, min=0.0, max=2.0, step=0.05,
                    tooltip="Direction-alignment strength multiplier "
                            "(repo first-stage value 1.0)."),
                io.Float.Input(
                    "beta_scale", default=0.5, min=0.0, max=2.0, step=0.05,
                    tooltip="Acceleration-alignment strength multiplier "
                            "(repo default 0.5)."),
                io.Combo.Input(
                    "upsampling", options=["latent", "pixel"],
                    default="latent",
                    tooltip="Per-step reference upsample: latent bicubic "
                            "(repo default) or pixel decode->sharpen->encode. "
                            "The stage-initialization anchor is always the "
                            "pixel round-trip of the previous final image."),
                io.Int.Input(
                    "target_resolution", default=2048, min=1024, max=8192,
                    step=128,
                    tooltip="Target resolution in pixels; stages double the "
                            "base per stage until reached (2048 = one 2x "
                            "stage from 1024)."),
            ],
            outputs=[
                io.Latent.Output(display_name="High-Res Latent"),
            ],
        )

    @classmethod
    def validate_inputs(cls, target_resolution):
        # Uninitialized graph state passes through (2026-08-25 fix pattern).
        if target_resolution is None:
            return True
        if int(target_resolution) < 16:
            return "target_resolution must be >= 16"
        return True

    @classmethod
    def execute(cls, model, vae, positive, negative, latent_image,
                cfg=3.5, steps=30, guidance=4.5, steps_per_stage=16,
                tau=0.6, filter_ratio=0.2, alpha_scale=1.0, beta_scale=0.5,
                upsampling="latent", target_resolution=2048,
                noise_seed=0, denoise=1.0) -> io.NodeOutput:
        import comfy.utils

        # Gate BEFORE any model calls: flow prediction + 2D latents only.
        _require_flow_model(model)

        if isinstance(latent_image, dict):
            initial_latent = latent_image["samples"]
        else:
            initial_latent = latent_image

        if initial_latent.ndim == 5 and initial_latent.shape[2] == 1:
            raise ValueError(
                "HiFlow received a 3D (video) latent. It supports 2D image "
                "latents only — for video models use PixelRush."
            )

        device = model.load_device if hasattr(model, "load_device") \
            else torch.device("cpu")
        initial_latent = initial_latent.to(device)

        # Channel handling for empty latents (EmptyLatentImage may produce 4
        # channels for a 16-channel model) — the PixelRush convention.
        model_latent_channels = getattr(
            model.model.latent_format, "latent_channels", None)
        if model_latent_channels is not None and \
                initial_latent.shape[1] != model_latent_channels:
            is_empty = torch.count_nonzero(initial_latent) == 0
            if is_empty:
                logger.info(
                    "HiFlow: empty input latent has %d channels, model "
                    "expects %d — repeating channels",
                    initial_latent.shape[1], model_latent_channels,
                )
                initial_latent = comfy.utils.repeat_to_batch_size(
                    initial_latent, model_latent_channels, dim=1,
                )
            else:
                logger.warning(
                    "HiFlow: non-empty input latent has %d channels, model "
                    "expects %d — results may be unexpected",
                    initial_latent.shape[1], model_latent_channels,
                )

        cfg_obj = HiFlowConfig(
            tau=float(tau), steps=int(steps),
            steps_per_stage=int(steps_per_stage),
            cfg=float(cfg), guidance_high=float(guidance),
            filter_ratio=float(filter_ratio),
            alpha_scale=float(alpha_scale), beta_scale=float(beta_scale),
            upsampling=str(upsampling),
        )

        predict_x0_base = _make_predict_x0(
            model, positive, negative, cfg_scale=float(cfg))
        predict_x0_stage = _make_predict_x0(
            model, positive, negative, cfg_scale=float(guidance))

        # VAE adapters are ALWAYS needed — the stage-initialization anchor
        # is the pixel round-trip of the previous final latent in both
        # upsampling modes (plan D2).
        vae_decode, vae_encode = _make_vae_adapters(vae, device)

        base_sigmas = _base_sigmas(model, int(steps))

        # A content latent with denoise=1.0 is a foot-gun: sigma_start == 1
        # zeroes the content weight entirely (from-noise generation).
        if (torch.count_nonzero(initial_latent) > 0
                and float(denoise) > 0.9999):
            logger.warning(
                "HiFlow: denoise=1.0 with a non-empty latent — the input "
                "image is ignored (the base starts from pure noise). Lower "
                "denoise (e.g. 0.6) to upscale the connected latent."
            )

        # Progress: base transitions + per-stage transitions (upper bound).
        from .hiflow import _stage_latent_sizes
        sizes = _stage_latent_sizes(
            initial_latent.shape[-2], initial_latent.shape[-1],
            int(target_resolution), _downscale_ratio(vae),
        )
        total = max(1, len(base_sigmas) - 1 + len(sizes) * int(steps_per_stage))
        pbar = comfy.utils.ProgressBar(total)
        counter = {"n": 0}

        def progress_callback(i, total_steps, stage):
            counter["n"] += 1
            pbar.update_absolute(min(counter["n"], total))

        result = hiflow_cascade(
            initial_latent=initial_latent,
            base_sigmas=base_sigmas,
            predict_x0_base=predict_x0_base,
            predict_x0_stage=predict_x0_stage,
            target_resolution=int(target_resolution),
            cfg=cfg_obj,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            sharpen=_sharpen,
            vae_downscale=_downscale_ratio(vae),
            progress_callback=progress_callback,
            noise_seed=int(noise_seed),
            denoise=float(denoise),
        )
        pbar.update_absolute(total)

        return io.NodeOutput({"samples": result})
