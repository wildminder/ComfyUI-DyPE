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
