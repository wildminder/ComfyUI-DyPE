"""
PixelRush ComfyUI node — cascade-based high-resolution generation.

Provides a ComfyUI node that applies the PixelRush algorithm to upscale
and refine images using partial DDIM inversion + patch-based denoising.
Works with any ComfyUI model (SDXL, SD1.5, FLUX, etc.).
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F

from comfy_api.latest import ComfyExtension, io

from .pixelrush import PixelRushConfig, pixelrush_cascade

logger = logging.getLogger("ComfyUI-DyPE")


def _make_predict_eps(model, positive, negative, cfg_scale):
    """Create a predict_eps adapter that runs the model with CFG.

    Uses ComfyUI's full conditioning pipeline (process_conds) to properly
    build model_conds (y, c_crossattn, etc.) from the conditioning input.

    Returns a callable: predict_eps(latent, timestep) -> eps [B, C, H, W]
    """
    import comfy.samplers

    # Pre-process conditioning using ComfyUI's full pipeline
    # process_conds expects {"positive": [...], "negative": [...]}
    # and returns processed conds with model_conds built
    device = model.load_device if hasattr(model, 'load_device') else torch.device("cpu")

    # We need a noise tensor for process_conds — use a dummy
    # The actual noise shape doesn't matter for conditioning processing
    # but process_conds uses it for area resolution
    _conds_processed = None

    def _get_processed_conds(latent):
        nonlocal _conds_processed
        if _conds_processed is not None:
            return _conds_processed
        noise = torch.zeros_like(latent)
        conds_dict = {"positive": positive, "negative": negative}
        _conds_processed = comfy.samplers.process_conds(
            model.model, noise, conds_dict, latent.device
        )
        return _conds_processed

    def predict_eps(latent: torch.Tensor, timestep: int) -> torch.Tensor:
        # Convert timestep to sigma
        sigmas = model.model.model_sampling.sigmas
        if timestep < len(sigmas):
            sigma_val = sigmas[timestep].item()
        else:
            sigma_val = sigmas[-1].item()
        B = latent.shape[0]
        sigma = torch.full((B,), sigma_val, device=latent.device, dtype=latent.dtype)

        # Get processed conditioning
        conds = _get_processed_conds(latent)

        def run_cond(prompt_type):
            cond_list = conds.get(prompt_type, [])
            if len(cond_list) == 0:
                return torch.zeros_like(latent)
            cond = cond_list[0]
            model_conds = cond.get("model_conds", {})
            # Build kwargs for apply_model from processed model_conds
            c = {}
            for k, v in model_conds.items():
                # COND objects have a process() method that returns the tensor
                processed = v.process(latent) if hasattr(v, 'process') else v
                if processed is not None:
                    c[k] = processed
            eps = model.model.apply_model(latent, sigma, **c)
            return eps

        eps_cond = run_cond("positive")
        eps_uncond = run_cond("negative")

        # CFG
        return eps_uncond + cfg_scale * (eps_cond - eps_uncond)

    return predict_eps


def _make_alpha_bar_at(model):
    """Create an alpha_bar_at adapter from the model's sigma schedule.

    Returns a callable: alpha_bar_at(timestep) -> float
    """
    sigmas = model.model.model_sampling.sigmas
    # alpha_bar = 1 / (sigma^2 + 1)
    alphas_cumprod = 1.0 / (sigmas ** 2 + 1.0)

    def alpha_bar_at(timestep: int) -> float:
        if timestep < len(alphas_cumprod):
            return alphas_cumprod[timestep].item()
        return alphas_cumprod[-1].item()

    return alpha_bar_at


def _make_vae_adapters(vae):
    """Create VAE decode/encode adapters.

    Returns (vae_decode, vae_encode) callables.
    """
    def vae_decode(latent: torch.Tensor) -> torch.Tensor:
        # latent: [B, C, H, W] — ComfyUI VAE expects [B, C, H, W]
        if isinstance(latent, dict):
            latent = latent["samples"]
        # VAE decode expects unscaled latent
        # ComfyUI VAEs handle scaling internally
        decoded = vae.decode(latent)
        # decoded: [B, H, W, C] → [B, C, H, W] for bicubic upscale
        if decoded.dim() == 4 and decoded.shape[-1] == 3:
            decoded = decoded.movedim(-1, 1)
        return decoded

    def vae_encode(image: torch.Tensor) -> torch.Tensor:
        # image: [B, C, H, W] → VAE expects [B, H, W, C]
        if image.dim() == 4 and image.shape[1] == 3:
            image = image.movedim(1, -1)
        encoded = vae.encode(image)
        if isinstance(encoded, dict):
            return encoded["samples"]
        return encoded

    return vae_decode, vae_encode


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
        # Get initial latent
        if isinstance(latent_image, dict):
            initial_latent = latent_image["samples"]
        else:
            initial_latent = latent_image

        # Auto-detect patch size from native resolution
        if patch_h == 0 or patch_w == 0:
            # Use the initial latent size as patch size
            _, _, h, w = initial_latent.shape
            patch_h = h if patch_h == 0 else patch_h
            patch_w = w if patch_w == 0 else patch_w

        cfg_obj = PixelRushConfig(
            patch_h=patch_h,
            patch_w=patch_w,
            overlap=overlap,
            k_timestep=k_timestep,
            noise_lambda=noise_lambda,
            gaussian_sigma=gaussian_sigma,
            gaussian_kernel_size=gaussian_kernel_size,
        )

        # Create adapters
        predict_eps = _make_predict_eps(model, positive, negative, cfg)
        alpha_bar_at = _make_alpha_bar_at(model)
        vae_decode, vae_encode = _make_vae_adapters(vae)

        # Run PixelRush cascade
        result_latent = pixelrush_cascade(
            initial_latent=initial_latent,
            num_cascade_stages=num_cascade_stages,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            predict_eps=predict_eps,
            alpha_bar_at=alpha_bar_at,
            cfg=cfg_obj,
        )

        return io.NodeOutput({"samples": result_latent})
