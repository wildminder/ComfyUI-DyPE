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

    Uses ComfyUI's full conditioning pipeline:
      1. ``convert_cond`` — convert tuple conditioning to dict format
      2. ``process_conds`` — build model_conds (y, c_crossattn, etc.)
      3. ``get_area_and_mult`` — extract processed conditioning tensors
      4. ``apply_model`` — run the model

    Returns a callable: predict_eps(latent, timestep) -> eps [B, C, H, W]
    """
    import comfy.samplers
    import comfy.sampler_helpers
    import comfy.model_management

    device = model.load_device if hasattr(model, 'load_device') else torch.device("cpu")

    # Ensure the model is loaded to GPU and pre_run is called
    # pre_run sets model.model.current_patcher = model (the ModelPatcher)
    # which is required by apply_hooks and prepare_state
    comfy.model_management.load_models_gpu([model])
    model.pre_run()

    # Cache for processed conditioning (built once, reused across calls)
    _processed = None

    def _get_processed(latent):
        """Build processed conditioning using ComfyUI's canonical pipeline.

        convert_cond converts tuple format [(tensor, dict), ...] to dict format
        [dict, ...] which process_conds expects.
        """
        nonlocal _processed
        if _processed is not None:
            return _processed
        # Step 1: Convert tuple conditioning to dict format
        pos_converted = comfy.sampler_helpers.convert_cond(positive)
        neg_converted = comfy.sampler_helpers.convert_cond(negative)
        conds_dict = {"positive": pos_converted, "negative": neg_converted}
        # Step 2: Process conds (builds model_conds via encode_model_conds)
        noise = torch.zeros_like(latent)
        _processed = comfy.samplers.process_conds(
            model.model, noise, conds_dict, device
        )
        return _processed

    def predict_eps(latent: torch.Tensor, timestep: int) -> torch.Tensor:
        # Move latent to model device for inference
        latent = latent.to(device)

        # Convert timestep to sigma
        sigmas = model.model.model_sampling.sigmas
        if timestep < len(sigmas):
            sigma_val = sigmas[timestep].item()
        else:
            sigma_val = sigmas[-1].item()
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
            eps = model.model.apply_model(p.input_x, sigma, **c)
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


def _make_vae_adapters(vae, device, model=None):
    """Create VAE decode/encode adapters.

    Returns (vae_decode, vae_encode) callables.
    All tensors are moved to ``device`` for GPU acceleration.
    Handles both 2D VAEs (latent_dim=2, 4D latents [B,C,H,W]) and
    3D/video VAEs (latent_dim=3, 5D latents [B,C,T,H,W]).
    Uses model.process_latent_out/in to convert between model latent
    format and VAE latent format (needed for Qwen, Krea2, etc.).
    """
    latent_dim = getattr(vae, 'latent_dim', 2)
    process_latent_out = None
    process_latent_in = None
    if model is not None and hasattr(model, 'model'):
        if hasattr(model.model, 'process_latent_out'):
            process_latent_out = model.model.process_latent_out
        if hasattr(model.model, 'process_latent_in'):
            process_latent_in = model.model.process_latent_in

    def vae_decode(latent: torch.Tensor) -> torch.Tensor:
        # latent: [B, C, H, W] — ComfyUI VAE expects [B, C, H, W]
        if isinstance(latent, dict):
            latent = latent["samples"]
        latent = latent.to(device)
        # Convert from model latent format to VAE latent format
        if process_latent_out is not None:
            latent = process_latent_out(latent)
        # For 3D VAEs (video), add temporal dimension: [B,C,H,W] -> [B,C,1,H,W]
        if latent_dim == 3 and latent.ndim == 4:
            latent = latent.unsqueeze(2)
        # VAE decode expects unscaled latent
        # ComfyUI VAEs handle scaling internally
        decoded = vae.decode(latent)
        # For 3D VAEs, decoded may be [B, T, H, W, C] — squeeze temporal dim
        if decoded.ndim == 5:
            decoded = decoded.squeeze(1)  # Remove T dimension (T=1)
        # decoded: [B, H, W, C] → [B, C, H, W] for bicubic upscale
        if decoded.dim() == 4 and decoded.shape[-1] == 3:
            decoded = decoded.movedim(-1, 1)
        return decoded

    def vae_encode(image: torch.Tensor) -> torch.Tensor:
        # image: [B, C, H, W] → VAE expects [B, H, W, C]
        image = image.to(device)
        if image.dim() == 4 and image.shape[1] == 3:
            image = image.movedim(1, -1)
        encoded = vae.encode(image)
        if isinstance(encoded, dict):
            encoded = encoded["samples"]
        # For 3D VAEs (video), remove temporal dimension: [B,C,1,H,W] -> [B,C,H,W]
        if latent_dim == 3 and encoded.ndim == 5:
            encoded = encoded.squeeze(2)
        # Convert from VAE latent format to model latent format
        if process_latent_in is not None:
            encoded = process_latent_in(encoded)
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
        device = model.load_device if hasattr(model, 'load_device') else torch.device("cpu")
        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        # Move initial latent to model device for GPU acceleration
        initial_latent = initial_latent.to(device)

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
