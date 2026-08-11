"""
PixelRush — training-free cascade-based high-resolution generation.

Turns high-resolution generation into a sequence of coarse-to-fine cascade
refinements: generate a native-resolution image, upscale it, then use a
single partial DDIM inversion + single denoising step per overlapping
latent patch to add detail rather than regenerate the whole image from noise.

Reference: PixelRush paper (arXiv:2602.12769).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Iterator, Tuple

import torch
import torch.nn.functional as F


Tensor = torch.Tensor

logger = logging.getLogger("ComfyUI-DyPE")


@dataclass
class PixelRushConfig:
    """Configuration for PixelRush cascade refinement.

    Parameters
    ----------
    patch_h, patch_w : int
        Latent-space patch dimensions.  For SDXL (native 1024px, VAE
        downscale 8×): 128×128.  For SD1.5 (native 512px): 64×64.
    overlap : float
        Fractional overlap along H and W (0.0–0.75).  Paper default: 0.50.
    k_timestep : int
        Partial-inversion timestep.  Must correspond to a valid timestep
        in the model's schedule.  Paper default: 249.
    noise_lambda : float
        Noise injection strength for slerp between predicted and random
        noise.  Paper default: 0.95.
    gaussian_sigma : float
        Gaussian feathering sigma.  Paper default: 8.0.
    gaussian_kernel_size : int
        Gaussian blur kernel size (must be odd).  Paper default: 41.
    eps : float
        Numerical stability epsilon.
    """

    patch_h: int
    patch_w: int
    overlap: float = 0.50
    k_timestep: int = 249
    noise_lambda: float = 0.95
    gaussian_sigma: float = 8.0
    gaussian_kernel_size: int = 41
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Spherical interpolation
# ---------------------------------------------------------------------------

def spherical_lerp(a: Tensor, b: Tensor, t: float, eps: float = 1e-7) -> Tensor:
    """Spherical interpolation (SLERP) between tensors ``a`` and ``b``.

    Treats each sample's complete latent tensor as one vector.
    Falls back to linear interpolation when vectors are nearly parallel.

    Parameters
    ----------
    a, b : Tensor
        Shape ``[B, C, H, W]`` (or any shape; flattened to ``[B, N]``).
    t : float
        Interpolation coefficient in ``[0, 1]``.  ``t=0`` → ``a``,
        ``t=1`` → ``b``.

    Returns
    -------
    Tensor
        Same shape as ``a``.
    """
    a_flat = a.flatten(1)
    b_flat = b.flatten(1)

    a_norm = a_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    b_norm = b_flat.norm(dim=1, keepdim=True).clamp_min(eps)

    a_unit = a_flat / a_norm
    b_unit = b_flat / b_norm

    cosine = (a_unit * b_unit).sum(dim=1, keepdim=True).clamp(-1 + eps, 1 - eps)
    omega = torch.acos(cosine)
    sin_omega = torch.sin(omega).clamp_min(eps)

    t_tensor = torch.full_like(omega, t)

    # Spherical direction (use raw vectors, matching reference code)
    direction = (
        torch.sin((1.0 - t_tensor) * omega) / sin_omega * a_flat
        + torch.sin(t_tensor * omega) / sin_omega * b_flat
    )

    # Interpolate magnitudes separately (linear)
    magnitude = (1.0 - t_tensor) * a_norm + t_tensor * b_norm
    return (direction * magnitude).view_as(a)


# ---------------------------------------------------------------------------
# Gaussian feathering
# ---------------------------------------------------------------------------

def gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Returns a normalized ``[1, 1, K, K]`` Gaussian convolution kernel."""
    assert kernel_size % 2 == 1, "Use an odd kernel size."

    coords = torch.arange(kernel_size, device=device, dtype=dtype)
    coords = coords - kernel_size // 2

    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()

    kernel = torch.outer(g, g)
    return kernel[None, None]  # [1, 1, K, K]


def gaussian_feather_mask(
    height: int,
    width: int,
    sigma: float,
    kernel_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create a ``[1, 1, H, W]`` smooth feather mask.

    Blurs an all-one patch with zero padding.  The center remains near one,
    and the boundaries smoothly decay — ideal for overlap-add blending.
    """
    hard_mask = torch.ones((1, 1, height, width), device=device, dtype=dtype)
    kernel = gaussian_kernel_2d(kernel_size, sigma, device, dtype)

    pad = kernel_size // 2
    blurred = F.conv2d(hard_mask, kernel, padding=pad)

    # Normalize peak to 1
    blurred = blurred / blurred.amax().clamp_min(1e-8)
    return blurred


# ---------------------------------------------------------------------------
# Patch positions
# ---------------------------------------------------------------------------

def patch_positions(
    full_h: int,
    full_w: int,
    patch_h: int,
    patch_w: int,
    overlap: float,
) -> Iterator[Tuple[int, int]]:
    """Yield ``(y, x)`` top-left coordinates covering a latent completely.

    The final patch in each dimension is shifted so it touches the edge.
    """
    assert 0.0 <= overlap < 1.0

    stride_h = max(1, int(round(patch_h * (1.0 - overlap))))
    stride_w = max(1, int(round(patch_w * (1.0 - overlap))))

    def starts(full_size: int, patch_size: int, stride: int) -> list[int]:
        if full_size <= patch_size:
            return [0]
        values = list(range(0, full_size - patch_size + 1, stride))
        last = full_size - patch_size
        if values[-1] != last:
            values.append(last)
        return values

    ys = starts(full_h, patch_h, stride_h)
    xs = starts(full_w, patch_w, stride_w)

    for y in ys:
        for x in xs:
            yield y, x


# ---------------------------------------------------------------------------
# DDIM inversion / denoising
# ---------------------------------------------------------------------------

def ddim_forward_one_step(
    z0: Tensor,
    eps0: Tensor,
    alpha_bar_k: Tensor | float,
) -> Tensor:
    """Deterministic DDIM inversion from timestep 0 to K.

    Since ``alpha_bar_0 = 1``::

        z_K = sqrt(alpha_bar_K) * z_0
              + sqrt(1 - alpha_bar_K) * eps(z_0, 0)
    """
    if not torch.is_tensor(alpha_bar_k):
        alpha_bar_k = torch.tensor(alpha_bar_k, device=z0.device, dtype=z0.dtype)

    a = alpha_bar_k.to(device=z0.device, dtype=z0.dtype)
    return a.sqrt() * z0 + (1.0 - a).sqrt() * eps0


def ddim_reverse_one_step_to_zero(
    z_k: Tensor,
    eps_k: Tensor,
    alpha_bar_k: Tensor | float,
) -> Tensor:
    """Deterministic DDIM denoising from K to 0.

    Since ``alpha_bar_0 = 1``::

        z_0_hat = (z_K - sqrt(1-alpha_bar_K) * eps_K) / sqrt(alpha_bar_K)
    """
    if not torch.is_tensor(alpha_bar_k):
        alpha_bar_k = torch.tensor(alpha_bar_k, device=z_k.device, dtype=z_k.dtype)

    a = alpha_bar_k.to(device=z_k.device, dtype=z_k.dtype)
    return (z_k - (1.0 - a).sqrt() * eps_k) / a.sqrt().clamp_min(1e-8)


# ---------------------------------------------------------------------------
# Single cascade stage
# ---------------------------------------------------------------------------

@torch.no_grad()
def refine_latent_once(
    coarse_latent: Tensor,
    predict_eps: Callable[[Tensor, int], Tensor],
    alpha_bar_at: Callable[[int], Tensor | float],
    cfg: PixelRushConfig,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Tensor:
    """Apply one PixelRush refinement stage to a coarse latent.

    Parameters
    ----------
    coarse_latent : Tensor
        ``[B, C, H, W]`` latent obtained by pixel-space upsampling and
        VAE encoding.
    predict_eps : callable
        ``predict_eps(latent, timestep) -> [B, C, H, W]`` epsilon prediction
        (should already include CFG).
    alpha_bar_at : callable
        ``alpha_bar_at(K) -> alpha_cumprod[K]``.
    cfg : PixelRushConfig
        Hyperparameters.
    progress_callback : callable, optional
        ``progress_callback(patch_idx, total_patches)`` called after each
        patch is refined.  Used for ComfyUI progress bar integration.

    Returns
    -------
    Tensor
        Refined latent ``[B, C, H, W]``.
    """
    b, c, full_h, full_w = coarse_latent.shape
    assert cfg.patch_h <= full_h and cfg.patch_w <= full_w, (
        "Patch dimensions must not exceed the latent."
    )

    alpha_k = alpha_bar_at(cfg.k_timestep)

    # Overlap-add buffers
    output_sum = torch.zeros_like(coarse_latent)
    weight_sum = torch.zeros_like(coarse_latent)

    feather = gaussian_feather_mask(
        cfg.patch_h,
        cfg.patch_w,
        sigma=cfg.gaussian_sigma,
        kernel_size=cfg.gaussian_kernel_size,
        device=coarse_latent.device,
        dtype=coarse_latent.dtype,
    )  # [1, 1, patch_h, patch_w]

    # Collect all patch positions for progress reporting
    positions = list(patch_positions(
        full_h=full_h,
        full_w=full_w,
        patch_h=cfg.patch_h,
        patch_w=cfg.patch_w,
        overlap=cfg.overlap,
    ))
    total_patches = len(positions)
    logger.info(
        "PixelRush: refining %dx%d latent with %dx%d patches, %d patches total (overlap=%.0f%%)",
        full_h, full_w, cfg.patch_h, cfg.patch_w, total_patches, cfg.overlap * 100,
    )

    for idx, (y, x) in enumerate(positions):
        if idx % 4 == 0:
            logger.info("PixelRush: patch %d/%d", idx + 1, total_patches)
        patch_0 = coarse_latent[:, :, y:y + cfg.patch_h, x:x + cfg.patch_w]

        # 1. Partial DDIM inversion: 0 -> K
        eps_inv = predict_eps(patch_0, timestep=0)
        # Ensure eps is on the same device as the patch
        eps_inv = eps_inv.to(patch_0.device)
        patch_k = ddim_forward_one_step(patch_0, eps_inv, alpha_k)

        # 2. One-step denoise: K -> 0
        eps_pred = predict_eps(patch_k, timestep=cfg.k_timestep)
        eps_pred = eps_pred.to(patch_k.device)

        # 3. Noise injection
        eps_rand = torch.randn_like(eps_pred)
        eps_injected = spherical_lerp(eps_pred, eps_rand, t=cfg.noise_lambda)

        # 4. Reverse DDIM step: K -> 0
        refined_patch = ddim_reverse_one_step_to_zero(patch_k, eps_injected, alpha_k)

        # 5. Gaussian-feather overlap-add
        output_sum[:, :, y:y + cfg.patch_h, x:x + cfg.patch_w] += (
            refined_patch * feather
        )
        weight_sum[:, :, y:y + cfg.patch_h, x:x + cfg.patch_w] += feather

        # 6. Progress callback
        if progress_callback is not None:
            progress_callback(idx + 1, total_patches)

    return output_sum / weight_sum.clamp_min(cfg.eps)


# ---------------------------------------------------------------------------
# Full cascade
# ---------------------------------------------------------------------------

@torch.no_grad()
def pixelrush_cascade(
    initial_latent: Tensor,
    num_cascade_stages: int,
    vae_decode: Callable[[Tensor], Tensor],
    vae_encode: Callable[[Tensor], Tensor],
    predict_eps: Callable[[Tensor, int], Tensor],
    alpha_bar_at: Callable[[int], Tensor | float],
    cfg: PixelRushConfig,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> Tensor:
    """PixelRush cascade: repeatedly upscale and refine.

    Parameters
    ----------
    initial_latent : Tensor
        Native-resolution base latent ``[B, C, H, W]``.
    num_cascade_stages : int
        1: native → 2×, 2: native → 4×, 3: native → 8×.
    vae_decode : callable
        ``vae_decode(latent) -> image`` (B, C_img, H_img, W_img).
    vae_encode : callable
        ``vae_encode(image) -> latent`` (B, C, H, W).
    predict_eps : callable
        ``predict_eps(latent, timestep) -> eps`` (with CFG).
    alpha_bar_at : callable
        ``alpha_bar_at(timestep) -> alpha_bar``.
    cfg : PixelRushConfig
    progress_callback : callable, optional
        ``progress_callback(patch_idx, total_patches, stage, num_stages)``
        called after each patch is refined.  Used for ComfyUI progress
        bar integration.

    Returns
    -------
    Tensor
        Final refined latent at target resolution.
    """
    z = initial_latent

    for stage in range(num_cascade_stages):
        logger.info(
            "PixelRush: cascade stage %d/%d — latent shape %s",
            stage + 1, num_cascade_stages, tuple(z.shape),
        )
        # Pixel-space cascade upsample: latent → RGB → 2× bicubic → latent
        image = vae_decode(z)

        image_up = F.interpolate(
            image,
            scale_factor=2.0,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        coarse_latent = vae_encode(image_up)
        # Ensure coarse_latent is on the same device as the model output
        # (VAE may return on CPU even if input was on GPU)
        coarse_latent = coarse_latent.to(image_up.device)
        # For 3D latent models, vae_encode may return 5D [B,C,T,H,W].
        # Squeeze temporal dim for the 4D spatial core algorithm.
        # predict_eps will unsqueeze back to 5D before calling apply_model.
        if coarse_latent.ndim == 5:
            coarse_latent = coarse_latent.squeeze(2)  # [B, C, H, W]
        logger.info(
            "PixelRush: upscaled to %s, starting patch refinement",
            tuple(coarse_latent.shape),
        )

        # Patch-based refinement
        if progress_callback is not None:
            def stage_callback(patch_idx, total_patches):
                progress_callback(patch_idx, total_patches, stage, num_cascade_stages)
        else:
            stage_callback = None

        z = refine_latent_once(
            coarse_latent=coarse_latent,
            predict_eps=predict_eps,
            alpha_bar_at=alpha_bar_at,
            cfg=cfg,
            progress_callback=stage_callback,
        )
        logger.info("PixelRush: stage %d complete", stage + 1)

    return z
