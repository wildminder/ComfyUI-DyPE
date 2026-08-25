"""
FreeScale — tuning-free higher-resolution generation via scale fusion.

Combines three inference-time operations:
  1. Self-cascade upscaling (decode → bicubic → encode → noise → denoise)
  2. Frequency-aware scale fusion (global high-freq + local low-freq)
  3. Detail-controlled latent blending (cosine decay schedule)

Reference: FreeScale paper (arXiv:2412.09626).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

logger = logging.getLogger("ComfyUI-DyPE")


# ---------------------------------------------------------------------------
# Gaussian blur (frequency decomposition)
# ---------------------------------------------------------------------------

def gaussian_kernel_2d(
    kernel_size: int,
    sigma: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Tensor:
    """Create a normalized 2D Gaussian kernel ``[K, K]``.

    Parameters
    ----------
    kernel_size : int
        Must be odd.
    sigma : float
        Standard deviation of the Gaussian.

    Returns
    -------
    Tensor
        ``[kernel_size, kernel_size]`` normalized kernel.
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)

    kernel = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # Outer product for 2D
    kernel_2d = torch.outer(kernel, kernel)
    return kernel_2d


def gaussian_blur_2d(
    x: Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> Tensor:
    """Depthwise Gaussian blur on ``[B, C, H, W]`` tensor.

    Used as an approximate low-pass filter for frequency decomposition
    in scale fusion.

    Parameters
    ----------
    x : Tensor
        Input ``[B, C, H, W]``.
    kernel_size : int
        Gaussian kernel size (must be odd).
    sigma : float
        Gaussian standard deviation.

    Returns
    -------
    Tensor
        Blurred tensor, same shape as input.
    """
    b, c, h, w = x.shape

    kernel = gaussian_kernel_2d(kernel_size, sigma, device=x.device, dtype=x.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(c, 1, 1, 1)

    padding = kernel_size // 2
    return F.conv2d(x, kernel, padding=padding, groups=c)


# ---------------------------------------------------------------------------
# Scale fusion
# ---------------------------------------------------------------------------

def scale_fusion(
    global_features: Tensor,
    local_features: Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> Tensor:
    """FreeScale frequency fusion: global high-freq + local low-freq.

    .. math::
        \\text{fused} = \\text{global} - G(\\text{global}) + G(\\text{local})

    where :math:`G` is Gaussian blur (low-pass filter).

    - Global high frequencies are retained (semantic detail positioned globally)
    - Local low frequencies replace global low frequencies (locally coherent structure)

    Parameters
    ----------
    global_features : Tensor
        ``[B, C, H, W]`` from global (full-resolution) attention.
    local_features : Tensor
        ``[B, C, H, W]`` from local (windowed) attention.
    kernel_size : int
        Gaussian kernel size (must be odd).
    sigma : float
        Gaussian standard deviation.

    Returns
    -------
    Tensor
        Fused features, same shape as inputs.
    """
    global_blurred = gaussian_blur_2d(global_features, kernel_size, sigma)
    local_blurred = gaussian_blur_2d(local_features, kernel_size, sigma)

    # global_high = global - blur(global)
    # fused = global_high + blur(local)
    #       = global - blur(global) + blur(local)
    return global_features - global_blurred + local_blurred


# ---------------------------------------------------------------------------
# Detail control (cosine decay)
# ---------------------------------------------------------------------------

def cosine_detail_weight(
    timestep_index: int,
    total_steps: int,
    alpha: float = 2.0,
) -> float:
    """Compute the detail-control coefficient :math:`c_t`.

    The paper uses :math:`t` counting down from :math:`T` to 0.
    Our ``timestep_index`` counts up from 0 to ``total_steps-1``.
    The conversion is: paper's :math:`t = T - \\text{timestep_index}`.

    .. math::
        c_t = \\alpha \\cdot \\frac{1 + \\cos\\left(\\frac{t}{T}\\pi\\right)}{2}

    where :math:`t = T - \\text{timestep_index}` (remaining steps).

    Early in denoising (``timestep_index=0``, paper's :math:`t=T`):
    :math:`c_t = \\alpha` (maximum upsampled signal).
    Late in denoising (``timestep_index=T-1``, paper's :math:`t=1`):
    :math:`c_t \\approx 0` (stabilize structure).

    Parameters
    ----------
    timestep_index : int
        Current step index (0 = first denoising step, total_steps-1 = last).
    total_steps : int
        Total number of denoising steps.
    alpha : float
        Detail strength. Paper default: 2.0. For 8K: ≤ 1.0.

    Returns
    -------
    float
        Blend coefficient in ``[0, alpha]``.
    """
    t = timestep_index
    T = total_steps
    cosine = 0.5 * (1.0 + math.cos(t / T * math.pi))
    return alpha * cosine


def blend_detail_latents(
    noisy_upsampled: Tensor,
    ordinary: Tensor,
    timestep_index: int,
    total_steps: int,
    alpha: float = 2.0,
) -> Tensor:
    """Blend upsampled noisy latent with ordinary diffusion latent.

    .. math::
        \\hat{z} = c_t \\cdot \\tilde{z} + (1 - c_t) \\cdot z

    Parameters
    ----------
    noisy_upsampled : Tensor
        The upsampled + noised latent ``[B, C, H, W]``.
    ordinary : Tensor
        The ordinary diffusion latent at the current step ``[B, C, H, W]``.
    timestep_index : int
        Current step index.
    total_steps : int
        Total denoising steps.
    alpha : float
        Detail strength.

    Returns
    -------
    Tensor
        Blended latent, same shape as inputs.
    """
    c = cosine_detail_weight(timestep_index, total_steps, alpha)
    return c * noisy_upsampled + (1.0 - c) * ordinary


# ---------------------------------------------------------------------------
# Forward noising
# ---------------------------------------------------------------------------

def forward_noise(
    z0: Tensor,
    eps: Tensor,
    alpha_bar_k: Tensor | float,
) -> Tensor:
    """Add noise at timestep K (DDIM forward).

    .. math::
        z_K = \\sqrt{\\bar{\\alpha}_K} \\cdot z_0 + \\sqrt{1 - \\bar{\\alpha}_K} \\cdot \\epsilon

    Parameters
    ----------
    z0 : Tensor
        Clean latent ``[B, C, H, W]``.
    eps : Tensor
        Gaussian noise, same shape as ``z0``.
    alpha_bar_k : Tensor or float
        Cumulative alpha at timestep K.

    Returns
    -------
    Tensor
        Noised latent, same shape as ``z0``.
    """
    if not torch.is_tensor(alpha_bar_k):
        alpha_bar_k = torch.tensor(alpha_bar_k, device=z0.device, dtype=z0.dtype)

    a = alpha_bar_k.to(device=z0.device, dtype=z0.dtype)
    return a.sqrt() * z0 + (1.0 - a).sqrt() * eps


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FreeScaleConfig:
    """Configuration for FreeScale cascade.

    Parameters
    ----------
    target_resolution : int
        Target resolution (e.g., 2048 for 2x from 1024).
    noise_timestep : int
        Timestep K for forward noising. Paper default: 700 (in 1000-step schedule).
    cosine_scale : float
        Detail control alpha. Paper default: 2.0. For 8K: ≤ 1.0.
    fast_mode : bool
        If True, use 4-global-window mode (faster, slightly lower quality).
    gaussian_kernel_size : int
        Gaussian blur kernel size for scale fusion. Must be odd.
    gaussian_sigma : float
        Gaussian blur sigma for scale fusion.
    num_inference_steps : int
        Number of denoising steps per resolution level.
    """

    target_resolution: int = 2048
    noise_timestep: int = 700
    cosine_scale: float = 2.0
    fast_mode: bool = True
    gaussian_kernel_size: int = 5
    gaussian_sigma: float = 1.0
    num_inference_steps: int = 50
