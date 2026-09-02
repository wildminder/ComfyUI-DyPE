"""
HiFlow — training-free high-resolution generation with flow-aligned guidance.

Turns a base-resolution flow-model generation into a higher-resolution one by
reusing the entire low-resolution denoising trajectory as a time-matched
reference for the high-resolution trajectory. Three alignments guide the
high-res stage: initialization (start at noise level tau from a noisy
time-matched reference prediction), direction (nudge the low frequencies of
the predicted clean sample toward the reference's), and acceleration (align
the change of the velocity with the reference's).

Reference: HiFlow paper (arXiv:2504.06232, NeurIPS 2025).

Space contract (plan 2026-09-03, D2): the core algorithm runs entirely in VAE
latent space (the ComfyUI LATENT convention, std ~ 1). The node-level x0
adapter owns the VAE<->model conversions around each model call; this module
never sees model-space tensors.
"""

from __future__ import annotations

import logging

import torch
import torch.fft as fft

Tensor = torch.Tensor

logger = logging.getLogger("ComfyUI-DyPE")


# ---------------------------------------------------------------------------
# Butterworth low-pass filter mask (frequency domain)
# ---------------------------------------------------------------------------

def butterworth_low_pass_filter_2d(
    shape: tuple[int, ...],
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    ratio: float = 0.2,
    n: int = 4,
) -> Tensor:
    """2D Butterworth low-pass mask for a latent of spatial shape (H, W).

    HiFlow reference formula (utils.py ``butterworth_low_pass_filter_2d``):

        mask[h, w] = 1 / (1 + (d2 / ratio**2) ** n)
        d2 = (2*h/H - 1)**2 + (2*w/W - 1)**2

    with n = 4 (larger n approaches the ideal low-pass; smaller approaches
    Gaussian). ``ratio`` is the normalized cutoff D from the paper (0.4 paper
    text, 0.2 repo default). The mask is built vectorized — the reference
    implementation loops h*w in Python, which takes seconds at 4K latents.

    Returns a [H, W] tensor broadcastable against an fftshifted spectrum.
    ``ratio <= 0`` returns an all-zero mask (reference behavior).
    """
    h, w = shape[-2], shape[-1]
    if h <= 0 or w <= 0:
        raise ValueError(f"butterworth mask needs positive H, W; got {shape!r}")
    if ratio <= 0.0:
        return torch.zeros((h, w), device=device, dtype=dtype)

    # Normalized, centered coordinates in [-1, 1): d=0 at the fftshifted DC.
    ys = torch.linspace(-1.0, 1.0 - 2.0 / h, h, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0 - 2.0 / w, w, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    d2 = grid_y.square() + grid_x.square()

    return 1.0 / (1.0 + (d2 / (ratio * ratio)) ** n)


# ---------------------------------------------------------------------------
# FFT frequency split
# ---------------------------------------------------------------------------

def split_frequency_components_fft(
    x: Tensor,
    freq_filter: Tensor,
    is_low: bool = True,
) -> Tensor:
    """Split ``x`` into low/high frequency components via a 2D FFT mask.

    Exact port of the HiFlow reference (utils.py
    ``split_frequency_components_fft``):

        low  = IFFT2(IFFTSHIFT(FFTSHIFT(FFT2(x)) * filter)).real
        high = same with (1 - filter)

    The filter must be [H, W] (or broadcastable); it multiplies the
    fftshifted spectrum, where DC sits at the array center. Computation runs
    in the filter's dtype (fp32 upcast) and the result is cast back to
    ``x.dtype`` — fp16 inputs do not produce NaNs from the FFT.
    """
    x_f32 = x.to(freq_filter.dtype)
    x_freq = fft.fftshift(fft.fft2(x_f32))
    masked = x_freq * freq_filter if is_low else x_freq * (1.0 - freq_filter)
    split = fft.ifft2(fft.ifftshift(masked)).real
    return split.to(x.dtype)
