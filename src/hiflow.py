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


# ---------------------------------------------------------------------------
# Stage sigma schedule + alignment scales
# ---------------------------------------------------------------------------

def build_stage_sigmas(
    full_sigmas: torch.Tensor,
    tau: float,
    steps: int,
) -> torch.Tensor:
    """Build the descending sigma schedule a high-res stage walks.

    Mirrors the HiFlow reference ``dlfg_timesteps = timesteps[-n:]`` (R4): the
    stage runs on the LAST ``steps`` sigmas of the model's full schedule —
    same spacing as the base run, entered at the sigma nearest ``tau`` from
    below — plus a trailing 0 so the walk lands on a clean sample.

    Rules:
      - the entry sigma must satisfy sigma_entry <= tau + 1e-6 (clamped to the
        largest schedule sigma below tau; if tau exceeds sigma_max the full
        schedule runs and a warning is logged);
      - the result is strictly descending except for the trailing 0 and has
        exactly ``steps`` transitions (len == steps + 1);
      - ``tau <= 0`` or ``steps < 1`` raises ValueError.
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be positive; got {tau!r}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps!r}")

    sigmas = full_sigmas.float()
    if sigmas.numel() < 2:
        raise ValueError(f"full_sigmas needs >= 2 entries; got {tuple(sigmas)}")
    if not bool(torch.all(sigmas[:-1] >= sigmas[1:])):
        raise ValueError("full_sigmas must be non-increasing")

    # Interior (non-zero) sigmas; the schedule convention ends at 0.
    interior = sigmas[sigmas > 0]
    if tau > float(interior.max()) + 1e-9:
        logger.warning(
            "HiFlow: tau %.4f above schedule max %.4f — clamping the stage "
            "entry to the full schedule", tau, float(interior.max()),
        )

    # Entry: the largest schedule sigma <= tau (fall back to the smallest
    # interior sigma when tau sits below the whole schedule — degenerate but
    # defined). Everything strictly below the entry is walkable; keep at most
    # steps-1 of the tail (the LAST ones, matching the reference's [-n:]
    # slice), then land on 0.
    below = interior[interior <= tau + 1e-9]
    entry = float(below.max()) if below.numel() > 0 else float(interior.min())

    tail = interior[interior < entry - 1e-12]
    if tail.numel() > steps - 1:
        tail = tail[-(steps - 1):]

    stage = torch.cat([torch.tensor([entry]), tail, torch.zeros(1)])
    return stage


def alignment_scales(
    stage_sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step direction (alpha) and acceleration (beta) scales.

    Theory form (plan D5): ``alpha_i = beta_i = sigma_i / sigma_entry`` for
    every entry sigma in the stage schedule — 1.0 at the stage entry,
    decaying to ~0 at the clean end (the trailing 0 gets an exact 0). The
    multiplier (cfg.alpha_scale / cfg.beta_scale) is applied by the caller.
    """
    sigmas = stage_sigmas.float()
    entry = float(sigmas[0])
    if entry <= 0.0:
        raise ValueError("stage schedule must enter at a positive sigma")
    scales = sigmas / entry
    scales = torch.clamp(scales, min=0.0)
    return scales, scales.clone()
