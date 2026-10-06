"""
I-Max — tuning-free resolution extrapolation for flow models (engine).

Two-pass resolution extrapolation for rectified-flow T2I models
(I-Max, arXiv 2410.07536): a low-resolution pass produces a fixed guidance
latent; the target-resolution pass then has its clean prediction pulled
toward the low-pass of that guidance with a CFG-shaped term whose strength
decays over the pass (Projected Flow, paper §2.2), while the schedule shift
re-balances the SNR between the two passes.

This module is the pure-torch engine (the ``src/hiflow.py`` contract):
ZERO comfy imports, ``@torch.no_grad()`` everywhere, dependency-injected
``predict_x0`` callables. The engine works on VAE-space latents end to end
— the node layer owns the VAE round trip of the guidance latent, the
model-space latent-format conversions and the per-pass model options.

Design decisions implemented here (plan 2026-10-05, D6/D7/D10/D14):

- D6 — Projected Flow runs in x0 space. ``sampling_function`` returns the
  denoised x̂₁, and with a linear low-pass ``P`` the reference's
  velocity-space correction ``v' = v + c·(P(v_G) − P(v))`` is exactly
  ``x̂₁' = x̂₁ + c·(P(G) − P(x̂₁))`` (the ``P(x_t)`` terms cancel). All four
  schedules of the reference are expressible from ``{x̂₁, G, P(x̂₁), P(G), c}``.
- D7 — ``P`` is a torch Haar low-pass. Zeroing every Haar detail band and
  inverting IS "iterated box average over 2^L, then nearest upsample" —
  see :func:`haar_lowpass`. ``pywt`` stays a test-only oracle.
- D10 — the low-resolution pass size is aspect-preserving and
  area-normalised (the reference's ``H // int(scale_factor + 0.5)``
  distorts aspect whenever ``round(s) != s``).
- D14 — ``denoise`` applies to pass B only: pass A always starts from pure
  noise; pass B enters at the denoise-truncated schedule's first sigma
  (the KSampler img2img convention used by HiFlow).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

logger = logging.getLogger("ComfyUI-DyPE")


# ---------------------------------------------------------------------------
# Haar low-pass — the I-Max projection P (paper §2.2)
# ---------------------------------------------------------------------------

def haar_lowpass(x: Tensor, level: int = 1) -> Tensor:
    """Low-pass ``x`` by zeroing every Haar detail band (the projection P).

    Haar DWT splits a signal into a coarse average band plus detail bands.
    The I-Max guidance projection keeps ONLY the coarse band: zeroing all
    detail coefficients and inverting the transform. For the Haar basis that
    inverse has a closed form — each 2^L × 2^L block of the input is replaced
    by its mean (L iterations of paired box averages), and the result is the
    nearest-neighbour upsample of the 2^L-downsampled average. This function
    computes exactly that, in pure torch:

        pad H/W up to a multiple of 2^L (periodic wrap)
        -> avg_pool2d(2^L)                (the coarse band)
        -> repeat_interleave(2^L)         (nearest upsample)
        -> crop back to (H, W)

    which is identical to ``pywt.wavedec2(x, 'haar', level=L)`` + zeroed
    details + ``waverec2`` up to the boundary mode (wrap here, pywt's
    symmetric there) — pinned by the pywt-oracle test.

    Contract: shape-preserving, ``[B, C, H, W]`` in/out, works for odd H/W
    (the tail blocks wrap around), returns ``x.dtype``/``x.device``. Half
    precision inputs are computed in fp32 and cast back (avg_pool2d half
    support is device-dependent). ``level < 1`` raises.
    """
    if x.dim() != 4:
        raise ValueError(
            f"haar_lowpass needs a 4D [B, C, H, W] tensor; got {tuple(x.shape)}"
        )
    if not x.is_floating_point():
        raise ValueError(
            f"haar_lowpass needs a float tensor; got dtype {x.dtype}"
        )
    if level < 1:
        raise ValueError(f"level must be >= 1; got {level!r}")

    k = 2 ** level
    h, w = x.shape[-2], x.shape[-1]
    # Half precision: avg_pool2d half/bfloat16 support is device-dependent —
    # compute in fp32 and cast back (the hiflow FFT-split precedent).
    compute_dtype = (
        torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
    )

    pad_h = (k - h % k) % k
    pad_w = (k - w % k) % k
    padded = x
    if pad_h or pad_w:
        # Periodic wrap: a tail block averages real pixels with the wrapped
        # ones from the opposite edge (pywt would mirror instead — the
        # pywt-oracle test therefore compares the interior only).
        padded = F.pad(x, (0, pad_w, 0, pad_h), mode="circular")

    coarse = F.avg_pool2d(padded.to(compute_dtype), kernel_size=k)
    upsampled = coarse.repeat_interleave(k, dim=-2).repeat_interleave(k, dim=-1)
    return upsampled[..., :h, :w].to(x.dtype)


# ---------------------------------------------------------------------------
# Flow sigma schedules + cosine decay (paper §2.2, static time shift)
# ---------------------------------------------------------------------------

_VALID_SCHEDULES = ("disable", "cosine_decay", "cosine_shift", "constant")


def build_flow_sigmas(steps: int, shift: float) -> Tensor:
    """Descending flow sigma schedule: ``steps`` interior sigmas + trailing 0.

    Flow times ``t`` are evenly spaced on ``[1, 1/steps]`` (the reference's
    ``np.linspace(1.0, 1/N, N)``, pipeline_flux_imax.py:636) and re-shifted
    by the static SNR balance

        sigma(t) = shift * t / (1 + (shift - 1) * t)

    — algebraically identical to ComfyUI's ``ModelSamplingFlux.sigma()``
    (``flux_time_shift(mu, 1.0, t)`` with ``mu = ln(shift)``,
    comfy/model_sampling.py:417,431-432). So no ``model_sampling`` patch is
    needed (plan D3): I-Max owns both schedules and the forward path of a
    CONST flow model never consults the shift.

    ``shift == 1`` is the identity linspace; larger shifts push more steps
    into the high-noise regime. Returns float32, length ``steps + 1``,
    ending in an exact ``0.0``.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps!r}")
    if shift <= 0.0:
        raise ValueError(f"shift must be > 0; got {shift!r}")
    t = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    sigmas = shift * t / (1.0 + (shift - 1.0) * t)
    return torch.cat([sigmas.to(torch.float32), torch.zeros(1)])


def cosine_factor(step_index: int, total_steps: int) -> float:
    """Projected-Flow guidance strength over pass B: 1.0 → ~0.

    Literal port of the reference (pipeline_flux_imax.py:808):
    ``0.5 * (1 + cos(pi * i / N))``. Note the reference form never reaches
    exactly 0 at the last step — ``cosine_factor(N-1, N) = pi^2/(4N^2)``
    (≈ 0.006 at the default N=20) — it is "≈ 0", per the plan's own
    ``# 1.0 → ~0`` contract. ``total_steps == 1`` degenerates to the first
    step (full guidance on the only transition).
    """
    if total_steps < 1:
        raise ValueError(f"total_steps must be >= 1; got {total_steps!r}")
    if not 0 <= step_index < total_steps:
        raise ValueError(
            f"step_index must be in [0, {total_steps}); got {step_index!r}"
        )
    return 0.5 * (1.0 + math.cos(math.pi * step_index / total_steps))


# ---------------------------------------------------------------------------
# Projected Flow in x0 space (plan D6)
# ---------------------------------------------------------------------------

def projected_flow_x0(
    x: Tensor,
    x0: Tensor,
    guidance: Tensor,
    sigma: float,
    cosine: float,
    schedule: str,
    dwt_level: int,
    p_guidance: Tensor | None = None,
) -> Tensor:
    """One Projected-Flow correction of the clean prediction x̂₁ (plan D6).

    The reference corrects the VELOCITY (pipeline_flux_imax.py:806-829):

        fp_v = -(G - x_t) / (t/1000 + 1e-6)
        v'   = v + c * (P(fp_v) - P(v))

    With ``v = (x_t - x̂₁)/σ``, ``v_G = (x_t - G)/σ`` and P linear, the
    ``P(x_t)`` terms cancel and Euler-stepping with ``v'`` is exactly
    stepping with the corrected clean prediction returned here
    (``test_cosine_decay_equals_velocity_space_reference`` pins it):

        cosine_decay : x̂₁' = x̂₁ + c·(P(G) − P(x̂₁))
        cosine_shift : x̂₁' = x̂₁ − c·(x̂₁ − G) − (1−c)·(P(x̂₁) − P(G))
        constant     : x̂₁' = x̂₁ + P(G) − P(x̂₁)
        disable      : x̂₁' = x̂₁

    ``x`` and ``sigma`` are accepted for call-site parity with the
    reference's velocity-space form — they cancel analytically in x0
    space. ``p_guidance`` may carry the once-computed ``P(G)`` (it is
    constant across pass B, so the dual-pass engine computes it a single
    time); when omitted it is derived from ``guidance`` here. Unknown
    schedules raise. Returns ``x̂₁'`` in ``x0.dtype``.
    """
    if schedule not in _VALID_SCHEDULES:
        raise ValueError(
            f"guidance_schedule must be one of {_VALID_SCHEDULES}; "
            f"got {schedule!r}"
        )
    if schedule == "disable":
        return x0
    p_x0 = haar_lowpass(x0, dwt_level)
    p_g = (
        p_guidance
        if p_guidance is not None
        else haar_lowpass(guidance, dwt_level)
    )
    if schedule == "cosine_decay":
        return x0 + cosine * (p_g - p_x0)
    if schedule == "cosine_shift":
        return x0 - cosine * (x0 - guidance) - (1.0 - cosine) * (p_x0 - p_g)
    # "constant" — full-strength pull toward the low-passed guidance.
    return x0 + p_g - p_x0


# ---------------------------------------------------------------------------
# Low-resolution pass size (plan D10)
# ---------------------------------------------------------------------------

def _snap(value: float, multiple: int) -> int:
    """Round HALF-UP to a multiple (Python's round() is banker's)."""
    return max(multiple, int(math.floor(value / multiple + 0.5)) * multiple)


def low_res_size(
    h: int,
    w: int,
    native: int = 1024,
    multiple: int = 16,
    scale: float = 1.0,
) -> tuple[int, int]:
    """Low-resolution pass size: aspect-preserving, area-normalised (D10).

    The reference divides both sides by ``int(scale_factor + 0.5)``
    (pipeline_flux_imax.py:627-628) — a rounded integer factor that
    distorts aspect whenever ``round(s) != s``. The fix keeps the exact
    area scale and snaps each side independently:

        h_low = snap(h * native / sqrt(h*w))      (round half-up)

    so the low pass preserves the target's aspect ratio and lands at
    roughly the NATIVE pixel area. ``scale`` multiplies the AREA
    (each side scales by ``sqrt(scale)``) — ``low_res_scale=0.5`` halves
    the guidance area.

    ``s = sqrt(h*w)/native <= 1`` (target at/below native): there is
    nothing to extrapolate — returns the target size unchanged and logs a
    WARNING (pass A runs at the target resolution). The result never
    exceeds the target size.

    Units: PIXELS (the caller converts latent dims through its VAE
    downscale factor; ``multiple=16`` px = one Flux latent row pair).
    """
    if h < 1 or w < 1:
        raise ValueError(f"low_res_size needs positive h, w; got {(h, w)!r}")
    if native < 1:
        raise ValueError(f"native must be >= 1; got {native!r}")
    if multiple < 1:
        raise ValueError(f"multiple must be >= 1; got {multiple!r}")
    if scale <= 0.0:
        raise ValueError(f"scale must be > 0; got {scale!r}")

    if math.sqrt(h * w) / native <= 1.0:
        logger.warning(
            "I-Max: target %dx%d px is at/below the native %d px — running "
            "the low pass at the target resolution (nothing to extrapolate)",
            w, h, native,
        )
        return h, w

    factor = native * math.sqrt(scale) / math.sqrt(h * w)
    h_low = min(_snap(h * factor, multiple), h)
    w_low = min(_snap(w * factor, multiple), w)
    return h_low, w_low
