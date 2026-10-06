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
