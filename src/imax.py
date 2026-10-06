"""
I-Max — tuning-free resolution extrapolation for flow models (engine).

Two-pass resolution extrapolation for rectified-flow T2I models
(I-Max, arXiv 2410.07536): a low-resolution pass produces a fixed guidance
latent; the target-resolution pass then has its clean prediction pulled
toward the low-pass of that guidance with a CFG-shaped term whose strength
decays over the pass (Projected Flow, paper §2.2), while the schedule shift
re-balances the SNR between the two passes.

This module is the pure-torch engine (the ``src/hiflow.py`` contract):
ZERO ComfyUI imports, ``@torch.no_grad()`` everywhere, dependency-injected
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
from dataclasses import dataclass
from typing import Callable

import numpy as np
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
    ComfyUI model_sampling.py:417,431-432). So no ``model_sampling`` patch is
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


# ---------------------------------------------------------------------------
# Dual-pass orchestration (plan Phase 3 / D14)
# ---------------------------------------------------------------------------

# Sigma floor in the Euler velocity (the reference's 1e-6 guard magnitude;
# interior schedule sigmas are >= 1/steps, so this only guards degenerates).
_EPS = 1e-6


@dataclass
class IMaxConfig:
    """I-Max hyperparameters (plan D10/D11/D14 defaults).

    steps_low/steps_high  sampling steps per pass (informational for the
                          node layer — the walk lengths follow the PASSED
                          sigma schedules)
    time_shift_low/high   static flow shifts for the two schedules (the
                          paper-consistent 3.0 / 6.0; the schedules
                          themselves are built by the caller via
                          :func:`build_flow_sigmas`)
    dwt_level             Haar low-pass level of the guidance projection P
    guidance_schedule     one of ``_VALID_SCHEDULES`` (default
                          "cosine_decay" — README + gradio agree on it)
    denoise               pass-B img2img strength (KSampler convention;
                          pass A always starts from pure noise)
    low_res_scale         multiplies the low pass AREA (1.0 = the paper's
                          native-area guidance)
    native_resolution     the model's native training resolution in PIXELS
    pixels_per_latent     latent->pixel conversion for the low-res size
                          derivation (8 for the Flux VAE) — the engine never
                          sees the VAE, so the caller supplies the ratio
    """

    steps_low: int = 30
    steps_high: int = 20
    time_shift_low: float = 3.0
    time_shift_high: float = 6.0
    dwt_level: int = 1
    guidance_schedule: str = "cosine_decay"
    denoise: float = 1.0
    low_res_scale: float = 1.0
    native_resolution: int = 1024
    pixels_per_latent: int = 8


def _truncate_sigmas_for_denoise(sigmas: Tensor, denoise: float) -> Tensor:
    """KSampler img2img truncation of pass B's schedule (plan D14).

    ComfyUI convention (KSampler): with denoise ``d < 1`` the schedule is
    re-derived at ``new_steps = int(steps / d)`` flow times and its LAST
    ``steps`` interior sigmas + a terminal 0 are kept — the walk enters at
    ``sigma_start = sigma(steps / new_steps)`` (≈ the denoise flow time),
    where the pass-B initialization keeps ``(1 - sigma_start)`` of the
    content. The dense grid is interpolated in FLOW-TIME space on the
    source grid's own spacing (``t_i = linspace(1, 1/steps, steps)`` — the
    :func:`build_flow_sigmas` convention), exact where the grids coincide;
    the same approach as ``src/hiflow.py`` ``calculate_full_sigmas``.
    """
    steps = sigmas.numel() - 1
    if not 0.0 < denoise <= 1.0:
        raise ValueError(f"denoise must be in (0, 1]; got {denoise!r}")
    if denoise > 0.9999:
        return sigmas
    new_steps = max(steps + 1, int(steps / denoise))
    if new_steps <= steps:
        return sigmas
    interior = sigmas[:-1].to(torch.float64)
    if interior.numel() < 2:
        return sigmas
    t_src = torch.linspace(1.0, 1.0 / steps, steps, dtype=torch.float64)
    t_new = torch.linspace(1.0, 1.0 / new_steps, new_steps, dtype=torch.float64)
    dense = torch.from_numpy(
        np.interp(t_new.numpy(), t_src.numpy()[::-1], interior.numpy()[::-1])
    )
    return torch.cat([dense[-steps:].to(sigmas.dtype), torch.zeros(1)])


def _validate_schedule(sigmas: Tensor, name: str) -> None:
    if sigmas.dim() != 1 or sigmas.numel() < 2:
        raise ValueError(
            f"{name} must be a 1D schedule with >= 2 entries; "
            f"got shape {tuple(sigmas.shape)}"
        )
    if not bool(torch.all(sigmas[:-1] >= sigmas[1:])):
        raise ValueError(f"{name} must be non-increasing")
    if float(sigmas[-1]) != 0.0:
        raise ValueError(f"{name} must end at an exact 0.0")


@torch.no_grad()
def imax_dual_pass(
    predict_x0_low: Callable[[Tensor, float], Tensor],
    predict_x0_high: Callable[[Tensor, float], Tensor],
    sigmas_low: Tensor,
    sigmas_high: Tensor,
    content_latent: Tensor,
    guidance: Tensor,
    seed: int,
    cfg: IMaxConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
    *,
    process_latent_in: Callable[[Tensor], Tensor] | None = None,
    process_latent_out: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Full I-Max run: low-res pass A, fixed guidance, target-res pass B.

    Parameters
    ----------
    predict_x0_low / predict_x0_high : ``(x_vae, sigma) -> x0_vae``
        Dependency-injected clean predictors (the node layer's adapters;
        the engine NEVER sees a VAE or a model — signature guard test).
        They may be the same adapter bound to different model options.
    sigmas_low / sigmas_high : :func:`build_flow_sigmas` outputs for the
        two passes (descending, terminal 0).
    content_latent : ``[B, C, H, W]`` target-resolution VAE-space latent.
        Shape source for pass B; its CONTENT is the pass-B img2img seed —
        mixed at ``(1 - sigma_start)`` (all-zero latent = pure txt2img,
        full schedule regardless of ``cfg.denoise``). 5D tensors are
        rejected: the node layer bridges 3D-format models to the 4D core.
    guidance : ``[B, C, H, W]`` guidance latent at TARGET resolution —
        the VAE round trip (decode -> bicubic -> encode) is the CALLER's
        job. RAW, not low-passed: the engine computes ``P(G)`` exactly
        once (it is constant across pass B) via :func:`haar_lowpass`.
    seed : one ``torch.Generator`` seeded once; sequential draws produce
        pass A's start noise, then pass B's — same seed, same result.
    cfg : :class:`IMaxConfig`.
    progress_callback : ``(step, total, stage)`` per step; stage is
        ``"low"`` (pass A), ``"roundtrip"`` (once, guidance preparation)
        and ``"high"`` (pass B).
    process_latent_in / process_latent_out : optional latent-format
        conversions. When both are given, the pass-B noising runs in MODEL
        space — ``sigma*eps + (1-sigma)*process_latent_in(content)``
        converted back (the HiFlow v2.12.1 fix: mixing in VAE space
        under-noises by the latent format's scale factor). None (tests /
        identity formats) keeps the VAE-space mix.

    Euler rule (paper Sec. 2): ``v = (x - x0)/sigma``,
    ``x <- x + v * (sigma_next - sigma)``, computed in fp32; the result is
    returned in ``content_latent``'s dtype.
    """
    if content_latent.dim() != 4:
        raise ValueError(
            "imax_dual_pass needs a 4D latent [B, C, H, W]; got "
            f"{tuple(content_latent.shape)}. 3D-format models must be "
            "squeezed to T=1 by the node layer; multi-frame (T>1) input is "
            "not supported."
        )
    if guidance.shape != content_latent.shape:
        raise ValueError(
            f"guidance shape {tuple(guidance.shape)} must match the target "
            f"latent {tuple(content_latent.shape)}"
        )
    if cfg.guidance_schedule not in _VALID_SCHEDULES:
        raise ValueError(
            f"guidance_schedule must be one of {_VALID_SCHEDULES}; "
            f"got {cfg.guidance_schedule!r}"
        )
    if not 0.0 < cfg.denoise <= 1.0:
        raise ValueError(f"denoise must be in (0, 1]; got {cfg.denoise!r}")
    _validate_schedule(sigmas_low, "sigmas_low")
    _validate_schedule(sigmas_high, "sigmas_high")

    device = content_latent.device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    x_target = content_latent.float()
    guidance = guidance.float()

    # ---- Pass A: low-resolution generation from pure noise (D14). -------
    b, c = content_latent.shape[0], content_latent.shape[1]
    h_t, w_t = content_latent.shape[-2], content_latent.shape[-1]
    ppl = int(cfg.pixels_per_latent)
    h_low_px, w_low_px = low_res_size(
        h_t * ppl, w_t * ppl,
        native=cfg.native_resolution, multiple=16, scale=cfg.low_res_scale,
    )
    h_low, w_low = max(1, h_low_px // ppl), max(1, w_low_px // ppl)

    n_low = sigmas_low.numel() - 1
    x = torch.randn(
        (b, c, h_low, w_low), device=device, dtype=torch.float32,
        generator=generator,
    )
    for i in range(n_low):
        sigma = float(sigmas_low[i])
        x0 = predict_x0_low(x, sigma).float()
        v = (x - x0) / max(sigma, _EPS)
        x = x + v * (float(sigmas_low[i + 1]) - sigma)
        if progress_callback is not None:
            progress_callback(i, n_low, "low")

    # ---- Guidance projection: P(G) computed ONCE (plan D6). -------------
    p_guidance = haar_lowpass(guidance, cfg.dwt_level)
    if progress_callback is not None:
        progress_callback(0, 1, "roundtrip")

    # ---- Pass B: target resolution with Projected-Flow guidance. --------
    # An empty content latent is pure txt2img: keep the full schedule
    # (truncating for zeros would waste high-noise steps regenerating
    # nothing — the HiFlow convention).
    content_empty = bool(torch.count_nonzero(x_target) == 0)
    effective_denoise = 1.0 if content_empty else float(cfg.denoise)
    sig_b = _truncate_sigmas_for_denoise(sigmas_high, effective_denoise)

    sigma_start = float(sig_b[0])
    noise = torch.randn(
        x_target.shape, device=device, dtype=torch.float32,
        generator=generator,
    )
    if process_latent_in is not None and process_latent_out is not None:
        noised_model = (
            sigma_start * noise + (1.0 - sigma_start) * process_latent_in(x_target)
        )
        x = process_latent_out(noised_model)
    else:
        # No conversions available (identity formats / tests): the VAE-space
        # mix. NOTE: only exact for scale==1, shift==0 latent formats.
        x = sigma_start * noise + (1.0 - sigma_start) * x_target

    n_high = sig_b.numel() - 1
    for i in range(n_high):
        sigma = float(sig_b[i])
        x0 = predict_x0_high(x, sigma).float()
        cosine = cosine_factor(i, n_high)
        x0 = projected_flow_x0(
            x, x0, guidance, sigma, cosine,
            cfg.guidance_schedule, cfg.dwt_level, p_guidance=p_guidance,
        )
        v = (x - x0) / max(sigma, _EPS)
        x = x + v * (float(sig_b[i + 1]) - sigma)
        if progress_callback is not None:
            progress_callback(i, n_high, "high")

    return x.to(content_latent.dtype)
