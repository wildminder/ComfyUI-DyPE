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
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F

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
    below — plus a trailing 0 so the walk lands on a clean sample. Stage
    sigmas MUST stay a subset of the base schedule: the reference lookups
    are exact-timestep keyed, and TrajectoryDict.nearest only tolerates
    small spacing drift.

    Rules:
      - the entry sigma must satisfy sigma_entry <= tau + 1e-6 (clamped to the
        largest schedule sigma below tau; if tau exceeds sigma_max the full
        schedule runs and a warning is logged);
      - the tail is the LAST (steps-1) interior sigmas strictly below the
        entry ("suffix-below-tau": when tau picks an early entry and the
        tail is capped, one enlarged first transition appears — accepted;
        the reference avoids it only because its entry IS schedule[-n]);
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


def denoise_sigmas(
    full_sigmas: torch.Tensor,
    denoise: float,
    steps: int,
) -> torch.Tensor:
    """Truncate the base schedule for img2img (denoise < 1).

    ComfyUI KSampler convention (samplers.py set_steps): with denoise d,
    compute ``new_steps = int(steps / d)`` sigmas and keep the LAST
    ``(steps + 1)`` of them. The stage then enters at sigma_start < 1, where
    the cascade's CONST noising ``sigma_start*eps + (1-sigma_start)*latent``
    keeps ``(1-sigma_start)`` of the content latent — at denoise=1 the
    schedule starts at sigma 1 and the content weight is 0 (from-noise
    generation, the reference pipeline's behavior with a random start).

    An EMPTY content latent (all zeros — EmptySD3LatentImage) always keeps
    the full schedule: truncating for zeros would waste high-noise steps
    re-generating nothing. Returns ``(sigmas, truncated)``.
    """
    if not (0.0 < denoise <= 1.0):
        raise ValueError(f"denoise must be in (0, 1]; got {denoise!r}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1; got {steps!r}")
    if denoise > 0.9999:
        return full_sigmas, False

    new_steps = max(steps + 1, int(steps / denoise))
    if new_steps <= steps:
        return full_sigmas, False

    full = calculate_full_sigmas(full_sigmas, new_steps, steps)
    if full is None:
        return full_sigmas, False
    return full, True


def calculate_full_sigmas(
    full_sigmas: torch.Tensor,
    new_steps: int,
    keep: int,
) -> torch.Tensor | None:
    """Re-derive a denser schedule and keep its last ``keep + 1`` sigmas.

    The model's sigma table is not available here (the node passes one
    concrete schedule), so the denser schedule is interpolated in FLOW-TIME
    space: sigma(t) is a monotone reparameterization of t, and the "simple"
    spacing the node uses is even in t. Interpolating on the given grid
    preserves the entry/exit behavior of ComfyUI's own denser-table slice
    (entry sigma lower than the steps-schedule's, same spacing class).
    Returns None when the source schedule is too short to interpolate
    (fewer than 2 interior sigmas).
    """
    interior = full_sigmas[full_sigmas > 0]
    if interior.numel() < 2:
        return None
    t = torch.linspace(1.0, 0.0, interior.numel(), dtype=torch.float64)
    t_new = torch.linspace(1.0, 0.0, new_steps, dtype=torch.float64)
    dense = torch.from_numpy(
        _interp_flow_time(t, interior.to(torch.float64), t_new)
    )
    # dense covers flow-times 1 -> 0 at new_steps points: drop the last
    # (t=0, sigma 0) and re-append an exact 0 so the kept slice has exactly
    # `keep` interior sigmas + the trailing 0 (KSampler's sigmas[-(steps+1):]).
    kept = dense[-(keep + 1):-1] if new_steps > keep else dense[:-1]
    return torch.cat(
        [kept.to(full_sigmas.dtype),
         torch.zeros(1, dtype=full_sigmas.dtype)]
    )


def _interp_flow_time(t_old, sigmas_old, t_new):
    """Monotone-in-sigma interpolation of sigma(t) at new flow times."""
    t_old_np = t_old.numpy()
    s_old_np = sigmas_old.numpy()
    t_new_np = t_new.numpy()
    return np.interp(t_new_np, t_old_np[::-1], s_old_np[::-1])


def alignment_scales(    stage_sigmas: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-step direction (alpha) and acceleration (beta) scales.

    Reference form (flux_pipeline_hiflow.py lines 903-904): LINEAR in the
    step index — ``(n - i) / n`` per transition, 1.0 at the stage entry
    decaying to 1/n at the last step — NOT the paper's alpha_t = t/tau =
    sigma_i/sigma_entry. The two coincide only when the schedule is evenly
    spaced in sigma; ComfyUI's "simple" spacing on a shifted model table
    (Z-Image: shift 3.0) is not, and the sigma-ratio form stays near 1.0
    mid-walk — over-locking low frequencies late (the Z-Image blur report,
    plan D6). A trailing 0 keeps length parity with the sigma schedule.
    """
    sigmas = stage_sigmas.float()
    entry = float(sigmas[0])
    if entry <= 0.0:
        raise ValueError("stage schedule must enter at a positive sigma")
    n = sigmas.numel() - 1
    if n < 1:
        raise ValueError("stage schedule needs at least one transition")
    idx = torch.arange(n, dtype=sigmas.dtype, device=sigmas.device)
    scales = (float(n) - idx) / float(n)
    scales = torch.cat(
        [scales, torch.zeros(1, dtype=sigmas.dtype, device=sigmas.device)]
    )
    return scales, scales.clone()


# ---------------------------------------------------------------------------
# Reference trajectory storage + base (Stage A) sampling
# ---------------------------------------------------------------------------

def _sigma_key(sigma: float) -> float:
    """Round a sigma to a stable dict key (float equality is fragile)."""
    return round(float(sigma), 6)


class TrajectoryDict:
    """sigma -> predicted-clean-x0 dictionary, the HiFlow reference flow.

    Stores every per-step clean prediction from a sampling run keyed by its
    sigma (6-decimal rounded). Tensors are parked on CPU to keep a 30-step
    4K trajectory (~1.5 GB fp32) off the GPU (plan D8); ``get``/``nearest``
    move them back to the caller's device.

    Time matching (plan D8): exact sigma first, then the nearest stored sigma
    within ``tol`` — robust across differently-spaced schedules, unlike the
    reference's exact-timestep lookup.
    """

    def __init__(self) -> None:
        self._entries: dict[float, torch.Tensor] = {}

    def put(self, sigma: float, x0: torch.Tensor) -> None:
        self._entries[_sigma_key(sigma)] = x0.detach().to("cpu")

    def get(self, sigma: float) -> torch.Tensor:
        key = _sigma_key(sigma)
        if key not in self._entries:
            raise KeyError(
                f"no trajectory entry at sigma={sigma!r}; stored sigmas: "
                f"{sorted(self._entries)[:8]}{'...' if len(self._entries) > 8 else ''}"
            )
        return self._entries[key]

    def nearest(
        self,
        sigma: float,
        tol: float = 1e-3,
        device: torch.device | None = None,
    ) -> tuple[float, torch.Tensor]:
        """Return (stored_sigma, x0) closest to ``sigma`` within ``tol``."""
        if not self._entries:
            raise ValueError(
                "no reference trajectory — run the base stage first"
            )
        keys = sorted(self._entries)
        target = float(sigma)
        best = min(keys, key=lambda k: abs(k - target))
        if abs(best - target) > tol:
            raise ValueError(
                f"no reference entry within {tol} of sigma={target!r} "
                f"(nearest stored: {best!r})"
            )
        x0 = self._entries[best]
        if device is not None:
            x0 = x0.to(device)
        return best, x0

    def sigmas(self) -> list[float]:
        return sorted(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def as_dict(self) -> dict[float, torch.Tensor]:
        return dict(self._entries)


@dataclass
class HiFlowConfig:
    """HiFlow hyperparameters (paper values; plan D7).

    tau            stage-entry noise level; paper cascade [0.6, 0.3, 0.3]
    steps          base-stage sampling steps (paper: 30)
    steps_per_stage guided-stage transitions per cascade stage (repo: 16/10)
    cfg            base-stage classifier-free guidance (FLUX-dev: 3.5)
    guidance_high  guided-stage CFG (repo: 4.5-6)
    filter_ratio   normalized Butterworth cutoff D (paper 0.4 / repo 0.2)
    alpha_scale    direction-alignment multiplier (repo first stage 1.0)
    beta_scale     acceleration-alignment multiplier (repo 0.5)
    upsampling     "latent" (bicubic on latents, repo default) | "pixel"
    eps_sigma      sigma floor in (x - x0)/sigma (reference uses 1e-6)
    """

    tau: float = 0.6
    steps: int = 30
    steps_per_stage: int = 16
    cfg: float = 3.5
    guidance_high: float = 4.5
    filter_ratio: float = 0.2
    alpha_scale: float = 1.0
    beta_scale: float = 0.5
    upsampling: str = "latent"
    eps_sigma: float = 1e-6


@torch.no_grad()
def base_trajectory(
    initial_latent: Tensor,
    sigmas: torch.Tensor,
    predict_x0: Callable[[Tensor, float], Tensor],
    cfg: HiFlowConfig,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> tuple[Tensor, TrajectoryDict]:
    """Stage A — ordinary rectified-flow sampling that records every step.

    Euler rule (paper Sec. 1): ``v = (x_t - x0_pred) / t``,
    ``x_{t-1} = x_t + v * (t_{t-1} - t)`` with the velocity derived from the
    predicted clean sample (``sigma`` clamped at ``cfg.eps_sigma`` — the
    reference divides by ``sigma + 1e-6``). The final sample is recorded at
    sigma 0 as the endpoint prediction, mirroring the reference's
    ``clean_predictions[0] = X``.

    Runs in fp32 regardless of the input dtype; returns the final latent in
    the input's dtype and the per-step trajectory (VAE space throughout —
    the predict_x0 adapter owns model-space conversions).
    """
    x = initial_latent.float()
    traj = TrajectoryDict()
    n_transitions = sigmas.numel() - 1

    for i in range(n_transitions):
        sigma = float(sigmas[i])
        x0 = predict_x0(x, sigma).float()
        traj.put(sigma, x0)

        sigma_safe = max(sigma, cfg.eps_sigma)
        v = (x - x0) / sigma_safe
        x = x + v * (float(sigmas[i + 1]) - sigma)

        if progress_callback is not None:
            progress_callback(i, n_transitions, -1)

    # Endpoint: the last state IS the sigma-0 clean prediction.
    traj.put(0.0, x)
    return x.to(initial_latent.dtype), traj


# ---------------------------------------------------------------------------
# Stage B — guided high-resolution sampling (three alignments)
# ---------------------------------------------------------------------------

@torch.no_grad()
def guided_stage(
    stage_latent: Tensor,
    init_anchor: Tensor,
    ref_traj: TrajectoryDict,
    stage_sigmas: torch.Tensor,
    predict_x0: Callable[[Tensor, float], Tensor],
    upsample_x0: Callable[[Tensor], Tensor],
    cfg: HiFlowConfig,
    freq_filter_factory: Callable[[tuple[int, ...]], Tensor] | None = None,
    progress_callback: Callable[[int, int, int], None] | None = None,
    stage_index: int = 0,
    generator: torch.Generator | None = None,
    process_latent_in: Callable[[Tensor], Tensor] | None = None,
    process_latent_out: Callable[[Tensor], Tensor] | None = None,
) -> tuple[Tensor, TrajectoryDict]:
    """One guided upscale stage: initialization + direction + acceleration.

    Parameters
    ----------
    stage_latent : Tensor
        VAE-space seed latent at the stage's TARGET size (used only for
        shape/device/dtype — its content is replaced by the initialization).
    init_anchor : Tensor
        Pre-upsampled initialization anchor at the stage's TARGET size in
        VAE space: the previous chain's FINAL latent (base output for the
        first stage, the previous stage's sigma-0 endpoint after that),
        pixel round-tripped (decode -> bicubic -> sharpen -> encode) by the
        caller — the reference pipeline always anchors on the final image
        regardless of upsampling_choice (plan D2).
    ref_traj : TrajectoryDict
        Reference flow (previous stage's or the base trajectory), RAW
        (uncorrected) x0 entries at the PREVIOUS (smaller) latent size.
    stage_sigmas : torch.Tensor
        Descending stage schedule from `build_stage_sigmas`; entry sigma is
        the effective tau.
    predict_x0 : callable
        ``(x_stage, sigma) -> x0`` at the stage size (guidance_high CFG) —
        the high-resolution model prediction.
    upsample_x0 : callable
        ``x0_prev_size -> x0_stage_size`` (VAE space; latent bicubic or the
        pixel decode/encode round trip, decided by the caller) — used ONLY
        for the per-step time-matched reference x0's.
    cfg : HiFlowConfig
    freq_filter_factory : callable, optional
        ``(shape) -> [H, W] Butterworth mask``; defaults to building one from
        ``cfg.filter_ratio`` at the stage latent shape. Injectable for tests.
    progress_callback : callable, optional
        ``cb(step_index, total_steps, stage_index)`` per guided step.
    stage_index : int
        Cascade index (0-based), forwarded to the progress callback.
    generator : torch.Generator, optional
        Drives the initialization noise. When None the global RNG is used
        (production); tests pass an explicit generator so the seed is
        reproducible regardless of RNG-stream leftovers.

    Returns
    -------
    (final_latent, stage_traj) : the walked latent and this stage's own
    per-sigma RAW-x0 trajectory (the NEXT stage's reference + its sigma-0
    endpoint is the next stage's anchor — both follow the reference code,
    plan D4).

    Alignment math (paper Sec. 4-6; plan D2-D4 for the code-vs-theory calls):

    - Initialization: ``x = sigma_e * eps + (1 - sigma_e) * init_anchor``
      where sigma_e is the stage entry sigma (== effective tau) and eps is
      fresh std-1 noise. The anchor is the previous chain's FINAL image
      (the reference pipeline's scale_noise on the pixel round-tripped
      final latent), not the time-matched ref[sigma_e] of the theory doc.
    - Direction: ``x0_hat = x0 + alpha_i * (LPF(ref_i) - LPF(x0))`` with
      ``alpha_i = alpha_scale * (n - i) / n`` (linear in step index).
    - Acceleration: ``v_ref = (x - ref_i) / sigma_i`` — the reference
      velocity derived from the HIGH-RES WALK's own current state, exactly
      the reference code's ``model_output_ref = (sample - pred_x0_ref) /
      (sigma + 1e-6)`` (there is no separately-integrated reference chain);
      the high-res velocity gains ``beta_i * beta_scale * (v_ref -
      prev_v_ref - v_hat + prev_v)``. The first guided step has no previous
      pair — acceleration skipped.
    - Recording: the RAW (pre-correction) model x0 per sigma (the reference
      returns original_pred_x0 to pred_x0_dict) — guidance does not
      compound across stages.
    """
    if len(ref_traj) == 0:
        raise ValueError(
            "guided_stage needs a non-empty reference trajectory"
        )

    device = stage_latent.device
    dtype = stage_latent.dtype
    x = stage_latent.float()

    def make_filter(shape: tuple[int, ...]) -> Tensor:
        if freq_filter_factory is not None:
            return freq_filter_factory(shape)
        return butterworth_low_pass_filter_2d(
            shape, device=device, ratio=cfg.filter_ratio,
        )

    filter_mask = make_filter(tuple(x.shape))
    alphas, betas = alignment_scales(stage_sigmas)
    sigma_e = float(stage_sigmas[0])

    # ---- Initialization alignment (plan D2) ------------------------------
    # Anchor already at the stage size; the seed latent only carries
    # shape/device/dtype. NOISING SPACE (v2.12.1): the sigma mix runs in
    # MODEL space (the reference's scheduler.scale_noise operates on its
    # model-scaled latents; ComfyUI's KSAMPLER.sample:993 likewise) — mixing
    # in VAE space under-scales the noise by the latent format's
    # scale_factor. The result converts back to VAE space for the walk.
    anchor = init_anchor.to(device=device, dtype=torch.float32)
    if anchor.shape != x.shape:
        raise ValueError(
            f"init_anchor shape {tuple(anchor.shape)} does not match the "
            f"stage latent {tuple(x.shape)}"
        )
    eps = torch.randn(
        x.shape, device=device, dtype=torch.float32, generator=generator,
    )
    if process_latent_in is not None and process_latent_out is not None:
        anchor_model = process_latent_in(anchor)
        x = process_latent_out(
            sigma_e * eps + (1.0 - sigma_e) * anchor_model
        )
    else:
        x = sigma_e * eps + (1.0 - sigma_e) * anchor

    ref_up: dict[float, Tensor] = {}

    prev_v_high: Tensor | None = None
    prev_v_ref: Tensor | None = None
    traj = TrajectoryDict()
    n_steps = stage_sigmas.numel() - 1

    for i in range(n_steps):
        sigma = float(stage_sigmas[i])
        sigma_safe = max(sigma, cfg.eps_sigma)

        # A. High-resolution model prediction (RAW — recorded before any
        # correction, plan D4: the reference's original_pred_x0).
        x0_high = predict_x0(x, sigma).float()
        traj.put(sigma, x0_high)

        # B. Direction alignment: nudge low frequencies toward the reference.
        ref_x0 = ref_up.get(_sigma_key(sigma))
        if ref_x0 is None:
            _, ref_raw = ref_traj.nearest(sigma, device=device)
            ref_x0 = upsample_x0(ref_raw.float()).to(device)
            ref_up[_sigma_key(sigma)] = ref_x0
        alpha = float(alphas[i]) * cfg.alpha_scale
        if alpha > 0.0:
            low_ref = split_frequency_components_fft(ref_x0, filter_mask, is_low=True)
            low_high = split_frequency_components_fft(x0_high, filter_mask, is_low=True)
            x0_high = x0_high + alpha * (low_ref - low_high)

        # C. Velocities — both from the walk's own current state x (plan D3).
        v_high = (x - x0_high) / sigma_safe
        v_ref = (x - ref_x0) / sigma_safe

        # Acceleration alignment (skipped on the first guided step).
        beta = float(betas[i]) * cfg.beta_scale
        if prev_v_high is not None and prev_v_ref is not None and beta > 0.0:
            v_high = v_high + beta * (
                v_ref - prev_v_ref - v_high + prev_v_high
            )
        prev_v_high = v_high
        prev_v_ref = v_ref

        # D. Euler update.
        dt = float(stage_sigmas[i + 1]) - sigma
        x = x + v_high * dt

        if progress_callback is not None:
            progress_callback(i, n_steps, stage_index)

    traj.put(0.0, x)
    return x.to(dtype), traj


# ---------------------------------------------------------------------------
# Latent upsampling + cascade driver
# ---------------------------------------------------------------------------

def upsample_latent(x0: Tensor, target_h: int, target_w: int) -> Tensor:
    """Antialiased bicubic upscale of a VAE-space x0 to (target_h, target_w).

    fp32 round trip for half dtypes (antialias is not implemented for fp16 —
    the PixelRush precedent).
    """
    orig_dtype = x0.dtype
    up = F.interpolate(
        x0.float(), size=(target_h, target_w),
        mode="bicubic", align_corners=False, antialias=True,
    )
    return up.to(orig_dtype)


def _snap_latent_dim(dim: int, vae_downscale: int, latent_multiple: int) -> int:
    """Snap a latent dim so pixels stay multiples of 16 and the latent dim a
    multiple of ``latent_multiple`` (FLUX packs 2x2 latent patches -> even
    latent dims). Rounds to the NEAREST valid value (up or down)."""
    px = dim * vae_downscale
    px = max(vae_downscale * latent_multiple, round(px / 16) * 16)
    snapped = px // vae_downscale
    # keep the latent dim a multiple of latent_multiple (round UP)
    return ((snapped + latent_multiple - 1) // latent_multiple) * latent_multiple


def _stage_latent_sizes(
    base_h: int, base_w: int, scale_factor: float, vae_downscale: int = 8,
    latent_multiple: int = 2,
) -> list[tuple[int, int]]:
    """Cascade stage sizes (latent H, W) for a SCALE FACTOR relative to the
    base latent (v2.13.0 — replaces the absolute target-resolution form).

    - ``scale > 1``: 2x doubling stages until the FINAL stage reaches the
      scaled size (the last stage may land slightly ABOVE the exact scale —
      doubling is quantized; the reference cascade is 2x stages only);
    - ``scale == 1``: no stages (the base output is returned unchanged);
    - ``0.25 <= scale < 1``: a SINGLE fractional stage walking toward the
      scaled-down size (bilinear ref upsampling then blends toward it).
    """
    if not (0.25 <= scale_factor <= 8.0):
        raise ValueError(
            f"scale_factor must be in [0.25, 8]; got {scale_factor!r}"
        )
    scale = float(scale_factor)

    def snap(dim: int) -> int:
        return _snap_latent_dim(dim, vae_downscale, latent_multiple)

    h0, w0 = snap(base_h), snap(base_w)
    if abs(scale - 1.0) < 1e-9:
        return []

    target_h = snap(round(h0 * scale))
    target_w = snap(round(w0 * scale))

    if scale < 1.0:
        # Fractional stage: one walk toward the smaller size. The stage is
        # guided exactly like an upscale stage (init anchor + per-step refs
        # both upsample FROM the smaller target TO the previous size would
        # invert roles, so the stage runs at the TARGET size with the
        # reference trajectory UPsampled to it — scale<1 simply means the
        # target is smaller; the reference chain is bicubic-downscaled).
        if target_h >= h0 and target_w >= w0:
            return []  # snapping ate the reduction — nothing to do
        return [(target_h, target_w)]

    sizes = []
    h, w = h0, w0
    while h < target_h or w < target_w:
        h, w = snap(h * 2), snap(w * 2)
        sizes.append((h, w))
        if len(sizes) > 4:
            raise ValueError(
                f"scale_factor {scale} needs more than 4 doubling stages"
            )
    return sizes


@torch.no_grad()
def hiflow_cascade(
    initial_latent: Tensor,
    base_sigmas: torch.Tensor,
    predict_x0_base: Callable[[Tensor, float], Tensor],
    predict_x0_stage: Callable[[Tensor, float], Tensor],
    scale_factor: float,
    cfg: HiFlowConfig,
    vae_decode: Callable[[Tensor], Tensor],
    vae_encode: Callable[[Tensor], Tensor],
    sharpen: Callable[[Tensor], Tensor] | None = None,
    vae_downscale: int = 8,
    progress_callback: Callable[[int, int, int], None] | None = None,
    noise_seed: int | None = None,
    denoise: float = 1.0,
    process_latent_in: Callable[[Tensor], Tensor] | None = None,
    process_latent_out: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Full HiFlow cascade: base trajectory, then guided stages.

    ``scale_factor`` (v2.13.0) is RELATIVE to the input latent: 2 doubles
    the resolution per side, 1 returns the base output unchanged, 0.5
    halves it (a single fractional stage — the reference trajectory is
    bicubic-resized to the smaller target; the stage itself walks exactly
    like an upscale stage). Doubling stages are quantized: scales between
    1 and 2 run ONE 2x stage (the paper's cascade is 2x stages only), so
    the final size may land above the exact scale.

    ``denoise`` (img2img, KSampler convention): 1.0 runs the full schedule
    from pure noise (sigma_start == 1 — the reference's randn start; an
    empty latent + noise_seed reproduces the reference pipeline). Values
    below 1 truncate the base schedule so sigma_start < 1 and the content
    latent survives with weight ``(1 - sigma_start)`` — a sampler latent
    connected here is upsampled faithfully. An EMPTY content latent
    (all zeros) always keeps the full schedule regardless of denoise
    (truncating for zeros would waste high-noise steps regenerating
    nothing).

    Each guided stage consumes the PREVIOUS stage's RAW-x0 trajectory
    (upsampled per-step: latent bicubic or the pixel round trip per
    ``cfg.upsampling``) and anchors its initialization on the previous
    chain's FINAL latent, ALWAYS pixel round tripped (decode -> bicubic ->
    sharpen -> encode) — the reference anchors on the final image
    regardless of upsampling_choice (plan D2).

    ``noise_seed`` seeds one generator that drives the base-start noise and
    every stage's initialization noise; None uses the global RNG (the
    reference passes a single generator through the whole pipeline).

    Returns the final VAE-space latent at the last stage's size (or the
    base latent when the input is already at the target).
    """
    if cfg.upsampling not in ("latent", "pixel"):
        raise ValueError(
            f"upsampling must be 'latent' or 'pixel'; got {cfg.upsampling!r}"
        )
    # The core works on 4D latents [B, C, H, W]; the node layer owns the
    # 5D<->4D bridging for 3D-format models (Krea2/Qwen-Image Wan21, the
    # PixelRush convention). Multi-frame (T>1) input is unsupported — the
    # paper's frequency alignment is 2D per-frame.
    if initial_latent.dim() != 4:
        shape = tuple(initial_latent.shape)
        raise ValueError(
            f"hiflow_cascade needs a 4D latent [B, C, H, W]; got {shape}. "
            f"3D-format models must be squeezed to T=1 by the node layer; "
            f"multi-frame (T>1) input is not supported."
        )

    device = initial_latent.device
    generator = None
    if noise_seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(noise_seed))

    # ---- Img2img: truncate the base schedule for a CONTENT latent. -------
    # KSampler convention: denoise < 1 keeps the last (steps+1) of a denser
    # schedule, entering at sigma_start < 1 where the noising below keeps
    # (1 - sigma_start) of the content. An empty latent skips truncation —
    # zeros carry no content to preserve.
    content_empty = bool(torch.count_nonzero(initial_latent) == 0)
    effective_denoise = 1.0 if content_empty else float(denoise)
    if effective_denoise < 1.0:
        base_sigmas, truncated = denoise_sigmas(
            base_sigmas, effective_denoise, cfg.steps,
        )
        if truncated:
            logger.info(
                "HiFlow: denoise %.2f — base schedule truncated, entry "
                "sigma %.4f (content weight %.1f%%)",
                effective_denoise, float(base_sigmas[0]),
                100.0 * (1.0 - float(base_sigmas[0])),
            )

    # ---- Stage A: base trajectory at native size, from a noised start. -----
    # NOISING SPACE (Z-Image round-3 fix, v2.12.1): ComfyUI noises in MODEL
    # space — samplers.py:1223 converts the content with process_latent_in
    # BEFORE KSAMPLER.sample:993 mixes sigma*eps + (1-sigma)*content. Mixing
    # in VAE space and converting the mixture scales the noise by the latent
    # format's scale_factor (Flux/Z-Image: 0.3611 — 2.77x under-noised) and
    # adds spurious shift offsets: the model then "corrects" an input that
    # looks far noisier than the sigma claims — the drastic img2img changes
    # at any usable denoise. Mix in model space, convert the result back.
    sigma_start = float(base_sigmas[0])
    start_noise = torch.randn(
        initial_latent.shape, device=device, dtype=torch.float32,
        generator=generator,
    )
    if process_latent_in is not None and process_latent_out is not None:
        content = process_latent_in(initial_latent.float())
        noised_model = sigma_start * start_noise + (1.0 - sigma_start) * content
        noised_start = process_latent_out(noised_model)
    else:
        # No conversions available (identity formats / tests): the VAE-space
        # mix. NOTE: this is only exact for scale==1, shift==0 formats.
        noised_start = (
            sigma_start * start_noise
            + (1.0 - sigma_start) * initial_latent.float()
        )
    noised_start = noised_start.to(initial_latent.dtype)
    final_base, ref_traj = base_trajectory(
        noised_start, base_sigmas, predict_x0_base, cfg,
        progress_callback=progress_callback,
    )

    base_h, base_w = initial_latent.shape[-2], initial_latent.shape[-1]
    sizes = _stage_latent_sizes(
        base_h, base_w, float(scale_factor), vae_downscale,
    )
    if not sizes:
        logger.info(
            "HiFlow: scale %.2f needs no stages (base %dx%d) — returning "
            "base output", scale_factor, base_h * vae_downscale,
            base_w * vae_downscale,
        )
        return final_base

    if vae_decode is None or vae_encode is None:
        raise ValueError(
            "hiflow_cascade needs vae_decode and vae_encode adapters — the "
            "initialization anchor is always the pixel round-tripped final "
            "latent, in both upsampling modes (plan D2)"
        )

    def make_ref_upsample(target_h: int, target_w: int) -> Callable[[Tensor], Tensor]:
        if cfg.upsampling == "latent":
            return lambda x: upsample_latent(x, target_h, target_w)

        def pixel_up(x: Tensor) -> Tensor:
            image = vae_decode(x)
            # The ComfyUI VAE boundary is channels-LAST ([B, H, W, 3],
            # Z-Image bugfix 2026-09-03); F.interpolate with size= assumes
            # channels-first and would resize the W and C axes — convert
            # around the bicubic.
            channels_last = image.dim() == 4 and image.shape[-1] == 3
            if channels_last:
                image = image.movedim(-1, 1)
            image_up = F.interpolate(
                image.float(), size=(
                    target_h * vae_downscale, target_w * vae_downscale),
                mode="bicubic", align_corners=False, antialias=True,
            ).to(image.dtype)
            if channels_last:
                image_up = image_up.movedim(1, -1)
            if sharpen is not None:
                image_up = sharpen(image_up)
            return vae_encode(image_up)

        return pixel_up

    def make_anchor_upsample(target_h: int, target_w: int) -> Callable[[Tensor], Tensor]:
        """ALWAYS the pixel round trip (decode -> bicubic -> sharpen ->
        encode) — the reference's initialization anchor path, regardless of
        cfg.upsampling (plan D2)."""
        def anchor_up(x: Tensor) -> Tensor:
            image = vae_decode(x)
            channels_last = image.dim() == 4 and image.shape[-1] == 3
            if channels_last:
                image = image.movedim(-1, 1)
            image_up = F.interpolate(
                image.float(), size=(
                    target_h * vae_downscale, target_w * vae_downscale),
                mode="bicubic", align_corners=False, antialias=True,
            ).to(image.dtype)
            if channels_last:
                image_up = image_up.movedim(1, -1)
            if sharpen is not None:
                image_up = sharpen(image_up)
            return vae_encode(image_up)

        return anchor_up

    x = final_base
    for stage_idx, (t_h, t_w) in enumerate(sizes):
        stage_sigmas = build_stage_sigmas(
            base_sigmas, cfg.tau, cfg.steps_per_stage,
        )
        # Anchor: the previous chain's final latent, pixel round tripped to
        # this stage's size (the previous stage's traj[0.0] == x for k > 0;
        # the base final for stage 0 — x already holds it).
        anchor = make_anchor_upsample(t_h, t_w)(x)
        logger.info(
            "HiFlow: stage %d/%d — latent %dx%d -> %dx%d, entry sigma %.4f",
            stage_idx + 1, len(sizes), x.shape[-2], x.shape[-1], t_h, t_w,
            float(stage_sigmas[0]),
        )
        # Seed built from the CURRENT chain latent's shape (S1: the input
        # latent's shape is the wrong template once stages resize — batch
        # and channels never change, but deriving from x keeps it local).
        seed = torch.zeros(
            x.shape[0], x.shape[1], t_h, t_w,
            device=device, dtype=x.dtype,
        )
        x, ref_traj = guided_stage(
            seed, anchor, ref_traj, stage_sigmas, predict_x0_stage,
            make_ref_upsample(t_h, t_w), cfg,
            progress_callback=progress_callback,
            stage_index=stage_idx,
            generator=generator,
            process_latent_in=process_latent_in,
            process_latent_out=process_latent_out,
        )
    return x
