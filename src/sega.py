"""
SEGA (Spectral-Energy Guided Attention) core algorithm.

Implements per-RoPE-dimension mscale computation from the latent's Fourier
spectrum.  All functions are pure (no model dependencies) and fully testable
in isolation.

Reference: SEGA paper (arXiv:2605.22668) and reference implementation in
``.dev/data/SEGA/sega/``.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Base mscale — resolution-dependent reference magnitude
# ---------------------------------------------------------------------------

def compute_base_mscale(
    target_res: int,
    training_res: int,
    formula: str = "power_res",
    coefficient: float | None = None,
) -> float:
    """Compute the shared reference magnitude ``m_ref``.

    .. math::
        m_{\\mathrm{ref}} = (R_{\\mathrm{target}} / R_{\\mathrm{train}})^{\\kappa}

    or the logarithmic alternative:

    .. math::
        m_{\\mathrm{ref}} = 1 + \\kappa \\ln(s)

    Parameters
    ----------
    target_res : int
        Target resolution in pixels (e.g. 4096).
    training_res : int
        Training resolution in pixels (e.g. 1024).
    formula : str
        ``"power_res"`` (paper default) or ``"log_res"``.
    coefficient : float | None
        Override for :math:`\\kappa`.  Defaults: 0.08 for ``power_res``,
        0.1 for ``log_res``.

    Returns
    -------
    float
        ``m_ref >= 1.0``.
    """
    s = max(float(target_res) / float(training_res), 1.0)
    if formula == "power_res":
        c = 0.08 if coefficient is None else float(coefficient)
        return s ** c
    if formula == "log_res":
        c = 0.1 if coefficient is None else float(coefficient)
        return 1.0 + c * math.log(s)
    raise ValueError(f"Unknown base_mscale formula: {formula!r}. Use 'power_res' or 'log_res'.")


# ---------------------------------------------------------------------------
# Spectral helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_spectral_energy_profile(
    hidden_states: torch.Tensor,
    height: int,
    width: int,
    n_bins: int,
) -> torch.Tensor:
    """Radial (isotropic) spectral energy profile ``E_iso``.

    Reshapes the leading ``height * width`` tokens into an ``H x W`` spatial
    map, averages over batch and channels, mean-centres, computes the 2D FFT
    power spectrum, and bins the power into ``n_bins`` concentric rings.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape ``(B, N, C)`` or ``(B, H, W, C)``.  If 3-D, ``N`` must be
        ``>= height * width``.
    height, width : int
        Spatial dimensions of the token grid.
    n_bins : int
        Number of radial frequency bins.

    Returns
    -------
    torch.Tensor
        Shape ``(n_bins,)`` — mean power per radial bin.
    """
    if hidden_states.dim() == 3:
        B, S, C = hidden_states.shape
        n_spatial = min(S, height * width)
        spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    elif hidden_states.dim() == 4:
        spatial = hidden_states
    else:
        raise ValueError(f"hidden_states must be 3-D or 4-D, got {hidden_states.dim()}-D")

    spatial_map = spatial.float().mean(dim=(0, -1))  # (H, W)
    spatial_map = spatial_map - spatial_map.mean()

    power = torch.fft.fftshift(torch.fft.fft2(spatial_map)).abs().pow(2)

    cy, cx = height / 2.0, width / 2.0
    y = torch.arange(height, device=power.device, dtype=torch.float32) - cy
    x = torch.arange(width, device=power.device, dtype=torch.float32) - cx
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    radius_norm = torch.sqrt(yy ** 2 + xx ** 2)
    radius_norm = radius_norm / (radius_norm.max() + 1e-8)

    bin_idx = (radius_norm * n_bins).long().clamp(0, n_bins - 1).flatten()
    flat_pw = power.flatten()

    energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
    energy_sum.scatter_add_(0, bin_idx, flat_pw)
    energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(flat_pw))
    return energy_sum / (energy_cnt + 1e-8)


@torch.no_grad()
def compute_axis_spectral_profiles(
    hidden_states: torch.Tensor,
    height: int,
    width: int,
    n_bins_h: int,
    n_bins_w: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis (height, width) 1-D spectral energy profiles.

    Computes separate 1-D FFT profiles along the height and width axes so
    that horizontal and vertical RoPE dimensions can be adjusted
    independently.

    Parameters
    ----------
    hidden_states : torch.Tensor
        Shape ``(B, N, C)`` or ``(B, H, W, C)``.
    height, width : int
        Spatial dimensions of the token grid.
    n_bins_h, n_bins_w : int
        Number of frequency bins for each axis profile.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(E_H, W)`` — each shape ``(n_bins_*,)``.
    """
    if hidden_states.dim() == 3:
        B, S, C = hidden_states.shape
        n_spatial = min(S, height * width)
        spatial = hidden_states[:, :n_spatial].reshape(B, height, width, C)
    elif hidden_states.dim() == 4:
        spatial = hidden_states
    else:
        raise ValueError(f"hidden_states must be 3-D or 4-D, got {hidden_states.dim()}-D")

    sm = spatial.float().mean(dim=(0, -1))  # (H, W)
    sm = sm - sm.mean()

    def _axis_profile(sm: torch.Tensor, axis: int, n_bins: int, length: int) -> torch.Tensor:
        fft = torch.fft.fft(sm, dim=axis)
        power = fft.abs().pow(2).mean(dim=1 - axis)
        half = length // 2 + 1
        power = power[:half]
        freq_norm = torch.linspace(0.0, 1.0, half, device=power.device)
        bin_idx = (freq_norm * n_bins).long().clamp(0, n_bins - 1)
        energy_sum = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_cnt = torch.zeros(n_bins, device=power.device, dtype=torch.float32)
        energy_sum.scatter_add_(0, bin_idx, power.float())
        energy_cnt.scatter_add_(0, bin_idx, torch.ones_like(power, dtype=torch.float32))
        return energy_sum / (energy_cnt + 1e-8)

    return (
        _axis_profile(sm, axis=0, n_bins=n_bins_h, length=height),
        _axis_profile(sm, axis=1, n_bins=n_bins_w, length=width),
    )


@torch.no_grad()
def compute_dynamic_spread(
    energy_profile: torch.Tensor,
    spread_min: float = 0.0,
    spread_max: float = 1.0,
    alpha: float = 1.5,
) -> float:
    """Spectral-flatness-driven spread in ``[spread_min, spread_max]``.

    .. math::
        \\mathrm{SF} = \\frac{\\exp(\\overline{\\ln E})}{\\overline{E}}
        \\qquad
        \\sigma = s_{\\min} + (s_{\\max} - s_{\\min})\\,(1 - (1 - c)^{\\alpha})

    where ``c = 1 - SF`` is the spectral concentration.

    * Flat (noise-like) spectrum → ``SF ≈ 1`` → ``σ ≈ spread_min``
    * Concentrated (structured) spectrum → ``SF < 1`` → ``σ → spread_max``
    """
    eps = 1e-8
    energy = energy_profile.clamp(min=eps)
    geo_mean = torch.exp(torch.log(energy).mean())
    arith_mean = energy.mean()
    flatness = (geo_mean / (arith_mean + eps)).clamp(0.0, 1.0)
    concentration = 1.0 - flatness.item()
    return spread_min + (spread_max - spread_min) * (1.0 - (1.0 - concentration) ** alpha)


# ---------------------------------------------------------------------------
# SEGA per-dimension mscale allocation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_sega_allocation(
    energy_profile: torch.Tensor,
    freqs: torch.Tensor,
    base_mscale: float,
    spread: float,
    alpha: float = 0.15,
    beta: float = 1.5,
    min_mscale: float = 1.0,
) -> torch.Tensor:
    """Per-dimension RoPE mscale from a spectral energy profile (SEGA).

    .. math::
        z_k = \\mathrm{standardise}(\\log E[\\mathrm{bin}(k)])
        s_k = \\tanh(\\beta \\, z_k) - \\overline{\\tanh(\\beta \\, z_k)}
        m_k = m_{\\mathrm{ref}} \\,(1 - \\alpha \\, \\sigma \\, s_k)

    High spectral-energy dimensions get *lower* ``m_k`` (sharpness-biased),
    while low-energy dimensions get *higher* ``m_k``.

    Parameters
    ----------
    energy_profile : torch.Tensor
        Shape ``(n_bins,)`` — axis-specific spectral energy.
    freqs : torch.Tensor
        Shape ``(D_half,)`` — inverse RoPE frequencies ``1/θ_d`` for the
        axis being scaled.
    base_mscale : float
        Reference magnitude ``m_ref``.
    spread : float
        Dynamic spread ``σ`` from :func:`compute_dynamic_spread`.
    alpha : float
        SEGA amplitude (default 0.15).
    beta : float
        tanh sharpness (default 1.5).
    min_mscale : float
        Floor for per-frequency mscale values (default 1.0).

    Returns
    -------
    torch.Tensor
        Shape ``(D_half,)`` — per-dimension mscale values.
    """
    D_half = freqs.shape[0]
    eps = 1e-8

    # Degenerate cases → uniform base_mscale
    if spread <= 0.0 or alpha <= 0.0:
        return torch.full((D_half,), float(base_mscale), device=freqs.device, dtype=torch.float32)

    # Map each RoPE dim to its FFT bin via log-period
    periods = 2.0 * math.pi / freqs.clamp(min=eps)
    log_periods = torch.log(periods)
    min_lp, max_lp = log_periods.min(), log_periods.max()
    if (max_lp - min_lp).item() > 1e-6:
        lp_norm = (log_periods - min_lp) / (max_lp - min_lp)  # 0=high-freq, 1=low-freq
    else:
        lp_norm = torch.zeros_like(log_periods)

    n_bins = energy_profile.shape[0]
    bin_pos = (1.0 - lp_norm) * (n_bins - 1)
    j_low = bin_pos.floor().long().clamp(0, n_bins - 1)
    j_high = (j_low + 1).clamp(0, n_bins - 1)
    frac = (bin_pos - j_low.to(bin_pos.dtype)).clamp(0.0, 1.0)

    E = energy_profile.to(freqs.device).clamp(min=eps)
    log_E = torch.log(E)
    raw = log_E[j_low] * (1.0 - frac) + log_E[j_high] * frac

    # Standardise + tanh + re-centre (zero-sum property)
    z = raw - raw.mean()
    z = z / z.std().clamp(min=eps)
    s = torch.tanh(float(beta) * z)
    s = s - s.mean()

    # Final per-dim mscale
    m = float(base_mscale) * (1.0 - float(alpha) * float(spread) * s)
    return m.clamp(min=float(min_mscale)).to(torch.float32)
