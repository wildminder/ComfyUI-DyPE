"""
SegAPosEmbed — SEGA-enhanced position embedding base class.

Extends :class:`DyPEBasePosEmbed` to add per-dimension spectral mscale
computation.  SEGA (Spectral-Energy Guided Attention) computes a
per-RoPE-dimension scaling factor from the latent's Fourier spectrum at
each denoising step, then applies it to the cos/sin of the NTK-scaled
RoPE frequencies.

The spectral data (energy profiles, dynamic spread) is set at runtime by
the ComfyUI wrapper before each forward pass via :meth:`set_spectral_data`.
"""

from __future__ import annotations

import torch

from .base import DyPEBasePosEmbed
from .rope import get_1d_ntk_pos_embed
from .sega import compute_base_mscale, compute_sega_allocation


class SegAPosEmbed(DyPEBasePosEmbed):
    """SEGA-enhanced position embedding with per-dim spectral mscale.

    Inherits NTK base frequency scaling from :class:`DyPEBasePosEmbed` and
    adds SEGA's per-dimension mscale on top.  The mscale is computed from
    the latent's Fourier spectrum (set via :meth:`set_spectral_data`) and
    applied to the cos/sin tensors before rotation matrix construction.

    Parameters
    ----------
    theta : int | list[float]
        RoPE base frequency (or per-axis list).
    axes_dim : list[int]
        Per-axis RoPE dimensions.
    method : str
        ``"sega"`` for SEGA (NTK + spectral mscale) or ``"ntk"`` for
        plain NTK (no spectral mscale).
    mscale_alpha : float
        SEGA amplitude (default 0.15).
    mscale_beta : float
        tanh sharpness (default 1.5).
    mscale_min : float
        Floor for per-frequency mscale values (default 1.0).
    spread_min, spread_max : float
        Dynamic spread range (default 0.0–1.0).
    spread_alpha : float
        Non-linear mapping exponent for spread (default 1.5).
    base_mscale_formula : str
        ``"power_res"`` or ``"log_res"`` (default ``"power_res"``).
    base_mscale_coefficient : float | None
        κ coefficient for base mscale (default 0.08 for power_res).
    training_res_pixels : int
        Training resolution in pixels (default 1024).
    **kwargs
        Forwarded to :class:`DyPEBasePosEmbed`.
    """

    def __init__(
        self,
        theta: int | list[float],
        axes_dim: list[int],
        method: str = "sega",
        yarn_alt_scaling: bool = False,
        dype: bool = True,
        dype_scale: float = 2.0,
        dype_exponent: float = 2.0,
        base_resolution: int = 1024,
        dype_start_sigma: float = 1.0,
        base_patch_grid: tuple[int, int] | int | None = None,
        # SEGA-specific parameters
        mscale_alpha: float = 0.15,
        mscale_beta: float = 1.5,
        mscale_min: float = 1.0,
        spread_min: float = 0.0,
        spread_max: float = 1.0,
        spread_alpha: float = 1.5,
        base_mscale_formula: str = "power_res",
        base_mscale_coefficient: float | None = None,
        training_res_pixels: int = 1024,
    ) -> None:
        super().__init__(
            theta=theta,
            axes_dim=axes_dim,
            method=method,
            yarn_alt_scaling=yarn_alt_scaling,
            dype=dype,
            dype_scale=dype_scale,
            dype_exponent=dype_exponent,
            base_resolution=base_resolution,
            dype_start_sigma=dype_start_sigma,
            base_patch_grid=base_patch_grid,
        )
        self.mscale_alpha = mscale_alpha
        self.mscale_beta = mscale_beta
        self.mscale_min = mscale_min
        self.spread_min = spread_min
        self.spread_max = spread_max
        self.spread_alpha = spread_alpha
        self.base_mscale_formula = base_mscale_formula
        self.base_mscale_coefficient = base_mscale_coefficient
        self.training_res_pixels = training_res_pixels

        # Runtime spectral state — set per-step by the wrapper
        self._energy_profile_h: torch.Tensor | None = None
        self._energy_profile_w: torch.Tensor | None = None
        self._dynamic_spread: float = 0.0
        self._target_res_h: int = 0
        self._target_res_w: int = 0

    # ------------------------------------------------------------------
    # Runtime state setters
    # ------------------------------------------------------------------

    def set_spectral_data(
        self,
        energy_profile_h: torch.Tensor | None,
        energy_profile_w: torch.Tensor | None,
        dynamic_spread: float,
        target_res_h: int = 0,
        target_res_w: int = 0,
    ) -> None:
        """Set per-step spectral data (called by the wrapper before forward)."""
        self._energy_profile_h = energy_profile_h
        self._energy_profile_w = energy_profile_w
        self._dynamic_spread = dynamic_spread
        self._target_res_h = target_res_h
        self._target_res_w = target_res_w

    # ------------------------------------------------------------------
    # SEGA component computation
    # ------------------------------------------------------------------

    def _compute_ntk_factor(self, axis_dim: int, scale_global: float) -> float:
        """Compute NTK factor for a given axis dimension and scale.

        Uses the SEGA-style NTK formula: ``s^(d/(d-2))`` (not the DyPE
        time-dependent variant).  SEGA's per-dim mscale handles the
        time-dependent adaptation instead.
        """
        if scale_global <= 1.0:
            return 1.0
        base_ntk = scale_global ** (axis_dim / (axis_dim - 2))
        return max(1.0, base_ntk)

    def _compute_per_dim_mscale(
        self,
        axis_idx: int,
        axis_dim: int,
        ntk_factor: float,
        device: torch.device,
    ) -> torch.Tensor | float:
        """Compute per-dimension SEGA mscale for a spatial axis.

        Returns a tensor of shape ``(axis_dim // 2,)`` if spectral data
        is available, or a scalar ``base_mscale`` float if not.
        """
        # Only spatial axes (idx > 0) get SEGA mscale
        if axis_idx == 0 or ntk_factor <= 1.0:
            return 1.0

        # Determine which energy profile to use
        if axis_idx == 1:
            energy_profile = self._energy_profile_h
            target_res = self._target_res_h
        elif axis_idx == 2:
            energy_profile = self._energy_profile_w
            target_res = self._target_res_w
        else:
            return 1.0

        # Compute base mscale from resolution ratio
        base_ms = compute_base_mscale(
            target_res=target_res if target_res > 0 else self.training_res_pixels * 2,
            training_res=self.training_res_pixels,
            formula=self.base_mscale_formula,
            coefficient=self.base_mscale_coefficient,
        )

        # If no spectral data or spread is zero, return uniform base_mscale
        if energy_profile is None or self._dynamic_spread <= 0.0 or base_ms <= 1.0 + 1e-8:
            return float(base_ms)

        # Compute inverse frequencies for this axis
        axis_theta = self.thetas[axis_idx] if self.thetas is not None else self.theta
        dim_indices = torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device)
        # NTK-scaled theta
        scaled_theta = axis_theta * ntk_factor
        freqs = 1.0 / (scaled_theta ** (dim_indices / axis_dim))

        # Compute per-dim mscale via SEGA allocation
        mscale = compute_sega_allocation(
            energy_profile=energy_profile.to(device),
            freqs=freqs,
            base_mscale=base_ms,
            spread=self._dynamic_spread,
            alpha=self.mscale_alpha,
            beta=self.mscale_beta,
            min_mscale=self.mscale_min,
        )
        return mscale

    def _calc_sega_components(
        self, pos: torch.Tensor, freqs_dtype: torch.dtype
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """SEGA method: NTK base frequencies + per-dim spectral mscale.

        For each axis:
        1. Compute NTK factor from the extrapolation scale.
        2. Compute per-dim SEGA mscale from the spectral energy profile.
        3. Generate cos/sin with NTK-scaled frequencies.
        4. Multiply cos/sin by the per-dim mscale.
        """
        n_axes = pos.shape[-1]
        components = []
        device = pos.device

        # Compute global scale (same logic as _calc_ntk_components)
        if n_axes >= 3:
            h_span = self._axis_token_span(pos[..., 1])
            w_span = self._axis_token_span(pos[..., 2])
            scale_global = max(1.0, max(h_span / self.base_patch_grid[0], w_span / self.base_patch_grid[1]))
        else:
            max_current_patches = self._axis_token_span(pos)
            scale_global = max(1.0, max_current_patches / self.base_patches)

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            axis_theta = self.thetas[i] if self.thetas is not None else self.theta

            # NTK factor for spatial axes
            ntk_factor = 1.0
            if i > 0 and scale_global > 1.0:
                ntk_factor = self._compute_ntk_factor(axis_dim, scale_global)

            # Generate cos/sin with NTK-scaled frequencies
            common_kwargs = {
                "dim": axis_dim,
                "pos": axis_pos,
                "theta": axis_theta,
                "use_real": True,
                "repeat_interleave_real": True,
                "freqs_dtype": freqs_dtype,
            }
            cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=ntk_factor)

            # Apply per-dim SEGA mscale to spatial axes
            if i > 0 and ntk_factor > 1.0:
                mscale = self._compute_per_dim_mscale(i, axis_dim, ntk_factor, device)
                if isinstance(mscale, torch.Tensor):
                    # mscale shape: (D/2,) — repeat_interleave to match (D,)
                    ms_expanded = mscale.repeat_interleave(2)
                    cos = cos * ms_expanded
                    sin = sin * ms_expanded
                elif mscale != 1.0:
                    cos = cos * mscale
                    sin = sin * mscale

            components.append((cos, sin))

        return components

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Dispatch to SEGA or parent method based on ``self.method``."""
        if self.method == "sega":
            return self._calc_sega_components(pos, freqs_dtype)
        # Fall back to parent DyPE methods for non-SEGA methods
        return super().get_components(pos, freqs_dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SegAPosEmbed is a base class. Use a model-specific subclass.")
