"""Tests for src/sega.py — SEGA core math functions (Tier 1: pure unit tests)."""
import math
import torch
import pytest

from src.sega import (
    compute_base_mscale,
    compute_spectral_energy_profile,
    compute_axis_spectral_profiles,
    compute_dynamic_spread,
    compute_sega_allocation,
)


# ---------------------------------------------------------------------------
# compute_base_mscale
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeBaseMscale:
    def test_power_res_scale_1_returns_1(self):
        result = compute_base_mscale(1024, 1024, formula="power_res")
        assert abs(result - 1.0) < 1e-6

    def test_power_res_larger_scale_increases(self):
        r1 = compute_base_mscale(2048, 1024, formula="power_res")
        r2 = compute_base_mscale(4096, 1024, formula="power_res")
        assert r2 > r1 > 1.0

    def test_power_res_formula(self):
        # s = 4096/1024 = 4, kappa=0.08 → 4^0.08
        expected = 4.0 ** 0.08
        result = compute_base_mscale(4096, 1024, formula="power_res")
        assert abs(result - expected) < 1e-6

    def test_power_res_coefficient_override(self):
        expected = 4.0 ** 0.15
        result = compute_base_mscale(4096, 1024, formula="power_res", coefficient=0.15)
        assert abs(result - expected) < 1e-6

    def test_log_res_formula(self):
        s = 4.0
        expected = 1.0 + 0.1 * math.log(s)
        result = compute_base_mscale(4096, 1024, formula="log_res")
        assert abs(result - expected) < 1e-6

    def test_log_res_coefficient_override(self):
        s = 4.0
        expected = 1.0 + 0.2 * math.log(s)
        result = compute_base_mscale(4096, 1024, formula="log_res", coefficient=0.2)
        assert abs(result - expected) < 1e-6

    def test_scale_clamped_to_1(self):
        # target < training → s clamped to 1.0
        result = compute_base_mscale(512, 1024, formula="power_res")
        assert abs(result - 1.0) < 1e-6

    def test_invalid_formula_raises(self):
        with pytest.raises(ValueError, match="Unknown base_mscale formula"):
            compute_base_mscale(4096, 1024, formula="invalid")


# ---------------------------------------------------------------------------
# compute_spectral_energy_profile
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeSpectralEnergyProfile:
    def test_output_shape(self):
        hs = torch.randn(1, 64, 16)  # (B, N=64, C=16)
        profile = compute_spectral_energy_profile(hs, height=8, width=8, n_bins=16)
        assert profile.shape == (16,)

    def test_non_negative_values(self):
        hs = torch.randn(1, 64, 16)
        profile = compute_spectral_energy_profile(hs, height=8, width=8, n_bins=8)
        assert (profile >= 0).all()

    def test_noise_gives_flat_profile(self):
        """Random noise should produce a relatively flat spectrum."""
        torch.manual_seed(42)
        hs = torch.randn(1, 256, 16)
        profile = compute_spectral_energy_profile(hs, height=16, width=16, n_bins=8)
        # Flatness = geo_mean / arith_mean should be close to 1 for noise
        eps = 1e-8
        flatness = torch.exp(torch.log(profile.clamp(min=eps)).mean()) / (profile.mean() + eps)
        assert flatness.item() > 0.5  # noise is fairly flat

    def test_structured_gives_peaked_profile(self):
        """A structured signal (low-frequency sine) should concentrate energy."""
        H, W = 16, 16
        y = torch.arange(H).float().unsqueeze(1).expand(H, W)
        x = torch.arange(W).float().unsqueeze(0).expand(H, W)
        # Low-frequency sine wave
        spatial = torch.sin(2 * math.pi * y / H).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)
        profile = compute_spectral_energy_profile(spatial, height=H, width=W, n_bins=8)
        # Energy should be concentrated in low-frequency bins
        assert profile[0] > profile[-1]

    def test_4d_input(self):
        hs = torch.randn(1, 8, 8, 16)  # (B, H, W, C)
        profile = compute_spectral_energy_profile(hs, height=8, width=8, n_bins=4)
        assert profile.shape == (4,)

    def test_invalid_dims_raises(self):
        hs = torch.randn(1, 64, 16, 8, 4)  # 5-D
        with pytest.raises(ValueError, match="must be 3-D or 4-D"):
            compute_spectral_energy_profile(hs, height=8, width=8, n_bins=4)


# ---------------------------------------------------------------------------
# compute_axis_spectral_profiles
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeAxisSpectralProfiles:
    def test_output_shapes(self):
        hs = torch.randn(1, 64, 16)
        eh, ew = compute_axis_spectral_profiles(hs, height=8, width=8, n_bins_h=8, n_bins_w=8)
        assert eh.shape == (8,)
        assert ew.shape == (8,)

    def test_non_negative(self):
        hs = torch.randn(1, 64, 16)
        eh, ew = compute_axis_spectral_profiles(hs, height=8, width=8, n_bins_h=4, n_bins_w=4)
        assert (eh >= 0).all()
        assert (ew >= 0).all()

    def test_non_square_different_bins(self):
        hs = torch.randn(1, 128, 16)
        eh, ew = compute_axis_spectral_profiles(hs, height=8, width=16, n_bins_h=4, n_bins_w=8)
        assert eh.shape == (4,)
        assert ew.shape == (8,)

    def test_vertical_structure_concentrates_h(self):
        """A pattern varying along H concentrates energy in H-axis non-DC bins.

        A sine wave along the H dimension (rows) produces frequency content
        in the H-axis 1-D FFT at the fundamental frequency (bin 1), while
        the W-axis FFT is flat (all DC) because the pattern is constant
        along W.
        """
        H, W = 16, 16
        y = torch.arange(H).float().unsqueeze(1).expand(H, W)
        spatial = torch.sin(2 * math.pi * y / H).unsqueeze(0).unsqueeze(-1)
        eh, ew = compute_axis_spectral_profiles(spatial, height=H, width=W, n_bins_h=8, n_bins_w=8)
        # H-axis should have energy in non-DC bins; W-axis should be DC-only
        assert eh[1:].sum() > ew[1:].sum()

    def test_4d_input(self):
        hs = torch.randn(1, 8, 8, 16)
        eh, ew = compute_axis_spectral_profiles(hs, height=8, width=8, n_bins_h=4, n_bins_w=4)
        assert eh.shape == (4,)
        assert ew.shape == (4,)


# ---------------------------------------------------------------------------
# compute_dynamic_spread
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeDynamicSpread:
    def test_flat_spectrum_returns_min(self):
        """Uniform energy → SF=1 → concentration=0 → spread=spread_min."""
        profile = torch.ones(16)
        spread = compute_dynamic_spread(profile, spread_min=0.0, spread_max=1.0, alpha=1.5)
        assert abs(spread - 0.0) < 1e-4

    def test_peaked_spectrum_returns_high(self):
        """Concentrated energy → low SF → high concentration → spread near max."""
        profile = torch.zeros(16)
        profile[0] = 100.0  # all energy in one bin
        spread = compute_dynamic_spread(profile, spread_min=0.0, spread_max=1.0, alpha=1.5)
        assert spread > 0.9

    def test_custom_min_max(self):
        profile = torch.ones(8)
        spread = compute_dynamic_spread(profile, spread_min=0.2, spread_max=0.8, alpha=1.5)
        assert abs(spread - 0.2) < 1e-4

    def test_alpha_controls_nonlinearity(self):
        """Higher alpha → more aggressive spread for moderate concentration."""
        # Moderate concentration: half energy in one bin
        profile = torch.tensor([10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        spread_low_alpha = compute_dynamic_spread(profile, alpha=0.5)
        spread_high_alpha = compute_dynamic_spread(profile, alpha=5.0)
        assert spread_high_alpha >= spread_low_alpha

    def test_returns_float(self):
        profile = torch.ones(8)
        spread = compute_dynamic_spread(profile)
        assert isinstance(spread, float)


# ---------------------------------------------------------------------------
# compute_sega_allocation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComputeSegaAllocation:
    def _make_freqs(self, D_half=28, theta=10000.0):
        """Standard RoPE inverse frequencies."""
        dim = D_half * 2
        dim_indices = torch.arange(0, dim, 2, dtype=torch.float32)
        return 1.0 / (theta ** (dim_indices / dim))

    def test_output_shape(self):
        freqs = self._make_freqs(28)
        energy = torch.ones(16)
        mscale = compute_sega_allocation(energy, freqs, base_mscale=1.2, spread=0.5)
        assert mscale.shape == (28,)

    def test_zero_spread_returns_base_mscale(self):
        """When spread=0, all dims should get base_mscale."""
        freqs = self._make_freqs(28)
        energy = torch.rand(16)
        mscale = compute_sega_allocation(energy, freqs, base_mscale=1.5, spread=0.0)
        assert torch.allclose(mscale, torch.full((28,), 1.5), atol=1e-5)

    def test_zero_alpha_returns_base_mscale(self):
        """When alpha=0, all dims should get base_mscale."""
        freqs = self._make_freqs(28)
        energy = torch.rand(16)
        mscale = compute_sega_allocation(energy, freqs, base_mscale=1.5, spread=0.5, alpha=0.0)
        assert torch.allclose(mscale, torch.full((28,), 1.5), atol=1e-5)

    def test_mscale_clamped_to_min(self):
        """mscale values should never go below min_mscale."""
        freqs = self._make_freqs(28)
        energy = torch.zeros(16)
        energy[0] = 100.0  # extreme concentration
        mscale = compute_sega_allocation(
            energy, freqs, base_mscale=1.0, spread=1.0, alpha=0.5, min_mscale=0.8
        )
        assert (mscale >= 0.8 - 1e-6).all()

    def test_zero_sum_s_d_property(self):
        """The s_d correction (before mscale) should be zero-sum.

        We verify this indirectly: if s_d is zero-sum, then the mean of
        m_d should equal base_mscale (since mean(1 - alpha*spread*s_d) = 1).
        """
        freqs = self._make_freqs(28)
        energy = torch.rand(16) * 10 + 1  # non-uniform
        mscale = compute_sega_allocation(
            energy, freqs, base_mscale=2.0, spread=0.5, alpha=0.15, beta=1.5
        )
        # mean(mscale) ≈ base_mscale because s_d is zero-sum
        assert abs(mscale.mean().item() - 2.0) < 0.01

    def test_uniform_energy_returns_base_mscale(self):
        """Uniform energy → z=0 → s=0 → m=base_mscale."""
        freqs = self._make_freqs(28)
        energy = torch.ones(16) * 5.0
        mscale = compute_sega_allocation(energy, freqs, base_mscale=1.3, spread=0.8)
        assert torch.allclose(mscale, torch.full((28,), 1.3), atol=1e-4)

    def test_non_uniform_energy_varies(self):
        """Non-uniform energy should produce varying mscale values."""
        freqs = self._make_freqs(28)
        energy = torch.zeros(16)
        energy[0] = 100.0
        energy[1:] = 1.0
        mscale = compute_sega_allocation(energy, freqs, base_mscale=1.5, spread=0.8, alpha=0.3)
        assert mscale.std().item() > 1e-4  # not all the same

    def test_higher_spread_more_variation(self):
        """Higher spread → more deviation from base_mscale."""
        freqs = self._make_freqs(28)
        energy = torch.rand(16) * 10 + 1
        mscale_low = compute_sega_allocation(energy, freqs, base_mscale=1.5, spread=0.1)
        mscale_high = compute_sega_allocation(energy, freqs, base_mscale=1.5, spread=1.0)
        assert mscale_high.std().item() > mscale_low.std().item()

    def test_returns_float32(self):
        freqs = self._make_freqs(28)
        energy = torch.ones(16)
        mscale = compute_sega_allocation(energy, freqs, base_mscale=1.0, spread=0.5)
        assert mscale.dtype == torch.float32
