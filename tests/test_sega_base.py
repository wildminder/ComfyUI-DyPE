"""Tests for src/sega_base.py — SegAPosEmbed base class (Tier 1: pure unit tests)."""
import pytest
import torch

from src.sega_base import SegAPosEmbed


class ConcreteSegAEmbed(SegAPosEmbed):
    """Minimal concrete subclass for testing (mirrors ConcreteEmbed in test_base.py)."""
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.float32
        components = self.get_components(pos, freqs_dtype)
        return torch.cat([cos for cos, sin in components], dim=-1)


def _make_pos(H=64, W=64, n_axes=3):
    """Create a position tensor for testing."""
    L = H * W
    pos = torch.zeros(1, L, n_axes)
    if n_axes >= 3:
        pos[..., 0] = torch.arange(L)
        pos[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        pos[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    return pos


@pytest.mark.unit
class TestSegAPosEmbedInit:
    def test_default_parameters(self):
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        assert emb.method == "sega"
        assert emb.mscale_alpha == 0.15
        assert emb.mscale_beta == 1.5
        assert emb.mscale_min == 1.0
        assert emb.spread_min == 0.0
        assert emb.spread_max == 1.0
        assert emb.spread_alpha == 1.5
        assert emb.base_mscale_formula == "power_res"
        assert emb.base_mscale_coefficient is None
        assert emb.training_res_pixels == 1024

    def test_custom_parameters(self):
        emb = ConcreteSegAEmbed(
            theta=10000, axes_dim=[16, 56, 56],
            mscale_alpha=0.3, mscale_beta=2.0, mscale_min=0.8,
            spread_min=0.1, spread_max=0.9, spread_alpha=2.0,
            base_mscale_formula="log_res", base_mscale_coefficient=0.15,
            training_res_pixels=512,
        )
        assert emb.mscale_alpha == 0.3
        assert emb.mscale_beta == 2.0
        assert emb.mscale_min == 0.8
        assert emb.spread_min == 0.1
        assert emb.spread_max == 0.9
        assert emb.spread_alpha == 2.0
        assert emb.base_mscale_formula == "log_res"
        assert emb.base_mscale_coefficient == 0.15
        assert emb.training_res_pixels == 512

    def test_inherits_dype_base(self):
        from src.base import DyPEBasePosEmbed
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        assert isinstance(emb, DyPEBasePosEmbed)
        assert emb.base_resolution == 1024
        assert emb.base_patch_grid == (64, 64)

    def test_initial_spectral_state_none(self):
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        assert emb._energy_profile_h is None
        assert emb._energy_profile_w is None
        assert emb._dynamic_spread == 0.0


@pytest.mark.unit
class TestSetSpectralData:
    def test_sets_data(self):
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        eh = torch.ones(16)
        ew = torch.ones(16)
        emb.set_spectral_data(eh, ew, 0.5, 2048, 2048)
        assert torch.equal(emb._energy_profile_h, eh)
        assert torch.equal(emb._energy_profile_w, ew)
        assert emb._dynamic_spread == 0.5
        assert emb._target_res_h == 2048
        assert emb._target_res_w == 2048

    def test_none_energy_profiles(self):
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        emb.set_spectral_data(None, None, 0.0)
        assert emb._energy_profile_h is None
        assert emb._energy_profile_w is None


@pytest.mark.unit
class TestComputeNtkFactor:
    def test_scale_1_returns_1(self):
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        assert emb._compute_ntk_factor(56, 1.0) == 1.0

    def test_scale_2_increases(self):
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        ntk = emb._compute_ntk_factor(56, 2.0)
        assert ntk > 1.0
        # s^(d/(d-2)) = 2^(56/54)
        expected = 2.0 ** (56 / 54)
        assert abs(ntk - expected) < 1e-5

    def test_clamped_to_1(self):
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        assert emb._compute_ntk_factor(56, 0.5) == 1.0


@pytest.mark.unit
class TestComputePerDimMscale:
    def test_axis_0_returns_1(self):
        """Text/sequential axis (idx=0) should not get SEGA mscale."""
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        mscale = emb._compute_per_dim_mscale(0, 16, 2.0, torch.device("cpu"))
        assert mscale == 1.0

    def test_no_ntk_returns_1(self):
        """When ntk_factor <= 1.0, no SEGA mscale."""
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        mscale = emb._compute_per_dim_mscale(1, 56, 1.0, torch.device("cpu"))
        assert mscale == 1.0

    def test_no_spectral_data_returns_base_mscale(self):
        """Without spectral data, returns uniform base_mscale."""
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], training_res_pixels=1024)
        emb.set_spectral_data(None, None, 0.0, target_res_h=4096, target_res_w=4096)
        mscale = emb._compute_per_dim_mscale(1, 56, 2.0, torch.device("cpu"))
        # base_mscale = (4096/1024)^0.08 = 4^0.08
        expected = 4.0 ** 0.08
        assert abs(mscale - expected) < 1e-5

    def test_zero_spread_returns_base_mscale(self):
        """With spread=0, returns uniform base_mscale."""
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], training_res_pixels=1024)
        eh = torch.rand(16)
        emb.set_spectral_data(eh, None, 0.0, target_res_h=4096, target_res_w=4096)
        mscale = emb._compute_per_dim_mscale(1, 56, 2.0, torch.device("cpu"))
        expected = 4.0 ** 0.08
        assert abs(mscale - expected) < 1e-5

    def test_with_spectral_data_returns_tensor(self):
        """With spectral data and spread > 0, returns a per-dim tensor."""
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], training_res_pixels=1024)
        eh = torch.rand(16) * 10 + 1
        emb.set_spectral_data(eh, None, 0.5, target_res_h=4096, target_res_w=4096)
        mscale = emb._compute_per_dim_mscale(1, 56, 2.0, torch.device("cpu"))
        assert isinstance(mscale, torch.Tensor)
        assert mscale.shape == (28,)  # 56 // 2

    def test_mscale_clamped_to_min(self):
        """Per-dim mscale should not go below mscale_min."""
        emb = ConcreteSegAEmbed(
            theta=10000, axes_dim=[16, 56, 56],
            mscale_alpha=0.5, mscale_min=0.9, training_res_pixels=1024,
        )
        # Extreme energy concentration
        eh = torch.zeros(16)
        eh[0] = 1000.0
        emb.set_spectral_data(eh, None, 1.0, target_res_h=4096, target_res_w=4096)
        mscale = emb._compute_per_dim_mscale(1, 56, 2.0, torch.device("cpu"))
        assert isinstance(mscale, torch.Tensor)
        assert (mscale >= 0.9 - 1e-6).all()


@pytest.mark.unit
class TestCalcSegaComponents:
    def test_output_shape(self):
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56])
        pos = _make_pos(64, 64)
        components = emb._calc_sega_components(pos, torch.float32)
        assert len(components) == 3
        assert components[0][0].shape == (1, 4096, 16)
        assert components[1][0].shape == (1, 4096, 56)
        assert components[2][0].shape == (1, 4096, 56)

    def test_no_extrapolation_no_mscale(self):
        """When positions are within base grid, no SEGA mscale applied."""
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], base_resolution=1024)
        pos = _make_pos(32, 32)  # 32 < 64 base patches
        components = emb._calc_sega_components(pos, torch.float32)
        # cos² + sin² should be 1 (no mscale)
        cos, sin = components[1]
        magnitude = cos ** 2 + sin ** 2
        assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-4)

    def test_with_spectral_data_applies_mscale(self):
        """With spectral data and extrapolation, mscale is applied."""
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], training_res_pixels=1024)
        eh = torch.rand(16) * 10 + 1
        ew = torch.rand(16) * 10 + 1
        emb.set_spectral_data(eh, ew, 0.5, target_res_h=4096, target_res_w=4096)
        pos = _make_pos(128, 128)  # 2x extrapolation
        components = emb._calc_sega_components(pos, torch.float32)
        # With mscale, cos² + sin² should NOT all be 1
        cos, sin = components[1]
        magnitude = cos ** 2 + sin ** 2
        assert not torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-3)

    def test_text_axis_not_scaled(self):
        """Axis 0 (text/sequential) should not get SEGA mscale."""
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], training_res_pixels=1024)
        eh = torch.rand(16) * 10 + 1
        emb.set_spectral_data(eh, None, 0.8, target_res_h=4096, target_res_w=4096)
        pos = _make_pos(128, 128)
        components = emb._calc_sega_components(pos, torch.float32)
        # Axis 0 should have unit magnitude (no mscale)
        cos0, sin0 = components[0]
        mag0 = cos0 ** 2 + sin0 ** 2
        assert torch.allclose(mag0, torch.ones_like(mag0), atol=1e-4)


@pytest.mark.unit
class TestGetComponentsDispatch:
    def test_sega_method_dispatches_to_sega(self):
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], method="sega")
        pos = _make_pos(64, 64)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3

    def test_ntk_method_dispatches_to_parent(self):
        """method='ntk' should use parent's _calc_ntk_components."""
        emb = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], method="ntk")
        pos = _make_pos(64, 64)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3

    def test_sega_and_ntk_differ_with_extrapolation(self):
        """With spectral data and extrapolation, SEGA should differ from NTK."""
        emb_sega = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], method="sega")
        emb_ntk = ConcreteSegAEmbed(theta=10000, axes_dim=[16, 56, 56], method="ntk")

        eh = torch.rand(16) * 10 + 1
        ew = torch.rand(16) * 10 + 1
        emb_sega.set_spectral_data(eh, ew, 0.8, target_res_h=4096, target_res_w=4096)
        emb_ntk.set_spectral_data(eh, ew, 0.8, target_res_h=4096, target_res_w=4096)

        pos = _make_pos(128, 128)
        comp_sega = emb_sega.get_components(pos, torch.float32)
        comp_ntk = emb_ntk.get_components(pos, torch.float32)

        # Spatial axes should differ due to SEGA mscale
        assert not torch.allclose(comp_sega[1][0], comp_ntk[1][0], atol=1e-4)
