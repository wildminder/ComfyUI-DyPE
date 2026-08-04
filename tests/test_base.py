"""Tests for src/base.py — DyPEBasePosEmbed (Tier 1: pure unit tests)."""
import math
import torch
import pytest

from src.base import DyPEBasePosEmbed


class ConcreteEmbed(DyPEBasePosEmbed):
    """Minimal concrete subclass for testing."""
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.float32
        components = self.get_components(pos, freqs_dtype)
        return torch.cat([cos for cos, sin in components], dim=-1)


@pytest.mark.unit
class TestAxisTokenSpan:
    def test_sequential_positions(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        pos = torch.arange(64, dtype=torch.float32)
        span = emb._axis_token_span(pos)
        assert span == 64.0

    def test_single_element(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        pos = torch.tensor([5.0])
        span = emb._axis_token_span(pos)
        assert span == 1.0

    def test_all_same_values(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        pos = torch.full((32,), 7.0)
        span = emb._axis_token_span(pos)
        assert span == 1.0

    def test_stepped_positions(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        pos = torch.arange(0, 128, 2, dtype=torch.float32)
        span = emb._axis_token_span(pos)
        assert span == 64.0

    def test_2d_tensor(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        pos = torch.arange(64, dtype=torch.float32).unsqueeze(0).expand(4, -1)
        span = emb._axis_token_span(pos)
        assert span == 64.0

    def test_cache_hit(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        pos = torch.arange(64, dtype=torch.float32)
        span1 = emb._axis_token_span(pos)
        span2 = emb._axis_token_span(pos)
        assert span1 == span2 == 64.0
        assert len(emb._span_cache) > 0

    def test_different_shape_different_cache(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        pos32 = torch.arange(32, dtype=torch.float32)
        pos64 = torch.arange(64, dtype=torch.float32)
        span32 = emb._axis_token_span(pos32)
        span64 = emb._axis_token_span(pos64)
        assert span32 == 32.0
        assert span64 == 64.0


@pytest.mark.unit
class TestDyPEBaseInit:
    def test_default_parameters(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        assert emb.theta == 10000
        assert emb.axes_dim == [16, 56, 56]
        assert emb.method == 'yarn'
        assert emb.dype is True
        assert emb.dype_scale == 2.0
        assert emb.dype_exponent == 2.0
        assert emb.base_resolution == 1024
        assert emb.current_timestep == 1.0

    def test_base_patch_grid_default(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], base_resolution=1024)
        assert emb.base_patch_grid == (64, 64)
        assert emb.base_patches == 64

    def test_base_patch_grid_custom_tuple(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], base_patch_grid=(32, 48))
        assert emb.base_patch_grid == (32, 48)
        assert emb.base_patches == 48

    def test_base_patch_grid_custom_int(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], base_patch_grid=32)
        assert emb.base_patch_grid == (32, 32)
        assert emb.base_patches == 32

    def test_vision_yarn_forces_dype(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='vision_yarn', dype=False)
        assert emb.dype is True

    def test_base_method_disables_dype(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='base', dype=True)
        assert emb.dype is False

    def test_dype_start_sigma_clamped_low(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], dype_start_sigma=0.0)
        assert emb.dype_start_sigma == 0.001

    def test_dype_start_sigma_clamped_high(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], dype_start_sigma=2.0)
        assert emb.dype_start_sigma == 1.0


@pytest.mark.unit
class TestSetTimestep:
    def test_sets_value(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        emb.set_timestep(0.5)
        assert emb.current_timestep == 0.5

    def test_sets_zero(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        emb.set_timestep(0.0)
        assert emb.current_timestep == 0.0


@pytest.mark.unit
class TestGetMscale:
    def test_mscale_at_t1(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], dype_exponent=2.0)
        emb.set_timestep(1.0)
        mscale = emb._get_mscale(4.0)
        expected = 1.0 + 0.1 * math.log(4.0)
        assert abs(mscale - expected) < 1e-5

    def test_mscale_at_t0(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], dype_exponent=2.0)
        emb.set_timestep(0.0)
        mscale = emb._get_mscale(4.0)
        assert abs(mscale - 1.0) < 1e-5

    def test_mscale_scale_1(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56])
        emb.set_timestep(1.0)
        mscale = emb._get_mscale(1.0)
        assert abs(mscale - 1.0) < 1e-5

    def test_mscale_midpoint(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], dype_exponent=2.0, dype_start_sigma=1.0)
        emb.set_timestep(0.5)
        mscale = emb._get_mscale(4.0)
        # t_norm = 0.5, pow(0.5, 2) = 0.25
        expected = 1.0 + (0.1 * math.log(4.0)) * 0.25
        assert abs(mscale - expected) < 1e-5


@pytest.mark.unit
class TestGetComponents:
    def _make_pos(self, H=64, W=64):
        L = H * W
        pos = torch.zeros(1, L, 3)
        pos[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        pos[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
        return pos

    def test_ntk_method_returns_3_components(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='ntk')
        pos = self._make_pos()
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3
        for cos, sin in components:
            assert cos.shape[-1] > 0

    def test_vision_yarn_method_returns_3_components(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='vision_yarn')
        pos = self._make_pos()
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3

    def test_yarn_method_returns_3_components(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='yarn')
        pos = self._make_pos()
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3

    def test_pi_method_returns_3_components(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='pi')
        pos = self._make_pos()
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3

    def test_base_method_returns_3_components(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='base')
        pos = self._make_pos()
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3

    def test_pi_timestep_affects_output(self):
        emb = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='pi', dype=True)
        pos = self._make_pos(128, 128)

        emb.set_timestep(1.0)
        comp_t1 = emb.get_components(pos, torch.float32)

        emb.set_timestep(0.1)
        comp_t01 = emb.get_components(pos, torch.float32)

        # Spatial axes (1, 2) should differ
        assert not torch.allclose(comp_t1[1][0], comp_t01[1][0], atol=1e-3)

    def test_vision_yarn_extrapolation_needed(self):
        """When resolution exceeds base, vision_yarn should produce different output than base."""
        emb_vy = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='vision_yarn', base_resolution=512)
        emb_base = ConcreteEmbed(theta=10000, axes_dim=[16, 56, 56], method='base', base_resolution=512)
        pos = self._make_pos(128, 128)  # 128 > 64 base patches for 512px

        comp_vy = emb_vy.get_components(pos, torch.float32)
        comp_base = emb_base.get_components(pos, torch.float32)

        # Axis 1 (height) should differ since extrapolation is needed
        assert not torch.allclose(comp_vy[1][0], comp_base[1][0], atol=1e-3)


@pytest.mark.unit
class TestPerAxisTheta:
    """Test per-axis theta support in base class."""

    def _make_pos(self, H=64, W=64):
        L = H * W
        pos = torch.zeros(1, L, 3)
        pos[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        pos[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
        return pos

    def test_list_theta_vision_yarn(self):
        """Vision YaRN should work with list theta."""
        emb = ConcreteEmbed(
            theta=[10000.0, 20000.0, 30000.0],
            axes_dim=[16, 56, 56],
            method='vision_yarn'
        )
        pos = self._make_pos(8, 8)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3

    def test_list_theta_yarn(self):
        """YaRN should work with list theta."""
        emb = ConcreteEmbed(
            theta=[10000.0, 20000.0, 30000.0],
            axes_dim=[16, 56, 56],
            method='yarn'
        )
        pos = self._make_pos(8, 8)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3

    def test_list_theta_ntk(self):
        """NTK should work with list theta."""
        emb = ConcreteEmbed(
            theta=[10000.0, 20000.0, 30000.0],
            axes_dim=[16, 56, 56],
            method='ntk'
        )
        pos = self._make_pos(8, 8)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3

    def test_list_theta_pi(self):
        """PI should work with list theta."""
        emb = ConcreteEmbed(
            theta=[10000.0, 20000.0, 30000.0],
            axes_dim=[16, 56, 56],
            method='pi'
        )
        pos = self._make_pos(8, 8)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3
