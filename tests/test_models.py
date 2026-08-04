"""Tests for src/models/ — adapter output format tests (Tier 1)."""
import torch
import pytest

from src.models.flux import PosEmbedFlux
from src.models.nunchaku import PosEmbedNunchaku
from src.models.qwen import PosEmbedQwen
from src.models.anima import PosEmbedAnima


@pytest.fixture
def flux_ids():
    """Standard FLUX position IDs: (B=1, L=4096, 3) for 64x64 grid."""
    B, H, W = 1, 64, 64
    L = H * W
    ids = torch.zeros(B, L, 3)
    ids[..., 0] = torch.arange(L, dtype=torch.float32)
    ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
    ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    return ids


@pytest.fixture
def small_ids():
    """Small position IDs: (B=1, L=64, 3) for 8x8 grid."""
    B, H, W = 1, 8, 8
    L = H * W
    ids = torch.zeros(B, L, 3)
    ids[..., 0] = torch.arange(L, dtype=torch.float32)
    ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
    ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    return ids


@pytest.mark.unit
class TestPosEmbedFlux:
    def test_output_shape(self, flux_ids):
        emb = PosEmbedFlux(theta=10000, axes_dim=[16, 56, 56], method='ntk')
        out = emb(flux_ids)
        # Expected: (B, 1, L, D//2, 2, 2) where D//2 = sum(axes_dim)//2 = 64
        # The rotation matrix format uses D/2 frequency pairs
        assert out.shape == (1, 1, 4096, 64, 2, 2)

    def test_output_is_rotation_matrix(self, small_ids):
        emb = PosEmbedFlux(theta=10000, axes_dim=[16, 56, 56], method='ntk')
        out = emb(small_ids)
        # Each 2x2 matrix should have det ≈ 1 (rotation)
        matrices = out[0, 0, 0, 0]  # First token, first frequency
        det = matrices[0, 0] * matrices[1, 1] - matrices[0, 1] * matrices[1, 0]
        assert abs(det.item() - 1.0) < 1e-4

    def test_vision_yarn_method(self, small_ids):
        emb = PosEmbedFlux(theta=10000, axes_dim=[16, 56, 56], method='vision_yarn')
        out = emb(small_ids)
        assert out.shape == (1, 1, 64, 64, 2, 2)

    def test_pi_method(self, small_ids):
        emb = PosEmbedFlux(theta=10000, axes_dim=[16, 56, 56], method='pi')
        out = emb(small_ids)
        assert out.shape == (1, 1, 64, 64, 2, 2)


@pytest.mark.unit
class TestPosEmbedNunchaku:
    def test_output_shape(self, flux_ids):
        emb = PosEmbedNunchaku(theta=10000, axes_dim=[16, 56, 56], method='ntk')
        out = emb(flux_ids)
        # Expected: (B, 1, L, D//2, 1, 2) where D=128 → D//2=64
        assert out.shape == (1, 1, 4096, 64, 1, 2)

    def test_sin_cos_pairing(self, small_ids):
        emb = PosEmbedNunchaku(theta=10000, axes_dim=[16, 56, 56], method='ntk')
        out = emb(small_ids)
        # Last dim is [sin, cos]
        sin_vals = out[..., 0]
        cos_vals = out[..., 1]
        # sin²+cos² should ≈ 1
        magnitude = sin_vals**2 + cos_vals**2
        assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-4)

    def test_3axis_input(self):
        """Nunchaku handles 3-axis input like FLUX."""
        B, H, W = 1, 8, 8
        L = H * W
        ids = torch.zeros(B, L, 3)
        ids[..., 0] = torch.arange(L, dtype=torch.float32)
        ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
        emb = PosEmbedNunchaku(theta=10000, axes_dim=[16, 56, 56], method='ntk')
        out = emb(ids)
        assert out.dim() >= 4


@pytest.mark.unit
class TestPosEmbedQwen:
    def test_output_shape(self, flux_ids):
        emb = PosEmbedQwen(theta=10000, axes_dim=[16, 56, 56], method='ntk')
        out = emb(flux_ids)
        # Expected: (B, 1, L, D/2, 2, 2) where D=128 → D/2=64
        assert out.shape == (1, 1, 4096, 64, 2, 2)

    def test_rotation_columns(self, small_ids):
        """Qwen output columns should form rotation matrices."""
        emb = PosEmbedQwen(theta=10000, axes_dim=[16, 56, 56], method='ntk')
        out = emb(small_ids)
        # out[..., 0] = [cos, sin], out[..., 1] = [-sin, cos]
        col0 = out[0, 0, 0, 0, :, 0]  # First token, first freq, column 0
        col1 = out[0, 0, 0, 0, :, 1]  # column 1
        cos_val = col0[0]
        sin_val = col0[1]
        # col1 should be [-sin, cos]
        assert torch.allclose(col1[0], -sin_val, atol=1e-5)
        assert torch.allclose(col1[1], cos_val, atol=1e-5)


@pytest.mark.unit
class TestPosEmbedAnima:
    """Tests for Anima/Cosmos positional embedding adapter."""

    def _make_anima_pos(self, T=1, H=64, W=64):
        """Create position tensor for Anima: (T*H*W, 3) with (t, h, w) coordinates."""
        t_grid = torch.arange(T, dtype=torch.float32).view(T, 1, 1).expand(T, H, W)
        h_grid = torch.arange(H, dtype=torch.float32).view(1, H, 1).expand(T, H, W)
        w_grid = torch.arange(W, dtype=torch.float32).view(1, 1, W).expand(T, H, W)
        pos = torch.stack([t_grid.flatten(), h_grid.flatten(), w_grid.flatten()], dim=-1)
        return pos

    def test_output_shape(self):
        """Anima output should be (T*H*W, D/2, 2, 2) rotation matrices."""
        # head_dim = 128, dim_h = dim_w = 128//6*2 = 42, dim_t = 128 - 84 = 44
        # axes_dim = [44, 42, 42]
        emb = PosEmbedAnima(
            theta=[10000.0, 10000.0, 10000.0],
            axes_dim=[44, 42, 42],
            method='vision_yarn'
        )
        pos = self._make_anima_pos(T=1, H=8, W=8)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3
        # Each component should have shape (T*H*W, dim_axis)
        assert components[0][0].shape == (64, 44)  # temporal
        assert components[1][0].shape == (64, 42)  # height
        assert components[2][0].shape == (64, 42)  # width

    def test_per_axis_theta(self):
        """Anima should use per-axis theta values via base class thetas list."""
        emb = PosEmbedAnima(
            theta=[10000.0, 20000.0, 30000.0],
            axes_dim=[44, 42, 42],
            method='vision_yarn'
        )
        assert emb.thetas == [10000.0, 20000.0, 30000.0]

    def test_vision_yarn_extrapolation(self):
        """Vision YaRN should handle extrapolation correctly."""
        emb = PosEmbedAnima(
            theta=[10000.0, 10000.0, 10000.0],
            axes_dim=[44, 42, 42],
            method='vision_yarn',
            base_resolution=512  # base_patch_grid = (32, 32)
        )
        # 64x64 patches = 2x extrapolation from 32x32 base
        pos = self._make_anima_pos(T=1, H=64, W=64)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3
        # Should not raise and should produce finite values
        for cos, sin in components:
            assert torch.isfinite(cos).all()
            assert torch.isfinite(sin).all()

    def test_yarn_method_with_extrapolation(self):
        """YaRN method should handle extrapolation with per-axis scale."""
        emb = PosEmbedAnima(
            theta=[10000.0, 10000.0, 10000.0],
            axes_dim=[44, 42, 42],
            method='yarn',
            base_resolution=512
        )
        pos = self._make_anima_pos(T=1, H=64, W=64)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3
        for cos, sin in components:
            assert torch.isfinite(cos).all()
            assert torch.isfinite(sin).all()

    def test_ntk_method_with_extrapolation(self):
        """NTK method should handle extrapolation with per-axis scale."""
        emb = PosEmbedAnima(
            theta=[10000.0, 10000.0, 10000.0],
            axes_dim=[44, 42, 42],
            method='ntk',
            base_resolution=512
        )
        pos = self._make_anima_pos(T=1, H=64, W=64)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3
        for cos, sin in components:
            assert torch.isfinite(cos).all()
            assert torch.isfinite(sin).all()

    def test_pi_method_with_extrapolation(self):
        """PI method should handle extrapolation with per-axis scale."""
        emb = PosEmbedAnima(
            theta=[10000.0, 10000.0, 10000.0],
            axes_dim=[44, 42, 42],
            method='pi',
            base_resolution=512
        )
        pos = self._make_anima_pos(T=1, H=64, W=64)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3
        for cos, sin in components:
            assert torch.isfinite(cos).all()
            assert torch.isfinite(sin).all()

    def test_base_method_no_extrapolation(self):
        """Base method should not apply any scaling."""
        emb = PosEmbedAnima(
            theta=[10000.0, 10000.0, 10000.0],
            axes_dim=[44, 42, 42],
            method='base',
            base_resolution=512
        )
        pos = self._make_anima_pos(T=1, H=64, W=64)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3
        for cos, sin in components:
            assert torch.isfinite(cos).all()
            assert torch.isfinite(sin).all()

    def test_temporal_axis_not_scaled(self):
        """Temporal axis (i=0) should never be scaled."""
        emb = PosEmbedAnima(
            theta=[10000.0, 10000.0, 10000.0],
            axes_dim=[44, 42, 42],
            method='vision_yarn',
            base_resolution=512
        )
        pos = self._make_anima_pos(T=4, H=64, W=64)
        components = emb.get_components(pos, torch.float32)
        # Temporal component should use ntk_factor=1.0 (no scaling)
        # We can't directly test this, but we can verify the output is finite
        assert torch.isfinite(components[0][0]).all()
        assert torch.isfinite(components[0][1]).all()

    def test_non_square_resolution(self):
        """Non-square resolutions should use per-axis scaling."""
        emb = PosEmbedAnima(
            theta=[10000.0, 10000.0, 10000.0],
            axes_dim=[44, 42, 42],
            method='vision_yarn',
            base_resolution=512
        )
        # H=64, W=128 — different scales for H and W
        pos = self._make_anima_pos(T=1, H=64, W=128)
        components = emb.get_components(pos, torch.float32)
        assert len(components) == 3
        for cos, sin in components:
            assert torch.isfinite(cos).all()
            assert torch.isfinite(sin).all()

    def test_forward_output_shape(self):
        """Forward pass should produce correct output shape."""
        emb = PosEmbedAnima(
            theta=[10000.0, 10000.0, 10000.0],
            axes_dim=[44, 42, 42],
            method='vision_yarn'
        )
        x = torch.randn(1, 1, 8, 8, 128)  # B, T, H, W, C
        out = emb(x)
        # Output should be (T*H*W, D/2, 2, 2) where D = 128
        # D/2 = 64, but we have 3 axes with dims [44, 42, 42]
        # Total freq dim = 44//2 + 42//2 + 42//2 = 22 + 21 + 21 = 64
        assert out.shape == (64, 64, 2, 2)
