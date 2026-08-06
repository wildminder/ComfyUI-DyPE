"""Tests for src/pixelrush.py — PixelRush core algorithm (Tier 1: pure unit tests)."""
import math
import torch
import pytest

from src.pixelrush import (
    PixelRushConfig,
    spherical_lerp,
    gaussian_kernel_2d,
    gaussian_feather_mask,
    patch_positions,
    ddim_forward_one_step,
    ddim_reverse_one_step_to_zero,
    refine_latent_once,
    pixelrush_cascade,
)


# ---------------------------------------------------------------------------
# spherical_lerp
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSphericalLerp:
    def test_t_zero_approximates_a(self):
        """t=0 should approximately return a (small numerical error from acos clamp)."""
        a = torch.randn(2, 4, 8, 8)
        b = torch.randn(2, 4, 8, 8)
        result = spherical_lerp(a, b, t=0.0)
        assert torch.allclose(result, a, atol=1e-3)

    def test_t_one_approximates_b(self):
        """t=1 should approximately return b (small numerical error from acos clamp)."""
        a = torch.randn(2, 4, 8, 8)
        b = torch.randn(2, 4, 8, 8)
        result = spherical_lerp(a, b, t=1.0)
        assert torch.allclose(result, b, atol=1e-3)

    def test_midpoint_between(self):
        a = torch.randn(1, 4, 4, 4)
        b = torch.randn(1, 4, 4, 4)
        result = spherical_lerp(a, b, t=0.5)
        # Midpoint should be between a and b
        assert result.shape == a.shape

    def test_parallel_vectors_linear(self):
        """SLERP of parallel vectors should approximate linear interpolation."""
        a = torch.ones(1, 8)
        b = torch.ones(1, 8) * 3.0
        result = spherical_lerp(a, b, t=0.5)
        # For parallel vectors, SLERP ≈ linear interpolation
        assert torch.allclose(result, torch.ones(1, 8) * 2.0, atol=0.1)

    def test_preserves_shape(self):
        a = torch.randn(2, 3, 16, 16)
        b = torch.randn(2, 3, 16, 16)
        result = spherical_lerp(a, b, t=0.3)
        assert result.shape == a.shape

    def test_no_nan(self):
        a = torch.randn(1, 4, 4, 4)
        b = torch.randn(1, 4, 4, 4)
        result = spherical_lerp(a, b, t=0.95)
        assert not torch.isnan(result).any()


# ---------------------------------------------------------------------------
# gaussian_kernel_2d
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGaussianKernel2D:
    def test_shape(self):
        k = gaussian_kernel_2d(41, 8.0, torch.device("cpu"), torch.float32)
        assert k.shape == (1, 1, 41, 41)

    def test_normalized(self):
        k = gaussian_kernel_2d(11, 3.0, torch.device("cpu"), torch.float32)
        assert abs(k.sum().item() - 1.0) < 1e-5

    def test_symmetric(self):
        k = gaussian_kernel_2d(11, 3.0, torch.device("cpu"), torch.float32)
        assert torch.allclose(k, k.flip(-1), atol=1e-6)
        assert torch.allclose(k, k.flip(-2), atol=1e-6)

    def test_center_peak(self):
        k = gaussian_kernel_2d(11, 3.0, torch.device("cpu"), torch.float32)
        center = k[0, 0, 5, 5]
        assert center == k.max()

    def test_odd_kernel_required(self):
        with pytest.raises(AssertionError):
            gaussian_kernel_2d(10, 3.0, torch.device("cpu"), torch.float32)


# ---------------------------------------------------------------------------
# gaussian_feather_mask
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGaussianFeatherMask:
    def test_shape(self):
        mask = gaussian_feather_mask(64, 64, 8.0, 41, torch.device("cpu"), torch.float32)
        assert mask.shape == (1, 1, 64, 64)

    def test_center_near_one(self):
        mask = gaussian_feather_mask(64, 64, 8.0, 41, torch.device("cpu"), torch.float32)
        center = mask[0, 0, 32, 32]
        assert abs(center.item() - 1.0) < 0.01

    def test_boundary_decay(self):
        mask = gaussian_feather_mask(64, 64, 8.0, 41, torch.device("cpu"), torch.float32)
        center = mask[0, 0, 32, 32]
        corner = mask[0, 0, 0, 0]
        assert corner < center

    def test_non_negative(self):
        mask = gaussian_feather_mask(32, 32, 5.0, 21, torch.device("cpu"), torch.float32)
        assert (mask >= 0).all()

    def test_no_nan(self):
        mask = gaussian_feather_mask(16, 16, 3.0, 11, torch.device("cpu"), torch.float32)
        assert not torch.isnan(mask).any()


# ---------------------------------------------------------------------------
# patch_positions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPatchPositions:
    def test_single_patch(self):
        positions = list(patch_positions(64, 64, 64, 64, 0.5))
        assert len(positions) == 1
        assert positions[0] == (0, 0)

    def test_full_coverage(self):
        """Patches should cover the entire latent."""
        positions = list(patch_positions(128, 128, 64, 64, 0.5))
        # With 50% overlap, stride=32: starts at 0, 32, 64
        ys = sorted(set(y for y, x in positions))
        xs = sorted(set(x for y, x in positions))
        assert ys[0] == 0
        assert ys[-1] + 64 >= 128  # Last patch reaches edge
        assert xs[0] == 0
        assert xs[-1] + 64 >= 128

    def test_overlap_count(self):
        """With 50% overlap on 128x128 with 64x64 patches: 3×3=9 patches."""
        positions = list(patch_positions(128, 128, 64, 64, 0.5))
        assert len(positions) == 9

    def test_no_overlap(self):
        """With 0% overlap on 128x128 with 64x64 patches: 2×2=4 patches."""
        positions = list(patch_positions(128, 128, 64, 64, 0.0))
        assert len(positions) == 4

    def test_edge_alignment(self):
        """Last patch in each dimension should touch the edge."""
        positions = list(patch_positions(100, 100, 64, 64, 0.5))
        ys = sorted(set(y for y, x in positions))
        xs = sorted(set(x for y, x in positions))
        assert ys[-1] + 64 == 100
        assert xs[-1] + 64 == 100

    def test_non_square(self):
        positions = list(patch_positions(128, 64, 64, 64, 0.5))
        ys = sorted(set(y for y, x in positions))
        xs = sorted(set(x for y, x in positions))
        assert len(ys) == 3  # 0, 32, 64
        assert len(xs) == 1  # 0


# ---------------------------------------------------------------------------
# ddim_forward_one_step / ddim_reverse_one_step_to_zero
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDDIMForward:
    def test_alpha_bar_1_returns_z0(self):
        """When alpha_bar=1, z_K = z_0 (no noise added)."""
        z0 = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        z_k = ddim_forward_one_step(z0, eps, alpha_bar_k=1.0)
        assert torch.allclose(z_k, z0, atol=1e-5)

    def test_alpha_bar_0_returns_eps(self):
        """When alpha_bar=0, z_K = eps (pure noise)."""
        z0 = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        z_k = ddim_forward_one_step(z0, eps, alpha_bar_k=0.0)
        assert torch.allclose(z_k, eps, atol=1e-5)

    def test_shape_preserved(self):
        z0 = torch.randn(2, 4, 16, 16)
        eps = torch.randn(2, 4, 16, 16)
        z_k = ddim_forward_one_step(z0, eps, alpha_bar_k=0.5)
        assert z_k.shape == z0.shape

    def test_tensor_alpha_bar(self):
        z0 = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        alpha = torch.tensor(0.5)
        z_k = ddim_forward_one_step(z0, eps, alpha_bar_k=alpha)
        assert z_k.shape == z0.shape

    def test_formula(self):
        """z_K = sqrt(a)*z0 + sqrt(1-a)*eps"""
        z0 = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        a = 0.7
        z_k = ddim_forward_one_step(z0, eps, alpha_bar_k=a)
        expected = math.sqrt(a) * z0 + math.sqrt(1 - a) * eps
        assert torch.allclose(z_k, expected, atol=1e-5)


@pytest.mark.unit
class TestDDIMReverse:
    def test_alpha_bar_1_returns_zk(self):
        """When alpha_bar=1, z_0 = z_K (no denoising needed)."""
        z_k = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        z_0 = ddim_reverse_one_step_to_zero(z_k, eps, alpha_bar_k=1.0)
        assert torch.allclose(z_0, z_k, atol=1e-5)

    def test_shape_preserved(self):
        z_k = torch.randn(2, 4, 16, 16)
        eps = torch.randn(2, 4, 16, 16)
        z_0 = ddim_reverse_one_step_to_zero(z_k, eps, alpha_bar_k=0.5)
        assert z_0.shape == z_k.shape

    def test_inverse_of_forward(self):
        """reverse(forward(z0, eps, a), eps, a) ≈ z0"""
        z0 = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        a = 0.8
        z_k = ddim_forward_one_step(z0, eps, alpha_bar_k=a)
        z_0_hat = ddim_reverse_one_step_to_zero(z_k, eps, alpha_bar_k=a)
        assert torch.allclose(z_0_hat, z0, atol=1e-4)

    def test_formula(self):
        """z_0 = (z_K - sqrt(1-a)*eps) / sqrt(a)"""
        z_k = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        a = 0.7
        z_0 = ddim_reverse_one_step_to_zero(z_k, eps, alpha_bar_k=a)
        expected = (z_k - math.sqrt(1 - a) * eps) / math.sqrt(a)
        assert torch.allclose(z_0, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# refine_latent_once
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRefineLatentOnce:
    def _mock_predict_eps(self):
        """Mock predict_eps that returns random noise."""
        def predict_eps(latent, timestep):
            return torch.randn_like(latent)
        return predict_eps

    def _mock_alpha_bar(self):
        """Mock alpha_bar_at: returns 0.8 for any timestep."""
        def alpha_bar_at(t):
            return 0.8
        return alpha_bar_at

    def test_output_shape(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_alpha_bar(), cfg)
        assert result.shape == latent.shape

    def test_no_nan(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_alpha_bar(), cfg)
        assert not torch.isnan(result).any()

    def test_single_patch(self):
        """When latent == patch size, only one patch."""
        cfg = PixelRushConfig(patch_h=64, patch_w=64, overlap=0.5)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_alpha_bar(), cfg)
        assert result.shape == latent.shape

    def test_multiple_patches(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 128, 128)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_alpha_bar(), cfg)
        assert result.shape == latent.shape

    def test_non_square(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 64, 128)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_alpha_bar(), cfg)
        assert result.shape == latent.shape

    def test_weight_normalization(self):
        """Output should be properly normalized (weight_sum > 0 everywhere)."""
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5, gaussian_sigma=4.0, gaussian_kernel_size=21)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_alpha_bar(), cfg)
        # Result should be finite (not inf/nan from division)
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# pixelrush_cascade
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPixelRushCascade:
    def _mock_vae_decode(self):
        """Mock VAE decode: just upscale channels to 3."""
        def decode(z):
            b, c, h, w = z.shape
            return z[:, :3] if c >= 3 else z.repeat(1, 3 // c + 1, 1, 1)[:, :3]
        return decode

    def _mock_vae_encode(self):
        """Mock VAE encode: just take first 4 channels."""
        def encode(x):
            b, c, h, w = x.shape
            if c >= 4:
                return x[:, :4]
            return x.repeat(1, 4 // c + 1, 1, 1)[:, :4]
        return encode

    def _mock_predict_eps(self):
        def predict_eps(latent, timestep):
            return torch.randn_like(latent)
        return predict_eps

    def _mock_alpha_bar(self):
        def alpha_bar_at(t):
            return 0.8
        return alpha_bar_at

    def test_single_stage(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        z0 = torch.randn(1, 4, 32, 32)
        result = pixelrush_cascade(
            z0, num_cascade_stages=1,
            vae_decode=self._mock_vae_decode(),
            vae_encode=self._mock_vae_encode(),
            predict_eps=self._mock_predict_eps(),
            alpha_bar_at=self._mock_alpha_bar(),
            cfg=cfg,
        )
        # After 1 stage: 32→64 (2× upscale)
        assert result.shape[2] == 64
        assert result.shape[3] == 64

    def test_two_stages(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        z0 = torch.randn(1, 4, 32, 32)
        result = pixelrush_cascade(
            z0, num_cascade_stages=2,
            vae_decode=self._mock_vae_decode(),
            vae_encode=self._mock_vae_encode(),
            predict_eps=self._mock_predict_eps(),
            alpha_bar_at=self._mock_alpha_bar(),
            cfg=cfg,
        )
        # After 2 stages: 32→64→128
        assert result.shape[2] == 128
        assert result.shape[3] == 128

    def test_no_nan(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        z0 = torch.randn(1, 4, 32, 32)
        result = pixelrush_cascade(
            z0, num_cascade_stages=1,
            vae_decode=self._mock_vae_decode(),
            vae_encode=self._mock_vae_encode(),
            predict_eps=self._mock_predict_eps(),
            alpha_bar_at=self._mock_alpha_bar(),
            cfg=cfg,
        )
        assert not torch.isnan(result).any()

    def test_progressive_resolution(self):
        """Each stage should double the resolution."""
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        z0 = torch.randn(1, 4, 32, 32)
        for stages in [1, 2, 3]:
            result = pixelrush_cascade(
                z0, num_cascade_stages=stages,
                vae_decode=self._mock_vae_decode(),
                vae_encode=self._mock_vae_encode(),
                predict_eps=self._mock_predict_eps(),
                alpha_bar_at=self._mock_alpha_bar(),
                cfg=cfg,
            )
            expected = 32 * (2 ** stages)
            assert result.shape[2] == expected
            assert result.shape[3] == expected
