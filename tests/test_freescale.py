"""Tests for FreeScale core algorithm (Tier 1: pure math tests)."""
import math

import pytest
import torch

from src.freescale import (
    FreeScaleConfig,
    blend_detail_latents,
    cosine_detail_weight,
    forward_noise,
    gaussian_blur_2d,
    gaussian_kernel_2d,
    scale_fusion,
)

# ---------------------------------------------------------------------------
# Gaussian kernel
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGaussianKernel2D:
    def test_shape(self):
        kernel = gaussian_kernel_2d(5, 1.0)
        assert kernel.shape == (5, 5)

    def test_normalized(self):
        kernel = gaussian_kernel_2d(5, 1.0)
        assert abs(kernel.sum().item() - 1.0) < 1e-6

    def test_symmetric(self):
        kernel = gaussian_kernel_2d(7, 2.0)
        # Symmetric along both axes
        assert torch.allclose(kernel, kernel.flip(0), atol=1e-6)
        assert torch.allclose(kernel, kernel.flip(1), atol=1e-6)

    def test_center_peak(self):
        kernel = gaussian_kernel_2d(5, 1.0)
        center = kernel[2, 2]
        assert center == kernel.max()

    def test_odd_kernel_required(self):
        with pytest.raises(ValueError, match="kernel_size must be odd"):
            gaussian_kernel_2d(4, 1.0)

    def test_device_dtype(self):
        kernel = gaussian_kernel_2d(5, 1.0, device=torch.device("cpu"), dtype=torch.float64)
        assert kernel.dtype == torch.float64


# ---------------------------------------------------------------------------
# Gaussian blur
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGaussianBlur2D:
    def test_shape(self):
        x = torch.randn(2, 4, 32, 32)
        blurred = gaussian_blur_2d(x, kernel_size=5, sigma=1.0)
        assert blurred.shape == x.shape

    def test_reduces_variance(self):
        """Blurring should reduce high-frequency variance."""
        x = torch.randn(1, 1, 64, 64)
        blurred = gaussian_blur_2d(x, kernel_size=5, sigma=1.0)
        # Blurred image should have lower variance (less high-freq)
        assert blurred.var() < x.var()

    def test_constant_input_center_unchanged(self):
        """Constant input should remain constant at the center (edges differ due to zero-padding)."""
        x = torch.ones(1, 3, 32, 32) * 5.0
        blurred = gaussian_blur_2d(x, kernel_size=5, sigma=1.0)
        # Check center region (away from padding effects)
        assert torch.allclose(blurred[:, :, 8:24, 8:24], x[:, :, 8:24, 8:24], atol=1e-5)

    def test_preserves_batch_channels(self):
        x = torch.randn(4, 8, 32, 32)
        blurred = gaussian_blur_2d(x, kernel_size=3, sigma=0.5)
        assert blurred.shape == (4, 8, 32, 32)

    def test_no_nan(self):
        x = torch.randn(2, 4, 16, 16)
        blurred = gaussian_blur_2d(x, kernel_size=7, sigma=2.0)
        assert not torch.isnan(blurred).any()


# ---------------------------------------------------------------------------
# Scale fusion
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestScaleFusion:
    def test_shape(self):
        global_feat = torch.randn(2, 4, 32, 32)
        local_feat = torch.randn(2, 4, 32, 32)
        fused = scale_fusion(global_feat, local_feat, kernel_size=5, sigma=1.0)
        assert fused.shape == global_feat.shape

    def test_formula(self):
        """fused = global - blur(global) + blur(local)"""
        global_feat = torch.randn(1, 4, 16, 16)
        local_feat = torch.randn(1, 4, 16, 16)
        k, s = 5, 1.0

        fused = scale_fusion(global_feat, local_feat, k, s)

        expected = (global_feat
                    - gaussian_blur_2d(global_feat, k, s)
                    + gaussian_blur_2d(local_feat, k, s))
        assert torch.allclose(fused, expected, atol=1e-5)

    def test_identical_inputs(self):
        """If global == local, fused = global (high-freq cancels, low-freq adds back)."""
        x = torch.randn(1, 4, 32, 32)
        fused = scale_fusion(x, x, kernel_size=5, sigma=1.0)
        # fused = x - blur(x) + blur(x) = x
        assert torch.allclose(fused, x, atol=1e-5)

    def test_no_nan(self):
        global_feat = torch.randn(2, 8, 64, 64)
        local_feat = torch.randn(2, 8, 64, 64)
        fused = scale_fusion(global_feat, local_feat, kernel_size=7, sigma=2.0)
        assert not torch.isnan(fused).any()

    def test_different_kernel_sizes(self):
        global_feat = torch.randn(1, 4, 32, 32)
        local_feat = torch.randn(1, 4, 32, 32)
        for ks in [3, 5, 7, 9]:
            fused = scale_fusion(global_feat, local_feat, kernel_size=ks, sigma=1.0)
            assert fused.shape == global_feat.shape


# ---------------------------------------------------------------------------
# Cosine detail weight
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCosineDetailWeight:
    def test_returns_float(self):
        w = cosine_detail_weight(0, 50, alpha=2.0)
        assert isinstance(w, float)

    def test_bounds(self):
        """Weight should be in [0, alpha]."""
        for t in range(50):
            w = cosine_detail_weight(t, 50, alpha=2.0)
            assert 0.0 <= w <= 2.0 + 1e-6

    def test_alpha_zero(self):
        """With alpha=0, weight should always be 0."""
        for t in range(50):
            assert cosine_detail_weight(t, 50, alpha=0.0) == 0.0

    def test_monotonic_decreasing(self):
        """Weight should decrease over timesteps (early = more upsampled signal)."""
        weights = [cosine_detail_weight(t, 50, alpha=2.0) for t in range(50)]
        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1] - 1e-6

    def test_first_step_max(self):
        """At t=0, weight should be alpha (maximum)."""
        w = cosine_detail_weight(0, 50, alpha=2.0)
        assert abs(w - 2.0) < 1e-6

    def test_last_step_min(self):
        """At t=T-1, weight should be near 0."""
        w = cosine_detail_weight(49, 50, alpha=2.0)
        assert w < 0.1

    def test_custom_alpha(self):
        w = cosine_detail_weight(0, 50, alpha=1.0)
        assert abs(w - 1.0) < 1e-6

    def test_formula(self):
        """Verify the cosine formula: alpha * 0.5 * (1 + cos(t/T * pi))."""
        t, T, alpha = 10, 50, 2.0
        expected = alpha * 0.5 * (1.0 + math.cos(t / T * math.pi))
        actual = cosine_detail_weight(t, T, alpha)
        assert abs(actual - expected) < 1e-6


# ---------------------------------------------------------------------------
# Blend detail latents
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBlendDetailLatents:
    def test_shape(self):
        noisy = torch.randn(2, 4, 32, 32)
        ordinary = torch.randn(2, 4, 32, 32)
        blended = blend_detail_latents(noisy, ordinary, 0, 50, alpha=2.0)
        assert blended.shape == noisy.shape

    def test_alpha_zero_returns_ordinary(self):
        """With alpha=0, blend should return ordinary latent."""
        noisy = torch.randn(1, 4, 16, 16)
        ordinary = torch.randn(1, 4, 16, 16)
        blended = blend_detail_latents(noisy, ordinary, 10, 50, alpha=0.0)
        assert torch.allclose(blended, ordinary, atol=1e-6)

    def test_first_step_returns_noisy(self):
        """At t=0 with high alpha, blend should be mostly noisy."""
        noisy = torch.randn(1, 4, 16, 16)
        ordinary = torch.randn(1, 4, 16, 16)
        blended = blend_detail_latents(noisy, ordinary, 0, 50, alpha=2.0)
        # c_0 = 2.0, so blended = 2.0 * noisy + (1-2.0) * ordinary = 2*noisy - ordinary
        expected = 2.0 * noisy - ordinary
        assert torch.allclose(blended, expected, atol=1e-5)

    def test_no_nan(self):
        noisy = torch.randn(2, 8, 64, 64)
        ordinary = torch.randn(2, 8, 64, 64)
        blended = blend_detail_latents(noisy, ordinary, 25, 50, alpha=1.5)
        assert not torch.isnan(blended).any()


# ---------------------------------------------------------------------------
# Forward noise
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestForwardNoise:
    def test_shape(self):
        z0 = torch.randn(2, 4, 32, 32)
        eps = torch.randn(2, 4, 32, 32)
        z_k = forward_noise(z0, eps, 0.5)
        assert z_k.shape == z0.shape

    def test_alpha_bar_1_returns_z0(self):
        """With alpha_bar=1, noise term vanishes, returns z0."""
        z0 = torch.randn(1, 4, 16, 16)
        eps = torch.randn(1, 4, 16, 16)
        z_k = forward_noise(z0, eps, 1.0)
        assert torch.allclose(z_k, z0, atol=1e-6)

    def test_alpha_bar_0_returns_eps(self):
        """With alpha_bar=0, signal term vanishes, returns eps."""
        z0 = torch.randn(1, 4, 16, 16)
        eps = torch.randn(1, 4, 16, 16)
        z_k = forward_noise(z0, eps, 0.0)
        assert torch.allclose(z_k, eps, atol=1e-6)

    def test_formula(self):
        """z_K = sqrt(alpha_bar) * z0 + sqrt(1-alpha_bar) * eps"""
        z0 = torch.randn(1, 4, 16, 16)
        eps = torch.randn(1, 4, 16, 16)
        alpha_bar = 0.7
        z_k = forward_noise(z0, eps, alpha_bar)
        expected = math.sqrt(alpha_bar) * z0 + math.sqrt(1 - alpha_bar) * eps
        assert torch.allclose(z_k, expected, atol=1e-5)

    def test_tensor_alpha_bar(self):
        z0 = torch.randn(1, 4, 16, 16)
        eps = torch.randn(1, 4, 16, 16)
        alpha_bar = torch.tensor(0.5)
        z_k = forward_noise(z0, eps, alpha_bar)
        assert z_k.shape == z0.shape

    def test_no_nan(self):
        z0 = torch.randn(2, 8, 32, 32)
        eps = torch.randn(2, 8, 32, 32)
        z_k = forward_noise(z0, eps, 0.3)
        assert not torch.isnan(z_k).any()


# ---------------------------------------------------------------------------
# FreeScaleConfig
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFreeScaleConfig:
    def test_defaults(self):
        cfg = FreeScaleConfig()
        assert cfg.target_resolution == 2048
        assert cfg.noise_timestep == 700
        assert cfg.cosine_scale == 2.0
        assert cfg.fast_mode is True
        assert cfg.gaussian_kernel_size == 5
        assert cfg.gaussian_sigma == 1.0
        assert cfg.num_inference_steps == 50

    def test_custom(self):
        cfg = FreeScaleConfig(
            target_resolution=4096,
            noise_timestep=500,
            cosine_scale=1.0,
            fast_mode=False,
            gaussian_kernel_size=7,
            gaussian_sigma=2.0,
            num_inference_steps=30,
        )
        assert cfg.target_resolution == 4096
        assert cfg.noise_timestep == 500
        assert cfg.cosine_scale == 1.0
        assert cfg.fast_mode is False
        assert cfg.gaussian_kernel_size == 7
        assert cfg.gaussian_sigma == 2.0
        assert cfg.num_inference_steps == 30
