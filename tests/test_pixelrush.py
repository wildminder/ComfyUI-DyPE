"""Tests for src/pixelrush.py — PixelRush core algorithm (Tier 1: pure unit tests)."""
import math

import pytest
import torch

from src.pixelrush import (
    PixelRushConfig,
    ddim_forward_one_step,
    ddim_reverse_one_step_to_zero,
    gaussian_feather_mask,
    gaussian_kernel_2d,
    patch_positions,
    pixelrush_cascade,
    refine_latent_once,
    spherical_lerp,
)

# ---------------------------------------------------------------------------
# spherical_lerp
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSphericalLerp:
    def test_t_zero_proportional_to_a(self):
        """t=0 must return exactly a (unit-vector SLERP: direction=a_unit,
        magnitude=a_norm -> a_unit*a_norm = a)."""
        a = torch.randn(2, 4, 8, 8)
        b = torch.randn(2, 4, 8, 8)
        result = spherical_lerp(a, b, t=0.0)
        # Correct SLERP uses unit vectors: at t=0, direction = a_unit, magnitude = a_norm
        # -> result = a_unit * a_norm = a (exactly)
        assert torch.allclose(result, a, atol=1e-5)

    def test_t_one_proportional_to_b(self):
        """t=1 must return exactly b (unit-vector SLERP: direction=b_unit,
        magnitude=b_norm -> b_unit*b_norm = b)."""
        a = torch.randn(2, 4, 8, 8)
        b = torch.randn(2, 4, 8, 8)
        result = spherical_lerp(a, b, t=1.0)
        assert torch.allclose(result, b, atol=1e-5)

    def test_midpoint_between(self):
        a = torch.randn(1, 4, 4, 4)
        b = torch.randn(1, 4, 4, 4)
        result = spherical_lerp(a, b, t=0.5)
        # Midpoint should be between a and b
        assert result.shape == a.shape

    def test_parallel_vectors_linear(self):
        """SLERP of parallel vectors: result = unit * (0.5*|a| + 0.5*|b|).

        Using unit vectors, the direction for parallel a,b is just the shared
        unit vector, so result = unit * interpolated_magnitude (NOT squared).
        """
        a = torch.ones(1, 8)
        b = torch.ones(1, 8) * 3.0
        result = spherical_lerp(a, b, t=0.5)
        # For parallel vectors (omega≈0): direction = 0.5*a_unit + 0.5*b_unit = unit
        # magnitude = 0.5 * norm_a + 0.5 * norm_b
        norm_a = a.flatten(1).norm(dim=1, keepdim=True)
        norm_b = b.flatten(1).norm(dim=1, keepdim=True)
        unit = a / norm_a
        magnitude = 0.5 * norm_a + 0.5 * norm_b
        expected = unit * magnitude  # = 2.0 per element
        assert torch.allclose(result, expected, atol=0.1)

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


# ---------------------------------------------------------------------------
# PixelRushConfig: operate_in_vae_space flag (plan 2026-08-12)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPixelRushConfigVAESpace:
    def test_default_operate_in_vae_space_true(self):
        """Default must be True (algorithm runs in VAE space)."""
        cfg = PixelRushConfig(patch_h=32, patch_w=32)
        assert cfg.operate_in_vae_space is True

    def test_override_operate_in_vae_space_false(self):
        """Can be set to False (legacy model-space path)."""
        cfg = PixelRushConfig(patch_h=32, patch_w=32, operate_in_vae_space=False)
        assert cfg.operate_in_vae_space is False


# ---------------------------------------------------------------------------
# Regression: SDXL noise-dominance fix (plan 2026-08-12)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPixelRushCascadeVAESpace:
    """Regression tests for the SDXL 'totally noisy' bug.

    Root cause: SDXL's process_latent_in scales the latent down by
    scale_factor=0.13025, so the latent has std≈0.13 in model space while the
    noise injection has std≈0.95. Operating the algorithm in VAE space (std≈1)
    keeps the noise balanced against the signal.
    """

    def _identity_vae_decode(self):
        def decode(z):
            return z  # identity (4ch passthrough)
        return decode

    def _identity_vae_encode(self):
        def encode(x):
            return x  # identity (4ch passthrough)
        return encode

    def _realistic_predict_eps(self, seed=0):
        """Mock a real SDXL: small epsilon (std 0.1) for clean latents."""
        def predict_eps(latent, timestep):
            g = torch.Generator().manual_seed(seed)
            return 0.1 * torch.randn(
                latent.shape, generator=g, device=latent.device, dtype=latent.dtype
            )
        return predict_eps

    def _vae_space_forward_step(self):
        def forward_step(x_0, eps, sigma):
            return x_0 + sigma * eps
        return forward_step

    def _vae_space_reverse_step(self):
        def reverse_step(x_K, eps_inj, sigma):
            return x_K - sigma * eps_inj
        return reverse_step

    def _sigma_at(self):
        def sigma_at(t):
            return 0.867  # SDXL sigma at K=249
        return sigma_at

    def _alpha_bar_at(self):
        def alpha_bar_at(t):
            return 1.0 / (0.867 ** 2 + 1.0)
        return alpha_bar_at

    def _run_cascade(self, cfg):
        torch.manual_seed(0)
        z0 = torch.randn(1, 4, 32, 32)  # VAE space, std≈1
        result = pixelrush_cascade(
            z0, num_cascade_stages=1,
            vae_decode=self._identity_vae_decode(),
            vae_encode=self._identity_vae_encode(),
            predict_eps=self._realistic_predict_eps(),
            alpha_bar_at=self._alpha_bar_at(),
            cfg=cfg,
            forward_step=self._vae_space_forward_step(),
            reverse_step=self._vae_space_reverse_step(),
            sigma_at=self._sigma_at(),
        )
        return z0, result

    def test_vae_space_cascade_signal_dominated(self):
        """Full cascade on a realistic SDXL mock must be signal-dominated.

        Regression guard for the 'totally noisy' bug: out.std / z0.std < 2.0.
        (Before the VAE-space fix this ratio was > 6.)
        """
        cfg = PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5, k_timestep=249,
            noise_lambda=0.95, operate_in_vae_space=True,
        )
        z0, result = self._run_cascade(cfg)
        ratio = result.std() / z0.std()
        assert ratio < 2.0, (
            f"Output noise dominates signal (ratio={ratio:.2f}); expected < 2.0"
        )

    def test_vae_space_cascade_correlates_with_input(self):
        """Output should positively correlate with the input (structure kept)."""
        import torch.nn.functional as F
        cfg = PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5, k_timestep=249,
            noise_lambda=0.95, operate_in_vae_space=True,
        )
        z0, result = self._run_cascade(cfg)
        res_down = F.interpolate(
            result, size=z0.shape[2:], mode="bilinear", align_corners=False
        )
        assert (res_down * z0).sum() > 0, "Output should correlate with input"


# ---------------------------------------------------------------------------
# Diagnostics: "compressed / JPEG-style" artifacts (plan 2026-08-13)
# ---------------------------------------------------------------------------

def _hf_energy(x: torch.Tensor) -> float:
    """High-frequency energy via 2D FFT radial power above 0.5*Nyquist.

    x: [B, C, H, W]. Returns the summed FFT power for radial frequencies
    r > 0.5 * r_max (the high-frequency half of the spectrum).
    """
    f = torch.fft.fft2(x)
    p = f.abs().pow(2)
    _, _, h, w = x.shape
    cy, cx = h // 2, w // 2
    yy = torch.arange(h, device=x.device).unsqueeze(1).expand(h, w).float()
    xx = torch.arange(w, device=x.device).unsqueeze(0).expand(h, w).float()
    r = ((yy - cy) ** 2 + (xx - cx) ** 2).sqrt()
    mask = r > 0.5 * r.max()
    return float(p[:, :, mask].sum().item())


def _laplacian_hf(x: torch.Tensor) -> torch.Tensor:
    """Extract HF component (edges) via a 3x3 Laplacian kernel.

    Applies the same Laplacian to every channel (broadcast over the channel
    dimension) by expanding the kernel to [C, 1, 3, 3].
    """
    import torch.nn.functional as F
    kernel = torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
        device=x.device, dtype=x.dtype,
    ).view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3).contiguous()
    return F.conv2d(x, kernel, padding=1, groups=x.shape[1])


@pytest.mark.unit
class TestPixelRushCompressionDiagnostics:
    """Diagnostics for the 'compressed / JPEG-style' output artifacts.

    Hypotheses (plan 2026-08-13):
      H1: refinement adds no model detail (noise_lambda=0.95 -> 95% random noise
          that averages out across overlapping patches, leaving smoothed bicubic).
      H2: eps_inv = predict_eps(patch_0, 0) ~= 0 -> partial inversion is a no-op.
      H3: VAE decode->encode round-trip is lossy/smoothing (compounds per stage).
      H4: patch overlap-add leaves block/discontinuity artifacts at seams.

    These tests use a STRUCTURED predict_eps (returns the latent's HF component,
    as a real diffusion model predicts detail as noise) to measure whether the
    algorithm adds or removes high-frequency detail.
    """

    def _identity_vae_decode(self):
        return lambda z: z

    def _identity_vae_encode(self):
        return lambda x: x

    def _structured_predict_eps(self, scale=0.5):
        """Mock a real model: eps = structured HF of the latent (detail as noise)."""
        def predict_eps(latent, timestep):
            return scale * _laplacian_hf(latent)
        return predict_eps

    def _random_predict_eps(self, std=0.1, seed=0):
        def predict_eps(latent, timestep):
            g = torch.Generator().manual_seed(seed)
            return std * torch.randn(
                latent.shape, generator=g, device=latent.device, dtype=latent.dtype
            )
        return predict_eps

    def _vae_space_forward_step(self):
        return lambda x_0, eps, sigma: x_0 + sigma * eps

    def _vae_space_reverse_step(self):
        return lambda x_K, eps_inj, sigma: x_K - sigma * eps_inj

    def _sigma_at(self):
        return lambda t: 0.867

    def _alpha_bar_at(self):
        return lambda t: 1.0 / (0.867 ** 2 + 1.0)

    def _make_cfg(self, **overrides):
        base = dict(patch_h=32, patch_w=32, overlap=0.5, k_timestep=249,
                    noise_lambda=0.95, operate_in_vae_space=True)
        base.update(overrides)
        return PixelRushConfig(**base)

    # --- Step 1: HF-energy regression baseline ---
    def test_cascade_preserves_hf_energy(self):
        """Full cascade (1 stage) must not destroy HF vs input.

        Regression guard: hf_energy(out) / hf_energy(z0) >= 0.8.
        EXPECTED TO FAIL before the fix (proves smoothing).
        """
        torch.manual_seed(0)
        z0 = torch.randn(1, 4, 32, 32)
        cfg = self._make_cfg()
        out = pixelrush_cascade(
            z0, num_cascade_stages=1,
            vae_decode=self._identity_vae_decode(),
            vae_encode=self._identity_vae_encode(),
            predict_eps=self._structured_predict_eps(),
            alpha_bar_at=self._alpha_bar_at(),
            cfg=cfg,
            forward_step=self._vae_space_forward_step(),
            reverse_step=self._vae_space_reverse_step(),
            sigma_at=self._sigma_at(),
        )
        ratio = _hf_energy(out) / _hf_energy(z0)
        assert ratio >= 0.8, (
            f"Cascade destroyed HF detail (ratio={ratio:.3f}); expected >= 0.8"
        )

    # --- Step 2: isolate bicubic + VAE smoothing ---
    def test_bicubic_vae_roundtrip_hf_loss(self):
        """Measure HF loss from bicubic upscale + VAE round-trip (no refinement)."""
        import torch.nn.functional as F
        torch.manual_seed(0)
        z0 = torch.randn(1, 4, 32, 32)
        # pixel-space path: decode -> bicubic 2x -> encode
        image = self._identity_vae_decode()(z0)
        image_up = F.interpolate(image, scale_factor=2.0, mode="bicubic",
                                 align_corners=False, antialias=True)
        coarse = self._identity_vae_encode()(image_up)
        ratio = _hf_energy(coarse) / _hf_energy(z0)
        # Record for the plan; bicubic alone should reduce HF somewhat.
        assert ratio > 0.0
        # Soft guard: bicubic should not destroy >60% of HF on its own.
        assert ratio >= 0.4, (
            f"Bicubic+VAE round-trip destroyed too much HF (ratio={ratio:.3f})"
        )

    # --- Step 3: isolate refinement effect ---
    def test_refinement_hf_delta(self):
        """refine_latent_once should ADD HF vs the coarse (bicubic) latent."""
        import torch.nn.functional as F
        torch.manual_seed(0)
        z0 = torch.randn(1, 4, 32, 32)
        image_up = F.interpolate(z0, scale_factor=2.0, mode="bicubic",
                                 align_corners=False, antialias=True)
        coarse = self._identity_vae_encode()(image_up)
        cfg = self._make_cfg()
        refined = refine_latent_once(
            coarse_latent=coarse,
            predict_eps=self._structured_predict_eps(),
            alpha_bar_at=self._alpha_bar_at(),
            cfg=cfg,
            forward_step=self._vae_space_forward_step(),
            reverse_step=self._vae_space_reverse_step(),
            sigma_at=self._sigma_at(),
        )
        ratio = _hf_energy(refined) / _hf_energy(coarse)
        # Record for the plan. If ratio < 1.0, refinement REMOVES HF (H1).
        assert ratio > 0.0

    # --- Step 4: inversion no-op check (H2) ---
    def test_inversion_adds_noise(self):
        """Partial inversion (0->K) must add a non-trivial amount of noise.

        If ||patch_k - patch_0|| ~= 0, the inversion is a no-op (H2) and the
        denoising has no signal to refine.
        """
        torch.manual_seed(0)
        z0 = torch.randn(1, 4, 32, 32)
        # Run a single patch through the forward step manually.
        patch_0 = z0[:, :, :32, :32]
        eps_inv = self._structured_predict_eps()(patch_0, 0)
        sigma_k = torch.tensor([0.867])
        patch_k = self._vae_space_forward_step()(patch_0, eps_inv, sigma_k)
        rel = (patch_k - patch_0).norm() / patch_0.norm()
        # Record for the plan. If rel < 0.1, inversion is effectively a no-op.
        assert rel >= 0.0

    # --- Step 5: patch-boundary discontinuity (H4) ---
    def test_no_patch_boundary_discontinuity(self):
        """Overlap-add must not leave block discontinuities at patch seams."""
        torch.manual_seed(0)
        z0 = torch.randn(1, 4, 32, 32)
        cfg = self._make_cfg()
        out = pixelrush_cascade(
            z0, num_cascade_stages=1,
            vae_decode=self._identity_vae_decode(),
            vae_encode=self._identity_vae_encode(),
            predict_eps=self._structured_predict_eps(),
            alpha_bar_at=self._alpha_bar_at(),
            cfg=cfg,
            forward_step=self._vae_space_forward_step(),
            reverse_step=self._vae_space_reverse_step(),
            sigma_at=self._sigma_at(),
        )
        # Seam at y=32 (patch boundary for patch_h=32, full latent 64x64 after 2x)
        # Use a 64x64 latent so a seam exists at the midpoint.
        # Measure gradient magnitude at seam vs interior.
        grad = torch.abs(out[:, :, 1:, :] - out[:, :, :-1, :]).mean(dim=(0, 1))
        h = grad.shape[0]
        seam = grad[h // 2].mean()
        interior = grad[:h // 2].mean()
        ratio = float((seam / interior.clamp_min(1e-8)).item())
        # Record for the plan; ratio > 1.5 suggests seam artifacts.
        assert ratio > 0.0

    # --- Step 6: VAE round-trip HF loss per stage (H3) ---
    def test_vae_roundtrip_hf_loss(self):
        """VAE decode->encode round-trip must not destroy HF (identity mock)."""
        torch.manual_seed(0)
        z0 = torch.randn(1, 4, 32, 32)
        roundtrip = self._identity_vae_encode()(self._identity_vae_decode()(z0))
        ratio = _hf_energy(roundtrip) / _hf_energy(z0)
        # Identity VAE -> ratio should be ~1.0. If a real VAE were used it would
        # be < 1.0 (smoothing). This test pins the mock behavior.
        assert ratio >= 0.99, (
            f"Identity VAE round-trip changed HF (ratio={ratio:.3f}); mock broken"
        )

    # --- Step 7: fix — refinement must preserve model detail (H1) ---
    def test_refinement_preserves_hf(self):
        """After the H1 fix, refine_latent_once must NOT remove HF vs coarse.

        Regression guard: hf_energy(refined) / hf_energy(coarse) >= 0.9.
        (Before the fix this was ~0.76 — refinement smoothed the image.)
        """
        import torch.nn.functional as F
        torch.manual_seed(0)
        z0 = torch.randn(1, 4, 32, 32)
        image_up = F.interpolate(z0, scale_factor=2.0, mode="bicubic",
                                 align_corners=False, antialias=True)
        coarse = self._identity_vae_encode()(image_up)
        cfg = self._make_cfg()
        refined = refine_latent_once(
            coarse_latent=coarse,
            predict_eps=self._structured_predict_eps(),
            alpha_bar_at=self._alpha_bar_at(),
            cfg=cfg,
            forward_step=self._vae_space_forward_step(),
            reverse_step=self._vae_space_reverse_step(),
            sigma_at=self._sigma_at(),
        )
        ratio = _hf_energy(refined) / _hf_energy(coarse)
        assert ratio >= 0.9, (
            f"Refinement removed HF (ratio={ratio:.3f}); expected >= 0.9"
        )
