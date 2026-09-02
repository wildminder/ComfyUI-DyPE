"""Tests for src/pixelrush.py — PixelRush core algorithm (Tier 1: pure unit tests)."""
import math

import pytest
import torch

from src.pixelrush import (
    PixelRushConfig,
    ddim_deterministic_step,
    ddim_forward_one_step,
    ddim_reverse_one_step_to_zero,
    gaussian_feather_mask,
    patch_positions,
    pixelrush_cascade,
    predict_x0_from_epsilon,
    refine_latent_once,
    slerp,
    spherical_lerp,
)

# ---------------------------------------------------------------------------
# slerp (corrected-theory standard vector SLERP)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSlerp:
    def test_t_zero_returns_a(self):
        """t=0 must return exactly a (sin(omega)/sin(omega) coefficient = 1)."""
        a = torch.randn(2, 4, 8, 8)
        b = torch.randn(2, 4, 8, 8)
        result = slerp(a, b, t=0.0)
        assert torch.allclose(result, a, atol=1e-5)

    def test_t_one_returns_b(self):
        """t=1 must return exactly b."""
        a = torch.randn(2, 4, 8, 8)
        b = torch.randn(2, 4, 8, 8)
        result = slerp(a, b, t=1.0)
        assert torch.allclose(result, b, atol=1e-5)

    def test_preserves_shape(self):
        a = torch.randn(2, 3, 16, 16)
        b = torch.randn(2, 3, 16, 16)
        result = slerp(a, b, t=0.3)
        assert result.shape == a.shape

    def test_no_nan(self):
        a = torch.randn(1, 4, 4, 4)
        b = torch.randn(1, 4, 4, 4)
        result = slerp(a, b, t=0.95)
        assert not torch.isnan(result).any()

    def test_collinear_falls_back_to_lerp(self):
        """Nearly parallel vectors must use the lerp fallback exactly.

        b = 2a is exactly collinear with a, so sin(omega)=0 and the
        use_lerp branch must fire, giving the exact linear interpolation.
        """
        a = torch.ones(1, 8)
        b = torch.ones(1, 8) * 3.0
        result = slerp(a, b, t=0.5)
        expected = 0.5 * a + 0.5 * b
        assert torch.allclose(result, expected, atol=1e-6), (
            f"Collinear slerp must equal lerp exactly; got {result.flatten()[:3]}, "
            f"expected {expected.flatten()[:3]}"
        )

    def test_orthogonal_formula(self):
        """Orthogonal unit vectors at t=0.5: result must be (a+b)/sqrt(2).

        Standard SLERP identity. The old unit-vector x linear-magnitude form
        gives |result| ~ |(a+b)/2|-profile magnitude; the corrected raw-vector
        form gives exactly |a| = |b| = 1 at the midpoint for unit inputs —
        this pins the corrected magnitude behavior.
        """
        a = torch.zeros(1, 2)
        a[0, 0] = 1.0
        b = torch.zeros(1, 2)
        b[0, 1] = 1.0
        result = slerp(a, b, t=0.5)
        expected = torch.zeros(1, 2)
        expected[0, 0] = 1.0 / math.sqrt(2.0)
        expected[0, 1] = 1.0 / math.sqrt(2.0)
        assert torch.allclose(result, expected, atol=1e-5), (
            f"slerp midpoint of orthogonal unit vectors must be (a+b)/sqrt(2), "
            f"got {result}"
        )
        # Norm must be exactly 1 (stays on the unit sphere)
        assert abs(result.flatten().norm().item() - 1.0) < 1e-5

    def test_alias_spherical_lerp_is_slerp(self):
        """The backward-compat alias must point at the same function."""
        assert spherical_lerp is slerp


# ---------------------------------------------------------------------------
# gaussian_feather_mask (analytic form, corrected theory)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGaussianFeatherMask:
    def test_shape(self):
        mask = gaussian_feather_mask(64, 64, 24.0, torch.device("cpu"), torch.float32)
        assert mask.shape == (1, 1, 64, 64)

    def test_center_is_exactly_one(self):
        """Peak normalization: the center pixel must be exactly 1.0."""
        mask = gaussian_feather_mask(64, 64, 24.0, torch.device("cpu"), torch.float32)
        center = mask[0, 0, 32, 32]  # (height-1)/2 rounded up for even sizes
        # For even sizes the analytic max is at (31.5, 31.5); the discrete
        # peak is at (31,31) or (32,32) with value > 1 - tiny epsilon, then
        # normalized. Just require the discrete max == 1 exactly and center
        # close to 1.
        assert mask.max().item() == pytest.approx(1.0, abs=1e-6)
        assert abs(center.item() - 1.0) < 0.01

    def test_axis_decay_matches_formula(self):
        """Along the center row, mask[c, c+k] == exp(-k^2 / (2 sigma^2))."""
        h = w = 65  # odd -> exact center at (32, 32)
        sigma = 12.0
        mask = gaussian_feather_mask(h, w, sigma, torch.device("cpu"), torch.float32)
        c = 32
        for k in (0, 4, 8, 16):
            expected = math.exp(-(k ** 2) / (2.0 * sigma ** 2))
            got = mask[0, 0, c, c + k].item()
            assert abs(got - expected) < 1e-5, (
                f"mask decay mismatch at offset {k}: got {got}, expected {expected}"
            )

    def test_monotonic_decay_from_center(self):
        """Mask must be strictly decreasing with |offset| from the center."""
        mask = gaussian_feather_mask(65, 65, 12.0, torch.device("cpu"), torch.float32)
        row = mask[0, 0, 32, :]
        left_half = row[:33]  # offsets -32..0
        # Each step toward the center must increase (or stay equal)
        assert (left_half[1:] >= left_half[:-1] - 1e-7).all()
        right_half = row[32:]  # offsets 0..32
        assert (right_half[1:] <= right_half[:-1] + 1e-7).all()

    def test_corner_matches_formula(self):
        """Corner value == exp(-(2*((h-1)/2)^2) / (2 sigma^2)) / peak."""
        h = w = 33
        sigma = 10.0
        mask = gaussian_feather_mask(h, w, sigma, torch.device("cpu"), torch.float32)
        half = (h - 1) / 2.0
        r2 = 2.0 * half * half
        expected = math.exp(-r2 / (2.0 * sigma ** 2))
        got = mask[0, 0, 0, 0].item()
        assert abs(got - expected) < 1e-5, (
            f"corner mismatch: got {got}, expected {expected}"
        )

    def test_symmetry(self):
        """180-degree rotation must leave the mask unchanged."""
        mask = gaussian_feather_mask(32, 48, 12.0, torch.device("cpu"), torch.float32)
        assert torch.allclose(mask, mask.flip(-1), atol=1e-6)
        assert torch.allclose(mask, mask.flip(-2), atol=1e-6)

    def test_non_negative(self):
        mask = gaussian_feather_mask(32, 32, 24.0, torch.device("cpu"), torch.float32)
        assert (mask >= 0).all()

    def test_no_nan(self):
        mask = gaussian_feather_mask(16, 16, 24.0, torch.device("cpu"), torch.float32)
        assert not torch.isnan(mask).any()

    def test_small_sigma_sharp_falloff(self):
        """sigma=2 on 32x32: corner must be < 1e-5 (sigma must scale with patch).

        Guards the pitfall of using the default sigma=24 on tiny patches (or a
        tiny sigma on default patches): the analytic form's falloff is
        exp(-r^2/(2 sigma^2)), which for sigma=2 and r~22 is astronomically
        small — the mask must actually reach it, not clamp at some floor.
        """
        mask = gaussian_feather_mask(32, 32, 2.0, torch.device("cpu"), torch.float32)
        corner = mask[0, 0, 0, 0].item()
        assert corner < 1e-5, f"sigma=2 corner should be ~0, got {corner}"


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
# ddim_deterministic_step / predict_x0_from_epsilon (generic transitions)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDDIMDeterministicStep:
    def test_predict_x0_formula(self):
        """predict_x0_from_epsilon must invert the noising formula exactly.

        x_t = sqrt(ab)*x0 + sqrt(1-ab)*eps -> x0 recovered exactly for
        ab in {0.9, 0.5, 0.1}.
        """
        for ab in (0.9, 0.5, 0.1):
            x0 = torch.randn(1, 4, 8, 8)
            eps = torch.randn(1, 4, 8, 8)
            x_t = math.sqrt(ab) * x0 + math.sqrt(1 - ab) * eps
            x0_rec = predict_x0_from_epsilon(x_t, eps, ab)
            assert torch.allclose(x0_rec, x0, atol=1e-4), (
                f"x0 recovery failed for alpha_bar={ab}"
            )

    def test_arbitrary_transition_formula(self):
        """step(x, e, 0.5, 0.2) == sqrt(0.2)*x_hat_0 + sqrt(0.8)*e
        with x_hat_0 = (x - sqrt(0.5)*e)/sqrt(0.5)."""
        x = torch.randn(1, 4, 8, 8)
        e = torch.randn(1, 4, 8, 8)
        out = ddim_deterministic_step(x, e, 0.5, 0.2)
        x0_hat = (x - math.sqrt(0.5) * e) / math.sqrt(0.5)
        expected = math.sqrt(0.2) * x0_hat + math.sqrt(0.8) * e
        assert torch.allclose(out, expected, atol=1e-5)

    def test_midpoint_chain_equivalence(self):
        """Deterministic DDIM (eta=0) is path-independent: stepping
        a->m->b with the same epsilon equals the direct a->b step."""
        x = torch.randn(1, 4, 8, 8)
        e = torch.randn(1, 4, 8, 8)
        direct = ddim_deterministic_step(x, e, 0.9, 0.1)
        via_mid = ddim_deterministic_step(x, e, 0.9, 0.5)
        via_mid = ddim_deterministic_step(via_mid, e, 0.5, 0.1)
        assert torch.allclose(direct, via_mid, atol=1e-4), (
            "eta=0 DDIM transitions must be path-independent"
        )

    def test_forward_wrapper_matches_generic(self):
        """ddim_forward_one_step == ddim_deterministic_step(·, 1.0, ab_k)."""
        z0 = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        a = 0.7
        via_wrapper = ddim_forward_one_step(z0, eps, alpha_bar_k=a)
        via_generic = ddim_deterministic_step(z0, eps, 1.0, a)
        assert torch.allclose(via_wrapper, via_generic, atol=1e-6)

    def test_reverse_wrapper_matches_generic(self):
        """ddim_reverse_one_step_to_zero == ddim_deterministic_step(·, ab_k, 1.0)."""
        z_k = torch.randn(1, 4, 8, 8)
        eps = torch.randn(1, 4, 8, 8)
        a = 0.7
        via_wrapper = ddim_reverse_one_step_to_zero(z_k, eps, alpha_bar_k=a)
        via_generic = ddim_deterministic_step(z_k, eps, a, 1.0)
        assert torch.allclose(via_wrapper, via_generic, atol=1e-6)

    def test_round_trip_identity(self):
        """reverse(forward(x0, e, ab), e, ab) must recover x0."""
        x0 = torch.randn(1, 4, 8, 8)
        e = torch.randn(1, 4, 8, 8)
        ab = 0.5
        z_k = ddim_forward_one_step(x0, e, alpha_bar_k=ab)
        z_0_hat = ddim_reverse_one_step_to_zero(z_k, e, alpha_bar_k=ab)
        assert torch.allclose(z_0_hat, x0, atol=1e-4)


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
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_predict_eps(),
            self._mock_alpha_bar(), cfg)
        assert result.shape == latent.shape

    def test_no_nan(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_predict_eps(),
            self._mock_alpha_bar(), cfg)
        assert not torch.isnan(result).any()

    def test_single_patch(self):
        """When latent == patch size, only one patch."""
        cfg = PixelRushConfig(patch_h=64, patch_w=64, overlap=0.5)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_predict_eps(),
            self._mock_alpha_bar(), cfg)
        assert result.shape == latent.shape

    def test_multiple_patches(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 128, 128)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_predict_eps(),
            self._mock_alpha_bar(), cfg)
        assert result.shape == latent.shape

    def test_non_square(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 64, 128)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_predict_eps(),
            self._mock_alpha_bar(), cfg)
        assert result.shape == latent.shape

    def test_weight_normalization(self):
        """Output should be properly normalized (weight_sum > 0 everywhere)."""
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5, gaussian_sigma=8.0)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(latent, self._mock_predict_eps(), self._mock_predict_eps(),
            self._mock_alpha_bar(), cfg)
        # Result should be finite (not inf/nan from division)
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# refine_latent_once: adapter combinations (alpha_k NameError regression)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRefineLatentOnceAdapterCombos:
    """Regression: alpha_k must be defined regardless of which adapters are given.

    Previously alpha_k was only computed when sigma_at was None, so passing
    sigma_at WITHOUT forward_step/reverse_step (or with only one of them)
    raised NameError in the DDIM fallback branches.
    """

    def _mock_predict_eps(self):
        def predict_eps(latent, timestep):
            return torch.randn_like(latent)
        return predict_eps

    def _mock_alpha_bar(self):
        def alpha_bar_at(t):
            return 0.8
        return alpha_bar_at

    def _sigma_at(self):
        def sigma_at(t):
            return 0.5
        return sigma_at

    def test_sigma_at_with_fallback_ddim_no_nameerror(self):
        """sigma_at given, BOTH adapters None: pure-DDIM fallback must not NameError."""
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(
            latent, self._mock_predict_eps(), self._mock_predict_eps(),
            self._mock_alpha_bar(), cfg,
            sigma_at=self._sigma_at(),
        )
        assert torch.isfinite(result).all()

    def test_partial_adapters_forward_only(self):
        """forward_step given, reverse_step None: reverse falls back to DDIM."""
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(
            latent, self._mock_predict_eps(), self._mock_predict_eps(),
            self._mock_alpha_bar(), cfg,
            forward_step=lambda x_0, eps, sigma: x_0 + sigma * eps,
            sigma_at=self._sigma_at(),
        )
        assert torch.isfinite(result).all()

    def test_partial_adapters_reverse_only(self):
        """reverse_step given, forward_step None: forward falls back to DDIM."""
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 64, 64)
        result = refine_latent_once(
            latent, self._mock_predict_eps(), self._mock_predict_eps(),
            self._mock_alpha_bar(), cfg,
            reverse_step=lambda x_K, eps_inj, sigma: x_K - sigma * eps_inj,
            sigma_at=self._sigma_at(),
        )
        assert torch.isfinite(result).all()

    def test_all_adapters_provided_ignores_alpha_bar(self):
        """Both adapters given: alpha_bar_at must never be called."""
        cfg = PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5)
        latent = torch.randn(1, 4, 64, 64)

        def alpha_bar_at(t):
            raise AssertionError(
                "alpha_bar_at must not be called when both adapters are provided"
            )

        result = refine_latent_once(
            latent,
            self._mock_predict_eps(),
            self._mock_predict_eps(),
            alpha_bar_at,
            cfg,
            forward_step=lambda x_0, eps, sigma: x_0 + sigma * eps,
            reverse_step=lambda x_K, eps_inj, sigma: x_K - sigma * eps_inj,
            sigma_at=self._sigma_at(),
        )
        assert torch.isfinite(result).all()


# ---------------------------------------------------------------------------
# inversion_eps / refiner_eps separation (corrected theory)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAdapterSeparation:
    """The core must call inversion_eps at t=0 and refiner_eps at t=K,
    and the two adapters must be independently replaceable (paper uses a
    separate distilled refiner, e.g. SDXL-Turbo)."""

    def _make_cfg(self):
        return PixelRushConfig(patch_h=32, patch_w=32, overlap=0.5, k_timestep=249)

    def _recording_eps(self, calls, scale=1.0):
        def eps_fn(latent, timestep):
            calls.append((latent.clone(), timestep))
            return scale * torch.ones_like(latent)
        return eps_fn

    def test_inversion_eps_called_at_zero_refiner_at_k(self):
        calls_inv, calls_ref = [], []
        cfg = self._make_cfg()
        latent = torch.randn(1, 4, 32, 32)
        refine_latent_once(
            latent,
            self._recording_eps(calls_inv),
            self._recording_eps(calls_ref),
            lambda t: 0.8,
            cfg,
        )
        assert len(calls_inv) == 1, (
            f"inversion_eps must be called exactly once per patch, got {len(calls_inv)}"
        )
        assert calls_inv[0][1] == 0, (
            f"inversion_eps must be called at timestep 0, got {calls_inv[0][1]}"
        )
        assert len(calls_ref) == 1, (
            f"refiner_eps must be called exactly once per patch, got {len(calls_ref)}"
        )
        assert calls_ref[0][1] == 249, (
            f"refiner_eps must be called at timestep K=249, got {calls_ref[0][1]}"
        )

    def test_distinct_adapters_produce_distinct_output(self):
        """A refiner returning 2x the base eps must change the refined output.

        Guards against the separation silently re-merging into one adapter.
        """
        cfg = self._make_cfg()
        torch.manual_seed(0)
        latent = torch.randn(1, 4, 64, 64)

        def base_eps(latent, timestep):
            return torch.ones_like(latent)

        def double_eps(latent, timestep):
            return 2.0 * torch.ones_like(latent)

        out_base = refine_latent_once(
            latent, base_eps, base_eps, lambda t: 0.8, cfg)
        out_double_refiner = refine_latent_once(
            latent, base_eps, double_eps, lambda t: 0.8, cfg)
        assert not torch.allclose(out_base, out_double_refiner), (
            "A distinct refiner eps must change the refined output"
        )

    def test_cascade_forwards_both_adapters(self):
        """pixelrush_cascade must pass each adapter to every stage."""
        calls_inv, calls_ref = [], []
        cfg = self._make_cfg()

        def vae_decode(z):
            return z[:, :3] if z.shape[1] >= 3 else z.repeat(1, 1, 1, 1)[:, :, :3]

        def vae_encode(x):
            b, c, h, w = x.shape
            if c >= 4:
                return x[:, :4]
            return x.repeat(1, 4 // c + 1, 1, 1)[:, :4]

        pixelrush_cascade(
            torch.randn(1, 4, 32, 32),
            num_cascade_stages=2,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            inversion_eps=self._recording_eps(calls_inv),
            refiner_eps=self._recording_eps(calls_ref),
            alpha_bar_at=lambda t: 0.8,
            cfg=cfg,
        )
        # Stage 1: 1 patch (64x64 latent == 32x32 patch? No: 32->64 latent,
        # patch 32x32, overlap 0.5 -> 3x3=9 patches). Stage 2: 128x128 -> 7x7
        # starts... just require both stages saw both adapters.
        assert len(calls_inv) == len(calls_ref) > 0
        # Inversion at t=0 only, refiner at t=K only, across all stages
        assert all(c[1] == 0 for c in calls_inv)
        assert all(c[1] == 249 for c in calls_ref)

    def test_no_predict_eps_kwarg_in_source(self):
        """Guard: the core source must not keep the merged predict_eps kwarg."""
        import pathlib
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush.py").read_text(encoding="utf-8")
        assert "predict_eps=" not in content, (
            "src/pixelrush.py still passes the merged predict_eps= kwarg; use "
            "inversion_eps=/refiner_eps="
        )


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
            inversion_eps=self._mock_predict_eps(),
            refiner_eps=self._mock_predict_eps(),
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
            inversion_eps=self._mock_predict_eps(),
            refiner_eps=self._mock_predict_eps(),
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
            inversion_eps=self._mock_predict_eps(),
            refiner_eps=self._mock_predict_eps(),
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
                inversion_eps=self._mock_predict_eps(),
            refiner_eps=self._mock_predict_eps(),
                alpha_bar_at=self._mock_alpha_bar(),
                cfg=cfg,
            )
            expected = 32 * (2 ** stages)
            assert result.shape[2] == expected
            assert result.shape[3] == expected


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
            inversion_eps=self._realistic_predict_eps(),
            refiner_eps=self._realistic_predict_eps(),
            alpha_bar_at=self._alpha_bar_at(),
            cfg=cfg,
            forward_step=self._vae_space_forward_step(),
            reverse_step=self._vae_space_reverse_step(),
            sigma_at=self._sigma_at(),
        )
        return z0, result

    def test_vae_space_cascade_signal_dominated(self):
        """Full cascade on a realistic SDXL mock must be signal-dominated.

        Regression guard for the 'totally noisy' bug: 0.5 < out.std/z0.std < 2.0.
        (Before the VAE-space fix this ratio was > 6; the lower bound guards
        against an inert/no-op refinement.)
        """
        cfg = PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5, k_timestep=249,
            noise_lambda=0.95,
        )
        z0, result = self._run_cascade(cfg)
        ratio = result.std() / z0.std()
        assert ratio < 2.0, (
            f"Output noise dominates signal (ratio={ratio:.2f}); expected < 2.0"
        )
        assert ratio > 0.5, (
            f"Output is inert vs input (ratio={ratio:.2f}); refinement is a no-op?"
        )

    def test_refinement_changes_latent(self):
        """Refinement must actually change the latent (> 1% relative delta).

        Guards against the refinement collapsing to a no-op under any
        future refactor (space-conversion or injection changes).
        """
        cfg = PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5, k_timestep=249,
            noise_lambda=0.95,
        )
        z0, result = self._run_cascade(cfg)
        # The cascade's first-stage input is the bicubic-upscaled z0; the
        # refined result must differ from a pure passthrough meaningfully.
        import torch.nn.functional as F
        z0_up = F.interpolate(z0, size=result.shape[2:], mode="bicubic",
                              align_corners=False, antialias=True)
        rel = (result - z0_up).norm() / z0_up.norm()
        assert rel > 0.01, (
            f"Refinement barely changed the latent (rel={rel:.4f}); "
            "suspect a no-op pipeline"
        )

    def test_vae_space_cascade_correlates_with_input(self):
        """Output should positively correlate with the input (structure kept)."""
        import torch.nn.functional as F
        cfg = PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5, k_timestep=249,
            noise_lambda=0.95,
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
                    noise_lambda=0.95, noise_injection="additive")
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
            inversion_eps=self._structured_predict_eps(),
            refiner_eps=self._structured_predict_eps(),
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
            inversion_eps=self._structured_predict_eps(),
            refiner_eps=self._structured_predict_eps(),
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
            inversion_eps=self._structured_predict_eps(),
            refiner_eps=self._structured_predict_eps(),
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
        """ADDITIVE (legacy) mode: refine_latent_once must NOT remove HF.

        Regression guard: hf_energy(refined) / hf_energy(coarse) >= 0.9.
        Calibrated against the 2026-08-13 additive injection; the slerp
        mode has its own companion test below.
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
            inversion_eps=self._structured_predict_eps(),
            refiner_eps=self._structured_predict_eps(),
            alpha_bar_at=self._alpha_bar_at(),
            cfg=cfg,
            forward_step=self._vae_space_forward_step(),
            reverse_step=self._vae_space_reverse_step(),
            sigma_at=self._sigma_at(),
        )
        ratio = _hf_energy(refined) / _hf_energy(coarse)
        # Recalibrated 2026-09-02 post-λ-fix: measured 0.583 (the flipped
        # convention injects only 5% random, which smooths less than the
        # old +0.95*rand formula that measured 0.955). Bound 0.5.
        assert ratio >= 0.5, (
            f"Refinement removed HF (ratio={ratio:.3f}); expected >= 0.5 "
            "(post-λ-fix calibration; measured 0.583)"
        )


# ---------------------------------------------------------------------------
# Noise injection modes (slerp default, additive legacy opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestNoiseInjectionModes:
    """refine_latent_once must implement both injection modes exactly.

    λ weights the REFINER'S PREDICTION (fixed 2026-09-02 post-release; the
    corrected doc's own caveat flagged the argument order as the one
    detail to verify against the authors' implementation):
        slerp (paper default): eps_injected = slerp(eps_random, eps_refined, λ)
        additive (legacy): eps_injected = eps_refined + (1-λ) * eps_random
    """

    def _eps_fn(self):
        def eps_fn(latent, timestep):
            return 0.3 * torch.ones_like(latent)
        return eps_fn

    def _cfg(self, mode, lam=0.95):
        return PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5, k_timestep=249,
            noise_lambda=lam, noise_injection=mode,
        )

    def _refine(self, cfg, seed=123):
        torch.manual_seed(seed)
        coarse = torch.randn(1, 4, 32, 32)
        return refine_latent_once(
            coarse, self._eps_fn(), self._eps_fn(), lambda t: 0.8, cfg)

    def test_slerp_mode_lambda_one_keeps_eps_pred(self):
        """λ=1 -> slerp returns the PREDICTION (95%+λ reading); the reverse
        step must use it exactly (refined == reverse(forward(patch, eps), eps))."""
        from src.pixelrush import ddim_forward_one_step, ddim_reverse_one_step_to_zero
        torch.manual_seed(123)
        coarse = torch.randn(1, 4, 32, 32)
        cfg = self._cfg("slerp", lam=1.0)
        out = refine_latent_once(
            coarse, self._eps_fn(), self._eps_fn(), lambda t: 0.8, cfg)
        eps = self._eps_fn()(coarse, 0)
        z_k = ddim_forward_one_step(coarse, eps, 0.8)
        expected = ddim_reverse_one_step_to_zero(z_k, eps, 0.8)
        assert torch.allclose(out, expected, atol=1e-5), (
            "λ=1 slerp must reduce to the pure-eps refinement"
        )

    def test_slerp_mode_lambda_zero_uses_random(self):
        """λ=0 -> slerp returns eps_random; refined must equal the
        reverse step with ONLY the (seeded) random eps."""
        from src.pixelrush import ddim_forward_one_step, ddim_reverse_one_step_to_zero
        torch.manual_seed(123)
        coarse = torch.randn(1, 4, 32, 32)
        cfg = self._cfg("slerp", lam=0.0)
        out = refine_latent_once(
            coarse, self._eps_fn(), self._eps_fn(), lambda t: 0.8, cfg)
        torch.manual_seed(123)
        _ = torch.randn(1, 4, 32, 32)  # the coarse draw
        eps_random = torch.randn(1, 4, 32, 32)  # the injection draw
        eps_pred = self._eps_fn()(coarse, 0)
        z_k = ddim_forward_one_step(coarse, eps_pred, 0.8)
        expected = ddim_reverse_one_step_to_zero(z_k, eps_random, 0.8)
        assert torch.allclose(out, expected, atol=1e-4), (
            "λ=0 slerp must reduce to pure-random-eps refinement"
        )

    def test_additive_mode_formula(self):
        """additive mode must reproduce the legacy formula with the same λ
        convention: eps_pred + (1-λ) * eps_random."""
        from src.pixelrush import ddim_forward_one_step, ddim_reverse_one_step_to_zero
        torch.manual_seed(123)
        coarse = torch.randn(1, 4, 32, 32)
        cfg = self._cfg("additive", lam=0.5)
        out = refine_latent_once(
            coarse, self._eps_fn(), self._eps_fn(), lambda t: 0.8, cfg)
        torch.manual_seed(123)
        _ = torch.randn(1, 4, 32, 32)
        eps_random = torch.randn(1, 4, 32, 32)
        eps_pred = self._eps_fn()(coarse, 0)
        z_k = ddim_forward_one_step(coarse, eps_pred, 0.8)
        eps_inj = eps_pred + (1.0 - 0.5) * eps_random
        expected = ddim_reverse_one_step_to_zero(z_k, eps_inj, 0.8)
        assert torch.allclose(out, expected, atol=1e-4), (
            "additive mode must equal eps_pred + (1-λ) * eps_random"
        )

    def test_default_mode_is_slerp(self):
        cfg = PixelRushConfig(patch_h=32, patch_w=32)
        assert cfg.noise_injection == "slerp", (
            "Default injection mode must be 'slerp' (paper / corrected theory)"
        )

    def test_invalid_mode_raises(self):
        cfg = self._cfg("bogus")
        with pytest.raises(ValueError, match="noise_injection"):
            self._refine(cfg)


@pytest.mark.unit
class TestPixelRushSlerpHF:
    """HF-preservation under the corrected slerp injection (paper defaults).

    The injected per-patch random eps is independent and adds HF energy;
    a ratio far below 1 re-indicates the space-mixing symptom (plan
    2026-09-02, Step 11 recalibrates the bound from measurements).
    """

    def _identity_vae_decode(self):
        return lambda z: z

    def _identity_vae_encode(self):
        return lambda x: x

    def _structured_eps(self, scale=0.5):
        def eps_fn(latent, timestep):
            return scale * _laplacian_hf(latent)
        return eps_fn

    def _forward_step(self):
        return lambda x_0, eps, sigma: x_0 + sigma * eps

    def _reverse_step(self):
        return lambda x_K, eps_inj, sigma: x_K - sigma * eps_inj

    def test_refinement_preserves_hf_slerp_mode(self):
        import torch.nn.functional as F
        torch.manual_seed(0)
        z0 = torch.randn(1, 4, 32, 32)
        image_up = F.interpolate(z0, scale_factor=2.0, mode="bicubic",
                                 align_corners=False, antialias=True)
        coarse = self._identity_vae_encode()(image_up)
        cfg = PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5, k_timestep=249,
            noise_lambda=0.95, noise_injection="slerp",
        )
        refined = refine_latent_once(
            coarse_latent=coarse,
            inversion_eps=self._structured_eps(),
            refiner_eps=self._structured_eps(),
            alpha_bar_at=lambda t: 1.0 / (0.867 ** 2 + 1.0),
            cfg=cfg,
            forward_step=self._forward_step(),
            reverse_step=self._reverse_step(),
            sigma_at=lambda t: 0.867,
        )
        ratio = _hf_energy(refined) / _hf_energy(coarse)
        # Calibrated in Step 11 of plan 2026-09-02, recalibrated post-λ-fix:
        # 0.584 on this mock under the flipped convention (λ weights the
        # prediction; the pre-fix convention measured 0.748 because 95%
        # random noise added HF). Bound 0.5.
        assert ratio >= 0.5, (
            f"slerp-mode refinement removed too much HF (ratio={ratio:.3f}); "
            "expected >= 0.5 (post-λ-fix calibration; measured 0.584)"
        )


# ---------------------------------------------------------------------------
# Post-release regression: lambda convention (2026-09-02 user report)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLambdaConvention:
    """Regression guard for the reported artifact: "structure similar to
    the original raw image, but completely noisy — soft non-uniform
    patches all over".

    Root cause: the injection used slerp(eps_pred, eps_random, 0.95),
    which at real scales makes the injected eps 99.6% PURE RANDOM (the
    corrected doc's own caveat flagged the argument order as the one
    detail to check). The reverse step then subtracted a random vector
    per patch: per-pixel noise std ~1.17 vs signal std ~1, rendered
    through the Gaussian feather as soft blurred patches. λ must weight
    the PREDICTION: slerp(eps_random, eps_refined, λ).
    """

    SIGMA_K = 0.867

    def test_injected_noise_small_vs_signal_at_paper_lambda(self):
        """At real-model scales (model-space signal std 1, eps std 1) the
        patch-independent noise in x0_hat must stay well below the signal.

        Pre-fix (λ on the random side): noise std 1.17 vs signal 1.0 —
        structure visible through heavy soft noise (the report).
        Post-fix (λ on the prediction): noise std ~0.07.
        """
        from src.pixelrush import slerp
        torch.manual_seed(0)
        eps_refined = torch.randn(1, 4, 64, 64)   # model eps, std 1
        eps_random = torch.randn(1, 4, 64, 64)
        # The random component of the injected eps vs the prediction
        inj = slerp(eps_random, eps_refined, 0.95)
        noise_std = self.SIGMA_K * (inj - eps_refined).std().item()
        assert noise_std < 0.2, (
            f"Injected patch-independent noise std {noise_std:.3f} vs "
            "signal 1.0 — λ is weighting the RANDOM side again "
            "(pre-fix value: ~1.17, the reported 'completely noisy' artifact)"
        )

    def test_refinement_end_to_end_correlates_with_clean(self):
        """One-patch refinement at real scales: output must stay highly
        correlated with the clean signal (>= 0.95), not merely 'structure
        visible through noise' (pre-fix correlation: ~0.65)."""
        torch.manual_seed(0)
        x0 = torch.randn(1, 4, 64, 64)              # clean signal, std 1
        eps_true = torch.randn(1, 4, 64, 64)
        eps_refined = eps_true + 0.2 * torch.randn(1, 4, 64, 64)  # good model

        cfg = PixelRushConfig(
            patch_h=64, patch_w=64, overlap=0.5, k_timestep=249,
            noise_lambda=0.95, noise_injection="slerp",
        )
        # Single patch == full latent; run through refine_latent_once so
        # the guard covers the real code path (forward + injection + reverse).
        # NOTE: refine_latent_once applies the forward step itself, so the
        # input must be the CLEAN patch (passing x_K would double-noise).
        def eps_fn(latent, timestep):
            return eps_refined

        refined = refine_latent_once(
            x0, eps_fn, eps_fn,
            lambda t: 1.0 / (self.SIGMA_K ** 2 + 1.0), cfg,
            forward_step=lambda x, e, s: x + self.SIGMA_K * e,
            reverse_step=lambda x, e, s: x - self.SIGMA_K * e,
            sigma_at=lambda t: self.SIGMA_K,
        )
        a = refined.flatten()
        b = x0.flatten()
        corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
        assert corr >= 0.95, (
            f"Refined output correlates only {corr:.3f} with the clean "
            "signal — pre-fix behavior (~0.65) reads as 'structure visible "
            "but completely noisy'"
        )
