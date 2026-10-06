"""Tests for src/imax.py — I-Max core engine (Tier 1: pure unit tests).

Phases 1-3 of plan 2026-10-05:
- Haar low-pass projection P (D7) — the pywt-oracle test is the fidelity
  tripwire against a real Haar wavedec2/waverec2 round trip;
- flow sigma schedules, cosine decay factor, x0-space Projected Flow (D6)
  and the low-resolution pass size (D10);
- the dual-pass orchestration (D14).
"""

import math

import pytest
import torch

from src.imax import (
    build_flow_sigmas,
    cosine_factor,
    haar_lowpass,
    low_res_size,
    projected_flow_x0,
)


# ---------------------------------------------------------------------------
# Haar low-pass projection (plan D7, paper §2.2 projection P)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHaarLowpass:
    def test_haar_level1_is_two_tap_box_average(self):
        """Level 1: every 2x2 block is replaced by its mean, broadcast."""
        x = torch.arange(16.0).reshape(1, 1, 4, 4)
        out = haar_lowpass(x, level=1)
        expected = torch.tensor([
            [2.5, 2.5, 4.5, 4.5],
            [2.5, 2.5, 4.5, 4.5],
            [10.5, 10.5, 12.5, 12.5],
            [10.5, 10.5, 12.5, 12.5],
        ]).reshape(1, 1, 4, 4)
        assert out.shape == x.shape
        assert torch.allclose(out, expected)

    def test_haar_level2_is_four_tap_box_average(self):
        """Level 2: the whole 4x4 input is one block — its overall mean."""
        x = torch.arange(16.0).reshape(1, 1, 4, 4)
        out = haar_lowpass(x, level=2)
        assert torch.allclose(out, torch.full_like(x, 7.5))

    @pytest.mark.parametrize("h,w", [(7, 9), (8, 8), (1, 1), (5, 1)])
    def test_haar_shape_preserved_for_odd_dims(self, h, w):
        x = torch.randn(2, 3, h, w)
        out = haar_lowpass(x, level=1)
        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_haar_preserves_dtype_and_device(self):
        for dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
            x = torch.randn(1, 1, 6, 6, dtype=dtype)
            out = haar_lowpass(x, level=1)
            assert out.dtype == dtype
            assert out.device == x.device

    def test_haar_is_identity_on_constant_field(self):
        """A constant field has zero detail energy — P must be the identity."""
        x = torch.full((1, 2, 5, 7), 3.25)
        assert torch.allclose(haar_lowpass(x, level=1), x)
        assert torch.allclose(haar_lowpass(x, level=2), x)

    def test_haar_level0_raises(self):
        with pytest.raises(ValueError, match="level"):
            haar_lowpass(torch.randn(1, 1, 4, 4), level=0)

    def test_haar_negative_level_raises(self):
        with pytest.raises(ValueError, match="level"):
            haar_lowpass(torch.randn(1, 1, 4, 4), level=-1)

    def test_haar_matches_pywt_reference(self):
        """Zeroed-details Haar round trip (PyWavelets) == torch box average.

        Boundary modes differ (circular wrap here, pywt's symmetric there),
        so the comparison crops 2**L pixels from each edge — interior blocks
        contain no boundary pixels and must agree exactly for a linear
        orthonormal transform. pywt is a TEST-ONLY oracle (plan D7): it must
        never appear in src/ or requirements.txt.
        """
        pytest.importorskip("pywt")
        import numpy as np
        import pywt

        torch.manual_seed(0)
        x = torch.randn(2, 3, 16, 16)
        for level in (1, 2):
            k = 2 ** level
            out = haar_lowpass(x, level=level)
            ref = np.empty_like(x.numpy())
            for b in range(x.shape[0]):
                for c in range(x.shape[1]):
                    arr = x[b, c].numpy()
                    coeffs = pywt.wavedec2(arr, wavelet="haar", level=level)
                    zeroed = [coeffs[0]] + [
                        tuple(np.zeros_like(a) for a in detail)
                        for detail in coeffs[1:]
                    ]
                    rec = pywt.waverec2(zeroed, wavelet="haar")
                    ref[b, c] = rec[: arr.shape[0], : arr.shape[1]]
            ref_t = torch.from_numpy(ref)
            interior = (slice(k, -k), slice(k, -k))
            assert torch.allclose(
                out[..., interior[0], interior[1]],
                ref_t[..., interior[0], interior[1]],
                atol=1e-5,
            ), f"haar_lowpass diverges from the pywt oracle at level {level}"

    def test_haar_does_not_import_comfy(self):
        """Engine purity guard: src.imax must stay comfy-free (plan D2)."""
        import src.imax as imax_module

        leaked = [name for name in vars(imax_module) if "comfy" in name.lower()]
        assert leaked == []


# ---------------------------------------------------------------------------
# Flow sigma schedules (plan D3 — closed form, no model_sampling patch)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFlowSigmas:
    def test_sigmas_length_is_steps_plus_one(self):
        for steps in (1, 2, 20, 30):
            assert build_flow_sigmas(steps, 3.0).numel() == steps + 1

    def test_sigmas_end_at_zero(self):
        assert build_flow_sigmas(20, 6.0)[-1].item() == 0.0

    def test_sigmas_are_strictly_decreasing(self):
        sig = build_flow_sigmas(30, 3.0)
        assert bool((sig[:-1] > sig[1:]).all())

    def test_sigmas_start_at_one(self):
        """t=1 gives shift*1/(1+(shift-1)*1) = 1 for ANY shift."""
        assert build_flow_sigmas(30, 3.0)[0].item() == pytest.approx(1.0)
        assert build_flow_sigmas(30, 6.0)[0].item() == pytest.approx(1.0)

    def test_shift_one_is_identity_linspace(self):
        steps = 10
        sig = build_flow_sigmas(steps, 1.0)
        expected = torch.linspace(1.0, 1.0 / steps, steps)
        assert torch.allclose(sig[:-1], expected.to(torch.float32))
        assert sig[-1].item() == 0.0

    def test_shift_is_monotonic_in_shift(self):
        """For fixed t < 1 the shifted sigma grows with the shift."""
        previous = -1.0
        for shift in (1.0, 1.5, 3.0, 6.0):
            current = build_flow_sigmas(2, shift)[1].item()  # t = 0.5
            assert current > previous
            previous = current

    def test_sigmas_match_flux_time_shift_closed_form(self):
        """Oracle: ComfyUI's flux_time_shift(mu=ln(s), 1.0, t)
        = exp(mu)/(exp(mu) + (1/t − 1)) (comfy/model_sampling.py:417,431-432)
        — the same function ModelSamplingFlux.sigma() applies."""
        for shift in (1.15, 3.0, 6.0):
            sig = build_flow_sigmas(8, shift)
            mu = math.log(shift)
            for i in range(8):
                t = 1.0 - i / 8.0
                oracle = math.exp(mu) / (math.exp(mu) + (1.0 / t - 1.0))
                assert sig[i].item() == pytest.approx(oracle, rel=1e-5), (
                    f"shift={shift}, i={i}"
                )

    def test_sigmas_match_comfy_simple_scheduler_ordering(self):
        """Shared contract with ComfyUI's index-sampled 'simple' grid
        (samplers.py:645-651): one interior sigma per step, strictly
        descending, entry 1.0, terminal exact 0.0."""
        sig = build_flow_sigmas(20, 3.0)
        interior = sig[:-1]
        assert interior.numel() == 20
        assert interior[0].item() == pytest.approx(1.0)
        assert bool((interior[1:] < interior[:-1]).all())
        assert sig[-1].item() == 0.0


# ---------------------------------------------------------------------------
# Cosine decay factor (reference pipeline_flux_imax.py:808)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCosineFactor:
    def test_cosine_factor_is_one_at_first_step(self):
        assert cosine_factor(0, 20) == pytest.approx(1.0)

    def test_cosine_factor_is_zero_at_last_step(self):
        """The reference form 0.5*(1+cos(pi*i/N)) never reaches exactly 0 at
        i = N-1 — it decays as pi^2/(4N^2). Pin the exact reference value
        and the operative bound for the default steps_high=20 regime."""
        for n in (20, 30):
            expected = 0.5 * (1.0 + math.cos(math.pi * (n - 1) / n))
            assert cosine_factor(n - 1, n) == pytest.approx(expected)
            assert cosine_factor(n - 1, n) < 0.01

    def test_cosine_factor_is_monotonic(self):
        values = [cosine_factor(i, 20) for i in range(20)]
        assert all(a >= b for a, b in zip(values, values[1:]))

    def test_cosine_factor_midpoint_is_half(self):
        assert cosine_factor(10, 20) == pytest.approx(0.5)

    def test_cosine_factor_total_steps_one(self):
        """Single step == the first step: full guidance on the only
        transition (reference cos(pi*0/1) = 1)."""
        assert cosine_factor(0, 1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Projected Flow in x0 space (plan D6 — hand-computed tables)
# ---------------------------------------------------------------------------

def _hand_case() -> tuple[torch.Tensor, torch.Tensor]:
    """x0 with P(x0) = 1.0 and guidance with P(G) = 3.0 (2x2, level 1:
    the whole field is one block, P = the block mean broadcast)."""
    x0 = torch.tensor([[[[0.0, 0.0], [0.0, 4.0]]]])
    guidance = torch.full((1, 1, 2, 2), 3.0)
    return x0, guidance


@pytest.mark.unit
class TestProjectedFlowX0:
    def test_disable_is_identity(self):
        x0, guidance = _hand_case()
        out = projected_flow_x0(x0, x0, guidance, 0.5, 0.5, "disable", 1)
        assert torch.equal(out, x0)

    def test_cosine_decay_matches_hand_computed(self):
        x0, guidance = _hand_case()
        out = projected_flow_x0(x0, x0, guidance, 0.5, 0.5, "cosine_decay", 1)
        expected = x0 + 0.5 * (3.0 - 1.0)  # [[1, 1], [1, 5]]
        assert torch.allclose(out, expected)

    def test_cosine_shift_matches_hand_computed(self):
        x0, guidance = _hand_case()
        out = projected_flow_x0(x0, x0, guidance, 0.5, 0.5, "cosine_shift", 1)
        # x0 - 0.5*(x0 - G) - 0.5*(P(x0) - P(G))
        #   = 0.5*x0 + 0.5*G + 0.5*(P(G) - P(x0))
        expected = 0.5 * x0 + 0.5 * guidance + 0.5 * (3.0 - 1.0)
        assert torch.allclose(out, expected)  # [[2.5, 2.5], [2.5, 4.5]]

    def test_constant_matches_hand_computed(self):
        x0, guidance = _hand_case()
        out = projected_flow_x0(x0, x0, guidance, 0.5, 0.5, "constant", 1)
        expected = x0 + (3.0 - 1.0)  # [[2, 2], [2, 6]]
        assert torch.allclose(out, expected)

    def test_unknown_schedule_raises(self):
        x0, guidance = _hand_case()
        with pytest.raises(ValueError, match="guidance_schedule"):
            projected_flow_x0(x0, x0, guidance, 0.5, 0.5, "linear", 1)

    def test_projected_flow_is_zero_when_guidance_equals_x0(self):
        x0, _ = _hand_case()
        out = projected_flow_x0(x0, x0, x0, 0.5, 0.7, "cosine_decay", 1)
        assert torch.allclose(out, x0)


# ---------------------------------------------------------------------------
# Parity with the reference velocity-space loop (plan D6)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProjectedFlowParity:
    def test_cosine_decay_equals_velocity_space_reference(self):
        """Literal transcription of pipeline_flux_imax.py:806-829
        (cosine_decay) in velocity space vs the x0-space engine call.

        The transcription normalizes the reference's scheduler convention
        ``t/1000 ≡ sigma`` and drops its ``+1e-6`` guard (plan D6): both
        contribute an O(1e-6/sigma) residual that makes atol=1e-6
        unattainable at small sigma; the schedule algebra under test is
        independent of both. The equivalence needs a LINEAR P — the Haar
        low-pass (avg_pool + nearest upsample) is linear, and the
        transcription applies exactly the P the engine applies.
        """
        torch.manual_seed(1)
        x_t = torch.randn(1, 1, 8, 8)
        x0 = torch.randn(1, 1, 8, 8)
        guidance = torch.randn(1, 1, 8, 8)
        sigma = 0.37
        cosine = 0.5 * (1.0 + math.cos(math.pi * 7 / 20))

        # Reference, velocity space (t/1000 ≡ sigma; +1e-6 guard dropped).
        v = (x_t - x0) / sigma
        fp_v = -(guidance - x_t) / sigma
        v_corrected = v + cosine * (
            haar_lowpass(fp_v, 1) - haar_lowpass(v, 1)
        )
        x0_reference = x_t - sigma * v_corrected

        x0_ours = projected_flow_x0(
            x_t, x0, guidance, sigma, cosine, "cosine_decay", 1,
        )
        assert torch.allclose(x0_ours, x0_reference, atol=1e-6)


# ---------------------------------------------------------------------------
# Low-resolution pass size (plan D10)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLowResSize:
    def test_square_4096_maps_to_1024(self):
        assert low_res_size(4096, 4096) == (1024, 1024)

    def test_aspect_is_preserved(self):
        """The reference's integer scale factor distorts aspect; D10 keeps
        it within the snap slack (16 px per side)."""
        h, w = 2048, 1024
        h_low, w_low = low_res_size(h, w)
        assert abs((h_low / w_low) - (h / w)) / (h / w) < 0.05

    def test_sizes_are_snapped_to_multiple_of_16(self):
        h_low, w_low = low_res_size(1500, 1000)
        assert h_low % 16 == 0
        assert w_low % 16 == 0

    def test_non_square_area_matches_native(self):
        h_low, w_low = low_res_size(2048, 1024)
        area_ratio = (h_low * w_low) / (1024 ** 2)
        assert area_ratio == pytest.approx(1.0, abs=0.05)

    def test_below_native_returns_target_size(self, caplog):
        with caplog.at_level("WARNING", logger="ComfyUI-DyPE"):
            assert low_res_size(512, 512) == (512, 512)
        assert "at/below the native" in caplog.text

    def test_low_res_scale_halves_the_area(self):
        base = low_res_size(4096, 4096)
        half = low_res_size(4096, 4096, scale=0.5)
        area_ratio = (half[0] * half[1]) / (base[0] * base[1])
        assert area_ratio == pytest.approx(0.5, abs=0.05)

    def test_low_res_never_exceeds_target(self):
        assert low_res_size(2048, 2048, scale=64.0) == (2048, 2048)
        assert low_res_size(1536, 1024) <= (1536, 1024)
