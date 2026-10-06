"""Tests for src/imax.py — I-Max core engine (Tier 1: pure unit tests).

Phases 1-3 of plan 2026-10-05:
- Haar low-pass projection P (D7) — the pywt-oracle test is the fidelity
  tripwire against a real Haar wavedec2/waverec2 round trip;
- flow sigma schedules, cosine decay factor, x0-space Projected Flow (D6)
  and the low-resolution pass size (D10);
- the dual-pass orchestration (D14).
"""

import inspect
import math

import pytest
import torch

from src.imax import (
    IMaxConfig,
    build_flow_sigmas,
    cosine_factor,
    haar_lowpass,
    imax_dual_pass,
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


# ---------------------------------------------------------------------------
# Dual-pass orchestration (plan Phase 3 / D14 — fake predict_x0 returning
# 0.5*x, the tests/test_hiflow_node.py convention)
# ---------------------------------------------------------------------------

def _recording_predict(factor: float = 0.5):
    """Fake predict_x0: returns factor*x and records shapes/sigmas plus the
    FIRST input it sees (each pass gets its own recorder)."""
    calls: dict = {"shapes": [], "sigmas": [], "first_input": None}

    def predict(x: torch.Tensor, sigma: float) -> torch.Tensor:
        if calls["first_input"] is None:
            calls["first_input"] = x.detach().clone()
        calls["shapes"].append(tuple(x.shape))
        calls["sigmas"].append(float(sigma))
        return factor * x

    return predict, calls


# 160 latent = 1280 px > native 1024 px: the low pass lands at 1024 px
# (= 128 latent) per D10's area-normalised formula.
_TARGET_LATENT = (160, 160)
_LOW_LATENT = (128, 128)
_SIGMAS_LOW = build_flow_sigmas(4, 3.0)
_SIGMAS_HIGH = build_flow_sigmas(3, 6.0)
_SEED_CONTENT = 7


def _make_content(seed: int = _SEED_CONTENT, h: int = 160, w: int = 160) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 4, h, w, generator=g)


def _make_guidance(h: int = 160, w: int = 160) -> torch.Tensor:
    g = torch.Generator().manual_seed(99)
    return torch.randn(1, 4, h, w, generator=g)


def _engine_cfg(**overrides) -> IMaxConfig:
    defaults = dict(
        steps_low=4, steps_high=3,
        native_resolution=1024, pixels_per_latent=8,
    )
    defaults.update(overrides)
    return IMaxConfig(**defaults)


def _run(content=None, guidance=None, cfg=None, seed=0, **kwargs):
    low, low_calls = _recording_predict()
    high, high_calls = _recording_predict()
    content = _make_content() if content is None else content
    guidance = _make_guidance() if guidance is None else guidance
    cfg = _engine_cfg() if cfg is None else cfg
    result = imax_dual_pass(
        low, high, _SIGMAS_LOW, _SIGMAS_HIGH, content, guidance,
        seed, cfg, **kwargs,
    )
    return result, low_calls, high_calls


@pytest.mark.unit
class TestIMaxDualPass:
    def test_total_forward_count_is_steps_low_plus_steps_high(self):
        _, low_calls, high_calls = _run()
        assert len(low_calls["shapes"]) == 4  # cfg.steps_low
        assert len(high_calls["shapes"]) == 3  # cfg.steps_high

    def test_low_pass_shape_is_low_res_shape(self):
        """160 latent = 1280 px target -> 1024 px (= 128 latent) low pass."""
        _, low_calls, _ = _run()
        assert low_calls["shapes"][0] == (1, 4, *_LOW_LATENT)

    def test_high_pass_shape_is_target_shape(self):
        result, _, high_calls = _run()
        assert high_calls["shapes"][0] == (1, 4, *_TARGET_LATENT)
        assert tuple(result.shape) == (1, 4, *_TARGET_LATENT)

    def test_same_seed_is_deterministic(self):
        content, guidance = _make_content(), _make_guidance()
        result_a, _, _ = _run(content=content, guidance=guidance, seed=123)
        result_b, _, _ = _run(content=content, guidance=guidance, seed=123)
        assert torch.equal(result_a, result_b)

    def test_different_seed_changes_result(self):
        content, guidance = _make_content(), _make_guidance()
        result_a, _, _ = _run(content=content, guidance=guidance, seed=1)
        result_b, _, _ = _run(content=content, guidance=guidance, seed=2)
        assert not torch.equal(result_a, result_b)

    def test_denoise_one_starts_from_pure_noise(self):
        content = _make_content(seed=42)
        result, _, high_calls = _run(
            content=content, cfg=_engine_cfg(denoise=1.0),
        )
        first = high_calls["first_input"]
        assert first.mean().abs() < 0.05
        assert (first.std() - 1.0).abs() < 0.05
        assert high_calls["sigmas"][0] == pytest.approx(1.0)

    def test_denoise_below_one_preserves_content(self):
        content = _make_content(seed=42)

        def correlation(first_input):
            a = first_input.flatten()
            b = content.flatten()
            a = a - a.mean()
            b = b - b.mean()
            return float((a * b).mean() / (a.std() * b.std()))

        _, _, high_calls = _run(
            content=content, cfg=_engine_cfg(denoise=0.3),
        )
        corr_noisy = correlation(high_calls["first_input"])
        assert high_calls["sigmas"][0] < 1.0  # truncated schedule entry
        assert corr_noisy > 0.05  # content survives the init mix

        _, _, high_calls_full = _run(
            content=content, cfg=_engine_cfg(denoise=1.0),
        )
        assert corr_noisy > correlation(high_calls_full["first_input"]) + 0.05

    def test_disable_schedule_equals_plain_euler(self):
        content, guidance = _make_content(), _make_guidance()
        cfg = _engine_cfg(guidance_schedule="disable")
        result, _, high_calls = _run(
            content=content, guidance=guidance, cfg=cfg,
        )
        # Manual Euler over the same schedule, from the engine's own
        # pass-B init, with the same fake predictor.
        x = high_calls["first_input"].clone()
        for i in range(_SIGMAS_HIGH.numel() - 1):
            sigma = float(_SIGMAS_HIGH[i])
            x0 = 0.5 * x
            v = (x - x0) / max(sigma, 1e-6)
            x = x + v * (float(_SIGMAS_HIGH[i + 1]) - sigma)
        assert torch.allclose(result, x, atol=1e-6)

    def test_progress_callback_reports_both_passes(self):
        events: list[tuple[int, int, str]] = []
        _run(progress_callback=lambda i, n, stage: events.append((i, n, stage)))
        stages = [stage for _, _, stage in events]
        assert stages.count("low") == 4
        assert stages.count("high") == 3
        assert all(n == 4 for i, n, s in events if s == "low")
        assert all(n == 3 for i, n, s in events if s == "high")

    def test_engine_never_calls_vae(self):
        """Signature guard: the engine takes injected predictors only —
        no VAE object ever reaches src/imax.py (plan D2)."""
        params = inspect.signature(imax_dual_pass).parameters
        assert [p for p in params if "vae" in p.lower()] == []

    def test_below_native_low_pass_runs_at_target(self, caplog):
        """64 latent = 512 px target <= native: pass A runs at the target
        and the D10 WARNING is logged."""
        content = _make_content(h=64, w=64)
        with caplog.at_level("WARNING", logger="ComfyUI-DyPE"):
            result, low_calls, _ = _run(
                content=content, guidance=_make_guidance(h=64, w=64),
            )
        assert low_calls["shapes"][0] == (1, 4, 64, 64)
        assert "at/below the native" in caplog.text
        assert tuple(result.shape) == (1, 4, 64, 64)

    def test_low_res_scale_shrinks_low_pass(self):
        """low_res_scale=0.25 quarters the guidance area (sqrt form)."""
        content = _make_content()
        _, low_calls, _ = _run(
            content=content, cfg=_engine_cfg(low_res_scale=0.25),
        )
        assert low_calls["shapes"][0] == (1, 4, 64, 64)

    def test_guidance_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="must match the target"):
            _run(guidance=_make_guidance(h=32, w=32))

    def test_multiframe_content_rejected(self):
        with pytest.raises(ValueError, match="4D latent"):
            _run(content=torch.randn(1, 4, 2, 64, 64))

    def test_denoise_out_of_range_raises(self):
        with pytest.raises(ValueError, match="denoise"):
            _run(cfg=_engine_cfg(denoise=0.0))
