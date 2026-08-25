"""Tests for src/rope.py — RoPE math functions (Tier 1: pure unit tests)."""
import pytest
import torch

from src.rope import (
    find_correction_factor,
    find_correction_range,
    find_newbase_ntk,
    get_1d_dype_yarn_pos_embed,
    get_1d_ntk_pos_embed,
    get_1d_yarn_pos_embed,
    linear_ramp_mask,
)


@pytest.mark.unit
class TestFindCorrectionFactor:
    def test_basic_computation(self):
        result = find_correction_factor(1.0, 128, 10000, 64)
        assert isinstance(result, float)
        assert result > 0

    def test_larger_max_pos_gives_larger_factor(self):
        r1 = find_correction_factor(1.0, 128, 10000, 64)
        r2 = find_correction_factor(1.0, 128, 10000, 128)
        assert r2 > r1

    def test_more_rotations_gives_smaller_factor(self):
        r1 = find_correction_factor(1.0, 128, 10000, 64)
        r2 = find_correction_factor(2.0, 128, 10000, 64)
        assert r2 < r1


@pytest.mark.unit
class TestFindCorrectionRange:
    def test_returns_within_bounds(self):
        low, high = find_correction_range(0.75, 1.25, 128, 10000, 64)
        assert low >= 0
        assert high <= 127

    def test_inverted_range_is_valid(self):
        # YaRN convention: low_ratio < 1 gives HIGHER correction factor
        # than high_ratio > 1, so low > high is expected behavior.
        # The linear_ramp_mask handles this via clamping.
        low, high = find_correction_range(0.1, 10.0, 128, 10000, 64)
        assert low >= 0
        assert high <= 127
        # low > high is valid — the ramp mask clamps to [0,1]

    def test_clamped_to_dim(self):
        low, high = find_correction_range(0.001, 1000.0, 64, 10000, 64)
        assert low >= 0
        assert high <= 63


@pytest.mark.unit
class TestLinearRampMask:
    def test_output_shape(self):
        mask = linear_ramp_mask(2.0, 10.0, 64)
        assert mask.shape == (64,)

    def test_values_in_0_1(self):
        mask = linear_ramp_mask(2.0, 10.0, 64)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_monotonic(self):
        mask = linear_ramp_mask(2.0, 10.0, 64)
        diffs = mask[1:] - mask[:-1]
        assert (diffs >= -1e-6).all()

    def test_equal_min_max_no_crash(self):
        mask = linear_ramp_mask(5.0, 5.0, 32)
        assert mask.shape == (32,)
        assert not torch.isnan(mask).any()

    def test_full_ramp(self):
        mask = linear_ramp_mask(0.0, 63.0, 64)
        assert mask[0].item() == pytest.approx(0.0, abs=1e-5)
        assert mask[-1].item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.unit
class TestFindNewbaseNtk:
    def test_scale_1_returns_base(self):
        result = find_newbase_ntk(128, 10000.0, 1.0)
        assert abs(result - 10000.0) < 1e-3

    def test_larger_scale_increases_base(self):
        result = find_newbase_ntk(128, 10000.0, 2.0)
        assert result > 10000.0

    def test_scale_2_formula(self):
        # base * scale^(dim/(dim-2)) = 10000 * 2^(128/126)
        expected = 10000.0 * (2.0 ** (128 / 126))
        result = find_newbase_ntk(128, 10000.0, 2.0)
        assert abs(result - expected) < 1e-3


@pytest.mark.unit
class TestGet1dNtkPosEmbed:
    def test_output_shapes(self):
        pos = torch.arange(64, dtype=torch.float32).unsqueeze(0)
        cos, sin = get_1d_ntk_pos_embed(
            dim=128, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32, ntk_factor=1.0
        )
        assert cos.shape == (1, 64, 128)
        assert sin.shape == (1, 64, 128)

    def test_ntk_factor_1_unit_circle(self):
        """With ntk_factor=1.0, cos²+sin²=1 (unit circle property)."""
        pos = torch.arange(32, dtype=torch.float32).unsqueeze(0)
        cos, sin = get_1d_ntk_pos_embed(
            dim=64, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32, ntk_factor=1.0
        )
        magnitude = cos**2 + sin**2
        assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-5)

    def test_higher_ntk_factor_changes_output(self):
        pos = torch.arange(32, dtype=torch.float32).unsqueeze(0)
        cos1, _ = get_1d_ntk_pos_embed(
            dim=64, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32, ntk_factor=1.0
        )
        cos2, _ = get_1d_ntk_pos_embed(
            dim=64, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32, ntk_factor=2.0
        )
        assert not torch.allclose(cos1, cos2)

    def test_complex_output_mode(self):
        pos = torch.arange(16, dtype=torch.float32).unsqueeze(0)
        result = get_1d_ntk_pos_embed(
            dim=32, pos=pos, theta=10000.0,
            use_real=False, repeat_interleave_real=False,
            freqs_dtype=torch.float32, ntk_factor=1.0
        )
        assert result.is_complex()


@pytest.mark.unit
class TestGet1dDypeYarnPosEmbed:
    def test_output_shapes(self):
        pos = torch.arange(64, dtype=torch.float32).unsqueeze(0)
        cos, sin = get_1d_dype_yarn_pos_embed(
            dim=128, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32,
            linear_scale=2.0, ntk_scale=2.0, ori_max_pe_len=64,
            dype=True, current_timestep=1.0,
            dype_scale=2.0, dype_exponent=2.0
        )
        assert cos.shape == (1, 64, 128)
        assert sin.shape == (1, 64, 128)

    def test_timestep_1_vs_0_differs(self):
        """DyPE at t=1 (full scaling) should differ from t≈0 (no scaling)."""
        pos = torch.arange(64, dtype=torch.float32).unsqueeze(0)
        kwargs = dict(
            dim=128, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32,
            linear_scale=2.0, ntk_scale=2.0, ori_max_pe_len=64,
            dype=True, dype_scale=2.0, dype_exponent=2.0
        )
        cos_t1, _ = get_1d_dype_yarn_pos_embed(current_timestep=1.0, **kwargs)
        cos_t0, _ = get_1d_dype_yarn_pos_embed(current_timestep=0.001, **kwargs)
        assert not torch.allclose(cos_t1, cos_t0, atol=1e-3)

    def test_dype_disabled_is_static(self):
        """With dype=False, timestep should not affect output."""
        pos = torch.arange(64, dtype=torch.float32).unsqueeze(0)
        kwargs = dict(
            dim=128, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32,
            linear_scale=2.0, ntk_scale=2.0, ori_max_pe_len=64,
            dype=False, dype_scale=2.0, dype_exponent=2.0
        )
        cos_t1, _ = get_1d_dype_yarn_pos_embed(current_timestep=1.0, **kwargs)
        cos_t0, _ = get_1d_dype_yarn_pos_embed(current_timestep=0.5, **kwargs)
        assert torch.allclose(cos_t1, cos_t0, atol=1e-6)

    def test_mscale_override(self):
        """override_mscale should directly scale the output."""
        pos = torch.arange(32, dtype=torch.float32).unsqueeze(0)
        kwargs = dict(
            dim=64, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32,
            linear_scale=2.0, ntk_scale=2.0, ori_max_pe_len=32,
            dype=True, current_timestep=1.0,
            dype_scale=2.0, dype_exponent=2.0
        )
        cos_a, sin_a = get_1d_dype_yarn_pos_embed(override_mscale=1.0, **kwargs)
        cos_b, sin_b = get_1d_dype_yarn_pos_embed(override_mscale=2.0, **kwargs)
        assert torch.allclose(cos_b, cos_a * 2.0, atol=1e-5)
        assert torch.allclose(sin_b, sin_a * 2.0, atol=1e-5)


@pytest.mark.unit
class TestGet1dYarnPosEmbed:
    def test_output_shapes(self):
        pos = torch.arange(64, dtype=torch.float32).unsqueeze(0)
        max_pe_len = torch.tensor(128.0)
        cos, sin = get_1d_yarn_pos_embed(
            dim=128, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32,
            max_pe_len=max_pe_len, ori_max_pe_len=64,
            dype=True, current_timestep=1.0,
            dype_scale=2.0, dype_exponent=2.0
        )
        assert cos.shape == (1, 64, 128)
        assert sin.shape == (1, 64, 128)

    def test_no_extrapolation_scale_1(self):
        """When max_pe_len == ori_max_pe_len, scale=1 → minimal change."""
        pos = torch.arange(32, dtype=torch.float32).unsqueeze(0)
        max_pe_len = torch.tensor(32.0)
        cos, sin = get_1d_yarn_pos_embed(
            dim=64, pos=pos, theta=10000.0,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=torch.float32,
            max_pe_len=max_pe_len, ori_max_pe_len=32,
            dype=False, current_timestep=1.0,
            dype_scale=2.0, dype_exponent=2.0
        )
        # mscale should be 1.0 when scale <= 1
        magnitude = cos**2 + sin**2
        assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-4)
