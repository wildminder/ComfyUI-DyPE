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

from src.imax import haar_lowpass


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
