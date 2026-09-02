"""Tests for src/hiflow.py — HiFlow core algorithm (Tier 1: pure unit tests).

Plan 2026-09-03 Step 1: Butterworth low-pass mask + FFT frequency split.
The loop-parity test is the fidelity tripwire against the authors'
reference implementation (.dev/data/HiFlow/HiFlow/utils.py).
"""

import time

import pytest
import torch

from src.hiflow import butterworth_low_pass_filter_2d, split_frequency_components_fft

# ---------------------------------------------------------------------------
# Butterworth low-pass mask
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestButterworthFilter:
    def test_mask_shape_and_range(self):
        for ratio in (0.2, 0.4, 0.95):
            mask = butterworth_low_pass_filter_2d((4, 16, 16), ratio=ratio)
            assert mask.shape == (16, 16)
            assert mask.min() >= 0.0
            assert mask.max() <= 1.0

    def test_ratio_zero_returns_zero_mask(self):
        """Reference behavior: ratio=0 -> all-zero (nothing passes)."""
        mask = butterworth_low_pass_filter_2d((16, 16), ratio=0.0)
        assert mask.shape == (16, 16)
        assert torch.count_nonzero(mask) == 0

    def test_high_ratio_nearly_passes_everything(self):
        mask = butterworth_low_pass_filter_2d((16, 16), ratio=0.95)
        assert mask.max().item() >= 0.9

    def test_center_passes_high_freq_blocked(self):
        """DC center (fftshifted layout) == 1; corners (highest |d|) < 0.05."""
        h = w = 16
        mask = butterworth_low_pass_filter_2d((h, w), ratio=0.2)
        assert mask[h // 2, w // 2].item() == pytest.approx(1.0, abs=1e-5)
        for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            assert mask[corner].item() < 0.05, (
                f"corner {corner} must be strongly attenuated at ratio 0.2"
            )

    def test_matches_loop_reference(self):
        """Vectorized mask == the authors' Python double-loop formula verbatim.

        Loop body copied from .dev/data/HiFlow/HiFlow/utils.py
        (butterworth_low_pass_filter_2d): d_square uses (2h/H - 1) terms with
        h the row index and H the full height.
        """
        for ratio in (0.2, 0.4):
            for h, w in ((16, 16), (8, 24)):
                mask_vec = butterworth_low_pass_filter_2d((h, w), ratio=ratio)

                loop = torch.zeros((h, w))
                for row in range(h):
                    for col in range(w):
                        d_square = ((2 * row / h - 1) ** 2 + (2 * col / w - 1) ** 2)
                        loop[row, col] = 1 / (1 + (d_square / ratio ** 2) ** 4)

                assert torch.allclose(mask_vec, loop, atol=1e-5), (
                    f"vectorized mask diverges from the reference loop "
                    f"(h={h}, w={w}, ratio={ratio})"
                )

    def test_vectorized_build_is_fast(self):
        """64x64 must build in < 50 ms on CPU (guards against regression to a
        Python loop; generous bound)."""
        t0 = time.perf_counter()
        butterworth_low_pass_filter_2d((64, 64), ratio=0.2)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.050, f"mask build took {elapsed * 1000:.1f} ms"

    def test_invalid_shape_rejected(self):
        with pytest.raises(ValueError, match="positive H, W"):
            butterworth_low_pass_filter_2d((0, 8))

    def test_symmetric_mask(self):
        """The mask is mirror-symmetric about the axis through the DC row pair.

        The reference grid g[h] = 2h/H - 1 satisfies g[H-h] = -g[h], so
        mask[h, :] == mask[H-h, :] for h in 1..H-1 (row 0's mirror would be
        row H, which does not exist — the grid is edge-asymmetric, NOT
        flip-symmetric, for even sizes).
        """
        h = w = 16
        mask = butterworth_low_pass_filter_2d((h, w), ratio=0.3)
        for row in range(1, h):
            assert torch.allclose(mask[row], mask[h - row], atol=1e-7), (
                f"row {row} must mirror row {h - row}"
            )
        for col in range(1, w):
            assert torch.allclose(mask[:, col], mask[:, w - col], atol=1e-7), (
                f"col {col} must mirror col {w - col}"
            )


# ---------------------------------------------------------------------------
# FFT frequency split
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFFTSplit:
    def test_low_plus_high_equals_x(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16, 16)
        f = butterworth_low_pass_filter_2d((16, 16), ratio=0.2)
        low = split_frequency_components_fft(x, f, is_low=True)
        high = split_frequency_components_fft(x, f, is_low=False)
        assert torch.allclose(low + high, x, atol=1e-4), (
            "low + high must reconstruct the input exactly (mask + (1-mask) = 1)"
        )

    def test_split_low_is_smooth(self):
        """A checkerboard is pure high frequency: its low component must carry
        little energy at a small cutoff."""
        x = torch.zeros(1, 1, 16, 16)
        x[0, 0, ::2, ::2] = 1.0
        x[0, 0, 1::2, 1::2] = 1.0
        f = butterworth_low_pass_filter_2d((16, 16), ratio=0.2)
        low = split_frequency_components_fft(x, f, is_low=True)
        assert low.std().item() < 0.2 * x.std().item(), (
            "low-pass component of a checkerboard should be near zero"
        )

    def test_zero_mask_kills_low_passes_everything(self):
        torch.manual_seed(1)
        x = torch.randn(1, 3, 8, 8)
        low = split_frequency_components_fft(x, torch.zeros(8, 8), is_low=True)
        high = split_frequency_components_fft(x, torch.zeros(8, 8), is_low=False)
        assert torch.count_nonzero(low) == 0
        assert torch.allclose(high, x, atol=1e-4)

    def test_ones_mask_passes_low_untouched(self):
        """Filter of ones: LPF(x) == x (up to FFT round trip)."""
        torch.manual_seed(2)
        x = torch.randn(1, 3, 8, 8)
        low = split_frequency_components_fft(x, torch.ones(8, 8), is_low=True)
        assert torch.allclose(low, x, atol=1e-4)

    def test_fp16_input_no_nan(self):
        """fp16 latents must not produce NaN (fp32 upcast inside the split)."""
        torch.manual_seed(3)
        x = torch.randn(1, 4, 16, 16, dtype=torch.float16)
        f = butterworth_low_pass_filter_2d((16, 16), ratio=0.2, dtype=torch.float32)
        out = split_frequency_components_fft(x, f)
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all()

    def test_dc_only_input_survives_low_pass(self):
        """A constant image is pure DC: the low-pass component is the image."""
        x = torch.full((1, 2, 8, 8), 0.7)
        f = butterworth_low_pass_filter_2d((8, 8), ratio=0.2)
        low = split_frequency_components_fft(x, f, is_low=True)
        assert torch.allclose(low, x, atol=1e-4)

    def test_parity_isotropic_norms(self):
        """Energy sanity: ||low||^2 + ||high||^2 == ||x||^2 for a real mask
        (Parseval-adjacent, exact only for 0/1 masks)."""
        torch.manual_seed(4)
        x = torch.randn(1, 2, 16, 16)
        h, w = 16, 16
        # Binary mask (ideal low-pass) keeps the energy split exact.
        ys = torch.linspace(-1.0, 1.0 - 2.0 / h, h)
        xs = torch.linspace(-1.0, 1.0 - 2.0 / w, w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        f = ((yy.square() + xx.square()) <= 0.2 ** 2).float()
        low = split_frequency_components_fft(x, f, is_low=True)
        high = split_frequency_components_fft(x, f, is_low=False)
        e_total = x.square().sum().item()
        e_split = low.square().sum().item() + high.square().sum().item()
        assert e_split == pytest.approx(e_total, rel=1e-3, abs=1e-3), (
            f"energy split {e_split} != total {e_total}"
        )

    def test_output_finite_random_shapes(self):
        torch.manual_seed(5)
        for h, w in ((8, 8), (12, 20), (32, 32)):
            x = torch.randn(2, 3, h, w)
            f = butterworth_low_pass_filter_2d((h, w), ratio=0.4)
            out = split_frequency_components_fft(x, f)
            assert out.shape == x.shape
            assert torch.isfinite(out).all()
