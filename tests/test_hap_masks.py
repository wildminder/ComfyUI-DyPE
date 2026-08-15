"""Tests for the HAP dense mask builder and compute-cost model (``src/hap.py``).

Plan T1.3 (``build_band_mask`` — the FlexAttention ``mask_mod`` oracle) and
T1.4 (``band_compute_cost`` — closed-form retained-pair count).

Markers: @pytest.mark.unit
"""

import pytest
import torch

from src import hap


def _scalar_mask_mod(q, k, text_len, half, anchor_stride, block=64):
    """Scalar (per q, k) evaluation of the reference ``mask_mod`` expression.

    Reference: hrdit/hap.py lines 34-41.  Anchors apply to image blocks only
    (``kb >= 0``); text keys are covered by ``text_k``.
    """
    text_q = q < text_len
    text_k = k < text_len
    qb = (q - text_len) // block
    kb = (k - text_len) // block
    band = abs(qb - kb) <= half
    anchor = kb >= 0 and anchor_stride > 0 and anchor_stride < hap.HAP_ANCHOR_OFF and (kb % anchor_stride) == 0
    return text_q or text_k or band or anchor


@pytest.mark.unit
class TestBuildBandMask:
    def test_mask_matches_scalar_mask_mod(self):
        """Brute-force parity with the reference mask_mod on a small config."""
        seq_len, text_len = 192, 64
        halves = [0, 1, 3]  # mixed half-widths, H=3
        anchor_stride = 2
        mask = hap.build_band_mask(seq_len, text_len, halves, anchor_stride)
        assert mask.shape == (3, seq_len, seq_len)
        assert mask.dtype == torch.bool
        for h, half in enumerate(halves):
            for q in range(seq_len):
                for k in range(seq_len):
                    expected = _scalar_mask_mod(q, k, text_len, half, anchor_stride)
                    assert bool(mask[h, q, k]) == expected, (h, q, k)

    def test_text_rows_and_cols_full(self):
        seq_len, text_len = 160, 32
        mask = hap.build_band_mask(seq_len, text_len, [0, 2], 0)
        for h in range(2):
            assert mask[h, :text_len, :].all(), "text query rows must attend everything"
            assert mask[h, :, :text_len].all(), "text keys must be visible to everyone"

    def test_anchor_periodicity(self):
        """Anchor blocks visible at exactly every stride-th image block."""
        block = 64
        text_len = 64
        num_img_blocks = 6
        seq_len = text_len + num_img_blocks * block
        stride = 2
        mask = hap.build_band_mask(seq_len, text_len, [0], stride, block=block)
        # Query from a far image block so only text_k / band(0) / anchor show.
        q = text_len + 5 * block + 10  # block 5
        for kb in range(num_img_blocks):
            k = text_len + kb * block + 3
            expected_anchor = (kb % stride) == 0
            expected_band = abs(5 - kb) <= 0
            assert bool(mask[0, q, k]) == (expected_anchor or expected_band), kb

    def test_anchor_disabled(self):
        """stride 0 and ANCHOR_OFF both disable anchors."""
        block = 64
        text_len = 64
        seq_len = text_len + 4 * block
        q = text_len + 3 * block + 1  # block 3, far from block 0
        for stride in (0, hap.HAP_ANCHOR_OFF, -1):
            mask = hap.build_band_mask(seq_len, text_len, [0], stride, block=block)
            # half=0: only the diagonal block + text visible; block 0 NOT visible.
            assert not bool(mask[0, q, text_len + 5]), f"stride={stride} must disable anchors"

    def test_band_edges(self):
        """Visibility flips exactly at |qb - kb| = half + 1."""
        block = 64
        text_len = 0
        seq_len = 10 * block
        half = 2
        mask = hap.build_band_mask(seq_len, text_len, [half], 0, block=block)
        q = 5 * block + 7  # block 5
        for kb in range(10):
            k = kb * block + 1
            expected = abs(5 - kb) <= half
            assert bool(mask[0, q, k]) == expected, kb

    def test_partial_last_block(self):
        """A non-multiple image length still yields correct per-token masks."""
        block = 64
        text_len = 32
        seq_len = text_len + 2 * block + 17  # partial last block
        mask = hap.build_band_mask(seq_len, text_len, [1], 0, block=block)
        for q in range(seq_len):
            for k in range(seq_len):
                expected = _scalar_mask_mod(q, k, text_len, 1, 0, block)
                assert bool(mask[0, q, k]) == expected, (q, k)


@pytest.mark.unit
class TestComputeCost:
    def _assert_cost_matches_mask(self, seq_len, text_len, halves, anchor_stride, block=64):
        mask = hap.build_band_mask(seq_len, text_len, halves, anchor_stride, block=block)
        expected = float(mask.sum().item()) / len(halves)
        got = hap.band_compute_cost(seq_len, text_len, halves, anchor_stride, block=block)
        assert got == expected, (seq_len, text_len, halves, anchor_stride, got, expected)

    def test_compute_cost_matches_mask_sum(self):
        """8 random configs: closed form == dense mask sum (exact)."""
        g = torch.Generator().manual_seed(99)
        for _ in range(8):
            block = 64
            n_img_blocks = int(torch.randint(1, 8, (1,), generator=g).item())
            tail = int(torch.randint(0, block, (1,), generator=g).item())
            text_len = int(torch.randint(0, 4, (1,), generator=g).item()) * block
            seq_len = text_len + n_img_blocks * block + tail
            H = int(torch.randint(1, 4, (1,), generator=g).item())
            halves = [int(torch.randint(0, n_img_blocks + 2, (1,), generator=g).item()) for _ in range(H)]
            stride_choice = int(torch.randint(0, 3, (1,), generator=g).item())
            anchor_stride = [0, 2, 3][stride_choice]
            self._assert_cost_matches_mask(seq_len, text_len, halves, anchor_stride, block)

    def test_compute_cost_fixed_cases(self):
        # Full attention: huge half-width -> every pair retained.
        seq_len, text_len = 192, 64
        full = hap.band_compute_cost(seq_len, text_len, [10**6], 0)
        assert full == float(seq_len * seq_len)
        # half=0, no anchors: text rows full + image diagonal blocks only.
        block = 64
        n_img = (seq_len - text_len) // block  # 2 full blocks
        expected = text_len * seq_len + n_img * (block * (text_len + block))
        got = hap.band_compute_cost(seq_len, text_len, [0], 0)
        assert got == float(expected)

    def test_compute_cost_rejects_bad_text_len(self):
        with pytest.raises(ValueError, match="exceeds seq_len"):
            hap.band_compute_cost(64, 128, [1], 0)

    def test_compute_cost_monotone_in_half(self):
        halves_small = hap.band_compute_cost(640, 64, [1], 0)
        halves_big = hap.band_compute_cost(640, 64, [4], 0)
        assert halves_big > halves_small
