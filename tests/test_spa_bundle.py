"""Regression tests for the SPA bundle/slide math (HRDiT-faithful, paper-N semantics).

SEMANTICS (2026-08-15 rewire): the user knob ``bundle_size`` is the PAPER's ``N``
(tokens per bundle; paper §4.1: ``N = 3`` at 2K, ``N = 5`` at 4K):

* ``N == 1``  -> off (single identity variant),
* ``N <= 0``  -> auto: minimal compression keeping every bundled position
  in-distribution (HRDiT ``group_num = 80`` ceiling, ``s_floor = ceil(max/79)``),
* ``N >= 2``  -> honoured, floored by the in-distribution minimum so bundled
  positions never exceed 79.

While the grid is inside the model's trained extent (``max_pos <= 64`` for
1024px-trained DiTs) SPA is an identity no-op.  The shared per-axis bundle size
``s`` comes from :func:`derive_bundle_s`; the boundary slides independently per
axis giving ``2*s - 1`` variants.

These tests lock that behaviour in (markers: @pytest.mark.unit).
"""

import pytest
import torch

from src.spa import (
    SPA_IN_DIST_MAX,
    SPA_MAX_PASSES,
    build_bundle_id_variants,
    derive_bundle_s,
)


def _flux_ids(H, W, B=1):
    L = H * W
    ids = torch.zeros(B, L, 3)
    ids[..., 0] = torch.arange(L)
    ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
    ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    return ids


def _effective_s(variants):
    return (len(variants) + 1) // 2


# ---------------------------------------------------------------------------
# T1.1 — derive_bundle_s pure-function table (plan §2.1)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeriveBundleS:
    """The full §2.1 behaviour table for ``derive_bundle_s``."""

    def test_identity_when_off_or_in_trained_extent(self):
        # N == 1 -> off
        assert derive_bundle_s(255, 1) == 1
        assert derive_bundle_s(63, 1) == 1
        # max_pos <= trained_extent -> identity regardless of N
        for N in (0, 2, 3, 5, 8):
            assert derive_bundle_s(63, N) == 1, f"N={N} at max_pos=63"
            assert derive_bundle_s(64, N) == 1, f"N={N} at max_pos=64"

    def test_auto_minimal_compression(self):
        # N == 0 -> s_floor = ceil(max_pos / 79)
        assert derive_bundle_s(95, 0) == 2    # ceil(95/79) = 2
        assert derive_bundle_s(127, 0) == 2   # ceil(127/79) = 2
        assert derive_bundle_s(191, 0) == 3   # ceil(191/79) = 3
        assert derive_bundle_s(255, 0) == 4   # ceil(255/79) = 4
        assert derive_bundle_s(511, 0) == 7   # ceil(511/79) = 7

    def test_explicit_n_honoured_with_in_dist_floor(self):
        # N >= 2 honoured but floored by s_floor (never OOD)
        assert derive_bundle_s(127, 3) == 3   # max(3, ceil(127/79)=2) = 3
        assert derive_bundle_s(127, 5) == 5   # max(5, 2) = 5
        assert derive_bundle_s(255, 3) == 4   # max(3, ceil(255/79)=4) = 4 (floor wins)
        assert derive_bundle_s(255, 5) == 5   # max(5, 4) = 5
        assert derive_bundle_s(191, 2) == 3   # max(2, ceil(191/79)=3) = 3 (floor wins)

    def test_pass_cap(self):
        # capped at (SPA_MAX_PASSES + 1) // 2 = 8
        assert derive_bundle_s(10000, 0) == 8
        assert derive_bundle_s(10000, 20) == 8

    def test_full_behaviour_table(self):
        """The exact §2.1 table (Krea-2 / FLUX-family, trained_extent = 64)."""
        # (max_pos, N) -> expected s
        table = [
            # 1024px (64x64), max_pos=63 -> identity for all N
            (63, 0, 1), (63, 2, 1), (63, 3, 1), (63, 5, 1),
            # 1536px (96x96), max_pos=95
            (95, 0, 2), (95, 2, 2), (95, 3, 3), (95, 5, 5),
            # 2048px (128x128), max_pos=127
            (127, 0, 2), (127, 2, 2), (127, 3, 3), (127, 5, 5),
            # 3072px (192x192), max_pos=191
            (191, 0, 3), (191, 2, 3), (191, 3, 3), (191, 5, 5),
            # 4096px (256x256), max_pos=255
            (255, 0, 4), (255, 2, 4), (255, 3, 4), (255, 5, 5),
        ]
        for (max_pos, N, expected) in table:
            got = derive_bundle_s(max_pos, N)
            assert got == expected, (
                f"derive_bundle_s({max_pos}, {N}) = {got}, expected {expected}")


# ---------------------------------------------------------------------------
# T1.2 — build_bundle_id_variants with paper-N semantics
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBundleIdVariiantsNSemantics:
    def test_n3_at_1k_is_identity(self):
        """INVERTED T0.1: knob 3 at 64x64 (1024px, in trained extent) -> identity.

        The big-patch collapse is GONE: the grid is inside the model's trained
        extent, so SPA is a no-op (1 identity variant, all 4096 positions kept).
        """
        ids = _flux_ids(64, 64)
        variants = build_bundle_id_variants(ids, 3)
        assert len(variants) == 1, (
            f"expected identity at 64x64 (in trained extent), got {len(variants)}")
        assert torch.equal(variants[0], ids)
        # All positions preserved.
        pairs = torch.stack([variants[0][0, :, 1], variants[0][0, :, 2]], dim=-1)
        assert int(torch.unique(pairs, dim=0).shape[0]) == 64 * 64

    def test_n3_at_2k_five_passes_fine_bundles(self):
        """INVERTED T0.2: knob 3 at 128x128 (2048px) -> 5 variants, fine bundles.

        s = max(3, ceil(127/79)=2) = 3 -> 2*3-1 = 5 passes.  Bundled max position
        stays in-distribution (<= 79) and the base variant keeps enough distinct
        positions that there are no big patches.
        """
        ids = _flux_ids(128, 128)
        variants = build_bundle_id_variants(ids, 3)
        assert len(variants) == 5, f"expected 5 variants (s=3), got {len(variants)}"
        # In-distribution: every bundled position <= 79.
        for v in variants:
            assert int(v[..., 1].max()) <= SPA_IN_DIST_MAX
            assert int(v[..., 2].max()) <= SPA_IN_DIST_MAX
        # Fine bundles: the base variant keeps a high distinct-position count.
        # With s=3 over 128 positions, distinct rows = ceil(128/3) ~ 43 -> ~43x43.
        base = variants[0]
        pairs = torch.stack([base[0, :, 1], base[0, :, 2]], dim=-1)
        distinct = int(torch.unique(pairs, dim=0).shape[0])
        assert distinct >= 1600, (
            f"base variant too coarse ({distinct} distinct); big patches present")

    def test_n5_at_4k_nine_passes(self):
        """knob 5 at 256x256 (4096px) -> s = max(5, ceil(255/79)=4) = 5 -> 9 passes."""
        ids = _flux_ids(256, 256)
        variants = build_bundle_id_variants(ids, 5)
        assert len(variants) == 9, f"expected 9 variants (s=5), got {len(variants)}"
        for v in variants:
            assert int(v[..., 1].max()) <= SPA_IN_DIST_MAX
            assert int(v[..., 2].max()) <= SPA_IN_DIST_MAX

    def test_auto_at_2k_three_passes(self):
        """knob 0 (auto) at 128x128 -> s = ceil(127/79) = 2 -> 3 passes."""
        ids = _flux_ids(128, 128)
        variants = build_bundle_id_variants(ids, 0)
        assert len(variants) == 3, f"expected 3 variants (s=2 auto), got {len(variants)}"

    def test_off_is_identity(self):
        ids = _flux_ids(128, 128)
        variants = build_bundle_id_variants(ids, 1)
        assert len(variants) == 1
        assert torch.equal(variants[0], ids)

    def test_in_distribution_guarantee_all_configs(self):
        """MOSAIC KILLER: every bundled position <= 79 for every active config."""
        for (H, W) in [(96, 96), (128, 128), (192, 192), (256, 256)]:
            for N in (0, 2, 3, 5, 8):
                ids = _flux_ids(H, W)
                variants = build_bundle_id_variants(ids, N)
                for v in variants:
                    assert int(v[..., 1].max()) <= SPA_IN_DIST_MAX, (
                        f"{H}x{W} N={N}: row OOD")
                    assert int(v[..., 2].max()) <= SPA_IN_DIST_MAX, (
                        f"{H}x{W} N={N}: col OOD")

    def test_slide_count_formula(self):
        """Number of variants = 2*s - 1 with the shared derive_bundle_s value."""
        for (H, W, N) in [(96, 96, 3), (128, 128, 3), (192, 192, 0), (256, 256, 5)]:
            ids = _flux_ids(H, W)
            variants = build_bundle_id_variants(ids, N)
            max_pos = max(H, W) - 1
            s = derive_bundle_s(max_pos, N)
            if s <= 1:
                assert len(variants) == 1
            else:
                assert len(variants) == 2 * s - 1, (
                    f"{H}x{W} N={N}: expected {2*s-1}, got {len(variants)}")

    def test_text_tokens_untouched(self):
        """Text tokens (row=col=0) stay at 0 across every variant."""
        ids = _flux_ids(128, 128)
        # prepend a text token
        txt = torch.zeros(1, 1, 3)
        ids = torch.cat([txt, ids], dim=1)
        for N in (0, 3, 5):
            variants = build_bundle_id_variants(ids, N)
            for v in variants:
                assert v[0, 0, 1] == 0 and v[0, 0, 2] == 0

    def test_non_square_aspect_preservation(self):
        """SQUISH FIX: shared s keeps non-square images undistorted."""
        H, W = 96, 144  # non-square, wider than tall (both > trained extent)
        ids = _flux_ids(H, W)
        variants = build_bundle_id_variants(ids, 3)
        row_max = max(int(v[..., 1].max()) for v in variants)
        col_max = max(int(v[..., 2].max()) for v in variants)
        assert row_max <= SPA_IN_DIST_MAX
        assert col_max <= SPA_IN_DIST_MAX
        # compression factor equal across axes (aspect preserved)
        row_ratio = row_max / H
        col_ratio = col_max / W
        assert abs(row_ratio - col_ratio) < 0.08

    def test_pass_count_bounded(self):
        """SLOWNESS FIX: pass count never exceeds SPA_MAX_PASSES."""
        for (H, W) in [(256, 256), (512, 512)]:
            for N in (0, 2, 3, 5, 8):
                ids = _flux_ids(H, W)
                variants = build_bundle_id_variants(ids, N)
                assert len(variants) <= SPA_MAX_PASSES, (
                    f"{H}x{W} N={N}: {len(variants)} passes > cap")

    def test_trained_extent_override(self):
        """A smaller trained_extent activates SPA at lower resolution."""
        ids = _flux_ids(64, 64)  # max_pos=63
        # default trained_extent=64 -> identity
        assert len(build_bundle_id_variants(ids, 3)) == 1
        # trained_extent=32 -> active (63 > 32), s = max(3, ceil(63/79)=1) = 3
        variants = build_bundle_id_variants(ids, 3, trained_extent=32)
        assert len(variants) == 5


# ---------------------------------------------------------------------------
# Legacy knob migration (decision M1) is exercised at the node/apply level;
# see tests/test_spa_node.py::test_legacy_group_num_values_flagged.
# ---------------------------------------------------------------------------
