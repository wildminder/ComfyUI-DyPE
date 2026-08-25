"""Unit tests for src/spa.py — SPA bundle/slide math and SPABasePosEmbed.

These are pure unit tests (marker: unit) and need no ComfyUI runtime.
"""
import pytest
import torch

from src.spa import (
    SPABasePosEmbed,
    _phi,
    build_bundle_id_variants,
    bundle_ids_1d,
)


@pytest.mark.unit
class TestPhi:
    def test_phi_shape(self):
        x = torch.arange(10)
        out = _phi(x, 3, 5)
        assert out.shape == x.shape
        assert out.dtype == torch.long

    def test_phi_below_n1_is_zero(self):
        x = torch.arange(20)
        out = _phi(x, 4, 5)
        # positions 0..3 are below n1=4 -> 0
        assert torch.equal(out[:4], torch.zeros(4, dtype=torch.long))

    def test_phi_above_n1_is_ceil(self):
        x = torch.arange(4, 20)
        out = _phi(x, 4, 5)
        # manually: ceil((i+1-4)/5) for i in 4..19
        expected = torch.tensor([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4], dtype=torch.long)
        assert torch.equal(out, expected)

    def test_phi_invalid_size(self):
        with pytest.raises(ValueError):
            _phi(torch.arange(5), 1, 0)

    def test_phi_invalid_n1(self):
        with pytest.raises(ValueError):
            _phi(torch.arange(5), 0, 3)


@pytest.mark.unit
class TestBundleIds1d:
    def test_range(self):
        ids = bundle_ids_1d(100, 5, 1)
        assert ids.min() >= 0
        assert ids.max() == 20  # ceil(99/5)

    def test_first_bundle_size_bounds(self):
        with pytest.raises(ValueError):
            bundle_ids_1d(10, 3, 0)
        with pytest.raises(ValueError):
            bundle_ids_1d(10, 3, 4)


@pytest.mark.unit
class TestBuildBundleIdVariants:
    def _ids(self, H=8, W=8, B=1):
        L = H * W
        ids = torch.zeros(B, L, 3)
        ids[..., 0] = torch.arange(L)
        ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
        return ids

    def test_variant_count(self):
        # Paper-N semantics: #variants = 2*s - 1 where s = derive_bundle_s.
        # For H=W=128 (max_pos=127 > trained_extent=64), N=5 ->
        # s = max(5, ceil(127/79)=2) = 5 -> 2*5 - 1 = 9 variants.
        ids = self._ids(H=128, W=128)
        variants = build_bundle_id_variants(ids, 5)
        assert len(variants) == 9
        for v in variants:
            assert v.shape == ids.shape

    def test_variant_count_formula(self):
        # #variants = 2*s - 1 with the shared derive_bundle_s value (paper-N
        # semantics, trained-extent gate + in-dist floor + pass cap).
        from src.spa import derive_bundle_s
        ids = self._ids(H=128, W=128)
        N = 3
        variants = build_bundle_id_variants(ids, N)
        max_pos = int(max(ids[..., 1].max(), ids[..., 2].max()))
        s = derive_bundle_s(max_pos, N)
        if s <= 1:
            assert len(variants) == 1
        else:
            assert len(variants) == 2 * s - 1

    def test_low_res_is_identity(self):
        # Trained-extent gate: a grid inside the trained extent (max_pos <= 64)
        # is in-distribution -> SPA is an identity no-op regardless of N.
        ids = self._ids(H=32, W=32)  # max_pos=31 <= 64
        for N in (0, 2, 3, 5, 8):
            variants = build_bundle_id_variants(ids, N)
            assert len(variants) == 1, f"N={N} at 32x32 must be identity"
            assert torch.equal(variants[0], ids)

    def test_text_rows_unchanged(self):
        """Text tokens (height=width=0) stay at 0 across every variant.

        In the FLUX id convention, token 0 is the text token (h=w=0).  After
        bundling, ``phi(0, n1, N) = 0`` for every ``n1 >= 1``, so it is left
        untouched; image tokens (h or w > 0) are compressed into bundles.
        """
        ids = self._ids(H=128, W=128)  # active bundling (max_pos=127 > 64)
        variants = build_bundle_id_variants(ids, 3)
        assert len(variants) > 1  # bundling is active
        for v in variants:
            assert v[0, 0, 1] == 0  # text token height preserved
            assert v[0, 0, 2] == 0  # text token width preserved
            # image tokens are never pushed below 0
            assert (v[..., 1:] >= 0).all()

    def test_bundle_size_one_is_identity(self):
        ids = self._ids()
        variants = build_bundle_id_variants(ids, 1)
        assert len(variants) == 1
        assert torch.equal(variants[0], ids)

    def test_deterministic(self):
        ids = self._ids()
        a = build_bundle_id_variants(ids, 4)
        b = build_bundle_id_variants(ids, 4)
        for x, y in zip(a, b):
            assert torch.equal(x, y)


@pytest.mark.unit
class TestAutoGroupNum:
    def test_default_is_eighty(self):
        from src.spa import SPA_DEFAULT_GROUP_NUM
        assert SPA_DEFAULT_GROUP_NUM == 80


# ---------------------------------------------------------------------------
# Concrete SPA embedder for base-class behaviour tests
# ---------------------------------------------------------------------------

class _DummySPAEmbed(SPABasePosEmbed):
    """Minimal adapter: format_components concatenates (cos, sin) per axis."""

    def format_components(self, components, ids):
        parts = []
        for cos, sin in components:
            parts.append(torch.cat([cos, sin], dim=-1))
        return torch.cat(parts, dim=-1).unsqueeze(1).to(ids.device)


@pytest.mark.unit
class TestSPABasePosEmbed:
    def _make(self, **kw):
        return _DummySPAEmbed(
            theta=10000, axes_dim=[16, 56, 56], method="ntk",
            yarn_alt_scaling=False, dype=False, dype_scale=2.0,
            dype_exponent=2.0, base_resolution=1024, dype_start_sigma=1.0,
            base_patch_grid=None, **kw,
        )

    def test_spa_components_shape(self):
        emb = self._make(bundle_size=3)
        pos = torch.zeros(1, 64, 3)
        comps = emb._spa_components(pos, torch.float32)
        assert len(comps) == 3
        # each axis: (..., axis_dim)
        assert comps[0][0].shape == (1, 64, 16)
        assert comps[1][0].shape == (1, 64, 56)
        assert comps[2][0].shape == (1, 64, 56)

    def test_spa_components_identity_axis(self):
        """Axis 0 (temporal/text) is never scaled: ntk_factor=1.0 -> plain cos/sin."""
        emb = self._make()
        pos = torch.zeros(1, 4, 3)
        pos[..., 1] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        pos[..., 2] = torch.tensor([0.0, 1.0, 2.0, 3.0])
        cos, sin = emb._spa_components(pos, torch.float32)[0]
        # base NTK with theta=10000 -> freqs = 1/theta^(2k/D); for pos=0 cos=1
        assert torch.allclose(cos[0, 0], torch.ones_like(cos[0, 0]), atol=1e-5)

    def test_forward_off_equals_base_components(self):
        emb = self._make(enable_spa=False, bundle_size=5)
        ids = torch.zeros(1, 64, 3)
        out = emb(ids)
        expected = emb.format_components(emb._spa_components(ids.float(), torch.float32), ids)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_forward_bundle_size_one_matches_base(self):
        emb = self._make(enable_spa=True, bundle_size=1)
        ids = torch.zeros(1, 64, 3)
        out = emb(ids)
        # bundle_size=1 -> single identity variant -> equals base
        expected = emb.format_components(emb._spa_components(ids.float(), torch.float32), ids)
        assert torch.allclose(out, expected, atol=1e-6)

    def _grid_ids(self, H, W):
        ids = torch.zeros(1, H * W, 3)
        ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
        return ids

    def test_forward_returns_base_not_mean(self):
        # T-P2-1: forward returns the BASE RoPE, not the averaged variant RoPE.
        # Use a grid OUTSIDE the trained extent (max_pos=127 > 64) so bundling
        # is active and the base-vs-mean distinction is meaningful.
        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._grid_ids(128, 128)
        out = emb(ids)
        base = emb.format_components(emb._spa_components(ids.float(), torch.float32), ids)
        assert torch.allclose(out, base, atol=1e-6)

    def test_forward_registers_variants(self):
        # T-P2-2: forward registers N variant RoPEs in the active context.
        # Paper-N semantics: bundle_size=3 on H=W=128 (max_pos=127 > 64):
        # s = max(3, ceil(127/79)=2) = 3 -> 2*3 - 1 = 5 variants.
        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._grid_ids(128, 128)
        out = emb(ids)
        from src.spa import get_spa_context
        ctx = get_spa_context()
        assert ctx is not None and ctx.active is True
        assert len(ctx.variant_pes) == 5
        assert torch.allclose(ctx.base_pe, out, atol=1e-6)

    def test_forward_in_trained_extent_is_identity_context(self):
        # Trained-extent gate: a grid inside the trained extent registers a
        # single identity variant (hook passthrough), even with an active knob.
        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._grid_ids(64, 64)  # max_pos=63 <= 64
        out = emb(ids)
        from src.spa import get_spa_context
        ctx = get_spa_context()
        assert ctx is not None and ctx.active is True
        assert len(ctx.variant_pes) == 1  # identity -> hook passthrough
        assert torch.allclose(ctx.base_pe, out, atol=1e-6)

    def test_forward_high_res_registers_changed_variants(self):
        # T-P2-4 mirror: at high resolution the hook's variant pes DO change vs base.
        emb = self._make(enable_spa=True, bundle_size=5)
        ids = self._grid_ids(128, 128)  # max_pos=127 > 64 -> active
        out = emb(ids)
        base = emb.format_components(emb._spa_components(ids.float(), torch.float32), ids)
        assert torch.allclose(out, base, atol=1e-6)  # forward returns base
        from src.spa import get_spa_context
        ctx = get_spa_context()
        assert len(ctx.variant_pes) > 1
        max_diff = max((vp - ctx.base_pe).abs().max().item() for vp in ctx.variant_pes)
        assert max_diff > 1e-4  # bundling changed the variant RoPEs

    def test_forward_is_finite_on_cpu(self):
        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._grid_ids(128, 128)
        out = emb(ids)
        assert torch.isfinite(out).all()


@pytest.mark.unit
class TestSPPeCaching:
    """T3.1 / T3.2 — static variant-PE cache (D2b).

    SPA is static (no timestep dependence), so the base + variant PEs depend only
    on the position-id grid and the bundle size.  The cache must (a) reuse the
    computed PEs across repeated forwards with the same grid and (b) recompute
    when the grid or bundle size changes.
    """

    def _make(self, **kw):
        return _DummySPAEmbed(
            theta=10000, axes_dim=[16, 56, 56], method="ntk",
            yarn_alt_scaling=False, dype=False, dype_scale=2.0,
            dype_exponent=2.0, base_resolution=1024, dype_start_sigma=1.0,
            base_patch_grid=None, **kw,
        )

    def _ids(self, H=16, W=16):
        ids = torch.zeros(1, H * W, 3)
        ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
        return ids

    def test_pe_caching_reuses_computation(self):
        """T3.1: a second forward with the same grid does NOT recompute the PEs."""
        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._ids()

        # Count _spa_components invocations (the expensive PE construction).
        calls = {"n": 0}
        orig = emb._spa_components

        def counting(pos, fdtype):
            calls["n"] += 1
            return orig(pos, fdtype)

        emb._spa_components = counting

        emb(ids)
        first = calls["n"]
        assert first > 0

        emb(ids)  # same grid -> cache hit
        assert calls["n"] == first, (
            f"cache miss: _spa_components ran {calls['n'] - first} extra times")

    def test_pe_cache_invalidated_on_grid_change(self):
        """T3.2a: a different grid shape recomputes (no stale PEs)."""
        emb = self._make(enable_spa=True, bundle_size=3)

        calls = {"n": 0}
        orig = emb._spa_components

        def counting(pos, fdtype):
            calls["n"] += 1
            return orig(pos, fdtype)

        emb._spa_components = counting

        emb(self._ids(16, 16))
        after_first = calls["n"]
        emb(self._ids(8, 8))  # different grid -> recompute
        assert calls["n"] > after_first, "grid change must recompute the PEs"

    def test_pe_cache_invalidated_on_bundle_change(self):
        """T3.2b: changing bundle_size on the same embedder recomputes."""
        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._ids()

        calls = {"n": 0}
        orig = emb._spa_components

        def counting(pos, fdtype):
            calls["n"] += 1
            return orig(pos, fdtype)

        emb._spa_components = counting

        emb(ids)
        after_first = calls["n"]
        emb.bundle_size = 5  # user changed the knob between runs
        emb(ids)
        assert calls["n"] > after_first, "bundle_size change must recompute the PEs"

    def test_cached_pes_match_fresh_computation(self):
        """Cache correctness: cached PEs are bit-identical to a fresh computation."""
        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._ids()

        base_a, variants_a = emb._cached_variant_pes(ids)
        base_b, variants_b = emb._cached_variant_pes(ids)
        assert torch.equal(base_a, base_b)
        assert len(variants_a) == len(variants_b)
        for a, b in zip(variants_a, variants_b):
            assert torch.equal(a, b)


@pytest.mark.unit
class TestSPADeltaCache:
    """T3.1 / T3.2 — P3 delta-rotation cache (the D5 per-call overhead fix).

    The composed deltas ``inv(base) @ variant`` are static per grid.  They must be
    composed exactly ONCE per unique grid (in ``_cached_variant_pes``) and reused on
    every subsequent forward / attention call — never recomposed per call.
    """

    def _make(self, **kw):
        return _DummySPAEmbed(
            theta=10000, axes_dim=[16, 56, 56], method="ntk",
            yarn_alt_scaling=False, dype=False, dype_scale=2.0,
            dype_exponent=2.0, base_resolution=1024, dype_start_sigma=1.0,
            base_patch_grid=None, **kw,
        )

    def _ids(self, H=128, W=128):
        ids = torch.zeros(1, H * W, 3)
        ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
        return ids

    def test_delta_cache_bit_identical(self):
        """T3.1a: cached deltas equal freshly composed ``inv(base) @ variant``."""
        from src.spa_attn import compose_rope, inv_rope

        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._ids()

        base_pe, variant_pes = emb._cached_variant_pes(ids)
        cached_deltas = emb._cached_variant_deltas(ids)

        inv_base = inv_rope(base_pe, emb._rope_fmt)
        fresh = [compose_rope(inv_base, vp, emb._rope_fmt) for vp in variant_pes]

        assert len(cached_deltas) == len(fresh) == len(variant_pes)
        for c, f in zip(cached_deltas, fresh):
            assert torch.equal(c, f), "cached delta differs from fresh composition"

    def test_delta_cache_composed_once(self):
        """T3.1b: over 2 steps x 2 layers, ``compose_rope`` runs exactly ``2*s-1``
        times total (once per variant for the single grid), not per call."""
        import src.spa as spa_mod
        from src.spa import derive_bundle_s

        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._ids()  # 128x128 -> max_pos=127 -> s=3 -> 5 variants
        s = derive_bundle_s(127, 3)
        n_variants = 2 * s - 1

        calls = {"compose": 0}
        orig_compose = spa_mod.compose_rope

        def counting_compose(*a, **k):
            calls["compose"] += 1
            return orig_compose(*a, **k)

        spa_mod.compose_rope = counting_compose
        try:
            # Simulate 2 steps x 2 layers: each forward registers variants (cache
            # hit after the first), and each attention call consumes cached deltas.
            for _step in range(2):
                emb(ids)  # forward -> _register_variants -> _cached_variant_deltas
                for _layer in range(2):
                    emb._cached_variant_deltas(ids)  # attention-call consumption
        finally:
            spa_mod.compose_rope = orig_compose

        assert calls["compose"] == n_variants, (
            f"compose_rope ran {calls['compose']} times; expected exactly "
            f"{n_variants} (once per variant, composed once per grid)")

    def test_cache_invalidated_on_grid_change(self):
        """T3.2a: a different grid recomposes the deltas (no stale deltas).

        W2.6 fix (plan 2026-08-25): shapes are legitimately EQUAL here (fixed
        axes_dim -> same PE layout); the deltas differ in VALUES.  The old
        shape-inequality assertion was the stale-mock-era bug.
        """
        emb = self._make(enable_spa=True, bundle_size=3)

        d_small = emb._cached_variant_deltas(self._ids(96, 96))
        d_big = emb._cached_variant_deltas(self._ids(128, 128))
        # Shapes are legitimately equal (fixed axes_dim).
        assert d_small[0].shape == d_big[0].shape
        # Different grids MUST produce different delta VALUES.
        diff = (d_small[0] - d_big[0]).abs().max()
        assert diff > 0, "different grids must produce different delta VALUES"

    def test_cache_reused_on_same_grid_returns_identical_deltas(self):
        """W2.6 companion: the SAME grid returns bit-identical cached deltas."""
        emb = self._make(enable_spa=True, bundle_size=3)

        ids = self._ids(96, 96)
        first = emb._cached_variant_deltas(ids)
        second = emb._cached_variant_deltas(ids)
        assert first is second or all(
            torch.equal(a, b) for a, b in zip(first, second)
        )

    def test_register_variants_populates_context_deltas(self):
        """T3.2b: ``_register_variants`` stores the cached deltas on the context."""
        from src.spa_context import get_spa_context, set_spa_context

        set_spa_context(None)
        emb = self._make(enable_spa=True, bundle_size=3)
        ids = self._ids()
        emb(ids)
        ctx = get_spa_context()
        assert ctx is not None and ctx.active
        assert ctx.variant_deltas is not None
        assert len(ctx.variant_deltas) == len(ctx.variant_pes)
        # The context deltas are the SAME cached tensors (no per-call recompute).
        assert ctx.variant_deltas[0] is emb._cached_variant_deltas(ids)[0]
