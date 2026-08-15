"""
Tests for the HAP runtime (``src/hap.py``).

Covers plan phases P0 (probes), P1 (ScopePlan + band math), P2 (backends +
runtime facade). See ``docs/plans/2026-08-15-hrdit-full-hap-implementation.md``.

All tests are CPU-safe; FlexAttention-specific tests are CUDA-gated and
auto-skip elsewhere.
"""

import pytest
import torch

from src import hap


@pytest.fixture
def mock_attn():
    """The conftest-provided (pristine SDPA) mock ``comfy.ldm.modules.attention`` module."""
    import comfy.ldm.modules.attention as attn_mod

    return attn_mod


# ---------------------------------------------------------------------------
# T0.1 — FlexAttention availability probe
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFlexProbe:
    def test_flex_probe_returns_bool(self):
        """The probe must always return a plain bool, whatever the env."""
        result = hap.hap_flex_available()
        assert isinstance(result, bool)

    def test_flex_probe_no_raise_on_cpu(self, monkeypatch):
        """Probe must not raise when CUDA is unavailable."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert hap.hap_flex_available() is False

    def test_flex_probe_false_when_import_fails(self, monkeypatch):
        """Probe returns False (never raises) if the flex_attention import blows up."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(hap, "_torch_version_at_least", lambda maj, mnr: True)

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if "flex_attention" in name:
                raise ImportError("simulated missing flex_attention")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert hap.hap_flex_available() is False

    def test_flex_probe_false_on_old_torch(self, monkeypatch):
        """Probe returns False when torch < 2.5 even with CUDA."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(hap, "_torch_version_at_least", lambda maj, mnr: False)
        assert hap.hap_flex_available() is False


@pytest.mark.unit
class TestTorchVersionParse:
    @pytest.mark.parametrize(
        "version,expected",
        [
            ("2.5.0", True),
            ("2.5.1+cu124", True),
            ("2.6.0.dev20241101", True),
            ("2.4.1", False),
            ("2.4", False),
            ("1.13.1+cpu", False),
            ("3.0.0", True),
        ],
    )
    def test_version_at_least(self, monkeypatch, version, expected):
        monkeypatch.setattr(torch, "__version__", version)
        assert hap._torch_version_at_least(2, 5) is expected

    def test_version_parse_never_raises(self, monkeypatch):
        monkeypatch.setattr(torch, "__version__", "garbage")
        assert isinstance(hap._torch_version_at_least(2, 5), bool)


# ---------------------------------------------------------------------------
# Constants sanity (reference parity)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestConstants:
    def test_block_size(self):
        assert hap.HAP_BLOCK == 64

    def test_default_text_len(self):
        assert hap.HAP_DEFAULT_TEXT_LEN == 512

    def test_anchor_off_sentinel(self):
        assert hap.HAP_ANCHOR_OFF == 1 << 30

    def test_train_seq_len(self):
        # FLUX training resolution: 64x64 image tokens + 512 text tokens.
        assert hap.HAP_TRAIN_SEQ_LEN == 4608


# ---------------------------------------------------------------------------
# T0.2 — Synthetic multi-layer DiT fixture
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestToyDiT:
    def test_toy_dit_call_order(self, mock_attn):
        """4 layers -> exactly 4 optimized_attention calls, in order 0..3."""
        from _hrdit_fixtures import CallRecorder, make_toy_dit

        rec = CallRecorder().install()
        try:
            dit = make_toy_dit(num_layers=4, heads=2, dim=16, text_len=8, img_hw=4, seed=7)
            dit.forward()
        finally:
            rec.uninstall()

        assert len(rec.calls) == 4
        assert [c["layer"] for c in dit.call_log] == [0, 1, 2, 3]

    def test_toy_dit_shapes_and_layout(self, mock_attn):
        """Each call sees (1, heads, text_len + img_hw^2, dim) tensors."""
        from _hrdit_fixtures import CallRecorder, make_toy_dit

        rec = CallRecorder().install()
        try:
            dit = make_toy_dit(num_layers=2, heads=3, dim=8, text_len=5, img_hw=3, seed=1)
            dit.forward()
        finally:
            rec.uninstall()

        expected_seq = 5 + 3 * 3
        for q, k, v, heads in rec.calls:
            assert tuple(q.shape) == (1, 3, expected_seq, 8)
            assert tuple(k.shape) == tuple(q.shape)
            assert tuple(v.shape) == tuple(q.shape)
            assert heads == 3
        assert dit.seq_len == expected_seq

    def test_toy_dit_deterministic(self, mock_attn):
        """Same seed -> identical outputs across two fresh instances."""
        from _hrdit_fixtures import make_toy_dit

        out1 = make_toy_dit(num_layers=3, seed=42).forward()
        out2 = make_toy_dit(num_layers=3, seed=42).forward()
        assert torch.equal(out1, out2)

    def test_toy_dit_output_finite(self, mock_attn):
        from _hrdit_fixtures import make_toy_dit

        out = make_toy_dit(num_layers=4, seed=3).forward()
        assert torch.isfinite(out).all()
        assert out.shape == (1, 8 + 16, 2 * 16)


# ---------------------------------------------------------------------------
# T1.1 — ScopePlan load / validate / round-trip
# ---------------------------------------------------------------------------

def _tiny_plan_dict():
    return {
        "alphas": [[2048.0, 0.0], [128.0, 64.0]],
        "betas": [[0.0, 0.25], [0.5, 0.0]],
    }


@pytest.mark.unit
class TestScopePlan:
    def test_scopeplan_roundtrip(self):
        d = _tiny_plan_dict()
        plan = hap.ScopePlan.from_dict(d)
        assert plan.to_dict() == d
        assert plan.num_layers == 2
        assert plan.num_heads == 2

    def test_scopeplan_loads_reference_flux_plan(self):
        """The REAL reference FLUX plan must load unchanged (format compat)."""
        import pathlib

        path = (
            pathlib.Path(__file__).parent.parent
            / ".dev" / "data" / "HRDit" / "HRDiT" / "configs" / "scope_plan_flux.json"
        )
        plan = hap.ScopePlan.load(path)
        assert plan.num_layers == 57
        assert plan.num_heads == 24
        assert all(a == 2048.0 for row in plan.alphas for a in row)
        assert all(b == 0.0 for row in plan.betas for b in row)

    def test_scopeplan_save_load_roundtrip(self, tmp_path):
        plan = hap.ScopePlan.from_dict(_tiny_plan_dict())
        out = tmp_path / "plan.json"
        plan.save(out)
        reloaded = hap.ScopePlan.load(out)
        assert reloaded.to_dict() == plan.to_dict()

    def test_scopeplan_rejects_ragged(self):
        d = {"alphas": [[1.0, 2.0], [3.0]], "betas": [[0.0, 0.0], [0.0]]}
        with pytest.raises(ValueError, match="ragged"):
            hap.ScopePlan.from_dict(d)

    def test_scopeplan_rejects_negative(self):
        d = {"alphas": [[-1.0]], "betas": [[0.0]]}
        with pytest.raises(ValueError, match=">= 0"):
            hap.ScopePlan.from_dict(d)

    def test_scopeplan_rejects_nonfinite(self):
        d = {"alphas": [[float("inf")]], "betas": [[0.0]]}
        with pytest.raises(ValueError, match="finite"):
            hap.ScopePlan.from_dict(d)

    def test_scopeplan_rejects_missing_key(self):
        with pytest.raises(ValueError, match="missing required key"):
            hap.ScopePlan.from_dict({"alphas": [[1.0]]})

    def test_scopeplan_rejects_layer_count_mismatch(self):
        d = {"alphas": [[1.0]], "betas": [[0.0], [0.0]]}
        with pytest.raises(ValueError, match="layers"):
            hap.ScopePlan.from_dict(d)

    def test_scopeplan_rejects_head_count_mismatch(self):
        d = {"alphas": [[1.0, 2.0]], "betas": [[0.0]]}
        with pytest.raises(ValueError, match="heads"):
            hap.ScopePlan.from_dict(d)

    def test_scopeplan_rejects_non_numeric(self):
        d = {"alphas": [["x"]], "betas": [[0.0]]}
        with pytest.raises(ValueError, match="number"):
            hap.ScopePlan.from_dict(d)

    def test_scopeplan_rejects_empty(self):
        with pytest.raises(ValueError, match="at least one layer"):
            hap.ScopePlan.from_dict({"alphas": [], "betas": []})

    def test_layer_bands(self):
        plan = hap.ScopePlan.from_dict(_tiny_plan_dict())
        # alpha=2048, beta=0, seq=66048 -> band 63 (reference FLUX numbers).
        assert plan.layer_bands(0, 66048)[0] == 63


# ---------------------------------------------------------------------------
# T1.2 — band_blocks (reference formula, exact)
# ---------------------------------------------------------------------------

def _reference_band_blocks(alphas, betas, seq_len, block=64):
    """Inline copy of hrdit/hap.py HapRuntime.band_blocks (parity oracle)."""
    nbx = seq_len // block
    return [max(2 * int(a // block + b * nbx) - 1, 1) for a, b in zip(alphas, betas)]


@pytest.mark.unit
class TestBandBlocks:
    def test_band_blocks_reference_flux_plan(self):
        # alpha=2048, beta=0, seq=66048 (4K FLUX) -> 2*int(2048/64)-1 = 63.
        assert hap.band_blocks([2048.0], [0.0], 66048) == [63]

    def test_band_blocks_beta_only(self):
        # alpha=0, beta=0.5, seq=66048 -> 2*int(0.5*1032)-1 = 1031.
        assert hap.band_blocks([0.0], [0.5], 66048) == [2 * int(0.5 * (66048 // 64)) - 1]
        assert hap.band_blocks([0.0], [0.5], 66048) == [1031]

    def test_band_blocks_min_one(self):
        assert hap.band_blocks([0.0], [0.0], 66048) == [1]

    def test_band_blocks_matches_reference_impl(self):
        """Property test vs the inline reference copy on 50 random cases."""
        g = torch.Generator().manual_seed(123)
        for _ in range(50):
            n = int(torch.randint(1, 8, (1,), generator=g).item())
            alphas = [float(torch.randint(0, 4097, (1,), generator=g).item()) for _ in range(n)]
            betas = [float(torch.rand(1, generator=g).item()) for _ in range(n)]
            seq = int(torch.randint(64, 70000, (1,), generator=g).item())
            assert hap.band_blocks(alphas, betas, seq) == _reference_band_blocks(alphas, betas, seq)

    def test_half_blocks(self):
        assert hap.half_blocks([63, 1, 1031, 4]) == [31, 0, 515, 1]


# ---------------------------------------------------------------------------
# T2.1 — HapContext + contextvars
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHapContextVars:
    def setup_method(self):
        hap.HapRuntime.reset()

    def teardown_method(self):
        from src.spa_context import set_hap_context, set_hrdit_layer_idx

        set_hap_context(None)
        set_hrdit_layer_idx(0)
        hap.HapRuntime.reset()

    def test_hap_context_default_inactive(self):
        from src.spa_context import get_hap_context

        assert get_hap_context() is None

    def test_hap_context_set_get_clear(self):
        from src.spa_context import get_hap_context, set_hap_context

        plan = hap.ScopePlan.from_dict(_tiny_plan_dict())
        ctx = hap.HapContext(active=True, plan=plan)
        set_hap_context(ctx)
        assert get_hap_context() is ctx
        set_hap_context(None)
        assert get_hap_context() is None

    def test_layer_counter_set_get_reset(self):
        from src.spa_context import get_hrdit_layer_idx, next_hrdit_layer_idx, set_hrdit_layer_idx

        assert get_hrdit_layer_idx() == 0
        assert next_hrdit_layer_idx() == 0
        assert next_hrdit_layer_idx() == 1
        assert get_hrdit_layer_idx() == 2
        set_hrdit_layer_idx(0)
        assert get_hrdit_layer_idx() == 0

    def test_context_isolation_across_copies(self):
        """Contextvar mutations inside a copied context don't leak out."""
        import contextvars

        from src.spa_context import get_hrdit_layer_idx, next_hrdit_layer_idx, set_hrdit_layer_idx

        set_hrdit_layer_idx(0)
        ctx = contextvars.copy_context()
        ctx.run(next_hrdit_layer_idx)
        ctx.run(next_hrdit_layer_idx)
        # The outer context is untouched.
        assert get_hrdit_layer_idx() == 0

    def test_hap_context_resolve_backend_auto(self, monkeypatch):
        plan = hap.ScopePlan.from_dict(_tiny_plan_dict())
        ctx = hap.HapContext(active=True, plan=plan, backend="auto")
        monkeypatch.setattr(hap, "hap_flex_available", lambda: False)
        assert ctx.resolve_backend() == "dense"
        monkeypatch.setattr(hap, "hap_flex_available", lambda: True)
        assert ctx.resolve_backend() == "flex"

    def test_hap_context_resolve_backend_explicit(self):
        plan = hap.ScopePlan.from_dict(_tiny_plan_dict())
        for backend in ("flex", "dense", "off"):
            ctx = hap.HapContext(active=True, plan=plan, backend=backend)
            assert ctx.resolve_backend() == backend


# ---------------------------------------------------------------------------
# T2.2 — Dense backend
# ---------------------------------------------------------------------------

def _rand_qkv(B=1, H=2, S=128, D=16, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g, dtype=dtype)
    k = torch.randn(B, H, S, D, generator=g, dtype=dtype)
    v = torch.randn(B, H, S, D, generator=g, dtype=dtype)
    return q, k, v


@pytest.mark.unit
class TestDenseBackend:
    def test_dense_backend_matches_manual_masked_softmax(self):
        """fp64: manual softmax over masked logits @ v == backend output."""
        S, H, text_len = 128, 2, 32
        q, k, v = _rand_qkv(H=H, S=S, seed=5)
        halves = [1, 3]
        mask = hap.build_band_mask(S, text_len, halves, anchor_stride=2)
        out = hap.hap_attn_dense(q, k, v, mask)

        scale = q.shape[-1] ** -0.5
        logits = (q @ k.transpose(-1, -2)) * scale
        neg_inf = torch.finfo(logits.dtype).min
        amask = torch.where(mask.unsqueeze(0), torch.zeros_like(logits), neg_inf)
        ref = torch.softmax(logits + amask, dim=-1) @ v
        assert torch.allclose(out, ref, atol=1e-12)

    def test_dense_backend_text_tokens_full_attention(self):
        """Text query rows equal plain SDPA output (mask all-True there)."""
        import torch.nn.functional as F

        S, H, text_len = 96, 2, 16
        q, k, v = _rand_qkv(H=H, S=S, seed=6)
        mask = hap.build_band_mask(S, text_len, [0], 0)
        out = hap.hap_attn_dense(q, k, v, mask)
        plain = F.scaled_dot_product_attention(q, k, v, scale=q.shape[-1] ** -0.5)
        assert torch.allclose(out[:, :, :text_len], plain[:, :, :text_len], atol=1e-12)

    def test_dense_backend_scale_passthrough(self):
        """An explicit scale is honoured (differs from the default)."""
        S, H, text_len = 64, 1, 0
        q, k, v = _rand_qkv(H=H, S=S, seed=7)
        mask = hap.build_band_mask(S, text_len, [10], 0)  # full attention
        out_default = hap.hap_attn_dense(q, k, v, mask)
        out_custom = hap.hap_attn_dense(q, k, v, mask, scale=2.0)
        assert not torch.allclose(out_default, out_custom, atol=1e-9)


# ---------------------------------------------------------------------------
# T2.3 — Flex backend (CUDA-gated; auto-skip elsewhere)
# ---------------------------------------------------------------------------

_FLEX_SKIP = pytest.mark.skipif(
    not hap.hap_flex_available(),
    reason="FlexAttention requires CUDA + torch>=2.5",
)


@pytest.mark.unit
class TestFlexBackend:
    @_FLEX_SKIP
    def test_flex_matches_dense_backend(self):
        S, H, text_len = 256, 2, 64
        q, k, v = _rand_qkv(H=H, S=S, seed=11, dtype=torch.float32)
        q, k, v = q.cuda(), k.cuda(), v.cuda()
        halves = [1, 2]
        # band = 2*int(beta*nbx)-1, half = (band-1)//2 -> beta = (half+1)/nbx.
        nbx = S // 64
        plan = hap.ScopePlan(
            alphas=[[0.0, 0.0]],
            betas=[[(halves[0] + 1) / nbx, (halves[1] + 1) / nbx]],
        )
        ctx = hap.HapContext(active=True, plan=plan, text_len=text_len, backend="flex")
        runtime = hap.HapRuntime.get()
        out_flex = runtime.attn(q, k, v, 0, ctx=ctx)
        mask = hap.build_band_mask(S, text_len, halves, 0)
        out_dense = hap.hap_attn_dense(q, k, v, mask.cuda())
        assert torch.allclose(out_flex, out_dense, rtol=1e-2, atol=1e-3)

    @_FLEX_SKIP
    def test_flex_mask_cache_reuse(self):
        S, H, text_len = 256, 2, 64
        q, k, v = _rand_qkv(H=H, S=S, seed=12, dtype=torch.float32)
        q, k, v = q.cuda(), k.cuda(), v.cuda()
        plan = hap.ScopePlan(alphas=[[0.0, 0.0]], betas=[[0.1, 0.1]])
        ctx = hap.HapContext(active=True, plan=plan, text_len=text_len, backend="flex")
        runtime = hap.HapRuntime.get()
        runtime.attn(q, k, v, 0, ctx=ctx)
        n_after_first = runtime.prepare_count
        runtime.attn(q, k, v, 0, ctx=ctx)
        assert runtime.prepare_count == n_after_first


# ---------------------------------------------------------------------------
# T2.4 — HapRuntime facade
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHapRuntime:
    def setup_method(self):
        hap.HapRuntime.reset()

    def teardown_method(self):
        from src.spa_context import set_hap_context

        set_hap_context(None)
        hap.HapRuntime.reset()

    def _ctx(self, num_layers=3, backend="dense"):
        plan = hap.ScopePlan(
            alphas=[[0.0, 0.0]] * num_layers,
            betas=[[0.5, 0.5]] * num_layers,
        )
        return hap.HapContext(active=True, plan=plan, text_len=0, backend=backend)

    def test_runtime_lazy_prepare_counts(self):
        """3 layers x 2 calls -> exactly 3 mask builds (one per distinct scope)."""
        from src.spa_context import set_hap_context

        ctx = self._ctx(num_layers=3)
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=2, S=128, seed=21)
        for _ in range(2):
            for layer in range(3):
                out = runtime.attn(q, k, v, layer)
                assert out is not None and out.shape == q.shape
        assert runtime.prepare_count == 3

    def test_runtime_seq_len_change_reprepares(self):
        from src.spa_context import set_hap_context

        ctx = self._ctx(num_layers=1)
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q1, k1, v1 = _rand_qkv(H=2, S=128, seed=22)
        runtime.attn(q1, k1, v1, 0)
        assert runtime.prepare_count == 1
        q2, k2, v2 = _rand_qkv(H=2, S=192, seed=23)
        runtime.attn(q2, k2, v2, 0)
        assert runtime.prepare_count == 2

    def test_runtime_inactive_returns_none(self):
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=2, S=64, seed=24)
        assert runtime.attn(q, k, v, 0) is None  # no context set

    def test_runtime_off_backend_falls_back_with_warning(self, caplog):
        import logging

        from src.spa_context import set_hap_context

        ctx = self._ctx(num_layers=1, backend="off")
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=2, S=64, seed=25)
        with caplog.at_level(logging.WARNING, logger="src.hap"):
            out = runtime.attn(q, k, v, 0)
        assert out is None
        assert any("off" in rec.message for rec in caplog.records)

    def test_runtime_layer_overflow_returns_none_with_warning(self, caplog):
        import logging

        from src.spa_context import set_hap_context

        ctx = self._ctx(num_layers=1)
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=2, S=64, seed=26)
        with caplog.at_level(logging.WARNING, logger="src.hap"):
            out = runtime.attn(q, k, v, 5)
        assert out is None
        assert any("exceeds" in rec.message for rec in caplog.records)

    def test_runtime_dense_output_matches_oracle(self):
        """End-to-end: runtime dense dispatch == manual masked softmax."""
        from src.spa_context import set_hap_context

        S, text_len = 128, 32
        # alpha=64 tokens -> band = 2*int(64/64)-1 = 1 -> half 0.
        plan = hap.ScopePlan(alphas=[[64.0, 64.0]], betas=[[0.0, 0.0]])
        ctx = hap.HapContext(active=True, plan=plan, text_len=text_len, backend="dense")
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=2, S=S, seed=27)
        out = runtime.attn(q, k, v, 0)
        mask = hap.build_band_mask(S, text_len, [0, 0], 0)
        ref = hap.hap_attn_dense(q, k, v, mask)
        assert torch.allclose(out, ref, atol=1e-12)

    def test_flops_ratio_bounds(self):
        plan = hap.ScopePlan.from_dict(_tiny_plan_dict())
        ratio = hap.flops_ratio(plan, seq_len=2048, text_len=512)
        assert 0.0 < ratio <= 1.0
        # Full-attention plan (huge alpha) -> ratio ~ 1.
        full = hap.ScopePlan(alphas=[[10**9]], betas=[[0.0]])
        assert hap.flops_ratio(full, seq_len=2048, text_len=0) == pytest.approx(1.0)
