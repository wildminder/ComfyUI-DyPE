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
        """Probe returns False (never raises) if the flex_attention import blows up.

        W3 note (2026-08-25): the probe now imports via
        ``from torch.nn.attention import flex_attention`` (the F823 fix), so
        the simulated failure must trigger on that module path.
        """
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(hap, "_torch_version_at_least", lambda maj, mnr: True)

        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch.nn.attention":
                raise ImportError("simulated missing flex_attention")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert hap.hap_flex_available() is False

    def test_flex_probe_false_on_old_torch(self, monkeypatch):
        """Probe returns False when torch < 2.5 even with CUDA."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(hap, "_torch_version_at_least", lambda maj, mnr: False)
        assert hap.hap_flex_available() is False

    def test_flex_probe_cuda_gate_reachable(self, monkeypatch):
        """W3 REGRESSION (ruff F823): the local
        ``import torch.nn.attention.flex_attention`` used to bind ``torch``
        function-locally, so ``torch.cuda.is_available()`` raised
        UnboundLocalError (swallowed by the except) and the probe returned
        False on EVERY environment — the CUDA gate was unreachable.  With a
        CUDA-mocked-positive env and an old-torch stub, the version gate must
        be what returns False (proving execution reached past the CUDA check).
        """
        reached = {"version_gate": False}

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        def _version_gate(maj, mnr):
            reached["version_gate"] = True
            return False

        monkeypatch.setattr(hap, "_torch_version_at_least", _version_gate)
        assert hap.hap_flex_available() is False
        assert reached["version_gate"], (
            "execution never reached the version gate — the CUDA check raised "
            "(UnboundLocalError regression)")


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
        """The REAL reference FLUX plan must load unchanged (format compat).

        Uses the SHIPPED plan (configs/scope_plan_flux.json) — version-
        controlled and present in CI. (An identical copy lives under the
        gitignored .dev/data research tree; do not reference that path.)"""
        import pathlib

        path = (
            pathlib.Path(__file__).parent.parent
            / "configs" / "scope_plan_flux.json"
        )
        assert path.exists(), f"shipped reference plan missing: {path}"
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

    # -- excluded_head_counts metadata (2026-08-23 head-count warning fix) ----

    def test_scopeplan_excluded_head_counts_roundtrip(self):
        """A plan WITH excluded_head_counts round-trips the field exactly."""
        d = dict(_tiny_plan_dict())
        d["excluded_head_counts"] = [20]
        plan = hap.ScopePlan.from_dict(d)
        assert plan.excluded_head_counts == [20]
        assert plan.to_dict() == d

    def test_scopeplan_excluded_head_counts_omitted_when_absent(self):
        """A legacy plan WITHOUT the field round-trips to the exact legacy
        shape (no spurious key) — full backward compatibility."""
        d = _tiny_plan_dict()
        plan = hap.ScopePlan.from_dict(d)
        assert plan.excluded_head_counts is None
        assert plan.to_dict() == d
        assert "excluded_head_counts" not in plan.to_dict()

    def test_scopeplan_excluded_head_counts_empty_omitted(self):
        """An EMPTY excluded list is treated as absent (omitted from to_dict)."""
        plan = hap.ScopePlan(
            alphas=_tiny_plan_dict()["alphas"],
            betas=_tiny_plan_dict()["betas"],
            excluded_head_counts=[],
        )
        assert "excluded_head_counts" not in plan.to_dict()

    def test_scopeplan_rejects_bad_excluded_head_counts(self):
        """excluded_head_counts must be a list of ints; else ValueError."""
        d = dict(_tiny_plan_dict())
        d["excluded_head_counts"] = [20, "x"]
        with pytest.raises(ValueError, match="excluded_head_counts"):
            hap.ScopePlan.from_dict(d)
        d2 = dict(_tiny_plan_dict())
        d2["excluded_head_counts"] = "not-a-list"
        with pytest.raises(ValueError, match="excluded_head_counts"):
            hap.ScopePlan.from_dict(d2)

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
        # W2.7 fix (2026-08-25): DISTINCT betas per layer.  The mask cache is
        # keyed by the RESOLVED halves, so a plan with identical per-layer
        # scopes shares one mask across all layers and a "one prepare per
        # distinct scope" count can never reach ``num_layers`` (the pre-fix
        # fixture used [0.5, 0.5] everywhere -> prepare_count == 1).
        # Cycle [0.5, 0.75, 1.0]: at nbx=4 (seq 256) these resolve to halves
        # {1, 2, 3} — three distinct masks.
        cycle = [0.5, 0.75, 1.0]
        plan = hap.ScopePlan(
            alphas=[[0.0, 0.0]] * num_layers,
            betas=[[cycle[i % len(cycle)]] * 2 for i in range(num_layers)],
        )
        return hap.HapContext(active=True, plan=plan, text_len=0, backend=backend)

    def test_runtime_lazy_prepare_counts(self):
        """3 layers x 2 calls -> exactly 3 mask builds (one per distinct scope)."""
        from src.spa_context import set_hap_context

        ctx = self._ctx(num_layers=3)
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        # S=256 -> nbx=4 so the cycled betas resolve to DISTINCT halves.
        q, k, v = _rand_qkv(H=2, S=256, seed=21)
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


# ---------------------------------------------------------------------------
# T3.1/T3.2 — Decline guards (plan 2026-08-16 G3)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeclineGuards:
    """HAP must DECLINE (return None -> plain attention) — never crash, never
    silent wrong math — for calls its square, plan-shaped mask cannot serve:

    * non-square attention (cross-attention, ``kv_len != q_len``), and
    * head-count mismatch (scope plan heads != model heads).

    Both decline with a one-time log latch (reset by ``HapRuntime.reset()``).
    """

    def setup_method(self):
        hap.HapRuntime.reset()

    def teardown_method(self):
        from src.spa_context import set_hap_context

        set_hap_context(None)
        hap.HapRuntime.reset()

    def _ctx(self, num_layers=3, num_heads=2, backend="dense"):
        plan = hap.ScopePlan(
            alphas=[[64.0] * num_heads for _ in range(num_layers)],
            betas=[[0.0] * num_heads for _ in range(num_layers)],
        )
        return hap.HapContext(active=True, plan=plan, text_len=0, backend=backend)

    def test_nonsquare_cross_attention_returns_none(self):
        """Cross-attention (kv_len != q_len) -> None (plain-attention fallback)."""
        from src.spa_context import set_hap_context

        ctx = self._ctx(num_heads=2)
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        # q has 64 tokens, k/v have 32 (cross-attention).
        g = torch.Generator().manual_seed(30)
        q = torch.randn(1, 2, 64, 16, generator=g)
        k = torch.randn(1, 2, 32, 16, generator=g)
        v = torch.randn(1, 2, 32, 16, generator=g)
        assert runtime.attn(q, k, v, 0) is None

    def test_nonsquare_decline_is_one_time_debug(self, caplog):
        """The non-square decline logs at most ONE debug line per runtime."""
        import logging

        from src.spa_context import set_hap_context

        ctx = self._ctx(num_heads=2)
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        g = torch.Generator().manual_seed(31)
        q = torch.randn(1, 2, 64, 16, generator=g)
        k = torch.randn(1, 2, 32, 16, generator=g)
        v = torch.randn(1, 2, 32, 16, generator=g)
        with caplog.at_level(logging.DEBUG, logger="src.hap"):
            runtime.attn(q, k, v, 0)
            runtime.attn(q, k, v, 1)  # second call must be silent
        nonsquare = [r for r in caplog.records if "non-square" in r.message]
        assert len(nonsquare) == 1

    def test_head_mismatch_returns_none(self):
        """Plan with 24 heads vs q with 16 heads -> None (wrong plan for model)."""
        from src.spa_context import set_hap_context

        ctx = self._ctx(num_heads=24)  # FLUX plan shape
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=16, S=64, seed=32)  # Anima runs 16 heads
        assert runtime.attn(q, k, v, 0) is None

    def test_head_mismatch_decline_is_one_time_warning(self, caplog):
        """The head-mismatch decline logs ONE warning naming both counts."""
        import logging

        from src.spa_context import set_hap_context

        ctx = self._ctx(num_heads=24)
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=16, S=64, seed=33)
        with caplog.at_level(logging.WARNING, logger="src.hap"):
            runtime.attn(q, k, v, 0)
            runtime.attn(q, k, v, 1)  # second call must be silent
        mismatch = [r for r in caplog.records if "heads" in r.message]
        assert len(mismatch) == 1
        # The warning names both the plan's and the model's head counts.
        assert "24" in mismatch[0].message
        assert "16" in mismatch[0].message

    def test_matching_heads_unaffected(self):
        """Matching head count still engages the kernel (existing behaviour)."""
        from src.spa_context import set_hap_context

        ctx = self._ctx(num_heads=2)
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=2, S=64, seed=34)
        out = runtime.attn(q, k, v, 0)
        assert out is not None and out.shape == q.shape

    # -- EXPECTED auxiliary fallback vs GENUINE wrong plan (2026-08-23) -------

    def _ctx_with_excluded(self, num_heads=2, excluded=(16,)):
        """A plan that DECLARED ``excluded`` head counts during calibration."""
        plan = hap.ScopePlan(
            alphas=[[64.0] * num_heads for _ in range(3)],
            betas=[[0.0] * num_heads for _ in range(3)],
            excluded_head_counts=list(excluded),
        )
        return hap.HapContext(active=True, plan=plan, text_len=0, backend="dense")

    def test_aux_head_mismatch_returns_none(self):
        """A head count in excluded_head_counts still declines to None."""
        from src.spa_context import set_hap_context

        ctx = self._ctx_with_excluded(num_heads=2, excluded=(16,))
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=16, S=64, seed=40)
        assert runtime.attn(q, k, v, 0) is None

    def test_aux_head_mismatch_decline_is_one_time_info(self, caplog):
        """An EXPECTED aux fallback logs ONE INFO (not a WARNING) naming the
        excluded head count, and stays silent on repeat calls."""
        import logging

        from src.spa_context import set_hap_context

        ctx = self._ctx_with_excluded(num_heads=2, excluded=(16,))
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=16, S=64, seed=41)
        with caplog.at_level(logging.INFO, logger="src.hap"):
            runtime.attn(q, k, v, 0)
            runtime.attn(q, k, v, 1)  # second call must be silent
        infos = [r for r in caplog.records
                 if r.levelno == logging.INFO and "EXCLUDED during" in r.message]
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "does not match" in r.message]
        assert len(infos) == 1
        assert warnings == []  # NOT a scary wrong-plan warning
        assert "16" in infos[0].message

    def test_genuine_head_mismatch_still_warning(self, caplog):
        """A head count NOT in excluded_head_counts still logs the WARNING
        (genuinely wrong plan) — the pre-fix behaviour is preserved."""
        import logging

        from src.spa_context import set_hap_context

        # Plan declares excluded=(99,) but the model runs 16 -> NOT excluded.
        ctx = self._ctx_with_excluded(num_heads=2, excluded=(99,))
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=16, S=64, seed=42)
        with caplog.at_level(logging.WARNING, logger="src.hap"):
            runtime.attn(q, k, v, 0)
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "does not match" in r.message]
        infos = [r for r in caplog.records
                 if r.levelno == logging.INFO and "EXCLUDED during" in r.message]
        assert len(warnings) == 1
        assert infos == []
        assert "2" in warnings[0].message and "16" in warnings[0].message

    def test_aux_and_genuine_latches_independent(self, caplog):
        """The aux INFO latch and the genuine WARNING latch are independent:
        an aux call then a genuine mismatch each log exactly once."""
        import logging

        from src.spa_context import set_hap_context

        ctx = self._ctx_with_excluded(num_heads=2, excluded=(16,))
        set_hap_context(ctx)
        runtime = hap.HapRuntime.get()
        q_aux, k_aux, v_aux = _rand_qkv(H=16, S=64, seed=43)
        q_bad, k_bad, v_bad = _rand_qkv(H=8, S=64, seed=44)  # not excluded
        with caplog.at_level(logging.INFO, logger="src.hap"):
            runtime.attn(q_aux, k_aux, v_aux, 0)   # aux -> INFO
            runtime.attn(q_bad, k_bad, v_bad, 1)   # genuine -> WARNING
        infos = [r for r in caplog.records
                 if r.levelno == logging.INFO and "EXCLUDED during" in r.message]
        warnings = [r for r in caplog.records
                    if r.levelno == logging.WARNING and "does not match" in r.message]
        assert len(infos) == 1
        assert len(warnings) == 1
        assert runtime._noted_aux_fallback is True
        assert runtime._warned_head_mismatch is True

    def test_decline_latches_reset_by_runtime_reset(self):
        """``HapRuntime.reset()`` drops the singleton -> fresh latches."""
        from src.spa_context import set_hap_context

        ctx = self._ctx(num_heads=24)
        set_hap_context(ctx)
        r1 = hap.HapRuntime.get()
        q, k, v = _rand_qkv(H=16, S=64, seed=35)
        r1.attn(q, k, v, 0)
        assert r1._warned_head_mismatch is True
        hap.HapRuntime.reset()
        r2 = hap.HapRuntime.get()
        assert r2 is not r1
        assert r2._warned_head_mismatch is False
        assert r2._warned_nonsquare is False
