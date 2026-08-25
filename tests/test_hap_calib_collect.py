"""Tests for the HAP calibration gradient path (``src/hap_calib.py``).

Plan phase P6:
- T6.1 ``chunked_attention`` — calibration-only attention exposing ``A.grad``
  per query-row chunk, output identical to SDPA;
- T6.2 ``collect_scope_scores`` — per-(layer, head, scope) score accumulation
  from ONE backward pass over a toy forward, without a dense ``T×T``.

The real-model calibration script (T6.3) is tested separately below
(``-k dry_run``) and in the user's venv (checklist A5).

Markers: @pytest.mark.unit / @pytest.mark.mock_integration
Accept (user-run):
    pytest tests/test_hap_calib_collect.py -k chunked
    pytest tests/test_hap_calib_collect.py -k collector
    pytest tests/test_hap_calib_collect.py -k dry_run
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from src.hap_calib import (
    chunked_attention,
    collect_scope_scores,
    estimate_head_scope_costs,
)


def _attn_module():
    return sys.modules["comfy.ldm.modules.attention"]


# ---------------------------------------------------------------------------
# T6.1 — chunked differentiable attention
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestChunkedAttention:
    def test_chunked_output_equals_sdpa(self):
        """Chunked forward output == ``F.scaled_dot_product_attention`` (same
        scale), for several chunk sizes including chunk > T."""
        torch.manual_seed(0)
        B, H, T, D = 1, 3, 40, 8
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        ref = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        for chunk in (7, 16, 40, 128):
            out, chunks = chunked_attention(q, k, v, scale=1.0, chunk=chunk)
            assert out.shape == (B, H, T, D)
            assert torch.allclose(out, ref, atol=1e-6)
            # Chunks tile the rows exactly, in order.
            assert sum(c.shape[2] for c in chunks) == T

    def test_chunked_grads_flow(self):
        """Scalar loss backward → every chunk has ``.grad`` with the right
        shape, and the concatenated chunk grads equal the single-shot
        materialized-softmax grad (fp64, atol 1e-12)."""
        torch.manual_seed(1)
        B, H, T, D = 1, 2, 24, 6
        q = torch.randn(B, H, T, D, dtype=torch.float64)
        k = torch.randn(B, H, T, D, dtype=torch.float64)
        v = torch.randn(B, H, T, D, dtype=torch.float64)
        weight = torch.randn(B, H, T, D, dtype=torch.float64)

        # Single-shot oracle: one materialized softmax leaf.
        A_full = torch.softmax(torch.matmul(q, k.transpose(-1, -2)), dim=-1)
        A_full = A_full.detach().requires_grad_(True)
        loss_full = (torch.matmul(A_full, v) * weight).sum()
        loss_full.backward()
        G_full = A_full.grad

        # Chunked.
        out, chunks = chunked_attention(q, k, v, scale=1.0, chunk=5)
        loss = (out * weight).sum()
        loss.backward()
        for c in chunks:
            assert c.grad is not None
            assert c.grad.shape == c.shape
        G_chunked = torch.cat([c.grad for c in chunks], dim=2)
        assert torch.allclose(G_chunked, G_full, atol=1e-12)

    def test_chunked_rejects_bad_inputs(self):
        q = torch.randn(1, 2, 8, 4)
        with pytest.raises(ValueError):
            chunked_attention(torch.randn(2, 8, 4), q, q)
        with pytest.raises(ValueError):
            chunked_attention(q, torch.randn(1, 2, 7, 4), q)
        with pytest.raises(ValueError):
            chunked_attention(q, q, q, chunk=0)

    def test_chunked_bf16_keeps_native_dtype_and_clean_grads(self):
        """OOM REVERT (2026-08-18): bf16 q/k/v must keep the NATIVE dtype for
        the attention chunks and their gradients — NOT be up-cast to fp32.

        An earlier revision up-cast to fp32 to "fix" the NaN quality table, but
        that DOUBLED the retained attention memory (bf16 3.6 GiB -> fp32
        7.19 GiB at seq=1198/H=48) plus a second fp32 ``.grad`` during
        ``backward()``, causing an OOM on a 16 GiB card.  De-risking
        (tmp/diag_oom_redesign.py, Q1) proved the attention-leaf backward
        ``grad_out @ vᵀ`` is CLEAN in bf16 even at 1e30 scale (bf16 shares
        fp32's exponent range), so the live NaN is UPSTREAM and up-casting
        ``A`` cannot fix it.  This test pins the memory-safe behaviour:
        native dtype preserved, grads clean.
        """
        torch.manual_seed(2)
        B, H, T, D = 1, 2, 16, 4
        q = torch.randn(B, H, T, D, dtype=torch.bfloat16)
        k = torch.randn(B, H, T, D, dtype=torch.bfloat16)
        v = torch.randn(B, H, T, D, dtype=torch.bfloat16)
        weight = torch.randn(B, H, T, D, dtype=torch.bfloat16)

        out, chunks = chunked_attention(q, k, v, scale=1.0, chunk=5)

        # Output stays in the model's dtype (bf16).
        assert out.dtype == torch.bfloat16
        assert out.shape == (B, H, T, D)

        # Chunks keep the native dtype (bf16) — NOT up-cast to fp32 (that
        # would double the retained memory and OOM).
        for c in chunks:
            assert c.dtype == torch.bfloat16, (
                f"chunk dtype {c.dtype} != bfloat16 — fp32 up-cast would "
                "double retained attention memory and OOM"
            )

        # Gradients keep the native dtype and are clean (no NaN/inf) — the
        # attention-leaf backward is robust in bf16 (same exponent range as
        # fp32).
        loss = (out * weight).sum()
        loss.backward()
        for c in chunks:
            assert c.grad is not None
            assert c.grad.dtype == torch.bfloat16, (
                f"chunk.grad dtype {c.grad.dtype} != bfloat16"
            )
            assert not torch.isnan(c.grad).any(), "NaN in chunk.grad"
            assert not torch.isinf(c.grad).any(), "inf in chunk.grad"


# ---------------------------------------------------------------------------
# P19 — fp16 logit-overflow fix (live Krea2 run #8 root cause)
# ---------------------------------------------------------------------------
#
# The live run proved: the model runs fp16 with |q|,|k| ~ 600 and head_dim=128,
# so the fp16 ``q @ kᵀ`` dot product reaches ~128 * 600^2 ~= 4.6e7 — ~700x OVER
# fp16's max (65504) -> ``inf`` logits -> ``softmax(inf - inf) = NaN`` rows ->
# the forward NaN cascade.  ComfyUI's own ``attention_basic`` computes
# ``einsum(q.float(), k.float())`` — logits are ALWAYS fp32 there.  The fix
# computes logits + softmax in fp32 (transient) and casts ``A`` back to the
# model dtype for storage (no retained-VRAM increase).

@pytest.mark.unit
class TestChunkedFp16OverflowFix:
    def test_fp16_large_magnitude_finite_attention(self):
        """PRIMARY regression: fp16 q/k with live-scale magnitudes (|q|,|k|
        in the hundreds, head_dim=128) used to overflow the fp16 ``q @ kᵀ``
        matmul -> inf logits -> NaN attention.  With the fp32-logits fix the
        attention chunks and output must be FINITE."""
        torch.manual_seed(3)
        B, H, T, D = 1, 2, 32, 128  # head_dim=128 like Krea2
        # randn * 100 -> element magnitudes in the hundreds; dot products over
        # 128 dims reach ~1e6, ~20x over fp16's 65504 limit.
        q = torch.randn(B, H, T, D, dtype=torch.float16) * 100
        k = torch.randn(B, H, T, D, dtype=torch.float16) * 100
        v = torch.randn(B, H, T, D, dtype=torch.float16)

        out, chunks = chunked_attention(q, k, v, scale=1.0, chunk=8)

        for c in chunks:
            assert torch.isfinite(c).all(), "NaN/inf in attention chunk (fp16 logit overflow)"
        assert torch.isfinite(out).all(), "NaN/inf in attention output"

    def test_fp16_matches_fp32_oracle(self):
        """The fp16 chunked output matches the SAME inputs computed in fp32
        (the fp32 path cannot overflow) — the fp32-logits fix makes fp16
        numerically equivalent to fp32 attention, up to fp16 storage rounding."""
        torch.manual_seed(4)
        B, H, T, D = 1, 2, 32, 128
        q32 = torch.randn(B, H, T, D) * 100
        k32 = torch.randn(B, H, T, D) * 100
        v32 = torch.randn(B, H, T, D)
        q16, k16, v16 = q32.half(), k32.half(), v32.half()

        out32, _ = chunked_attention(q32, k32, v32, scale=1.0, chunk=8)
        out16, _ = chunked_attention(q16, k16, v16, scale=1.0, chunk=8)

        # fp16 storage rounding of A and v bounds the difference; with the
        # large logits the attention is near one-hot so out ~ a v row.
        assert torch.allclose(out16.float(), out32, rtol=2e-2, atol=2.0)

    def test_fp16_large_magnitude_grads_clean(self):
        """Under live-scale fp16 magnitudes the chunk gradients stay FINITE
        (the old fp16-matmul path produced NaN grads via the NaN attention)."""
        torch.manual_seed(5)
        B, H, T, D = 1, 2, 32, 128
        q = torch.randn(B, H, T, D, dtype=torch.float16) * 100
        k = torch.randn(B, H, T, D, dtype=torch.float16) * 100
        v = torch.randn(B, H, T, D, dtype=torch.float16)

        out, chunks = chunked_attention(q, k, v, scale=1.0, chunk=8)
        out.sum().backward()
        for c in chunks:
            assert c.grad is not None
            assert torch.isfinite(c.grad).all(), "NaN/inf in chunk.grad"

    def test_fp16_chunks_keep_fp16_dtype(self):
        """The stored attention chunks stay fp16 (only the TRANSIENT logits are
        fp32) — the retained-VRAM footprint must NOT double (the reverted P10
        upcast stored A in fp32 and OOM'd)."""
        torch.manual_seed(6)
        B, H, T, D = 1, 2, 16, 128
        q = torch.randn(B, H, T, D, dtype=torch.float16) * 100
        k = torch.randn(B, H, T, D, dtype=torch.float16) * 100
        v = torch.randn(B, H, T, D, dtype=torch.float16)
        out, chunks = chunked_attention(q, k, v, scale=1.0, chunk=8)
        assert out.dtype == torch.float16
        for c in chunks:
            assert c.dtype == torch.float16, (
                f"chunk dtype {c.dtype} != float16 — storing A in fp32 would "
                "double retained attention memory and OOM"
            )

    def test_fp32_input_not_downcast(self):
        """fp32 (and fp64) inputs are NEVER down-cast: compute stays in the
        input dtype and the chunks keep it."""
        torch.manual_seed(7)
        B, H, T, D = 1, 2, 16, 8
        for dt in (torch.float32, torch.float64):
            q = torch.randn(B, H, T, D, dtype=dt)
            k = torch.randn(B, H, T, D, dtype=dt)
            v = torch.randn(B, H, T, D, dtype=dt)
            out, chunks = chunked_attention(q, k, v, scale=1.0, chunk=4)
            assert out.dtype == dt
            for c in chunks:
                assert c.dtype == dt


# ---------------------------------------------------------------------------
# T6.2 — stats collector over a toy forward
# ---------------------------------------------------------------------------

def _dense_reference_scores(dit, loss_fn, num_scopes, text_len, scale=1.0):
    """Dense single-shot oracle: materialize the FULL ``A`` once per layer as
    a grad leaf, run the same forward+loss+backward, and score with
    :func:`estimate_head_scope_costs`.  Returns ``(L, H, S)`` fp64."""
    attn_mod = _attn_module()
    orig = attn_mod.optimized_attention
    dense_As = []

    def dense_attn(q, k, v, heads, *args, **kwargs):
        A = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        A = A.detach().requires_grad_(True)
        dense_As.append(A)
        return torch.matmul(A, v)

    attn_mod.optimized_attention = dense_attn
    try:
        out = dit.forward()
        loss = loss_fn(out)
        loss.backward()
    finally:
        attn_mod.optimized_attention = orig

    layers = []
    for A in dense_As:
        A2 = A[0].to(torch.float64)
        G = A.grad[0].to(torch.float64)
        layers.append(estimate_head_scope_costs(A2, G, num_scopes, text_len))
    return torch.stack(layers, dim=0)


def _make_loss_fn(dit):
    """Deterministic differentiable loss: MSE of the hidden state vs a fixed
    seeded target of the same shape."""
    g = torch.Generator().manual_seed(123)
    target = torch.randn(1, dit.seq_len, dit.heads * dit.dim,
                         generator=g, dtype=torch.float64)

    def loss_fn(output):
        return torch.nn.functional.mse_loss(output, target)

    return loss_fn


@pytest.mark.unit
class TestCollector:
    def _toy(self, seed=5):
        from _hrdit_fixtures import make_toy_dit
        # 2 layers, seq = text_len(8) + 4*4 = 24 <= 192 (plan bound).
        return make_toy_dit(num_layers=2, heads=3, dim=8, text_len=8,
                            img_hw=4, seed=seed, dtype=torch.float64)

    def test_collector_equals_dense_reference(self):
        """Collected (chunked) scores == dense single-shot scores (fp64,
        atol 1e-8) for a 2-layer toy DiT with an MSE loss."""
        num_scopes, text_len = 6, 8
        dit = self._toy()
        loss_fn = _make_loss_fn(dit)
        quality, compute = collect_scope_scores(
            dit.forward, loss_fn, num_scopes, text_len=text_len, chunk=5,
            scale=1.0,
        )
        assert quality.shape == (2, 3, num_scopes)
        assert compute.shape == (2, 3, num_scopes)

        ref = _dense_reference_scores(self._toy(), _make_loss_fn(self._toy()),
                                      num_scopes, text_len)
        assert torch.allclose(quality, ref, atol=1e-8)

    def test_collector_memory_chunking_invariant(self):
        """chunk=3 vs chunk=4096 (> seq) give IDENTICAL scores — chunking is a
        pure memory knob and never changes the result.

        W2.7 re-baseline (2026-08-25): asserted as fp64 ``allclose`` at
        1e-12 instead of bitwise ``torch.equal``.  Each query row's softmax
        spans the full key dimension regardless of chunking (the math IS
        chunk-invariant), but the logits matmul runs through differently
        SHAPED BLAS calls per chunk size, whose summation order may differ in
        the last ulp — a bitwise comparison over-asserts the guarantee.
        """
        num_scopes, text_len = 5, 8
        q_small, _ = collect_scope_scores(
            self._toy().forward, _make_loss_fn(self._toy()),
            num_scopes, text_len=text_len, chunk=3, scale=1.0,
        )
        q_big, _ = collect_scope_scores(
            self._toy().forward, _make_loss_fn(self._toy()),
            num_scopes, text_len=text_len, chunk=4096, scale=1.0,
        )
        assert torch.allclose(q_small, q_big, rtol=0.0, atol=1e-12)

    def test_collector_restores_original_attention(self):
        """After collection the module's ``optimized_attention`` is the
        original object (no leak), even if the forward raises."""
        attn_mod = _attn_module()
        orig = attn_mod.optimized_attention
        dit = self._toy()
        collect_scope_scores(dit.forward, _make_loss_fn(dit), 4,
                             text_len=8, chunk=6, scale=1.0)
        assert attn_mod.optimized_attention is orig

        # Restore also holds on failure inside the forward.
        def boom():
            raise RuntimeError("calibration forward failed")
        with pytest.raises(RuntimeError):
            collect_scope_scores(boom, lambda o: o.sum(), 4)
        assert attn_mod.optimized_attention is orig

    def test_collector_full_scope_column_is_zero(self):
        """The last scope column (full attention) scores exactly 0 per layer."""
        num_scopes = 4
        quality, _ = collect_scope_scores(
            self._toy().forward, _make_loss_fn(self._toy()),
            num_scopes, text_len=8, chunk=7, scale=1.0,
        )
        assert torch.all(quality[:, :, -1] == 0.0)

    def test_collector_no_calls_raises(self):
        """A forward that never calls ``optimized_attention`` raises a clear
        error (nothing to calibrate)."""
        with pytest.raises(RuntimeError, match="no optimized_attention calls"):
            collect_scope_scores(lambda: torch.zeros(1), lambda o: o.sum(), 4)


# ---------------------------------------------------------------------------
# text_len clamp (Krea2 live crash: band_compute_cost text_len=512 exceeds
# seq_len=430 — the FLUX-ism text_len knob overruns the observed sequence)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCollectorTextLenClamp:
    def _toy(self, seed=5):
        from _hrdit_fixtures import make_toy_dit
        # seq_len = text_len(8) + 4*4 = 24.
        return make_toy_dit(num_layers=2, heads=3, dim=8, text_len=8,
                            img_hw=4, seed=seed, dtype=torch.float64)

    def test_clamp_no_crash_when_knob_exceeds_seq(self):
        """A ``text_len`` knob larger than the observed sequence no longer raises
        ``ValueError: band_compute_cost: text_len exceeds seq_len``; it returns a
        valid rectangular table."""
        num_scopes = 5
        dit = self._toy()
        quality, compute = collect_scope_scores(
            dit.forward, _make_loss_fn(dit), num_scopes, text_len=512,
            chunk=4096, scale=1.0,
        )
        assert quality.shape == (2, 3, num_scopes)
        assert compute.shape == (2, 3, num_scopes)
        assert torch.isfinite(quality).all()
        assert torch.isfinite(compute).all()

    def test_clamp_equals_explicit_boundary(self):
        """``text_len=512`` (clamped to seq=24) == passing ``text_len=24``
        explicitly — the clamp is exactly the boundary value."""
        num_scopes = 5
        q_clamped, c_clamped = collect_scope_scores(
            self._toy().forward, _make_loss_fn(self._toy()),
            num_scopes, text_len=512, chunk=4096, scale=1.0,
        )
        q_explicit, c_explicit = collect_scope_scores(
            self._toy().forward, _make_loss_fn(self._toy()),
            num_scopes, text_len=24, chunk=4096, scale=1.0,
        )
        assert torch.allclose(q_clamped, q_explicit, atol=1e-10)
        assert torch.allclose(c_clamped, c_explicit, atol=1e-10)

    def test_clamp_logs_warning(self, caplog):
        """When the knob exceeds the observed sequence, a WARNING names the knob,
        the observed length, and the clamped value."""
        import logging
        dit = self._toy()
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            collect_scope_scores(
                dit.forward, _make_loss_fn(dit), 4, text_len=512,
                chunk=4096, scale=1.0,
            )
        warns = [r.getMessage() for r in caplog.records
                 if "exceeds the observed attention" in r.getMessage()]
        assert len(warns) == 1
        m = warns[0]
        assert "512" in m
        assert "(24)" in m
        assert "clamped to 24" in m

    def test_no_warning_when_knob_valid(self, caplog):
        """A ``text_len`` within ``[0, seq]`` triggers NO clamp warning."""
        import logging
        dit = self._toy()
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            collect_scope_scores(
                dit.forward, _make_loss_fn(dit), 4, text_len=8,
                chunk=4096, scale=1.0,
            )
        warns = [r.getMessage() for r in caplog.records
                 if "exceeds the observed attention" in r.getMessage()]
        assert warns == []


# ---------------------------------------------------------------------------
# T6.3 — calibration script dry-run (mock_integration)
# ---------------------------------------------------------------------------

def _load_calibrate_module():
    """Import ``calibration/calibrate_hap.py`` by path (it is not a package)."""
    import importlib.util
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "calibration", "calibrate_hap.py")
    spec = importlib.util.spec_from_file_location("calibrate_hap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.mock_integration
class TestCalibrateScriptDryRun:
    def test_calibrate_script_dry_run(self, tmp_path):
        """``--dry_run`` runs the full collector → solver → JSON pipeline on the
        toy model and writes a valid, round-trippable plan file (CI-safe, no
        GPU model)."""
        mod = _load_calibrate_module()
        out = tmp_path / "scope_plan_dry.json"
        rc = mod.main([
            "--dry_run",
            "--num_prompts", "2",
            "--num_scopes", "6",
            "--budget_ratio", "0.5",
            "--dry_layers", "3",
            "--dry_heads", "4",
            "--out", str(out),
        ])
        assert rc == 0
        assert out.exists()

        # The written plan is valid and round-trips through ScopePlan.
        from src.hap import ScopePlan
        plan = ScopePlan.load(str(out))
        assert plan.num_layers == 3
        assert plan.num_heads == 4

        import json
        with open(out, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        assert set(d.keys()) == {"alphas", "betas"}
        # All alphas are 0; betas in the valid scope→beta range (0, 1].
        assert all(all(a == 0.0 for a in row) for row in d["alphas"])
        assert all(0.0 < b <= 1.0 for row in d["betas"] for b in row)

    def test_calibrate_script_dry_run_deterministic(self, tmp_path):
        """Two dry-runs with identical args produce byte-identical plans."""
        mod = _load_calibrate_module()
        outs = []
        for i in range(2):
            out = tmp_path / f"plan_{i}.json"
            mod.main([
                "--dry_run", "--num_prompts", "2", "--num_scopes", "5",
                "--budget_ratio", "0.5", "--dry_layers", "2", "--dry_heads", "3",
                "--out", str(out),
            ])
            outs.append(out.read_text(encoding="utf-8"))
        assert outs[0] == outs[1]
