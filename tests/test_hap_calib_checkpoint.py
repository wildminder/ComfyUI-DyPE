"""Tests for memory-bounded HAP calibration (per-block gradient checkpointing).

Covers the OOM fix (``.dev/docs/oom.md``): the calibration backward retained
O(layers) of autograd ACTIVATION memory (~0.81 GiB/layer on Krea2).  The fix
wraps each transformer block in ``torch.utils.checkpoint(..., use_reentrant=False)``
so peak activation memory is bounded to ~one block, and records the
chunked-attention leaves created during the BACKWARD RECOMPUTE (the only ones
that receive ``.grad`` under checkpointing) via a forward-capture /
backward-replay layer-key scheme.

Tested here (all CPU-safe, no real model / no CUDA required):

- :func:`src.hap_calib_node._find_block_list` — generic block detection
  (single list, multi-list Flux-style, identity dedup, empty).
- :func:`src.hap_calib_node._install_block_checkpointing` /
  :func:`_uninstall_block_checkpointing` — wrap + restore block forwards.
- :func:`src.hap_calib_node._flush_gpu_allocator` — no-op safe without CUDA.
- **Correctness (the critical test):** ``collect_scope_scores_for_model`` on a
  block-structured toy DiT (checkpointing auto-active) produces scores IDENTICAL
  to (a) the same collector with checkpointing disabled and (b) a dense
  single-shot oracle.  This validates ``use_reentrant=False`` recompute-with-grad
  and the forward-capture/backward-replay layer keys.
- Chunk-size invariance under checkpointing.

Markers: @pytest.mark.unit
Accept (user-run):
    pytest tests/test_hap_calib_checkpoint.py
"""

import sys
import types

import pytest
import torch

from src import hap_calib_node as hcn

# ---------------------------------------------------------------------------
# Toy block-structured DiT (real nn.Module blocks so checkpointing activates)
# ---------------------------------------------------------------------------

class _CalibBlock(torch.nn.Module):
    """One transformer block: a single patched-attention call folded into ``h``.

    q/k/v are derived from the block INPUT ``h`` through a learnable projection
    (``self.proj``), exactly like a real DiT block's qkv-projection.  This is
    ESSENTIAL: gradient checkpointing only triggers the backward RECOMPUTE when
    the block output depends non-trivially on its inputs/parameters.  A block
    whose attention used fresh random q/k/v (independent of ``h``) would have
    only the trivial identity residual ``h + out`` as input-dependence, so
    backward would never recompute it — and the recomputed attention leaves (the
    ones that receive ``.grad`` under checkpointing) would never be created.

    The projection is seeded at construction so the forward is deterministic.
    """

    def __init__(self, heads, seq_len, dim, seed, dtype):
        super().__init__()
        self.heads = heads
        self.seq_len = seq_len
        self.dim = dim
        self.dtype = dtype
        c = heads * dim
        g = torch.Generator().manual_seed(seed)
        # Learnable qkv projection: makes out depend on h AND on a parameter,
        # forcing the checkpoint recompute during backward.
        self.proj = torch.nn.Parameter(
            torch.randn(c, c, generator=g, dtype=dtype) * 0.1
        )

    def forward(self, h):
        attn_mod = sys.modules["comfy.ldm.modules.attention"]
        # h: (1, T, C).  Project to q (=k=v) so the attention is square and
        # depends on h + self.proj.
        x = h @ self.proj                                   # (1, T, C)
        q = x.reshape(1, self.seq_len, self.heads, self.dim).permute(0, 2, 1, 3)
        out = attn_mod.optimized_attention(q, q, q, self.heads)
        # Normalize to (1, seq_len, heads*dim) regardless of the attention's
        # output convention: the pristine mock returns (B, H, T, D) while the
        # patched ``chunked_attn`` returns (B, T, H*D).  Reshape only the 4D
        # case so the downstream arithmetic is bit-identical across oracles.
        if out.dim() == 4:
            b, hh, t, d = out.shape
            out = out.permute(0, 2, 1, 3).reshape(b, t, hh * d)
        return h + out


class _BlockDiT(torch.nn.Module):
    """Minimal DiT exposing a ``blocks`` ModuleList (Krea2/FLUX-like)."""

    def __init__(self, num_layers, heads, dim, text_len, img_hw, seed, dtype):
        super().__init__()
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.seq_len = text_len + img_hw * img_hw
        self.blocks = torch.nn.ModuleList([
            _CalibBlock(heads, self.seq_len, dim, seed + i, dtype)
            for i in range(num_layers)
        ])
        # A NON-ZERO learnable input that seeds the hidden state.  This is
        # ESSENTIAL for the checkpoint recompute to fire: the block outputs must
        # depend non-trivially on a grad-requiring tensor so backward has a real
        # path to traverse (and thus re-runs each block's forward, creating the
        # recomputed attention leaves that receive ``.grad``).  A zero seed would
        # make every ``h @ proj`` zero and the output independent of the block
        # parameters, so backward would never recompute.
        g = torch.Generator().manual_seed(seed + 999)
        self.input_param = torch.nn.Parameter(
            torch.randn(1, self.seq_len, heads * dim, generator=g, dtype=dtype)
        )

    def forward(self):
        h = self.input_param  # non-zero, requires_grad -> roots the graph
        for blk in self.blocks:
            h = blk(h)
        return h


class _MultiListDiT(_BlockDiT):
    """FLUX-style model with TWO block lists (double then single)."""

    def __init__(self, num_double, num_single, heads, dim, text_len, img_hw, seed, dtype):
        # Bypass _BlockDiT.__init__'s single ``blocks`` list.
        torch.nn.Module.__init__(self)
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.seq_len = text_len + img_hw * img_hw
        self.double_blocks = torch.nn.ModuleList([
            _CalibBlock(heads, self.seq_len, dim, seed + i, dtype)
            for i in range(num_double)
        ])
        self.single_blocks = torch.nn.ModuleList([
            _CalibBlock(heads, self.seq_len, dim, seed + 100 + i, dtype)
            for i in range(num_single)
        ])

    def forward(self):
        h = torch.zeros(1, self.seq_len, self.heads * self.dim, dtype=self.dtype)
        for blk in self.double_blocks:
            h = blk(h)
        for blk in self.single_blocks:
            h = blk(h)
        return h


class _FakeModelPatcher:
    """Minimal ModelPatcher stand-in exposing ``model.diffusion_model``."""

    def __init__(self, diffusion_model):
        self.model = types.SimpleNamespace(diffusion_model=diffusion_model)


def _make_loss_fn(dit):
    """Deterministic differentiable MSE loss vs a seeded target."""
    g = torch.Generator().manual_seed(123)
    target = torch.randn(1, dit.seq_len, dit.heads * dit.dim, generator=g,
                         dtype=torch.float64)

    def loss_fn(output):
        return torch.nn.functional.mse_loss(output, target)

    return loss_fn


def _make_case(num_layers=3, heads=2, dim=6, text_len=4, img_hw=3, seed=7):
    """Build (model_patcher, dit, loss_fn) in fp64 for exact comparison."""
    dit = _BlockDiT(num_layers, heads, dim, text_len, img_hw, seed,
                    dtype=torch.float64)
    return _FakeModelPatcher(dit), dit, _make_loss_fn(dit)


def _dense_reference(dit, loss_fn, num_scopes, text_len, scale=1.0):
    """Dense single-shot oracle: materialize the FULL ``A`` once per layer as a
    grad leaf, run the same forward+loss+backward, score with
    :func:`estimate_head_scope_costs`.  Returns ``(L, H, S)`` fp64.

    The dense attention mirrors the patched attention's DEFAULT output convention
    (``(B, T, H*D)``) so the block's downstream arithmetic is bit-identical.
    """
    from src.hap_calib import estimate_head_scope_costs

    attn_mod = sys.modules["comfy.ldm.modules.attention"]
    orig = attn_mod.optimized_attention
    dense_As = []

    def dense_attn(q, k, v, heads, *args, **kwargs):
        A = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * scale, dim=-1)
        A = A.detach().requires_grad_(True)
        dense_As.append(A)
        out4 = torch.matmul(A, v)  # (B, H, T, D)
        b, h, t, d = out4.shape
        return out4.permute(0, 2, 1, 3).reshape(b, t, h * d)

    attn_mod.optimized_attention = dense_attn
    try:
        out = dit.forward()
        loss = loss_fn(out)
        loss.backward()
    finally:
        attn_mod.optimized_attention = orig

    layers = []
    for A in dense_As:
        layers.append(estimate_head_scope_costs(
            A[0].to(torch.float64), A.grad[0].to(torch.float64),
            num_scopes, text_len,
        ))
    return torch.stack(layers, dim=0)


# ---------------------------------------------------------------------------
# _find_block_list — generic block detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFindBlockList:
    def test_single_modulelist(self):
        dit = _BlockDiT(3, 2, 4, 2, 2, 0, torch.float32)
        blocks = hcn._find_block_list(dit)
        assert len(blocks) == 3
        assert all(isinstance(b, torch.nn.Module) for b in blocks)
        # Forward order preserved.
        assert blocks == list(dit.blocks)

    def test_multi_list_flux_order_and_dedup(self):
        dit = _MultiListDiT(2, 3, 2, 4, 2, 2, 0, torch.float32)
        blocks = hcn._find_block_list(dit)
        # double_blocks first, then single_blocks (forward order).
        assert len(blocks) == 5
        assert blocks[:2] == list(dit.double_blocks)
        assert blocks[2:] == list(dit.single_blocks)

    def test_dedup_by_identity(self):
        """A model exposing the SAME list under two attribute names is deduped."""
        dit = _BlockDiT(2, 2, 4, 2, 2, 0, torch.float32)
        # Alias the same ModuleList under a second probed name.
        dit.layers = dit.blocks
        blocks = hcn._find_block_list(dit)
        assert len(blocks) == 2  # not 4

    def test_no_blocks_returns_empty(self):
        empty = torch.nn.Module()
        assert hcn._find_block_list(empty) == []
        assert hcn._find_block_list(None) == []

    def test_plain_list_of_modules_accepted(self):
        dit = torch.nn.Module()
        dit.blocks = [torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)]
        blocks = hcn._find_block_list(dit)
        assert len(blocks) == 2


# ---------------------------------------------------------------------------
# install / uninstall block checkpointing
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestInstallCheckpointing:
    def test_install_wraps_and_uninstall_restores(self):
        dit = _BlockDiT(2, 2, 4, 2, 2, 0, torch.float32)
        originals = [b.forward for b in dit.blocks]
        installed = hcn._install_block_checkpointing(dit)
        assert len(installed) == 2
        # Forwards are replaced.
        for b, orig in zip(dit.blocks, originals):
            assert b.forward is not orig
        hcn._uninstall_block_checkpointing(installed)
        # Forwards restored to the exact original bound methods.
        for b, orig in zip(dit.blocks, originals):
            assert b.forward == orig

    def test_install_on_model_without_blocks_is_noop(self):
        empty = torch.nn.Module()
        assert hcn._install_block_checkpointing(empty) == []
        assert hcn._install_block_checkpointing(None) == []

    def test_checkpointed_forward_matches_eager(self):
        """Wrapping blocks in checkpoint does NOT change the forward output."""
        torch.manual_seed(0)
        dit = _BlockDiT(2, 2, 4, 2, 2, 3, torch.float64)
        # Eager output (attention = pristine SDPA from the conftest fixture).
        eager = dit.forward().detach().clone()
        installed = hcn._install_block_checkpointing(dit)
        try:
            ckpt = dit.forward().detach().clone()
        finally:
            hcn._uninstall_block_checkpointing(installed)
        assert torch.allclose(eager, ckpt, atol=1e-12)


# ---------------------------------------------------------------------------
# _flush_gpu_allocator — safe no-op without CUDA
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFlushAllocator:
    def test_flush_does_not_raise(self):
        # Must be a safe no-op on CPU-only environments.
        hcn._flush_gpu_allocator()


# ---------------------------------------------------------------------------
# Correctness: checkpointed collector == disabled-checkpoint == dense oracle
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCheckpointedCollectorCorrectness:
    def test_checkpointed_equals_disabled_checkpoint(self, monkeypatch):
        """Scores with checkpointing auto-active == scores with checkpointing
        forcibly disabled (same model, same loss).  Isolates the effect of the
        checkpoint + forward-capture/backward-replay machinery."""
        num_scopes, text_len = 5, 4
        model, dit, loss_fn = _make_case()

        # Checkpointing active (blocks present -> auto-installed).
        q_ckpt, c_ckpt, seq_ckpt = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=1.0,
        )

        # Checkpointing disabled via monkeypatch.
        monkeypatch.setattr(hcn, "_install_block_checkpointing", lambda dm: [])
        model2, dit2, loss_fn2 = _make_case()
        q_off, c_off, seq_off = hcn.collect_scope_scores_for_model(
            model=model2, model_type="flux", forward_fn=dit2.forward,
            loss_fn=loss_fn2, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=1.0,
        )

        assert q_ckpt.shape == q_off.shape == (3, 2, num_scopes)
        assert seq_ckpt == seq_off
        assert torch.allclose(q_ckpt, q_off, atol=1e-8)
        assert torch.allclose(c_ckpt, c_off, atol=1e-8)

    def test_checkpointed_equals_dense_oracle(self):
        """Checkpointed collected scores == dense single-shot oracle (fp64,
        atol 1e-8).  The gold-standard correctness check."""
        num_scopes, text_len = 5, 4
        model, dit, loss_fn = _make_case()

        quality, compute, seq = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=1.0,
        )

        ref = _dense_reference(_make_case()[1], _make_case()[2],
                               num_scopes, text_len)
        assert quality.shape == ref.shape
        assert torch.allclose(quality, ref, atol=1e-8)

    def test_chunk_invariance_under_checkpointing(self):
        """chunk=2 vs chunk=4096 give IDENTICAL scores under checkpointing —
        chunking remains a pure memory knob."""
        num_scopes, text_len = 4, 4
        m1, d1, l1 = _make_case()
        q_small, _, _ = hcn.collect_scope_scores_for_model(
            model=m1, model_type="flux", forward_fn=d1.forward, loss_fn=l1,
            num_scopes=num_scopes, text_len=text_len, chunk=2, scale=1.0,
        )
        m2, d2, l2 = _make_case()
        q_big, _, _ = hcn.collect_scope_scores_for_model(
            model=m2, model_type="flux", forward_fn=d2.forward, loss_fn=l2,
            num_scopes=num_scopes, text_len=text_len, chunk=4096, scale=1.0,
        )
        # Chunking is a pure memory knob: scores are invariant to within fp64
        # accumulation-order rounding (sub-ulp).  Use a tight allclose rather
        # than bitwise equal.
        assert torch.allclose(q_small, q_big, atol=1e-10)

    def test_block_forwards_restored_after_collection(self):
        """Collection restores every block's original forward (no leak), even
        though checkpointing was active during the run."""
        model, dit, loss_fn = _make_case()
        originals = [b.forward for b in dit.blocks]
        hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
        )
        for b, orig in zip(dit.blocks, originals):
            assert b.forward == orig

    def test_multi_list_model_collects_all_layers(self):
        """A Flux-style double+single model yields one calibrated layer per
        block across BOTH lists, in forward order."""
        heads, dim, text_len, img_hw = 2, 6, 4, 3
        num_scopes = 4
        dit = _MultiListDiT(2, 3, heads, dim, text_len, img_hw, seed=11,
                            dtype=torch.float64)
        model = _FakeModelPatcher(dit)
        quality, _, seq = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=_make_loss_fn(dit), num_scopes=num_scopes,
            text_len=text_len, chunk=4096, scale=1.0,
        )
        # 2 double + 3 single = 5 calibrated layers.
        assert quality.shape == (5, heads, num_scopes)


# ---------------------------------------------------------------------------
# Backward-recompute recording gate
# (plan 2026-08-17-hap-calib-backward-recompute-no-grad-fix)
# ---------------------------------------------------------------------------
#
# ``use_reentrant=False`` checkpointing RE-RUNS each block's forward during
# ``loss.backward()``; the still-patched ``chunked_attn`` fires again.  Without a
# gate those recompute calls would append spurious ``phase='backward'`` records
# whose orphaned leaves never receive ``.grad`` (the live Krea2 crash at layer
# 32).  The fix keeps running ``chunked_attention`` during the recompute (checkpoint
# requires identical ops/shapes — a passthrough raises ``CheckpointError``) but
# SKIPS the recording + counter bump when ``phase != "forward"``.

def _parse_diag(caplog):
    """Extract ``(total, forward, backward_recompute, chunks, with_grad,
    missing)`` from the ``[HAP calib][diag]`` log line, or ``None``."""
    import re
    for r in caplog.records:
        m = r.getMessage()
        if "[HAP calib][diag]" not in m:
            continue
        g = re.search(
            r"total_records=(\d+) \(forward=(\d+), backward_recompute=(\d+)\) "
            r"total_chunks=(\d+) chunks_with_grad=(\d+) chunks_missing_grad=(\d+)",
            m,
        )
        if g:
            return tuple(int(x) for x in g.groups())
    return None


@pytest.mark.unit
class TestBackwardRecomputeGate:
    def test_recompute_fires_patched_attention(self, monkeypatch):
        """P0/mechanism: the backward recompute calls ``chunked_attention`` again
        — total invocations == 2 * num_blocks (forward + recompute).  This is the
        precondition that, ungated, would append ``phase='backward'`` records.
        NOTE: the gate must NOT stop this call (checkpoint needs identical ops);
        it only suppresses the *recording*."""
        from src import hap_calib
        calls = [0]
        orig = hap_calib.chunked_attention

        def counting(*a, **k):
            calls[0] += 1
            return orig(*a, **k)

        monkeypatch.setattr(hap_calib, "chunked_attention", counting)

        num_layers = 3
        model, dit, loss_fn = _make_case(num_layers=num_layers)
        hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
        )
        # Forward (3) + backward recompute (3) = 6.
        assert calls[0] == 2 * num_layers

    def test_gate_excludes_backward_records(self, caplog):
        """P2: with the gate active, EVERY record is ``phase='forward'`` and the
        record count equals the block count (no recompute duplicates); all chunk
        leaves received ``.grad``."""
        import logging
        model, dit, loss_fn = _make_case(num_layers=3)
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=dit.forward,
                loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
            )
        diag = _parse_diag(caplog)
        assert diag is not None
        total, forward, backward_recompute, chunks, with_grad, missing = diag
        assert backward_recompute == 0
        assert forward == total == 3
        assert missing == 0
        assert with_grad == chunks

    def test_call_counter_not_polluted_by_recompute(self, caplog):
        """P2: observed layer keys are ``0..num_blocks-1`` in order — the
        recompute never bumps the counter (asserted via the ``[mem]`` layer_key
        sequence)."""
        import logging
        import re
        model, dit, loss_fn = _make_case(num_layers=3)
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=dit.forward,
                loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
            )
        keys = []
        for r in caplog.records:
            m = r.getMessage()
            if "[HAP calib][mem]" not in m:
                continue
            g = re.search(r"layer_key=(\d+)", m)
            if g:
                keys.append(int(g.group(1)))
        assert keys == [0, 1, 2]

    def test_non_checkpointed_path_unchanged(self, monkeypatch, caplog):
        """P2: with checkpointing disabled (no recompute) the gate is a no-op —
        record count == num_blocks, all forward, all grad."""
        import logging
        monkeypatch.setattr(hcn, "_install_block_checkpointing", lambda dm: [])
        model, dit, loss_fn = _make_case(num_layers=3)
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=dit.forward,
                loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
            )
        diag = _parse_diag(caplog)
        assert diag is not None
        total, forward, backward_recompute, chunks, with_grad, missing = diag
        assert backward_recompute == 0
        assert forward == total == 3
        assert missing == 0

    def test_scores_unchanged_by_gate(self):
        """P3: scores with the gate active == dense single-shot oracle (fp64,
        atol 1e-8).  The gate must not alter the calibrated values."""
        num_scopes, text_len = 5, 4
        model, dit, loss_fn = _make_case()
        quality, _, _ = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=1.0,
        )
        ref = _dense_reference(_make_case()[1], _make_case()[2],
                               num_scopes, text_len)
        assert quality.shape == ref.shape
        assert torch.allclose(quality, ref, atol=1e-8)


# ---------------------------------------------------------------------------
# Heterogeneous head-count filter
# (Krea2 live crash: stack got [20, 50] vs [48, 50] — auxiliary projector
#  attention has a different head count than the main transformer blocks)
# ---------------------------------------------------------------------------

class _HeteroDiT(torch.nn.Module):
    """DiT with MIXED attention head counts.

    3 MAIN blocks (heads=2, dim=6) + 1 AUXILIARY block (heads=3, dim=4).  All
    share hidden dim C=12 so the residual chain ``h + out`` stays shape-
    compatible, but the attention HEAD counts differ (2 vs 3) — reproducing the
    Krea2 live crash (20-head auxiliary vs 48-head main).  The auxiliary block
    sits at forward index 1 so its layer key is 1.
    """

    def __init__(self, text_len, img_hw, seed, dtype):
        super().__init__()
        self.dtype = dtype
        self.seq_len = text_len + img_hw * img_hw
        self.hidden = 12  # heads*dim for EVERY block (chain-compatible)
        C = self.hidden
        self.blocks = torch.nn.ModuleList([
            _CalibBlock(2, self.seq_len, 6, seed + 0, dtype),  # main, key 0
            _CalibBlock(3, self.seq_len, 4, seed + 1, dtype),  # aux,  key 1
            _CalibBlock(2, self.seq_len, 6, seed + 2, dtype),  # main, key 2
            _CalibBlock(2, self.seq_len, 6, seed + 3, dtype),  # main, key 3
        ])
        g = torch.Generator().manual_seed(seed + 999)
        self.input_param = torch.nn.Parameter(
            torch.randn(1, self.seq_len, C, generator=g, dtype=dtype)
        )

    def forward(self):
        h = self.input_param
        for blk in self.blocks:
            h = blk(h)
        return h


class _HeteroTieDiT(_HeteroDiT):
    """Tie-break fixture: 2 blocks of heads=2 then 2 blocks of heads=3 (equal
    counts; heads=2 appears FIRST so it must win the tie)."""

    def __init__(self, text_len, img_hw, seed, dtype):
        torch.nn.Module.__init__(self)
        self.dtype = dtype
        self.seq_len = text_len + img_hw * img_hw
        self.hidden = 12
        C = self.hidden
        self.blocks = torch.nn.ModuleList([
            _CalibBlock(2, self.seq_len, 6, seed + 0, dtype),  # heads=2, key 0
            _CalibBlock(2, self.seq_len, 6, seed + 1, dtype),  # heads=2, key 1
            _CalibBlock(3, self.seq_len, 4, seed + 2, dtype),  # heads=3, key 2
            _CalibBlock(3, self.seq_len, 4, seed + 3, dtype),  # heads=3, key 3
        ])
        g = torch.Generator().manual_seed(seed + 999)
        self.input_param = torch.nn.Parameter(
            torch.randn(1, self.seq_len, C, generator=g, dtype=dtype)
        )


def _make_hetero_case(dit_cls=_HeteroDiT, text_len=4, img_hw=3, seed=7):
    """Build (model_patcher, dit, loss_fn) for a heterogeneous-head toy."""
    dit = dit_cls(text_len, img_hw, seed, dtype=torch.float64)
    g = torch.Generator().manual_seed(123)
    target = torch.randn(1, dit.seq_len, dit.hidden, generator=g,
                         dtype=torch.float64)

    def loss_fn(output):
        return torch.nn.functional.mse_loss(output, target)

    return _FakeModelPatcher(dit), dit, loss_fn


@pytest.mark.unit
class TestHeterogeneousHeadFilter:
    def test_hetero_produces_rectangular_plan(self):
        """The collector no longer crashes on mixed head counts; it returns a
        rectangular ``(L, H, S)`` table over the DOMINANT head count only."""
        num_scopes, text_len = 5, 4
        model, dit, loss_fn = _make_hetero_case()
        quality, compute, seq = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=1.0,
        )
        # Dominant head count = 2 (3 main layers); the single 3-head aux layer
        # is excluded.  Rectangular (3, 2, S).
        assert quality.shape == (3, 2, num_scopes)
        assert compute.shape == (3, 2, num_scopes)

    def test_hetero_selects_dominant_not_total(self):
        """Layer count == dominant head-count population (3), NOT the total
        collected calls (4)."""
        num_scopes, text_len = 4, 4
        model, dit, loss_fn = _make_hetero_case()
        quality, _, _ = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=1.0,
        )
        assert quality.shape[0] == 3  # not 4

    def test_hetero_logs_exclusion(self, caplog):
        """The filter logs the detected head-count histogram, the dominant
        choice, and the excluded auxiliary layer keys."""
        import logging
        model, dit, loss_fn = _make_hetero_case()
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=dit.forward,
                loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
            )
        msgs = [r.getMessage() for r in caplog.records
                if "heterogeneous head counts" in r.getMessage()]
        assert len(msgs) == 1
        m = msgs[0]
        assert "2: 3" in m and "3: 1" in m   # histogram heads->count
        assert "dominant head count (2 heads" in m
        assert "keys=[1]" in m               # aux layer key excluded

    def test_tie_break_first_occurrence_wins(self):
        """Equal head-count populations -> the head count appearing FIRST in
        forward order wins (deterministic)."""
        num_scopes, text_len = 4, 4
        model, dit, loss_fn = _make_hetero_case(dit_cls=_HeteroTieDiT)
        quality, _, _ = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=1.0,
        )
        # heads=2 and heads=3 each have 2 layers; heads=2 appears first -> kept.
        assert quality.shape == (2, 2, num_scopes)

    def test_uniform_model_no_filter_log(self, caplog):
        """A uniform-head model triggers NO heterogeneous filtering (no log)."""
        import logging
        model, dit, loss_fn = _make_case(num_layers=3)
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            quality, _, _ = hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=dit.forward,
                loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
            )
        msgs = [r.getMessage() for r in caplog.records
                if "heterogeneous head counts" in r.getMessage()]
        assert msgs == []
        assert quality.shape[0] == 3  # all layers kept

    # -- excluded_head_counts metadata (2026-08-23 head-count warning fix) ----

    def test_hetero_meta_records_excluded_head_counts(self):
        """The collector records the NON-dominant head counts into ``meta``
        so the runtime can log a friendly INFO instead of a scary WARNING."""
        model, dit, loss_fn = _make_hetero_case()
        meta = {}
        hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
            meta=meta,
        )
        # Dominant = 2 heads (3 layers); the single 3-head aux layer is excluded.
        assert meta["excluded_head_counts"] == [3]

    def test_uniform_meta_empty_excluded(self):
        """A uniform-head model records an EMPTY excluded list (no aux)."""
        model, dit, loss_fn = _make_case(num_layers=3)
        meta = {}
        hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
            meta=meta,
        )
        assert meta["excluded_head_counts"] == []

    def test_meta_none_backward_compatible(self):
        """Omitting ``meta`` (the pre-fix call convention) still works and
        returns the same 3-tuple."""
        model, dit, loss_fn = _make_hetero_case()
        result = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
        )
        assert len(result) == 3  # (quality, compute, seq)


# ---------------------------------------------------------------------------
# Early GPU-leaf release during scoring (plan 2026-08-24 P1)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEarlyLeafRelease:
    """After a chunk's A/G are copied to CPU and scored, the GPU leaf must be
    released immediately instead of staying resident until function exit."""

    def test_free_scored_chunk_clears_grad(self):
        """``_free_scored_chunk`` drops the ``.grad`` reference."""
        t = torch.randn(2, 3, 4, requires_grad=True)
        g = torch.randn_like(t)
        t.grad = g
        hcn._free_scored_chunk(t)
        assert t.grad is None

    def test_free_scored_chunk_none_safe(self):
        """``_free_scored_chunk(None)`` is a no-op (defensive)."""
        hcn._free_scored_chunk(None)  # must not raise

    def test_chunks_freed_after_scoring(self):
        """Every recorded chunk slot is cleared once scored — verified via the
        ``meta["chunks_freed"]`` flag and by monkeypatching
        ``_free_scored_chunk`` to count calls equal to total chunks."""
        num_scopes = 4
        model, dit, loss_fn = _make_case(num_layers=2)
        calls = {"n": 0}
        orig = hcn._free_scored_chunk

        def spy(chunk):
            if chunk is not None:
                calls["n"] += 1
            return orig(chunk)

        monkey = pytest.MonkeyPatch()
        monkey.setattr(hcn, "_free_scored_chunk", spy)
        try:
            meta = {}
            quality, _, _ = hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=dit.forward,
                loss_fn=loss_fn, num_scopes=num_scopes, text_len=4,
                chunk=4096, scale=1.0, meta=meta,
            )
        finally:
            monkey.undo()
        assert meta.get("chunks_freed") is True
        # 2 layers x 1 call each; chunk=4096 > seq_len=13 -> 1 chunk per call.
        assert calls["n"] == 2
        assert quality.shape == (2, 2, num_scopes)

    def test_scores_identical_with_early_release(self):
        """Early release must NOT change results: scores with the freeing spy
        active equal scores from an untouched run (same seeds)."""
        num_scopes = 5
        m1, d1, l1 = _make_case()
        q_ref, c_ref, _ = hcn.collect_scope_scores_for_model(
            model=m1, model_type="flux", forward_fn=d1.forward, loss_fn=l1,
            num_scopes=num_scopes, text_len=4, chunk=4096, scale=1.0,
        )
        m2, d2, l2 = _make_case()
        q_new, c_new, _ = hcn.collect_scope_scores_for_model(
            model=m2, model_type="flux", forward_fn=d2.forward, loss_fn=l2,
            num_scopes=num_scopes, text_len=4, chunk=4096, scale=1.0,
        )
        assert torch.equal(q_ref, q_new)
        assert torch.equal(c_ref, c_new)

    def test_missing_grad_still_raises(self):
        """The no-grad error path still fires; freeing logic must not mask it."""
        model, dit, loss_fn = _make_case(num_layers=2)

        def grad_killer():
            out = dit.forward()
            return out.detach().requires_grad_(True)  # decouples attention

        with pytest.raises(RuntimeError, match="no gradient"):
            hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=grad_killer,
                loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096,
                scale=1.0,
            )


# ---------------------------------------------------------------------------
# Post-scoring purge (plan 2026-08-24 P2/P3)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCalibrationMemoryPurge:
    def test_purge_prefers_comfy_soft_empty_cache(self, monkeypatch):
        """With comfy available, soft_empty_cache is used and raw empty_cache
        is NOT."""
        import sys
        called = {"soft": 0, "raw": 0}
        fake_mm = types.ModuleType("comfy.model_management")
        fake_mm.soft_empty_cache = lambda: called.__setitem__("soft", 1)
        fake_pkg = types.ModuleType("comfy")
        fake_pkg.model_management = fake_mm
        monkeypatch.setitem(sys.modules, "comfy", fake_pkg)
        monkeypatch.setitem(sys.modules, "comfy.model_management", fake_mm)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
        monkeypatch.setattr(torch.cuda, "empty_cache",
                            lambda: called.__setitem__("raw", 1), raising=False)
        hcn._purge_calibration_memory()
        assert called["soft"] == 1
        assert called["raw"] == 0

    def test_purge_falls_back_to_torch_when_comfy_missing(self, monkeypatch):
        """Without comfy, raw torch.cuda.empty_cache runs when CUDA exists."""
        import builtins
        import sys
        called = {"raw": 0}
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name == "comfy.model_management" or name == "comfy":
                raise ImportError("no comfy")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for mod in ("comfy", "comfy.model_management"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True, raising=False)
        monkeypatch.setattr(torch.cuda, "empty_cache",
                            lambda: called.__setitem__("raw", 1), raising=False)
        hcn._purge_calibration_memory()
        assert called["raw"] == 1

    def test_purge_never_raises(self, monkeypatch):
        """Both paths raising still leaves _purge silent."""
        import builtins
        import sys
        real_import = builtins.__import__

        def fake_import(name, *a, **kw):
            if name.startswith("comfy"):
                raise RuntimeError("boom")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        for mod in ("comfy", "comfy.model_management"):
            monkeypatch.delitem(sys.modules, mod, raising=False)

        def boom():
            raise RuntimeError("cuda boom")

        monkeypatch.setattr(torch.cuda, "is_available", boom, raising=False)
        hcn._purge_calibration_memory()  # must not raise

    def test_purge_noop_without_cuda(self, monkeypatch):
        """No CUDA -> fallback path does nothing (and never raises)."""
        import sys
        monkeypatch.setitem(sys.modules, "comfy", None)  # import fails
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False, raising=False)
        hcn._purge_calibration_memory()  # must not raise

    def test_collector_calls_purge_after_scoring(self, monkeypatch):
        """The collector purges once after scoring completes."""
        n = {"purges": 0}
        monkeypatch.setattr(hcn, "_purge_calibration_memory",
                            lambda: n.__setitem__("purges", n["purges"] + 1))
        model, dit, loss_fn = _make_case(num_layers=2)
        hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
        )
        assert n["purges"] == 1


# ---------------------------------------------------------------------------
# text_len clamp (Krea2 live crash: band_compute_cost text_len=512 exceeds
# seq_len=430 — the FLUX-ism text_len knob overruns the observed sequence when
# calibration runs at a reduced resolution / the model's real text length is
# below the knob)
# ---------------------------------------------------------------------------
#
# The cost model requires ``text_len <= seq_len`` (seq = text + image).  The fix
# clamps ``text_len`` to ``[0, observed_seq]`` — mirroring the HAP runtime's
# ``max(0, min(text_len, seq_len))`` (src/hap.py HapRuntime.attn) — and threads
# the clamped value through BOTH the quality-scoring loop and the cost table.

@pytest.mark.unit
class TestTextLenClamp:
    def test_clamp_no_crash_when_knob_exceeds_seq(self):
        """The primary regression: a ``text_len`` knob larger than the observed
        sequence no longer raises ``ValueError: band_compute_cost: text_len
        exceeds seq_len``; it returns a valid rectangular table."""
        num_scopes = 5
        model, dit, loss_fn = _make_case()  # seq_len = 4 + 3*3 = 13
        # text_len=512 >> seq_len=13 — the exact live-crash shape.
        quality, compute, seq = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=512,
            chunk=4096, scale=1.0,
        )
        assert seq == 13
        assert quality.shape == (3, 2, num_scopes)
        assert compute.shape == (3, 2, num_scopes)
        assert torch.isfinite(quality).all()
        assert torch.isfinite(compute).all()

    def test_clamp_equals_explicit_boundary(self):
        """``text_len=512`` (clamped to seq) produces scores IDENTICAL to passing
        ``text_len=seq_len`` explicitly — the clamp is exactly the boundary value,
        not an approximation."""
        num_scopes = 5
        m1, d1, l1 = _make_case()
        q_clamped, c_clamped, seq = hcn.collect_scope_scores_for_model(
            model=m1, model_type="flux", forward_fn=d1.forward, loss_fn=l1,
            num_scopes=num_scopes, text_len=512, chunk=4096, scale=1.0,
        )
        m2, d2, l2 = _make_case()
        q_explicit, c_explicit, _ = hcn.collect_scope_scores_for_model(
            model=m2, model_type="flux", forward_fn=d2.forward, loss_fn=l2,
            num_scopes=num_scopes, text_len=seq, chunk=4096, scale=1.0,
        )
        assert torch.allclose(q_clamped, q_explicit, atol=1e-10)
        assert torch.allclose(c_clamped, c_explicit, atol=1e-10)

    def test_clamp_logs_warning(self, caplog):
        """When the knob exceeds the observed sequence, a WARNING names the knob,
        the observed length, and the clamped value."""
        import logging
        model, dit, loss_fn = _make_case()
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=dit.forward,
                loss_fn=loss_fn, num_scopes=4, text_len=512, chunk=4096, scale=1.0,
            )
        warns = [r.getMessage() for r in caplog.records
                 if "exceeds the observed attention" in r.getMessage()]
        assert len(warns) == 1
        m = warns[0]
        assert "512" in m      # the knob
        assert "(13)" in m     # the observed sequence length
        assert "clamped to 13" in m

    def test_no_warning_when_knob_valid(self, caplog):
        """A ``text_len`` within ``[0, seq]`` triggers NO clamp warning, and the
        geometry log reports effective == knob."""
        import logging
        model, dit, loss_fn = _make_case()
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=dit.forward,
                loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
            )
        warns = [r.getMessage() for r in caplog.records
                 if "exceeds the observed attention" in r.getMessage()]
        assert warns == []
        geoms = [r.getMessage() for r in caplog.records
                 if "[HAP calib][geom]" in r.getMessage()]
        assert len(geoms) == 1
        assert "knob text_len=4 effective text_len=4" in geoms[0]


# ---------------------------------------------------------------------------
# P17 — Diagnostic no_grad regression (CheckpointError: forward vs recompute)
# ---------------------------------------------------------------------------
#
# The live 2026-08-18 run #7 crashed with:
#   torch.utils.checkpoint.CheckpointError: A different number of tensors was
#   saved during the original forward and recomputation. (298 vs 280)
# Root cause: the ``[mag]`` magnitude/logit diagnostic ran autograd-TRACKED ops
# (``q4.abs().max()``, ``torch.matmul(q4, k4ᵀ)``) during the FORWARD only
# (phase-gated).  Those ops saved tensors via the checkpoint pack hook; during
# the backward RECOMPUTE the diagnostic is skipped (``phase[0]=='backward'``),
# so those tensors were never saved -> count mismatch.  Fix: wrap every
# diagnostic tensor op in ``torch.no_grad()`` so it saves nothing.
#
# The existing correctness tests use tiny sequences (T=13), so the ``[mag]``
# diagnostic (gated to T>=512) NEVER fires there and cannot catch this.  These
# tests use a large sequence so the diagnostic path is actually exercised.

def _make_large_case(num_layers=2, heads=2, dim=6, text_len=4, img_hw=23, seed=7):
    """A case with seq_len = text_len + img_hw^2 = 4 + 529 = 533 >= 512 so the
    ``[mag]`` diagnostic (T>=512 gate) FIRES during the forward."""
    dit = _BlockDiT(num_layers, heads, dim, text_len, img_hw, seed,
                    dtype=torch.float64)
    assert dit.seq_len >= 512, "test requires seq_len >= 512 to fire [mag]"
    return _FakeModelPatcher(dit), dit, _make_loss_fn(dit)


@pytest.mark.unit
class TestDiagnosticNoGradRegression:
    def test_mag_diagnostic_does_not_break_checkpoint(self, caplog):
        """PRIMARY regression: with a large sequence (T>=512) the ``[mag]``
        diagnostic FIRES during the forward, yet ``loss.backward()`` completes
        WITHOUT ``CheckpointError`` (forward and recompute save identical tensor
        counts because the diagnostic ops are under ``no_grad``)."""
        import logging
        model, dit, loss_fn = _make_large_case()
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            quality, compute, seq = hcn.collect_scope_scores_for_model(
                model=model, model_type="flux", forward_fn=dit.forward,
                loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
            )
        # The diagnostic actually fired (so this test exercises the path).
        mags = [r.getMessage() for r in caplog.records
                if "[HAP calib][mag]" in r.getMessage()]
        assert len(mags) >= 1, "expected the [mag] diagnostic to fire at T>=512"
        # And the collection completed with valid, finite scores.
        assert quality.shape == (2, 2, 4)
        assert seq == dit.seq_len
        assert torch.isfinite(quality).all()

    def test_qkv_nan_diagnostic_does_not_break_checkpoint(self, caplog):
        """The ``[qkv-nan]`` diagnostic (isnan/isinf probes) also runs under
        ``no_grad`` and must not perturb the checkpoint tensor counts.  A clean
        fp64 model produces no NaN, so this asserts the probe path is exercised
        without error even when it does NOT log."""
        model, dit, loss_fn = _make_large_case()
        quality, _, _ = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=4, text_len=4, chunk=4096, scale=1.0,
        )
        assert torch.isfinite(quality).all()

    def test_mag_scores_match_dense_oracle(self):
        """Even with the ``[mag]`` diagnostic firing (T>=512), the collected
        scores still match the dense single-shot oracle — the diagnostic is a
        pure observer and never perturbs the math."""
        num_scopes, text_len = 4, 4
        model, dit, loss_fn = _make_large_case()
        quality, _, _ = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=1.0,
        )
        m2, d2, l2 = _make_large_case()
        ref = _dense_reference(d2, l2, num_scopes, text_len)
        assert quality.shape == ref.shape
        assert torch.allclose(quality, ref, atol=1e-8)


# ---------------------------------------------------------------------------
# P19 — Scale-convention fix (ComfyUI ``attention_basic`` parity)
# ---------------------------------------------------------------------------
#
# The collector previously hardcoded ``scale=1.0``, but ComfyUI's
# ``attention_basic`` applies ``scale = kwargs.get("scale", dim_head ** -0.5)``
# INSIDE the attention function.  So during calibration the model ran with
# logits ``sqrt(dim_head)``x too large (~11.3x at head_dim=128) — calibrating
# against a WRONG attention distribution AND contributing to the fp16 logit
# overflow.  The fix mirrors the ComfyUI convention:
#   1. ``kwargs["scale"]`` (the model's explicit per-call scale) wins,
#   2. else the outer ``scale`` parameter (explicit override; tests),
#   3. else ``dim_head ** -0.5`` (ComfyUI's default).
# The orchestrator now passes ``scale=None`` (= use the convention).

@pytest.mark.unit
class TestScaleConvention:
    def test_scale_none_uses_dim_head_default(self):
        """``scale=None`` (the new orchestrator default) applies ComfyUI's
        ``dim_head ** -0.5`` convention — scores match a dense oracle that uses
        the SAME convention, NOT the old hardcoded 1.0."""
        from src.hap_calib import estimate_head_scope_costs

        num_scopes, text_len = 4, 4
        model, dit, loss_fn = _make_case()
        quality, _, _ = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=None,  # => dim_head ** -0.5
        )

        # Dense oracle using the ComfyUI convention (dim_head ** -0.5).
        attn_mod = sys.modules["comfy.ldm.modules.attention"]
        orig = attn_mod.optimized_attention
        dense_As = []

        def dense_attn(q, k, v, heads, *args, **kwargs):
            dim_head = q.shape[-1]
            s = kwargs.get("scale", dim_head ** -0.5)
            A = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * s, dim=-1)
            A = A.detach().requires_grad_(True)
            dense_As.append(A)
            out4 = torch.matmul(A, v)
            b, h, t, d = out4.shape
            return out4.permute(0, 2, 1, 3).reshape(b, t, h * d)

        attn_mod.optimized_attention = dense_attn
        try:
            m2, d2, l2 = _make_case()
            out = d2.forward()
            loss = l2(out)
            loss.backward()
        finally:
            attn_mod.optimized_attention = orig

        layers = [
            estimate_head_scope_costs(
                A[0].to(torch.float64), A.grad[0].to(torch.float64),
                num_scopes, text_len,
            )
            for A in dense_As
        ]
        ref = torch.stack(layers, dim=0)
        assert quality.shape == ref.shape
        assert torch.allclose(quality, ref, atol=1e-8)

    def test_explicit_scale_still_respected(self):
        """An explicit ``scale`` override (tests' convention) is still honored:
        ``scale=1.0`` scores match the dense oracle at scale=1.0 (unchanged
        behaviour for the existing test suite)."""
        num_scopes, text_len = 4, 4
        model, dit, loss_fn = _make_case()
        quality, _, _ = hcn.collect_scope_scores_for_model(
            model=model, model_type="flux", forward_fn=dit.forward,
            loss_fn=loss_fn, num_scopes=num_scopes, text_len=text_len,
            chunk=4096, scale=1.0,
        )
        m2, d2, l2 = _make_case()
        ref = _dense_reference(d2, l2, num_scopes, text_len)  # scale=1.0 oracle
        assert quality.shape == ref.shape
        assert torch.allclose(quality, ref, atol=1e-8)

    def test_scale_none_differs_from_scale_one(self):
        """``scale=None`` (dim_head**-0.5) produces DIFFERENT scores than the
        old hardcoded ``scale=1.0`` — proving the convention actually changed
        the calibration math (not a silent no-op)."""
        num_scopes, text_len = 4, 4
        m1, d1, l1 = _make_case()
        q_none, _, _ = hcn.collect_scope_scores_for_model(
            model=m1, model_type="flux", forward_fn=d1.forward, loss_fn=l1,
            num_scopes=num_scopes, text_len=text_len, chunk=4096, scale=None,
        )
        m2, d2, l2 = _make_case()
        q_one, _, _ = hcn.collect_scope_scores_for_model(
            model=m2, model_type="flux", forward_fn=d2.forward, loss_fn=l2,
            num_scopes=num_scopes, text_len=text_len, chunk=4096, scale=1.0,
        )
        # The two attention distributions differ (sqrt(dim_head)x logit scale),
        # so the Taylor scores must differ too.
        assert not torch.allclose(q_none, q_one, atol=1e-6)
