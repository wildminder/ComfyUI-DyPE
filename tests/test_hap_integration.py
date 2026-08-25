"""End-to-end integration tests for the composed HRDiT pipeline (plan phase P9).

- T9.1 ``test_e2e_spa_plus_hap_2k`` — SPA + HAP composed over 8 simulated
  denoising steps with step gating (``spa_steps=3``): the leading steps run the
  full averaged-variant passes through the HAP kernel; the later steps run a
  single kernel pass each; the ORIGINAL dense attention is never called while
  HAP is live.
- T9.2 ``test_e2e_hap_standalone`` — SPA off (``bundle_size=1``), HAP on:
  every step is a single kernel pass whose output equals the dense-mask
  reference, and differs from the no-HAP baseline.
- T5.1 ``TestE2EAnimaRegression`` — Anima-style cosmos-convention calls
  (``skip_reshape=True`` + ``transformer_options`` kwargs) through the REAL
  shared unet wrapper: matching plan engages the kernel (output == dense-mask
  reference, orig never called); mismatched plan falls back gracefully with a
  one-time warning (the exact scenario that crashed pre-fix).
- T5.2 ``test_e2e_cross_attention_mix`` — a forward mixing square self-attn
  and non-square cross-attn calls: self-attn HAP-masked, cross-attn plain,
  layer counter advances for BOTH (alignment preserved).

These drive the REAL shared unet wrapper (``_hrdit_install_hook``) with a
decreasing-sigma schedule so the step-count gate behaves exactly as in a real
ComfyUI forward.  All math is on the mock SDPA backend (no CUDA required).

Markers: @pytest.mark.mock_integration
Accept (user-run):
    pytest tests/test_hap_integration.py -k e2e_spa
    pytest tests/test_hap_integration.py -k e2e_hap_standalone
    pytest tests/test_hap_integration.py -k anima
    pytest tests/test_hap_integration.py -k cross_attention
"""


import pytest
import torch
import torch.nn.functional as F

from src import hap
from src.spa import _hrdit_install_hook
from src.spa_context import (
    SPAContext,
    set_hap_context,
    set_hrdit_layer_idx,
    set_hrdit_proportional,
    set_spa_context,
    set_spa_layer_filter,
    set_spa_step_gate,
)

NUM_LAYERS = 4
HEADS = 2
DIM = 16
SEQ_LEN = 64


def _rand_qkv(seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, HEADS, SEQ_LEN, DIM, generator=g)
    k = torch.randn(1, HEADS, SEQ_LEN, DIM, generator=g)
    v = torch.randn(1, HEADS, SEQ_LEN, DIM, generator=g)
    return q, k, v


def _identity_spa_ctx(num_variants=5):
    """SPA context with IDENTITY variant rotations (s = (n+1)//2 = 3 -> 5 passes).

    Identity rotations make every variant pass mathematically plain attention,
    so the test isolates the PASS COUNT / kernel-routing behaviour without
    depending on RoPE numerics.
    """
    L, P = SEQ_LEN, DIM // 2
    eye = torch.eye(2).expand(1, 1, L, P, 2, 2).clone()
    return SPAContext(
        active=True,
        bundle_size=(num_variants + 1) // 2,
        base_pe=eye.clone(),
        variant_pes=[eye.clone() for _ in range(num_variants)],
        variant_deltas=[eye.clone() for _ in range(num_variants)],
        pre_roped=True,
        fmt="flux",
        text_len=0,
    )


def _plan(num_layers=NUM_LAYERS, num_heads=HEADS, alpha=64.0, beta=0.0):
    return hap.ScopePlan(
        alphas=[[alpha] * num_heads for _ in range(num_layers)],
        betas=[[beta] * num_heads for _ in range(num_layers)],
    )


def _hap_ctx(num_layers=NUM_LAYERS, text_len=0, backend="dense"):
    return hap.HapContext(active=True, plan=_plan(num_layers),
                          text_len=text_len, backend=backend)


class _MockModel:
    def __init__(self):
        self._unet_wrapper = None
        self._spa_installed = None
        self._spa_orig_optimized_attention = None
        self._hrdit_consumers = None
        self._hap_ctx = None
        # Step-gating state (normally set by apply_spa_to_model).
        self._spa_steps = 0
        self._spa_start_sigma = 1.0
        self._spa_step_counter = 0
        self._spa_last_sigma = None
        self._spa_layer_filter = None
        self._hrdit_proportional_attention = False

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


@pytest.fixture
def mock_attn():
    import comfy.ldm.modules.attention as attn_mod

    return attn_mod


@pytest.fixture(autouse=True)
def _clean_state():
    hap.HapRuntime.reset()
    set_hrdit_layer_idx(0)
    set_spa_layer_filter(None)
    set_hrdit_proportional(False)
    yield
    set_hap_context(None)
    set_spa_context(None)
    set_spa_step_gate(True)
    set_spa_layer_filter(None)
    set_hrdit_layer_idx(0)
    set_hrdit_proportional(False)
    hap.HapRuntime.reset()


def _install_spy_orig(mock_attn):
    """Wrap the pristine SDPA so we can assert it is NEVER called while HAP is
    live (the wrapper's ``orig`` fallback must not fire)."""
    pristine = mock_attn.optimized_attention
    orig_calls = []

    def spy_orig(q, k, v, heads, *a, **kw):
        orig_calls.append(1)
        return pristine(q, k, v, heads, *a, **kw)

    mock_attn.optimized_attention = spy_orig
    return orig_calls


# ---------------------------------------------------------------------------
# T9.1 — SPA + HAP composed over a gated 8-step schedule
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestE2ESpaPlusHap:
    def test_e2e_spa_plus_hap_2k(self, mock_attn):
        orig_calls = _install_spy_orig(mock_attn)

        m = _MockModel()
        m._spa_steps = 3          # SPA on the 3 LEADING steps only
        m._spa_start_sigma = 1.0   # no sigma-threshold gating
        m._hap_ctx = _hap_ctx()    # HAP live on EVERY step
        _hrdit_install_hook(m, "flux", consumer="spa")
        _hrdit_install_hook(m, "flux", consumer="hap")  # shared wrapper
        assert m._hrdit_consumers == {"spa", "hap"}

        q, k, v = _rand_qkv()

        kernel_calls = []
        real_attn = hap.HapRuntime.attn

        def spy(self, qq, kk, vv, layer, **kw):
            kernel_calls.append(layer)
            return real_attn(self, qq, kk, vv, layer, **kw)

        def model_fn(x, t, **c):
            # Simulate the embedder forward: register the SPA variants, then run
            # one attention call per layer through the patched symbol.
            set_spa_context(_identity_spa_ctx(num_variants=5))
            for _ in range(NUM_LAYERS):
                mock_attn.optimized_attention(q, k, v, HEADS)
            return x

        # Decreasing sigma -> no jump-up -> the step counter runs 0..7.
        sigmas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
        per_step_kernel_counts = []

        hap.HapRuntime.attn = spy
        try:
            for sigma in sigmas:
                kernel_calls.clear()
                m._unet_wrapper(model_fn, {
                    "input": torch.zeros(1),
                    "timestep": torch.tensor(sigma),
                    "c": {},
                })
                per_step_kernel_counts.append(len(kernel_calls))
        finally:
            hap.HapRuntime.attn = real_attn

        # Leading 3 steps: SPA active -> 5 variant passes x 4 layers = 20 each.
        # Later 5 steps: SPA gated off -> 1 pass x 4 layers = 4 each.
        assert per_step_kernel_counts == [20, 20, 20, 4, 4, 4, 4, 4]
        # The original dense attention never fires while HAP is live.
        assert orig_calls == []

    def test_e2e_spa_plus_hap_kernel_layer_sequence(self, mock_attn):
        """Within one SPA-active step the kernel sees each layer index repeated
        per variant pass (5x), in block order."""
        _install_spy_orig(mock_attn)
        m = _MockModel()
        m._spa_steps = 1
        m._hap_ctx = _hap_ctx()
        _hrdit_install_hook(m, "flux", consumer="spa")
        _hrdit_install_hook(m, "flux", consumer="hap")

        q, k, v = _rand_qkv()
        kernel_calls = []
        real_attn = hap.HapRuntime.attn

        def spy(self, qq, kk, vv, layer, **kw):
            kernel_calls.append(layer)
            return real_attn(self, qq, kk, vv, layer, **kw)

        def model_fn(x, t, **c):
            set_spa_context(_identity_spa_ctx(num_variants=5))
            for _ in range(NUM_LAYERS):
                mock_attn.optimized_attention(q, k, v, HEADS)
            return x

        hap.HapRuntime.attn = spy
        try:
            m._unet_wrapper(model_fn, {
                "input": torch.zeros(1),
                "timestep": torch.tensor(1.0),
                "c": {},
            })
        finally:
            hap.HapRuntime.attn = real_attn

        assert kernel_calls == [0] * 5 + [1] * 5 + [2] * 5 + [3] * 5


# ---------------------------------------------------------------------------
# T9.2 — HAP standalone (SPA off)
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestE2EHapStandalone:
    def test_e2e_hap_standalone(self, mock_attn):
        """SPA off (no SPA context), HAP on: single kernel pass per layer whose
        output equals the dense-mask reference and differs from the baseline."""
        orig_calls = _install_spy_orig(mock_attn)
        m = _MockModel()
        m._hap_ctx = _hap_ctx(text_len=0)
        _hrdit_install_hook(m, "flux", consumer="hap")
        assert m._hrdit_consumers == {"hap"}

        q, k, v = _rand_qkv(seed=7)

        kernel_calls = []
        captured = {}
        real_attn = hap.HapRuntime.attn

        def spy(self, qq, kk, vv, layer, **kw):
            kernel_calls.append(layer)
            return real_attn(self, qq, kk, vv, layer, **kw)

        def model_fn(x, t, **c):
            # No SPA context registered -> spa_active is False on every layer.
            # skip_output_reshape=True -> wrapper returns head format (B, H, T, D)
            # to match the hap_attn_dense reference below (math comparison).
            for layer in range(NUM_LAYERS):
                out = mock_attn.optimized_attention(q, k, v, HEADS, skip_output_reshape=True)
                if layer == 0:
                    captured["out"] = out  # capture while HAP ctx is live
            return x

        hap.HapRuntime.attn = spy
        try:
            # Two steps; HAP is not step-gated, so both run the kernel.
            for sigma in (1.0, 0.5):
                kernel_calls.clear()
                m._unet_wrapper(model_fn, {
                    "input": torch.zeros(1),
                    "timestep": torch.tensor(sigma),
                    "c": {},
                })
                # Single pass per layer, in block order.
                assert kernel_calls == [0, 1, 2, 3]
        finally:
            hap.HapRuntime.attn = real_attn

        # Output (captured inside the forward, while HAP was live) equals the
        # dense-mask reference (alpha=64 -> half=0 band).
        out = captured["out"]
        mask = hap.build_band_mask(SEQ_LEN, 0, [0] * HEADS, 0)
        ref = hap.hap_attn_dense(q, k, v, mask)
        assert torch.allclose(out, ref, atol=1e-6)
        # ...and differs from the unmasked baseline (pruning is observable).
        baseline = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        assert not torch.allclose(out, baseline, atol=1e-6)
        # The wrapper's orig fallback never fired while HAP was live.
        assert orig_calls == []


# ---------------------------------------------------------------------------
# T5.1 / T5.2 — Anima regression e2e (plan 2026-08-16 P5)
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestE2EAnimaRegression:
    """Anima-style cosmos-convention calls through the REAL shared unet wrapper.

    Anima (``comfy.ldm.cosmos.predict2``) calls ``optimized_attention`` UNMASKED
    with ``skip_reshape=True`` and ``transformer_options`` as keyword args.  The
    pre-fix wrapper fed ``skip_reshape`` into the real ``mask`` slot ->
    ``mask.ndim`` -> ``AttributeError``.  These tests prove the fixed wrapper
    serves the cosmos convention end-to-end:

    - T5.1 matching plan (heads == q heads): the HAP kernel engages, the output
      equals the dense-mask reference, and the original attention never fires.
    - T5.1 mismatched plan (FLUX 24-head plan on a 2-head model): graceful
      plain-attention fallback, no exception, one-time warning naming both counts.
    - T5.2 mixed self/cross-attention: self-attn is HAP-masked, cross-attn is
      plain, and the layer counter advances for BOTH (alignment is sacred).
    """

    def test_e2e_anima_matching_plan(self, mock_attn):
        """T5.1 matching: cosmos convention + SPA active + HAP active (heads match)
        -> kernel engages, output == dense-mask reference, orig never called."""
        orig_calls = _install_spy_orig(mock_attn)
        m = _MockModel()
        m._spa_steps = 1
        m._spa_start_sigma = 1.0
        m._hap_ctx = _hap_ctx()  # HEADS=2 == q heads -> kernel engages
        _hrdit_install_hook(m, "flux", consumer="spa")
        _hrdit_install_hook(m, "flux", consumer="hap")
        assert m._hrdit_consumers == {"spa", "hap"}

        q, k, v = _rand_qkv(seed=11)
        captured = {}

        def model_fn(x, t, **c):
            set_spa_context(_identity_spa_ctx(num_variants=5))
            # Cosmos convention: skip_reshape + transformer_options kwargs, NO mask.
            # skip_output_reshape=True -> wrapper returns head format (B, H, T, D)
            # to match the hap_attn_dense reference below (math comparison).
            out = mock_attn.optimized_attention(
                q, k, v, HEADS, skip_reshape=True, skip_output_reshape=True,
                transformer_options={}
            )
            captured["out"] = out
            return x

        m._unet_wrapper(model_fn, {
            "input": torch.zeros(1),
            "timestep": torch.tensor(1.0),
            "c": {},
        })

        # Identity SPA rotations -> every variant pass is the same HAP-masked
        # attention, so the averaged output equals the dense-mask reference.
        out = captured["out"]
        mask = hap.build_band_mask(SEQ_LEN, 0, [0] * HEADS, 0)
        ref = hap.hap_attn_dense(q, k, v, mask)
        assert torch.allclose(out, ref, atol=1e-6)
        # The original dense attention never fires while HAP is live.
        assert orig_calls == []

    def test_e2e_anima_mismatched_plan_fallback(self, mock_attn, caplog):
        """T5.1 mismatched: FLUX plan (24 heads) on a 2-head model -> graceful
        plain-attention fallback, no exception, one-time warning naming both counts."""
        import logging

        orig_calls = _install_spy_orig(mock_attn)
        m = _MockModel()
        m._spa_steps = 1
        # FLUX-shaped plan (24 heads) vs q with HEADS=2 -> head-count mismatch.
        mismatched_plan = hap.ScopePlan(
            alphas=[[64.0] * 24 for _ in range(NUM_LAYERS)],
            betas=[[0.0] * 24 for _ in range(NUM_LAYERS)],
        )
        m._hap_ctx = hap.HapContext(
            active=True, plan=mismatched_plan, text_len=0, backend="dense"
        )
        _hrdit_install_hook(m, "flux", consumer="hap")

        q, k, v = _rand_qkv(seed=12)
        captured = {}

        def model_fn(x, t, **c):
            out = mock_attn.optimized_attention(
                q, k, v, HEADS, skip_reshape=True, transformer_options={}
            )
            captured["out"] = out
            return x

        with caplog.at_level(logging.WARNING, logger="src.hap"):
            m._unet_wrapper(model_fn, {
                "input": torch.zeros(1),
                "timestep": torch.tensor(1.0),
                "c": {},
            })

        # No exception (the pre-fix crash) and a graceful plain-attention fallback.
        out = captured["out"]
        baseline = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        assert torch.allclose(out, baseline, atol=1e-6)
        # The wrapper fell back to the original attention exactly once.
        assert len(orig_calls) == 1
        # One-time warning naming both the plan's and the model's head counts.
        mismatch = [r for r in caplog.records if "heads" in r.message]
        assert len(mismatch) == 1
        assert "24" in mismatch[0].message
        assert "2" in mismatch[0].message

    def test_e2e_cross_attention_mix(self, mock_attn):
        """T5.2: a forward mixing square self-attn and non-square cross-attn calls.
        Self-attn is HAP-masked, cross-attn is plain, and the RAW layer counter
        advances for BOTH (alignment is sacred).

        UPDATED (2026-08-19, plan-layer ordinal fix): the HAP PLAN-LAYER ORDINAL
        advances ONLY for plan-covered calls (square + unmasked + head-match),
        mirroring calibration — calibration SKIPS non-square cross-attention, so
        the runtime must not consume a plan slot for it either.  The self-attn
        calls therefore receive plan layers [0, 1] (not [0, 2]), and the
        cross-attn calls keep their raw indices [1, 3] but decline to plain
        attention via the non-square guard.  The pre-fix expectation
        ``[(0,..),(1,..),(2,..),(3,..)]`` encoded the buggy raw-counter indexing
        that shifted Krea2's main blocks by its 4 aux calls.
        """
        orig_calls = _install_spy_orig(mock_attn)
        m = _MockModel()
        m._hap_ctx = _hap_ctx()  # HEADS=2 == q heads
        _hrdit_install_hook(m, "flux", consumer="hap")

        q_self, k_self, v_self = _rand_qkv(seed=21)  # square (SEQ_LEN x SEQ_LEN)
        g = torch.Generator().manual_seed(22)
        q_cross = torch.randn(1, HEADS, SEQ_LEN, DIM, generator=g)
        k_cross = torch.randn(1, HEADS, SEQ_LEN // 2, DIM, generator=g)  # shorter kv
        v_cross = torch.randn(1, HEADS, SEQ_LEN // 2, DIM, generator=g)

        kernel_calls = []
        captured = {}
        real_attn = hap.HapRuntime.attn

        def spy(self, qq, kk, vv, layer, **kw):
            kernel_calls.append((layer, int(qq.shape[-2]), int(kk.shape[-2])))
            return real_attn(self, qq, kk, vv, layer, **kw)

        def model_fn(x, t, **c):
            # Block 0: self-attn (square) then cross-attn (non-square).
            # Self-attn: skip_output_reshape=True -> head format (B, H, T, D) to
            # match the hap_attn_dense reference below (math comparison).
            # Cross-attn declines HAP (non-square) -> orig fallback (pristine SDPA,
            # head format) — unaffected by the output-reshape fix.
            captured["self0"] = mock_attn.optimized_attention(
                q_self, k_self, v_self, HEADS, skip_reshape=True, skip_output_reshape=True
            )
            captured["cross0"] = mock_attn.optimized_attention(
                q_cross, k_cross, v_cross, HEADS, skip_reshape=True
            )
            # Block 1: self-attn then cross-attn.
            mock_attn.optimized_attention(q_self, k_self, v_self, HEADS, skip_reshape=True, skip_output_reshape=True)
            mock_attn.optimized_attention(q_cross, k_cross, v_cross, HEADS, skip_reshape=True)
            return x

        hap.HapRuntime.attn = spy
        try:
            m._unet_wrapper(model_fn, {
                "input": torch.zeros(1),
                "timestep": torch.tensor(1.0),
                "c": {},
            })
        finally:
            hap.HapRuntime.attn = real_attn

        # The RAW layer counter advances for BOTH self-attn and cross-attn
        # (alignment is sacred), but the HAP PLAN-LAYER ORDINAL advances only for
        # the covered (square) self-attn calls.  So the self-attn calls receive
        # plan layers [0, 1] and the cross-attn calls keep their raw indices
        # [1, 3] (declined to plain attention via the non-square guard).
        assert kernel_calls == [
            (0, SEQ_LEN, SEQ_LEN),          # block 0 self-attn (square) -> plan 0
            (1, SEQ_LEN, SEQ_LEN // 2),     # block 0 cross-attn (non-square) -> raw 1, declines
            (1, SEQ_LEN, SEQ_LEN),          # block 1 self-attn (square) -> plan 1
            (3, SEQ_LEN, SEQ_LEN // 2),     # block 1 cross-attn (non-square) -> raw 3, declines
        ]
        # Cross-attn (non-square) declined to plain attention: 2 orig fallbacks.
        assert len(orig_calls) == 2
        # Self-attn output is HAP-masked (equals the dense-mask reference).
        mask = hap.build_band_mask(SEQ_LEN, 0, [0] * HEADS, 0)
        ref_self = hap.hap_attn_dense(q_self, k_self, v_self, mask)
        assert torch.allclose(captured["self0"], ref_self, atol=1e-6)
        # Cross-attn output is plain attention (the pristine SDPA).
        ref_cross = F.scaled_dot_product_attention(q_cross, k_cross, v_cross, scale=1.0)
        assert torch.allclose(captured["cross0"], ref_cross, atol=1e-6)

    def test_e2e_anima_spa_cross_attention_no_crash(self, mock_attn):
        """REGRESSION (2026-08-16): the exact Anima production crash path.

        SPA active (identity variants) + a NON-SQUARE cosmos-convention call
        (image queries vs text keys) through the REAL shared unet wrapper.
        Pre-fix this raised ``RuntimeError: einsum(): subscript l has size 512
        for operand 1 ... previously seen size 6300`` because the averaged
        passes applied the image-sized RoPE rotations to the text ``k``.  The
        non-square guard must decline SPA and run plain attention.
        """
        m = _MockModel()
        m._spa_steps = 1
        m._spa_start_sigma = 1.0
        _hrdit_install_hook(m, "flux", consumer="spa")

        # Identity SPA variants over the IMAGE query length (SEQ_LEN) — exactly
        # what PosEmbedSPAAnima registers for the T*H*W grid.
        q_img = torch.randn(1, HEADS, SEQ_LEN, DIM,
                            generator=torch.Generator().manual_seed(31))
        k_text = torch.randn(1, HEADS, SEQ_LEN // 2, DIM,
                             generator=torch.Generator().manual_seed(32))
        v_text = torch.randn(1, HEADS, SEQ_LEN // 2, DIM,
                             generator=torch.Generator().manual_seed(33))
        captured = {}

        def model_fn(x, t, **c):
            set_spa_context(_identity_spa_ctx(num_variants=5))
            # Cosmos convention: skip_reshape kw, NO mask, q_len != k_len.
            captured["out"] = mock_attn.optimized_attention(
                q_img, k_text, v_text, HEADS,
                skip_reshape=True, transformer_options={},
            )
            return x

        # Pre-fix: RuntimeError in einsum.  Post-fix: must complete.
        m._unet_wrapper(model_fn, {
            "input": torch.zeros(1),
            "timestep": torch.tensor(1.0),
            "c": {},
        })

        # The declined cross-attn call ran PLAIN attention (pristine SDPA).
        ref = F.scaled_dot_product_attention(q_img, k_text, v_text, scale=1.0)
        assert torch.allclose(captured["out"], ref, atol=1e-6)
