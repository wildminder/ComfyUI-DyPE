"""End-to-end integration tests for the composed HRDiT pipeline (plan phase P9).

- T9.1 ``test_e2e_spa_plus_hap_2k`` — SPA + HAP composed over 8 simulated
  denoising steps with step gating (``spa_steps=3``): the leading steps run the
  full averaged-variant passes through the HAP kernel; the later steps run a
  single kernel pass each; the ORIGINAL dense attention is never called while
  HAP is live.
- T9.2 ``test_e2e_hap_standalone`` — SPA off (``bundle_size=1``), HAP on:
  every step is a single kernel pass whose output equals the dense-mask
  reference, and differs from the no-HAP baseline.

These drive the REAL shared unet wrapper (``_hrdit_install_hook``) with a
decreasing-sigma schedule so the step-count gate behaves exactly as in a real
ComfyUI forward.  All math is on the mock SDPA backend (no CUDA required).

Markers: @pytest.mark.mock_integration
Accept (user-run):
    pytest tests/test_hap_integration.py -k e2e_spa
    pytest tests/test_hap_integration.py -k e2e_hap_standalone
"""

import types

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
            for layer in range(NUM_LAYERS):
                out = mock_attn.optimized_attention(q, k, v, HEADS)
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
