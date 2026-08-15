"""Tests for the unified HRDiT attention wrapper (plan P3: T3.1-T3.4).

Covers:
  * T3.1 — behaviour-preserving refactor + per-forward layer counter,
  * T3.2 — HAP dispatch inside the wrapper (decision matrix §2.1),
  * T3.3 — ref-counted shared install policy (SPA + HAP share ONE wrapper),
  * T3.4 — ``text_len`` derivation + seq_len safety.

Markers: @pytest.mark.unit / @pytest.mark.mock_integration
"""

import types

import pytest
import torch

from src import hap
from src.spa import (
    _hrdit_install_hook,
    _hrdit_uninstall_hook,
    _spa_derive_text_len,
    _spa_install_hook,
    restore_spa_attention_hook,
)
from src.spa_context import (
    SPAContext,
    get_hrdit_layer_idx,
    set_hap_context,
    set_hrdit_layer_idx,
    set_spa_context,
    set_spa_step_gate,
)


@pytest.fixture
def mock_attn():
    """The conftest-provided (pristine SDPA) mock ``comfy.ldm.modules.attention`` module."""
    import comfy.ldm.modules.attention as attn_mod

    return attn_mod


class _MockModel:
    def __init__(self):
        self._unet_wrapper = None
        self._spa_installed = None
        self._spa_orig_optimized_attention = None
        self._hrdit_consumers = None
        self._hap_ctx = None

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def _plan(num_layers=4, num_heads=2, alpha=64.0, beta=0.0):
    """A uniform scope plan.  alpha=64 -> band = 2*int(64/64)-1 = 1 -> half 0."""
    return hap.ScopePlan(
        alphas=[[alpha] * num_heads for _ in range(num_layers)],
        betas=[[beta] * num_heads for _ in range(num_layers)],
    )


def _hap_ctx(num_layers=4, text_len=0, backend="dense"):
    return hap.HapContext(active=True, plan=_plan(num_layers), text_len=text_len, backend=backend)


def _rand_qkv(B=1, H=2, S=128, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g)
    k = torch.randn(B, H, S, D, generator=g)
    v = torch.randn(B, H, S, D, generator=g)
    return q, k, v


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset singletons + contextvars around every test."""
    hap.HapRuntime.reset()
    set_hrdit_layer_idx(0)
    yield
    set_hap_context(None)
    set_spa_context(None)
    set_spa_step_gate(True)
    set_hrdit_layer_idx(0)
    hap.HapRuntime.reset()


# ---------------------------------------------------------------------------
# T3.1 — layer counter
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestLayerCounter:
    def test_layer_counter_sequence(self, mock_attn):
        """4 attention calls -> per-forward indices 0,1,2,3 (observed via HAP)."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=4))

        seen = []
        real_attn = hap.HapRuntime.attn

        def spy(self, q, k, v, layer, **kw):
            seen.append(layer)
            return real_attn(self, q, k, v, layer, **kw)

        hap.HapRuntime.attn = spy
        try:
            q, k, v = _rand_qkv()
            for _ in range(4):
                mock_attn.optimized_attention(q, k, v, 2)
        finally:
            hap.HapRuntime.attn = real_attn

        assert seen == [0, 1, 2, 3]
        assert get_hrdit_layer_idx() == 4

    def test_layer_counter_resets_between_forwards(self, mock_attn):
        """Simulating two forwards (manual reset) restarts the index at 0."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=4))
        q, k, v = _rand_qkv()
        mock_attn.optimized_attention(q, k, v, 2)
        mock_attn.optimized_attention(q, k, v, 2)
        assert get_hrdit_layer_idx() == 2
        set_hrdit_layer_idx(0)  # what the unet wrapper does per forward
        mock_attn.optimized_attention(q, k, v, 2)
        assert get_hrdit_layer_idx() == 1

    def test_layer_counter_increments_when_gate_closed(self, mock_attn):
        """SPA step gate closed -> early-return path STILL advances the counter."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        set_spa_step_gate(False)  # gate closed -> plain attention
        q, k, v = _rand_qkv()
        mock_attn.optimized_attention(q, k, v, 2)
        mock_attn.optimized_attention(q, k, v, 2)
        # Alignment guard: the counter advanced even though SPA was gated off.
        assert get_hrdit_layer_idx() == 2


# ---------------------------------------------------------------------------
# T3.2 — HAP dispatch (decision matrix)
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestHapDispatch:
    def test_hap_only_single_kernel_pass_per_layer(self, mock_attn):
        """HAP on, SPA off -> exactly one kernel pass per attention call."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=3))

        calls = []
        real_attn = hap.HapRuntime.attn

        def spy(self, q, k, v, layer, **kw):
            calls.append(layer)
            return real_attn(self, q, k, v, layer, **kw)

        hap.HapRuntime.attn = spy
        try:
            q, k, v = _rand_qkv()
            for _ in range(3):
                mock_attn.optimized_attention(q, k, v, 2)
        finally:
            hap.HapRuntime.attn = real_attn
        assert calls == [0, 1, 2]

    def test_hap_output_equals_dense_mask_reference(self, mock_attn):
        """Wrapper HAP output == manual dense-mask attention (alpha=64 -> half 0)."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=1, text_len=0))
        q, k, v = _rand_qkv(S=128, seed=3)
        out = mock_attn.optimized_attention(q, k, v, 2)
        mask = hap.build_band_mask(128, 0, [0, 0], 0)
        ref = hap.hap_attn_dense(q, k, v, mask)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_kernel_none_falls_back_to_orig(self, mock_attn):
        """HAP runtime returning None -> wrapper falls back to orig attention."""
        import torch.nn.functional as F

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        # Plan with too few layers -> layer 5 exceeds it -> runtime returns None.
        set_hap_context(_hap_ctx(num_layers=1))
        q, k, v = _rand_qkv(seed=4)
        # Burn 5 calls so the next is layer index 5 (>= num_layers=1).
        for _ in range(5):
            mock_attn.optimized_attention(q, k, v, 2)
        out = mock_attn.optimized_attention(q, k, v, 2)
        ref = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_no_hap_no_spa_is_plain(self, mock_attn):
        """Neither active -> bit-identical to the original attention."""
        import torch.nn.functional as F

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        q, k, v = _rand_qkv(seed=5)
        out = mock_attn.optimized_attention(q, k, v, 2)
        ref = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_spa_plus_hap_variants_through_kernel(self, mock_attn):
        """SPA active (N variants) + HAP -> EVERY variant pass runs through the kernel."""
        from src.spa_attn import apply_rope_matrix

        try:
            from tests._spa_math_helpers import angles_to_blocks
        except ImportError:
            from _spa_math_helpers import angles_to_blocks

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        _hrdit_install_hook(m, "flux", consumer="hap")  # shared wrapper, 2nd consumer

        L, H, D = 128, 2, 16
        N = 5  # 2s-1 variant passes (s=3) as produced by build_bundle_id_variants
        P = D // 2
        g = torch.Generator().manual_seed(0)
        base = torch.randn(L, P, generator=g) * 0.3
        variants = [torch.randn(L, P, generator=g) * 0.3 for _ in range(N)]
        base_R = angles_to_blocks(base)[None, None]
        variant_Rs = [angles_to_blocks(a)[None, None] for a in variants]
        spa_ctx = SPAContext(active=True, bundle_size=3, base_pe=base_R,
                             variant_pes=variant_Rs, pre_roped=True, fmt="flux",
                             model_key=0, text_len=0)
        set_spa_context(spa_ctx)
        set_hap_context(_hap_ctx(num_layers=1, text_len=0))

        kernel_calls = []
        real_attn = hap.HapRuntime.attn

        def spy(self, q, k, v, layer, **kw):
            kernel_calls.append(layer)
            return real_attn(self, q, k, v, layer, **kw)

        hap.HapRuntime.attn = spy
        try:
            q, k, v = _rand_qkv(H=H, S=L, D=D, seed=6)
            q_base = apply_rope_matrix(q, base_R, "flux")
            k_base = apply_rope_matrix(k, base_R, "flux")
            mock_attn.optimized_attention(q_base, k_base, v, H)
        finally:
            hap.HapRuntime.attn = real_attn

        # N = 5 variant passes, each through the HAP kernel at layer 0.
        assert kernel_calls == [0] * N


# ---------------------------------------------------------------------------
# T3.3 — ref-counted shared install policy
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestInstallPolicy:
    def test_hap_only_installs_wrapper(self, mock_attn):
        """Inverts T0.3: HAP-standalone MUST install the wrapper."""
        orig = mock_attn.optimized_attention
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        assert getattr(m, "_spa_installed", None)
        assert mock_attn.optimized_attention is not orig
        assert m._hrdit_consumers == {"hap"}

    def test_spa_then_hap_single_wrapper(self, mock_attn):
        """SPA then HAP -> ONE wrapper, both consumers recorded."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        wrapper_after_spa = mock_attn.optimized_attention
        _hrdit_install_hook(m, "flux", consumer="hap")
        assert mock_attn.optimized_attention is wrapper_after_spa  # not re-wrapped
        assert m._hrdit_consumers == {"spa", "hap"}

    def test_hap_then_spa_single_wrapper(self, mock_attn):
        """Order-independent: HAP then SPA also shares one wrapper."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        wrapper = mock_attn.optimized_attention
        _hrdit_install_hook(m, "flux", consumer="spa")
        assert mock_attn.optimized_attention is wrapper
        assert m._hrdit_consumers == {"spa", "hap"}

    def test_restore_requires_both_consumers(self, mock_attn):
        """Unpatch SPA while HAP active -> wrapper stays; unpatch HAP -> restored."""
        orig = mock_attn.optimized_attention
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        _hrdit_install_hook(m, "flux", consumer="hap")
        # Remove SPA: HAP still needs the wrapper.
        restore_spa_attention_hook(m, mock_attn)
        assert mock_attn.optimized_attention is not orig
        assert m._hrdit_consumers == {"hap"}
        # Remove HAP: last consumer -> full restore.
        _hrdit_uninstall_hook(m, "hap")
        assert mock_attn.optimized_attention is orig

    def test_spa_only_restore_still_works(self, mock_attn):
        """Legacy single-consumer (SPA-only) restore is unchanged."""
        orig = mock_attn.optimized_attention
        m = _MockModel()
        _spa_install_hook(m, "flux")
        assert mock_attn.optimized_attention is not orig
        restore_spa_attention_hook(m, mock_attn)
        assert mock_attn.optimized_attention is orig


# ---------------------------------------------------------------------------
# T3.4 — text_len derivation + seq_len safety
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTextLenDerivation:
    def _flux_ids(self, text_len, hw):
        """FLUX-style ids: ``text_len`` leading text tokens then an hw x hw grid."""
        L = text_len + hw * hw
        ids = torch.zeros(1, L, 3)
        ids[..., 0] = torch.arange(L)
        # text tokens: row == col == 0 (already zero)
        grid = torch.arange(hw * hw)
        ids[0, text_len:, 1] = (grid // hw).float() + 1  # +1 so (0,0) pixel != text
        ids[0, text_len:, 2] = (grid % hw).float() + 1
        return ids

    def test_text_len_from_spa_ids(self):
        ids = self._flux_ids(text_len=128, hw=8)
        assert _spa_derive_text_len(ids) == 128

    def test_text_len_zero_when_no_leading_text(self):
        ids = self._flux_ids(text_len=0, hw=8)
        assert _spa_derive_text_len(ids) == 0

    def test_text_len_none_for_bad_shape(self):
        assert _spa_derive_text_len(torch.zeros(4)) is None
        assert _spa_derive_text_len(None) is None

    def test_text_len_default_512_without_spa(self):
        """No live SPA context -> HapContext.text_len (node default) is used."""
        hctx = hap.HapContext(active=True, plan=_plan(1), text_len=512)
        from src.spa import _hrdit_resolve_text_len

        assert _hrdit_resolve_text_len(hctx, seq_len=4096) == 512

    def test_text_len_prefers_spa_derived(self):
        """A live SPA context's derived text_len wins over the node default."""
        from src.spa import _hrdit_resolve_text_len

        hctx = hap.HapContext(active=True, plan=_plan(1), text_len=512)
        spa_ctx = SPAContext(active=True, text_len=77)
        set_spa_context(spa_ctx)
        assert _hrdit_resolve_text_len(hctx, seq_len=4096) == 77

    def test_text_len_clamped_to_seq(self):
        """text_len 512 > seq 256 -> clamped to 256 (degenerate-safe)."""
        from src.spa import _hrdit_resolve_text_len

        hctx = hap.HapContext(active=True, plan=_plan(1), text_len=512)
        assert _hrdit_resolve_text_len(hctx, seq_len=256) == 256

    def test_no_stale_mask_on_resolution_change(self, mock_attn):
        """A resolution change mid-session builds a fresh mask (no stale reuse)."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=1, text_len=0))
        runtime = hap.HapRuntime.get()
        q1, k1, v1 = _rand_qkv(S=128, seed=7)
        mock_attn.optimized_attention(q1, k1, v1, 2)
        assert runtime.prepare_count == 1
        q2, k2, v2 = _rand_qkv(S=192, seed=8)
        mock_attn.optimized_attention(q2, k2, v2, 2)
        assert runtime.prepare_count == 2
