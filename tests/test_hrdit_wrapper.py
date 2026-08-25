"""Tests for the unified HRDiT attention wrapper (plan P3: T3.1-T3.4).

Covers:
  * T3.1 — behaviour-preserving refactor + per-forward layer counter,
  * T3.2 — HAP dispatch inside the wrapper (decision matrix §2.1),
  * T3.3 — ref-counted shared install policy (SPA + HAP share ONE wrapper),
  * T3.4 — ``text_len`` derivation + seq_len safety.

Markers: @pytest.mark.unit / @pytest.mark.mock_integration
"""


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
        """Wrapper HAP output == manual dense-mask attention (alpha=64 -> half 0).

        Uses ``skip_output_reshape=True`` so the wrapper returns head format
        ``(B, H, T, D)`` — the layout ``hap_attn_dense`` produces — for a direct
        element-wise comparison.
        """
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=1, text_len=0))
        q, k, v = _rand_qkv(S=128, seed=3)
        out = mock_attn.optimized_attention(q, k, v, 2, skip_output_reshape=True)
        mask = hap.build_band_mask(128, 0, [0, 0], 0)
        ref = hap.hap_attn_dense(q, k, v, mask)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_hap_output_flattened_when_skip_output_reshape_false(self, mock_attn):
        """REGRESSION (2026-08-18, krea2 inference crash ``dim 3: 128 vs 6144``).

        Krea2 calls ``optimized_attention_masked(..., skip_reshape=True)`` WITHOUT
        ``skip_output_reshape`` (defaults False) and then does
        ``out * F.sigmoid(gate)`` where ``gate`` is ``(B, T, H*D)`` — so it NEEDS
        the flattened ``(B, T, H*D)`` layout.  The pre-fix wrapper returned the
        head-format ``(B, H, T, D)`` output as-is, crashing the elementwise
        multiply.  With ``skip_output_reshape=False`` the wrapper must flatten the
        HAP output to ``(B, T, H*D)``, matching the flattened dense-mask reference.
        """
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=1, text_len=0))
        B, H, S, D = 1, 2, 128, 16
        q, k, v = _rand_qkv(B=B, H=H, S=S, D=D, seed=3)
        # Default skip_output_reshape=False -> caller expects flattened (B, T, H*D).
        out = mock_attn.optimized_attention(q, k, v, H)
        assert out.shape == (B, S, H * D), (
            f"skip_output_reshape=False must flatten to (B, T, H*D); got {tuple(out.shape)}"
        )
        mask = hap.build_band_mask(S, 0, [0, 0], 0)
        ref_head = hap.hap_attn_dense(q, k, v, mask)          # (B, H, S, D)
        ref_flat = ref_head.permute(0, 2, 1, 3).reshape(B, S, H * D)
        assert torch.allclose(out, ref_flat, atol=1e-6)

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
# HAP plan-layer ordinal (2026-08-19 runtime layer-index mismatch fix)
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestHapPlanLayerOrdinal:
    """REGRESSION (2026-08-19): the runtime must index the scope plan by the
    DOMINANT-HEAD-ONLY ordinal (mirroring calibration's heterogeneous-head-count
    filter), NOT the raw all-call counter.

    Krea2 runs 4 auxiliary 20-head projector calls before its 28 main 48-head
    blocks.  The pre-fix wrapper fed the RAW counter into the plan, so the 4 aux
    calls consumed indices 0-3, shifting every main block by 4 (block 0 read plan
    layer 4 instead of 0) and pushing the last 4 main blocks past the 28-layer
    plan ("layer 28 exceeds the scope plan").  Calibration enumerates plan layers
    by the dominant-head-only ordinal, so the runtime must do the same.
    """

    def test_aux_head_mismatch_calls_do_not_shift_plan_index(self, mock_attn):
        """4 aux calls (3 heads) + 4 main calls (2 heads; plan has 2 heads).

        The main calls must receive plan layers [0,1,2,3] — NOT [4,5,6,7] — and
        the ordinal must advance only for the covered (main) calls.
        """
        from src.spa_context import get_hap_layer_idx

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=4))  # num_heads=2 (dominant)

        seen = []  # (layer, heads)
        real_attn = hap.HapRuntime.attn

        def spy(self, q, k, v, layer, **kw):
            seen.append((layer, q.shape[1]))
            return real_attn(self, q, k, v, layer, **kw)

        hap.HapRuntime.attn = spy
        try:
            # 4 auxiliary calls with a DIFFERENT head count (3 heads).
            qa, ka, va = _rand_qkv(H=3, seed=10)
            for _ in range(4):
                mock_attn.optimized_attention(qa, ka, va, 3)
            # 4 main calls matching the plan's head count (2 heads).
            qm, km, vm = _rand_qkv(H=2, seed=11)
            for _ in range(4):
                mock_attn.optimized_attention(qm, km, vm, 2)
        finally:
            hap.HapRuntime.attn = real_attn

        aux = [l for (l, h) in seen if h == 3]
        main = [l for (l, h) in seen if h == 2]
        # Aux calls are non-covered: they keep the raw index and decline via the
        # head-mismatch guard inside ``HapRuntime.attn``.
        assert aux == [0, 1, 2, 3]
        # MAIN calls: plan ordinal 0-3 — NOT shifted by the 4 aux calls.
        assert main == [0, 1, 2, 3]
        # The ordinal advanced only for the 4 covered (main) calls.
        assert get_hap_layer_idx() == 4
        # The raw counter advanced for all 8 calls (alignment is sacred).
        assert get_hrdit_layer_idx() == 8

    def test_main_calls_engage_hap_aux_calls_plain(self, mock_attn):
        """Main (dominant-head) calls engage HAP (masked output); aux calls
        decline to plain SDPA — proving correct routing, not just indexing."""
        import torch.nn.functional as F

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=4, text_len=0))  # 2-head plan

        # Aux call (3 heads): declines -> plain SDPA.
        qa, ka, va = _rand_qkv(H=3, S=128, seed=20)
        out_aux = mock_attn.optimized_attention(qa, ka, va, 3, skip_output_reshape=True)
        ref_aux = F.scaled_dot_product_attention(qa, ka, va, scale=1.0)
        assert torch.allclose(out_aux, ref_aux, atol=1e-6)

        # Main call (2 heads): engages HAP -> dense-mask reference.
        qm, km, vm = _rand_qkv(H=2, S=128, seed=21)
        out_main = mock_attn.optimized_attention(qm, km, vm, 2, skip_output_reshape=True)
        mask = hap.build_band_mask(128, 0, [0, 0], 0)
        ref_main = hap.hap_attn_dense(qm, km, vm, mask)
        assert torch.allclose(out_main, ref_main, atol=1e-6)

    def test_plan_ordinal_resets_between_forwards(self, mock_attn):
        """Simulating two forwards (manual reset, as the unet wrapper does)
        restarts the plan ordinal at 0."""
        from src.spa_context import get_hap_layer_idx, set_hap_layer_idx

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=4))
        q, k, v = _rand_qkv(H=2)
        mock_attn.optimized_attention(q, k, v, 2)
        mock_attn.optimized_attention(q, k, v, 2)
        assert get_hap_layer_idx() == 2
        set_hap_layer_idx(0)  # what the unet wrapper does per forward
        mock_attn.optimized_attention(q, k, v, 2)
        assert get_hap_layer_idx() == 1

    def test_non_square_and_masked_calls_do_not_consume_ordinal(self, mock_attn):
        """Cross-attention (non-square) and masked calls are non-covered: they
        must NOT advance the plan ordinal (calibration skips them too)."""
        from src.spa_context import get_hap_layer_idx

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=4))

        q, k, v = _rand_qkv(H=2, S=128)
        # Non-square (cross-attention): k has a different sequence length.
        k_short = k[:, :, :64, :]
        v_short = v[:, :, :64, :]
        mock_attn.optimized_attention(q, k_short, v_short, 2)
        assert get_hap_layer_idx() == 0  # not consumed

        # Masked call: an external mask is present.
        mask = torch.ones(2, 128, 128, dtype=torch.bool)
        mock_attn.optimized_attention(q, k, v, 2, mask=mask)
        assert get_hap_layer_idx() == 0  # still not consumed

        # A covered (square, unmasked, head-match) call DOES consume it.
        mock_attn.optimized_attention(q, k, v, 2)
        assert get_hap_layer_idx() == 1

    def test_exceeds_warning_fires_once(self, mock_attn, caplog):
        """A covered call overrunning the plan logs the 'exceeds' warning ONCE
        (latched), not once per call/step (the live Krea2 spam)."""
        import logging

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=1))  # only 1 layer

        q, k, v = _rand_qkv(H=2, seed=30)
        with caplog.at_level(logging.WARNING, logger="src.hap"):
            for _ in range(4):  # ordinals 0,1,2,3 -> 3 exceed the 1-layer plan
                mock_attn.optimized_attention(q, k, v, 2)

        exceeds = [r for r in caplog.records if "exceeds the scope plan" in r.getMessage()]
        assert len(exceeds) == 1


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
        """A resolution change mid-session builds a fresh mask (no stale reuse).

        W2.5 re-baseline (2026-08-25): the runtime singleton is reset FIRST so
        the count reflects only THIS test's two prepares (a polluted singleton
        shared across tests made the absolute counts order-dependent).
        W2.7 note: the HAP plan-layer ordinal is a per-forward contextvar that
        the UNET wrapper resets — calling ``optimized_attention`` directly
        twice leaves the ordinal at 1 after the first call, which exceeds the
        1-layer plan and declines call 2 before any mask work.  Reset BOTH
        counters before each call to emulate fresh forwards.
        """
        from src.spa_context import set_hap_layer_idx, set_hrdit_layer_idx

        hap.HapRuntime.reset()
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        set_hap_context(_hap_ctx(num_layers=1, text_len=0))
        runtime = hap.HapRuntime.get()
        q1, k1, v1 = _rand_qkv(S=128, seed=7)
        n_before = runtime.prepare_count
        set_hrdit_layer_idx(0)
        set_hap_layer_idx(0)
        mock_attn.optimized_attention(q1, k1, v1, 2)
        assert runtime.prepare_count == n_before + 1
        q2, k2, v2 = _rand_qkv(S=192, seed=8)
        set_hrdit_layer_idx(0)
        set_hap_layer_idx(0)
        mock_attn.optimized_attention(q2, k2, v2, 2)
        assert runtime.prepare_count == n_before + 2


# ---------------------------------------------------------------------------
# T4.1 — clone-state carry-over helper (plan 2026-08-16 G4)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCarryState:
    """Unit tests for :func:`src.spa._hrdit_carry_state` (plan G4/T4.1)."""

    def test_copies_every_listed_attr_when_present(self):
        from src.spa import _HRDIT_PATCHER_ATTRS, _hrdit_carry_state

        class _P:
            pass

        src, dst = _P(), _P()
        sentinel = object()
        for attr in _HRDIT_PATCHER_ATTRS:
            if attr == "_hrdit_state_ref":
                setattr(src, attr, [src])  # a real 1-element ref list
            else:
                setattr(src, attr, sentinel)
        _hrdit_carry_state(src, dst)
        for attr in _HRDIT_PATCHER_ATTRS:
            assert hasattr(dst, attr), f"{attr} was not carried"
            if attr == "_hrdit_state_ref":
                # The ref is re-pointed to the NEW authoritative patcher (dst).
                assert getattr(dst, attr)[0] is dst
            else:
                assert getattr(dst, attr) is sentinel

    def test_copies_nothing_when_absent(self):
        from src.spa import _HRDIT_PATCHER_ATTRS, _hrdit_carry_state

        class _P:
            pass

        src, dst = _P(), _P()
        _hrdit_carry_state(src, dst)
        for attr in _HRDIT_PATCHER_ATTRS:
            assert not hasattr(dst, attr), f"{attr} should not be created"

    def test_no_raise_on_bare_objects(self):
        from src.spa import _hrdit_carry_state

        class _P:
            pass

        # Must not raise on objects with no HRDiT state at all.
        _hrdit_carry_state(_P(), _P())

    def test_state_ref_repointed_to_dst(self):
        """Carrying a state ref re-points its single slot to the new patcher."""
        from src.spa import _hrdit_carry_state

        class _P:
            pass

        src, dst = _P(), _P()
        ref = [src]
        src._hrdit_state_ref = ref
        _hrdit_carry_state(src, dst)
        # dst shares the SAME list object, now pointing at dst.
        assert dst._hrdit_state_ref is ref
        assert ref[0] is dst


# ---------------------------------------------------------------------------
# Non-square guard (2026-08-16 Anima cross-attention crash fix)
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestSpaNonSquareGuard:
    """SPA must decline non-square (cross-attention) calls and run plain attention.

    Anima (cosmos.predict2) runs cross-attention — image queries (T*H*W tokens)
    against text/context keys — through the SAME patched ``optimized_attention``
    symbol.  The averaged passes apply the spatial RoPE rotations to BOTH q and
    k, which is only valid for square self-attention; pre-fix the einsum
    broadcast crashed (``subscript l has size 512 for operand 1 ... previously
    seen size 6300``).  The guard declines SPA for q_len != k_len.
    """

    def _spa_ctx_len(self, L, N=5, D=16):
        """An active SPA context whose variant rotations span ``L`` tokens."""
        try:
            from tests._spa_math_helpers import angles_to_blocks
        except ImportError:
            from _spa_math_helpers import angles_to_blocks

        P = D // 2
        g = torch.Generator().manual_seed(0)
        base = torch.randn(L, P, generator=g) * 0.3
        variants = [torch.randn(L, P, generator=g) * 0.3 for _ in range(N)]
        base_R = angles_to_blocks(base)[None, None]
        variant_Rs = [angles_to_blocks(a)[None, None] for a in variants]
        return SPAContext(active=True, bundle_size=3, base_pe=base_R,
                          variant_pes=variant_Rs, pre_roped=True, fmt="flux",
                          model_key=0, text_len=0)

    def test_nonsquare_call_declines_spa_no_crash(self, mock_attn):
        """q_len=128 vs k_len=64 (cross-attn) -> plain attention, no einsum crash."""
        import torch.nn.functional as F

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        set_spa_context(self._spa_ctx_len(L=128))

        g = torch.Generator().manual_seed(1)
        q = torch.randn(1, 2, 128, 16, generator=g)   # image queries
        k = torch.randn(1, 2, 64, 16, generator=g)    # text/context keys
        v = torch.randn(1, 2, 64, 16, generator=g)
        # Pre-fix this raised RuntimeError in einsum; now it must return plain attention.
        out = mock_attn.optimized_attention(q, k, v, 2)
        ref = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_nonsquare_decline_is_one_time_debug(self, mock_attn, caplog):
        """The non-square decline logs at most ONE debug line per SPA context."""
        import logging

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        ctx = self._spa_ctx_len(L=128)
        set_spa_context(ctx)

        g = torch.Generator().manual_seed(2)
        q = torch.randn(1, 2, 128, 16, generator=g)
        k = torch.randn(1, 2, 64, 16, generator=g)
        v = torch.randn(1, 2, 64, 16, generator=g)
        with caplog.at_level(logging.DEBUG, logger="ComfyUI-DyPE"):
            mock_attn.optimized_attention(q, k, v, 2)
            mock_attn.optimized_attention(q, k, v, 2)  # second call must be silent
        nonsquare = [r for r in caplog.records if "non-square" in r.message]
        assert len(nonsquare) == 1

    def test_nonsquare_counter_still_advances(self, mock_attn):
        """The layer counter advances for declined cross-attn calls (alignment)."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        set_spa_context(self._spa_ctx_len(L=128))

        g = torch.Generator().manual_seed(3)
        q = torch.randn(1, 2, 128, 16, generator=g)
        k = torch.randn(1, 2, 64, 16, generator=g)
        v = torch.randn(1, 2, 64, 16, generator=g)
        mock_attn.optimized_attention(q, k, v, 2)
        mock_attn.optimized_attention(q, k, v, 2)
        assert get_hrdit_layer_idx() == 2

    def test_square_call_still_runs_spa(self, mock_attn):
        """Square self-attention (q_len == k_len) is unaffected by the guard."""
        from src.spa_attn import apply_rope_matrix

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        ctx = self._spa_ctx_len(L=128)
        set_spa_context(ctx)

        g = torch.Generator().manual_seed(4)
        q = torch.randn(1, 2, 128, 16, generator=g)
        k = torch.randn(1, 2, 128, 16, generator=g)
        v = torch.randn(1, 2, 128, 16, generator=g)
        q_base = apply_rope_matrix(q, ctx.base_pe, "flux")
        k_base = apply_rope_matrix(k, ctx.base_pe, "flux")
        out = mock_attn.optimized_attention(q_base, k_base, v, 2)
        # SPA active -> averaged output differs from plain attention of the
        # base-RoPE'd inputs (the variants perturb the rotations).
        import torch.nn.functional as F

        plain = F.scaled_dot_product_attention(q_base, k_base, v, scale=1.0)
        assert not torch.allclose(out, plain, atol=1e-6)

    def test_mixed_self_and_cross_forward(self, mock_attn):
        """A forward mixing square self-attn and non-square cross-attn: self-attn
        runs SPA, cross-attn runs plain, and the counter advances for BOTH."""
        import torch.nn.functional as F

        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        ctx = self._spa_ctx_len(L=128)
        set_spa_context(ctx)

        g = torch.Generator().manual_seed(5)
        q_self = torch.randn(1, 2, 128, 16, generator=g)
        k_self = torch.randn(1, 2, 128, 16, generator=g)
        v_self = torch.randn(1, 2, 128, 16, generator=g)
        q_cross = torch.randn(1, 2, 128, 16, generator=g)
        k_cross = torch.randn(1, 2, 64, 16, generator=g)
        v_cross = torch.randn(1, 2, 64, 16, generator=g)

        # Block: self-attn (square) then cross-attn (non-square), twice.
        out_cross = None
        for _ in range(2):
            mock_attn.optimized_attention(q_self, k_self, v_self, 2)
            out_cross = mock_attn.optimized_attention(q_cross, k_cross, v_cross, 2)

        # Counter advanced for BOTH call kinds (4 calls total).
        assert get_hrdit_layer_idx() == 4
        # Cross-attn output is plain attention (SPA declined).
        ref_cross = F.scaled_dot_product_attention(q_cross, k_cross, v_cross, scale=1.0)
        assert torch.allclose(out_cross, ref_cross, atol=1e-6)
