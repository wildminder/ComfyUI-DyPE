"""Call-convention matrix + mock-signature conformance tripwire (plan 2026-08-16).

T1.2 / T2.2 of the Anima-crash-fix plan.  Two responsibilities:

1. **Convention matrix** — for every real backend call convention (FLUX kw,
   Anima kw-no-mask, Qwen masked positional, Krea masked kw, Z-Image masked
   positional, CrossAttention with ``attn_precision``), assert that the wrapper
   forwards to ``orig`` EXACTLY what the backend sent (positional slots 5-8 ==
   ``mask``, ``attn_precision``, ``skip_reshape``, ``skip_output_reshape``;
   everything else via ``**kw``).  With SPA active, EVERY variant pass uses the
   same correct convention.

2. **Mock-signature tripwire** — assert the conftest mock's parameter names/order
   equal the canonical ComfyUI order, so a future drift back to the inverted
   (pre-fix) signature fails loudly instead of silently re-hiding a wrapper bug.

Markers: @pytest.mark.unit / @pytest.mark.mock_integration
"""

import inspect
import sys
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
    set_spa_context,
    set_spa_joint_mode,
    set_spa_step_gate,
)

try:
    from tests._hrdit_fixtures import make_recording_orig
except ImportError:  # namespace-package import fallback
    from _hrdit_fixtures import make_recording_orig

#: Canonical real-ComfyUI attention parameter order (attention_pytorch).
CANONICAL_PARAMS = (
    "q", "k", "v", "heads", "mask", "attn_precision",
    "skip_reshape", "skip_output_reshape",
)


@pytest.fixture
def mock_attn():
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


@pytest.fixture(autouse=True)
def _clean_state():
    hap.HapRuntime.reset()
    set_hrdit_layer_idx(0)
    yield
    set_hap_context(None)
    set_spa_context(None)
    set_spa_step_gate(True)
    set_spa_joint_mode(None)
    set_hrdit_layer_idx(0)
    hap.HapRuntime.reset()


def _install_recording_orig(mock_attn):
    """Install a REAL-signature recording orig (touches mask.ndim like the real fn)."""
    calls = []

    def recording_orig(q, k, v, heads, mask=None, attn_precision=None,
                       skip_reshape=False, skip_output_reshape=False, **kwargs):
        if mask is not None:
            _ = mask.ndim  # real attention_pytorch behaviour
        calls.append({
            "mask": mask,
            "attn_precision": attn_precision,
            "skip_reshape": skip_reshape,
            "skip_output_reshape": skip_output_reshape,
            "kwargs": kwargs,
        })
        return F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=0.0,
                                              is_causal=False)

    mock_attn.optimized_attention = recording_orig
    mock_attn.optimized_attention_masked = recording_orig
    return calls


def _rand_qkv(B=1, H=2, S=64, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g)
    k = torch.randn(B, H, S, D, generator=g)
    v = torch.randn(B, H, S, D, generator=g)
    return q, k, v


# ---------------------------------------------------------------------------
# T2.2 — mock-signature conformance tripwire
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestMockSignatureTripwire:
    def test_canonical_fixture_signature_locked(self):
        """W2.1 (CRIT-002/IMP-108): the shared ``make_recording_orig`` fixture
        must carry the canonical real signature, and the lock helper must
        reject any drift."""
        from _hrdit_fixtures import assert_real_signature, make_recording_orig

        orig = make_recording_orig()
        assert_real_signature(orig)  # must not raise

        # A deliberately wrong mock (pre-fix inverted order) MUST be rejected.
        def bad_orig(q, k, v, heads, skip_reshape=False, mask=None,
                     transformer_options=None, **kw):
            return F.scaled_dot_product_attention(q, k, v)

        with pytest.raises(AssertionError, match="signature drifted"):
            assert_real_signature(bad_orig)

    def test_conftest_mock_matches_real_signature(self, mock_attn):
        """The conftest mock's first-8 params MUST equal the canonical order.

        This is the tripwire against the G2 closed-loop mock-fidelity failure:
        the pre-fix mock had ``skip_reshape`` before ``mask`` and therefore
        agreed with the buggy wrapper instead of real ComfyUI.
        """
        sig = inspect.signature(mock_attn.optimized_attention)
        names = tuple(sig.parameters.keys())[:8]
        assert names == CANONICAL_PARAMS, (
            "The mock attention signature drifted from real ComfyUI "
            f"(expected {CANONICAL_PARAMS}, got {names}).  Tests would again "
            "mask positional wrapper bugs."
        )

    def test_masked_alias_is_same_function(self, mock_attn):
        """Real ComfyUI: ``optimized_attention_masked = optimized_attention``."""
        assert mock_attn.optimized_attention_masked is mock_attn.optimized_attention


# ---------------------------------------------------------------------------
# T1.2 — call-convention matrix (all 6 real backend conventions)
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestCallConventionMatrix:
    def _run(self, mock_attn, backend, call):
        """Install the hook for ``backend`` and run ``call(symbol)``; return calls."""
        calls = _install_recording_orig(mock_attn)
        m = _MockModel()
        _hrdit_install_hook(m, backend, consumer="hap")  # no hap ctx -> orig path
        call(mock_attn)
        return calls

    def test_flux_kw_convention(self, mock_attn):
        """FLUX: ``optimized_attention(q,k,v,heads, skip_reshape=True, mask=mask,
        transformer_options=to)`` — mask as KEYWORD."""
        q, k, v = _rand_qkv(seed=10)
        mask = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        to = {"flux": True}
        calls = self._run(
            mock_attn, "flux",
            lambda mod: mod.optimized_attention(
                q, k, v, 2, skip_reshape=True, mask=mask, transformer_options=to),
        )
        assert len(calls) == 1
        rec = calls[0]
        assert rec["mask"] is mask
        assert rec["skip_reshape"] is True
        assert rec["attn_precision"] is None
        assert rec["kwargs"]["transformer_options"] == to

    def test_anima_kw_no_mask_convention(self, mock_attn):
        """Anima: ``optimized_attention(q,k,v,heads, skip_reshape=True,
        transformer_options=to)`` — NO mask at all."""
        q, k, v = _rand_qkv(seed=11)
        to = {"anima": True}
        calls = self._run(
            mock_attn, "anima",
            lambda mod: mod.optimized_attention(
                q, k, v, 2, skip_reshape=True, transformer_options=to),
        )
        assert len(calls) == 1
        rec = calls[0]
        assert rec["mask"] is None
        assert rec["skip_reshape"] is True
        assert rec["kwargs"]["transformer_options"] == to

    def test_qwen_masked_positional_convention(self, mock_attn):
        """Qwen: ``optimized_attention_masked(q,k,v,heads, attn_mask,
        transformer_options=to)`` — mask POSITIONAL slot 5."""
        q, k, v = _rand_qkv(seed=12)
        mask = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        to = {"qwen": True}
        calls = self._run(
            mock_attn, "qwen",
            lambda mod: mod.optimized_attention_masked(
                q, k, v, 2, mask, transformer_options=to),
        )
        assert len(calls) == 1
        rec = calls[0]
        assert rec["mask"] is mask
        assert rec["skip_reshape"] is False
        assert rec["kwargs"]["transformer_options"] == to

    def test_krea_masked_kw_convention(self, mock_attn):
        """Krea-2: ``optimized_attention_masked(q,k,v,heads, mask=mask,
        skip_reshape=True, transformer_options=to)``."""
        q, k, v = _rand_qkv(seed=13)
        mask = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        to = {"krea": True}
        calls = self._run(
            mock_attn, "krea2",
            lambda mod: mod.optimized_attention_masked(
                q, k, v, 2, mask=mask, skip_reshape=True, transformer_options=to),
        )
        assert len(calls) == 1
        rec = calls[0]
        assert rec["mask"] is mask
        assert rec["skip_reshape"] is True
        assert rec["kwargs"]["transformer_options"] == to

    def test_zimage_masked_positional_convention(self, mock_attn):
        """Z-Image: ``optimized_attention_masked(xq,xk,xv,heads, x_mask,
        skip_reshape=True, transformer_options=to)`` — mask positional + kw skip."""
        q, k, v = _rand_qkv(seed=14)
        mask = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        to = {"zimage": True}
        calls = self._run(
            mock_attn, "zimage",
            lambda mod: mod.optimized_attention_masked(
                q, k, v, 2, mask, skip_reshape=True, transformer_options=to),
        )
        assert len(calls) == 1
        rec = calls[0]
        assert rec["mask"] is mask
        assert rec["skip_reshape"] is True
        assert rec["kwargs"]["transformer_options"] == to

    def test_cross_attention_attn_precision_convention(self, mock_attn):
        """CrossAttention: ``optimized_attention(q,k,v,heads, attn_precision=...,
        transformer_options=to)`` — attn_precision forwarded untouched."""
        q, k, v = _rand_qkv(seed=15)
        to = {"cross": True}
        calls = self._run(
            mock_attn, "flux",
            lambda mod: mod.optimized_attention(
                q, k, v, 2, attn_precision="high", transformer_options=to),
        )
        assert len(calls) == 1
        rec = calls[0]
        assert rec["attn_precision"] == "high"
        assert rec["mask"] is None
        assert rec["kwargs"]["transformer_options"] == to

    def test_extra_kwargs_forwarded(self, mock_attn):
        """``skip_output_reshape`` / ``enable_gqa`` / ``scale`` ride **kw."""
        q, k, v = _rand_qkv(seed=16)
        calls = self._run(
            mock_attn, "flux",
            lambda mod: mod.optimized_attention(
                q, k, v, 2, skip_output_reshape=True, enable_gqa=True, scale=0.5),
        )
        assert len(calls) == 1
        rec = calls[0]
        assert rec["skip_output_reshape"] is True
        assert rec["kwargs"]["enable_gqa"] is True
        assert rec["kwargs"]["scale"] == 0.5


# ---------------------------------------------------------------------------
# T1.2 — SPA-active variant passes use the same correct convention
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestSpaActiveConvention:
    def test_spa_variant_passes_use_correct_convention(self, mock_attn):
        """With SPA active (N variant passes), EVERY pass forwards the caller's
        args with the real positional convention (no mis-forwarding on any pass)."""
        calls = _install_recording_orig(mock_attn)
        m = _MockModel()
        _hrdit_install_hook(m, "anima", consumer="spa")

        # Identity-variant SPA context (3 variants -> 3 passes) so the math is
        # plain attention but the PASS COUNT / convention is observable.
        L, D = 64, 16
        P = D // 2
        eye = torch.eye(2).expand(1, 1, L, P, 2, 2).clone()
        ctx = SPAContext(
            active=True, bundle_size=2, base_pe=eye.clone(),
            variant_pes=[eye.clone() for _ in range(3)],
            variant_deltas=[eye.clone() for _ in range(3)],
            pre_roped=True, fmt="flux", text_len=0,
        )
        set_spa_context(ctx)

        q, k, v = _rand_qkv(S=L, D=D, seed=17)
        to = {"spa": True}
        out = mock_attn.optimized_attention(
            q, k, v, 2, skip_reshape=True, transformer_options=to)

        # 3 variant passes, each reaching orig with the correct convention.
        assert len(calls) == 3
        for rec in calls:
            assert rec["mask"] is None
            assert rec["skip_reshape"] is True
            assert rec["attn_precision"] is None
            assert rec["kwargs"]["transformer_options"] == to
        # Output is finite and shaped correctly (averaged plain attention).
        assert out.shape == q.shape
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Qwen-Image-2.1 conventions (its OWN bound symbol) + the prefill-cache path
# ---------------------------------------------------------------------------

#: Small-but-real 2.1 geometry: 2 text tokens + a 2x2 image grid, 4 heads of
#: dim 8 -> head-FLATTENED width 32 and rotations over P = D // 2 = 4.
QWEN21_TEXT_LEN = 2
QWEN21_TOTAL_LEN = QWEN21_TEXT_LEN + 4
QWEN21_HEADS = 4
QWEN21_HEAD_DIM = 8


def _qwen21_orig(record):
    """A 2.1-shaped ``orig``: head-flattened in, ComfyUI's head split inside.

    Built ON TOP of the mandated :func:`make_recording_orig` factory (the real
    ComfyUI signature is still asserted at construction time).  The shim adds
    only the two conventions 2.1 relies on and the factory does not model: the
    ``not skip_reshape`` head split that ``attention_pytorch`` performs on
    ``(B, N, H*D)`` inputs, and the flatten-back matching
    ``skip_output_reshape=False``.  The 8 recorded slots are captured BEFORE the
    split, so they describe the convention as the caller sent it.
    """
    base = make_recording_orig(record=record)

    def orig(q, k, v, heads, mask=None, attn_precision=None,
             skip_reshape=False, skip_output_reshape=False, **kwargs):
        if not skip_reshape:
            d = q.shape[-1] // heads
            q = q.reshape(q.shape[0], q.shape[1], heads, d).transpose(1, 2)
            k = k.reshape(k.shape[0], k.shape[1], heads, d).transpose(1, 2)
            v = v.reshape(v.shape[0], v.shape[1], heads, d).transpose(1, 2)
        out = base(q, k, v, heads, mask, attn_precision,
                   skip_reshape, skip_output_reshape, **kwargs)
        if not skip_output_reshape:
            b, h, n, d = out.shape
            out = out.permute(0, 2, 1, 3).reshape(b, n, h * d)
        return out

    return orig


def _qwen21_ctx(total_len=QWEN21_TOTAL_LEN, n_variants=3):
    """Identity-variant SPA context in 2.1's flux layout, with ``total_len``.

    Identity rotations keep the math plain attention while the PASS COUNT and
    the segment arithmetic (``end == total_len`` -> target segment) stay
    observable.
    """
    eye = torch.eye(2).expand(1, 1, total_len, QWEN21_HEAD_DIM // 2, 2, 2).clone()
    return SPAContext(
        active=True, bundle_size=n_variants, base_pe=eye.clone(),
        variant_pes=[eye.clone() for _ in range(n_variants)],
        variant_deltas=[eye.clone() for _ in range(n_variants)],
        pre_roped=True, fmt="flux", text_len=QWEN21_TEXT_LEN,
        total_len=total_len,
    )


def _qwen21_qkv(length, seed):
    """Head-flattened ``(B, N, H*D)`` q/k/v exactly as 2.1 hands them over."""
    g = torch.Generator().manual_seed(seed)
    width = QWEN21_HEADS * QWEN21_HEAD_DIM
    return (
        torch.randn(1, length, width, generator=g),
        torch.randn(1, length, width, generator=g),
        torch.randn(1, length, width, generator=g),
    )


@pytest.fixture
def qwen21_backend(monkeypatch):
    """Install the hook on ``comfy.ldm.qwen_image21.model``'s OWN symbol.

    2.1 binds the unmasked ``optimized_attention`` into its own module and
    never looks at ``comfy.ldm.modules.attention``, so the convention has to be
    asserted on THAT symbol — a test driving the global module attribute would
    pass even if 2.1's real call site were left unpatched.  Yields
    ``(module, calls, patcher)``; uninstalls on teardown.
    """
    from src.spa import _spa_restore_installed

    record = []
    mod = types.ModuleType("comfy.ldm.qwen_image21.model")
    mod.optimized_attention = _qwen21_orig(record)
    monkeypatch.setitem(sys.modules, "comfy.ldm.qwen_image21", types.ModuleType("p"))
    monkeypatch.setitem(sys.modules, "comfy.ldm.qwen_image21.model", mod)

    m = _MockModel()
    m._object_patches = {}
    _hrdit_install_hook(m, "qwen21", consumer="spa")
    try:
        yield mod, record, m
    finally:
        _spa_restore_installed(m)


@pytest.mark.mock_integration
class TestQwen21CallConventions:
    """2.1's real call convention, as ``build_sequence``/``prefix_cached_attention``
    emit it (``comfy/ldm/qwen_image21/model.py``).
    """

    def test_block_causal_text_segment_convention(self, qwen21_backend):
        """Text segment: ``(..., heads, mask=mask, transformer_options=...,
        preferred_attention=...)`` — mask as KEYWORD, heads positional slot 4."""
        mod, record, _ = qwen21_backend
        to = {"qwen21": True}
        mask = torch.ones(QWEN21_TEXT_LEN, QWEN21_TEXT_LEN, dtype=torch.bool).tril()
        q, k, v = _qwen21_qkv(QWEN21_TEXT_LEN, seed=30)

        out = mod.optimized_attention(q, k, v, QWEN21_HEADS, mask=mask,
                                      transformer_options=to, preferred_attention=None)

        assert len(record) == 1  # a masked text segment declines SPA
        rec = record[0]
        assert rec[4] is mask          # slot 5 == mask
        assert rec[5] is None          # slot 6 == attn_precision (never sent)
        assert rec[6] is False         # slot 7 == skip_reshape (never sent)
        assert rec[7] is False         # slot 8 == skip_output_reshape
        assert out.shape == (1, QWEN21_TEXT_LEN, QWEN21_HEADS * QWEN21_HEAD_DIM)

    def test_block_causal_image_segment_runs_spa_with_the_same_convention(self, qwen21_backend):
        """Image target segment: mask=None, and EVERY SPA pass forwards the
        caller's convention unchanged."""
        mod, record, _ = qwen21_backend
        set_hrdit_layer_idx(0)
        set_spa_joint_mode("causal_prefix")
        ctx = _qwen21_ctx()
        set_spa_context(ctx)
        to = {"qwen21": True}
        q, k, v = _qwen21_qkv(QWEN21_TOTAL_LEN, seed=31)
        try:
            out = mod.optimized_attention(q[:, QWEN21_TEXT_LEN:], k, v, QWEN21_HEADS,
                                          mask=None, transformer_options=to,
                                          preferred_attention=None)
        finally:
            set_spa_context(None)
            set_spa_joint_mode(None)

        assert len(record) == len(ctx.variant_pes)  # SPA ran, not a silent no-op
        for rec in record:
            assert rec[4] is None
            assert rec[5] is None
            assert rec[6] is False
            assert rec[7] is False
        assert out.shape == (1, QWEN21_TOTAL_LEN - QWEN21_TEXT_LEN,
                             QWEN21_HEADS * QWEN21_HEAD_DIM)

    def test_prefill_cache_call_convention(self, qwen21_backend):
        """Cached step: ONE unmasked call, ``q`` target rows over
        ``k = [cached prefix, target]`` — it lands on the target branch and runs
        SPA with the same convention.

        ``prefix_cached_attention`` is not special-cased anywhere; the claim that
        it is compatible is pinned here rather than assumed.
        """
        mod, record, _ = qwen21_backend
        set_hrdit_layer_idx(0)
        set_spa_joint_mode("causal_prefix")
        ctx = _qwen21_ctx()
        set_spa_context(ctx)
        to = {"qwen21": True}
        target_q, target_k, target_v = _qwen21_qkv(
            QWEN21_TOTAL_LEN - QWEN21_TEXT_LEN, seed=32)
        # ``prefix_cached_attention``: the cached prefix's k/v are concatenated in
        # front of the step's TARGET-ONLY k/v, so k spans the full sequence.
        prefix_k, prefix_v = _qwen21_qkv(QWEN21_TEXT_LEN, seed=33)[1:]
        k = torch.cat([prefix_k, target_k], dim=1)
        v = torch.cat([prefix_v, target_v], dim=1)
        try:
            out = mod.optimized_attention(target_q, k, v, QWEN21_HEADS,
                                          transformer_options=to,
                                          preferred_attention=None)
        finally:
            set_spa_context(None)
            set_spa_joint_mode(None)

        assert len(record) == len(ctx.variant_pes)  # ran SPA, did NOT decline
        for rec in record:
            assert rec[4] is None       # no mask on the cached path at all
            assert rec[6] is False      # skip_reshape never sent by 2.1
            assert rec[7] is False
            # The recording happens after ``attention_pytorch``'s own head split,
            # so the layouts are observed in head form: k spans [prefix, target],
            # q stays target-only.
            assert rec[1].shape == (1, QWEN21_HEADS, QWEN21_TOTAL_LEN, QWEN21_HEAD_DIM)
            assert rec[0].shape == (1, QWEN21_HEADS,
                                    QWEN21_TOTAL_LEN - QWEN21_TEXT_LEN, QWEN21_HEAD_DIM)
        assert out.shape == target_q.shape
        assert torch.isfinite(out).all()

    def test_qwen_1_0_masked_target_is_not_patched(self, qwen21_backend):
        """Negative control: 2.1's own module keeps the 1.0 symbol untouched."""
        mod, _, _ = qwen21_backend
        assert not hasattr(mod, "optimized_attention_masked")
