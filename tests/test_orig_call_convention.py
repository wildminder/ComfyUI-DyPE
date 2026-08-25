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
    set_spa_step_gate,
)

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
