"""Root-cause characterization + fix proof for the Anima HAP crash (plan 2026-08-16).

The production traceback (HAP works on Krea/Z-Image, crashes on Anima)::

    File ".../comfy/ldm/cosmos/predict2.py", line 68, in torch_attention_op
        return optimized_attention(q, k, v, heads, skip_reshape=True,
                                   transformer_options=transformer_options)
    File ".../src/spa.py", line 984, in _attn  -> falls through to orig(...)
    File ".../comfy/ldm/modules/attention.py", line 556, in attention_pytorch
        if mask.ndim == 2:
    AttributeError: 'bool' object has no attribute 'ndim'

Four compounding gaps are fixed and proven here:

* **G1** — :func:`src.spa._make_hrdit_wrapper` called ``orig`` with a wrong
  positional order.  The unmasked variant fed ``skip_reshape`` (a bool) into the
  real ``mask`` parameter -> ``mask.ndim`` -> the Anima crash; the masked variant
  mis-forwarded ``skip_reshape``->``attn_precision`` and ``transformer_options``->
  ``skip_reshape``.  Fix: the wrapper now mirrors the real ComfyUI signature
  bit-for-bit (positional slots 5-8 == ``mask``, ``attn_precision``,
  ``skip_reshape``, ``skip_output_reshape``).
* **G2** — the pytest mock mirrored the wrapper's *invented* signature, so the
  suite never caught G1 (closed-loop mock-fidelity failure).  Fix: the conftest
  mock now mirrors the real signature (tripwire in test_orig_call_convention.py).
* **G3** — :meth:`src.hap.HapRuntime.attn` had no guards for Anima's realities
  (cross-attention ``kv_len != q_len``; head-count mismatch vs the FLUX plan).
  Fix: decline-guards (covered in test_hap_runtime.py).
* **G4** — real ``ModelPatcher.clone()`` drops custom attrs, so chaining
  SPA<->HAP lost the other node's state depending on order.  Fix:
  :func:`src.spa._hrdit_carry_state` + the ``_hrdit_state_ref`` indirection.

These tests are written as **fix proofs**: each asserts the CORRECT post-fix
behaviour and is constructed so that re-introducing the bug makes it fail (the
recording ``orig`` mimics the real ``attention_pytorch``'s ``mask.ndim`` access,
so a positional regression crashes the test exactly like production).

Markers: @pytest.mark.mock_integration
"""


import pytest
import torch
import torch.nn.functional as F

from src import hap
from src.spa import _hrdit_install_hook
from src.spa_context import (
    set_hap_context,
    set_hrdit_layer_idx,
    set_spa_context,
    set_spa_step_gate,
)


@pytest.fixture
def mock_attn():
    """The conftest-provided (pristine, REAL-signature) mock attention module."""
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
    """Replace ``optimized_attention`` with a REAL-signature recording ``orig``.

    The recorder mirrors ``comfy/ldm/modules/attention.py::attention_pytorch``:
    it touches ``mask.ndim`` whenever a mask is present — exactly the line that
    crashed when the pre-fix wrapper fed the ``skip_reshape`` bool into the
    ``mask`` slot.  Re-introducing the positional bug therefore raises here.

    Returns the list of per-call records (dicts of what ``orig`` received).
    """
    calls = []

    def recording_orig(q, k, v, heads, mask=None, attn_precision=None,
                       skip_reshape=False, skip_output_reshape=False, **kwargs):
        if mask is not None:
            # Real attention_pytorch: ``if mask.ndim == 2:`` — crashes on a bool.
            _ = mask.ndim
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
    mock_attn.optimized_attention_masked = recording_orig  # real alias
    return calls


def _rand_qkv(B=1, H=2, S=64, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g)
    k = torch.randn(B, H, S, D, generator=g)
    v = torch.randn(B, H, S, D, generator=g)
    return q, k, v


# ---------------------------------------------------------------------------
# G1 — the Anima crash (unmasked cosmos convention)
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestAnimaCrashFix:
    def test_anima_cosmos_convention_no_crash(self, mock_attn):
        """The exact production call runs without exception after the fix.

        Anima (cosmos predict2) calls the UNMASKED symbol with ``skip_reshape``
        and ``transformer_options`` as keywords and NO mask.  With the wrapper
        installed and HAP/SPA inactive (so control falls through to ``orig``),
        this must NOT raise and must produce plain SDPA output.
        """
        calls = _install_recording_orig(mock_attn)
        m = _MockModel()
        _hrdit_install_hook(m, "anima", consumer="hap")  # no hap ctx -> dispatch None

        q, k, v = _rand_qkv(seed=1)
        # The cosmos convention (predict2.py:68).
        out = mock_attn.optimized_attention(
            q, k, v, 2, skip_reshape=True, transformer_options={}
        )
        ref = F.scaled_dot_product_attention(q, k, v, scale=1.0)
        assert torch.allclose(out, ref, atol=1e-6)
        # orig was reached exactly once...
        assert len(calls) == 1
        # ...with the mask slot EMPTY (Anima sends no mask) — pre-fix it was True.
        assert calls[0]["mask"] is None
        assert calls[0]["skip_reshape"] is True
        assert calls[0]["kwargs"].get("transformer_options") == {}

    def test_orig_receives_exact_caller_args_unmasked(self, mock_attn):
        """The wrapper forwards EVERY slot exactly as the backend sent it."""
        calls = _install_recording_orig(mock_attn)
        m = _MockModel()
        _hrdit_install_hook(m, "anima", consumer="hap")

        q, k, v = _rand_qkv(seed=2)
        to = {"some": "option"}
        mock_attn.optimized_attention(
            q, k, v, 2, skip_reshape=True, skip_output_reshape=True,
            transformer_options=to, enable_gqa=True,
        )
        assert len(calls) == 1
        rec = calls[0]
        assert rec["mask"] is None
        assert rec["attn_precision"] is None
        assert rec["skip_reshape"] is True
        assert rec["skip_output_reshape"] is True
        # Extras ride **kw untouched.
        assert rec["kwargs"]["transformer_options"] == to
        assert rec["kwargs"]["enable_gqa"] is True


# ---------------------------------------------------------------------------
# G1 — masked-variant mis-forwarding (Krea/Qwen/Z-Image convention)
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestMaskedConventionFix:
    def test_masked_call_forwards_correctly(self, mock_attn):
        """A masked call forwards mask/skip_reshape/attn_precision to the right slots.

        Pre-fix the masked variant sent ``skip_reshape`` into ``attn_precision``
        and ``transformer_options`` into ``skip_reshape``.  After the fix, orig
        receives exactly what the caller sent.
        """
        calls = _install_recording_orig(mock_attn)
        m = _MockModel()
        _hrdit_install_hook(m, "krea2", consumer="spa")  # masked backend target

        q, k, v = _rand_qkv(S=64, seed=3)
        mask = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        to = {"k": 1}
        mock_attn.optimized_attention_masked(
            q, k, v, 2, mask, skip_reshape=True, transformer_options=to,
        )
        assert len(calls) == 1
        rec = calls[0]
        # The mask tensor lands in the mask slot (slot 5), not skipped.
        assert rec["mask"] is mask
        # attn_precision was NOT sent -> must be None (pre-fix it was True).
        assert rec["attn_precision"] is None
        # skip_reshape lands in slot 7 (pre-fix it received the to-dict).
        assert rec["skip_reshape"] is True
        assert rec["kwargs"]["transformer_options"] == to

    def test_masked_positional_mask_only(self, mock_attn):
        """Qwen/Z-Image positional-mask convention: ``(q,k,v,heads,mask)``."""
        calls = _install_recording_orig(mock_attn)
        m = _MockModel()
        _hrdit_install_hook(m, "qwen", consumer="spa")

        q, k, v = _rand_qkv(S=64, seed=4)
        mask = torch.ones(1, 1, 64, 64, dtype=torch.bool)
        mock_attn.optimized_attention_masked(q, k, v, 2, mask)
        assert len(calls) == 1
        assert calls[0]["mask"] is mask
        assert calls[0]["skip_reshape"] is False
        assert calls[0]["attn_precision"] is None
