"""Tests for proportional attention scaling (plan phase P7, G4).

- T7.1 ``proportional_scale_ratio()`` pure function (``src/hap.py``);
- T7.2 wrapper integration (q pre-scaling) — see ``TestWrapperIntegration``;
- T7.3 node knobs + plumbing — see ``TestNodeKnobs``.

Reference: ``hrdit/attention.py`` lines 89-93; plan §2.6.

Markers: @pytest.mark.unit / @pytest.mark.mock_integration
Accept (user-run):
    pytest tests/test_proportional_scale.py -k ratio
    pytest tests/test_proportional_scale.py -k wrapper
    pytest tests/test_proportional_scale.py tests/test_spa_node.py tests/test_hap_node.py
"""

import math
import pathlib
import types

import pytest
import torch
import torch.nn.functional as F

from src.hap import (
    HAP_TRAIN_SEQ_LEN,
    ScopePlan,
    apply_hap_to_model,
    proportional_scale_ratio,
)
from src.spa import _hrdit_install_hook, _make_hrdit_wrapper, apply_spa_to_model
from src.spa_context import (
    SPAContext,
    get_hrdit_proportional,
    set_hap_context,
    set_hrdit_layer_idx,
    set_hrdit_proportional,
    set_spa_context,
    set_spa_step_gate,
)

_INIT = pathlib.Path(__file__).parent.parent / "__init__.py"


# ---------------------------------------------------------------------------
# T7.1 — proportional_scale_ratio()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProportionalScaleRatio:
    def test_ratio_one_at_train_seq_len(self):
        """seq 4608 (1024px FLUX: 4096 image + 512 text) → exactly 1.0."""
        assert HAP_TRAIN_SEQ_LEN == 64 * 64 + 512
        assert proportional_scale_ratio(HAP_TRAIN_SEQ_LEN) == 1.0

    def test_ratio_reference_4k(self):
        """seq 66048 (4K FLUX) → sqrt(ln(66048)/ln(4608)), recomputed."""
        expected = math.sqrt(math.log(66048) / math.log(4608))
        assert proportional_scale_ratio(66048) == pytest.approx(expected, abs=1e-12)
        # Sanity: ~1.147 for 4K vs 1024px (W2.7 re-baseline 2026-08-25: the
        # pre-fix bound 1.2 < r < 1.4 was a hand-waved guess; the exact value
        # is sqrt(ln(66048)/ln(4608)) ~= 1.1470, already asserted above).
        assert 1.1 < proportional_scale_ratio(66048) < 1.2

    def test_ratio_monotone_increasing(self):
        seqs = [1024, 2048, 4608, 8192, 16384, 32768, 66048]
        ratios = [proportional_scale_ratio(s) for s in seqs]
        for lo, hi in zip(ratios, ratios[1:]):
            assert hi > lo

    def test_ratio_below_one_under_train_len(self):
        """Below the training extent the ratio is < 1 (faithful to the
        formula; the feature is only intended for seq >= train_seq_len)."""
        assert proportional_scale_ratio(1024) < 1.0
        assert proportional_scale_ratio(2) < 1.0

    def test_ratio_custom_train_seq_len(self):
        assert proportional_scale_ratio(100, train_seq_len=100) == 1.0
        expected = math.sqrt(math.log(400) / math.log(100))
        assert proportional_scale_ratio(400, train_seq_len=100) == pytest.approx(expected, abs=1e-12)

    def test_ratio_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            proportional_scale_ratio(0)
        with pytest.raises(ValueError):
            proportional_scale_ratio(-5)
        with pytest.raises(ValueError):
            proportional_scale_ratio(4608, train_seq_len=1)
        with pytest.raises(ValueError):
            proportional_scale_ratio(4608, train_seq_len=0)


# ---------------------------------------------------------------------------
# T7.2 — wrapper integration (q pre-scaling)
# ---------------------------------------------------------------------------

def _sdpa_orig(q, k, v, heads, mask=None, attn_precision=None,
               skip_reshape=False, skip_output_reshape=False, **kw):
    """Pristine SDPA reference (scale=1.0, matches the conftest mock).

    W2.1: signature-locked via the canonical fixture helper so any drift
    fails at construction (the pre-fix 4-7-arg order caused the stale-mock
    rot fixed in plan 2026-08-25 W2).
    """
    from _hrdit_fixtures import assert_real_signature

    assert_real_signature(_sdpa_orig)
    return F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=0.0,
                                          is_causal=False)


def _rand_qkv(B=1, H=2, S=64, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g)
    k = torch.randn(B, H, S, D, generator=g)
    v = torch.randn(B, H, S, D, generator=g)
    return q, k, v


@pytest.fixture
def mock_attn():
    """The conftest-provided (pristine SDPA) mock attention module."""
    import comfy.ldm.modules.attention as attn_mod

    return attn_mod


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset contextvars around every test (no cross-test leaks)."""
    set_hrdit_layer_idx(0)
    set_hrdit_proportional(False)
    yield
    set_hap_context(None)
    set_spa_context(None)
    set_spa_step_gate(True)
    set_hrdit_layer_idx(0)
    set_hrdit_proportional(False)


@pytest.mark.mock_integration
class TestWrapperIntegration:
    def test_proportional_equals_scaled_logits(self):
        """Enabled: the backend receives ``q * ratio`` (logits scaled by ratio)."""
        seen = []

        def rec(q, k, v, heads, mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False, **kw):
            seen.append(q)
            return q

        wrapper = _make_hrdit_wrapper(rec, is_masked=False)
        q, k, v = _rand_qkv(S=64)
        set_hrdit_proportional(True)
        wrapper(q, k, v, 2)
        ratio = proportional_scale_ratio(64)
        assert len(seen) == 1
        assert torch.equal(seen[0], q * ratio)

    def test_proportional_end_to_end_matches_scaled_sdpa(self):
        """Enabled: full wrapper output == SDPA(q*ratio, k, v) (small seq)."""
        wrapper = _make_hrdit_wrapper(_sdpa_orig, is_masked=False)
        q, k, v = _rand_qkv(S=64)
        set_hrdit_proportional(True)
        out = wrapper(q, k, v, 2)
        ratio = proportional_scale_ratio(64)
        expected = F.scaled_dot_product_attention(q * ratio, k, v, scale=1.0)
        assert torch.allclose(out, expected, atol=0.0, rtol=0.0)

    def test_proportional_noop_at_train_seq_len(self):
        """At seq == train_seq_len the ratio is exactly 1.0 -> q passes through
        bit-identically (multiplying by 1.0 is exact in IEEE)."""
        seen = []

        def rec(q, k, v, heads, mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False, **kw):
            seen.append(q)
            return q

        wrapper = _make_hrdit_wrapper(rec, is_masked=False)
        q, k, v = _rand_qkv(S=HAP_TRAIN_SEQ_LEN)
        set_hrdit_proportional(True)
        wrapper(q, k, v, 2)
        assert len(seen) == 1
        assert torch.equal(seen[0], q)

    def test_proportional_disabled_is_bit_identical(self):
        """Flag off (default): output bit-identical to the pre-feature baseline."""
        wrapper = _make_hrdit_wrapper(_sdpa_orig, is_masked=False)
        q, k, v = _rand_qkv(S=64)
        out = wrapper(q, k, v, 2)
        expected = _sdpa_orig(q, k, v, 2)
        assert torch.equal(out, expected)

    def test_proportional_applies_to_masked_variant(self):
        """The masked backend variant pre-scales q the same way."""
        seen = []

        def rec(q, k, v, heads, mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False, **kw):
            seen.append(q)
            return q

        wrapper = _make_hrdit_wrapper(rec, is_masked=True)
        q, k, v = _rand_qkv(S=64)
        set_hrdit_proportional(True)
        wrapper(q, k, v, 2, None)
        ratio = proportional_scale_ratio(64)
        assert len(seen) == 1
        assert torch.equal(seen[0], q * ratio)

    def test_proportional_applies_to_all_spa_variants(self):
        """SPA active: EVERY variant pass receives the pre-scaled q."""
        seen_q = []

        def recording_orig(q, k, v, heads, mask=None, attn_precision=None,
                           skip_reshape=False, skip_output_reshape=False, **kw):
            seen_q.append(q.clone())
            return _sdpa_orig(q, k, v, heads)

        wrapper = _make_hrdit_wrapper(recording_orig, is_masked=False)
        q, k, v = _rand_qkv(S=64)
        ratio = proportional_scale_ratio(64)

        # Synthetic SPA context with 3 variant rotations (s=2 -> 2*s-1=3 passes).
        # IDENTITY rotations: the variant passes are mathematically plain
        # attention, so the only observable difference is the q pre-scaling.
        L = q.shape[-2]
        P = q.shape[-1] // 2
        eye = torch.eye(2).expand(1, 1, L, P, 2, 2).clone()
        ctx = SPAContext(
            active=True,
            bundle_size=2,
            base_pe=eye.clone(),
            variant_pes=[eye.clone(), eye.clone(), eye.clone()],
            variant_deltas=[eye.clone(), eye.clone(), eye.clone()],
            pre_roped=True,
            fmt="flux",
        )
        set_spa_context(ctx)
        set_hrdit_proportional(True)

        out = wrapper(q, k, v, 2)

        # 3 variant passes, each receiving the pre-scaled q (identity-rotated).
        assert len(seen_q) == 3
        for qq in seen_q:
            assert torch.allclose(qq, q * ratio, atol=1e-6)
        # All passes identical -> the averaged output equals one scaled pass.
        expected = _sdpa_orig(q * ratio, k, v, 2)
        assert torch.allclose(out, expected, atol=1e-6)


class _MockModel:
    """Minimal ModelPatcher stand-in (hook bookkeeping attrs, custom attrs kept)."""

    def __init__(self):
        self._unet_wrapper = None
        self._spa_installed = None
        self._spa_orig_optimized_attention = None
        self._hrdit_consumers = None
        self._hap_ctx = None

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


@pytest.mark.mock_integration
class TestUnetWrapperPlumbing:
    def test_unet_wrapper_sets_flag_from_model_attr(self, mock_attn):
        """The unet wrapper activates the contextvar from ``m._hrdit_proportional_attention``."""
        m = _MockModel()
        m._hrdit_proportional_attention = True
        _hrdit_install_hook(m, "flux", consumer="hap")
        assert m._unet_wrapper is not None

        observed = []

        def model_fn(x, t, **c):
            observed.append(get_hrdit_proportional())
            return x

        m._unet_wrapper(model_fn, {"input": torch.zeros(1),
                                   "timestep": torch.tensor(1.0), "c": {}})
        assert observed == [True]
        # Cleared after the forward -> no leak into the next model's forward.
        assert get_hrdit_proportional() is False

    def test_unet_wrapper_flag_default_off(self, mock_attn):
        """No attr on the model -> the flag stays off during the forward."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="hap")
        observed = []

        def model_fn(x, t, **c):
            observed.append(get_hrdit_proportional())
            return x

        m._unet_wrapper(model_fn, {"input": torch.zeros(1),
                                   "timestep": torch.tensor(1.0), "c": {}})
        assert observed == [False]
        assert get_hrdit_proportional() is False


# ---------------------------------------------------------------------------
# T7.3 — node knobs + plumbing (OR-semantics)
# ---------------------------------------------------------------------------

class _PatcherMock:
    """ModelPatcher stand-in whose ``clone()`` DROPS custom attributes.

    Mimics the REAL ``ModelPatcher.clone()`` (verified against ComfyUI source):
    only KNOWN fields are copied; custom attrs like
    ``_hrdit_proportional_attention`` do NOT survive.  The plumbing must
    therefore read the flag from the SOURCE patcher before cloning.
    """

    def __init__(self):
        self.model = types.SimpleNamespace()
        self.model.diffusion_model = types.SimpleNamespace()
        self._object_patches = {}
        self._unet_wrapper = None

    def clone(self):
        new = _PatcherMock()
        dst = types.SimpleNamespace()
        for k, v in vars(self.model.diffusion_model).items():
            setattr(dst, k, v)
        new.model.diffusion_model = dst
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        # Deliberately NOT copying custom attributes (like the real clone()).
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def _make_flux_patcher():
    m = _PatcherMock()
    m.model.diffusion_model.pe_embedder = types.SimpleNamespace(
        theta=10000, axes_dim=[16, 56, 56]
    )
    return m


def _tiny_plan():
    return ScopePlan(alphas=[[64.0, 64.0]], betas=[[0.0, 0.0]])


@pytest.mark.unit
class TestNodeKnobs:
    def _content(self):
        return _INIT.read_text(encoding="utf-8")

    def test_spa_schema_has_proportional_input(self):
        content = self._content()
        start = content.index("class SPA(io.ComfyNode):")
        end = content.index("class HAP(io.ComfyNode):")
        assert '"proportional_attention"' in content[start:end]

    def test_hap_schema_has_proportional_input(self):
        content = self._content()
        start = content.index("class HAP(io.ComfyNode):")
        end = content.index("class DyPEExtension")
        assert '"proportional_attention"' in content[start:end]

    def test_proportional_default_off_both_nodes(self):
        content = self._content()
        for cls_start, cls_end in (
            ("class SPA(io.ComfyNode):", "class HAP(io.ComfyNode):"),
            ("class HAP(io.ComfyNode):", "class DyPEExtension"),
        ):
            section = content[content.index(cls_start):content.index(cls_end)]
            idx = section.index('"proportional_attention"')
            assert "default=False" in section[idx:idx + 200]

    def test_execute_signatures_plumb_the_flag(self):
        content = self._content()
        assert "proportional_attention: bool = False" in content
        assert "proportional_attention=bool(proportional_attention)" in content


@pytest.mark.mock_integration
class TestPlumbingOrSemantics:
    def test_spa_sets_flag(self, mock_attn):
        m = apply_spa_to_model(_make_flux_patcher(), "flux", 4096, 4096, "ntk",
                               enable_spa=True, proportional_attention=True)
        assert m._hrdit_proportional_attention is True

    def test_spa_default_flag_off(self, mock_attn):
        m = apply_spa_to_model(_make_flux_patcher(), "flux", 4096, 4096, "ntk",
                               enable_spa=True)
        assert getattr(m, "_hrdit_proportional_attention", False) is False

    def test_hap_sets_flag(self, mock_attn):
        m = apply_hap_to_model(_make_flux_patcher(), "flux", _tiny_plan(),
                               proportional_attention=True)
        assert m._hrdit_proportional_attention is True

    def test_hap_default_flag_off(self, mock_attn):
        m = apply_hap_to_model(_make_flux_patcher(), "flux", _tiny_plan())
        assert getattr(m, "_hrdit_proportional_attention", False) is False

    def test_or_semantics_spa_then_hap(self, mock_attn):
        """SPA enables it; a later HAP apply (its own knob off) keeps it on."""
        m1 = apply_spa_to_model(_make_flux_patcher(), "flux", 4096, 4096, "ntk",
                                enable_spa=True, proportional_attention=True)
        m2 = apply_hap_to_model(m1, "flux", _tiny_plan(),
                                proportional_attention=False)
        assert m2._hrdit_proportional_attention is True

    def test_or_semantics_hap_then_spa(self, mock_attn):
        """HAP enables it; a later SPA apply (its own knob off) keeps it on."""
        m1 = apply_hap_to_model(_make_flux_patcher(), "flux", _tiny_plan(),
                                proportional_attention=True)
        m2 = apply_spa_to_model(m1, "flux", 4096, 4096, "ntk",
                                enable_spa=True, proportional_attention=False)
        assert m2._hrdit_proportional_attention is True

    def test_flag_survives_clone_drop(self, mock_attn):
        """The real clone() drops custom attrs; the flag is re-applied from the
        SOURCE patcher (read-before-clone), so a pre-set flag survives."""
        src = _make_flux_patcher()
        src._hrdit_proportional_attention = True
        m = apply_spa_to_model(src, "flux", 4096, 4096, "ntk",
                               enable_spa=True, proportional_attention=False)
        assert m._hrdit_proportional_attention is True
