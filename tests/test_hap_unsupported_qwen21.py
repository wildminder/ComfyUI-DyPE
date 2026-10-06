"""HAP refuses Qwen-Image-2.1 with an ACTIONABLE message (plan T6).

HAP is head-adaptive *pruning* driven by a scope plan whose layer ordinals come
from one attention call per block.  Qwen-Image-2.1's block-causal attention
issues one call per SEQUENCE SEGMENT (mask-carrying text chunks plus a
non-square image segment), so the plan would be indexed by segments and the
pruned head set would be wrong — silently, which is worse than a refusal.

The repo's rule (``src/spa.py``: an unsupported-model error must name the
offender AND the recovery path) is what these tests pin.  "Reload the model" is
explicitly NOT an acceptable remedy — that was a documented failure mode once
already; the fix is another node, not another load.

Markers: @pytest.mark.unit / @pytest.mark.mock_integration
"""

import types

import pytest

from src.hap import ScopePlan, apply_hap_to_model


class _MockModel:
    """Minimal stand-in for comfy.model_patcher.ModelPatcher (self-contained)."""

    def __init__(self):
        self.model = types.SimpleNamespace()
        self.model.diffusion_model = types.SimpleNamespace()
        self._object_patches = {}
        self._unet_wrapper = None
        self._spa_installed = None
        self._spa_orig_optimized_attention = None
        self._hrdit_consumers = None
        self._hap_ctx = None

    def clone(self):
        new = _MockModel()
        src = self.model.diffusion_model
        # Detection reads ``dm.__class__.__name__``, so the clone must PRESERVE
        # the diffusion model's class (a plain SimpleNamespace copy would make
        # every model look unrecognisable).
        if isinstance(src, types.SimpleNamespace):
            dm = types.SimpleNamespace(**vars(src))
        else:
            dm = type(src)()
            dm.__dict__.update(vars(src))
        new.model.diffusion_model = dm
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


class QwenImage21Transformer2DModel:  # noqa: N801 - mirrors ComfyUI's class name
    """Only the class name matters: detection reads ``type(dm).__name__``."""

    pe_embedder = types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56])


class QwenImageTransformer2DModel:  # noqa: N801 - the 1.0 backend
    """Negative control: 1.0 keeps working."""

    pe_embedder = types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56])


def _make_model(dm_cls=QwenImage21Transformer2DModel):
    m = _MockModel()
    m.model.diffusion_model = dm_cls()
    return m


def _tiny_plan():
    return ScopePlan(alphas=[[64.0, 64.0]], betas=[[0.0, 0.0]])


@pytest.fixture
def mock_attn():
    import comfy.ldm.modules.attention as attn_mod

    return attn_mod


@pytest.mark.mock_integration
class TestHapRefusesQwen21:
    def test_qwen21_model_raises(self, mock_attn):
        with pytest.raises(ValueError) as exc:
            apply_hap_to_model(_make_model(), "auto", _tiny_plan())
        assert "Qwen-Image-2.1" in str(exc.value)

    def test_explicit_qwen21_selection_raises_too(self, mock_attn):
        """Selecting the key by hand must not bypass the refusal."""
        with pytest.raises(ValueError):
            apply_hap_to_model(_make_model(QwenImageTransformer2DModel), "qwen21",
                               _tiny_plan())

    def test_refusal_installs_nothing(self, mock_attn):
        """No partial state: no context, no hook, no patched attention symbol."""
        orig = mock_attn.optimized_attention
        m = _make_model()
        with pytest.raises(ValueError):
            apply_hap_to_model(m, "auto", _tiny_plan())
        assert m._hap_ctx is None                       # caller patcher untouched
        assert not getattr(m, "_spa_installed", None)
        assert not m._unet_wrapper
        assert mock_attn.optimized_attention is orig    # symbol never patched

    def test_hap_message_does_not_suggest_reload_only(self, mock_attn):
        """The message must name the offender, the mechanism and the way out."""
        with pytest.raises(ValueError) as exc:
            apply_hap_to_model(_make_model(), "auto", _tiny_plan())
        msg = str(exc.value)

        # 1. names the offender
        assert "Qwen-Image-2.1" in msg
        # 2. explains the mechanism (why a scope plan cannot apply)
        assert "segment" in msg.lower()
        # 3. offers the recovery path: another node, by name
        assert "SPA" in msg and "DyPE" in msg and "SEGA" in msg
        # 4. NOT the "reload the model" failure mode the repo already hit once
        low = msg.lower()
        assert "reload" not in low
        assert "re-load" not in low
        assert "try again" not in low
        assert "unsupported" not in low.replace("is not supported", "")


@pytest.mark.mock_integration
class TestHapStillWorksForOtherBackends:
    """Negative controls: the refusal must not leak onto a supported model."""

    def test_hap_still_works_for_qwen_1_0(self, mock_attn):
        """Qwen-Image 1.0 resolves to ``qwen`` and HAP applies normally."""
        m = apply_hap_to_model(_make_model(QwenImageTransformer2DModel), "auto",
                               _tiny_plan())
        assert m._hap_ctx is not None and m._hap_ctx.active is True
        assert getattr(m, "_spa_installed", None)

    def test_refusal_also_covers_a_disabled_hap_node(self, mock_attn):
        """``enable_hap=False`` is NOT a bypass — the refusal still fires.

        The pack's rule is "never a silent broken path": a HAP node wired into a
        2.1 graph with HAP switched off would otherwise sit there doing nothing,
        indistinguishable from a hook that failed to install.  The same guard the
        Nunchaku case applies (unsupported backend is rejected regardless of the
        enabled flag) is used here.
        """
        with pytest.raises(ValueError):
            apply_hap_to_model(_make_model(), "auto", _tiny_plan(), enable_hap=False)