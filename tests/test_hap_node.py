"""Tests for the HAP node + ``apply_hap_to_model`` (plan P4: T4.1-T4.4).

The functional (patching) tests exercise ``apply_hap_to_model`` directly rather
than loading the extension's ``__init__.py`` (which transitively imports nodes
that require the *real* ``comfy`` package).  The node *wiring* (schema, inputs,
registration) is covered by text-checks that read ``__init__.py`` directly —
the same pattern as ``tests/test_spa_node.py``.

Markers: @pytest.mark.unit / @pytest.mark.mock_integration
"""

import pathlib
import types

import pytest

from src.hap import ScopePlan, apply_hap_to_model, restore_hap_attention_hook

_INIT = pathlib.Path(__file__).parent.parent / "__init__.py"
_SHIPPED_PLAN = pathlib.Path(__file__).parent.parent / "configs" / "scope_plan_flux.json"


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

    def _copy_dm(self, src):
        dst = types.SimpleNamespace()
        for k, v in vars(src).items():
            setattr(dst, k, v)
        return dst

    def clone(self):
        new = _MockModel()
        new.model.diffusion_model = self._copy_dm(self.model.diffusion_model)
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def _make_flux_mock():
    m = _MockModel()
    m.model.diffusion_model.pe_embedder = types.SimpleNamespace(
        theta=10000, axes_dim=[16, 56, 56]
    )
    return m


def _make_nunchaku_mock():
    m = _MockModel()

    class _NunchakuInner:
        def __init__(self):
            self.pos_embed = types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56])

    # W2.7 fix (2026-08-25): build the dm AS a mutable class instance instead
    # of assigning ``__class__`` on a SimpleNamespace (which raises
    # "assignment only supported for mutable types" on Python 3.13).
    dm = _NunchakuInner()
    dm.model = _NunchakuInner()
    m.model.diffusion_model = dm
    return m


def _tiny_plan():
    return ScopePlan(alphas=[[64.0, 64.0]], betas=[[0.0, 0.0]])


@pytest.fixture
def mock_attn():
    import comfy.ldm.modules.attention as attn_mod

    return attn_mod


# ---------------------------------------------------------------------------
# T4.1 — apply_hap_to_model
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestApplyHap:
    def test_apply_hap_stores_state(self, mock_attn):
        m = apply_hap_to_model(_make_flux_mock(), "flux", _tiny_plan(),
                               anchor_stride=32, text_len=512)
        assert m._hap_ctx is not None
        assert m._hap_ctx.active is True
        assert m._hap_ctx.anchor_stride == 32
        assert m._hap_ctx.text_len == 512
        assert m._hap_plan is m._hap_ctx.plan
        assert getattr(m, "_spa_installed", None)  # shared hook installed
        assert m._hrdit_consumers == {"hap"}

    def test_apply_hap_disabled_passthrough(self, mock_attn):
        orig = mock_attn.optimized_attention
        m = apply_hap_to_model(_make_flux_mock(), "flux", _tiny_plan(), enable_hap=False)
        assert m._hap_ctx is None
        assert not getattr(m, "_spa_installed", None)
        assert mock_attn.optimized_attention is orig

    def test_apply_hap_nunchaku_guard(self, mock_attn, caplog):
        import logging

        orig = mock_attn.optimized_attention
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            m = apply_hap_to_model(_make_nunchaku_mock(), "nunchaku", _tiny_plan())
        assert m._hap_ctx is None
        assert not getattr(m, "_spa_installed", None)
        assert mock_attn.optimized_attention is orig
        assert any("Nunchaku" in rec.message for rec in caplog.records)

    def test_apply_hap_accepts_dict_and_path(self, mock_attn, tmp_path):
        # dict input
        m1 = apply_hap_to_model(_make_flux_mock(), "flux", _tiny_plan().to_dict())
        assert m1._hap_ctx.plan.num_layers == 1
        # path input
        p = tmp_path / "plan.json"
        _tiny_plan().save(p)
        m2 = apply_hap_to_model(_make_flux_mock(), "flux", str(p))
        assert m2._hap_ctx.plan.num_layers == 1

    def test_apply_hap_rejects_bad_type(self, mock_attn):
        with pytest.raises(ValueError, match="scope_plan must be"):
            apply_hap_to_model(_make_flux_mock(), "flux", 42)

    def test_restore_hap_clears_ctx_and_consumer(self, mock_attn):
        orig = mock_attn.optimized_attention
        m = apply_hap_to_model(_make_flux_mock(), "flux", _tiny_plan())
        assert mock_attn.optimized_attention is not orig
        restore_hap_attention_hook(m)
        assert m._hap_ctx is None
        assert mock_attn.optimized_attention is orig


# ---------------------------------------------------------------------------
# T4.2 — HAP node schema + execute (text-checks; __init__ needs real comfy)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHapNodeSchema:
    def _content(self):
        return _INIT.read_text(encoding="utf-8")

    def test_hap_node_class_defined(self):
        assert "class HAP(io.ComfyNode):" in self._content()

    def test_hap_node_registered(self):
        assert "return [DyPE_FLUX, SEGA, SPA, HAP," in self._content()

    def test_hap_imports(self):
        content = self._content()
        assert "apply_hap_to_model" in content
        assert "ScopePlan" in content

    def test_hap_has_scope_plan_path_input(self):
        assert '"scope_plan_path"' in self._content()

    def test_hap_has_anchor_stride_input(self):
        assert '"anchor_stride"' in self._content()

    def test_hap_has_text_len_input(self):
        assert '"text_len"' in self._content()

    def test_hap_has_enable_hap_input(self):
        assert '"enable_hap"' in self._content()

    def test_hap_default_plan_path(self):
        assert "configs/scope_plan_flux.json" in self._content()

    def test_hap_category_and_output(self):
        content = self._content()
        start = content.index("class HAP(io.ComfyNode):")
        end = content.index("class DyPEExtension")
        section = content[start:end]
        assert "model_patches/position_encoding" in section
        assert "io.Model.Output" in section


# ---------------------------------------------------------------------------
# T4.3 — shipped reference FLUX scope plan
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestShippedPlan:
    def test_shipped_plan_exists(self):
        assert _SHIPPED_PLAN.exists(), "configs/scope_plan_flux.json must ship with the pack"

    def test_shipped_plan_loads(self):
        plan = ScopePlan.load(_SHIPPED_PLAN)
        assert plan.num_layers == 57
        assert plan.num_heads == 24
        assert all(a == 2048.0 for row in plan.alphas for a in row)
        assert all(b == 0.0 for row in plan.betas for b in row)


# ---------------------------------------------------------------------------
# T4.2 — clone-state carry-over: chain order independence (plan 2026-08-16 G4)
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestChainOrderIndependence:
    """SPA<->HAP chaining must be behaviour-identical regardless of node order.

    The mock ``_MockModel.clone()`` drops custom attrs exactly like the real
    ``ModelPatcher.clone()``, so these tests prove :func:`_hrdit_carry_state`
    re-applies the other node's state across the clone (plan G4).
    """

    def test_hap_then_spa_carries_both_states(self, mock_attn):
        """HAP -> SPA: the final patcher keeps ``_hap_ctx`` AND SPA gating."""
        from src.spa import apply_spa_to_model

        base = _make_flux_mock()
        m_hap = apply_hap_to_model(base, "flux", _tiny_plan(), text_len=512)
        assert m_hap._hap_ctx is not None
        # Chain SPA on top (clones m_hap, dropping custom attrs, then carries).
        m_final = apply_spa_to_model(
            m_hap, "flux", 2048, 2048, "ntk",
            enable_spa=True, bundle_size=2, spa_steps=3,
        )
        # HAP state survived the clone (the exact scenario that crashed pre-fix).
        assert m_final._hap_ctx is not None
        assert m_final._hap_ctx is m_hap._hap_ctx
        # SPA gating is configured on the final patcher.
        assert m_final._spa_steps == 3
        assert m_final._spa_layer_filter is None
        # Shared hook: both consumers, single install.
        assert m_final._hrdit_consumers == {"spa", "hap"}

    def test_spa_then_hap_carries_both_states(self, mock_attn):
        """SPA -> HAP: the final patcher keeps ``_spa_steps`` AND ``_hap_ctx``."""
        from src.spa import apply_spa_to_model

        base = _make_flux_mock()
        m_spa = apply_spa_to_model(
            base, "flux", 2048, 2048, "ntk",
            enable_spa=True, bundle_size=2, spa_steps=5,
        )
        assert m_spa._spa_steps == 5
        # Chain HAP on top (clones m_spa, dropping custom attrs, then carries).
        m_final = apply_hap_to_model(m_spa, "flux", _tiny_plan(), text_len=512)
        # SPA gating survived the clone.
        assert m_final._spa_steps == 5
        # HAP state is configured on the final patcher.
        assert m_final._hap_ctx is not None
        # Shared hook: both consumers, single install (fast path fired).
        assert m_final._hrdit_consumers == {"spa", "hap"}

    def test_spa_then_hap_no_double_wrapper(self, mock_attn):
        """SPA -> HAP must NOT re-wrap the attention symbol (install fast path)."""
        from src.spa import apply_spa_to_model

        orig = mock_attn.optimized_attention
        base = _make_flux_mock()
        m_spa = apply_spa_to_model(
            base, "flux", 2048, 2048, "ntk", enable_spa=True, bundle_size=2,
        )
        wrapper_after_spa = mock_attn.optimized_attention
        assert wrapper_after_spa is not orig
        # Chain HAP: the carried ``_spa_installed`` triggers the fast path.
        apply_hap_to_model(m_spa, "flux", _tiny_plan())
        assert mock_attn.optimized_attention is wrapper_after_spa  # not re-wrapped

    def test_state_ref_points_at_final_patcher(self, mock_attn):
        """After chaining, ``_hrdit_state_ref`` points at the newest clone so the
        shared unet wrapper reads the combined (authoritative) state."""
        from src.spa import apply_spa_to_model

        base = _make_flux_mock()
        m_hap = apply_hap_to_model(base, "flux", _tiny_plan())
        m_final = apply_spa_to_model(
            m_hap, "flux", 2048, 2048, "ntk", enable_spa=True, bundle_size=2,
        )
        ref = getattr(m_final, "_hrdit_state_ref", None)
        assert ref is not None
        assert ref[0] is m_final

    def test_proportional_or_semantics_across_chain(self, mock_attn):
        """Either node enabling proportional scaling survives the chain (OR)."""
        from src.spa import apply_spa_to_model

        base = _make_flux_mock()
        # HAP enables proportional; SPA does not -> flag must stay True.
        m_hap = apply_hap_to_model(base, "flux", _tiny_plan(), proportional_attention=True)
        assert m_hap._hrdit_proportional_attention is True
        m_final = apply_spa_to_model(
            m_hap, "flux", 2048, 2048, "ntk",
            enable_spa=True, bundle_size=2, proportional_attention=False,
        )
        assert m_final._hrdit_proportional_attention is True
