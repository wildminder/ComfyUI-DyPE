"""P0/T0.3 — Characterize the current install policy (documents the G1 blocker).

Proves that TODAY the averaged-attention wrapper is installed ONLY when SPA is
active (``enable_spa`` and ``bundle_size != 1``).  Consequence: HAP-standalone
(no SPA) is impossible without the P3 install-policy generalization.

``test_no_wrapper_when_spa_off`` PASSES on current code.  Phase P3 (T3.3) adds
the inverted companion test: with HAP enabled the wrapper MUST be installed
even when SPA is off.

Markers: @pytest.mark.mock_integration
"""

import types

import pytest

from src.spa import apply_spa_to_model


@pytest.fixture
def mock_attn():
    """The conftest-provided (pristine SDPA) mock ``comfy.ldm.modules.attention`` module."""
    import comfy.ldm.modules.attention as attn_mod

    return attn_mod


class _MockModel:
    """Minimal stand-in for comfy.model_patcher.ModelPatcher (self-contained)."""

    def __init__(self):
        self.model = types.SimpleNamespace()
        self.model.diffusion_model = types.SimpleNamespace()
        self._object_patches = {}
        self._unet_wrapper = None

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


@pytest.mark.mock_integration
class TestInstallPolicyCharacterization:
    def test_no_wrapper_when_spa_off(self, mock_attn):
        """bundle_size=1 (SPA off) -> no attention wrapper installed.

        This is the G1 blocker: with SPA off there is no hook in the attention
        path, so HAP-standalone has nothing to dispatch through.  PASSES on
        current code; T3.3 inverts it for the HAP-enabled case.
        """
        orig_attn = mock_attn.optimized_attention
        m = apply_spa_to_model(
            _make_flux_mock(), "flux", 2048, 2048, "ntk",
            enable_spa=True, bundle_size=1,
        )
        assert not getattr(m, "_spa_installed", None)
        # The module-level attention symbol is untouched.
        assert mock_attn.optimized_attention is orig_attn

    def test_no_wrapper_when_spa_disabled(self, mock_attn):
        """enable_spa=False -> no wrapper either (same blocker path)."""
        orig_attn = mock_attn.optimized_attention
        m = apply_spa_to_model(
            _make_flux_mock(), "flux", 2048, 2048, "ntk",
            enable_spa=False,
        )
        assert not getattr(m, "_spa_installed", None)
        assert mock_attn.optimized_attention is orig_attn

    def test_wrapper_installed_when_spa_active(self, mock_attn):
        """Control: active SPA (bundle_size=2) DOES install the wrapper today."""
        orig_attn = mock_attn.optimized_attention
        m = apply_spa_to_model(
            _make_flux_mock(), "flux", 2048, 2048, "ntk",
            enable_spa=True, bundle_size=2,
        )
        assert getattr(m, "_spa_installed", None)
        assert mock_attn.optimized_attention is not orig_attn
