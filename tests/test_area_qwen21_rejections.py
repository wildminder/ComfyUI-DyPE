"""Qwen-Image-2.1 — the install-time rejection contract (plan 2026-10-21, §6, §7/S9).

Every negative path of the Qwen-Image-2.1 adaptation, in one place:

* the install-time REFUSALS (N5 a fused kernel, N6 a contradicting model type,
  N8 a missing ``pe_embedder``, N10 a foreign wrapper) raise a ``ValueError``
  that names the offending condition and the remedy, and they are checked
  BEFORE the seam is rebound — so a refused install leaves the module
  pristine;
* the FORWARD-time refusals (N11 a patched forward) run plain attention with
  a one-time warning: the patches are per-forward transformer options, so
  the install cannot know them;
* the prefix K/V cache (N12) is switched OFF on AReA's own clone: the cached
  path is length-indistinguishable from the plain image segment, so with the
  cache on AReA would engage on no call at all.

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from src import area
from src.area_arch import resolve_area_arch
from src.area_qwen21 import QWEN21_SEAM_MODULE, QWEN21_SEAM_SYMBOL

try:
    from tests._qwen21_seam_fixtures import (
        QwenImage21Transformer2DModel,
        qwen21_backend_module,
    )
except ImportError:  # namespace-package import fallback
    from _qwen21_seam_fixtures import (
        QwenImage21Transformer2DModel,
        qwen21_backend_module,
    )

pytestmark = pytest.mark.unit

#: A canvas past the native extent, so the install really runs.
WIDTH, HEIGHT = 1040, 1040


class _Patcher:
    """A patcher mock with the attributes the install reads.

    ``model_options`` is the real patcher's own field (comfy deep-copies it on
    ``clone()``); the mock copies it the same way, so AReA's prefix-cache
    switch lands on the clone and not on the caller's patcher.
    """

    def __init__(self, dm):
        self.model = types.SimpleNamespace(diffusion_model=dm)
        self.model_options = {"transformer_options": {}}

    def clone(self):
        new = _Patcher(self.model.diffusion_model)
        new.model_options = {
            "transformer_options": dict(self.model_options.get("transformer_options") or {})
        }
        return new

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


@pytest.fixture(autouse=True)
def _clean_area_state():
    """AReA's seam state is process-global (the ref-counts and the runtimes)."""
    area._warned.clear()
    area._QSEAM.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    area._QSEAM["recorder"] = None
    area._QRUNTIME = None
    yield
    area._warned.clear()
    area._QSEAM.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    area._QSEAM["recorder"] = None
    area._QRUNTIME = None


def _model():
    return QwenImage21Transformer2DModel(layers=1, seed=11)


def _fake_backend(symbol):
    """A backend module that IS comfy's, by name, but binds a foreign symbol.

    The mock harness registers its own module under comfy's name; this one is
    the opposite case — the module's own ``__name__`` is comfy's, so the
    fused-kernel check applies to it.
    """
    module = types.ModuleType(QWEN21_SEAM_MODULE)
    module.optimized_attention = symbol
    return module


def test_a_fused_seam_is_refused():
    """N5 — the seam symbol must be comfy's CURRENT attention implementation."""
    def fused(q, k, v, heads, **kw):  # pragma: no cover - never called
        return None

    backend = _fake_backend(fused)
    saved = sys.modules.get(QWEN21_SEAM_MODULE)
    sys.modules[QWEN21_SEAM_MODULE] = backend
    try:
        with pytest.raises(ValueError, match="fused or quantized"):
            area._qwen21_install_seam(_Patcher(_model()), _model())
    finally:
        if saved is None:
            sys.modules.pop(QWEN21_SEAM_MODULE, None)
        else:
            sys.modules[QWEN21_SEAM_MODULE] = saved


def test_a_foreign_wrapper_is_refused():
    """N10 — SPA's marker on the bound symbol is a refusal, not a silent wrap."""
    with qwen21_backend_module() as backend:
        def foreign(q, k, v, heads, **kw):  # pragma: no cover - never called
            return None

        foreign._spa_wrapper = True
        backend.optimized_attention = foreign
        with pytest.raises(ValueError, match="SPA"):
            area._qwen21_install_seam(_Patcher(_model()), _model())


def test_a_contradicting_model_type_is_refused():
    """N6 — ``model_type`` that contradicts the backbone is a loud refusal."""
    dm = _model()
    with pytest.raises(ValueError, match="Qwen-Image-2.1"):
        resolve_area_arch(dm, "flux")
    with pytest.raises(ValueError, match="Qwen-Image-2.1"):
        resolve_area_arch(dm, "krea2")


def test_a_missing_pe_embedder_is_refused():
    """N8 — without the model's own embedder there is no rope table to invert."""
    dm = _model()
    dm.pe_embedder = None
    with qwen21_backend_module() as backend:
        pristine = backend.optimized_attention
        with pytest.raises(ValueError, match="pe_embedder"):
            area._qwen21_install_seam(_Patcher(dm), dm)
        # the refusal happened BEFORE the rebind: the module is untouched
        assert backend.optimized_attention is pristine


def test_the_patches_are_refused():
    """N11 — a patched forward runs plain attention, once, loudly."""
    dm = _model()
    rt = area._AreaRuntime(area.AreaConfig(native=4), dm, area.QWEN21_PROFILE)
    table = torch.linspace(1.0, 0.0, 21)
    params = {
        "input": torch.randn(1, dm.channels, 8, 8),
        "c": {
            "c_crossattn": torch.randn(1, 3, dm.txt_in.in_features),
            "transformer_options": {
                "sample_sigmas": table,
                "sigmas": table[0],
                "patches": {"attn1_patch": [object()]},
            },
        },
    }
    ctx = area._area_resolve_forward(rt, params)
    assert ctx.phase is None
    # the same forward without the patches resolves a phase
    params["c"]["transformer_options"].pop("patches")
    ctx = area._area_resolve_forward(rt, params)
    assert ctx.phase is not None


def test_the_prefix_cache_is_switched_off():
    """N12 — the clone carries the cache switch; the caller's patcher does not."""
    with qwen21_backend_module():
        patcher = _Patcher(_model())
        patched = area.apply_area_to_model(patcher, "qwen21", WIDTH, HEIGHT)
    assert patched.model_options["transformer_options"]["qwen_image21_cache"] == {
        "device": "off"
    }
    # the caller's own patcher is untouched
    assert "transformer_options" not in patcher.model_options or not (
        patcher.model_options.get("transformer_options") or {}
    ).get("qwen_image21_cache")


def test_a_refused_install_leaves_the_model_untouched():
    """A refused install rebinds NOTHING: the module keeps the pristine symbol."""
    def fused(q, k, v, heads, **kw):  # pragma: no cover - never called
        return None

    backend = _fake_backend(fused)
    saved = sys.modules.get(QWEN21_SEAM_MODULE)
    sys.modules[QWEN21_SEAM_MODULE] = backend
    try:
        with pytest.raises(ValueError):
            area._qwen21_install_seam(_Patcher(_model()), _model())
        assert backend.optimized_attention is fused
    finally:
        if saved is None:
            sys.modules.pop(QWEN21_SEAM_MODULE, None)
        else:
            sys.modules[QWEN21_SEAM_MODULE] = saved


def test_the_seam_is_live_after_the_install():
    """The install proves the seam is live before it claims the model."""
    with qwen21_backend_module() as backend:
        dm = _model()
        patched = area.apply_area_to_model(_Patcher(dm), "qwen21", WIDTH, HEIGHT)
        try:
            bound = getattr(backend, QWEN21_SEAM_SYMBOL)
            assert getattr(bound, "_area_wrapper", False)
            assert area._qwen21_seam_is_live(dm, bound)
            assert patched._area_seam == "qwen21"
        finally:
            area.restore_area_attention_hook(patched)


def test_text_segment_is_delegated_untouched():
    """N1 — the caption segment (a causal mask) is DELEGATED."""
    from src.area_qwen21 import qwen21_segment_eligible
    mask = torch.ones((3, 3), dtype=torch.bool).tril()
    assert qwen21_segment_eligible(mask, 3, 3, 3, 3, 4, 4) is False


def test_reference_chunk_is_delegated():
    """N2 — a reference chunk's key is longer than the segment geometry."""
    from src.area_qwen21 import qwen21_segment_eligible
    assert qwen21_segment_eligible(None, 9, 13 + 1024, 13 + 1024, 4, 3, 3) is False


def test_reference_latent_is_delegated():
    """N3 — a reference-latent forward is not one rectangular canvas."""
    from src.area_qwen21 import qwen21_layout
    with pytest.raises(ValueError, match="key"):
        qwen21_layout(4, 3, 3, 9, 13 + 1024)


def test_five_dim_latent_raises_at_install():
    """N4 — a 5-D (video) latent is refused LOUDLY, never silently.

    The latent is known only in the forward, so the refusal happens at the
    forward-resolve — the earliest point it can be detected at all — and
    BEFORE the seam sees a single tensor.
    """
    dm = _model()
    rt = area._AreaRuntime(area.AreaConfig(native=4), dm, area.QWEN21_PROFILE)
    params = {
        "input": torch.randn(1, dm.channels, 4, 4, 2),
        "c": {"c_crossattn": torch.randn(1, 3, dm.txt_in.in_features)},
    }
    with pytest.raises(ValueError, match="5-D"):
        area._area_resolve_forward(rt, params)


def test_fused_seam_raises_at_install():
    """N5 — a fused kernel is refused at install, and the module is untouched."""
    def fused(q, k, v, heads, **kw):  # pragma: no cover - never called
        return None

    backend = _fake_backend(fused)
    saved = sys.modules.get(QWEN21_SEAM_MODULE)
    sys.modules[QWEN21_SEAM_MODULE] = backend
    try:
        with pytest.raises(ValueError, match="fused or quantized"):
            area._qwen21_install_seam(_Patcher(_model()), _model())
        assert backend.optimized_attention is fused
    finally:
        if saved is None:
            sys.modules.pop(QWEN21_SEAM_MODULE, None)
        else:
            sys.modules[QWEN21_SEAM_MODULE] = saved


def test_contradicting_model_type_raises():
    """N6 — a contradicting ``model_type`` names both sides."""
    dm = _model()
    with pytest.raises(ValueError, match="Qwen-Image-2.1"):
        resolve_area_arch(dm, "flux")


def test_missing_pe_embedder_raises():
    """N8 — without the model's own embedder the install refuses."""
    dm = _model()
    dm.pe_embedder = None
    with qwen21_backend_module() as backend:
        pristine = backend.optimized_attention
        with pytest.raises(ValueError, match="pe_embedder"):
            area._qwen21_install_seam(_Patcher(dm), dm)
        assert backend.optimized_attention is pristine


def test_native_extent_is_delegated():
    """N7 — at or below the native extent AReA is a provable no-op."""
    dm = _model()
    rt = area._AreaRuntime(area.AreaConfig(native=64), dm, area.QWEN21_PROFILE)
    table = torch.linspace(1.0, 0.0, 21)
    params = {
        "input": torch.randn(1, dm.channels, 4, 4),
        "c": {
            "c_crossattn": torch.randn(1, 3, dm.txt_in.in_features),
            "transformer_options": {"sample_sigmas": table, "sigmas": table[0]},
        },
    }
    assert area._area_resolve_forward(rt, params).phase is None


def test_patches_are_refused():
    """N11 — a model-level patch is refused AT INSTALL.

    A patch that arrives per FORWARD (through ``transformer_options``) cannot
    be known at install time, so it is refused at forward time instead
    (``test_the_patches_are_refused``).
    """
    dm = _model()
    dm.attn1_patch = [object()]
    with qwen21_backend_module() as backend:
        pristine = backend.optimized_attention
        with pytest.raises(ValueError, match="attn1_patch"):
            area._qwen21_install_seam(_Patcher(dm), dm)
        assert backend.optimized_attention is pristine


def test_prefix_cache_is_disabled():
    """N12 — the clone carries the prefix-cache switch."""
    with qwen21_backend_module():
        patched = area.apply_area_to_model(_Patcher(_model()), "qwen21",
                                           WIDTH, HEIGHT)
    assert patched.model_options["transformer_options"]["qwen_image21_cache"] == {
        "device": "off"
    }


def test_unknown_head_count_is_delegated():
    """N13 — a foreign head count is DELEGATED; it never raises."""
    from src.area_qwen21 import qwen21_head_geometry, qwen21_segment_eligible
    x = torch.zeros(1, 5, 32)
    assert qwen21_head_geometry(x, 3) is None       # 32 does not split into 3
    assert qwen21_head_geometry(x, 4) == 8
    assert qwen21_segment_eligible(None, 9, 13, 13, 4, 3, 3,
                                   heads=16, dm_heads=32) is False


def test_non_trailing_image_span_is_delegated():
    """N14 — the image span must be the trailing ``h*w`` rows of the key."""
    from src.area_qwen21 import qwen21_layout
    with pytest.raises(ValueError, match="key"):
        qwen21_layout(4, 3, 3, 9, 9)


def test_a_non_2_1_model_is_never_touched():
    """AReA never installs the 2.1 seam on another backbone (N6)."""
    class NotQwen21:
        patch_size = 2

    with qwen21_backend_module() as backend:
        pristine = backend.optimized_attention
        with pytest.raises(ValueError):
            resolve_area_arch(NotQwen21(), "qwen21")
        assert backend.optimized_attention is pristine


def test_foreign_wrapper_is_refused():
    """N10 — SPA's wrapper on the bound symbol is refused, module untouched."""
    with qwen21_backend_module() as backend:
        def foreign(q, k, v, heads, **kw):  # pragma: no cover - never called
            return None

        foreign._spa_wrapper = True
        backend.optimized_attention = foreign
        with pytest.raises(ValueError, match="SPA"):
            area._qwen21_install_seam(_Patcher(_model()), _model())
        assert backend.optimized_attention is foreign
