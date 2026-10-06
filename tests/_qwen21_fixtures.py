"""Shared fixtures for Qwen-Image-2.1 tests.

Qwen-Image-2.1 ships as its own module ``comfy/ldm/qwen_image21/model.py``
with class ``QwenImage21Transformer2DModel``.  Three properties of that class
drive everything the pack has to special-case, and all three are re-stated here
so a single fixture can pin them:

1. The class name CONTAINS the substring ``"QwenImage"``.  Without an explicit
   class-name check the generic probe would claim 2.1 for Qwen-Image 1.0.
2. There is **no** ``patch_size`` attribute and no patchify — one transformer
   token per latent position.
3. Its position embedder is the very same ``comfy.ldm.flux.layers.EmbedND``
   class 1.0 uses, so it exposes ``theta`` and ``axes_dim`` but no ``thetas``.

:func:`make_qwen21_dm` therefore builds a REAL (mutable) class instance — a
``SimpleNamespace`` is fine too, but a real class is required for class-name
detection and mirrors what actually runs in ComfyUI.  ``tests/test_qwen21_fixtures.py``
asserts property 2 explicitly so an upstream change that adds ``patch_size``
fails here rather than silently flipping the geometry branch.
"""

from __future__ import annotations

import types

#: The class name of the Qwen-Image-2.1 transformer in core ComfyUI.
QWEN21_CLASS_NAME = "QwenImage21Transformer2DModel"

#: The class name of the Qwen-Image 1.0 transformer (must stay "qwen").
QWEN10_CLASS_NAME = "QwenImageTransformer2DModel"


def make_qwen21_dm(cls_name: str = QWEN21_CLASS_NAME, **attrs):
    """A Qwen-Image-2.1-shaped diffusion model for detection/geometry tests.

    Exposes ``pe_embedder(theta=10000, axes_dim=[16, 56, 56])`` like the real
    model and — deliberately — **no** ``patch_size``.  Extra keyword arguments
    override or extend the shape.
    """
    shape = {
        "pe_embedder": types.SimpleNamespace(
            dim=64, theta=10000, axes_dim=[16, 56, 56],
        ),
    }
    shape.update(attrs)
    return type(cls_name, (), shape)()


def make_qwen10_dm(**attrs):
    """A Qwen-Image 1.0-shaped diffusion model (has ``patch_size == 2``)."""
    shape = {
        "patch_size": 2,
        "pe_embedder": types.SimpleNamespace(
            dim=64, theta=10000, axes_dim=[16, 56, 56],
        ),
    }
    shape.update(attrs)
    return type(QWEN10_CLASS_NAME, (), shape)()


class Qwen21Patcher:
    """Minimal ``ModelPatcher`` stand-in carrying a 2.1 diffusion model.

    Mirrors the patcher used by ``tests/test_model_detection_parity.py``:
    ``model_sampling`` lives UNDER ``model`` (both installers read
    ``m.model.model_sampling.sigma_max``) and ``clone()`` returns a fresh
    patcher over the same diffusion model.
    """

    def __init__(self, dm):
        self.model = types.SimpleNamespace(
            diffusion_model=dm,
            model_config=types.SimpleNamespace(),
            model_sampling=types.SimpleNamespace(
                sigma_max=types.SimpleNamespace(item=lambda: 1.0)),
        )
        self._object_patches = {}
        self._unet_wrapper = None
        self._spa_installed = None
        self._spa_orig_optimized_attention = None
        self._hrdit_consumers = None
        self._hap_ctx = None

    def clone(self):
        new = Qwen21Patcher(self.model.diffusion_model)
        new.model.model_config = self.model.model_config
        new.model.model_sampling = self.model.model_sampling
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def make_qwen21_patcher(dm=None):
    """A :class:`Qwen21Patcher` over a default 2.1 dm (or one supplied)."""
    return Qwen21Patcher(dm if dm is not None else make_qwen21_dm())