"""Qwen-Image-2.1 fixture shape guards.

The 2.1 geometry branch in ``src/patch_utils.py`` exists because
``QwenImage21Transformer2DModel`` has **no** ``patch_size`` (there is no
patchify) and uses a **16x** VAE.  These tests pin the premises that justify
that branch.  If core ComfyUI ever adds ``patch_size`` to the 2.1 class, the
first test fails loudly and the geometry branch must be revisited rather than
silently shadowed by the generic ``dm.patch_size`` probe.

Markers: @pytest.mark.unit
"""

import pytest

from _qwen21_fixtures import (
    QWEN10_CLASS_NAME,
    QWEN21_CLASS_NAME,
    make_qwen10_dm,
    make_qwen21_dm,
    make_qwen21_patcher,
)


@pytest.mark.unit
class TestQwen21FixtureShape:
    def test_qwen21_dm_has_no_patch_size(self):
        """The premise of the ``patch_size = 1`` branch (no patchify)."""
        dm = make_qwen21_dm()
        assert not hasattr(dm, "patch_size"), (
            "Qwen-Image-2.1 gained a patch_size attribute upstream — the "
            "explicit patch_size=1 / latent_downscale=16 geometry branch must "
            "be revisited before it silently shadows the generic probe."
        )

    def test_qwen21_dm_class_name(self):
        dm = make_qwen21_dm()
        assert type(dm).__name__ == QWEN21_CLASS_NAME
        assert "QwenImage" in QWEN21_CLASS_NAME, (
            "this is the substring collision the explicit class-name check "
            "exists to defeat — if upstream ever renames the class, the "
            "detector's class-name check must be updated with it."
        )

    def test_qwen21_pe_embedder_shape(self):
        pe = make_qwen21_dm().pe_embedder
        assert pe.theta == 10000
        assert list(pe.axes_dim) == [16, 56, 56]
        assert not hasattr(pe, "thetas"), (
            "2.1's EmbedND has no multi-theta attribute (1.0's family does); "
            "an embedder that grows .thetas needs the adapters re-checked."
        )

    def test_qwen10_dm_has_patch_size_two(self):
        dm = make_qwen10_dm()
        assert dm.patch_size == 2
        assert type(dm).__name__ == QWEN10_CLASS_NAME

    def test_patcher_carries_model_sampling_under_model(self):
        """Both installers read ``m.model.model_sampling.sigma_max``."""
        p = make_qwen21_patcher()
        assert p.model.model_sampling.sigma_max.item() == 1.0

    def test_patcher_clone_is_independent(self):
        p = make_qwen21_patcher()
        p.add_object_patch("a", 1)
        c = p.clone()
        c.add_object_patch("b", 2)
        assert "b" in c._object_patches
        assert "b" not in p._object_patches
        assert c.model.diffusion_model is p.model.diffusion_model