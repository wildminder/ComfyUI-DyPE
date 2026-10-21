"""S2 — the Qwen-Image-2.1 geometry constants, with their provenance labels.

The pixel factor is MEASURED (the latent format downsamples 16x and 2.1 does
not patchify on top of it — ``comfy/latent_formats.py:957-960``,
``comfy/ldm/qwen_image21/model.py:260``).  The trained extent in TOKENS is
ASSUMED: it is the pack's own hypothesis (``src/spa.py:86-91``) and is NOT
derivable from the source.  The labels are not decoration — the tests read
them, so an unlabelled constant fails the build (the Krea-2 rule,
``src/area_krea.py:135-146``).
"""
import pytest
import torch

from src.area_qwen21 import (
    QWEN21_HEADS,
    QWEN21_HEAD_DIM,
    QWEN21_LAYERS,
    QWEN21_NATIVE_NOTE,
    QWEN21_NATIVE_PROVENANCE,
    QWEN21_NATIVE_TOKENS,
    QWEN21_PATCH,
    QWEN21_PX_NOTE,
    QWEN21_PX_PER_PATCH,
    QWEN21_PX_PROVENANCE,
    QWEN21_ROPE_AXES,
    QWEN21_TRAINED_RESOLUTION_PX,
    QWEN21_VAE_DOWNSCALE,
    qwen21_latent_grid,
)

pytestmark = pytest.mark.unit


class TestConstants:
    def test_px_per_token_is_sixteen_and_is_measured(self):
        """16 px per token — read off the latent format, not guessed."""
        assert QWEN21_PX_PER_PATCH == 16
        assert QWEN21_PX_PROVENANCE == "MEASURED"
        assert "latent_formats.py:957-960" in QWEN21_PX_NOTE

    def test_native_token_count_is_labelled_assumed(self):
        """64 tokens (1024 px) — the pack's hypothesis, NOT derivable."""
        assert QWEN21_NATIVE_TOKENS == 64
        assert QWEN21_NATIVE_PROVENANCE == "ASSUMED"
        assert "src/spa.py:86-91" in QWEN21_NATIVE_NOTE

    def test_the_two_constants_agree_on_the_trained_canvas(self):
        """A trained extent in tokens and a pixel factor that disagree would
        mis-size every advisory, so the cross-check is a test."""
        assert QWEN21_NATIVE_TOKENS * QWEN21_PX_PER_PATCH == QWEN21_TRAINED_RESOLUTION_PX


class TestTheLatentGrid:
    def test_the_latent_grid_is_the_latent_shape(self):
        """2.1 does not patchify — the token grid IS the latent's (H, W)."""
        latent = torch.zeros(1, 64, 8, 10)
        assert qwen21_latent_grid(latent) == (8, 10)

    def test_a_five_dimensional_latent_is_refused(self):
        """2.1's format is 2-D (``latent_dimensions = 2``), so a 5-D latent is
        a foreign one — refused loudly, not silently mis-sized."""
        with pytest.raises(ValueError, match="4-D"):
            qwen21_latent_grid(torch.zeros(1, 64, 1, 8, 10))

    def test_a_non_tensor_is_refused(self):
        with pytest.raises(ValueError, match="tensor"):
            qwen21_latent_grid([1, 64, 8, 10])


def test_the_head_geometry_matches_the_model():
    """32 heads of 128 over 32 blocks (model.py:240-242)."""
    assert QWEN21_HEADS == 32
    assert QWEN21_HEAD_DIM == 128
    assert QWEN21_LAYERS == 32


def test_the_rope_axes_are_fifty_six_plus_fifty_six_plus_sixteen():
    """(16, 56, 56) — the narrow axis is the FIRST one (model.py:244)."""
    assert QWEN21_ROPE_AXES == (16, 56, 56)
    # the axes carry the head dim, so the table's frequency axis is 64
    assert sum(QWEN21_ROPE_AXES) == QWEN21_HEAD_DIM


def test_the_patch_size_is_one_and_the_vae_downscale_is_sixteen():
    """No patchify on top of a 16x VAE — one token per latent cell."""
    assert QWEN21_PATCH == 1
    assert QWEN21_VAE_DOWNSCALE == 16
    assert QWEN21_PX_PER_PATCH == QWEN21_VAE_DOWNSCALE * QWEN21_PATCH
