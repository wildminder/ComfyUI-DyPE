"""S5 — the AReA architecture profiles, with Qwen-Image-2.1 among them.

The profile is the ONE place that knows which backbone a model is; the
backbone's geometry lives in its adapter (``src/area_qwen21.py`` for 2.1).
Qwen-Image-2.1 is a FOURTH backbone: 16 px per token (MEASURED — the latent
format downsamples 16x and 2.1 does not patchify, ``comfy/latent_formats.py:
957-960``), no sequence padding at all (``model.py:311-339``) and a
shape-aware bridge (the Krea-2 lesson: the paper's square ``(rho*w)^2``
bridge count over-provisions on a non-square canvas).

The contradictions are refusals, not silent mis-patches: a ``flux`` pin on a
2.1 model, a ``qwen21`` pin on a model that is not a 2.1 one.
"""
import pytest
import torch

from _qwen21_seam_fixtures import QwenImage21Transformer2DModel
from src.area_arch import (
    FLUX_PROFILE,
    KREA_PROFILE,
    NEXT_DIT_BACKBONE,
    QWEN21_PROFILE,
    ZIMAGE_PROFILE,
    resolve_area_arch,
)
from src.model_detect import resolve_model_type

pytestmark = pytest.mark.unit


class _Flux(torch.nn.Module):
    """A FLUX-shaped backbone: the ``pe_embedder`` probe is the detector's."""

    def __init__(self):
        super().__init__()
        self.pe_embedder = torch.nn.Identity()


class _Krea(torch.nn.Module):
    """A Krea-2 backbone — the class name IS the marker."""


_Krea.__name__ = "SingleStreamDiT"


class _ZImage(torch.nn.Module):
    """A Z-Image backbone — the class name plus the ``rope_embedder`` probe."""

    def __init__(self):
        super().__init__()
        self.rope_embedder = torch.nn.Identity()


_ZImage.__name__ = NEXT_DIT_BACKBONE


class TestTheQwen21Profile:
    def test_the_qwen21_profile_carries_the_measured_px_per_patch(self):
        assert QWEN21_PROFILE.model_type == "qwen21"
        assert QWEN21_PROFILE.unet_backbone == "QwenImage21Transformer2DModel"
        assert QWEN21_PROFILE.px_per_patch == 16
        assert QWEN21_PROFILE.seq_pad_multiple == 0
        assert QWEN21_PROFILE.native_tokens == 64
        assert QWEN21_PROFILE.bridge_shape_aware is True

    def test_the_qwen21_profile_resolves(self):
        model = QwenImage21Transformer2DModel()
        assert resolve_area_arch(model, "auto") is QWEN21_PROFILE
        assert resolve_area_arch(model, "qwen21") is QWEN21_PROFILE

    def test_a_contradictory_pin_is_refused(self):
        model = QwenImage21Transformer2DModel()
        with pytest.raises(ValueError, match="Qwen-Image-2.1"):
            resolve_area_arch(model, "flux")

    def test_a_qwen21_pin_on_another_backbone_is_refused(self):
        with pytest.raises(ValueError, match="qwen21"):
            resolve_area_arch(_Flux(), "qwen21")

class TestTheOtherProfiles:
    def test_flux_still_resolves(self):
        assert resolve_area_arch(_Flux(), "auto") is FLUX_PROFILE

    def test_krea2_still_resolves(self):
        assert resolve_area_arch(_Krea(), "auto") is KREA_PROFILE

    def test_zimage_still_resolves(self):
        assert resolve_area_arch(_ZImage(), "auto") is ZIMAGE_PROFILE


def test_the_qwen21_profile_is_not_confused_with_qwen_image_1():
    """A 1.0-shaped model never resolves to the 2.1 profile.

    The trap is a ``"QwenImage" in class_name`` probe: 2.1's class name
    contains it, but so does 1.0's — so the 2.1 branch is matched EXACTLY.
    """
    from src.area_arch import QWEN21_PROFILE, resolve_area_arch

    class QwenImageTransformer2DModel:  # Qwen-Image 1.0's class name
        patch_size = 2

    try:
        profile = resolve_area_arch(QwenImageTransformer2DModel(), "auto")
    except ValueError:
        return  # refused outright — which is not the 2.1 profile either
    assert profile is not QWEN21_PROFILE


def test_the_bridge_count_is_shape_aware():
    """The 2.1 profile counts the bridge by AREA, not by ``w^2``."""
    from src.area_arch import QWEN21_PROFILE
    assert QWEN21_PROFILE.bridge_shape_aware is True


def test_the_existing_profiles_are_unchanged():
    """The FLUX / Z-Image / Krea-2 profiles keep their exact numbers."""
    from src.area_arch import FLUX_PROFILE, KREA_PROFILE, ZIMAGE_PROFILE
    for profile in (FLUX_PROFILE, ZIMAGE_PROFILE, KREA_PROFILE):
        assert profile.px_per_patch == 16
        assert profile.native_tokens == 64
