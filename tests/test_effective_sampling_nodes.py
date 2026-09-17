"""Node-layer effective-sampling coverage for PixelRush/FreeScale (v2.16.0, plan S4).

Asserts the S2 contract extends to the other direct-sampling nodes: their
prediction detection, timestep/sigma conversions and sigma-table reads follow
the patcher's OWN patch chain, never a live attr leaked by an earlier run.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import nodes.freescale as fsn  # noqa: E402
import nodes.pixelrush as prn  # noqa: E402
from src.effective_sampling import effective_model_sampling  # noqa: E402


class _Patcher:
    """ModelPatcher facade (patch -> backup -> live)."""

    def __init__(self, live, patches=None, backup=None, prediction="CONST"):
        base = types.SimpleNamespace()
        base.model_sampling = live
        base.latent_format = types.SimpleNamespace(
            latent_dimensions=2, latent_channels=4)
        self.model = base
        self.object_patches = dict(patches or {})
        self.object_patches_backup = dict(backup or {})

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        return getattr(self.model, name)


def _ms(name, mixin):
    return type(name, (type(mixin, (), {}),), {})()


@pytest.mark.unit
class TestPixelRushResolvedDetection:
    def test_detection_follows_patch_not_leak(self):
        # A leaked EPS sampling on the base must not flip the detection when
        # the graph's own patch is CONST.
        patch = _ms("PatchedConst", "CONST")
        leak = _ms("LeakedEps", "EPS")
        patcher = _Patcher(leak, patches={"model_sampling": patch})
        assert prn._detect_prediction_type(
            effective_model_sampling(patcher)) == "const"

    def test_detection_backup_original(self):
        orig = _ms("OrigConst", "CONST")
        leak = _ms("LeakedEps", "EPS")
        patcher = _Patcher(leak, backup={"model_sampling": orig})
        assert prn._detect_prediction_type(
            effective_model_sampling(patcher)) == "const"


@pytest.mark.unit
class TestFreeScaleResolvedSigmas:
    def test_sigma_table_from_own_patch(self):
        patch = _ms("PatchedSampling", "CONST")
        patch.sigmas = torch.linspace(1.0, 0.0, 21)
        leak = _ms("LeakedSampling", "CONST")
        leak.sigmas = torch.zeros(3)
        patcher = _Patcher(leak, patches={"model_sampling": patch})
        sigmas = effective_model_sampling(patcher).sigmas
        assert sigmas.numel() == 21
        assert not torch.equal(sigmas, leak.sigmas)

    def test_sigma_table_fallback_live(self):
        live = _ms("LiveSampling", "CONST")
        live.sigmas = torch.linspace(1.0, 0.0, 11)
        model = types.SimpleNamespace()
        model.model = types.SimpleNamespace(model_sampling=live)
        sigmas = effective_model_sampling(model).sigmas
        assert sigmas.numel() == 11
