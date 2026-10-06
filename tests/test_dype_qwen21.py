"""Qwen-Image-2.1 wiring into the DyPE / SEGA installers (commit 2).

The adapters themselves are unit-tested in ``tests/test_models.py`` and
``tests/test_sega_models.py``.  What is pinned here is the *installer* half:

1. a ``qwen21``-detected model gets the fp32-preserving adapter class, not the
   1.0 one;
2. ``is_qwen21`` participates in the DyPE schedule-cache key, so a 2.1 install
   and a 1.0 install can never share one cache entry;
3. the noise schedule is patched for 2.1 exactly as for 1.0 — the ``is_qwen21``
   flag carries it there, because 2.1's live ``model_sampling`` is a
   discrete-flow object the ``ModelSamplingFlux`` isinstance test rejects.

The patcher stand-in comes from ``tests/_qwen21_fixtures.py`` (owned by the
detection/geometry batch) so both halves describe the same 2.1 model.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.qwen import PosEmbedQwen  # noqa: E402
from src.models.qwen21 import PosEmbedQwen21  # noqa: E402
from src.models.sega_qwen import SegAPosEmbedQwen  # noqa: E402
from src.models.sega_qwen21 import SegAPosEmbedQwen21  # noqa: E402
from src.patch_utils import (  # noqa: E402
    _DYPE_PARAMS_ATTR,
    apply_dype_to_model,
    apply_sega_to_model,
)
from _qwen21_fixtures import (  # noqa: E402
    Qwen21Patcher,
    make_qwen10_dm,
    make_qwen21_dm,
)

#: Slot layout of the DyPE schedule-cache key.  The exact index of
#: ``is_qwen21`` is part of the contract with the test below.
_CACHE_KEY_FIELDS = (
    "width", "height", "base_shift", "max_shift", "method",
    "yarn_alt_scaling", "base_resolution", "dype_start_sigma",
    "is_nunchaku", "is_qwen", "is_qwen21", "is_z_image", "is_anima",
)
_IS_QWEN21_SLOT = _CACHE_KEY_FIELDS.index("is_qwen21")

_INSTALL_KWARGS = dict(method="yarn", yarn_alt_scaling=False, enable_dype=True,
                       dype_scale=2.0, dype_exponent=2.0, base_shift=0.5,
                       max_shift=1.15)


class _CacheAwarePatcher(Qwen21Patcher):
    """The fixture patcher, plus a schedule-cache attribute that survives clone().

    ``Qwen21Patcher.clone`` rebuilds from scratch and drops any attribute the
    installer stored on it, so the ``enable_dype=False`` restore branch — which
    only fires when a previous install left an entry — would be unreachable.
    """

    def clone(self):
        new = super().clone()
        if hasattr(self, _DYPE_PARAMS_ATTR):
            setattr(new, _DYPE_PARAMS_ATTR, getattr(self, _DYPE_PARAMS_ATTR))
        return new


def _patcher(dm=None):
    """A 2.1 (or supplied) patcher with a model_config ModelSamplingFlux accepts.

    ``tests/_qwen21_fixtures.py`` carries a bare ``SimpleNamespace`` config;
    ``ModelSamplingFlux.__init__`` reads ``model_config.sampling_settings`` and
    would raise AttributeError (swallowed by the installer's except clause),
    making every schedule assertion vacuous.  The config is only needed on the
    paths that DO patch the schedule.
    """
    patcher = _CacheAwarePatcher(dm if dm is not None else make_qwen21_dm())
    patcher.model.model_config = types.SimpleNamespace(
        sampling_settings={"shift": 1.15})
    return patcher


def _dype(model_type="qwen21", dm=None, **overrides):
    kwargs = {**_INSTALL_KWARGS, **overrides}
    return apply_dype_to_model(_patcher(dm), model_type, 1024, 1024, **kwargs)


def _sega(model_type="qwen21", dm=None, **overrides):
    return apply_sega_to_model(_patcher(dm), model_type, 1024, 1024, **overrides)


def _dype_1_0(**overrides):
    """Same installer against a 1.0-shaped model (the negative control)."""
    kwargs = {**_INSTALL_KWARGS, **overrides}
    return apply_dype_to_model(_patcher(make_qwen10_dm()), "qwen", 1024, 1024,
                               **kwargs)


def _repatch(installed):
    """Re-wrap an installer result so a second install sees what it left.

    ``apply_*_to_model`` returns a clone, so anything the installer stored
    (the schedule cache entry) lives on the result, not on the input patcher.
    """
    patcher = _patcher(installed.model.diffusion_model)
    patcher.model.model_sampling = installed.model.model_sampling
    if hasattr(installed, _DYPE_PARAMS_ATTR):
        setattr(patcher, _DYPE_PARAMS_ATTR, getattr(installed, _DYPE_PARAMS_ATTR))
    return patcher


def _installed(result):
    return result._object_patches["diffusion_model.pe_embedder"]


@pytest.mark.unit
class TestDypeQwen21Wiring:
    def test_installs_the_fp32_adapter(self):
        embedder = _installed(_dype())
        assert type(embedder) is PosEmbedQwen21

    def test_qwen_1_0_still_installs_the_bf16_capable_adapter(self):
        """Negative control: the 2.1 adapter must not leak into 1.0."""
        embedder = _installed(_dype_1_0())
        assert type(embedder) is PosEmbedQwen
        assert not isinstance(embedder, PosEmbedQwen21)

    def test_installed_embedder_output_is_fp32(self):
        out = _installed(_dype())(_image_ids(16))
        assert out.dtype is torch.float32
        assert out.shape == (1, 1, 16 * 16, 64, 2, 2)

    def test_installed_embedder_computes_frequencies_in_fp32(self, monkeypatch):
        """The frequency dtype the INSTALLED adapter uses is pinned by spying.

        ``freqs_dtype`` does not survive into the output dtype (rope.py builds
        its arange in it but every downstream op promotes back to fp32), so the
        argument itself is the only observable.
        """
        seen = []
        original = PosEmbedQwen21.get_components

        def spy(self, pos, freqs_dtype):
            seen.append(freqs_dtype)
            return original(self, pos, freqs_dtype)

        monkeypatch.setattr(PosEmbedQwen21, "get_components", spy)
        _installed(_dype())(_image_ids(8))
        assert seen == [torch.float32]

    def test_pe_embedder_patch_path_unchanged(self):
        """2.1 keeps EmbedND at ``pe_embedder`` — the default patch path."""
        assert "diffusion_model.pe_embedder" in _dype()._object_patches

    def test_unet_wrapper_is_installed(self):
        assert _dype()._unet_wrapper is not None


@pytest.mark.unit
class TestSegaQwen21Wiring:
    def test_installs_the_fp32_adapter(self):
        embedder = _installed(_sega())
        assert type(embedder) is SegAPosEmbedQwen21

    def test_qwen_1_0_still_installs_the_1_0_adapter(self):
        """Negative control."""
        embedder = _installed(_sega(model_type="qwen", dm=make_qwen10_dm()))
        assert type(embedder) is SegAPosEmbedQwen
        assert not isinstance(embedder, SegAPosEmbedQwen21)

    def test_installed_embedder_output_is_fp32(self):
        out = _installed(_sega())(_image_ids(16).squeeze(0))
        assert out.dtype is torch.float32
        assert out.shape == (16 * 16, 1, 64, 2, 2)

    def test_spectral_data_still_reaches_the_embedder(self):
        """The SEGA wrapper must still drive set_spectral_data for 2.1."""
        result = _sega()
        embedder = _installed(result)
        result._unet_wrapper(lambda x, t, **k: x, {
            "input": torch.randn(1, 64, 64, 64),  # 16x VAE → 64 latents @ 1024px
            "timestep": torch.tensor([0.5]),
            "c": {},
        })
        assert embedder._energy_profile_h is not None


@pytest.mark.unit
class TestQwen21CacheKey:
    def test_is_qwen21_is_part_of_the_key(self):
        """The key carries the flag, at a fixed slot, on the patched model."""
        key = getattr(_dype(), _DYPE_PARAMS_ATTR)
        assert len(key) == len(_CACHE_KEY_FIELDS)
        assert key[_IS_QWEN21_SLOT] is True
        assert key[_CACHE_KEY_FIELDS.index("is_qwen")] is False

    def test_a_1_0_entry_does_not_suppress_a_2_1_install(self):
        """The hazard the extra slot exists for.

        Same 1.0-class diffusion model (so the requested model_type is what
        selects the family), installed as 2.1 first and then as 1.0.  Without
        ``is_qwen21`` in the key the second install would see an identical
        entry, skip the schedule patch, and silently keep the 1.0 shift.
        """
        patcher = _patcher(make_qwen10_dm())
        as_2_1 = apply_dype_to_model(patcher, "qwen21", 1024, 1024,
                                     **_INSTALL_KWARGS)
        assert getattr(as_2_1, _DYPE_PARAMS_ATTR)[_IS_QWEN21_SLOT] is True

        result = apply_dype_to_model(_repatch(as_2_1), "qwen", 1024, 1024,
                                     **_INSTALL_KWARGS)
        assert "model_sampling" in result._object_patches
        assert getattr(result, _DYPE_PARAMS_ATTR)[_IS_QWEN21_SLOT] is False

    def test_reinstalling_the_same_family_is_suppressed(self):
        """The cache still works: an identical second install does not repatch."""
        first = _dype()
        patcher = _repatch(first)
        patcher._object_patches.pop("model_sampling", None)
        result = apply_dype_to_model(patcher, "qwen21", 1024, 1024,
                                     **_INSTALL_KWARGS)
        assert "model_sampling" not in result._object_patches


@pytest.mark.unit
class TestQwen21SchedulePatched:
    """2.1 follows 1.0 down the schedule path (16x VAE + patch_size 1)."""

    def test_dype_patches_model_sampling(self):
        assert "model_sampling" in _dype()._object_patches

    def test_sega_patches_model_sampling(self):
        assert "model_sampling" in _sega()._object_patches

    def test_patched_without_a_flux_live_sampling(self):
        """The flag is what carries 2.1 here, not the live attribute.

        2.1's native sampling is a discrete-flow shift, so the isinstance test
        would leave it unpatched while 1.0 is patched — hence ``is_qwen21``.
        """
        from comfy import model_sampling as comfy_ms

        class DiscreteFlowLike:
            """Not a ModelSamplingFlux — 2.1's native sampling shape."""

            sigma_max = types.SimpleNamespace(item=lambda: 1.0)

        patcher = _patcher()
        patcher.model.model_sampling = DiscreteFlowLike()
        assert not isinstance(patcher.model.model_sampling, comfy_ms.ModelSamplingFlux)
        result = apply_dype_to_model(patcher, "qwen21", 1024, 1024,
                                     **_INSTALL_KWARGS)
        assert "model_sampling" in result._object_patches

    def test_shift_matches_qwen_1_0_at_the_same_resolution(self):
        """The 16x / patch_size=1 geometry reproduces 1.0's token count.

        Both families then land on the same shift, so the geometry swap moves
        nothing about the schedule.
        """
        shift_2_1 = _dype()._object_patches["model_sampling"]._shift
        shift_1_0 = _dype_1_0()._object_patches["model_sampling"]._shift
        assert shift_2_1 == pytest.approx(shift_1_0)

    def test_disabled_dype_does_not_restore_a_flux_sampler(self):
        """``enable_dype=False`` with no prior entry must install nothing."""
        assert "model_sampling" not in _dype(enable_dype=False)._object_patches

    def test_qwen_1_0_still_patches_the_schedule(self):
        """Negative control."""
        assert "model_sampling" in _dype_1_0()._object_patches

    def test_qwen_1_0_still_restores_the_default_sampler_when_disabled(self):
        patcher = _repatch(_dype_1_0())
        result = apply_dype_to_model(patcher, "qwen", 1024, 1024,
                                     **{**_INSTALL_KWARGS, "enable_dype": False})
        assert "model_sampling" in result._object_patches


def _image_ids(n):
    """(1, n*n, 3) image-token position ids, matching what 2.1 builds."""
    grid = torch.arange(n, dtype=torch.float32)
    h = grid.unsqueeze(1).expand(n, n).reshape(-1)
    w = grid.unsqueeze(0).expand(n, n).reshape(-1)
    seq = torch.arange(n * n, dtype=torch.float32)
    return torch.stack([seq, h, w], dim=-1).unsqueeze(0)