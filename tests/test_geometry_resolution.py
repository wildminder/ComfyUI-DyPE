"""W4.4 — ModelGeometry resolution characterization (IMP-007 safety net).

``resolve_model_geometry`` replaces two byte-identical inline blocks in
``apply_dype_to_model`` / ``apply_sega_to_model``.  These tests pin the exact
field values the OLD code produced for each mock shape, so the refactor is
provably behaviour-preserving:

* FLUX default: patch_size=2, derived_base_patches=(1024//8)//2 = 64,
  derived_base_seq_len = 64*64 = 4096.
* Z-Image with axes_lens=[128,64,64]: base h/w tokens (64,64),
  derived_base_patches = max(64,64) = 64, seq_len = 64*64 = 4096.
* Nunchaku: patch_size read from ``model.config.patch_size``.
* Anima: patch_size from ``patch_spatial``.
* Missing patch_size attribute: warning + default 2.

Markers: @pytest.mark.unit
"""

import types

import pytest

from src.patch_utils import resolve_model_geometry

from _qwen21_fixtures import make_qwen10_dm, make_qwen21_dm, make_qwen21_patcher


def _make_dm(cls_name=None, **attrs):
    if cls_name:
        cls = type(cls_name, (), dict(attrs))
        return cls()
    return types.SimpleNamespace(**attrs)


class _Patcher:
    def __init__(self, dm):
        self.model = types.SimpleNamespace(
            diffusion_model=dm, model_config=types.SimpleNamespace())

    def clone(self):
        new = _Patcher(self.model.diffusion_model)
        new.model.model_config = self.model.model_config
        new.model.model_sampling = getattr(self.model, "model_sampling", None)
        return new

    def add_object_patch(self, path, obj):
        self._object_patches = getattr(self, "_object_patches", {})
        self._object_patches[path] = obj


@pytest.mark.unit
class TestModelGeometryResolution:
    def test_flux_defaults(self):
        dm = _make_dm(None, patch_size=2,
                      pe_embedder=types.SimpleNamespace(theta=10000))
        geo = resolve_model_geometry(_Patcher(dm), "auto", base_resolution=1024)
        assert geo.detected == "flux"
        assert geo.patch_size == 2
        assert geo.base_patch_h_tokens is None
        assert geo.base_patch_w_tokens is None
        # The plan's documented value: (1024 // 8) // 2 == 64.
        assert geo.derived_base_patches == 64
        assert geo.derived_base_seq_len == 64 * 64

    def test_qwen_class_name(self):
        dm = _make_dm("QwenImageDiT", patch_size=2,
                      pe_embedder=types.SimpleNamespace(theta=10000))
        geo = resolve_model_geometry(_Patcher(dm), "auto")
        assert geo.detected == "qwen"
        assert geo.patch_size == 2
        assert geo.derived_base_patches == 64

    def test_zimage_axes_lens(self):
        dm = _make_dm(None, patch_size=2,
                      rope_embedder=types.SimpleNamespace(),
                      axes_lens=[128, 64, 64])
        geo = resolve_model_geometry(_Patcher(dm), "auto")
        assert geo.detected == "zimage"
        assert geo.base_patch_h_tokens == 64
        assert geo.base_patch_w_tokens == 64
        assert geo.derived_base_patches == 64
        assert geo.derived_base_seq_len == 64 * 64

    def test_zimage_without_axes_lens_falls_back(self):
        dm = _make_dm(None, patch_size=2,
                      rope_embedder=types.SimpleNamespace())
        geo = resolve_model_geometry(_Patcher(dm), "auto")
        assert geo.detected == "zimage"
        assert geo.base_patch_h_tokens is None
        assert geo.derived_base_patches == 64

    def test_nunchaku_reads_config_patch_size(self):
        inner = types.SimpleNamespace(
            config=types.SimpleNamespace(patch_size=4),
            pos_embed=types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56]),
        )
        dm = _make_dm(None, model=inner)
        geo = resolve_model_geometry(_Patcher(dm), "nunchaku")
        assert geo.detected == "nunchaku"
        assert geo.patch_size == 4

    def test_anima_reads_patch_spatial(self):
        cls = type("AnimaDIT", (), {
            "patch_spatial": 3,
            "pos_embedder": types.SimpleNamespace(dim_spatial_range=[0, 1, 2]),
        })
        geo = resolve_model_geometry(_Patcher(cls()), "anima")
        assert geo.detected == "anima"
        assert geo.patch_size == 3

    def test_missing_patch_size_defaults_to_two(self):
        # FLUX-shaped but no patch_size attr -> warning + default 2.
        dm = _make_dm(None, pe_embedder=types.SimpleNamespace(theta=10000))
        geo = resolve_model_geometry(_Patcher(dm), "flux")
        assert geo.patch_size == 2

    def test_custom_base_resolution(self):
        dm = _make_dm(None, patch_size=2,
                      pe_embedder=types.SimpleNamespace(theta=10000))
        geo = resolve_model_geometry(_Patcher(dm), "flux", base_resolution=2048)
        # (2048 // 8) // 2 == 128 patches; seq == patches^2.
        assert geo.derived_base_patches == 128
        assert geo.derived_base_seq_len == 128 * 128

    def test_krea2_detected_by_class_name(self):
        dm = _make_dm("SingleStreamDiT", patch_size=2,
                      pe_embedder=types.SimpleNamespace(theta=10000))
        geo = resolve_model_geometry(_Patcher(dm), "auto")
        assert geo.detected == "krea2"

    def test_geometry_is_immutable(self):
        dm = _make_dm(None, patch_size=2,
                      pe_embedder=types.SimpleNamespace(theta=10000))
        geo = resolve_model_geometry(_Patcher(dm), "flux")
        with pytest.raises(Exception):
            geo.patch_size = 3


# ---------------------------------------------------------------------------
# Qwen-Image-2.1 — 16x VAE, no patchify
#
# Historically 2.1 resolved to "qwen" and then read ``dm.patch_size`` from an
# attribute it does not have (AttributeError, swallowed, default 2) while the
# installers divided by a hardcoded 8.  The two errors cancelled:
# ``H/8/2 == H/16``.  These tests make the cancellation explicit so it cannot
# drift apart, and pin the negative controls that protect the other five
# architectures.
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQwen21Geometry:
    def test_qwen21_patch_size_is_one(self):
        """2.1 has no patchify — one transformer token per latent position."""
        geo = resolve_model_geometry(_Patcher(make_qwen21_dm()), "auto")
        assert geo.detected == "qwen21"
        assert geo.patch_size == 1

    def test_qwen21_latent_downscale_is_sixteen(self):
        geo = resolve_model_geometry(_Patcher(make_qwen21_dm()), "auto")
        assert geo.latent_downscale == 16

    def test_qwen21_base_patches_match_qwen_1_0(self):
        """THE cancellation pin: both architectures land on a 64-token base
        grid side at 1024, by two different routes."""
        geo21 = resolve_model_geometry(
            _Patcher(make_qwen21_dm()), "auto", base_resolution=1024)
        geo10 = resolve_model_geometry(
            _Patcher(make_qwen10_dm()), "auto", base_resolution=1024)
        assert geo21.derived_base_patches == 64            # (1024 // 16) // 1
        assert geo10.derived_base_patches == 64            # (1024 // 8) // 2
        assert geo21.derived_base_patches == geo10.derived_base_patches
        assert geo21.derived_base_seq_len == geo10.derived_base_seq_len

    def test_qwen21_custom_base_resolution(self):
        geo = resolve_model_geometry(
            _Patcher(make_qwen21_dm()), "auto", base_resolution=2048)
        assert geo.derived_base_patches == 128            # (2048 // 16) // 1
        assert geo.derived_base_seq_len == 128 * 128

    def test_no_warning_logged_for_qwen21_patch_size(self, caplog):
        """The explicit ``patch_size = 1`` branch exists so the
        "Could not read patch_size" warning must NOT fire for 2.1 — a
        warning there means the branch was bypassed."""
        import logging

        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            geo = resolve_model_geometry(_Patcher(make_qwen21_dm()), "auto")
        assert geo.patch_size == 1
        patch_size_warnings = [
            r for r in caplog.records
            if "Could not read patch_size" in r.getMessage()
        ]
        assert not patch_size_warnings, (
            "the qwen21 branch did not take effect: "
            f"{[r.getMessage() for r in patch_size_warnings]}"
        )

    def test_qwen21_image_seq_len_matches_pixel_count(self):
        """End-to-end: the schedule patch the installer applies for a
        2048x1024 target must be derived from a 128x64 LATENT grid, i.e.
        8192 tokens — recovered from the shift that was actually applied.

        The shift is linear in ``image_seq_len``, so inverting it recovers the
        token count the installer used without reaching into its internals.
        """
        from comfy import model_sampling

        from src.patch_utils import apply_dype_to_model

        base_shift, max_shift, base_resolution = 0.5, 1.15, 1024
        width, height = 1024, 2048

        patcher = make_qwen21_patcher()
        # ``_should_patch_schedule`` needs a live ModelSamplingFlux instance
        # (or the qwen flags, which land in a later batch).
        patcher.model.model_sampling = model_sampling.ModelSamplingFlux(
            patcher.model.model_config)

        out = apply_dype_to_model(
            patcher, "auto", width, height, "ntk", False,
            enable_dype=True, dype_scale=1.0, dype_exponent=1.0,
            base_shift=base_shift, max_shift=max_shift,
            base_resolution=base_resolution,
        )
        sampler = out._object_patches["model_sampling"]

        base_seq_len = (base_resolution // 16) // 1
        base_seq_len = base_seq_len * base_seq_len
        max_seq_len = base_seq_len * 4
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        intercept = base_shift - slope * base_seq_len
        image_seq_len = (sampler._shift - intercept) / slope

        assert round(image_seq_len) == (height // 16) * (width // 16) == 128 * 64


# ---------------------------------------------------------------------------
# Negative controls — the other five architectures must be untouched
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGeometryNegativeControls:
    def test_flux_patch_size_still_two(self):
        dm = _make_dm(None, patch_size=2,
                      pe_embedder=types.SimpleNamespace(theta=10000))
        geo = resolve_model_geometry(_Patcher(dm), "flux")
        assert geo.patch_size == 2
        assert geo.latent_downscale == 8

    def test_zimage_downscale_still_eight(self):
        dm = _make_dm(None, patch_size=2, rope_embedder=types.SimpleNamespace(),
                      axes_lens=[128, 64, 64])
        geo = resolve_model_geometry(_Patcher(dm), "zimage")
        assert geo.latent_downscale == 8
        assert geo.derived_base_patches == 64

    def test_qwen_1_0_downscale_still_eight(self):
        geo = resolve_model_geometry(_Patcher(make_qwen10_dm()), "auto")
        assert geo.detected == "qwen"
        assert geo.latent_downscale == 8
        assert geo.patch_size == 2

    @pytest.mark.parametrize("requested", [
        "flux", "nunchaku", "zimage", "anima", "krea2",
    ])
    def test_every_other_backend_keeps_downscale_eight(self, requested):
        dm = _make_dm(None, patch_size=2,
                      pe_embedder=types.SimpleNamespace(theta=10000))
        geo = resolve_model_geometry(_Patcher(dm), requested)
        assert geo.latent_downscale == 8, (
            f"{requested} unexpectedly got a {geo.latent_downscale}x VAE"
        )

    def test_latent_downscale_defaults_to_eight(self):
        """The field is appended LAST WITH A DEFAULT so existing positional and
        keyword construction keeps working."""
        from src.patch_utils import ModelGeometry

        geo = ModelGeometry(
            patch_size=2, base_patch_h_tokens=None, base_patch_w_tokens=None,
            derived_base_patches=64, derived_base_seq_len=4096,
            detected="flux",
        )
        assert geo.latent_downscale == 8
