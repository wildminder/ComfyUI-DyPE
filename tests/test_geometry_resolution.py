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


def _make_dm(cls_name=None, **attrs):
    if cls_name:
        cls = type(cls_name, (), dict(attrs))
        return cls()
    return types.SimpleNamespace(**attrs)


class _Patcher:
    def __init__(self, dm):
        self.model = types.SimpleNamespace(diffusion_model=dm)

    def clone(self):
        return _Patcher(self.model.diffusion_model)


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
