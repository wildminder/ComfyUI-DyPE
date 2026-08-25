"""Tests for Qwen2D VAE architecture and patching."""
import pathlib

import pytest


@pytest.mark.unit
class TestQwen2DVAEArchitecture:
    """Tests for Qwen2D VAE architecture (src/qwen2d_vae.py)."""

    def _read_source(self):
        return (pathlib.Path(__file__).parent.parent / "src" / "qwen2d_vae.py").read_text(encoding="utf-8")

    def test_qwen2d_vae_class_exists(self):
        content = self._read_source()
        assert "class Qwen2DVAE" in content

    def test_default_config_exists(self):
        content = self._read_source()
        assert "QWEN2D_DEFAULT_CONFIG" in content

    def test_default_config_z_dim_16(self):
        content = self._read_source()
        assert '"z_dim": 16' in content

    def test_has_encode_method(self):
        content = self._read_source()
        assert "def encode" in content

    def test_has_decode_method(self):
        content = self._read_source()
        assert "def decode" in content

    def test_has_flatten_frames(self):
        content = self._read_source()
        assert "_flatten_frames" in content

    def test_has_restore_frames(self):
        content = self._read_source()
        assert "_restore_frames" in content

    def test_alias_autoencoder_exists(self):
        content = self._read_source()
        assert "AutoencoderKLQwenImage2D" in content


@pytest.mark.unit
class TestQwen2DVAEPatching:
    """Tests for Qwen2D VAE patching module (src/qwen2d_vae_patch.py)."""

    def _read_source(self):
        return (pathlib.Path(__file__).parent.parent / "src" / "qwen2d_vae_patch.py").read_text(encoding="utf-8")

    def test_install_qwen2d_patch_exists(self):
        content = self._read_source()
        assert "def install_qwen2d_patch" in content

    def test_is_qwen2d_state_dict_exists(self):
        content = self._read_source()
        assert "def _is_qwen2d_state_dict" in content

    def test_init_qwen2d_vae_exists(self):
        content = self._read_source()
        assert "def _init_qwen2d_vae" in content

    def test_has_decode_method(self):
        content = self._read_source()
        assert "def _qwen2d_decode" in content

    def test_has_encode_method(self):
        content = self._read_source()
        assert "def _qwen2d_encode" in content

    def test_has_tiled_methods(self):
        content = self._read_source()
        assert "def _qwen2d_decode_tiled" in content
        assert "def _qwen2d_encode_tiled" in content

    def test_has_strip_singleton_temporal(self):
        content = self._read_source()
        assert "_strip_singleton_temporal" in content

    def test_has_flatten_temporal_batch(self):
        content = self._read_source()
        assert "_flatten_temporal_batch" in content

    def test_patches_vae_init(self):
        content = self._read_source()
        assert "comfy_sd.VAE.__init__" in content

    def test_sets_latent_dim_3(self):
        content = self._read_source()
        assert "vae.latent_dim = 3" in content

    def test_sets_not_video_true(self):
        content = self._read_source()
        assert "vae.not_video = True" in content

    def test_imports_qwen2d_vae(self):
        content = self._read_source()
        assert "from .qwen2d_vae import Qwen2DVAE" in content


@pytest.mark.unit
class TestQwen2DVAERegistration:
    """Tests for Qwen2D VAE patch registration in __init__.py."""

    def test_import_in_init(self):
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "install_qwen2d_patch" in content

    def test_on_load_calls_install(self):
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "async def on_load" in content
        assert "install_qwen2d_patch()" in content
