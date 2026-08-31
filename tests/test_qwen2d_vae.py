"""Tests for Qwen2D VAE architecture and patching."""
import os
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

    def test_disabled_by_default_gate_exists(self):
        """v2.8.3: the patch must be opt-in via DYPE_ENABLE_QWEN2D_VAE."""
        content = self._read_source()
        assert "DYPE_ENABLE_QWEN2D_VAE" in content
        assert "def _qwen2d_patch_enabled" in content

    def test_install_gated_on_env_var(self):
        """install_qwen2d_patch() returns early (no patching) unless the
        opt-in env var is set — the default-off regression guard."""
        import importlib
        import sys
        import types

        def _fake_mod(name, **attrs):
            # Additive only: never replace the root conftest's comfy mocks.
            if name in sys.modules:
                return sys.modules[name]
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
            # Wire the attribute onto the parent module (manual sys.modules
            # registration skips the import system's parent-attr step).
            parent, _, child = name.rpartition(".")
            if parent and parent in sys.modules:
                setattr(sys.modules[parent], child, m)
            return m

        # Minimal fake comfy chain so qwen2d_vae_patch imports standalone.
        # (Pre-registering each full submodule name in sys.modules makes the
        # imports resolve without the parent being a real package.)
        _fake_mod("comfy.model_management",
                  is_amd=lambda: False, dtype_size=lambda d: 4,
                  OOM_EXCEPTION=RuntimeError)
        _fake_mod("comfy.utils")
        _fake_mod("comfy.ops", disable_weight_init=object)
        _fake_mod("comfy.ldm")
        _fake_mod("comfy.ldm.modules")
        _fake_mod("comfy.ldm.modules.diffusionmodules")
        _fake_mod("comfy.ldm.modules.diffusionmodules.model",
                  vae_attention=lambda *a, **k: None)
        sd_mod = _fake_mod("comfy.sd", VAE=type("VAE", (), {}))
        # ("comfy" and "comfy.model_patcher" already exist via the root
        # conftest mock.)

        mod = importlib.import_module("src.qwen2d_vae_patch")

        # Default: env var unset -> no install.
        os.environ.pop("DYPE_ENABLE_QWEN2D_VAE", None)
        mod.install_qwen2d_patch()
        assert getattr(sd_mod.VAE, "_anzhc_qwen2d_patch_installed", False) is False

        # Opt-in: env var set -> installs.
        os.environ["DYPE_ENABLE_QWEN2D_VAE"] = "1"
        try:
            mod.install_qwen2d_patch()
            assert getattr(sd_mod.VAE, "_anzhc_qwen2d_patch_installed", False) is True
        finally:
            os.environ.pop("DYPE_ENABLE_QWEN2D_VAE", None)


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
