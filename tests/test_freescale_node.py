"""Tests for FreeScale node schema (Tier 2: node schema tests)."""
import pathlib

import pytest
import torch


@pytest.mark.unit
class TestFreeScaleNodeSchema:
    def _read_source(self):
        return (pathlib.Path(__file__).parent.parent / "src" / "freescale_node.py").read_text(encoding="utf-8")

    def test_node_class_exists(self):
        content = self._read_source()
        assert "class FreeScaleNode" in content

    def test_node_has_inputs(self):
        content = self._read_source()
        for inp in ["model", "vae", "positive", "negative", "latent_image", "cfg",
                     "num_inference_steps", "target_resolution", "cosine_scale",
                     "noise_timestep", "fast_mode"]:
            assert inp in content, f"FreeScale node should have input: {inp}"

    def test_node_has_output(self):
        content = self._read_source()
        assert "io.Latent.Output" in content

    def test_node_category(self):
        content = self._read_source()
        assert "image/upscaling" in content

    def test_node_defaults_match_paper(self):
        content = self._read_source()
        assert "default=7.5" in content  # cfg
        assert "default=50" in content  # num_inference_steps
        assert "default=2048" in content  # target_resolution
        assert "default=2.0" in content  # cosine_scale
        assert "default=700" in content  # noise_timestep
        assert "default=True" in content  # fast_mode

    def test_node_registered_in_extension(self):
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "FreeScale" in content

    def test_imports_freescale(self):
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "freescale_node" in content or "FreeScaleNode" in content


@pytest.mark.unit
class TestFreeScaleAttentionPatching:
    """Tests for the attention patching functions."""

    def _read_source(self):
        return (pathlib.Path(__file__).parent.parent / "src" / "freescale_node.py").read_text(encoding="utf-8")

    def test_has_patch_scale_attention(self):
        content = self._read_source()
        assert "def patch_scale_attention" in content

    def test_has_unpatch_scale_attention(self):
        content = self._read_source()
        assert "def unpatch_scale_attention" in content

    def test_patch_stores_originals(self):
        content = self._read_source()
        assert "stored" in content

    def test_uses_scale_fusion(self):
        """The node implements scale fusion via its own 3D Gaussian filter
        (``_gaussian_filter_3d``) rather than importing ``scale_fusion`` /
        ``gaussian_blur_2d`` from src/freescale.py — pin the actual symbols.
        (W3 ruff auto-fix removed a stale unused import this check relied on;
        the check now targets the real implementation, 2026-08-25.)"""
        content = self._read_source()
        assert ("scale_fusion" in content or "gaussian_blur" in content
                or "_gaussian_filter_3d" in content)

    def test_has_vae_adapters(self):
        content = self._read_source()
        assert "_make_vae_adapters" in content

    def test_handles_3d_vae(self):
        content = self._read_source()
        assert "latent_dim" in content

    def test_handles_5d_latent_for_3d_models(self):
        """Verify the node adds temporal dimension for 3D latent models."""
        content = self._read_source()
        assert "latent_dimensions" in content
        assert "unsqueeze(2)" in content

    def test_uses_repeat_to_batch_size_for_empty_latent(self):
        """Verify empty latent channels are repeated, not zero-padded."""
        content = self._read_source()
        assert "repeat_to_batch_size" in content
        assert "is_empty" in content

    def test_vae_encode_returns_5d_for_3d_models(self):
        """Verify vae_encode returns 5D latents for 3D latent models."""
        content = self._read_source()
        # The vae_encode should NOT squeeze the temporal dimension for 3D models
        assert "process_latent_in" in content


@pytest.mark.unit
class TestFreeScaleVAEAdapters5D:
    """Tests for VAE adapter 5D latent handling (Krea2/Qwen/Anima)."""

    def _read_source(self):
        return (pathlib.Path(__file__).parent.parent / "src" / "freescale_node.py").read_text(encoding="utf-8")

    def test_vae_decode_accepts_5d_latent(self):
        """vae_decode should handle 5D [B,C,T,H,W] input from sampler output."""
        content = self._read_source()
        # The vae_decode should handle 5D input via process_latent_out
        assert "latent.ndim == 5" in content or "latent_dim == 3" in content

    def test_vae_encode_returns_5d_for_3d_latent_dim(self):
        """vae_encode should return 5D for 3D latent models (latent_dim == 3)."""
        content = self._read_source()
        # For 3D latent models, vae_encode should keep the 5D shape
        # (not squeeze to 4D) so it can be passed to the sampler
        assert "encoded.unsqueeze(2)" in content

    def test_no_broadcasting_misalignment(self):
        """Verify process_latent_out/in is called on 5D, not 4D, for 3D models."""
        content = self._read_source()
        # The code should NOT squeeze 5D to 4D before calling process_latent_out
        # for 3D latent models (that was the old buggy behavior)
        assert "encoded_5d[:, :, 0]" not in content or "latent_5d[:, :, 0]" not in content


@pytest.mark.unit
class TestFreeScaleVAEAdaptersFunctional:
    """Functional tests for VAE adapter 5D latent handling with mock objects."""

    def _make_mock_vae_3d(self):
        """Create a mock 3D VAE (like Qwen2D/WanVAE) for testing."""
        import types

        vae = types.SimpleNamespace()
        vae.latent_dim = 3
        vae.downscale_ratio = 8

        def decode(latent):
            # Simulate VAE decode: [B, C, T, H, W] -> [B, 3, T, H, W]
            # Flatten T into B for 2D processing
            if latent.ndim == 5:
                b, c, t, h, w = latent.shape
                latent = latent.reshape(b * t, c, h, w)
            # Simple identity decode for testing
            out = latent[:, :3]  # Take first 3 channels as image
            return out

        def encode(image):
            # Simulate VAE encode: [B, 3, H, W] -> [B, 16, 1, H, W]
            b = image.shape[0]
            h, w = image.shape[-2], image.shape[-1]
            # Return 5D with 16 channels
            return torch.randn(b, 16, 1, h, w)

        vae.decode = decode
        vae.encode = encode
        return vae

    def _make_mock_model_3d(self):
        """Create a mock model with 3D latent format (Wan21-like)."""
        import types

        model = types.SimpleNamespace()
        model.model = types.SimpleNamespace()

        # Mock latent_format
        model.model.latent_format = types.SimpleNamespace()
        model.model.latent_format.latent_channels = 16
        model.model.latent_format.latent_dimensions = 3

        # Mock process_latent_out/in (Wan21-style: 5D latents_mean/std)
        latents_mean = torch.zeros(1, 16, 1, 1, 1)
        latents_std = torch.ones(1, 16, 1, 1, 1)

        def process_latent_out(latent):
            # Should receive 5D input
            assert latent.ndim == 5, f"process_latent_out should receive 5D, got {latent.ndim}D"
            return (latent - latents_mean) / latents_std

        def process_latent_in(latent):
            # Should receive 5D input
            assert latent.ndim == 5, f"process_latent_in should receive 5D, got {latent.ndim}D"
            return latent * latents_std + latents_mean

        model.model.process_latent_out = process_latent_out
        model.model.process_latent_in = process_latent_in

        return model

    def test_vae_decode_5d_calls_process_latent_out_on_5d(self):
        """vae_decode should call process_latent_out on 5D tensor, not 4D."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.freescale_node import _make_vae_adapters

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        # 5D latent from sampler output
        latent_5d = torch.randn(1, 16, 1, 64, 64)
        # Should not raise assertion error from process_latent_out
        result = vae_decode(latent_5d)
        assert result.ndim == 4  # Decoded image is 4D [B, 3, H, W]

    def test_vae_decode_4d_adds_temporal_before_process(self):
        """vae_decode should add temporal dim to 4D before process_latent_out."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.freescale_node import _make_vae_adapters

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        # 4D latent (should be converted to 5D before process_latent_out)
        latent_4d = torch.randn(1, 16, 64, 64)
        # Should not raise assertion error from process_latent_out
        result = vae_decode(latent_4d)
        assert result.ndim == 4

    def test_vae_encode_returns_5d_for_3d_model(self):
        """vae_encode should return 5D latent for 3D latent models."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.freescale_node import _make_vae_adapters

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        # 4D image [B, 3, H, W]
        image = torch.randn(1, 3, 64, 64)
        result = vae_encode(image)
        # Should be 5D for 3D latent models
        assert result.ndim == 5, f"Expected 5D output for 3D model, got {result.ndim}D"
        assert result.shape[2] == 1  # Temporal dim = 1
        assert result.shape[1] == 16  # 16 latent channels

    def test_vae_encode_5d_passes_process_latent_in_on_5d(self):
        """vae_encode should call process_latent_in on 5D tensor."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.freescale_node import _make_vae_adapters

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        # 4D image [B, 3, H, W]
        image = torch.randn(1, 3, 64, 64)
        # Should not raise assertion error from process_latent_in
        result = vae_encode(image)
        assert result.ndim == 5

    def test_no_batch_size_corruption(self):
        """Verify 5D handling doesn't corrupt batch dimension (the Krea2 bug)."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.freescale_node import _make_vae_adapters

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        # Simulate the cascade: decode -> upscale -> encode
        latent_5d = torch.randn(1, 16, 1, 64, 64)
        image = vae_decode(latent_5d)
        assert image.shape[0] == 1, f"Batch should be 1, got {image.shape[0]}"

        import torch.nn.functional as F
        image_up = F.interpolate(image, size=(128, 128), mode="bicubic", align_corners=False)
        z_up = vae_encode(image_up)
        # Batch should still be 1, not 16
        assert z_up.shape[0] == 1, f"Batch should be 1, got {z_up.shape[0]}"
        assert z_up.ndim == 5, f"Should be 5D, got {z_up.ndim}D"
