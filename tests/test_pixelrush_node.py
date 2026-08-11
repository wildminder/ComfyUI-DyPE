"""Tests for PixelRush node schema (Tier 2: node schema tests)."""
import pytest
import pathlib
import torch


@pytest.mark.unit
class TestPixelRushNodeSchema:
    def test_node_class_exists(self):
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")
        assert "class PixelRushNode" in content

    def test_node_has_inputs(self):
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")
        for inp in ["model", "vae", "positive", "negative", "latent_image", "cfg",
                     "num_cascade_stages", "k_timestep", "noise_lambda", "overlap"]:
            assert inp in content, f"PixelRush node should have input: {inp}"

    def test_node_has_output(self):
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")
        assert "io.Latent.Output" in content

    def test_node_category(self):
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")
        assert "image/upscaling" in content

    def test_node_defaults_match_paper(self):
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")
        assert "default=0.95" in content  # noise_lambda
        assert "default=0.50" in content  # overlap
        assert "default=249" in content  # k_timestep
        assert "default=8.0" in content  # gaussian_sigma
        assert "default=41" in content  # gaussian_kernel_size

    def test_node_registered_in_extension(self):
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "PixelRush" in content

    def test_imports_pixelrush(self):
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "pixelrush_node" in content or "PixelRushNode" in content


@pytest.mark.unit
class TestPredictEpsConditioningPipeline:
    """Tests for the conditioning pipeline in _make_predict_eps.

    Verifies that the predict_eps adapter uses ComfyUI's canonical
    conditioning pipeline: convert_cond → process_conds → get_area_and_mult → apply_model.
    """

    def _read_source(self):
        return (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")

    def test_uses_convert_cond(self):
        """convert_cond must be called to convert tuple conditioning to dict format."""
        content = self._read_source()
        assert "convert_cond" in content, (
            "_make_predict_eps must call convert_cond to convert tuple conditioning "
            "to dict format before passing to process_conds"
        )

    def test_uses_process_conds(self):
        """process_conds must be called to build model_conds."""
        content = self._read_source()
        assert "process_conds" in content, (
            "_make_predict_eps must call process_conds to build model_conds"
        )

    def test_uses_get_area_and_mult(self):
        """get_area_and_mult must be used instead of manual process() calls."""
        content = self._read_source()
        assert "get_area_and_mult" in content, (
            "_make_predict_eps must use get_area_and_mult to properly process "
            "COND objects (calls process_cond with batch_size and area)"
        )

    def test_does_not_use_manual_process(self):
        """Must not use the incorrect v.process(latent) pattern."""
        content = self._read_source()
        assert "v.process(latent)" not in content, (
            "_make_predict_eps must not use v.process(latent) — COND objects "
            "use process_cond(batch_size, area), not process(latent)"
        )

    def test_does_not_pass_raw_tuples_to_process_conds(self):
        """Must not pass raw positive/negative directly to process_conds."""
        content = self._read_source()
        # The old buggy code passed positive/negative directly:
        # conds_dict = {"positive": positive, "negative": negative}
        # The fixed code converts first:
        # conds_dict = {"positive": pos_converted, "negative": neg_converted}
        assert 'conds_dict = {"positive": positive' not in content, (
            "_make_predict_eps must not pass raw positive/negative tuples to "
            "process_conds — must convert via convert_cond first"
        )

    def test_passes_transformer_options_to_apply_model(self):
        """apply_model requires transformer_options in the conditioning dict."""
        content = self._read_source()
        assert "transformer_options" in content, (
            "_make_predict_eps must include transformer_options in the conditioning "
            "dict passed to apply_model"
        )

    def test_uses_p_input_x_not_raw_latent(self):
        """Should use p.input_x from get_area_and_mult, not raw latent."""
        content = self._read_source()
        assert "p.input_x" in content, (
            "_make_predict_eps should use p.input_x from get_area_and_mult "
            "instead of raw latent (handles area cropping)"
        )

    def test_loads_model_to_gpu(self):
        """Model must be loaded to GPU before calling apply_model."""
        content = self._read_source()
        assert "load_models_gpu" in content, (
            "_make_predict_eps must call load_models_gpu to ensure the model "
            "is on GPU before calling apply_model"
        )

    def test_calls_pre_run(self):
        """pre_run must be called to set current_patcher on the model."""
        content = self._read_source()
        assert "pre_run" in content, (
            "_make_predict_eps must call model.pre_run() to set "
            "current_patcher before apply_hooks is called"
        )

    def test_uses_model_apply_hooks_not_current_patcher(self):
        """Should use model.apply_hooks, not model.model.current_patcher.apply_hooks."""
        content = self._read_source()
        assert "model.apply_hooks" in content, (
            "_make_predict_eps should use model.apply_hooks (ModelPatcher) "
            "directly, not model.model.current_patcher.apply_hooks"
        )

    def test_uses_cond_cat_to_extract_tensors(self):
        """Should use cond_cat to extract tensors from COND objects."""
        content = self._read_source()
        assert "cond_cat" in content, (
            "_make_predict_eps must use cond_cat to extract raw tensors from "
            "COND objects (p.conditioning contains CONDCrossAttn etc., not tensors)"
        )


@pytest.mark.unit
class TestPixelRush5DLatentHandling:
    """Tests for 5D latent handling in PixelRush (Krea2/Qwen/Anima support).

    Verifies that the node correctly handles 3D latent models by:
    - Adding temporal dimension (unsqueeze(2)) before process_latent_in
    - Using process_latent_out/in on 5D tensors in VAE adapters
    - Using repeat_to_batch_size for empty latent channels
    - Unsqeezing/squeezing between 4D (core algorithm) and 5D (model)
    """

    def _read_source(self):
        return (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")

    def test_execute_uses_process_latent_in(self):
        """execute must call process_latent_in to convert initial latent to model format."""
        content = self._read_source()
        assert "process_latent_in" in content, (
            "PixelRush execute must call process_latent_in to convert the initial "
            "latent to model format (the guider normally does this, but PixelRush "
            "calls apply_model directly)"
        )

    def test_execute_adds_temporal_dim_for_3d(self):
        """execute must unsqueeze 4D to 5D for 3D latent models before process_latent_in."""
        content = self._read_source()
        assert "unsqueeze(2)" in content, (
            "PixelRush execute must add temporal dimension (unsqueeze(2)) for 3D "
            "latent models before calling process_latent_in"
        )

    def test_execute_uses_repeat_to_batch_size(self):
        """execute must use repeat_to_batch_size for empty latent channel mismatch."""
        content = self._read_source()
        assert "repeat_to_batch_size" in content, (
            "PixelRush execute must use repeat_to_batch_size for empty latent "
            "channel mismatch (not zero-padding)"
        )

    def test_execute_gets_latent_dimensions(self):
        """execute must read latent_dimensions from model.model.latent_format."""
        content = self._read_source()
        assert "latent_dimensions" in content, (
            "PixelRush execute must read latent_dimensions from the model's "
            "latent_format to detect 3D latent models"
        )

    def test_execute_squeezes_5d_to_4d_for_core(self):
        """execute must squeeze 5D to 4D before passing to pixelrush_cascade."""
        content = self._read_source()
        assert "squeeze(2)" in content, (
            "PixelRush execute must squeeze 5D to 4D before passing to "
            "pixelrush_cascade (core algorithm works in 4D spatial)"
        )

    def test_execute_unsqueezes_4d_to_5d_for_output(self):
        """execute must unsqueeze 4D result back to 5D for 3D latent models."""
        content = self._read_source()
        # The output unsqueeze is separate from the input unsqueeze
        assert content.count("unsqueeze(2)") >= 2, (
            "PixelRush execute must unsqueeze to 5D both for process_latent_in "
            "and for the final output (for 3D latent models)"
        )

    def test_vae_adapters_use_process_latent_out(self):
        """VAE decode adapter must call process_latent_out to convert from model format."""
        content = self._read_source()
        assert "process_latent_out" in content, (
            "PixelRush VAE decode adapter must call process_latent_out to convert "
            "from model latent format to VAE latent format"
        )

    def test_vae_adapters_use_process_latent_in(self):
        """VAE encode adapter must call process_latent_in to convert to model format."""
        content = self._read_source()
        assert "process_latent_in" in content, (
            "PixelRush VAE encode adapter must call process_latent_in to convert "
            "from VAE latent format to model latent format"
        )

    def test_vae_adapters_handle_5d_for_3d_models(self):
        """VAE adapters must handle 5D tensors for 3D latent models."""
        content = self._read_source()
        assert "latent_dim == 3" in content, (
            "PixelRush VAE adapters must check latent_dim == 3 to handle 5D tensors"
        )

    def test_predict_eps_accepts_latent_dimensions(self):
        """_make_predict_eps must accept latent_dimensions parameter."""
        content = self._read_source()
        assert "latent_dimensions" in content, (
            "_make_predict_eps must accept latent_dimensions to know if the model "
            "is 3D latent (for unsqueezing 4D patches to 5D before apply_model)"
        )

    def test_predict_eps_unsqueezes_4d_to_5d(self):
        """predict_eps must unsqueeze 4D patches to 5D for 3D latent models."""
        content = self._read_source()
        assert "is_3d" in content, (
            "predict_eps must track is_3d flag to unsqueeze 4D patches to 5D "
            "before calling apply_model for 3D latent models"
        )

    def test_predict_eps_squeezes_5d_eps_back(self):
        """predict_eps must squeeze 5D eps output back to 4D for the core algorithm."""
        content = self._read_source()
        assert "eps.squeeze(2)" in content or "eps.ndim == 5" in content, (
            "predict_eps must squeeze 5D eps output back to 4D for the core algorithm"
        )


@pytest.mark.unit
class TestPixelRushVAEAdaptersFunctional:
    """Functional tests for VAE adapter 5D latent handling with mock objects."""

    def _make_mock_vae_3d(self):
        """Create a mock 3D VAE (like Qwen2D/WanVAE) for testing."""
        import types

        vae = types.SimpleNamespace()
        vae.latent_dim = 3
        vae.downscale_ratio = 8

        def decode(latent):
            if latent.ndim == 5:
                b, c, t, h, w = latent.shape
                latent = latent.reshape(b * t, c, h, w)
            out = latent[:, :3]
            return out

        def encode(image):
            b = image.shape[0]
            h, w = image.shape[-2], image.shape[-1]
            return torch.randn(b, 16, 1, h, w)

        vae.decode = decode
        vae.encode = encode
        return vae

    def _make_mock_model_3d(self):
        """Create a mock model with 3D latent format (Wan21-like)."""
        import types

        model = types.SimpleNamespace()
        model.model = types.SimpleNamespace()

        model.model.latent_format = types.SimpleNamespace()
        model.model.latent_format.latent_channels = 16
        model.model.latent_format.latent_dimensions = 3

        latents_mean = torch.zeros(1, 16, 1, 1, 1)
        latents_std = torch.ones(1, 16, 1, 1, 1)

        def process_latent_out(latent):
            assert latent.ndim == 5, f"process_latent_out should receive 5D, got {latent.ndim}D"
            return (latent - latents_mean) / latents_std

        def process_latent_in(latent):
            assert latent.ndim == 5, f"process_latent_in should receive 5D, got {latent.ndim}D"
            return latent * latents_std + latents_mean

        model.model.process_latent_out = process_latent_out
        model.model.process_latent_in = process_latent_in

        return model

    def test_vae_decode_5d_calls_process_latent_out_on_5d(self):
        """vae_decode should call process_latent_out on 5D tensor, not 4D."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_vae_adapters

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        latent_5d = torch.randn(1, 16, 1, 64, 64)
        result = vae_decode(latent_5d)
        assert result.ndim == 4

    def test_vae_decode_4d_adds_temporal_before_process(self):
        """vae_decode should add temporal dim to 4D before process_latent_out."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_vae_adapters

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        latent_4d = torch.randn(1, 16, 64, 64)
        result = vae_decode(latent_4d)
        assert result.ndim == 4

    def test_vae_encode_returns_5d_for_3d_model(self):
        """vae_encode should return 5D latent for 3D latent models."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_vae_adapters

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        image = torch.randn(1, 3, 64, 64)
        result = vae_encode(image)
        assert result.ndim == 5, f"Expected 5D output for 3D model, got {result.ndim}D"
        assert result.shape[2] == 1
        assert result.shape[1] == 16

    def test_vae_encode_5d_passes_process_latent_in_on_5d(self):
        """vae_encode should call process_latent_in on 5D tensor."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_vae_adapters

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        image = torch.randn(1, 3, 64, 64)
        result = vae_encode(image)
        assert result.ndim == 5

    def test_no_batch_size_corruption(self):
        """Verify 5D handling doesn't corrupt batch dimension (the Krea2 bug)."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_vae_adapters
        import torch.nn.functional as F

        vae = self._make_mock_vae_3d()
        model = self._make_mock_model_3d()
        device = torch.device("cpu")

        vae_decode, vae_encode = _make_vae_adapters(vae, device, model)

        latent_5d = torch.randn(1, 16, 1, 64, 64)
        image = vae_decode(latent_5d)
        assert image.shape[0] == 1, f"Batch should be 1, got {image.shape[0]}"

        image_up = F.interpolate(image, size=(128, 128), mode="bicubic", align_corners=False)
        z_up = vae_encode(image_up)
        assert z_up.shape[0] == 1, f"Batch should be 1, got {z_up.shape[0]}"
        assert z_up.ndim == 5, f"Should be 5D, got {z_up.ndim}D"

    def test_cascade_squeezes_5d_to_4d(self):
        """pixelrush_cascade should squeeze 5D vae_encode output to 4D for refine_latent_once."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush import pixelrush_cascade, PixelRushConfig
        import torch.nn.functional as F

        # Track shapes passed to predict_eps
        eps_shapes = []

        def vae_decode(latent):
            if isinstance(latent, dict):
                latent = latent["samples"]
            if latent.ndim == 5:
                latent = latent[:, :, 0]
            return latent[:, :3]  # [B, 3, H, W]

        def vae_encode(image):
            b = image.shape[0]
            h, w = image.shape[-2], image.shape[-1]
            # Return 5D (as the fixed vae_encode adapter does for 3D models)
            return torch.randn(b, 16, 1, h, w)

        def predict_eps(latent, timestep):
            eps_shapes.append(latent.shape)
            return torch.randn_like(latent)

        def alpha_bar_at(timestep):
            return 0.5

        cfg = PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5,
            k_timestep=249, noise_lambda=0.95,
            gaussian_sigma=8.0, gaussian_kernel_size=41,
        )

        initial_latent = torch.randn(1, 16, 32, 32)
        result = pixelrush_cascade(
            initial_latent=initial_latent,
            num_cascade_stages=1,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            predict_eps=predict_eps,
            alpha_bar_at=alpha_bar_at,
            cfg=cfg,
        )

        # Result should be 4D (core algorithm works in 4D)
        assert result.ndim == 4, f"Result should be 4D, got {result.ndim}D"
        # predict_eps should have received 4D patches
        for shape in eps_shapes:
            assert len(shape) == 4, f"predict_eps should receive 4D, got {len(shape)}D shape {shape}"
