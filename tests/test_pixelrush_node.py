"""Tests for PixelRush node schema (Tier 2: node schema tests)."""
import pathlib

import pytest
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
        import torch.nn.functional as F

        from src.pixelrush_node import _make_vae_adapters

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

        from src.pixelrush import PixelRushConfig, pixelrush_cascade

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


@pytest.mark.unit
class TestPixelRushProgressBar:
    """Tests for progress bar integration in PixelRush node."""

    def _read_source(self):
        return (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")

    def test_execute_creates_progress_bar(self):
        """execute must create a comfy.utils.ProgressBar."""
        content = self._read_source()
        assert "ProgressBar" in content, (
            "PixelRush execute must create a comfy.utils.ProgressBar for "
            "native ComfyUI progress tracking"
        )

    def test_execute_passes_progress_callback(self):
        """execute must pass progress_callback to pixelrush_cascade."""
        content = self._read_source()
        assert "progress_callback" in content, (
            "PixelRush execute must pass a progress_callback to pixelrush_cascade"
        )

    def test_execute_calls_update_absolute(self):
        """execute must call pbar.update_absolute to update progress."""
        content = self._read_source()
        assert "update_absolute" in content, (
            "PixelRush execute must call pbar.update_absolute to update the progress bar"
        )

    def test_execute_precomputes_total_patches(self):
        """execute must pre-compute total patches across all cascade stages."""
        content = self._read_source()
        assert "total_patches" in content, (
            "PixelRush execute must pre-compute total_patches for the progress bar"
        )

    def test_refine_latent_once_accepts_progress_callback(self):
        """refine_latent_once must accept a progress_callback parameter."""
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush.py").read_text(encoding="utf-8")
        assert "progress_callback" in content, (
            "refine_latent_once must accept a progress_callback parameter"
        )

    def test_pixelrush_cascade_accepts_progress_callback(self):
        """pixelrush_cascade must accept a progress_callback parameter."""
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush.py").read_text(encoding="utf-8")
        # The function signature should include progress_callback
        assert "progress_callback" in content, (
            "pixelrush_cascade must accept a progress_callback parameter"
        )

    def test_progress_callback_called_per_patch(self):
        """refine_latent_once should call progress_callback after each patch."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush import PixelRushConfig, refine_latent_once

        # Track callback invocations
        callback_calls = []

        def progress_callback(patch_idx, total_patches):
            callback_calls.append((patch_idx, total_patches))

        coarse_latent = torch.randn(1, 4, 64, 64)

        def predict_eps(latent, timestep):
            return torch.randn_like(latent)

        def alpha_bar_at(timestep):
            return 0.5

        cfg = PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5,
            k_timestep=249, noise_lambda=0.95,
            gaussian_sigma=8.0, gaussian_kernel_size=41,
        )

        refine_latent_once(
            coarse_latent=coarse_latent,
            predict_eps=predict_eps,
            alpha_bar_at=alpha_bar_at,
            cfg=cfg,
            progress_callback=progress_callback,
        )

        # Should have been called once per patch
        assert len(callback_calls) > 0, "progress_callback should have been called"
        # Last call should have patch_idx == total_patches
        last_idx, last_total = callback_calls[-1]
        assert last_idx == last_total, (
            f"Last callback should have patch_idx == total_patches, "
            f"got {last_idx} != {last_total}"
        )

    def test_cascade_progress_callback_receives_stage_info(self):
        """pixelrush_cascade should pass stage info to progress_callback."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush import PixelRushConfig, pixelrush_cascade

        callback_calls = []

        def progress_callback(patch_idx, total_patches, stage, num_stages):
            callback_calls.append((patch_idx, total_patches, stage, num_stages))

        def vae_decode(latent):
            if isinstance(latent, dict):
                latent = latent["samples"]
            return latent[:, :3]

        def vae_encode(image):
            b = image.shape[0]
            h, w = image.shape[-2], image.shape[-1]
            return torch.randn(b, 4, h, w)

        def predict_eps(latent, timestep):
            return torch.randn_like(latent)

        def alpha_bar_at(timestep):
            return 0.5

        cfg = PixelRushConfig(
            patch_h=32, patch_w=32, overlap=0.5,
            k_timestep=249, noise_lambda=0.95,
            gaussian_sigma=8.0, gaussian_kernel_size=41,
        )

        initial_latent = torch.randn(1, 4, 32, 32)
        pixelrush_cascade(
            initial_latent=initial_latent,
            num_cascade_stages=2,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            predict_eps=predict_eps,
            alpha_bar_at=alpha_bar_at,
            cfg=cfg,
            progress_callback=progress_callback,
        )

        # Should have calls from both stages
        stages_seen = set(call[2] for call in callback_calls)
        assert 0 in stages_seen, "Should have calls from stage 0"
        assert 1 in stages_seen, "Should have calls from stage 1"
        # All calls should have num_stages == 2
        for call in callback_calls:
            assert call[3] == 2, f"num_stages should be 2, got {call[3]}"


@pytest.mark.unit
class TestPixelRushInferenceBugFix:
    """Tests for the inference bug fixes (timestep/sigma conversion, x0→epsilon).

    Bug 1: apply_model returns x0, not epsilon — must convert via (x - x0) / sigma
    Bug 2: timestep used as array index — must use model_sampling.sigma(timestep)
    Bug 3: spherical_lerp uses unit vectors — must use raw vectors to match reference
    """

    def _read_source(self):
        return (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")

    # --- Bug 2: timestep → sigma conversion ---

    def test_predict_eps_uses_model_sampling_sigma(self):
        """predict_eps must use model_sampling.sigma() for timestep conversion."""
        content = self._read_source()
        assert "model_sampling.sigma" in content, (
            "predict_eps must use model_sampling.sigma(timestep) to convert "
            "timestep (0-999) to sigma, not use timestep as array index"
        )

    def test_predict_eps_does_not_use_timestep_as_index(self):
        """predict_eps must not use timestep as an index into sigmas array."""
        content = self._read_source()
        assert "sigmas[timestep]" not in content, (
            "predict_eps must not use sigmas[timestep] — timestep is a value "
            "in 0-999 range, not an index into the sigmas array"
        )

    def test_predict_eps_clamps_sigma_minimum(self):
        """predict_eps must clamp sigma to avoid division by zero at timestep 0."""
        content = self._read_source()
        assert "1e-6" in content or "max(sigma_val" in content, (
            "predict_eps must clamp sigma to a minimum (1e-6) to avoid "
            "division by zero when extracting epsilon at timestep 0"
        )

    def test_alpha_bar_at_uses_model_sampling_sigma(self):
        """alpha_bar_at must use model_sampling.sigma() for timestep conversion."""
        content = self._read_source()
        assert "model_sampling.sigma" in content, (
            "alpha_bar_at must use model_sampling.sigma(timestep) to convert "
            "timestep to sigma before computing alpha_bar"
        )

    def test_alpha_bar_at_does_not_use_index(self):
        """alpha_bar_at must not use timestep as an array index."""
        content = self._read_source()
        assert "alphas_cumprod[timestep]" not in content, (
            "alpha_bar_at must not use alphas_cumprod[timestep] — timestep is "
            "a value in 0-999 range, not an index"
        )

    # --- Bug 1: bypass calculate_denoised to get raw epsilon ---

    def test_predict_eps_bypasses_calculate_denoised(self):
        """predict_eps must bypass calculate_denoised to get raw model output (epsilon).

        apply_model returns calculate_denoised(sigma, model_output, x) = x0,
        which is useless at sigma≈0 (returns x trivially, making eps≈0).
        Instead, predict_eps must call diffusion_model directly to get the raw
        model_output (which IS epsilon for EPS prediction type).
        """
        content = self._read_source()
        assert "diffusion_model(" in content, (
            "predict_eps must call m.diffusion_model() directly to get raw "
            "model output (epsilon), bypassing calculate_denoised"
        )
        assert "calculate_input" in content, (
            "predict_eps must call ms.calculate_input() to scale the input "
            "before feeding to diffusion_model"
        )

    def test_predict_eps_does_not_call_apply_model(self):
        """predict_eps must not call apply_model (it returns x0, not epsilon)."""
        content = self._read_source()
        assert "model.model.apply_model" not in content, (
            "predict_eps must not call apply_model — it returns x0 (via "
            "calculate_denoised), not the raw epsilon we need"
        )

    def test_predict_eps_does_not_divide_by_sigma(self):
        """predict_eps must not divide by sigma (no x0→epsilon conversion needed).

        Since we get the raw model_output (epsilon) directly from diffusion_model,
        there's no need to convert x0→epsilon via (x - x0) / sigma.
        """
        content = self._read_source()
        assert "(p.input_x - x0)" not in content, (
            "predict_eps must not do x0→epsilon conversion — we get raw "
            "epsilon directly from diffusion_model"
        )
        assert "sigma_reshaped" not in content, (
            "predict_eps must not reshape sigma for division — no division "
            "is needed when getting raw epsilon directly"
        )

    # --- Bug 3: spherical_lerp must use UNIT vectors (not raw) ---

    def test_spherical_lerp_uses_unit_vectors(self):
        """spherical_lerp must use a_unit/b_unit in the direction, not raw a_flat/b_flat.

        Using raw vectors squares the norm whenever |a| != |b| (always true for
        eps_pred≈0 vs eps_rand≈1), making eps_inj ~60x too large -> pure noise.
        """
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush.py").read_text(encoding="utf-8")
        assert "a_unit" in content, "spherical_lerp should define a_unit"
        assert "b_unit" in content, "spherical_lerp should define b_unit"
        # The direction term must use a_unit/b_unit, NOT a_flat/b_flat.
        assert "sin_omega * a_flat" not in content, (
            "spherical_lerp direction must use a_unit (unit vector), not a_flat (raw)"
        )
        assert "sin_omega * b_flat" not in content, (
            "spherical_lerp direction must use b_unit (unit vector), not b_flat (raw)"
        )

    def test_spherical_lerp_does_not_explode_norm(self):
        """Regression: slerp of two different-magnitude vectors must not square the norm.

        slerp(eps_pred (norm~6), eps_rand (norm~64), 0.95) must yield a result
        whose norm is ~ the interpolated magnitude (~61), NOT ~3900 (squared).
        """
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        import torch

        from src.pixelrush import spherical_lerp
        torch.manual_seed(0)
        a = 0.1 * torch.randn(1, 4, 32, 32)   # eps_pred-like (small norm)
        b = torch.randn(1, 4, 32, 32)          # eps_rand-like (large norm)
        out = spherical_lerp(a, b, t=0.95)
        out_norm = out.flatten(1).norm(dim=1).item()
        # Interpolated magnitude should be ~ (1-0.95)*||a|| + 0.95*||b||
        expected_mag = 0.05 * a.flatten(1).norm().item() + 0.95 * b.flatten(1).norm().item()
        # Allow 2x tolerance; a squared norm would be ~60x larger.
        assert out_norm < 2.0 * expected_mag, (
            f"slerp exploded the norm: got {out_norm:.1f}, expected ~{expected_mag:.1f} "
            f"(squared-norm bug would give ~{expected_mag**2:.1f})"
        )


@pytest.mark.unit
class TestPixelRushInferenceBugFixFunctional:
    """Functional tests for the inference bug fixes with mock objects."""

    def _make_mock_model_sampling(self):
        """Create a mock model_sampling with sigma() method.

        Simulates ModelSamplingDiscrete.sigma() which maps timestep (0-999)
        to sigma by interpolating in log-space across the sigma schedule.
        The schedule has ~20 entries, but timesteps can be 0-999.
        """
        import types

        ms = types.SimpleNamespace()
        # Mock sigmas array (typical 20-step schedule)
        # Use a small minimum instead of 0 to avoid log(0) = -inf
        ms.sigmas = torch.linspace(0.01, 14.0, 20)
        ms.log_sigmas = ms.sigmas.log()

        def sigma(timestep):
            # ModelSamplingDiscrete.sigma() maps timestep (0-999) to sigma
            # by treating timestep as a continuous index into log_sigmas.
            # timestep=0 → sigmas[0], timestep=999 → sigmas[-1]
            # Scale timestep from 0-999 to 0-(len-1)
            max_ts = 999.0
            t = torch.clamp(timestep.float(), min=0, max=max_ts)
            t_scaled = t * (len(ms.sigmas) - 1) / max_ts
            low_idx = t_scaled.floor().long().clamp(0, len(ms.sigmas) - 1)
            high_idx = t_scaled.ceil().long().clamp(0, len(ms.sigmas) - 1)
            w = t_scaled.frac()
            log_sigma = (1 - w) * ms.log_sigmas[low_idx] + w * ms.log_sigmas[high_idx]
            return log_sigma.exp()

        ms.sigma = sigma
        return ms

    def test_alpha_bar_at_returns_reasonable_values(self):
        """alpha_bar_at should return reasonable values for various timesteps."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        import types

        from src.pixelrush_node import _make_alpha_bar_at

        model = types.SimpleNamespace()
        model.model = types.SimpleNamespace()
        model.model.model_sampling = self._make_mock_model_sampling()
        model.load_device = torch.device("cpu")

        alpha_bar_at = _make_alpha_bar_at(model)

        # timestep=0 should give alpha_bar ≈ 1.0 (clean)
        ab_0 = alpha_bar_at(0)
        assert 0.9 < ab_0 <= 1.0, f"alpha_bar(0) should be ~1.0, got {ab_0}"

        # timestep=249 should give alpha_bar between 0 and 1 (not 0!)
        ab_249 = alpha_bar_at(249)
        assert 0 < ab_249 < 1.0, f"alpha_bar(249) should be in (0, 1), got {ab_249}"

        # timestep=999 should give small alpha_bar (high noise)
        ab_999 = alpha_bar_at(999)
        assert 0 < ab_999 < ab_249, f"alpha_bar(999) should be < alpha_bar(249), got {ab_999}"

        # Different timesteps should give different alpha_bars
        assert ab_0 != ab_249, "Different timesteps should give different alpha_bars"

    def test_alpha_bar_at_249_not_near_zero(self):
        """alpha_bar_at(249) should NOT be near zero (the old bug returned 0)."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        import types

        from src.pixelrush_node import _make_alpha_bar_at

        model = types.SimpleNamespace()
        model.model = types.SimpleNamespace()
        model.model.model_sampling = self._make_mock_model_sampling()
        model.load_device = torch.device("cpu")

        alpha_bar_at = _make_alpha_bar_at(model)
        ab_249 = alpha_bar_at(249)

        # The old bug returned alpha_bar ≈ 0 (using sigmas[-1] which is max sigma)
        # The fix should return a reasonable value
        assert ab_249 > 0.01, (
            f"alpha_bar(249) should be > 0.01, got {ab_249} — "
            "if this is near 0, the timestep is being used as an array index (Bug 2)"
        )

    def test_raw_model_output_is_epsilon(self):
        """Verify that bypassing calculate_denoised gives raw epsilon.

        For EPS models, the raw model_output from diffusion_model IS epsilon.
        calculate_denoised would convert it to x0 via: x0 = x - eps * sigma.
        At sigma≈0, x0 ≈ x (trivially), making (x - x0)/sigma ≈ 0/0 = garbage.
        By bypassing calculate_denoised, we get the true epsilon directly.
        """
        # Simulate: model predicts epsilon directly
        x = torch.randn(1, 4, 8, 8)
        eps_true = torch.randn_like(x)
        sigma = torch.tensor([1.0])

        # The raw model_output IS epsilon (no conversion needed)
        model_output = eps_true  # What diffusion_model returns
        assert torch.equal(model_output, eps_true), (
            "Raw model_output should be epsilon, no conversion needed"
        )

        # Verify calculate_denoised would give x0 (which is NOT what we want)
        sigma_reshaped = sigma.reshape(sigma.shape + (1,) * (x.ndim - sigma.ndim))
        x0 = x - model_output * sigma_reshaped  # This is what apply_model returns
        assert not torch.equal(x0, eps_true), (
            "x0 (from calculate_denoised) should NOT equal epsilon — "
            "this is why we bypass it"
        )

        # At sigma≈0, x0 ≈ x (trivially), making (x - x0)/sigma numerically unstable.
        # The subtraction x - x0 loses precision (two nearly-equal numbers),
        # and dividing by 1e-6 amplifies the error. This is WHY we bypass
        # calculate_denoised and get raw epsilon directly.
        sigma_near_zero = torch.tensor([1e-6])
        sigma_rz = sigma_near_zero.reshape(sigma_near_zero.shape + (1,) * (x.ndim - sigma_near_zero.ndim))
        x0_near_zero = x - eps_true * sigma_rz  # ≈ x since sigma≈0
        eps_via_x0 = (x - x0_near_zero) / sigma_rz  # Numerically unstable!
        # The conversion should NOT match the true epsilon closely —
        # this demonstrates the numerical instability at low sigma.
        max_err = (eps_via_x0 - eps_true).abs().max().item()
        assert max_err > 1e-3, (
            f"At sigma≈0, x0→eps conversion should be numerically unstable "
            f"(max error should be > 1e-3, got {max_err}) — this is why we "
            f"bypass calculate_denoised and get raw epsilon directly"
        )

    def test_ddim_forward_with_correct_alpha_bar(self):
        """DDIM forward with correct alpha_bar should produce partially noised latent."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush import ddim_forward_one_step, ddim_reverse_one_step_to_zero

        z_0 = torch.randn(1, 4, 8, 8)
        eps = torch.randn_like(z_0)

        # With correct alpha_bar (e.g., 0.5), z_K should be a mix of z_0 and eps
        alpha_bar = 0.5
        z_K = ddim_forward_one_step(z_0, eps, alpha_bar)
        expected = (0.5 ** 0.5) * z_0 + (0.5 ** 0.5) * eps
        assert torch.allclose(z_K, expected, atol=1e-5), (
            "DDIM forward with alpha_bar=0.5 should produce sqrt(0.5)*z_0 + sqrt(0.5)*eps"
        )

        # Reverse should recover z_0 (approximately, with different eps)
        z_0_hat = ddim_reverse_one_step_to_zero(z_K, eps, alpha_bar)
        assert torch.allclose(z_0_hat, z_0, atol=1e-4), (
            "DDIM reverse with same eps should recover z_0"
        )

    def test_ddim_forward_with_alpha_bar_zero_produces_noise(self):
        """DDIM forward with alpha_bar=0 (old bug) produces pure noise."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush import ddim_forward_one_step

        z_0 = torch.randn(1, 4, 8, 8)
        eps = torch.randn_like(z_0)

        # With alpha_bar=0 (the old bug), z_K = 0*z_0 + 1*eps = eps (pure noise!)
        z_K = ddim_forward_one_step(z_0, eps, 0.0)
        assert torch.allclose(z_K, eps, atol=1e-5), (
            "DDIM forward with alpha_bar=0 produces pure noise (this was the bug)"
        )


@pytest.mark.unit
class TestPixelRushPredictionTypeDetection:
    """Tests for prediction-type detection and epsilon conversion.

    The critical bug: FLUX uses CONST (flow-matching) prediction where the
    raw model output is velocity v = eps - x0, NOT epsilon. Treating velocity
    as epsilon produces pure noise. We must detect the prediction type and
    convert correctly.
    """

    def _make_mock_model_sampling(self, prediction_type):
        """Create a mock model_sampling with the given prediction type's MRO."""
        if prediction_type == "const":
            ConstClass = type("CONST", (), {})
            ModelSampling = type("ModelSampling", (ConstClass,), {})
        elif prediction_type == "eps":
            EpsClass = type("EPS", (), {})
            ModelSampling = type("ModelSampling", (EpsClass,), {})
        elif prediction_type == "v_prediction":
            EpsClass = type("EPS", (), {})
            VPClass = type("V_PREDICTION", (EpsClass,), {})
            ModelSampling = type("ModelSampling", (VPClass,), {})
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")
        ms = ModelSampling()
        ms.sigma_data = 1.0
        return ms

    def test_detect_const_prediction_type(self):
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _detect_prediction_type

        ms = self._make_mock_model_sampling("const")
        assert _detect_prediction_type(ms) == "const"

    def test_detect_eps_prediction_type(self):
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _detect_prediction_type

        ms = self._make_mock_model_sampling("eps")
        assert _detect_prediction_type(ms) == "eps"

    def test_detect_v_prediction_type(self):
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _detect_prediction_type

        ms = self._make_mock_model_sampling("v_prediction")
        assert _detect_prediction_type(ms) == "v_prediction"

    def test_const_conversion_velocity_to_epsilon(self):
        """For CONST/flow, raw output is velocity v = eps - x0.

        eps = x_t + v * (1 - sigma)  (stable at sigma≈0)
        """
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_model_output_to_eps

        ms = self._make_mock_model_sampling("const")
        converter = _make_model_output_to_eps(ms, "const")

        x_t = torch.randn(1, 4, 8, 8)
        eps_true = torch.randn_like(x_t)
        sigma = torch.tensor([0.5])

        # velocity v = eps - x0, and x0 = x_t - v*sigma
        # So v = eps - (x_t - v*sigma) → v = (eps - x_t) / (1 - sigma)
        v = (eps_true - x_t) / (1.0 - 0.5)

        eps_converted = converter(v, x_t, sigma)
        assert torch.allclose(eps_converted, eps_true, atol=1e-5), (
            "CONST conversion should recover epsilon from velocity"
        )

    def test_const_conversion_at_sigma_zero(self):
        """At sigma=0, CONST conversion: eps = x_t + v * 1 = x_t + v.

        For a clean image, v ≈ -x_t (since eps≈0), so eps ≈ 0. This is
        stable (no division by sigma).
        """
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_model_output_to_eps

        ms = self._make_mock_model_sampling("const")
        converter = _make_model_output_to_eps(ms, "const")

        x_t = torch.randn(1, 4, 8, 8)
        sigma = torch.tensor([0.0])

        # velocity for clean image: v = eps - x0 = 0 - x_t = -x_t
        v = -x_t
        eps_converted = converter(v, x_t, sigma)
        # eps = x_t + v*1 = x_t - x_t = 0
        assert torch.allclose(eps_converted, torch.zeros_like(x_t), atol=1e-5), (
            "CONST conversion at sigma=0 with clean image should give eps≈0"
        )

    def test_eps_conversion_is_identity(self):
        """For EPS, raw output IS epsilon (identity)."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_model_output_to_eps

        ms = self._make_mock_model_sampling("eps")
        converter = _make_model_output_to_eps(ms, "eps")

        x_t = torch.randn(1, 4, 8, 8)
        model_output = torch.randn_like(x_t)
        sigma = torch.tensor([0.5])

        eps_converted = converter(model_output, x_t, sigma)
        assert torch.equal(eps_converted, model_output), (
            "EPS conversion should be identity (raw output IS epsilon)"
        )

    def test_const_eps_to_x0(self):
        """For CONST/flow, x0 = (x_t - sigma*eps) / (1 - sigma)."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_eps_to_x0

        ms = self._make_mock_model_sampling("const")
        converter = _make_eps_to_x0(ms, "const")

        x_t = torch.randn(1, 4, 8, 8)
        eps = torch.randn_like(x_t)
        sigma = torch.tensor([0.5])

        # x_t = sigma*eps + (1-sigma)*x0 → x0 = (x_t - sigma*eps) / (1-sigma)
        x0_true = (x_t - 0.5 * eps) / (1.0 - 0.5)
        x0_converted = converter(x_t, eps, sigma)
        assert torch.allclose(x0_converted, x0_true, atol=1e-5), (
            "CONST eps_to_x0 should match (x_t - sigma*eps) / (1-sigma)"
        )

    def test_eps_eps_to_x0(self):
        """For EPS, x0 = x_t - sigma*eps."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _make_eps_to_x0

        ms = self._make_mock_model_sampling("eps")
        converter = _make_eps_to_x0(ms, "eps")

        x_t = torch.randn(1, 4, 8, 8)
        eps = torch.randn_like(x_t)
        sigma = torch.tensor([0.5])

        x0_true = x_t - 0.5 * eps
        x0_converted = converter(x_t, eps, sigma)
        assert torch.allclose(x0_converted, x0_true, atol=1e-5), (
            "EPS eps_to_x0 should match x_t - sigma*eps"
        )

    def test_source_uses_prediction_type_detection(self):
        """Source must detect prediction type and convert model output."""
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")
        assert "_detect_prediction_type" in content, (
            "predict_eps must detect the model's prediction type"
        )
        assert "_make_model_output_to_eps" in content, (
            "predict_eps must convert raw model output to epsilon using the "
            "prediction type (CONST/flow, EPS, V_PREDICTION, X0)"
        )
        assert "model_output_to_eps(" in content, (
            "run_cond must call model_output_to_eps to convert raw output"
        )

    def test_source_uses_noise_scaling_for_forward(self):
        """Source must use model's noise_scaling for the forward step."""
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")
        assert "_make_forward_step" in content, (
            "Node must create a forward_step adapter using noise_scaling"
        )
        assert "ms.noise_scaling" in content, (
            "forward_step must use model_sampling.noise_scaling"
        )

    def test_source_uses_eps_to_x0_for_reverse(self):
        """Source must use eps_to_x0 for the reverse step."""
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")
        assert "_make_reverse_step" in content, (
            "Node must create a reverse_step adapter using eps_to_x0"
        )
        assert "_make_eps_to_x0" in content, (
            "reverse_step must use _make_eps_to_x0 (inverse of noise_scaling)"
        )

    def test_refine_latent_once_accepts_adapters(self):
        """refine_latent_once must accept forward_step/reverse_step/sigma_at."""
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush.py").read_text(encoding="utf-8")
        assert "forward_step:" in content, "refine_latent_once must accept forward_step"
        assert "reverse_step:" in content, "refine_latent_once must accept reverse_step"
        assert "sigma_at:" in content, "refine_latent_once must accept sigma_at"

    def test_pixelrush_cascade_passes_adapters(self):
        """pixelrush_cascade must pass adapters to refine_latent_once."""
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush.py").read_text(encoding="utf-8")
        assert "forward_step=forward_step" in content, (
            "pixelrush_cascade must pass forward_step to refine_latent_once"
        )
        assert "reverse_step=reverse_step" in content, (
            "pixelrush_cascade must pass reverse_step to refine_latent_once"
        )
        assert "sigma_at=sigma_at" in content, (
            "pixelrush_cascade must pass sigma_at to refine_latent_once"
        )


@pytest.mark.unit
class TestPixelRushKTimestepScaling:
    """Tests for k_timestep scaling to model's native timestep range.

    FLUX/CONST-flow models use 0-1 timestep range; EPS/SD models use 0-999.
    Passing k_timestep=249 (paper default for EPS) to a FLUX model gives
    sigma>1 (invalid), making 1-sigma negative -> pure noise.
    """

    def _make_mock_model(self, timestep_range):
        """Create a mock model with the given timestep range at sigma_max."""
        import types

        model = types.SimpleNamespace()
        model_sampling = types.SimpleNamespace()
        model_sampling.sigma_max = 1.0 if timestep_range == "01" else 14.6
        model_sampling.timestep = lambda sigma: sigma if timestep_range == "01" else sigma * 999.0 / 14.6
        model.model = types.SimpleNamespace()
        model.model.model_sampling = model_sampling
        return model

    def test_scale_k_timestep_flux_01_range(self):
        """For FLUX (0-1 range), k_timestep=249 should scale to ~0.249."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _scale_k_timestep

        model = self._make_mock_model("01")
        scaled = _scale_k_timestep(model, 249)
        assert abs(scaled - 249 / 999.0) < 1e-6, (
            f"FLUX k_timestep should scale 249 -> {249/999.0:.4f}, got {scaled}"
        )

    def test_scale_k_timestep_eps_0999_range(self):
        """For EPS (0-999 range), k_timestep=249 should be unchanged."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _scale_k_timestep

        model = self._make_mock_model("0999")
        scaled = _scale_k_timestep(model, 249)
        assert scaled == 249, (
            f"EPS k_timestep should be unchanged (249), got {scaled}"
        )

    def test_scale_k_timestep_flux_small_value(self):
        """For FLUX, k_timestep=50 should scale to ~0.05."""
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _scale_k_timestep

        model = self._make_mock_model("01")
        scaled = _scale_k_timestep(model, 50)
        assert abs(scaled - 50 / 999.0) < 1e-6, (
            f"FLUX k_timestep should scale 50 -> {50/999.0:.4f}, got {scaled}"
        )

    def test_source_uses_scale_k_timestep(self):
        """Source must use _scale_k_timestep in execute."""
        content = (pathlib.Path(__file__).parent.parent / "src" / "pixelrush_node.py").read_text(encoding="utf-8")
        assert "_scale_k_timestep(" in content, (
            "execute must call _scale_k_timestep to scale k_timestep to model range"
        )


@pytest.mark.unit
class TestPrepareInitialLatent:
    """Tests for _prepare_initial_latent (regression guard for the SDXL
    UnboundLocalError: cfg_obj referenced before assignment in execute).

    The guard that decides whether to apply process_latent_in to the initial
    latent was previously inlined in execute and referenced cfg_obj (defined
    later). Extracting it into this helper makes operate_in_vae_space an
    explicit parameter, so it can never be undefined.
    """

    def _import_helper(self):
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
        from src.pixelrush_node import _prepare_initial_latent
        return _prepare_initial_latent

    def test_vae_space_skips_process_latent_in(self):
        """operate_in_vae_space=True must NOT call process_latent_in (SDXL fix)."""
        helper = self._import_helper()
        latent = torch.randn(1, 4, 32, 32)
        calls = []
        def process_latent_in(x):
            calls.append(1)
            return x * 0.13025
        out = helper(latent, process_latent_in, latent_dimensions=2,
                     operate_in_vae_space=True)
        assert len(calls) == 0, "process_latent_in must be skipped in VAE space"
        assert torch.equal(out, latent), "latent must be unchanged in VAE space"

    def test_model_space_applies_process_latent_in(self):
        """operate_in_vae_space=False must call process_latent_in (legacy path)."""
        helper = self._import_helper()
        latent = torch.randn(1, 4, 32, 32)
        calls = []
        def process_latent_in(x):
            calls.append(1)
            return x * 0.13025
        out = helper(latent, process_latent_in, latent_dimensions=2,
                     operate_in_vae_space=False)
        assert len(calls) == 1, "process_latent_in must be called in model space"
        assert torch.allclose(out, latent * 0.13025)

    def test_none_process_latent_in_is_noop(self):
        """process_latent_in=None must be a no-op in both modes."""
        helper = self._import_helper()
        latent = torch.randn(1, 4, 32, 32)
        out_vae = helper(latent, None, latent_dimensions=2, operate_in_vae_space=True)
        out_model = helper(latent, None, latent_dimensions=2, operate_in_vae_space=False)
        assert torch.equal(out_vae, latent)
        assert torch.equal(out_model, latent)

    def test_3d_unsqueezes_before_process_latent_in(self):
        """3D model-space path must unsqueeze 4D -> 5D before process_latent_in."""
        helper = self._import_helper()
        latent = torch.randn(1, 4, 32, 32)  # 4D
        seen_shape = {}
        def process_latent_in(x):
            seen_shape["shape"] = tuple(x.shape)
            return x
        out = helper(latent, process_latent_in, latent_dimensions=3,
                     operate_in_vae_space=False)
        assert seen_shape["shape"] == (1, 4, 1, 32, 32), (
            f"3D process_latent_in should receive 5D, got {seen_shape['shape']}"
        )
        assert tuple(out.shape) == (1, 4, 1, 32, 32)

    def test_3d_vae_space_skips_process_latent_in(self):
        """3D VAE-space path must NOT call process_latent_in and keep 4D."""
        helper = self._import_helper()
        latent = torch.randn(1, 4, 32, 32)
        calls = []
        def process_latent_in(x):
            calls.append(1)
            return x
        out = helper(latent, process_latent_in, latent_dimensions=3,
                     operate_in_vae_space=True)
        assert len(calls) == 0
        assert tuple(out.shape) == (1, 4, 32, 32)
