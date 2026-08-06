"""Tests for PixelRush node schema (Tier 2: node schema tests)."""
import pytest
import pathlib


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
