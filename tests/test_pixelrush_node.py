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
