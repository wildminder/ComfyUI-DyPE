"""Tests for the SEGA ComfyUI node definition (Tier 2: node schema tests)."""
import pytest
import inspect
import pathlib


@pytest.mark.unit
class TestSegaNodeSchema:
    def _get_node_class(self):
        """Import the SEGA node class from __init__.py."""
        import importlib
        spec = importlib.util.spec_from_file_location(
            "dype_init", pathlib.Path(__file__).parent.parent / "__init__.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # We need comfy_api which may not be available in test env
        try:
            spec.loader.exec_module(mod)
            return mod.SEGA
        except Exception:
            pytest.skip("comfy_api not available in test environment")

    def test_node_class_exists(self):
        """SEGA node class should be defined in __init__.py."""
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "class SEGA" in content

    def test_node_registered_in_extension(self):
        """SEGA should be in the extension's node list."""
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "SEGA" in content
        # Check that get_node_list includes SEGA
        assert "return [DyPE_FLUX, SEGA]" in content

    def test_imports_apply_sega(self):
        """__init__.py should import apply_sega_to_model."""
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "apply_sega_to_model" in content

    def test_node_has_sega_inputs(self):
        """The SEGA node definition should include SEGA-specific inputs."""
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        sega_inputs = [
            "mscale_alpha",
            "mscale_beta",
            "mscale_min",
            "spread_min",
            "spread_max",
            "spread_alpha",
            "base_mscale_formula",
            "base_mscale_coefficient",
        ]
        for inp in sega_inputs:
            assert inp in content, f"SEGA node should have input: {inp}"

    def test_node_has_method_combo(self):
        """The SEGA node should have a method combo with 'sega' and 'ntk' options."""
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert '"sega"' in content
        assert '"ntk"' in content

    def test_node_has_model_type_combo(self):
        """The SEGA node should have a model_type combo."""
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert '"auto"' in content
        assert '"flux"' in content
        assert '"qwen"' in content

    def test_node_defaults_match_paper(self):
        """Default SEGA parameters should match the paper's recommendations."""
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        # mscale_alpha = 0.15 (paper default)
        assert "default=0.15" in content
        # mscale_beta = 1.5 (paper default)
        assert "default=1.5" in content
        # base_mscale_coefficient = 0.08 (paper kappa)
        assert "default=0.08" in content
        # spread_alpha = 1.5 (paper gamma)
        assert "default=1.5" in content

    def test_node_has_output(self):
        """The SEGA node should output a patched model."""
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "io.Model.Output" in content

    def test_node_category(self):
        """The SEGA node should be in the model_patches category."""
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "model_patches/position_encoding" in content
