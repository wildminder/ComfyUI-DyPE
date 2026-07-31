"""Code quality meta-tests — verify structural invariants (Tier 1)."""
import ast
import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"


@pytest.mark.unit
class TestNoBareExcept:
    def test_no_bare_except_in_src(self):
        """Ensure no bare 'except:' clauses exist in src/."""
        violations = []
        for py_file in SRC_DIR.rglob("*.py"):
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    violations.append(f"{py_file.relative_to(PROJECT_ROOT)}:{node.lineno}")
        assert violations == [], f"Bare except found at: {violations}"


@pytest.mark.unit
class TestMagicNumbersDocumented:
    def test_yarn_constants_documented(self):
        """Magic numbers beta_0/beta_1 must have documentation comments."""
        rope_file = SRC_DIR / "rope.py"
        content = rope_file.read_text(encoding="utf-8")
        idx = content.index("beta_0, beta_1 = 1.25, 0.75")
        preceding = content[max(0, idx - 500):idx]
        assert "YaRN" in preceding or "Peng" in preceding, \
            "Magic numbers beta_0/beta_1 lack documentation"


@pytest.mark.unit
class TestZImageScaleHintDocumented:
    def test_zimage_scale_hint_has_comment(self):
        """The Z-Image scale hint computation must be documented."""
        patch_file = SRC_DIR / "patch_utils.py"
        content = patch_file.read_text(encoding="utf-8")
        assert "zimage_freq_scale_factor" in content
        # Verify there's a comment explaining the approach
        assert "PosEmbedZImage" in content or "scale hint" in content, \
            "Missing documentation for Z-Image scale hint approach"


@pytest.mark.unit
class TestTypeAnnotations:
    def test_rope_functions_have_return_annotations(self):
        import inspect
        from src.rope import (
            find_correction_factor, find_correction_range,
            linear_ramp_mask, find_newbase_ntk,
            get_1d_dype_yarn_pos_embed, get_1d_yarn_pos_embed, get_1d_ntk_pos_embed
        )
        functions = [
            find_correction_factor, find_correction_range,
            linear_ramp_mask, find_newbase_ntk,
            get_1d_dype_yarn_pos_embed, get_1d_yarn_pos_embed, get_1d_ntk_pos_embed
        ]
        for fn in functions:
            sig = inspect.signature(fn)
            assert sig.return_annotation != inspect.Parameter.empty, \
                f"{fn.__name__} missing return annotation"

    def test_base_class_methods_have_annotations(self):
        import inspect
        from src.base import DyPEBasePosEmbed
        methods = ['set_timestep', '_get_mscale', 'get_components', 'forward']
        for name in methods:
            method = getattr(DyPEBasePosEmbed, name)
            sig = inspect.signature(method)
            assert sig.return_annotation != inspect.Parameter.empty, \
                f"DyPEBasePosEmbed.{name} missing return annotation"


@pytest.mark.unit
class TestPackaging:
    def test_pyproject_has_pytest_config(self):
        import tomllib
        pyproject = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        assert "pytest" in data.get("tool", {})

    def test_pyproject_has_markers(self):
        import tomllib
        pyproject = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        markers = data["tool"]["pytest"]["ini_options"]["markers"]
        assert any("comfyui_integration" in m for m in markers)

    def test_requires_comfyui_present(self):
        import tomllib
        pyproject = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        tool_comfy = data.get("tool", {}).get("comfy", {})
        assert "requires-comfyui" in tool_comfy, "Missing requires-comfyui in [tool.comfy]"

    def test_ruff_config_present(self):
        import tomllib
        pyproject = PROJECT_ROOT / "pyproject.toml"
        with open(pyproject, "rb") as f:
            data = tomllib.load(f)
        assert "ruff" in data.get("tool", {}), "Missing [tool.ruff] in pyproject.toml"


@pytest.mark.unit
class TestNodeCategory:
    def test_category_does_not_reference_unet(self):
        """DiT models are not UNets; category should not say 'unet'."""
        init_file = PROJECT_ROOT / "__init__.py"
        content = init_file.read_text(encoding="utf-8")
        assert "model_patches/unet" not in content, \
            "Category still references 'unet'"
