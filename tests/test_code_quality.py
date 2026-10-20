"""Code quality meta-tests — verify structural invariants (Tier 1)."""
import ast
import pathlib
import re

import pytest

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"
TESTS_DIR = PROJECT_ROOT / "tests"

#: The markers the CI subset selects (``.github/workflows/tests.yml``:
#: ``python -m pytest -q -m "unit or mock_integration"``).  A test file that
#: carries NEITHER is silently DESELECTED by that command — the failure mode
#: this project has already been bitten by, which is why the guard below
#: exists.
CI_MARKERS = ("unit", "mock_integration")


def ci_markers_in(path: pathlib.Path) -> set:
    """The CI markers a test file actually uses.

    Read from the AST, not from the text: a marker merely MENTIONED in a
    docstring must not pass for a marker that is APPLIED.  Both spellings
    count — ``@pytest.mark.unit`` on a test and a module-level
    ``pytestmark = pytest.mark.unit``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in CI_MARKERS:
            continue
        mark = node.value
        if (isinstance(mark, ast.Attribute) and mark.attr == "mark"
                and isinstance(mark.value, ast.Name) and mark.value.id == "pytest"):
            found.add(node.attr)
    return found


def area_test_files_without_ci_marker() -> list:
    """Every AReA test file that the CI subset would silently deselect."""
    missing = []
    for path in sorted(TESTS_DIR.glob("test_area_*.py")):
        if not ci_markers_in(path):
            missing.append(path.name)
    return missing


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
class TestRetiredDebugArtifacts:
    """W9.e (NTH-106): temporary diagnostics must not linger in production."""

    def test_shape_diag_latch_retired(self):
        """The 2026-08-18 ``_shape_diag`` diagnostic latch was removed once its
        hypothesis was confirmed and fixed."""
        spa_src = (SRC_DIR / "spa.py").read_text(encoding="utf-8")
        assert "_shape_diag" not in spa_src, (
            "the retired _shape_diag diagnostic latch reappeared in spa.py"
        )

    def test_tmp_dir_is_gitignored(self):
        """Scratch files live in tmp/, which must stay untracked."""
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert re.search(r"^tmp/$", gitignore, re.MULTILINE), (
            "tmp/ is missing from .gitignore"
        )


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
            find_correction_factor,
            find_correction_range,
            find_newbase_ntk,
            get_1d_dype_yarn_pos_embed,
            get_1d_ntk_pos_embed,
            get_1d_yarn_pos_embed,
            linear_ramp_mask,
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
    def _load_pyproject(self):
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)

    def test_pyproject_has_pytest_config(self):
        data = self._load_pyproject()
        assert "pytest" in data.get("tool", {})

    def test_pyproject_has_markers(self):
        data = self._load_pyproject()
        markers = data["tool"]["pytest"]["ini_options"]["markers"]
        assert any("comfyui_integration" in m for m in markers)

    def test_requires_comfyui_present(self):
        data = self._load_pyproject()
        tool_comfy = data.get("tool", {}).get("comfy", {})
        assert "requires-comfyui" in tool_comfy, "Missing requires-comfyui in [tool.comfy]"

    def test_ruff_config_present(self):
        data = self._load_pyproject()
        assert "ruff" in data.get("tool", {}), "Missing [tool.ruff] in pyproject.toml"


@pytest.mark.unit
class TestNodeCategory:
    def test_category_does_not_reference_unet(self):
        """DiT models are not UNets; category should not say 'unet'."""
        init_file = PROJECT_ROOT / "__init__.py"
        content = init_file.read_text(encoding="utf-8")
        assert "model_patches/unet" not in content, \
            "Category still references 'unet'"


@pytest.mark.unit
class TestCiMarkerGuard:
    """Every AReA test file must be SELECTED by the CI's marker subset.

    The CI runs ``pytest -q -m "unit or mock_integration"``
    (``.github/workflows/tests.yml``).  A test file that carries neither marker
    is not a failure — it is SILENTLY deselected, so a whole file can stop
    being run without anyone noticing.  The AReA family grew by six files
    during the Krea-2 adaptation, so the guard covers all of them.
    """

    def test_the_ci_markers_are_declared_in_pyproject(self):
        """An undeclared marker is an error under ``--strict-markers``."""
        with open(PROJECT_ROOT / "pyproject.toml", "rb") as handle:
            data = tomllib.load(handle)
        markers = data["tool"]["pytest"]["ini_options"]["markers"]
        for marker in CI_MARKERS:
            assert any(marker in m for m in markers), (
                f"the {marker!r} marker is not declared in pyproject.toml"
            )

    def test_every_area_test_file_carries_a_ci_marker(self):
        """No AReA test file may fall out of the CI selection."""
        missing = area_test_files_without_ci_marker()
        assert missing == [], (
            "these AReA test files carry no @pytest.mark.unit / "
            f"@pytest.mark.mock_integration marker and are silently deselected "
            f"by the CI subset: {missing}"
        )

    def test_the_guard_covers_the_new_krea_files(self):
        """The guard's own scope includes the files the Krea-2 work added."""
        covered = {path.name for path in TESTS_DIR.glob("test_area_*.py")}
        for name in ("test_area_krea.py", "test_area_krea_seam.py",
                     "test_area_krea_seam_probe.py", "test_area_rope.py",
                     "test_area_flux_identity.py"):
            assert name in covered, f"{name} is not covered by the marker guard"
