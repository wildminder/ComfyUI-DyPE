"""Code quality meta-tests — verify structural invariants (Tier 1)."""
import ast
import pathlib
import re
import subprocess
import sys

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

#: The Qwen-Image-2.1 test files the CI subset must SELECT.  The 2.1
#: adaptation (S1–S11) added eight of them; a file that carries NEITHER CI
#: marker is silently DESELECTED by the CI's ``-m "unit or mock_integration"``
#: — a test that stops running without a single failure.  Every one of them is
#: named here so that the hazard is checked EXPLICITLY, not only through the
#: ``test_area_*`` glob.
NEW_QWEN21_FILES = (
    "test_area_qwen21_seam.py",          # S1  the seam-verification harness
    "test_area_qwen21_geometry.py",      # S2  the geometry constants
    "test_area_qwen21_adapter.py",       # S3  the 3-D <-> 4-D adapter
    "test_area_qwen21_rope.py",          # S4  the rope recorder
    "test_area_qwen21_phase1.py",        # S6/S11  the Phase I seam
    "test_area_qwen21_phase2.py",        # S8/S11  the Phase II seam
    "test_area_qwen21_rejections.py",    # S9   the rejection contract
    "test_area_attn_text_keys_only.py",  # S7   the engine keyword
)

#: The byte-identity goldens of the pre-2.1 baseline — the five seam answers
#: that must not move while the 2.1 seam is added.  They live in
#: ``tests/test_area_grid_fallback.py``'s ``GOLDENS`` and, as a SECOND copy,
#: in ``tests/test_area_flux_identity.py``; the values are repeated HERE so
#: that an edit of EITHER copy cannot silently bless a regression.  The recipe
#: is the sha256 of a tensor's contiguous float32 bytes, truncated to 32 hex
#: characters.  The BEHAVIOURAL re-derivation lives in the two files named
#: above — this is the fast, structural pin.
BASELINE_GOLDENS = {
    "flux80_coarse": "dca3a04b5655b81b6a40049a010c0d94",
    "flux80_fine": "1c5d27e285761eef13ec76dc10638022",
    "flux160_coarse": "52ed0106c10bd149ea66b16728efd973",
    "zimage65_coarse": "394b3e8a2529bb35436fbaea4618d808",
    "zimage65_fine": "b16a038f4050d776d9c77a4a91192d0a",
}


def module_literal(path: pathlib.Path, name: str):
    """A module-level constant of a Python file, read from its AST.

    ``ast.literal_eval`` — the constants are plain literals, so reading them
    needs neither an import nor torch.  A missing constant is a FAILURE, not
    a ``None``: a renamed table must not pass for an empty one.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level constant of {path.name}")


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


@pytest.mark.unit
class TestTheQwen21CiMarkerGuard:
    """The Qwen-Image-2.1 files must be SELECTED by the CI's marker subset.

    The Krea-2 precedent, in the same shape: the static checks below prove
    the marker is APPLIED, the dynamic one runs the CI's OWN command over
    the new files and proves the tests are actually COLLECTED — a marker on
    the wrong node, or a file pytest does not collect at all, would slip past
    the static check.
    """

    def test_the_guard_covers_the_new_qwen21_files(self):
        """Every 2.1 test file is inside the guard's scope."""
        covered = {path.name for path in TESTS_DIR.glob("test_area_*.py")}
        for name in NEW_QWEN21_FILES:
            assert name in covered, f"{name} is not covered by the marker guard"

    def test_every_new_qwen21_file_carries_a_ci_marker(self):
        """Each 2.1 test file carries one of the CI markers.

        The existing guard (``test_every_area_test_file_carries_a_ci_marker``)
        covers the same ground through the ``test_area_*`` glob; this spells
        the eight names out, so a file that is RENAMED out of the glob (or
        added without being listed above) fails HERE, by name.
        """
        for name in NEW_QWEN21_FILES:
            path = TESTS_DIR / name
            assert path.exists(), f"{name} is missing from {TESTS_DIR}"
            assert ci_markers_in(path), (
                f"{name} carries no @pytest.mark.unit / "
                "@pytest.mark.mock_integration marker and is silently "
                "deselected by the CI subset"
            )

    def test_the_marker_subset_selects_every_new_test(self):
        """The CI's OWN selection picks every new test up.

        The static checks above prove the marker TEXT is there; this runs the
        CI's own command (``-m "unit or mock_integration"``) over the new
        files and proves the tests are actually COLLECTED.
        """
        command = [sys.executable, "-m", "pytest", "--collect-only", "-q",
                   "-p", "no:cacheprovider",
                   # the project's addopts carry ``-v`` (pyproject.toml), which
                   # would turn the node-id list into a tree; only the import
                   # mode matters for the collection
                   "-o", "addopts=--import-mode=importlib",
                   "-m", "unit or mock_integration",
                   *[f"tests/{name}" for name in NEW_QWEN21_FILES]]
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                                text=True)
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "deselected" not in output, output
        for name in NEW_QWEN21_FILES:
            assert f"tests/{name}::" in output, (
                f"{name} was not selected by the CI marker subset:\n{output}"
            )


@pytest.mark.unit
class TestTheFinalGoldensGate:
    """The pre-2.1 byte-identity goldens are still the baseline ones.

    The 2.1 work touched the SHARED engine (``windowed_attention``'s
    ``text_keys_only`` keyword) and added a THIRD seam, so the FLUX / Z-Image
    answers are the canary.  Their BEHAVIOURAL re-derivation lives in
    ``tests/test_area_flux_identity.py`` and in
    ``tests/test_area_qwen21_phase2.py``; the checks below are the fast,
    structural pin: the golden table and its second copy must still carry the
    documented values, so an edit of either one cannot silently bless a
    regression.
    """

    def test_the_golden_table_still_carries_the_baseline_hashes(self):
        """``tests/test_area_grid_fallback.py``'s GOLDENS is the baseline."""
        goldens = module_literal(TESTS_DIR / "test_area_grid_fallback.py",
                                 "GOLDENS")
        for tag, digest in BASELINE_GOLDENS.items():
            assert goldens.get(tag) == digest, (
                f"the {tag} golden moved: {goldens.get(tag)} != {digest}"
            )

    def test_the_two_golden_copies_agree(self):
        """The golden table and the identity file's copy do not drift.

        ``tests/test_area_flux_identity.py`` keeps its OWN copy of the five
        hashes, so that editing the table cannot bless a regression; the two
        copies must hold the same values.
        """
        table = module_literal(TESTS_DIR / "test_area_grid_fallback.py",
                               "GOLDENS")
        identity = dict(module_literal(TESTS_DIR / "test_area_flux_identity.py",
                                       "FLUX_GOLDENS"))
        identity.update(module_literal(TESTS_DIR / "test_area_flux_identity.py",
                                       "ZIMAGE_GOLDENS"))
        for tag in BASELINE_GOLDENS:
            assert identity.get(tag) == table.get(tag), (
                f"the two golden copies disagree on {tag}: "
                f"{identity.get(tag)} != {table.get(tag)}"
            )
