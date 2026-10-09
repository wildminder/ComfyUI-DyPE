"""README documentation pins for I-Max Z-Image support (plan 2026-10-06, P5).

v2.19.0 widened I-Max from FLUX-only to FLUX + Z-Image (nodes/imax.py gate,
per-group NTK clip, ``optimized_attention_override`` seam). These tests pin
the USER-FACING documentation of that widening: the support-matrix cell, the
I-Max section's Z-Image paragraph, and the changelog entry — so a later edit
cannot silently revert the docs to the FLUX-only claim.

Version parity itself is owned by ``tests/test_version_sync.py`` and
``tests/test_hiflow_node.py::test_version_bumped``; these pins stay
version-agnostic where possible (a ``>= (2, 19, 0)`` floor, not a literal)
so a future patch release does not break them.
"""
from __future__ import annotations

import pathlib
import re

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).parent.parent
_README_PATH = _PROJECT_ROOT / "README.md"
_PYPROJECT_PATH = _PROJECT_ROOT / "pyproject.toml"


def _readme() -> str:
    return _README_PATH.read_text(encoding="utf-8")


def _pyproject_version() -> str:
    text = _PYPROJECT_PATH.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert m, "pyproject.toml has no parseable 'version = \"...\"' line"
    return m.group(1)


def _imax_section() -> str:
    """The I-Max section body (last anchored section, up to the next '## ')."""
    m = re.search(r'<a id="user-content-imax"></a>(.*?)\n## ',
                  _readme(), re.DOTALL)
    assert m, "README I-Max section (user-content-imax anchor) not found"
    return m.group(1)


@pytest.mark.unit
class TestIMaxZImageDocs:
    def test_version_is_bumped(self):
        """pyproject carries the Z-Image release version (2.19.0 or newer)."""
        version = _pyproject_version()
        key = tuple(int(p) for p in version.split("."))
        assert key >= (2, 19, 0), (
            f"pyproject.toml version {version!r} predates the I-Max "
            "Z-Image release (2.19.0)."
        )

    def test_readme_changelog_entry_exists(self):
        """The v2.19.0 changelog entry exists and covers the Z-Image support."""
        m = re.search(r"^### v2\.19\.0\b.*?(?=\n### |\Z)",
                      _readme(), re.DOTALL | re.MULTILINE)
        assert m, "README changelog has no '### v2.19.0' entry"
        entry = m.group(0)
        assert "Z-Image" in entry, (
            "the v2.19.0 changelog entry does not mention Z-Image"
        )

    def test_readme_matrix_documents_zimage_imax(self):
        """Support matrix + methods table no longer claim FLUX-only I-Max."""
        readme = _readme()
        # "Which method when?" methods table lists Z-Image next to FLUX.
        row = re.search(r"^\| \*\*I-Max\*\* \| (.+?) \|",
                        readme, re.MULTILINE)
        assert row, "methods table has no I-Max row"
        assert "FLUX" in row.group(1) and "Z-Image" in row.group(1), (
            f"I-Max methods-table row lists {row.group(1)!r}, expected "
            "FLUX and Z-Image"
        )
        # Model-support matrix: locate the I-Max column from the header row,
        # then read the Z-Image row's cell.
        lines = readme.splitlines()
        header_idx = next(
            (i for i, ln in enumerate(lines)
             if ln.startswith("| Architecture |")),
            None,
        )
        assert header_idx is not None, "support matrix header row not found"
        header = [c.strip() for c in lines[header_idx].split("|")]
        imax_col = header.index("I-Max")
        zimage_row = next(
            (ln for ln in lines[header_idx + 1:]
             if ln.startswith("| Z-Image")),
            None,
        )
        assert zimage_row, "support matrix has no Z-Image row"
        cells = [c.strip() for c in zimage_row.split("|")]
        imax_cell = cells[imax_col]
        assert imax_cell.startswith("✅"), (
            f"I-Max cell for the Z-Image row is {imax_cell!r}, expected ✅"
        )
        # The cell claims support for the SHARED "Z-Image, Anima/Cosmos" row;
        # the * footnote must narrow it to Z-Image or the row over-claims.
        assert imax_cell.endswith("*"), (
            f"I-Max cell for the Z-Image row ({imax_cell!r}) must carry the "
            "'*' footnote marker"
        )
        assert "\\* **Z-Image only**" in readme, (
            "support matrix is missing the '* Z-Image only' footnote that "
            "narrows the shared Z-Image/Anima row"
        )

    def test_readme_imax_section_has_zimage_note(self):
        """I-Max section documents the Z-Image wiring and drops FLUX-only."""
        section = _imax_section()
        assert "Z-Image" in section, "I-Max section does not mention Z-Image"
        # The per-arch behavior paragraph must name the two skipped extras.
        assert "cross-attention" in section, (
            "I-Max section does not explain why text duplication is "
            "skipped for Z-Image (Lumina-Next cross-attention)"
        )
        assert "guidance-distilled" in section, (
            "I-Max section does not explain why the guidance embeds are "
            "FLUX-only (Z-Image is guidance-distilled)"
        )
        # The v2.18.0 FLUX-only rejection claim must not survive the widening.
        assert "Non-FLUX models are rejected" not in section, (
            "I-Max section still claims non-FLUX models are rejected"
        )
