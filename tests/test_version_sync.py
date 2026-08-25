"""Version-sync guard (plan 2026-08-25 W1.1, tracker CRIT-001).

``pyproject.toml`` is the SINGLE SOURCE OF TRUTH for the pack version.
The README changelog's top ``#### vX.Y.Z`` entry must mirror it.

These tests make drift impossible to merge unnoticed: any future version
bump that forgets the changelog (or vice versa) fails here — and therefore
fails CI (see .github/workflows/tests.yml, job ``meta``).

Also runnable standalone for pre-commit:

    python tests/test_version_sync.py
"""
from __future__ import annotations

import os
import re
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYPROJECT = os.path.join(_PROJECT_ROOT, "pyproject.toml")
_README = os.path.join(_PROJECT_ROOT, "README.md")

_SEMVER_RE = re.compile(r"\d+\.\d+\.\d+")
# Accept both changelog heading styles: '#### vX.Y.Z' (legacy) and
# '### vX.Y.Z' (condensed README format adopted 2026-08-25).
_CHANGELOG_ENTRY_RE = re.compile(r"^#{3,4}\s+v(\d+\.\d+\.\d+)", re.MULTILINE)


def _pyproject_version() -> str:
    """Read ``[project] version`` from pyproject.toml.

    Prefers stdlib ``tomllib`` (Python >= 3.11); falls back to a line regex
    on Python 3.10 where tomllib is unavailable.
    """
    with open(_PYPROJECT, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        import tomllib  # Python >= 3.11
    except ImportError:
        m = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        if m is None:
            raise AssertionError(
                "pyproject.toml has no parseable 'version = \"...\"' line"
            )
        return m.group(1)
    data = tomllib.loads(text)
    version = data.get("project", {}).get("version")
    if not version:
        raise AssertionError("pyproject.toml [project] table has no 'version'")
    return version


def _changelog_versions() -> list:
    """Return every ``vX.Y.Z`` from the README changelog, top-first."""
    with open(_README, "r", encoding="utf-8") as fh:
        text = fh.read()
    return _CHANGELOG_ENTRY_RE.findall(text)


def test_pyproject_version_matches_changelog():
    """CRIT-001 guard: pyproject version == top README changelog entry."""
    py_version = _pyproject_version()
    entries = _changelog_versions()
    assert entries, (
        "README.md contains no '### vX.Y.Z' / '#### vX.Y.Z' changelog "
        "entries; add one matching the pyproject.toml version."
    )
    top = entries[0]
    assert py_version == top, (
        f"VERSION DRIFT detected:\n"
        f"  pyproject.toml version : {py_version!r}\n"
        f"  README changelog top   : v{top!r}\n"
        f"Fix: bump pyproject.toml to {top} OR add a matching "
        f"'#### v{py_version}' changelog entry at the top of the README "
        f"Changelog section."
    )


def test_version_is_semver():
    """Both versions must be full X.Y.Z semver strings (catches '2.8' typos)."""
    py_version = _pyproject_version()
    entries = _changelog_versions()
    assert _SEMVER_RE.fullmatch(py_version), (
        f"pyproject.toml version {py_version!r} is not X.Y.Z semver."
    )
    for entry in entries:
        assert _SEMVER_RE.fullmatch(entry), (
            f"README changelog entry v{entry!r} is not X.Y.Z semver."
        )


def test_changelog_entries_are_descending():
    """Changelog entries must be strictly descending top->bottom.

    Catches inserting a new release entry in the wrong place (e.g. below an
    older entry), which would silently break the parity guard's meaning.
    """
    entries = _changelog_versions()
    assert len(entries) >= 2, "expected at least two changelog entries"

    def key(v):
        return tuple(int(p) for p in v.split("."))

    for prev, cur in zip(entries, entries[1:]):
        assert key(prev) > key(cur), (
            f"Changelog order violation: v{prev} must be NEWER than the "
            f"entry directly below it (v{cur}). Newest releases go on TOP."
        )


if __name__ == "__main__":  # pre-commit / standalone runner
    failures = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures.append((name, str(exc)))
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
