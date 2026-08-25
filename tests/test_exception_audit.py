"""W7.1 — Broad ``except Exception`` audit guard (IMP-006, permanent).

Every broad ``except Exception`` handler in ``src/`` must carry an INTENT TAG
on the except line or the line immediately above:

* ``# probe:``   — capability detection; failure IS the answer (e.g.
  FlexAttention availability).  Narrow to a specific exception where possible.
* ``# degrade:`` — log-and-continue on an optional enhancement; the feature
  degrades gracefully and the user may or may not need to act.
* ``# leak-guard:`` — best-effort cleanup/diagnostic in OOM-adjacent or
  teardown paths; swallowing is correct-by-design.

The tag forces every future broad handler to state WHY it is broad — the
audit's purpose is not to eliminate handlers but to make each one deliberate.

Inventory at tagging time (2026-08-25): see git history for the pre-tag list;
categories: probes (~6), degrade (~5), leak-guard (~8+).
"""

import ast
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
SRC_DIR = PROJECT_ROOT / "src"

_INTENT_TAGS = ("# probe:", "# degrade:", "# leak-guard:")


def _iter_broad_handlers():
    """Yield (file, lineno) of every ``except Exception`` handler in src/."""
    for py_file in sorted(SRC_DIR.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                exc = node.type
                # Match `except Exception` (Name) and `except (Exception, ...)` (Tuple)
                names = []
                if isinstance(exc, ast.Name):
                    names = [exc.id]
                elif isinstance(exc, ast.Tuple):
                    names = [e.id for e in exc.elts if isinstance(e, ast.Name)]
                if "Exception" in names:
                    yield py_file.relative_to(PROJECT_ROOT), node.lineno


def _has_intent_tag(source_lines, lineno):
    """Check the except line and the line above for an intent tag."""
    for idx in (lineno - 1, lineno - 2):  # 0-based: line above + the line itself
        if 0 <= idx < len(source_lines):
            stripped = source_lines[idx].strip().lower()
            if any(stripped.startswith(tag.lower()) for tag in _INTENT_TAGS):
                return True
            # Tag may trail code on the same line as `except ...:`:
            if any(tag.lower() in source_lines[idx].lower() for tag in _INTENT_TAGS):
                return True
    return False


@pytest.mark.unit
class TestBroadExceptionAudit:
    def test_every_broad_handler_has_intent_tag(self):
        """All broad handlers carry # probe:/# degrade:/# leak-guard:."""
        offenders = []
        for rel_path, lineno in _iter_broad_handlers():
            lines = (PROJECT_ROOT / rel_path).read_text(encoding="utf-8").splitlines()
            if not _has_intent_tag(lines, lineno):
                offenders.append(f"{rel_path}:{lineno}")
        assert not offenders, (
            "Untagged broad `except Exception` handlers found — add an intent "
            "tag (# probe:/# degrade:/# leak-guard:) stating why the handler "
            "is broad:\n  " + "\n  ".join(offenders)
        )

    def test_inventory_is_bounded(self):
        """Sanity bound: the number of broad handlers should stay small (<40).
        A jump past this bound means broad handling is spreading — review."""
        count = sum(1 for _ in _iter_broad_handlers())
        assert count < 40, (
            f"{count} broad `except Exception` handlers in src/ — audit them "
            f"before adding more"
        )
