"""Tests for plan persistence + output-dir resolution (plan P4: T4.1-T4.2).

Markers: @pytest.mark.unit
Accept (user-run):
    pytest tests/test_hap_calib_io.py -q
"""

import json
import os
import sys

import pytest

from src.hap import ScopePlan
from src.hap_calib_node import resolve_output_dir, write_scope_plan


def _tiny_plan_dict():
    return {
        "alphas": [[0.0, 0.0], [0.0, 0.0]],
        "betas": [[0.5, 1.0], [0.25, 0.75]],
    }


# ---------------------------------------------------------------------------
# T4.1 — write_scope_plan
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestWriteScopePlan:
    def test_writes_valid_json_roundtrip(self, tmp_path):
        plan_dict = _tiny_plan_dict()
        path = write_scope_plan(plan_dict, str(tmp_path), "plan.json")
        assert os.path.isabs(path)
        assert os.path.exists(path)
        # Round-trips through ScopePlan.load.
        plan = ScopePlan.load(path)
        assert plan.num_layers == 2
        assert plan.num_heads == 2
        # JSON content matches.
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        assert d == plan_dict

    def test_creates_nested_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        path = write_scope_plan(_tiny_plan_dict(), str(nested), "plan.json")
        assert os.path.exists(path)

    def test_appends_json_extension(self, tmp_path):
        path = write_scope_plan(_tiny_plan_dict(), str(tmp_path), "myplan")
        assert path.endswith("myplan.json")

    @pytest.mark.parametrize("bad_name", ["a/b.json", "a\\b.json", "../x.json"])
    def test_rejects_path_separators(self, tmp_path, bad_name):
        with pytest.raises(ValueError, match="path"):
            write_scope_plan(_tiny_plan_dict(), str(tmp_path), bad_name)

    def test_writes_excluded_head_counts_roundtrip(self, tmp_path):
        """A plan WITH excluded_head_counts persists the field and reloads it
        (2026-08-23 head-count warning fix)."""
        plan_dict = dict(_tiny_plan_dict())
        plan_dict["excluded_head_counts"] = [20]
        path = write_scope_plan(plan_dict, str(tmp_path), "plan.json")
        plan = ScopePlan.load(path)
        assert plan.excluded_head_counts == [20]
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        assert d["excluded_head_counts"] == [20]


# ---------------------------------------------------------------------------
# T4.2 — resolve_output_dir fallback
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolveOutputDir:
    def test_fallback_when_folder_paths_missing(self, monkeypatch):
        """Without ``folder_paths`` the resolver falls back to <pack>/tmp."""
        # Ensure folder_paths is NOT importable.
        monkeypatch.setitem(sys.modules, "folder_paths", None)
        # Force re-import failure by removing any cached module.
        saved = sys.modules.pop("folder_paths", None)
        try:
            # Make import fail.
            import builtins
            real_import = builtins.__import__

            def fake_import(name, *args, **kwargs):
                if name == "folder_paths":
                    raise ImportError("no folder_paths")
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", fake_import)
            out = resolve_output_dir()
            assert out.endswith("tmp")
        finally:
            if saved is not None:
                sys.modules["folder_paths"] = saved

    def test_uses_folder_paths_when_available(self, monkeypatch):
        import types
        fake = types.ModuleType("folder_paths")
        fake.get_output_directory = lambda: "/fake/output"
        monkeypatch.setitem(sys.modules, "folder_paths", fake)
        assert resolve_output_dir() == "/fake/output"
