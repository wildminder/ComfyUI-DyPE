"""Pack-layout structure guards (plan 2026-09-08: nodes/ + src/ separation).

`nodes/` — exclusively ComfyUI node definitions (schema + execute wiring).
`src/`   — exclusively engines/implementation.
Entry `__init__.py` — extension registration only.

These guards are text/AST-based where the entry module cannot be imported in
the GPU-free test env, and runtime-based where the mock-comfy conftest makes
live imports possible (same split as test_hap_node.py / test_hiflow_node.py).

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
ENTRY = PROJECT_ROOT / "__init__.py"
NODES_DIR = PROJECT_ROOT / "nodes"
NODES_INIT = NODES_DIR / "__init__.py"

ALL_NODE_CLASSES = [
    "DyPE_FLUX", "SEGA", "SPA", "HAP",
    "HAPCalibrate", "PixelRushNode", "FreeScaleNode", "HiFlowNode",
]

# Node modules that should exist once every layout-plan step has landed.
# S1 ships dype/sega/spa/hap; the *_node moves append their entries (S2/S3).
EXPECTED_NODE_MODULES = [
    "dype", "sega", "spa", "hap",
]


def _entry_src() -> str:
    return ENTRY.read_text(encoding="utf-8")


@pytest.mark.unit
class TestStructureGuard:
    def test_entry_imports_from_nodes_package(self):
        """The entry must define NO schemas itself — nodes come from .nodes."""
        src = _entry_src()
        assert "from .nodes import" in src, (
            "entry __init__.py must import node classes from the .nodes package"
        )
        assert "io.Schema(" not in src, (
            "entry __init__.py must not define schemas — node definitions "
            "live in nodes/ (pack layout plan 2026-09-08)"
        )
        assert "define_schema" not in src, (
            "entry __init__.py must not define node classes"
        )

    def test_entry_node_list_is_8_classes(self):
        """get_node_list() must return exactly the 8 registered classes."""
        src = _entry_src()
        start = src.index("async def get_node_list")
        body = src[start:]
        marker = "return ["
        bracket = body.index(marker) + len(marker) - 1
        close = body.index("]", bracket)
        listed = [c.strip() for c in body[bracket + 1:close].split(",") if c.strip()]
        assert sorted(listed) == sorted(ALL_NODE_CLASSES), (
            f"get_node_list must return exactly {sorted(ALL_NODE_CLASSES)}, "
            f"got {sorted(listed)}"
        )

    def test_nodes_init_exports_all_classes(self):
        """import nodes (flat, mock-comfy env) exposes all 8 node classes."""
        import nodes  # repo root on sys.path via conftest

        for cls in ALL_NODE_CLASSES:
            assert hasattr(nodes, cls), (
                f"nodes/__init__.py must re-export {cls}; missing"
            )
            assert getattr(nodes, cls) is not None

    def test_entry_registers_qwen2d_patch_on_load(self):
        """on_load keeps installing the Qwen2D VAE patch (entry's only job
        besides registration)."""
        src = _entry_src()
        assert "install_qwen2d_patch" in src
        assert "async def on_load" in src

    def test_pack_node_modules_exist(self):
        """Every expected node module file is present in nodes/."""
        for name in EXPECTED_NODE_MODULES:
            mod = NODES_DIR / f"{name}.py"
            assert mod.exists(), f"nodes/{name}.py missing (layout plan step incomplete)"
