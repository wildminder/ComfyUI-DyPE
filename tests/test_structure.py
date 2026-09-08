"""Pack-layout structure guards (plan 2026-09-08: nodes/ + src/ separation).

`nodes/` — exclusively ComfyUI node definitions (schema + execute wiring).
`src/`   — exclusively engines/implementation.
Entry `__init__.py` — extension registration only.

These guards are text/AST-based where the entry module cannot be imported in
the GPU-free test env, and runtime-based where the mock-comfy conftest makes
live imports possible (same split as test_hap_node.py / test_hiflow_node.py).
The loader regression test replicates ComfyUI's ``load_custom_node``
(``spec_from_file_location`` on the pack ``__init__.py``, executed as a
module in ``sys.modules``) — the mechanism verified empirically 2026-09-08.

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import asyncio
import importlib.util
import pathlib
import sys
import types

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
ENTRY = PROJECT_ROOT / "__init__.py"
NODES_DIR = PROJECT_ROOT / "nodes"
NODES_INIT = NODES_DIR / "__init__.py"
SRC_DIR = PROJECT_ROOT / "src"

ALL_NODE_CLASSES = [
    "DyPE_FLUX", "SEGA", "SPA", "HAP",
    "HAPCalibrate", "PixelRushNode", "FreeScaleNode", "HiFlowNode",
]

# Expected node modules once every layout-plan step has landed.
EXPECTED_NODE_MODULES = [
    "dype", "sega", "spa", "hap",
    "hap_calibrate", "freescale", "pixelrush", "hiflow",
]

_OLD_CATEGORIES = ("model_patches/position_encoding", "image/upscaling")


def _entry_src() -> str:
    return ENTRY.read_text(encoding="utf-8")


def _fake_mod(name, **attrs):
    """Additive-only mock module (pattern from tests/test_qwen2d_vae.py):
    never replaces the root conftest's comfy mocks, registers in sys.modules
    and wires the parent attribute."""
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    parent, _, child = name.rpartition(".")
    if parent and parent in sys.modules:
        setattr(sys.modules[parent], child, m)
    return m


def _ensure_entry_comfy_chain():
    """The pack entry imports src.qwen2d_vae_patch, which imports the comfy
    chain (comfy.sd, comfy.model_management, ...). Install additive mocks
    so the loader replica can execute the entry in the GPU-free env."""
    _fake_mod("comfy.sd", VAE=type("VAE", (), {}))
    _fake_mod("comfy.model_management",
              is_amd=lambda: False, dtype_size=lambda d: 4,
              OOM_EXCEPTION=RuntimeError)
    _fake_mod("comfy.utils")
    _fake_mod("comfy.ops", disable_weight_init=object)
    _fake_mod("comfy.ldm")
    _fake_mod("comfy.ldm.modules")
    _fake_mod("comfy.ldm.modules.diffusionmodules")
    _fake_mod("comfy.ldm.modules.diffusionmodules.model",
              vae_attention=lambda *a, **k: None)


def _load_entry_via_comfy_loader():
    """Replicate ComfyUI nodes.load_custom_node for a DIRECTORY pack:
    spec_from_file_location(sys_module_name, <dir>/__init__.py), module in
    sys.modules, exec. Returns the loaded module (caller pops sys.modules)."""
    sys_module_name = str(PROJECT_ROOT).replace(".", "_x_")
    module_spec = importlib.util.spec_from_file_location(
        sys_module_name, str(PROJECT_ROOT / "__init__.py")
    )
    _ensure_entry_comfy_chain()
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[sys_module_name] = module
    module_spec.loader.exec_module(module)
    return module


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

    def test_nodes_dir_exclusively_nodes(self):
        """Every nodes/*.py module defines an io.ComfyNode subclass; src/
        never defines a node schema."""
        for name in EXPECTED_NODE_MODULES:
            src = (NODES_DIR / f"{name}.py").read_text(encoding="utf-8")
            assert "io.ComfyNode" in src or "_ComfyNodeBase = io.ComfyNode" in src, (
                f"nodes/{name}.py must define an io.ComfyNode node class"
            )
        for py in SRC_DIR.rglob("*.py"):
            src = py.read_text(encoding="utf-8")
            assert "io.Schema(" not in src, (
                f"engine file {py.name} must not define node schemas — "
                f"node definitions live in nodes/ (layout plan 2026-09-08)"
            )

    def test_category_wmnodes_everywhere(self):
        """Every node module pins category=\"WMNodes/image\"."""
        for name in EXPECTED_NODE_MODULES:
            src = (NODES_DIR / f"{name}.py").read_text(encoding="utf-8")
            assert 'category="WMNodes/image"' in src, (
                f"nodes/{name}.py must use category=\"WMNodes/image\""
            )

    def test_no_category_strings_in_engines(self):
        """No category strings (old or new) may appear in src/."""
        for py in SRC_DIR.rglob("*.py"):
            src = py.read_text(encoding="utf-8")
            for cat in _OLD_CATEGORIES + ("WMNodes/image",):
                assert cat not in src, f"{py.name} mentions category {cat!r}"

    def test_no_legacy_node_files_in_src(self):
        """src/ holds no *_node.py leftovers."""
        leftovers = sorted(str(p.name) for p in SRC_DIR.glob("*_node.py"))
        assert not leftovers, f"legacy node modules remain in src/: {leftovers}"


@pytest.mark.unit
class TestCategoryGrepGuard:
    def test_no_old_category_strings_repo_wide(self):
        """No tracked file mentions the old category paths. Uses git
        ls-files so untracked/ignored files never mask a miss."""
        import subprocess

        out = subprocess.run(
            ["git", "ls-files"], cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        offenders = []
        for rel in out:
            if rel.replace("\\", "/") == "tests/test_structure.py":
                continue  # this guard holds the reference strings
            if not rel.endswith((".py", ".md", ".toml", ".yml", ".json")):
                continue
            path = PROJECT_ROOT / rel
            if not path.exists():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for cat in _OLD_CATEGORIES:
                if cat in text:
                    offenders.append(f"{rel}: {cat}")
        assert not offenders, f"old category strings remain: {offenders}"


@pytest.mark.unit
class TestComfyLoaderRegression:
    """Replicates ComfyUI's load_custom_node for a DIRECTORY pack and pins
    that the pack keeps loading through that exact mechanism."""

    def test_entry_exposes_v3_contract(self):
        """Loader mechanics: module loads, exports comfy_entrypoint +
        DyPEExtension."""
        sys_module_name = str(PROJECT_ROOT).replace(".", "_x_")
        try:
            module = _load_entry_via_comfy_loader()
            assert hasattr(module, "comfy_entrypoint"), (
                "pack entry must export comfy_entrypoint() for the V3 loader"
            )
            assert hasattr(module, "DyPEExtension"), (
                "pack entry must export the DyPEExtension class"
            )
        finally:
            sys.modules.pop(sys_module_name, None)

    def test_entrypoint_returns_8_node_extension(self):
        """comfy_entrypoint() → get_node_list() yields the 8 node classes
        (runtime import through loader mechanics, mock-comfy env)."""
        sys_module_name = str(PROJECT_ROOT).replace(".", "_x_")
        try:
            module = _load_entry_via_comfy_loader()
            ext = asyncio.run(module.comfy_entrypoint())
            nodes_list = asyncio.run(ext.get_node_list())
            names = sorted(c.__name__ for c in nodes_list)
            assert names == sorted(ALL_NODE_CLASSES), (
                f"loaded pack registers {names}, expected {sorted(ALL_NODE_CLASSES)}"
            )
        finally:
            sys.modules.pop(sys_module_name, None)

    def test_extension_on_load_installs_qwen2d_patch(self):
        """on_load() must install the Qwen2D VAE patch (monkeypatched to
        observe the call; the real patcher is engine-level and covered by
        test_qwen2d_vae.py)."""
        sys_module_name = str(PROJECT_ROOT).replace(".", "_x_")
        try:
            module = _load_entry_via_comfy_loader()
            ext = asyncio.run(module.comfy_entrypoint())
            called = {}

            import src.qwen2d_vae_patch as qwen_patch
            orig = qwen_patch.install_qwen2d_patch

            def spy():
                called["yes"] = True
                return orig()

            # The loaded entry module resolved install_qwen2d_patch into ITS
            # namespace at import time (from .src.qwen2d_vae_patch import);
            # on_load() looks the name up there. Patch BOTH namespaces.
            qwen_patch.install_qwen2d_patch = spy
            entry_ns_orig = getattr(module, "install_qwen2d_patch", None)
            module.install_qwen2d_patch = spy
            try:
                asyncio.run(ext.on_load())
            finally:
                qwen_patch.install_qwen2d_patch = orig
                if entry_ns_orig is not None:
                    module.install_qwen2d_patch = entry_ns_orig
            assert called.get("yes") is True, (
                "extension on_load() did not call install_qwen2d_patch"
            )
        finally:
            sys.modules.pop(sys_module_name, None)
