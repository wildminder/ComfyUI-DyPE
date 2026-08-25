"""W8 — Example-workflow validation + docs-claims guards (IMP-005).

Two permanent guards:

1. ``TestExampleWorkflows``: every ``example_workflows/*.json`` parses and
   references only node ids that THIS pack registers (via the V3
   ``define_schema().node_id`` of each exported node class) plus standard
   ComfyUI core nodes.  Prevents drift between shipped workflows and schemas.
2. ``TestDocsClaims``: every ``example_workflows/<name>.<ext>`` reference in
   README.md points at a file that exists on disk (kills phantom-file claims
   like the pre-fix ``SPA_basic.json`` line).

Markers: @pytest.mark.unit
"""

import json
import pathlib
import re

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent
WORKFLOWS_DIR = PROJECT_ROOT / "example_workflows"
README = PROJECT_ROOT / "README.md"

# Node ids registered by THIS pack (V3 schema exports across all sources).
# Parsed from source so the guard cannot silently rot when nodes are added.
_PACK_NODE_IDS = None


def _pack_node_ids():
    global _PACK_NODE_IDS
    if _PACK_NODE_IDS is None:
        ids = set()
        for py in PROJECT_ROOT.rglob("*.py"):
            if ".dev" in py.parts or "tmp" in py.parts:
                continue
            src = py.read_text(encoding="utf-8")
            ids.update(re.findall(r'node_id\s*=\s*"([^"]+)"', src))
        _PACK_NODE_IDS = ids
    return _PACK_NODE_IDS


# ComfyUI core nodes + well-known third-party loader nodes commonly used in
# example workflows (only what our shipped workflows actually reference).
_CORE_NODES = {
    # core sampling / io
    "CheckpointLoaderSimple", "UNETLoader", "CLIPLoader", "VAELoader",
    "DualCLIPLoader", "CLIPTextEncode", "EmptyLatentImage",
    "EmptySD3LatentImage", "KSampler", "KSamplerAdvanced", "VAEDecode",
    "VAEDecodeTiled", "VAEEncode", "SaveImage", "PreviewImage",
    "ModelSamplingFlux", "ModelSamplingAuraFlow", "LoraLoaderModelOnly",
    "ConditioningZeroOut",
    # UI-format-only helper nodes (never in API graphs)
    "Reroute", "PrimitiveNode", "Note", "MarkdownNote",
    # third-party (documented optional dependency)
    "NunchakuFluxDiTLoader",
}


@pytest.mark.unit
class TestExampleWorkflows:
    def test_workflows_dir_has_json_files(self):
        jsons = list(WORKFLOWS_DIR.glob("*.json"))
        assert jsons, "no example workflows shipped"

    @pytest.mark.parametrize("wf", sorted(
        p.name for p in WORKFLOWS_DIR.glob("*.json")))
    def test_workflow_json_parses(self, wf):
        data = json.loads((WORKFLOWS_DIR / wf).read_text(encoding="utf-8"))
        assert isinstance(data, dict), f"{wf}: not a JSON object"

    @pytest.mark.parametrize("wf", sorted(
        p.name for p in WORKFLOWS_DIR.glob("*.json")))
    def test_workflow_references_known_nodes(self, wf):
        """Every class_type / node type in the workflow must be a node this
        pack registers or a known ComfyUI core node."""
        data = json.loads((WORKFLOWS_DIR / wf).read_text(encoding="utf-8"))
        # API format: {"nodes": {id: {"type": ...}}}; UI format: list-ish dict
        types = set()
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, dict) and "class_type" in v:
                    types.add(v["class_type"])
                elif isinstance(v, dict) and v.get("type"):
                    types.add(v["type"])
            nodes_list = data.get("nodes")
            if isinstance(nodes_list, list):
                for n in nodes_list:
                    if isinstance(n, dict) and n.get("type"):
                        types.add(n["type"])
        unknown = types - _CORE_NODES - _pack_node_ids()
        assert not unknown, (
            f"{wf} references unregistered/unknown node types: {sorted(unknown)}"
        )


@pytest.mark.unit
class TestDocsClaims:
    def test_readme_workflow_file_refs_exist(self):
        """Every example_workflows/<file> mentioned in README must exist."""
        text = README.read_text(encoding="utf-8")
        refs = set(re.findall(r"example_workflows/([A-Za-z0-9_\-]+\.\w+)", text))
        missing = [r for r in sorted(refs)
                   if not (WORKFLOWS_DIR / r).exists()]
        assert not missing, (
            f"README references workflow files that do not exist: {missing}"
        )

    def test_pack_node_display_names_documented(self):
        """Each registered node's id appears somewhere in README (catches
        future undocumented nodes)."""
        text = README.read_text(encoding="utf-8")
        for node_id in sorted(_pack_node_ids()):
            assert node_id in text, (
                f"node {node_id!r} is registered but never mentioned in README"
            )
