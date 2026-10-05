"""Tests for src/prefix_cache.py — the cascade prefix-K/V-cache switch (v2.17.0).

Qwen-Image 2.1 keys its prefix K/V cache on the target latent shape, so a
cascade (which changes shape between stages) allocates several slots.  Upstream
evicts the touched slot with ``list.remove`` over dicts holding tensors, which
raises once two or more slots exist.  The cascade nodes therefore switch the
cache off; these tests cover the switch and pin the upstream behaviour that
makes it necessary.

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.prefix_cache as pc  # noqa: E402


class FakePatcher:
    """Minimal ModelPatcher stand-in: ``clone`` deep-copies model_options."""

    def __init__(self, model_options=None):
        self.model_options = model_options if model_options is not None else {}
        self.clone_count = 0

    def clone(self):
        self.clone_count += 1
        out = FakePatcher()
        out.model_options = {
            k: (v.copy() if isinstance(v, dict) else v)
            for k, v in self.model_options.items()
        }
        return out


def _slot(key):
    return {"key": key, "blocks": {}}


def _upsert(slots, key):
    """Body of PoseBranchCache.select — verbatim eviction semantics."""
    for s in slots:
        if s["key"].shape == key.shape and torch.equal(s["key"], key):
            slots.remove(s)
            slots.append(s)
            return True
    return False


class TestDisablePrefixKvCache:
    def test_sets_device_off(self):
        out = pc.disable_prefix_kv_cache(FakePatcher())
        assert out.model_options["transformer_options"]["qwen_image21_cache"] == {
            "device": "off"
        }

    def test_clones_before_writing(self):
        original = FakePatcher()
        pc.disable_prefix_kv_cache(original)
        assert original.clone_count == 1
        assert "transformer_options" not in original.model_options

    def test_creates_transformer_options_when_absent(self):
        out = pc.disable_prefix_kv_cache(FakePatcher({}))
        assert out.model_options["transformer_options"]["qwen_image21_cache"] == {
            "device": "off"
        }

    def test_preserves_sibling_transformer_options(self):
        original = FakePatcher(
            {"transformer_options": {"rope_options": {"max_shift": 1.15}}}
        )
        out = pc.disable_prefix_kv_cache(original)
        assert out.model_options["transformer_options"]["rope_options"] == {
            "max_shift": 1.15
        }
        assert out.model_options["transformer_options"]["qwen_image21_cache"] == {
            "device": "off"
        }

    def test_overrides_an_explicitly_configured_cache(self):
        """A user-placed QwenImage21Cache node must not re-enable it mid-cascade.

        2.1 reads the option per forward, so a stale ``auto``/``gpu`` setting
        would resurrect the multi-slot eviction the cascade is avoiding.
        """
        original = FakePatcher(
            {"transformer_options": {"qwen_image21_cache": {"device": "gpu"}}}
        )
        out = pc.disable_prefix_kv_cache(original)
        assert out.model_options["transformer_options"]["qwen_image21_cache"] == {
            "device": "off"
        }

    def test_preserves_other_model_option_keys(self):
        original = FakePatcher({"sampler_pre_cfg_function": ["hook"]})
        out = pc.disable_prefix_kv_cache(original)
        assert out.model_options["sampler_pre_cfg_function"] == ["hook"]


class TestUpstreamEvictionDefect:
    """Why the switch exists — pinned so the reason cannot silently rot."""

    def test_two_slots_raise_on_tensor_dict_comparison(self):
        """The match must NOT be the head of the list.

        ``list.remove`` compares elements left-to-right and short-circuits on
        identity, so a slot at index 0 is removed without ever touching ``==``.
        The crash needs an older, non-matching slot in front of the match — which
        is exactly the LRU state a cascade builds (each shape change appends a
        slot, so the reused one ends up behind others).
        """
        older, target = _slot(torch.ones(4)), _slot(torch.zeros(4))
        slots = [older, target]
        with pytest.raises(RuntimeError, match="ambiguous"):
            _upsert(slots, torch.zeros(4))

    def test_head_slot_survives_even_with_others_present(self):
        """Control: same two slots, match at the head — no tensor comparison."""
        target, newer = _slot(torch.zeros(4)), _slot(torch.ones(4))
        slots = [target, newer]
        assert _upsert(slots, torch.zeros(4)) is True

    def test_single_slot_survives_identity_fast_path(self):
        """Why an ordinary one-prompt sampler run never trips this."""
        a = _slot(torch.zeros(4))
        slots = [a]
        assert _upsert(slots, torch.zeros(4)) is True

    def test_distinct_keys_do_not_raise(self):
        """Slot CREATION (no eviction) is fine — the crash needs a later match."""
        a, b = _slot(torch.zeros(4)), _slot(torch.ones(4))
        slots = [a]
        assert _upsert(slots, torch.ones(4)) is False
        slots.append(b)


class TestCascadeWiring:
    """Each cascade changes latent shape, so each must disable the cache."""

    @pytest.mark.parametrize(
        "module_name",
        ["hiflow", "pixelrush", "freescale"],
    )
    def test_cascade_disables_prefix_cache(self, module_name):
        import importlib
        import inspect

        mod = importlib.import_module(f"nodes.{module_name}")
        source = inspect.getsource(mod)
        assert "disable_prefix_kv_cache(model)" in source, (
            f"{module_name} must disable 2.1's prefix K/V cache — a cascade "
            f"changes the latent shape and trips the upstream eviction bug"
        )