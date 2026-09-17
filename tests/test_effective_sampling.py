"""Tests for src/effective_sampling.py (v2.16.0, plan S1).

Covers the resolution order (patch -> backup -> live), the mock fallback and
the stale-leak detector, using fake patchers that replicate the ModelPatcher
get_model_object semantics (object_patches -> object_patches_backup -> live
attr; comfy/model_patcher.py:758-768).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.effective_sampling import (  # noqa: E402
    STALE_LEAK_CLASS_NAMES,
    effective_model_sampling,
    is_stale_dype_leak,
)


class FakeBaseModel:
    def __init__(self, model_sampling):
        self.model_sampling = model_sampling


class FakePatcher:
    """Minimal ModelPatcher stand-in with the real get_model_object order."""

    def __init__(self, live, patches=None, backup=None):
        self.model = FakeBaseModel(live)
        self.object_patches = dict(patches or {})
        self.object_patches_backup = dict(backup or {})

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        return getattr(self.model, name)


class PlainNode:
    """Object WITHOUT get_model_object (the mock-fallback path)."""

    def __init__(self, live):
        self.model = FakeBaseModel(live)


def _make_leak_class(name: str) -> type:
    return type(name, (), {})


ORIG = _make_leak_class("ModelSamplingContinuousFlow")()
LEAK = _make_leak_class("DypeModelSamplingFlux")()
PATCH = _make_leak_class("DypeModelSamplingFlux")()


class TestEffectiveModelSampling:
    def test_own_patch_wins_over_leaked_live(self):
        patcher = FakePatcher(LEAK, patches={"model_sampling": PATCH})
        assert effective_model_sampling(patcher) is PATCH

    def test_backup_used_when_no_patch(self):
        patcher = FakePatcher(LEAK, backup={"model_sampling": ORIG})
        assert effective_model_sampling(patcher) is ORIG

    def test_live_fallback_when_patch_and_backup_empty(self):
        patcher = FakePatcher(ORIG)
        assert effective_model_sampling(patcher) is ORIG

    def test_live_leak_returned_when_nothing_else_available(self):
        # Documented residual: patcher without patch/backup sees the leak —
        # the same object a stock KSampler would resolve (KSampler semantics).
        patcher = FakePatcher(LEAK)
        assert effective_model_sampling(patcher) is LEAK

    def test_plain_object_fallback(self):
        node = PlainNode(ORIG)
        assert effective_model_sampling(node) is ORIG

    def test_plain_object_without_model_attr_returns_none(self):
        class Empty:
            pass

        assert effective_model_sampling(Empty()) is None


class TestIsStaleDypeLeak:
    def test_true_for_leak_class_with_empty_patcher(self):
        assert is_stale_dype_leak(FakePatcher(LEAK)) is True

    @pytest.mark.parametrize("name", STALE_LEAK_CLASS_NAMES)
    def test_true_for_every_known_leak_class(self, name):
        leak = _make_leak_class(name)()
        assert is_stale_dype_leak(FakePatcher(leak)) is True

    def test_false_when_patcher_carries_own_patch(self):
        # A DyPE clone in THIS graph legitimately patches model_sampling —
        # the same class name live is not stale there.
        patcher = FakePatcher(PATCH, patches={"model_sampling": PATCH})
        assert is_stale_dype_leak(patcher) is False

    def test_false_when_backup_holds_the_key(self):
        patcher = FakePatcher(LEAK, backup={"model_sampling": ORIG})
        assert is_stale_dype_leak(patcher) is False

    def test_false_for_foreign_live_class(self):
        assert is_stale_dype_leak(FakePatcher(ORIG)) is False

    def test_false_without_get_model_object(self):
        # Mock safety: no resolution contract -> never claim a leak.
        assert is_stale_dype_leak(PlainNode(LEAK)) is False

    def test_false_when_live_attr_missing(self):
        class BarePatcher:
            object_patches = {}
            object_patches_backup = {}

            def get_model_object(self, name):
                return None

        assert is_stale_dype_leak(BarePatcher()) is False
