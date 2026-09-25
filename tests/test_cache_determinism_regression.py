"""Cache-determinism regression: the reported Krea2 repro, simulated (v2.16.0, plan S3).

User report (2026-09-17): with identical HiFlow parameters, results differ
unless model AND node caches are cleared before each run; the HiFlow console
shows a different stage entry sigma (0.3872 vs 0.5055) for identical params.

Root cause (verified against the installed ComfyUI source, see
src/effective_sampling.py docstring): object patches applied by a patch-node
clone STAY on the shared BaseModel after that clone is superseded
(``load_models_gpu`` detaches stale entries with ``detach(unpatch_all=False)``,
model_management.py:962; ``partially_unload`` never restores object patches).
Restore-to-original only happens through ``object_patches_backup`` — a dict
SHARED between a patcher and its clones (model_patcher.py:428/462) — so a
patcher whose own backup is empty and that carries no ``model_sampling`` patch
resolves the live (leaked) attribute.

This file encodes those semantics in a FakeComfyLifecycle harness and asserts
the v2.16.0 contract: HiFlow's resolved model_sampling / stage schedule is a
function of the GRAPH's patch chain, not of what ran earlier in the session —
even when the live BaseModel attribute has diverged.

If a future ComfyUI changes these lifecycle semantics, only THIS harness needs
updating.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import nodes.hiflow as hfn  # noqa: E402
from src.effective_sampling import effective_model_sampling  # noqa: E402
from src.hiflow import build_stage_sigmas  # noqa: E402


# ---------------------------------------------------------------------------
# Fake ComfyUI lifecycle
# ---------------------------------------------------------------------------

class FakeBaseModel:
    """The SHARED BaseModel: object patches mutate its live attributes."""

    def __init__(self, model_sampling):
        self.model_sampling = model_sampling


class FakeModelPatcher:
    """ModelPatcher stand-in with the lifecycle semantics that matter.

    - clone() shares the parent's object_patches_backup dict object (via
      get_clone_model_override, model_patcher.py:428 + clone 462) and copies
      object_patches;
    - patch_model() applies object patches to the SHARED base model, backing
      up only keys not already backed up (model_patcher.py:1113-1120);
    - unpatch_model() restores the backup and CLEARS it (1165-1169);
    - detach(unpatch_all=False) skips the unpatch -> leak (called from
      model_management.py:962).
    """

    def __init__(self, base_model, object_patches=None):
        self.model = base_model
        self.object_patches = dict(object_patches or {})
        self.object_patches_backup = {}  # replaced by SHARED dict on clone()

    def clone(self):
        child = FakeModelPatcher(self.model, self.object_patches)
        child.object_patches_backup = self.object_patches_backup  # SHARED
        return child

    def patch_model(self):
        for key, patch in self.object_patches.items():
            old = getattr(self.model, key)
            if key not in self.object_patches_backup:
                self.object_patches_backup[key] = old
            setattr(self.model, key, patch)

    def unpatch_model(self):
        for key in list(self.object_patches_backup.keys()):
            setattr(self.model, key, self.object_patches_backup[key])
        self.object_patches_backup.clear()

    def partially_load(self):
        # partially_load = unpatch(own backup) + patch(own patches)
        # (model_patcher.py:1256+). A patcher with no patch for a key and an
        # empty backup restores nothing for it.
        self.unpatch_model()
        self.patch_model()

    def detach(self, unpatch_all=True):
        if unpatch_all:
            self.unpatch_model()

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        return getattr(self.model, name)


class FakeLoadedModels:
    """load_models_gpu semantics (model_management.py:913-1014).

    Identity-keyed entries; stale same-base entries are detached with
    ``unpatch_all=False`` (LEAK); the requested patcher always runs
    model_load -> partially_load (re-applies ITS OWN patches).
    """

    def __init__(self):
        self.entries = []

    def load_models_gpu(self, patcher):
        stale = [e for e in self.entries
                 if e is not patcher and e.model is patcher.model]
        for e in stale:
            e.detach(unpatch_all=False)  # LEAK by design
            self.entries.remove(e)
        if patcher not in self.entries:
            self.entries.append(patcher)
        patcher.partially_load()


def _install_fake_calculate_sigmas(monkeypatch):
    """comfy.samplers.calculate_sigmas stand-in: indexes the sampling's own
    table so the schedule genuinely depends on WHICH object is resolved."""

    def calculate_sigmas(ms, scheduler, steps):
        assert scheduler == "simple"
        table = getattr(ms, "sigma_table")
        idx = torch.linspace(0, len(table) - 1, steps + 1).round().long()
        return table[idx]

    fake = types.ModuleType("comfy.samplers")
    fake.calculate_sigmas = calculate_sigmas
    monkeypatch.setitem(sys.modules, "comfy.samplers", fake)
    comfy_mod = sys.modules.get("comfy")
    if comfy_mod is not None:
        monkeypatch.setattr(
            comfy_mod, "samplers", fake, raising=False)


# ---------------------------------------------------------------------------
# Sampling stand-ins with distinct, measurable schedules
# ---------------------------------------------------------------------------

def _make_sampling(name, shift):
    class _Sampling:
        def __init__(self, shift):
            self.shift = shift
            t = torch.linspace(1.0, 0.0, 101)
            self.sigma_table = shift * t / (1.0 + (shift - 1.0) * t)

        def timestep(self, sigma):
            return sigma * 1000.0

    _Sampling.__name__ = name
    return _Sampling(shift)


def _checkpoint_patcher(shift=2.0):
    """A fresh Load-Checkpoint output: patcher over a fresh BaseModel."""
    return FakeModelPatcher(
        FakeBaseModel(_make_sampling("ModelSamplingContinuousFlow", shift)))


def _install_dype_patch(patcher):
    """Stand-in for apply_dype_to_model's schedule patch (patch_utils.py:166-171):
    a NEW clone + an object patch whose shift differs from the original."""
    patcher.object_patches["model_sampling"] = _make_sampling(
        "DypeModelSamplingFlux", 3.5)


def _stage_entry_sigma(model, steps=30, tau=0.6):
    """What HiFlow's console line reports: the guided stage's entry sigma."""
    base_sigmas = hfn._base_sigmas(model, steps)
    stage = build_stage_sigmas(base_sigmas, tau, 16)
    return float(stage[0])


# ---------------------------------------------------------------------------
# The reported repro
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReportedRepro:
    """Run1 (plain checkpoint) -> Run2 (DyPE clone, different schedule) ->
    Run3 (restore run-1 graph, caches NOT cleared).

    Contract delivered by v2.16.0:
    - a patcher with its OWN patch (DyPE/SEGA in the graph) resolves that
      patch — always, regardless of live-attr history;
    - an unpatched patcher resolves the backup original when the lineage
      holds one (the ComfyUI shared-backup heal path, model_patcher.py:428);
    - ONLY the un-healable residue (no patch, empty backup, live leak — a
      live attr inherited from outside the patcher's lineage) falls back to
      the live attr, exactly what a stock KSampler on the same input sees;
      the S6 stale-leak warning flags that case instead of silent drift.
    """

    def test_p0_reload_heals_via_shared_backup(self, monkeypatch):
        """Run3 re-requests the SAME checkpoint patcher object (the normal
        Load-Checkpoint cache hit): the shared backup restores the original
        and the schedule matches run 1."""
        _install_fake_calculate_sigmas(monkeypatch)
        loaded = FakeLoadedModels()

        p1 = _checkpoint_patcher()
        loaded.load_models_gpu(p1)
        s1 = _stage_entry_sigma(p1)

        p2 = p1.clone()
        _install_dype_patch(p2)
        loaded.load_models_gpu(p2)
        s2 = _stage_entry_sigma(p2)

        # Run 3: the cached checkpoint patcher object again (node cache hit),
        # DyPE node removed from the graph.
        loaded.load_models_gpu(p1)
        s3 = _stage_entry_sigma(p1)

        assert s2 != s1, "harness sanity: the patched schedule must differ"
        assert s3 == pytest.approx(s1), (
            "re-loading the checkpoint patcher must restore its original "
            "schedule via the shared backup (unpatch_model on partially_load)"
        )

    def test_restored_dype_params_resolve_own_schedule(self, monkeypatch):
        """Run3 re-executes the patch node with restored params (a NEW clone):
        its own patch decides the schedule — not the live attr left by run 2's
        clone, which had DIFFERENT shift parameters."""
        _install_fake_calculate_sigmas(monkeypatch)
        loaded = FakeLoadedModels()
        p1 = _checkpoint_patcher()
        loaded.load_models_gpu(p1)

        # Run 2: DyPE clone at shift 3.5.
        p2 = p1.clone()
        _install_dype_patch(p2)
        loaded.load_models_gpu(p2)
        s2 = _stage_entry_sigma(p2)

        # Run 3: restored params -> fresh clone whose patch is re-derived for
        # the restored parameters. The live attr is POISONED with a foreign
        # schedule (as if another run's patch had leaked): resolution must
        # ignore it.
        p3 = p1.clone()
        _install_dype_patch(p3)
        p3.model.model_sampling = _make_sampling("Leaked", 0.5)
        loaded.load_models_gpu(p3)
        s3 = _stage_entry_sigma(p3)

        assert s3 == pytest.approx(s2), (
            "a patcher with its own schedule patch must resolve that patch "
            "regardless of the live attribute's history"
        )

    def test_unhealable_live_leak_is_ksampler_consistent(self, monkeypatch):
        """The documented residual: a patcher with no patch and an empty
        backup over a leaked base resolves the live attr — the SAME schedule
        a stock KSampler on that input would use (consistent, not silent
        divergence between our node and ComfyUI's own)."""
        _install_fake_calculate_sigmas(monkeypatch)
        loaded = FakeLoadedModels()
        p1 = _checkpoint_patcher()
        loaded.load_models_gpu(p1)
        p2 = p1.clone()
        _install_dype_patch(p2)
        loaded.load_models_gpu(p2)

        # A patcher outside the leak's backup lineage over the same base.
        p3 = FakeModelPatcher(p1.model)
        loaded.load_models_gpu(p3)

        assert effective_model_sampling(p3) is p3.model.model_sampling


@pytest.mark.unit
class TestResolutionContract:
    def test_same_patcher_reuse_keeps_schedule_stable(self, monkeypatch):
        _install_fake_calculate_sigmas(monkeypatch)
        loaded = FakeLoadedModels()
        p1 = _checkpoint_patcher()
        loaded.load_models_gpu(p1)
        s1 = _stage_entry_sigma(p1)
        loaded.load_models_gpu(p1)  # cache HIT: same patcher object
        assert _stage_entry_sigma(p1) == pytest.approx(s1)

    def test_dype_in_chain_is_always_honored(self, monkeypatch):
        """A DyPE clone in THIS graph always resolves to its own patched
        schedule — before and after unrelated model loads."""
        _install_fake_calculate_sigmas(monkeypatch)
        loaded = FakeLoadedModels()
        dype_patcher = _checkpoint_patcher()
        _install_dype_patch(dype_patcher)
        loaded.load_models_gpu(dype_patcher)
        s_direct = _stage_entry_sigma(dype_patcher)

        # Unrelated load in between (another model's patcher).
        other = FakeModelPatcher(
            FakeBaseModel(_make_sampling("Other", 1.5)))
        loaded.load_models_gpu(other)

        assert _stage_entry_sigma(dype_patcher) == pytest.approx(s_direct)
        assert s_direct > 0.0

    def test_schedule_matches_resolved_object_table(self, monkeypatch):
        """The derived base schedule indexes the RESOLVED object's table —
        the schedule a stock KSampler on the same input would compute."""
        _install_fake_calculate_sigmas(monkeypatch)
        loaded = FakeLoadedModels()
        p1 = _checkpoint_patcher(shift=2.0)
        loaded.load_models_gpu(p1)

        p2 = p1.clone()
        _install_dype_patch(p2)
        loaded.load_models_gpu(p2)

        p3 = FakeModelPatcher(p1.model)
        sigmas = hfn._base_sigmas(p3, steps=30)
        resolved = effective_model_sampling(p3)
        expected = resolved.sigma_table[
            torch.linspace(0, 100, 31).round().long()]
        assert torch.allclose(sigmas, expected)
