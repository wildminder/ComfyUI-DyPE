"""AReA × Krea-2 — the FLUX/Z-Image byte-identity guard and the CI-marker guard
(plan 2026-10-20, §7/S10).

The Krea-2 adaptation added a THIRD attention seam, a THIRD arch profile and
one additive keyword to the engine (``windowed_attention(..., img_axis0=None)``
in ``src/area_attn.py``).  Each of those is a chance to move a backbone that
already worked, so this file is the explicit regression guard the brief asks
for.  What it pins:

* the FLUX and the Z-Image golden hashes are re-derived from the repo's OWN
  seam doubles and must equal the values captured on the pre-Krea tree.  The
  golden literals live HERE as well as in
  ``tests/test_area_grid_fallback.py``, so editing that file's ``GOLDENS``
  cannot silently bless a regression;
* installing and releasing a Krea-2 seam leaves ``comfy.ldm.flux.layers
  .attention``, ``comfy.ldm.lumina.model.optimized_attention_masked`` and the
  shared DEFINING module (``comfy.ldm.modules.attention``) byte-identical to
  their pristine objects;
* a Krea-2 forward never touches the FLUX runtime global (``_AREA_RUNTIME``) —
  the three backbones have three separate runtimes and three separate
  ref-counts, so a cross-backbone leak would fire on the wrong seam;
* the FLUX and Z-Image profile constants (``src/area_arch.py``) and the
  module-level geometry constants (``src/area.py``) are unchanged;
* every AReA test file carries a CI marker, and the new Krea-2 files are
  actually SELECTED by the CI's own ``-m "unit or mock_integration"`` subset —
  an unmarked file is silently deselected, the failure mode this project has
  already been bitten by.

The goldens are the ones in ``tests/test_area_grid_fallback.py:63-92``; the
recipe is the sha256 of a tensor's contiguous float32 bytes, truncated to 32
hex characters.

Markers: @pytest.mark.unit
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
import types

import pytest
import torch

import src.area as area
import test_area_grid_fallback as fallback
import test_area_krea_seam as krea
import test_area_node as flux
import test_area_zimage_seam as zimage
from _krea_fixtures import KREA_SEAM_SYMBOL, krea_backend_module
from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_arch import FLUX_PROFILE, ZIMAGE_PROFILE
from src.area_attn import pool_keys_values
from test_code_quality import (  # the shared marker guard (S10)
    PROJECT_ROOT,
    TESTS_DIR,
    area_test_files_without_ci_marker,
)

#: The FLUX golden hashes — the seam's own answer on the 80x80 and the 160x160
#: canvas, coarse and fine step.  Captured on the pre-Krea tree (HEAD
#: ``56f90a3``) with the CI interpreter; they must not move.
FLUX_GOLDENS = {
    "flux80_coarse": "dca3a04b5655b81b6a40049a010c0d94",
    "flux80_fine": "1c5d27e285761eef13ec76dc10638022",
    "flux160_coarse": "52ed0106c10bd149ea66b16728efd973",
}

#: The Z-Image golden hashes — the same canvas, coarse and fine step.  The
#: FINE one moved with area-fix3 (the GTCN's caption anchoring); it is pinned
#: at its post-fix3 value.
ZIMAGE_GOLDENS = {
    "zimage65_coarse": "394b3e8a2529bb35436fbaea4618d808",
    "zimage65_fine": "b16a038f4050d776d9c77a4a91192d0a",
}

#: The pooled keys / values of the 80x80 grid, per pooling mode — the engine's
#: pooling recipe, frozen.
POOL_GOLDENS = {
    "pool_subsample_k": "93fb5f62bc218d3d2a6deab3124eff6f",
    "pool_subsample_v": "a5392ef83c215b60d9c89862258107c9",
    "pool_mean_k": "8c6db78b59148a21bfca9a17b3333a53",
    "pool_mean_v": "b7fc0cc00fb961d62cfaa75bd7b3be25",
    "pool_medoid_k": "8d27a65069c7f1fb774b2c60c778ac6a",
    "pool_medoid_v": "9731af70c28bb4a9097e898dbeae38de",
    "pool_random_k": "780ac6ff520fc8bce97a0ca408284f6d",
    "pool_random_v": "951c32eae697798dc595c41f015c25b7",
}

#: The FLUX coarse rope ids of the 80x80 grid (query side / key side).
ROPE_GOLDENS = {
    "rope_q_ids": "b1b7b8ada72c8841fea5c88fad54463d",
    "rope_kv_ids": "6cc6716177cb8b2ec132fd89dd0ac7d6",
}

#: The canvases the goldens were captured on: 1280 px (an 80x80 patch grid) and
#: 2560 px (a 160x160 one).
GOLD_SIDE, BIG_SIDE = 1280, 2560
GOLD_GRID = GOLD_SIDE // 16                       # 80
TXT_LEN = 32
PHASE1 = 8

#: The new Krea-2 test files — the ones the CI subset must SELECT.
NEW_KREA_FILES = (
    "test_area_krea.py",
    "test_area_krea_seam.py",
    "test_area_krea_seam_probe.py",
    "test_area_rope.py",
    "test_area_flux_identity.py",
)


def _hash(tensor: torch.Tensor) -> str:
    """The golden recipe: sha256 of the tensor's contiguous float32 bytes."""
    data = tensor.detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flux_layers():
    """A mock ``comfy.ldm.flux.layers`` with a pristine ``attention``.

    AReA patches the *bound* symbol, so the module must hold the original
    function as a plain attribute — the same shape the real backend leaves
    behind (the repo's own mock, ``tests/test_area_node.py:291-312``).
    """
    saved = sys.modules.get("comfy.ldm.flux.layers")
    saved_parent = getattr(sys.modules.get("comfy.ldm.flux"), "layers", None)
    parent = sys.modules.setdefault("comfy.ldm.flux", types.ModuleType("comfy.ldm.flux"))
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.attention = flux.plain_flux_attention
    sys.modules["comfy.ldm.flux.layers"] = layers
    parent.layers = layers
    try:
        yield layers
    finally:
        sys.modules["comfy.ldm.flux.layers"] = saved
        parent.layers = saved_parent
        if saved is None:
            sys.modules.pop("comfy.ldm.flux.layers", None)


@pytest.fixture
def lumina_model():
    """A mock ``comfy.ldm.lumina.model`` with a pristine bound symbol."""
    saved = sys.modules.get("comfy.ldm.lumina.model")
    saved_parent = getattr(sys.modules.get("comfy.ldm.lumina"), "model", None)
    parent = sys.modules.setdefault("comfy.ldm.lumina", types.ModuleType("comfy.ldm.lumina"))
    mod = types.ModuleType("comfy.ldm.lumina.model")
    mod.optimized_attention_masked = zimage.base_lumina_attention
    sys.modules["comfy.ldm.lumina.model"] = mod
    parent.model = mod
    try:
        yield mod
    finally:
        sys.modules["comfy.ldm.lumina.model"] = saved
        parent.model = saved_parent
        if saved is None:
            sys.modules.pop("comfy.ldm.lumina.model", None)


@pytest.fixture
def krea_backend():
    """The mock Krea-2 backend, registered as ``comfy.ldm.krea2.model``."""
    with krea_backend_module() as backend:
        yield backend


@pytest.fixture(autouse=True)
def _clean_area_state():
    """AReA's seam state is process-global: three ref-counts, three runtimes."""
    area._warned.clear()
    for seam in (area._SEAM, area._ZSEAM, area._KSEAM):
        seam.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    area._SEAM.pop("recorder", None)
    area._ZSEAM["recorder"] = None
    area._KSEAM["recorder"] = None
    area._AREA_RUNTIME = None
    area._ZRUNTIME = None
    area._KRUNTIME = None
    yield
    area._warned.clear()
    for seam in (area._SEAM, area._ZSEAM, area._KSEAM):
        seam.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    area._SEAM.pop("recorder", None)
    area._ZSEAM["recorder"] = None
    area._KSEAM["recorder"] = None
    area._AREA_RUNTIME = None
    area._ZRUNTIME = None
    area._KRUNTIME = None


# ---------------------------------------------------------------------------
# The byte-identity guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestByteIdentity:
    """The FLUX and the Z-Image answers did not move."""

    def test_flux_goldens_are_unchanged_after_the_krea_work(self, flux_layers):
        """ANY change to the FLUX output fails here.

        The three FLUX seam hashes are re-derived from the repo's own seam
        doubles and compared with the literals captured on the pre-Krea tree.
        """
        for tag, side, step in (("flux80_coarse", GOLD_SIDE, 0),
                                ("flux80_fine", GOLD_SIDE, 27),
                                ("flux160_coarse", BIG_SIDE, 0)):
            m = apply_area_to_model(flux.mock_flux_model(), "flux", side, side,
                                    phase1_steps=PHASE1, seed=7)
            try:
                sim, _out = flux._run_forward(m, step=step, side=side)
            finally:
                restore_area_attention_hook(m)
            assert _hash(sim.calls[0]) == FLUX_GOLDENS[tag], tag
            # the same value the golden file pins — the two guards agree
            assert fallback.GOLDENS[tag] == FLUX_GOLDENS[tag], tag

    def test_zimage_goldens_are_unchanged_after_the_krea_work(self, lumina_model):
        """ANY change to the Z-Image output fails here.

        The Z-Image seam's rope table is captured by the Z-Image recorder, and
        the Krea-2 work added a THIRD recorder with its own pause guard — a
        leak between the two would move this hash.
        """
        for tag, step in (("zimage65_coarse", 0), ("zimage65_fine", 27)):
            m = apply_area_to_model(zimage.mock_zimage_model(), "zimage",
                                    zimage.WIDTH, zimage.HEIGHT,
                                    phase1_steps=zimage.PHASE1, seed=3)
            try:
                _sim, out = zimage._run_forward(m, step=step)
            finally:
                restore_area_attention_hook(m)
            assert _hash(out) == ZIMAGE_GOLDENS[tag], tag
            assert fallback.GOLDENS[tag] == ZIMAGE_GOLDENS[tag], tag

    def test_the_pooling_recipe_is_unchanged_after_the_krea_work(self):
        """Every pooling mode pools the 80x80 grid exactly as before.

        The Krea-2 work added a pooling call path (Phase I over the
        UN-ROTATED keys) — it must not have changed the pooling itself.
        """
        gen = torch.Generator().manual_seed(0)
        n = GOLD_GRID * GOLD_GRID
        key = torch.randn(1, 2, TXT_LEN + n, 64, generator=gen)
        value = torch.randn(1, 2, TXT_LEN + n, 64, generator=gen)
        for pooling in ("subsample", "mean", "medoid", "random"):
            k, v = pool_keys_values(key, value, TXT_LEN, GOLD_GRID, GOLD_GRID, 2,
                                    pooling, seed=7, step=0, block=0)
            assert _hash(k) == POOL_GOLDENS[f"pool_{pooling}_k"], pooling
            assert _hash(v) == POOL_GOLDENS[f"pool_{pooling}_v"], pooling
            assert fallback.GOLDENS[f"pool_{pooling}_k"] == POOL_GOLDENS[f"pool_{pooling}_k"]
            assert fallback.GOLDENS[f"pool_{pooling}_v"] == POOL_GOLDENS[f"pool_{pooling}_v"]

    def test_the_coarse_rope_ids_are_unchanged_after_the_krea_work(self):
        """The FLUX coarse tables' position ids did not move.

        Krea-2 builds its coarse tables through the MODEL's own embedder (the
        same FLUX-layout ``EmbedND``), so a change to the coarse id builder
        would move the FLUX ones too.
        """
        n = GOLD_GRID * GOLD_GRID
        q_ids = flux._ids(TXT_LEN, torch.arange(n) // GOLD_GRID // 2,
                          torch.arange(n) % GOLD_GRID // 2)
        kv_ids = flux._ids(TXT_LEN, torch.arange(n // 4) // (GOLD_GRID // 2),
                           torch.arange(n // 4) % (GOLD_GRID // 2))
        assert _hash(q_ids) == ROPE_GOLDENS["rope_q_ids"]
        assert _hash(kv_ids) == ROPE_GOLDENS["rope_kv_ids"]
        assert fallback.GOLDENS["rope_q_ids"] == ROPE_GOLDENS["rope_q_ids"]
        assert fallback.GOLDENS["rope_kv_ids"] == ROPE_GOLDENS["rope_kv_ids"]


@pytest.mark.unit
class TestCrossBackboneIsolation:
    """The Krea-2 seam touches the Krea-2 seam and NOTHING else."""

    def test_krea_install_does_not_touch_the_flux_seam(self, krea_backend,
                                                       flux_layers):
        """A cross-backbone install would move FLUX's answer.

        The FLUX seam is ``comfy.ldm.flux.layers.attention``; the Krea-2 seam
        is ``comfy.ldm.krea2.model.optimized_attention_masked``.  They are
        different modules and different symbols — the Krea-2 install must
        leave the FLUX one pristine, and must not claim its ref-count.
        """
        pristine = flux_layers.attention
        patched = apply_area_to_model(krea._mock_krea_model(), "krea2",
                                      krea.WIDTH, krea.HEIGHT)
        try:
            assert flux_layers.attention is pristine
            assert area._SEAM["consumers"] == 0
            assert area._SEAM["orig"] is None
            # the Krea-2 seam IS installed — the guard is not vacuous
            assert area._KSEAM["consumers"] == 1
            assert getattr(krea_backend, KREA_SEAM_SYMBOL) is not pristine
        finally:
            restore_area_attention_hook(patched)
        assert flux_layers.attention is pristine

    def test_krea_install_does_not_touch_the_zimage_seam(self, krea_backend,
                                                         lumina_model):
        """The same, for the Z-Image seam.

        Z-Image's bound symbol is ``comfy.ldm.lumina.model
        .optimized_attention_masked`` — the SAME NAME Krea-2 binds in its own
        module, so this is the seam a copy-paste install would get wrong.
        """
        pristine = lumina_model.optimized_attention_masked
        patched = apply_area_to_model(krea._mock_krea_model(), "krea2",
                                      krea.WIDTH, krea.HEIGHT)
        try:
            assert lumina_model.optimized_attention_masked is pristine
            assert area._ZSEAM["consumers"] == 0
            assert area._ZSEAM["orig"] is None
            assert area._KSEAM["consumers"] == 1
        finally:
            restore_area_attention_hook(patched)
        assert lumina_model.optimized_attention_masked is pristine

    def test_krea_install_does_not_touch_the_defining_module(self, krea_backend):
        """``comfy.ldm.modules.attention`` is shared by every backbone.

        It is the module the symbol is DEFINED in (``comfy/ldm/modules
        /attention.py``); patching it would move FLUX, Qwen and Z-Image at
        once.  The Krea-2 seam rebinds the BOUND name only.
        """
        defining = sys.modules["comfy.ldm.modules.attention"]
        before = defining.optimized_attention_masked
        patched = apply_area_to_model(krea._mock_krea_model(), "krea2",
                                      krea.WIDTH, krea.HEIGHT)
        try:
            assert defining.optimized_attention_masked is before
        finally:
            restore_area_attention_hook(patched)
        assert defining.optimized_attention_masked is before

    def test_krea_forward_leaves_the_flux_runtime_global_unset(self, krea_backend):
        """The three backbones have three SEPARATE runtime globals.

        ``_make_area_unet_wrapper`` sets exactly one of ``_AREA_RUNTIME`` /
        ``_ZRUNTIME`` / ``_KRUNTIME`` per forward (``src/area.py:1911-1928``).
        A shared global would let a Krea-2 forward fire the FLUX engine's
        code path (or read the FLUX runtime's state) — so the FLUX global
        must still be ``None`` WHILE the Krea-2 forward runs.
        """
        dm = krea.SingleStreamDiT()
        patched = apply_area_to_model(krea._mock_krea_model(dm), "krea2",
                                      krea.WIDTH, krea.HEIGHT)
        seen = {}

        def _model_function(x, timestep, **c):
            # inside the wrapper: the runtime globals are live right now
            seen["flux"] = area._AREA_RUNTIME
            seen["zimage"] = area._ZRUNTIME
            seen["krea"] = area._KRUNTIME
            if x.dim() == 5:
                # comfy flattens the temporal axis into the batch before the
                # model sees it (``comfy/ldm/krea2/model.py:293-297``)
                b, ch, t, h, w = x.shape
                x = x.reshape(b * t, ch, h, w)
            options = c.get("transformer_options") or {}
            return dm(x, timestep, c["c_crossattn"], transformer_options=options)

        try:
            wrapper = krea._unet_wrapper(patched)
            assert wrapper is not None
            params = krea._params(dm, step=0)
            wrapper(_model_function, params)
        finally:
            restore_area_attention_hook(patched)
        assert seen, "the forward never reached the model"
        assert seen["flux"] is None, "a Krea-2 forward set the FLUX runtime"
        assert seen["zimage"] is None, "a Krea-2 forward set the Z-Image runtime"
        assert seen["krea"] is not None, "the Krea-2 runtime was not set"
        # and the globals are cleared again once the forward is over
        assert area._AREA_RUNTIME is None
        assert area._ZRUNTIME is None
        assert area._KRUNTIME is None


@pytest.mark.unit
class TestProfileConstants:
    """The FLUX and the Z-Image profiles did not move."""

    def test_flux_profile_constants_are_unchanged(self):
        """A silent change to ``FLUX_PROFILE`` would move the geometry.

        The profile decides the pixel factor, the sequence padding and the
        trained extent — the three numbers the FLUX and Z-Image answers are
        built on.
        """
        assert FLUX_PROFILE.model_type == "flux"
        assert FLUX_PROFILE.unet_backbone == "Flux"
        assert FLUX_PROFILE.px_per_patch == 16
        assert FLUX_PROFILE.seq_pad_multiple == 0
        assert FLUX_PROFILE.native_tokens == 64
        assert FLUX_PROFILE.accepts == "FLUX.1 (Flux) diffusion model"
        # the module-level geometry constants the FLUX path reads
        assert area.PATCH_PX == 16
        assert area.NATIVE_PATCHES == 64

    def test_zimage_profile_constants_are_unchanged(self):
        """The same, for the Z-Image profile (its pad policy is 32)."""
        assert ZIMAGE_PROFILE.model_type == "zimage"
        assert ZIMAGE_PROFILE.unet_backbone == "NextDiT"
        assert ZIMAGE_PROFILE.px_per_patch == 16
        assert ZIMAGE_PROFILE.seq_pad_multiple == 32
        assert ZIMAGE_PROFILE.native_tokens == 64
        assert ZIMAGE_PROFILE.accepts == "Lumina2 (NextDiT) diffusion model"


@pytest.mark.unit
class TestCiMarkerGuard:
    """No AReA test file may fall out of the CI selection."""

    def test_every_area_test_file_carries_a_ci_marker(self):
        """The static guard: every ``tests/test_area_*.py`` is marked.

        The CI runs ``pytest -q -m "unit or mock_integration"``; a file with
        neither marker is silently DESELECTED — not a failure, just a test
        that stopped running.
        """
        assert area_test_files_without_ci_marker() == []
        covered = {path.name for path in TESTS_DIR.glob("test_area_*.py")}
        for name in NEW_KREA_FILES:
            assert name in covered, f"{name} is not covered by the marker guard"

    def test_new_krea_files_are_collected_by_the_ci_marker_subset(self):
        """The dynamic guard: the CI's OWN selection picks the new files up.

        The static check above proves the marker TEXT is there; this runs the
        CI's own command (``-m "unit or mock_integration"``) over the new
        files and proves the tests are actually COLLECTED — a marker on the
        wrong node, or a file pytest does not collect at all, would slip past
        the static check.
        """
        command = [sys.executable, "-m", "pytest", "--collect-only", "-q",
                   "-p", "no:cacheprovider",
                   # the project's addopts carry ``-v`` (pyproject.toml), which
                   # would turn the node-id list into a tree; only the import
                   # mode matters for the collection
                   "-o", "addopts=--import-mode=importlib",
                   "-m", "unit or mock_integration",
                   *[f"tests/{name}" for name in NEW_KREA_FILES]]
        result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True,
                                text=True)
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "deselected" not in output, output
        for name in NEW_KREA_FILES:
            assert f"tests/{name}::" in output, (
                f"{name} was not selected by the CI marker subset:\n{output}"
            )
