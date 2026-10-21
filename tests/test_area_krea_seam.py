"""AReA × Krea-2 — the seam install and the forward-time resolution
(plan §7/S6 and §7/S7).

Krea-2 binds its OWN copy of the attention symbol
(``from comfy.ldm.modules.attention import ... optimized_attention_masked``,
``comfy/ldm/krea2/model.py:21``) and calls it at ``:99-100`` as
``optimized_attention_masked(q, k, v, self.heads, mask=mask, skip_reshape=True,
preferred_attention=self.comfy_attention, transformer_options=...)`` — FOUR
positional slots, the q/k **rotated** by the model itself (``model.py:93``), and
the answer in comfy's flattened ``[b, n, heads*d]`` layout (no
``skip_output_reshape`` is passed).

So the seam is ``comfy.ldm.krea2.model.optimized_attention_masked`` — the BOUND
name, not the defining module: patching ``comfy.ldm.modules.attention`` would
move FLUX/Qwen/Z-Image too and would never reach Krea-2 (the silent no-op SPA
documents, ``src/spa.py:558-567``).  The mock harness
(``tests/_krea_fixtures.py``) registers ITSELF under that name, so a rebind of
the bound symbol is seen by the mock's own attention class — the same trick the
Z-Image harness uses.

What this file pins (each test names the mutation it kills):

* the install rebinds the BOUND symbol, leaves the defining module alone, and
  leaves the FLUX and Z-Image seams alone;
* the delegating wrapper is **byte-identical** to the pristine model: it
  forwards the original arguments, consumes no container, and lets the four
  20-head auxiliary calls through untouched;
* the SPA collision raises an actionable error, and a wrapper with no recorded
  pristine is refused instead of wrapped;
* a quantized/fused attention kernel (the model's class does not resolve the
  symbol) leaves the model UNCHANGED;
* the ref-count: a second install re-uses the wrapper, and the symbol comes
  back only on the last restore;
* the detail-map hook on ``dm.first`` fires once per forward and captures the
  image embedding;
* the forward-time resolution: a 5-D ``T == 1`` latent is squeezed and its
  grid read from the LATENT, a video (``T > 1``) and a forward carrying
  reference latents are REFUSED with a one-time warning, the native-extent
  gate uses the profile's tokens, the schedule is cached per canvas, and the
  step index comes from the sampler's own sigma table.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_krea_seam.py tests/test_area_grid_fallback.py
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

import src.area as area
from _krea_fixtures import (
    HEAD_DIM,
    HEADS,
    KREA_SEAM_MODULE,
    KREA_SEAM_SYMBOL,
    LAYERS,
    SEQ,
    TXT_HEADS,
    SingleStreamDiT,
    krea_backend_module,
)
from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_arch import FLUX_PROFILE, KREA_PROFILE, ZIMAGE_PROFILE
from src.area_attn import apply_rope1

try:
    from comfy.model_patcher import ModelPatcher
except ImportError:  # pragma: no cover - mock env always provides it
    from tests.conftest import MockModelPatcher as ModelPatcher

#: A canvas past the native extent, so ``apply_area_to_model`` really installs
#: the seam (at or below 64 patches AReA is a provable no-op and returns the
#: clone untouched).  The FORWARDS below use small latents, so the mock's 28
#: blocks stay cheap.
WIDTH, HEIGHT = 1040, 1040

#: The prompt length of the mock forward — the joint stream is
#: ``[prompt | image]``, and the refiner calls' stream length is this number
#: (a PREFIX of the joint one).
TXT_LEN = SEQ

#: The reference's FLUX schedule: 28 steps, 8 of them Phase I.
STEPS, PHASE1 = 28, 8

#: The latent's cells of the small canvas the forwards use — 12x12 cells is a
#: 6x6 TOKEN grid (the model's own patch factor is 2).
CELLS = 12
GRID = CELLS // 2


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def krea_backend():
    """The mock backend, registered as ``comfy.ldm.krea2.model`` for the test."""
    with krea_backend_module() as backend:
        yield backend


@pytest.fixture(autouse=True)
def _clean_area_state():
    """AReA's seam state is process-global (three ref-counts, three runtimes)."""
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


def _mock_krea_model(dm=None) -> ModelPatcher:
    """A patcher whose diffusion model is the mock Krea-2 backbone."""
    m = ModelPatcher()
    m.model.diffusion_model = dm if dm is not None else SingleStreamDiT()
    m.model.model_sampling = types.SimpleNamespace()
    return m


def _sigmas(steps: int = STEPS) -> torch.Tensor:
    """A denoising sigma table with the sampler's terminal zero."""
    return torch.linspace(1.0, 0.0, steps + 1)


def _latent(b: int = 1, cells: int = CELLS, frames: int = 1) -> torch.Tensor:
    """The Krea-2 latent ``[B, 16, T, H, W]`` — the shape comfy hands the model."""
    gen = torch.Generator().manual_seed(7)
    return torch.randn(b, 16, frames, cells, cells, generator=gen)


def _cross(b: int = 1, seq: int = TXT_LEN, dm=None) -> torch.Tensor:
    """The prompt conditioning ``[B, seq, txtlayers * txtdim]``."""
    dm = dm if dm is not None else SingleStreamDiT()
    gen = torch.Generator().manual_seed(11)
    return torch.randn(b, seq, dm.txtlayers * dm.txtdim, generator=gen)


def _params(dm, step: int = 0, steps: int = STEPS, frames: int = 1,
            cells: int = CELLS, refs=None, sigmas: bool = True,
            batch: int = 1) -> dict:
    """The unet wrapper's ``params`` — the shape comfy hands the sampler.

    ``batch`` is the CFG batch: comfy runs cond and uncond in ONE batched
    forward (``comfy/samplers.py:309``), so a batch of two is the two branches
    of a single forward.
    """
    table = _sigmas(steps)
    opts = {"sample_sigmas": table, "sigmas": table[step]}
    if not sigmas:
        opts = {"sigmas": table[step]}
    conds = {"c_crossattn": _cross(b=batch, dm=dm), "transformer_options": opts}
    if refs is not None:
        conds["ref_latents"] = refs
    return {
        "input": _latent(b=batch, frames=frames, cells=cells),
        "timestep": torch.tensor([float(table[step])]),
        "c": conds,
    }


class _KreaModel:
    """The model's own forward, as the unet wrapper calls it.

    ``comfy/ldm/krea2/model.py:293-297`` flattens a 5-D latent into the batch
    (``temporal = x.ndim == 5``), so the model itself only ever sees a 4-D
    latent — the mock backbone's forward is 4-D only, and this adapter does
    the same reshape.
    """

    def __init__(self, dm):
        self.dm = dm
        self.calls = 0

    def __call__(self, x, timestep, **c):
        self.calls += 1
        if x.dim() == 5:
            b, ch, t, h, w = x.shape
            x = x.reshape(b * t, ch, h, w)
        opts = c.get("transformer_options") or {}
        return self.dm(x, timestep, c["c_crossattn"], transformer_options=opts)


def _unet_wrapper(m):
    """The wrapper the patcher installed (mock and real ModelPatcher)."""
    try:
        return m.model_options["model_function_wrapper"]
    except (AttributeError, KeyError, TypeError):
        return getattr(m, "_unet_wrapper", None)


def _runtime(dm, native: int = 4, phase1_steps: int = PHASE1, **config):
    """A Krea-2 runtime built by hand.

    The native extent is lowered so a SMALL canvas is already past it: the
    mock's forward over a 65x65-token canvas (the real native extent) would
    need a 4233-token attention matrix per block — far too slow on CPU.  The
    resolution code under test is the same either way.
    """
    return area._AreaRuntime(
        area.AreaConfig(native=native, phase1_steps=phase1_steps, **config),
        dm, KREA_PROFILE)


# ---------------------------------------------------------------------------
# The install
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeamInstall:
    def test_install_rebinds_the_bound_symbol_in_krea2_model(self, krea_backend):
        """The seam is the BOUND name — the one the model's class resolves."""
        pristine = krea_backend.optimized_attention_masked
        patched = apply_area_to_model(_mock_krea_model(), "krea2", WIDTH, HEIGHT)
        try:
            bound = getattr(krea_backend, KREA_SEAM_SYMBOL)
            assert bound is not pristine
            assert getattr(bound, "_area_wrapper", False)
            assert bound._area_orig is pristine
            assert area._KSEAM["module"] is krea_backend
            assert area._KSEAM["attr"] == KREA_SEAM_SYMBOL
            assert area._KSEAM["orig"] is pristine
            assert area._KSEAM["consumers"] == 1
            assert patched._area_seam == "krea2"
            # the model's own attention class resolves the symbol we rebound
            dm = patched.model.diffusion_model
            namespace = type(dm.blocks[0].attn).forward.__globals__
            assert namespace[KREA_SEAM_SYMBOL] is bound
        finally:
            restore_area_attention_hook(patched)

    def test_install_leaves_the_defining_module_untouched(self, krea_backend):
        """The DEFINING module is shared with FLUX/Qwen/Z-Image — never touched."""
        defining = sys.modules["comfy.ldm.modules.attention"]
        before = defining.optimized_attention_masked
        patched = apply_area_to_model(_mock_krea_model(), "krea2", WIDTH, HEIGHT)
        try:
            assert defining.optimized_attention_masked is before
            assert getattr(krea_backend, KREA_SEAM_SYMBOL) is not before
        finally:
            restore_area_attention_hook(patched)

    def test_install_does_not_touch_the_flux_or_zimage_seams(self, krea_backend):
        """Three seams, three ref-counts: the other two stay pristine."""
        patched = apply_area_to_model(_mock_krea_model(), "krea2", WIDTH, HEIGHT)
        try:
            assert area._SEAM["consumers"] == 0
            assert area._ZSEAM["consumers"] == 0
            assert area._KSEAM["consumers"] == 1
        finally:
            restore_area_attention_hook(patched)

    def test_install_registers_the_rope_capture_and_the_detail_map_hook(self,
                                                                       krea_backend):
        """The capture sits on the model's OWN embedder, the map on ``first``."""
        patched = apply_area_to_model(_mock_krea_model(), "krea2", WIDTH, HEIGHT)
        try:
            dm = patched.model.diffusion_model
            assert len(dm.pe_embedder._forward_hooks) == 1
            assert len(dm.first._forward_hooks) == 1
            assert area._KSEAM["hook"]
            assert area._KSEAM["recorder"] is not None
        finally:
            restore_area_attention_hook(patched)
        for handle in (area._KSEAM["hook"] or ()):
            assert handle is not None

    def test_install_refuses_when_spa_owns_the_same_symbol(self, krea_backend):
        """SPA patches THIS symbol for Krea-2 — the collision is refused."""
        foreign = lambda *a, **kw: None      # noqa: E731 - a marker, not a call
        foreign._spa_wrapper = True
        krea_backend.optimized_attention_masked = foreign
        with pytest.raises(ValueError, match="already wrapped by SPA") as excinfo:
            apply_area_to_model(_mock_krea_model(), "krea2", WIDTH, HEIGHT)
        message = str(excinfo.value)
        # names the offender AND the remedy (house rule, src/hap.py:876-895)
        assert KREA_SEAM_MODULE in message
        assert "src/spa.py:583-589" in message
        assert krea_backend.optimized_attention_masked is foreign   # nothing rebound
        assert area._KSEAM["consumers"] == 0                        # no state leaked

    def test_install_refuses_a_wrapper_without_a_recorded_pristine(self, krea_backend):
        """Our own wrapper with no recorded original is refused, not re-wrapped."""
        broken = lambda *a, **kw: None       # noqa: E731 - a marker, not a call
        broken._area_wrapper = True
        krea_backend.optimized_attention_masked = broken
        with pytest.raises(ValueError, match="no recorded pristine symbol"):
            apply_area_to_model(_mock_krea_model(), "krea2", WIDTH, HEIGHT)
        assert krea_backend.optimized_attention_masked is broken
        assert area._KSEAM["consumers"] == 0

    def test_fused_kernel_model_is_returned_unchanged(self, krea_backend):
        """A quantized/fused attention class bypasses the seam — the model is left alone.

        The install proves the model's own attention class still resolves the
        seam symbol.  A fused kernel replaces the class, so the proof fails and
        AReA would be a SILENT no-op — the model must come back UNCHANGED.
        """
        pristine = krea_backend.optimized_attention_masked
        dm = SingleStreamDiT()

        class FusedAttention(torch.nn.Module):
            """A fused attention class: it never resolves the seam symbol."""

            def forward(self, x, freqs=None, mask=None, transformer_options={}):
                return x

        dm.blocks[0].attn = FusedAttention()
        patched = apply_area_to_model(_mock_krea_model(dm), "krea2", WIDTH, HEIGHT)
        try:
            assert patched._area_config is None                 # UNCHANGED
            assert krea_backend.optimized_attention_masked is pristine
            assert area._KSEAM["consumers"] == 0
            assert "krea-fused" in area._warned
        finally:
            restore_area_attention_hook(patched)

    def test_restore_returns_the_pristine_symbol_and_is_idempotent(self, krea_backend):
        pristine = krea_backend.optimized_attention_masked
        patched = apply_area_to_model(_mock_krea_model(), "krea2", WIDTH, HEIGHT)
        dm = patched.model.diffusion_model
        assert area._KSEAM["hook"]
        restore_area_attention_hook(patched)
        assert krea_backend.optimized_attention_masked is pristine
        assert area._KSEAM == {"module": None, "attr": None, "orig": None,
                               "hook": None, "recorder": None, "consumers": 0}
        assert dm.pe_embedder._forward_hooks == {}
        assert dm.first._forward_hooks == {}
        assert patched._area_config is None
        # idempotent — a second restore is a no-op, not a crash
        restore_area_attention_hook(patched)

    def test_ref_counted_install_needs_matching_restores(self, krea_backend):
        pristine = krea_backend.optimized_attention_masked
        first = apply_area_to_model(_mock_krea_model(), "krea2", WIDTH, HEIGHT, seed=1)
        wrapped = krea_backend.optimized_attention_masked
        second = apply_area_to_model(first, "krea2", WIDTH, HEIGHT, seed=2)
        try:
            # a second apply re-uses the wrapper instead of wrapping our own wrapper
            assert krea_backend.optimized_attention_masked is wrapped
            assert area._KSEAM["consumers"] == 2
            assert second._area_config.seed == 2          # newest apply wins
            restore_area_attention_hook(second)
            assert area._KSEAM["consumers"] == 1
            assert krea_backend.optimized_attention_masked is wrapped
            restore_area_attention_hook(first)
            assert area._KSEAM["consumers"] == 0
            assert krea_backend.optimized_attention_masked is pristine
        finally:
            restore_area_attention_hook(first)
            restore_area_attention_hook(second)


# ---------------------------------------------------------------------------
# The delegating wrapper
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDelegatingWrapper:
    def _install_with_spy(self, krea_backend):
        """Install the seam over a spy on the pristine symbol.

        The spy reads the containers BEFORE the pristine consumes them, so a
        wrapper that took a container (the v2.19.1 crash class) would be
        visible here as a failed ``peek()``.
        """
        seen = []
        real = krea_backend.optimized_attention_masked

        def spy(q, k, v, heads, *args, **kw):
            # ``transformer_options`` is MUTATED later in the forward (the model
            # publishes ``img_slice`` and ``block_index`` on the very dict it
            # passes to every call), so its CONTENT is snapshotted HERE — at
            # call time — exactly as the probe's ``SeamCall`` does.
            options = dict(kw.get("transformer_options") or {})
            seen.append({
                "heads": heads,
                "q": tuple(q.peek().shape),
                "k": tuple(k.peek().shape),
                "v": tuple(v.peek().shape),
                "args": args,
                "kw": dict(kw),
                "options": options,
            })
            return real(q, k, v, heads, *args, **kw)

        krea_backend.optimized_attention_masked = spy
        patched = apply_area_to_model(_mock_krea_model(), "krea2", WIDTH, HEIGHT)
        return patched, seen, real

    def test_delegating_forward_is_byte_identical_to_the_pristine_model(self,
                                                                       krea_backend):
        """The install changes NOTHING: the forward answers the same bits.

        The reference is the pristine model's own answer on the same latent and
        prompt; the patched run must be bit-for-bit equal to it.
        """
        dm = SingleStreamDiT()
        params = _params(dm)
        model = _KreaModel(dm)
        reference = model(params["input"], params["timestep"], **params["c"])

        patched = apply_area_to_model(_mock_krea_model(dm), "krea2", WIDTH, HEIGHT)
        try:
            out = _unet_wrapper(patched)(model, params)
            assert torch.equal(out, reference)
        finally:
            restore_area_attention_hook(patched)

    def test_wrapper_delegates_with_the_original_positional_slots(self, krea_backend):
        """``heads`` is the FOURTH positional slot, and the kwargs ride along.

        Krea-2 calls the seam with ``(q, k, v, heads)`` and the keywords
        ``mask`` / ``skip_reshape`` / ``preferred_attention`` /
        ``transformer_options`` (``model.py:99-100``) — a wrapper that
        mis-slots ``heads`` into ``mask`` fails silently.
        """
        dm = SingleStreamDiT()
        patched, seen, _real = self._install_with_spy(krea_backend)
        try:
            model = _KreaModel(dm)
            _unet_wrapper(patched)(model, _params(dm))
        finally:
            restore_area_attention_hook(patched)
        assert seen, "the seam was never reached"
        for call in seen:
            assert call["heads"] in (HEADS, TXT_HEADS)
            assert call["args"] == ()
            assert call["kw"]["skip_reshape"] is True
            assert call["kw"]["mask"] is None
            assert "preferred_attention" in call["kw"]
            assert "transformer_options" in call["kw"]

    def test_wrapper_consumes_no_container(self, krea_backend):
        """The containers reach the pristine symbol ALIVE.

        Taking a container and handing the taken tensor back is the v2.19.1
        crash class; the delegating wrapper must not touch them at all.
        """
        dm = SingleStreamDiT()
        patched, seen, _real = self._install_with_spy(krea_backend)
        try:
            model = _KreaModel(dm)
            _unet_wrapper(patched)(model, _params(dm))
        finally:
            restore_area_attention_hook(patched)
        # every recorded call PEEKED the container successfully — a wrapper that
        # had taken it would have raised "already been consumed" here
        assert seen
        for call in seen:
            assert len(call["q"]) == 4

    def test_aux_calls_reach_the_wrapper_and_are_delegated(self, krea_backend):
        """The four 20-head text-fusion calls reach the seam and are delegated.

        They run BEFORE ``transformer_options["img_slice"]`` is published
        (``model.py:341`` vs ``:369``), so they carry no ``img_slice`` — which
        is what tells them apart from the 48-head main calls.
        """
        dm = SingleStreamDiT()
        patched, seen, _real = self._install_with_spy(krea_backend)
        try:
            model = _KreaModel(dm)
            _unet_wrapper(patched)(model, _params(dm))
        finally:
            restore_area_attention_hook(patched)
        aux = [c for c in seen if c["heads"] == TXT_HEADS]
        main = [c for c in seen if c["heads"] == HEADS]
        assert len(aux) == 4, "the four auxiliary calls never reached the seam"
        assert main, "the main blocks never reached the seam"
        for call in aux:
            assert "img_slice" not in call["options"]
        for call in main:
            assert "img_slice" in call["options"]

    def test_an_area_step_replaces_the_pristine_attention(self, krea_backend):
        """An AReA step REPLACES the attention (S8) — the engine is no longer a no-op.

        The runtime's native extent is lowered so the small canvas IS an AReA
        step (the phase is resolved, not ``None``), and the answer must now
        DIFFER from the pristine model's own: the phases run.
        """
        dm = SingleStreamDiT()
        params = _params(dm)
        model = _KreaModel(dm)
        reference = model(params["input"], params["timestep"], **params["c"])

        patched = apply_area_to_model(_mock_krea_model(dm), "krea2", WIDTH, HEIGHT)
        try:
            rt = _runtime(dm, native=4)
            wrapper = area._make_area_unet_wrapper(rt, None)
            out = wrapper(model, params)
            assert rt.ctx is not None and rt.ctx.phase is not None
            assert not torch.equal(out, reference)
        finally:
            restore_area_attention_hook(patched)


# ---------------------------------------------------------------------------
# The detail-map hook
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDetailMapHook:
    def test_detail_map_hook_on_first_fires_once_per_forward(self, krea_backend):
        """``dm.first`` is Krea-2's image embedder — the detail map's input.

        The hook is registered on the module itself (``first`` is CALLED, not
        iterated), and it captures the image embedding ``[b, h*w, features]`` —
        the layout :func:`~src.area_attn.detail_map` reads.
        """
        dm = SingleStreamDiT()
        patched = apply_area_to_model(_mock_krea_model(dm), "krea2", WIDTH, HEIGHT)
        try:
            hooks = list(patched.model.diffusion_model.first._forward_hooks.values())
            assert len(hooks) == 1
            assert hooks[0] is area._krea_first_hook
            # a FINE step captures the image embedding
            rt = _runtime(dm, native=4, phase1_steps=0)
            wrapper = area._make_area_unet_wrapper(rt, None)
            model = _KreaModel(dm)
            wrapper(model, _params(dm))
            assert rt.ctx.phase == "fine"
            assert rt.hidden is not None
            assert tuple(rt.hidden.shape) == (1, GRID * GRID, dm.first.out_features)
        finally:
            restore_area_attention_hook(patched)


# ---------------------------------------------------------------------------
# The forward-time resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestForwardResolution:
    def test_five_d_latent_with_t1_resolves_the_grid(self, krea_backend):
        """``[B, 16, 1, H, W]`` is the NORMAL image case — it must not be dropped.

        AReA's own guard is ``x.ndim != 4`` (``src/area.py``), so without the
        squeeze a Krea-2 image run would be a SILENT NO-OP.
        """
        dm = SingleStreamDiT()
        rt = _runtime(dm, native=4)
        area._area_resolve_forward(rt, _params(dm))
        assert rt.ctx.h == GRID and rt.ctx.w == GRID
        assert rt.ctx.phase is not None

    def test_four_d_latent_still_resolves(self, krea_backend):
        """A 4-D latent (a loader that hands one over) is fine too."""
        dm = SingleStreamDiT()
        rt = _runtime(dm, native=4)
        params = _params(dm)
        params["input"] = params["input"][:, :, 0]
        area._area_resolve_forward(rt, params)
        assert rt.ctx.h == GRID and rt.ctx.w == GRID

    def test_video_latent_with_t_gt_1_refuses_with_a_warning(self, krea_backend):
        """A VIDEO is refused: the frames are not one rectangular canvas."""
        dm = SingleStreamDiT()
        rt = _runtime(dm, native=4)
        area._area_resolve_forward(rt, _params(dm, frames=3))
        assert rt.ctx.phase is None
        assert "krea-video" in area._warned

    def test_reference_latents_refuse_with_a_warning(self, krea_backend):
        """Reference latents append tokens to the canvas — the grid breaks."""
        dm = SingleStreamDiT()
        rt = _runtime(dm, native=4)
        refs = [torch.zeros(1, 16, 8, 8)]
        area._area_resolve_forward(rt, _params(dm, refs=refs))
        assert rt.ctx.phase is None
        assert "krea-ref-latents" in area._warned

    def test_grid_comes_from_the_latent_not_the_widget(self, krea_backend):
        """The runtime grid is the LATENT's — the widget may disagree."""
        dm = SingleStreamDiT()
        rt = _runtime(dm, native=4)
        params = _params(dm, cells=20)        # 20x20 cells -> 10x10 tokens
        area._area_resolve_forward(rt, params)
        assert (rt.ctx.h, rt.ctx.w) == (10, 10)

    def test_native_extent_gate_uses_the_profile_native_tokens(self, krea_backend):
        """The gate is the PROFILE's trained extent — and the config carries it."""
        dm = SingleStreamDiT()
        patched = apply_area_to_model(_mock_krea_model(dm), "krea2", WIDTH, HEIGHT)
        try:
            assert patched._area_config.native == KREA_PROFILE.native_tokens
            assert KREA_PROFILE.native_tokens == 64
            # the other backbones keep their own numbers
            assert FLUX_PROFILE.native_tokens == 64
            assert ZIMAGE_PROFILE.native_tokens == 64
        finally:
            restore_area_attention_hook(patched)
        # at/below the native extent the forward is a provable no-op
        rt = _runtime(dm, native=64)
        area._area_resolve_forward(rt, _params(dm))
        assert rt.ctx.phase is None

    def test_schedule_is_built_once_per_canvas_and_cached(self, krea_backend):
        """One schedule per canvas — the reference's per-canvas cache."""
        dm = SingleStreamDiT()
        rt = _runtime(dm, native=4)
        area._area_resolve_forward(rt, _params(dm, step=0))
        area._area_resolve_forward(rt, _params(dm, step=1))
        assert len(rt.schedules) == 1

    def test_phase_switch_follows_the_schedule(self, krea_backend):
        """The leading ``phase1_steps`` steps are Phase I, the rest Phase II."""
        dm = SingleStreamDiT()
        rt = _runtime(dm, native=4, phase1_steps=2)
        area._area_resolve_forward(rt, _params(dm, step=0))
        assert rt.ctx.phase == "coarse"
        area._area_resolve_forward(rt, _params(dm, step=1))
        assert rt.ctx.phase == "coarse"
        area._area_resolve_forward(rt, _params(dm, step=2))
        assert rt.ctx.phase == "fine"

    def test_step_index_located_by_the_sigma_table(self, krea_backend):
        """The step comes from the sampler's own sigma table, as for FLUX."""
        dm = SingleStreamDiT()
        rt = _runtime(dm, native=4)
        area._area_resolve_forward(rt, _params(dm, step=5))
        assert rt.ctx.step == 5
        assert rt.ctx.total == STEPS

    def test_missing_sigma_table_falls_back_to_plain_attention(self, krea_backend):
        """No sigma table -> no step index -> plain attention (never a guess)."""
        dm = SingleStreamDiT()
        rt = _runtime(dm, native=4)
        area._area_resolve_forward(rt, _params(dm, sigmas=False))
        assert rt.ctx.phase is None
        assert "no-sigmas" in area._warned

    def test_plain_step_delegates_to_the_pristine_base(self, krea_backend,
                                                       monkeypatch):
        """A plain step (no sigma table) runs the PRISTINE attention.

        The engine is monkeypatched to raise: a plain step must not compute.
        """
        dm = SingleStreamDiT()
        params = _params(dm, sigmas=False)
        model = _KreaModel(dm)
        reference = model(params["input"], params["timestep"], **params["c"])
        patched = apply_area_to_model(_mock_krea_model(dm), "krea2", WIDTH, HEIGHT)
        try:
            def _engine(*a, **kw):        # pragma: no cover - must never run
                raise AssertionError("the engine computed on a plain step")

            monkeypatch.setattr(area, "pool_keys_values", _engine)
            monkeypatch.setattr(area, "windowed_attention", _engine)
            out = _unet_wrapper(patched)(model, params)
            assert torch.equal(out, reference)
        finally:
            restore_area_attention_hook(patched)


# ---------------------------------------------------------------------------
# The two phases on the Krea-2 seam (S8)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKreaPhases:
    """S8 — Phase I and Phase II run on the Krea-2 seam.

    Krea-2 rotates q/k ITSELF (``comfy/ldm/krea2/model.py:93``) and hands the
    seam no rope table, so both phases start from the UN-ROTATED frame; the
    answer is comfy's flattened ``[b, n, heads*d]`` — the layout the model's
    ``out * F.sigmoid(gate)`` multiplies (``model.py:106``).
    """

    @pytest.fixture
    def seam(self, krea_backend):
        """The seam installed on the mock backend, and the model behind it."""
        dm = SingleStreamDiT()
        patched = apply_area_to_model(_mock_krea_model(dm), "krea2", WIDTH, HEIGHT)
        yield dm, patched, krea_backend
        restore_area_attention_hook(patched)

    def _run(self, dm, step, rt=None, params=None):
        """One AReA forward through the Krea-2 seam; returns ``(runtime, output)``."""
        rt = rt if rt is not None else _runtime(dm, native=4)
        wrapper = area._make_area_unet_wrapper(rt, None)
        out = wrapper(_KreaModel(dm),
                      params if params is not None else _params(dm, step=step))
        return rt, out

    def _pre_rope_keys(self, dm, block: int) -> torch.Tensor:
        """The block's OWN keys, before the model rotated them.

        ``MockAttention`` keeps the pre-rotation pair (``model.py:93``'s
        counterpart); the GQA repeat is applied after it, so the keys are
        repeated to the query head count here.
        """
        attn = dm.blocks[block].attn
        keys = attn.pre_rope[1]
        return keys.repeat_interleave(dm.heads // attn.kvheads, dim=1)

    def test_coarse_phase_pools_unrotated_keys(self, seam, monkeypatch):
        """Phase I pools the keys the model had NOT rotated yet.

        The model rotates q/k before the seam (``model.py:93``), so pooling
        the seam's keys would pool keys rotated by DIFFERENT angles inside one
        block (``src/area_attn.py:364-368``).
        """
        dm, _patched, _backend = seam
        seen = {}
        real = area.pool_keys_values

        def spy(key, value, txt_len, h, w, f, pooling, **kw):
            seen["key"] = key.clone()
            seen["block"] = kw.get("block")
            return real(key, value, txt_len, h, w, f, pooling, **kw)

        monkeypatch.setattr(area, "pool_keys_values", spy)
        rt, _out = self._run(dm, 0)
        assert seen, "Phase I never pooled"
        pre = self._pre_rope_keys(dm, seen["block"])
        rotated = apply_rope1(pre, rt.recorder.table)
        # the pooled keys are the model's PRE-rotation ones ...
        assert torch.allclose(seen["key"], pre, atol=1e-3)
        # ... and NOT the rotated ones the model handed to the seam
        assert not torch.allclose(seen["key"], rotated, atol=1e-3)

    def test_coarse_phase_never_pools_the_text_block(self, seam, monkeypatch):
        """The caption tokens pass through the pooling untouched."""
        dm, _patched, _backend = seam
        seen = {}
        real = area.pool_keys_values

        def spy(key, value, txt_len, h, w, f, pooling, **kw):
            k, v = real(key, value, txt_len, h, w, f, pooling, **kw)
            seen["key"] = key.clone()
            seen["pooled"] = k.clone()
            seen["txt_len"] = txt_len
            seen["f"] = f
            return k, v

        monkeypatch.setattr(area, "pool_keys_values", spy)
        self._run(dm, 0)
        t = seen["txt_len"]
        assert t == TXT_LEN
        # the caption block is byte-identical in the pooled keys
        assert torch.equal(seen["pooled"][:, :, :t], seen["key"][:, :, :t])
        # and the canvas shrank to one token per f x f block
        assert seen["pooled"].shape[2] == t + (GRID // seen["f"]) ** 2

    def test_coarse_phase_uses_the_models_coarse_tables(self, seam, monkeypatch):
        """The coarse tables come from the model's OWN embedder, in FLUX's shape."""
        dm, _patched, _backend = seam
        seen = {}
        ids = []
        real = area.coarse_ropes_krea
        real_embedder = dm.pe_embedder.forward

        def spy_embedder(ids_arg):
            ids.append(ids_arg.clone())
            return real_embedder(ids_arg)

        def spy_coarse(embedder, layout, f, device, recorder=None):
            seen["embedder"] = embedder
            seen["recorder"] = recorder
            out = real(embedder, layout, f, device, recorder)
            seen["q"] = tuple(out[0].shape)
            seen["kv"] = tuple(out[1].shape)
            return out

        monkeypatch.setattr(area, "coarse_ropes_krea", spy_coarse)
        dm.pe_embedder.forward = spy_embedder
        rt, _out = self._run(dm, 0)
        assert seen["embedder"] is dm.pe_embedder
        assert seen["recorder"] is rt.recorder
        # the q side covers the whole joint stream, the kv side the pooled grid
        assert seen["q"][2] == TXT_LEN + GRID * GRID
        assert seen["kv"][2] == TXT_LEN + (GRID // rt.ctx.size) ** 2
        # the ids are FLUX-shaped: the caption at the origin, axis 0 (the frame
        # index) zero everywhere (comfy/ldm/krea2/model.py:291-294, :346)
        coarse_ids = ids[-2]
        assert coarse_ids.shape == (1, TXT_LEN + GRID * GRID, 3)
        assert torch.allclose(coarse_ids[0, :, 0], torch.zeros(TXT_LEN + GRID * GRID))
        assert torch.allclose(coarse_ids[0, :TXT_LEN], torch.zeros(TXT_LEN, 3))
        # the pause guard: the coarse tables are not extra tables of the forward
        assert rt.recorder.extra_tables == 0

    def test_fine_phase_keeps_the_models_rotation(self, seam, monkeypatch):
        """Phase II attends at the model's OWN positions — no second rotation."""
        dm, _patched, _backend = seam
        seen = {}
        real = area.windowed_attention

        def spy(q_txt, k_txt, v_txt, q_img, k_img, v_img, k_unrot, layout, **kw):
            seen["k_img"] = k_img.clone()
            seen["k_unrot"] = k_unrot.clone()
            seen["kw"] = kw
            return real(q_txt, k_txt, v_txt, q_img, k_img, v_img, k_unrot, layout, **kw)

        monkeypatch.setattr(area, "windowed_attention", spy)
        rt, _out = self._run(dm, 10, rt=_runtime(dm, native=4, phase1_steps=0))
        assert seen, "Phase II never ran"
        pre = self._pre_rope_keys(dm, len(dm.blocks) - 1)
        rotated = apply_rope1(pre, rt.recorder.table)[:, :, TXT_LEN:]
        # the image keys the seam passed are the model's ROTATED ones ...
        assert torch.allclose(seen["k_img"], rotated, atol=1e-3)
        # ... and the bridge's keys are the UNROTATED ones (bridge.py:10-24)
        assert torch.allclose(seen["k_unrot"], pre[:, :, TXT_LEN:], atol=1e-3)

    def test_fine_phase_bridge_uses_axis_zero_zero(self, seam, monkeypatch):
        """The bridge keys are re-roped at Krea-2's frame index (axis 0 = 0).

        Z-Image's convention is ``cap_len + 1`` (the first token after the
        caption block); Krea-2's image tokens carry the FRAME index on axis 0
        (``comfy/ldm/krea2/model.py:291-294``), which is 0 for an image.  The
        seam hands the engine that value; the engine's own use of it is pinned
        in ``tests/test_area_attn.py`` (``test_img_axis0_ropes_the_bridge_keys_at
        _the_backbones_axis_zero``).
        """
        dm, _patched, _backend = seam
        seen = {}
        real = area.windowed_attention

        def spy(q_txt, k_txt, v_txt, q_img, k_img, v_img, k_unrot, layout, **kw):
            seen["kw"] = kw
            return real(q_txt, k_txt, v_txt, q_img, k_img, v_img, k_unrot, layout, **kw)

        monkeypatch.setattr(area, "windowed_attention", spy)
        self._run(dm, 10, rt=_runtime(dm, native=4, phase1_steps=0))
        assert seen["kw"]["img_axis0"] == 0.0
        assert seen["kw"]["cap_len"] == TXT_LEN
        assert seen["kw"]["rope_embedder"] is dm.pe_embedder

    def test_fine_phase_caption_is_anchored_at_the_window_origin(self, seam,
                                                                  monkeypatch):
        """GTCN: the caption is re-roped at EACH window's own origin."""
        dm, _patched, _backend = seam
        ids = []
        real_embedder = dm.pe_embedder.forward

        def spy_embedder(ids_arg):
            ids.append(ids_arg.clone())
            return real_embedder(ids_arg)

        dm.pe_embedder.forward = spy_embedder
        rt, _out = self._run(dm, 10, rt=_runtime(dm, native=4, phase1_steps=0))
        anchors = [i for i in ids if i.dim() == 3 and i.shape[1] == TXT_LEN]
        assert anchors, "the caption anchor was never built"
        assert len(anchors) == 1, "the anchor table was rebuilt per block"
        anchor = anchors[0]
        # axis 0 is zero — an identity rotation, Krea-2's frame index
        assert torch.allclose(anchor[:, :, 0], torch.zeros_like(anchor[:, :, 0]))
        layout = rt.ctx.data["layout"]
        for row, win in enumerate(layout.windows):
            assert float(anchor[row, 0, 1]) == float(win.r0)
            assert float(anchor[row, 0, 2]) == float(win.c0)

    def test_output_is_the_flattened_layout(self, seam):
        """The seam answers comfy's flattened ``[b, n, heads*d]``."""
        dm, _patched, backend = seam
        wrapped = backend.optimized_attention_masked
        seen = []

        def spy(*args, **kwargs):
            out = wrapped(*args, **kwargs)
            seen.append((int(args[3]), tuple(out.shape)))
            return out

        backend.optimized_attention_masked = spy
        self._run(dm, 0)
        main = [shape for heads, shape in seen if heads == HEADS]
        assert len(main) == LAYERS
        for shape in main:
            assert shape == (1, TXT_LEN + GRID * GRID, HEADS * HEAD_DIM)

    def test_output_shape_matches_the_pristine_seam(self, seam):
        """The answer has the shape the model's ``out * sigmoid(gate)`` needs."""
        dm, _patched, _backend = seam
        params = _params(dm)
        reference = _KreaModel(dm)(params["input"], params["timestep"], **params["c"])
        _rt, out = self._run(dm, 0, params=params)
        assert out.shape == reference.shape

    def test_aux_calls_do_not_advance_the_block_counter(self, seam, monkeypatch):
        """The four 20-head calls must not eat the block ordinals 0..3.

        The main blocks are numbered 0..27 exactly as the reference numbers its
        blocks; if the auxiliary calls consumed ordinals, every block's random
        pooling stream would shift (the HAP bug class, ``src/spa.py:1033-1052``).
        """
        dm, _patched, _backend = seam
        blocks = []
        real = area.pool_keys_values

        def spy(key, value, txt_len, h, w, f, pooling, **kw):
            blocks.append(kw.get("block"))
            return real(key, value, txt_len, h, w, f, pooling, **kw)

        monkeypatch.setattr(area, "pool_keys_values", spy)
        rt, _out = self._run(dm, 0)
        assert blocks[0] == 0, "the first main block did not get ordinal 0"
        assert rt.block == LAYERS, "the counter ran past the 28 main blocks"

    def test_block_counter_numbers_the_main_blocks_0_to_27(self, seam, monkeypatch):
        """The ordinals are exactly 0..27, in order, once per main block."""
        dm, _patched, _backend = seam
        blocks = []
        real = area.pool_keys_values

        def spy(key, value, txt_len, h, w, f, pooling, **kw):
            blocks.append(kw.get("block"))
            return real(key, value, txt_len, h, w, f, pooling, **kw)

        monkeypatch.setattr(area, "pool_keys_values", spy)
        self._run(dm, 0)
        assert blocks == list(range(LAYERS))

    def test_missing_rope_table_delegates_to_the_pristine_base(self, seam,
                                                                monkeypatch):
        """A forward whose table was not captured runs the PRISTINE attention."""
        dm, _patched, _backend = seam
        params = _params(dm)
        model = _KreaModel(dm)
        reference = model(params["input"], params["timestep"], **params["c"])

        monkeypatch.setattr(area.KreaRopeRecorder, "record",
                            lambda self, table: None)
        _rt, out = self._run(dm, 0, params=params)
        assert torch.equal(out, reference)
        assert "krea-table" in area._warned

    def test_rng_streams_are_keyed_by_seed_and_step(self, seam):
        """No global RNG is read, and the same (seed, step) rebuilds the same run."""
        dm, _patched, _backend = seam
        state = torch.random.get_rng_state()
        _rt1, out1 = self._run(dm, 10, rt=_runtime(dm, native=4, phase1_steps=0))
        assert torch.equal(torch.random.get_rng_state(), state)
        _rt2, out2 = self._run(dm, 10, rt=_runtime(dm, native=4, phase1_steps=0))
        assert torch.equal(out1, out2)

    def test_cond_and_uncond_share_one_layout(self, seam, monkeypatch):
        """One batched forward -> ONE layout for both branches."""
        dm, _patched, _backend = seam
        layouts = []
        real = area.windowed_attention

        def spy(q_txt, k_txt, v_txt, q_img, k_img, v_img, k_unrot, layout, **kw):
            layouts.append(id(layout))
            return real(q_txt, k_txt, v_txt, q_img, k_img, v_img, k_unrot, layout, **kw)

        monkeypatch.setattr(area, "windowed_attention", spy)
        params = _params(dm, step=10, batch=2)
        self._run(dm, 10, rt=_runtime(dm, native=4, phase1_steps=0), params=params)
        assert layouts
        assert len(set(layouts)) == 1
