"""AReA — training-free, time-dependent replacement of FLUX image self-attention.

This module is the model-side wiring (the reference's ``area/backbones/flux.py``
and ``area/backbones/base.py``): it patches a ComfyUI FLUX model so that every
block's attention runs the two AReA phases built from :mod:`src.area_schedule`
and :mod:`src.area_attn` instead of dense attention.  The node
(``nodes/area.py``) is a thin schema/execute shell over
:func:`apply_area_to_model`.

**Two seams, one engine.**  FLUX and Z-Image (Lumina2) are different backbones with
different seams, so :func:`apply_area_to_model` resolves an
:class:`~src.area_arch._ArchProfile` once and installs exactly one of them:

* FLUX — the bound ``attention`` of ``comfy.ldm.flux.layers`` (below);
* Z-Image — the bound ``optimized_attention_masked`` of
  ``comfy.ldm.lumina.model`` (``comfy/ldm/lumina/model.py:14`` binds the name, ``:182``
  calls it), because ``JointAttention.forward`` applies rope ITSELF (``:159-173``) and
  hands the seam **rotated** q/k with no rope table.  The z-image adapter
  (:mod:`src.area_zimage`) un-rotates them with the model's own captured table, slices
  the 32-token pad spans out of the stream, and rebuilds the coarse tables from
  lumina's ids; this module owns the ref-counted install and the per-forward state.
  Unlike FLUX — where SPA binds a *different* symbol — SPA patches this very symbol for
  Z-Image too (``src/spa.py:590-591``), so the install refuses to wrap a foreign hook.

Phase I ("coarse") pools image K/V to one token per ``f x f`` block and runs
cross-scale attention on a coarse rope grid while the queries stay at full
resolution.  Phase II ("fine") runs windowed attention whose windows come from
the Haar-DWT detail map of the embedded image tokens, each window extended by a
randomized Halo ring and a global-KV bridge of distinctive tokens.  At or below
the native extent (64 patches = 1024 px) AReA is a *provable* no-op: the
schedule is not built and attention is delegated untouched (the reference's
``Backbone.schedule``, ``backbones/base.py:59-68``).

**The seam.** AReA must *replace* the attention computation and Phase I must pool
K/V *before* rope with *different* q/k tables, so neither the ``attn1_patch``
seam (which only rewrites q/k/v/pe before a fixed dense attention,
``comfy/ldm/flux/layers.py:236-241``) nor the ``optimized_attention`` seam SPA
uses (which sees q/k *after* native rope) fits.  The seam is the module-level
function ``attention`` **as bound into the backend module**:
``comfy/ldm/flux/layers.py:10`` does ``from .math import attention``, so the live
symbol is ``comfy.ldm.flux.layers.attention``, called at ``layers.py:244``
(double-stream) and ``layers.py:349`` (single-stream) — one symbol covers both.
Patching the *defining* module (``comfy.ldm.flux.math.attention``) would be a
silent no-op, the trap SPA documents at ``src/spa.py:558-567``.  The wrapper
receives ``(q, k, v, pe, mask, transformer_options, preferred_attention)`` with
q/k/v as ``AttentionTensorContainer``s holding the joint ``[b, heads, T + h*w,
d]`` stream *unrotated* — everything Phase I needs.  Because SPA/HAP bind
``comfy.ldm.flux.math.optimized_attention`` (a different symbol) AReA does not
collide with them; it simply takes precedence on the steps where it runs (it
does not call down), and on plain-attention steps it delegates so an installed
SPA/HAP hook still fires.

**Per-forward state.** One unet wrapper (``model_function_wrapper``,
``comfy/samplers.py:332-335``) resolves one AReA context per forward: the step
index and total from ``transformer_options["sample_sigmas"]`` / ``["sigmas"]``
(``comfy/samplers.py:316,1229``), the patch grid from the latent, the text
length from ``c_crossattn``, and the coarse rope tables from the *installed*
``pe_embedder`` (so a chained DyPE/SEGA embedder composes).  Cond and uncond
share ONE batched forward and timestep (``comfy/samplers.py:309``), so both
branches resolve the identical step and attend through identical
windows/halos/bridge layouts (theory.md:496).  The Phase II detail map comes
from a forward hook on ``diffusion_model.img_in`` — the comfy analogue of the
reference's ``x_embedder`` hook (``backbones/flux.py:113,150-163``).  The
context is set before and cleared after every forward (the SPA discipline,
``src/spa.py:1385-1476``), so no layout may leak across models or steps.

Deviations from the reference, all deliberate and pinned by the Phase 3 tests:

* **Output layout.** ``comfy.ldm.flux.layers.attention`` returns
  ``[b, N, heads*dim]``, NOT ``[b, heads, N, d]``: ``math.attention`` calls
  ``optimized_attention(..., skip_reshape=True)`` without
  ``skip_output_reshape`` (``ldm/modules/attention.py:303-314``), and the FLUX
  blocks slice the result on axis 1 (``layers.py:250``).  This module therefore
  returns ``out.permute(0, 2, 1, 3).reshape(b, n, heads * dim)`` — the exact
  flatten order comfy uses — while all the AReA math stays heads-first.
* **Step location.** The reference compares *timesteps*; comfy hands the
  attention call the current **sigma** (``transformer_options["sigmas"]``,
  ``samplers.py:316``) plus the whole sigma table, so the same ``argmin`` rule
  (``backbones/base.py:48-57``) is evaluated in sigma space against
  ``sample_sigmas[:-1]`` — the trailing terminal zero is dropped exactly as the
  reference drops ``scheduler.sigmas[:-1]``.  This avoids depending on
  ``model_sampling.timestep`` (a discrete index map for FLUX's
  ``ModelSamplingDiscrete``) and gives the identical answer for any sigma the
  sampler actually uses.
* **CFG batching.** comfy batches cond and uncond into ONE forward, so the
  reference's batch reductions are kept verbatim (:func:`detail_map` means over
  the batch, :func:`token_importance` sums over it) and one layout serves the
  whole batch.
* **Detail-map fallback.** When ``diffusion_model.img_in`` cannot be hooked
  (not an ``nn.Module``), the Phase II detail map falls back to the
  post-norm projected image keys of the first fine call, with a one-time
  warning.  Everything else is unchanged.
* **Short-schedule bridge floor (A1) and geometry (A4).** Deliberate, opt-out
  deviations on short (distilled) schedules only — ``bridge_floor_steps`` leading
  fine steps keep the paper's full bridge/halo above the cost cap, and
  ``window_start`` / ``window_hold_steps`` reshape the window ramp
  (``docs/2026-10-18-area-zimage-adaptation-research.md`` §4-§5).  Both are gated
  on the shared :func:`~src.area_schedule.is_short_schedule` predicate, so the
  28-step FLUX path — and every long schedule — is byte-identical.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .area_attn import (
    POOLING,
    AreaContext,
    FineLayout,
    adaptive_tiling,
    apply_rope1,
    bridge_registers,
    build_fine_layout,
    detail_map,
    fit_extra_keys,
    pool_keys_values,
    tiling_rng,
    windowed_attention,
    windowed_step_cost,
)
from .area_arch import (
    FLUX_PROFILE,
    ZIMAGE_PROFILE,
    _ArchProfile,
    resolve_area_arch,
)
from .area_schedule import AreaSchedule, build_schedule, initial_factor, is_short_schedule
from .area_zimage import (
    ZIMAGE_GROUP_ORDER,
    ZIMAGE_SEAM_MODULE,
    ZIMAGE_SEAM_SYMBOL,
    ZIMAGE_SIGMA_STAR,
    ZImageRopeRecorder,
    capture_tables,
    coarse_ropes_zimage,
    sigma_prefix_count,
    stitch_fine_zeros,
    unrotate1,
    zimage_layout,
)
from .model_detect import resolve_model_type
from .patch_utils import _snap_to_multiple

logger = logging.getLogger("ComfyUI-DyPE")

# FLUX constants (reference backbones/flux.py:15-16, plan §1.1)
NATIVE_PATCHES = 64    # 1024 px / 16 px per patch — the model's trained extent
PHASE1_STEPS = 8       # of 28
FLUX_PATCH_SIZE = 2    # ldm/flux/model.py: patchify on an 8x-downscaled latent (16 px/token)
PATCH_PX = 16          # VAE 8 x patch 2 — the node's geometry multiple

# The seam: the bound symbol the FLUX blocks actually call (layers.py:10,244,349).
FLUX_ATTENTION_MODULE = "comfy.ldm.flux.layers"
FLUX_ATTENTION_SYMBOL = "attention"

# The Z-Image seam: the symbol ``JointAttention.forward`` calls after applying rope
# itself — imported into the backend module at comfy/ldm/lumina/model.py:14, called
# at :182, so the *bound* name is the only symbol patching it moves (the same trap
# SPA documents at src/spa.py:558-567).  Named by the adapter (plan §4,
# ``src/area_zimage.ZIMAGE_SEAM_MODULE`` / ``ZIMAGE_SEAM_SYMBOL``) and imported
# above, so the interface has one definition.

#: AReA-private patcher attrs, carried across a later clone (``ModelPatcher.clone()``
#: drops custom attrs; same reason as SPA/HAP's carry-over).
_AREA_PATCHER_ATTRS = ("_area_config",)

# One-time warnings (module scoped, like SPA's legacy-knob migration warning).
_warned: set = set()


def _warn_once(key: str, message: str, *args) -> None:
    """Emit ``message`` at WARNING level the first time ``key`` is seen."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


def _warn_dynamic_embedder(embedder) -> None:
    """Warn once about a timestep-dependent positional embedder (both seams).

    A DyPE-family embedder carries dynamic, per-step scaling, so AReA's coarse
    positions now ride on it and its scaling also moves the pooled keys — the
    I-Max takeover warning (``nodes/imax.py:287-294``).
    """
    if hasattr(embedder, "set_timestep"):
        _warn_once(
            "dynamic-embedder",
            "AReA: the installed positional embedder is a dynamic scaling "
            "wrapper (%s) — AReA's coarse-grid rope positions now ride on it, so "
            "its per-step scaling also moves the pooled keys. Chain only one of "
            "DyPE / I-Max / SEGA positional scaling.",
            type(embedder).__name__,
        )


@dataclass(frozen=True)
class AreaConfig:
    """Everything AReA needs that is not resolved per forward."""

    phase1_steps: int = PHASE1_STEPS
    #: How the Phase I -> II switch is expressed: ``"steps"`` (the reference's
    #: ``phase1_steps``) or ``"sigma"`` (the leading steps whose noise level is
    #: still at or above :attr:`phase1_sigma`, theory.md:240-246 / Q4).
    phase1_mode: str = "steps"
    phase1_sigma: float = ZIMAGE_SIGMA_STAR
    pooling: str = "subsample"
    halo: int = 8
    bridge_ratio: float = 0.5
    #: A1 — how many leading Phase II steps keep the paper's full bridge and
    #: halo instead of the cost cap's trimmed ones.  Short schedules only
    #: (:func:`~src.area_schedule.is_short_schedule`); ``0`` disables the floor,
    #: which restores the capped behaviour on every step.
    bridge_floor_steps: int = 2
    #: A4 — window side Phase II starts its ramp at (``0`` = the Eq. 8
    #: cost-optimal choice), clamped into ``[w_min, extent]``; short only.
    window_start: int = 0
    #: A4 — extra steps that hold the start side before the ramp shrinks it;
    #: short only, clamped to ``[0, fine_steps - 2]``.
    window_hold_steps: int = 0
    seed: int = 0
    native: int = NATIVE_PATCHES


class _AreaRuntime:
    """One patcher's AReA state plus the per-forward per-call caches.

    ``dm`` is the diffusion model the patcher patches (resolved once at apply
    time so the per-forward rope lookup sees the *installed* embedder — comfy
    applies ``object_patches`` in place around model use, so a chained DyPE/SEGA
    embedder is visible at forward time).  ``profile`` is the backbone that
    patcher resolved to; the seam install, the rope tables and the pad policy
    all branch on it.

    Per forward: ``ctx`` is the current step (``None`` phase = plain attention,
    delegated untouched), ``hidden`` is the ``img_in`` output of a FLUX forward or
    the ``noise_refiner`` output of a Z-Image one (its Phase II detail-map input),
    and ``block`` counts attention calls so the reference's per-block ``random``
    pooling stream is reproduced (double-stream blocks 0..N then single-stream,
    the same order as the reference's
    ``list(transformer_blocks) + list(single_transformer_blocks)``).
    ``ropes`` / ``schedules`` are per-canvas caches (the reference's
    ``self._ropes`` / ``self._schedule``).

    Z-Image adds two fields: ``recorder`` captures the model's own rope groups in
    call order (``src/area_zimage.py``, built by its :func:`capture_tables`) and
    owns the arming window through :attr:`~src.area_zimage.ZImageRopeRecorder.armed`
    — open from the start of the forward and closed by the ``noise_refiner``
    hook, so only ``embed_cap`` / ``embed_all``'s two groups are ever recorded —
    and ``joint`` is the resolved pad layout of the current forward (``None``
    until the joint stream is identified).
    """

    __slots__ = ("config", "dm", "profile", "ctx", "hidden", "ropes", "schedules",
                 "block", "recorder", "joint")

    def __init__(self, config: AreaConfig, dm, profile: Optional[_ArchProfile] = None):
        self.config = config
        self.dm = dm
        self.profile = profile
        self.ctx: Optional[AreaContext] = None
        self.hidden: Optional[torch.Tensor] = None
        self.ropes: dict = {}
        self.schedules: dict = {}
        self.block = 0
        self.recorder: Optional[ZImageRopeRecorder] = None
        self.joint = None


#: The runtime of the FLUX forward currently running, read by the FLUX attention
#: seam.  Set and cleared around every forward (SPA discipline) — ``None`` between
#: forwards, which is what makes an unpatched/inactive model a pure no-op.
_AREA_RUNTIME: Optional[_AreaRuntime] = None

#: The same for Z-Image: the two globals are set exclusively (the unet wrapper
#: assigns exactly one per forward), so neither backbone's seam can fire on the
#: other's forward and the two never share a runtime.
_ZRUNTIME: Optional[_AreaRuntime] = None

# Seam install bookkeeping: the patched module, the original symbol, the
# ``img_in`` hook handle and the number of patchers consuming them.  A second
# AReA apply re-uses the wrapper instead of wrapping our own wrapper, and the
# symbol is restored only when the last consumer unpatches.
_SEAM = {"module": None, "attr": None, "orig": None, "hook": None, "consumers": 0}

# The Z-Image twin of ``_SEAM`` — a *different* module and symbol, so a different
# ref-count: FLUX and Z-Image installs never share one.  ``hook`` holds EVERY
# handle this install registered on the first consumer's diffusion model (the
# rope recorder and the ``noise_refiner`` layout hook), which is why it is a
# tuple here and a single handle on the FLUX side, and ``recorder`` is the
# capture :func:`src.area_zimage.capture_tables` built for it — shared by every
# consumer of this install, since the hook that feeds it is registered once on
# the first consumer's model (the same reason the symbol is ref-counted).
_ZSEAM = {"module": None, "attr": None, "orig": None, "hook": None, "recorder": None,
          "consumers": 0}


# ---------------------------------------------------------------------------
# Apply / unpatch
# ---------------------------------------------------------------------------


def _area_carry_state(src, dst) -> None:
    """Copy AReA's private patcher attrs from ``src`` to ``dst``.

    ``ModelPatcher.clone()`` keeps only its known fields, so an AReA-patched
    source patcher would silently lose its config on any later node's clone.
    Called right after ``clone()`` and before any early return, so node order
    never decides whether AReA is live.
    """
    for attr in _AREA_PATCHER_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def _grid_from_latent(x: torch.Tensor, patch_size: int) -> tuple[int, int]:
    """Patch grid ``(h, w)`` of a FLUX latent, padding included.

    ``ldm/flux/model.py:352-356`` pads the latent to the patch size and
    rearranges it into ``(h * w)`` tokens of ``ph x pw`` patches, so the grid is
    ``ceil(H / patch_size) x ceil(W / patch_size)``.
    """
    steps = (x.shape[-2] + patch_size // 2) // patch_size
    return steps, (x.shape[-1] + patch_size // 2) // patch_size


def _area_locate(sigmas, current) -> tuple[Optional[int], Optional[int]]:
    """``(step, total)`` of this forward, by the reference's ``argmin`` rule.

    ``backbones/base.py:48-57`` matches the running timestep against the
    scheduler's schedule (minus its terminal entry) and returns the index and
    the length.  Comfy hands the attention call the current **sigma**
    (``transformer_options["sigmas"]``, ``samplers.py:316``) and the whole table
    as ``sample_sigmas`` (``samplers.py:1229``), so the same rule is evaluated in
    sigma space here — see the module docstring.
    """
    try:
        if not torch.is_tensor(sigmas) or sigmas.numel() < 2:
            raise ValueError("sample_sigmas missing or too short")
        value = float(current.detach().float().flatten()[0]) if torch.is_tensor(current) else float(current)
        table = sigmas.detach().float().flatten().cpu()[:-1]
        step = int(torch.argmin((table - value).abs()))
        return step, int(table.numel())
    except (TypeError, ValueError, RuntimeError, IndexError) as exc:
        # A deterministic no-op beats a wrong schedule: run plain attention and
        # say so once (plan §0, "remaining risks").
        _warn_once(
            "no-sigmas",
            "AReA: the full sigma schedule is not available on this sampler "
            "(transformer_options['sample_sigmas'] is missing or unreadable), so "
            "the step index — and with it the AReA schedule — cannot be located. "
            "This forward runs plain attention instead of a guessed one. (%s)",
            exc,
        )
        return None, None


def _area_schedule(rt: _AreaRuntime, h: int, w: int, total: int, phase1: int,
                   txt_len: int) -> AreaSchedule:
    """The schedule of one canvas, cached per ``(h, w, total, phase1, txt_len)``.

    The reference caches on exactly this key (``backbones/base.py:64-66``): the
    windows of a step depend on the canvas and on how many text tokens the
    prompt carries, both constant across a generation.  ``phase1`` is the
    *resolved* Phase I count (``_phase1_count``), not the raw knob, so sigma mode
    — whose count comes from the schedule itself — is cached correctly.

    The A4 geometry knobs join the key because they change the windows they
    produce: two runtimes differing only in ``window_start`` / ``window_hold_steps``
    must not share one entry.  ``bridge_floor_steps`` is deliberately NOT in the
    key — the floor is applied per step by the layout builder, so it never
    changes the schedule (the resolution that produced it is irrelevant to it).
    """
    cfg = rt.config
    phase1 = max(0, int(phase1))
    key = (h, w, total, phase1, txt_len, cfg.window_start, cfg.window_hold_steps)
    schedule = rt.schedules.get(key)
    if schedule is None:
        schedule = build_schedule(h, w, total, phase1, cfg.native, txt_len=txt_len,
                                  window_start=cfg.window_start,
                                  window_hold_steps=cfg.window_hold_steps)
        rt.schedules[key] = schedule
    return schedule


def _phase1_count(rt: _AreaRuntime, opts: dict) -> int:
    """How many leading steps of this run are Phase I — the node's switch.

    ``"steps"`` is the reference's fixed count (``config.phase1_steps``);
    ``"sigma"`` counts the leading steps whose noise level is still at or above
    ``phase1_sigma`` (:func:`~src.area_zimage.sigma_prefix_count`, ``theory.md:
    240-246``), which follows the schedule rather than a fixed count (Q4).  The
    trailing terminal zero of ``sample_sigmas`` is dropped exactly as
    :func:`_area_locate` drops it, so the count and the step index read the same
    table.
    """
    cfg = rt.config
    if cfg.phase1_mode == "bootstrap":
        # The paper's Appendix F mode: Phase I is replaced by a separate
        # native-resolution pass (the canvas is never pooled, so there is no
        # pooling footprint to ripple), so this run runs Phase II only.
        if int(cfg.phase1_steps):
            _warn_once(
                "area-bootstrap-steps",
                "AReA: phase1_mode is 'bootstrap' but phase1_steps is %d, which "
                "bootstrap ignores — Phase I runs no steps (Appendix F replaces "
                "it with a native-resolution pre-pass, see the plan "
                ".dev/docs/plans/2026-10-18-area-zimage-bootstrap.md).",
                int(cfg.phase1_steps),
            )
        return 0
    if cfg.phase1_mode != "sigma":
        return max(0, int(cfg.phase1_steps))
    table = opts.get("sample_sigmas")
    if not torch.is_tensor(table) or table.numel() < 2:
        _warn_once(
            "sigma-mode-no-table",
            "AReA: phase1_mode is 'sigma' but this sampler did not publish a "
            "usable sigma table, so the Phase I -> II switch cannot be located. "
            "This forward runs all Phase II (a deterministic choice).",
        )
        return 0
    return sigma_prefix_count(table.detach().float().flatten().cpu()[:-1], cfg.phase1_sigma)


def _area_rope_ids(txt_len: int, rows: torch.Tensor, cols: torch.Tensor,
                   device, dtype) -> torch.Tensor:
    """``[1, txt_len + n, 3]`` rope position ids: zero text ids, image rows/cols.

    ``ldm/flux/model.py:192-194`` builds exactly this (text ids all zero, image
    ids carrying row/col on axes 1/2 with the axis-0 index at 0), so a stock
    ``pe_embedder`` returns the same table the model itself computes for the
    canvas.
    """
    ids = torch.zeros(1, txt_len + int(rows.numel()), 3, device=device, dtype=dtype)
    ids[0, txt_len:, 1] = rows.to(dtype)
    ids[0, txt_len:, 2] = cols.to(dtype)
    return ids


def _area_coarse_ropes(rt: _AreaRuntime, h: int, w: int, f: int, txt_len: int,
                       device) -> tuple[torch.Tensor, torch.Tensor]:
    """The two Phase I rope tables (the reference's ``_coarse_ropes``).

    Queries keep full resolution and ride on their block's coarse position, the
    pooled keys sit on the ``(h/f) x (w/f)`` grid, so both are measured in the
    same units.  Tables come from the *installed* embedder so a chained DyPE/SEGA
    embedder composes (the reference calls its own ``pos_embed``).
    """
    embedder = getattr(rt.dm, "pe_embedder", None)
    if embedder is None:
        raise ValueError(
            "AReA: this model has no diffusion_model.pe_embedder, so the Phase I "
            "coarse rope tables cannot be built. Use an unmodified FLUX loader."
        )
    _warn_dynamic_embedder(embedder)
    key = ("coarse", h, w, f, txt_len, id(embedder), str(device))
    cached = rt.ropes.get(key)
    if cached is not None:
        return cached
    pos = torch.arange(h * w, device=device)
    coarse = torch.arange((h // f) * (w // f), device=device)
    rope_q = embedder(_area_rope_ids(txt_len, pos // w // f, pos % w // f, device, torch.float32))
    rope_kv = embedder(
        _area_rope_ids(txt_len, coarse // (w // f), coarse % (w // f), device, torch.float32)
    )
    rt.ropes[key] = (rope_q, rope_kv)
    return rope_q, rope_kv


def _area_fine_layout(rt: _AreaRuntime, ctx: AreaContext, k_img: torch.Tensor,
                      hook_name: str = "diffusion_model.img_in") -> Optional[FineLayout]:
    """Build (once per step) the Phase II layout — the reference's ``_after_embedding``."""
    layout = ctx.data.get("layout")
    if layout is not None:
        return layout
    cfg = rt.config
    h, w, t = ctx.h, ctx.w, ctx.txt_len
    hidden = rt.hidden
    if hidden is None:
        # Fallback: the detail map is derived from the post-norm projected image
        # keys instead of the hooked hook's output (see the module docstring).
        _warn_once(
            "no-img-in",
            "AReA: %s could not be hooked, so the Phase II detail map is derived "
            "from the projected image keys of the first fine call. Results differ "
            "slightly from a hooked model.",
            hook_name,
        )
        # [b, heads, n, d] -> [b, n, heads*d], the img_in output's layout
        hidden = k_img.transpose(1, 2).reshape(*k_img.shape[:1], k_img.shape[2], -1)
    energy = detail_map(hidden, h, w)
    w_min = min(cfg.native, max(h, w))
    windows = adaptive_tiling(energy, h, w, ctx.size, w_min, t, rng=tiling_rng(cfg.seed, ctx.step))
    if ctx.fine_step < ctx.floor_steps:
        # A1 — the bridge floor (``docs/2026-10-18-area-zimage-adaptation-research.md``
        # §4/§5-A1): on a SHORT schedule the cost cap cuts the bridge to zero and
        # the Halo to a sliver on exactly the steps where composition is still
        # forming, which the paper's own ablation links to window duplication.  The
        # leading ``floor_steps`` fine steps therefore keep the paper's full
        # ``round(rho*w)^2`` bridge and the configured halo, exempt from the cap —
        # an explicit, warned cost decision, not a silent one.
        registers, halo = bridge_registers(ctx.size, cfg.bridge_ratio), cfg.halo
        cost = windowed_step_cost(windows, h, w, t, registers, halo)
        if cost > 1.0:
            _warn_once(
                "area-bridge-floor",
                "AReA: the first %d fine step(s) of this short schedule keep the "
                "full bridge (R=%d per window) and halo=%d, which costs %.2fx a "
                "full-attention step instead of the cost-capped 1.0x — the cap is "
                "deliberately waived there, because it would cut the bridge to "
                "zero on the composition-critical steps. Set bridge_floor_steps=0 "
                "to cap every step instead.",
                ctx.floor_steps, registers, halo, cost,
            )
    else:
        registers = bridge_registers(ctx.size, cfg.bridge_ratio)
        # Never cost more than a full-attention step: cut bridge tokens, then the Halo.
        registers, halo = fit_extra_keys(windows, h, w, t, registers, cfg.halo)
    layout = build_fine_layout(windows, h, w, halo, registers, cfg.seed, ctx.step,
                               k_img.dtype, k_img.device)
    ctx.data["layout"] = layout
    return layout


def _take(x):
    """Consume an attention container once (``ldm/modules/attention.py:193-209``).

    Duck-typed on purpose: the engine imports no comfy module at runtime, and the
    conformance test supplies its own ``peek``/``take`` container.
    """
    return x.take() if hasattr(x, "take") else x


def _area_delegate(orig, q, k, v, pe, mask, transformer_options, preferred_attention, cls):
    """Call the original attention with FRESH containers.

    The incoming containers are already consumed by :func:`_take`, and a taken
    container must never be re-wrapped (a second ``take`` raises,
    ``ldm/modules/attention.py:207``) — that is the v2.19.1 crash class.
    """
    if cls is not None:
        q, k, v = cls(q), cls(k), cls(v)
    return orig(q, k, v, pe=pe, mask=mask,
                transformer_options={} if transformer_options is None else transformer_options,
                preferred_attention=preferred_attention)


def _area_attention(rt: _AreaRuntime, ctx: AreaContext, q, k, v, pe):
    """The AReA computation of one attention call, or ``None`` to delegate.

    q/k/v are the taken ``[b, heads, N, d]`` joint tensors, unrotated.  Returns
    the comfy-shaped ``[b, N, heads*d]`` result (see the module docstring).
    """
    b, heads, n, dim = q.shape
    t = ctx.txt_len
    if n != t + ctx.h * ctx.w:
        _warn_once(
            f"geometry:{n}:{t}:{ctx.h}:{ctx.w}",
            "AReA: the attention stream has %d tokens but the %dx%d patch grid "
            "plus %d text tokens needs %d (a reference-token or non-standard "
            "FLUX path). This call runs plain attention.",
            n, ctx.h, ctx.w, t, t + ctx.h * ctx.w,
        )
        return None
    block = rt.block
    rt.block += 1
    if ctx.phase == "coarse":
        k_pooled, v_pooled = pool_keys_values(
            k, v, t, ctx.h, ctx.w, ctx.size, rt.config.pooling,
            seed=rt.config.seed, step=ctx.step, block=block,
        )
        rope_q, rope_kv = ctx.data["rope_q"], ctx.data["rope_kv"]
        out = F.scaled_dot_product_attention(
            apply_rope1(q, rope_q), apply_rope1(k_pooled, rope_kv), v_pooled
        )
    else:
        if pe is None:
            _warn_once(
                "no-pe",
                "AReA: this model emitted a fine step with no rope table (pe is "
                "None), so the windowed phase has no positions to attend by. "
                "This call runs plain attention.",
            )
            return None
        layout = _area_fine_layout(rt, ctx, k[:, :, t:])
        if layout is None:
            return None
        q_r, k_r = apply_rope1(q, pe), apply_rope1(k, pe)
        out_txt, out_img = windowed_attention(
            q_r[:, :, :t], k_r[:, :, :t], v[:, :, :t],
            q_r[:, :, t:], k_r[:, :, t:], v[:, :, t:],
            # the bridge ranks UNROTATED keys (bridge.py:10-24)
            k[:, :, t:], layout,
        )
        out = torch.cat([out_txt, out_img], dim=2)
    # comfy's own flatten: heads then head-dim, sequence back on axis 1
    # (ldm/modules/attention.py:309-314).
    return out.permute(0, 2, 1, 3).reshape(b, n, heads * dim)


def _make_area_wrapper(orig):
    """Wrap the FLUX ``attention`` seam so AReA replaces it.

    The wrapper is installed on the *bound* symbol
    ``comfy.ldm.flux.layers.attention``; a delegating call hands the job back to
    the original, which is what keeps SPA/HAP (installed one level below, on
    ``comfy.ldm.flux.math.optimized_attention``) alive on plain steps.
    """
    def attention(q, k, v, pe=None, mask=None, transformer_options=None, preferred_attention=None):
        rt = _AREA_RUNTIME
        ctx = None if rt is None else rt.ctx
        if ctx is None or ctx.phase is None:
            # No forward of ours, or a plain-attention step: delegate untouched.
            return orig(q, k, v, pe=pe, mask=mask, transformer_options=transformer_options,
                        preferred_attention=preferred_attention)
        cls = type(q) if hasattr(q, "take") else None
        q_t, k_t, v_t = _take(q), _take(k), _take(v)
        out = _area_attention(rt, ctx, q_t, k_t, v_t, pe)
        if out is None:
            return _area_delegate(orig, q_t, k_t, v_t, pe, mask, transformer_options,
                                  preferred_attention, cls)
        return out

    attention._area_wrapper = True
    return attention


# ---------------------------------------------------------------------------
# The Z-Image seam
# ---------------------------------------------------------------------------


def _zimage_take(x):
    """Consume a seam argument once — the engine duck-types its own, comfy's
    ``AttentionTensorContainer`` is single-owner (``ldm/modules/attention.py:193-209``)."""
    return x.take() if hasattr(x, "take") else x


def _zimage_delegate(orig, q, k, v, heads, mask, attn_precision, skip_reshape,
                     skip_output_reshape, kw, cls=None):
    """Call the pristine attention with FRESH containers.

    Mirrors :func:`_area_delegate`: a taken container must never be re-wrapped
    (a second ``take`` raises), which is the v2.19.1 crash class.  Everything the
    seam was handed is forwarded untouched — ``transformer_options`` and
    ``preferred_attention`` ride ``kw`` — so a delegating call is bit-identical
    to the unpatched model.
    """
    if cls is not None:
        q, k, v = cls(q), cls(k), cls(v)
    return orig(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kw)


def _zimage_check_foreign(symbol, when: str) -> None:
    """Refuse to install over another pack's hook on the same bound symbol.

    SPA's Z-Image target is this very symbol (``src/spa.py:590-591`` returns
    ``("comfy.ldm.lumina.model", "optimized_attention_masked", True)``) and it
    marks its wrapper ``_spa_wrapper`` (``src/spa.py:1362-1363``).  Unlike FLUX —
    where SPA binds ``comfy.ldm.flux.math.optimized_attention``, a *different*
    symbol, so AReA simply takes precedence on the steps it runs — there is ONE
    symbol here, so the two cannot coexist.  AReA never wraps another pack's hook
    silently: the install raises and the user reorders (plan §0, "Silent no-op
    avoided by ... foreign-binding collision ValueError").
    """
    if getattr(symbol, "_spa_wrapper", False):
        raise ValueError(
            f"AReA: {ZIMAGE_SEAM_MODULE}.{ZIMAGE_SEAM_SYMBOL} is already "
            "wrapped by SPA (src/spa.py:590-591 patches this same bound symbol "
            "for Z-Image, while for FLUX SPA binds comfy.ldm.flux.math."
            "optimized_attention instead — a different symbol). AReA never "
            "wraps another pack's hook silently: restore SPA before applying "
            "AReA, or run the two on separate models. On a shared symbol "
            "whichever pack installed last would own every step."
        )


def _zimage_pristine(bound):
    """The pristine symbol behind ``bound`` — an AReA wrapper carries its own.

    A re-apply after a patcher clone, or an install whose ref-count bookkeeping
    was reset underneath it, can find our own wrapper still bound with no recorded
    original.  The wrapper stows the symbol it wrapped (``_area_orig``), so the
    pristine is recoverable instead of wrapping our own wrapper.
    """
    if not getattr(bound, "_area_wrapper", False):
        return bound
    orig = getattr(bound, "_area_orig", None)
    if orig is None:
        raise ValueError(
            f"AReA: {ZIMAGE_SEAM_MODULE}.{ZIMAGE_SEAM_SYMBOL} is an AReA "
            "wrapper with no recorded pristine symbol, so the Z-Image attention "
            "seam cannot be installed safely. Restart ComfyUI to clear it."
        )
    return orig


def _make_zimage_wrapper(orig):
    """Wrap the Z-Image ``optimized_attention_masked`` seam so AReA replaces it.

    The signature mirrors the real call bit-for-bit
    (``comfy/ldm/lumina/model.py:182``) and comfy's own
    ``(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
    skip_output_reshape=False, **kwargs)``
    (``comfy/ldm/modules/attention.py``): ``heads`` and ``mask`` arrive
    positionally — a wrapper that mis-slots them fails silently, which is what
    ``tests/conftest.py:113-129`` and ``tests/test_orig_call_convention.py`` pin.
    The containers are consumed exactly once (``ldm/modules/attention.py:207``),
    and the output is comfy's sequence-first ``[b, n, heads*d]`` (``model.py:184``).
    """
    def optimized_attention_masked(q, k, v, heads, mask=None, attn_precision=None,
                                   skip_reshape=False, skip_output_reshape=False, **kw):
        rt = _ZRUNTIME
        ctx = None if rt is None else rt.ctx
        layout = _zimage_layout_for(rt, q)
        if ctx is None or ctx.phase is None or layout is None:
            # No forward of ours, a plain-attention step, or a refiner pass: delegate.
            return _zimage_delegate(orig, q, k, v, heads, mask, attn_precision,
                                    skip_reshape, skip_output_reshape, kw)
        cls = type(q) if hasattr(q, "take") else None
        q_t, k_t, v_t = _zimage_take(q), _zimage_take(k), _zimage_take(v)
        out = _zimage_attention(rt, ctx, layout, q_t, k_t, v_t, heads, kw)
        if out is None:
            return _zimage_delegate(orig, q_t, k_t, v_t, heads, mask, attn_precision,
                                    skip_reshape, skip_output_reshape, kw, cls)
        return out

    optimized_attention_masked._area_wrapper = True
    optimized_attention_masked._area_orig = orig
    return optimized_attention_masked


def _zimage_stream_len(q) -> Optional[int]:
    """The sequence length of a seam argument, without consuming its container."""
    tensor = q.peek() if hasattr(q, "peek") else q
    if torch.is_tensor(tensor) and tensor.ndim == 4:
        return int(tensor.shape[2])
    return None


def _zimage_layout_for(rt: Optional[_AreaRuntime], q):
    """The joint layout of THIS call, or ``None`` when it is not the joint stream.

    A plain Z-Image forward hits this symbol three ways: ``context_refiner``
    carries the caption block alone (``comfy/ldm/lumina/model.py:784-786``),
    ``noise_refiner`` the image block alone (``:815-822``), and the joint layer
    loop the concatenated stream (``:880-881``).  The joint stream is identified
    by its length — the padded caption block plus the padded image, which is
    strictly longer than either half — and by the recorder holding the model's
    own two rope groups in call order (the plan's group-identity rule, Q2).  Only
    that stream is eligible; everything else delegates untouched.
    """
    if rt is None or rt.ctx is None or rt.ctx.phase is None or rt.profile is None:
        return None
    ctx = rt.ctx
    if ctx.h * ctx.w < 1 or ctx.txt_len < 1:
        return None
    n = _zimage_stream_len(q)
    if n is None:
        return None
    if rt.joint is not None:
        return rt.joint if rt.joint.seq_len == n else None
    recorder = rt.recorder
    if recorder is None:
        return None
    try:
        cap_len, _img_len, _full, _groups = recorder.reads(ZIMAGE_GROUP_ORDER)
    except ValueError:
        # A refiner pass, or a rope call the arming window kept out: the model
        # has not made both joint groups yet, so this is not the joint stream.
        return None
    if n != cap_len + rt.profile.padded_seq_len(ctx.h * ctx.w):
        return None
    try:
        layout = zimage_layout(ctx.txt_len, cap_len, ctx.h, ctx.w, n,
                               rt.profile.seq_pad_multiple)
    except ValueError as exc:
        # A stream that does not fit the pad policy (a Ming-Image-style prompt
        # mask, a non-Z-Image backend): refuse the pooling rather than guess, and
        # say so once.
        _warn_once("zimage-layout", "AReA: %s", exc)
        return None
    rt.joint = layout
    # The joint stream's two groups are already in, so anything the model ropes
    # from here on is the seam's own coarse builder, never AReA's business.
    # Closing the window here too — not only in the refiner hook — keeps the
    # arming independent of that hook's firing.
    if recorder is not None:
        recorder.armed = False
    return layout


def _zimage_coarse_ropes(rt: _AreaRuntime, layout, f: int, device, kw: dict):
    """The two Phase I rope tables, cached per (canvas, factor, rope options).

    ``coarse_ropes_zimage`` rebuilds them through the model's OWN
    ``rope_embedder`` with lumina's ids, so a chained positional-scaling embedder
    composes and the model's own rope options (``scale_x/y``, ``shift_x/y`` —
    ``pos_ids_x``, ``comfy/ldm/lumina/model.py:421-437``) are honoured instead of
    assumed away.
    """
    embedder = getattr(rt.dm, "rope_embedder", None)
    _warn_dynamic_embedder(embedder)
    opts = kw.get("transformer_options")
    rope_options = opts.get("rope_options") if isinstance(opts, dict) else None
    key = ("zcoarse", layout, f, tuple(sorted((rope_options or {}).items())))
    cached = rt.ropes.get(key)
    if cached is not None:
        return cached
    try:
        ropes = coarse_ropes_zimage(embedder, layout, f, device, rope_options)
    except ValueError as exc:
        _warn_once("zimage-ropes", "AReA: %s This forward runs plain attention.", exc)
        return None
    rt.ropes[key] = ropes
    return ropes


def _zimage_attention(rt: _AreaRuntime, ctx: AreaContext, layout, q, k, v, heads, kw):
    """The AReA computation of one Z-Image attention call, or ``None`` to delegate.

    The seam's q/k arrive ROTATED — ``JointAttention`` applies rope itself before
    the call (``comfy/ldm/lumina/model.py:159-182``) — so both phases start in the
    un-rotated frame:

    * **Phase I** pools *unrotated* keys (pooling rotated keys would mix keys
      rotated by different angles inside one block, ``src/area_attn.py:180-184``)
      and rides the coarse tables, whose q side covers the whole padded stream in
      stream order and whose kv side the engine's key order.  The caption block
      and the image pad ride through unchanged (``pass_through``, plan Q6).
    * **Phase II** attends at the model's own full-resolution positions, which IS
      the table the recorder captured — so the rotated pair passes through
      untouched (a second rotation would be a float round trip of the same table)
      and only the bridge's keys are un-rotated (``bridge.py:10-24``).  The
      answers go back in stream order with zeros in the pad spans Phase II
      discards (``stitch_fine_zeros``).
    """
    table = rt.recorder.full_table()
    if table.shape[2] != layout.seq_len:
        _warn_once(
            "zimage-table",
            "AReA: the recorded Z-Image rope table covers %d tokens but the "
            "stream is %d, so the seam's rotation cannot be undone. This call "
            "runs plain attention.",
            table.shape[2], layout.seq_len,
        )
        return None
    block = rt.block
    rt.block += 1
    if ctx.phase == "coarse":
        k_unrot = unrotate1(k, table)
        ropes = _zimage_coarse_ropes(rt, layout, ctx.size, q.device, kw)
        if ropes is None:
            return None
        rope_q, rope_kv = ropes
        k_pooled, v_pooled = pool_keys_values(
            layout.engine_stream(k_unrot), layout.engine_stream(v),
            layout.pass_through_len, ctx.h, ctx.w, ctx.size, rt.config.pooling,
            seed=rt.config.seed, step=ctx.step, block=block,
        )
        out = F.scaled_dot_product_attention(
            apply_rope1(unrotate1(q, table), rope_q),
            apply_rope1(k_pooled, rope_kv), v_pooled,
        )
    else:
        first, last = layout.img_start, layout.img_end
        fine = _area_fine_layout(rt, ctx, unrotate1(k, table)[:, :, first:last],
                                 hook_name="diffusion_model.noise_refiner")
        if fine is None:
            return None
        out_txt, out_img = windowed_attention(
            q[:, :, :ctx.txt_len], k[:, :, :ctx.txt_len], v[:, :, :ctx.txt_len],
            q[:, :, first:last], k[:, :, first:last], v[:, :, first:last],
            unrotate1(k, table)[:, :, first:last], fine,
        )
        out = stitch_fine_zeros(out_txt, out_img, layout)
    # comfy's own flatten: heads then head-dim, sequence back on axis 1
    # (ldm/modules/attention.py:913-917).
    return out.permute(0, 2, 1, 3).reshape(q.shape[0], q.shape[2], heads * q.shape[3])


def _zimage_refiner_handles(refiner) -> tuple:
    """Forward-hook handles covering every ``noise_refiner`` layer.

    A plain Z-Image forward runs the refiner as ``for layer in
    self.noise_refiner: x = layer(x, ...)`` (``comfy/ldm/lumina/model.py:816``),
    so a hook registered on the ModuleList itself never fires — only the layers'
    hooks do.  A refiner that is a plain module (not a container) is its own
    target, and a container with no hookable children yields nothing so the
    caller can warn instead of staying silent.

    The FIRST hookable layer is tagged ``_area_refiner_entry``: the Phase II
    detail map's input is that layer's INPUT — the image embedding that enters
    refinement, the faithful Z-Image analogue of the reference's
    ``x_embedder`` (``backbones/flux.py:111-114,150-163``) — so the hook
    captures from the tagged module and only that one.
    """
    if refiner is None:
        return ()
    layers = [
        layer for layer in (list(refiner.children()) or [refiner])
        if hasattr(layer, "register_forward_hook")
    ]
    if layers:
        # The same module object is the first layer every forward calls, so the
        # tag identifies it per forward — no ordering re-discovery inside the
        # hook, no mutable global counter.
        layers[0]._area_refiner_entry = True
    return tuple(layer.register_forward_hook(_zimage_refiner_hook) for layer in layers)


def _zimage_refiner_hook(module, args, output):
    """Close the rope window, and capture the Phase II detail-map input.

    The arming window it closes is the plan's: the tagged (first) layer fires
    after ``embed_all``'s two rope groups and before the joint layer loop
    (``comfy/ldm/lumina/model.py:815-822,880-881``), and every later layer
    closes it again harmlessly.
    The detail map's input is the image embedding that ENTERS the FIRST refiner
    layer (``args[0]`` of the module tagged ``_area_refiner_entry``), sliced to
    the real ``h*w`` so no pad token reaches the detail map
    (``src/area_attn.py:361``): the faithful Z-Image analogue of the reference's
    ``x_embedder`` — layer-0 features, before any transformer block
    (``backbones/flux.py:111-114,150-163``), not any layer's transformed output.
    Later refiner layers keep closing the arming window but never re-capture.
    """
    rt = _ZRUNTIME
    if rt is None or rt.ctx is None:
        return
    if rt.recorder is not None:
        # Close the rope capture's arming window: the joint stream's two groups
        # are in, so anything the model ropes later is none of AReA's business.
        rt.recorder.armed = False
    if (getattr(module, "_area_refiner_entry", False) and rt.ctx.phase == "fine"
            and args and torch.is_tensor(args[0])):
        rt.hidden = args[0][:, : rt.ctx.h * rt.ctx.w]


def _zimage_install_seam(m, dm):
    """Patch ``comfy.ldm.lumina.model.optimized_attention_masked`` (ref-counted).

    Returns the rope capture :func:`src.area_zimage.capture_tables` built for
    this install (or the live one on a later consumer), so the new runtime feeds
    the same recorder the registered hook writes to — the FLUX path returns
    ``None``.

    The symbol is process-global and shared by every patcher, so the install is a
    module-global rebind plus a ref-count — the proven in-pack pattern
    (``src/area.py``'s FLUX seam and ``src/spa.py:1341-1372``): comfy's
    ``add_object_patch`` is a per-patch record re-applied around model use
    (``comfy/model_patcher.py:739``), which would double-wrap a shared symbol on
    every clone/re-apply.  A second AReA apply re-uses the wrapper instead of
    wrapping our own wrapper, and the symbol is restored on the last release.
    """
    import importlib

    if _ZSEAM["orig"] is None:
        try:
            mod = importlib.import_module(ZIMAGE_SEAM_MODULE)
        except Exception as exc:  # probe: backend module not importable here
            raise ValueError(
                f"AReA: cannot import {ZIMAGE_SEAM_MODULE} to patch the "
                f"Z-Image attention seam ({exc}). AReA supports FLUX and Z-Image "
                "(Lumina2); load an unmodified Z-Image checkpoint on a current "
                "ComfyUI."
            ) from exc
        bound = getattr(mod, ZIMAGE_SEAM_SYMBOL, None)
        if bound is None:
            raise ValueError(
                f"AReA: {ZIMAGE_SEAM_MODULE}.{ZIMAGE_SEAM_SYMBOL} is "
                "missing in this ComfyUI version, so the Z-Image attention seam "
                "cannot be installed. Update ComfyUI (or load an unmodified "
                "Z-Image model)."
            )
        _zimage_check_foreign(bound, "install")
        orig = _zimage_pristine(bound)
        setattr(mod, ZIMAGE_SEAM_SYMBOL, _make_zimage_wrapper(orig))
        _ZSEAM.update(module=mod, attr=ZIMAGE_SEAM_SYMBOL, orig=orig)
        handles = []
        recorder = None
        rope = getattr(dm, "rope_embedder", None)
        if rope is not None and hasattr(rope, "register_forward_hook"):
            hook, recorder = capture_tables(rope)
            handles.append(hook.handle)
        else:
            _warn_once(
                "no-zimage-rope",
                "AReA: this Z-Image model has no diffusion_model.rope_embedder "
                "to hook, so the seam's rope table cannot be captured. Every "
                "forward runs plain attention.",
            )
        _ZSEAM["recorder"] = recorder
        # lumina ITERATES the refiner ModuleList
        # (``comfy/ldm/lumina/model.py:816``) instead of calling it, so a forward
        # hook on the container never fires — register on every layer.  A plain
        # module (a refiner that is not a container) is its own target.
        refiner_hooks = _zimage_refiner_handles(getattr(dm, "noise_refiner", None))
        if refiner_hooks:
            handles.extend(refiner_hooks)
        else:
            _warn_once(
                "no-zimage-refiner",
                "AReA: this Z-Image model has no diffusion_model.noise_refiner "
                "to hook, so the Phase II detail map falls back to the projected "
                "image keys of the first fine call.",
            )
        _ZSEAM["hook"] = tuple(handles)
    _ZSEAM["consumers"] += 1
    m._area_seam_consumer = True
    m._area_seam = "zimage"
    # The capture is process-wide for this install (its hook is registered once,
    # on the first consumer's embedder), so a later consumer re-uses it rather
    # than building a second recorder the registered hook never feeds.
    return _ZSEAM["recorder"]


def _area_install_seam(m, dm, profile):
    """Patch the backbone's bound attention seam and hook the model (ref-counted).

    FLUX and Z-Image are different symbols in different modules, so each takes its
    own install path and its own ref-count — the two never share one.  Returns
    the Z-Image rope capture (``None`` on the FLUX side), which the caller hands
    to the runtime it builds.
    """
    if profile is ZIMAGE_PROFILE:
        return _zimage_install_seam(m, dm)
    import importlib

    if _SEAM["orig"] is None:
        try:
            mod = importlib.import_module(FLUX_ATTENTION_MODULE)
        except Exception as exc:  # probe: backend module not importable here
            raise ValueError(
                f"AReA: cannot import {FLUX_ATTENTION_MODULE} to patch the FLUX "
                f"attention seam ({exc}). AReA supports FLUX and Z-Image; use an "
                "unmodified FLUX/Qwen loader (no fused attention kernels)."
            ) from exc
        orig = getattr(mod, FLUX_ATTENTION_SYMBOL, None)
        if orig is None or getattr(orig, "_area_wrapper", False):
            raise ValueError(
                f"AReA: {FLUX_ATTENTION_MODULE}.{FLUX_ATTENTION_SYMBOL} is missing "
                "in this ComfyUI version, so the attention seam cannot be "
                "installed. Update ComfyUI (or load an unmodified FLUX model)."
            )
        setattr(mod, FLUX_ATTENTION_SYMBOL, _make_area_wrapper(orig))
        _SEAM.update(module=mod, attr=FLUX_ATTENTION_SYMBOL, orig=orig)
        img_in = getattr(dm, "img_in", None)
        if img_in is not None and hasattr(img_in, "register_forward_hook"):
            _SEAM["hook"] = (img_in.register_forward_hook(_area_img_in_hook),)
        else:
            _warn_once(
                "no-img-in",
                "AReA: diffusion_model.img_in could not be hooked, so the Phase "
                "II detail map is derived from the projected image keys of the "
                "first fine call. Results differ slightly from a hooked model.",
            )
    _SEAM["consumers"] += 1
    m._area_seam_consumer = True
    m._area_seam = "flux"
    return None


def _area_img_in_hook(module, args, output):
    """Capture the ``img_in`` output of the running forward (the reference's
    ``x_embedder`` hook, ``backbones/flux.py:150-163``)."""
    rt = _AREA_RUNTIME
    if rt is None or rt.ctx is None or rt.ctx.phase != "fine":
        return
    rt.hidden = output


def _warn_regime(total: int, phase1: int, phase: Optional[str]) -> None:
    """Warn once when Phase I would flood a short (distilled) schedule.

    The node default (``phase1_steps=8``) is the reference's FLUX constant *of 28*.
    On a short distilled schedule — Z-Image Turbo's 8-step samplers — it makes
    EVERY step Phase I: pooled K/V everywhere, no fine windowed steps to restore
    local detail, and the distilled model hallucinates around the loss (measured
    2026-10-17: 8/8 coarse at 2048 px, periodic ripple/mosaic + swirls; see
    ``docs/2026-10-17-area-zimage-ripple-diagnosis.md``).  Advisory only, no
    behavior change: a 28-step FLUX schedule never trips it (the shared
    :func:`~src.area_schedule.is_short_schedule` gate returns early), and a user
    who wants the flood keeps it by setting ``phase1_steps`` explicitly — the
    same discipline as I-Max's z-image schedule advisory (``src/imax.py:311-349``).
    """
    if phase != "coarse" or not is_short_schedule(total):
        return
    if phase1 * 3 <= total:  # up to a third of the steps coarse: the paper's share
        return
    _warn_once(
        "area-regime",
        "AReA: %d of %d denoising steps run Phase I (coarse pooled attention) on "
        "a short schedule — a distilled Turbo sampler cannot re-detail after "
        "pooled keys, which shows as periodic ripple/mosaic artifacts. Set "
        "phase1_steps to 1-3, or phase1_mode to 'sigma', to give this run a fine "
        "(windowed) majority. (8 of 28 is the FLUX paper constant, not a Z-Image "
        "one.)",
        phase1, total,
    )


def _area_resolve_forward(rt: _AreaRuntime, params: dict) -> AreaContext:
    """Resolve one forward's step, geometry, ropes and phase into a context."""
    ctx = AreaContext()
    rt.ctx = ctx
    rt.block = 0
    rt.hidden = None
    if rt.profile is ZIMAGE_PROFILE:
        # Per-forward Z-Image state (the SPA discipline): the rope groups are
        # emptied, the joint layout is unresolved, and the recording window opens
        # — ``embed_cap`` / ``embed_all`` are about to call ``rope_embedder``.
        if rt.recorder is None:
            rt.recorder = _ZSEAM["recorder"] or ZImageRopeRecorder()
        rt.recorder.reset()
        rt.recorder.armed = True
        rt.joint = None
    x = params.get("input")
    conds = params.get("c") or {}
    if not torch.is_tensor(x) or x.ndim != 4:
        return ctx  # nothing pool-able at this shape -> plain attention
    patch_size = int(getattr(rt.dm, "patch_size", FLUX_PATCH_SIZE) or FLUX_PATCH_SIZE)
    h, w = _grid_from_latent(x, patch_size)
    cross = conds.get("c_crossattn")
    if torch.is_tensor(cross) and cross.ndim >= 2:
        ctx.txt_len = int(cross.shape[1])
    opts = conds.get("transformer_options") or {}
    step, total = _area_locate(opts.get("sample_sigmas"), opts.get("sigmas"))
    if step is None or max(h, w) <= rt.config.native:
        return ctx  # at/below the native extent AReA is a provable no-op
    phase1 = _phase1_count(rt, opts)
    cfg = rt.config
    schedule = _area_schedule(rt, h, w, total, phase1, ctx.txt_len)
    phase, size = schedule.at(step)
    ctx.phase, ctx.size, ctx.step, ctx.h, ctx.w = phase, size, step, h, w
    # The Phase II ordinal (how many fine steps ran before this one) and how many
    # of its leading steps the bridge floor exempts from the cost cap.  The floor
    # is keyed on the RESOLVED schedule's geometry, so steps / sigma / bootstrap
    # all behave identically: a resolution that spends more steps coarse simply
    # reaches fewer of the floored fine steps, and a floored step is floored
    # whichever mode resolved the schedule.
    ctx.fine_step = step - len(schedule.factors)
    ctx.floor_steps = (cfg.bridge_floor_steps
                       if phase == "fine" and is_short_schedule(total) else 0)
    _warn_regime(total, phase1, phase)
    if step == 0 and rt.config.phase1_mode == "bootstrap" and phase == "fine":
        # Appendix F: Phase II must ENTER at the re-noise level the native
        # pre-pass handed over (sigma_star), so check the sampler's own table
        # against phase1_sigma and say so once when the graph re-uses a table
        # that enters somewhere else.
        table = opts.get("sample_sigmas")
        if torch.is_tensor(table) and table.numel():
            first = float(table.detach().float().flatten().cpu()[0])
            if abs(first - rt.config.phase1_sigma) > 0.05:
                _warn_once(
                    "zimage-bootstrap-entry",
                    "AReA: bootstrap mode runs Phase II from the first step, "
                    "but this sampler enters at sigma %.2f while phase1_sigma "
                    "is %.2f — the native-resolution pre-pass must hand the "
                    "latent over re-noised to sigma_star (plan 2026-10-18-area-"
                    "zimage-bootstrap, step B2), or this run starts windowed "
                    "attention on a canvas composed for a different noise "
                    "level.",
                    first, rt.config.phase1_sigma,
                )
    if phase == "coarse" and rt.profile is not ZIMAGE_PROFILE:
        # FLUX resolves its coarse tables here; the Z-Image ones need the model's
        # own rope groups, which only exist once the forward has run its embed,
        # so the seam resolves them lazily (``_zimage_coarse_ropes``).
        try:
            ctx.data["rope_q"], ctx.data["rope_kv"] = _area_coarse_ropes(
                rt, h, w, size, ctx.txt_len, x.device
            )
        except ValueError as exc:
            _warn_once("no-ropes", "AReA: %s This forward runs plain attention.", exc)
            ctx.phase = None
    return ctx


def _make_area_unet_wrapper(rt: _AreaRuntime, previous_wrapper):
    """The per-forward wrapper: resolve AReA's state, then call through.

    ``previous_wrapper`` (a chained DyPE/SEGA/HAP/SPA wrapper found on the cloned
    ``model_options``) is called THROUGH — replacing it outright would drop its
    per-forward state updates.
    """

    def _call_through(model_function: Callable, params: dict):
        if previous_wrapper is not None:
            return previous_wrapper(model_function, params)
        return model_function(params.get("input"), params.get("timestep"), **params.get("c", {}))

    def wrapper(model_function: Callable, params: dict):
        # NOTE: the ``global`` belongs HERE, not in the factory — a ``global`` in
        # the enclosing function does not reach a nested scope, so the assignment
        # would otherwise bind a local of ``wrapper`` and the seam would never
        # see the runtime (the whole patch would silently delegate).
        global _AREA_RUNTIME, _ZRUNTIME

        # One runtime per forward, and only the backbone this patcher resolved:
        # the FLUX and Z-Image seams read different globals, so neither can fire
        # on the other's forward.
        _AREA_RUNTIME = None if rt.profile is ZIMAGE_PROFILE else rt
        _ZRUNTIME = rt if rt.profile is ZIMAGE_PROFILE else None
        try:
            _area_resolve_forward(rt, params)
            return _call_through(model_function, params)
        finally:
            _AREA_RUNTIME = None
            _ZRUNTIME = None

    return wrapper


def apply_area_to_model(
    model,
    model_type: str,
    width: int,
    height: int,
    enable_area: bool = True,
    phase1_steps: int = PHASE1_STEPS,
    pooling: str = "subsample",
    halo: int = 8,
    bridge_ratio: float = 0.5,
    bridge_floor_steps: int = 2,
    window_start: int = 0,
    window_hold_steps: int = 0,
    seed: int = 0,
    phase1_mode: str = "steps",
    phase1_sigma: float = ZIMAGE_SIGMA_STAR,
):
    """Patch a ComfyUI FLUX or Z-Image model with AReA (arXiv 2610.17227).

    The backbone is resolved once into an :class:`~src.area_arch._ArchProfile`
    (``src/area_arch.py``) and exactly one attention seam is installed for it —
    FLUX's bound ``attention`` or Z-Image's bound
    ``optimized_attention_masked``.  Every other architecture raises an actionable
    :class:`ValueError` (house rule, ``src/hap.py:876-895``), and Nunchaku (fused
    kernels bypass the seam) warns and returns the clone UNCHANGED
    (``src/hap.py:897-903``).  At or below the native 64-patch extent (1024 px)
    nothing is patched — attention would be replaced by itself.

    ``phase1_mode`` / ``phase1_sigma`` are the Phase I -> II switch (Q4):
    ``"steps"`` (the default) is the reference's fixed ``phase1_steps``, and
    ``"sigma"`` counts the leading steps whose noise level is still at or above
    ``phase1_sigma`` (``theory.md:240-246``) so the switch follows the schedule.
    ``phase1_steps`` is clamped to ``[0, total]`` and ``0`` means "all Phase II"
    (the reference's ``config.phase1_steps``, ported without its
    ``phase1_steps=0`` assert).  ``pooling`` is one of :data:`POOLING`
    (``subsample`` / ``mean`` / ``medoid`` / ``random``), ``halo`` the widest
    Halo ring, ``bridge_ratio`` rho of ``R = round(rho * w)^2`` bridge tokens per
    window, and ``seed`` the key of every draw (windows, halos, random pooling).

    ``bridge_floor_steps`` (A1) is the short-schedule bridge floor: the first
    that many fine steps keep the paper's full bridge and halo instead of the
    cost cap's trimmed ones, on a short schedule only (a 28-step FLUX run is
    never floored), with a one-time warning naming the cost multiplier it pays.
    ``window_start`` / ``window_hold_steps`` (A4) re-shape the Phase II window
    ramp on short schedules only — a start side instead of Eq. 8's, and extra
    steps holding it — and are ignored on a long one.

    Determinism: no global RNG is read — every stream is keyed by
    ``(seed, step)`` in :mod:`src.area_schedule` / :mod:`src.area_attn`.
    """
    from .spa import _hrdit_carry_state

    width = _snap_to_multiple(width, PATCH_PX)
    height = _snap_to_multiple(height, PATCH_PX)

    # ``ModelPatcher.clone()`` drops custom attrs: carry the AReA config (and
    # SPA/HAP's, so a chained DyPE/SEGA/HAP node never loses its state) onto the
    # clone right away, before any early return below.
    m = model.clone()
    _area_carry_state(model, m)
    _hrdit_carry_state(model, m)

    dm = m.model.diffusion_model
    detected_type = resolve_model_type(dm, model_type)
    logger.info("AReA: Detected model type: %s", detected_type)
    if detected_type == "nunchaku":
        # Fused/quantized attention kernels bypass the seam entirely.
        _warn_once(
            "nunchaku",
            "AReA: unsupported on Nunchaku (quantized/fused attention kernels "
            "bypass the attention seam); returning the model UNCHANGED. "
            "Use a non-quantized FLUX / Z-Image (Lumina2) loader.",
        )
        m._area_config = None
        return m
    if detected_type not in ("flux", "zimage"):
        raise ValueError(
            f"AReA supports FLUX and Z-Image (Lumina2 / NextDiT) — it covered "
            f"FLUX only in v1 — but the resolved model type is "
            f"{detected_type!r}. Load an unmodified (non-quantized) FLUX model, "
            "or use SPA / DyPE / SEGA for this architecture."
        )
    # The arch profile validates the pin against the model's own backbone and
    # refuses the Lumina2 siblings AReA cannot patch (pixel space, Ming-Image).
    profile = resolve_area_arch(dm, model_type)

    m._area_config = AreaConfig(
        phase1_steps=max(0, int(phase1_steps)),
        phase1_mode=phase1_mode if phase1_mode in ("steps", "sigma", "bootstrap") else "steps",
        phase1_sigma=float(phase1_sigma),
        pooling=pooling if pooling in POOLING else "subsample",
        halo=max(0, int(halo)),
        bridge_ratio=float(bridge_ratio),
        # A1/A4: clamped to the node's widget ranges (the node clamps the same
        # way, so a graph and a script agree on what is stored).
        bridge_floor_steps=max(0, min(16, int(bridge_floor_steps))),
        window_start=max(0, min(1024, int(window_start))),
        window_hold_steps=max(0, min(8, int(window_hold_steps))),
        seed=int(seed),
    )

    if not enable_area:
        return m

    if getattr(model, "_spa_installed", None) or getattr(m, "_hrdit_consumers", None):
        _warn_once(
            "spa-hap",
            "AReA: an SPA/HAP hook is installed on this model. AReA replaces "
            "attention outright on the steps it runs, so SPA's averaged passes "
            "and HAP's pruned heads are bypassed there; plain-attention steps "
            "(and the native extent) still delegate to them.",
        )

    h, w = height // PATCH_PX, width // PATCH_PX
    if max(h, w) <= NATIVE_PATCHES:
        _warn_once(
            "native",
            "AReA: %dx%d px builds a %dx%d patch grid, at or below the native "
            "64-patch (1024 px) extent where AReA is a provable no-op — nothing "
            "is patched. Raise the resolution past 1024 px on the longer side "
            "(and keep this node's width/height in step with your latent) for "
            "AReA to engage.",
            width, height, w, h,
        )
        return m

    # GEOMETRY PRE-CHECK (schedule.py:39-48): the pooling ramp needs a common
    # divisor >= 2 of the patch grid to tile it exactly, and a grid with none
    # would otherwise fail three steps into the sampling run.  Fail here, at
    # apply time, with the reference's actionable message.
    initial_factor(h, w, NATIVE_PATCHES)

    _area_install_seam(m, dm, profile)

    previous_wrapper = None
    try:
        previous_wrapper = m.model_options.get("model_function_wrapper")
    except AttributeError:  # probe: mock patchers without model_options
        previous_wrapper = getattr(m, "_unet_wrapper", None)
    m.set_model_unet_function_wrapper(
        _make_area_unet_wrapper(_AreaRuntime(m._area_config, dm, profile), previous_wrapper)
    )
    logger.info(
        "AReA: installed on a %dx%d patch grid (%dx%d px) on the %s seam, "
        "phase1_steps=%d (mode %s), pooling=%s, halo=%d, bridge_ratio=%s, seed=%d",
        w, h, width, height, "Z-Image" if profile is ZIMAGE_PROFILE else "FLUX",
        m._area_config.phase1_steps, m._area_config.phase1_mode,
        m._area_config.pooling, m._area_config.halo, m._area_config.bridge_ratio,
        m._area_config.seed,
    )
    return m


def _release_seam(seam) -> None:
    """Drop one consumer of a shared seam; restore the symbol on the last one.

    Every handle the install registered is removed, and the bound symbol goes
    back to the pristine function it was rebound from — `None` only when nothing
    was ever installed (an install that raised before binding).
    """
    seam["consumers"] = max(0, seam["consumers"] - 1)
    if seam["consumers"] > 0:
        return
    for handle in seam["hook"] or ():
        try:
            handle.remove()
        except Exception:  # probe: handle already gone with its module
            pass
    if seam["orig"] is not None:
        mod, attr, orig = seam["module"], seam["attr"], seam["orig"]
        bound = getattr(mod, attr, None)
        if bound is not orig and not getattr(bound, "_area_wrapper", False):
            # neither our own wrapper nor the pristine symbol: something else
            # rebound it while we were installed.
            _warn_once(
                "seam-rebound",
                "AReA: %s.%s was rebound by something else since AReA installed "
                "its hook; restoring the pristine symbol anyway.",
                getattr(mod, "__name__", mod), attr,
            )
        if bound is not orig:
            setattr(mod, attr, orig)
    reset = {"module": None, "attr": None, "orig": None, "hook": None}
    if "recorder" in seam:
        # the Z-Image seam's rope capture; the FLUX twin has none to clear.
        reset["recorder"] = None
    seam.update(**reset)


def restore_area_attention_hook(m) -> None:
    """Undo an AReA patch recorded on ``m`` (idempotent, consumer-aware).

    The seam symbol and the forward hooks are shared by every patcher, so they
    are restored only when the last consumer leaves — mirroring
    ``src/spa.py:1485``.  FLUX and Z-Image have separate ref-counts, so the
    patcher's own seam (``m._area_seam``) is the one released.
    """
    if not getattr(m, "_area_seam_consumer", False):
        m._area_config = None
        return
    m._area_seam_consumer = False
    _release_seam(_ZSEAM if getattr(m, "_area_seam", None) == "zimage" else _SEAM)
    m._area_seam = None
    m._area_config = None
