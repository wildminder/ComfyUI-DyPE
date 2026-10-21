"""AReA — training-free, time-dependent replacement of FLUX image self-attention.

This module is the model-side wiring (the reference's ``area/backbones/flux.py``
and ``area/backbones/base.py``): it patches a ComfyUI FLUX model so that every
block's attention runs the two AReA phases built from :mod:`src.area_schedule`
and :mod:`src.area_attn` instead of dense attention.  The node
(``nodes/area.py``) is a thin schema/execute shell over
:func:`apply_area_to_model`.

**Four seams, one engine.**  FLUX, Z-Image (Lumina2), Krea-2 and
Qwen-Image-2.1 are different backbones with different seams, so
:func:`apply_area_to_model` resolves an
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
* Krea-2 — the bound ``optimized_attention_masked`` of
  ``comfy.ldm.krea2.model`` (``comfy/ldm/krea2/model.py:21`` binds the name, ``:99-100``
  calls it), a single-stream MMDiT whose joint stream is ``[text | image]`` with NO
  padding (``model.py:355``), whose 48-head main blocks are told apart from the four
  20-head text-fusion calls by the head count and the published ``img_slice``
  (``model.py:369``), and whose latent is **5-D** in the normal image case (the Wan21
  latent format has three dimensions — ``comfy/latent_formats.py:731`` — so comfy
  unsqueezes the sampler's 4-D latent, ``comfy/sample.py:58-59``).  The krea-2 adapter
  (:mod:`src.area_krea`) owns the rope capture, the joint layout and the eligibility
  guard; this module owns the ref-counted install, the per-forward state, the
  forward-time resolution and the two phases themselves (``_krea_attention``) — the
  model rotates q/k itself, so both phases start from the UN-ROTATED frame, and the
  answer is the flattened ``[b, n, heads*d]`` the model's gate multiplies.  SPA patches
  this very symbol for Krea-2 too
  (``src/spa.py:583-589``), so the install refuses to wrap a foreign hook — and a
  quantized/fused attention kernel bypasses the seam altogether, so the install proves
  the model's own attention class still resolves the symbol and leaves the model
  UNCHANGED when it does not.
* Qwen-Image-2.1 — the bound ``optimized_attention`` of
  ``comfy.ldm.qwen_image21.model`` (``comfy/ldm/qwen_image21/model.py:15`` binds
  the name, ``:211`` calls it), a single-stream MMDiT with a BLOCK-CAUSAL
  stream: the caption is a segment of its own and the image another, and the
  image segment's queries are the ``h*w`` image rows while its keys are the
  JOINT set — the caption AND the image (``model.py:206-208``), so the
  eligible call has NO text queries at all.  The seam's input is 3-D
  (``[b, n, heads*dim]``, read head-major) and the ``mask`` KEYWORD's
  presence separates the plain path from the length-indistinguishable
  prefix-cache path (``model.py:222``), so the wrapper's ``mask`` default is
  the ``QWEN21_NO_MASK`` sentinel.  The qwen-2.1 adapter
  (:mod:`src.area_qwen21`) owns the geometry, the ids and the rope capture;
  this module owns the ref-counted install, the per-forward state and the
  Phase I computation (``_qwen21_attention``).

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
    caption_window_ids,
    detail_map,
    fit_extra_keys,
    pool_keys_values,
    tiling_rng,
    windowed_attention,
    windowed_step_cost,
)
from .area_arch import (
    FLUX_PROFILE,
    KREA_PROFILE,
    QWEN21_PROFILE,
    ZIMAGE_PROFILE,
    _ArchProfile,
    resolve_area_arch,
)
from .area_krea import (
    KREA_PATCH,
    KREA_SEAM_MODULE,
    KREA_SEAM_SYMBOL,
    KreaRopeRecorder,
    capture_krea_table,
    coarse_ropes_krea,
    krea_latent_grid,
    krea_layout,
    krea_squeeze_latent,
    krea_stream_eligible,
)
from .area_qwen21 import (
    QWEN21_HEADS,
    QWEN21_NO_MASK,
    QWEN21_SEAM_MODULE,
    QWEN21_SEAM_SYMBOL,
    Qwen21RopeRecorder,
    capture_qwen21_table,
    coarse_ropes_qwen21,
    qwen21_detail_map,
    qwen21_head_geometry,
    qwen21_layout,
    qwen21_latent_grid,
    qwen21_peek,
    qwen21_segment_eligible,
    qwen21_slice_table,
    qwen21_to_heads,
)
from .area_rope import flux_style_ids
from .area_schedule import (
    AreaSchedule,
    build_schedule,
    coarse_grid,
    is_short_schedule,
    resolve_factor,
)
from .area_zimage import (
    ZIMAGE_GROUP_ORDER,
    ZIMAGE_SEAM_MODULE,
    ZIMAGE_SEAM_SYMBOL,
    ZIMAGE_SIGMA_STAR,
    ZImageRopeRecorder,
    _head_shared,
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


def _warn_grid_adjusted(h: int, w: int, width_px: int, height_px: int, factor: int) -> None:
    """Tell the user once that the patch grid could not be tiled exactly.

    The pooling ramp needs a common divisor >= 2 of the patch grid to tile it
    exactly; a grid with none (125x32, from 512x2000 px) is *adjusted* instead
    of refused: Phase I pools ``f x f`` blocks whose last block on a partial
    axis is a partial block, and the canvas, the latent and the output
    resolution are untouched.  The key is the grid, so every resolution is
    announced once and a grid that already tiles is never announced.
    """
    nh, nw = coarse_grid(h, w, factor)
    _warn_once(
        f"area-grid-factor:{h}x{w}",
        "AReA: the %dx%d patch grid (%dx%d px) has no common divisor >= 2, so the "
        "pooling ramp cannot tile it exactly. Phase I pools %dx%d blocks with a "
        "partial edge block (pooled grid %dx%d) instead of refusing to run. The "
        "canvas, the latent and the output resolution are unchanged (%dx%d px).",
        h, w, width_px, height_px, factor, factor, nh, nw, width_px, height_px,
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
    #: Delay — leading Phase I steps run at factor 1 (no pooling). Measured on
    #: the GPU to cut final 32px energy 2.3-4.4x, but the user's real-checkpoint
    #: check found it WORSE visually (more hallucination, worse quality): the
    #: mid-composition pooled steps after full-attention steps are the harmful
    #: mix. Default 0 (today's pooling-from-step-0) until that is understood.
    delay: int = 0
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

    Krea-2 reuses the same two fields with its own types: ``recorder`` is a
    :class:`~src.area_krea.KreaRopeRecorder` (the model's ONE rope table per
    forward, captured through a forward hook on its own ``pe_embedder``) and
    ``joint`` is the resolved joint layout of the current forward.
    """

    __slots__ = ("config", "dm", "profile", "ctx", "hidden", "ropes", "schedules",
                 "block", "recorder", "joint", "img_in_calls", "announced")

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
        #: How many times the backbone's image embedder fired in this forward.
        #: Qwen-Image-2.1's ``img_in`` is called ONCE PER IMAGE (the target
        #: last, ``comfy/ldm/qwen_image21/model.py:319``), so a count above one
        #: means the forward carries reference latents (D6).
        self.img_in_calls = 0
        #: The schedules whose EFFECTIVE coarse-step count was already
        #: announced — the log line is emitted ONCE per schedule, not once per
        #: step (:func:`_announce_effective_coarse`).
        self.announced = set()


#: The runtime of the FLUX forward currently running, read by the FLUX attention
#: seam.  Set and cleared around every forward (SPA discipline) — ``None`` between
#: forwards, which is what makes an unpatched/inactive model a pure no-op.
_AREA_RUNTIME: Optional[_AreaRuntime] = None

#: The same for Z-Image: the two globals are set exclusively (the unet wrapper
#: assigns exactly one per forward), so neither backbone's seam can fire on the
#: other's forward and the two never share a runtime.
_ZRUNTIME: Optional[_AreaRuntime] = None

#: The same for Krea-2 — a THIRD seam, a third module and a third symbol, so
#: its runtime never reaches FLUX's or Z-Image's wrapper.  The unet wrapper
#: assigns exactly one of the three per forward.
_KRUNTIME: Optional[_AreaRuntime] = None

#: The same for Qwen-Image-2.1 — a FOURTH seam, a fourth module and a fourth
#: symbol (``comfy.ldm.qwen_image21.model``'s own bound ``optimized_attention``),
#: so its runtime never reaches FLUX's, Z-Image's or Krea-2's wrapper.  The
#: unet wrapper assigns exactly one of the four per forward.
_QRUNTIME: Optional[_AreaRuntime] = None

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

# The Krea-2 twin — a THIRD module and symbol (``comfy.ldm.krea2.model``'s own
# bound ``optimized_attention_masked``), so a THIRD ref-count: the three
# installs never share one.  ``hook`` holds every handle this install
# registered on the first consumer's diffusion model (the rope capture on
# ``pe_embedder`` and the image-embedder hook on ``first``), and ``recorder``
# is the capture :func:`src.area_krea.capture_krea_table` built for it.
_KSEAM = {"module": None, "attr": None, "orig": None, "hook": None, "recorder": None,
          "consumers": 0}

#: The Qwen-Image-2.1 twin — a FOURTH module and symbol
#: (``comfy.ldm.qwen_image21.model``'s own bound ``optimized_attention``), so a
#: FOURTH ref-count: the four installs never share one.  ``hook`` holds every
#: handle this install registered on the first consumer's diffusion model (the
#: rope capture on ``pe_embedder``), and ``recorder`` is the capture
#: :func:`src.area_qwen21.capture_qwen21_table` built for it.
_QSEAM = {"module": None, "attr": None, "orig": None, "hook": None, "recorder": None,
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


def _schedule_key(rt: _AreaRuntime, h: int, w: int, total: int, phase1: int,
                   txt_len: int) -> tuple:
    """The cache key of ONE schedule — the reference's key plus the A4 knobs.

    Shared by :func:`_area_schedule` (the cache) and
    :func:`_announce_effective_coarse` (the once-per-schedule log line), so
    the line is emitted exactly once per cached schedule and never twice for
    the same one.
    """
    cfg = rt.config
    return (h, w, int(total), max(0, int(phase1)), int(txt_len), cfg.window_start,
            cfg.window_hold_steps, cfg.delay)


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
    key = _schedule_key(rt, h, w, total, phase1, txt_len)
    schedule = rt.schedules.get(key)
    if schedule is None:
        cfg = rt.config
        schedule = build_schedule(h, w, total, max(0, int(phase1)), cfg.native,
                                  txt_len=txt_len,
                                  window_start=cfg.window_start,
                                  window_hold_steps=cfg.window_hold_steps,
                                  delay=cfg.delay)
        rt.schedules[key] = schedule
    return schedule


def _announce_effective_coarse(rt: _AreaRuntime, key: tuple, total: int,
                               cap: Optional[int], effective: int) -> None:
    """The EFFECTIVE coarse-step count of a schedule — ONCE per schedule (Q10).

    The user's knob and the schedule the run really follows are two different
    numbers whenever the heal cap bites (a short schedule keeps at least two
    fine steps, or the pooling footprint never heals), and a SILENTLY
    overridden knob is unreadable — the Krea-2 lesson (v).  The line names all
    of them: the knob, the schedule total, the cap, the effective coarse count
    and the effective fine count.

    The cap is reported as ``n/a`` on a long schedule, where the heal rule
    does not apply at all.  The line is emitted when the schedule is first
    resolved and never again for the same one (the key is the schedule's own
    cache key), so a 28-step run logs one line, not twenty-eight.
    """
    if key in rt.announced:
        return
    rt.announced.add(key)
    logger.info(
        "AReA: effective coarse count %d of %d steps (phase1_steps knob %d, "
        "heal cap %s, effective fine count %d)",
        int(effective), int(total), int(rt.config.phase1_steps),
        "n/a" if cap is None else str(int(cap)),
        max(0, int(total) - int(effective)),
    )


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


#: ``[1, txt_len + n, 3]`` rope position ids: zero text ids, image rows/cols.
#:
#: ``ldm/flux/model.py:192-194`` builds exactly this (text ids all zero, image
#: ids carrying row/col on axes 1/2 with the axis-0 index at 0), so a stock
#: ``pe_embedder`` returns the same table the model itself computes for the
#: canvas.
#:
#: The builder is shared with the Krea-2 adapter (whose ids have the same FLUX
#: shape) and lives in ``src/area_rope.py`` — this is a thin ALIAS of
#: :func:`src.area_rope.flux_style_ids`, not a copy, so the two can never drift
#: apart (plan §5.1/S3).
_area_rope_ids = flux_style_ids


def _area_coarse_ropes(rt: _AreaRuntime, h: int, w: int, f: int, txt_len: int,
                       device) -> tuple[torch.Tensor, torch.Tensor]:
    """The two Phase I rope tables (the reference's ``_coarse_ropes``).

    Queries keep full resolution and ride on their block's coarse position, the
    pooled keys sit on the pooled grid (``ceil(h/f) x ceil(w/f)`` — the ceil
    only differs from ``(h/f) x (w/f)`` on a grid the blocks do not tile, where
    the last block on a partial axis is a partial block), so both are measured
    in the same units.  Tables come from the *installed* embedder so a chained
    DyPE/SEGA embedder composes (the reference calls its own ``pos_embed``).
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
    nh, nw = coarse_grid(h, w, f)
    pos = torch.arange(h * w, device=device)
    coarse = torch.arange(nh * nw, device=device)
    rope_q = embedder(_area_rope_ids(txt_len, pos // w // f, pos % w // f, device, torch.float32))
    rope_kv = embedder(
        _area_rope_ids(txt_len, coarse // nw, coarse % nw, device, torch.float32)
    )
    rt.ropes[key] = (rope_q, rope_kv)
    return rope_q, rope_kv


def _area_fine_layout(rt: _AreaRuntime, ctx: AreaContext, k_img: torch.Tensor,
                      hook_name: Optional[str] = None,
                      energy=None) -> Optional[FineLayout]:
    """Build (once per step) the Phase II layout — the reference's ``_after_embedding``.

    ``energy`` is the detail map, when the caller has already built it (the
    Qwen-Image-2.1 path: the adapter owns 2.1's grid, so the map is built by
    :func:`~src.area_qwen21.qwen21_detail_map` and handed over).  Without one
    the map is derived here from the hooked hook's output — or, when that hook
    never fired, from the projected image keys (the fallback below).
    """
    layout = ctx.data.get("layout")
    if layout is not None:
        return layout
    cfg = rt.config
    h, w, t = ctx.h, ctx.w, ctx.txt_len
    hidden = rt.hidden
    if hook_name is None:
        # The detail map's source is the backbone's own image embedder: FLUX's
        # ``img_in``, Z-Image's refiner entry (passed explicitly), Krea-2's
        # ``first`` (``comfy/ldm/krea2/model.py:333``).
        hook_name = ("diffusion_model.first" if rt.profile is KREA_PROFILE
                     else "diffusion_model.img_in")
    if energy is None:
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
    # Window integrity on short (distilled) schedules (area-fix2 mechanism 3):
    # an 8-step model has not composed anything yet at sigma ~ 0.8, so the DWT
    # energy map fires on noise and fragments the canvas into 44x44 quadrants
    # that each hallucinate their own subject — keep the base windows whole
    # (no splits) and the boundaries still (no jitter) instead.
    short = is_short_schedule(ctx.total)
    windows = adaptive_tiling(
        energy, h, w, ctx.size, w_min, t,
        rng=None if short else tiling_rng(cfg.seed, ctx.step),
        allow_splits=(not short),
    )
    # The bridge's register count follows the window's ACTUAL SHAPE on the
    # backbones whose profile asks for it (Krea-2): the paper's ``(rho*w)^2``
    # is the count for a SQUARE window, and a window clipped by a canvas
    # narrower than its side holds fewer tokens, so the square formula would
    # over-provision the bridge and dilute the window's own content.  The
    # FLUX and Z-Image paths keep the square count, bit for bit.
    shape = (h, w) if getattr(rt.profile, "bridge_shape_aware", False) else (None, None)
    if ctx.fine_step < ctx.floor_steps:
        # A1 — the bridge floor (``docs/2026-10-18-area-zimage-adaptation-research.md``
        # §4/§5-A1): on a SHORT schedule the cost cap cuts the bridge to zero and
        # the Halo to a sliver on exactly the steps where composition is still
        # forming, which the paper's own ablation links to window duplication.  The
        # leading ``floor_steps`` fine steps therefore keep the paper's full
        # ``round(rho*w)^2`` bridge and the configured halo, exempt from the cap —
        # an explicit, warned cost decision, not a silent one.
        registers, halo = bridge_registers(ctx.size, cfg.bridge_ratio, *shape), cfg.halo
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
        registers = bridge_registers(ctx.size, cfg.bridge_ratio, *shape)
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


def _zimage_caption_rope(rt: _AreaRuntime, ctx: AreaContext, fine: FineLayout,
                          t: int, device):
    """The GTCN caption anchor table, built ONCE per step (cached on the context).

    The table depends only on the step's windows, the caption length and the
    model's rope embedder — all of them fixed for the step — so the step's
    attention calls (one per refiner layer) share it instead of rebuilding it
    per call.  The cache lives on the STEP's context, so a new step (new
    windows, new timestep) always rebuilds it: a dynamic positional embedder
    (DyPE / I-Max / SEGA) is therefore never served a stale table.
    """
    cached = ctx.data.get("caption_rope")
    if cached is not None:
        return cached
    embedder = getattr(rt.dm, "rope_embedder", None)
    if embedder is None:
        return None
    _warn_dynamic_embedder(embedder)
    table = _head_shared(embedder(caption_window_ids(fine, t, device)))
    ctx.data["caption_rope"] = table
    return table


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
      and only the bridge's keys are un-rotated (``bridge.py:10-24``).  The text
      block is the whole PADDED caption block — ``[0, cap_len)``, the prompt's
      tokens plus the model's own learned caption pad
      (``cap_pad_token``, ``comfy/ldm/lumina/model.py:620-623``) — the same span
      Phase I passes through, so both phases see the same caption and the pad
      tokens keep their attention outputs.  The answers go back in stream order
      with zeros only in the image pad span Phase II discards
      (``stitch_fine_zeros``).
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
            q[:, :, :layout.cap_len], k[:, :, :layout.cap_len], v[:, :, :layout.cap_len],
            q[:, :, first:last], k[:, :, first:last], v[:, :, first:last],
            unrotate1(k, table)[:, :, first:last], fine,
            rope_embedder=getattr(rt.dm, "rope_embedder", None),
            cap_len=layout.cap_len,
            caption_rope=_zimage_caption_rope(rt, ctx, fine, layout.cap_len, q.device),
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


# ---------------------------------------------------------------------------
# The Krea-2 seam
# ---------------------------------------------------------------------------


def _krea_check_foreign(symbol) -> None:
    """Refuse to install over another pack's hook on the same bound symbol.

    SPA's Krea-2 target is this very symbol (``src/spa.py:583-589`` returns
    ``("comfy.ldm.krea2.model", "optimized_attention_masked", True)``) and it
    marks its wrapper ``_spa_wrapper`` (``src/spa.py:1363``) — the same
    collision the Z-Image seam refuses (``_zimage_check_foreign``).  Unlike
    FLUX — where SPA binds ``comfy.ldm.flux.math.optimized_attention``, a
    *different* symbol — there is ONE symbol here, so the two cannot coexist.
    AReA never wraps another pack's hook silently: the install raises and the
    user reorders (plan §0, "foreign-binding collision ValueError").
    """
    if getattr(symbol, "_spa_wrapper", False):
        raise ValueError(
            f"AReA: {KREA_SEAM_MODULE}.{KREA_SEAM_SYMBOL} is already wrapped by "
            "SPA (src/spa.py:583-589 patches this same bound symbol for Krea-2, "
            "while for FLUX SPA binds comfy.ldm.flux.math.optimized_attention "
            "instead — a different symbol). AReA never wraps another pack's hook "
            "silently: restore SPA before applying AReA, or run the two on "
            "separate models. On a shared symbol whichever pack installed last "
            "would own every step."
        )


def _krea_pristine(bound):
    """The pristine symbol behind ``bound`` — an AReA wrapper carries its own.

    A re-apply after a patcher clone, or an install whose ref-count bookkeeping
    was reset underneath it, can find our own wrapper still bound with no
    recorded original.  The wrapper stows the symbol it wrapped
    (``_area_orig``), so the pristine is recoverable instead of wrapping our own
    wrapper.
    """
    if not getattr(bound, "_area_wrapper", False):
        return bound
    orig = getattr(bound, "_area_orig", None)
    if orig is None:
        raise ValueError(
            f"AReA: {KREA_SEAM_MODULE}.{KREA_SEAM_SYMBOL} is an AReA wrapper "
            "with no recorded pristine symbol, so the Krea-2 attention seam "
            "cannot be installed safely. Restart ComfyUI to clear it."
        )
    return orig


def _krea_delegate(orig, q, k, v, heads, mask, attn_precision, skip_reshape,
                   skip_output_reshape, kw, cls=None):
    """Hand the call back to the pristine symbol, UNTOUCHED.

    Nothing is consumed here: the containers reach the pristine symbol exactly
    as the model built them, so a delegating call is bit-identical to the
    unpatched model.  (The opposite mistake — taking a container and handing
    the taken tensor back — is the v2.19.1 crash class, ``src/area.py:589-600``.)

    ``cls`` re-wraps the tensors in FRESH containers: it is used when the
    engine has already taken the containers and then declined to compute (a
    missing rope table), so the pristine symbol still receives containers it
    can consume — a taken container must never be re-wrapped, and a raw tensor
    would make comfy's ``wrap_attn`` skip the take entirely
    (``comfy/ldm/modules/attention.py:215-253``).
    """
    if cls is not None:
        q, k, v = cls(q), cls(k), cls(v)
    return orig(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                **kw)


def _make_krea_wrapper(orig):
    """Wrap the Krea-2 ``optimized_attention_masked`` seam so AReA replaces it.

    The signature mirrors the real call bit-for-bit
    (``comfy/ldm/krea2/model.py:99-100``) and comfy's own
    ``(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
    skip_output_reshape=False, **kwargs)``
    (``comfy/ldm/modules/attention.py:256``): ``heads`` is the FOURTH POSITIONAL
    slot and Krea-2 passes ``mask`` / ``skip_reshape`` /
    ``preferred_attention`` / ``transformer_options`` as keywords.  A wrapper
    that mis-slots ``heads`` into ``mask`` fails silently, which is what
    ``tests/test_orig_call_convention.py`` pins.

    The q/k that arrive are **rotated** — the model applies its own rope before
    the call (``model.py:93``) — and the answer is comfy's flattened
    ``[b, n, heads*d]`` (``model.py:99-100`` passes no ``skip_output_reshape``,
    and the model multiplies it by a ``[b, n, dim]`` gate, ``model.py:106``).

    The seam replaces the attention on the steps AReA runs and DELEGATES
    untouched otherwise: a plain step, an auxiliary text-fusion call (20 heads
    over a stream that is a PREFIX of the joint one), a foreign stream, or a
    forward whose rope table was not captured.
    """
    def optimized_attention_masked(q, k, v, heads, mask=None, attn_precision=None,
                                   skip_reshape=False, skip_output_reshape=False, **kw):
        rt = _KRUNTIME
        ctx = None if rt is None else rt.ctx
        if ctx is None or ctx.phase is None:
            # No forward of ours, or a plain-attention step: delegate untouched.
            return _krea_delegate(orig, q, k, v, heads, mask, attn_precision,
                                  skip_reshape, skip_output_reshape, kw)
        layout = _krea_layout_for(rt, q, heads, kw)
        if layout is None:
            # An auxiliary call (20 heads) or a foreign stream: the model gets
            # its own attention back, untouched and with its containers alive.
            return _krea_delegate(orig, q, k, v, heads, mask, attn_precision,
                                  skip_reshape, skip_output_reshape, kw)
        cls = type(q) if hasattr(q, "take") else None
        q_t, k_t, v_t = _take(q), _take(k), _take(v)
        out = _krea_attention(rt, ctx, layout, q_t, k_t, v_t, heads, kw)
        if out is None:
            return _krea_delegate(orig, q_t, k_t, v_t, heads, mask, attn_precision,
                                  skip_reshape, skip_output_reshape, kw, cls)
        return out

    optimized_attention_masked._area_wrapper = True
    optimized_attention_masked._area_orig = orig
    return optimized_attention_masked


def _krea_stream_len(q) -> Optional[int]:
    """The sequence length of a seam argument, without consuming its container."""
    tensor = q.peek() if hasattr(q, "peek") else q
    if torch.is_tensor(tensor) and tensor.ndim == 4:
        return int(tensor.shape[2])
    return None


def _krea_layout_for(rt: Optional[_AreaRuntime], q, heads, kw: dict):
    """The joint layout of THIS call, or ``None`` when it is not a main block.

    Krea-2's forward reaches the seam four times before the main blocks: the
    text-fusion adapter's FOUR 20-head calls (``comfy/ldm/krea2/model.py:341``).
    Their stream length is the PROMPT length — a PREFIX of the joint stream's —
    so a length check alone cannot tell them apart.  The exact guard is
    :func:`~src.area_krea.krea_stream_eligible`: the head count (48 for the
    main blocks, 20 for the auxiliary ones — the GQA repeat is resolved BEFORE
    the seam, ``model.py:94-97``), the published ``img_slice``
    (``model.py:369``) and the stream length together.

    An ineligible call is DELEGATED UNCHANGED, and — crucially — it does NOT
    advance the block counter: the 28 main blocks must be numbered 0..27
    exactly as the reference numbers its blocks, or every block's random
    pooling stream would shift (the HAP bug class, ``src/spa.py:1033-1052``).
    """
    if rt is None or rt.ctx is None or rt.ctx.phase is None:
        return None
    ctx = rt.ctx
    if ctx.h * ctx.w < 1 or ctx.txt_len < 1:
        return None
    n = _krea_stream_len(q)
    if n is None:
        return None
    options = kw.get("transformer_options") or {}
    dm_heads = int(getattr(rt.dm, "heads", 0) or 0)
    if not krea_stream_eligible(heads, dm_heads, n, ctx.txt_len, ctx.h, ctx.w,
                                options.get("img_slice")):
        return None
    if rt.joint is not None:
        return rt.joint if rt.joint.seq_len == n else None
    try:
        layout = krea_layout(ctx.txt_len, ctx.h, ctx.w, n)
    except ValueError as exc:
        # A stream that does not fit the joint geometry: refuse the pooling
        # rather than guess, and say so once.
        _warn_once("krea-layout", "AReA: %s", exc)
        return None
    rt.joint = layout
    return layout


def _krea_coarse_ropes(rt: _AreaRuntime, layout, f: int, device):
    """The two Phase I rope tables, cached per (canvas, factor).

    Krea-2's rope ids ARE FLUX-shaped — the caption at the origin, every image
    token at ``(0, row, col)`` with axis 0 the FRAME index
    (``comfy/ldm/krea2/model.py:291-294``, ``:346``) — so the tables are
    FLUX's own recipe (floor-divided block coordinates) built through the
    model's OWN embedder, which lets a chained DyPE/SEGA embedder compose.

    They are built LAZILY, here at the seam, and never at forward-resolve time:
    the model's own table is the FIRST table of the forward (the recorder keeps
    it and never overwrites it), so a resolve-time build would be captured as
    the model's table.  The capture is PAUSED around the two embedder calls, so
    the coarse tables are not mistaken for extra tables of the forward.
    """
    embedder = getattr(rt.dm, "pe_embedder", None)
    _warn_dynamic_embedder(embedder)
    key = ("kcoarse", layout, f, str(device))
    cached = rt.ropes.get(key)
    if cached is not None:
        return cached
    try:
        ropes = coarse_ropes_krea(embedder, layout, f, device, rt.recorder)
    except ValueError as exc:
        _warn_once("krea-ropes", "AReA: %s This forward runs plain attention.", exc)
        return None
    rt.ropes[key] = ropes
    return ropes


def _krea_caption_rope(rt: _AreaRuntime, ctx: AreaContext, fine: FineLayout,
                        t: int, device):
    """The GTCN caption anchor table, built ONCE per step (cached on the context).

    The table depends only on the step's windows, the caption length and the
    model's own embedder — all of them fixed for the step — so the step's 28
    attention calls share it instead of rebuilding it per call (the Z-Image
    pattern, ``src/area.py``'s ``_zimage_caption_rope``).  The cache lives on
    the STEP's context, so a new step (new windows, new timestep) always
    rebuilds it.

    The anchor's axis 0 is zero, which is Krea-2's frame index
    (``comfy/ldm/krea2/model.py:292``) — :func:`~src.area_attn.caption_window_ids`
    needs no change for it.  The capture is paused around the embedder call, so
    the anchor table is not recorded as an extra table of the forward.
    """
    cached = ctx.data.get("caption_rope")
    if cached is not None:
        return cached
    embedder = getattr(rt.dm, "pe_embedder", None)
    if embedder is None:
        return None
    _warn_dynamic_embedder(embedder)
    ids = caption_window_ids(fine, t, device)
    recorder = rt.recorder
    if recorder is not None:
        with recorder.paused():
            table = _head_shared(embedder(ids))
    else:
        table = _head_shared(embedder(ids))
    ctx.data["caption_rope"] = table
    return table


def _krea_attention(rt: _AreaRuntime, ctx: AreaContext, layout, q, k, v, heads, kw):
    """The AReA computation of one Krea-2 main block, or ``None`` to delegate.

    q/k/v are the taken ``[b, heads, N, d]`` joint tensors, **rotated** — the
    model applies its own rope before the seam (``model.py:93``) — and the
    answer is comfy's flattened ``[b, n, heads*d]`` (``model.py:99-100`` passes
    no ``skip_output_reshape``, and the model's ``out * F.sigmoid(gate)``
    multiplies a ``[b, n, dim]`` gate, ``model.py:106``).

    Both phases start in the UNROTATED frame, exactly as Z-Image's do:

    * **Phase I** pools *unrotated* keys (pooling rotated keys would mix keys
      rotated by different angles inside one block, ``src/area_attn.py:364-368``)
      and rides the coarse tables, whose q side covers the whole joint stream
      (caption first, then the canvas) and whose kv side the engine's own key
      order.  The caption block is never pooled.
    * **Phase II** attends at the model's own full-resolution positions, which
      IS the table the recorder captured — so the rotated pair passes through
      untouched (a second rotation would be a float round trip of the same
      table) and only the bridge's keys are un-rotated and re-roped at their
      saturated positions.
    """
    table = rt.recorder.table if rt.recorder is not None else None
    if table is None or table.shape[2] != layout.seq_len:
        _warn_once(
            "krea-table",
            "AReA: the captured Krea-2 rope table covers %s tokens but the "
            "stream is %d, so the seam's rotation cannot be undone. This call "
            "runs plain attention.",
            "no" if table is None else int(table.shape[2]), layout.seq_len,
        )
        return None
    block = rt.block
    rt.block += 1
    t = ctx.txt_len
    k_unrot = unrotate1(k, table)
    if ctx.phase == "coarse":
        ropes = _krea_coarse_ropes(rt, layout, ctx.size, q.device)
        if ropes is None:
            return None
        rope_q, rope_kv = ropes
        k_pooled, v_pooled = pool_keys_values(
            k_unrot, v, t, ctx.h, ctx.w, ctx.size, rt.config.pooling,
            seed=rt.config.seed, step=ctx.step, block=block,
        )
        out = F.scaled_dot_product_attention(
            apply_rope1(unrotate1(q, table), rope_q),
            apply_rope1(k_pooled, rope_kv), v_pooled,
        )
    else:
        fine = _area_fine_layout(rt, ctx, k_unrot[:, :, t:],
                                 hook_name="diffusion_model.first")
        if fine is None:
            return None
        out_txt, out_img = windowed_attention(
            q[:, :, :t], k[:, :, :t], v[:, :, :t],
            q[:, :, t:], k[:, :, t:], v[:, :, t:],
            # the bridge ranks UNROTATED keys (bridge.py:10-24)
            k_unrot[:, :, t:], fine,
            rope_embedder=getattr(rt.dm, "pe_embedder", None),
            cap_len=t,
            caption_rope=_krea_caption_rope(rt, ctx, fine, t, q.device),
            # Krea-2's image tokens carry axis 0 = 0 (the frame index)
            img_axis0=0.0,
        )
        out = torch.cat([out_txt, out_img], dim=2)
    # comfy's own flatten: heads then head-dim, sequence back on axis 1
    # (ldm/modules/attention.py:309-314).
    b, n = out.shape[0], out.shape[2]
    dim = out.shape[3]
    return out.permute(0, 2, 1, 3).reshape(b, n, heads * dim)


def _krea_first_hook(module, args, output):
    """Capture the image embedding — the Phase II detail map's input.

    ``dm.first`` is Krea-2's image embedder (``comfy/ldm/krea2/model.py:333``):
    the analogue of FLUX's ``img_in`` hook (``src/area.py:1150-1156``) and of
    Z-Image's refiner entry (``src/area.py:999-1023``).  Its output is
    ``[b, h*w, features]`` — the image tokens in the model's own layout, which
    is the layout :func:`~src.area_attn.detail_map` reads.
    """
    rt = _KRUNTIME
    if rt is None or rt.ctx is None or rt.ctx.phase != "fine":
        return
    rt.hidden = output


def _krea_latent(x):
    """The Krea-2 latent of this forward, squeezed to 4-D — or ``None``.

    Krea-2's latent format is ``Wan21`` (``comfy/supported_models.py:2009-2035``),
    whose ``latent_dimensions == 3`` (``comfy/latent_formats.py:731``) makes
    comfy unsqueeze the sampler's 4-D latent to ``[B, 16, T, H, W]``
    (``comfy/sample.py:58-59``) — so the 5-D shape is the NORMAL image case,
    and AReA's own 4-D guard (``x.ndim != 4``) would make AReA a SILENT NO-OP on
    every Krea-2 image run.

    A latent with more than one frame is a VIDEO and is REFUSED (a one-time
    warning): the frames share their positions
    (``comfy/ldm/krea2/model.py:299-303``) and do not form one rectangular
    canvas, so AReA's windowed attention — a latent-IMAGE construction —
    cannot be applied to it.
    """
    if not torch.is_tensor(x):
        return None
    if x.dim() == 5 and int(x.shape[2]) > 1:
        _warn_once(
            "krea-video",
            "AReA: this Krea-2 forward carries a %d-frame video latent "
            "[B, C, T, H, W], and AReA's windowed attention is a latent-IMAGE "
            "construction: the frames share their positions "
            "(comfy/ldm/krea2/model.py:299-303) and do not form one rectangular "
            "canvas. This forward runs plain attention — render single images "
            "for AReA.",
            int(x.shape[2]),
        )
        return None
    try:
        return krea_squeeze_latent(x)
    except ValueError as exc:
        _warn_once("krea-latent", "AReA: %s", exc)
        return None


def _krea_ref_latents(conds) -> bool:
    """Does this forward carry REFERENCE latents?

    Krea-2 appends the reference images' tokens to the canvas tokens
    (``comfy/ldm/krea2/model.py:313-331``), so the image tokens are then NOT one
    rectangular canvas and pooling them as canvas tokens would corrupt the
    geometry silently.  The reference latents ride in the cond dict
    (``comfy/model_base.py:2780-2789``).
    """
    refs = conds.get("ref_latents")
    if refs is None:
        return False
    try:
        return len(refs) > 0
    except TypeError:
        return True


def _krea_seam_is_live(dm, wrapper) -> bool:
    """Does the model's own attention class still call OUR seam symbol?

    A quantized / fused attention kernel replaces the class, so its forward
    resolves no ``optimized_attention_masked`` at all — AReA would then be a
    SILENT no-op.  The class's forward looks the symbol up in its OWN module's
    globals (``comfy/ldm/krea2/model.py:21`` binds the name, ``:99-100`` calls
    it), so the check reads them: the seam is live only when the class resolves
    the symbol we just rebound (the Nunchaku precedent, ``src/area.py:1411-1420``).
    """
    blocks = getattr(dm, "blocks", None)
    if blocks is None:
        return False
    try:
        attn = blocks[0].attn
    except (AttributeError, IndexError, KeyError, TypeError):
        return False
    forward = getattr(type(attn), "forward", None)
    namespace = getattr(forward, "__globals__", None)
    if namespace is None:
        return False
    return namespace.get(KREA_SEAM_SYMBOL) is wrapper


def _krea_install_seam(m, dm) -> bool:
    """Patch ``comfy.ldm.krea2.model.optimized_attention_masked`` (ref-counted).

    Returns ``True`` when the seam is live and ``False`` when the model must be
    returned UNCHANGED — a quantized / fused attention kernel bypasses the seam
    entirely, so AReA would be a silent no-op there (the Nunchaku precedent).

    The symbol is process-global and shared by every patcher, so the install is
    a module-global rebind plus a ref-count — the proven in-pack pattern
    (``src/area.py``'s FLUX seam and ``src/spa.py:1341-1372``): comfy's
    ``add_object_patch`` is a per-patch record re-applied around model use
    (``comfy/model_patcher.py:739``), which would double-wrap a shared symbol on
    every clone/re-apply.  A second AReA apply re-uses the wrapper instead of
    wrapping our own wrapper, and the symbol is restored on the last release.
    """
    import importlib

    if _KSEAM["orig"] is None:
        try:
            mod = importlib.import_module(KREA_SEAM_MODULE)
        except Exception as exc:  # probe: backend module not importable here
            raise ValueError(
                f"AReA: cannot import {KREA_SEAM_MODULE} to patch the Krea-2 "
                f"attention seam ({exc}). AReA supports FLUX, Z-Image (Lumina2) "
                "and Krea-2; load an unmodified Krea-2 checkpoint on a current "
                "ComfyUI."
            ) from exc
        bound = getattr(mod, KREA_SEAM_SYMBOL, None)
        if bound is None:
            raise ValueError(
                f"AReA: {KREA_SEAM_MODULE}.{KREA_SEAM_SYMBOL} is missing in this "
                "ComfyUI version, so the Krea-2 attention seam cannot be "
                "installed. Update ComfyUI (or load an unmodified Krea-2 model)."
            )
        _krea_check_foreign(bound)
        orig = _krea_pristine(bound)
        wrapper = _make_krea_wrapper(orig)
        setattr(mod, KREA_SEAM_SYMBOL, wrapper)
        if not _krea_seam_is_live(dm, wrapper):
            # A fused / quantized attention class does not resolve the symbol:
            # put the pristine symbol back and leave the model alone.
            setattr(mod, KREA_SEAM_SYMBOL, orig)
            _warn_once(
                "krea-fused",
                "AReA: this Krea-2 model's attention class does not call "
                "optimized_attention_masked (a quantized/fused attention kernel "
                "bypasses the attention seam), so AReA would be a silent no-op; "
                "returning the model UNCHANGED. Use a non-quantized Krea-2 loader.",
            )
            return False
        _KSEAM.update(module=mod, attr=KREA_SEAM_SYMBOL, orig=orig)
        handles = []
        recorder = None
        embedder = getattr(dm, "pe_embedder", None)
        if embedder is not None and hasattr(embedder, "register_forward_hook"):
            hook, recorder = capture_krea_table(embedder)
            handles.append(hook.handle)
        else:
            _warn_once(
                "no-krea-rope",
                "AReA: this Krea-2 model has no diffusion_model.pe_embedder to "
                "hook, so the seam's rope table cannot be captured. Every forward "
                "runs plain attention.",
            )
        first = getattr(dm, "first", None)
        if first is not None and hasattr(first, "register_forward_hook"):
            handles.append(first.register_forward_hook(_krea_first_hook))
        else:
            _warn_once(
                "no-krea-first",
                "AReA: this Krea-2 model has no diffusion_model.first to hook, so "
                "the Phase II detail map falls back to the projected image keys of "
                "the first fine call.",
            )
        _KSEAM["recorder"] = recorder
        _KSEAM["hook"] = tuple(handles)
    _KSEAM["consumers"] += 1
    m._area_seam_consumer = True
    m._area_seam = "krea2"
    # The capture is process-wide for this install (its hook is registered once,
    # on the first consumer's embedder), so a later consumer re-uses it rather
    # than building a second recorder the registered hook never feeds.
    return True


# ---------------------------------------------------------------------------
# The Qwen-Image-2.1 seam
# ---------------------------------------------------------------------------


def _qwen21_check_foreign(symbol) -> None:
    """Refuse to install over another pack's hook on the same bound symbol.

    SPA's Qwen-Image-2.1 target is this very symbol (``src/spa.py:575-582``
    returns ``("comfy.ldm.qwen_image21.model", "optimized_attention", False)``)
    and it marks its wrapper ``_spa_wrapper`` (``src/spa.py:1363``) — the same
    collision the Krea-2 seam refuses (``_krea_check_foreign``).  Unlike FLUX —
    where SPA binds ``comfy.ldm.flux.math.optimized_attention``, a *different*
    symbol — there is ONE symbol here, so the two cannot coexist.  AReA never
    wraps another pack's hook silently: the install raises and the user
    reorders (plan §0, "foreign-binding collision ValueError").
    """
    if getattr(symbol, "_spa_wrapper", False):
        raise ValueError(
            f"AReA: {QWEN21_SEAM_MODULE}.{QWEN21_SEAM_SYMBOL} is already wrapped "
            "by SPA (src/spa.py:575-582 patches this same bound symbol for "
            "Qwen-Image-2.1, while for FLUX SPA binds comfy.ldm.flux.math."
            "optimized_attention instead — a different symbol). AReA never "
            "wraps another pack's hook silently: restore SPA before applying "
            "AReA, or run the two on separate models. On a shared symbol "
            "whichever pack installed last would own every step."
        )


def _qwen21_pristine(bound):
    """The pristine symbol behind ``bound`` — an AReA wrapper carries its own.

    A re-apply after a patcher clone, or an install whose ref-count bookkeeping
    was reset underneath it, can find our own wrapper still bound with no
    recorded original.  The wrapper stows the symbol it wrapped
    (``_area_orig``), so the pristine is recoverable instead of wrapping our
    own wrapper (the ``_krea_pristine`` pattern).
    """
    if not getattr(bound, "_area_wrapper", False):
        return bound
    orig = getattr(bound, "_area_orig", None)
    if orig is None:
        raise ValueError(
            f"AReA: {QWEN21_SEAM_MODULE}.{QWEN21_SEAM_SYMBOL} is an AReA wrapper "
            "with no recorded pristine symbol, so the Qwen-Image-2.1 attention "
            "seam cannot be installed safely. Restart ComfyUI to clear it."
        )
    return orig


def _qwen21_current_attention():
    """The CURRENT ``comfy.ldm.modules.attention.optimized_attention``.

    ``optimized_attention`` is REASSIGNED at import across the backends
    (sage / flash / xformers / pytorch / split / sub_quad /
    comfy_kitchen_int8 — ``comfy/ldm/modules/attention.py:923-948``), so the
    check follows the CURRENT binding and never a hardcoded
    ``attention_basic`` (D4) — which would refuse every non-basic backend.
    ``None`` when comfy's attention module is not importable here.
    """
    import importlib

    try:
        module = importlib.import_module("comfy.ldm.modules.attention")
    except Exception:  # probe: comfy's attention module is not importable here
        return None
    return getattr(module, "optimized_attention", None)


def _qwen21_check_fused(bound, backend) -> None:
    """N5 — the seam symbol must be comfy's CURRENT attention implementation.

    A quantized / fused attention kernel replaces the symbol the backend
    module calls, and AReA would then patch a function the model never calls
    — a SILENT no-op.  The check compares the backend module's own binding
    against the CURRENT ``comfy.ldm.modules.attention.optimized_attention``
    (D4), so a backend that imported a different variant is refused too.

    The mock harness (``tests/_qwen21_seam_fixtures.py``) registers ITS OWN
    module under comfy's name — that is the whole point of the harness: the
    rebind must reach the model's globals — and its symbol is not comfy's, so
    the check is about comfy's own backend and does not apply there.
    """
    if getattr(backend, "__name__", "") != QWEN21_SEAM_MODULE:
        return
    current = _qwen21_current_attention()
    if current is None or bound is current:
        return
    raise ValueError(
        f"AReA: {QWEN21_SEAM_MODULE}.{QWEN21_SEAM_SYMBOL} is not comfy's "
        "current attention implementation (comfy/ldm/modules/attention.py:"
        "923-948 reassigns optimized_attention per backend), so it is a "
        "fused or quantized kernel and AReA would patch a function the model "
        "never calls. Use a non-quantized Qwen-Image-2.1 loader."
    )


def _qwen21_disable_prefix_cache(m) -> None:
    """Switch 2.1's prefix K/V cache OFF on AReA's own clone (N12).

    The cache is ON by default (``comfy/model_base.py:2723-2724``) and the
    cached path is LENGTH-INDISTINGUISHABLE from the plain image segment
    (``model.py:222`` passes no ``mask`` keyword at all), so with the cache
    on AReA would engage on NO call: every seam call would be either the
    caption (masked) or the cached path.  The switch is the pack's
    established mechanism (``src/prefix_cache.py:26-49``) and it is a HARD
    PRECONDITION of the adaptation, not an optimisation.

    The option is REPLACED, not mutated in place: the clone's
    ``model_options`` is a fresh dict, so the caller's own transformer
    options keep their own cache setting.
    """
    try:
        model_options = m.model_options
    except AttributeError:  # probe: a mock patcher without model_options
        return
    options = dict(model_options.get("transformer_options") or {})
    options["qwen_image21_cache"] = {"device": "off"}
    model_options["transformer_options"] = options


def _qwen21_delegate(orig, q, k, v, heads, mask, attn_precision, skip_reshape,
                     skip_output_reshape, kw, cls=None):
    """Hand the call back to the pristine symbol, UNTOUCHED.

    Nothing is consumed here: the containers reach the pristine symbol exactly
    as the model built them, so a delegating call is bit-identical to the
    unpatched model.  (The opposite mistake — taking a container and handing
    the taken tensor back — is the v2.19.1 crash class, ``src/area.py:652-660``.)

    ``cls`` re-wraps the tensors in FRESH containers: it is used when the
    engine has already taken the containers and then declined to compute (a
    missing rope table), so the pristine symbol still receives containers it
    can consume — a taken container must never be re-wrapped, and a raw tensor
    would make comfy's ``wrap_attn`` skip the take entirely
    (``comfy/ldm/modules/attention.py:215-253``).

    The ``mask`` SENTINEL (``QWEN21_NO_MASK``) is the prefix-cache path's
    "no keyword at all"; the pristine's own default for ``mask`` is ``None``
    (``comfy/ldm/modules/attention.py:256``), so the call is the same one.
    """
    if cls is not None:
        q, k, v = cls(q), cls(k), cls(v)
    if mask is QWEN21_NO_MASK:
        mask = None
    return orig(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape,
                **kw)


def _make_qwen21_wrapper(orig):
    """Wrap the Qwen-Image-2.1 ``optimized_attention`` seam so AReA replaces it.

    The signature mirrors comfy's own
    ``(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False,
    skip_output_reshape=False, **kwargs)``
    (``comfy/ldm/modules/attention.py:255``) and the model's own call
    (``comfy/ldm/qwen_image21/model.py:211``): ``heads`` is the FOURTH
    POSITIONAL slot and 2.1 passes ``mask`` / ``transformer_options`` /
    ``preferred_attention`` as keywords.

    **``mask`` carries a SENTINEL default** (``QWEN21_NO_MASK``): the plain
    path ALWAYS passes the keyword and the prefix-cache path NEVER does
    (``model.py:222``), and the two are length-indistinguishable — the cached
    key is ``txt_len + h*w`` over ``h*w`` queries, exactly the plain image
    segment's signature.  Only the keyword's PRESENCE tells them apart, so
    the absent case needs a value of its own (``None`` is a legal mask).

    The seam's q/k/v are **3-D** ``[b, n, heads*dim]`` (2.1 flattens
    ``(B, n, H, D)`` itself, ``model.py:206-208``) and arrive **rotated** (the
    model applies its own rope before the call, ``model.py:99-100``); the
    answer is the FLATTENED ``[b, n, heads*dim]`` (``model.py:212``
    concatenates the per-segment answers on dim 1).
    """
    def optimized_attention(q, k, v, heads, mask=QWEN21_NO_MASK, attn_precision=None,
                            skip_reshape=False, skip_output_reshape=False, **kw):
        rt = _QRUNTIME
        ctx = None if rt is None else rt.ctx
        if ctx is None or ctx.phase is None:
            # No forward of ours, or a plain-attention step: delegate untouched.
            return _qwen21_delegate(orig, q, k, v, heads, mask, attn_precision,
                                    skip_reshape, skip_output_reshape, kw)
        layout = _qwen21_layout_for(rt, q, k, heads, mask)
        if layout is None:
            # The caption segment (a causal mask), the prefix-cache path (NO
            # mask keyword) or a foreign stream: the model gets its own
            # attention back, untouched and with its containers alive.
            return _qwen21_delegate(orig, q, k, v, heads, mask, attn_precision,
                                    skip_reshape, skip_output_reshape, kw)
        cls = type(q) if hasattr(q, "take") else None
        q_t, k_t, v_t = _take(q), _take(k), _take(v)
        if ctx.phase == "coarse":
            out = _qwen21_attention(rt, ctx, layout, q_t, k_t, v_t, heads)
        else:
            out = _qwen21_fine(rt, ctx, layout, q_t, k_t, v_t, heads)
        if out is None:
            # The engine took the containers and then declined to compute: the
            # pristine symbol gets FRESH containers around the same tensors.
            return _qwen21_delegate(orig, q_t, k_t, v_t, heads, mask,
                                    attn_precision, skip_reshape,
                                    skip_output_reshape, kw, cls)
        return out

    optimized_attention._area_wrapper = True
    optimized_attention._area_orig = orig
    return optimized_attention


def _qwen21_stream_len(x) -> Optional[int]:
    """The sequence length of a seam argument, without consuming its container.

    2.1's seam input is 3-D ``[b, n, heads*dim]`` (``model.py:206-208``), so
    the sequence is axis 1 — unlike every other AReA seam, which sees 4-D.
    """
    tensor = qwen21_peek(x)
    if torch.is_tensor(tensor) and tensor.ndim == 3:
        return int(tensor.shape[1])
    return None


def _qwen21_to_heads(q, k, v, heads):
    """The three seam tensors in the engine's layout, or ``None``.

    A flat width that does not split into the seam's heads is a FOREIGN call
    (N13): it is reported as ``None`` and the seam DELEGATES it, so a sampling
    run is never aborted by an unknown variant of the backbone.
    """
    try:
        return (qwen21_to_heads(q, heads), qwen21_to_heads(k, heads),
                qwen21_to_heads(v, heads))
    except ValueError as exc:
        _warn_once("qwen21-heads", "AReA: %s", exc)
        return None


def _qwen21_model_geometry(dm) -> tuple[Optional[int], Optional[int]]:
    """The backbone's own ``(head count, head dim)``, or ``(None, None)``.

    2.1's constructor keeps neither ``num_attention_heads`` nor
    ``attention_head_dim`` on the model (``model.py:236-251`` — only
    ``inner_dim`` survives), but every attention block carries its head count
    (``model.py:73`` — ``self.heads = heads``) and the head dim follows from
    ``inner_dim // heads``.  The mock harness keeps the head count on the model
    itself (``tests/_qwen21_seam_fixtures.py``).

    The pair is the N13 discriminator: a seam call whose head count (or head
    width) is not the model's own belongs to an UNKNOWN VARIANT, and such a
    call is DELEGATED — never engaged with, and never a reason to abort a
    sampling run.
    """
    heads = getattr(dm, "heads", None)
    if not (isinstance(heads, int) and heads > 0):
        heads = None
        blocks = getattr(dm, "transformer_blocks", None)
        if blocks:
            attn = getattr(blocks[0], "attn", None)
            value = getattr(attn, "heads", None)
            if isinstance(value, int) and value > 0:
                heads = value
    if heads is None:
        return None, None
    inner = getattr(dm, "inner_dim", None)
    head_dim = None
    if isinstance(inner, int) and inner > 0 and inner % heads == 0:
        head_dim = inner // heads
    return heads, head_dim


def _qwen21_layout_for(rt: Optional[_AreaRuntime], q, k, heads, mask):
    """The image segment's layout of THIS call, or ``None`` when it is not it.

    2.1's forward reaches the seam three ways (the design's H2): the PLAIN
    path (the caption segment with a causal mask, then the image segment with
    ``mask=None``), the PREFIX pass (the caption segments only, also masked)
    and the PREFIX-CACHE path (NO ``mask`` keyword at all, ``model.py:222``).
    The cached path is LENGTH-INDISTINGUISHABLE from the plain image segment
    (its key is ``txt_len + h*w`` over ``h*w`` queries), so only the keyword's
    presence separates them — which is why the wrapper's ``mask`` default is
    the ``QWEN21_NO_MASK`` sentinel.

    An ineligible call is DELEGATED UNCHANGED, and — crucially — it does NOT
    advance the block counter: the 32 blocks must be numbered 0..31 exactly as
    the reference numbers its blocks, or every block's random pooling stream
    would shift (the HAP bug class, ``src/spa.py:1033-1052``).
    """
    if rt is None or rt.ctx is None or rt.ctx.phase is None:
        return None
    ctx = rt.ctx
    if ctx.h * ctx.w < 1 or ctx.txt_len < 1:
        return None
    q_len = _qwen21_stream_len(q)
    k_len = _qwen21_stream_len(k)
    if q_len is None or k_len is None:
        return None
    # N13 — the head geometry must be the MODEL's own.  A flat width that does
    # not split into the seam's heads, or a head count the model does not
    # declare, is an UNKNOWN VARIANT: the call is DELEGATED here, BEFORE the
    # containers are taken, so the pristine symbol still gets them alive and
    # a sampling run is never aborted by a foreign call.
    head_dim = qwen21_head_geometry(qwen21_peek(k), heads)
    if head_dim is None:
        return None
    dm_heads, dm_head_dim = _qwen21_model_geometry(rt.dm)
    if dm_head_dim is not None and head_dim != dm_head_dim:
        return None
    # The forward's whole sequence: the model's own rope table covers exactly
    # the joint stream, so its length IS the total (``model.py:338``).
    table = rt.recorder.table if rt.recorder is not None else None
    total_len = int(table.shape[2]) if table is not None else None
    if not qwen21_segment_eligible(mask, q_len, k_len, total_len, ctx.txt_len,
                                    ctx.h, ctx.w, heads, dm_heads):
        return None
    try:
        layout = qwen21_layout(ctx.txt_len, ctx.h, ctx.w, q_len, k_len)
    except ValueError as exc:
        # A stream that does not fit the segment geometry: refuse the pooling
        # rather than guess, and say so once.
        _warn_once("qwen21-layout", "AReA: %s", exc)
        return None
    return layout


def _qwen21_coarse_ropes(rt: _AreaRuntime, layout, f: int, device):
    """The two Phase I rope tables, cached per (canvas, factor).

    2.1's own ids are neither FLUX- nor Z-Image-shaped: the caption rides a
    DIAGONAL at the model's own running counter and the image's spatial axes
    are CENTRED on the canvas (``comfy/ldm/qwen_image21/model.py:323``,
    ``:330-332``), so the tables are built by the adapter through the model's
    OWN embedder — a chained DyPE/SEGA embedder composes.

    They are built LAZILY, here at the seam, and never at forward-resolve
    time: the model's own table is the FIRST table of the forward (the
    recorder keeps it and never overwrites it), so a resolve-time build would
    be captured as the model's table.  The capture is PAUSED around the two
    embedder calls, so the coarse tables are not mistaken for extra tables of
    the forward.
    """
    embedder = getattr(rt.dm, "pe_embedder", None)
    _warn_dynamic_embedder(embedder)
    key = ("qcoarse", layout, f, str(device))
    cached = rt.ropes.get(key)
    if cached is not None:
        return cached
    try:
        ropes = coarse_ropes_qwen21(embedder, layout, f, device, rt.recorder)
    except ValueError as exc:
        _warn_once("qwen21-ropes", "AReA: %s This forward runs plain attention.", exc)
        return None
    rt.ropes[key] = ropes
    return ropes


def _qwen21_attention(rt: _AreaRuntime, ctx: AreaContext, layout, q, k, v, heads):
    """The AReA computation of one Qwen-Image-2.1 image segment, or ``None``.

    q/k/v are the taken ``[b, n, heads*dim]`` tensors, read HEAD-MAJOR (2.1
    flattens ``(B, n, H, D)`` itself, ``model.py:206-208``), and the model's
    q/k arrive **rotated** (it applies its own rope before the call,
    ``model.py:99-100``), so both phases start in the UNROTATED frame.

    **Phase I** pools *unrotated* keys (pooling rotated keys would mix keys
    rotated by different angles inside one block, ``src/area_attn.py:364-368``)
    and rides the coarse tables, whose two sides have DIFFERENT lengths on 2.1
    (the design's H5): ``rope_q`` covers the ``h*w`` IMAGE QUERIES only — the
    eligible call has NO text queries — and ``rope_kv`` covers
    ``txt_len + nh*nw`` rows, the caption at the model's own diagonal ids and
    the pooled keys at their block's continuous centroid.  The caption's keys
    are never pooled, so the prompt stays visible to the image queries.
    """
    table = rt.recorder.table if rt.recorder is not None else None
    if table is None or table.shape[2] != layout.k_len:
        _warn_once(
            "qwen21-table",
            "AReA: the captured Qwen-Image-2.1 rope table covers %s tokens but "
            "the joint stream is %d, so the seam's rotation cannot be undone. "
            "This call runs plain attention.",
            "no" if table is None else int(table.shape[2]), layout.k_len,
        )
        return None
    block = rt.block
    rt.block += 1
    t = layout.txt_len
    heads_t = _qwen21_to_heads(q, k, v, heads)
    if heads_t is None:
        return None
    q_h, k_h, v_h = heads_t
    # The model rotated the WHOLE joint stream and sliced afterwards, so the
    # image queries were rotated with the table's TRAILING rows and the joint
    # key with its leading ones — the model's own table, sliced, is the exact
    # inverse of that rotation.
    q_unrot = unrotate1(q_h, qwen21_slice_table(table, t, layout.k_len))
    k_unrot = unrotate1(k_h, qwen21_slice_table(table, 0, layout.k_len))
    ropes = _qwen21_coarse_ropes(rt, layout, ctx.size, q.device)
    if ropes is None:
        return None
    rope_q, rope_kv = ropes
    k_pooled, v_pooled = pool_keys_values(
        k_unrot, v_h, t, layout.h, layout.w, ctx.size, rt.config.pooling,
        seed=rt.config.seed, step=ctx.step, block=block,
    )
    out = F.scaled_dot_product_attention(
        apply_rope1(q_unrot, rope_q), apply_rope1(k_pooled, rope_kv), v_pooled
    )
    # comfy's own flatten: heads then head-dim, sequence back on axis 1
    # (comfy/ldm/modules/attention.py:309-314).
    b, n = out.shape[0], out.shape[2]
    dim = out.shape[3]
    return out.permute(0, 2, 1, 3).reshape(b, n, heads * dim)


def _qwen21_fine(rt: _AreaRuntime, ctx: AreaContext, layout, q, k, v, heads):
    """The Phase II computation of one Qwen-Image-2.1 image segment, or ``None``.

    q/k/v are the taken ``[b, n, heads*dim]`` tensors of the IMAGE segment:
    the queries are the ``h*w`` image rows and the keys are the JOINT set —
    the caption AND the image (``comfy/ldm/qwen_image21/model.py:206-208``).

    **The caption contributes KEYS and VALUES only.**  2.1's eligible call has
    no text queries at all, so the engine is called with ``text_keys_only=True``
    (the S7 keyword): the caption stays VISIBLE to the image queries and the
    text answer is empty — the model concatenates the per-segment answers on
    dim 1 (``model.py:212``), and the caption's own answer comes from the
    DELEGATED caption segment.

    **The paper's caption placement** (the design's H7): the GTCN caption
    anchor assumes a caption at the spatial origin, and 2.1's caption rides a
    DIAGONAL at the model's own running counter (``model.py:323``) — so no
    embedder is passed, the caption keeps its own true positions, and the
    bridge keys keep their true positions too (the horizon-saturation branch
    lives inside the ``rope_embedder is not None`` branch of
    :func:`~src.area_attn.windowed_attention`, so it never runs here and
    ``img_axis0`` is not used for 2.1 at all).
    """
    table = rt.recorder.table if rt.recorder is not None else None
    if table is None or table.shape[2] != layout.k_len:
        _warn_once(
            "qwen21-fine-table",
            "AReA: the captured Qwen-Image-2.1 rope table covers %s tokens but "
            "the joint stream is %d, so the seam's rotation cannot be undone. "
            "This call runs plain attention.",
            "no" if table is None else int(table.shape[2]), layout.k_len,
        )
        return None
    if rt.hidden is None:
        # N9 — no image embedder output, so there is no detail map and no
        # Phase II.  Phase I only, never a faked map.
        _warn_once(
            "qwen21-no-detail",
            "AReA: this Qwen-Image-2.1 forward carries no img_in output, so "
            "there is no detail map and Phase II is disabled. This call runs "
            "plain attention.",
        )
        return None
    try:
        energy = qwen21_detail_map(rt.hidden, ctx.h, ctx.w)
    except ValueError as exc:
        _warn_once("qwen21-detail", "AReA: %s", exc)
        return None
    rt.block += 1
    t = layout.txt_len
    heads_t = _qwen21_to_heads(q, k, v, heads)
    if heads_t is None:
        return None
    q_h, k_h, v_h = heads_t
    # The model rotated the WHOLE joint stream with its own table and sliced
    # afterwards, so the keys arrive ROTATED at their true positions (the
    # caption at its diagonal ids, the image at its centred ones) and the
    # bridge's ranking needs them UNROTATED.
    k_unrot = unrotate1(k_h, qwen21_slice_table(table, 0, layout.k_len))
    fine = _area_fine_layout(rt, ctx, k_h[:, :, t:], hook_name="diffusion_model.img_in",
                             energy=energy)
    if fine is None:
        return None
    out_txt, out_img = windowed_attention(
        # no text queries at all: the caption is keys and values only
        q_h[:, :, :0],
        k_h[:, :, :t], v_h[:, :, :t],
        q_h, k_h[:, :, t:], v_h[:, :, t:],
        # the bridge ranks UNROTATED keys (bridge.py:10-24)
        k_unrot[:, :, t:], fine,
        # 2.1's caption rides a diagonal, so the GTCN caption anchor does not
        # apply: the paper's own placement, the bridge keys at their true
        # positions, and no img_axis0 (the image's axis 0 is the model's own
        # running counter, ``model.py:330-332``).
        rope_embedder=None,
        cap_len=t,
        caption_rope=None,
        img_axis0=None,
        text_keys_only=True,
    )
    # comfy's own flatten: heads then head-dim, sequence back on axis 1
    # (comfy/ldm/modules/attention.py:309-314).
    b, n = out_img.shape[0], out_img.shape[2]
    dim = out_img.shape[3]
    return out_img.permute(0, 2, 1, 3).reshape(b, n, heads * dim)


def _qwen21_img_in_hook(module, args, output):
    """Capture the image embedding — the Phase II detail map's input.

    ``dm.img_in`` is 2.1's image embedder (``comfy/ldm/qwen_image21/model.py:
    260``): a plain ``Linear`` over the latent's channels, applied to the
    flattened latent (``model.py:328``), so its output is ``[b, h*w, dim]`` —
    the image tokens in the model's own layout, which is the layout
    :func:`~src.area_attn.detail_map` reads.

    **The hook fires ONCE PER IMAGE, not once per forward** (D6):
    ``build_sequence`` embeds every reference latent and then the target, and
    the target is appended LAST (``model.py:319``).  So the hook keeps the
    LAST call and counts them; more than one call means the forward carries
    reference latents, the image tokens are then not one rectangular canvas,
    and the detail map would be a map of the wrong image — so the capture is
    dropped and the forward runs plain attention (the seam's own length
    guard refuses that call anyway).
    """
    rt = _QRUNTIME
    if rt is None or rt.ctx is None or rt.ctx.phase != "fine":
        return
    rt.img_in_calls += 1
    if rt.img_in_calls > 1:
        _warn_once(
            "qwen21-ref-latents",
            "AReA: this Qwen-Image-2.1 forward embeds %d images (comfy/ldm/"
            "qwen_image21/model.py:319 embeds every reference latent and then "
            "the target), so the image tokens are not one rectangular canvas "
            "and AReA's Phase II detail map would be a map of the wrong "
            "image. This forward runs plain attention — render a "
            "reference-free image for AReA.",
            rt.img_in_calls,
        )
        rt.hidden = None
        return
    rt.hidden = output


def _qwen21_hooked(opts: dict) -> bool:
    """Does this forward carry a model patch that rewrites 2.1's rope or states?

    ``comfy/ldm/qwen_image21/model.py:356-359`` runs the ``post_input`` patches
    over the built stream — and they may REPLACE the rope table ``pe`` — and
    ``model.py:88-98`` runs the ``attn1_patch`` patches over q/k/v.  Either one
    makes AReA's captured table (the model's own ``pe_embedder`` output) the
    wrong table for the tensors the seam receives, so the forward must run
    plain attention (N11).  The patches are per-FORWARD transformer options,
    so this is a forward-time refusal — the install cannot know them.
    """
    patches = opts.get("patches") or {}
    for name in ("post_input", "single_block", "attn1_patch"):
        if patches.get(name):
            return True
    replace = opts.get("patches_replace") or {}
    return bool(replace.get("dit"))


def _qwen21_seam_is_live(dm, wrapper) -> bool:
    """Does the model's own backend module still call OUR seam symbol?

    2.1 calls the seam from the MODULE-LEVEL ``block_causal_attention``
    (``comfy/ldm/qwen_image21/model.py:197-213``) and
    ``prefix_cached_attention`` (``:216-223``), which resolve the bound name
    in the BACKEND MODULE's globals — so the check reads those globals: the
    seam is live only when they resolve the symbol we just rebound (the
    Nunchaku precedent, the ``_krea_seam_is_live`` pattern).  A quantized /
    fused attention kernel replaces the call, so AReA would then be a SILENT
    no-op.
    """
    import importlib

    try:
        mod = importlib.import_module(QWEN21_SEAM_MODULE)
    except Exception:  # probe: the backend module is not importable here
        return False
    for name in ("block_causal_attention", "prefix_cached_attention"):
        function = getattr(mod, name, None)
        namespace = getattr(function, "__globals__", None)
        if namespace is None:
            continue
        if namespace.get(QWEN21_SEAM_SYMBOL) is wrapper:
            return True
    return False


def _qwen21_install_seam(m, dm):
    """Patch ``comfy.ldm.qwen_image21.model.optimized_attention`` (ref-counted).

    Returns ``True`` when the seam is live and ``False`` when the model must be
    returned UNCHANGED — a quantized / fused attention kernel bypasses the seam
    entirely, so AReA would be a silent no-op there (the Nunchaku precedent,
    the ``_krea_install_seam`` pattern).

    The symbol is process-global and shared by every patcher, so the install is
    a module-global rebind plus a ref-count — the proven in-pack pattern
    (``src/area.py``'s FLUX seam and ``src/spa.py:1341-1372``): comfy's
    ``add_object_patch`` is a per-patch record re-applied around model use
    (``comfy/model_patcher.py:739``), which would double-wrap a shared symbol on
    every clone/re-apply.  A second AReA apply re-uses the wrapper instead of
    wrapping our own wrapper, and the symbol is restored on the last release.
    """
    import importlib

    # N11 — a model that carries an attention or post-input patch OF ITS OWN
    # would have its rope table (or its hidden states) REPLACED behind AReA's
    # back (``model.py:88-98``, ``:356-359``), so the captured table would not
    # describe the rotation the seam receives.  Refuse BEFORE the rebind, so a
    # refused install leaves the module pristine.  (A patch that arrives per
    # FORWARD, through ``transformer_options``, is refused at forward time
    # instead — the install cannot know it.)
    for name in ("attn1_patch", "post_input"):
        if getattr(dm, name, None) is not None:
            raise ValueError(
                f"AReA: this Qwen-Image-2.1 model carries a {name!r} patch, "
                "which may replace the rope table or the hidden states "
                "(comfy/ldm/qwen_image21/model.py:88-98, :356-359), so AReA's "
                "captured rope table would not match the tensors the seam "
                "receives. Use an unmodified Qwen-Image-2.1 loader."
            )

    if _QSEAM["orig"] is None:
        try:
            mod = importlib.import_module(QWEN21_SEAM_MODULE)
        except Exception as exc:  # probe: backend module not importable here
            raise ValueError(
                f"AReA: cannot import {QWEN21_SEAM_MODULE} to patch the "
                f"Qwen-Image-2.1 attention seam ({exc}). AReA supports FLUX, "
                "Z-Image (Lumina2), Krea-2 and Qwen-Image-2.1; load an "
                "unmodified Qwen-Image-2.1 checkpoint on a current ComfyUI."
            ) from exc
        bound = getattr(mod, QWEN21_SEAM_SYMBOL, None)
        if bound is None:
            raise ValueError(
                f"AReA: {QWEN21_SEAM_MODULE}.{QWEN21_SEAM_SYMBOL} is missing in "
                "this ComfyUI version, so the Qwen-Image-2.1 attention seam "
                "cannot be installed. Update ComfyUI (or load an unmodified "
                "Qwen-Image-2.1 model)."
            )
        _qwen21_check_foreign(bound)
        orig = _qwen21_pristine(bound)
        _qwen21_check_fused(orig, mod)
        embedder = getattr(dm, "pe_embedder", None)
        if embedder is None or not hasattr(embedder, "register_forward_hook"):
            # N8 — without the model's own embedder there is no rope table to
            # capture, so the seam's rotation cannot be undone and Phase I
            # would pool rotated keys.  Refuse BEFORE the rebind, so the
            # module is left pristine.
            raise ValueError(
                "AReA: this Qwen-Image-2.1 model has no diffusion_model."
                "pe_embedder to hook, so AReA cannot capture the model's own "
                "rope table and cannot undo the rotation the seam receives "
                "(comfy/ldm/qwen_image21/model.py:99-100). Use an unmodified "
                "Qwen-Image-2.1 loader."
            )
        wrapper = _make_qwen21_wrapper(orig)
        setattr(mod, QWEN21_SEAM_SYMBOL, wrapper)
        if not _qwen21_seam_is_live(dm, wrapper):
            # A fused / quantized attention class does not resolve the symbol:
            # put the pristine symbol back and leave the model alone.
            setattr(mod, QWEN21_SEAM_SYMBOL, orig)
            _warn_once(
                "qwen21-fused",
                "AReA: this Qwen-Image-2.1 model's backend module does not call "
                "optimized_attention (a quantized/fused attention kernel "
                "bypasses the attention seam), so AReA would be a silent no-op; "
                "returning the model UNCHANGED. Use a non-quantized "
                "Qwen-Image-2.1 loader.",
            )
            return False
        _QSEAM.update(module=mod, attr=QWEN21_SEAM_SYMBOL, orig=orig)
        handles = []
        hook, recorder = capture_qwen21_table(embedder)
        handles.append(hook.handle)
        # The Phase II detail map's source: 2.1's own image embedder.  A model
        # without one is Phase-I-only (N9) — the fine steps then delegate.
        img_in = getattr(dm, "img_in", None)
        if img_in is not None and hasattr(img_in, "register_forward_hook"):
            handles.append(img_in.register_forward_hook(_qwen21_img_in_hook))
        else:
            _warn_once(
                "no-qwen21-img-in",
                "AReA: this Qwen-Image-2.1 model has no diffusion_model.img_in "
                "to hook, so there is no detail map and Phase II is disabled — "
                "Phase I only. Every fine step runs plain attention.",
            )
        _QSEAM["recorder"] = recorder
        _QSEAM["hook"] = tuple(handles)
    _QSEAM["consumers"] += 1
    m._area_seam_consumer = True
    m._area_seam = "qwen21"
    # The capture is process-wide for this install (its hook is registered once,
    # on the first consumer's embedder), so a later consumer re-uses it rather
    # than building a second recorder the registered hook never feeds.
    return True


def _area_install_seam(m, dm, profile):
    """Patch the backbone's bound attention seam and hook the model (ref-counted).

    FLUX, Z-Image and Krea-2 are different symbols in different modules, so each
    takes its own install path and its own ref-count — the three never share one.
    Returns the Z-Image rope capture (``None`` on the FLUX side), which the
    caller hands to the runtime it builds, or ``False`` when a Krea-2 model
    carries a fused attention kernel and must be left UNCHANGED.
    """
    if profile is ZIMAGE_PROFILE:
        return _zimage_install_seam(m, dm)
    if profile is KREA_PROFILE:
        return _krea_install_seam(m, dm)
    if profile is QWEN21_PROFILE:
        return _qwen21_install_seam(m, dm)
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


def _warn_regime(total: int, phase1: int, phase: Optional[str], mode: str = "steps") -> None:
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

    ``mode`` is the RESOLVED switch (the caller passes ``cfg.phase1_mode``).  A
    sigma-resolved schedule is itself the remedy the advisory recommends: its
    count already follows the schedule, and on the 8-step Turbo table
    ``phase1_sigma = 0.89`` resolves 3 coarse steps, which trips the
    third-of-steps threshold and would make the DEFAULT Z-Image run warn on
    every forward — so sigma mode returns early.
    """
    if phase != "coarse" or not is_short_schedule(total):
        return
    if mode == "sigma":
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
    if rt.profile is KREA_PROFILE:
        # Per-forward Krea-2 state (the SPA discipline): the rope table is
        # captured anew every forward, and the arming window opens NOW — the
        # model's own embedder is called ONCE per forward, after the text
        # fusion (``comfy/ldm/krea2/model.py:364``).
        if rt.recorder is None:
            rt.recorder = _KSEAM["recorder"] or KreaRopeRecorder()
        rt.recorder.reset()
        rt.recorder.armed = True
        rt.joint = None
    if rt.profile is QWEN21_PROFILE:
        # Per-forward Qwen-Image-2.1 state (the SPA discipline): the rope table
        # is captured anew every forward, and the arming window opens NOW — the
        # model's own embedder is called ONCE per forward, over the whole
        # joint stream (``comfy/ldm/qwen_image21/model.py:338``).
        if rt.recorder is None:
            rt.recorder = _QSEAM["recorder"] or Qwen21RopeRecorder()
        rt.recorder.reset()
        rt.recorder.armed = True
        rt.joint = None
        # The image embedder fires ONCE PER IMAGE (2.1 embeds every reference
        # latent and then the target, ``model.py:319``), so the count starts
        # at zero for every forward (D6).
        rt.img_in_calls = 0
    x = params.get("input")
    conds = params.get("c") or {}
    if rt.profile is KREA_PROFILE:
        # A Krea-2 latent is 4-D or 5-D, and the 5-D one with a single frame is
        # the NORMAL image case (the Wan21 latent format has three dimensions,
        # so comfy unsqueezes the sampler's latent).  A video (T > 1) and a
        # forward carrying reference latents are REFUSED — neither is one
        # rectangular canvas, and pooling them would corrupt the geometry
        # silently.
        x = _krea_latent(x)
        if x is None:
            return ctx
        if _krea_ref_latents(conds):
            _warn_once(
                "krea-ref-latents",
                "AReA: this Krea-2 forward carries reference latents, so the "
                "image tokens are not one rectangular canvas any more "
                "(comfy/ldm/krea2/model.py:313-331 appends the reference "
                "images' tokens to the canvas ones) and AReA cannot pool them "
                "without corrupting the geometry. This forward runs plain "
                "attention — render a reference-free image for AReA.",
            )
            return ctx
        patch = int(getattr(rt.dm, "patch", KREA_PATCH) or KREA_PATCH)
        h, w = krea_latent_grid(x, patch)
    elif rt.profile is QWEN21_PROFILE:
        # A Qwen-Image-2.1 latent is 4-D [B, C, H, W] and its token grid IS
        # the latent's own (H, W): 2.1 does not patchify (``img_in`` is a plain
        # Linear, ``comfy/ldm/qwen_image21/model.py:260``, and the model reads
        # the grid off the latent's own shape, ``model.py:327``).  A 5-D latent
        # is a VIDEO latent and is REFUSED LOUDLY (N4): 2.1's format is 2-D
        # (``latent_formats.py:959``), and a silent no-op would leave the user
        # with an unpatched run and no explanation (the Krea-2 lesson).  The
        # latent is known only HERE — never at install time — so this is the
        # earliest point the refusal can happen, and it happens BEFORE the
        # seam sees a single tensor.
        if torch.is_tensor(x) and x.dim() != 4:
            raise ValueError(
                "AReA does not support video / 5-D latents for Qwen-Image-2.1: "
                f"the latent is {x.dim()}-D with shape {tuple(x.shape)}, but "
                "2.1's latent format is 4-D [B, C, H, W] (comfy/latent_formats."
                "py:957-960 — latent_dimensions = 2). AReA's windowed attention "
                "is a latent-IMAGE construction: render a still image for AReA, "
                "or leave AReA off for the video run."
            )
        try:
            h, w = qwen21_latent_grid(x)
        except ValueError as exc:
            _warn_once("qwen21-latent", "AReA: %s", exc)
            return ctx
    else:
        if not torch.is_tensor(x) or x.ndim != 4:
            return ctx  # nothing pool-able at this shape -> plain attention
        patch_size = int(getattr(rt.dm, "patch_size", FLUX_PATCH_SIZE) or FLUX_PATCH_SIZE)
        h, w = _grid_from_latent(x, patch_size)
    cross = conds.get("c_crossattn")
    if torch.is_tensor(cross) and cross.ndim >= 2:
        ctx.txt_len = int(cross.shape[1])
    opts = conds.get("transformer_options") or {}
    if rt.profile is QWEN21_PROFILE and _qwen21_hooked(opts):
        # N11 — a patched forward rewrites the rope table or the hidden
        # states, so AReA's captured table would not match the seam's
        # tensors.  The patches are per-FORWARD options, so this is a
        # forward-time refusal: plain attention, once, loudly.
        _warn_once(
            "qwen21-patches",
            "AReA: this Qwen-Image-2.1 forward carries model patches "
            "(transformer_options['patches'] or 'patches_replace'), which "
            "rewrite the rope table or the hidden states (comfy/ldm/"
            "qwen_image21/model.py:88-98, :356-359), so AReA's captured rope "
            "table would not match the tensors the seam receives. This "
            "forward runs plain attention.",
        )
        return ctx
    step, total = _area_locate(opts.get("sample_sigmas"), opts.get("sigmas"))
    if step is None or max(h, w) <= rt.config.native:
        return ctx  # at/below the native extent AReA is a provable no-op
    # The runtime grid comes from the LATENT, not from the widget, so the
    # apply-time check cannot cover it: a grid that cannot be tiled exactly is
    # adjusted (never refused) and announced here — once per grid, so a widget
    # and a latent that agree say it once in total.
    factor, adjusted = resolve_factor(h, w, rt.config.native)
    if adjusted:
        # The pixel factor is the PROFILE's (16 px per token on every backbone
        # today), so a Krea-2 canvas is announced in its own units.
        px = int(rt.profile.px_per_patch) if rt.profile is not None else PATCH_PX
        _warn_grid_adjusted(h, w, w * px, h * px, factor)
    phase1 = _phase1_count(rt, opts)
    cap = None
    if is_short_schedule(total):
        # The "+2 healing rule" (area-fix2): an Euler step only displaces the
        # latent by 15-25% of its trajectory, so fewer than two fine steps
        # cannot overwrite the 32 px footprint Phase I bakes in — the ripple
        # would survive to the final step.  Cap the coarse prefix so at least
        # two fine steps always remain, and say so once.
        max_allowed_phase1 = max(1, total - 2)
        if phase1 > max_allowed_phase1:
            _warn_once(
                "area-heal-cap",
                "AReA-Z: phase1_steps (%d) capped to %d on a %d-step schedule. "
                "Euler trajectory dynamics require at least 2 fine steps to completely "
                "heal the coarse pooling footprint.",
                phase1, max_allowed_phase1, total,
            )
            phase1 = max_allowed_phase1
        cap = max_allowed_phase1
    cfg = rt.config
    schedule = _area_schedule(rt, h, w, total, phase1, ctx.txt_len)
    # The EFFECTIVE coarse-step count, announced once per schedule (Q10): the
    # knob, the cap and what the run really does, so a capped run is readable.
    _announce_effective_coarse(rt, _schedule_key(rt, h, w, total, phase1, ctx.txt_len),
                               total, cap, phase1)
    phase, size = schedule.at(step)
    ctx.phase, ctx.size, ctx.step, ctx.h, ctx.w = phase, size, step, h, w
    # The schedule length rides on the context so the layout builder can tell a
    # short (distilled) schedule from a long one without peeking into caches
    # (area-fix2 mechanism 3).
    ctx.total = int(total) if total is not None else 0
    # The Phase II ordinal (how many fine steps ran before this one) and how many
    # of its leading steps the bridge floor exempts from the cost cap.  The floor
    # is keyed on the RESOLVED schedule's geometry, so steps / sigma / bootstrap
    # all behave identically: a resolution that spends more steps coarse simply
    # reaches fewer of the floored fine steps, and a floored step is floored
    # whichever mode resolved the schedule.
    ctx.fine_step = step - len(schedule.factors)
    ctx.floor_steps = (cfg.bridge_floor_steps
                       if phase == "fine" and is_short_schedule(total) else 0)
    _warn_regime(total, phase1, phase, cfg.phase1_mode)
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
                    "latent over re-noised to sigma_star (plan-area-"
                    "zimage-bootstrap, step B2), or this run starts windowed "
                    "attention on a canvas composed for a different noise "
                    "level.",
                    first, rt.config.phase1_sigma,
                )
    if (phase == "coarse" and rt.profile is not ZIMAGE_PROFILE
            and rt.profile is not KREA_PROFILE and rt.profile is not QWEN21_PROFILE):
        # FLUX resolves its coarse tables here; the Z-Image ones need the model's
        # own rope groups, which only exist once the forward has run its embed,
        # so the seam resolves them lazily (``_zimage_coarse_ropes``).  The
        # Krea-2 ones are lazy too — and for a second reason: the model's own
        # table is the FIRST table of the forward (the recorder keeps it and
        # never overwrites it), so a resolve-time build through the same
        # embedder would be captured AS the model's table
        # (``_krea_coarse_ropes``).
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
        global _AREA_RUNTIME, _ZRUNTIME, _KRUNTIME, _QRUNTIME

        # One runtime per forward, and only the backbone this patcher resolved:
        # the four seams read four different globals, so none of them can fire
        # on another's forward.
        if rt.profile is ZIMAGE_PROFILE:
            _AREA_RUNTIME, _ZRUNTIME, _KRUNTIME, _QRUNTIME = None, rt, None, None
        elif rt.profile is KREA_PROFILE:
            _AREA_RUNTIME, _ZRUNTIME, _KRUNTIME, _QRUNTIME = None, None, rt, None
        elif rt.profile is QWEN21_PROFILE:
            _AREA_RUNTIME, _ZRUNTIME, _KRUNTIME, _QRUNTIME = None, None, None, rt
        else:
            _AREA_RUNTIME, _ZRUNTIME, _KRUNTIME, _QRUNTIME = rt, None, None, None
        try:
            _area_resolve_forward(rt, params)
            return _call_through(model_function, params)
        finally:
            _AREA_RUNTIME = None
            _ZRUNTIME = None
            _KRUNTIME = None
            _QRUNTIME = None

    return wrapper


def apply_area_to_model(
    model,
    model_type: str,
    width: int,
    height: int,
    enable_area: bool = True,
    phase1_steps: int = PHASE1_STEPS,
    pooling: str = "auto",
    halo: int = 8,
    bridge_ratio: float = 0.5,
    bridge_floor_steps: int = 2,
    window_start: int = 0,
    window_hold_steps: int = 0,
    delay: int = 0,
    seed: int = 0,
    phase1_mode: str = "auto",
    phase1_sigma: float = ZIMAGE_SIGMA_STAR,
):
    """Patch a ComfyUI FLUX, Z-Image or Krea-2 model with AReA (arXiv 2610.17227).

    The backbone is resolved once into an :class:`~src.area_arch._ArchProfile`
    (``src/area_arch.py``) and exactly one attention seam is installed for it —
    FLUX's bound ``attention``, Z-Image's bound ``optimized_attention_masked``
    or Krea-2's bound ``optimized_attention_masked``.  Every other architecture
    raises an actionable :class:`ValueError` (house rule,
    ``src/hap.py:876-895``), and Nunchaku (fused kernels bypass the seam) warns
    and returns the clone UNCHANGED (``src/hap.py:897-903``) — as does a
    Krea-2 model whose attention class does not resolve the seam symbol at all.
    At or below the native extent (64 patches = 1024 px on FLUX and Z-Image)
    nothing is patched — attention would be replaced by itself.

    ``phase1_mode`` / ``phase1_sigma`` are the Phase I -> II switch (Q4):
    ``"steps"`` is the reference's fixed ``phase1_steps``, and ``"sigma"`` counts
    the leading steps whose noise level is still at or above
    ``phase1_sigma`` (``theory.md:240-246``) so the switch follows the schedule.
    Both default to ``"auto"``, which resolves PER BACKBONE right after the profile is known
    (AReA-Z, ``.dev/area-debug/area-fix.txt`` §3-File 3): Z-Image runs the
    norm-preserving centroid pooling and the schedule-aware sigma handoff
    (``phase1_sigma``, default ``ZIMAGE_SIGMA_STAR``), FLUX keeps the reference's
    constants exactly — ``subsample`` pooling and ``steps`` mode with
    ``phase1_steps`` — so the FLUX default path is byte-identical to before.
    Krea-2 runs the norm-preserving centroid pooling (the reference's
    ``subsample`` leaves the pooled values on an f-stride lattice, and with the
    floor-divided block positions the block's queries collapse onto one rope
    position — the pack's measured Z-Image 32 px ripple, which the Krea-2 run
    reproduces) and the ``steps`` mode, so the user's ``phase1_steps`` knob
    stays authoritative.
    ``phase1_steps`` is clamped to ``[0, total]`` and ``0`` means "all Phase II"
    (the reference's ``config.phase1_steps``, ported without its
    ``phase1_steps=0`` assert).  ``pooling`` is one of :data:`POOLING`
    (``subsample`` / ``mean`` / ``medoid`` / ``random``; in ``mean`` mode the
    values are the plain block average and the keys the norm-preserving
    centroid, ``src/area_attn._norm_preserved_keys``), ``halo`` the widest
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
    if detected_type not in ("flux", "zimage", "krea2", "qwen21"):
        raise ValueError(
            f"AReA supports FLUX, Z-Image (Lumina2 / NextDiT), Krea-2 "
            f"(SingleStreamDiT) and Qwen-Image-2.1 "
            f"(QwenImage21Transformer2DModel) — it covered FLUX only in v1 — but "
            f"the resolved model type is {detected_type!r}. Load an unmodified "
            "(non-quantized) FLUX, Z-Image, Krea-2 or Qwen-Image-2.1 model, or "
            "use SPA / DyPE / SEGA for this architecture."
        )
    # The arch profile validates the pin against the model's own backbone and
    # refuses the Lumina2 siblings AReA cannot patch (pixel space, Ming-Image).
    profile = resolve_area_arch(dm, model_type)
    if profile is QWEN21_PROFILE:
        # N12 — 2.1's prefix K/V cache is ON by default, and the cached path
        # is length-indistinguishable from the plain image segment, so with
        # the cache on AReA would engage on no call at all.  The switch is a
        # HARD PRECONDITION of this adaptation (the design's H2).
        _qwen21_disable_prefix_cache(m)
    # The pixel multiple and the trained extent are the PROFILE's — they are
    # per-backbone facts (a 16x VAE would make a Krea-2 token 32 px wide).
    # FLUX and Z-Image keep 16 px and 64 tokens, so their numbers do not move.
    px_per_patch = int(profile.px_per_patch)
    native_tokens = int(profile.native_tokens)
    width = _snap_to_multiple(width, px_per_patch)
    height = _snap_to_multiple(height, px_per_patch)

    # AReA-Z defaults (.dev/area-debug/area-fix.txt §3-File 3): "auto" resolves per
    # backbone — Z-Image runs the norm-preserving centroid pooling and the
    # schedule-aware sigma handoff, FLUX keeps the reference's constants exactly.
    # Krea-2 joins Z-Image on the POOLING: "subsample" keeps one token of every
    # f x f block, so the pooled VALUES live on an f-stride lattice, and the
    # block's f^2 queries — which the reference's floor division put on ONE rope
    # position — then receive nearly the same output, i.e. a piecewise-constant
    # field with the block's period (32 px at f = 2).  That is the pack's
    # measured Z-Image ripple (area-fix.txt Flaw 3) and the Krea-2 run
    # reproduces it; the norm-preserving centroid keys and the averaged values
    # remove it (area-fix.txt mechanism 2).  FLUX keeps "subsample" — its path
    # is byte-frozen.
    if pooling == "auto":
        pooling = "subsample" if profile is FLUX_PROFILE else "mean"
    if phase1_mode == "auto":
        phase1_mode = "sigma" if profile is ZIMAGE_PROFILE else "steps"

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
        delay=max(0, min(int(phase1_steps), int(delay))),
        seed=int(seed),
        # D4: the trained extent in TOKENS is the profile's — FLUX and Z-Image
        # keep 64, so their numbers do not move.
        native=native_tokens,
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

    h, w = height // px_per_patch, width // px_per_patch
    if max(h, w) <= native_tokens:
        native_px = native_tokens * px_per_patch
        _warn_once(
            "native",
            "AReA: %dx%d px builds a %dx%d patch grid, at or below the native "
            "%d-patch (%d px) extent where AReA is a provable no-op — nothing "
            "is patched. Raise the resolution past %d px on the longer side "
            "(and keep this node's width/height in step with your latent) for "
            "AReA to engage.",
            width, height, w, h, native_tokens, native_px, native_px,
        )
        return m

    # GEOMETRY (schedule.py:39-48): the pooling ramp tiles the patch grid with
    # a common divisor >= 2; a grid with none (125x32, from 512x2000 px) has
    # no exact tiling, and used to fail here at apply time — and again, from
    # the latent, three steps into the sampling run.  The engine never raises
    # for a grid: the factor is resolved (the extent rule without the
    # divisibility constraint) and the last pooling block on a partial axis is
    # a partial block.  The canvas, the latent and the output resolution are
    # untouched; the user is told once.
    factor, adjusted = resolve_factor(h, w, native_tokens)
    if adjusted:
        _warn_grid_adjusted(h, w, width, height, factor)

    installed = _area_install_seam(m, dm, profile)
    if installed is False:
        # A quantized / fused attention kernel bypasses the seam: AReA would be
        # a silent no-op, so the model is returned UNCHANGED (the Nunchaku
        # precedent above).
        m._area_config = None
        return m

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
        w, h, width, height,
        {"flux": "FLUX", "zimage": "Z-Image", "krea2": "Krea-2",
         "qwen21": "Qwen-Image-2.1"}.get(
            profile.model_type, "FLUX"),
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
    ``src/spa.py:1485``.  FLUX, Z-Image and Krea-2 have separate ref-counts, so
    the patcher's own seam (``m._area_seam``) is the one released.
    """
    if not getattr(m, "_area_seam_consumer", False):
        m._area_config = None
        return
    m._area_seam_consumer = False
    seam = getattr(m, "_area_seam", None)
    if seam == "zimage":
        _release_seam(_ZSEAM)
    elif seam == "krea2":
        _release_seam(_KSEAM)
    elif seam == "qwen21":
        _release_seam(_QSEAM)
    else:
        _release_seam(_SEAM)
    m._area_seam = None
    m._area_config = None
