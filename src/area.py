"""AReA — training-free, time-dependent replacement of FLUX image self-attention.

This module is the model-side wiring (the reference's ``area/backbones/flux.py``
and ``area/backbones/base.py``): it patches a ComfyUI FLUX model so that every
block's attention runs the two AReA phases built from :mod:`src.area_schedule`
and :mod:`src.area_attn` instead of dense attention.  The node
(``nodes/area.py``) is a thin schema/execute shell over
:func:`apply_area_to_model`.

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
)
from .area_schedule import AreaSchedule, build_schedule, initial_factor
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


@dataclass(frozen=True)
class AreaConfig:
    """Everything AReA needs that is not resolved per forward."""

    phase1_steps: int = PHASE1_STEPS
    pooling: str = "subsample"
    halo: int = 8
    bridge_ratio: float = 0.5
    seed: int = 0
    native: int = NATIVE_PATCHES


class _AreaRuntime:
    """One patcher's AReA state plus the per-forward per-call caches.

    ``dm`` is the diffusion model the patcher patches (resolved once at apply
    time so the per-forward rope lookup sees the *installed* embedder — comfy
    applies ``object_patches`` in place around model use, so a chained DyPE/SEGA
    embedder is visible at forward time).

    Per forward: ``ctx`` is the current step (``None`` phase = plain attention,
    delegated untouched), ``hidden`` is the ``img_in`` output of that forward,
    and ``block`` counts attention calls so the reference's per-block ``random``
    pooling stream is reproduced (double-stream blocks 0..N then single-stream,
    the same order as the reference's
    ``list(transformer_blocks) + list(single_transformer_blocks)``).
    ``ropes`` / ``schedules`` are per-canvas caches (the reference's
    ``self._ropes`` / ``self._schedule``).
    """

    __slots__ = ("config", "dm", "ctx", "hidden", "ropes", "schedules", "block")

    def __init__(self, config: AreaConfig, dm):
        self.config = config
        self.dm = dm
        self.ctx: Optional[AreaContext] = None
        self.hidden: Optional[torch.Tensor] = None
        self.ropes: dict = {}
        self.schedules: dict = {}
        self.block = 0


#: The runtime of the forward currently running, read by the attention seam.  Set
#: and cleared around every forward (SPA discipline) — ``None`` between forwards,
#: which is what makes an unpatched/inactive model a pure no-op.
_AREA_RUNTIME: Optional[_AreaRuntime] = None

# Seam install bookkeeping: the patched module, the original symbol, the
# ``img_in`` hook handle and the number of patchers consuming them.  A second
# AReA apply re-uses the wrapper instead of wrapping our own wrapper, and the
# symbol is restored only when the last consumer unpatches.
_SEAM = {"module": None, "attr": None, "orig": None, "hook": None, "consumers": 0}


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


def _area_schedule(rt: _AreaRuntime, h: int, w: int, total: int, txt_len: int) -> AreaSchedule:
    """The schedule of one canvas, cached per ``(h, w, total, phase1, txt_len)``.

    The reference caches on exactly this key (``backbones/base.py:64-66``): the
    windows of a step depend on the canvas and on how many text tokens the
    prompt carries, both constant across a generation.
    """
    cfg = rt.config
    key = (h, w, total, cfg.phase1_steps, txt_len)
    schedule = rt.schedules.get(key)
    if schedule is None:
        schedule = build_schedule(h, w, total, cfg.phase1_steps, cfg.native, txt_len=txt_len)
        rt.schedules[key] = schedule
    return schedule


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
    if hasattr(embedder, "set_timestep"):
        # Takeover warning (the I-Max pattern, nodes/imax.py:287-294): a
        # DyPE-family embedder carries dynamic, timestep-dependent scaling.
        _warn_once(
            "dynamic-embedder",
            "AReA: the installed positional embedder is a dynamic scaling "
            "wrapper (%s) — AReA's coarse-grid rope positions now ride on it, so "
            "its per-step scaling also moves the pooled keys. Chain only one of "
            "DyPE / I-Max / SEGA positional scaling.",
            type(embedder).__name__,
        )
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


def _area_fine_layout(rt: _AreaRuntime, ctx: AreaContext, k_img: torch.Tensor) -> Optional[FineLayout]:
    """Build (once per step) the Phase II layout — the reference's ``_after_embedding``."""
    layout = ctx.data.get("layout")
    if layout is not None:
        return layout
    cfg = rt.config
    h, w, t = ctx.h, ctx.w, ctx.txt_len
    hidden = rt.hidden
    if hidden is None:
        # Fallback: the detail map is derived from the post-norm projected image
        # keys instead of the img_in output (see the module docstring).
        _warn_once(
            "no-img-in",
            "AReA: diffusion_model.img_in could not be hooked, so the Phase II "
            "detail map is derived from the projected image keys of the first "
            "fine call. Results differ slightly from a hooked model.",
        )
        # [b, heads, n, d] -> [b, n, heads*d], the img_in output's layout
        hidden = k_img.transpose(1, 2).reshape(*k_img.shape[:1], k_img.shape[2], -1)
    energy = detail_map(hidden, h, w)
    w_min = min(cfg.native, max(h, w))
    windows = adaptive_tiling(energy, h, w, ctx.size, w_min, t, rng=tiling_rng(cfg.seed, ctx.step))
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


def _area_img_in_hook(module, args, output):
    """Capture the ``img_in`` output of the running forward (the reference's
    ``x_embedder`` hook, ``backbones/flux.py:150-163``)."""
    rt = _AREA_RUNTIME
    if rt is None or rt.ctx is None or rt.ctx.phase != "fine":
        return
    rt.hidden = output


def _area_install_seam(m, dm) -> None:
    """Patch ``comfy.ldm.flux.layers.attention`` and hook ``img_in`` (ref-counted)."""
    import importlib

    if _SEAM["orig"] is None:
        try:
            mod = importlib.import_module(FLUX_ATTENTION_MODULE)
        except Exception as exc:  # probe: backend module not importable here
            raise ValueError(
                f"AReA: cannot import {FLUX_ATTENTION_MODULE} to patch the FLUX "
                f"attention seam ({exc}). AReA supports FLUX only; use an "
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
            _SEAM["hook"] = img_in.register_forward_hook(_area_img_in_hook)
        else:
            _warn_once(
                "no-img-in",
                "AReA: diffusion_model.img_in could not be hooked, so the Phase "
                "II detail map is derived from the projected image keys of the "
                "first fine call. Results differ slightly from a hooked model.",
            )
    _SEAM["consumers"] += 1
    m._area_seam_consumer = True


def _area_resolve_forward(rt: _AreaRuntime, params: dict) -> AreaContext:
    """Resolve one forward's step, geometry, ropes and phase into a context."""
    ctx = AreaContext()
    rt.ctx = ctx
    rt.block = 0
    rt.hidden = None
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
    phase, size = _area_schedule(rt, h, w, total, ctx.txt_len).at(step)
    ctx.phase, ctx.size, ctx.step, ctx.h, ctx.w = phase, size, step, h, w
    if phase == "coarse":
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
        global _AREA_RUNTIME

        _AREA_RUNTIME = rt
        try:
            _area_resolve_forward(rt, params)
            return _call_through(model_function, params)
        finally:
            _AREA_RUNTIME = None

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
    seed: int = 0,
):
    """Patch a ComfyUI FLUX model with AReA (arXiv 2610.17227).

    FLUX only, v1: every other architecture raises an actionable
    :class:`ValueError` (house rule, ``src/hap.py:876-895``), and Nunchaku (fused
    kernels bypass the seam) warns and returns the clone UNCHANGED
    (``src/hap.py:897-903``).  At or below the native 64-patch extent (1024 px)
    nothing is patched — attention would be replaced by itself.

    ``phase1_steps`` is clamped to ``[0, total]`` and ``0`` means "all Phase II"
    (the reference's ``config.phase1_steps``, ported without its
    ``phase1_steps=0`` assert).  ``pooling`` is one of :data:`POOLING`
    (``subsample`` / ``mean`` / ``medoid`` / ``random``), ``halo`` the widest
    Halo ring, ``bridge_ratio`` rho of ``R = round(rho * w)^2`` bridge tokens per
    window, and ``seed`` the key of every draw (windows, halos, random pooling).

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
            "bypass the FLUX attention seam); returning the model UNCHANGED. "
            "Use a non-quantized FLUX loader.",
        )
        m._area_config = None
        return m
    if detected_type != "flux":
        raise ValueError(
            f"AReA supports FLUX only in v1, but the resolved model type is "
            f"{detected_type!r}. Load an unmodified (non-quantized) FLUX model, "
            "or use SPA / DyPE / SEGA for this architecture."
        )

    m._area_config = AreaConfig(
        phase1_steps=max(0, int(phase1_steps)),
        pooling=pooling if pooling in POOLING else "subsample",
        halo=max(0, int(halo)),
        bridge_ratio=float(bridge_ratio),
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
    # would otherwise fail three steps into a sampling run.  Fail here, at
    # apply time, with the reference's actionable message.
    initial_factor(h, w, NATIVE_PATCHES)

    _area_install_seam(m, dm)

    previous_wrapper = None
    try:
        previous_wrapper = m.model_options.get("model_function_wrapper")
    except AttributeError:  # probe: mock patchers without model_options
        previous_wrapper = getattr(m, "_unet_wrapper", None)
    m.set_model_unet_function_wrapper(
        _make_area_unet_wrapper(_AreaRuntime(m._area_config, dm), previous_wrapper)
    )
    logger.info(
        "AReA: installed on a %dx%d patch grid (%dx%d px), phase1_steps=%d, pooling=%s, "
        "halo=%d, bridge_ratio=%s, seed=%d",
        w, h, width, height, m._area_config.phase1_steps, m._area_config.pooling,
        m._area_config.halo, m._area_config.bridge_ratio, m._area_config.seed,
    )
    return m


def restore_area_attention_hook(m) -> None:
    """Undo an AReA patch recorded on ``m`` (idempotent, consumer-aware).

    The seam symbol and the ``img_in`` hook are shared by every patcher, so they
    are restored only when the last consumer leaves — mirroring
    ``src/spa.py:1485``.
    """
    if not getattr(m, "_area_seam_consumer", False):
        m._area_config = None
        return
    m._area_seam_consumer = False
    _SEAM["consumers"] = max(0, _SEAM["consumers"] - 1)
    if _SEAM["consumers"] == 0:
        handle, _SEAM["hook"] = _SEAM["hook"], None
        if handle is not None:
            try:
                handle.remove()
            except Exception:  # probe: handle already gone with its module
                pass
        if _SEAM["orig"] is not None:
            mod, attr, orig = _SEAM["module"], _SEAM["attr"], _SEAM["orig"]
            if getattr(mod, attr, None) is not orig:
                setattr(mod, attr, orig)
            _SEAM.update(module=None, attr=None, orig=None)
    m._area_config = None
