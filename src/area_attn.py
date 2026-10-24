"""AReA attention phases — port of the reference's ``area/attention.py`` + ``area/bridge.py``
and the tensor parts of ``area/tiling.py``.

ComfyUI FLUX reaches the attention seam as ``[b, heads, T + n, d]``
(``comfy/ldm/flux/layers.py:229-244``) while the reference works in diffusers'
``[batch, sequence, heads, head_dim]``, so every sequence axis here is axis 2 and
heads live on axis 1.  Boxes, token indices and merge weights are layout-agnostic
and come from :mod:`src.area_schedule` (Phase 1); this module adds only what
produces or consumes tensors, and owns nothing about the model itself.

The two phases, in the seam's terms:

* **Phase I (coarse)** — :func:`pool_keys_values` keeps one pooled key/value per
  ``f x f`` block of the image tokens (text is never pooled), so queries keep full
  resolution while the key/value canvas shrinks to ``(h/f)(w/f)``.  The two rope
  tables — full grid for q, coarse grid for k/v — are applied by the caller with
  :func:`apply_rope1` (the reference's ``backbones/flux.py:180-190``).
  A patch grid whose counts share no factor >= 2 has no exact tiling; there the
  pooled canvas is the ceil-based grid and the last block on a partial axis is a
  *partial* block, pooled over its real members only (the canvas and the token
  stream are untouched either way).
* **Phase II (fine)** — :func:`build_fine_layout` turns one step's window tiling
  into the grouped token/halo/weight buffers, and :func:`windowed_attention` runs
  each group as one ``F.scaled_dot_product_attention`` call on ``[b*g, heads,
  T+n, d]`` with the keys extended by that group's Halo ring and bridge tokens.

Not here: the per-model schedule, the seam that calls these functions, and the
node — ``src/area.py`` and ``nodes/area.py``, Phase 3.

Deviations from the reference, all deliberate and pinned by the Phase 2 tests:

* Layout is comfy's ``[b, heads, N, d]`` instead of ``[B, N, heads, d]``: every
  ``key[:, txt_len:]`` becomes ``key[:, :, txt_len:]`` and every pooled canvas is
  returned heads-first.
* ``dispatch_attention_fn`` (a diffusers import, and diffusers is not a
  dependency) is replaced by ``torch.nn.functional.scaled_dot_product_attention``
  on the grouped call — deterministic, and it lets the backend pick its own kernel.
* RoPE is applied by :func:`apply_rope1`, a local port of comfy's own
  ``ldm/flux/math.py:36-49`` (``_apply_rope1``) so this engine needs no runtime
  comfy import.  It takes the table the installed ``pe_embedder`` returns — ``[b, 1,
  N, d/2, 2, 2]``, head-shared (``ldm/flux/model.py:17-27``) — so a chained
  DyPE/SEGA embedder composes; the tables themselves are resolved per forward by
  the caller (Phase 3), which is also why the coarse q/k tables differ.
* The grid is the model's token grid: patches in the reference (``f`` blocks of
  patches) and the packed 2x2-patch token grid in comfy FLUX
  (``ldm/flux/model.py:318``), so ``h*w`` is the image sequence length either way.
* Comfy batches cond and uncond into ONE forward, so the reference's batch
  reductions are kept exactly as shipped — :func:`detail_map` means over the batch
  and :func:`token_importance` sums over it — and one layout serves the whole
  batch: both CFG branches attend through identical graphs (theory.md:496).
* :func:`windowed_step_cost` exposes the pricing of :func:`fit_extra_keys`
  without applying it, for the engine's bridge-floor exemption (which deliberately
  runs above the cap on a short schedule's leading fine steps).  ``fit_extra_keys``
  itself is untouched, so the cost-capped path is byte-identical.  Both live in
  :mod:`src.area_schedule` now — the Phase II COST MODEL is pure schedule math,
  and the Phase II side choice has to price a window before this module's
  tensors exist.  They are re-exported here under the same names, so every
  call site (and every test) keeps working; the import graph runs ONE way only
  (``area_attn`` imports FROM ``area_schedule``, never the reverse).

Determinism: nothing here reads the global RNG.  Every stream is a
``numpy.random.default_rng`` keyed by ``(seed, step, salt)`` through
:func:`src.area_schedule.step_rng`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .area_schedule import (
    Window,
    _dedupe,
    _grown_box,
    bridge_registers,
    coarse_grid,
    even_starts,
    fit_extra_keys,
    ring_size,
    step_rng,
    window_tokens,
    windowed_step_cost,
)

POOLING = ("subsample", "mean", "medoid", "random")

# tiling constants (reference tiling.py:12-18)
DWT_LEVELS = 3
DWT_PAD = 8
GATE_LO, GATE_HI = 0.05, 0.25
SPLIT_QUANTILE = 0.5
BLEND_FLOOR = 0.05
_CHUNK_TOKENS = 8192

# bridge and pooling chunking (reference bridge.py:5-6, attention.py:17)
CELLS = 8
_CHUNK_ELEMS = 8_000_000
_MEDOID_CHUNK_ELEMS = 8_000_000

# salts of the keyed streams: the split draw belongs to the backbone, the halo widths to the
# layout builder, the random pooling to one (seed, step, block); all are keyed by the step so a
# layout cannot drift between calls (theory.md:496)
TILING_SALT = 0x5EA1        # adaptive splits (reference backbones/flux.py:158)
HALO_SALT = 0x4A10          # halo widths and reaches (reference attention.py:145)
_POOL_SALT = 0xC0DA         # random pooling (reference attention.py:31)


# ---------------------------------------------------------------------------
# Per-step state (reference backbones/base.py:12-21)
# ---------------------------------------------------------------------------


@dataclass
class AreaContext:
    """The current denoising step of one forward, read by every attention call.

    ``phase`` is ``None`` for a plain-attention step (at or below the native
    extent), ``"coarse"`` for Phase I and ``"fine"`` for Phase II; ``size`` is the
    pooling factor ``f`` (Phase I) or the window side ``w`` (Phase II).  ``data``
    carries the payload the caller resolved for that phase: ``rope_q``/``rope_kv``
    for a coarse step, a :class:`FineLayout` under ``"layout"`` for a fine one.

    ``fine_step`` is the step's ordinal WITHIN Phase II (``0`` on the first fine
    step, negative on a coarse one) and ``floor_steps`` how many of those leading
    steps the bridge floor exempts from the cost cap (``0`` off a short schedule,
    or when the floor is disabled) — the pair the layout builder compares to
    decide whether this step keeps the paper's full bridge (``src/area.py``).

    Phase 3 sets and clears this around every forward (the SPA discipline,
    ``src/spa.py:1385-1476``) so no layout may leak across models or steps.
    """

    phase: Optional[str] = None
    step: int = 0
    size: int = 0
    h: int = 0
    w: int = 0
    txt_len: int = 0
    #: ordinal within Phase II — how many fine steps ran before this one
    fine_step: int = 0
    #: how many leading fine steps the bridge floor covers (0 = no floor)
    floor_steps: int = 0
    #: the schedule's total step count, so the layout builder can tell a short
    #: (distilled) schedule from a long one without peeking into caches
    #: (``0`` when a context is built by hand)
    total: int = 0
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase I: cross-scale coarse attention
# ---------------------------------------------------------------------------


def _partial_axis(h: int, w: int, f: int) -> bool:
    """Whether the ``f x f`` pooling blocks do not tile the patch grid exactly.

    The reference's Eq. 5 refuses a grid whose patch counts share no factor
    >= 2, so on such a grid the blocks cannot tile it: the last block on a
    partial axis owns fewer than ``f`` rows (or columns) and the cells it does
    not own are *padding*, never canvas.  Every grid that tiles keeps the
    reference's path, bit for bit — this predicate is the guard, not a rewrite.
    """
    f = max(1, int(f))
    return int(h) % f != 0 or int(w) % f != 0


def _block_mask(h: int, w: int, f: int, device) -> torch.Tensor:
    """``[nh, nw, f, f]`` — 1 on the REAL cells of every block, 0 on the padded ones.

    Block ``r`` owns rows ``[r*f, min(r*f+f-1, h-1)]`` and block ``c`` owns
    columns ``[c*f, min(c*f+f-1, w-1)]``, so the blocks together cover every
    row and every column of the canvas exactly once: no token is dropped and
    none is pooled twice.  The mask is what turns a partial block's mean into
    a mean over its REAL members only — an unmasked mean over the padded cells
    would shrink that block's key norm and bias it.
    """
    nh, nw = coarse_grid(h, w, f)
    rows = torch.arange(nh * f, device=device) < int(h)
    cols = torch.arange(nw * f, device=device) < int(w)
    mask = (rows[:, None] & cols[None, :]).reshape(nh, f, nw, f)
    return mask.permute(0, 2, 1, 3).float()


def _block_selection(k_img: torch.Tensor, h: int, w: int, f: int, pooling: str,
                     seed: int, step: int, block: int) -> Optional[torch.Tensor]:
    """Index in ``[0, f*f)`` of the token that represents each block, for the selecting poolings.

    ``k_img`` is ``[b, heads, h*w, d]``; the result is ``[b, h/f, w/f]`` and is
    shared by keys and values, so K and V always come from the same token.

    On a grid the blocks do not tile (a partial edge block) the index is a
    cell of the ``f x f`` block, mapped onto the block's real extent by
    :func:`_pool_partial` — the padding cells are never selectable.
    """
    if _partial_axis(h, w, f):
        return _block_selection_partial(k_img, h, w, f, pooling, seed, step, block)
    b, heads, _, dim = k_img.shape
    nh, nw = h // f, w // f
    if pooling == "random":
        mix = (int(seed) & 0x7FFFFFFF) + int(step) * 2_654_435_761 + int(block) * 40_503 + _POOL_SALT
        sel = np.random.default_rng(mix).integers(0, f * f, size=(nh, nw), dtype=np.int64)
        return torch.from_numpy(sel).to(k_img.device).unsqueeze(0).expand(b, nh, nw).contiguous()
    if pooling == "medoid":
        grid = k_img.reshape(b, heads, h, w, dim)
        rows_per_chunk = max(1, _MEDOID_CHUNK_ELEMS // max(1, nw * f * f * heads * dim))
        parts = []
        for r0 in range(0, nh, rows_per_chunk):
            r1 = min(nh, r0 + rows_per_chunk)
            kb = (grid[:, :, r0 * f : r1 * f]
                  .reshape(b, heads, r1 - r0, f, nw, f, dim)
                  .permute(0, 2, 4, 3, 5, 1, 6)
                  .reshape(b, r1 - r0, nw, f * f, heads, dim)
                  .float())
            parts.append((kb - kb.mean(dim=3, keepdim=True)).pow(2).sum(dim=(-2, -1)).argmin(dim=3))
        return torch.cat(parts, dim=1)
    return None


def _block_selection_partial(k_img: torch.Tensor, h: int, w: int, f: int, pooling: str,
                              seed: int, step: int, block: int) -> Optional[torch.Tensor]:
    """The selecting poolings on a grid the ``f x f`` blocks do not tile exactly.

    Same contract as :func:`_block_selection` — one index in ``[0, f*f)`` per
    block, shared by keys and values — but the block's mean is the mean over
    its REAL members and the padded cells are masked out of the ``medoid``
    argmin, so a partial block can never select a padding cell (which would
    pool a zero vector).  The ``random`` draw keeps its ``(seed, step, block)``
    key, so determinism is unchanged.
    """
    b, heads, _, dim = k_img.shape
    nh, nw = coarse_grid(h, w, f)
    if pooling == "random":
        mix = (int(seed) & 0x7FFFFFFF) + int(step) * 2_654_435_761 + int(block) * 40_503 + _POOL_SALT
        sel = np.random.default_rng(mix).integers(0, f * f, size=(nh, nw), dtype=np.int64)
        return torch.from_numpy(sel).to(k_img.device).unsqueeze(0).expand(b, nh, nw).contiguous()
    if pooling == "medoid":
        mask = _block_mask(h, w, f, k_img.device).reshape(nh, nw, f * f)
        grid = k_img.reshape(b, heads, h, w, dim)
        rows_per_chunk = max(1, _MEDOID_CHUNK_ELEMS // max(1, nw * f * f * heads * dim))
        parts = []
        for r0 in range(0, nh, rows_per_chunk):
            r1 = min(nh, r0 + rows_per_chunk)
            nrows = r1 - r0
            real = min(r1 * f, h) - r0 * f
            kb = (F.pad(grid[:, :, r0 * f : min(r1 * f, h)], (0, 0, 0, nw * f - w, 0, nrows * f - real))
                  .reshape(b, heads, nrows, f, nw, f, dim)
                  .permute(0, 1, 2, 4, 3, 5, 6)
                  .reshape(b, heads, nrows, nw, f * f, dim)
                  .float())
            real_cells = mask[r0:r1][None, None, :, :, :, None]
            count = real_cells.sum(dim=4).clamp(min=1.0)
            mean = (kb * real_cells).sum(dim=4) / count
            dist = (kb - mean[:, :, :, :, None, :]).pow(2).sum(dim=-1)
            dist = dist.masked_fill(real_cells.squeeze(-1) == 0, float("inf"))
            parts.append(dist.sum(dim=1).argmin(dim=-1))
        return torch.cat(parts, dim=1)
    return None


def _pool(x_img: torch.Tensor, h: int, w: int, f: int, pooling: str,
          sel: Optional[torch.Tensor]) -> torch.Tensor:
    """One token per ``f x f`` block of the image tokens, returned ``[b, heads, h*w/f^2, d]``."""
    if _partial_axis(h, w, f):
        return _pool_partial(x_img, h, w, f, pooling, sel)
    b, heads, _, dim = x_img.shape
    nh, nw = h // f, w // f
    grid = x_img.reshape(b, heads, h, w, dim)
    if pooling == "subsample":
        off = f // 2
        pooled = grid[:, :, off::f, off::f][:, :, :nh, :nw]
    elif pooling == "mean":
        pooled = grid.reshape(b, heads, nh, f, nw, f, dim).mean(dim=(3, 5))
    else:
        blocks = grid.reshape(b, heads, nh, f, nw, f, dim).permute(0, 1, 2, 4, 3, 5, 6)
        idx = sel[:, None, :, :, None, None].expand(b, heads, nh, nw, 1, dim)
        pooled = blocks.reshape(b, heads, nh, nw, f * f, dim).gather(4, idx).squeeze(4)
    return pooled.reshape(b, heads, nh * nw, dim)


def _pool_partial(x_img: torch.Tensor, h: int, w: int, f: int, pooling: str,
                  sel: Optional[torch.Tensor]) -> torch.Tensor:
    """One token per block on a grid the blocks do not tile exactly.

    ``subsample`` takes the block centre clamped to the block's real extent,
    ``mean`` averages over the block's REAL members only (the padded cells are
    excluded, so a partial block is not shrunk by its own padding), and the
    selecting poolings take the cell ``sel`` names, mapped onto the block's
    real extent — a padded cell is never pooled.
    """
    b, heads, _, dim = x_img.shape
    nh, nw = coarse_grid(h, w, f)
    if pooling == "subsample":
        off = f // 2
        rows = (torch.arange(nh, device=x_img.device) * f + off).clamp(max=h - 1)
        cols = (torch.arange(nw, device=x_img.device) * f + off).clamp(max=w - 1)
        return x_img.index_select(2, (rows[:, None] * w + cols[None, :]).reshape(-1))
    if pooling == "mean":
        mask = _block_mask(h, w, f, x_img.device)
        count = mask.sum(dim=(-2, -1)).clamp(min=1.0)
        grid = x_img.reshape(b, heads, h, w, dim)
        padded = F.pad(grid, (0, 0, 0, nw * f - w, 0, nh * f - h))
        blocks = padded.reshape(b, heads, nh, f, nw, f, dim).float()
        total = (blocks * mask.permute(0, 2, 1, 3)[None, None, :, :, :, :, None]).sum(dim=(3, 5))
        return (total / count[None, None, :, :, None]).reshape(b, heads, nh * nw, dim).to(x_img.dtype)
    # medoid / random: the selection names a cell of the f x f block, mapped onto its real extent
    row = (torch.arange(nh, device=x_img.device).view(1, nh, 1) * f + sel // f).clamp(max=h - 1)
    col = (torch.arange(nw, device=x_img.device).view(1, 1, nw) * f + sel % f).clamp(max=w - 1)
    idx = (row * w + col).reshape(b, 1, nh * nw, 1).expand(b, heads, nh * nw, dim)
    return x_img.gather(2, idx)


def _norm_preserved_keys(k_img: torch.Tensor, h: int, w: int, f: int) -> torch.Tensor:
    """The ``mean`` pooling's keys: the block-mean DIRECTION at the mean MEMBER NORMS.

    ``subsample`` pooling discards ``f^2 - 1`` of every block's tokens, so the
    survivors' values form a Dirac comb on the ``f``-stride grid and drive the
    latent toward a piecewise-constant mosaic (Flaw 3).  Averaging the block
    removes the comb but also shrinks the key norm by roughly ``sqrt(f^2)``,
    flattening the attention temperature.  This keeps the centroid direction —
    every token of the block still contributes equally — and rescales it to the
    mean of the block's member norms, so the softmax scale stays the un-pooled
    one.  ``f == 1`` returns the keys untouched, bit for bit.
    """
    if _partial_axis(h, w, f):
        return _norm_preserved_keys_partial(k_img, h, w, f)
    b, heads, _, dim = k_img.shape
    nh, nw = h // f, w // f
    if f == 1:
        return k_img
    grid = k_img.reshape(b, heads, nh, f, nw, f, dim).float()
    centroid = grid.mean(dim=(3, 5))
    norms = grid.norm(dim=-1).mean(dim=(3, 5))
    pooled = centroid / (centroid.norm(dim=-1, keepdim=True) + 1e-8) * norms[..., None]
    return pooled.reshape(b, heads, nh * nw, dim).to(k_img.dtype)


def _norm_preserved_keys_partial(k_img: torch.Tensor, h: int, w: int, f: int) -> torch.Tensor:
    """The norm-preserving keys on a grid the blocks do not tile exactly.

    The centroid is the mean over the block's REAL members and the norm is the
    mean of the REAL members' norms, so a partial block keeps the un-pooled
    temperature instead of the shrunk one an unmasked mean would give it.
    """
    b, heads, _, dim = k_img.shape
    nh, nw = coarse_grid(h, w, f)
    mask = _block_mask(h, w, f, k_img.device)
    count = mask.sum(dim=(-2, -1)).clamp(min=1.0)
    grid = k_img.reshape(b, heads, h, w, dim)
    padded = F.pad(grid, (0, 0, 0, nw * f - w, 0, nh * f - h))
    blocks = padded.reshape(b, heads, nh, f, nw, f, dim).float()
    cells = mask.permute(0, 2, 1, 3)[None, None, :, :, :, :, None]
    centroid = (blocks * cells).sum(dim=(3, 5)) / count[None, None, :, :, None]
    norms = (blocks.norm(dim=-1) * mask.permute(0, 2, 1, 3)[None, None]).sum(dim=(3, 5))
    norms = norms / count[None, None]
    pooled = centroid / (centroid.norm(dim=-1, keepdim=True) + 1e-8) * norms[..., None]
    return pooled.reshape(b, heads, nh * nw, dim).to(k_img.dtype)


def pool_keys_values(key: torch.Tensor, value: torch.Tensor, txt_len: int, h: int, w: int, f: int,
                     pooling: str, seed: int = 0, step: int = 0, block: int = 0
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep one token per ``f x f`` block of the image keys and values (text is never pooled).

    ``subsample`` keeps the block centre, ``mean`` averages the VALUES and keeps
    the norm-preserving CENTROID keys (:func:`_norm_preserved_keys` — the block-mean
    direction at the mean member norm, so neither a Dirac comb nor a flattened
    temperature; the centroid-key mechanism), ``medoid`` keeps the
    token whose key is closest to the block-mean key, and ``random`` keeps a token
    drawn per ``(seed, step, block)``.  Keys must be unrotated — pooling after
    RoPE would rotate the survivors away from the query's frame — and K and V
    always come from the same token for the selecting poolings.

    On a grid the blocks do not tile (a patch grid with no common divisor
    >= 2) the pooled canvas is the ceil-based one and the last block on a
    partial axis is a *partial* block: it owns fewer than ``f`` rows or
    columns, is pooled over its real members only, and never contributes a
    padded cell to the keys or the values.
    """
    k_img, v_img = key[:, :, txt_len:], value[:, :, txt_len:]
    sel = _block_selection(k_img, h, w, f, pooling, seed, step, block)
    k_pooled = (_norm_preserved_keys(k_img, h, w, f) if pooling == "mean"
                else _pool(k_img, h, w, f, pooling, sel))
    k = torch.cat([key[:, :, :txt_len], k_pooled], dim=2)
    v = torch.cat([value[:, :, :txt_len], _pool(v_img, h, w, f, pooling, sel)], dim=2)
    return k, v


def apply_rope1(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` ``[b, heads, N, d]`` by a head-shared table ``[b, 1, N, d//2, 2, 2]``.

    Local port of comfy's ``_apply_rope1`` (``ldm/flux/math.py:36-49``) so the
    engine imports no comfy module at runtime: comfy's own ``apply_rope1``
    dispatches to ``comfy.quant_ops.ck.apply_rope`` outside training, while this
    is the same math in the branch every environment can run.  The math is done in
    the table's dtype (comfy builds it in float32) and cast back, exactly as comfy.
    """
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    if x_.shape[2] != 1 and freqs_cis.shape[2] != 1 and x_.shape[2] != freqs_cis.shape[2]:
        freqs_cis = freqs_cis[:, :, :x_.shape[2]]
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out.addcmul_(freqs_cis[..., 1], x_[..., 1])
    return x_out.reshape(*x.shape).type_as(x)


# ---------------------------------------------------------------------------
# Phase II: the windowed layout
# ---------------------------------------------------------------------------


@dataclass
class WindowGroup:
    """Windows of one step that share a token count and a Halo size, run as one batched call."""

    tokens: torch.Tensor                 # [G, n]
    halo: Optional[torch.Tensor]         # [G, n_halo]
    weights: torch.Tensor                # [G * n] merge weights
    #: indices into :attr:`FineLayout.windows`, one per member.  ``None`` when a
    #: hand-built group does not declare the mapping — the mapping is then
    #: recovered from the group's own token rows (see ``_group_window_ids``).
    window_ids: Optional[torch.Tensor] = None      # [G]


@dataclass
class FineLayout:
    """Everything Phase II needs for one denoising step, shared by all transformer blocks."""

    h: int
    w: int
    windows: list[Window]
    groups: list[WindowGroup]
    coverage: torch.Tensor               # [N] number of windows covering each token
    registers: int                       # R bridge tokens per window (0 = no bridge)


def tiling_rng(seed: int, step: int) -> np.random.Generator:
    """The keyed stream of one step's adaptive splits (the backbone's salt)."""
    return step_rng(seed, step, TILING_SALT)


def build_fine_layout(windows: list[Window], h: int, w: int, halo: int, registers: int,
                      seed: int, step: int, dtype: torch.dtype, device) -> FineLayout:
    """Group one step's windows by token count and Halo size, and build their merge weights."""
    n_img = h * w
    tokens = [window_tokens(win, w, device) for win in windows]
    coverage = torch.zeros(n_img, dtype=dtype, device=device)
    for idx in tokens:
        coverage[idx] += 1.0
    # every side of every window draws its own Halo width in [0, halo] fresh at each step, and
    # every row or column along that side its own reach; the ring keeps the straight ring's size
    rng = step_rng(seed, step, HALO_SALT) if halo > 0 else None
    rings = [jagged_halo_ring(win, tuple(int(d) for d in rng.integers(0, halo + 1, size=4)), halo, h, w,
                              rng, device)
             if rng is not None else None for win in windows]
    weights = merge_weights(windows, tokens, n_img, dtype, device)
    by_key: dict = {}
    for i, idx in enumerate(tokens):
        ring = rings[i]
        by_key.setdefault((int(idx.numel()), 0 if ring is None else int(ring.numel())), []).append(i)
    groups = [
        WindowGroup(
            tokens=torch.stack([tokens[i] for i in members]),
            halo=torch.stack([rings[i] for i in members]) if ring_len else None,
            weights=torch.cat([weights[i] for i in members]),
            window_ids=torch.tensor(members, dtype=torch.long, device=device),
        )
        for (_, ring_len), members in by_key.items()
    ]
    return FineLayout(h, w, windows, groups, coverage, registers)


# The rope horizon of Z-Image's trained extent: rope theta=256 makes the base rotary
# frequency 1.0 rad per token, so a bridge hop of the canvas' own diameter (100+ tokens
# at 2048 px) wraps the rotation ~16 times and turns the query-key dot product into
# pseudorandom noise — distant features leak into empty regions and hallucinate there
# (the horizon-saturation mechanism).  56 tokens stays inside the trained 64-token extent.
D_MAX_HORIZON = 56.0


def saturate_bridge_positions(reg_indices: torch.Tensor, win: Window, h: int, w: int,
                              device) -> tuple[torch.Tensor, torch.Tensor]:
    """The saturated ``(rows, cols)`` of one window's bridge tokens (the horizon-saturation mechanism).

    The register's true offset from the window centre is compressed through a smooth
    ``D_MAX * tanh(delta / D_MAX)``: the direction survives — and the axis ratio too,
    while the offset stays inside the horizon — but no query ever meets a bridge key
    through more rope phase than the model was trained on.
    """
    win_r_center = (win.r0 + win.r1) * 0.5
    win_c_center = (win.c0 + win.c1) * 0.5
    reg_r = (reg_indices // w).to(torch.float32)
    reg_c = (reg_indices % w).to(torch.float32)
    delta_r = reg_r - win_r_center
    delta_c = reg_c - win_c_center
    sat_r = win_r_center + D_MAX_HORIZON * torch.tanh(delta_r / D_MAX_HORIZON)
    sat_c = win_c_center + D_MAX_HORIZON * torch.tanh(delta_c / D_MAX_HORIZON)
    return sat_r, sat_c


def bridge_true_positions(reg_indices: torch.Tensor, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The TRUE ``(rows, cols)`` of one window's bridge tokens on the grid.

    The counterpart of :func:`saturate_bridge_positions`: the register keeps
    the cell it occupies on the canvas, so its hop from the window's centre is
    the REAL one.  A backbone whose rope horizon covers that hop reads the far
    side at its true distance instead of at a compressed one — the frame
    Qwen-Image-2.1 takes (theta=10000, where a 63-92 token hop is a
    well-defined phase and not the wrapped noise the saturation was built for).
    """
    reg_r = (reg_indices // w).to(torch.float32)
    reg_c = (reg_indices % w).to(torch.float32)
    return reg_r, reg_c


def _group_window_ids(group: WindowGroup, layout: FineLayout) -> list[int]:
    """The layout windows a group's members belong to.

    ``group.window_ids`` is the declared mapping — the builder fills it in.
    A hand-built group may leave it unset (``None``); the mapping is then
    recovered by matching the group's token rows against the layout's windows
    (the same windows, found the long way round), so such a group runs instead
    of raising.  A member that matches nothing keeps its own index.
    """
    if group.window_ids is not None:
        return [int(i) for i in group.window_ids.reshape(-1).tolist()]
    found: list[int] = []
    for member, row in enumerate(group.tokens):
        match = None
        for i, win in enumerate(layout.windows):
            if win.area != int(row.numel()):
                continue
            if torch.equal(window_tokens(win, layout.w, row.device), row):
                match = i
                break
        found.append(member if match is None else match)
    return found


def caption_window_ids(layout: FineLayout, t: int, device) -> torch.Tensor:
    """``[n_windows, t, 3]`` — the caption's rope ids, anchored to EACH window.

    **GTCN, Phase II (the GTCN Phase II mechanism).**  Lumina2
    puts every caption token at the spatial origin (``(t, 0, 0)``,
    ``comfy/ldm/lumina/model.py:664-667``), so a window far from the origin
    meets the prompt through a rope phase the model was never trained on — at
    Z-Image's theta (256, one rad per token) a 200-token distance wraps the
    rotation ~32 times and the text cross-attention collapses into noise.  Each
    window therefore sees the caption at ITS OWN origin: the spatial axes
    carry that window's ``(r0, c0)`` and the caption's distance inside the
    window is bounded by the window's side (<= 63 tokens).

    The **t axis is zero — an identity rotation** — because the seam's caption
    keys are ALREADY rotated in the t axis by the model itself
    (``embed_cap``: ``cap_pos_ids[:, :, 0] = arange(len) + 1``) and the rope
    table's axes are INDEPENDENT frequency bands (``EmbedND`` concatenates one
    band per axis, ``comfy/ldm/flux/layers.py:15-29``).  An identity t slot
    therefore preserves the model's own t rotation and adds only the
    window-origin rotation on the spatial bands; the product is exactly
    ``R(t_i, r_win, c_win)`` — the caption as the model would have placed it
    over that window, with its own t positions (1..t) intact.

    One row per window of the layout, in the layout's own order, so a group
    picks its members' rows through the group's window mapping
    (:func:`_group_window_ids`) — never through the window's position in the
    group.
    """
    n = len(layout.windows)
    ids = torch.zeros(n, int(t), 3, device=device, dtype=torch.float32)
    for i, win in enumerate(layout.windows):
        ids[i, :, 1] = float(win.r0)
        ids[i, :, 2] = float(win.c0)
    return ids


def _rope_per_group(x: torch.Tensor, rope: torch.Tensor, b: int, heads: int, g: int,
                    t: int, dim: int) -> torch.Tensor:
    """``[b, heads, g, t, d]`` — ``x`` rotated by each group's OWN rope table.

    The table is head-shared (``[g, 1, t, d/2, 2, 2]``), so the group axis and
    the head axis travel together as the batch: ``[g, b * heads, t, d]``.
    """
    flat = (x.unsqueeze(2).expand(b, heads, g, t, dim)
            .permute(2, 0, 1, 3, 4).reshape(g, b * heads, t, dim))
    return apply_rope1(flat, rope).reshape(g, b, heads, t, dim).permute(1, 2, 0, 3, 4)


def windowed_attention(q_txt: torch.Tensor, k_txt: torch.Tensor, v_txt: torch.Tensor,
                       q_img: torch.Tensor, k_img: torch.Tensor, v_img: torch.Tensor,
                       k_img_unrotated: torch.Tensor, layout: FineLayout,
                       rope_embedder=None, cap_len: int = 32, caption_rope=None,
                       img_axis0: Optional[float] = None, text_keys_only: bool = False,
                       bridge_saturate: bool = True, caption_anchor: bool = True,
                       img_origin: Optional[tuple[float, float]] = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention inside each window, its keys extended by the Halo ring and the bridge tokens.

    Queries and keys are already rotated at their true grid positions (Phase I's
    coarse tables do not apply here); ``k_img_unrotated`` ranks the bridge tokens.
    Window outputs merge with the centre weights, and the text output is the mean
    over windows: every window re-runs the same text queries and keys, so averaging
    is the refusal to let window count leak into the text stream.

    ``rope_embedder`` (with ``cap_len``) is the BACKBONE'S OWN rope embedder:
    with one, the bridge's keys are re-roped through it — and the caption too,
    when the caption anchor is on.  Without one (the FLUX call) the paper's
    behaviour is kept untouched, bit for bit.

    Two mechanisms ride on it.  Both were built for Z-Image, but NEITHER is
    Z-Image-only any more: each is a per-backbone switch whose DEFAULT is
    Z-Image's behaviour, so the Z-Image golden cannot move.

    * the bridge ROPE HORIZON SATURATION (the horizon-saturation mechanism, the default
      ``bridge_saturate=True``): the bridge keys are taken from the UNROTATED
      keys and re-roped at the saturated positions instead of riding the
      true-position rotation, so no query meets a distant key through more
      rope phase than the model was trained on.  A backbone whose rope horizon
      covers the REAL hop opts out (``bridge_saturate=False``): the registers
      are then re-roped at their TRUE positions, so the bridge keys carry the
      model's own true-position rotation — Qwen-Image-2.1 does (theta=10000,
      where a 63-92 token hop is a well-defined phase);
    * the GTCN's CAPTION ANCHORING (the GTCN mechanism, the default
      ``caption_anchor=True``): the caption is re-roped at each window's own
      origin, so the prompt stays inside the model's trained phase even for a
      window far from the canvas' top-left corner — see
      :func:`caption_window_ids` for the route and its evidence.  A backbone
      whose caption does NOT ride the spatial axes opts out
      (``caption_anchor=False``): Qwen-Image-2.1's caption rides a DIAGONAL
      at the model's own running counter (``model.py:323``), so there is no
      per-window spatial origin to anchor it to.

    ``caption_rope`` is that anchor table, ALREADY built: the caller may pass it
    (the Z-Image seam builds it once per step and shares it between the step's
    attention calls — the table depends only on the step's windows, the caption
    length and the model's embedder, all of them fixed for the step).  Without
    it the table is built here, which is what every other caller does.

    ``img_axis0`` is the rope AXIS-0 position the bridge keys are re-roped at
    (the horizon-saturation branch above).  It is the backbone's own axis-0
    value for an image token: ``None`` keeps Z-Image's convention
    (``cap_len + 1`` — the first token after the caption block), and a backbone
    that puts its image tokens at the frame index passes that index instead
    (Krea-2's image tokens carry axis 0 = 0,
    ``comfy/ldm/krea2/model.py:291-294``).  The default reproduces today's
    behaviour byte for byte, so the Z-Image golden cannot move.

    ``caption_anchor`` switches the CAPTION ANCHORING mechanism off (see the
    bullet above) — for a backbone whose caption does not ride the spatial
    axes, so that there is no per-window origin to anchor it to.

    ``img_origin`` is the SHIFT between the grid's own 0-based positions and
    the frame the backbone's image tokens ride: the bridge's rows and columns
    are shifted by it before they are handed to the embedder.  A backbone
    whose image tokens ride CENTRED spatial axes needs the shift —
    Qwen-Image-2.1's own centred ramp is ``index - (size - size // 2)``
    (``comfy/ldm/qwen_image21/model.py:330-331``), so the seam passes
    ``(h - h // 2, w - w // 2)``; ``None`` keeps the grid's own 0-based frame
    (Z-Image / Krea-2), so their goldens cannot move.

    ``text_keys_only`` splits the ONE length this function used to couple the
    text queries and the text keys by (``t = q_txt.shape[2]``) into the two
    lengths they really have: ``t_q`` (the text QUERIES) and ``t_k`` (the text
    KEYS).  A call that carries the caption's KEYS and NO text queries at all —
    Qwen-Image-2.1's image segment, where the queries are the ``h*w`` image
    rows and the keys are the JOINT set (``comfy/ldm/qwen_image21/model.py:
    206-211``) — needs the caption to stay VISIBLE to the image queries while
    producing no text answer of its own.  With the flag on, the query cat is
    ``t_q + n`` rows, the key/value cat is ``t_k + n + extra`` rows and the
    answer is split at ``t_q`` (``merged = out[:, :, :, t_q:]``); with the flag
    OFF (the default) the two lengths are asserted equal and every line is the
    line that is there today — the FLUX / Z-Image / Krea-2 goldens are the
    proof.
    """
    b, heads, n_img, dim = q_img.shape
    # The text QUERIES and the text KEYS are two lengths, not one: a call with
    # text keys and no text queries passes an EMPTY q_txt (``text_keys_only``).
    t_q = q_txt.shape[2]
    t_k = k_txt.shape[2]
    if t_q != t_k and not text_keys_only:
        raise ValueError(
            f"AReA: the windowed attention received {t_q} text queries and {t_k} "
            "text keys, but the default path couples them into one length. Pass "
            "text_keys_only=True for a call that carries text keys and no text "
            "queries (Qwen-Image-2.1's image segment)."
        )
    out_img = torch.zeros(b, heads, n_img, dim, dtype=q_img.dtype, device=q_img.device)
    out_txt = torch.zeros(b, heads, t_q, dim, dtype=q_img.dtype, device=q_img.device)
    score = token_importance(k_img_unrotated) if layout.registers > 0 else None

    # GTCN Phase II (the GTCN mechanism): the caption's anchor table, ONE call for the
    # whole layout — one row per window, the caption at that window's origin.
    # The table's t axis is zero (an identity rotation), so the model's own t
    # rotation of the caption survives and only the window-origin rotation is
    # added — see ``caption_window_ids``.  Without an embedder (the FLUX call)
    # the caption keeps the paper's own positions, bit for bit.
    cap_rope = None
    if rope_embedder is not None and caption_anchor:
        if caption_rope is None:
            # ``area_zimage`` imports this module, so the helper is imported
            # here (the same discipline the bridge branch below uses) — the
            # FLUX path never imports it at all.
            from .area_zimage import _head_shared
            caption_rope = _head_shared(
                rope_embedder(caption_window_ids(layout, t_k, q_img.device)))
        cap_rope = caption_rope

    for group in layout.groups:
        g, n = group.tokens.shape
        flat = group.tokens.reshape(-1)

        if cap_rope is not None:
            # the group's OWN windows (never ``layout.windows[:g]``)
            win_ids = torch.tensor(_group_window_ids(group, layout),
                                   device=q_img.device, dtype=torch.long)
            rope = cap_rope.index_select(0, win_ids)
            q_txt_g = _rope_per_group(q_txt, rope, b, heads, g, t_q, dim)
            k_txt_g = _rope_per_group(k_txt, rope, b, heads, g, t_k, dim)
        else:
            q_txt_g = q_txt.unsqueeze(2).expand(b, heads, g, t_q, dim)
            k_txt_g = k_txt.unsqueeze(2).expand(b, heads, g, t_k, dim)
        v_txt_g = v_txt.unsqueeze(2).expand(b, heads, g, t_k, dim)

        extra_k, extra_v = [], []
        if score is not None:
            reg = select_registers(score, group.tokens, layout.registers, layout.h, layout.w)
            if rope_embedder is not None:
                # The bridge FRAME — the positions the registers are re-roped
                # at.  There are two of them:
                #
                # * SATURATED (the horizon-saturation mechanism, the DEFAULT): the
                #   register's offset from the window centre is compressed
                #   through ``D_MAX * tanh(delta / D_MAX)``, so no query meets
                #   a distant key through more rope phase than the model was
                #   trained on.  Z-Image's rationale — theta=256 wraps a 100+
                #   token hop into pseudorandom noise — and its goldens are
                #   frozen on it.
                # * TRUE (``bridge_saturate=False``): the register keeps the
                #   cell it occupies on the canvas.  Qwen-Image-2.1's theta is
                #   10000, where a 63-92 token hop is a well-defined phase —
                #   and the compression only MANUFACTURED the far-half copy it
                #   was meant to prevent: it pulled the other half's tokens
                #   inside the window's own horizon, so the far half was read
                #   as local.
                #
                # The VALUES are the registers' own either way, and a caller
                # without an embedder (FLUX) keeps the paper's true-position
                # keys untouched.
                rows, cols = [], []
                for j, win_index in enumerate(_group_window_ids(group, layout)):
                    if bridge_saturate:
                        r, c = saturate_bridge_positions(
                            reg[j], layout.windows[win_index], layout.h, layout.w, q_img.device)
                    else:
                        r, c = bridge_true_positions(reg[j], layout.w)
                    rows.append(r)
                    cols.append(c)
                ids = torch.zeros(g, layout.registers, 3, device=q_img.device,
                                  dtype=torch.float32)
                # The bridge keys are re-roped at the backbone's own axis-0
                # position for an image token: Z-Image's convention is
                # ``cap_len + 1`` (the first token after the caption block) and
                # it is the DEFAULT, so the Z-Image golden cannot move; a
                # backbone whose image tokens carry the frame index instead
                # (Krea-2: axis 0 = 0) passes its own value.
                ids[:, :, 0] = float(cap_len + 1) if img_axis0 is None else float(img_axis0)
                # The positions live in the grid's OWN 0-based frame; a
                # backbone whose image tokens ride CENTRED spatial axes
                # (Qwen-Image-2.1) needs them shifted into that frame.
                org_r, org_c = (0.0, 0.0) if img_origin is None else img_origin
                ids[:, :, 1] = torch.stack(rows) - float(org_r)
                ids[:, :, 2] = torch.stack(cols) - float(org_c)
                # ``area_zimage`` imports this module, so the helper is imported
                # here — the FLUX path never reaches this branch and never
                # imports it.
                from .area_zimage import _head_shared
                rope = _head_shared(rope_embedder(ids))
                reg_flat = reg.reshape(-1)
                unrot = (k_img_unrotated[:, :, reg_flat]
                         .reshape(b, heads, g, -1, dim)
                         .permute(2, 0, 1, 3, 4).reshape(g, b * heads, -1, dim))
                extra_v.append(v_img[:, :, reg_flat].reshape(b, heads, g, -1, dim))
                extra_k.append(
                    apply_rope1(unrot, rope).reshape(g, b, heads, -1, dim).permute(1, 2, 0, 3, 4))
            else:
                reg_flat = reg.reshape(-1)
                extra_k.append(k_img[:, :, reg_flat].reshape(b, heads, g, -1, dim))
                extra_v.append(v_img[:, :, reg_flat].reshape(b, heads, g, -1, dim))
        if group.halo is not None:
            ring = group.halo.reshape(-1)
            extra_k.append(k_img[:, :, ring].reshape(b, heads, g, -1, dim))
            extra_v.append(v_img[:, :, ring].reshape(b, heads, g, -1, dim))

        # the group runs as one SDPA call on [b*g, heads, t_q+n, d]; comfy keeps heads on axis 1, so
        # the group axis has to travel past the head axis to reach a shape SDPA accepts
        local = q_img[:, :, flat].reshape(b, heads, g, n, dim)
        q = torch.cat([q_txt_g, local], dim=3).permute(0, 2, 1, 3, 4).reshape(b * g, heads, t_q + n, dim)
        k = torch.cat([k_txt_g, k_img[:, :, flat].reshape(b, heads, g, n, dim)] + extra_k,
                      dim=3).permute(0, 2, 1, 3, 4).reshape(b * g, heads, -1, dim)
        v = torch.cat([v_txt_g, v_img[:, :, flat].reshape(b, heads, g, n, dim)] + extra_v,
                      dim=3).permute(0, 2, 1, 3, 4).reshape(b * g, heads, -1, dim)
        out = F.scaled_dot_product_attention(q, k, v).reshape(b, g, heads, t_q + n, dim)

        out_txt += out[:, :, :, :t_q].sum(dim=1)
        merged = out[:, :, :, t_q:].permute(0, 2, 1, 3, 4).reshape(b, heads, g * n, dim)
        out_img.index_add_(2, flat, merged * group.weights.view(1, 1, -1, 1))

    out_img = out_img / layout.coverage.clamp(min=1.0).view(1, 1, n_img, 1)
    out_txt = out_txt / max(float(len(layout.windows)), 1.0)
    return out_txt, out_img


# ---------------------------------------------------------------------------
# Phase II: the tiling
# ---------------------------------------------------------------------------


@torch.no_grad()
def detail_map(hidden_states: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Normalised wavelet detail map E (Eqs. 12 and 13) from the image tokens ``[b, h*w, C]``.

    ``hidden_states`` is what the model produces before attention — in comfy the
    ``img_in`` output, in the reference the ``x_embedder`` output.  The map is
    mean-subtracted, so a flat canvas maps to all zeros; the reference's mean over
    the batch is kept (comfy's one forward carries cond and uncond).
    """
    n = h * w
    s = torch.empty(n, device=hidden_states.device, dtype=torch.float32)
    for i0 in range(0, n, _CHUNK_TOKENS):
        i1 = min(n, i0 + _CHUNK_TOKENS)
        s[i0:i1] = hidden_states[:, i0:i1].float().std(dim=-1).mean(dim=0)
    s = s.reshape(h, w)
    x = (s - s.mean()).cpu().numpy()

    multiple = (2 ** DWT_LEVELS) * DWT_PAD
    x = np.pad(x, ((0, (-h) % multiple), (0, (-w) % multiple)), mode="edge")
    energy = np.zeros_like(x, dtype=np.float32)
    cur = x.astype(np.float64)
    for level in range(DWT_LEVELS):
        ch, cw = cur.shape
        if ch < 2 or cw < 2:
            break
        cur = cur[: ch - ch % 2, : cw - cw % 2]
        a, b, c, d = cur[0::2, 0::2], cur[0::2, 1::2], cur[1::2, 0::2], cur[1::2, 1::2]
        detail = (np.abs((a + b - c - d) * 0.25) + np.abs((a - b + c - d) * 0.25)
                  + np.abs((a - b - c + d) * 0.25))
        up = detail.astype(np.float64)
        while up.shape[0] < x.shape[0] or up.shape[1] < x.shape[1]:
            up = np.repeat(np.repeat(up, 2, axis=0), 2, axis=1)
        energy += (up[: x.shape[0], : x.shape[1]].astype(np.float32) * (0.5 ** level)).astype(np.float32)
        cur = (a + b + c + d) * 0.25
    energy = energy[:h, :w]
    if energy.max() > 0:
        energy = energy / (energy.max() + 1e-8)
    return energy.astype(np.float32)


def jittered_starts(total: int, size: int, rng: np.random.Generator) -> list[int]:
    """A uniformly random valid placement of the base windows on one axis.

    Same count and size as ``even_starts`` with both end windows pinned, so the
    axis stays covered.  Valid placements are the integer compositions of
    ``total - size`` into gaps in ``[1, size]``; one is drawn uniformly with an
    exact counting table, since drawing each start in turn would pile the slack at
    the far edge.
    """
    base = even_starts(total, size)
    T, m = int(total) - int(size), len(base) - 1
    if m <= 1 or T <= 0:
        return base
    counts = np.zeros((m + 1, T + 1), dtype=np.float64)
    counts[0, 0] = 1.0
    idx = np.arange(T + 1)
    for i in range(1, m + 1):
        pre = np.cumsum(counts[i - 1])
        a, b = idx - 1, idx - size - 1
        counts[i] = (np.where(a >= 0, pre[np.clip(a, 0, T)], 0.0)
                     - np.where(b >= 0, pre[np.clip(b, 0, T)], 0.0))
    if counts[m, T] <= 0.0:
        return base
    starts, pos, rem = [0], 0, T
    for i in range(m, 0, -1):
        cand = np.arange(1, min(size, rem - (i - 1)) + 1)
        weight = counts[i - 1][rem - cand]
        total_w = float(weight.sum())
        gap = int(cand[-1]) if total_w <= 0.0 else int(rng.choice(cand, p=weight / total_w))
        pos += gap
        rem -= gap
        starts.append(pos)
    return starts


def _mean_energy(energy: np.ndarray, win: Window) -> float:
    block = energy[win.r0 : win.r1 + 1, win.c0 : win.c1 + 1]
    return float(block.mean()) if block.size else 0.0


def _quadrants(win: Window) -> list[Window]:
    rm = win.r0 + win.height // 2 - 1
    cm = win.c0 + win.width // 2 - 1
    return [Window(r0, c0, r1, c1, split=True)
            for r0, r1 in ((win.r0, rm), (rm + 1, win.r1))
            for c0, c1 in ((win.c0, cm), (cm + 1, win.c1))]


def adaptive_tiling(energy: np.ndarray, h: int, w: int, size: int, w_min: int, txt_len: int,
                    rng: Optional[np.random.Generator] = None,
                    allow_splits: bool = True) -> list[Window]:
    """Base tiling of side ``size`` with the smooth windows split into quadrants (Appendix C).

    The gate alpha comes from the tile-energy coefficient of variation, so a flat
    canvas never splits; candidates are the low-energy half that sit below the
    window-mean energy, and a split is only taken while the step stays inside its
    pre-split cost budget.  Any split pair that covers each other by at least
    ``DEDUPE_COVER`` is merged back, and both candidates and splits are drawn from
    ``rng`` when one is given.
    """
    rows = jittered_starts(h, size, rng) if rng is not None else even_starts(h, size)
    cols = jittered_starts(w, size, rng) if rng is not None else even_starts(w, size)
    boxes = [Window(r, c, min(h - 1, r + size - 1), min(w - 1, c + size - 1)) for r in rows for c in cols]
    if not allow_splits:
        # Window integrity (the short-schedule window-integrity rule): the DWT energy map reads noise as
        # detail before an 8-step distilled model has composed anything, and its
        # quadrants are 11% of the canvas each — small enough to hallucinate a whole
        # subject of their own.  A short schedule therefore keeps the base windows.
        return boxes
    e = np.array([_mean_energy(energy, box) for box in boxes], dtype=np.float64).reshape(len(rows), len(cols))

    tile = max(1, size // 2)
    th, tw = energy.shape[0] // tile, energy.shape[1] // tile
    tiles = (energy[: th * tile, : tw * tile].reshape(th, tile, tw, tile).mean(axis=(1, 3))
             if th * tw >= 4 else e)
    t_mean = float(tiles.mean())
    u = float(tiles.std() / t_mean) if t_mean > 1e-12 else 0.0
    alpha = min(1.0, max(0.0, (u - GATE_LO) / max(1e-6, GATE_HI - GATE_LO)))
    max_splits = int(alpha * len(boxes))
    if alpha <= 0.0 or len(boxes) < 2 or max_splits == 0:
        return boxes

    def cost(win: Window) -> float:
        # text tokens are queries and keys of every window
        return float(txt_len + win.area) ** 2

    budget = total = sum(cost(box) for box in boxes)
    smooth = -e
    threshold = float(np.quantile(smooth, SPLIT_QUANTILE))
    mean = float(smooth.mean())
    split = [False] * len(boxes)
    n_split = 0
    for k in sorted(range(len(boxes)), key=lambda k: -smooth.flat[k]):
        if n_split >= max_splits or smooth.flat[k] < threshold or smooth.flat[k] <= mean:
            break
        box = boxes[k]
        if min(box.height // 2, box.width // 2) < max(1, w_min // 2):
            continue
        delta = sum(cost(q) for q in _quadrants(box)) - cost(box)
        if total + delta > budget:
            continue
        split[k] = True
        total += delta
        n_split += 1

    windows: list[Window] = []
    for box, is_split in zip(boxes, split):
        windows.extend(_quadrants(box) if is_split else [box])
    return _dedupe(windows)


def halo_ring(win: Window, depth, h: int, w: int, device) -> Optional[torch.Tensor]:
    """Tokens of the window's box grown by ``depth`` and clipped to the grid, minus the window.

    ``depth`` is one width for all four sides or ``(top, bottom, left, right)``.
    """
    if ring_size(win, depth, h, w) <= 0:
        return None
    r0, r1, c0, c1 = _grown_box(win, depth, h, w)
    rows = torch.arange(r0, r1 + 1, device=device)
    cols = torch.arange(c0, c1 + 1, device=device)
    rr, cc = rows[:, None], cols[None, :]
    outside = (rr < win.r0) | (rr > win.r1) | (cc < win.c0) | (cc > win.c1)
    return (rr * w + cc)[outside]


def _side_profile(total: int, length: int, cap: int, rng: np.random.Generator) -> np.ndarray:
    """``length`` random reaches in ``[0, cap]`` that sum to ``total``."""
    depth = rng.integers(0, cap + 1, size=length)
    gap = total - int(depth.sum())
    while gap != 0:
        room = np.flatnonzero(depth < cap) if gap > 0 else np.flatnonzero(depth > 0)
        pick = rng.choice(room, size=min(abs(gap), room.size), replace=False)
        depth[pick] += 1 if gap > 0 else -1
        gap -= int(np.sign(gap)) * pick.size
    return depth


def jagged_halo_ring(win: Window, depth: Sequence[int], halo: int, h: int, w: int,
                     rng: np.random.Generator, device) -> Optional[torch.Tensor]:
    """``halo_ring`` with an uneven edge: every row (left, right) or column (top, bottom) of a side
    reaches its own random distance in ``[0, halo]``, and the reaches of a side sum to what the
    straight edge of ``depth`` would hold. The ring keeps ``ring_size(win, depth, ...)`` tokens, and
    its corners stay the blocks between the two sides' widths.
    """
    if ring_size(win, depth, h, w) <= 0:
        return None
    r0, r1, c0, c1 = _grown_box(win, depth, h, w)
    top, bottom, left, right = win.r0 - r0, r1 - win.r1, win.c0 - c0, c1 - win.c1   # widths after clipping
    rooms = (win.r0, h - 1 - win.r1, win.c0, w - 1 - win.c1)
    rows_in = np.arange(win.r0, win.r1 + 1)
    cols_in = np.arange(win.c0, win.c1 + 1)
    parts = []
    for width, room, along, side in ((top, rooms[0], cols_in, "top"), (bottom, rooms[1], cols_in, "bottom"),
                                     (left, rooms[2], rows_in, "left"), (right, rooms[3], rows_in, right)):
        if width <= 0:
            continue
        cap = min(halo, room)
        reach = _side_profile(width * along.size, along.size, cap, rng)
        k = np.arange(1, cap + 1)[:, None]
        keep = k <= reach[None, :]
        if side == "top":
            rr, cc = win.r0 - k, along[None, :]
        elif side == "bottom":
            rr, cc = win.r1 + k, along[None, :]
        elif side == "left":
            rr, cc = along[None, :], win.c0 - k
        else:
            rr, cc = along[None, :], win.c1 + k
        parts.append(np.broadcast_to(rr * w + cc, keep.shape)[keep])
    for rows, cols in ((np.arange(r0, win.r0), np.arange(c0, win.c0)), (np.arange(r0, win.r0), np.arange(win.c1 + 1, c1 + 1)),
                       (np.arange(win.r1 + 1, r1 + 1), np.arange(c0, win.c0)),
                       (np.arange(win.r1 + 1, r1 + 1), np.arange(win.c1 + 1, c1 + 1))):
        parts.append((rows[:, None] * w + cols[None, :]).reshape(-1))
    return torch.from_numpy(np.concatenate(parts)).to(device)


def merge_weights(windows: list[Window], tokens: list[torch.Tensor], n_img: int,
                  dtype: torch.dtype, device) -> list[torch.Tensor]:
    """Centre-weighted merge: a separable raised cosine per window, rescaled so the weights
    covering a token sum to its coverage count (a token in one window keeps weight 1).
    """
    raw = []
    for win in windows:
        r = torch.arange(win.height, device=device, dtype=torch.float32)
        c = torch.arange(win.width, device=device, dtype=torch.float32)
        fr = torch.sin(math.pi * (r + 0.5) / win.height) ** 2
        fc = torch.sin(math.pi * (c + 0.5) / win.width) ** 2
        raw.append((BLEND_FLOOR + (1.0 - BLEND_FLOOR) * fr[:, None] * fc[None, :]).reshape(-1))
    total = torch.zeros(n_img, dtype=torch.float32, device=device)
    count = torch.zeros(n_img, dtype=torch.float32, device=device)
    for idx, weight in zip(tokens, raw):
        total.index_add_(0, idx, weight)
        count[idx] += 1.0
    scale = count / total.clamp(min=1e-8)
    return [(weight * scale[idx]).to(dtype) for idx, weight in zip(tokens, raw)]


# ---------------------------------------------------------------------------
# The bridge
# ---------------------------------------------------------------------------


def _token_matrix(x: torch.Tensor, c: int) -> torch.Tensor:
    """A ``[b, heads, n, d]`` chunk summed over the batch, as ``[n, heads*d]`` (one row per token).

    Comfy's layout puts heads before the sequence, so the head axis is moved before
    the flatten; the ordering inside a row is irrelevant to a distance, but it must
    be the same in the mean and in the score.
    """
    return x.sum(dim=0).transpose(0, 1).reshape(-1, c)


@torch.no_grad()
def token_importance(k_img: torch.Tensor) -> torch.Tensor:
    """Eq. 17: squared distance of each (unrotated) image key ``[b, heads, N, d]`` from the canvas-mean key.

    The mean is taken over the whole batch (comfy's one forward carries cond and
    uncond), so one bridge ranking serves both CFG branches.
    """
    n = int(k_img.shape[2])
    c = int(k_img.shape[1]) * int(k_img.shape[3])
    chunk = max(1, _CHUNK_ELEMS // c)
    total = torch.zeros(c, device=k_img.device, dtype=torch.float32)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        total += _token_matrix(k_img[:, :, i0:i1].float(), c).sum(0)
    mean = total / float(n)
    score = torch.empty(n, device=k_img.device, dtype=torch.float32)
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        score[i0:i1] = (_token_matrix(k_img[:, :, i0:i1].float(), c) - mean).pow(2).sum(dim=1)
    return score


@torch.no_grad()
def select_registers(score: torch.Tensor, windows: torch.Tensor, count: int, h: int, w: int) -> torch.Tensor:
    """``[G, count]`` bridge tokens for ``G`` windows given their token indices ``[G, n]``.

    The canvas is split into a ``CELLS x CELLS`` grid. Each cell gets a quota
    proportional to how many of its tokens lie outside the window (leftovers go to
    the cells with the most headroom), and is filled with its highest-scoring
    outside tokens.  A window's own tokens are never offered as bridge.
    """
    dev = score.device
    n = int(score.shape[0])
    pos = torch.arange(n, device=dev)
    cr = ((pos // w).float() * CELLS / max(h, 1)).long().clamp_(0, CELLS - 1)
    cc = ((pos % w).float() * CELLS / max(w, 1)).long().clamp_(0, CELLS - 1)
    cell = cr * CELLS + cc
    out = torch.empty(int(windows.shape[0]), count, dtype=torch.long, device=dev)
    for g in range(int(windows.shape[0])):
        sg = score.index_fill(0, windows[g], float("-inf"))
        avail = torch.isfinite(sg)
        cnt = torch.bincount(cell[avail], minlength=CELLS * CELLS).float()
        quota = torch.minimum(torch.floor(cnt / cnt.sum().clamp_min(1.0) * count).long(), cnt.long())
        short = count - int(quota.sum())
        if short > 0:
            order = torch.argsort(cnt - quota.float(), descending=True)
            head = (cnt.long() - quota)[order]
            take = torch.minimum(torch.clamp(short - (torch.cumsum(head, 0) - head), min=0), head)
            quota[order] += take
        ai = torch.nonzero(avail, as_tuple=False).squeeze(1)
        if ai.numel():
            ranked = ai[torch.sort(sg[ai], descending=True, stable=True)[1]]
            ranked = ranked[torch.sort(cell[ranked], stable=True)[1]]
            start = torch.cumsum(cnt.long(), 0) - cnt.long()
            rank_in_cell = torch.arange(ranked.numel(), device=dev) - start[cell[ranked]]
            sel = ranked[rank_in_cell < quota[cell[ranked]]]
        else:
            sel = torch.empty(0, dtype=torch.long, device=dev)
        if sel.numel() < count:
            extra = sg.topk(min(n, count)).indices
            sel = torch.cat([sel, extra[~torch.isin(extra, sel)][: count - sel.numel()]])
        out[g] = sel[:count]
    return out
