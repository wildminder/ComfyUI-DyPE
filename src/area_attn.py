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
    even_starts,
    ring_size,
    step_rng,
    window_tokens,
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

    Phase 3 sets and clears this around every forward (the SPA discipline,
    ``src/spa.py:1385-1476``) so no layout may leak across models or steps.
    """

    phase: Optional[str] = None
    step: int = 0
    size: int = 0
    h: int = 0
    w: int = 0
    txt_len: int = 0
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase I: cross-scale coarse attention
# ---------------------------------------------------------------------------


def _block_selection(k_img: torch.Tensor, h: int, w: int, f: int, pooling: str,
                     seed: int, step: int, block: int) -> Optional[torch.Tensor]:
    """Index in ``[0, f*f)`` of the token that represents each block, for the selecting poolings.

    ``k_img`` is ``[b, heads, h*w, d]``; the result is ``[b, h/f, w/f]`` and is
    shared by keys and values, so K and V always come from the same token.
    """
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


def _pool(x_img: torch.Tensor, h: int, w: int, f: int, pooling: str,
          sel: Optional[torch.Tensor]) -> torch.Tensor:
    """One token per ``f x f`` block of the image tokens, returned ``[b, heads, h*w/f^2, d]``."""
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


def pool_keys_values(key: torch.Tensor, value: torch.Tensor, txt_len: int, h: int, w: int, f: int,
                     pooling: str, seed: int = 0, step: int = 0, block: int = 0
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep one token per ``f x f`` block of the image keys and values (text is never pooled).

    ``subsample`` keeps the block centre, ``mean`` averages the block, ``medoid``
    keeps the token whose key is closest to the block-mean key, and ``random``
    keeps a token drawn per ``(seed, step, block)``.  Keys must be unrotated —
    pooling after RoPE would rotate the survivors away from the query's frame —
    and K and V always come from the same token.
    """
    k_img, v_img = key[:, :, txt_len:], value[:, :, txt_len:]
    sel = _block_selection(k_img, h, w, f, pooling, seed, step, block)
    k = torch.cat([key[:, :, :txt_len], _pool(k_img, h, w, f, pooling, sel)], dim=2)
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


@dataclass
class FineLayout:
    """Everything Phase II needs for one denoising step, shared by all transformer blocks."""

    h: int
    w: int
    windows: list[Window]
    groups: list[WindowGroup]
    coverage: torch.Tensor               # [N] number of windows covering each token
    registers: int                       # R bridge tokens per window (0 = no bridge)


def fit_extra_keys(windows: list[Window], h: int, w: int, txt_len: int, registers: int,
                   halo: int) -> tuple[int, int]:
    """Bridge count and Halo width that keep one Phase II step at or below full attention.

    Each window runs ``T + n`` queries (text and its image tokens) against ``T + n``
    keys plus its Halo ring and ``R`` bridge tokens, so the step costs
    ``sum (T+n)(T+n+ring+R)`` against ``(T + h*w)^2``.  The Halo is priced at its
    full width, the most any window can draw.  The bridge is cut first, to the
    largest count that fits; the Halo narrows only if the step is still above full
    attention without any bridge tokens.
    """
    full = float(txt_len + h * w) ** 2
    rows = [float(txt_len + win.area) for win in windows]

    def local(depth: int) -> float:
        return sum(q * (q + ring_size(win, depth, h, w)) for q, win in zip(rows, windows))

    room = full - local(halo)
    if room >= registers * sum(rows):
        return registers, halo
    if room >= 0:
        return int(room // sum(rows)), halo
    depth = next((d for d in range(halo, -1, -1) if local(d) <= full), 0)
    return 0, depth


def tiling_rng(seed: int, step: int) -> np.random.Generator:
    """The keyed stream of one step's adaptive splits (the backbone's salt)."""
    return step_rng(seed, step, TILING_SALT)


def bridge_registers(size: int, bridge_ratio: float) -> int:
    """R of the bridge: ``round(rho * w)^2`` tokens per window, 0 when the bridge is off."""
    return max(1, int(round(float(bridge_ratio) * int(size)))) ** 2 if bridge_ratio > 0 else 0


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
        )
        for (_, ring_len), members in by_key.items()
    ]
    return FineLayout(h, w, windows, groups, coverage, registers)


def windowed_attention(q_txt: torch.Tensor, k_txt: torch.Tensor, v_txt: torch.Tensor,
                       q_img: torch.Tensor, k_img: torch.Tensor, v_img: torch.Tensor,
                       k_img_unrotated: torch.Tensor, layout: FineLayout
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention inside each window, its keys extended by the Halo ring and the bridge tokens.

    Queries and keys are already rotated at their true grid positions (Phase I's
    coarse tables do not apply here); ``k_img_unrotated`` ranks the bridge tokens.
    Window outputs merge with the centre weights, and the text output is the mean
    over windows: every window re-runs the same text queries and keys, so averaging
    is the refusal to let window count leak into the text stream.
    """
    b, heads, n_img, dim = q_img.shape
    t = q_txt.shape[2]
    out_img = torch.zeros(b, heads, n_img, dim, dtype=q_img.dtype, device=q_img.device)
    out_txt = torch.zeros(b, heads, t, dim, dtype=q_img.dtype, device=q_img.device)
    score = token_importance(k_img_unrotated) if layout.registers > 0 else None

    for group in layout.groups:
        g, n = group.tokens.shape
        flat = group.tokens.reshape(-1)

        def gather(x_txt, x_img):
            # [b, heads, g, t+n, d]: the text tokens attend in every window of the group
            local = x_img[:, :, flat].reshape(b, heads, g, n, dim)
            return torch.cat([x_txt.unsqueeze(2).expand(b, heads, g, t, dim), local], dim=3)

        extra_k, extra_v = [], []
        if score is not None:
            reg = select_registers(score, group.tokens, layout.registers, layout.h, layout.w).reshape(-1)
            extra_k.append(k_img[:, :, reg].reshape(b, heads, g, -1, dim))
            extra_v.append(v_img[:, :, reg].reshape(b, heads, g, -1, dim))
        if group.halo is not None:
            ring = group.halo.reshape(-1)
            extra_k.append(k_img[:, :, ring].reshape(b, heads, g, -1, dim))
            extra_v.append(v_img[:, :, ring].reshape(b, heads, g, -1, dim))

        # the group runs as one SDPA call on [b*g, heads, t+n, d]; comfy keeps heads on axis 1, so
        # the group axis has to travel past the head axis to reach a shape SDPA accepts
        q = gather(q_txt, q_img).permute(0, 2, 1, 3, 4).reshape(b * g, heads, t + n, dim)
        k = torch.cat([gather(k_txt, k_img)] + extra_k, dim=3).permute(0, 2, 1, 3, 4).reshape(b * g, heads, -1, dim)
        v = torch.cat([gather(v_txt, v_img)] + extra_v, dim=3).permute(0, 2, 1, 3, 4).reshape(b * g, heads, -1, dim)
        out = F.scaled_dot_product_attention(q, k, v).reshape(b, g, heads, t + n, dim)

        out_txt += out[:, :, :, :t].sum(dim=1)
        merged = out[:, :, :, t:].permute(0, 2, 1, 3, 4).reshape(b, heads, g * n, dim)
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
                    rng: Optional[np.random.Generator] = None) -> list[Window]:
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
