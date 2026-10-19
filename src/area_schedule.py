"""AReA per-step schedule and Phase II window math — port of ``area/schedule.py``.

Literal port of the reference's ``schedule.py``
(``.dev/data/area/AReA/area/schedule.py``) plus the pure-math tiling primitives
of its ``tiling.py``: the window box, the clipped-box/geometry helpers and the
coverage/dedupe predicates.  The tensor-producing tiling (DWT detail map,
jittered starts, adaptive splits, halo rings, merge weights) lives in
:mod:`src.area_attn`; the model patcher that builds one schedule per forward
lives in :mod:`src.area`.

Everything here is pure numpy/int math with no tensor layout, so it is testable
GPU-free and shared by both attention phases.  The reference works in
diffusers' ``[B, S, H, D]`` layout while ComfyUI FLUX reaches the attention seam
as ``[b, heads, S, D]``; nothing in this module touches the axis order — the
boxes and token indices it produces are layout-agnostic and serve both.

Determinism: the only randomness is :func:`step_rng`, keyed by
``(seed, step, salt)``.  ComfyUI batches cond and uncond into ONE forward
(``comfy/samplers.py``), so a keyed layout must serve both CFG branches and
nothing here may read the global RNG (theory.md:496, "Synchronize randomness
under classifier-free guidance").

Deviations from the reference, all deliberate and pinned by
``tests/test_area_schedule.py``:
* ``Schedule`` is named :class:`AreaSchedule` — ``src`` is a flat shared
  namespace and a bare ``Schedule`` would collide with the next port.
* :meth:`AreaSchedule.at` holds the last **factor** on an all-coarse schedule
  (``phase1_steps == steps``) instead of indexing ``windows[-1]`` on an empty
  tuple the way the reference does; every in-range answer is unchanged.
* :func:`build_schedule` does not short-circuit at or below the native extent;
  the caller (``src/area.py``) owns that gate, mirroring the reference's
  ``Backbone.schedule`` (``backbones/base.py:59-68``).  A native-size grid still
  builds a well-formed schedule here (one full-grid window per fine step).
* :func:`build_schedule` takes ``window_start`` / ``window_hold_steps``
  short-schedule geometry overrides and :func:`window_sizes` a ``hold`` rung —
  both absent from the reference, both ignored (zeroed) on a long schedule, so
  the FLUX default stays byte-identical.
* The engine never raises for a patch grid with no common divisor >= 2.
  :func:`initial_factor` keeps the reference's strict raise (it is the literal
  port of the reference's ``schedule.py:39-49``), and the engine-facing
  :func:`resolve_factor` replaces it: on such a grid the factor comes from the
  extent rule instead of the divisibility rule, the pooling blocks no longer
  tile the grid exactly (the last block on a partial axis is a *partial*
  block) and the caller is told the grid was *adjusted*.  The canvas, the
  token stream and the latent are untouched by the adjustment.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

F_MIN = 2          # Eq. 4: Phase I never pools past 2x2 blocks
DEDUPE_COVER = 0.8  # Appendix C: split windows covering each other by this much merge

#: A run is a SHORT (distilled) schedule when its total step count sits at or below
#: this.  The paper calibrated on 28 steps; Turbo-class samplers run 8, where the
#: same constants mean something else (the regime advisory and the Phase II bridge
#: floor both key on this one definition, so the two gates can never disagree).
SHORT_SCHEDULE_TOTAL = 12


# ---------------------------------------------------------------------------
# Window geometry (reference tiling.py)
# ---------------------------------------------------------------------------


@dataclass
class Window:
    """Inclusive box on the patch grid."""

    r0: int
    c0: int
    r1: int
    c1: int
    split: bool = False

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def area(self) -> int:
        return self.height * self.width


def step_rng(seed: int, step: int, salt: int) -> np.random.Generator:
    """The RNG stream of one (seed, step, salt) draw — never the global stream.

    Keying on the step (not a running cursor) is what lets cond and uncond share
    one layout: both CFG branches resolve the same step index, so both draw from
    the identical stream.
    """
    return np.random.default_rng((int(seed) & 0x7FFFFFFF) + int(step) * 2_654_435_761 + salt)


def _grown_box(win: Window, depth: int | Sequence[int], h: int, w: int) -> tuple[int, int, int, int]:
    """The window's box grown by ``depth`` and clipped to the grid, as inclusive ``(r0, r1, c0, c1)``.

    ``depth`` is one width for all four sides or ``(top, bottom, left, right)``.
    """
    top, bottom, left, right = (depth,) * 4 if isinstance(depth, int) else depth
    return (max(0, win.r0 - top), min(h - 1, win.r1 + bottom),
            max(0, win.c0 - left), min(w - 1, win.c1 + right))


def ring_size(win: Window, depth: int | Sequence[int], h: int, w: int) -> int:
    """Number of tokens in ``halo_ring(win, depth, ...)`` (the Halo, sans the window itself)."""
    r0, r1, c0, c1 = _grown_box(win, depth, h, w)
    return (r1 - r0 + 1) * (c1 - c0 + 1) - win.area


def window_tokens(win: Window, w: int, device) -> torch.Tensor:
    """Row-major flat token indices of ``win`` on an ``h x w`` grid."""
    rows = torch.arange(win.r0, win.r1 + 1, device=device)
    cols = torch.arange(win.c0, win.c1 + 1, device=device)
    return (rows[:, None] * w + cols[None, :]).reshape(-1)


def _cover(a: Window, b: Window) -> float:
    """Fraction of the smaller window that the overlap of ``a`` and ``b`` fills."""
    r0, c0, r1, c1 = max(a.r0, b.r0), max(a.c0, b.c0), min(a.r1, b.r1), min(a.c1, b.c1)
    if r1 < r0 or c1 < c0:
        return 0.0
    return (r1 - r0 + 1) * (c1 - c0 + 1) / min(a.area, b.area)


def _dedupe(windows: list[Window]) -> list[Window]:
    """Replace split windows that cover each other by at least ``DEDUPE_COVER`` with their union."""
    split = [win for win in windows if win.split]
    kept = [win for win in windows if not win.split]
    while len(split) >= 2:
        best, pair = DEDUPE_COVER, None
        for i in range(len(split)):
            for j in range(i + 1, len(split)):
                f = _cover(split[i], split[j])
                if f >= best:
                    best, pair = f, (i, j)
        if pair is None:
            break
        b = split.pop(pair[1])
        a = split.pop(pair[0])
        split.append(Window(min(a.r0, b.r0), min(a.c0, b.c0), max(a.r1, b.r1), max(a.c1, b.c1), split=True))
    return kept + split


# ---------------------------------------------------------------------------
# Phase I: the pooling ramp (reference schedule.py)
# ---------------------------------------------------------------------------


def even_starts(total: int, size: int) -> list[int]:
    """Starts of the fewest ``size``-wide windows covering ``total``, spaced evenly.

    Both end windows are pinned (first start 0, last start ``total - size``) so
    the axis stays covered, and the slack between them is spread evenly.
    Python's :func:`round` (banker's rounding) is part of the reference's
    contract — the jittered placement in :mod:`src.area_attn` reproduces these
    exact starts, so rounding them differently would change the draw's support.
    """
    total, size = int(total), int(size)
    if total <= size:
        return [0]
    n = (total - 1) // size + 1
    return [round(i * (total - size) / (n - 1)) for i in range(n)]


def window_count(h: int, w: int, size: int) -> int:
    """N(w) of Eq. 6: base windows of one Phase II step."""
    return len(even_starts(h, size)) * len(even_starts(w, size))


def window_step_cost(h: int, w: int, size: int, txt_len: int = 0) -> float:
    """C(w) of Eq. 7: one windowed step in units of one full-attention pass.

    Every window runs its ``txt_len`` text tokens as queries and keys, so it
    costs ``(size^2 + T)^2`` against ``(h*w + T)^2`` for full attention.
    """
    t = float(txt_len)
    return window_count(h, w, size) * (float(size) ** 2 + t) ** 2 / (float(h * w) + t) ** 2


def grid_divisors(h: int, w: int) -> list[int]:
    """Every divisor of gcd(h, w): the pooling factors that tile the grid exactly."""
    g = math.gcd(int(h), int(w))
    return [d for d in range(1, g + 1) if g % d == 0]


def initial_factor(h: int, w: int, native: int) -> int:
    """f_0 of Eq. 5: the smallest grid divisor >= 2 whose pooled grid fits the native extent.

    This is the reference's contract, ported verbatim, and it **raises** on a
    grid with no common divisor >= 2.  The engine must never raise, so the
    engine-facing entry point is :func:`resolve_factor` — a grid that reaches
    :func:`initial_factor` directly re-introduces the reference's crash.
    """
    divisors = [d for d in grid_divisors(h, w) if d >= 2]
    if not divisors:
        raise ValueError(
            f"the patch grid {h}x{w} has no common divisor >= 2; choose a height and width "
            "whose patch counts share a factor"
        )
    fits = [d for d in divisors if max(h, w) / d <= native]
    return min(fits) if fits else max(divisors)


def coarse_grid(h: int, w: int, f: int) -> tuple[int, int]:
    """The pooled grid ``(ceil(h/f), ceil(w/f))`` — Phase I's canvas for a factor ``f``.

    Ceil, not floor: a factor that does not divide the grid leaves a *partial*
    edge block, and the pooled canvas is one cell larger on that axis.  For a
    factor that tiles the grid exactly this is ``(h//f, w//f)``, so the
    tiling path is unchanged.
    """
    f = max(1, int(f))
    return (-(-int(h) // f), -(-int(w) // f))


def resolve_factor(h: int, w: int, native: int) -> tuple[int, bool]:
    """The engine's ``f_0`` — the factor of the pooling ramp, and whether it was *adjusted*.

    Never raises: any grid gets a factor.  ``(factor, adjusted)`` is

    * ``(f_0, False)`` on a grid with a common divisor >= 2 — exactly
      :func:`initial_factor`, byte for byte;
    * ``(f, True)`` on a grid with none: the factor is then the smallest one
      ``>= 2`` whose pooled grid still fits the native extent (Eq. 5's extent
      rule without the divisibility constraint), so the pooling blocks no
      longer tile the grid exactly and the last block on a partial axis is a
      *partial* block.

    The canvas, the token stream and the latent are untouched either way — only
    the block partition of the pooling changes.
    """
    h, w = int(h), int(w)
    divisors = [d for d in grid_divisors(h, w) if d >= 2]
    if divisors:                                    # the reference's Eq. 5, unchanged
        fits = [d for d in divisors if max(h, w) / d <= native]
        return (min(fits) if fits else max(divisors)), False
    native = max(1, int(native))
    return max(2, -(-max(h, w) // native)), True


def legal_factors(h: int, w: int, f0: int, adjusted: bool) -> list[int]:
    """The factor set the ramp may draw from.

    On a grid that tiles, that is the grid's own divisors up to ``f_0`` — the
    reference's set, unchanged.  On an adjusted grid every block size is
    legal (the partial edge block covers the remainder), so the set is every
    integer in ``[2, f_0]``.
    """
    if not adjusted:
        return [d for d in grid_divisors(h, w) if d <= f0]
    return list(range(2, int(f0) + 1))


def pooling_ramp(f0: int, steps: int, h: int, w: int, f_min: int = F_MIN,
                  factors: Sequence[int] | None = None) -> list[int]:
    """Eqs. 2 and 4: f decreases linearly from f_0 to f_min, snapped to non-increasing legal factors.

    ``factors`` is the *legal* factor set the ramp snaps to; ``None`` keeps the
    reference's set (the grid divisors up to ``f_0``).  On a grid that does not
    tile, the caller passes every legal block size instead
    (:func:`legal_factors`), and the ramp's shape is unchanged.
    """
    if factors is None:
        divisors = [d for d in grid_divisors(h, w) if d <= f0]
    else:
        divisors = sorted({int(d) for d in factors if int(d) <= int(f0)})
    if not divisors:
        divisors = [int(f0)]
    if f_min not in divisors:
        f_min = min(divisors, key=lambda d: (abs(d - f_min), -d))
    if steps <= 1:
        return [f0]
    plan, prev = [], f0
    for target in np.linspace(float(f0), float(f_min), steps):
        candidates = [d for d in divisors if d <= prev]
        plan.append(min(candidates, key=lambda d: (abs(d - target), -d)))
        prev = plan[-1]
    plan[0], plan[-1] = f0, f_min
    return plan


def start_window(h: int, w: int, lo: int, txt_len: int = 0) -> int:
    """w_start of Eq. 8: the largest window whose step is cheaper than full attention.

    ``lo`` is the floor — Phase II must not start below the pooled grid Phase I
    handed over (``extent // f_last``).  Falls back to ``lo`` when no size below
    the full grid is both cheaper than full attention and split into >1 window.
    """
    for size in range(max(h, w), lo - 1, -1):
        if window_step_cost(h, w, size, txt_len) < 1.0 and window_count(h, w, size) > 1:
            return size
    return lo


def is_short_schedule(total: int) -> bool:
    """Whether ``total`` denoising steps is a short (distilled) schedule.

    The paper's calibration is a 28-step FLUX run; Z-Image Turbo's 8-step table is
    a different regime, and every "short schedule" rule in the engine (the Phase I
    regime advisory, the Phase II bridge floor and the short-schedule geometry
    overrides) gates on this one predicate so they cannot disagree.
    """
    return 0 < int(total) <= SHORT_SCHEDULE_TOTAL


def _geometric_sizes(steps: int, start: int, floor: int) -> list[int]:
    if steps == 1:
        return [start]
    sizes = [
        max(floor, int(round(np.exp(np.log(start) + float(t) * (np.log(floor) - np.log(start))))))
        for t in np.linspace(0.0, 1.0, steps)
    ]
    sizes[0], sizes[-1] = start, floor
    return sizes


def window_sizes(h: int, w: int, steps: int, start: int, floor: int, txt_len: int = 0,
                 hold: int = 0) -> list[int]:
    """Eqs. 10 and 11: geometric shrink from w_start to w_min, each rung snapped down so it
    never costs more than the first one and sizes strictly decrease.

    ``hold`` repeats the START rung ``hold`` extra times before the ramp shrinks
    (clamped so the shortened ramp still spans at least two rungs, and never
    negative).  On a short schedule the leading fine steps are the
    composition-critical ones, so a start-heavy ramp hands them the budget the
    20-step geometric ramp would have spread over the whole phase
    (``docs/2026-10-18-area-zimage-adaptation-research.md`` §4-A4).  ``hold=0``
    reproduces the reference's ramp exactly.
    """
    if steps <= 0:
        return []
    hold = max(0, min(int(hold), max(0, steps - 2)))
    if hold:
        return [start] * hold + window_sizes(h, w, steps - hold, start, floor, txt_len)
    ideal = _geometric_sizes(steps, start, floor)
    if start - floor < steps - 1:
        return ideal
    cap = window_step_cost(h, w, start, txt_len)
    sizes: list[int] = []
    for i, target in enumerate(ideal):
        lo = floor + steps - i - 1
        hi = start if not sizes else min(int(target), sizes[-1] - 1)
        hi = max(hi, lo)
        pick = next((s for s in range(hi, lo - 1, -1) if window_step_cost(h, w, s, txt_len) <= cap + 1e-12),
                    None)
        if pick is None:
            pick = min(range(lo, hi + 1), key=lambda s: window_step_cost(h, w, s, txt_len))
        sizes.append(pick)
    return sizes


# ---------------------------------------------------------------------------
# The schedule (reference schedule.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AreaSchedule:
    """What every denoising step runs: Phase I with pooling factor f, or Phase II with window side w."""

    grid: tuple[int, int]
    factors: tuple[int, ...]
    windows: tuple[int, ...]

    @property
    def steps(self) -> int:
        return len(self.factors) + len(self.windows)

    def at(self, step: int) -> tuple[str, int]:
        """``("coarse", f)`` for the first steps, then ``("fine", w)``.

        An index past the end holds the last rung, so a sampler that runs more
        steps than the schedule was built for degrades instead of raising.  (The
        reference indexes ``windows[-1]`` unconditionally, which raises
        ``IndexError`` on an all-coarse schedule; this port reports the last
        factor instead.)
        """
        if step < len(self.factors):
            return "coarse", self.factors[step]
        if self.windows:
            return "fine", self.windows[min(step - len(self.factors), len(self.windows) - 1)]
        if self.factors:
            return "coarse", self.factors[-1]
        raise ValueError("the schedule covers no denoising steps")


def build_schedule(
    h: int, w: int, steps: int, phase1_steps: int, native: int, w_min: int | None = None,
    txt_len: int = 0, window_start: int = 0, window_hold_steps: int = 0, delay: int = 0,
) -> AreaSchedule:
    """Schedule for an ``h x w`` patch grid over ``steps`` denoising steps.

    ``phase1_steps`` is clamped to ``[0, steps]`` (the node clamps its widget the
    same way, and 0 means "all Phase II"), ``w_min`` defaults to the native
    extent, and the Phase I/II handoff is the pooled grid the last ramp rung
    leaves behind (``extent // f_last``) — Phase II never starts below it.

    ``window_start`` / ``window_hold_steps`` are the short-schedule Phase II
    geometry overrides (§4-A4 of the adaptation research): a start side to begin
    the ramp at instead of Eq. 8's cost-optimal one (``0`` = Eq. 8, clamped into
    ``[w_min, extent]``) and extra steps to hold that first side.  Both are gated
    on :func:`is_short_schedule` — on a long schedule the Eq. 8 start is the
    calibrated choice and the two knobs are ignored (zeroed) rather than applied.

    ``delay`` is the Phase I start delay (§12 of the research: the measured 32 px
    ripple pulse lives in the first coarse steps at sigma ~ 1, and pooling from
    step ``delay`` instead of 0 removes it).  Steps before it run factor 1 — the
    engine's own no-pooling representation — so Phase I begins on an established
    canvas.  Gated like the knobs above, and clamped so the ramp never shrinks
    below ``phase1_steps`` pooled rungs.

    The schedule is built for **any** grid: the pooling factor comes from
    :func:`resolve_factor`, which never raises.  On a grid with no common
    divisor >= 2 the factor is the extent rule's, the ramp snaps to the legal
    set (:func:`legal_factors`) and the last pooling block on a partial axis is
    a partial block — the canvas and the token stream are unchanged.
    """
    phase1_steps = max(0, min(int(phase1_steps), int(steps)))
    extent = max(h, w)
    w_min = min(int(w_min or native), extent)
    # Leave at least one pooled rung: delay == phase1_steps would make the last
    # ramp factor 1, and the Phase II handoff reads that factor, so a full
    # delay would silently start Phase II on the whole grid instead of the
    # pooled handoff ("no Phase I at all" is phase1_steps=0 / bootstrap).
    delay = max(0, min(int(delay), phase1_steps - 1)) if is_short_schedule(steps) else 0
    f0, adjusted = resolve_factor(h, w, native)
    ramp = (pooling_ramp(f0, phase1_steps, h, w, factors=legal_factors(h, w, f0, adjusted))
            if phase1_steps else [])
    factors = [1] * delay + ramp[:phase1_steps - delay]
    phase2_steps = steps - phase1_steps
    windows: list[int] = []
    if phase2_steps:
        handoff = extent // (factors[-1] if factors else 1)
        lo = max(w_min, min(handoff, extent))
        if not is_short_schedule(steps):
            # The overrides are a property of SHORT schedules only: a long ramp
            # already starts where the geometry paper calibrated it, so the two
            # knobs are ignored there (never applied, never clamped away).
            window_start, window_hold_steps = 0, 0
        start = (start_window(h, w, lo, txt_len) if int(window_start) <= 0
                 else min(max(int(window_start), w_min), extent))
        windows = window_sizes(h, w, phase2_steps, start, w_min, txt_len, window_hold_steps)
    return AreaSchedule((h, w), tuple(factors), tuple(windows))
