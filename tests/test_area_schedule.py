"""Tests for the AReA schedule and Phase II window math (``src/area_schedule.py``).

Plan 2026-10-15, Phase 1 (``docs/2026-10-15-area-aReA-node.md``): the port of the
reference's ``area/schedule.py`` plus the pure-math tiling primitives of
``area/tiling.py``.  What these tests pin:

* **Determinism** — identical inputs give identical schedules; :func:`step_rng`
  is keyed by ``(seed, step, salt)`` and never touches the global RNG, so one
  keyed layout can serve both CFG branches of a single forward (theory.md:496).
* **Ramp invariants** — non-increasing, first rung ``f_0``, last rung the grid
  divisor nearest ``F_MIN`` under the reference's tie rule; ``phase1_steps`` is
  clamped to ``[0, steps]``.
* **Window-cost ceiling** — every produced window side satisfies
  ``window_step_cost <= cost(w_start)`` in the roomy regime the FLUX defaults
  live in.  The ceiling is a per-rung *search* result, not a theorem: shrinking a
  window can add a row/column of windows and cost *more* (theory.md:142), so in
  the cramped regime a rung is forced past the ceiling and the port keeps the
  reference's behaviour instead of silently inflating windows — pinned by
  :class:`TestWindowSizesCaveat`.
* **Geometry** — ``even_starts`` parity with hand-computed vectors, full axis
  coverage, the Phase I/II handoff lower bound, and the ``_grown_box`` /
  ``ring_size`` / ``window_tokens`` / ``_cover`` / ``_dedupe`` primitives.
* **Boundary** — a native-size (or smaller) grid still builds a well-formed
  schedule here; the ``max(h, w) <= native`` short-circuit belongs to the caller
  (``src/area.py``, Phase 3), mirroring ``backbones/base.py:59-68``.
* **Errors** — :func:`initial_factor` raises the reference's actionable
  ``ValueError`` on a grid with no common divisor >= 2 (that is the ported
  contract, kept verbatim), while the engine-facing :func:`resolve_factor` and
  :func:`build_schedule` never raise: they fall back to the extent rule, report
  the grid as *adjusted* and pool ``f x f`` blocks with a partial edge block.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_schedule.py
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch

from src.area_schedule import (
    DEDUPE_COVER,
    F_MIN,
    SHORT_SCHEDULE_TOTAL,
    AreaSchedule,
    Window,
    _cover,
    _dedupe,
    _geometric_sizes,
    _grown_box,
    build_schedule,
    coarse_grid,
    even_starts,
    grid_divisors,
    initial_factor,
    is_short_schedule,
    legal_factors,
    pooling_ramp,
    resolve_factor,
    ring_size,
    start_window,
    step_rng,
    window_count,
    window_sizes,
    window_step_cost,
    window_tokens,
)
from src.area import AreaConfig

NATIVE = 64          # FLUX: 64 patches (1024 px) per side
TEXT = 333           # a plausible T5 text-token count


def _grids():
    """Patch grids that share a factor (gcd >= 2) — enough to build a schedule."""
    return [(h, w) for h in range(65, 145, 13) for w in range(65, 145, 11) if math.gcd(h, w) >= 2]


def _handoff_floor(sched: AreaSchedule, txt_len: int) -> int:
    """``lo`` of :func:`build_schedule`: Phase II must not start below the pooled
    grid the last Phase I rung hands over (``extent // f_last``)."""
    extent = max(sched.grid)
    handoff = extent // (sched.factors[-1] if sched.factors else 1)
    return max(min(NATIVE, extent), min(handoff, extent))


@pytest.mark.unit
class TestEvenStarts:
    def test_hand_computed_vectors(self):
        """Parity with hand-computed covers (both end windows pinned, slack spread).

        The plan's Phase 1 bullet illustrates this with "100/32 -> 5 windows";
        the port's rule is ``n = (total - 1) // size + 1``, which gives 4 windows
        for 100/32 and 5 windows for 100/20 — both asserted below.
        """
        assert even_starts(100, 20) == [0, 20, 40, 60, 80]   # exact tiling, 5 windows
        assert even_starts(100, 32) == [0, 23, 45, 68]       # 4 windows: ceil(100/32) == 4
        assert even_starts(65, 32) == [0, 16, 33]
        assert even_starts(10, 32) == [0]                    # one window, clipped
        assert even_starts(64, 64) == [0]

    def test_covers_every_token_of_the_axis(self):
        for total in range(1, 200):
            for size in range(1, 70):
                starts = even_starts(total, size)
                assert starts[0] == 0 and starts[-1] + size >= total
                assert len(starts) == (total - 1) // size + 1
                covered = {t for s in starts for t in range(s, min(total, s + size))}
                assert covered == set(range(total)), (total, size, starts)

    def test_starts_are_spaced_evenly(self):
        starts = even_starts(100, 32)
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        assert max(gaps) - min(gaps) <= 1


@pytest.mark.unit
class TestWindowCost:
    def test_hand_computed(self):
        """N(w) of Eq. 6 and C(w) of Eq. 7 on a 128x128 grid with no text tokens."""
        assert window_count(128, 128, 32) == 16                       # 4 x 4 windows
        assert window_step_cost(128, 128, 32, 0) == pytest.approx(0.0625)
        # a full-grid window is exactly one full-attention pass
        assert window_step_cost(128, 128, 128, 0) == pytest.approx(1.0)

    def test_cost_only_widens_with_text(self):
        """Eq. 7: more text tokens can only raise a windowed step's relative cost
        (every window re-runs the text queries/keys).  The claim needs
        ``size^2 <= h*w`` — a window wider than the grid it tiles (N(w) == 1)
        starts at a cost above 1 and text tokens then pull it *down* toward 1."""
        for h, w in _grids():
            for size in (16, 48, 96):
                if size * size > h * w:
                    continue
                costs = [window_step_cost(h, w, size, t) for t in range(0, 2049, 128)]
                assert all(b >= a - 1e-12 for a, b in zip(costs, costs[1:])), (h, w, size)

    def test_txt_len_does_not_move_the_pooling_ramp(self):
        """Only the Phase II windows depend on ``txt_len``; Phase I's ramp is a
        function of the grid alone (Eqs. 2 and 4)."""
        for (h, w), steps, phase1 in itertools.product(_grids(), (28,), (0, 4, 8, 28)):
            base = build_schedule(h, w, steps, phase1, NATIVE)
            wide = build_schedule(h, w, steps, phase1, NATIVE, txt_len=2048)
            assert base.factors == wide.factors
            if base.windows:
                assert wide.windows[0] <= base.windows[0]     # more text -> smaller head start


@pytest.mark.unit
class TestGridDivisors:
    def test_hand_computed(self):
        assert grid_divisors(96, 144) == [1, 2, 3, 4, 6, 8, 12, 16, 24, 48]
        assert grid_divisors(63, 65) == [1]                    # coprime: no factor to pool

    def test_initial_factor_is_the_smallest_fitting_divisor(self):
        assert initial_factor(128, 128, NATIVE) == 2
        assert initial_factor(130, 130, NATIVE) == 5           # 2 does not fit: 130/2 > 64
        assert initial_factor(99, 99, NATIVE) == 3

    def test_no_fitting_divisor_falls_back_to_the_largest(self):
        """``native=1`` fits no divisor, so f_0 is the whole grid (Eq. 5 fallback)."""
        assert initial_factor(128, 128, 1) == 128

    def test_coprime_grid_raises_an_actionable_error(self):
        """The reference's contract, ported verbatim: ``initial_factor`` raises.

        The *engine* must never raise, so it goes through
        :func:`resolve_factor` instead (see :class:`TestResolveFactor`).
        """
        with pytest.raises(ValueError, match="no common divisor >= 2") as err:
            initial_factor(63, 65, NATIVE)
        message = str(err.value)
        assert "63x65" in message                     # names the offending grid
        assert "share a factor" in message            # names the remedy


@pytest.mark.unit
class TestResolveFactor:
    """The engine-facing resolver: a factor for *every* grid, never a raise."""

    def test_matches_initial_factor_on_a_tiling_grid(self):
        for h, w in _grids():
            factor, adjusted = resolve_factor(h, w, NATIVE)
            assert not adjusted
            assert factor == initial_factor(h, w, NATIVE)

    def test_hand_computed(self):
        assert resolve_factor(128, 128, NATIVE) == (2, False)
        assert resolve_factor(130, 130, NATIVE) == (5, False)
        assert resolve_factor(99, 99, NATIVE) == (3, False)
        # no common divisor >= 2: the extent rule takes over and the grid is
        # reported as adjusted (the blocks no longer tile it exactly)
        assert resolve_factor(125, 32, NATIVE) == (2, True)
        assert resolve_factor(63, 65, NATIVE) == (2, True)

    def test_never_raises(self):
        for h in range(1, 20):
            for w in range(1, 20):
                factor, adjusted = resolve_factor(h, w, NATIVE)
                assert factor >= 2
                assert isinstance(adjusted, bool)

    def test_adjusted_factor_keeps_the_pooled_grid_inside_the_native_extent(self):
        """The fallback follows Eq. 5's *extent* rule: pool just enough that
        the coarse grid still fits the native extent."""
        for h, w in [(125, 32), (63, 65), (127, 33), (200, 101), (129, 100)]:
            factor, adjusted = resolve_factor(h, w, NATIVE)
            assert adjusted
            nh, nw = coarse_grid(h, w, factor)
            assert max(nh, nw) <= NATIVE
            assert factor == max(2, -(-max(h, w) // NATIVE))   # the smallest such factor

    def test_adjusted_factor_is_never_larger_than_it_has_to_be(self):
        """A factor of 4 when 3 would do is not the extent rule."""
        for h, w in [(200, 101), (129, 100), (193, 101)]:
            factor, _ = resolve_factor(h, w, NATIVE)
            assert factor >= 3
            assert max(coarse_grid(h, w, factor - 1)) > NATIVE


@pytest.mark.unit
class TestCoarseGrid:
    """The pooled grid: ceil on a partial axis, floor on a tiling one."""

    def test_tiling_grid_is_floor_division(self):
        assert coarse_grid(128, 128, 2) == (64, 64)
        assert coarse_grid(130, 130, 5) == (26, 26)
        assert coarse_grid(96, 144, 3) == (32, 48)

    def test_partial_axis_rounds_up(self):
        assert coarse_grid(125, 32, 2) == (63, 16)      # 125 = 62 blocks + 1 row
        assert coarse_grid(63, 65, 2) == (32, 33)
        assert coarse_grid(10, 10, 3) == (4, 4)

    def test_the_blocks_partition_the_canvas(self):
        """Every canvas cell belongs to exactly one block: no row and no column
        is dropped by the partial edge block."""
        for h, w, f in [(125, 32, 2), (63, 65, 2), (200, 101, 4), (10, 7, 3), (1, 1, 2)]:
            nh, nw = coarse_grid(h, w, f)
            rows = [row for r in range(nh) for row in range(r * f, min(r * f + f, h))]
            cols = [col for c in range(nw) for col in range(c * f, min(c * f + f, w))]
            assert rows == list(range(h))
            assert cols == list(range(w))
            # the blocks cover the canvas exactly: one pooled token per block
            # of the coarse grid, and every canvas cell in exactly one of them
            covered = sum(min(r * f + f, h) - r * f for r in range(nh)) * \
                sum(min(c * f + f, w) - c * f for c in range(nw))
            assert covered == h * w
            assert nh * nw == len(range(nh)) * len(range(nw))

    def test_pooled_token_count(self):
        """The user's grid: 125x32 pooled by 2 -> 63x16 = 1008 of 4000 tokens."""
        nh, nw = coarse_grid(125, 32, 2)
        assert (nh, nw) == (63, 16)
        assert nh * nw == 1008


@pytest.mark.unit
class TestLegalFactors:
    """The factor set the ramp may draw from."""

    def test_tiling_grid_keeps_the_grid_divisors(self):
        assert legal_factors(128, 128, 2, False) == [d for d in grid_divisors(128, 128) if d <= 2]
        assert legal_factors(130, 130, 5, False) == [1, 2, 5]

    def test_adjusted_grid_allows_every_block_size(self):
        assert legal_factors(125, 32, 2, True) == [2]
        assert legal_factors(200, 101, 4, True) == [2, 3, 4]


@pytest.mark.unit
class TestPoolingRamp:
    def test_ends_at_f_min(self):
        """Eqs. 2 and 4: f ramps down to the grid divisor nearest F_MIN."""
        assert pooling_ramp(2, 8, 128, 128) == [2] * 8                 # already at 2
        assert pooling_ramp(5, 8, 130, 130) == [5, 5, 5, 5, 2, 2, 2, 2]
        assert pooling_ramp(5, 8, 195, 195) == [5, 5, 5, 5, 3, 3, 3, 3]  # gcd odd: 2 not a divisor

    def test_pinned_and_snapped(self):
        for h, w in _grids():
            f0 = initial_factor(h, w, NATIVE)
            divisors = [d for d in grid_divisors(h, w) if d <= f0]
            snapped = F_MIN if F_MIN in divisors else min(divisors, key=lambda d: (abs(d - F_MIN), -d))
            for steps in (2, 5, 8, 12):
                plan = pooling_ramp(f0, steps, h, w)
                assert len(plan) == steps
                assert plan[0] == f0                       # starts at f_0
                assert plan[-1] == snapped                 # ends at the snapped f_min
                assert all(b <= a for a, b in zip(plan, plan[1:]))   # non-increasing
                assert all(d in divisors for d in plan)    # always a real grid divisor

    def test_single_step_is_f0(self):
        assert pooling_ramp(5, 1, 130, 130) == [5]
        assert pooling_ramp(5, 0, 130, 130) == [5]

    def test_the_legal_set_replaces_the_grid_divisors(self):
        """On a grid that does not tile, the ramp snaps to the legal set (every
        block size) instead of the grid's divisors — whose only member <= f_0
        would be 1."""
        assert legal_factors(125, 32, 2, False) == [1]     # the tiling set: no factor
        legal = legal_factors(125, 32, 2, True)
        assert pooling_ramp(2, 8, 125, 32, factors=legal) == [2] * 8

    def test_the_ramp_invariants_hold_on_the_legal_set(self):
        for h, w in [(125, 32), (200, 101), (63, 65)]:
            f0, adjusted = resolve_factor(h, w, NATIVE)
            legal = legal_factors(h, w, f0, adjusted)
            for steps in (2, 5, 8, 12):
                plan = pooling_ramp(f0, steps, h, w, factors=legal)
                assert len(plan) == steps
                assert plan[0] == f0                       # starts at f_0
                assert plan[-1] == F_MIN                   # ends at f_min
                assert all(b <= a for a, b in zip(plan, plan[1:]))   # non-increasing
                assert all(d in legal for d in plan)       # always a legal block size

    def test_schedule_clamps_phase1_steps(self):
        """The node clamps its widget the same way; 0 means "all Phase II"."""
        for h, w in _grids():
            assert build_schedule(h, w, 28, -5, NATIVE).factors == ()      # -> 0 coarse steps
            assert len(build_schedule(h, w, 28, -5, NATIVE).windows) == 28
            all_coarse = build_schedule(h, w, 28, 999, NATIVE)             # -> all coarse steps
            assert len(all_coarse.factors) == 28 and all_coarse.windows == ()
            assert build_schedule(h, w, 28, 0, NATIVE).factors == ()


@pytest.mark.unit
class TestAdjustedSchedule:
    """A grid with no common divisor >= 2 still builds a well-formed schedule."""

    def test_the_users_grid_builds(self):
        """512x2000 px -> a 125x32 patch grid (gcd 1)."""
        sched = build_schedule(125, 32, 28, 8, NATIVE)
        assert sched.factors == (2,) * 8
        assert len(sched.windows) == 20
        assert sched.at(0) == ("coarse", 2)
        assert sched.at(8) == ("fine", sched.windows[0])

    def test_never_raises(self):
        for h in range(1, 17):
            for w in range(1, 17):
                sched = build_schedule(h, w, 28, 8, NATIVE)
                assert len(sched.factors) == 8
                assert len(sched.windows) == 20

    def test_the_tiling_path_is_unchanged(self):
        """The schedule goldens of the tiling path — the fallback must not move
        a grid that already tiles."""
        assert build_schedule(80, 80, 28, 8, NATIVE).factors == (2,) * 8
        assert build_schedule(96, 144, 28, 8, NATIVE).factors == (3, 3, 3, 3, 2, 2, 2, 2)
        assert build_schedule(130, 130, 28, 8, NATIVE).factors == (5, 5, 5, 5, 2, 2, 2, 2)


@pytest.mark.unit
class TestStartWindow:
    def test_hand_computed(self):
        """w_start of Eq. 8 on a 128x128 grid: the largest window cheaper than a
        full-attention step, and text tokens push it down."""
        assert start_window(128, 128, 64, 0) == 90
        assert start_window(128, 128, 64, 64) == 90
        assert start_window(128, 128, 64, 256) == 89
        assert start_window(128, 128, 64, 1024) == 87

    def test_never_below_the_handoff(self):
        """The largest window is exactly the Eq. 8 size, and it never starts
        below the pooled grid Phase I handed over (``extent // f_last``)."""
        for h, w in _grids():
            for steps, phase1 in ((28, 0), (28, 4), (28, 8), (20, 8)):
                for txt_len in (0, TEXT):
                    sched = build_schedule(h, w, steps, phase1, NATIVE, txt_len=txt_len)
                    if not sched.windows:
                        continue
                    lo = _handoff_floor(sched, txt_len)
                    assert sched.windows[0] == start_window(h, w, lo, txt_len)
                    assert sched.windows[0] >= lo
                    assert sched.windows[-1] >= min(NATIVE, max(h, w))    # never below w_min


@pytest.mark.unit
class TestWindowSizesCeiling:
    def test_direct_call_reproduces_the_built_schedule(self):
        """``window_sizes`` on its own is the Phase II half of ``build_schedule``."""
        assert window_sizes(128, 128, 20, 90, 64, 0) == list(build_schedule(128, 128, 28, 8, NATIVE).windows)
        assert window_sizes(128, 128, 0, 90, 64) == []        # no fine steps, no windows

    @pytest.mark.parametrize(
        "h,w,steps,phase1",
        [
            (128, 128, 28, 8),      # FLUX default: 8 coarse + 20 fine steps
            (128, 128, 28, 4),
            (128, 128, 20, 8),
            (128, 128, 28, 20),
            (160, 96, 28, 8),
            (96, 160, 28, 8),
            (96, 128, 28, 8),
            (128, 96, 28, 8),
            (144, 144, 28, 8),
            (120, 160, 30, 8),
            (160, 160, 40, 12),
        ],
    )
    def test_every_window_fits_the_start_cost_ceiling(self, h, w, steps, phase1):
        """Eqs. 10/11: each rung is snapped against ``cost(w_start)``, so the whole
        fine schedule stays within the first rung's budget in the roomy regime."""
        for txt_len in (0, TEXT):
            sched = build_schedule(h, w, steps, phase1, NATIVE, txt_len=txt_len)
            cap = window_step_cost(h, w, sched.windows[0], txt_len)
            assert cap < 1.0 or sched.windows[0] >= max(h, w)
            for size in sched.windows:
                assert window_step_cost(h, w, size, txt_len) <= cap + 1e-12

    def test_rungs_shrink_and_end_at_w_min(self):
        for h, w in _grids():
            for steps, phase1 in ((28, 8), (12, 4), (40, 12)):
                sched = build_schedule(h, w, steps, phase1, NATIVE, txt_len=TEXT)
                sizes = list(sched.windows)
                assert sizes, (h, w)
                assert all(b <= a for a, b in zip(sizes, sizes[1:]))       # non-increasing
                assert sizes[-1] == min(NATIVE, max(h, w))                 # ends at w_min
                assert sizes[0] == start_window(h, w, _handoff_floor(sched, TEXT), TEXT)


@pytest.mark.unit
class TestWindowSizesCaveat:
    def test_a_rung_never_costs_more_than_its_admissible_range(self):
        """The honest form of the ceiling: a rung is forced past ``cost(w_start)``
        when the strictly-decreasing constraint leaves no cheaper size, but it is
        never worse than the CHEAPEST size its rung allows (the reference's
        ``pick is None`` fallback, ``schedule.py:102-103``).  Shrinking a window
        can add a whole row/column of windows, so smaller is not always cheaper
        (theory.md:142)."""
        for h, w in _grids():
            for steps, phase1 in ((28, 8), (10, 2), (40, 12)):
                sched = build_schedule(h, w, steps, phase1, NATIVE, txt_len=TEXT)
                n = len(sched.windows)
                start, floor_ = sched.windows[0], sched.windows[-1]
                if start - floor_ < n - 1:
                    continue        # cramped: the geometric path ignores cost entirely
                ideal = _geometric_sizes(n, start, floor_)
                cap = window_step_cost(h, w, start, TEXT)
                prev = None
                for i, chosen in enumerate(sched.windows):
                    lo = floor_ + n - i - 1                       # room left for the rest
                    hi = start if prev is None else min(int(ideal[i]), prev - 1)
                    hi = max(hi, lo)
                    options = range(lo, hi + 1)
                    cheapest = min(options, key=lambda s: window_step_cost(h, w, s, TEXT))
                    over = window_step_cost(h, w, chosen, TEXT) > cap + 1e-12
                    if over:
                        assert chosen == cheapest, (h, w, steps, phase1, i, chosen, cheapest)
                    prev = chosen

    def test_cramped_regime_can_exceed_the_start_cost(self):
        """Pinned instance of the ceiling caveat: on a 68x100 grid with 6 fine
        steps the handoff leaves only 5 rungs of room, so the last ones are forced
        past the ceiling rather than inflated past each other."""
        sched = build_schedule(68, 100, 8, 2, NATIVE)
        assert sched.windows[0] == 69
        cap = window_step_cost(68, 100, 69, 0)
        forced = 67                                    # rung 2: no admissible size fits the cap
        assert window_step_cost(68, 100, forced, 0) > cap
        assert forced in sched.windows
        assert list(sched.windows) == sorted(sched.windows, reverse=True)


@pytest.mark.unit
class TestShortScheduleGate:
    """``SHORT_SCHEDULE_TOTAL`` / :func:`is_short_schedule`: the one definition of
    "short schedule" every short-schedule rule in the engine gates on."""

    def test_boundary_of_the_predicate(self):
        assert SHORT_SCHEDULE_TOTAL == 12
        # 0 and negatives are NOT short: they carry no steps to be short on
        assert is_short_schedule(0) is False
        assert is_short_schedule(-3) is False
        for total in range(1, SHORT_SCHEDULE_TOTAL + 1):
            assert is_short_schedule(total) is True
        assert is_short_schedule(SHORT_SCHEDULE_TOTAL + 1) is False
        assert is_short_schedule(28) is False            # the FLUX default


@pytest.mark.unit
class TestWindowStartOverride:
    """A4: ``window_start`` — Phase II begins at a chosen side, not Eq. 8's.

    "Strictly decreasing" is the roomy-regime guarantee: it holds whenever the
    override leaves at least ``fine_steps - 1`` rungs of room between itself and
    ``w_min`` (each rung's admissible lower bound is distinct), and degrades to
    non-increasing in the cramped regime exactly as the un-overridden ramp does.
    """

    def _fine(self, **kwargs):
        """The fine window tuple of a 5-fine-step 128 grid schedule."""
        return list(build_schedule(128, 128, 8, 3, NATIVE, **kwargs).windows)

    @pytest.mark.parametrize("override", [96, 80])
    @pytest.mark.parametrize("txt_len", [0, TEXT])
    def test_override_strictly_decreases_to_w_min(self, override, txt_len):
        sizes = self._fine(txt_len=txt_len, window_start=override)
        assert sizes[0] == override                        # the override leads
        assert all(b < a for a, b in zip(sizes, sizes[1:]))   # strictly decreasing
        assert sizes[-1] == NATIVE                         # ... and ends at w_min
        # the override really changed the ramp, it is not a no-op
        assert sizes != self._fine(txt_len=txt_len)

    def test_override_is_clamped_into_the_grid(self):
        # above the extent: one full-grid window per fine step
        assert self._fine(window_start=4096)[0] == 128
        # below w_min: clamped UP to w_min, never under it
        assert self._fine(window_start=1)[0] == NATIVE

    def test_override_is_ignored_on_a_long_schedule(self):
        """A4's knobs are short-schedule geometry; a 28-step run keeps Eq. 8."""
        plain = build_schedule(128, 128, 28, 3, NATIVE)
        overridden = build_schedule(128, 128, 28, 3, NATIVE,
                                    window_start=96, window_hold_steps=2)
        assert overridden.factors == plain.factors
        assert overridden.windows == plain.windows
        assert plain.windows[0] == start_window(128, 128, 64, 0) == 90


@pytest.mark.unit
class TestWindowHoldSteps:
    """A4: ``hold`` repeats the start rung, clamped so the ramp still runs."""

    def _fine(self, **kwargs):
        return list(build_schedule(128, 128, 8, 3, NATIVE, **kwargs).windows)

    @pytest.mark.parametrize("hold", [1, 2])
    def test_hold_repeats_the_start_rung(self, hold):
        sizes = self._fine(window_hold_steps=hold)
        assert len(sizes) == 5                              # the total is preserved
        assert sizes[:hold + 1] == [sizes[0]] * (hold + 1)  # k extra + the rung itself
        assert sizes[-1] == NATIVE                          # the ramp still lands on w_min
        # the held head IS the un-held schedule's start rung
        assert sizes[0] == self._fine()[0]

    def test_hold_is_clamped_to_two_rungs_of_ramp(self):
        sizes = self._fine(window_hold_steps=99)
        assert len(sizes) == 5
        assert sizes[:4] == [sizes[0]] * 4                  # clamped to steps - 2
        assert sizes[4] == NATIVE
        # the clamp is the documented one: [0, max(0, steps - 2)]
        assert window_sizes(128, 128, 5, 90, 64, 0, 99) == [90] * 3 + window_sizes(128, 128, 2, 90, 64, 0)
        assert window_sizes(128, 128, 5, 90, 64, 0, -4) == window_sizes(128, 128, 5, 90, 64, 0)
        # a two-rung schedule cannot hold at all (steps - 2 == 0)
        assert window_sizes(128, 128, 2, 90, 64, 0, 1) == [90, 64]

    @pytest.mark.parametrize("phase1,txt_len", list(itertools.product((0, 1, 3), (0, 64, 288))))
    def test_zero_knobs_reproduce_the_current_tuples(self, phase1, txt_len):
        """Byte-identical: explicit zeros and the historical call are one thing."""
        plain = build_schedule(128, 128, 8, phase1, NATIVE, txt_len=txt_len)
        zeroed = build_schedule(128, 128, 8, phase1, NATIVE, txt_len=txt_len,
                                window_start=0, window_hold_steps=0)
        assert zeroed.factors == plain.factors
        assert zeroed.windows == plain.windows
        # ... and so is a NEGATIVE window_start (the "unset" spelling)
        assert (build_schedule(128, 128, 8, phase1, NATIVE, txt_len=txt_len,
                               window_start=-1).windows == plain.windows)


@pytest.mark.unit
class TestBuildSchedule:
    def test_is_a_pure_function_of_its_inputs(self):
        for h, w in _grids():
            kwargs = dict(native=NATIVE, txt_len=TEXT)
            one = build_schedule(h, w, 28, 8, **kwargs)
            two = build_schedule(h, w, 28, 8, **kwargs)
            assert repr(one) == repr(two)
            assert one is not two
            assert hash(one) == hash(two)          # frozen: cacheable per forward

    def test_at_maps_steps_to_phase_and_size(self):
        sched = build_schedule(128, 128, 28, 8, NATIVE)
        assert sched.steps == 28
        for step, factor in enumerate(sched.factors):
            assert sched.at(step) == ("coarse", factor)
        for step, size in enumerate(sched.windows, start=len(sched.factors)):
            assert sched.at(step) == ("fine", size)

    def test_at_holds_the_last_rung_past_the_end(self):
        sched = build_schedule(128, 128, 28, 8, NATIVE)
        assert sched.at(27) == ("fine", sched.windows[-1])
        assert sched.at(999) == ("fine", sched.windows[-1])   # over-run degrades, never raises
        # an all-coarse schedule (phase1_steps == steps) holds its last factor
        all_coarse = build_schedule(128, 128, 4, 4, NATIVE)
        assert all_coarse.windows == () and all_coarse.at(99) == ("coarse", 2)
        # a schedule that covers no step at all has no phase to report
        with pytest.raises(ValueError, match="no denoising steps"):
            build_schedule(128, 128, 0, 0, NATIVE).at(0)

    def test_native_grid_still_builds_a_schedule(self):
        """The short-circuit at or below the native extent belongs to the caller
        (``src/area.py``); here a native grid degrades to one full-grid window per
        fine step, which is a no-op graph, not an error."""
        for h, w in ((64, 64), (48, 48), (64, 32)):
            sched = build_schedule(h, w, 28, 8, NATIVE)
            assert sched.steps == 28
            assert set(sched.windows) <= {max(h, w)}
            assert sched.at(27) == ("fine", max(h, w))

    def test_grid_is_carried(self):
        sched = build_schedule(160, 96, 28, 8, NATIVE)
        assert sched.grid == (160, 96)


@pytest.mark.unit
class TestWindowGeometry:
    def test_hand_computed_box_and_ring(self):
        win = Window(2, 3, 5, 7)                       # 4x5 box on an 8x10 grid
        assert (win.height, win.width, win.area) == (4, 5, 20)
        assert _grown_box(win, 1, 8, 10) == (1, 6, 2, 8)
        assert ring_size(win, 1, 8, 10) == 6 * 7 - 20 == 22
        # per-side depths (top, bottom, left, right), clipped to the grid
        assert _grown_box(win, (1, 2, 3, 4), 8, 10) == (1, 7, 0, 9)
        assert ring_size(win, (1, 2, 3, 4), 8, 10) == 7 * 10 - 20 == 50
        # growing past the edge clips: the ring is the rest of the grid
        assert _grown_box(win, 9, 8, 10) == (0, 7, 0, 9)
        assert ring_size(win, 9, 8, 10) == 8 * 10 - 20 == 60

    def test_window_spanning_the_grid_has_no_ring(self):
        win = Window(0, 0, 7, 9)
        assert ring_size(win, 1, 8, 10) == 0

    def test_window_tokens_are_row_major(self):
        win = Window(2, 3, 5, 7)
        tokens = window_tokens(win, 10, "cpu")
        assert tokens.dtype == torch.long
        assert tokens.numel() == win.area
        assert tokens[0].item() == 2 * 10 + 3
        assert tokens[-1].item() == 5 * 10 + 7
        assert torch.equal(tokens, (torch.arange(2, 6)[:, None] * 10 + torch.arange(3, 8)[None, :]).reshape(-1))

    def test_cover_is_the_smaller_side_ratio(self):
        assert _cover(Window(0, 0, 3, 3), Window(2, 2, 5, 5)) == pytest.approx(0.25)
        assert _cover(Window(0, 0, 3, 3), Window(0, 0, 2, 2)) == pytest.approx(1.0)
        assert _cover(Window(0, 0, 3, 3), Window(5, 5, 6, 6)) == 0.0

    def test_dedupe_merges_overlapping_splits_only(self):
        assert DEDUPE_COVER == 0.8
        merged = _dedupe([Window(0, 0, 3, 3), Window(0, 0, 3, 3, split=True), Window(0, 0, 2, 2, split=True)])
        assert merged == [Window(0, 0, 3, 3), Window(0, 0, 3, 3, split=True)]   # union of the split pair
        apart = _dedupe([Window(0, 0, 3, 3, split=True), Window(10, 10, 11, 11, split=True)])
        assert apart == [Window(0, 0, 3, 3, split=True), Window(10, 10, 11, 11, split=True)]
        # non-split windows are never merged away
        assert _dedupe([Window(0, 0, 3, 3), Window(0, 0, 2, 2)]) == [Window(0, 0, 3, 3), Window(0, 0, 2, 2)]


@pytest.mark.unit
class TestStepRng:
    def test_keyed_by_seed_step_and_salt(self):
        def draws(seed, step, salt):
            return step_rng(seed, step, salt).integers(0, 1000, size=8).tolist()

        assert draws(7, 3, 0x4A10) == draws(7, 3, 0x4A10)
        assert draws(7, 3, 0x4A10) != draws(7, 4, 0x4A10)     # a new step is a new stream
        assert draws(7, 3, 0x4A10) != draws(7, 3, 0x5EA1)     # ... and so is a new salt
        assert draws(7, 3, 0x4A10) != draws(8, 3, 0x4A10)

    def test_does_not_touch_the_global_stream(self):
        """Cond and uncond share one forward; both branches must draw the SAME
        layout, so no draw may perturb the global RNG (theory.md:496)."""
        np.random.seed(1234)
        before = np.random.get_state()
        step_rng(7, 3, 0x4A10).integers(0, 1000, size=8)
        again = step_rng(7, 3, 0x4A10).integers(0, 1000, size=8)
        after = np.random.get_state()
        assert before[1].tolist() == after[1].tolist() and before[2] == after[2]
        assert again.tolist() == step_rng(7, 3, 0x4A10).integers(0, 1000, size=8).tolist()

    def test_large_seed_is_masked(self):
        assert (step_rng(2 ** 40, 0, 0).bit_generator.seed_seq.entropy
                == step_rng(0, 0, 0).bit_generator.seed_seq.entropy)


class TestPhaseIDelay:
    """The Phase I start delay (``delay``) — engine-level, DEFAULT OFF (§13/§14).

    The knob measured 2.3-4.4x lower final 32px energy but was visually WORSE on
    the real checkpoint (more hallucination): pooling introduced mid-composition
    hurts. So its default is 0 in ``AreaConfig`` and it ships opt-in; these tests
    pin both the schedule semantics and the "never a default regression" rule.
    """

    def test_default_is_off(self):
        """The shipped default never shifts the ramp (no silent behavior change)."""
        assert AreaConfig().delay == 0
        plain = build_schedule(128, 128, 10, 8, 64, txt_len=64)
        assert build_schedule(128, 128, 10, 8, 64, txt_len=64).factors == plain.factors

    def test_delay_shifts_the_pooled_ramp(self):
        """D full-attention steps then the pooled ramp, same coarse count."""
        s = build_schedule(128, 128, 10, 8, 64, txt_len=64, delay=2)
        assert s.factors == (1, 1) + (2,) * 6

    def test_one_pooled_rung_survives(self):
        """A delay may never eat the whole ramp: the handoff reads the last factor."""
        assert build_schedule(128, 128, 10, 2, 64, txt_len=64, delay=9).factors == (1, 2)
        assert build_schedule(128, 128, 10, 1, 64, txt_len=64, delay=9).factors == (2,)

    def test_long_schedules_ignore_it(self):
        """A long schedule is not delayed — FLUX's calibrated ramp is intact."""
        plain = build_schedule(128, 128, 28, 8, 64, txt_len=64)
        assert build_schedule(128, 128, 28, 8, 64, txt_len=64, delay=4).factors == plain.factors
