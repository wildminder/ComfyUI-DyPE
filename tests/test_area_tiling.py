"""Tests for the AReA Phase II tiling (``src/area_attn.py``).

Plan 2026-10-15, Phase 2 (``docs/2026-10-15-area-aReA-node.md``): the tensor half of the
reference's ``area/tiling.py``.  What these tests pin:

* **DWT detail map** — a flat canvas maps to zeros (which is what keeps the gate alpha
  at 0 on smooth regions), the response sits where the detail is, the Haar's odd-sized
  padding produces a usable ``(h, w)`` map, and the reference's mean over the batch is
  kept so one layout serves cond and uncond.
* **Adaptive tiling** — the windows always cover every token after the jitter and the
  splits, a split is only offered from the low half of the windows that sit below the
  mean (recomputed here from the map, not copied from the implementation), the split
  budget is the gate's ``int(alpha * windows)``, and no output box invents geometry the
  base tiling did not have.
* **Halo** — :func:`halo_ring` and :func:`jagged_halo_ring` return exactly the tokens
  outside the window, the jagged ring keeps the straight ring's token count while its
  reach is randomised, every row of a side is reached, and clipping at the grid edge
  never reaches outside the grid.
* **Merge weights** — a separable raised cosine per window, rescaled so the weights
  covering a token sum to that token's coverage count, which is what makes a token that
  lives in a single window keep weight 1 and what lets a centre token dominate its edge.
* **Determinism** — every draw comes from ``step_rng`` keyed by ``(seed, step, salt)``;
  none of these functions reads the global numpy or torch stream.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_tiling.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import torch

from src.area_attn import (
    BLEND_FLOOR,
    DWT_LEVELS,
    DWT_PAD,
    GATE_HI,
    GATE_LO,
    _side_profile,
    adaptive_tiling,
    detail_map,
    halo_ring,
    jagged_halo_ring,
    jittered_starts,
    merge_weights,
    tiling_rng,
)
from src.area_schedule import Window, even_starts, ring_size, step_rng, window_tokens

GRID = 32
TEXT = 100


def _covered(windows, h: int, w: int) -> np.ndarray:
    """Coverage count of every token of an ``h x w`` grid by ``windows``."""
    cov = np.zeros(h * w, dtype=int)
    for win in windows:
        cov[window_tokens(win, w, "cpu").numpy()] += 1
    return cov


def _energy(rng: np.random.Generator, h: int, w: int, blocks: int = 8) -> np.ndarray:
    """A structured detail map: blocky high-detail patches on a quiet canvas."""
    e = np.zeros((h, w), dtype=np.float32)
    spots = rng.integers(0, max(1, h - blocks), size=(12, 2))
    for r, c in spots:
        e[r : r + blocks, c : c + blocks] = rng.uniform(0.4, 1.0)
    if e.max() == 0:
        e[0, 0] = 1.0
    return e / e.max()


@pytest.mark.unit
class TestDetailMap:
    def test_flat_canvas_maps_to_zeros(self):
        """Eq. 12 subtracts the mean, so a canvas with no detail has no energy at all."""
        for shape in ((8, 8), (16, 10), (13, 7)):
            flat = torch.full((1, shape[0] * shape[1], 6), 0.25)
            e = detail_map(flat, *shape)
            assert float(e.max()) == 0.0 and float(e.min()) == 0.0

    def test_shape_and_dtype_survive_odd_sizes(self):
        """The Haar levels need a multiple of ``2**DWT_LEVELS * DWT_PAD``; the map is padded
        and cropped back, so any grid size produces a usable ``(h, w)`` map."""
        assert (DWT_LEVELS, DWT_PAD) == (3, 8)
        for h, w in ((8, 10), (13, 7), (17, 31), (64, 64)):
            e = detail_map(torch.randn(2, h * w, 5), h, w)
            assert e.shape == (h, w)
            assert e.dtype == np.float32
            assert np.isfinite(e).all()

    def test_max_normalised(self):
        e = detail_map(torch.randn(1, 16 * 16, 8), 16, 16)
        assert e.max() > 0.0
        assert float(e.max()) == pytest.approx(1.0, abs=1e-6)

    def test_response_needs_detail_inside_a_block(self):
        """A token's detail is the spread across its channels.  The Haar pairs (even, odd)
        rows and columns, so a step that lands exactly on a dyadic boundary is invisible while
        the same busier region one column earlier lights the map up."""
        h = w = 16

        def canvas(start: int) -> torch.Tensor:
            c = torch.zeros(1, h * w, 4)
            busy = torch.from_numpy(np.arange(h * w) % w >= start)
            c[0, busy] = torch.tensor([0.0, 1.0, 0.0, 1.0])
            return c

        on_edge = detail_map(canvas(8), h, w)       # the step sits between col 7 and 8
        inside = detail_map(canvas(7), h, w)        # the step sits between col 6 and 7
        assert float(on_edge.sum()) == 0.0
        assert float(inside.sum()) > 0.0
        assert float(inside.max()) > 0.5

    def test_batch_is_reduced_by_the_mean(self):
        """Comfy batches cond and uncond into one forward, so the reference's mean over the
        batch is kept: the per-token channel spread of a doubled batch is the same as one
        half, and the reduction does not care which half comes first."""
        half = torch.randn(1, GRID * GRID, 8)
        other = torch.randn(1, GRID * GRID, 8)
        assert np.allclose(detail_map(half, GRID, GRID), detail_map(torch.cat([half, half]), GRID, GRID))
        assert np.allclose(detail_map(torch.cat([half, other]), GRID, GRID),
                           detail_map(torch.cat([other, half]), GRID, GRID))

    def test_where_the_detail_sits_moves_the_map(self):
        """The same 20 detailed tokens spread over the canvas and packed into its first two
        rows give different maps — the map is a property of the canvas, not of the grid."""
        h = w = 16

        def canvas(idx) -> torch.Tensor:
            c = torch.zeros(1, h * w, 4)
            c[0, torch.as_tensor(idx)] = torch.tensor([0.0, 1.0, 0.0, 1.0])
            return c

        spread = detail_map(canvas(range(0, h * w, 13)), h, w)
        corner = detail_map(canvas(range(20)), h, w)
        assert float(spread.sum()) > 0.0 and float(corner.sum()) > 0.0
        assert not np.allclose(spread, corner)


@pytest.mark.unit
class TestJitteredStarts:
    def test_same_count_and_size_as_even_starts(self):
        """Same count and size as ``even_starts`` with both end windows pinned: the axis
        stays covered no matter where the slack lands."""
        for total, size in ((32, 8), (32, 16), (48, 20), (17, 6)):
            base = even_starts(total, size)
            for seed in range(6):
                starts = jittered_starts(total, size, np.random.default_rng(seed))
                assert len(starts) == len(base)
                assert starts[0] == 0 and starts[-1] == base[-1]
                assert all(0 <= s <= total - size for s in starts)
                assert all(1 <= b - a <= size for a, b in zip(starts, starts[1:]))

    def test_the_slack_is_really_sampled(self):
        """``30/8`` leaves 22 to split into 3 gaps of at most 8, so six placements are valid and
        the draws must use more than one of them — and never leave that set."""
        total, size = 30, 8
        valid = {(0, 6, 14, 22), (0, 8, 14, 22), (0, 8, 16, 22),
                 (0, 7, 14, 22), (0, 7, 15, 22), (0, 8, 15, 22)}
        draws = [tuple(jittered_starts(total, size, np.random.default_rng(seed))) for seed in range(200)]
        assert even_starts(total, size) == [0, 7, 15, 22]
        assert set(draws) <= valid, (sorted(set(draws)), sorted(valid))
        assert len(set(draws)) > 1, draws[:8]

    def test_degenerate_axes_fall_back_to_even_starts(self):
        """A single window, or a full-width one, has no slack to distribute."""
        for total, size in ((16, 16), (8, 16), (24, 24)):
            assert jittered_starts(total, size, np.random.default_rng(0)) == even_starts(total, size)

    def test_keyed_stream_does_not_touch_the_global_rng(self):
        before = np.random.get_state()
        draws = [jittered_starts(GRID, 8, np.random.default_rng(s)) for s in range(4)]
        after = np.random.get_state()
        assert before[1].tolist() == after[1].tolist() and before[2] == after[2]
        assert draws == [jittered_starts(GRID, 8, np.random.default_rng(s)) for s in range(4)]


@pytest.mark.unit
class TestAdaptiveTiling:
    @staticmethod
    def _boxes(size: int) -> list[Window]:
        """The base tiling of side ``size`` on the grid, unclipped windows included."""
        return [Window(r, c, min(GRID - 1, r + size - 1), min(GRID - 1, c + size - 1))
                for r in even_starts(GRID, size) for c in even_starts(GRID, size)]

    def test_no_rng_is_the_even_placement(self):
        """No jitter and no gate on a flat canvas: the base boxes come out in the even order."""
        flat = np.zeros((GRID, GRID), dtype=np.float32)
        windows = adaptive_tiling(flat, GRID, GRID, 16, 8, TEXT, rng=None)
        rows, cols = even_starts(GRID, 16), even_starts(GRID, 16)
        assert [(win.r0, win.c0, win.r1, win.c1) for win in windows] == \
               [(r, c, min(GRID - 1, r + 15), min(GRID - 1, c + 15)) for r in rows for c in cols]

    def test_flat_map_never_splits(self):
        """Gate alpha is 0 when the tile energy has no variation, so the base tiling comes
        back untouched — AReA does not spend tokens on a smooth canvas."""
        flat = np.zeros((GRID, GRID), dtype=np.float32)
        windows = adaptive_tiling(flat, GRID, GRID, 16, 8, TEXT, rng=step_rng(3, 7, 0x5EA1))
        assert len(windows) == len(even_starts(GRID, 16)) ** 2
        assert not any(win.split for win in windows)

    def test_windows_cover_every_token_after_the_splits(self):
        """The full-coverage guarantee: the jitter may move the boxes and the splits may cut
        them, but no token of the grid is ever left unattended."""
        for size, w_min in ((16, 8), (32, 16), (24, 12)):
            for step, seed in ((0, 1), (1, 2), (7, 9)):
                energy = _energy(np.random.default_rng(step * 10 + seed), GRID, GRID)
                windows = adaptive_tiling(energy, GRID, GRID, size, w_min, TEXT,
                                          rng=tiling_rng(seed, step))
                cov = _covered(windows, GRID, GRID)
                assert cov.min() >= 1, (size, step, seed, int((cov == 0).sum()))
                assert all(win.height <= size and win.width <= size for win in windows)

    def test_every_output_box_sits_inside_a_base_box(self):
        """Splits are quadrant cuts and dedupe only ever merges, so no window can reach
        outside the base box it came from."""
        for size in (16, 24):
            energy = _energy(np.random.default_rng(4), GRID, GRID)
            windows = adaptive_tiling(energy, GRID, GRID, size, 8, TEXT, rng=tiling_rng(11, 2))
            base = self._boxes(size)
            for win in windows:
                assert any(b.r0 <= win.r0 and win.r1 <= b.r1 and b.c0 <= win.c0 and win.c1 <= b.c1
                           for b in base), win

    def test_split_budget_is_the_gate_allowance(self):
        """Recomputed from the map: alpha from the tile-energy CV, ``max_splits = int(alpha
        * windows)``, and one split makes four quadrant windows."""
        for size in (16, 32):
            energy = _energy(np.random.default_rng(4), GRID, GRID)
            windows = adaptive_tiling(energy, GRID, GRID, size, 8, TEXT, rng=tiling_rng(11, 2))
            n_boxes = len(self._boxes(size))
            t_mean = float(energy.mean())
            u = float(energy.std() / t_mean) if t_mean > 1e-12 else 0.0
            alpha = min(1.0, max(0.0, (u - GATE_LO) / max(1e-6, GATE_HI - GATE_LO)))
            assert alpha > 0.0, "a structured map must open the gate"
            n_split = sum(1 for win in windows if win.split) // 4
            assert n_split <= int(alpha * n_boxes)

    def test_only_low_energy_windows_are_split(self):
        """Candidates are the low half of the windows *and* below the mean, so the base box
        behind every split sits strictly below the mean window energy of the base tiling."""
        size = 16
        energy = _energy(np.random.default_rng(6), GRID, GRID)
        windows = adaptive_tiling(energy, GRID, GRID, size, 8, TEXT, rng=tiling_rng(5, 3))
        base = self._boxes(size)
        box_e = np.array([energy[w.r0 : w.r1 + 1, w.c0 : w.c1 + 1].mean() for w in base])
        split_box_e = [energy[b.r0 : b.r1 + 1, b.c0 : b.c1 + 1].mean()
                       for b in base for w in windows
                       if w.split and b.r0 <= w.r0 and w.r1 <= b.r1 and b.c0 <= w.c0 and w.c1 <= b.c1]
        assert split_box_e, "a structured map with a 16-wide window must split somewhere"
        assert max(split_box_e) < float(box_e.mean()) + 1e-9

    def test_split_boxes_respect_the_w_min_floor(self):
        """The guard skips a box whose quadrant side would fall below ``w_min // 2``, so a
        ``w_min`` whose half exceeds the quadrant side blocks every split."""
        for size, w_min in ((16, 8), (24, 8)):
            energy = _energy(np.random.default_rng(8), GRID, GRID)
            windows = adaptive_tiling(energy, GRID, GRID, size, w_min, TEXT, rng=tiling_rng(1, 1))
            assert any(win.split for win in windows), (size, w_min)
        for size, w_min in ((16, 32), (8, 24), (16, 33)):
            energy = _energy(np.random.default_rng(8), GRID, GRID)
            windows = adaptive_tiling(energy, GRID, GRID, size, w_min, TEXT, rng=tiling_rng(1, 1))
            assert not any(win.split for win in windows), (size, w_min)

    def test_the_same_seed_and_step_rebuild_the_same_tiling(self):
        """A 12-wide window on this grid leaves slack in the gaps, so the jitter is really drawn
        and the keyed stream must reproduce it (and a new step must not)."""
        energy = _energy(np.random.default_rng(0), GRID, GRID)

        def tiling(step):
            return [(w.r0, w.c0, w.r1, w.c1, w.split)
                    for w in adaptive_tiling(energy, GRID, GRID, 12, 8, TEXT, rng=tiling_rng(4, step))]

        assert tiling(6) == tiling(6)
        assert tiling(6) != tiling(7)


@pytest.mark.unit
class TestHaloRing:
    def test_straight_ring_returns_exactly_the_outside_tokens(self):
        win = Window(4, 6, 9, 11)                       # 6x6 box on a 12x16 grid
        depth = (2, 1, 3, 2)
        ring = halo_ring(win, depth, 12, 16, "cpu")
        assert ring is not None
        grown = {r * 16 + c for r in range(2, 11) for c in range(3, 14)}
        expect = grown - set(window_tokens(win, 16, "cpu").tolist())
        assert set(ring.tolist()) == expect
        assert ring.numel() == ring_size(win, depth, 12, 16) == 63

    def test_depth_zero_and_grid_spanning_windows_have_no_ring(self):
        assert halo_ring(Window(3, 3, 8, 8), 0, 12, 16, "cpu") is None
        assert halo_ring(Window(0, 0, 11, 15), 3, 12, 16, "cpu") is None

    def test_edge_windows_clip_into_the_grid(self):
        win = Window(0, 0, 3, 3)
        ring = halo_ring(win, 5, 12, 16, "cpu")
        assert ring is not None and int(ring.max()) < 12 * 16
        # the grown box is rows 0..8, cols 0..8 (clipped at 3+5): nothing outside it
        assert {int(t) // 16 for t in ring.tolist()} <= set(range(9))
        assert {int(t) % 16 for t in ring.tolist()} <= set(range(9))
        assert not set(ring.tolist()) & set(window_tokens(win, 16, "cpu").tolist())

    def test_jagged_ring_keeps_the_straight_ring_s_size(self):
        """The whole point of the uneven edge: the reach is randomised, the token count is not."""
        win = Window(4, 6, 9, 11)
        depth = (3, 2, 2, 4)
        for step in range(6):
            ring = jagged_halo_ring(win, depth, 5, 12, 16, step_rng(4, step, 0x4A10), "cpu")
            assert ring is not None
            assert ring.numel() == ring_size(win, depth, 12, 16), step
            assert ring.min() >= 0 and ring.max() < 12 * 16
            assert not set(ring.tolist()) & set(window_tokens(win, 16, "cpu").tolist())

    def test_jagged_ring_reaches_every_row_of_each_side(self):
        """Every row (left, right) or column (top, bottom) of a side takes part, so the ring is
        a ring and not just the corners between two straight edges."""
        win = Window(4, 6, 9, 11)
        ring = jagged_halo_ring(win, (4, 4, 4, 4), 6, 12, 16, step_rng(9, 1, 0x4A10), "cpu")
        assert ring.dtype == torch.long
        assert {int(t) // 16 for t in ring.tolist()} >= set(range(4))       # top side, 4 rows
        assert {int(t) % 16 for t in ring.tolist()} >= set(range(6))        # left side, 6 columns

    def test_side_profile_sums_and_stays_inside_the_cap(self):
        for cap in (0, 1, 3, 7):
            for length in (1, 2, 5):
                for step in range(5):
                    total = int(np.random.default_rng(step).integers(0, cap * length + 1))
                    d = _side_profile(total, length, cap, step_rng(2, step, 0x4A10))
                    assert int(d.sum()) == total
                    assert d.min() >= 0 and d.max() <= cap
                    assert len(d) == length

    def test_rings_are_identical_for_the_same_step_and_differ_across_steps(self):
        win = Window(2, 2, 7, 7)
        one = jagged_halo_ring(win, (2, 2, 2, 2), 4, 16, 16, step_rng(7, 1, 0x4A10), "cpu")
        two = jagged_halo_ring(win, (2, 2, 2, 2), 4, 16, 16, step_rng(7, 1, 0x4A10), "cpu")
        other = jagged_halo_ring(win, (2, 2, 2, 2), 4, 16, 16, step_rng(7, 2, 0x4A10), "cpu")
        assert torch.equal(one, two)
        assert not torch.equal(one, other)


@pytest.mark.unit
class TestMergeWeights:
    def test_a_lone_window_is_a_plain_average(self):
        """One window over a token means weight 1: the merge cannot dim what nothing overlaps."""
        win = Window(0, 0, 7, 7)
        weight = merge_weights([win], [window_tokens(win, 8, "cpu")], 64, torch.float32, "cpu")[0]
        assert weight.shape == (64,)
        assert torch.allclose(weight, torch.ones(64))
        assert weight.dtype == torch.float32

    def test_weights_are_centre_weighted(self):
        """Two windows offset horizontally share the middle columns; the take of a shared token
        goes to the window whose centre is nearer."""
        h, w = 8, 12
        wins = [Window(0, 0, 7, 7), Window(0, 4, 7, 11)]
        tokens = [window_tokens(x, w, "cpu") for x in wins]
        weights = merge_weights(wins, tokens, h * w, torch.float32, "cpu")

        def share(win: int, pos: int) -> float:
            return float(weights[win][tokens[win].tolist().index(pos)])

        for row in (0, 3, 7):
            left, right = row * w + 5, row * w + 6
            assert share(0, left) > share(1, left) > BLEND_FLOOR
            assert share(1, right) > share(0, right) > BLEND_FLOOR
            assert share(0, left) + share(1, left) == pytest.approx(2.0)

    def test_every_token_of_a_window_is_weighted(self):
        """A token covered once keeps weight 1; a token covered twice is shared between the
        windows that cover it and nothing is lost (the sum invariant is the next test)."""
        wins = [Window(0, 0, 3, 3), Window(0, 0, 2, 2), Window(4, 4, 7, 7)]
        tokens = [window_tokens(x, 8, "cpu") for x in wins]
        weights = merge_weights(wins, tokens, 64, torch.float32, "cpu")
        cov = Counter(t for tk in tokens for t in tk.tolist())
        for idx, w in zip(tokens, weights):
            for pos, tok in enumerate(idx.tolist()):
                if cov[tok] == 1:
                    assert float(w[pos]) == pytest.approx(1.0)
                else:
                    assert 0.0 < float(w[pos]) <= 2.0

    def test_weights_covering_a_token_sum_to_its_coverage_count(self):
        wins = [Window(0, 0, 5, 5), Window(3, 3, 8, 8)]
        tokens = [window_tokens(w, 9, "cpu") for w in wins]
        weights = merge_weights(wins, tokens, 81, torch.float32, "cpu")
        total = torch.zeros(81)
        for idx, w in zip(tokens, weights):
            total.index_add_(0, idx, w)
        cov = Counter(t for tk in tokens for t in tk.tolist())
        overlap = set(tokens[0].tolist()) & set(tokens[1].tolist())
        assert overlap
        for tok in range(81):
            assert float(total[tok]) == pytest.approx(float(cov.get(tok, 0)))

    def test_dtype_is_cast(self):
        win = Window(0, 0, 3, 9)
        w = merge_weights([win], [window_tokens(win, 10, "cpu")], 100, torch.bfloat16, "cpu")[0]
        assert w.dtype == torch.bfloat16 and w.shape == (win.area,)


@pytest.mark.unit
class TestNoGlobalRngReads:
    def test_tiling_never_reads_the_global_streams(self):
        """Cond and uncond share one forward, so both branches must be able to rebuild the
        same layout — which rules out the global numpy and torch streams (theory.md:496)."""
        np_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
        energy = _energy(np.random.default_rng(0), GRID, GRID)
        adaptive_tiling(energy, GRID, GRID, 16, 8, TEXT, rng=tiling_rng(2, 3))
        jittered_starts(GRID, 8, tiling_rng(2, 3))
        halo_ring(Window(2, 2, 7, 7), (2, 2, 2, 2), GRID, GRID, "cpu")
        jagged_halo_ring(Window(2, 2, 7, 7), (2, 2, 2, 2), 4, GRID, GRID,
                         step_rng(2, 3, 0x4A10), "cpu")
        assert np_state[1].tolist() == np.random.get_state()[1].tolist()
        assert np_state[2] == np.random.get_state()[2]
        assert torch.equal(torch_state, torch.random.get_rng_state())

    def test_tiling_rng_is_keyed_by_step(self):
        first = tiling_rng(5, 1).integers(0, 1000, size=4).tolist()
        assert first == tiling_rng(5, 1).integers(0, 1000, size=4).tolist()
        assert first != tiling_rng(5, 2).integers(0, 1000, size=4).tolist()
