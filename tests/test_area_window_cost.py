"""TSK 2 — the Phase II side selection: the TRUE clipped price and the bridge reserve.

The Phase II side is chosen by a cost model, and the model had two defects that
only show up on a canvas that is NOT square (the Qwen-Image-2.1 canvas the user
runs: 512x2000 px -> a 125x32 token grid):

* **B1 — the square price.**  A window wider than the canvas is CLIPPED by it,
  so it holds ``side x width`` tokens, not ``side^2``.  The reference's square
  price over-prices it by ``side / width``: at 125x32 the side-64 window really
  costs 0.5861 of a full pass and is priced 2.0912 (3.568x).  Under the square
  price NO side qualifies, so the schedule falls back to its floor — the
  literal vertical cut.
* **B2 — no bridge reserve.**  The side the truthful price picks (86 at 125x32)
  cannot afford even ONE bridge token per window, so ``fit_extra_keys`` returns
  R = 0 (and the Halo collapses): the windows lose their only channel to each
  other (the far half of the canvas is read as local).  The side choice must
  RESERVE the bridge/halo budget.

These pins fail when either fix is reverted: the price pin on B1, the side and
the bridge pins on B2.  The frozen profiles (FLUX / Z-Image / Krea-2) keep the
reference's square price and no reserve, bit for bit — the engine picks the
cost model per profile (``src/area.py``), so only Qwen-Image-2.1 is touched.

Markers: @pytest.mark.unit
"""
from __future__ import annotations

import types

import numpy as np
import pytest

import src.area as area
from src.area_arch import FLUX_PROFILE, QWEN21_PROFILE
from src.area_attn import adaptive_tiling, bridge_registers, fit_extra_keys
from src.area_attn import bridge_registers as bridge_registers_from_attn
from src.area_attn import fit_extra_keys as fit_extra_keys_from_attn
from src.area_attn import windowed_step_cost as windowed_step_cost_from_attn
from src.area_schedule import (build_schedule, bridge_registers as bridge_registers_from_schedule,
                                fit_extra_keys as fit_extra_keys_from_schedule,
                                start_window, window_step_cost,
                                windowed_step_cost as windowed_step_cost_from_schedule)

#: The 2.1 canvas: 512x2000 px -> a 125x32 token grid (``gcd == 1``).
H, W = 125, 32
#: The 4K canvas: 4096x4096 px -> a 256x256 token grid.
BIG_H, BIG_W = 256, 256
#: The caption length the design measured on (the 2.1 caption block).
TXT = 256
#: The trained extent in tokens (1024 px) — the node's ``native``.
NATIVE = 64
#: The node's default halo width and bridge ratio.
HALO, RATIO = 8, 0.5


def _layout(h: int, w: int, size: int, txt_len: int = TXT):
    """The Phase II layout of one side: the base windows and their budget."""
    windows = adaptive_tiling(np.zeros((h, w), dtype=np.float32), h, w, size, NATIVE, txt_len,
                              rng=None, allow_splits=False)
    registers, halo = fit_extra_keys(windows, h, w, txt_len,
                                     bridge_registers(size, RATIO, h, w), HALO)
    return windows, registers, halo


@pytest.mark.unit
class TestTheTruePrice:
    """B1 — a clipped window is priced by its TRUE area."""

    def test_a_clipped_window_is_priced_by_its_true_area(self):
        """The side-64 window of a 125x32 canvas holds 64x32 tokens, not 64x64."""
        assert window_step_cost(H, W, 64, TXT) == pytest.approx(0.5861, abs=1e-4)
        assert window_step_cost(H, W, 64, TXT, clipped=False) == pytest.approx(2.0912, abs=1e-4)

    def test_the_square_price_is_three_and_a_half_times_the_true_one(self):
        """The measured over-pricing of the reference's model on this canvas."""
        true = window_step_cost(H, W, 64, TXT)
        square = window_step_cost(H, W, 64, TXT, clipped=False)
        assert square / true == pytest.approx(3.568, abs=1e-3)

    def test_a_square_canvas_prices_the_same_either_way(self):
        """The golden canvases (80x80, 160x160, 65x65) cannot move.

        Only a side that fits the canvas is compared — a window WIDER than the
        canvas is the whole canvas, which the square price cannot see.
        """
        for side in (32, 56, 64, 80):
            assert (window_step_cost(80, 80, side, TXT)
                    == window_step_cost(80, 80, side, TXT, clipped=False))

    def test_the_full_grid_costs_a_full_pass(self):
        """A window that covers the canvas IS a full-attention pass.

        The square price says 13.9 passes for the 125x32 canvas' own full grid
        — the over-pricing this fix removes.
        """
        assert window_step_cost(H, W, max(H, W), TXT) == pytest.approx(1.0)
        assert window_step_cost(H, W, max(H, W), TXT, clipped=False) > 1.0


@pytest.mark.unit
class TestTheBridgeReserve:
    """B2 — the side choice reserves the bridge/halo budget."""

    def test_the_chosen_side_keeps_the_bridge_alive(self):
        """125x32: the chosen side leaves room for bridge tokens and the Halo."""
        schedule = build_schedule(H, W, 28, 1, NATIVE, txt_len=TXT,
                                  clipped=True, halo=HALO, bridge_ratio=RATIO)
        _, registers, halo = _layout(H, W, schedule.windows[0])
        assert registers >= 1
        assert halo >= HALO

    def test_the_side_without_the_reserve_would_starve_the_bridge(self):
        """The side the truthful price alone picks (86) leaves R = 0 and no Halo."""
        start = start_window(H, W, NATIVE, TXT, clipped=True)
        _, registers, halo = _layout(H, W, start)
        assert registers == 0
        assert halo < HALO

    def test_the_reserve_is_what_changes_the_side(self):
        """The reserve rejects the starving side; the price rejects every side.

        Reverting B1 (the square price) collapses both sides onto the floor;
        reverting B2 (the reserve) makes them equal.  Either revert fails here.
        """
        reserved = build_schedule(H, W, 28, 1, NATIVE, txt_len=TXT,
                                  clipped=True, halo=HALO, bridge_ratio=RATIO)
        bare = build_schedule(H, W, 28, 1, NATIVE, txt_len=TXT,
                              clipped=True, halo=0, bridge_ratio=0.0)
        assert reserved.windows[0] < bare.windows[0]
        assert reserved.windows[0] != NATIVE

    def test_the_4k_canvas_keeps_its_bridge(self):
        """256x256: the leading fine step leaves room for the bridge too."""
        schedule = build_schedule(BIG_H, BIG_W, 28, 1, NATIVE, txt_len=TXT,
                                  clipped=True, halo=HALO, bridge_ratio=RATIO)
        _, registers, halo = _layout(BIG_H, BIG_W, schedule.windows[0])
        assert registers >= 1
        assert halo >= HALO

    def test_a_bridge_that_is_off_reserves_nothing(self):
        """With no bridge declared the side choice is the plain cost choice."""
        assert (start_window(H, W, NATIVE, TXT, clipped=True, halo=0, bridge_ratio=0.0)
                == start_window(H, W, NATIVE, TXT, clipped=True))


@pytest.mark.unit
class TestTheFrozenProfiles:
    """The frozen profiles keep the reference's square price, bit for bit."""

    def test_the_square_price_still_chooses_the_floor_on_the_2_1_canvas(self):
        """The Krea-2 canvas (125x32) keeps the side 64 it has always run."""
        schedule = build_schedule(H, W, 8, 1, NATIVE, txt_len=32,
                                  clipped=False, halo=0, bridge_ratio=0.0)
        assert set(schedule.windows) == {NATIVE}

    def test_the_engine_scopes_the_cost_model_to_the_2_1_profile(self):
        """2.1 gets the truthful price; FLUX keeps the square one."""
        def runtime(profile):
            return types.SimpleNamespace(profile=profile,
                                         config=area.AreaConfig(native=NATIVE, halo=HALO,
                                                                bridge_ratio=RATIO),
                                         schedules={})

        qwen = area._area_schedule(runtime(QWEN21_PROFILE), H, W, 28, 1, TXT)
        flux = area._area_schedule(runtime(FLUX_PROFILE), H, W, 28, 1, TXT)
        assert qwen.windows[0] != flux.windows[0]
        assert flux.windows[0] == NATIVE          # the floor, as today
        assert qwen.windows[0] > flux.windows[0]  # the truthful price picks a real side


@pytest.mark.unit
class TestTheMovedPrimitives:
    """The cost primitives live in ``area_schedule`` and are re-exported."""

    def test_area_attn_re_exports_the_same_objects(self):
        assert bridge_registers_from_attn is bridge_registers_from_schedule
        assert fit_extra_keys_from_attn is fit_extra_keys_from_schedule
        assert windowed_step_cost_from_attn is windowed_step_cost_from_schedule
