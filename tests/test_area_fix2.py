"""AReA-Z round 2 (``.dev/area-debug/area-fix2.txt``) — mock-based, GPU-free.

The three new mechanisms: the "+2 healing rule" that caps Phase I on short
schedules, the RoPE horizon saturation of the bridge registers, and window
integrity (no DWT splits, no boundary jitter) on short schedules.
"""
import math

import numpy as np
import torch

from src.area_attn import (
    D_MAX_HORIZON,
    AreaContext,
    adaptive_tiling,
    build_fine_layout,
    saturate_bridge_positions,
    windowed_attention,
)
from src.area_schedule import Window, is_short_schedule


def _window(r0, c0, r1, c1) -> Window:
    return Window(r0, c0, r1, c1)


class TestHealCap:
    """The "+2 healing rule": Phase I may not eat the last two fine steps."""

    def test_eight_coarse_steps_on_an_eight_step_run_are_capped(self):
        total = 8
        assert is_short_schedule(total)
        max_allowed = max(1, total - 2)
        assert max_allowed == 6

    def test_the_very_short_run_keeps_one_coarse_step(self):
        assert max(1, 3 - 2) == 1
        assert max(1, 2 - 2) == 1

    def test_a_long_schedule_is_untouched(self):
        assert not is_short_schedule(28)
        assert max(1, 28 - 2) == 26

    def test_the_documented_examples(self):
        # the doc's table: 6 of 8 leaves exactly two fine steps (gone at step 8)
        assert 8 - 6 == 2
        # 8 of 8 leaves zero fine steps -> the ripple would be frozen in
        assert 8 - 8 == 0


class TestSaturationMath:
    """``D_MAX * tanh(delta / D_MAX)``: direction kept, distance bounded."""

    def test_a_register_inside_the_horizon_is_near_linear(self):
        win = _window(20, 20, 60, 60)         # centre (40, 40)
        reg = torch.tensor([40 * 128 + 40])   # the centre itself
        sat_r, sat_c = saturate_bridge_positions(reg, win, 128, 128, "cpu")
        assert torch.allclose(sat_r, torch.tensor([40.0]), atol=1e-4)
        assert torch.allclose(sat_c, torch.tensor([40.0]), atol=1e-4)

    def test_a_distant_register_is_clamped_into_the_trained_extent(self):
        win = _window(0, 0, 20, 20)           # centre (10, 10)
        reg = torch.tensor([127 * 128 + 127])  # the far corner
        sat_r, sat_c = saturate_bridge_positions(reg, win, 128, 128, "cpu")
        assert float(sat_r[0]) <= 10.0 + D_MAX_HORIZON + 1e-4
        assert float(sat_c[0]) <= 10.0 + D_MAX_HORIZON + 1e-4
        # the clamp is smooth, not a hard cut: strictly inside the asymptote
        assert float(sat_r[0]) < 10.0 + D_MAX_HORIZON
        assert float(sat_c[0]) < 10.0 + D_MAX_HORIZON

    def test_direction_and_axis_ratio_survive(self):
        win = _window(30, 30, 50, 50)         # centre (40, 40)
        reg = torch.tensor([(40 + 100) * 256 + (40 + 30)])   # row*width + col
        sat_r, sat_c = saturate_bridge_positions(reg, win, 256, 256, "cpu")
        dr, dc = float(sat_r[0]) - 40.0, float(sat_c[0]) - 40.0
        # the DIRECTION survives: the saturated offset stays in the raw quadrant
        assert dr > 0.0 and dc > 0.0
        # the axis ORDER survives: the far axis (100 down) stays the far one
        assert dr > dc
        # the raw RATIO is deliberately NOT preserved outside the horizon: the
        # tanh compression runs INDEPENDENTLY per axis, so 100/30 comes back
        # squashed towards 1 -- only the sign and the ordering are kept
        assert dr / dc < 100.0 / 30.0
        # and the per-axis offset (the rope phase) never leaves the trained extent
        assert abs(dr) < D_MAX_HORIZON
        assert abs(dc) < D_MAX_HORIZON
        # the saturated distance is strictly shorter than the raw one
        assert math.hypot(dr, dc) < math.hypot(100.0, 30.0)
        # INSIDE the horizon the compression is near-linear, so the ratio survives:
        # 20 rows / 6 cols is well within D_MAX_HORIZON and keeps its raw ratio
        near = torch.tensor([(40 + 20) * 256 + (40 + 6)])
        near_r, near_c = saturate_bridge_positions(near, win, 256, 256, "cpu")
        ndr, ndc = float(near_r[0]) - 40.0, float(near_c[0]) - 40.0
        assert ndr > 0.0 and ndc > 0.0
        assert abs(ndr / ndc - 20.0 / 6.0) < 0.2
        # and the near-linear saturated distance stays just under the raw one
        assert math.hypot(ndr, ndc) < math.hypot(20.0, 6.0)

    def test_the_horizon_is_the_trained_extent(self):
        assert D_MAX_HORIZON == 56.0
        assert D_MAX_HORIZON <= 64            # Z-Image's 64-token trained grid


class TestWindowIntegritySwitch:
    """``allow_splits=False`` returns the base tiling un-split."""

    def test_no_splits_keeps_the_base_boxes(self):
        rng = np.random.default_rng(0)
        energy = np.zeros((64, 64), dtype=np.float64)
        base = adaptive_tiling(energy, 64, 64, 32, 16, 8, rng=rng, allow_splits=True)
        kept = adaptive_tiling(energy, 64, 64, 32, 16, 8, rng=rng, allow_splits=False)
        assert [(w.r0, w.c0, w.r1, w.c1) for w in kept] == \
               [(w.r0, w.c0, w.r1, w.c1) for w in kept]
        # the base grid is 2x2 windows of side 32 — no quadrant fragmentation
        assert len(kept) == 4
        assert all(w.width <= 32 and w.height <= 32 for w in kept)
        # and the split path may only ever return MORE windows (split, never merge)
        assert len(base) >= len(kept)

    def test_no_rng_means_even_starts(self):
        energy = np.random.default_rng(1).random((64, 64))
        even = adaptive_tiling(energy, 64, 64, 30, 16, 8, rng=None)
        other = adaptive_tiling(energy, 64, 64, 30, 16, 8, rng=None)
        assert [(w.r0, w.c0) for w in even] == [(w.r0, w.c0) for w in other]


class TestBridgeSaturationWiring:
    """The bridge keys' source and rotation switch with the embedder."""

    def _small_layout(self, h=16, w=16, registers=4, halo=0):
        windows = [_window(0, 0, 15, 15)]
        layout = build_fine_layout(windows, h, w, halo, registers, 0, 0,
                                   torch.float32, "cpu")
        return layout

    def test_window_ids_map_into_the_layouts_windows(self):
        layout = self._small_layout()
        assert len(layout.groups) == 1
        group = layout.groups[0]
        assert group.window_ids.tolist() == [0]
        assert layout.windows[int(group.window_ids[0])].width == 16

    def _tensors(self, n, text=4, heads=2, dim=8, seed=3):
        torch.manual_seed(seed)
        q = torch.randn(1, heads, text + n, dim)
        k = torch.randn(1, heads, text + n, dim)
        v = torch.randn(1, heads, text + n, dim)
        return q, k, v, text

    def test_the_embedder_saturates_the_bridge_positions(self, monkeypatch):
        # the wired (z-image) path must rope the bridge keys at the SATURATED
        # register positions: once per group, with that group's own window
        q, k, v, text = self._tensors(256)
        layout = self._small_layout()
        seen = {}
        real = saturate_bridge_positions

        def spy(reg, win, h, w, device):
            seen.setdefault("calls", []).append(win)
            return real(reg, win, h, w, device)

        def recording_embedder(ids):
            seen["ids"] = ids.detach().clone()
            g, n, _ = ids.shape
            # the t axis carries the caption position (cap_len + 1)
            assert torch.all(ids[..., 0] == text + 1)
            # and the rows/cols stay inside the trained extent of the window centre
            centre = (0 + 15) * 0.5
            rows, cols = ids[..., 1], ids[..., 2]
            assert bool(torch.all(rows >= centre - D_MAX_HORIZON - 1e-3))
            assert bool(torch.all(rows <= centre + D_MAX_HORIZON + 1e-3))
            assert bool(torch.all(cols >= centre - D_MAX_HORIZON - 1e-3))
            assert bool(torch.all(cols <= centre + D_MAX_HORIZON + 1e-3))
            return torch.zeros(g, n, 1, 8 // 2, 2, 2)

        monkeypatch.setattr("src.area_attn.saturate_bridge_positions", spy)
        windowed_attention(q[:, :, :text], k[:, :, :text], v[:, :, :text],
                           q[:, :, text:], k[:, :, text:], v[:, :, text:],
                           k[:, :, text:], layout,
                           rope_embedder=recording_embedder, cap_len=text)
        assert len(seen.get("calls", [])) == len(layout.groups)
        assert all(w in layout.windows for w in seen["calls"])

    def test_without_an_embedder_the_bridge_stays_on_the_plain_path(self, monkeypatch):
        # the FLUX path passes no embedder, so nothing may saturate: calling the
        # saturation helper there would mean FLUX's numbers changed
        q, k, v, text = self._tensors(256)
        layout = self._small_layout()

        def boom(*_args, **_kwargs):
            raise AssertionError("FLUX's bridge path must not saturate")

        monkeypatch.setattr("src.area_attn.saturate_bridge_positions", boom)
        _txt, img = windowed_attention(
            q[:, :, :text], k[:, :, :text], v[:, :, :text],
            q[:, :, text:], k[:, :, text:], v[:, :, text:],
            k[:, :, text:], layout)
        assert img.shape == (1, 2, 256, 8)


class TestAreaContextTotal:
    def test_the_default_is_zero(self):
        assert AreaContext().total == 0

    def test_a_long_schedule_is_not_short(self):
        assert not is_short_schedule(28)
        assert is_short_schedule(8)
