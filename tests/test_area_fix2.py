"""AReA round 2 (the second hardening round) — mock-based, GPU-free.

The three new mechanisms, pinned BEHAVIOURALLY: every test below drives the
real engine code (the resolver, the layout builder, the windowed attention) and
asserts what the engine PRODUCES — never a literal, never two calls compared
against each other.

* mechanism 1 — the "+2 healing rule": the resolver caps the Phase I count so
  that at least two fine steps remain on a short schedule
  (``src.area._area_resolve_forward``);
* mechanism 2 — the RoPE horizon saturation of the bridge registers, the
  group -> window mapping it is computed against, and the tensor the re-roped
  bridge keys are taken from (``src.area_attn``);
* mechanism 3 — window integrity on short schedules: the base tiling, unsplit
  and un-jittered (``src.area._area_fine_layout``).

Markers: @pytest.mark.unit
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import src.area as area
from src.area import AreaConfig
from src.area_attn import (
    D_MAX_HORIZON,
    AreaContext,
    FineLayout,
    WindowGroup,
    adaptive_tiling,
    apply_rope1,
    build_fine_layout,
    saturate_bridge_positions,
    select_registers,
    token_importance,
    windowed_attention,
)
from src.area_schedule import Window, even_starts, is_short_schedule

# ---------------------------------------------------------------------------
# Mocks (comfy-faithful, GPU-free) — the same shapes as tests/test_area_node.py
# ---------------------------------------------------------------------------

TXT = 4              # caption tokens of the tiny streams
HEADS, DIM = 2, 8    # heads / head dim of the tiny streams


def _rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """``comfy.ldm.flux.math.rope`` — the table builder of a stock ``EmbedND``."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2)


class MockPeEmbedder(torch.nn.Module):
    """A stock FLUX ``EmbedND`` (``ldm/flux/layers.py:17-27``): head-shared table."""

    def __init__(self, axes_dim=(16, 56, 56), theta=10000):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([_rope(ids[..., i], self.axes_dim[i], self.theta)
                         for i in range(ids.shape[-1])], dim=-3)
        return emb.unsqueeze(1)


class MockFluxDiT(torch.nn.Module):
    """The FLUX pieces the resolver touches: ``pe_embedder`` and ``patch_size``."""

    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.pe_embedder = MockPeEmbedder()


@pytest.fixture(autouse=True)
def _clean_area_state():
    """AReA's one-time warnings are process-global — reset them around a test."""
    area._warned.clear()
    yield
    area._warned.clear()


@pytest.fixture
def sdpa_capture(monkeypatch):
    """The keys every grouped SDPA call runs on (the windowed attention's input)."""
    calls: list = []
    real = torch.nn.functional.scaled_dot_product_attention

    def spy(query, key, value, *args, **kwargs):
        calls.append((query.detach().clone(), key.detach().clone(), value.detach().clone()))
        return real(query, key, value, *args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
    return calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _window(r0, c0, r1, c1) -> Window:
    return Window(r0, c0, r1, c1)


def _runtime(**cfg) -> area._AreaRuntime:
    """A runtime on the FLUX profile (no arch profile -> the FLUX code paths)."""
    return area._AreaRuntime(AreaConfig(**cfg), MockFluxDiT())


def _sigmas(steps: int) -> torch.Tensor:
    """A sampler's sigma table (``samplers.py:1229``) — ``steps + 1`` entries."""
    return torch.linspace(1.0, 0.0, steps + 1)


def _resolve(rt, steps: int, step: int = 0, grid: int = 80, txt_len: int = 32) -> AreaContext:
    """Drive the REAL resolver (``src.area._area_resolve_forward``) for one step.

    The latent is ``grid x grid`` patches (two latent pixels per patch token),
    so the patch grid is ``grid x grid`` — past the native 64-patch extent, so
    the resolver does not short-circuit to plain attention.
    """
    sigmas = _sigmas(steps)
    latent = torch.randn(1, 16, grid * 2, grid * 2)
    cross = torch.randn(1, txt_len, 77)
    params = {
        "input": latent,
        "timestep": sigmas[step],
        "c": {
            "c_crossattn": cross,
            "transformer_options": {"sample_sigmas": sigmas, "sigmas": sigmas[step]},
        },
    }
    return area._area_resolve_forward(rt, params)


def _resolved_phase1(rt):
    """The Phase I count the resolver actually scheduled with.

    The schedule cache is keyed by ``(h, w, total, phase1, txt_len, ...)``
    (``src.area._area_schedule``), so the key carries the RESOLVED count.
    """
    assert len(rt.schedules) == 1, "the resolver must have built exactly one schedule"
    key, schedule = next(iter(rt.schedules.items()))
    return key[3], schedule


def _base_boxes(h: int, w: int, size: int) -> list:
    """The unsplit base tiling: even starts, full-size windows, no quadrants."""
    return [(r, c, min(h - 1, r + size - 1), min(w - 1, c + size - 1))
            for r in even_starts(h, size) for c in even_starts(w, size)]


def _rope_table(ids: torch.Tensor, dim: int = DIM) -> torch.Tensor:
    """A rope table built from the position ids (the shape ``_head_shared`` wants)."""
    g, r, _ = ids.shape
    angle = ids[..., 1] * 0.01 + ids[..., 2] * 0.02
    cos = torch.cos(angle).unsqueeze(1).unsqueeze(-1)
    sin = torch.sin(angle).unsqueeze(1).unsqueeze(-1)
    table = torch.zeros(g, 1, r, dim // 2, 2, 2)
    table[..., 0, 0] = cos
    table[..., 0, 1] = -sin
    table[..., 1, 0] = sin
    table[..., 1, 1] = cos
    return table


# ---------------------------------------------------------------------------
# Mechanism 1 — the "+2 healing rule"
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestHealCap:
    """The "+2 healing rule": Phase I may not eat the last two fine steps.

    Driven through the real resolver, so the cap is observed where it happens —
    in the schedule the resolver builds and in the advisory it emits.
    """

    def test_eight_coarse_steps_on_an_eight_step_run_are_capped(self):
        rt = _runtime(phase1_steps=8, seed=7)
        _resolve(rt, steps=8)
        phase1, schedule = _resolved_phase1(rt)
        # the RESOLVED count is the cap, not the knob
        assert phase1 == max(1, 8 - 2) == 6
        assert phase1 != 8
        # six coarse rungs and TWO fine steps left to heal the footprint
        assert len(schedule.factors) == 6
        assert len(schedule.windows) == 2
        # and the advisory says so
        assert "area-heal-cap" in area._warned

    def test_the_very_short_run_keeps_one_coarse_step(self):
        for steps in (2, 3):
            rt = _runtime(phase1_steps=8, seed=7)
            _resolve(rt, steps=steps)
            phase1, schedule = _resolved_phase1(rt)
            # the count never drops below one
            assert phase1 == max(1, steps - 2) == 1
            assert len(schedule.factors) == 1
            assert "area-heal-cap" in area._warned

    def test_a_long_schedule_is_untouched(self):
        rt = _runtime(phase1_steps=8, seed=7)
        ctx = _resolve(rt, steps=28)
        phase1, schedule = _resolved_phase1(rt)
        # a long schedule is never capped
        assert phase1 == 8
        assert len(schedule.factors) == 8
        assert "area-heal-cap" not in area._warned
        assert ctx.total == 28

    def test_a_short_schedule_below_the_cap_is_untouched(self):
        rt = _runtime(phase1_steps=3, seed=7)
        _resolve(rt, steps=8)
        phase1, schedule = _resolved_phase1(rt)
        assert phase1 == 3
        assert len(schedule.factors) == 3
        assert "area-heal-cap" not in area._warned

    def test_the_cap_leaves_two_fine_steps_on_a_short_run(self):
        # the rule the doc states: S_c = min(phase1_steps, S - 2)
        rt = _runtime(phase1_steps=8, seed=7)
        _resolve(rt, steps=8)
        _, schedule = _resolved_phase1(rt)
        assert len(schedule.factors) + len(schedule.windows) == 8
        assert len(schedule.windows) >= 2


# ---------------------------------------------------------------------------
# Mechanism 2 — the RoPE horizon saturation
# ---------------------------------------------------------------------------


@pytest.mark.unit
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


# ---------------------------------------------------------------------------
# Mechanism 3 — window integrity
# ---------------------------------------------------------------------------

#: the patch grid of the layout test — 200 is NOT a multiple of the window
#: side, so the boundary jitter has room to move (a multiple would pin it).
GRID = 200
#: the Phase II window side of that run
WIN = 64


def _checkerboard_hidden(h: int = GRID, w: int = GRID, channels: int = 16) -> torch.Tensor:
    """A hidden state whose detail map fires the DWT split gate.

    ``detail_map`` measures the per-token channel std, so a checkerboard of
    "constant over the channels" and "spread over the channels" tokens is the
    DWT's favourite input: the Haar detail is large inside the checked region
    and zero outside it, so the tile energy varies strongly and the split gate
    opens.  This is the noise-like energy map the mechanism defends against.
    """
    hidden = torch.zeros(1, h * w, channels)
    for r in range(0, 63):
        for c in range(0, 63):
            idx = r * w + c
            if (r + c) % 2 == 0:
                hidden[0, idx, :] = 1.0
            else:
                hidden[0, idx, 0] = 3.0
                hidden[0, idx, 1] = -3.0
    return hidden


@pytest.mark.unit
class TestWindowIntegrity:
    """Mechanism 3: a short schedule keeps the base windows whole and still."""

    def _layout(self, total: int, step: int = 3):
        """The layout the REAL builder produces for a context of this length."""
        rt = _runtime(seed=7)
        rt.hidden = _checkerboard_hidden()
        ctx = AreaContext(phase="fine", step=step, size=WIN, h=GRID, w=GRID,
                          txt_len=32, fine_step=0, floor_steps=0, total=total)
        k_img = torch.randn(1, HEADS, GRID * GRID, DIM)
        return area._area_fine_layout(rt, ctx, k_img)

    def test_the_short_schedule_keeps_the_unsplit_base_tiling(self):
        layout = self._layout(total=8)
        # exactly the base tiling of even starts and full-size windows
        assert [(w.r0, w.c0, w.r1, w.c1) for w in layout.windows] == _base_boxes(GRID, GRID, WIN)
        # no DWT quadrant split anywhere
        assert not any(w.split for w in layout.windows)
        # and the starts are the EVEN ones (no boundary jitter)
        assert sorted({w.r0 for w in layout.windows}) == even_starts(GRID, WIN)
        assert sorted({w.c0 for w in layout.windows}) == even_starts(GRID, WIN)

    def test_the_long_schedule_keeps_the_split_and_jitter_path(self):
        layout = self._layout(total=28)
        # the DWT split path is still active ...
        assert any(w.split for w in layout.windows)
        # ... and the boundaries still jitter (a start off the even grid)
        assert not {w.r0 for w in layout.windows} <= set(even_starts(GRID, WIN))
        # so the long layout is NOT the base tiling the short one keeps
        assert [(w.r0, w.c0, w.r1, w.c1) for w in layout.windows] != _base_boxes(GRID, GRID, WIN)

    def test_the_two_schedules_really_produce_different_layouts(self):
        short = self._layout(total=8)
        long = self._layout(total=28)
        assert [(w.r0, w.c0, w.r1, w.c1) for w in short.windows] != \
               [(w.r0, w.c0, w.r1, w.c1) for w in long.windows]


@pytest.mark.unit
class TestAdaptiveTilingSwitch:
    """``allow_splits=False`` returns the base tiling un-split."""

    def test_no_splits_keeps_the_base_boxes(self):
        energy = np.zeros((64, 64), dtype=np.float64)
        kept = adaptive_tiling(energy, 64, 64, 32, 16, 8, rng=None, allow_splits=False)
        assert [(w.r0, w.c0, w.r1, w.c1) for w in kept] == _base_boxes(64, 64, 32)
        # the base grid is 2x2 windows of side 32 — no quadrant fragmentation
        assert len(kept) == 4
        assert all(w.width <= 32 and w.height <= 32 for w in kept)
        # and the split path may only ever return MORE windows (split, never merge)
        base = adaptive_tiling(energy, 64, 64, 32, 16, 8, rng=None, allow_splits=True)
        assert len(base) >= len(kept)

    def test_no_rng_means_even_starts(self):
        energy = np.random.default_rng(1).random((64, 64))
        windows = adaptive_tiling(energy, 64, 64, 30, 16, 8, rng=None)
        # the starts ARE the even ones — not merely equal across two calls
        assert [(w.r0, w.c0) for w in windows] == \
               [(r, c) for r in even_starts(64, 30) for c in even_starts(64, 30)]

    def test_a_split_can_actually_happen_when_the_energy_fires(self):
        # the gate the mechanism switches off is real: a high-contrast energy map
        # splits the base windows, and the unsplit path never does
        energy = np.zeros((64, 64), dtype=np.float64)
        energy[:32, :32] = 1.0
        energy[32:, 32:] = 0.1
        split = adaptive_tiling(energy, 64, 64, 32, 16, 8, rng=None, allow_splits=True)
        kept = adaptive_tiling(energy, 64, 64, 32, 16, 8, rng=None, allow_splits=False)
        assert any(w.split for w in split)
        assert not any(w.split for w in kept)


# ---------------------------------------------------------------------------
# Mechanism 2 — the wiring
# ---------------------------------------------------------------------------

#: a MULTI-GROUP layout: two windows of different sizes, so the builder puts
#: them in two groups (one member each) with DIFFERENT centres —
#: (15.5, 15.5) and (11.5, 47.5).  The upstream port mapped a group's members
#: to ``layout.windows[:g]`` — the first g windows — which is only right when
#: every group holds the first windows; this layout is the case that breaks it.
GRP_H, GRP_W = 32, 64
GRP_WINDOWS = (_window(0, 0, 31, 31), _window(0, 32, 23, 63))
GRP_REGISTERS = 4


def _two_group_layout():
    """A layout with two groups, one window each, of different token counts."""
    return build_fine_layout(list(GRP_WINDOWS), GRP_H, GRP_W, 0, GRP_REGISTERS,
                              0, 0, torch.float32, "cpu")


def _stream(seed: int = 5):
    """The joint stream: ``[b, heads, T + h*w, d]`` q/k/v plus the unrotated keys."""
    torch.manual_seed(seed)
    shape = (1, HEADS, TXT + GRP_H * GRP_W, DIM)
    q = torch.randn(shape)
    k = torch.randn(shape)
    v = torch.randn(shape)
    k_unrot = torch.randn((1, HEADS, GRP_H * GRP_W, DIM))
    return q, k, v, k_unrot


@pytest.mark.unit
class TestBridgeSaturationWiring:
    """The bridge keys' source, the group -> window mapping, and the FLUX call site."""

    def test_window_ids_map_into_the_layouts_windows(self):
        layout = build_fine_layout([_window(0, 0, 15, 15)], 16, 16, 0, 4, 0, 0,
                                   torch.float32, "cpu")
        assert len(layout.groups) == 1
        group = layout.groups[0]
        assert group.window_ids.tolist() == [0]
        assert layout.windows[int(group.window_ids[0])].width == 16

    def test_the_layout_really_has_two_groups_with_different_centres(self):
        layout = _two_group_layout()
        assert len(layout.groups) == 2
        assert [g.window_ids.tolist() for g in layout.groups] == [[0], [1]]
        centres = [((w.r0 + w.r1) * 0.5, (w.c0 + w.c1) * 0.5) for w in layout.windows]
        assert centres == [(15.5, 15.5), (11.5, 47.5)]

    def test_each_group_saturates_against_its_own_window(self):
        """The saturated ids of a group are measured from THAT group's window."""
        layout = _two_group_layout()
        q, k, v, k_unrot = _stream()
        calls: list = []

        def recording_embedder(ids):
            # EVERY call is recorded: the caption's anchor table (the GTCN mechanism) is
            # built once before the group loop, the bridge's table once per
            # group inside it.  The bridge's tables are therefore the calls
            # AFTER the first — selected by that ROLE, never by a predicate on
            # the axis the assertions below check.
            calls.append(ids.detach().clone())
            return _rope_table(ids)

        windowed_attention(q[:, :, :TXT], k[:, :, :TXT], v[:, :, :TXT],
                           q[:, :, TXT:], k[:, :, TXT:], v[:, :, TXT:],
                           k_unrot, layout, rope_embedder=recording_embedder, cap_len=TXT)
        seen = calls[1:]
        assert len(seen) == len(layout.groups)

        score = token_importance(k_unrot)
        for j, group in enumerate(layout.groups):
            reg = select_registers(score, group.tokens, layout.registers, GRP_H, GRP_W)
            win = layout.windows[int(group.window_ids[0])]
            exp_r, exp_c = saturate_bridge_positions(reg, win, GRP_H, GRP_W, "cpu")
            assert torch.allclose(seen[j][..., 1], exp_r, atol=1e-4)
            assert torch.allclose(seen[j][..., 2], exp_c, atol=1e-4)
            # the ids of a group are measured from ITS OWN centre
            centre_r, centre_c = (win.r0 + win.r1) * 0.5, (win.c0 + win.c1) * 0.5
            assert bool(torch.all((seen[j][..., 1] - centre_r).abs() <= D_MAX_HORIZON + 1e-3))
            assert bool(torch.all((seen[j][..., 2] - centre_c).abs() <= D_MAX_HORIZON + 1e-3))

        # the two groups measure different offsets — the mapping is per group
        assert not torch.allclose(seen[0][..., 2], seen[1][..., 2], atol=1e-4)

    def test_the_wrong_window_would_saturate_differently(self):
        """The upstream mapping (``layout.windows[:g]``) is observably wrong here.

        Without this, the mapping test above would also pass on the broken
        port — the two windows' centres are close enough for a weak assertion.
        """
        layout = _two_group_layout()
        q, k, v, k_unrot = _stream()
        calls: list = []

        def recording_embedder(ids):
            # every call is recorded (the bridge's tables are the calls after
            # the first — see the test above)
            calls.append(ids.detach().clone())
            return _rope_table(ids)

        windowed_attention(q[:, :, :TXT], k[:, :, :TXT], v[:, :, :TXT],
                           q[:, :, TXT:], k[:, :, TXT:], v[:, :, TXT:],
                           k_unrot, layout, rope_embedder=recording_embedder, cap_len=TXT)
        seen = calls[1:]
        score = token_importance(k_unrot)
        group = layout.groups[1]
        reg = select_registers(score, group.tokens, layout.registers, GRP_H, GRP_W)
        wrong_r, wrong_c = saturate_bridge_positions(reg, layout.windows[0], GRP_H, GRP_W, "cpu")
        # the upstream port would have produced THESE ids for the second group
        assert not torch.allclose(seen[1][..., 1], wrong_r, atol=1e-3)
        assert not torch.allclose(seen[1][..., 2], wrong_c, atol=1e-3)

    def test_the_bridge_keys_come_from_the_unrotated_tensor(self, sdpa_capture):
        """The re-roped bridge keys are the UNROTATED keys, not the rotated ones."""
        layout = _two_group_layout()
        q, k, v, k_unrot = _stream()
        calls: list = []

        def recording_embedder(ids):
            # every call is recorded (the bridge's tables are the calls after
            # the first — see the test above)
            calls.append(ids.detach().clone())
            return _rope_table(ids)

        windowed_attention(q[:, :, :TXT], k[:, :, :TXT], v[:, :, :TXT],
                           q[:, :, TXT:], k[:, :, TXT:], v[:, :, TXT:],
                           k_unrot, layout, rope_embedder=recording_embedder, cap_len=TXT)
        seen = calls[1:]
        assert len(sdpa_capture) == len(layout.groups)

        score = token_importance(k_unrot)
        for j, group in enumerate(layout.groups):
            n = int(group.tokens.shape[1])
            # one register row per member of the group (this group holds one)
            reg = select_registers(score, group.tokens, layout.registers, GRP_H, GRP_W)[0]
            rope = _rope_table(seen[j])
            expected = apply_rope1(k_unrot[0][:, reg, :].unsqueeze(0), rope)[0]
            # the group's slice of the batched SDPA keys: [b*g, heads, L, dim]
            captured = sdpa_capture[j][1][0]
            got = captured[:, TXT + n: TXT + n + layout.registers, :]
            assert torch.allclose(got, expected, atol=1e-5)
            # and NOT the rotated tensor's registers
            wrong = apply_rope1(k[0][:, reg, :].unsqueeze(0), rope)[0]
            assert not torch.allclose(got, wrong, atol=1e-3)

    def test_the_embedder_saturates_the_bridge_positions(self, monkeypatch):
        # the wired (z-image) path must rope the bridge keys at the SATURATED
        # register positions: once per group, with that group's own window
        q, k, v, k_unrot = _stream()
        layout = _two_group_layout()
        seen = {}
        real = saturate_bridge_positions

        def spy(reg, win, h, w, device):
            seen.setdefault("calls", []).append(win)
            return real(reg, win, h, w, device)

        def recording_embedder(ids):
            seen.setdefault("ids", []).append(ids.detach().clone())
            # The FIRST call is the caption's anchor table (the GTCN mechanism), built
            # once before the group loop; every call AFTER it is the bridge's
            # own table, one per group.  The role is decided by the call
            # ORDER — a predicate on the t axis would filter out the very
            # thing the assertion below checks, and make it unable to fail.
            if len(seen["ids"]) == 1:
                return _rope_table(ids)
            # the t axis carries the caption position (cap_len + 1)
            assert torch.all(ids[..., 0] == TXT + 1)
            # and the rows/cols stay inside the trained extent of the window centre
            centre = (0 + 15) * 0.5
            rows, cols = ids[..., 1], ids[..., 2]
            assert bool(torch.all(rows >= centre - D_MAX_HORIZON - 1e-3))
            assert bool(torch.all(rows <= centre + D_MAX_HORIZON + 1e-3))
            assert bool(torch.all(cols >= centre - D_MAX_HORIZON - 1e-3))
            assert bool(torch.all(cols <= centre + D_MAX_HORIZON + 1e-3))
            return _rope_table(ids)

        monkeypatch.setattr("src.area_attn.saturate_bridge_positions", spy)
        windowed_attention(q[:, :, :TXT], k[:, :, :TXT], v[:, :, :TXT],
                           q[:, :, TXT:], k[:, :, TXT:], v[:, :, TXT:],
                           k_unrot, layout,
                           rope_embedder=recording_embedder, cap_len=TXT)
        assert len(seen.get("calls", [])) == len(layout.groups)
        assert all(w in layout.windows for w in seen["calls"])

    def test_without_an_embedder_the_bridge_stays_on_the_plain_path(self, monkeypatch):
        # no embedder -> nothing may saturate: calling the saturation helper
        # there would mean the plain path's numbers changed
        q, k, v, k_unrot = _stream()
        layout = _two_group_layout()

        def boom(*_args, **_kwargs):
            raise AssertionError("the plain bridge path must not saturate")

        monkeypatch.setattr("src.area_attn.saturate_bridge_positions", boom)
        _txt, img = windowed_attention(
            q[:, :, :TXT], k[:, :, :TXT], v[:, :, :TXT],
            q[:, :, TXT:], k[:, :, TXT:], v[:, :, TXT:],
            k_unrot, layout)
        assert img.shape == (1, HEADS, GRP_H * GRP_W, DIM)

    def test_a_hand_built_group_without_window_ids_still_runs(self):
        """A group that does not declare its mapping runs (and maps the same).

        ``WindowGroup.window_ids`` is optional: a hand-built group may leave it
        unset, and the mapping is then recovered from the group's own tokens.
        """
        layout = _two_group_layout()
        q, k, v, k_unrot = _stream()
        seen: list = []

        def recording_embedder(ids):
            seen.append(ids.detach().clone())
            return _rope_table(ids)

        base = windowed_attention(q[:, :, :TXT], k[:, :, :TXT], v[:, :, :TXT],
                                  q[:, :, TXT:], k[:, :, TXT:], v[:, :, TXT:],
                                  k_unrot, layout, rope_embedder=recording_embedder, cap_len=TXT)
        stripped = FineLayout(
            layout.h, layout.w, layout.windows,
            [WindowGroup(tokens=g.tokens, halo=g.halo, weights=g.weights)
             for g in layout.groups],
            layout.coverage, layout.registers)
        assert all(g.window_ids is None for g in stripped.groups)

        seen2: list = []

        def recording_embedder2(ids):
            seen2.append(ids.detach().clone())
            return _rope_table(ids)

        out = windowed_attention(q[:, :, :TXT], k[:, :, :TXT], v[:, :, :TXT],
                                 q[:, :, TXT:], k[:, :, TXT:], v[:, :, TXT:],
                                 k_unrot, stripped, rope_embedder=recording_embedder2, cap_len=TXT)
        # the recovered mapping is the declared one -> the same numbers
        assert len(seen2) == len(seen)
        for a, b in zip(seen, seen2):
            assert torch.allclose(a, b, atol=1e-6)
        assert torch.allclose(out[1], base[1], atol=1e-6)


@pytest.mark.unit
class TestFluxCallSite:
    """The FLUX path must keep the paper's true-position bridge keys.

    The pin is on the CALL SITE (``src.area._area_attention``, the FLUX branch)
    — not on ``windowed_attention`` called without an embedder, which would
    pass even if the seam handed FLUX an embedder.
    """

    def _stream(self, grid: int = 32, size: int = 16):
        rt = _runtime(halo=0, seed=7)
        ctx = AreaContext(phase="fine", step=3, size=size, h=grid, w=grid,
                          txt_len=TXT, fine_step=0, floor_steps=0, total=28)
        torch.manual_seed(11)
        shape = (1, HEADS, TXT + grid * grid, DIM)
        q = torch.randn(shape)
        k = torch.randn(shape)
        v = torch.randn(shape)
        ids = torch.zeros(1, TXT + grid * grid, 3)
        pos = torch.arange(grid * grid)
        ids[0, TXT:, 1] = (pos // grid).float()
        ids[0, TXT:, 2] = (pos % grid).float()
        # the rope table of a head dim of DIM: the axes sum to DIM (half of it
        # per rope axis), exactly as a stock ``EmbedND`` builds it
        pe = MockPeEmbedder(axes_dim=(2, 4, 2))(ids)
        return rt, ctx, q, k, v, pe

    def test_the_flux_call_site_passes_no_rope_embedder(self, monkeypatch):
        rt, ctx, q, k, v, pe = self._stream()
        seen = {}

        def fake_windowed_attention(*args, **kwargs):
            seen["kwargs"] = kwargs
            b, heads, t, dim = args[0].shape
            n = args[3].shape[2]
            return torch.zeros(b, heads, t, dim), torch.zeros(b, heads, n, dim)

        monkeypatch.setattr("src.area.windowed_attention", fake_windowed_attention)
        area._area_attention(rt, ctx, q, k, v, pe)
        assert "rope_embedder" not in seen["kwargs"]
        assert seen["kwargs"].get("rope_embedder") is None

    def test_the_flux_bridge_keys_stay_on_the_rotated_true_positions(self, sdpa_capture):
        rt, ctx, q, k, v, pe = self._stream()
        out = area._area_attention(rt, ctx, q, k, v, pe)
        assert out.shape == (1, TXT + 32 * 32, HEADS * DIM)
        assert len(sdpa_capture) == 1
        layout = ctx.data["layout"]
        group = layout.groups[0]
        n = int(group.tokens.shape[1])
        score = token_importance(k[:, :, TXT:])
        reg = select_registers(score, group.tokens, layout.registers, 32, 32)
        captured = sdpa_capture[0][1]
        k_rotated = apply_rope1(k, pe)
        # every member of the group keeps the paper's true-position (rotated) keys
        for j in range(int(group.tokens.shape[0])):
            bridge = captured[j][:, TXT + n: TXT + n + layout.registers, :]
            assert torch.allclose(bridge, k_rotated[0][:, TXT:][:, reg[j], :], atol=1e-5)


# ---------------------------------------------------------------------------
# The context's total
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAreaContextTotal:
    def test_the_default_is_zero(self):
        assert AreaContext().total == 0

    def test_the_resolver_populates_the_total(self):
        rt = _runtime(phase1_steps=3, seed=7)
        ctx = _resolve(rt, steps=8)
        # NOT the dataclass default: the resolver writes the schedule's length
        assert ctx.total == 8
        assert ctx.total != AreaContext().total

    def test_a_long_schedule_populates_the_total_too(self):
        rt = _runtime(phase1_steps=8, seed=7)
        ctx = _resolve(rt, steps=28)
        assert ctx.total == 28

    def test_a_short_schedule_is_a_short_schedule(self):
        assert is_short_schedule(8)
        assert not is_short_schedule(28)
