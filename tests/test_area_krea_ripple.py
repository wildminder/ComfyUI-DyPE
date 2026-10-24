"""AReA × Krea-2 — the 32 px stair-step ripple and the bridge's share.

**The bug this file reproduces.**  The Krea-2 path ran the reference's FLUX
recipe for Phase I: the query's position was FLOOR-DIVIDED into its block's
integer coordinate (``row // f``), and the pooling was the reference's
``subsample``.  Together those are the two flaws the pack's own AReA
analysis identified as the cause of the Z-Image 32 px ripple
(the block-position flaw and Flaw 3):

* **Flaw 2 — the integer stair-step.**  All ``f x f`` tokens of a block shared
  ONE rope position, and the phase jumped by a whole unit at every block
  boundary — one RADIAN in the top frequency band, and the top band is
  ``theta ** 0 = 1`` rad per token for EVERY theta
  (``comfy/ldm/flux/math.py:20-32``), so the jump is a property of the floor
  division, not of the backbone's theta.  The period is the block: ``f``
  tokens, i.e. **32 px at ``f = 2``** on a 16 px/token canvas.
* **Flaw 3 — the Dirac comb.**  ``subsample`` keeps ONE token of every block,
  so the pooled values live on an f-stride lattice; and because the block's
  queries shared one rope position, their attention distributions — and
  therefore their outputs — were nearly identical, i.e. the Phase I field was
  piecewise-constant over 32 px blocks.

The user's Krea-2 run reproduces the Z-Image ripple verbatim ("the ripple
artifacts appear (the same as z-image)") and the 32 px period is exactly the
"vertically extended by duplicating the same conceptual element" offset the
report describes.  Plan Q9 had scoped the Z-Image refinement OUT of the
Krea-2 adapter and recorded it as an explicit follow-up
(the Krea-2 adaptation note) — this file
pins that follow-up.

**The canvas.**  512 x 2000 px -> a **125 x 32** token grid (``gcd == 1``, so
the factor is the extent rule's ``f = 2``), a coarse canvas of 63 x 16, and a
Phase II window side of 64 — WIDER than the canvas, so the Phase II tiling
degenerates into two full-width strips split at row 61 (pixel 976, the
canvas' middle).  That geometry is measured at the bottom of this file and is
REPORTED, not changed; the bridge's share of a window's content is fixed
here.

Everything here is CPU-only and mock-based: no checkpoint, no GPU.

Markers: @pytest.mark.unit
Accept: pytest tests/test_area_krea_ripple.py
"""
from __future__ import annotations

import sys
import types

import pytest
import torch

import src.area as area
from _krea_fixtures import MockEmbedND, SingleStreamDiT, krea_backend_module
from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_arch import FLUX_PROFILE, KREA_PROFILE, ZIMAGE_PROFILE
from src.area_attn import bridge_registers, pool_keys_values
from src.area_krea import KREA_NATIVE_TOKENS, KREA_PX_PER_PATCH, coarse_ropes_krea, krea_layout
from src.area_schedule import build_schedule, coarse_grid, even_starts, resolve_factor

try:
    from comfy.model_patcher import ModelPatcher
except ImportError:  # pragma: no cover - the mock env always provides it
    from tests.conftest import MockModelPatcher as ModelPatcher

#: The user's canvas: 512 x 2000 px -> a 125 x 32 token grid.
GRID_H, GRID_W = 125, 32
TXT_LEN = 32
#: The pooling factor the engine picks on that grid (the extent rule, because
#: ``gcd(125, 32) == 1`` leaves the grid untileable).
F = resolve_factor(GRID_H, GRID_W, KREA_NATIVE_TOKENS)[0]
#: The block's period in PIXELS — the ripple's wavelength.
BLOCK_PX = F * KREA_PX_PER_PATCH


class _SpyEmbedder(torch.nn.Module):
    """An embedder that RECORDS the ids it was handed (the positions)."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, ids):
        self.calls.append(ids)
        return torch.full((1, 1, ids.shape[1], 3, 2, 2), float(len(self.calls)))


def _axis_angles(table: torch.Tensor, axis: int) -> torch.Tensor:
    """The rotation angle of one rope axis, per token.

    The table is the head-shared ``[b, 1, n, axes, 2, 2]`` an ``EmbedND``
    returns; its 2x2 block is ``[[cos, -sin], [sin, cos]]`` (``rope()``,
    ``comfy/ldm/flux/math.py:20-32``), so the angle is ``atan2(sin, cos)``.
    """
    m = table[0, 0, :, axis]
    return torch.atan2(m[:, 1, 0], m[:, 0, 0])


def _phase_steps(angles: torch.Tensor) -> torch.Tensor:
    """The phase advance between consecutive tokens, wrapped to ``(-pi, pi]``."""
    delta = angles[1:] - angles[:-1]
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def _krea_ids(h: int = GRID_H, w: int = GRID_W, f: int = F):
    """The two id tables the coarse rope builder hands to the embedder."""
    spy = _SpyEmbedder()
    layout = krea_layout(txt_len=TXT_LEN, h=h, w=w, stream_len=TXT_LEN + h * w)
    coarse_ropes_krea(spy, layout, f, torch.device("cpu"))
    return spy.calls


@pytest.mark.unit
class TestKreaCoarseStairStep:
    """The Phase I rope must not be a square wave (the block-position flaw)."""

    def test_the_phase_advance_per_token_is_uniform(self):
        """REPRODUCTION: the advance is 1/f rad per token — at EVERY token.

        The floor-divided recipe advanced the phase by 0 inside a block and by
        a whole unit (one radian) at the block boundary, i.e. a square wave
        with the block's period (32 px here).  The continuous recipe advances
        it uniformly, so the step across a boundary equals the step inside
        the block.

        The mock's embedder reproduces the top band faithfully: its row axis
        has ``dim = 2``, so ``scale = linspace(0, 0, 1) = [0]`` and
        ``omega = theta ** 0 = 1`` — one radian per position unit, exactly
        like the top band of the real 48-dim row axis.
        """
        layout = krea_layout(txt_len=TXT_LEN, h=GRID_H, w=GRID_W,
                             stream_len=TXT_LEN + GRID_H * GRID_W)
        rope_q, _ = coarse_ropes_krea(MockEmbedND(), layout, F, torch.device("cpu"))
        column = rope_q[:, :, TXT_LEN::GRID_W, :]
        steps = _phase_steps(_axis_angles(column, 1))
        assert steps.numel() == GRID_H - 1
        # every step is the SAME, and it is 1/f rad
        assert torch.allclose(steps, torch.full_like(steps, 1.0 / F), atol=1e-5), (
            "the Phase I rope phase is not uniform: the step per fine token "
            "varies between %.4f and %.4f rad — a square wave with the "
            "block's period (%d px)"
            % (float(steps.min()), float(steps.max()), BLOCK_PX)
        )

    def test_the_tokens_of_a_block_do_not_share_one_position(self):
        """The block's f^2 tokens keep distinct positions (the collapse is gone)."""
        ids = _krea_ids()
        rows = ids[0][0, TXT_LEN:, 1]
        cols = ids[0][0, TXT_LEN:, 2]
        assert len(torch.unique(rows)) == GRID_H          # one position per row
        assert len(torch.unique(cols)) == GRID_W          # one position per column
        # and the positions are exactly row/f (continuous), not row//f
        assert torch.allclose(rows[::GRID_W],
                              torch.arange(GRID_H, dtype=torch.float32) / F)

    def test_the_pooled_key_sits_on_its_block_centroid(self):
        """The pooled key is the MEAN of its block's query positions.

        That is what makes the phase step across a block boundary equal the
        step inside it — and it is what the floor-divided recipe could not
        do (the key sat on the block's integer coordinate, the very position
        the block's queries had collapsed onto).
        """
        ids = _krea_ids()
        q_rows = ids[0][0, TXT_LEN:, 1].reshape(GRID_H, GRID_W)
        q_cols = ids[0][0, TXT_LEN:, 2].reshape(GRID_H, GRID_W)
        kv_rows = ids[1][0, TXT_LEN:, 1]
        kv_cols = ids[1][0, TXT_LEN:, 2]
        nh, nw = coarse_grid(GRID_H, GRID_W, F)
        for block in range(nh * nw):
            r, c = divmod(block, nw)
            cells = q_rows[r * F:(r + 1) * F, c * F:(c + 1) * F]
            cells_c = q_cols[r * F:(r + 1) * F, c * F:(c + 1) * F]
            assert cells.numel() > 0
            assert abs(float(kv_rows[block]) - float(cells.mean())) < 1e-6
            assert abs(float(kv_cols[block]) - float(cells_c.mean())) < 1e-6
            # and it is the block's INTERIOR: a quarter of a block in from its
            # first cell — not the block's integer coordinate, which is where
            # the floor-divided recipe put it (and where the block's queries
            # had collapsed onto).
            if cells.shape[0] == F and cells.shape[1] == F:
                assert abs(float(kv_rows[block]) - (r + (F - 1) / (2.0 * F))) < 1e-6
                assert abs(float(kv_cols[block]) - (c + (F - 1) / (2.0 * F))) < 1e-6

    def test_the_caption_still_rides_at_the_origin(self):
        """The continuous positions touch the IMAGE ids only."""
        for ids in _krea_ids():
            assert torch.equal(ids[0, :TXT_LEN], torch.zeros(TXT_LEN, 3))
            assert torch.equal(ids[0, :, 0], torch.zeros(ids.shape[1]))


@pytest.mark.unit
class TestKreaCanvasExtentRatio:
    """GTCN — the canvas extent ratio (the GTCN mechanism) on the Krea coarse table."""

    def test_the_multiplier_is_inert_inside_the_trained_extent(self):
        """The block units already hold the canvas inside the trained extent.

        On the user's canvas the coarse grid is 63 x 16 — 63 rows against the
        trained 64 — so the multiplier is exactly 1.0 and the table is the
        plain continuous one.  This pins that the GTCN is NOT what changes
        the user's run: it cannot be.
        """
        ids = _krea_ids()
        rows = ids[0][0, TXT_LEN:, 1]
        assert torch.allclose(rows[::GRID_W],
                              torch.arange(GRID_H, dtype=torch.float32) / F)

    def test_a_canvas_beyond_the_trained_extent_is_compressed(self):
        """A canvas the block units cannot hold is compressed back into it.

        A 250 x 250 token canvas at ``f = 2`` has a 125 x 125 coarse grid —
        twice the trained extent — so the coordinates are scaled by
        ``f / max(f, s)`` with ``s = 250 / 64``, and the coarse canvas fits
        the trained domain again.
        """
        side = 250
        ids = _krea_ids(h=side, w=side, f=2)
        rows = ids[0][0, TXT_LEN:, 1]
        norm = 2.0 / max(2.0, side / float(KREA_NATIVE_TOKENS))
        assert norm < 1.0
        assert float(rows.max()) == pytest.approx((side - 1) / 2.0 * norm, abs=1e-4)
        # the compressed coarse canvas fits the trained extent again
        assert float(rows.max()) <= float(KREA_NATIVE_TOKENS)


@pytest.mark.unit
class TestKreaPooling:
    """The Dirac comb (the Dirac-comb flaw) — the pooling Krea-2 runs."""

    def test_the_krea_default_pooling_is_the_centroid(self, krea_backend):
        """Krea-2 pools with the norm-preserving centroid, not ``subsample``."""
        m = apply_area_to_model(_mock_krea_model(), "krea2", 512, 2000,
                                phase1_steps=6, seed=7)
        try:
            assert m._area_config.pooling == "mean"
        finally:
            restore_area_attention_hook(m)

    def test_the_flux_default_pooling_is_unchanged(self, flux_layers):
        """FLUX keeps the reference's ``subsample`` — its path is frozen."""
        import test_area_node as flux

        m = apply_area_to_model(flux.mock_flux_model(), "flux", 1280, 1280,
                                phase1_steps=8, seed=7)
        try:
            assert m._area_config.pooling == "subsample"
        finally:
            restore_area_attention_hook(m)

    def test_the_pooled_keys_are_not_a_dirac_comb(self):
        """The pooled key of a block is a BLEND of its members.

        With ``subsample`` the pooled key IS one member of the block, so the
        block's values live on an f-stride lattice; with the centroid the key
        is the block's mean direction at the mean member norm.
        """
        # a 4 x 4 canvas at f = 2 -> four blocks of four members each; every
        # member has its own direction, so no block's mean is one of them
        import math

        members = torch.tensor([[[[math.cos(2 * math.pi * i / 16),
                                   math.sin(2 * math.pi * i / 16)]
                                  for i in range(16)]]])        # [1,1,16,2]
        comb_k, comb_v = pool_keys_values(members, members, 0, 4, 4, 2, "subsample")
        blend_k, blend_v = pool_keys_values(members, members, 0, 4, 4, 2, "mean")
        assert comb_k.shape == (1, 1, 4, 2)
        # the subsample pool is one of the block's members, verbatim
        for b in range(4):
            r, c = divmod(b, 2)
            mem = [8 * r + 2 * c + m for m in (0, 1, 4, 5)]
            assert any(torch.allclose(comb_k[0, 0, b], members[0, 0, m]) for m in mem)
        assert torch.equal(comb_v, comb_k)
        # the centroid pool is none of the members — it is their mean direction
        for b in range(4):
            r, c = divmod(b, 2)
            mem = [8 * r + 2 * c + m for m in (0, 1, 4, 5)]
            assert not any(torch.allclose(blend_k[0, 0, b], members[0, 0, m])
                           for m in mem)
        assert not torch.equal(blend_k, comb_k)


@pytest.mark.unit
class TestKreaBridgeShare:
    """The bridge's share of a window's content on a canvas narrower than it."""

    def test_the_bridge_count_follows_the_window_shape(self):
        """On the user's canvas the bridge is a QUARTER of the window, not half.

        The Phase II window side is 64 and the canvas is 32 tokens wide, so a
        window holds 64 x 32 = 2048 tokens — half the square window the
        paper's ``R = (rho * w)^2`` assumes.  The square formula would hand
        the bridge 1024 registers, i.e. 50 % of the window's own content;
        the designed share (``rho^2``) is 25 %, i.e. 512 registers.
        """
        assert bridge_registers(64, 0.5) == 1024                    # the square one
        assert bridge_registers(64, 0.5, GRID_H, GRID_W) == 512       # the window's shape
        assert bridge_registers(64, 0.5, GRID_H, GRID_W) * 4 == 2048  # a quarter of it

    def test_the_square_count_is_unchanged_without_the_canvas(self):
        """Without the canvas the count is the paper's square one, bit for bit."""
        assert bridge_registers(64, 0.5) == bridge_registers(64, 0.5, 64, 64)
        assert bridge_registers(65, 0.5) == bridge_registers(65, 0.5, 65, 65)
        assert bridge_registers(8, 0.5) == bridge_registers(8, 0.5, 8, 8)

    def test_only_the_krea_profile_asks_for_the_shape_aware_count(self):
        """The FLUX and Z-Image paths keep the square bridge (they are frozen)."""
        assert KREA_PROFILE.bridge_shape_aware is True
        assert FLUX_PROFILE.bridge_shape_aware is False
        assert ZIMAGE_PROFILE.bridge_shape_aware is False


@pytest.mark.unit
class TestTheUserCanvasGeometry:
    """The geometry the user's runs actually ran (the effective config).

    These are MEASUREMENTS, not fixes: they are what the GPU checklist is
    built on, and they pin the numbers the report quotes.
    """

    def test_the_grid_and_the_factor(self):
        """125 x 32 tokens, ``gcd == 1``, so the factor is the extent rule's 2."""
        assert (GRID_H, GRID_W) == (125, 32)
        factor, adjusted = resolve_factor(GRID_H, GRID_W, KREA_NATIVE_TOKENS)
        assert adjusted is True
        assert factor == 2
        assert coarse_grid(GRID_H, GRID_W, factor) == (63, 16)

    def test_the_phase_two_tiling_is_two_full_width_strips(self):
        """The Phase II window side is 64 — WIDER than the 32-token canvas.

        So the tiling is a single window column and the canvas is cut in two
        strips that meet at row 61 (pixel 976 of 2000 — the canvas' middle,
        which is where the user sees the split).
        """
        schedule = build_schedule(GRID_H, GRID_W, 8, 1, KREA_NATIVE_TOKENS,
                                  txt_len=TXT_LEN)
        assert schedule.factors == (2,)
        assert set(schedule.windows) == {64}
        assert even_starts(GRID_H, 64) == [0, 61]
        assert even_starts(GRID_W, 64) == [0]        # ONE window across the width

    def test_the_heal_cap_moves_the_users_phase_one_counts(self):
        """The "+2 healing law" rewrites the knob — but only on a SHORT schedule.

        ``src/area.py:1827-1845``: the cap ``max(1, total - 2)`` is inside the
        ``is_short_schedule(total)`` branch, so it applies at 8 kSampler steps
        (the knob 7 and 8 both RUN AS 6) and NOT at 14 (a long schedule — the
        knob is honoured exactly, so ``phase1_steps=14`` at 14 steps is 100 %
        coarse and Phase II never runs at all).
        """
        from src.area_schedule import is_short_schedule

        assert is_short_schedule(8) and not is_short_schedule(14)
        assert max(1, 8 - 2) == 6
        # 14 steps, 14 coarse: the whole run is Phase I
        sch = build_schedule(GRID_H, GRID_W, 14, 14, KREA_NATIVE_TOKENS,
                             txt_len=TXT_LEN)
        assert len(sch.factors) == 14
        assert sch.windows == ()


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def krea_backend():
    """The mock backend, registered as ``comfy.ldm.krea2.model`` for the test."""
    with krea_backend_module() as backend:
        yield backend


@pytest.fixture
def flux_layers():
    """A mock ``comfy.ldm.flux.layers`` with a pristine ``attention``."""
    import test_area_node as flux

    saved = sys.modules.get("comfy.ldm.flux.layers")
    saved_parent = getattr(sys.modules.get("comfy.ldm.flux"), "layers", None)
    parent = sys.modules.setdefault("comfy.ldm.flux", types.ModuleType("comfy.ldm.flux"))
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.attention = flux.plain_flux_attention
    sys.modules["comfy.ldm.flux.layers"] = layers
    parent.layers = layers
    try:
        yield layers
    finally:
        sys.modules["comfy.ldm.flux.layers"] = saved
        parent.layers = saved_parent
        if saved is None:
            sys.modules.pop("comfy.ldm.flux.layers", None)


@pytest.fixture(autouse=True)
def _clean_area_state():
    """AReA's seam state and its one-time warning keys are process-global."""
    area._warned.clear()
    for seam in (area._SEAM, area._ZSEAM, area._KSEAM):
        seam.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    area._AREA_RUNTIME = None
    area._ZRUNTIME = None
    area._KRUNTIME = None
    yield
    area._warned.clear()
    for seam in (area._SEAM, area._ZSEAM, area._KSEAM):
        seam.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    area._AREA_RUNTIME = None
    area._ZRUNTIME = None
    area._KRUNTIME = None


def _mock_krea_model() -> ModelPatcher:
    """A patcher whose diffusion model is the mock Krea-2 backbone."""
    m = ModelPatcher()
    m.model.diffusion_model = SingleStreamDiT()
    m.model.model_sampling = types.SimpleNamespace()
    return m
