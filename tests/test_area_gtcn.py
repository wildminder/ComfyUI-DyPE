"""AReA — the Global Text-Coordinate Normalization (the caption-anchoring round).

Lumina2 puts EVERY caption token at the spatial origin (``(t, 0, 0)``,
``comfy/ldm/lumina/model.py:664-667``), so the rope distance between an image
token and the prompt is that token's own distance from the origin.  At Z-Image's
rope theta (256 — one rad per token) a distance beyond the trained 64 tokens
wraps the rotation several times and the text cross-attention collapses into
noise, which is what the user sees as hallucination outside the 1024x1024 box.

The fix has two halves, and this file pins both:

* **Phase I** (``src/area_zimage.py``): the coarse rope coordinates are scaled
  by the canvas extent ratio, so the canvas always fits the trained domain;
  the caption keeps the origin (the thing that bounds the distance to it) and
  the t axis is never touched;
* **Phase II** (``src/area_attn.py``): the caption is anchored to EACH window's
  own origin, so a window far from the top-left corner still meets the prompt
  inside the trained phase.

The pins are deliberately mutation-resistant: reverting the Phase I scaling or
the Phase II anchoring makes them FAIL (the mutation checks are described in
the caption-anchoring design note).

The FLUX path is untouched by both halves — only the Z-Image seam passes a rope
embedder — so the byte-identity goldens of ``tests/test_area_grid_fallback.py``
still hold and are re-checked here.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_gtcn.py
"""
from __future__ import annotations

import pytest
import torch

from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_attn import (
    D_MAX_HORIZON,
    apply_rope1,
    build_fine_layout,
    caption_window_ids,
    saturate_bridge_positions,
    select_registers,
    token_importance,
    windowed_attention,
)
from src.area_schedule import Window, coarse_grid
from src.area_zimage import ZImageJointLayout, coarse_ropes_zimage

# The repo's own seam doubles and the real EmbedND mock (lumina's axes) —
# reused rather than forked, so the rope structure these pins assert against is
# the one the model really uses.
import test_area_grid_fallback as grid
from test_area_grid_fallback import flux_layers  # noqa: F401
from test_area_zimage import MockEmbedND

#: The trained extent, in tokens (1024 px over the 16 px patch grid).
TRAINED = 64

#: A canvas past the trained extent: 128x128 tokens, ``s_y = s_x = 2.0``.
BIG = 128
#: A canvas far past it: 256x256 tokens, ``s = 4.0`` — the ramp's late rungs
#: pool with ``f = 2`` there, which is where the canvas leaves the domain.
HUGE = 256
#: The native extent: ``s = 1`` — nothing may move there.
NATIVE = 64

#: The caption block of the tiny streams (the model's own pad rule: 32).
CAP = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class PassthroughEmbedder:
    """The ids the adapter builds, handed back unchanged.

    ``coarse_ropes_zimage`` embeds ids, so a passthrough stand-in is the only
    way to read the POSITIONS it computed (the returned "tables" are the
    ``(t, row, col)`` ids themselves).
    """

    def __init__(self):
        self.calls: list = []

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(ids.clone())
        return ids


def _layout(h: int, w: int, txt_len: int = 20, cap_len: int = CAP) -> ZImageJointLayout:
    """A Z-Image joint layout of an ``h x w`` patch grid (no pad)."""
    seq = cap_len + h * w
    return ZImageJointLayout(txt_len=txt_len, cap_len=cap_len, h=h, w=w, seq_len=seq)


def _ids(embedder: PassthroughEmbedder, index: int) -> torch.Tensor:
    """The ids of one of the embedder's calls (``[1, n, 3]``)."""
    return embedder.calls[index]


def _caption_ids(cap_len: int) -> torch.Tensor:
    """The caption block's ids — the model's own ``(t, 0, 0)``."""
    ids = torch.zeros(cap_len, 3, dtype=torch.float32)
    ids[:, 0] = torch.arange(1, cap_len + 1, dtype=torch.float32)
    return ids


def _pre_change_ids(h: int, w: int, f: int, cap_len: int = CAP) -> torch.Tensor:
    """The QUERY ids the tree built BEFORE the GTCN — the byte-identity reference.

    The pre-change table: the caption at the origin on the t axis, then one
    query per image token at ``r / f`` — the block units the pooling factor
    produces.
    """
    pos = torch.arange(h * w)
    ids = torch.zeros(cap_len + h * w, 3, dtype=torch.float32)
    ids[:cap_len] = _caption_ids(cap_len)
    ids[cap_len:, 0] = float(cap_len + 1)
    ids[cap_len:, 1] = (pos // w).to(torch.float32) / float(f)
    ids[cap_len:, 2] = (pos % w).to(torch.float32) / float(f)
    return ids.unsqueeze(0)


def _pre_change_kv_ids(h: int, w: int, f: int, cap_len: int = CAP) -> torch.Tensor:
    """The KEY ids the tree built BEFORE the GTCN.

    The caption block, then one key per pooled block at its GEOMETRIC CENTROID
    ``R + (f - 1) / (2f)`` — a partial edge block (the grid does not tile) at
    the mean of its REAL members (``_block_offsets``).
    """
    nh, nw = coarse_grid(h, w, f)
    uniform = (f - 1.0) / (2.0 * f)
    rows = torch.arange(nh, dtype=torch.float32) + uniform
    cols = torch.arange(nw, dtype=torch.float32) + uniform
    if h % f:
        rows[-1] = nh - 1 + ((h - (nh - 1) * f) - 1) / (2.0 * f)
    if w % f:
        cols[-1] = nw - 1 + ((w - (nw - 1) * f) - 1) / (2.0 * f)
    block_rows = rows.repeat_interleave(nw)
    block_cols = cols.repeat(nh)
    ids = torch.zeros(cap_len + nh * nw, 3, dtype=torch.float32)
    ids[:cap_len] = _caption_ids(cap_len)
    ids[cap_len:, 0] = float(cap_len + 1)
    ids[cap_len:, 1] = block_rows
    ids[cap_len:, 2] = block_cols
    return ids.unsqueeze(0)


# ---------------------------------------------------------------------------
# Phase I: the canvas extent ratio
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhaseICoordinateMapping:
    """Phase I: the coarse coordinates are scaled by the canvas extent ratio."""

    def test_a_query_at_row_127_of_a_128_grid_maps_to_63_5(self):
        """The pin: h=128 (s=2) puts the far corner at 63.5 — inside the domain.

        The canvas extent ratio is the compression the GTCN guarantees.  The
        tree's own block units are kept only where they already hold the
        canvas inside the trained domain, so the pin holds for every factor
        that does not compress the canvas harder than the canvas ratio does —
        and where the factor does not compress at all (``f = 1``) the pin is
        the GTCN's own work.
        """
        for f in (1, 2):
            embedder = PassthroughEmbedder()
            coarse_ropes_zimage(embedder, _layout(BIG, BIG), f, "cpu")
            ids = _ids(embedder, 0)
            # the query at the last row, last column
            assert float(ids[0, CAP + BIG * BIG - 1, 1]) == pytest.approx(63.5)
            assert float(ids[0, CAP + BIG * BIG - 1, 2]) == pytest.approx(63.5)
            # ... and the whole column above it is scaled the same way
            assert float(ids[0, CAP + 127 * BIG, 1]) == pytest.approx(127 / 2.0)

    def test_the_scaling_reaches_the_canvas_the_pooling_left_outside(self):
        """A late ramp rung (``f < s``) is where the canvas leaves the domain.

        A 256x256 canvas pools with ``f = 4`` at the start of the ramp and
        with ``f = 2`` at its end.  The late rung's block units still span
        127.5 tokens — outside the trained 64 — so there the GTCN pulls the
        canvas back to 63.75.
        """
        embedder = PassthroughEmbedder()
        coarse_ropes_zimage(embedder, _layout(HUGE, HUGE), 2, "cpu")
        ids = _ids(embedder, 0)
        assert float(ids[0, CAP + HUGE * HUGE - 1, 1]) == pytest.approx(63.75)
        assert float(ids[0, CAP + HUGE * HUGE - 1, 2]) == pytest.approx(63.75)
        # the first rung (f = 4) already fits the domain and is untouched
        embedder = PassthroughEmbedder()
        coarse_ropes_zimage(embedder, _layout(HUGE, HUGE), 4, "cpu")
        assert torch.equal(_ids(embedder, 0), _pre_change_ids(HUGE, HUGE, 4))
        # the ramp's own late rung on a 130x130 canvas: the block units reach
        # 64.5 — just outside the trained extent — so the GTCN pulls them back
        embedder = PassthroughEmbedder()
        coarse_ropes_zimage(embedder, _layout(130, 130), 2, "cpu")
        ids = _ids(embedder, 0)
        assert float(ids[0, CAP + 130 * 130 - 1, 1]) == pytest.approx(
            (129 / 2.0) * (2.0 / (130 / 64.0)), abs=1e-3)

    def test_the_caption_stays_at_the_origin_and_the_t_axis_is_untouched(self):
        """The text keeps the origin (the distance to it is what must stay bounded)."""
        embedder = PassthroughEmbedder()
        coarse_ropes_zimage(embedder, _layout(BIG, BIG), 2, "cpu")
        ids = _ids(embedder, 0)
        cap = ids[0, :CAP]
        assert torch.all(cap[:, 0] == torch.arange(1, CAP + 1, dtype=torch.float32))
        assert torch.all(cap[:, 1] == 0.0)
        assert torch.all(cap[:, 2] == 0.0)
        # the image tokens keep the model's own t value (cap_len + 1)
        assert torch.all(ids[0, CAP:, 0] == float(CAP + 1))

    def test_the_pooled_keys_are_scaled_consistently_with_the_queries(self):
        """A pooled key is still the MEAN of its block's query positions."""
        embedder = PassthroughEmbedder()
        coarse_ropes_zimage(embedder, _layout(BIG, BIG), 2, "cpu")
        q_ids = _ids(embedder, 0)
        kv_ids = _ids(embedder, 1)
        nw = BIG // 2
        for blk in (0, 1, 37, nw * nw - 1):
            rows, cols = [], []
            for dr in (0, 1):
                for dc in (0, 1):
                    token = CAP + (2 * (blk // nw) + dr) * BIG + 2 * (blk % nw) + dc
                    rows.append(float(q_ids[0, token, 1]))
                    cols.append(float(q_ids[0, token, 2]))
            assert float(kv_ids[0, CAP + blk, 1]) == pytest.approx(sum(rows) / 4.0)
            assert float(kv_ids[0, CAP + blk, 2]) == pytest.approx(sum(cols) / 4.0)

    def test_the_distance_to_the_caption_stays_inside_the_trained_domain(self):
        """Every image position is within the trained extent of the origin."""
        for h, w, f in ((BIG, BIG, 1), (BIG, BIG, 2), (BIG, BIG, 4),
                        (HUGE, HUGE, 2), (HUGE, HUGE, 4)):
            embedder = PassthroughEmbedder()
            coarse_ropes_zimage(embedder, _layout(h, w), f, "cpu")
            ids = _ids(embedder, 0)
            assert float(ids[0, CAP:, 1].max()) <= TRAINED - 0.25
            assert float(ids[0, CAP:, 2].max()) <= TRAINED - 0.25

    def test_the_native_extent_is_byte_identical(self):
        """A 64x64 canvas (s=1) produces EXACTLY the pre-GTCN ids."""
        embedder = PassthroughEmbedder()
        coarse_ropes_zimage(embedder, _layout(NATIVE, NATIVE), 2, "cpu")
        assert torch.equal(_ids(embedder, 0), _pre_change_ids(NATIVE, NATIVE, 2))
        assert torch.equal(_ids(embedder, 1), _pre_change_kv_ids(NATIVE, NATIVE, 2))

    def test_a_grid_inside_the_trained_extent_is_byte_identical(self):
        """Every canvas that already fits the domain is untouched, bit for bit."""
        for h, w, f in ((65, 65, 5), (32, 125, 2), (80, 80, 2), (48, 48, 2)):
            embedder = PassthroughEmbedder()
            coarse_ropes_zimage(embedder, _layout(h, w), f, "cpu")
            assert torch.equal(_ids(embedder, 0), _pre_change_ids(h, w, f)), (h, w, f)
            assert torch.equal(_ids(embedder, 1), _pre_change_kv_ids(h, w, f)), (h, w, f)

    def test_the_scaling_is_applied_where_the_block_units_do_not_fit(self):
        """A pooling factor below the canvas ratio pulls the canvas back in."""
        embedder = PassthroughEmbedder()
        coarse_ropes_zimage(embedder, _layout(BIG, BIG), 1, "cpu")
        ids = _ids(embedder, 0)
        # f = 1 < s = 2: the far corner lands on (h - 1) / s
        assert float(ids[0, CAP + BIG * BIG - 1, 1]) == pytest.approx((BIG - 1) / 2.0)
        # ... while the block units of a grid that already fits are untouched
        embedder = PassthroughEmbedder()
        coarse_ropes_zimage(embedder, _layout(BIG, BIG), 2, "cpu")
        assert torch.equal(_ids(embedder, 0), _pre_change_ids(BIG, BIG, 2))


# ---------------------------------------------------------------------------
# Phase II: the caption is anchored to each window's own origin
# ---------------------------------------------------------------------------

#: A 16x16 canvas split into four 8x8 windows — every window has its own
#: origin, so the anchoring is observable.
H, W = 16, 16
WINDOWS = (Window(0, 0, 7, 7), Window(0, 8, 7, 15),
           Window(8, 0, 15, 7), Window(8, 8, 15, 15))
TXT, HEADS, DIM = 4, 2, 128          # DIM matches the mock embedder's head dim


def _four_window_layout(registers: int = 0):
    """The layout of the four windows above (one group, four members)."""
    return build_fine_layout(list(WINDOWS), H, W, 0, registers, 0, 0,
                             torch.float32, "cpu")


def _model_caption_table(embedder, t: int) -> torch.Tensor:
    """The model's OWN caption table — ``(t, 0, 0)`` (``embed_cap``)."""
    ids = torch.zeros(1, t, 3, dtype=torch.float32)
    ids[0, :, 0] = torch.arange(1, t + 1, dtype=torch.float32)
    return embedder(ids)


def _stream(h: int = H, w: int = W, txt: int = TXT, seed: int = 5):
    """The joint stream: the caption half (rotated by the model) and the image half.

    Returns the UNROTATED caption keys as well — the engine's input is the
    model's own rotation of them, and the pins compare against the rotation
    the model would have used.
    """
    torch.manual_seed(seed)
    k_txt_unrot = torch.randn((1, HEADS, txt, DIM))
    q_img = torch.randn((1, HEADS, h * w, DIM))
    k_img = torch.randn((1, HEADS, h * w, DIM))
    v_img = torch.randn((1, HEADS, h * w, DIM))
    k_unrot = torch.randn((1, HEADS, h * w, DIM))
    return k_txt_unrot, q_img, k_img, v_img, k_unrot


@pytest.mark.unit
class TestPhaseIICaptionAnchoring:
    """Phase II: every window sees the caption at its own origin."""

    def test_the_caption_ids_carry_each_window_origin(self):
        """One row per window, the caption at that window's (r0, c0)."""
        layout = _four_window_layout()
        ids = caption_window_ids(layout, TXT, "cpu")
        assert tuple(ids.shape) == (len(layout.windows), TXT, 3)
        for i, win in enumerate(layout.windows):
            assert torch.all(ids[i, :, 1] == float(win.r0))
            assert torch.all(ids[i, :, 2] == float(win.c0))
        # the t axis is the IDENTITY: the model's own t rotation is kept
        assert torch.all(ids[..., 0] == 0.0)

    def test_different_windows_get_different_caption_rotations(self):
        """The rotations differ per window — the anchoring is not a constant."""
        embedder = MockEmbedND()
        layout = _four_window_layout()
        ids = caption_window_ids(layout, TXT, "cpu")
        table = embedder(ids)
        assert not torch.allclose(table[0], table[1])
        assert not torch.allclose(table[0], table[3])

    def test_the_composed_rotation_is_the_models_own(self, monkeypatch):
        """The re-roped caption equals R(t_i, r_win, c_win) of the UNROTATED keys.

        The seam's caption keys arrive already rotated in the t axis by the
        model (``embed_cap``: ``(t_i, 0, 0)``).  The anchor table's t axis is
        the identity, so the composition is exactly the table the model would
        have built for the caption over that window.
        """
        embedder = MockEmbedND()
        layout = _four_window_layout()
        group = layout.groups[0]
        k_txt_unrot, q_img, k_img, v_img, k_unrot = _stream()
        cap_table = _model_caption_table(embedder, TXT)
        k_txt = apply_rope1(k_txt_unrot, cap_table)     # the seam's caption keys

        captured: list = []
        real = torch.nn.functional.scaled_dot_product_attention

        def spy(query, key, value, *args, **kwargs):
            captured.append(key.detach().clone())
            return real(query, key, value, *args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
        windowed_attention(k_txt, k_txt, k_txt, q_img, k_img, v_img, k_unrot,
                           layout, rope_embedder=embedder, cap_len=TXT)
        assert len(captured) == 1                    # one group
        keys = captured[0]                           # [b*g, heads, t + n, dim]
        for j, win_index in enumerate(group.window_ids.tolist()):
            win = layout.windows[win_index]
            expected_ids = torch.zeros(1, TXT, 3, dtype=torch.float32)
            expected_ids[0, :, 0] = torch.arange(1, TXT + 1, dtype=torch.float32)
            expected_ids[0, :, 1] = float(win.r0)
            expected_ids[0, :, 2] = float(win.c0)
            expected = apply_rope1(k_txt_unrot, embedder(expected_ids))
            assert torch.allclose(keys[j][:, :TXT, :], expected, atol=1e-5), j

    def test_each_group_gets_its_own_caption_rotation(self, monkeypatch):
        """Two groups, two windows, two DIFFERENT caption rotations.

        The group -> window mapping is the group's own (``group.window_ids``),
        so the second group's caption sits at the SECOND window's origin — not
        at the first window's, and not at the canvas origin.
        """
        embedder = MockEmbedND()
        windows = (Window(0, 0, 15, 15), Window(0, 16, 7, 31))
        layout = build_fine_layout(list(windows), 16, 32, 0, 0, 0, 0,
                                   torch.float32, "cpu")
        assert len(layout.groups) == 2
        k_txt_unrot, q_img, k_img, v_img, k_unrot = _stream(h=16, w=32)
        cap_table = _model_caption_table(embedder, TXT)
        k_txt = apply_rope1(k_txt_unrot, cap_table)

        captured: list = []
        real = torch.nn.functional.scaled_dot_product_attention

        def spy(query, key, value, *args, **kwargs):
            captured.append(key.detach().clone())
            return real(query, key, value, *args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
        windowed_attention(k_txt, k_txt, k_txt, q_img, k_img, v_img, k_unrot,
                           layout, rope_embedder=embedder, cap_len=TXT)
        assert len(captured) == 2                     # one call per group
        seen = []
        for j, group in enumerate(layout.groups):
            win = layout.windows[int(group.window_ids[0])]
            expected_ids = torch.zeros(1, TXT, 3, dtype=torch.float32)
            expected_ids[0, :, 0] = torch.arange(1, TXT + 1, dtype=torch.float32)
            expected_ids[0, :, 1] = float(win.r0)
            expected_ids[0, :, 2] = float(win.c0)
            expected = apply_rope1(k_txt_unrot, embedder(expected_ids))
            assert torch.allclose(captured[j][:, :, :TXT, :], expected, atol=1e-5), j
            seen.append(captured[j][:, :, :TXT, :])
        # the two groups really do see the prompt at different places
        assert not torch.allclose(seen[0], seen[1], atol=1e-4)

    def test_the_anchor_table_must_not_repeat_the_t_rotation(self):
        """A t axis of 1..t on the ALREADY-ROTATED keys would double the t phase."""
        embedder = MockEmbedND()
        cap_table = _model_caption_table(embedder, TXT)
        k_rot = apply_rope1(_stream()[0], cap_table)
        wrong_ids = torch.zeros(1, TXT, 3, dtype=torch.float32)
        wrong_ids[0, :, 0] = torch.arange(1, TXT + 1, dtype=torch.float32)
        wrong_ids[0, :, 1] = 8.0
        wrong_ids[0, :, 2] = 8.0
        right_ids = wrong_ids.clone()
        right_ids[0, :, 0] = 0.0
        wrong = apply_rope1(k_rot, embedder(wrong_ids))
        right = apply_rope1(k_rot, embedder(right_ids))
        assert not torch.allclose(wrong, right)

    def test_the_image_tokens_are_not_re_roped(self, monkeypatch):
        """The image half of the keys rides the model's own rotation, untouched."""
        layout = build_fine_layout([Window(0, 0, 15, 15)], H, W, 0, 0, 0, 0,
                                   torch.float32, "cpu")
        k_txt_unrot, q_img, k_img, v_img, k_unrot = _stream()
        captured: list = []
        real = torch.nn.functional.scaled_dot_product_attention

        def spy(query, key, value, *args, **kwargs):
            captured.append(key.detach().clone())
            return real(query, key, value, *args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
        windowed_attention(k_txt_unrot, k_txt_unrot, k_txt_unrot, q_img, k_img, v_img,
                           k_unrot, layout, rope_embedder=MockEmbedND(), cap_len=TXT)
        assert captured
        # the image keys of the only group are the ones the caller passed
        assert torch.equal(captured[0][:, :, TXT:, :], k_img)

    def test_the_bridge_branch_still_uses_the_groups_own_window(self):
        """The the horizon saturation is untouched: the group's own window."""
        windows = (Window(0, 0, 31, 31), Window(0, 32, 23, 63))
        layout = build_fine_layout(list(windows), 32, 64, 0, 4, 0, 0,
                                   torch.float32, "cpu")
        assert len(layout.groups) == 2
        k_txt_unrot, q_img, k_img, v_img, k_unrot = _stream(h=32, w=64)
        seen: list = []

        def recording_embedder(ids):
            # the bridge's table carries the image's t position (cap_len + 1);
            # the caption's anchor table has an identity t axis
            if torch.all(ids[..., 0] == TXT + 1):
                seen.append(ids.detach().clone())
            return MockEmbedND()(ids)

        windowed_attention(k_txt_unrot, k_txt_unrot, k_txt_unrot, q_img, k_img, v_img,
                           k_unrot, layout, rope_embedder=recording_embedder, cap_len=TXT)
        assert len(seen) == len(layout.groups)
        score = token_importance(k_unrot)
        for j, group in enumerate(layout.groups):
            reg = select_registers(score, group.tokens, layout.registers, 32, 64)
            win = layout.windows[int(group.window_ids[0])]
            exp_r, exp_c = saturate_bridge_positions(reg, win, 32, 64, "cpu")
            assert torch.allclose(seen[j][..., 1], exp_r, atol=1e-4)
            assert torch.allclose(seen[j][..., 2], exp_c, atol=1e-4)
            centre_r = (win.r0 + win.r1) * 0.5
            assert bool(torch.all((seen[j][..., 1] - centre_r).abs() <= D_MAX_HORIZON + 1e-3))


# ---------------------------------------------------------------------------
# The FLUX path is untouched
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnchangedPaths:
    """The GTCN is Z-Image-only: FLUX stays byte-identical."""

    def test_the_flux_seam_answers_are_byte_identical(self, request):
        """The FLUX goldens of ``tests/test_area_grid_fallback.py`` still hold."""
        # the grid-fallback module's own mock of comfy's flux layers — pulled in
        # by name (the fixture is imported, not requested, so the name is free)
        request.getfixturevalue("flux_layers")
        for tag, side, step in (("flux80_coarse", grid.GOLD_SIDE, 0),
                                ("flux80_fine", grid.GOLD_SIDE, grid.STEPS - 1),
                                ("flux160_coarse", grid.BIG_SIDE, 0)):
            m = apply_area_to_model(grid.flux.mock_flux_model(), "flux", side, side,
                                    phase1_steps=grid.PHASE1, seed=7)
            try:
                sim, _out = grid.flux._run_forward(m, step=step, side=side)
            finally:
                restore_area_attention_hook(m)
            assert grid._hash(sim.calls[0]) == grid.GOLDENS[tag], tag

    def test_the_zimage_phase_i_of_the_native_extent_is_byte_identical(self):
        """A 64x64 Z-Image canvas produces exactly the pre-GTCN coarse ids."""
        embedder = PassthroughEmbedder()
        coarse_ropes_zimage(embedder, _layout(NATIVE, NATIVE), 2, "cpu")
        assert torch.equal(_ids(embedder, 0), _pre_change_ids(NATIVE, NATIVE, 2))
        assert torch.equal(_ids(embedder, 1), _pre_change_kv_ids(NATIVE, NATIVE, 2))
