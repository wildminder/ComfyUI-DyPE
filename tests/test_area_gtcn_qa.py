"""QA round 2 — an INDEPENDENT re-derivation of the area-fix3 (GTCN) criteria.

The engineer's own pins live in ``tests/test_area_gtcn.py``.  This file does
not trust them: it re-checks the acceptance criteria of the QA brief against
the implementation, with the REAL vendored rope code where the brief demands
it (comfy's own ``EmbedND``/``rope``, loaded in isolation).

Markers: @pytest.mark.unit
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

import pytest
import torch

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
from src.area_zimage import ZIMAGE_TRAINED_TOKENS, ZImageJointLayout, coarse_ropes_zimage

#: The ComfyUI install the vendored rope code lives in.
COMFY_FLUX = os.environ.get(
    "AQA_COMFY_FLUX",
    "C:/_Dev/ComfyUI_dev/ComfyUI_t211c130p313/ComfyUI/comfy/ldm/flux",
)

#: Z-Image's OWN configuration (comfy/model_detection.py:614-620):
#: dim=3840, n_heads=30 -> head_dim=128, axes_dims=[32, 48, 48], rope_theta=256.
Z_HEAD_DIM = 3840 // 30
Z_AXES = [32, 48, 48]
Z_THETA = 256.0


# ---------------------------------------------------------------------------
# comfy's REAL EmbedND, loaded in isolation
# ---------------------------------------------------------------------------


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def real_embednd():
    """comfy's real ``EmbedND`` (``comfy/ldm/flux/layers.py``), in isolation.

    The repo's conftest mocks the whole ``comfy`` package, so the vendored
    files are loaded here with the two modules they import stubbed out.  The
    rope maths itself is UNMODIFIED — only ``comfy.model_management``'s
    capability probe and the attention helpers are stubbed.
    """
    if not os.path.isdir(COMFY_FLUX):
        pytest.skip("the vendored comfy rope code is not available")
    import comfy.model_management as mm

    keys = ("comfy.ldm", "comfy.ldm.flux", "comfy.ldm.flux.math", "comfy.ldm.flux.layers",
            "comfy.ldm.modules", "comfy.ldm.modules.attention", "comfy.quant_ops")
    saved = {k: sys.modules.get(k) for k in keys}
    had_fp64 = hasattr(mm, "supports_fp64")
    old_fp64 = getattr(mm, "supports_fp64", None)
    try:
        mm.supports_fp64 = lambda device: True
        for name in ("comfy.ldm", "comfy.ldm.flux", "comfy.ldm.modules"):
            pkg = types.ModuleType(name)
            pkg.__path__ = []
            sys.modules[name] = pkg
        attn = types.ModuleType("comfy.ldm.modules.attention")
        attn.AttentionTensorContainer = object
        attn.ComfyAttention = object
        attn.optimized_attention = lambda *a, **k: None
        sys.modules["comfy.ldm.modules.attention"] = attn
        sys.modules["comfy.quant_ops"] = types.ModuleType("comfy.quant_ops")
        math_mod = _load_module("comfy.ldm.flux.math", os.path.join(COMFY_FLUX, "math.py"))
        sys.modules["comfy.ldm.flux"].math = math_mod
        layers = _load_module("comfy.ldm.flux.layers", os.path.join(COMFY_FLUX, "layers.py"))
        yield layers.EmbedND
    finally:
        if had_fp64:
            mm.supports_fp64 = old_fp64
        else:
            try:
                del mm.supports_fp64
            except AttributeError:
                pass
        for k in keys:
            if saved[k] is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = saved[k]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _Ids:
    """A passthrough embedder: the returned 'table' IS the ids."""

    def __init__(self):
        self.calls: list = []

    def __call__(self, ids: torch.Tensor) -> torch.Tensor:
        self.calls.append(ids.clone())
        return ids


def _layout(h: int, w: int, cap_len: int = 32, txt_len: int = 20) -> ZImageJointLayout:
    return ZImageJointLayout(txt_len=txt_len, cap_len=cap_len, h=h, w=w,
                             seq_len=cap_len + h * w)


def _ids_of(t: torch.Tensor, r: float, c: float) -> torch.Tensor:
    out = torch.zeros(1, int(t.numel()), 3, dtype=torch.float32)
    out[0, :, 0] = t.to(torch.float32)
    out[0, :, 1] = float(r)
    out[0, :, 2] = float(c)
    return out


def _mock_embednd():
    """The repo's mock of comfy's ``EmbedND`` (a faithful port of the real one)."""
    from test_area_zimage import MockEmbedND
    return MockEmbedND()


TXT, HEADS, DIM = 4, 2, 128


def _stream(h: int, w: int, txt: int = TXT, seed: int = 11):
    torch.manual_seed(seed)
    shape = (1, HEADS, txt + h * w, DIM)
    return (torch.randn(shape), torch.randn(shape), torch.randn(shape),
            torch.randn((1, HEADS, h * w, DIM)))


# ---------------------------------------------------------------------------
# A — the caption rotation route (c1: compose)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCaptionRotationRoute:
    """The anchor table composes with the model's own caption rotation."""

    def test_the_composition_is_exact_with_the_real_embedder(self, real_embednd):
        """R(t,0,0) then R(0,r,c) == R(t,r,c) — with comfy's OWN code.

        The brief demands this against the real embedder: the model's own
        caption table, the anchor table with the t axis zero, and the single
        table the model would have built for the caption over that window.
        """
        emb = real_embednd(dim=Z_HEAD_DIM, theta=Z_THETA, axes_dim=Z_AXES)
        torch.manual_seed(0)
        k_unrot = torch.randn(1, HEADS, TXT, Z_HEAD_DIM)
        t_axis = torch.arange(1, TXT + 1, dtype=torch.float32)
        r, c = 37.0, 11.0

        k_model = apply_rope1(k_unrot, emb(_ids_of(t_axis, 0.0, 0.0)))
        k_anchor = apply_rope1(k_model, emb(_ids_of(torch.zeros(TXT), r, c)))
        k_full = apply_rope1(k_unrot, emb(_ids_of(t_axis, r, c)))

        assert torch.allclose(k_anchor, k_full, atol=1e-6), \
            (k_anchor - k_full).abs().max().item()
        # the double-t alternative is observably different
        k_wrong = apply_rope1(k_model, emb(_ids_of(t_axis, r, c)))
        assert not torch.allclose(k_wrong, k_full, atol=1e-3)

    def test_the_anchor_table_is_the_identity_on_the_t_band(self, real_embednd):
        """The t band of the ANCHOR table (t = 0) must be the identity."""
        emb = real_embednd(dim=Z_HEAD_DIM, theta=Z_THETA, axes_dim=Z_AXES)
        zero = torch.zeros(TXT)
        anchor = emb(_ids_of(zero, 21.0, 13.0))
        n_t = Z_AXES[0] // 2
        assert torch.allclose(anchor[..., :n_t, 0, 0], torch.ones_like(anchor[..., :n_t, 0, 0]))
        assert torch.allclose(anchor[..., :n_t, 0, 1], torch.zeros_like(anchor[..., :n_t, 0, 1]))
        assert torch.allclose(anchor[..., :n_t, 1, 0], torch.zeros_like(anchor[..., :n_t, 1, 0]))
        assert torch.allclose(anchor[..., :n_t, 1, 1], torch.ones_like(anchor[..., :n_t, 1, 1]))
        # the model's own caption table has the same band layout but R(t_i)
        model = emb(_ids_of(torch.arange(1, TXT + 1, dtype=torch.float32), 0.0, 0.0))
        assert not torch.allclose(model[..., :n_t, 0, 0], anchor[..., :n_t, 0, 0])

    def test_the_axes_are_disjoint_frequency_bands(self, real_embednd):
        """One band per axis: each band depends on its own axis only."""
        emb = real_embednd(dim=Z_HEAD_DIM, theta=Z_THETA, axes_dim=Z_AXES)
        t_axis = torch.arange(1, TXT + 1, dtype=torch.float32)
        a = emb(_ids_of(t_axis, 5.0, 0.0))     # t=1..t, r=5, c=0
        b = emb(_ids_of(t_axis, 5.0, 9.0))     # t=1..t, r=5, c=9
        d = emb(_ids_of(t_axis, 9.0, 0.0))     # t=1..t, r=9, c=0
        n_t = Z_AXES[0] // 2
        n_y = n_t + Z_AXES[1] // 2
        n_x = n_y + Z_AXES[2] // 2
        assert a.shape[-3] == n_x
        # the t band never moves with the spatial axes
        assert torch.equal(a[..., :n_t, :, :], b[..., :n_t, :, :])
        assert torch.equal(a[..., :n_t, :, :], d[..., :n_t, :, :])
        # the y band depends on r only
        assert torch.equal(a[..., n_t:n_y, :, :], b[..., n_t:n_y, :, :])
        assert not torch.equal(a[..., n_t:n_y, :, :], d[..., n_t:n_y, :, :])
        # the x band depends on c only
        assert torch.equal(a[..., n_y:, :, :], d[..., n_y:, :, :])
        assert not torch.equal(a[..., n_y:, :, :], b[..., n_y:, :, :])

    def test_the_captions_own_t_positions_survive(self):
        """The composed caption keeps the model's own t positions 1..t."""
        embedder = _mock_embednd()
        layout = build_fine_layout([Window(0, 0, 15, 15)], 16, 16, 0, 0, 0, 0,
                                   torch.float32, "cpu")
        q, k, v, k_unrot = _stream(16, 16)
        cap_table = _mock_embednd()(_ids_of(torch.arange(1, TXT + 1, dtype=torch.float32),
                                            0.0, 0.0))
        k_rot = apply_rope1(k[:, :, :TXT], cap_table)
        ids = caption_window_ids(layout, TXT, "cpu")
        assert torch.all(ids[..., 0] == 0.0)            # the identity slot
        anchor = embedder(ids)
        composed = apply_rope1(k_rot, anchor)
        # the same vector the model would have produced for (t_i, r0, c0)
        expected = apply_rope1(
            k[:, :, :TXT],
            embedder(_ids_of(torch.arange(1, TXT + 1, dtype=torch.float32), 0.0, 0.0)),
        )
        assert torch.allclose(composed, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# B — the Phase I scaling
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhaseIScaling:
    """The Phase I coordinates and their relation to the spec's ``r / s``."""

    def _rows(self, h, w, f, rope_options=None):
        emb = _Ids()
        coarse_ropes_zimage(emb, _layout(h, w), f, "cpu", rope_options)
        return emb.calls

    def test_the_query_rows_are_block_units(self):
        """The tree's coordinates are ``r_canvas / f`` — block units."""
        emb = _Ids()
        coarse_ropes_zimage(emb, _layout(128, 128), 2, "cpu")
        ids = emb.calls[0][0]
        # the token at row 127, column 0
        assert float(ids[32 + 127 * 128, 1]) == pytest.approx(127 / 2.0)
        assert float(ids[32 + 127 * 128, 2]) == pytest.approx(0.0)
        # the last token: row 127, column 127
        assert float(ids[32 + 128 * 128 - 1, 1]) == pytest.approx(127 / 2.0)
        assert float(ids[32 + 128 * 128 - 1, 2]) == pytest.approx(127 / 2.0)

    def test_the_scaling_equals_the_spec_when_the_canvas_ratio_dominates(self):
        """``s >= f``: the result IS the spec's canvas-level ``r / s``."""
        for h, w, f in ((256, 256, 2), (256, 256, 1), (130, 130, 2), (192, 192, 2)):
            s_y = max(1.0, h / float(ZIMAGE_TRAINED_TOKENS))
            s_x = max(1.0, w / float(ZIMAGE_TRAINED_TOKENS))
            assert s_y >= f and s_x >= f, (h, w, f)
            emb = _Ids()
            coarse_ropes_zimage(emb, _layout(h, w), f, "cpu")
            ids = emb.calls[0][0]
            last = 32 + h * w - 1
            assert float(ids[last, 1]) == pytest.approx((h - 1) / s_y)
            assert float(ids[last, 2]) == pytest.approx((w - 1) / s_x)

    def test_the_scaling_is_a_no_op_when_the_pooling_dominates(self):
        """``s < f``: the tree's own block units are kept (bit for bit)."""
        for h, w, f in ((65, 65, 5), (80, 80, 2), (64, 64, 2), (32, 125, 2)):
            emb = _Ids()
            coarse_ropes_zimage(emb, _layout(h, w), f, "cpu")
            ids = emb.calls[0][0]
            last = 32 + h * w - 1
            assert float(ids[last, 1]) == pytest.approx((h - 1) / f)
            assert float(ids[last, 2]) == pytest.approx((w - 1) / f)

    def test_the_keys_carry_the_same_factor_as_the_queries(self):
        """A pooled key stays the MEAN of its block's query positions."""
        h = w = 256
        f = 2
        emb = _Ids()
        coarse_ropes_zimage(emb, _layout(h, w), f, "cpu")
        q_ids, kv_ids = emb.calls[0][0], emb.calls[1][0]
        nw = w // f
        for blk in (0, 1, 1000, nw * nw - 1):
            rows, cols = [], []
            for dr in range(f):
                for dc in range(f):
                    tok = 32 + ((blk // nw) * f + dr) * w + (blk % nw) * f + dc
                    rows.append(float(q_ids[tok, 1]))
                    cols.append(float(q_ids[tok, 2]))
            assert float(kv_ids[32 + blk, 1]) == pytest.approx(sum(rows) / len(rows))
            assert float(kv_ids[32 + blk, 2]) == pytest.approx(sum(cols) / len(cols))

    def test_the_caption_keeps_the_origin_and_the_t_axis_is_untouched(self):
        emb = _Ids()
        coarse_ropes_zimage(emb, _layout(256, 256), 2, "cpu")
        ids = emb.calls[0][0]
        assert torch.all(ids[:32, 0] == torch.arange(1, 33, dtype=torch.float32))
        assert torch.all(ids[:32, 1] == 0.0)
        assert torch.all(ids[:32, 2] == 0.0)
        assert torch.all(ids[32:, 0] == 33.0)

    def test_the_rope_options_are_applied_in_the_specs_order(self):
        """``(coord * norm) * h_scale + h_start`` — the spec's order."""
        opts = {"scale_y": 2.0, "shift_y": 3.0, "scale_x": 0.5, "shift_x": -1.0}
        emb = _Ids()
        coarse_ropes_zimage(emb, _layout(256, 256), 2, "cpu", opts)
        ids = emb.calls[0][0]
        last = 32 + 256 * 256 - 1
        # r_canvas/s = 255/4, then * 2.0 + 3.0
        assert float(ids[last, 1]) == pytest.approx((255 / 4.0) * 2.0 + 3.0)
        assert float(ids[last, 2]) == pytest.approx((255 / 4.0) * 0.5 - 1.0)

    def test_the_distance_to_the_caption_stays_inside_the_trained_domain(self):
        for h, w, f in ((128, 128, 1), (128, 128, 2), (256, 256, 2), (256, 256, 4),
                        (130, 130, 2), (192, 192, 3)):
            emb = _Ids()
            coarse_ropes_zimage(emb, _layout(h, w), f, "cpu")
            ids = emb.calls[0][0]
            assert float(ids[32:, 1].max()) <= ZIMAGE_TRAINED_TOKENS - 0.25, (h, w, f)
            assert float(ids[32:, 2].max()) <= ZIMAGE_TRAINED_TOKENS - 0.25, (h, w, f)


# ---------------------------------------------------------------------------
# C — the Phase II anchoring semantics
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhaseIIAnchoring:
    """The caption is anchored per group, at THAT group's window origin."""

    def _capture(self, monkeypatch, q, k, v, k_unrot, layout, embedder):
        captured: list = []
        real = torch.nn.functional.scaled_dot_product_attention

        def spy(query, key, value, *args, **kwargs):
            captured.append((query.detach().clone(), key.detach().clone(),
                             value.detach().clone()))
            return real(query, key, value, *args, **kwargs)

        monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", spy)
        out_txt, out_img = windowed_attention(q[:, :, :TXT], k[:, :, :TXT], v[:, :, :TXT],
                                               q[:, :, TXT:], k[:, :, TXT:], v[:, :, TXT:],
                                               k_unrot, layout,
                                               rope_embedder=embedder, cap_len=TXT)
        return captured, out_txt, out_img

    def test_a_multi_member_group_uses_its_own_windows(self, monkeypatch):
        """A group of two members must use ITS two windows, not the first two.

        The windows are ordered so that the two-member group holds the LAST
        two windows — the spec's ``layout.windows[:g]`` would hand it the
        first two, which sit elsewhere on the canvas.
        """
        windows = (Window(0, 0, 7, 7), Window(16, 16, 31, 31), Window(32, 32, 47, 47))
        layout = build_fine_layout(list(windows), 48, 48, 0, 0, 0, 0,
                                   torch.float32, "cpu")
        # the two big windows share a token count -> one group of two members
        sizes = sorted(int(g.tokens.shape[0]) for g in layout.groups)
        assert sizes == [1, 2], sizes
        multi = max(layout.groups, key=lambda g: int(g.tokens.shape[0]))
        assert multi.window_ids.tolist() == [1, 2]

        embedder = _mock_embednd()
        q, k, v, k_unrot = _stream(48, 48)
        captured, _, _ = self._capture(monkeypatch, q, k, v, k_unrot, layout, embedder)
        call = captured[1]
        for j, win_index in enumerate(multi.window_ids.tolist()):
            win = layout.windows[win_index]
            # the anchor table: the t axis ZERO (the model's own t rotation is
            # already inside the incoming keys)
            expected = apply_rope1(
                k[:, :, :TXT],
                embedder(_ids_of(torch.zeros(TXT), win.r0, win.c0)),
            )
            assert torch.allclose(call[1][j, :, :TXT, :], expected[0], atol=1e-5), j
            # and the composition with the model's own caption table is the
            # table the model would have built for the caption over that window
            cap = embedder(_ids_of(torch.arange(1, TXT + 1, dtype=torch.float32),
                                   0.0, 0.0))
            composed = apply_rope1(apply_rope1(k[:, :, :TXT], cap),
                                   embedder(_ids_of(torch.zeros(TXT), win.r0, win.c0)))
            full = apply_rope1(
                k[:, :, :TXT],
                embedder(_ids_of(torch.arange(1, TXT + 1, dtype=torch.float32),
                                 win.r0, win.c0)),
            )
            assert torch.allclose(composed, full, atol=1e-5), j

    def test_the_caption_t_axis_is_the_identity(self):
        """The anchor's t slot is an identity, so the model's own t stays."""
        layout = build_fine_layout([Window(8, 8, 15, 15)], 16, 16, 0, 0, 0, 0,
                                   torch.float32, "cpu")
        ids = caption_window_ids(layout, TXT, "cpu")
        assert torch.all(ids[..., 0] == 0.0)
        assert torch.all(ids[0, :, 1] == 8.0) and torch.all(ids[0, :, 2] == 8.0)

    def test_the_image_tokens_are_not_re_roped(self, monkeypatch):
        layout = build_fine_layout([Window(0, 0, 15, 15)], 16, 16, 0, 0, 0, 0,
                                   torch.float32, "cpu")
        q, k, v, k_unrot = _stream(16, 16)
        captured, _, _ = self._capture(monkeypatch, q, k, v, k_unrot, layout,
                                       _mock_embednd())
        assert torch.equal(captured[0][1][:, :, TXT:, :], k[:, :, TXT:, :])

    def test_the_bridge_saturation_still_uses_the_groups_own_window(self, monkeypatch):
        """area-fix2's bridge: the group's own window, inside the horizon."""
        windows = (Window(0, 0, 31, 31), Window(0, 32, 23, 63))
        layout = build_fine_layout(list(windows), 32, 64, 0, 4, 0, 0,
                                   torch.float32, "cpu")
        q, k, v, k_unrot = _stream(32, 64)
        seen: list = []

        def recording_embedder(ids):
            if torch.all(ids[..., 0] == TXT + 1):     # the bridge's table only
                seen.append(ids.detach().clone())
            return _mock_embednd()(ids)

        windowed_attention(q[:, :, :TXT], k[:, :, :TXT], v[:, :, :TXT],
                           q[:, :, TXT:], k[:, :, TXT:], v[:, :, TXT:],
                           k_unrot, layout, rope_embedder=recording_embedder,
                           cap_len=TXT)
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

    def test_the_text_output_is_the_mean_over_the_windows(self, monkeypatch):
        """``out_txt`` = the sum over the windows, divided by their count."""
        windows = (Window(0, 0, 15, 15), Window(0, 16, 7, 31))
        layout = build_fine_layout(list(windows), 16, 32, 0, 0, 0, 0,
                                   torch.float32, "cpu")
        assert len(layout.groups) == 2
        q, k, v, k_unrot = _stream(16, 32)
        captured, out_txt, _ = self._capture(monkeypatch, q, k, v, k_unrot, layout,
                                             _mock_embednd())
        manual = []
        for j, group in enumerate(layout.groups):
            query, key, value = captured[j]
            g = int(group.tokens.shape[0])
            o = torch.nn.functional.scaled_dot_product_attention(query, key, value)
            for m in range(g):
                manual.append(o[m, :, :TXT, :])
        expected = torch.stack(manual).sum(dim=0) / float(len(layout.windows))
        assert torch.allclose(out_txt, expected, atol=1e-5)

    def test_the_text_queries_are_rotated_per_window(self, monkeypatch):
        """The spec rotates the text QUERIES per window too — the impl does."""
        windows = (Window(0, 0, 15, 15), Window(0, 16, 7, 31))
        layout = build_fine_layout(list(windows), 16, 32, 0, 0, 0, 0,
                                   torch.float32, "cpu")
        q, k, v, k_unrot = _stream(16, 32)
        captured, _, _ = self._capture(monkeypatch, q, k, v, k_unrot, layout,
                                       _mock_embednd())
        text_q = [c[0][:, :, :TXT, :] for c in captured]
        assert not torch.allclose(text_q[0], text_q[1], atol=1e-4)
        for j, group in enumerate(layout.groups):
            win = layout.windows[int(group.window_ids[0])]
            # the anchor table: the t axis ZERO (the model's own t rotation is
            # already inside the incoming query)
            expected = apply_rope1(
                q[:, :, :TXT],
                _mock_embednd()(_ids_of(torch.zeros(TXT), win.r0, win.c0)),
            )
            assert torch.allclose(text_q[j][0, :, :TXT, :], expected[0], atol=1e-5)
            # and the composition with the model's own caption table equals the
            # table the model would have built for the caption over that window
            cap = _mock_embednd()(_ids_of(torch.arange(1, TXT + 1, dtype=torch.float32),
                                          0.0, 0.0))
            composed = apply_rope1(apply_rope1(q[:, :, :TXT], cap),
                                   _mock_embednd()(_ids_of(torch.zeros(TXT), win.r0, win.c0)))
            full = apply_rope1(
                q[:, :, :TXT],
                _mock_embednd()(_ids_of(torch.arange(1, TXT + 1, dtype=torch.float32),
                                        win.r0, win.c0)),
            )
            assert torch.allclose(composed, full, atol=1e-5)

    def test_without_an_embedder_the_caption_is_not_rotated(self, monkeypatch):
        """The FLUX call site: no embedder -> the caption keeps its own frame."""
        layout = build_fine_layout([Window(0, 0, 15, 15)], 16, 16, 0, 0, 0, 0,
                                   torch.float32, "cpu")
        q, k, v, k_unrot = _stream(16, 16)
        captured, _, _ = self._capture(monkeypatch, q, k, v, k_unrot, layout, None)
        assert torch.equal(captured[0][1][:, :, :TXT, :], k[:, :, :TXT, :])
