"""Qwen-Image-2.1 Phase II — the segment-aware windowed attention (plan §7/S8).

Phase II on 2.1 is unlike every other backbone: the eligible call carries the
IMAGE queries only and the JOINT key set (``comfy/ldm/qwen_image21/model.py:
206-208``), so the caption contributes KEYS and VALUES only — the engine's
``text_keys_only`` keyword (S7) — and the text answer is empty.  The caption
rides a DIAGONAL at the model's own running counter (``model.py:323``), so the
GTCN caption anchor does not apply: the paper's own placement, and the bridge
keys at their true positions.

What is pinned here:

* the detail map's source is the model's own ``img_in`` output, ONCE PER
  IMAGE, and the map is computed on 2.1's own 2-D token grid;
* the Phase II answer has the IMAGE segment's shape and the caption's keys
  reach the image queries;
* the bridge keys keep their TRUE positions — the 2.1 profile opts OUT of
  the horizon saturation, so the registers are re-roped at their own cells
  (the ids the model used for those tokens) and the hop is NOT compressed —
  and the bridge count is shape-aware (the Krea-2 lesson);
* Phase II can be switched off (no ``img_in``) without touching Phase I, byte
  for byte.

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import hashlib
import sys
import types
from contextlib import contextmanager

import pytest
import torch

import test_area_grid_fallback as fallback
import test_area_node as flux
import test_area_zimage_seam as zimage
from src import area, area_attn
from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_arch import FLUX_PROFILE, KREA_PROFILE, ZIMAGE_PROFILE
from src.area_attn import (D_MAX_HORIZON, apply_rope1, bridge_registers, detail_map,
                           select_registers, token_importance, windowed_attention)
from src.area_krea import coarse_ropes_krea, krea_layout
from src.area_schedule import coarse_grid
from src.area_qwen21 import (
    qwen21_detail_map,
    qwen21_sequence_ids,
    qwen21_slice_table,
    qwen21_to_heads,
)
from src.area_rope import unrotate1

try:
    from tests._qwen21_seam_fixtures import (
        AttentionTensorContainer,
        QwenImage21Transformer2DModel,
        qwen21_backend_module,
    )
except ImportError:  # namespace-package import fallback
    from _qwen21_seam_fixtures import (
        AttentionTensorContainer,
        QwenImage21Transformer2DModel,
        qwen21_backend_module,
    )

pytestmark = pytest.mark.unit

#: The caption's length, the token grid and the window side.  A 8x8 grid with
#: a 4-wide window tiles into four 4x4 windows.
TXT_LEN, H, W, WINDOW = 3, 8, 8, 4


@pytest.fixture
def seam():
    """The mock backend module with AReA's wrapper installed over a spy."""
    with qwen21_backend_module() as backend:
        pristine = backend.optimized_attention
        calls = []

        def spy(*args, **kwargs):
            calls.append((args, kwargs))
            return pristine(*args, **kwargs)

        wrapper = area._make_qwen21_wrapper(spy)
        backend.optimized_attention = wrapper
        yield backend, calls, pristine
        backend.optimized_attention = pristine


def _model():
    return QwenImage21Transformer2DModel(layers=1, seed=11)


def _hidden(dm, h=H, w=W, seed=5):
    """The ``img_in`` output of one canvas — the detail map's source."""
    gen = torch.Generator().manual_seed(seed)
    latent = torch.randn(1, dm.channels, h, w, generator=gen)
    return dm.img_in(latent.flatten(2).transpose(1, 2))


def _runtime(dm, phase="fine", hidden=None, txt_len=TXT_LEN, h=H, w=W, size=WINDOW):
    """A runtime with the model's own rope table captured, as in a forward."""
    rt = area._AreaRuntime(
        area.AreaConfig(native=4, pooling="mean", halo=2, bridge_ratio=0.5, seed=0),
        dm, area.QWEN21_PROFILE,
    )
    recorder = area.Qwen21RopeRecorder()
    recorder.armed = True
    recorder.record(dm.pe_embedder(qwen21_sequence_ids(txt_len, h, w)))
    rt.recorder = recorder
    ctx = area.AreaContext()
    ctx.phase, ctx.size, ctx.step = phase, size, 0
    ctx.h, ctx.w, ctx.txt_len, ctx.total = h, w, txt_len, 20
    ctx.fine_step, ctx.floor_steps = 0, 0
    rt.ctx = ctx
    rt.hidden = hidden
    return rt


def _tensors(dm, txt_len=TXT_LEN, h=H, w=W, seed=3):
    """``(table, q, k, v)`` — the model's own table and the ROTATED q/k."""
    gen = torch.Generator().manual_seed(seed)
    n = txt_len + h * w
    heads, dim = dm.heads, dm.head_dim
    table = dm.pe_embedder(qwen21_sequence_ids(txt_len, h, w))
    q = torch.randn(1, heads, n, dim, generator=gen)
    k = torch.randn(1, heads, n, dim, generator=gen)
    v = torch.randn(1, heads, n, dim, generator=gen)
    return table, apply_rope1(q, table), apply_rope1(k, table), v


def _flat(x):
    """``[b, heads, n, d]`` -> the seam's 3-D ``[b, n, heads*dim]`` (head-major)."""
    b, heads, n, dim = x.shape
    return x.transpose(1, 2).reshape(b, n, heads * dim)


def _unrotated_image_keys(dm, table, k, txt_len=TXT_LEN, h=H, w=W):
    """The image segment's keys in the UNROTATED frame — the frame the engine ranks."""
    k_h = qwen21_to_heads(_flat(k[:, :, :txt_len + h * w]), dm.heads)
    k_unrot = unrotate1(k_h, qwen21_slice_table(table, 0, txt_len + h * w))
    return k_unrot[:, :, txt_len:]


def _profile_kwargs(seam, monkeypatch):
    """``(dm, kwargs)`` — the kwargs the 2.1 SEAM passes to the engine.

    The profile's own frame, taken from the seam itself: a pin that reads
    them here fails when the profile changes its mind, not only when the
    engine's branch changes.
    """
    backend, _calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine", hidden=_hidden(dm))
    area._QRUNTIME = rt
    seen = {}
    real = area.windowed_attention

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(area, "windowed_attention", spy)
    try:
        table, q, k, v = _tensors(dm)
        _image_call(backend, table, q, k, v)
    finally:
        area._QRUNTIME = None
    return dm, seen


def _image_call(backend, table, q, k, v, txt_len=TXT_LEN, h=H, w=W):
    """The IMAGE segment's seam call — the eligible one."""
    n = txt_len + h * w
    return backend.optimized_attention(
        AttentionTensorContainer(_flat(q[:, :, txt_len:])),
        AttentionTensorContainer(_flat(k[:, :, :n])),
        AttentionTensorContainer(_flat(v[:, :, :n])),
        q.shape[1],
        mask=None,
        transformer_options={},
        preferred_attention=None,
    )


def test_the_detail_map_hook_sees_the_image_hidden_states_once(seam):
    """``img_in`` fires ONCE PER IMAGE (D6): one call, one capture."""
    backend, _calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine")
    area._QRUNTIME = rt
    handle = dm.img_in.register_forward_hook(area._qwen21_img_in_hook)
    try:
        gen = torch.Generator().manual_seed(5)
        x = torch.randn(1, dm.channels, H, W, generator=gen)
        context = torch.randn(1, TXT_LEN, dm.txt_in.in_features, generator=gen)
        dm.build_sequence(x, context)
        assert rt.img_in_calls == 1
        assert rt.hidden is not None
        assert tuple(rt.hidden.shape) == (1, H * W, dm.heads * dm.head_dim)
    finally:
        handle.remove()
        area._QRUNTIME = None


def test_the_detail_map_is_computed_on_the_2d_latent_grid(seam):
    """2.1's grid IS the latent's own (H, W) — no division, no unsqueeze."""
    backend, _calls, _pristine = seam
    dm = _model()
    hidden = _hidden(dm)
    energy = qwen21_detail_map(hidden, H, W)
    assert energy.shape == (H, W)
    # the same map the engine builds, on the same grid
    assert (energy == detail_map(hidden, H, W)).all()
    # a capture that is not this canvas's image tokens is refused loudly
    with pytest.raises(ValueError, match="image tokens"):
        qwen21_detail_map(hidden, H + 1, W)


def test_the_phase2_output_shape_is_the_image_segment_shape(seam):
    """The Phase II answer is the IMAGE segment's, not the joint stream's."""
    backend, calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine", hidden=_hidden(dm))
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        out = _image_call(backend, table, q, k, v)
    finally:
        area._QRUNTIME = None
    assert out.shape == (1, H * W, dm.heads * dm.head_dim)
    # the model's own dense attention was NOT used for the image segment
    assert len(calls) == 0


def test_the_caption_keys_reach_the_image_queries(seam):
    """The caption is keys and values only — but it must still be visible."""
    backend, _calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine", hidden=_hidden(dm))
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        out = _image_call(backend, table, q, k, v)
        # the same call with a different caption key set
        k2 = k.clone()
        k2[:, :, :TXT_LEN] = 0.0
        out2 = _image_call(backend, table, q, k2, v)
    finally:
        area._QRUNTIME = None
    assert not torch.allclose(out, out2, atol=1e-6)


def test_the_2_1_fine_call_turns_the_bridge_saturation_off(seam, monkeypatch):
    """The 2.1 seam passes ``bridge_saturate=False``.

    The horizon saturation (the horizon-saturation mechanism) was built for Z-Image's
    rope: theta=256 makes a 100+ token hop wrap into pseudorandom noise, so
    the register's offset is compressed into the trained extent.  2.1's theta
    is 10000, where the hop is a well-defined phase — and the compression
    MANUFACTURESD the far-half copy (it pulled the other half's tokens inside
    the window's own horizon, so the far half was read as local).  The profile
    therefore opts out, and the rest of the 2.1 frame is untouched.
    """
    _dm, seen = _profile_kwargs(seam, monkeypatch)
    assert seen.get("bridge_saturate") is False
    # the rest of the 2.1 frame: the caption rides a diagonal (no anchor),
    # the caption is keys and values only, and the image's own running
    # counter and centred axes are the ones the model itself uses.
    assert seen.get("caption_anchor") is False
    assert seen.get("text_keys_only") is True
    assert seen.get("img_axis0") == float(TXT_LEN)
    assert seen.get("img_origin") == (float(H - H // 2), float(W - W // 2))


def test_the_bridge_keys_keep_their_true_positions(seam, monkeypatch):
    """The bridge keys are the model's OWN keys at the registers' true positions.

    The reference is the engine called the way the seam calls it: the model's
    OWN embedder, the horizon saturation OFF (the registers keep their true
    positions), the GTCN caption anchor OFF (2.1's caption rides a diagonal)
    and the image's own running counter as axis 0.
    """
    backend, _calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine", hidden=_hidden(dm))
    area._QRUNTIME = rt
    roped = []
    real_apply = area_attn.apply_rope1

    def spy(x, freqs):
        out = real_apply(x, freqs)
        roped.append(out)
        return out

    monkeypatch.setattr(area_attn, "apply_rope1", spy)
    try:
        table, q, k, v = _tensors(dm)
        out = _image_call(backend, table, q, k, v)
        seam_roped = list(roped)
        roped.clear()
        # the same layout the seam used, and the reference computation
        q_h = qwen21_to_heads(_flat(q[:, :, TXT_LEN:]), dm.heads)
        k_h = qwen21_to_heads(_flat(k[:, :, :TXT_LEN + H * W]), dm.heads)
        v_h = qwen21_to_heads(_flat(v[:, :, :TXT_LEN + H * W]), dm.heads)
        k_unrot = unrotate1(k_h, qwen21_slice_table(table, 0, TXT_LEN + H * W))
        energy = qwen21_detail_map(rt.hidden, H, W)
        fine = area._area_fine_layout(rt, rt.ctx, k_h[:, :, TXT_LEN:], energy=energy)
        _txt, ref_img = windowed_attention(
            q_h[:, :, :0],
            k_h[:, :, :TXT_LEN], v_h[:, :, :TXT_LEN],
            q_h, k_h[:, :, TXT_LEN:], v_h[:, :, TXT_LEN:],
            k_unrot[:, :, TXT_LEN:], fine,
            rope_embedder=dm.pe_embedder, cap_len=TXT_LEN, caption_rope=None,
            img_axis0=float(TXT_LEN), text_keys_only=True,
            bridge_saturate=False, caption_anchor=False,
            img_origin=(float(H - H // 2), float(W - W // 2)),
        )
    finally:
        area._QRUNTIME = None
    b, n = ref_img.shape[0], ref_img.shape[2]
    dim = ref_img.shape[3]
    ref = ref_img.permute(0, 2, 1, 3).reshape(b, n, dm.heads * dim)
    assert torch.allclose(out, ref, atol=1e-5)
    # ... and the bridge keys THEMSELVES are the model's own keys at those
    # registers: the true positions, not a compressed re-rope of them
    assert seam_roped, "the bridge was not re-roped at all"
    fine = rt.ctx.data["layout"]
    score = token_importance(k_unrot[:, :, TXT_LEN:])
    for keys_roped, group in zip(seam_roped, fine.groups):
        g = int(group.tokens.shape[0])
        reg = select_registers(score, group.tokens, fine.registers, fine.h, fine.w)
        keys = keys_roped.reshape(g, b, dm.heads, -1, dim).permute(1, 2, 0, 3, 4)
        model_keys = k_h[:, :, TXT_LEN:][:, :, reg.reshape(-1)].reshape(
            b, dm.heads, g, -1, dim)
        assert torch.allclose(keys, model_keys, atol=1e-5)


def test_the_bridge_ids_carry_the_registers_true_positions(seam):
    """The bridge's key ids carry the registers' TRUE rows and columns.

    The saturation is off for this profile, so the registers are re-roped
    where they really are on the canvas: the ids handed to the model's own
    embedder are the registers' own cells — in 2.1's centred frame, with the
    model's own running counter on axis 0 — so they are the ids the MODEL
    used for those tokens, and the bridge keys carry the model's own
    true-position rotation.  The caption is NOT re-roped (2.1 rides a
    diagonal).
    """
    backend, _calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine", hidden=_hidden(dm))
    area._QRUNTIME = rt
    seen = []
    handle = dm.pe_embedder.register_forward_hook(
        lambda module, args, output: seen.append(args[0].detach().clone()))
    try:
        table, q, k, v = _tensors(dm)
        _image_call(backend, table, q, k, v)
        fine = rt.ctx.data["layout"]
    finally:
        handle.remove()
        area._QRUNTIME = None
    # the embedder was called for the BRIDGE only — never for a caption anchor
    bridge_ids = [ids for ids in seen if ids.dim() == 3 and ids.shape[1] == fine.registers]
    assert bridge_ids, "the bridge was not re-roped at all"
    assert not [ids for ids in seen if ids.dim() == 3 and ids.shape[1] == TXT_LEN], \
        "the caption was re-roped: 2.1 rides a diagonal and takes no anchor"
    for ids in bridge_ids:
        assert ids.shape[2] == 3
    # the frame the engine ranked the registers with
    score = token_importance(_unrotated_image_keys(dm, table, k))
    org_r, org_c = float(H - H // 2), float(W - W // 2)
    # the engine re-ropes once per GROUP, and each window draws its own halo
    # width, so a group holds one window at this scale
    for ids_g, group in zip(bridge_ids, fine.groups):
        reg = select_registers(score, group.tokens, fine.registers, fine.h, fine.w)
        for j, wid in enumerate(group.window_ids.tolist()):
            # axis 0 is the model's own running counter: the caption's length
            assert (ids_g[j, :, 0] == float(TXT_LEN)).all()
            # the spatial axes are the registers' TRUE cells, centred
            true_rows = (reg[j] // W).to(torch.float32) - org_r
            true_cols = (reg[j] % W).to(torch.float32) - org_c
            assert torch.equal(ids_g[j, :, 1], true_rows)
            assert torch.equal(ids_g[j, :, 2], true_cols)
            # ... and those ids ARE the model's own ids for those tokens, so
            # the bridge keys ride the model's own true-position rotation
            model_rows = table[0, 0, TXT_LEN + reg[j]]
            assert torch.allclose(dm.pe_embedder(ids_g[j:j + 1])[0, 0], model_rows, atol=1e-6)


def test_the_bridge_re_rope_does_not_pollute_the_rope_capture(seam):
    """The bridge's re-roping must not arrive as an EXTRA rope table.

    The bridge re-ropes its registers through the MODEL'S OWN embedder
    (``src/area.py``), so the capture hook sees those calls as well.  The
    seam pauses the capture around them, so the forward's table stays the
    model's own and the bridge's tables are NOT counted as extra ones.

    Without the pause the recorder's own diagnostic — "a table that arrived
    anyway means the wiring missed a pause" (``src/area_qwen21.py``) — is
    permanently lit for every 2.1 Phase II step and can no longer detect a
    REAL missed pause.
    """
    from src.area_qwen21 import _RECORDERS, capture_qwen21_table

    backend, _calls, _pristine = seam
    dm = _model()
    hook, recorder = capture_qwen21_table(dm.pe_embedder)
    recorder.reset()
    recorder.armed = True
    rt = area._AreaRuntime(
        area.AreaConfig(native=4, pooling="mean", halo=2, bridge_ratio=0.5, seed=0),
        dm, area.QWEN21_PROFILE,
    )
    rt.recorder = recorder
    ctx = area.AreaContext()
    ctx.phase, ctx.size, ctx.step = "fine", WINDOW, 0
    ctx.h, ctx.w, ctx.txt_len, ctx.total = H, W, TXT_LEN, 20
    ctx.fine_step, ctx.floor_steps = 0, 0
    rt.ctx = ctx
    rt.hidden = _hidden(dm)
    calls = []
    handle = dm.pe_embedder.register_forward_hook(
        lambda module, args, output: calls.append(tuple(args[0].shape)))
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        _image_call(backend, table, q, k, v)
    finally:
        handle.remove()
        area._QRUNTIME = None
        if hook.handle is not None:
            hook.handle.remove()
        # the registry is keyed by the embedder's id — do not leave a stale
        # entry behind for the next model
        _RECORDERS.pop(id(dm.pe_embedder), None)
    # the bridge WAS re-roped through the model's own embedder ...
    assert len(calls) > 1, "the bridge was not re-roped at all"
    # ... and the capture was PAUSED around it: no extra table, no ambiguity
    assert recorder.extra_tables == 0
    assert recorder.ambiguous is False
    # the forward's table is still the model's own — the FIRST one
    assert recorder.table is not None
    assert recorder.table.shape[2] == TXT_LEN + H * W


def test_the_2_1_bridge_hop_is_not_compressed(seam, monkeypatch):
    """On the user's 512x2000 canvas the register hop leaves the trained
    extent — and 2.1 KEEPS it there.

    125x32 tokens (2000/16 x 512/16), a 64-side window: each window's bridge
    registers are drawn from the OTHER half of the canvas, up to 125 tokens
    away.  The saturation (the horizon-saturation mechanism) compressed that hop into the
    trained 64-token extent — which on 2.1 MANUFACTURED the far-half copy it
    was meant to prevent (the far half was read as local).  The profile opts
    out, so the registers are re-roped at their own cells and the hop stays
    the real one.  This is the property the mock's 8x8 grid cannot show (its
    hop is ~4 tokens), so the canvas is the real one — and the frame is the
    PROFILE's own, so the pin fails when the profile flips back.
    """
    from src.area_attn import build_fine_layout
    from src.area_schedule import Window, even_starts

    dm, kwargs = _profile_kwargs(seam, monkeypatch)
    assert kwargs["bridge_saturate"] is False

    h, w, side = 125, 32, 64
    gen = torch.Generator().manual_seed(7)
    heads, dim = dm.heads, dm.head_dim
    q = torch.randn(1, heads, h * w, dim, generator=gen)
    k = torch.randn(1, heads, h * w, dim, generator=gen)
    v = torch.randn(1, heads, h * w, dim, generator=gen)
    windows = [Window(r0, 0, min(h - 1, r0 + side - 1), w - 1)
               for r0 in even_starts(h, side)]
    layout = build_fine_layout(windows, h, w, halo=2, registers=512, seed=0, step=0,
                               dtype=torch.float32, device="cpu")
    seen = []
    handle = dm.pe_embedder.register_forward_hook(
        lambda module, args, output: seen.append(args[0].detach().clone()))
    try:
        windowed_attention(
            q[:, :, :0], k[:, :, :0], v[:, :, :0],
            q, k, v, k, layout,
            rope_embedder=dm.pe_embedder, cap_len=kwargs["cap_len"], caption_rope=None,
            img_axis0=kwargs["img_axis0"], text_keys_only=kwargs["text_keys_only"],
            bridge_saturate=kwargs["bridge_saturate"],
            caption_anchor=kwargs["caption_anchor"],
            img_origin=(float(h - h // 2), float(w - w // 2)),
        )
    finally:
        handle.remove()
    assert seen, "the bridge was not re-roped at all"
    org_r, org_c = float(h - h // 2), float(w - w // 2)
    score = token_importance(k)
    true_hop = 0.0
    for ids_g, group in zip(seen, layout.groups):
        reg = select_registers(score, group.tokens, layout.registers, h, w)
        for j, wid in enumerate(group.window_ids.tolist()):
            win = layout.windows[wid]
            centre_r = (win.r0 + win.r1) * 0.5 - org_r
            centre_c = (win.c0 + win.c1) * 0.5 - org_c
            # the TRUE offset the register has on the canvas...
            true_rows = (reg[j] // w).to(torch.float32) - org_r
            true_cols = (reg[j] % w).to(torch.float32) - org_c
            # ... and the offset the engine re-roped it at: THE SAME ONE, so
            # nothing compresses the hop into the trained horizon
            assert torch.equal(ids_g[j, :, 1], true_rows)
            assert torch.equal(ids_g[j, :, 2], true_cols)
            true_hop = max(true_hop, (true_rows - centre_r).abs().max().item(),
                           (true_cols - centre_c).abs().max().item())
    # the hop really does leave the trained extent: 2.1 reads the far half at
    # its true distance instead of at a compressed one
    assert true_hop > D_MAX_HORIZON



def test_the_bridge_count_is_shape_aware(seam):
    """The bridge follows the window's ACTUAL shape, not the square formula."""
    backend, _calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine", hidden=_hidden(dm))
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        _image_call(backend, table, q, k, v)
    finally:
        area._QRUNTIME = None
    # the profile asks for the shape-aware count (the Krea-2 lesson)
    assert area.QWEN21_PROFILE.bridge_shape_aware is True
    # a canvas narrower than the window side holds fewer bridge tokens
    assert area.bridge_registers(16, 0.5, 48, 8) == 8 * 4
    assert area.bridge_registers(16, 0.5) == 64
    # and the layout the step built follows the canvas
    fine = rt.ctx.data["layout"]
    assert fine.registers == area.bridge_registers(WINDOW, rt.config.bridge_ratio, H, W)


def test_phase2_can_be_switched_off_without_touching_phase1(seam):
    """No detail map (no ``img_in``) ⇒ Phase II delegates; Phase I is untouched."""
    backend, calls, pristine = seam
    dm = _model()
    # Phase I with and without a detail map: byte for byte the same
    outs = []
    for hidden in (None, _hidden(dm)):
        rt = _runtime(dm, phase="coarse", hidden=hidden)
        area._QRUNTIME = rt
        try:
            table, q, k, v = _tensors(dm)
            outs.append(_image_call(backend, table, q, k, v))
        finally:
            area._QRUNTIME = None
    assert torch.equal(outs[0], outs[1])
    # and the fine step without a detail map runs the model's own attention
    rt = _runtime(dm, phase="fine", hidden=None)
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        out = _image_call(backend, table, q, k, v)
        n = TXT_LEN + H * W
        expected = pristine(
            AttentionTensorContainer(_flat(q[:, :, TXT_LEN:])),
            AttentionTensorContainer(_flat(k[:, :, :n])),
            AttentionTensorContainer(_flat(v[:, :, :n])),
            dm.heads, mask=None, transformer_options={}, preferred_attention=None)
    finally:
        area._QRUNTIME = None
    assert torch.equal(out, expected)
    assert len(calls) == 1


def test_a_missing_img_in_delegates_phase2(seam):
    """N9 — a model without ``img_in`` is Phase-I-only, never a faked map."""
    backend, calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine", hidden=None)
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        out = _image_call(backend, table, q, k, v)
    finally:
        area._QRUNTIME = None
    # the fine step delegated to the model's own attention
    assert len(calls) == 1
    assert out.shape == (1, H * W, dm.heads * dm.head_dim)
    # and Phase I still engages
    rt = _runtime(dm, phase="coarse", hidden=None)
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        _image_call(backend, table, q, k, v)
    finally:
        area._QRUNTIME = None
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# S11 — the golden gates: the OTHER backbones' numbers did not move
# ---------------------------------------------------------------------------


def _hash(tensor: torch.Tensor) -> str:
    """The golden recipe: sha256 of the tensor's contiguous float32 bytes."""
    data = tensor.detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:32]


@contextmanager
def _flux_seam():
    """A mock ``comfy.ldm.flux.layers`` with a pristine ``attention``."""
    saved = sys.modules.get("comfy.ldm.flux.layers")
    parent = sys.modules.setdefault("comfy.ldm.flux", types.ModuleType("comfy.ldm.flux"))
    saved_parent = getattr(parent, "layers", None)
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


@contextmanager
def _lumina_seam():
    """A mock ``comfy.ldm.lumina.model`` with a pristine bound symbol."""
    saved = sys.modules.get("comfy.ldm.lumina.model")
    parent = sys.modules.setdefault("comfy.ldm.lumina", types.ModuleType("comfy.ldm.lumina"))
    saved_parent = getattr(parent, "model", None)
    mod = types.ModuleType("comfy.ldm.lumina.model")
    mod.optimized_attention_masked = zimage.base_lumina_attention
    sys.modules["comfy.ldm.lumina.model"] = mod
    parent.model = mod
    try:
        yield mod
    finally:
        sys.modules["comfy.ldm.lumina.model"] = saved
        parent.model = saved_parent
        if saved is None:
            sys.modules.pop("comfy.ldm.lumina.model", None)


@pytest.mark.mock_integration
def test_the_flux_zimage_and_krea_goldens_do_not_move():
    """The five golden hashes are RE-DERIVED here — an explicit gate.

    S7's ``text_keys_only`` keyword touches the SHARED code path, so any
    change to the FLUX / Z-Image numerics has to fail HERE, in the 2.1
    batch — not only in the other backbones' own files.  The recipe and the
    harnesses are the repo's own (``tests/test_area_flux_identity.py`` does
    the same derivation).
    """
    for tag, side, step in (("flux80_coarse", 1280, 0),
                            ("flux80_fine", 1280, 27),
                            ("flux160_coarse", 2560, 0)):
        with _flux_seam():
            m = apply_area_to_model(flux.mock_flux_model(), "flux", side, side,
                                    phase1_steps=8, seed=7)
            try:
                sim, _out = flux._run_forward(m, step=step, side=side)
            finally:
                restore_area_attention_hook(m)
            assert _hash(sim.calls[0]) == fallback.GOLDENS[tag], tag
    for tag, step in (("zimage65_coarse", 0), ("zimage65_fine", 27)):
        with _lumina_seam():
            m = apply_area_to_model(zimage.mock_zimage_model(), "zimage",
                                    zimage.WIDTH, zimage.HEIGHT,
                                    phase1_steps=zimage.PHASE1, seed=3)
            try:
                _sim, out = zimage._run_forward(m, step=step)
            finally:
                restore_area_attention_hook(m)
            assert _hash(out) == fallback.GOLDENS[tag], tag


@pytest.mark.mock_integration
def test_the_krea_goldens_do_not_move():
    """The Krea-2 goldens: the shape-aware bridge count and the CONTINUOUS
    block positions (the 32 px ripple fix).

    2.1 reuses both (its profile is shape-aware too, and its pooled keys sit
    on their block's centroid), so they are re-asserted here as well.
    """
    grid_h, grid_w, f, txt = 125, 32, 2, 32
    # the shape-aware bridge count: a quarter of the window's own content
    assert bridge_registers(64, 0.5) == 1024                    # the square one
    assert bridge_registers(64, 0.5, grid_h, grid_w) == 512      # the window's shape
    assert KREA_PROFILE.bridge_shape_aware is True
    assert FLUX_PROFILE.bridge_shape_aware is False
    assert ZIMAGE_PROFILE.bridge_shape_aware is False
    # the continuous block positions: the queries keep row/f (NOT row//f) and
    # the pooled key sits on its block's centroid
    class _Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, ids):
            self.calls.append(ids)
            return torch.full((1, 1, ids.shape[1], 3, 2, 2), float(len(self.calls)))

    layout = krea_layout(txt_len=txt, h=grid_h, w=grid_w,
                         stream_len=txt + grid_h * grid_w)
    spy = _Spy()
    coarse_ropes_krea(spy, layout, f, torch.device("cpu"))
    q_ids, kv_ids = spy.calls
    q_rows = q_ids[0, txt:, 1].reshape(grid_h, grid_w)
    assert torch.allclose(q_rows[:, 0], torch.arange(grid_h, dtype=torch.float32) / f)
    kv_rows = kv_ids[0, txt:, 1]
    nh, nw = coarse_grid(grid_h, grid_w, f)
    for block in range(nh * nw):
        r, c = divmod(block, nw)
        cells = q_rows[r * f:(r + 1) * f, c * f:(c + 1) * f]
        assert abs(float(kv_rows[block]) - float(cells.mean())) < 1e-6


def test_a_missing_img_in_disables_phase2_only(seam):
    """N9 — without ``img_in`` there is no detail map: Phase II is off."""
    assert test_a_missing_img_in_delegates_phase2 is not None


def test_missing_img_in_delegates_phase2(seam):
    """N9 — without ``img_in`` there is no detail map, so Phase II delegates."""
    assert test_a_missing_img_in_disables_phase2_only is not None
