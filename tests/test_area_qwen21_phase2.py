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
* the bridge keys keep their true positions and the bridge count is
  shape-aware (the Krea-2 lesson);
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
from src import area
from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_arch import FLUX_PROFILE, KREA_PROFILE, ZIMAGE_PROFILE
from src.area_attn import apply_rope1, bridge_registers, detail_map, windowed_attention
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


def test_the_bridge_keys_keep_their_true_positions(seam):
    """The bridge keys are the model's OWN rotated keys, at their true positions.

    The reference is the engine called the way the design prescribes: the
    paper's caption placement (``rope_embedder=None`` — so the bridge is NOT
    re-roped by the horizon-saturation branch), the caption's own rotated keys
    and the model's own rotated image keys.
    """
    backend, _calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine", hidden=_hidden(dm))
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        out = _image_call(backend, table, q, k, v)
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
            rope_embedder=None, cap_len=TXT_LEN, caption_rope=None,
            img_axis0=None, text_keys_only=True,
        )
    finally:
        area._QRUNTIME = None
    b, n = ref_img.shape[0], ref_img.shape[2]
    dim = ref_img.shape[3]
    ref = ref_img.permute(0, 2, 1, 3).reshape(b, n, dm.heads * dim)
    assert torch.allclose(out, ref, atol=1e-5)


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
