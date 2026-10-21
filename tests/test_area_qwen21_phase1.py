"""Qwen-Image-2.1 Phase I — the seam install and the coarse pooled attention
(plan §7/S6).

The Qwen-Image-2.1 seam is a FOURTH binding: 2.1 imports the *unmasked*
``optimized_attention`` into its OWN module (``comfy/ldm/qwen_image21/model.py:
15``) and calls the BOUND name per segment (``model.py:211``).  The eligible
call is the IMAGE segment: the queries are the ``h*w`` image rows and the keys
are the JOINT set — the caption AND the image (``model.py:206-208``).

What is pinned here:

* the eligible image segment IS AReA'd (coarse pooled attention), and its
  answer matches the hand-computed reference;
* every other call is DELEGATED UNCHANGED — the caption segment (a causal
  mask) and the prefix-cache path (NO ``mask`` keyword) — and the delegate is
  byte-identical to the unpatched model;
* the pristine symbol receives the ORIGINAL arguments (the same container
  objects, the same mask, the same heads);
* the wrapper never consumes a container it then delegates, and the Phase II
  (fine) steps delegate untouched.

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

from src import area
from src.area_attn import apply_rope1, pool_keys_values
from src.area_qwen21 import (
    QWEN21_NO_MASK,
    coarse_ropes_qwen21,
    qwen21_layout,
    qwen21_sequence_ids,
    qwen21_slice_table,
    qwen21_to_heads,
)
from src.area_rope import unrotate1

try:
    from tests._qwen21_seam_fixtures import (
        AttentionTensorContainer,
        QwenImage21Transformer2DModel,
        install_recorder,
        qwen21_backend_module,
    )
except ImportError:  # namespace-package import fallback
    from _qwen21_seam_fixtures import (
        AttentionTensorContainer,
        QwenImage21Transformer2DModel,
        install_recorder,
        qwen21_backend_module,
    )

try:
    from comfy.model_patcher import ModelPatcher
except ImportError:  # pragma: no cover - the mock env always provides it
    from tests.conftest import MockModelPatcher as ModelPatcher

pytestmark = pytest.mark.unit

TXT_LEN, H, W, FACTOR = 3, 4, 4, 2


@pytest.fixture
def seam():
    """The mock backend module with AReA's wrapper installed over a spy.

    The spy records the arguments the PRISTINE symbol receives, so the
    "the pristine got the original arguments" contract can be asserted.
    """
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


def _unet_wrapper(m):
    """The wrapper the patcher installed (mock and real ModelPatcher)."""
    try:
        return m.model_options["model_function_wrapper"]
    except (AttributeError, KeyError, TypeError):
        return getattr(m, "_unet_wrapper", None)


def _runtime(dm, phase="coarse", txt_len=TXT_LEN, h=H, w=W, factor=FACTOR):
    """A runtime with the model's own rope table captured, as in a forward."""
    rt = area._AreaRuntime(area.AreaConfig(pooling="mean"), dm, area.QWEN21_PROFILE)
    recorder = area.Qwen21RopeRecorder()
    recorder.armed = True
    ids = qwen21_sequence_ids(txt_len, h, w)
    recorder.record(dm.pe_embedder(ids))
    rt.recorder = recorder
    ctx = area.AreaContext()
    ctx.phase, ctx.size, ctx.step = phase, factor, 0
    ctx.h, ctx.w, ctx.txt_len, ctx.total = h, w, txt_len, 20
    rt.ctx = ctx
    return rt


def _tensors(dm, txt_len=TXT_LEN, h=H, w=W, seed=3):
    """``(table, q, k, v)`` — the model's own table and the ROTATED q/k.

    The q/k are built heads-first and rotated with the engine's own rotation
    (a byte-for-byte port of comfy's ``_apply_rope1``), which is what the
    model does to its ``(B, N, H, D)`` tensors (``model.py:99-100``).
    """
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


def _image_call(backend, table, q, k, v, txt_len=TXT_LEN):
    """The IMAGE segment's seam call — the eligible one."""
    n = txt_len + H * W
    return backend.optimized_attention(
        AttentionTensorContainer(_flat(q[:, :, txt_len:])),
        AttentionTensorContainer(_flat(k[:, :, :n])),
        AttentionTensorContainer(_flat(v[:, :, :n])),
        q.shape[1],
        mask=None,
        transformer_options={},
        preferred_attention=None,
    )


def _reference(dm, rt, q, k, v, heads, txt_len=TXT_LEN, h=H, w=W, factor=FACTOR):
    """The Phase I computation, written out."""
    table = rt.recorder.table
    layout = qwen21_layout(txt_len, h, w, h * w, txt_len + h * w)
    q_h = qwen21_to_heads(_flat(q[:, :, txt_len:]), heads)
    k_h = qwen21_to_heads(_flat(k[:, :, :txt_len + h * w]), heads)
    v_h = qwen21_to_heads(_flat(v[:, :, :txt_len + h * w]), heads)
    q_un = unrotate1(q_h, qwen21_slice_table(table, txt_len, layout.k_len))
    k_un = unrotate1(k_h, qwen21_slice_table(table, 0, layout.k_len))
    rope_q, rope_kv = coarse_ropes_qwen21(dm.pe_embedder, layout, factor, q.device,
                                          rt.recorder)
    k_pooled, v_pooled = pool_keys_values(
        k_un, v_h, txt_len, h, w, factor, rt.config.pooling,
        seed=rt.config.seed, step=ctx_step(rt), block=0)
    out = F.scaled_dot_product_attention(
        apply_rope1(q_un, rope_q), apply_rope1(k_pooled, rope_kv), v_pooled)
    b, n = out.shape[0], out.shape[2]
    return out.permute(0, 2, 1, 3).reshape(b, n, heads * out.shape[3])


def ctx_step(rt):
    return rt.ctx.step


def test_the_image_segment_is_rea_pooled(seam):
    """The eligible call is AReA's, not the model's own dense attention."""
    backend, calls, pristine = seam
    dm = _model()
    rt = _runtime(dm)
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        out = _image_call(backend, table, q, k, v)
        n = TXT_LEN + H * W
        dense = pristine(
            AttentionTensorContainer(_flat(q[:, :, TXT_LEN:])),
            AttentionTensorContainer(_flat(k[:, :, :n])),
            AttentionTensorContainer(_flat(v[:, :, :n])),
            dm.heads, mask=None, transformer_options={}, preferred_attention=None)
    finally:
        area._QRUNTIME = None
    assert out.shape == (1, H * W, dm.heads * dm.head_dim)
    # the AReA answer differs from the model's own dense attention
    assert not torch.allclose(out, dense, atol=1e-6)


def test_the_image_segment_matches_the_hand_computed_reference(seam):
    """The Phase I numbers: unrotated pooling, the coarse tables, the flatten."""
    backend, calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm)
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        out = _image_call(backend, table, q, k, v)
        ref = _reference(dm, rt, q, k, v, dm.heads)
    finally:
        area._QRUNTIME = None
    assert torch.allclose(out, ref, atol=1e-5)


def test_the_text_segment_output_is_bit_identical_to_the_pristine(seam):
    """The caption carries a causal mask: it is DELEGATED, bit for bit."""
    backend, calls, pristine = seam
    dm = _model()
    rt = _runtime(dm)
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        n = TXT_LEN
        mask = torch.ones((n, n), dtype=torch.bool).tril()
        args = (AttentionTensorContainer(_flat(q[:, :, :n])),
                AttentionTensorContainer(_flat(k[:, :, :n])),
                AttentionTensorContainer(_flat(v[:, :, :n])))
        out = backend.optimized_attention(*args, dm.heads, mask=mask,
                                           transformer_options={},
                                           preferred_attention=None)
        expected = pristine(
            AttentionTensorContainer(_flat(q[:, :, :n])),
            AttentionTensorContainer(_flat(k[:, :, :n])),
            AttentionTensorContainer(_flat(v[:, :, :n])),
            dm.heads, mask=mask, transformer_options={}, preferred_attention=None)
    finally:
        area._QRUNTIME = None
    assert torch.equal(out, expected)
    assert len(calls) == 1


def test_the_prefix_cache_path_is_delegated(seam):
    """The cached path passes NO ``mask`` keyword: it is delegated too."""
    backend, calls, pristine = seam
    dm = _model()
    rt = _runtime(dm)
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        n = TXT_LEN + H * W
        out = backend.optimized_attention(
            AttentionTensorContainer(_flat(q[:, :, TXT_LEN:])),
            AttentionTensorContainer(_flat(k[:, :, :n])),
            AttentionTensorContainer(_flat(v[:, :, :n])),
            dm.heads, transformer_options={}, preferred_attention=None)
        expected = pristine(
            AttentionTensorContainer(_flat(q[:, :, TXT_LEN:])),
            AttentionTensorContainer(_flat(k[:, :, :n])),
            AttentionTensorContainer(_flat(v[:, :, :n])),
            dm.heads, transformer_options={}, preferred_attention=None)
    finally:
        area._QRUNTIME = None
    assert torch.equal(out, expected)
    # the absent keyword became the pristine's own default
    assert calls[0][1]["mask"] is None


def test_the_delegated_call_is_byte_identical(seam):
    """The delegate hands back the ORIGINAL containers, mask and heads."""
    backend, calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm)
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        n = TXT_LEN
        mask = torch.ones((n, n), dtype=torch.bool).tril()
        q_c = AttentionTensorContainer(_flat(q[:, :, :n]))
        k_c = AttentionTensorContainer(_flat(k[:, :, :n]))
        v_c = AttentionTensorContainer(_flat(v[:, :, :n]))
        backend.optimized_attention(q_c, k_c, v_c, dm.heads, mask=mask,
                                    transformer_options={"block_index": 3},
                                    preferred_attention=None)
    finally:
        area._QRUNTIME = None
    args, kwargs = calls[0]
    assert args[0] is q_c and args[1] is k_c and args[2] is v_c
    assert args[3] == dm.heads
    assert kwargs["mask"] is mask
    assert kwargs["transformer_options"] == {"block_index": 3}


def test_a_fine_step_delegates_untouched(seam):
    """Phase II is not installed yet (S8): a fine step runs plain attention."""
    backend, calls, _pristine = seam
    dm = _model()
    rt = _runtime(dm, phase="fine")
    area._QRUNTIME = rt
    try:
        table, q, k, v = _tensors(dm)
        out = _image_call(backend, table, q, k, v)
    finally:
        area._QRUNTIME = None
    assert len(calls) == 1
    assert out.shape == (1, H * W, dm.heads * dm.head_dim)


def test_the_seam_is_live_for_the_backend_module(seam):
    """The install proves the seam is live before it claims the model."""
    backend, _calls, _pristine = seam
    dm = _model()
    wrapper = backend.optimized_attention
    assert area._qwen21_seam_is_live(dm, wrapper)


def test_a_foreign_wrapper_is_refused():
    """SPA's marker on the bound symbol is a refusal, not a silent wrap."""
    def foreign(q, k, v, heads, **kw):  # pragma: no cover - never called
        return None

    foreign._spa_wrapper = True
    with pytest.raises(ValueError, match="SPA"):
        area._qwen21_check_foreign(foreign)


def test_the_sentinel_is_not_a_mask():
    """``None`` is a legal mask, so the absent keyword needs its own value."""
    assert QWEN21_NO_MASK is not None
    assert repr(QWEN21_NO_MASK) == "<QWEN21_NO_MASK>"


# ---------------------------------------------------------------------------
# S11 — the end-to-end mock forward
# ---------------------------------------------------------------------------

#: The end-to-end canvas: 80 x 16 TOKENS (2.1's token grid IS the latent's
#: own — no patchify), a 64-token caption and 32 blocks.  The canvas is ABOVE
#: the native extent (64 tokens per axis) on purpose: at or below it AReA is a
#: provable no-op, and a no-op forward would prove nothing.
E2E_H, E2E_W = 80, 16
E2E_TXT = 64
E2E_LAYERS = 32
E2E_STEPS = 8
E2E_PHASE1 = 2


@pytest.mark.mock_integration
def test_a_full_mock_forward_runs_and_keeps_the_text_segment_pristine():
    """A WHOLE Qwen-Image-2.1 forward through the patched seam.

    32 blocks x 2 segments = 64 seam calls: the 32 IMAGE-segment calls went
    through AReA (the pooled, coarse attention) and the 32 caption-segment
    calls are the model's own, bit for bit.
    """
    with qwen21_backend_module() as backend:
        pristine = backend.optimized_attention
        dm = QwenImage21Transformer2DModel(layers=E2E_LAYERS, seed=5)
        source = ModelPatcher()
        source.model.diffusion_model = dm
        source.model.model_sampling = types.SimpleNamespace()
        patched = area.apply_area_to_model(
            source, "auto", E2E_W * 16, E2E_H * 16,
            phase1_steps=E2E_PHASE1, seed=7)
        # The spy goes IN FRONT of AReA's wrapper, so it records the calls the
        # MODEL made — the image segment's AReA'd answer included.
        recorder, restore = install_recorder()
        outputs = []

        def spy(*args, **kwargs):
            out = recorder(*args, **kwargs)
            outputs.append(out)
            return out

        backend.optimized_attention = spy
        try:
            sigmas = torch.linspace(1.0, 0.0, E2E_STEPS + 1)
            gen = torch.Generator().manual_seed(7)
            latent = torch.randn(1, dm.channels, E2E_H, E2E_W, generator=gen)
            context = torch.randn(1, E2E_TXT, dm.txt_in.in_features, generator=gen)

            def model_function(x, timestep, **c):
                return dm(x, timestep, context=c["context"],
                          transformer_options=c.get("transformer_options") or {})

            wrapper = _unet_wrapper(patched)
            assert wrapper is not None
            params = {
                "input": latent,
                "timestep": torch.tensor([float(sigmas[0])]),
                "c": {"context": context, "c_crossattn": context,
                      "transformer_options": {"sample_sigmas": sigmas,
                                              "sigmas": sigmas[0]}},
            }
            out = wrapper(model_function, params)
        finally:
            restore()
            area.restore_area_attention_hook(patched)

    # the forward produced the model's own answer
    assert out.shape == (1, dm.channels, E2E_H, E2E_W)
    # 32 blocks x 2 segments
    assert len(recorder.calls) == 2 * E2E_LAYERS
    assert len(outputs) == len(recorder.calls)
    image_calls = recorder.image_calls()
    caption_calls = recorder.caption_calls()
    assert len(image_calls) == E2E_LAYERS
    assert len(caption_calls) == E2E_LAYERS
    # the caption segment is the model's own, bit for bit; the image segment
    # is AReA's, NOT the model's dense attention
    for index, (call, answer) in enumerate(zip(recorder.calls, outputs)):
        dense = pristine(call.q, call.k, call.v, call.heads, mask=call.mask)
        if call.mask is None:                       # the image segment
            # AReA's pooled answer is NOT the model's dense attention.  The
            # mock's weights are scaled by 0.05, so the answers collapse with
            # depth (1e-7 in the first block, 1e-32 in the last): the check
            # is EXACT for every block (a delegate would hand back the
            # pristine's bit-identical answer) and RELATIVE for the first
            # one, where the numbers are still readable.
            assert not torch.equal(answer, dense)
            if index < 2:
                assert not torch.allclose(answer, dense, rtol=1e-2, atol=0.0)
            assert answer.shape == (1, E2E_H * E2E_W, dm.heads * dm.head_dim)
        else:                                      # the caption segment
            assert torch.equal(answer, dense)


def test_the_phase1_output_shape_matches_the_pristine(seam):
    """The answer keeps comfy's own layout: ``[b, n, heads*dim]``."""
    backend, calls, pristine = seam
    dm = _model()
    rt = _runtime(dm)
    table, q, k, v = _tensors(dm)
    area._QRUNTIME = rt
    try:
        out = _image_call(backend, table, q, k, v)
    finally:
        area._QRUNTIME = None
    assert out.shape == (1, H * W, dm.heads * dm.head_dim)


def test_the_phase1_output_differs_from_the_pristine(seam):
    """Phase I is not a no-op: the coarse answer differs from the model's."""
    backend, calls, pristine = seam
    dm = _model()
    rt = _runtime(dm)
    table, q, k, v = _tensors(dm)
    area._QRUNTIME = rt
    try:
        out = _image_call(backend, table, q, k, v)
    finally:
        area._QRUNTIME = None
    plain = pristine(
        AttentionTensorContainer(_flat(q[:, :, TXT_LEN:])),
        AttentionTensorContainer(_flat(k[:, :, :TXT_LEN + H * W])),
        AttentionTensorContainer(_flat(v[:, :, :TXT_LEN + H * W])),
        dm.heads, mask=None, transformer_options={}, preferred_attention=None)
    assert out.shape == plain.shape
    assert not torch.equal(out, plain)


def test_the_pooling_is_mean_not_subsample():
    """A ramp's pooled VALUE is the block MEAN, not its first member."""
    from src.area_attn import pool_keys_values
    h = w = 4
    f = 2
    ramp = torch.arange(h * w, dtype=torch.float32).view(1, 1, h * w, 1)
    _k, v_pooled = pool_keys_values(ramp.clone(), ramp, 0, h, w, f, "mean")
    assert v_pooled.shape[2] == (h // f) * (w // f)
    assert torch.allclose(v_pooled.flatten(),
                          torch.tensor([2.5, 4.5, 10.5, 12.5]))


def test_the_coarse_positions_are_continuous():
    """The pooled key sits at its block's CONTINUOUS centroid.

    The floor division would put every block on one INTEGER position
    (``-1``, ``0``) and inject a square wave; the continuous centroid is
    ``-0.75`` and ``0.25`` for a 4x4 canvas with ``f = 2``.
    """
    from src.area_qwen21 import qwen21_coarse_key_ids
    ids = qwen21_coarse_key_ids(0, 0, 4, 4, 2)[0]
    assert torch.allclose(ids[:, 1], torch.tensor([-0.75, -0.75, 0.25, 0.25]))
    assert torch.allclose(ids[:, 2], torch.tensor([-0.75, 0.25, -0.75, 0.25]))


def test_the_caption_rows_never_move(seam):
    """The coarse table's caption rows are the model's OWN text ids."""
    from src.area_qwen21 import coarse_ropes_qwen21, qwen21_layout, qwen21_text_ids
    dm = _model()
    rt = _runtime(dm)
    layout = qwen21_layout(TXT_LEN, H, W, H * W, TXT_LEN + H * W)
    _rope_q, rope_kv = coarse_ropes_qwen21(dm.pe_embedder, layout, FACTOR,
                                           torch.device("cpu"), rt.recorder)
    own = dm.pe_embedder(qwen21_text_ids(0, TXT_LEN))
    assert torch.equal(rope_kv[:, :, :TXT_LEN], own[:, :, :TXT_LEN])


def test_the_coarse_table_comes_from_the_models_own_embedder(seam):
    """The coarse tables go through the MODEL'S OWN embedder (no second one)."""
    from src.area_qwen21 import (coarse_ropes_qwen21, qwen21_coarse_key_ids,
                                 qwen21_coarse_query_ids, qwen21_layout)
    dm = _model()
    rt = _runtime(dm)
    layout = qwen21_layout(TXT_LEN, H, W, H * W, TXT_LEN + H * W)
    rope_q, rope_kv = coarse_ropes_qwen21(dm.pe_embedder, layout, FACTOR,
                                          torch.device("cpu"), rt.recorder)
    assert torch.equal(rope_q, dm.pe_embedder(
        qwen21_coarse_query_ids(TXT_LEN, H, W, FACTOR)))
    assert torch.equal(rope_kv, dm.pe_embedder(
        qwen21_coarse_key_ids(TXT_LEN, TXT_LEN, H, W, FACTOR)))
