"""AReA × Krea-2 — the SEAM PROBE (plan §7/S1).

This step writes NO AReA code: it produces the harness and the proof every
later step depends on — that the seam is REACHED, WITH WHICH CONVENTION, and
that the four auxiliary calls are distinguishable from the 28 main ones.

What the probe pins (each test names the mutation it kills):

* the seam is the symbol the BACKEND module binds
  (``comfy.ldm.krea2.model.optimized_attention_masked``, ``model.py:21``) and
  the model's own attention class resolves it through that module's globals —
  rebinding the DEFINING module (``comfy.ldm.modules.attention``) instead
  would be a silent no-op for Krea-2 (the trap SPA documents,
  ``src/spa.py:558-567``);
* the call convention: FOUR POSITIONAL arguments ``(q, k, v, heads)`` plus the
  kwargs ``mask``, ``skip_reshape=True``, ``preferred_attention`` and
  ``transformer_options`` — ``attn_precision`` and ``skip_output_reshape`` are
  absent (a wrapper that mis-slots ``heads`` into ``mask`` would fail here);
* the q/k that reach the seam are ROTATED by the model's own table
  (``model.py:93``) — a wrapper that assumes unrotated q/k would fail here;
* the four auxiliary text-fusion calls carry 20 heads and NO ``img_slice``
  (they run before ``model.py:369``), and the two refiner calls' stream length
  is the PROMPT length — a PREFIX of the joint stream's length, so a
  length-only eligibility guard would let them through;
* the 28 main calls carry 48 heads, ``img_slice`` and ``block_index``, and
  their stream length is ``txt_len + h*w``;
* the pristine seam answers in the FLATTENED ``[b, n, heads*dim]`` layout —
  the one Krea-2's ``out * F.sigmoid(gate)`` multiplies (``model.py:106``).

Everything is mock-based and GPU-free (the house convention): the harness is
``tests/_krea_fixtures.py``.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_krea_seam_probe.py
"""
from __future__ import annotations

import sys

import pytest
import torch

from _krea_fixtures import (
    CHANNELS,
    HEAD_DIM,
    HEADS,
    KREA_SEAM_MODULE,
    KREA_SEAM_SYMBOL,
    KV_HEADS,
    LAYERS,
    PATCH,
    SEQ,
    TXT_HEADS,
    TXT_LAYERS,
    SingleStreamDiT,
    install_recorder,
    krea_backend_module,
    run_krea_forward,
)
from src.area_attn import apply_rope1

#: The probe's canvas: an 8x8 LATENT (8 latent cells) -> a 4x4 token grid, so
#: the joint stream is ``8 prompt tokens + 16 image tokens``.
LATENT = 8
GRID = LATENT // PATCH
IMG_TOKENS = GRID * GRID
JOINT = SEQ + IMG_TOKENS


@pytest.fixture
def krea_backend():
    """The mock backend, registered as ``comfy.ldm.krea2.model`` for the test."""
    with krea_backend_module() as backend:
        yield backend


@pytest.fixture
def probe(krea_backend):
    """One recorded Krea-2 forward: the seam's call log, the model and its output."""
    recorder, restore = install_recorder()
    try:
        dm = SingleStreamDiT()
        out = run_krea_forward(dm, batch=1, seq=SEQ, height=LATENT, width=LATENT)
        yield recorder, dm, out
    finally:
        restore()


@pytest.mark.unit
class TestKreaSeamProbe:
    def test_seam_symbol_is_the_backend_bound_name(self, probe, krea_backend):
        """The seam is the BOUND name, and the model resolves it through its own globals."""
        recorder, dm, _out = probe
        assert krea_backend is sys.modules[KREA_SEAM_MODULE]
        assert getattr(krea_backend, KREA_SEAM_SYMBOL) is recorder
        # the model's attention class looks the symbol up in ITS module's globals
        forward_globals = type(dm.blocks[0].attn).forward.__globals__
        assert forward_globals[KREA_SEAM_SYMBOL] is getattr(krea_backend, KREA_SEAM_SYMBOL)
        # ... and the DEFINING module is a DIFFERENT binding: patching it would
        # never reach Krea-2 (the silent no-op of src/spa.py:558-567)
        defining = sys.modules.get("comfy.ldm.modules.attention")
        assert defining is not None
        assert getattr(defining, KREA_SEAM_SYMBOL) is not getattr(krea_backend, KREA_SEAM_SYMBOL)

    def test_seam_call_convention_is_four_positional_args(self, probe):
        """Four positional slots (q, k, v, heads) and exactly four kwargs."""
        recorder, _dm, _out = probe
        assert recorder.calls
        for call in recorder.calls:
            assert call.n_positional == 4
            assert set(call.kwargs) == {"mask", "skip_reshape", "preferred_attention",
                                        "transformer_options"}
            assert call.kwargs["skip_reshape"] is True
            assert call.kwargs["mask"] is None
            # the two slots Krea-2 does NOT pass (comfy/ldm/krea2/model.py:99-100)
            assert "attn_precision" not in call.kwargs
            assert "skip_output_reshape" not in call.kwargs
            # ``heads`` is the FOURTH positional slot, not a kwarg
            assert call.heads in (HEADS, TXT_HEADS)

    def test_seam_receives_rotated_qk(self, probe):
        """The q/k that reach the seam are the model's OWN rotation of its projections."""
        recorder, dm, _out = probe
        main = recorder.by_heads(HEADS)[-1]
        attn = dm.blocks[-1].attn
        pre_q, pre_k = attn.pre_rope
        assert torch.equal(main.q, apply_rope1(pre_q, attn.freqs))
        # the keys carry the GQA repeat, which the model resolves BEFORE the seam
        # (comfy/ldm/krea2/model.py:94-97) — so the seam's k is the repeated one
        repeat = HEADS // KV_HEADS
        assert torch.equal(main.k, apply_rope1(pre_k, attn.freqs).repeat_interleave(repeat, dim=1))
        # not vacuous: a non-zero position really moves the tensors
        assert not torch.allclose(main.q, pre_q)

    def test_aux_calls_carry_20_heads_and_no_img_slice(self, probe):
        """The four text-fusion calls: 20 heads, no img_slice, no block_index."""
        recorder, _dm, _out = probe
        aux = recorder.by_heads(TXT_HEADS)
        assert len(aux) == 4
        for call in aux:
            assert call.img_slice is None
            assert call.block_index is None
        # the two layerwise calls run over the 12-layer Qwen3-VL stack, the two
        # refiner calls over the PROMPT length (comfy/ldm/krea2/model.py:159-165)
        assert [call.stream_len for call in aux] == [TXT_LAYERS, TXT_LAYERS, SEQ, SEQ]
        # the refiner's length is a PREFIX of the joint stream's — which is why
        # a length-only eligibility guard would let it through
        assert SEQ < JOINT

    def test_main_calls_carry_48_heads_and_img_slice(self, probe):
        """The 28 main calls: 48 heads, the published img_slice, their block index."""
        recorder, _dm, _out = probe
        main = recorder.by_heads(HEADS)
        assert len(main) == LAYERS
        for index, call in enumerate(main):
            assert call.img_slice == [SEQ, JOINT]
            assert call.block_index == index

    def test_main_call_stream_length_is_txt_plus_hw(self, probe):
        """The joint stream is text FIRST: the image tokens start at txt_len."""
        recorder, dm, _out = probe
        for call in recorder.by_heads(HEADS):
            assert call.stream_len == SEQ + GRID * GRID
        # the model's own ids agree: the text block sits at the origin, the
        # image tokens on (0, row, col) (comfy/ldm/krea2/model.py:291-294,346)
        img, imgpos, h, w = dm.process_img(torch.zeros(1, CHANNELS, LATENT, LATENT))
        assert (h, w) == (GRID, GRID)
        assert img.shape[1] == IMG_TOKENS
        assert torch.equal(imgpos[0, :, 0], torch.zeros(IMG_TOKENS))
        assert torch.equal(imgpos[0, :, 1], torch.arange(h, dtype=torch.float32).repeat_interleave(w))
        assert torch.equal(imgpos[0, :, 2], torch.arange(w, dtype=torch.float32).repeat(h))

    def test_pristine_seam_returns_flattened_layout(self, probe):
        """The pristine answer is the FLATTENED [b, n, heads*dim] Krea-2 multiplies."""
        recorder, _dm, out = probe
        call = recorder.by_heads(HEADS)[0]
        answer = recorder.pristine(call.q, call.k, call.v, call.heads, skip_reshape=True)
        assert answer.shape == (1, call.stream_len, call.heads * call.q.shape[-1])
        # the model's gate is [b, n, dim] with dim == heads*dim_head, so the
        # flattened layout is the only one the model can multiply (model.py:106)
        assert call.heads * call.q.shape[-1] == HEADS * HEAD_DIM
        assert out.shape == (1, CHANNELS, LATENT, LATENT)

    def test_aux_call_count_is_four_and_main_is_twenty_eight(self, probe):
        """The call census: 4 auxiliary calls, then 28 main ones — no more, no fewer."""
        recorder, _dm, _out = probe
        assert len(recorder.calls) == 4 + LAYERS
        assert len(recorder.by_heads(TXT_HEADS)) == 4
        assert len(recorder.by_heads(HEADS)) == LAYERS
        # the four aux calls come FIRST (the text fusion precedes the blocks)
        assert [call.heads for call in recorder.calls[:4]] == [TXT_HEADS] * 4
        assert [call.heads for call in recorder.calls[4:]] == [HEADS] * LAYERS
