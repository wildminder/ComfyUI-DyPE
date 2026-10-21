"""S1 — the Qwen-Image-2.1 seam contract, proved against a comfy-faithful mock.

The mock (``tests/_qwen21_seam_fixtures.py``) mirrors
``comfy/ldm/qwen_image21/model.py`` closely enough that these seven facts can
be pinned WITHOUT a checkpoint and WITHOUT a GPU:

1. the seam symbol is bound in the 2.1 module's OWN namespace, and a rebind of
   that name reaches the model (the trap SPA documents at ``src/spa.py:558-567``);
2. the image segment's call carries the ``mask`` KEYWORD, with the value
   ``None`` (``model.py:211``, ``model.py:333``);
3. the caption segment's call carries a CAUSAL BOOL mask (``model.py:324``);
4. the prefix-cache path's call carries NO ``mask`` keyword at all
   (``model.py:222``) — the only thing that separates it from the plain image
   segment, which has the SAME length signature;
5. the seam's input is 3-D ``[b, n, heads*dim]`` and comfy reads it
   HEAD-MAJOR (``comfy/ldm/modules/attention.py:147-158``);
6. the answer is the FLATTENED ``[b, n, heads*dim]``;
7. a container is single-owner: a second ``take()`` raises.
"""
import pytest
import torch

from src.area_qwen21 import QWEN21_NO_MASK
from _qwen21_seam_fixtures import (
    AttentionTensorContainer,
    _apply_rope_bnd,
    QwenImage21Transformer2DModel,
    install_recorder,
    optimized_attention,
    qwen21_backend_module,
    run_qwen21_forward,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def qwen21_backend():
    """This module IS ``comfy.ldm.qwen_image21.model`` for the test's duration."""
    with qwen21_backend_module() as backend:
        yield backend


@pytest.fixture
def model():
    return QwenImage21Transformer2DModel()


class TestSeamBinding:
    def test_the_seam_symbol_is_reached(self, qwen21_backend):
        """The 2.1 module binds the UNMASKED ``optimized_attention`` itself."""
        assert qwen21_backend.optimized_attention is optimized_attention
        assert "optimized_attention" in vars(qwen21_backend)

    def test_rebinding_the_bound_symbol_reaches_the_model(self, qwen21_backend, model):
        """A rebind of the BOUND name is what the model calls.

        This is the whole point of the harness: a class defined in the test
        file would resolve the symbol in the TEST's globals and the seam's
        rebind would never reach the model.
        """
        recorder, restore = install_recorder()
        try:
            run_qwen21_forward(model)
        finally:
            restore()
        assert recorder.calls, "the model never reached the bound seam symbol"
        assert all(call.heads == model.heads for call in recorder.calls)


class TestCallPaths:
    def test_the_image_segment_carries_no_mask(self, qwen21_backend, model):
        """``model.py:211`` always passes ``mask=``; the image segment's is None."""
        recorder, restore = install_recorder()
        try:
            run_qwen21_forward(model, seq=4, height=4, width=4)
        finally:
            restore()
        image = recorder.image_calls()
        assert image, "the image segment never called the seam"
        for call in image:
            assert call.mask_present is True
            assert call.mask is None

    def test_the_text_segment_carries_the_causal_mask(self, qwen21_backend, model):
        """``model.py:324`` — the caption segment carries a causal BOOL mask."""
        recorder, restore = install_recorder()
        try:
            run_qwen21_forward(model, seq=4, height=4, width=4)
        finally:
            restore()
        caption = recorder.caption_calls()
        assert caption, "the caption segment never called the seam"
        for call in caption:
            assert call.mask_present is True
            assert call.mask.dtype == torch.bool
            assert call.mask.shape == (call.q_len, call.k_len)
            # a CAUSAL mask: the lower triangle, nothing above it
            assert torch.equal(call.mask, call.mask.tril())

    def test_the_cached_path_passes_no_mask_keyword(self, qwen21_backend, model):
        """``model.py:222`` — the prefix-cache call has NO ``mask`` at all.

        It is LENGTH-INDISTINGUISHABLE from the plain image segment (the key
        is ``txt_len + h*w`` over ``h*w`` queries), so only the keyword's
        absence separates the two paths.
        """
        recorder, restore = install_recorder()
        try:
            run_qwen21_forward(model, seq=4, height=4, width=4, cached=True)
        finally:
            restore()
        cached = recorder.cached_calls()
        assert cached, "the cached path never called the seam"
        for call in cached:
            assert call.mask_present is False
            assert call.mask is QWEN21_NO_MASK
            assert call.q_len == 16 and call.k_len == 20


class TestTheSeamIO:
    def test_the_seam_input_is_three_dimensional(self, qwen21_backend, model):
        """``model.py:206-208`` — the seam sees ``[b, n, heads*dim]``, not 4-D."""
        recorder, restore = install_recorder()
        try:
            run_qwen21_forward(model, seq=4, height=4, width=4)
        finally:
            restore()
        for call in recorder.calls:
            assert call.q.dim() == 3
            assert call.q.shape == (1, call.q_len, model.heads * model.head_dim)
            assert call.k.dim() == 3
            assert call.v.dim() == 3

    def test_comfy_reads_the_flat_index_head_major(self, qwen21_backend):
        """``comfy/ldm/modules/attention.py:147-158`` — ``j = head*dim + d``.

        The head-major reading is the one that reproduces comfy's answer; the
        head-minor one does not.
        """
        heads, dim, n = 3, 2, 4
        gen = torch.Generator().manual_seed(3)
        q = torch.randn(1, n, heads * dim, generator=gen)
        k = torch.randn(1, n, heads * dim, generator=gen)
        v = torch.randn(1, n, heads * dim, generator=gen)
        out = optimized_attention(q, k, v, heads)
        # the answer is the FLATTENED [b, n, heads*dim]
        assert out.shape == (1, n, heads * dim)
        head_major = torch.nn.functional.scaled_dot_product_attention(
            q.view(1, n, heads, dim).transpose(1, 2),
            k.view(1, n, heads, dim).transpose(1, 2),
            v.view(1, n, heads, dim).transpose(1, 2),
        ).transpose(1, 2).reshape(1, n, heads * dim)
        assert torch.allclose(out, head_major, atol=1e-6)
        head_minor = torch.nn.functional.scaled_dot_product_attention(
            q.view(1, n, dim, heads).transpose(1, 2),
            k.view(1, n, dim, heads).transpose(1, 2),
            v.view(1, n, dim, heads).transpose(1, 2),
        ).transpose(1, 2).reshape(1, n, heads * dim)
        assert not torch.allclose(out, head_minor, atol=1e-6)

    def test_a_container_can_be_taken_only_once(self):
        """``comfy/ldm/modules/attention.py:196-209`` — single owner.

        The wrapper INSPECTS with ``peek()`` and takes exactly once, so this is
        the crash class a second take would produce.
        """
        container = AttentionTensorContainer(torch.zeros(1, 2, 3))
        assert container.peek().shape == (1, 2, 3)
        assert container.take().shape == (1, 2, 3)
        with pytest.raises(RuntimeError):
            container.take()


def test_a_segment_call_is_distinguishable_from_a_whole_stream_call(qwen21_backend, model):
    """A SEGMENT call is not a whole-stream call: the key is LONGER.

    The image segment's key is ``[caption | image]`` over the image's queries
    alone (``model.py:206-208``), so ``k_len > q_len`` — the discriminator the
    guard uses.  A whole-stream call would have ``k_len == q_len`` and would
    be a DIFFERENT (and wrong) thing to coarsen.
    """
    recorder, restore = install_recorder()
    try:
        run_qwen21_forward(model, seq=4, height=4, width=4)
    finally:
        restore()
    image = recorder.image_calls()
    assert image, "the image segment never called the seam"
    for call in image:
        assert call.k_len > call.q_len
        assert call.k_len == call.q_len + 4      # the caption's rows
        assert call.q_len == 16                  # the canvas' own rows
    # the caption segment is a whole-stream call (its key is its query set)
    for call in recorder.caption_calls():
        assert call.k_len == call.q_len


def test_the_model_rotates_before_the_seam(qwen21_backend, model):
    """The seam's q/k are the model's OWN ROTATED tensors (model.py:99-100).

    The mock keeps the pre-rotation q/k (``MockAttention.pre_rope``) and the
    model rotates them BEFORE the seam — so the seam never sees the
    un-rotated tensors, and an adapter that re-rotated them would be wrong.
    """
    gen = torch.Generator().manual_seed(5)
    x = torch.randn(1, model.channels, 4, 4, generator=gen)
    context = torch.randn(1, 4, model.txt_in.in_features, generator=gen)
    hidden, pe, segments = model.build_sequence(x, context)
    block = model.transformer_blocks[0]
    pre_q = block.to_q(hidden).view(1, hidden.shape[1], model.heads, -1)
    rotated = _apply_rope_bnd(pre_q, pe)
    recorder, restore = install_recorder()
    try:
        model(x, torch.zeros(1), context)
    finally:
        restore()
    image = recorder.image_calls()[0]
    expected = rotated[:, -image.q_len:].flatten(2)
    assert torch.allclose(image.q, expected, atol=1e-6)
    # the model's own probe agrees: the seam's tensors are NOT the
    # pre-rotation ones
    probe_q, _probe_k = block.pre_rope
    assert not torch.allclose(image.q, probe_q[:, -image.q_len:].flatten(2), atol=1e-6)


def test_the_recorder_snapshots_at_call_time(qwen21_backend, model):
    """The recorded block index belongs to THAT call.

    ``transformer_options`` is MUTATED per block (``model.py:388``), so a
    recorder that read it after the forward would see the LAST block's index
    on every call.
    """
    recorder, restore = install_recorder()
    try:
        run_qwen21_forward(model, seq=4, height=4, width=4)
    finally:
        restore()
    assert recorder.calls
    for index, call in enumerate(recorder.calls):
        # two segments per block: the caption and then the image
        assert call.block_index == index // 2


def test_the_image_segment_shapes(qwen21_backend, model):
    """The image segment's call: ``[b, h*w, heads*dim]`` over ``[b, txt+h*w, ...]``.

    A whole-stream call would have the queries and the key the same length
    (``q[:, :end]``) — the segment call slices the queries to the canvas'
    own rows (``model.py:206-208``).
    """
    recorder, restore = install_recorder()
    try:
        run_qwen21_forward(model, seq=4, height=4, width=4)
    finally:
        restore()
    image = recorder.image_calls()[0]
    width = model.heads * model.head_dim
    assert image.q.shape == (1, 16, width)
    assert image.k.shape == (1, 20, width)
    assert image.v.shape == (1, 20, width)
