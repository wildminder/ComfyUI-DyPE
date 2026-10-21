"""S4 — the Qwen-Image-2.1 rope recorder and the coarse tables.

2.1 builds its rope table ONCE per forward (``comfy/ldm/qwen_image21/model.py:
338``) and applies it itself BEFORE the seam (``model.py:99-100``), so the seam
receives rotated q/k and NO table — Phase I has to pool *unrotated* keys, which
means the model's own table has to be captured and inverted.

**The two Phase-I tables have DIFFERENT lengths on 2.1** (the asymmetry the
plan leaves under-specified): the eligible call has no text queries, so
``rope_q`` covers the ``h*w`` image queries and ``rope_kv`` covers
``txt_len + nh*nw`` rows — the caption at the model's own diagonal ids and the
pooled keys at their block's CONTINUOUS centroid in the centred frame.
"""
import pytest
import torch

from _qwen21_seam_fixtures import MockEmbedND, QwenImage21Transformer2DModel
from src.area_qwen21 import (
    Qwen21RopeRecorder,
    capture_qwen21_table,
    coarse_ropes_qwen21,
    qwen21_coarse_key_ids,
    qwen21_coarse_query_ids,
    qwen21_layout,
    qwen21_text_ids,
)

pytestmark = pytest.mark.unit


def _table(n: int = 5) -> torch.Tensor:
    """A stand-in for an ``EmbedND`` table — ``[b, 1, n, axes, 2, 2]``."""
    return torch.zeros(1, 1, n, 4, 2, 2)


class TestTheRecorder:
    def test_the_recorder_counts_repeated_calls(self):
        recorder = Qwen21RopeRecorder()
        recorder.armed = True
        first = torch.ones(1, 1, 5, 4, 2, 2)
        recorder.record(first)
        recorder.record(torch.zeros(1, 1, 5, 4, 2, 2))
        assert torch.equal(recorder.table, first)
        assert recorder.extra_tables == 1
        assert recorder.ambiguous is True

    def test_a_disarmed_recorder_records_nothing(self):
        recorder = Qwen21RopeRecorder()
        recorder.record(_table())
        assert recorder.table is None

    def test_the_recorder_can_be_paused(self):
        """The pause guard — the coarse tables go through the SAME embedder."""
        recorder = Qwen21RopeRecorder()
        recorder.armed = True
        with recorder.paused():
            recorder.record(_table())
        assert recorder.table is None
        recorder.record(_table())
        assert recorder.table is not None

    def test_a_foreign_output_is_refused(self):
        recorder = Qwen21RopeRecorder()
        recorder.armed = True
        with pytest.raises(ValueError, match="EmbedND"):
            recorder.record("not a table")

    def test_the_hook_drops_a_foreign_output_without_raising(self):
        """A forward must keep running plain attention, not crash the sampler."""
        class Foreign(torch.nn.Module):
            def forward(self, x):
                return "not a table"

        embedder = Foreign()
        hook, recorder = capture_qwen21_table(embedder)
        try:
            recorder.armed = True
            embedder(torch.zeros(1))
        finally:
            if hook.handle is not None:
                hook.handle.remove()
        assert recorder.table is None


class TestTheCapture:
    def test_the_recorder_captures_the_models_own_table(self):
        """The recorder takes the model's OWN table, head-shared."""
        model = QwenImage21Transformer2DModel()
        hook, recorder = capture_qwen21_table(model.pe_embedder)
        try:
            recorder.armed = True
            model.build_sequence(torch.zeros(1, model.channels, 4, 4),
                                 torch.zeros(1, 3, model.txt_in.in_features))
            table = recorder.table
            assert table is not None
            # the head-shared layout the engine consumes: [b, 1, n, axes, 2, 2]
            assert table.dim() == 6
            assert table.shape[1] == 1
            assert table.shape[2] == 3 + 16
        finally:
            if hook.handle is not None:
                hook.handle.remove()


class TestTheCoarseTables:
    def test_the_two_tables_have_different_lengths(self):
        """The asymmetry: no text queries, so the q side is the image only."""
        embedder = MockEmbedND(axes_dim=(4, 6, 6))
        layout = qwen21_layout(4, 4, 4, 16, 20)
        rope_q, rope_kv = coarse_ropes_qwen21(embedder, layout, 2, torch.device("cpu"))
        assert rope_q.shape[2] == 16          # h*w image queries
        assert rope_kv.shape[2] == 4 + 4      # the caption + 2x2 pooled blocks

    def test_the_caption_keeps_the_models_own_diagonal_ids(self):
        """``qwen21_coarse_key_ids`` — the caption never moves."""
        ids = qwen21_coarse_key_ids(4, 4, 4, 4, 2)
        assert torch.equal(ids[0, :4], qwen21_text_ids(0, 4)[0])

    def test_the_pooled_keys_sit_at_their_blocks_continuous_centroid(self):
        """The ripple fix — the key is the MEAN of its block's query positions.

        The reference floor-divides, so every token of a block shares ONE rope
        position and the phase jumps a whole unit at every block boundary — the
        pack's measured Z-Image square wave.
        """
        h = w = 4
        f = 2
        q_ids = qwen21_coarse_query_ids(4, h, w, f)[0]
        kv_ids = qwen21_coarse_key_ids(4, 4, h, w, f)[0]
        nw = 2
        for block in range(4):
            br, bc = divmod(block, nw)
            members = [r * w + c
                       for r in range(br * f, (br + 1) * f)
                       for c in range(bc * f, (bc + 1) * f)]
            assert kv_ids[4 + block, 1] == pytest.approx(
                float(sum(q_ids[m, 1] for m in members)) / len(members))
            assert kv_ids[4 + block, 2] == pytest.approx(
                float(sum(q_ids[m, 2] for m in members)) / len(members))

    def test_the_pooled_key_is_not_the_floor_divided_position(self):
        """The floor division would put the block's key at its FIRST token."""
        q_ids = qwen21_coarse_query_ids(4, 4, 4, 2)[0]
        kv_ids = qwen21_coarse_key_ids(4, 4, 4, 4, 2)[0]
        # the first block's key row is the MEAN of its two query rows, not the
        # first one
        assert kv_ids[4, 1] == pytest.approx(float((q_ids[0, 1] + q_ids[4, 1]) / 2))
        assert kv_ids[4, 1] != pytest.approx(float(q_ids[0, 1]))

    def test_the_axis_zero_value_is_the_models_own_counter(self):
        """The image tokens do not move along axis 0, so no ``img_axis0``."""
        kv_ids = qwen21_coarse_key_ids(6, 4, 4, 4, 2)[0]
        assert torch.equal(kv_ids[4:, 0], torch.full((4,), 6.0))


def test_the_captured_table_has_the_models_layout():
    """``[b, 1, n, sum(axes)//2, 2, 2]`` (comfy/ldm/flux/layers.py:14-29)."""
    from _qwen21_seam_fixtures import MockEmbedND
    from src.area_qwen21 import capture_qwen21_table
    embedder = MockEmbedND()
    hook, recorder = capture_qwen21_table(embedder)
    try:
        recorder.armed = True
        ids = torch.zeros(1, 5, 3)
        recorder.record(embedder(ids))
    finally:
        handle = getattr(hook, "handle", None)
        if handle is not None:
            handle.remove()
    captured = recorder.table
    assert captured is not None
    assert captured.dim() == 6
    assert captured.shape[1] == 1                 # the head-shared layout
    assert captured.shape[2] == 5                 # the sequence
    assert captured.shape[3] == sum(embedder.axes_dim) // 2
    assert captured.shape[4:] == (2, 2)


def test_the_unrotate_round_trip_is_exact():
    """``unrotate1`` undoes the engine's own rotation exactly."""
    from _qwen21_seam_fixtures import MockEmbedND
    from src.area_attn import apply_rope1
    from src.area_rope import unrotate1, unrotate_table
    embedder = MockEmbedND()
    table = embedder(torch.zeros(1, 6, 3))
    x = torch.randn(1, 2, 6, 128)
    rotated = apply_rope1(x, table)
    assert torch.allclose(unrotate1(rotated, table), x, atol=1e-5)
    # the inverse table negates the off-diagonals of the 2x2 blocks
    inverse = unrotate_table(table)
    assert torch.equal(inverse[..., 0, 1], -table[..., 0, 1])
    assert torch.equal(inverse[..., 1, 0], -table[..., 1, 0])
