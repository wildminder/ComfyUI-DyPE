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

from _qwen21_seam_fixtures import QwenImage21Transformer2DModel
from src.area_qwen21 import (
    Qwen21RopeRecorder,
    capture_qwen21_table,
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
