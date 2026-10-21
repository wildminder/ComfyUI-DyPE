"""S3 — the Qwen-Image-2.1 ids, the layout, the 3-D conversion and the guard.

**The ids.**  2.1's caption rides a DIAGONAL (``(pos+j, pos+j, pos+j)``,
``comfy/ldm/qwen_image21/model.py:323``) and its image tokens sit at
``(pos, hh(r), ww(c))`` with the spatial axes CENTRED on the canvas and a
half-token parity term (``model.py:330-332``).  The builder mirrors the model
exactly — the reference-free case (``(j,j,j)`` and ``row - h//2``) is only its
special case.

**The guard.**  Only the IMAGE segment of the PLAIN path is eligible: the
caption segment carries a causal mask, the prefix-cache path carries no
``mask`` keyword at all, and the two are length-indistinguishable.
"""
import pytest
import torch

from _qwen21_seam_fixtures import QwenImage21Transformer2DModel
from src.area_qwen21 import (
    QWEN21_NO_MASK,
    qwen21_from_heads,
    qwen21_image_ids,
    qwen21_layout,
    qwen21_segment_eligible,
    qwen21_sequence_ids,
    qwen21_text_ids,
    qwen21_to_heads,
)

pytestmark = pytest.mark.unit


class TestTheIds:
    def test_the_text_ids_are_the_diagonal(self):
        """``model.py:323`` — ``arange(pos, pos+n).expand(n, 3)``."""
        ids = qwen21_text_ids(0, 3)
        assert ids.shape == (1, 3, 3)
        assert torch.equal(ids[0], torch.tensor([[0.0, 0.0, 0.0],
                                                 [1.0, 1.0, 1.0],
                                                 [2.0, 2.0, 2.0]]))

    def test_the_caption_ids_follow_the_running_counter(self):
        """``pos`` is the model's running counter — ``txt_len`` after the caption."""
        ids = qwen21_text_ids(5, 2)
        assert torch.equal(ids[0], torch.tensor([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]))

    def test_the_image_ids_are_centred_and_carry_the_counter(self):
        """``model.py:330`` — ``r - (h - h//2)``, the canvas' own centre."""
        ids = qwen21_image_ids(7, 4, 3)
        assert ids.shape == (1, 12, 3)
        # the axis-0 value is the counter, for every row
        assert torch.equal(ids[0, :, 0], torch.full((12,), 7.0))
        # the rows: -2, -1, 0, 1 (h=4 -> h - h//2 = 2), one row per w tokens
        assert torch.equal(ids[0, 0, 1], torch.tensor(-2.0))
        assert torch.equal(ids[0, 3, 1], torch.tensor(-1.0))
        assert torch.equal(ids[0, 6, 1], torch.tensor(0.0))
        assert torch.equal(ids[0, 9, 1], torch.tensor(1.0))
        # the columns: -2, -1, 0 (w=3 -> w - w//2 = 2)
        assert torch.equal(ids[0, :3, 2], torch.tensor([-2.0, -1.0, 0.0]))

    def test_the_parity_term_is_reproduced(self):
        """``model.py:330`` — ``0.5 * (h % 2 - target_h % 2)``.

        A grid whose parity differs from the target's is centred on it by HALF
        a token; the builder keeps the term instead of only the special case.
        """
        ids = qwen21_image_ids(0, 3, 3, target_h=4, target_w=4)
        # hh = arange(3) - (3 - 1) + 0.5 * (1 - 0) = -1.5, -0.5, 0.5
        assert torch.equal(ids[0, 0, 1], torch.tensor(-1.5))
        assert torch.equal(ids[0, 3, 1], torch.tensor(-0.5))
        assert torch.equal(ids[0, 6, 1], torch.tensor(0.5))
        # the target's own grid: the term vanishes
        own = qwen21_image_ids(0, 3, 3, target_h=3, target_w=3)
        assert torch.equal(own[0, 0, 1], torch.tensor(-2.0))
        assert torch.equal(own[0, 3, 1], torch.tensor(-1.0))
        assert torch.equal(own[0, 6, 1], torch.tensor(0.0))

    def test_the_builder_mirrors_the_model_exactly(self):
        """The model's OWN ids, read out of ``build_sequence``."""
        model = QwenImage21Transformer2DModel()
        x = torch.zeros(1, model.channels, 4, 4)
        context = torch.zeros(1, 3, model.txt_in.in_features)
        model.build_sequence(x, context)
        assert torch.equal(model.last_ids, qwen21_sequence_ids(3, 4, 4)[0])


class TestTheLayout:
    def test_a_fitting_segment_is_accepted(self):
        layout = qwen21_layout(4, 3, 3, 9, 13)
        assert layout.img_start == 4
        assert layout.img_end == 13
        assert layout.img_tokens == 9

    def test_a_wrong_query_length_is_refused(self):
        with pytest.raises(ValueError, match="queries"):
            qwen21_layout(4, 3, 3, 8, 13)

    def test_a_wrong_key_length_is_refused(self):
        with pytest.raises(ValueError, match="key"):
            qwen21_layout(4, 3, 3, 9, 12)

    def test_an_empty_grid_is_refused(self):
        with pytest.raises(ValueError, match="grid"):
            qwen21_layout(4, 0, 3, 0, 4)


class TestTheGuard:
    def test_the_eligibility_guard_accepts_the_image_segment(self):
        """No mask value, the plain path's lengths."""
        assert qwen21_segment_eligible(None, 9, 13, 13, 4, 3, 3) is True

    def test_the_eligibility_guard_rejects_the_text_segment(self):
        """``model.py:324`` — the caption carries a causal bool mask."""
        mask = torch.ones((4, 4), dtype=torch.bool).tril()
        assert qwen21_segment_eligible(mask, 4, 4, 4, 3, 3, 3) is False

    def test_the_cached_path_is_refused(self):
        """``model.py:222`` — no ``mask`` keyword at all.

        The cached path's lengths are the plain image segment's lengths, so
        ONLY the keyword's absence separates the two.
        """
        assert qwen21_segment_eligible(QWEN21_NO_MASK, 9, 13, 13, 4, 3, 3) is False

    def test_a_foreign_length_is_refused(self):
        assert qwen21_segment_eligible(None, 9, 14, 14, 4, 3, 3) is False
        assert qwen21_segment_eligible(None, 8, 13, 13, 4, 3, 3) is False


class TestTheHeadConversion:
    def test_the_conversion_is_an_exact_inverse(self):
        x = torch.randn(2, 5, 4 * 3)
        heads = qwen21_to_heads(x, 4)
        assert heads.shape == (2, 4, 5, 3)
        assert torch.equal(qwen21_from_heads(heads), x)

    def test_the_conversion_is_head_major(self):
        """The flat index of a row is ``head * dim_head + d``."""
        x = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        heads = qwen21_to_heads(x, 2)
        assert torch.equal(heads[0, 0], torch.tensor([[0.0, 1.0], [4.0, 5.0], [8.0, 9.0]]))
        assert torch.equal(heads[0, 1], torch.tensor([[2.0, 3.0], [6.0, 7.0], [10.0, 11.0]]))

    def test_a_bad_head_count_is_refused(self):
        with pytest.raises(ValueError, match="heads"):
            qwen21_to_heads(torch.zeros(1, 2, 5), 2)


def test_the_layout_rejects_a_reference_stream():
    """A reference-image forward is not ``txt_len + h*w`` long (N2/N3)."""
    with pytest.raises(ValueError, match="key"):
        qwen21_layout(4, 3, 3, 9, 9 + 1024)


def test_the_eligibility_guard_never_raises():
    """Garbage in, ``False`` out — a guard must never abort a forward."""
    assert qwen21_segment_eligible(None, "x", None, 1, "y", 0, 0) is False
    assert qwen21_segment_eligible(object(), 9, 13, 13, 4, 3, 3) is False
    # the healthy case is still True
    assert qwen21_segment_eligible(None, 9, 13, 13, 4, 3, 3) is True


def test_the_guard_rejects_a_foreign_head_count():
    """N13 — an unknown variant of the backbone is DELEGATED, not engaged."""
    assert qwen21_segment_eligible(None, 9, 13, 13, 4, 3, 3,
                                   heads=16, dm_heads=32) is False
    assert qwen21_segment_eligible(None, 9, 13, 13, 4, 3, 3,
                                   heads=32, dm_heads=32) is True
