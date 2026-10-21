"""The additive ``text_keys_only`` keyword of the AReA engine (plan §7/S7).

The engine's :func:`~src.area_attn.windowed_attention` used to couple the text
queries and the text keys through ONE length (``t = q_txt.shape[2]``) — the
query cat, the key cat, the value expand and the output split all read it.
Qwen-Image-2.1's eligible call breaks that coupling: its image segment has NO
text queries (the queries are the ``h*w`` image rows) while the caption's KEYS
must stay visible to them (``comfy/ldm/qwen_image21/model.py:206-211``).  The
keyword splits the length into ``t_q`` (text queries) and ``t_k`` (text keys).

**The flag is OFF by default and every default line is the line that is there
today** — the FLUX / Z-Image / Krea-2 goldens are the proof.  What is pinned
here:

* with equal lengths the flag ON and OFF give BIT-IDENTICAL answers (the only
  difference between the two branches is the split, and it vanishes);
* the default path REFUSES a length mismatch (the explicit assertion);
* with the flag ON the caption's keys stay VISIBLE to the image queries, the
  text answer has ZERO rows, and the image answer starts at ``t_q`` — not at
  ``t_k``.

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.area_attn import FineLayout, WindowGroup, windowed_attention
from src.area_schedule import Window

pytestmark = pytest.mark.unit

B, HEADS, DIM = 1, 2, 4


def _layout(h: int, w: int, windows: list) -> FineLayout:
    """A hand-built layout — one group per window, no Halo, no bridge.

    The merge weights are ONE, so the merged image answer is the plain
    attention of the group: the reference below can then be written by hand.
    """
    groups = []
    coverage = torch.zeros(h * w, dtype=torch.float32)
    for index, win in enumerate(windows):
        rows = torch.arange(win.r0, win.r1 + 1)
        cols = torch.arange(win.c0, win.c1 + 1)
        idx = (rows[:, None] * w + cols[None, :]).reshape(-1)
        coverage[idx] += 1.0
        groups.append(WindowGroup(
            tokens=idx[None, :],
            halo=None,
            weights=torch.ones(1, idx.numel()),
            window_ids=torch.tensor([index]),
        ))
    return FineLayout(h, w, windows, groups, coverage, 0)


def _tensors(n_txt: int, n_img: int, seed: int = 5):
    """``(q_txt, k_txt, v_txt, q_img, k_img, v_img)`` in the engine's layout."""
    gen = torch.Generator().manual_seed(seed)
    shape = (B, HEADS, n_img, DIM)
    q_img = torch.randn(shape, generator=gen)
    k_img = torch.randn(shape, generator=gen)
    v_img = torch.randn(shape, generator=gen)
    tshape = (B, HEADS, n_txt, DIM)
    return (torch.randn(tshape, generator=gen), torch.randn(tshape, generator=gen),
            torch.randn(tshape, generator=gen), q_img, k_img, v_img)


def _reference(q_txt, k_txt, v_txt, q_img, k_img, v_img):
    """The plain attention of one window over ``[text | image]`` keys."""
    q = torch.cat([q_txt, q_img], dim=2) if q_txt.shape[2] else q_img
    k = torch.cat([k_txt, k_img], dim=2)
    v = torch.cat([v_txt, v_img], dim=2)
    out = F.scaled_dot_product_attention(q, k, v)
    return out[:, :, :q_txt.shape[2]], out[:, :, q_txt.shape[2]:]


def test_the_default_path_is_unchanged():
    """The flag only splits a length that is one length: nothing else moves."""
    h, w, t = 4, 4, 3
    layout = _layout(h, w, [Window(0, 0, h - 1, w - 1)])
    tensors = _tensors(t, h * w)
    off = windowed_attention(*tensors, tensors[3], layout)
    on = windowed_attention(*tensors, tensors[3], layout, text_keys_only=True)
    assert torch.equal(off[0], on[0])
    assert torch.equal(off[1], on[1])


def test_a_mismatched_text_length_raises_with_the_default():
    """The default path ASSERTS the two lengths are equal (the design's S7)."""
    h, w = 4, 4
    layout = _layout(h, w, [Window(0, 0, h - 1, w - 1)])
    q_txt, k_txt, v_txt, q_img, k_img, v_img = _tensors(2, h * w)
    # a caption with more KEYS than queries — the 2.1 shape, refused by default
    k_txt = torch.cat([k_txt, k_txt[:, :, :1]], dim=2)
    v_txt = torch.cat([v_txt, v_txt[:, :, :1]], dim=2)
    with pytest.raises(ValueError, match="text keys"):
        windowed_attention(q_txt, k_txt, v_txt, q_img, k_img, v_img, q_img, layout)


def test_the_text_keys_are_still_visible_to_the_queries():
    """With the flag on, the caption's KEYS are in the image queries' key set."""
    h, w, t = 4, 4, 3
    layout = _layout(h, w, [Window(0, 0, h - 1, w - 1)])
    tensors = _tensors(t, h * w)
    q_txt, k_txt, v_txt, q_img, k_img, v_img = tensors
    out_txt, out_img = windowed_attention(
        q_txt[:, :, :0], k_txt, v_txt, q_img, k_img, v_img, q_img, layout,
        text_keys_only=True)
    _ref_txt, ref_img = _reference(q_txt[:, :, :0], k_txt, v_txt, q_img, k_img, v_img)
    assert torch.allclose(out_img, ref_img, atol=1e-6)
    # and the caption really contributes: the image-only answer differs
    _only_txt, only_img = _reference(q_txt[:, :, :0], k_txt[:, :, :0], v_txt[:, :, :0],
                                     q_img, k_img, v_img)
    assert not torch.allclose(out_img, only_img, atol=1e-6)


def test_the_text_output_has_zero_rows():
    """No text queries means no text answer — the shape says so."""
    h, w, t = 4, 4, 3
    layout = _layout(h, w, [Window(0, 0, h - 1, w - 1)])
    q_txt, k_txt, v_txt, q_img, k_img, v_img = _tensors(t, h * w)
    out_txt, out_img = windowed_attention(
        q_txt[:, :, :0], k_txt, v_txt, q_img, k_img, v_img, q_img, layout,
        text_keys_only=True)
    assert out_txt.shape[2] == 0
    assert out_img.shape[2] == h * w


def test_the_image_output_is_the_tail_after_the_text_queries():
    """The split offset is ``t_q`` — the image rows, not the caption's rows."""
    h, w, t_k = 4, 4, 5
    layout = _layout(h, w, [Window(0, 0, h - 1, w - 1)])
    q_txt, k_txt, v_txt, q_img, k_img, v_img = _tensors(t_k, h * w)
    answers = []
    for t_q in (0, 3):
        out_txt, out_img = windowed_attention(
            q_txt[:, :, :t_q], k_txt, v_txt, q_img, k_img, v_img, q_img, layout,
            text_keys_only=True)
        answers.append(out_img)
    # the image queries' answer does not depend on how many text queries came
    # along — so the split is at t_q, not at t_k
    assert torch.allclose(answers[0], answers[1], atol=1e-6)
    _ref_txt, ref_img = _reference(q_txt[:, :, :0], k_txt, v_txt, q_img, k_img, v_img)
    assert torch.allclose(answers[0], ref_img, atol=1e-6)


def test_the_text_answer_is_the_mean_over_the_windows():
    """The default path is unchanged: the text answer is the mean over windows."""
    h, w, t = 4, 4, 2
    windows = [Window(0, 0, 1, 3), Window(2, 0, 3, 3)]
    layout = _layout(h, w, windows)
    q_txt, k_txt, v_txt, q_img, k_img, v_img = _tensors(t, h * w)
    out_txt, _out_img = windowed_attention(
        q_txt, k_txt, v_txt, q_img, k_img, v_img, q_img, layout)
    # every window re-runs the SAME text queries against the caption's keys
    # and its OWN tokens, and the answer is the mean over the windows
    answers = []
    for win in windows:
        rows = torch.arange(win.r0, win.r1 + 1)
        cols = torch.arange(win.c0, win.c1 + 1)
        idx = (rows[:, None] * w + cols[None, :]).reshape(-1)
        k = torch.cat([k_txt, k_img[:, :, idx]], dim=2)
        v = torch.cat([v_txt, v_img[:, :, idx]], dim=2)
        answers.append(F.scaled_dot_product_attention(q_txt, k, v))
    ref_txt = torch.stack(answers).mean(dim=0)
    assert torch.allclose(out_txt, ref_txt, atol=1e-6)
