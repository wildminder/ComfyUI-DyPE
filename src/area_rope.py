"""Shared rope helpers for the AReA adapters (plan 2026-10-20, §5.1/S3).

The backbones that need them disagree about the LAYOUT of a rope table, not
about the maths, so the maths lives here once:

* **FLUX** (``comfy/ldm/flux/model.py``) and **Krea-2**
  (``comfy/ldm/krea2/model.py``) build their position ids the same way — the
  caption sits at the origin and every image token carries ``(0, row, col)``,
  axis 0 being the FRAME index — so one id builder serves both
  (:func:`flux_style_ids`).
* **Z-Image / Lumina** (``comfy/ldm/lumina/model.py``) views the very same
  ``EmbedND`` output in the other axis order — ``[b, N, 1, axes, 2, 2]`` after
  ``movedim(1, 2)`` (``model.py:666``) — so a table has to be normalised before
  the engine's rotation consumes it (:func:`head_shared`).

Two backbones also need the INVERSE rotation: their seam receives q/k that the
model already rotated (``comfy/ldm/krea2/model.py:93``, ``comfy/ldm/lumina/
model.py:159-173``) and no rope table, so Phase I must pool *unrotated* keys
(:func:`unrotate_table`, :func:`unrotate1`).

``src/area_zimage.py`` re-exports these under its historical names
(``unrotate_table``, ``unrotate1``, ``_head_shared``), so the existing wiring
(``src/area.py``) and the existing tests keep importing from there unchanged.

The only pack import is the engine's own rotation, :func:`src.area_attn.apply_rope1`
— a byte-for-byte port of comfy's ``_apply_rope1`` (``comfy/ldm/flux/math.py:
36-49``) — because the inverse must be the exact inverse of the forward
rotation.  ``src/area_attn.py`` imports nothing from here at module level (its
own use of :func:`head_shared` stays a lazy, in-function import), so the two
modules do not form an import cycle.
"""
from __future__ import annotations

import torch

from .area_attn import apply_rope1

__all__ = [
    "flux_style_ids",
    "head_shared",
    "unrotate1",
    "unrotate_table",
]


def head_shared(table: torch.Tensor) -> torch.Tensor:
    """A rope table in the head-shared layout ``[b, 1, N, axes, 2, 2]``.

    ``EmbedND`` already returns this layout (``emb.unsqueeze(1)``,
    ``comfy/ldm/flux/layers.py:15-29``) and lumina views the very same tensor as
    ``[b, N, 1, axes, 2, 2]`` after ``movedim(1, 2)`` (``model.py:666``), so a
    table may legitimately arrive in either.  The two are told apart by which of
    axes 1/2 is the singleton, and the flip is a view (no copy).  This is the
    layout :func:`src.area_attn.apply_rope1` consumes.
    """
    if table.dim() >= 3 and table.shape[1] != 1 and table.shape[2] == 1:
        return table.movedim(1, 2)
    return table


def unrotate_table(table: torch.Tensor) -> torch.Tensor:
    """The inverse of a rope table — the rotation by the NEGATED angle.

    ``math.rope()`` (``comfy/ldm/flux/math.py:20-32``) stacks
    ``[cos, -sin, sin, cos]`` into the 2x2 block ``[[cos, -sin], [sin, cos]]``, a
    rotation by ``+position * omega``; its inverse is that block with the two
    off-diagonals negated (``[[cos, sin], [-sin, cos]]``), which for a rotation
    matrix is also the transpose.  Negating — rather than transposing — keeps the
    intent explicit and is what plan §4 pins.

    The caller's table is not touched (the result is a fresh tensor), so a table
    can be inverted and then reused for the forward rotation.
    """
    inverse = table.clone()
    inverse[..., 0, 1] = -table[..., 0, 1]
    inverse[..., 1, 0] = -table[..., 1, 0]
    return inverse


def unrotate1(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """Undo one rope rotation of ``x`` ``[b, heads, N, d]`` by ``table``.

    The inverse of :func:`src.area_attn.apply_rope1` — the engine's own rotation
    applied with :func:`unrotate_table` — so the seam's already-rotated q/k come
    back to the unrotated frame Phase I must pool from, and a pooled key can be
    re-roped exactly (plan §4).  Values and dtype follow comfy: the math runs in
    the table's dtype and is cast back.
    """
    return apply_rope1(x, unrotate_table(head_shared(table)))


def flux_style_ids(txt_len: int, rows: torch.Tensor, cols: torch.Tensor,
                   device, dtype) -> torch.Tensor:
    """``[1, txt_len + n, 3]`` rope position ids: zero text ids, image rows/cols.

    ``ldm/flux/model.py:192-194`` builds exactly this (text ids all zero, image
    ids carrying row/col on axes 1/2 with the axis-0 index at 0), and so does
    Krea-2 — ``txtpos`` is zeros (``comfy/ldm/krea2/model.py:346``) and the image
    ids are ``(index, row, col)`` with the frame index on axis 0
    (``model.py:291-294``).  So a stock ``pe_embedder`` returns the same table
    the model itself computes for the canvas.

    Axis 0 stays ZERO for the image tokens: it is the frame index, and a
    single-frame image carries frame 0 (a video frame's index would be the
    frame's own number).
    """
    ids = torch.zeros(1, txt_len + int(rows.numel()), 3, device=device, dtype=dtype)
    ids[0, txt_len:, 1] = rows.to(dtype)
    ids[0, txt_len:, 2] = cols.to(dtype)
    return ids
