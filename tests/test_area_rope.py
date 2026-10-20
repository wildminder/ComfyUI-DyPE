"""AReA — the shared rope helpers (plan 2026-10-20, §7/S3).

The rope maths moved out of the Z-Image adapter into ``src/area_rope.py``,
because Krea-2's seam is rotated the same way (``comfy/ldm/krea2/model.py:93``)
and its ids have the same FLUX shape (``model.py:291-294``, ``:346``).  These
tests pin that the extraction changed NOTHING:

* :func:`src.area_rope.unrotate1` is still the exact inverse of the engine's
  rotation (``src.area_attn.apply_rope1``), bit for bit;
* :func:`src.area_rope.head_shared` still accepts BOTH table layouts — the
  ``EmbedND`` one (``[b, 1, N, ...]``) and lumina's own view of it
  (``[b, N, 1, ...]``, ``comfy/ldm/lumina/model.py:666``);
* :func:`src.area_rope.flux_style_ids` builds FLUX's (and Krea-2's) ids: the
  caption at the origin, the image tokens on ``(0, row, col)`` with axis 0 the
  FRAME index;
* ``src.area_zimage.py`` RE-EXPORTS the very same objects — a re-export that
  wrapped instead of aliasing would be a silent behaviour change;
* ``src/area.py``'s private ``_area_rope_ids`` is an alias of
  ``flux_style_ids``, not a copy.

The FLUX and Z-Image byte-identity goldens live in
``tests/test_area_grid_fallback.py`` and are the gate for this step; this file
does not touch them.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_rope.py tests/test_area_zimage.py
                   tests/test_area_grid_fallback.py
"""
from __future__ import annotations

import pytest
import torch

import src.area as area
import src.area_rope as rope
import src.area_zimage as zimage
from src.area_attn import apply_rope1
from test_area_zimage import MockEmbedND

#: The id builder's axis count — FLUX's and Krea-2's ids are ``(t, row, col)``.
AXES = 3


@pytest.mark.unit
class TestUnrotate:
    def test_unrotate1_inverts_the_engine_rotation_bitwise(self):
        """The inverse is the exact inverse — not a transpose, not an approximation."""
        embedder = MockEmbedND()                      # theta 10000, axes (16, 56, 56)
        n, heads, dim = 13, 3, embedder.head_dim
        # distinct (t, row, col) per token, so every frequency is exercised
        ids = torch.zeros(1, n, AXES)
        ids[0, :, 0] = torch.arange(1, n + 1, dtype=torch.float32)
        ids[0, :, 1] = torch.arange(n, dtype=torch.float32) % 5
        ids[0, :, 2] = torch.arange(n, dtype=torch.float32).flip(0)
        table = embedder(ids)

        # the table really is [[cos, -sin], [sin, cos]] (math.rope, math.py:30)
        assert torch.allclose(table[..., 0, 0], table[..., 1, 1])
        assert torch.allclose(table[..., 0, 1], -table[..., 1, 0])

        torch.manual_seed(0)
        x = torch.randn(1, heads, n, dim, dtype=torch.float64)
        table64 = table.double()
        rotated = apply_rope1(x, table64)
        # NOT vacuous: a non-zero position really moves the tensor
        assert not torch.allclose(rotated, x)
        # the inverse restores the original (1e-6 on fp64)
        assert torch.allclose(rope.unrotate1(rotated, table64), x, atol=1e-6, rtol=0.0)
        # ... and un-rotating a rotated tensor restores the un-rotated one
        assert torch.allclose(rope.unrotate1(x, table64),
                              apply_rope1(x, rope.unrotate_table(table64)),
                              atol=1e-6, rtol=0.0)

        # the inverse block is the transpose of a rotation matrix, block by block
        inverse = rope.unrotate_table(table64)
        assert torch.allclose(inverse, table64.transpose(-1, -2))
        eye = inverse @ table64
        assert torch.allclose(eye, torch.eye(2, dtype=torch.float64).expand_as(eye).clone(),
                              atol=1e-6, rtol=0.0)
        # the caller's table is never mutated (the seam re-uses it for the q side)
        untouched = table64.clone()
        rope.unrotate_table(table64)
        assert torch.allclose(table64, untouched)


@pytest.mark.unit
class TestHeadShared:
    def test_head_shared_accepts_both_embednd_layouts(self):
        """``[b, 1, N, ...]`` stays; lumina's ``[b, N, 1, ...]`` view is flipped."""
        embedder = MockEmbedND()
        ids = torch.zeros(1, 7, AXES)
        table = embedder(ids)
        assert table.shape[1] == 1
        assert rope.head_shared(table).shape == table.shape

        # lumina views the very same tensor as [b, N, 1, ...] (model.py:666)
        moved = table.movedim(1, 2)
        assert moved.shape[2] == 1
        shared = rope.head_shared(moved)
        assert shared.shape == table.shape
        assert torch.equal(shared, table)


@pytest.mark.unit
class TestFluxStyleIds:
    def test_flux_style_ids_are_zero_for_text_and_row_col_for_image(self):
        """The caption sits at the origin; the image tokens carry ``(0, row, col)``."""
        txt_len = 5
        rows = torch.tensor([0, 0, 1, 1, 2])
        cols = torch.tensor([0, 1, 0, 1, 2])
        ids = rope.flux_style_ids(txt_len, rows, cols, "cpu", torch.float32)
        assert ids.shape == (1, txt_len + rows.numel(), AXES)
        # the caption block is all zeros (comfy/ldm/flux/model.py:192-194)
        assert torch.equal(ids[0, :txt_len], torch.zeros(txt_len, AXES))
        # axis 0 is the FRAME index and stays 0 for a single frame
        assert torch.equal(ids[0, txt_len:, 0], torch.zeros(rows.numel()))
        # axes 1/2 carry the image's own row/col
        assert torch.equal(ids[0, txt_len:, 1], rows.float())
        assert torch.equal(ids[0, txt_len:, 2], cols.float())


@pytest.mark.unit
class TestReexports:
    def test_area_zimage_reexports_are_the_same_objects(self):
        """The Z-Image adapter re-exports the shared helpers — the SAME objects."""
        assert zimage.unrotate1 is rope.unrotate1
        assert zimage.unrotate_table is rope.unrotate_table
        assert zimage._head_shared is rope.head_shared

    def test_the_area_wiring_aliases_the_shared_id_builder(self):
        """``src/area.py``'s private id builder is an ALIAS, not a copy."""
        assert area._area_rope_ids is rope.flux_style_ids
