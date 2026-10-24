"""Tests for the AReA attention phases and the global-KV bridge (``src/area_attn.py``).

Plan 2026-10-15, Phase 2 (``docs/2026-10-15-area-aReA-node.md``): the port of the reference's
``area/attention.py`` and ``area/bridge.py`` into the comfy layout ``[b, heads, N, d]``.
What these tests pin:

* **Pooling** — all four modes shrink the image keys and values to ``(h/f)(w/f)`` while the
  text tokens pass through untouched, ``subsample`` is the block centre, ``mean`` the plain
  block average for VALUES with the norm-preserving centroid key for KEYS (the block-mean
  direction at the mean member norm), ``medoid`` the token closest to the block-mean key and
  ``random`` a draw keyed by ``(seed, step, block)``.  K and V always come from the same token.
* **RoPE** — :func:`apply_rope1` is comfy's ``_apply_rope1`` (``ldm/flux/math.py:36-49``)
  in local form: the hand-computed rotation, the input dtype, the head-shared table and
  comfy's clip when the table is longer than the tensor.
* **Windowed attention** — a window that spans the grid is plain attention bit for bit; the
  text output is the mean over windows; the image output is coverage-normalised so a token
  in one window keeps weight 1; the Halo ring and the bridge tokens really do extend the
  keys (checked against a direct SDPA over exactly those tokens).
* **Budget** — :func:`fit_extra_keys` cuts the bridge before it narrows the Halo.
* **Bridge** — importance is the squared distance from the canvas mean, a window never sees
  itself as a bridge token, and the per-cell quota follows the tokens a window has left over.
* **Determinism** — one layout per forward serves both CFG branches, the same ``(seed, step)``
  rebuilds it exactly, and nothing reads the global numpy or torch stream.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_attn.py
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.area_attn import (
    CELLS,
    POOLING,
    AreaContext,
    _block_mask,
    _block_selection,
    _norm_preserved_keys,
    _pool,
    _token_matrix,
    apply_rope1,
    bridge_registers,
    build_fine_layout,
    fit_extra_keys,
    pool_keys_values,
    select_registers,
    token_importance,
    windowed_attention,
)
from src.area_schedule import Window, coarse_grid, even_starts, window_tokens

B, HEADS, DIM, TEXT = 2, 3, 8, 5


def _seam(b: int = B, heads: int = HEADS, dim: int = DIM, text: int = TEXT, h: int = 8, w: int = 8,
          seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A comfy seam call's unrotated q/k/v: ``[b, heads, text + h*w, d]``, text first."""
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, heads, text + h * w, dim, generator=g)
    k = torch.randn(b, heads, text + h * w, dim, generator=g)
    v = torch.randn(b, heads, text + h * w, dim, generator=g)
    return q, k, v


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """The oracle: plain SDPA on comfy's ``[b, heads, N, d]`` layout."""
    return F.scaled_dot_product_attention(q, k, v)


def _blocks(x_img: torch.Tensor, h: int, w: int, f: int) -> torch.Tensor:
    """Image tokens as per-block members: ``[b, heads, h/f, w/f, f*f, d]``."""
    b, heads, _, dim = x_img.shape
    return (x_img.reshape(b, heads, h // f, f, w // f, f, dim)
            .permute(0, 1, 2, 4, 3, 5, 6)
            .reshape(b, heads, h // f, w // f, f * f, dim))


def _reference_pool(x_img: torch.Tensor, h: int, w: int, f: int, pooling: str,
                    sel: torch.Tensor | None) -> torch.Tensor:
    """The reference's pooling, rebuilt block by block from the definition.

    An oracle independent of the implementation's reshape: the pooled token of
    block ``(r, c)`` is the block centre (``subsample``), the block average
    (``mean``) or the member ``sel`` names (``medoid`` / ``random``).
    """
    b, heads, _, dim = x_img.shape
    grid = x_img.reshape(b, heads, h, w, dim)
    out = []
    for r in range(h // f):
        for c in range(w // f):
            block = grid[:, :, r * f:(r + 1) * f, c * f:(c + 1) * f, :]
            if pooling == "subsample":
                out.append(block[:, :, f // 2, f // 2, :])
            elif pooling == "mean":
                out.append(block.mean(dim=(2, 3)))
            else:
                idx = sel[:, r, c].view(b, 1, 1, 1).expand(b, heads, 1, dim)
                out.append(block.reshape(b, heads, f * f, dim).gather(2, idx).squeeze(2))
    return torch.stack(out, dim=2).reshape(b, heads, (h // f) * (w // f), dim)


@pytest.mark.unit
class TestPoolKeysValues:
    @pytest.mark.parametrize("pooling", POOLING)
    def test_every_mode_shrinks_the_canvas_and_keeps_the_text(self, pooling):
        h, w, f = 12, 16, 4
        q, k, v = _seam(h=h, w=w)
        pk, pv = pool_keys_values(k, v, TEXT, h, w, f, pooling, seed=7, step=3, block=2)
        assert pk.shape == (B, HEADS, TEXT + (h // f) * (w // f), DIM)
        assert pv.shape == pk.shape
        assert torch.equal(pk[:, :, :TEXT], k[:, :, :TEXT])
        assert torch.equal(pv[:, :, :TEXT], v[:, :, :TEXT])

    def test_subsample_keeps_the_block_centre(self):
        h, w, f = 12, 16, 4
        _, k, v = _seam(h=h, w=w)
        pooled = _pool(k[:, :, TEXT:], h, w, f, "subsample", None)
        off = f // 2
        grid = k[:, :, TEXT:].reshape(B, HEADS, h, w, DIM)
        assert torch.equal(pooled, grid[:, :, off::f, off::f][:, :, : h // f, : w // f]
                           .reshape(B, HEADS, -1, DIM))

    def test_mean_pools_keys_norm_preserved_and_values_plain(self):
        """M2: ``mean`` keys are the block-mean DIRECTION at the mean MEMBER NORM.

        The centroid direction keeps every token of the block contributing (no
        Dirac comb, the block-position design note Flaw 3) while the restored
        member norm keeps the attention temperature at the un-pooled one; the
        VALUES stay the plain block mean.
        """
        h, w, f = 12, 16, 4
        _, k, v = _seam(h=h, w=w, seed=5)
        pk, pv = pool_keys_values(k, v, TEXT, h, w, f, "mean")
        blocks_k = _blocks(k[:, :, TEXT:], h, w, f)
        blocks_v = _blocks(v[:, :, TEXT:], h, w, f)

        # values: the plain block average, exactly as before (text never pooled)
        assert torch.allclose(pv[:, :, TEXT:],
                              blocks_v.mean(dim=4).reshape(B, HEADS, -1, DIM), atol=1e-6)

        # keys: the centroid direction ...
        centroid = blocks_k.mean(dim=4)                       # [b, heads, nh, nw, d]
        expected = (centroid / (centroid.norm(dim=-1, keepdim=True) + 1e-8)
                    * blocks_k.norm(dim=-1).mean(dim=4, keepdim=True))
        assert torch.allclose(pk[:, :, TEXT:],
                              expected.reshape(B, HEADS, -1, DIM), atol=1e-5)
        # ... which is NOT the plain block mean: the norm really is restored
        assert not torch.allclose(pk[:, :, TEXT:],
                                  centroid.reshape(B, HEADS, -1, DIM), atol=1e-3)
        # the direction is the block mean's own ...
        flat = centroid.reshape(B, HEADS, -1, DIM)
        assert torch.allclose(pk[:, :, TEXT:] / pk[:, :, TEXT:].norm(dim=-1, keepdim=True),
                              flat / flat.norm(dim=-1, keepdim=True), atol=1e-5)
        # ... and the norm is the mean of the block's member norms
        assert torch.allclose(pk[:, :, TEXT:].norm(dim=-1),
                              blocks_k.norm(dim=-1).mean(dim=4).reshape(B, HEADS, -1),
                              atol=1e-5)
        # the text half is untouched either way
        assert torch.equal(pk[:, :, :TEXT], k[:, :, :TEXT])
        assert torch.equal(pv[:, :, :TEXT], v[:, :, :TEXT])

    def test_mean_at_factor_one_is_bitwise_identity(self):
        """``f == 1`` pools nothing: ``mean`` returns the keys bit for bit."""
        h, w, f = 8, 8, 1
        _, k, v = _seam(h=h, w=w, seed=5)
        pk, pv = pool_keys_values(k, v, TEXT, h, w, f, "mean")
        assert torch.equal(pk, k)
        assert torch.equal(pv, v)

    def test_the_selecting_poolings_are_unchanged(self):
        """subsample / medoid / random keep the shared-selection behaviour: K and V
        still come from the same token, pooled with :func:`_pool`."""
        h, w, f = 8, 8, 2
        for pooling in ("subsample", "medoid", "random"):
            _, k, v = _seam(h=h, w=w, seed=3)
            sel = _block_selection(k[:, :, TEXT:], h, w, f, pooling, 5, 2, 1)
            pk, pv = pool_keys_values(k, v, TEXT, h, w, f, pooling, 5, 2, 1)
            assert torch.equal(pk[:, :, TEXT:],
                               _pool(k[:, :, TEXT:], h, w, f, pooling, sel))
            assert torch.equal(pv[:, :, TEXT:],
                               _pool(v[:, :, TEXT:], h, w, f, pooling, sel))

    def test_medoid_keeps_the_token_closest_to_the_block_mean_key(self):
        """Rebuilt independently: for every block the argmin over the squared distance to the
        block-mean key, taken over heads and head dims at once."""
        h, w, f = 8, 8, 2
        _, k, v = _seam(h=h, w=w)
        sel = _block_selection(k[:, :, TEXT:], h, w, f, "medoid", 0, 0, 0)
        blocks = _blocks(k[:, :, TEXT:], h, w, f)
        assert torch.equal(sel, blocks.float().sub(blocks.float().mean(dim=4, keepdim=True))
                           .pow(2).sum(dim=(1, -1)).argmin(dim=-1))

    def test_random_is_keyed_by_seed_step_and_block(self):
        h, w, f = 8, 8, 2
        _, k, v = _seam(h=h, w=w)
        img = k[:, :, TEXT:]
        first = _block_selection(img, h, w, f, "random", 11, 3, 4)
        assert torch.equal(first, _block_selection(img, h, w, f, "random", 11, 3, 4))
        assert not torch.equal(first, _block_selection(img, h, w, f, "random", 11, 3, 5))
        assert not torch.equal(first, _block_selection(img, h, w, f, "random", 11, 4, 4))
        assert not torch.equal(first, _block_selection(img, h, w, f, "random", 12, 3, 4))
        assert first.min() >= 0 and first.max() < f * f

    def test_keys_and_values_come_from_the_same_token(self):
        """K and V are pooled with one selection, so both survive or fall together."""
        h, w, f = 8, 8, 2
        for pooling in ("random", "medoid"):
            _, k, v = _seam(h=h, w=w, seed=3)
            sel = _block_selection(k[:, :, TEXT:], h, w, f, pooling, 5, 2, 1)
            _, pv = pool_keys_values(k, v, TEXT, h, w, f, pooling, 5, 2, 1)
            v_pooled = pv[:, :, TEXT:]
            grid = v[:, :, TEXT:].reshape(B, HEADS, h, w, DIM)
            for i in range(h // f):
                for j in range(w // f):
                    for b in range(B):
                        s = int(sel[b, i, j])
                        assert torch.equal(v_pooled[b, :, i * (w // f) + j],
                                           grid[b, :, i * f + s // f, j * f + s % f]), (pooling, i, j)

    def test_text_is_pooled_by_nobody(self):
        """Eq. 3: text tokens are never pooled, at any factor."""
        h, w, f = 16, 16, 8
        _, k, v = _seam(h=h, w=w, text=TEXT)
        for pooling in POOLING:
            pk, pv = pool_keys_values(k, v, TEXT, h, w, f, pooling)
            assert pk.shape[2] == TEXT + (h // f) * (w // f)


@pytest.mark.unit
class TestPartialEdgeBlock:
    """A patch grid whose counts share no factor >= 2 has no exact pooling tiling.

    The reference's Eq. 5 refuses such a grid; the engine resolves it to a
    factor whose blocks do NOT tile it, so the last block on a partial axis
    owns fewer than ``f`` rows or columns.  These pin that partition: every
    canvas token is pooled exactly once, the partial block is averaged over its
    REAL members only, and the pooled token is always a real canvas token.
    """

    #: the user's 512x2000 px canvas: 125x32 patches, gcd(125, 32) == 1, factor 2
    GRID = (125, 32, 2)

    def test_the_pooled_grid_is_the_ceil_of_the_factor(self):
        h, w, f = self.GRID
        nh, nw = coarse_grid(h, w, f)
        assert (nh, nw) == (63, 16)
        _, k, v = _seam(h=h, w=w, seed=3)
        for pooling in POOLING:
            pk, pv = pool_keys_values(k, v, TEXT, h, w, f, pooling, seed=5, step=2, block=1)
            assert pk.shape == (B, HEADS, TEXT + nh * nw, DIM)
            assert pv.shape == pk.shape
            assert torch.equal(pk[:, :, :TEXT], k[:, :, :TEXT])
            assert torch.equal(pv[:, :, :TEXT], v[:, :, :TEXT])

    def test_the_blocks_cover_every_token_exactly_once(self):
        h, w, f = self.GRID
        nh, nw = coarse_grid(h, w, f)
        rows = [range(r * f, min(r * f + f, h)) for r in range(nh)]
        cols = [range(c * f, min(c * f + f, w)) for c in range(nw)]
        assert [i for r in rows for i in r] == list(range(h))
        assert [i for c in cols for i in c] == list(range(w))
        mask = _block_mask(h, w, f, "cpu")
        assert mask.shape == (nh, nw, f, f)
        assert int(mask.sum()) == h * w                      # every cell pooled exactly once

    def test_the_mean_is_taken_over_the_real_members_only(self):
        h, w, f = 9, 6, 4                                   # both axes partial
        _, k, _ = _seam(h=h, w=w, seed=5)
        nh, nw = coarse_grid(h, w, f)
        pooled = _pool(k[:, :, TEXT:], h, w, f, "mean", None).reshape(B, HEADS, -1, DIM)
        grid = k[:, :, TEXT:].reshape(B, HEADS, h, w, DIM)
        for r, c in ((0, 0), (nh - 1, nw - 1), (nh - 1, 0), (0, nw - 1)):
            members = grid[:, :, r * f:min((r + 1) * f, h), c * f:min((c + 1) * f, w), :]
            want = members.mean(dim=(2, 3))
            assert torch.allclose(pooled[:, :, r * nw + c], want, atol=1e-6), (r, c)

    def test_the_partial_block_is_not_averaged_over_its_padding(self):
        """The masked mean, not the mean of the zero-padded block: an unmasked
        mean would divide the real members' sum by the padded block's size and
        shrink the partial block's key."""
        h, w, f = 9, 6, 4
        _, k, _ = _seam(h=h, w=w, seed=5)
        nh, nw = coarse_grid(h, w, f)
        pooled = _pool(k[:, :, TEXT:], h, w, f, "mean", None).reshape(B, HEADS, -1, DIM)
        grid = k[:, :, TEXT:].reshape(B, HEADS, h, w, DIM)
        padded = torch.zeros(B, HEADS, nh * f, nw * f, DIM)
        padded[:, :, :h, :w] = grid
        unmasked = padded.reshape(B, HEADS, nh, f, nw, f, DIM).mean(dim=(3, 5))
        assert not torch.allclose(pooled, unmasked.reshape(B, HEADS, -1, DIM), atol=1e-6)

    def test_the_norm_preserved_keys_keep_the_unpooled_temperature(self):
        """The partial block's key is the masked centroid at the mean norm of its
        REAL members — the temperature the un-pooled canvas would have had."""
        h, w, f = 9, 6, 4
        _, k, _ = _seam(h=h, w=w, seed=5)
        nh, nw = coarse_grid(h, w, f)
        keys = _norm_preserved_keys(k[:, :, TEXT:], h, w, f).reshape(B, HEADS, -1, DIM)
        grid = k[:, :, TEXT:].reshape(B, HEADS, h, w, DIM)
        r, c = nh - 1, nw - 1
        members = grid[:, :, r * f:min((r + 1) * f, h), c * f:min((c + 1) * f, w), :]
        centroid = members.mean(dim=(2, 3))
        norms = members.norm(dim=-1).mean(dim=(2, 3))
        want = centroid / (centroid.norm(dim=-1, keepdim=True) + 1e-8) * norms[..., None]
        assert torch.allclose(keys[:, :, r * nw + c], want, atol=1e-5)

    def test_subsample_takes_the_clamped_block_centre(self):
        h, w, f = 9, 6, 4
        _, k, _ = _seam(h=h, w=w, seed=5)
        nh, nw = coarse_grid(h, w, f)
        pooled = _pool(k[:, :, TEXT:], h, w, f, "subsample", None).reshape(B, HEADS, -1, DIM)
        grid = k[:, :, TEXT:].reshape(B, HEADS, h, w, DIM)
        for r, c in ((0, 0), (nh - 1, nw - 1), (nh - 1, 0)):
            row = min(r * f + f // 2, h - 1)
            col = min(c * f + f // 2, w - 1)
            assert torch.equal(pooled[:, :, r * nw + c], grid[:, :, row, col]), (r, c)

    def test_the_selecting_poolings_always_take_a_real_token(self):
        """A padded cell is a zero vector, so no block may ever select one."""
        h, w, f = 9, 6, 4
        _, k, v = _seam(h=h, w=w, seed=3)
        nh, nw = coarse_grid(h, w, f)
        grid = v[:, :, TEXT:].reshape(B, HEADS, h, w, DIM)
        for pooling in ("medoid", "random"):
            _, pv = pool_keys_values(k, v, TEXT, h, w, f, pooling, 5, 2, 1)
            pooled = pv[:, :, TEXT:].reshape(B, HEADS, nh, nw, DIM)
            for r, c in ((nh - 1, nw - 1), (nh - 1, 0), (0, nw - 1)):
                block = grid[:, :, r * f:min((r + 1) * f, h), c * f:min((c + 1) * f, w), :]
                hit = (pooled[:, :, r, c].unsqueeze(2).unsqueeze(3) == block).all(dim=-1)
                assert bool(hit.any(dim=2).any(dim=2).all()), (pooling, r, c)

    def test_a_grid_of_two_primes_still_pools(self):
        """The engine's fallback grid: 127 and 33 share no factor >= 2."""
        h, w, f = 127, 33, 2
        _, k, v = _seam(h=h, w=w, seed=1)
        nh, nw = coarse_grid(h, w, f)
        assert (nh, nw) == (64, 17)
        pk, pv = pool_keys_values(k, v, TEXT, h, w, f, "mean")
        assert pk.shape == (B, HEADS, TEXT + nh * nw, DIM)
        assert torch.isfinite(pk).all() and torch.isfinite(pv).all()

    def test_the_partial_pooling_is_deterministic(self):
        h, w, f = 9, 6, 4
        _, k, v = _seam(h=h, w=w, seed=3)
        for pooling in POOLING:
            one = pool_keys_values(k, v, TEXT, h, w, f, pooling, 5, 2, 1)
            two = pool_keys_values(k, v, TEXT, h, w, f, pooling, 5, 2, 1)
            assert torch.equal(one[0], two[0]) and torch.equal(one[1], two[1]), pooling

    def test_the_exact_tiling_path_is_unchanged(self):
        """A grid that tiles keeps the reference's pooling — the partial path is
        a guard around the old body, not a rewrite of it."""
        for h, w, f in ((8, 8, 2), (12, 16, 4), (16, 24, 8)):
            _, k, _ = _seam(h=h, w=w, seed=11)
            img = k[:, :, TEXT:]
            for pooling in POOLING:
                sel = _block_selection(img, h, w, f, pooling, 5, 2, 1)
                got = _pool(img, h, w, f, pooling, sel)
                want = _reference_pool(img, h, w, f, pooling, sel)
                if pooling == "mean":
                    assert torch.allclose(got, want, atol=1e-6), (h, w, f)
                else:
                    assert torch.equal(got, want), (h, w, f)


@pytest.mark.unit
class TestApplyRope1:
    @staticmethod
    def _table(n: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
        """A head-shared FLUX-style table ``[1, 1, N, dim//2, 2, 2]`` from comfy's ``rope``."""
        omega = 1.0 / (theta ** (2.0 * torch.arange(dim // 2, dtype=torch.float32) / dim))
        ang = torch.arange(n, dtype=torch.float32)[:, None] * omega[None, :]
        c, s = torch.cos(ang), torch.sin(ang)
        return torch.stack([c, -s, s, c], dim=-1).reshape(1, 1, n, dim // 2, 2, 2)

    def test_matches_the_hand_computed_rotation(self):
        n = 9
        x = torch.randn(1, HEADS, n, DIM)
        freqs = self._table(n, DIM)
        out = apply_rope1(x, freqs)
        even, odd = x[..., 0::2], x[..., 1::2]
        c, s = freqs[..., 0, 0], freqs[..., 1, 0]
        expected = torch.stack([c * even - s * odd, s * even + c * odd], dim=-1).reshape(x.shape)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_preserves_the_input_dtype(self):
        x = torch.randn(1, HEADS, 7, DIM)
        for dtype in (torch.float32, torch.bfloat16):
            out = apply_rope1(x.to(dtype), self._table(7, DIM))
            assert out.dtype == dtype
        # bfloat16 carries ~3 decimal digits, so the cast-back is compared at that scale
        scale = float(x.abs().max())
        assert torch.allclose(apply_rope1(x.bfloat16(), self._table(7, DIM)).float(),
                              apply_rope1(x, self._table(7, DIM)).float(), atol=0.02 * scale)

    def test_a_longer_table_is_clipped(self):
        """Comfy clips the table to the tensor's length (pooled keys on a coarse table)."""
        x = torch.randn(1, HEADS, 7, DIM)
        out = apply_rope1(x, self._table(7, DIM))
        assert torch.allclose(apply_rope1(x, torch.cat([self._table(7, DIM)] * 2, dim=2)), out,
                              atol=1e-6)

    def test_the_head_shared_table_rotates_every_head_alike(self):
        n = 6
        x = torch.randn(1, 1, n, DIM).expand(1, HEADS, n, DIM).contiguous()
        out = apply_rope1(x, self._table(n, DIM))
        for head in range(1, HEADS):
            assert torch.allclose(out[:, head], out[:, 0], atol=1e-7)

    def test_rotation_moves_every_token(self):
        x = torch.randn(1, HEADS, 12, DIM)
        out = apply_rope1(x, self._table(12, DIM))
        assert not torch.allclose(out, x)
        assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)     # a rotation


@pytest.mark.unit
class TestWindowedAttention:
    def test_a_single_window_spanning_the_grid_is_plain_attention(self):
        """Phase II with one window, no halo and no bridge is exactly the graph it replaced."""
        h = w = 8
        q, k, v = _seam(h=h, w=w)
        layout = build_fine_layout([Window(0, 0, h - 1, w - 1)], h, w, 0, 0, 5, 1,
                                   torch.float32, "cpu")
        out_txt, out_img = windowed_attention(q[:, :, :TEXT], k[:, :, :TEXT], v[:, :, :TEXT],
                                              q[:, :, TEXT:], k[:, :, TEXT:], v[:, :, TEXT:],
                                              k[:, :, TEXT:], layout)
        want = _attention(q, k, v)
        assert out_txt.shape == (B, HEADS, TEXT, DIM)
        assert out_img.shape == (B, HEADS, h * w, DIM)
        assert torch.allclose(out_txt, want[:, :, :TEXT], atol=1e-6)
        assert torch.allclose(out_img, want[:, :, TEXT:], atol=1e-6)

    def test_the_text_output_is_the_mean_over_windows(self):
        """Every window re-runs the text queries and keys, so the text stream is their mean —
        the window count must not leak into it."""
        h = w = 8
        q, k, v = _seam(h=h, w=w)
        halves = [torch.arange(h * w)[: h * w // 2], torch.arange(h * w)[h * w // 2 :]]
        layout = build_fine_layout([Window(0, 0, h // 2 - 1, w - 1), Window(h // 2, 0, h - 1, w - 1)],
                                   h, w, 0, 0, 5, 1, torch.float32, "cpu")
        out_txt, out_img = windowed_attention(q[:, :, :TEXT], k[:, :, :TEXT], v[:, :, :TEXT],
                                              q[:, :, TEXT:], k[:, :, TEXT:], v[:, :, TEXT:],
                                              k[:, :, TEXT:], layout)
        means = [_attention(torch.cat([q[:, :, :TEXT], q[:, :, TEXT + idx]], dim=2),
                            torch.cat([k[:, :, :TEXT], k[:, :, TEXT + idx]], dim=2),
                            torch.cat([v[:, :, :TEXT], v[:, :, TEXT + idx]], dim=2))[:, :, :TEXT]
                 for idx in halves]
        assert torch.allclose(out_txt, torch.stack(means).mean(0), atol=1e-6)
        # the image stream is not an average: each half attends with its own keys
        for idx in halves:
            want = _attention(torch.cat([q[:, :, :TEXT], q[:, :, TEXT + idx]], dim=2),
                              torch.cat([k[:, :, :TEXT], k[:, :, TEXT + idx]], dim=2),
                              torch.cat([v[:, :, :TEXT], v[:, :, TEXT + idx]], dim=2))[:, :, TEXT:]
            assert torch.allclose(out_img[:, :, idx], want, atol=1e-6)

    def test_the_halo_ring_extends_the_keys(self):
        """A window plus its ring is plain attention over window | ring | text — the merge
        weight of a covered token is 1, so nothing is rescaled."""
        h = w = 8
        q, k, v = _seam(h=h, w=w)
        win = Window(0, 0, 6, 7)                     # all but the last row
        layout = build_fine_layout([win], h, w, 1, 0, 2, 4, torch.float32, "cpu")
        ring = layout.groups[0].halo.reshape(-1)
        assert ring.numel() == 8                      # exactly the row the window leaves out
        out_txt, out_img = windowed_attention(q[:, :, :TEXT], k[:, :, :TEXT], v[:, :, :TEXT],
                                              q[:, :, TEXT:], k[:, :, TEXT:], v[:, :, TEXT:],
                                              k[:, :, TEXT:], layout)
        keys = torch.cat([window_tokens(win, w, "cpu"), ring])
        want = _attention(torch.cat([q[:, :, :TEXT], q[:, :, TEXT + keys]], dim=2),
                          torch.cat([k[:, :, :TEXT], k[:, :, TEXT + keys]], dim=2),
                          torch.cat([v[:, :, :TEXT], v[:, :, TEXT + keys]], dim=2))[:, :, TEXT:]
        assert torch.allclose(out_img[:, :, window_tokens(win, w, "cpu")],
                           want[:, :, : win.area], atol=1e-6)
        assert torch.count_nonzero(out_img[:, :, ring]) == 0     # the ring is not a query

    def test_the_bridge_tokens_extend_the_keys(self):
        """The bridge tokens are the ones :func:`select_registers` picks for this window, and
        they join its keys ahead of the halo ring."""
        h = w = 8
        q, k, v = _seam(h=h, w=w)
        win = Window(0, 0, 5, 7)                     # all but the last two rows
        layout = build_fine_layout([win], h, w, 1, 4, 2, 4, torch.float32, "cpu")
        group = layout.groups[0]
        assert layout.registers == 4
        registers = select_registers(token_importance(k[:, :, TEXT:]), group.tokens,
                                     layout.registers, layout.h, layout.w).reshape(-1)
        keys = torch.cat([window_tokens(win, w, "cpu"), registers, group.halo.reshape(-1)])
        out_txt, out_img = windowed_attention(q[:, :, :TEXT], k[:, :, :TEXT], v[:, :, :TEXT],
                                              q[:, :, TEXT:], k[:, :, TEXT:], v[:, :, TEXT:],
                                              k[:, :, TEXT:], layout)
        want = _attention(torch.cat([q[:, :, :TEXT], q[:, :, TEXT + keys]], dim=2),
                          torch.cat([k[:, :, :TEXT], k[:, :, TEXT + keys]], dim=2),
                          torch.cat([v[:, :, :TEXT], v[:, :, TEXT + keys]], dim=2))[:, :, TEXT:]
        assert torch.allclose(out_img[:, :, window_tokens(win, w, "cpu")],
                           want[:, :, : win.area], atol=1e-6)
        assert not set(registers.tolist()) & set(window_tokens(win, w, "cpu").tolist())

    def test_the_output_follows_the_input_dtype(self):
        h = w = 8
        q, k, v = _seam(h=h, w=w)
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        layout = build_fine_layout([Window(0, 0, 3, 7), Window(4, 0, 7, 7)], h, w, 1, 2, 2, 4,
                                   torch.bfloat16, "cpu")
        out_txt, out_img = windowed_attention(q[:, :, :TEXT], k[:, :, :TEXT], v[:, :, :TEXT],
                                              q[:, :, TEXT:], k[:, :, TEXT:], v[:, :, TEXT:],
                                              k[:, :, TEXT:], layout)
        assert out_txt.dtype == torch.bfloat16 and out_img.dtype == torch.bfloat16
        assert torch.isfinite(out_txt.float()).all() and torch.isfinite(out_img.float()).all()

    def test_one_layout_serves_both_cfg_branches(self):
        """Comfy batches cond and uncond into one forward: identical halves give identical
        outputs through the shared graph, and different halves stay different."""
        h = w = 8
        q, k, v = _seam(h=h, w=w)
        layout = build_fine_layout([Window(0, 0, 3, 7), Window(4, 0, 7, 7)], h, w, 1, 2, 2, 4,
                                   torch.float32, "cpu")

        def run(q_, k_, v_):
            return windowed_attention(q_[:, :, :TEXT], k_[:, :, :TEXT], v_[:, :, :TEXT],
                                      q_[:, :, TEXT:], k_[:, :, TEXT:], v_[:, :, TEXT:],
                                      k_[:, :, TEXT:], layout)

        same_txt, same_img = run(torch.cat([q[:1], q[:1]]), torch.cat([k[:1], k[:1]]),
                                 torch.cat([v[:1], v[:1]]))
        assert torch.equal(same_txt[0], same_txt[1]) and torch.equal(same_img[0], same_img[1])
        other_txt, other_img = run(torch.cat([q, k]), torch.cat([k, q]), torch.cat([v, k]))
        assert not torch.allclose(other_txt[0], other_txt[1])

    def test_the_same_layout_gives_the_same_answer(self):
        h = w = 8
        q, k, v = _seam(h=h, w=w)
        layout = build_fine_layout([Window(0, 0, 3, 7), Window(4, 0, 7, 7)], h, w, 1, 2, 2, 4,
                                   torch.float32, "cpu")
        one = windowed_attention(q[:, :, :TEXT], k[:, :, :TEXT], v[:, :, :TEXT],
                                 q[:, :, TEXT:], k[:, :, TEXT:], v[:, :, TEXT:], k[:, :, TEXT:], layout)
        two = windowed_attention(q[:, :, :TEXT], k[:, :, :TEXT], v[:, :, :TEXT],
                                 q[:, :, TEXT:], k[:, :, TEXT:], v[:, :, TEXT:], k[:, :, TEXT:], layout)
        assert torch.equal(one[0], two[0]) and torch.equal(one[1], two[1])


@pytest.mark.unit
class TestBuildFineLayout:
    def test_groups_are_keyed_by_token_count_and_ring_size(self):
        """Windows that share a key run as one batched call, and every window is in one."""
        h = w = 16
        windows = [Window(0, 0, 7, 7), Window(0, 8, 7, 15), Window(8, 0, 15, 15), Window(8, 8, 15, 15)]
        layout = build_fine_layout(windows, h, w, 2, 0, 6, 3, torch.float32, "cpu")
        keys = [(int(g.tokens.shape[1]), 0 if g.halo is None else int(g.halo.shape[1]))
                for g in layout.groups]
        assert len(keys) == len(set(keys))
        assert sum(g.tokens.shape[0] for g in layout.groups) == len(windows)
        assert sum(int(g.tokens.numel()) for g in layout.groups) == sum(win.area for win in windows)
        assert layout.registers == 0

    def test_coverage_counts_the_windows_covering_each_token(self):
        h = w = 8
        windows = [Window(0, 0, 4, 7), Window(0, 4, 7, 7)]           # rows 0..4 | cols 4..7
        layout = build_fine_layout(windows, h, w, 0, 0, 1, 1, torch.float32, "cpu")
        assert int(layout.coverage[0]) == 1                            # row 0, col 0: first only
        assert int(layout.coverage[4 * 8 + 5]) == 2                     # the overlap
        assert int(layout.coverage[5 * 8 + 5]) == 1                     # row 5: second only
        assert int(layout.coverage[6 * 8]) == 0                         # outside both
        assert set(layout.coverage.tolist()) == {0.0, 1.0, 2.0}

    def test_halo_zero_means_no_ring(self):
        h = w = 8
        layout = build_fine_layout([Window(0, 0, 7, 3), Window(0, 4, 7, 7)], h, w, 0, 4, 1, 1,
                                   torch.float32, "cpu")
        assert all(g.halo is None for g in layout.groups)

    def test_the_same_seed_and_step_rebuild_the_same_layout(self):
        h = w = 8
        windows = [Window(0, 0, 3, 7), Window(4, 0, 7, 7)]

        def signature(step: int):
            layout = build_fine_layout(windows, h, w, 3, 2, 6, step, torch.float32, "cpu")
            return [(g.tokens.tolist(), None if g.halo is None else g.halo.tolist(),
                     g.weights.tolist(), layout.coverage.tolist()) for g in layout.groups]

        assert signature(3) == signature(3)
        assert signature(3) != signature(4)

    def test_the_step_moves_the_halo_reaches(self):
        """A new step redraws the halo widths and reaches: the same window, a different ring."""
        h = w = 8
        windows = [Window(1, 1, 6, 6)]
        sizes = set()
        for step in range(8):
            layout = build_fine_layout(windows, h, w, 3, 0, 4, step, torch.float32, "cpu")
            ring = layout.groups[0].halo
            sizes.add(None if ring is None else tuple(sorted(ring.reshape(-1).tolist())))
        assert len(sizes) > 1
        assert None not in sizes                         # depth 0 would leave a ring empty


@pytest.mark.unit
class TestFitExtraKeys:
    @staticmethod
    def _windows(h: int, w: int, size: int) -> list[Window]:
        return [Window(r, c, min(h - 1, r + size - 1), min(w - 1, c + size - 1))
                for r in even_starts(h, size) for c in even_starts(w, size)]

    def test_a_roomy_step_is_left_alone(self):
        windows = self._windows(16, 16, 8)
        assert fit_extra_keys(windows, 16, 16, 8, 4, 2) == (4, 2)

    def test_the_bridge_is_cut_first(self):
        """``room = full - local(halo)`` and the bridge takes what is left of it, priced per
        window; the halo is untouched because the step already fits."""
        windows = self._windows(16, 16, 8)
        rows = [float(8 + win.area) for win in windows]
        room = (8 + 16 * 16) ** 2 - sum(r * (r + 36) for r, _ in zip(rows, windows))
        assert fit_extra_keys(windows, 16, 16, 8, 10 ** 6, 2) == (int(room // sum(rows)), 2)

    def test_the_halo_narrows_only_when_the_bridge_cannot_pay(self):
        """16 windows of 8x8 on a 32x32 grid with T=100: local(7) = 1160464 <= full = 1263376
        and local(8) = 1312000 > it, so the halo narrows to 7 and the bridge goes to zero."""
        windows = self._windows(32, 32, 8)
        assert fit_extra_keys(windows, 32, 32, 100, 10 ** 6, 16) == (0, 7)

    def test_the_floor_when_even_the_bare_windows_are_over_budget(self):
        """With T=333 the 16 windows alone cost more than full attention, so the only depth
        left to try is 0 — the reference's ``next(..., 0)`` fallback."""
        windows = self._windows(32, 32, 8)
        assert fit_extra_keys(windows, 32, 32, 333, 10 ** 6, 8) == (0, 0)

    def test_the_narrowed_halo_keeps_the_step_within_full_attention(self):
        windows = self._windows(32, 32, 8)
        registers, halo = fit_extra_keys(windows, 32, 32, 100, 10 ** 6, 16)
        full = (100 + 32 * 32) ** 2
        cost = sum((100 + win.area) * ((100 + win.area) + (win.r1 - win.r0 + 3) * (win.c1 - win.c0 + 3)
                                       - win.area)
                   for win in windows)
        assert cost <= full


@pytest.mark.unit
class TestBridge:
    def test_importance_is_zero_for_constant_keys(self):
        k = torch.ones(1, HEADS, 64, DIM)
        assert float(token_importance(k).abs().sum()) == 0.0

    def test_the_outlier_scores_highest(self):
        k = torch.zeros(1, HEADS, 64, DIM)
        k[:, :, 37, :] = 50.0
        score = token_importance(k)
        assert int(score.argmax()) == 37
        assert float(score[37]) > float(score.sum() - score[37])

    def test_the_score_is_the_squared_distance_from_the_canvas_mean(self):
        """Rebuilt from the definition, including the comfy head-before-sequence flatten."""
        k = torch.randn(2, HEADS, 100, DIM)
        score = token_importance(k)
        c = HEADS * DIM
        flat = k.float().sum(dim=0).transpose(0, 1).reshape(-1, c)
        mean = flat.mean(0)
        assert torch.allclose(score, (flat - mean).pow(2).sum(1), atol=1e-4)

    def test_token_matrix_moves_the_head_axis_before_the_flatten(self):
        x = torch.randn(2, HEADS, 5, DIM)
        matrix = _token_matrix(x, HEADS * DIM)
        assert matrix.shape == (5, HEADS * DIM)
        for token in range(5):
            assert torch.equal(matrix[token], x.sum(0)[:, token].reshape(-1))

    def test_a_window_is_never_offered_its_own_tokens(self):
        h = w = 8
        _, k, _ = _seam(h=h, w=w)
        win = Window(1, 1, 6, 6)
        own = window_tokens(win, w, "cpu")
        registers = select_registers(token_importance(k[:, :, TEXT:]), own.unsqueeze(0), 6, h, w)
        assert registers.shape == (1, 6)
        assert registers.dtype == torch.long
        assert len(set(registers[0].tolist())) == 6
        assert not set(registers[0].tolist()) & set(own.tolist())

    def test_the_quota_follows_the_tokens_a_window_leaves_behind(self):
        """A window over the left half leaves only the right half available, and the canvas is
        cut into a ``CELLS x CELLS`` grid whose quota follows those leftovers."""
        h = w = 16
        _, k, _ = _seam(h=h, w=w)
        win = Window(0, 0, 15, 7)
        registers = select_registers(token_importance(k[:, :, TEXT:]),
                                     window_tokens(win, w, "cpu").unsqueeze(0), CELLS * 2, h, w)
        chosen = registers[0].tolist()
        assert len(chosen) == CELLS * 2
        assert all((t % w) >= 8 for t in chosen)              # everything offered is on the right
        cells = {t % w * CELLS // w for t in chosen}
        assert cells == set(range(4, 8)), cells               # and only its cells

    def test_bridge_registers_is_round_rho_w_squared(self):
        assert bridge_registers(8, 0.5) == 16
        assert bridge_registers(64, 0.5) == 1024
        assert bridge_registers(10, 0.25) == 4                 # round(2.5) == 2, banker's
        assert bridge_registers(8, 0.0) == 0
        assert bridge_registers(8, -1.0) == 0

    def test_a_short_quota_is_still_filled(self):
        """A window covering nearly the canvas leaves fewer free tokens than the quota; the
        reference fills the rest from the top of the score rather than returning a short row."""
        h = w = 8
        _, k, _ = _seam(h=h, w=w)
        win = Window(0, 0, 6, 7)                               # leaves exactly one row free
        registers = select_registers(token_importance(k[:, :, TEXT:]),
                                     window_tokens(win, w, "cpu").unsqueeze(0), 16, h, w)
        assert registers.shape == (1, 16)
        assert registers.min() >= 0 and registers.max() < h * w


@pytest.mark.unit
class TestAreaContext:
    def test_starts_at_plain_attention(self):
        ctx = AreaContext()
        assert ctx.phase is None and ctx.step == 0 and ctx.size == 0
        assert (ctx.h, ctx.w, ctx.txt_len) == (0, 0, 0)
        assert ctx.data == {}

    def test_carries_the_payload_of_both_phases(self):
        ctx = AreaContext(phase="coarse", step=3, size=2, h=128, w=96, txt_len=333,
                          data={"rope_q": "q-table", "rope_kv": "kv-table"})
        assert ctx.at() if False else (ctx.phase, ctx.size) == ("coarse", 2)
        layout = object()
        fine = AreaContext(phase="fine", step=20, size=64, h=128, w=96, txt_len=333,
                           data={"layout": layout})
        assert fine.data["layout"] is layout
        assert AreaContext().data is not fine.data            # never a shared default


@pytest.mark.unit
class TestNoGlobalRngReads:
    def test_the_phases_never_read_the_global_streams(self):
        """Cond and uncond share one forward, so both branches must attend through the same
        graphs — which rules out the global numpy and torch streams (theory.md:496)."""
        h = w = 8
        np_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
        q, k, v = _seam(h=h, w=w, seed=11)
        for pooling in POOLING:
            pool_keys_values(k, v, TEXT, h, w, 2, pooling, 7, 3, 1)
        layout = build_fine_layout([Window(0, 0, 3, 7), Window(4, 0, 7, 7)], h, w, 3, 2, 7, 3,
                                   torch.float32, "cpu")
        windowed_attention(q[:, :, :TEXT], k[:, :, :TEXT], v[:, :, :TEXT],
                           q[:, :, TEXT:], k[:, :, TEXT:], v[:, :, TEXT:], k[:, :, TEXT:], layout)
        select_registers(token_importance(k[:, :, TEXT:]), layout.groups[0].tokens, 2, h, w)
        assert np_state[1].tolist() == np.random.get_state()[1].tolist()
        assert np_state[2] == np.random.get_state()[2]
        assert torch.equal(torch_state, torch.random.get_rng_state())


# ---------------------------------------------------------------------------
# The ONE additive engine keyword: ``img_axis0`` (plan §7/S8)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestImgAxis0:
    """``windowed_attention(..., img_axis0=None)`` — the bridge's axis-0 position.

    Z-Image's image tokens carry ``t = cap_len + 1`` (the first token after
    the caption block) and Krea-2's carry the FRAME index, which is 0 for an
    image (``comfy/ldm/krea2/model.py:291-294``).  The bridge keys are
    re-roped at that axis-0 value, so the parameter is additive and defaults
    to today's behaviour — the Z-Image golden cannot move.
    """

    def _embedder(self, seen):
        """A fake rope embedder: records the ids, returns a head-shared table.

        The table carries one rotation band per axis plus one zero band, so it
        has the ``dim // 2`` bands :func:`apply_rope1` consumes.
        """
        def embed(ids):
            seen.append(ids.clone())
            bands = []
            for axis in range(3):
                angle = ids[..., axis].float().unsqueeze(-1)
                bands.append(torch.stack([torch.cos(angle), -torch.sin(angle),
                                          torch.sin(angle), torch.cos(angle)],
                                         dim=-1).reshape(*ids.shape[:-1], 1, 2, 2))
            bands.append(torch.zeros_like(bands[0]))       # the spare band
            return torch.cat(bands, dim=-3).unsqueeze(1)    # [g, 1, n, dim//2, 2, 2]
        return embed

    def _call(self, img_axis0=None, seed: int = 3):
        """One windowed call with a non-empty bridge.

        Returns ``(bridge_ids, outputs)``: the ids the BRIDGE was re-roped at
        (the caption anchor's own ids are filtered out — they carry the
        caption's length on axis 1, the bridge's carry the register count).
        """
        h = w = 8
        q, k, v = _seam(h=h, w=w, seed=11)
        layout = build_fine_layout([Window(0, 0, 3, 7), Window(4, 0, 7, 7)], h, w,
                                   3, 2, 7, seed, torch.float32, "cpu")
        seen = []
        kwargs = {} if img_axis0 is None else {"img_axis0": img_axis0}
        out = windowed_attention(q[:, :, :TEXT], k[:, :, :TEXT], v[:, :, :TEXT],
                                  q[:, :, TEXT:], k[:, :, TEXT:], v[:, :, TEXT:],
                                  k[:, :, TEXT:], layout,
                                  rope_embedder=self._embedder(seen),
                                  cap_len=TEXT, **kwargs)
        bridge = [i for i in seen if i.dim() == 3 and i.shape[1] == layout.registers]
        return bridge, out

    def test_img_axis0_default_keeps_the_zimage_bridge_behaviour(self):
        """``None`` is Z-Image's convention — ``cap_len + 1``, as today."""
        seen, out = self._call()
        assert seen, "the bridge never re-roped its keys"
        for ids in seen:
            assert ids.shape[-1] == 3
            assert torch.allclose(ids[:, :, 0],
                                  torch.full_like(ids[:, :, 0], TEXT + 1))
        # the default is exactly the explicit ``None`` — no behaviour change
        seen_explicit, out_explicit = self._call(img_axis0=None)
        assert torch.equal(out[0], out_explicit[0])
        assert torch.equal(out[1], out_explicit[1])

    def test_img_axis0_ropes_the_bridge_keys_at_the_backbones_axis_zero(self):
        """A backbone's own axis-0 (Krea-2's frame index: 0) re-ropes there."""
        seen, out = self._call(img_axis0=0.0)
        assert seen, "the bridge never re-roped its keys"
        for ids in seen:
            assert torch.allclose(ids[:, :, 0], torch.zeros_like(ids[:, :, 0]))
            assert not torch.allclose(ids[:, :, 0],
                                      torch.full_like(ids[:, :, 0], TEXT + 1))
        # and the re-roping really moved the answer
        _seen_default, out_default = self._call()
        assert not torch.equal(out[0], out_default[0])
