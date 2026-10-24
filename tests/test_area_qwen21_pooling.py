"""Qwen-Image-2.1 Phase I — the pooled key/value PROVENANCE (the R1 fix).

Phase I's answer is ``softmax(q·K)·V`` (``src/area.py:2161-2167``): whatever
VALUE sits behind a pooled KEY lands in the latent.  The auto default used to
be ``mean`` on 2.1, and ``mean`` pools a BLEND — the block's values are
averaged (``src/area_attn.py:268-269``) and the key is replaced by a synthetic
centroid direction that belongs to no token (``_norm_preserved_keys``,
``src/area_attn.py:309-331``).  A block that straddles a face boundary then
contributes an AVERAGE OF TWO FACES behind ONE key: the doubled / imploded
character.

``medoid`` keeps ONE REAL TOKEN's key AND value (``src/area_attn.py:200-212``),
so every pooled position carries exactly one content.

What is pinned here:

* the PROVENANCE — the medoid's pooled key and value are byte-identical to ONE
  member's key and value (the SAME member), while ``mean``'s value is the
  blend: the two clusters contribute (0.5, 0.5) on a two-vs-two block;
* the REPRESENTATIVENESS — on a three-vs-one block the medoid keeps the
  MAJORITY cluster, while ``subsample`` (the block centre) keeps the MINORITY;
* the LATTICE — the medoid must not trade the ghost for a comb: the Phase I
  output's block-explained ratio on 2.1's own continuous ids is not worse than
  ``mean``'s (measured 98.9 % for both — the block lattice is 2.1's own, and
  the medoid neither adds nor removes it).

The per-backbone auto default is pinned in ``tests/test_area_node.py``
(``TestAutoDefaults``); the pooling SEMANTICS live here.

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.area_attn import apply_rope1, pool_keys_values
from src.area_qwen21 import qwen21_coarse_key_ids, qwen21_coarse_query_ids

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# The synthetic block: ONE 2 x 2 block of a 2 x 2 canvas (f = 2)
# ---------------------------------------------------------------------------

#: The fixture's dimension.  It carries the cluster direction, the two spread
#: axes and the two cluster values — five orthogonal directions.
BLOCK_DIM = 8


def _axes() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Three orthogonal unit axes: the cluster direction ``p`` (the clusters sit
    at ``+p`` and ``-p``), the intra-cluster spread ``e`` and a second spread
    axis ``q``."""
    p = torch.zeros(BLOCK_DIM)
    p[0] = 1.0
    e = torch.zeros(BLOCK_DIM)
    e[1] = 1.0
    q = torch.zeros(BLOCK_DIM)
    q[2] = 1.0
    return p, e, q


def _cluster_values() -> tuple[torch.Tensor, torch.Tensor]:
    """The two clusters' values — ORTHOGONAL unit vectors, so a blend's
    coefficients are recovered exactly by the dot products."""
    u_a = torch.zeros(BLOCK_DIM)
    u_a[3] = 1.0
    u_b = torch.zeros(BLOCK_DIM)
    u_b[4] = 1.0
    return u_a, u_b


def _block(keys: list[torch.Tensor], values: list[torch.Tensor]):
    """``[1, 1, 4, BLOCK_DIM]`` keys/values of the ONE block of a 2 x 2 canvas
    (the member order is the grid order: row-major inside the block)."""
    return (torch.stack(keys).reshape(1, 1, 4, BLOCK_DIM),
            torch.stack(values).reshape(1, 1, 4, BLOCK_DIM))


def _two_vs_two():
    """A block with TWO clusters of two members: members 0, 1 are cluster A
    (keys around ``+p``), members 2, 3 are cluster B (keys around ``-p``).

    The spreads are asymmetric on purpose: the block MEAN is then a small
    vector along ``e``, so the ``mean`` pooling's centroid direction is
    (nearly) ORTHOGONAL to both clusters — the synthetic key points at neither
    of them, which is the measured signature of the defect.
    """
    p, e, _ = _axes()
    u_a, u_b = _cluster_values()
    keys = [p + 0.10 * e, p - 0.05 * e, -p + 0.30 * e, -p + 0.20 * e]
    values = [u_a, u_a, u_b, u_b]
    return _block(keys, values), (u_a, u_b)


def _three_vs_one():
    """A block with THREE members of cluster A and ONE of cluster B — the
    minority cluster the block centre (``subsample``) keeps."""
    p, e, q = _axes()
    u_a, u_b = _cluster_values()
    keys = [p + 0.05 * e, p - 0.05 * e, p + 0.02 * q, -p + 0.30 * e]
    values = [u_a, u_a, u_a, u_b]
    return _block(keys, values), (u_a, u_b)


def _pool(pooling: str, block) -> tuple[torch.Tensor, torch.Tensor]:
    """``pool_keys_values`` on the 2 x 2 canvas (f = 2, no caption)."""
    key, value = block
    return pool_keys_values(key, value, 0, 2, 2, 2, pooling)


# ---------------------------------------------------------------------------
# The Phase I answer on 2.1's own canvas
# ---------------------------------------------------------------------------

#: The canvas the user runs: 512 x 2000 px -> a 125 x 32 token grid (16 px per
#: token).  125 is ODD, so the pooling blocks do NOT tile it and the
#: PARTIAL-edge code path is the one the real canvas runs.
CANVAS_H, CANVAS_W, FACTOR = 125, 32, 2

#: 2.1's head dimension (``comfy/ldm/qwen_image21/model.py``).
HEAD_DIM = 128


def _rope(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """comfy's own rope (``comfy/ldm/flux/math.py:20-33``), CLONED so the test
    imports no comfy module and loads no model."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).to(torch.float32)


def _embed_nd(ids: torch.Tensor, theta: float = 10000.0,
              axes_dim: tuple[int, int, int] = (16, 56, 56)) -> torch.Tensor:
    """2.1's own embedder: ``EmbedND`` (``comfy/ldm/flux/layers.py:15-29``) with
    2.1's theta and axis split."""
    emb = torch.cat(
        [_rope(ids[..., i], axes_dim[i], theta) for i in range(ids.shape[-1])], dim=-3
    )
    return emb.unsqueeze(1)


def _canvas(seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """The surrogate canvas: smooth spatial keys and values (so the attention
    stays local) with a FACE written into the values."""
    gen = torch.Generator().manual_seed(seed)
    rows = torch.arange(CANVAS_H, dtype=torch.float32)[:, None]
    cols = torch.arange(CANVAS_W, dtype=torch.float32)[None, :]
    freq = torch.rand(8, 2, generator=gen) * 0.25 + 0.05
    basis = torch.randn(8, 16, generator=gen) / 4 ** 0.5
    key = torch.zeros(CANVAS_H, CANVAS_W, HEAD_DIM)
    value = torch.zeros(CANVAS_H, CANVAS_W, HEAD_DIM)
    for j in range(8):
        phase = freq[j, 0] * rows + freq[j, 1] * cols
        key[..., j * 16:(j + 1) * 16] = torch.sin(phase)[..., None] * basis[j]
        value[..., j * 16:(j + 1) * 16] = torch.cos(phase)[..., None] * basis[j]
    face = ((rows >= 18) & (rows <= 42) & (cols >= 6) & (cols <= 24)).float()
    face_vec = torch.randn(32, generator=gen) / 5.657
    value[..., :32] += 3.0 * face[..., None] * face_vec
    key = key / key.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    shape = (1, 1, CANVAS_H * CANVAS_W, HEAD_DIM)
    return key.reshape(shape), value.reshape(shape)


def _queries(seed: int = 11) -> torch.Tensor:
    """The model's own queries: smooth and INDEPENDENT of the keys."""
    gen = torch.Generator().manual_seed(seed)
    rows = torch.arange(CANVAS_H, dtype=torch.float32)[:, None]
    cols = torch.arange(CANVAS_W, dtype=torch.float32)[None, :]
    freq = torch.rand(8, 2, generator=gen) * 0.25 + 0.05
    basis = torch.randn(8, 16, generator=gen) / 4 ** 0.5
    query = torch.zeros(CANVAS_H, CANVAS_W, HEAD_DIM)
    for j in range(8):
        phase = freq[j, 0] * rows + freq[j, 1] * cols
        query[..., j * 16:(j + 1) * 16] = torch.sin(phase)[..., None] * basis[j]
    query = query / query.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    return query.reshape(1, 1, CANVAS_H * CANVAS_W, HEAD_DIM)


def _phase_one(pooling: str) -> torch.Tensor:
    """2.1's Phase I on the surrogate canvas: the pooled keys and values, the
    model's own CONTINUOUS coarse ids and the attention that produces the
    answer (``src/area.py:2161-2167``)."""
    key, value = _canvas()
    query = _queries()
    k_pooled, v_pooled = pool_keys_values(
        key, value, 0, CANVAS_H, CANVAS_W, FACTOR, pooling
    )
    rope_q = _embed_nd(qwen21_coarse_query_ids(0, CANVAS_H, CANVAS_W, FACTOR))
    rope_k = _embed_nd(qwen21_coarse_key_ids(0, 0, CANVAS_H, CANVAS_W, FACTOR))
    return F.scaled_dot_product_attention(
        apply_rope1(query, rope_q), apply_rope1(k_pooled, rope_k), v_pooled
    )


def _block_explained(out: torch.Tensor) -> float:
    """The share of the Phase I output's variance that sits BETWEEN the ``f x
    f`` blocks.  A piecewise-constant field — the block comb — scores 1.0; a
    field that varies inside every block scores 0.0."""
    field = out[0, 0].reshape(CANVAS_H, CANVAS_W, HEAD_DIM)
    nh, nw = -(-CANVAS_H // FACTOR), -(-CANVAS_W // FACTOR)
    means, within = [], []
    for i in range(nh):
        for j in range(nw):
            block = field[i * FACTOR:min(CANVAS_H, (i + 1) * FACTOR),
                          j * FACTOR:min(CANVAS_W, (j + 1) * FACTOR)]
            mean = block.mean(dim=(0, 1))
            means.append(mean)
            within.append(((block - mean) ** 2).sum(-1).sqrt().mean())
    means = torch.stack(means)
    within = torch.stack(within).mean()
    across = ((means - means.mean(0)) ** 2).sum(-1).sqrt().mean()
    return float(across ** 2 / (across ** 2 + within ** 2))


# ---------------------------------------------------------------------------
# The pins
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestProvenance:
    """Phase I's answer is ``softmax(q·K)·V``: the VALUE behind the pooled KEY
    is what lands in the latent."""

    def test_the_medoid_keeps_ONE_real_tokens_key_and_value(self):
        """The medoid's pooled key and value are ONE member's — byte for byte.

        This is the whole fix: a pooled position carries exactly ONE content,
        so a block that straddles a face boundary cannot average two faces
        into one.
        """
        block, _ = _two_vs_two()
        k_pooled, v_pooled = _pool("medoid", block)
        members = [
            i for i in range(4) if torch.equal(k_pooled[0, 0, 0], block[0][0, 0, i])
        ]
        assert len(members) == 1, "the pooled key must be exactly one member's key"
        assert torch.equal(v_pooled[0, 0, 0], block[1][0, 0, members[0]]), \
            "the pooled value must belong to the SAME member as the pooled key"

    def test_the_mean_pooling_BLENDS_the_values(self):
        """``mean`` is the blend 2.1 must not ship: on a two-vs-two block the
        two clusters contribute (0.5, 0.5) of the pooled value."""
        block, (u_a, u_b) = _two_vs_two()
        _, v_pooled = _pool("mean", block)
        assert float(v_pooled[0, 0, 0] @ u_a) == pytest.approx(0.5, abs=1e-6)
        assert float(v_pooled[0, 0, 0] @ u_b) == pytest.approx(0.5, abs=1e-6)

    def test_the_mean_pooling_SYNTHESISES_the_key(self):
        """The ``mean`` key belongs to NO token and points at NEITHER cluster."""
        block, _ = _two_vs_two()
        k_pooled, _ = _pool("mean", block)
        assert not any(
            torch.equal(k_pooled[0, 0, 0], block[0][0, 0, i]) for i in range(4)
        ), "the centroid key must not be a member's key"
        cos = float(F.cosine_similarity(k_pooled[0, 0, 0], block[0][0, 0, 0], dim=0))
        assert abs(cos) < 0.5, f"the centroid key points at cluster A (cos {cos:.3f})"


@pytest.mark.unit
class TestRepresentativeness:
    """Which member represents the block."""

    def test_the_medoid_keeps_the_MAJORITY_cluster(self):
        """Three members of A and one of B: the medoid keeps A — the
        representative — with BOTH its key and its value."""
        block, (u_a, _) = _three_vs_one()
        k_pooled, v_pooled = _pool("medoid", block)
        assert torch.equal(v_pooled[0, 0, 0], u_a), "the medoid must keep the majority"
        assert torch.equal(k_pooled[0, 0, 0], block[0][0, 0, 0])

    def test_the_block_centre_keeps_the_MINORITY_cluster(self):
        """``subsample`` is a real token too, but not a REPRESENTATIVE one: the
        block centre is the minority here — which is why 2.1 does not run it."""
        block, (_, u_b) = _three_vs_one()
        k_pooled, v_pooled = _pool("subsample", block)
        assert torch.equal(v_pooled[0, 0, 0], u_b), "the block centre is the minority here"
        assert torch.equal(k_pooled[0, 0, 0], block[0][0, 0, 3])

    def test_the_mean_pooling_blends_three_to_one(self):
        """On the same block ``mean`` still blends: (0.75, 0.25)."""
        block, (u_a, u_b) = _three_vs_one()
        _, v_pooled = _pool("mean", block)
        assert float(v_pooled[0, 0, 0] @ u_a) == pytest.approx(0.75, abs=1e-6)
        assert float(v_pooled[0, 0, 0] @ u_b) == pytest.approx(0.25, abs=1e-6)


@pytest.mark.unit
class TestLattice:
    """The fix must not trade the ghost for a comb (the block lattice is 2.1's
    own: it is 99 % between-block for EVERY pooling on the continuous ids)."""

    def test_the_medoid_does_not_make_the_block_lattice_worse(self):
        """The medoid's Phase I output is no MORE piecewise-constant than
        ``mean``'s — so the fix removes the blended ghost without adding a
        comb."""
        mean = _block_explained(_phase_one("mean"))
        medoid = _block_explained(_phase_one("medoid"))
        assert medoid <= mean + 0.02, (
            f"the medoid makes the field more piecewise-constant: "
            f"{medoid:.3f} vs {mean:.3f}"
        )
        assert 0.5 <= medoid <= 1.0, f"the metric is degenerate ({medoid:.3f})"
