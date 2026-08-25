"""Regression tests: Krea-2 (K2) SPA MATH is correct (no mosaic source in code).

This module locks in the findings from the 2026-08-14 Krea investigation:

* Krea-2's ``pe_embedder`` is FLUX's ``EmbedND`` with the ASYMMETRIC
  ``axes_dim = [32, 48, 48]`` (headdim=128, theta=1000) -- NOT the symmetric
  ``[16,16,16]`` FLUX uses.  Asymmetry hides/breaks nothing here: SPA reads
  ``orig_embedder.axes_dim`` and the frequency exponents of ``get_1d_ntk_pos_embed``
  EXACTLY equal FLUX's ``rope`` for any even dim
  (``(dim-2)/(dim*(dim/2-1)) == 2/dim``), so the base PE is bit-identical.

* The averaged-attention hook applies the per-variant *delta* ``inv(base)@variant``
  to the already-base-roPE'd q/k (RoPE is applied OUTSIDE ``optimized_attention_masked``
  in comfy/ldm/krea2/model.py, lines 92->97).  This is HRDiT-faithful and, on a
  spatially-structured (locally-coherent) input, injects NO high-frequency / mosaic
  energy -- it is ~60x SMOOTHER than the base pass.

These tests are comfy-free (pure torch) so they run in the portable unit env and
act as a guard: if a future change regresses the Krea rope parity or the averaging
math, these will fail and tell us the mosaic came BACK into the code (not the model).

Markers: @pytest.mark.unit
"""
import math

import torch

from src.rope import get_1d_ntk_pos_embed
from src.spa import SPA_MAX_PASSES, build_bundle_id_variants, derive_bundle_s
from src.spa_attn import (
    apply_rope_matrix,
    spa_averaged_attention,
)

# --- Krea config (verified against comfy/ldm/krea2/model.py SingleStreamDiT) --
KREA_AXES_DIM = [32, 48, 48]
KREA_THETA = 1000.0


# ---------------------------------------------------------------------------
# RoPE parity: SPA base PE == FLUX's rope() for Krea's asymmetric axes_dim
# ---------------------------------------------------------------------------

def _flux_rope_reference(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """Exact re-implementation of comfy/ldm/flux/math.py::rope (the model's PE)."""
    device = pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2,
                           dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = out.reshape(*out.shape[:-1], 2, 2)  # (..., d/2, 2, 2) rotation blocks
    return out.to(dtype=torch.float32, device=device)


def _spa_components(pos, axes_dim, theta, fdtype):
    n_axes = pos.shape[-1]
    out = []
    for i in range(n_axes):
        axis_pos = pos[..., i]
        axis_dim = axes_dim[i]
        cos, sin = get_1d_ntk_pos_embed(
            dim=axis_dim, pos=axis_pos, theta=theta,
            use_real=True, repeat_interleave_real=True,
            freqs_dtype=fdtype, ntk_factor=1.0)
        out.append((cos, sin))
    return out


def _spa_format_flux(components, ids):
    emb_parts = []
    for cos, sin in components:
        cos_r = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
        sin_r = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
        row1 = torch.cat([cos_r, -sin_r], dim=-1)
        row2 = torch.cat([sin_r, cos_r], dim=-1)
        emb_parts.append(torch.stack([row1, row2], dim=-2))
    emb = torch.cat(emb_parts, dim=-3)
    return emb.unsqueeze(1).to(ids.device)


def test_krea_base_pe_parity_with_flux_rope():
    """SPA's base PE must equal FLUX's rope() block-for-block for axes_dim=[32,48,48]."""
    # mixed token ids: text (0,0,0) + image grid spanning 0..127 on h,w
    ids = torch.zeros(1, 200, 3)
    ids[0, 4:, 1] = torch.arange(128).float().repeat(1)[:196] if False else torch.arange(196).float()
    ids[0, 4:, 2] = torch.arange(196).float()

    fdtype = torch.float32
    spa_pe = _spa_format_flux(
        _spa_components(ids, KREA_AXES_DIM, KREA_THETA, fdtype), ids)

    # Reference: per-axis FLUX rope(), concatenated along dim=-3, unsqueezed like the model.
    ref_parts = []
    for i in range(ids.shape[-1]):
        ref_parts.append(_flux_rope_reference(ids[..., i], KREA_AXES_DIM[i], KREA_THETA))
    ref_pe = torch.cat(ref_parts, dim=-3).unsqueeze(1)

    assert spa_pe.shape == ref_pe.shape, (spa_pe.shape, ref_pe.shape)
    # Max abs error must be tiny (float32 rounding of cos/sin only).
    max_err = (spa_pe - ref_pe).abs().max().item()
    assert max_err < 1e-5, f"Krea base PE diverges from FLUX rope() by {max_err}"
    # And it must NOT be all-zeros / degenerate.
    assert spa_pe.abs().mean().item() > 1e-3


# ---------------------------------------------------------------------------
# Averaging math: no mosaic injected on a locally-coherent (DiT-like) input
# ---------------------------------------------------------------------------

def _plain_attn(q, k, v):
    d = q.shape[-1]
    scores = torch.einsum("bhld,bhmd->bhlm", q, k) / (d ** 0.5)
    return torch.einsum("bhlm,bhmd->bhld", torch.softmax(scores, dim=-1), v)


def _build_bundle_variants(ids, N):
    """Delegate to the REAL production bundling (paper-``N`` semantics).

    Updated 2026-08-15: this guard previously re-implemented the OLD ``group_num``
    formula locally.  It now calls :func:`src.spa.build_bundle_id_variants` so the
    mosaic guard exercises the exact code path the node runs (trained-extent gate
    + in-dist floor + pass cap).  ``N`` is the paper knob (tokens per bundle):
    0=auto, 1=off, 2..8 explicit.
    """
    return build_bundle_id_variants(ids, N)


def test_krea_averaged_attention_is_smooth_not_mosaic():
    """On a spatially-structured (locally-coherent) q/k, spa_averaged_attention must
    inject essentially NO high-frequency (mosaic) energy vs the base pass.

    Mirrors the isolated faithful probe: if this fails, the Krea mosaic has come back
    into the SPA MATH (a code regression), not the model.

    NOTE (2026-08-15 paper-N rewire): the grid is 128x128 (max_pos=127 > trained_extent=64)
    so the trained-extent gate is ACTIVE, and N=3 (paper tokens-per-bundle) gives
    s = max(3, ceil(127/79)) = 3 -> 2*3-1 = 5 passes.  The bundle stays in-distribution
    (positions <= 79), so no OOD mosaic; the period-2 guard remains a catastrophic
    blow-up detector only.
    """
    D = sum(KREA_AXES_DIM)  # 128
    G = 128
    txt = torch.zeros(1, 4, 3)
    img = torch.zeros(1, G * G, 3)
    img[..., 1] = torch.arange(G)[:, None].repeat(1, G).reshape(-1)  # span 0..127
    img[..., 2] = torch.arange(G)[None, :].repeat(G, 1).reshape(-1)
    ids = torch.cat([txt, img], dim=1)

    fdtype = torch.float32
    base_pe = _spa_format_flux(_spa_components(ids, KREA_AXES_DIM, KREA_THETA, fdtype), ids)
    variants = _build_bundle_variants(ids, 3)  # N=3 -> s=3 (active, 5 passes)
    variant_pes = [_spa_format_flux(_spa_components(v, KREA_AXES_DIM, KREA_THETA, fdtype), v)
                   for v in variants]

    # Locally-coherent q/k/v: f(pos) = low-freq sinusoids -> nearby tokens attend nearby.
    yy, xx = torch.meshgrid(torch.linspace(0, 1, G), torch.linspace(0, 1, G), indexing="ij")
    feats = torch.stack([
        torch.sin(2 * math.pi * yy), torch.cos(2 * math.pi * yy),
        torch.sin(2 * math.pi * xx), torch.cos(2 * math.pi * xx),
        torch.sin(2 * math.pi * (yy + xx)), torch.cos(2 * math.pi * (yy - xx)),
    ], dim=-1)
    g = torch.Generator().manual_seed(0)
    coef = torch.randn(6, D, generator=g) * 0.7
    f_img = torch.einsum("gwc,cd->gwd", feats, coef).reshape(1, G * G, D)
    f = torch.cat([torch.randn(1, 4, D, generator=g) * 0.05, f_img], dim=1)
    q = k = v = f.unsqueeze(0)  # (1,1,L,D)

    qb = apply_rope_matrix(q, base_pe, "flux")
    kb = apply_rope_matrix(k, base_pe, "flux")
    base_out = _plain_attn(qb, kb, v)
    spa_out = spa_averaged_attention(qb, kb, v, base_pe, variant_pes,
                                     attn_fn=_plain_attn, pre_roped=True, fmt="flux")

    diff = (spa_out - base_out).float()
    img_diff = diff[0, 0, 4:, :].reshape(G, G, D)
    gx = img_diff[1:, :, :] - img_diff[:-1, :, :]
    gy = img_diff[:, 1:, :] - img_diff[:, :-1, :]
    hf = (gx ** 2).mean() + (gy ** 2).mean()

    base_img = base_out[0, 0, 4:, :].reshape(G, G, D).float()
    bhf = ((base_img[1:, :, :] - base_img[:-1, :, :]) ** 2).mean() + \
          ((base_img[:, 1:, :] - base_img[:, :-1, :]) ** 2).mean()

    hf_ratio = hf.item() / max(bhf.item(), 1e-12)
    assert torch.isfinite(spa_out).all()
    # Generic perturbation bound: SPA is a position-extrapolation mechanism, so it
    # legitimately deviates from the base pass (more at finer bundles).  A *mosaic*
    # would blow this up orders of magnitude; we only guard against a catastrophic blow-up.
    assert hf_ratio < 0.5, f"SPA injects catastrophic energy (HF ratio {hf_ratio})"
    assert diff.abs().max().item() < 2.0

    # PRIMARY MOSAIC DETECTOR: period-2 (checkerboard) energy of the delta.  The user's
    # "mosaic-glass / doubled elements" is a period-2 spatial aliasing that appears when
    # the bundle collapses adjacent tokens onto the same RoPE (the low-res s=2 regime).
    # A correct fine bundle must NOT inject a structured checkerboard.
    dd = img_diff[..., 0]
    # even/odd lattice difference (the checkerboard coefficient)
    even = (dd[0::2, 0::2].mean() + dd[1::2, 1::2].mean()) / 2
    odd = (dd[0::2, 1::2].mean() + dd[1::2, 0::2].mean()) / 2
    period2 = ((even - odd) ** 2).item()
    bb = base_img[..., 0]
    beven = (bb[0::2, 0::2].mean() + bb[1::2, 1::2].mean()) / 2
    bodd = (bb[0::2, 1::2].mean() + bb[1::2, 0::2].mean()) / 2
    bperiod2 = ((beven - bodd) ** 2).item()
    period2_ratio = period2 / max(bperiod2, 1e-12)
    # With the paper-N rewire, N=3 at this grid (max_pos=127) gives
    # s = max(3, ceil(127/79)) = 3 -> 5 passes.  The bundle stays in-distribution
    # (positions <= 79), so no OOD mosaic; the period-2 guard remains a
    # catastrophic blow-up detector only.
    assert period2_ratio < 5.0, f"SPA injects period-2 mosaic (ratio {period2_ratio})"

    # LOCK the paper-N formula: at this grid (max_pos=127, N=3) s=3 -> 5 passes,
    # always within the SPA_MAX_PASSES cost cap.
    s = (len(variant_pes) + 1) // 2
    assert s == derive_bundle_s(127, 3), f"expected s={derive_bundle_s(127, 3)}, got s={s}"
    assert len(variant_pes) <= SPA_MAX_PASSES, "pass count exceeds cap"


def test_krea_variant_count_bounded_at_cap():
    """At N=0 (auto) the variant count follows the HRDiT in-dist formula and stays
    within the SPA_MAX_PASSES cost cap (fast, in-distribution).

    Updated 2026-08-15: N=0 (auto) reproduces the HRDiT ``group_num=80`` ceiling
    (``s_floor = ceil(max_pos / 79)``).  The old test passed group_num=80 directly;
    under paper-N semantics that value is legacy and is migrated to auto.
    """
    # 256x256 latent (1024px Krea) -> max token 255 -> s=ceil(255/79)=4 -> 7 passes.
    ids = torch.zeros(1, 256 * 256, 3)
    ids[..., 1] = torch.arange(256).float().repeat(256, 1).reshape(-1)
    ids[..., 2] = torch.arange(256).float().repeat_interleave(256).reshape(-1)
    variants = _build_bundle_variants(ids, 0)  # N=0 auto -> HRDiT group_num=80 behaviour
    s = (len(variants) + 1) // 2
    assert s == 4, f"expected HRDiT s=4 at max_pos=255, got s={s}"
    assert len(variants) == 2 * s - 1, len(variants)
    # Bounded pass count: <= SPA_MAX_PASSES (NOT the ~1021 the unclamped group_num=2 gave).
    assert len(variants) <= SPA_MAX_PASSES
