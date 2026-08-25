"""Shared numeric helpers for SPA tests (pure torch, no ComfyUI).

Contains:
  * rotation-matrix <-> angle utilities, and ``random_variants`` (P0),
  * ``apply_rotary_emb`` + ``reference_spa_attention`` copied *verbatim* from
    HRDiT ``hrdit/attention.py`` (lines 127-196) for exact equivalence (P1),
  * the local-coherence metric used by the mosaic regression test (P4).

These bridge the two RoPE representations: our attention core consumes rotation
matrices, while the HRDiT reference consumes ``(cos, sin)`` pairs.  Both are built
from the *same* per-pair angle vectors, so the bridge is exact.
"""
import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Angle <-> rotation matrix / (cos, sin) utilities
# ---------------------------------------------------------------------------

def angles_to_blocks(angles: torch.Tensor) -> torch.Tensor:
    """Build per-pair 2x2 rotation blocks from a ``(..., P)`` angle tensor.

    Returns ``(..., P, 2, 2)`` with block ``[[cos, -sin], [sin, cos]]``.
    """
    c = torch.cos(angles).unsqueeze(-1)
    s = torch.sin(angles).unsqueeze(-1)
    row1 = torch.cat([c, -s], dim=-1)
    row2 = torch.cat([s, c], dim=-1)
    return torch.stack([row1, row2], dim=-2)


def angles_to_cos_sin(angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """HRDiT-style *real-interleaved* ``(cos, sin)`` from a ``(..., P)`` angle tensor.

    Returns ``(cos, sin)`` each of shape ``(..., 2P)`` (``repeat_interleave(2)``),
    matching ``get_1d_ntk_pos_embed(use_real=True, repeat_interleave_real=True)``.
    """
    c = torch.cos(angles)
    s = torch.sin(angles)
    cos = c.repeat_interleave(2, dim=-1)
    sin = s.repeat_interleave(2, dim=-1)
    return cos, sin


def random_variants(L: int, D: int, N: int, seed: int) -> List[torch.Tensor]:
    """Reproducible random RoPE rotation-matrix stacks for P0 characterization.

    Returns ``N`` tensors of shape ``(L, D // 2, 2, 2)``; each is a stack of
    independent random 2x2 rotation blocks (one per head-pair and token).  The
    seed makes the construction fully deterministic.
    """
    g = torch.Generator().manual_seed(seed)
    P = D // 2
    out = []
    for _ in range(N):
        angles = torch.randn(L, P, generator=g) * 0.5
        out.append(angles_to_blocks(angles))
    return out


# ---------------------------------------------------------------------------
# HRDiT reference implementation (verbatim from hrdit/attention.py 127-196)
# ---------------------------------------------------------------------------

def apply_rotary_emb(x, freqs_cis, use_real: bool = True, use_real_unbind_dim: int = -1):
    """HRDiT ``apply_rotary_emb`` — copied verbatim for exact reference (P1/T-P1-7)."""
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]  # equal to .unsqueeze(0).unsqueeze(0)
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)
        return x_out.type_as(x)


def reference_spa_attention(query, key, value, rope_variants, attention_scale=1.0):
    """HRDiT ``_spa_attention`` — copied verbatim (mean of N attention outputs).

    ``query, key, value`` : ``(B, H, S, D)``.  ``rope_variants`` : list of
    ``(cos, sin)`` tuples each ``(S, D)`` (real-interleaved).  Faithful reference
    for the SPA fix.
    """
    acc = None
    for rope_v in rope_variants:
        q_v = apply_rotary_emb(query, rope_v)
        k_v = apply_rotary_emb(key, rope_v)
        out = F.scaled_dot_product_attention(
            q_v, k_v, value, dropout_p=0.0, is_causal=False, scale=attention_scale
        )
        acc = out if acc is None else acc + out
    return acc / len(rope_variants)


# ---------------------------------------------------------------------------
# Local-coherence metric (P4 mosaic regression)
# ---------------------------------------------------------------------------

def smooth_qk(H: int, W: int, D: int, seed: int, device="cpu") -> torch.Tensor:
    """Smooth low-frequency 2D-sin field -> per-token Q/K vectors ``(1, 1, L, D)``.

    Adjacent tokens get similar vectors so a *correct* attention map is locally
    coherent; a corrupted (buggy) map scatters and the metric drops.
    """
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(H, device=device).float()
    j = torch.arange(W, device=device).float()
    # two low spatial frequencies -> smooth field
    field = (
        torch.sin(0.35 * i[:, None] + 0.20 * j[None, :])
        + 0.5 * torch.sin(0.15 * i[:, None] - 0.10 * j[None, :])
    )  # (H, W)
    field = field - field.mean()
    L = H * W
    vec = field.reshape(L).unsqueeze(-1).repeat(1, D)  # (L, D)
    vec = vec + 0.01 * torch.randn(L, D, generator=g)
    return vec.unsqueeze(0).unsqueeze(0)  # (1, 1, L, D)


def grid_adjacency(H: int, W: int, device="cpu") -> torch.Tensor:
    """4-neighbourhood adjacency (right, down) over an ``H x W`` grid -> ``(E, 2)``."""
    idx = torch.arange(H * W, device=device).reshape(H, W)
    edges = []
    # right
    if W > 1:
        edges.append(torch.stack([idx[:, :-1].reshape(-1), idx[:, 1:].reshape(-1)], dim=-1))
    # down
    if H > 1:
        edges.append(torch.stack([idx[:-1, :].reshape(-1), idx[1:, :].reshape(-1)], dim=-1))
    return torch.cat(edges, dim=0)


def attention_weights(q, k, scale: float):
    """Row-normalised attention weight matrix ``W`` from ``q, k`` (``(..., L, D)``)."""
    logits = (q @ k.transpose(-1, -2)) / scale
    return torch.softmax(logits, dim=-1)


def local_coherence(W: torch.Tensor, adjacency: torch.Tensor) -> float:
    """Cosine similarity of attention rows across adjacent tokens, averaged.

    ``W`` : ``(B, H, L, L)`` (rows sum to 1).  ``adjacency`` : ``(E, 2)``.
    """
    rows = W[..., adjacency[:, 0], :]
    cols = W[..., adjacency[:, 1], :]
    num = (rows * cols).sum(-1)
    den = rows.norm(dim=-1) * cols.norm(dim=-1) + 1e-9
    sim = num / den
    return float(sim.mean().item())


# ---------------------------------------------------------------------------
# Krea-2 PE construction (shared by the ripple detector + math locks)
# ---------------------------------------------------------------------------

# Krea-2 (SingleStreamDiT) RoPE config, verified against comfy/ldm/krea2/model.py.
KREA_AXES_DIM = [32, 48, 48]
KREA_THETA = 1000.0


def krea_components(pos: torch.Tensor, fdtype=torch.float32):
    """Per-axis ``(cos, sin)`` for Krea-2's asymmetric axes (pure, ntk_factor=1).

    Mirrors ``SPABasePosEmbed._spa_components`` so the ripple detector exercises
    the SAME PE math the node uses in production.
    """
    from src.rope import get_1d_ntk_pos_embed

    out = []
    for i in range(pos.shape[-1]):
        cos, sin = get_1d_ntk_pos_embed(
            dim=KREA_AXES_DIM[i],
            pos=pos[..., i],
            theta=KREA_THETA,
            use_real=True,
            repeat_interleave_real=True,
            freqs_dtype=fdtype,
            ntk_factor=1.0,
        )
        out.append((cos, sin))
    return out


def krea_format_flux(components, ids: torch.Tensor) -> torch.Tensor:
    """FLUX-layout rotation blocks ``(1, 1, L, D//2, 2, 2)`` from ``(cos, sin)``.

    Mirrors ``PosEmbedFlux.format_components`` (real-interleaved pairs).
    """
    parts = []
    for cos, sin in components:
        cos_r = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
        sin_r = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
        row1 = torch.cat([cos_r, -sin_r], dim=-1)
        row2 = torch.cat([sin_r, cos_r], dim=-1)
        parts.append(torch.stack([row1, row2], dim=-2))
    return torch.cat(parts, dim=-3).unsqueeze(1).to(ids.device)


def krea_pe(ids: torch.Tensor) -> torch.Tensor:
    """Full Krea-2 base RoPE blocks for a ``(B, L, 3)`` position-id tensor."""
    return krea_format_flux(krea_components(ids.float()), ids)


# ---------------------------------------------------------------------------
# Period-s ripple detector (T0.4 calibration / P4 quality gate)
# ---------------------------------------------------------------------------

def structured_qkv(H: int, W: int, D: int, seed: int, device="cpu"):
    """Spatially-structured q/k/v for the ripple detector.

    q/k = low-frequency sinusoidal surface + controlled mid-band components (so
    energy exists near the bundle frequency and a period-``s`` band CAN be
    excited); v = smooth low-frequency grid (DiT-like local coherence).  Returns
    ``(q, k, v)`` each ``(1, 1, H*W, D)``.  Fully deterministic via ``seed``.

    NOTE: the 2026-08-14 isolation probes used ONLY the smooth low-frequency part,
    which cannot excite the bundle band — that is why the math "looked clean"
    while real outputs rippled.  The mid-band terms are the calibration fix.
    """
    g = torch.Generator().manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing="ij",
    )
    feats = torch.stack([
        # low-frequency surface (smooth content)
        torch.sin(2 * math.pi * yy), torch.cos(2 * math.pi * yy),
        torch.sin(2 * math.pi * xx), torch.cos(2 * math.pi * xx),
        torch.sin(2 * math.pi * (yy + xx)), torch.cos(2 * math.pi * (yy - xx)),
        # controlled mid-band components (excite the bundle band)
        torch.sin(2 * math.pi * 4 * yy), torch.cos(2 * math.pi * 4 * xx),
        torch.sin(2 * math.pi * 5 * (yy + xx) / 2),
    ], dim=-1)  # (H, W, 9)
    coef = torch.randn(feats.shape[-1], D, generator=g) * 0.7
    img = torch.einsum("gwc,cd->gwd", feats, coef).reshape(H * W, D)
    q = img.unsqueeze(0).unsqueeze(0)
    k = img.unsqueeze(0).unsqueeze(0)
    # v: smooth low-frequency grid (local coherence, no mid-band energy)
    vfeats = torch.stack([
        torch.sin(2 * math.pi * yy), torch.cos(2 * math.pi * xx),
    ], dim=-1)
    vcoef = torch.randn(2, D, generator=g) * 0.7
    vimg = torch.einsum("gwc,cd->gwd", vfeats, vcoef).reshape(H * W, D)
    v = vimg.unsqueeze(0).unsqueeze(0)
    return q, k, v


def _ring_mask(H: int, W: int, s: int, device="cpu") -> torch.Tensor:
    """Boolean FFT-bin mask for the ring at spatial period ``s ± 1`` (cycles/px)."""
    ky = torch.fft.fftfreq(H, device=device)
    kx = torch.fft.fftfreq(W, device=device)
    r = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    if s <= 1:
        return torch.zeros(H, W, dtype=torch.bool, device=device)
    lo, hi = 1.0 / (s + 1), 1.0 / (s - 1)
    ring = (r >= lo) & (r <= hi)
    ring[0, 0] = False  # never count DC
    return ring


def period_band_power(delta: torch.Tensor, H: int, W: int, s: int,
                      max_channels: int = 16) -> float:
    """Fraction of total AC power in the period-``s ± 1`` ring of ``delta``.

    ``delta`` : ``(H*W, D)`` or ``(H, W, D)`` — the ``(SPA − base)`` output delta
    over the image tokens.  The power spectrum is computed per channel (first
    ``max_channels``) and the per-channel fractions are averaged, so a ripple that
    is strong in some channels and weak in others is still detected.
    """
    if delta.dim() == 2:
        delta = delta.reshape(H, W, -1)
    x = delta.float()
    C = min(max_channels, x.shape[-1])
    ring = _ring_mask(H, W, s, device=x.device)
    fracs = []
    for c in range(C):
        power = torch.fft.fft2(x[..., c]).abs() ** 2
        total = power.sum() - power[0, 0]
        if float(total) <= 0:
            fracs.append(0.0)
            continue
        fracs.append(float(power[ring].sum() / total))
    return float(sum(fracs) / len(fracs))


def period_band_density(delta: torch.Tensor, H: int, W: int, s: int,
                        max_channels: int = 16) -> float:
    """Mean power PER BIN in the period-``s ± 1`` ring (density, not fraction).

    Density normalises for ring size (number of bins), which makes different
    periods comparable — a wide ring (e.g. period 2) collects more total power
    simply because it contains more bins.
    """
    if delta.dim() == 2:
        delta = delta.reshape(H, W, -1)
    x = delta.float()
    C = min(max_channels, x.shape[-1])
    ring = _ring_mask(H, W, s, device=x.device)
    n_bins = int(ring.sum())
    if n_bins == 0:
        return 0.0
    dens = []
    for c in range(C):
        power = torch.fft.fft2(x[..., c]).abs() ** 2
        dens.append(float(power[ring].mean()))
    return float(sum(dens) / len(dens))


def band_dominance(delta: torch.Tensor, H: int, W: int, s: int,
                   other_periods=(2, 3, 4, 5, 6, 7, 10, 12, 16),
                   max_channels: int = 16) -> float:
    """Period-``s`` band density divided by the mean density of ``other_periods``.

    Values ``>> 1`` mean the delta's energy concentrates at period ``s`` (the
    ripple signature); values near 1 mean no structured band (clean).
    """
    d_s = period_band_density(delta, H, W, s, max_channels)
    others = [
        period_band_density(delta, H, W, p, max_channels)
        for p in other_periods if p != s and p > 1
    ]
    mean_other = sum(others) / max(len(others), 1)
    return d_s / max(mean_other, 1e-30)


def period_peak_ratio(delta: torch.Tensor, H: int, W: int, s: int,
                      max_channels: int = 16) -> float:
    """Detrended spectral-peak detector for a period-``s`` ripple.

    The ``(SPA − base)`` delta is inherently smooth (attention outputs are
    smooth), so its raw power spectrum decays ~1/f with frequency and a naive
    band fraction/density is confounded by that trend.  This detector removes
    the trend: it bins the 2D power spectrum by radial frequency, takes a
    running-median of the radial profile (the smooth 1/f trend), and returns
    the mean power in the period-``s ± 1`` ring DIVIDED by the trend power at
    that radius.  Values near 1.0 mean the ring sits on the smooth trend (no
    structured ripple); values ``>> 1`` mean a genuine peak at period ``s``.
    """
    if delta.dim() == 2:
        delta = delta.reshape(H, W, -1)
    x = delta.float()
    C = min(max_channels, x.shape[-1])
    ky = torch.fft.fftfreq(H, device=x.device)
    kx = torch.fft.fftfreq(W, device=x.device)
    r = torch.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    rmax = float(r.max())
    n_bins = max(H, W) // 2
    edges = torch.linspace(0.0, rmax * 1.0001, n_bins + 1, device=x.device)
    ring = _ring_mask(H, W, s, device=x.device)
    if int(ring.sum()) == 0:
        return 1.0
    ratios = []
    for c in range(C):
        power = torch.fft.fft2(x[..., c]).abs() ** 2
        power = power.clone()
        power[0, 0] = 0.0  # exclude DC from trend + ring
        # radial profile (mean power per radial bin) = the smooth 1/f trend
        prof = torch.zeros(n_bins, device=x.device)
        for b in range(n_bins):
            m = (r >= edges[b]) & (r < edges[b + 1])
            if int(m.sum()) > 0:
                prof[b] = power[m].mean()
        # running median (window 5) -> robust trend estimate
        pad = 2
        pe = torch.cat([prof[:1].repeat(pad), prof, prof[-1:].repeat(pad)])
        med = pe.unfold(0, 2 * pad + 1, 1).median(dim=-1).values
        # trend value at each ring bin (interpolated by radial bin index)
        idx = torch.bucketize(r[ring], edges) - 1
        idx = idx.clamp(0, n_bins - 1)
        trend = med[idx]
        ring_power = power[ring]
        denom = trend.clamp(min=1e-30)
        ratios.append(float((ring_power / denom).mean()))
    return float(sum(ratios) / len(ratios))
