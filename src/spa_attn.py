"""SPA delta-RoPE composition + averaged-attention core.

Pure-torch, backend-agnostic.  Faithful to HRDiT
``hrdit/attention.py::_spa_attention``: instead of averaging the per-variant RoPE
*rotation matrices* and running a single softmax, we run ``N`` attention passes —
one per bundled RoPE variant — and average the **attention outputs**
(``mean_n softmax(A_n) @ V == mean_n(softmax(A_n) @ V)`` because ``V`` is shared).

Tensor formats (``fmt``), see remediation decision 5:
  * ``"flux"``     : rotation matrices ``(..., L, D//2, 2, 2)``  (FLUX / Qwen / Z-Image)
  * ``"anima"``    : rotation matrices ``(L, D//2, 2, 2)`` (no batch / head dims)
  * ``"nunchaku"`` : packed ``(..., L, D//2, 1, 2)`` ``[sin, cos]`` vector layout
"""
import torch


def _nunchaku_to_blocks(pe: torch.Tensor) -> torch.Tensor:
    """Convert Nunchaku ``[sin, cos]`` packed layout to 2x2 rotation blocks.

    ``pe`` : ``(..., L, P, 1, 2)`` (last dim = ``[sin, cos]`` packed with a
    structural singleton).  Returns ``(..., L, P, 2, 2)`` rotation blocks
    compatible with :func:`apply_rope_matrix`'s ``"...lpij,...lpj->...lpi"`` einsum.
    """
    s = pe[..., 0]
    c = pe[..., 1]
    row0 = torch.stack([c, -s], dim=-1)  # (..., L, P, 1, 2)
    row1 = torch.stack([s, c], dim=-1)    # (..., L, P, 1, 2)
    # Stack the two rows along the structural singleton axis -> (..., L, P, 2, 2).
    return torch.cat([row0, row1], dim=-2)


def _blocks_to_nunchaku(blocks: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_nunchaku_to_blocks`."""
    s = blocks[..., 1, 0]
    c = blocks[..., 0, 0]
    pe = torch.stack([s, c], dim=-1)
    return pe.unsqueeze(-2)


def apply_rope_matrix(x: torch.Tensor, R: torch.Tensor, fmt: str = "flux") -> torch.Tensor:
    """Apply rotation matrices ``R`` to the last dim of ``x``.

    ``x`` : ``(..., L, D)`` (``D`` even).  ``R`` : ``(..., L, P, 2, 2)`` where
    ``P = D // 2``.  Leading dims of ``R`` broadcast over leading dims of ``x``
    (e.g. the FLUX head-shared ``R`` has shape ``(B, 1, L, P, 2, 2)`` while ``x``
    is ``(B, H, L, D)``).  For ``fmt="nunchaku"`` the packed ``[sin, cos]`` layout
    is normalised to 2x2 blocks first.
    """
    if fmt == "nunchaku":
        R = _nunchaku_to_blocks(R)
    D = x.shape[-1]
    P = D // 2
    xr = x.reshape(*x.shape[:-1], P, 2)
    # The rotation blocks are built in fp32/bf16 for numerically stable cos/sin
    # values, but the model runs attention in a different (often fp16/Half) dtype.
    # The contraction must stay in the activations' dtype to remain contiguous with
    # the surrounding attention; the precision-sensitive delta composition
    # (inv(base) @ variant) already ran upstream in the PE's own dtype.
    if R.dtype != xr.dtype:
        R = R.to(xr.dtype)
    out = torch.einsum("...lpij,...lpj->...lpi", R, xr)
    return out.reshape(*x.shape[:-1], D)


def inv_rope(pe: torch.Tensor, fmt: str = "flux") -> torch.Tensor:
    """Inverse of a stack of 2x2 rotation matrices (transpose each block)."""
    if fmt == "nunchaku":
        blocks = _nunchaku_to_blocks(pe)
        inv_blocks = blocks.transpose(-1, -2)
        return _blocks_to_nunchaku(inv_blocks)
    return pe.transpose(-1, -2)


def compose_rope(inv_base: torch.Tensor, variant: torch.Tensor, fmt: str = "flux") -> torch.Tensor:
    """Compose two rotation matrices: ``delta = inv(base) @ variant`` per block."""
    if fmt == "nunchaku":
        inv_base = _nunchaku_to_blocks(inv_base)
        variant = _nunchaku_to_blocks(variant)
    delta = torch.einsum("...ij,...jk->...ik", inv_base, variant)
    if fmt == "nunchaku":
        delta = _blocks_to_nunchaku(delta)
    return delta


def _rotate_flat(
    x: torch.Tensor,
    R: torch.Tensor,
    fmt: str,
    heads: int,
) -> torch.Tensor:
    """Rotate a head-FLATTENED ``(..., L, H*D)`` tensor and flatten it back.

    Qwen-Image-2.1 calls ``optimized_attention`` with ``q/k`` already collapsed
    to ``(B, N, H*D)`` and WITHOUT ``skip_reshape``, so the backend splits the
    heads itself.  :func:`apply_rope_matrix` derives ``P`` from the LAST dim, so
    applying a per-head rotation to ``(B, N, H*D)`` would use ``P = H*D/2``
    against a rotation built for ``P = D/2`` — the einsum raises.

    Splitting the heads out is not an optimisation: its einsum labels the final
    dims ``(l, p, j)``, i.e. it reads ``x`` as ``(..., L, D)`` with EVERY leading
    dim broadcastable.  The head dim therefore has to become a leading dim —
    ``(B, H, N, D)``, exactly where ``attention_pytorch``'s own reshape-then-
    transpose puts it — and not stay inline at ``(B, N, H, D)``.  The rotation is
    head-SHARED (``R``'s head dim is a singleton), so it broadcasts over the
    restored head dim and the round trip is exact (pinned by the flatten/unflatten
    equivalence test).
    """
    hd = x.shape[-1] // heads
    lead = x.shape[:-1]
    x4 = x.reshape(*lead, heads, hd).transpose(-3, -2)  # (..., H, L, D)
    out = apply_rope_matrix(x4, R, fmt)
    return out.transpose(-3, -2).reshape(*lead, heads * hd)  # (..., L, H*D)


def spa_averaged_attention(
    q,
    k,
    v,
    base_pe,
    variant_pes,
    attn_fn,
    pre_roped: bool = True,
    fmt: str = "flux",
    flatten_heads=None,
    q_slice=None,
    k_slice=None,
):
    """Run ``N`` attention passes (one per bundled RoPE variant) and average outputs.

    Faithful to HRDiT ``_spa_attention``.  When ``pre_roped`` is True the incoming
    ``q,k`` are already base-RoPE'd (ComfyUI FLUX applies ``pe`` before calling
    ``optimized_attention``); we then apply the per-variant *delta*
    ``inv(base) @ variant``.  When ``pre_roped`` is False we apply the full variant
    ``pe`` directly (variant RoPE is applied inside ``optimized_attention``).  A single
    variant (or none) is a transparent passthrough.

    Qwen-Image-2.1 (``flatten_heads``): its block-causal attention hands us a
    PREFIX of the sequence per segment — ``q`` spans ``[start, end)`` while ``k``
    spans ``[0, end)`` — and both arrive head-flattened.  ``q_slice`` / ``k_slice``
    (plain ``slice`` objects, ellipsis-first so they work for both the ``"flux"``
    ``(B, 1, L, P, 2, 2)`` and the ``"anima"`` ``(L, P, 2, 2)`` rotation layouts)
    select the matching rows of ``R``; ``flatten_heads`` is the head count the
    caller flattened to and enables the unflatten/rotate/reflatten round trip.
    The two are independent.  All three default to ``None``, which keeps every
    other backend byte-for-byte on the pre-existing path.
    """
    if not variant_pes or len(variant_pes) <= 1:
        return attn_fn(q, k, v)
    if pre_roped:
        inv_base = inv_rope(base_pe, fmt)
        rotations = [compose_rope(inv_base, vp, fmt) for vp in variant_pes]
    else:
        rotations = list(variant_pes)

    def _rows(rot, sl):
        """Select a segment's rows from a rotation (ellipsis-first)."""
        return rot if sl is None else rot[..., sl, :, :, :]

    # Slicing and unflattening are INDEPENDENT knobs: a caller may hand us a
    # segment of a head-format tensor, a flattened segment, or both.
    if flatten_heads is not None:
        outs = [
            attn_fn(
                _rotate_flat(q, _rows(rot, q_slice), fmt, flatten_heads),
                _rotate_flat(k, _rows(rot, k_slice), fmt, flatten_heads),
                v,
            )
            for rot in rotations
        ]
        return torch.stack(outs, dim=0).mean(dim=0)
    outs = [
        attn_fn(
            apply_rope_matrix(q, _rows(rot, q_slice), fmt),
            apply_rope_matrix(k, _rows(rot, k_slice), fmt),
            v,
        )
        for rot in rotations
    ]
    return torch.stack(outs, dim=0).mean(dim=0)
