"""Qwen-Image-2.1 adapter — the geometry, the ids and the rope capture.

Qwen-Image-2.1 (``comfy/ldm/qwen_image21/model.py``) is a single-stream MMDiT
with 32 blocks of ONE attention call each, a FLUX-layout ``EmbedND`` rope
embedder called ONCE per forward (``model.py:257``, called at ``model.py:338``)
and a **block-causal** stream: the caption is a segment of its own and the
image is another (``model.py:311-339``).

**The seam is a DIFFERENT binding from Qwen-Image 1.0.**  2.1 imports the
*unmasked* ``optimized_attention`` into its OWN module (``model.py:15``) and
calls the BOUND name at ``model.py:211`` (the per-segment call) and
``model.py:222`` (the prefix-cache call).  Patching
``comfy.ldm.qwen_image21.model.optimized_attention`` therefore hits 2.1 only —
the trap SPA documents (``src/spa.py:558-567``) and avoids
(``src/spa.py:575-582``).

**The eligible call is the IMAGE segment, and it has NO text queries.**
``block_causal_attention`` slices the joint stream per segment
(``model.py:205-211``): the caption segment carries a causal mask and the image
segment carries ``mask=None``; the image segment's queries are the ``h*w``
image rows and its keys are ``k[:, :end]`` — the caption AND the image.  So the
key set is JOINT (text first, image last — the engine's own convention,
``src/area_attn.py:380``) and there are no text queries at all.

**Geometry — MEASURED, and one ASSUMED constant.**  ``QWEN21_PX_PER_PATCH = 16``
is measured: the latent format is 64-channel, 2-dimensional and downscaled 16x
(``comfy/latent_formats.py:957-960``) and 2.1 does NOT patchify
(``model.py:260``, ``model.py:328``), so one token covers 16 px and the token
grid is the latent's OWN ``(H, W)``.  ``QWEN21_NATIVE_TOKENS = 64`` (1024 px)
is **ASSUMED** — it is the pack's own hypothesis (``src/spa.py:86-91``) and is
NOT derivable from the source — so it ships with a provenance label the tests
assert, exactly as the Krea-2 constants do (``src/area_krea.py:135-146``).

**The ids are the model's own, and they are neither FLUX- nor Z-Image-shaped.**
The caption's token ``j`` sits at ``(pos+j, pos+j, pos+j)`` — a DIAGONAL
(``model.py:323``) — where ``pos`` is a running counter; the image's token
``(r, c)`` sits at ``(pos, hh(r), ww(c))`` with the spatial axes CENTRED on the
canvas and a half-token parity term (``model.py:330-332``).  ``flux_style_ids``
is not reusable: the caption is not at the origin and the image's axis 0 is the
counter, not the frame index.

**The seam's input is 3-D.**  ``model.py:206-208`` flattens ``(B, n, H, D)`` to
``(B, n, H*D)`` and calls the seam WITHOUT ``skip_reshape``; comfy reads that
head-major (``comfy/ldm/modules/attention.py:147-158`` — the flat index ``j`` of
a row is ``head * dim_head + d``).  Every other AReA seam sees the 4-D layout,
so :func:`qwen21_to_heads` and :func:`qwen21_from_heads` convert both ways.

**The ``mask`` KEYWORD separates the call paths.**  The plain path always
passes ``mask=`` (``model.py:211``), the prefix-cache path passes NO mask at
all (``model.py:222``) — and the two are LENGTH-INDISTINGUISHABLE (the cached
path's key is ``txt_len + h*w`` long over ``h*w`` queries, exactly the plain
image segment's signature).  Only the keyword's PRESENCE tells them apart, so
the guard takes a SENTINEL for an absent keyword (:data:`QWEN21_NO_MASK`) and
the cached path is DELEGATED — its caption K/V are stale (an earlier step or a
separate prefix pass, ``model.py:392``), so a cross-scale attention over them
would be wrong.

**No comfy import here.**  The adapter names the seam's identity and owns the
tensors; the install lives in ``src/area.py``.  This module imports torch (and
the pack's own rope helpers) only, like every other adapter in the pack.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .area_attn import detail_map
from .area_rope import block_offsets, head_shared
from .area_schedule import coarse_grid

logger = logging.getLogger("ComfyUI-DyPE")

__all__ = [
    "QWEN21_BACKBONE",
    "QWEN21_HEADS",
    "QWEN21_HEAD_DIM",
    "QWEN21_LAYERS",
    "QWEN21_NATIVE_NOTE",
    "QWEN21_NATIVE_PROVENANCE",
    "QWEN21_NATIVE_TOKENS",
    "QWEN21_NO_MASK",
    "QWEN21_PATCH",
    "QWEN21_PX_NOTE",
    "QWEN21_PX_PER_PATCH",
    "QWEN21_PX_PROVENANCE",
    "QWEN21_ROPE_AXES",
    "QWEN21_SEAM_MODULE",
    "QWEN21_SEAM_SYMBOL",
    "QWEN21_TRAINED_RESOLUTION_PX",
    "QWEN21_VAE_DOWNSCALE",
    "Qwen21Layout",
    "Qwen21RopeRecorder",
    "capture_qwen21_table",
    "coarse_ropes_qwen21",
    "qwen21_centred_axis",
    "qwen21_coarse_key_ids",
    "qwen21_coarse_query_ids",
    "qwen21_detail_map",
    "qwen21_from_heads",
    "qwen21_head_geometry",
    "qwen21_image_ids",
    "qwen21_latent_grid",
    "qwen21_layout",
    "qwen21_segment_eligible",
    "qwen21_sequence_ids",
    "qwen21_slice_table",
    "qwen21_text_ids",
    "qwen21_to_heads",
]

#: One-time warnings (module scoped, like ``src/area_zimage.py``'s).
_warned: set = set()


def _warn_once(key: str, message: str, *args) -> None:
    """Emit ``message`` at WARNING level the first time ``key`` is seen."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


#: The seam's identity — the module that BINDS the symbol and the symbol
#: itself.  Qwen-Image-2.1 imports its own copy of the *unmasked*
#: ``optimized_attention`` (``from comfy.ldm.modules.attention import ...
#: optimized_attention``, ``comfy/ldm/qwen_image21/model.py:15``) and calls the
#: BOUND name at ``model.py:211`` and ``model.py:222``, so patching the
#: defining module (``comfy.ldm.modules.attention``) would move EVERY backend
#: while patching the 2.1 module moves only 2.1.  Named here (plan §4) so the
#: adapter's interface has one definition; ``src/area.py`` does the import and
#: the rebind.
QWEN21_SEAM_MODULE = "comfy.ldm.qwen_image21.model"
QWEN21_SEAM_SYMBOL = "optimized_attention"

#: The backbone's class name — the MRO marker the arch profile reads
#: (``src/area_arch.py``) and the detector's own class-name check
#: (``src/model_detect.py:50-51``).
QWEN21_BACKBONE = "QwenImage21Transformer2DModel"

#: Pixels per token — MEASURED from the source.  The latent format is
#: ``latent_channels = 64``, ``latent_dimensions = 2`` and
#: ``spacial_downscale_ratio = 16`` (``comfy/latent_formats.py:957-960``), and
#: 2.1 does not patchify on top of it (``model.py:260`` embeds the latent with
#: a plain ``Linear``, ``model.py:328`` flattens it to ``h*w`` tokens), so one
#: token covers 16 px — the same token size as FLUX, Z-Image and Krea-2.
QWEN21_PX_PER_PATCH = 16

#: The patch size — MEASURED.  2.1 does NOT patchify: one token per latent
#: cell (``comfy/ldm/qwen_image21/model.py:260`` — ``img_in`` is a plain
#: ``Linear`` over the latent's channels — and ``model.py:328``, which flattens
#: the latent to ``h*w`` tokens).  So the patch size is 1 and the token grid IS
#: the latent's own ``(H, W)``.
QWEN21_PATCH = 1

#: The VAE's spatial downscale — MEASURED
#: (``comfy/latent_formats.py:960`` — ``spacial_downscale_ratio = 16``).  With
#: no patchify on top of it, one token covers 16 px.
QWEN21_VAE_DOWNSCALE = 16

#: The query head count — MEASURED (``comfy/ldm/qwen_image21/model.py:242`` —
#: ``num_attention_heads = 32``; there is NO grouped-query attention, so the
#: key and the value heads are the same 32).
QWEN21_HEADS = 32

#: The head dimension — MEASURED (``model.py:241`` — ``attention_head_dim =
#: 128``).  The seam's flat input is ``heads * head_dim`` wide
#: (``model.py:206-208``).
QWEN21_HEAD_DIM = 128

#: The block count — MEASURED (``model.py:240`` — ``num_layers = 32``).
QWEN21_LAYERS = 32

#: The rope axes — MEASURED (``model.py:244`` — ``axes_dims_rope = (16, 56,
#: 56)``).  They sum to the head dimension (128), so the rope table's
#: frequency axis is ``128 // 2 = 64`` (``comfy/ldm/flux/layers.py:14-29``).
#: The FIRST axis is the narrow one (16) and the two spatial axes carry the
#: resolution.
QWEN21_ROPE_AXES = (16, 56, 56)

#: The provenance of :data:`QWEN21_PX_PER_PATCH`: read from the source, so it is
#: MEASURED.  The tests read it, so a silent constant fails the build.
QWEN21_PX_PROVENANCE = "MEASURED"

#: The label the constant carries, verbatim — what the provenance rests on.
QWEN21_PX_NOTE = (
    "MEASURED — latent_channels=64, latent_dimensions=2, "
    "spacial_downscale_ratio=16 (comfy/latent_formats.py:957-960); no patchify "
    "(comfy/ldm/qwen_image21/model.py:260, :328)"
)

#: The model's TRAINED extent, in TOKENS — ASSUMED.  1024 px over the 16 px
#: token grid is 64 tokens, the same number the pack uses for FLUX, Z-Image and
#: Krea-2.  It is NOT derivable from the source: it is the pack's own working
#: hypothesis (``src/spa.py:86-91``) and no Qwen-Image-2.1 checkpoint is
#: available in the dev environment.
QWEN21_NATIVE_TOKENS = 64

#: The provenance of :data:`QWEN21_NATIVE_TOKENS`: ASSUMED, not measured.
QWEN21_NATIVE_PROVENANCE = "ASSUMED"

#: The label the constant carries, verbatim — the plan's own hypothesis.
QWEN21_NATIVE_NOTE = (
    "ASSUMED — not measured; the trained extent is the pack's own hypothesis "
    "(src/spa.py:86-91), no Qwen-Image-2.1 checkpoint in the dev environment"
)

#: The trained canvas in PIXELS — ASSUMED (1024 px, the same hypothesis).  It
#: is the cross-check of the two constants above: a trained extent in tokens
#: and a pixel factor that disagree would mis-size every advisory, so the
#: tests assert ``native_tokens * px_per_patch == trained_resolution``.
QWEN21_TRAINED_RESOLUTION_PX = 1024


class _NoMask:
    """The sentinel for an ABSENT ``mask`` keyword.

    The plain path always passes ``mask=`` (``comfy/ldm/qwen_image21/model.py:
    211``) and the prefix-cache path never does (``model.py:222``), and the two
    are length-indistinguishable — the cached path's key is ``txt_len + h*w``
    long over ``h*w`` queries, exactly the plain image segment's signature.  So
    the ONLY discriminator is the keyword's presence, and a wrapper cannot see
    it unless the absent case has a value of its own: ``None`` is a legal
    ``mask`` (the image segment's), so the absent case needs a sentinel.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<QWEN21_NO_MASK>"


#: The sentinel handed to the guard when the seam call carried NO ``mask``
#: keyword at all — the prefix-cache path (``model.py:222``).
QWEN21_NO_MASK = _NoMask()


# ---------------------------------------------------------------------------
# The geometry — the token grid of a 2.1 latent
# ---------------------------------------------------------------------------


def qwen21_latent_grid(latent: torch.Tensor) -> tuple[int, int]:
    """The ``(h, w)`` TOKEN grid of a Qwen-Image-2.1 latent.

    2.1 does not patchify: ``img_in`` is a plain ``Linear`` over the latent's
    channels (``comfy/ldm/qwen_image21/model.py:260``) and the model reads the
    grid off the latent's OWN shape (``model.py:327``, ``h, w = img.shape[-2:]``)
    — there is no ``pad_to_patch_size`` and no division anywhere.  The latent
    format agrees: ``latent_dimensions = 2`` and ``spacial_downscale_ratio =
    16`` (``comfy/latent_formats.py:957-960``), so one token covers 16 px and
    the grid is the latent's own ``(H, W)``.

    A 4-D latent ``[B, C, H, W]`` is the NORMAL case (2.1's format is 2-D, so
    comfy never unsqueezes — the opposite of Krea-2's ``krea_squeeze_latent``,
    ``src/area_krea.py:179-223``).  Anything else is refused loudly: a 5-D
    latent is a VIDEO latent, and 2.1 has no video path at all, so accepting
    one would silently mis-size the canvas.

    Raises:
        ValueError: when the latent is not a tensor or is not 4-D.
    """
    if not torch.is_tensor(latent):
        raise ValueError(
            "AReA: the Qwen-Image-2.1 latent must be a tensor, but got %s. This "
            "forward runs plain attention." % type(latent).__name__
        )
    if latent.dim() != 4:
        raise ValueError(
            f"AReA: a Qwen-Image-2.1 latent is 4-D [B, C, H, W] "
            f"(comfy/latent_formats.py:957-960 — latent_dimensions = 2), but "
            f"this one is {latent.dim()}-D with shape {tuple(latent.shape)}. "
            "AReA's windowed attention is a latent-IMAGE construction, so this "
            "forward runs plain attention."
        )
    return int(latent.shape[-2]), int(latent.shape[-1])


# ---------------------------------------------------------------------------
# The ids — the model's own positions
# ---------------------------------------------------------------------------


def qwen21_centred_axis(size: int, target_size: Optional[int] = None,
                        device=None, dtype=torch.float32) -> torch.Tensor:
    """The model's own CENTRED coordinate of every row/column index.

    ``comfy/ldm/qwen_image21/model.py:330-331``::

        hh = arange(h) - (h - h // 2) + 0.5 * (h % 2 - target_h % 2)
        ww = arange(w) - (w - w // 2) + 0.5 * (w % 2 - target_w % 2)

    The spatial axes are centred on the canvas, and the half-token parity term
    centres a REFERENCE grid on the target.  For the target itself
    (``size == target_size``) the parity term vanishes, so the axis is the
    plain centred ramp — but the builder keeps the term, because the ids must
    mirror the model exactly rather than its single-image special case.

    Args:
        size: the grid's own size on this axis (``h`` or ``w``).
        target_size: the TARGET's size on this axis — the parity term's other
            half.  ``None`` means "the target is this grid", which zeroes the
            term.
    """
    size = int(size)
    target = size if target_size is None else int(target_size)
    index = torch.arange(size, device=device, dtype=dtype)
    return index - float(size - size // 2) + 0.5 * float(size % 2 - target % 2)


def qwen21_text_ids(pos: int, n: int, device=None,
                    dtype=torch.float32) -> torch.Tensor:
    """``[1, n, 3]`` — the caption's own rope ids, the DIAGONAL ``(pos+j)*3``.

    ``comfy/ldm/qwen_image21/model.py:323``::

        ids.append(torch.arange(pos, pos + n, ...).unsqueeze(1).expand(n, 3))

    ``pos`` is the model's running counter — 0 for the first text segment, and
    ``txt_len`` after it (``model.py:325``).  The caption is therefore NOT at
    the origin (FLUX's shape) and NOT on a single axis (Z-Image's shape): it
    rides a diagonal, and "the caption never moves" means this exact table.
    """
    n = int(n)
    j = torch.arange(n, device=device, dtype=dtype) + float(pos)
    return torch.stack([j, j, j], dim=-1).unsqueeze(0)


def qwen21_image_ids(pos: int, h: int, w: int, target_h: Optional[int] = None,
                     target_w: Optional[int] = None, device=None,
                     dtype=torch.float32) -> torch.Tensor:
    """``[1, h*w, 3]`` — the image tokens' own rope ids ``(pos, hh(r), ww(c))``.

    ``comfy/ldm/qwen_image21/model.py:330-332`` — the axis-0 value is the
    running counter (the caption's length in the reference-free case) and the
    two spatial axes are the centred ones of :func:`qwen21_centred_axis`.
    """
    h, w = int(h), int(w)
    hh = qwen21_centred_axis(h, target_h, device, dtype)
    ww = qwen21_centred_axis(w, target_w, device, dtype)
    axis0 = torch.full((h, w), float(pos), device=device, dtype=dtype)
    return torch.stack(
        [axis0, hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1
    ).reshape(1, h * w, 3)


def qwen21_sequence_ids(txt_len: int, h: int, w: int, target_h: Optional[int] = None,
                        target_w: Optional[int] = None, device=None,
                        dtype=torch.float32) -> torch.Tensor:
    """``[1, txt_len + h*w, 3]`` — the model's own ids for ONE image and a caption.

    This is ``build_sequence``'s id table for the reference-free case
    (``comfy/ldm/qwen_image21/model.py:317-332``): the caption's diagonal ids
    with the counter at 0, then the image's ids with the counter at ``txt_len``.
    The image's grid is the TARGET's grid, so the parity term is zero unless
    the caller says otherwise.
    """
    txt_len = int(txt_len)
    text = qwen21_text_ids(0, txt_len, device, dtype)
    image = qwen21_image_ids(txt_len, h, w, target_h, target_w, device, dtype)
    return torch.cat([text, image], dim=1)


def qwen21_coarse_query_ids(pos: int, h: int, w: int, f: int,
                            target_h: Optional[int] = None,
                            target_w: Optional[int] = None, device=None,
                            dtype=torch.float32) -> torch.Tensor:
    """``[1, h*w, 3]`` — the image queries' COMPRESSED ids.

    Phase I keeps the queries at full resolution but rides them on the coarse
    grid, so their positions are the model's own CENTRED positions scaled by
    ``1/f`` — the continuous step (the continuous-position mechanism) that the reference's
    floor division breaks.  The axis-0 value stays the model's own counter: the
    image tokens do not move along that axis, so no ``img_axis0`` keyword is
    needed for 2.1.
    """
    h, w, f = int(h), int(w), int(f)
    hh = qwen21_centred_axis(h, target_h, device, dtype) / float(f)
    ww = qwen21_centred_axis(w, target_w, device, dtype) / float(f)
    axis0 = torch.full((h, w), float(pos), device=device, dtype=dtype)
    return torch.stack(
        [axis0, hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1
    ).reshape(1, h * w, 3)


def qwen21_coarse_key_ids(img_pos: int, txt_len: int, h: int, w: int, f: int,
                          target_h: Optional[int] = None,
                          target_w: Optional[int] = None, device=None,
                          dtype=torch.float32) -> torch.Tensor:
    """``[1, txt_len + nh*nw, 3]`` — the pooled keys' ids.

    Two halves, and they are NOT the same length (the Phase-I tables are
    ASYMMETRIC on 2.1, because the eligible call has no text queries at all):

    * the caption's ``txt_len`` rows keep the model's own DIAGONAL ids — the
      caption never moves (``src/area_krea.py:592``);
    * the ``nh*nw`` pooled blocks sit at their block's CONTINUOUS centroid in
      the CENTRED frame — the mean of the block's query positions, which is the
      Krea-2 ripple fix (``src/area_krea.py:497-623``; the floor-division
      square wave is the measured Z-Image ripple).  A PARTIAL block (the last
      one on an axis the grid does not tile) owns fewer members, so its
      centroid is the mean of *its* members (``block_offsets``).

    The axis-0 value of the image rows is the model's own counter
    (``img_pos``); the caption's rows carry the caption's own diagonal.
    """
    txt_len, h, w, f = int(txt_len), int(h), int(w), int(f)
    text = qwen21_text_ids(0, txt_len, device, dtype)
    nh, nw = coarse_grid(h, w, f)
    uniform = (float(f) - 1.0) / (2.0 * float(f))
    row_off = block_offsets(nh, h, f, uniform, device)
    col_off = block_offsets(nw, w, f, uniform, device)
    blocks = torch.arange(nh * nw, device=device)
    block_rows = blocks // nw
    block_cols = blocks % nw
    # the block's mean ROW INDEX, then the model's own centred coordinate of it
    mean_rows = (block_rows.to(dtype) * f + row_off[block_rows].to(dtype) * f)
    mean_cols = (block_cols.to(dtype) * f + col_off[block_cols].to(dtype) * f)
    centred_rows = mean_rows - float(h - h // 2) + 0.5 * float(h % 2 - (h if target_h is None else int(target_h)) % 2)
    centred_cols = mean_cols - float(w - w // 2) + 0.5 * float(w % 2 - (w if target_w is None else int(target_w)) % 2)
    axis0 = torch.full((nh * nw,), float(img_pos), device=device, dtype=dtype)
    image = torch.stack([axis0, centred_rows / float(f), centred_cols / float(f)], dim=-1)
    return torch.cat([text, image.unsqueeze(0)], dim=1)


# ---------------------------------------------------------------------------
# The layout — the joint key set of the image segment
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Qwen21Layout:
    """The geometry of one Qwen-Image-2.1 image segment.

    2.1's eligible call has NO text queries: the image segment's queries are
    the ``h*w`` image rows and its keys are the JOINT set — the caption AND the
    image (``comfy/ldm/qwen_image21/model.py:206-208``, ``k[:, :end]``).  The
    engine's own convention ("text first, image last",
    ``src/area_attn.py:380``) therefore applies verbatim: the caption's rows
    are the first ``txt_len`` rows of the key and the image's rows are the
    trailing ``h*w``.

    ``txt_len``  the caption's length (``context.shape[1]``).
    ``h``, ``w``  the token grid of the canvas (the latent's own ``(H, W)``).
    ``q_len``  the queries of THIS call (``h*w`` — the image rows only).
    ``k_len``  the keys of THIS call (``txt_len + h*w`` — the joint set).
    """

    txt_len: int
    h: int
    w: int
    q_len: int
    k_len: int

    # --- the spans ----------------------------------------------------------

    @property
    def img_start(self) -> int:
        """``txt_len`` under its slicing name: the first image token of the KEY."""
        return self.txt_len

    @property
    def img_end(self) -> int:
        """One past the last image token of the KEY."""
        return self.txt_len + self.h * self.w

    @property
    def img_tokens(self) -> int:
        """``h * w`` — how many image tokens the canvas owns."""
        return self.h * self.w


def qwen21_layout(txt_len: int, h: int, w: int, q_len: int,
                  k_len: int) -> Qwen21Layout:
    """Resolve — and check — the geometry of one 2.1 image segment.

    Raises:
        ValueError: when the call does not fit the segment geometry — an empty
            grid, or a query/key length that is not ``h*w`` /
            ``txt_len + h*w``.  The message names the offender and the remedy
            (house rule, ``src/hap.py:876-895``), so a foreign call is refused
            instead of silently pooling the wrong tokens.
    """
    txt_len, h, w = int(txt_len), int(h), int(w)
    q_len, k_len = int(q_len), int(k_len)
    if min(h, w) < 1:
        raise ValueError(
            f"AReA: a Qwen-Image-2.1 image segment needs a non-empty token "
            f"grid, but got {h}x{w}. This forward runs plain attention."
        )
    if txt_len < 0:
        raise ValueError(
            f"AReA: a Qwen-Image-2.1 image segment cannot have a negative "
            f"caption length ({txt_len}). This forward runs plain attention."
        )
    if q_len != h * w:
        raise ValueError(
            f"AReA: the Qwen-Image-2.1 image segment carries {q_len} queries, "
            f"but a {h}x{w} token grid owns {h * w} (comfy/ldm/qwen_image21/"
            "model.py:206). The prompt and the latent do not belong to one "
            "forward, so the image tokens cannot be located. This forward runs "
            "plain attention."
        )
    if k_len != txt_len + h * w:
        raise ValueError(
            f"AReA: the Qwen-Image-2.1 image segment's key is {k_len} rows "
            f"long, but a {txt_len}-token caption behind a {h}x{w} canvas needs "
            f"{txt_len + h * w} (comfy/ldm/qwen_image21/model.py:207 — "
            "k[:, :end]). This forward runs plain attention."
        )
    return Qwen21Layout(txt_len, h, w, q_len, k_len)


# ---------------------------------------------------------------------------
# The eligibility guard — the image segment of the plain path, and nothing else
# ---------------------------------------------------------------------------


def qwen21_segment_eligible(mask, q_len, k_len, total_len, txt_len, h, w,
                            heads=None, dm_heads=None) -> bool:
    """Is this seam call 2.1's IMAGE segment of the PLAIN path? (plan Q2)

    Five conditions, all necessary:

    * **the ``mask`` keyword was PRESENT** — the prefix-cache path calls the
      seam with no ``mask`` at all (``comfy/ldm/qwen_image21/model.py:222``),
      and that path is length-indistinguishable from this one (its key is
      ``txt_len + h*w`` long over ``h*w`` queries), so the keyword's presence
      is the ONLY discriminator.  :data:`QWEN21_NO_MASK` marks the absent case.
    * **the mask is ``None``** — the caption segment carries a causal bool
      mask (``model.py:324``) and the image segment carries ``None``
      (``model.py:333``).  The caption is DELEGATED untouched.
    * **the lengths fit** — ``q_len == h*w`` and ``k_len == txt_len + h*w``.
      A reference-latent forward makes the joint stream longer than that, so
      it is refused here as well (the forward refuses it too).
    * **the key is the WHOLE stream** — ``k_len == total_len`` when the
      caller knows the forward's sequence length (the captured rope table's
      length is exactly that).  A reference chunk's key stops short of it.
    * **the head count is the model's own** — ``heads == dm_heads`` (N13).
      An unknown variant of the backbone is DELEGATED, never engaged with.

    Returns ``False`` — never raises — so a call that fails any of the five
    is simply DELEGATED UNCHANGED by the seam.
    """
    try:
        if mask is QWEN21_NO_MASK:
            # the prefix-cache path: no mask keyword at all.
            return False
        if mask is not None:
            # the caption segment: a causal bool mask.
            return False
        if heads is not None and dm_heads is not None and int(heads) != int(dm_heads):
            # N13 — an unknown head count is an unknown variant: delegate.
            return False
        q_len = int(q_len)
        k_len = int(k_len)
        txt_len = int(txt_len)
        h = int(h)
        w = int(w)
        if h < 1 or w < 1:
            return False
        if q_len != h * w:
            return False
        if k_len != txt_len + h * w:
            return False
        if total_len is not None and k_len != int(total_len):
            return False
    except (TypeError, ValueError):
        return False
    return True


def qwen21_head_geometry(x, heads) -> Optional[int]:
    """The per-head width of a 3-D seam tensor, or ``None`` when it is foreign.

    2.1's seam input is ``[b, n, heads*dim]`` (``model.py:206-208``), so the
    flat width must split into the seam's ``heads`` — and the head width is
    what the geometry is checked against.  A width that does NOT split is a
    FOREIGN call (an unknown variant of the backbone), and it is reported as
    ``None`` instead of raising: the seam then DELEGATES the call, which keeps
    a sampling run alive instead of aborting it (N13).
    """
    if not torch.is_tensor(x) or x.dim() != 3:
        return None
    try:
        heads = int(heads)
    except (TypeError, ValueError):
        return None
    inner = int(x.shape[-1])
    if heads < 1 or inner % heads:
        return None
    return inner // heads


# ---------------------------------------------------------------------------
# The seam's layout — 3-D in, 3-D out
# ---------------------------------------------------------------------------


def qwen21_to_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    """``[b, n, heads*dim]`` -> ``[b, heads, n, dim]`` — the engine's layout.

    The seam's input is 3-D: ``model.py:206-208`` flattens ``(B, n, H, D)`` to
    ``(B, n, H*D)`` and calls the seam without ``skip_reshape``, and comfy
    reads that head-major (``comfy/ldm/modules/attention.py:147-158`` — the
    flat index ``j`` of a row is ``head * dim_head + d``).  Every other AReA
    seam sees the 4-D layout, so the conversion is here — and it is the exact
    inverse of :func:`qwen21_from_heads`.
    """
    b, n, inner = x.shape
    heads = int(heads)
    if heads < 1 or inner % heads:
        raise ValueError(
            f"AReA: a Qwen-Image-2.1 seam input of shape {tuple(x.shape)} does "
            f"not split into {heads} heads, so it cannot be read head-major. "
            "This call runs plain attention."
        )
    return x.reshape(b, n, heads, inner // heads).transpose(1, 2)


def qwen21_from_heads(x: torch.Tensor) -> torch.Tensor:
    """``[b, heads, n, dim]`` -> ``[b, n, heads*dim]`` — comfy's seam output.

    The inverse of :func:`qwen21_to_heads`, and the layout the model expects
    back: ``block_causal_attention`` concatenates the per-segment answers on
    dim 1 (``model.py:212``) and the block multiplies the answer by a
    ``[b, n, dim]`` gate (``model.py:146``).
    """
    b, heads, n, dim = x.shape
    return x.transpose(1, 2).reshape(b, n, heads * dim)


def qwen21_peek(x):
    """The tensor inside a seam argument, WITHOUT consuming its container.

    ``AttentionTensorContainer`` is single-owner: ``take()`` consumes it once
    and a second take raises (``comfy/ldm/modules/attention.py:196-209``).  The
    seam therefore INSPECTS with ``peek()`` and takes exactly once — on the
    AReA path only.
    """
    return x.peek() if hasattr(x, "peek") else x


def qwen21_slice_table(table: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """``table[:, :, start:end]`` — the model's own table for a span of tokens.

    The model rotates the WHOLE joint stream with the whole table and slices
    afterwards (``comfy/ldm/qwen_image21/model.py:99-100``, ``:206-208``), so
    the image segment's QUERIES were rotated with the table's TRAILING rows
    (the image's own) and the joint KEY with its leading ones.  Slicing the
    model's own table is therefore the exact inverse of that rotation — the
    same rows, in the same order, with the same phases.
    """
    return table[:, :, int(start):int(end)]


# ---------------------------------------------------------------------------
# The detail map — the Phase II source, on 2.1's own 2-D grid
# ---------------------------------------------------------------------------


def qwen21_detail_map(hidden: torch.Tensor, h: int, w: int) -> np.ndarray:
    """The Phase II detail map over 2.1's OWN 2-D token grid.

    **The grid IS the latent's ``(H, W)``** (D7): 2.1 does not patchify
    (``img_in`` is a plain ``Linear`` over the latent's channels,
    ``comfy/ldm/qwen_image21/model.py:260``) and its latent format is
    2-dimensional (``comfy/latent_formats.py:959``), so there is no division
    and no unsqueeze — the opposite of Krea-2's ``krea_squeeze_latent``
    (``src/area_krea.py:179-223``), which exists because Krea-2's latent is
    5-D and comfy unsqueezes it.

    ``hidden`` is the captured ``img_in`` output: ``[b, h*w, dim]`` — the image
    tokens in the model's own layout, which is the layout
    :func:`~src.area_attn.detail_map` reads.

    Raises:
        ValueError: when the capture is not this canvas's image-token grid (a
            reference latent, a padded row, another canvas) — the map would
            then be a map of the wrong image, which is worse than no map.
    """
    if not torch.is_tensor(hidden):
        raise ValueError(
            "AReA: the Qwen-Image-2.1 detail-map source must be a tensor, but "
            "got %s. This forward runs plain attention." % type(hidden).__name__
        )
    if hidden.dim() != 3:
        raise ValueError(
            f"AReA: the Qwen-Image-2.1 detail-map source must be 3-D "
            f"[b, h*w, dim], but it is {hidden.dim()}-D with shape "
            f"{tuple(hidden.shape)}. This forward runs plain attention."
        )
    expected = int(h) * int(w)
    if int(hidden.shape[1]) != expected:
        raise ValueError(
            f"AReA: the Qwen-Image-2.1 detail-map source holds "
            f"{int(hidden.shape[1])} image tokens, but a {int(h)}x{int(w)} "
            f"canvas owns {expected} (comfy/ldm/qwen_image21/model.py:327 "
            "reads the grid off the latent's own shape). This forward runs "
            "plain attention."
        )
    return detail_map(hidden, int(h), int(w))


# ---------------------------------------------------------------------------
# The rope capture — the model's own table, once per forward
# ---------------------------------------------------------------------------


class Qwen21RopeRecorder:
    """Per-forward capture of the ONE rope table the model's own embedder built.

    2.1 builds its table ONCE per forward — ``self.pe_embedder(ids)`` over the
    whole joint stream (``comfy/ldm/qwen_image21/model.py:338``) — and applies
    it itself BEFORE the seam (``model.py:99-100``), so the seam receives
    rotated q/k and NO table.  Phase I has to pool *unrotated* keys, which
    means the model's own table has to be inverted
    (:func:`src.area_rope.unrotate_table`).

    **The first table wins, and it is never overwritten.**  The coarse Phase I
    tables are built through the very SAME embedder (so a chained DyPE/SEGA
    embedder composes), which means the capture hook would see them too.  Two
    mechanisms keep that from polluting the forward:

    * :meth:`paused` suspends the capture around a coarse build — the pause
      guard;
    * a table that arrives anyway (an embedder call the wiring did not know
      about) is IGNORED: the first table stays, and the extra one is only
      counted (:attr:`extra_tables`, :attr:`ambiguous`).

    The recorder stores tensors only; it never rebinds a module symbol.  It is
    armed and reset per forward (the SPA discipline,
    ``src/spa.py:1385-1476``), so no table leaks across models or steps.
    """

    __slots__ = ("_extra", "_paused", "_table", "armed")

    def __init__(self) -> None:
        self._table: Optional[torch.Tensor] = None
        self._extra = 0
        self._paused = 0
        # The caller's arming window: the recorder only records while the
        # wiring has armed it, so an embedder call outside the forward of
        # interest (a refiner pass, a chained embedder's own call) is ignored.
        self.armed = False

    def reset(self) -> None:
        """Forget this forward's table — called at the start of each forward."""
        self._table = None
        self._extra = 0

    def record(self, table: torch.Tensor) -> None:
        """Capture one table, if this forward's table has not been seen yet.

        ``table`` is what the embedder's forward hook sees: an ``EmbedND``
        result ``[b, 1, n, axes, 2, 2]`` (``comfy/ldm/flux/layers.py:14-29``).
        It is normalised to the head-shared layout on the way in, so every
        accessor reads the sequence length off axis 2.

        The FIRST table of the forward is kept; any later one is counted and
        dropped — it never replaces the model's own table (the recorder is
        paused around the coarse builds, so a second table means the wiring
        missed a pause).
        """
        if not self.armed or self._paused:
            return
        if not torch.is_tensor(table) or table.dim() < 3:
            raise ValueError(
                "AReA: the Qwen-Image-2.1 rope recorder was handed %s, which "
                "is not an EmbedND table ([b, 1, n, axes, 2, 2]). A pe_embedder "
                "forward hook must record the embedder's output."
                % type(table).__name__
            )
        if self._table is None:
            self._table = head_shared(table)
        else:
            self._extra += 1

    @property
    def table(self) -> Optional[torch.Tensor]:
        """This forward's table — the model's OWN, or ``None`` if none was seen."""
        return self._table

    @property
    def extra_tables(self) -> int:
        """How many tables arrived AFTER the first one and were dropped."""
        return self._extra

    @property
    def ambiguous(self) -> bool:
        """Whether an extra table arrived (informational — it is NOT a refusal).

        The recorder never refuses and never overwrites: the model's own table
        is the first one and it stays.  This flag only says that the forward
        produced more than one table, which the wiring may log.
        """
        return self._extra > 0

    @contextmanager
    def paused(self):
        """Suspend the capture for the duration of the block.

        The coarse Phase I tables are built through the model's OWN embedder
        (``src/area_qwen21.py``'s :func:`coarse_ropes_qwen21`) — so without
        this guard the capture would record them as extra tables of the
        forward.  The pause is re-entrant and always unwinds.
        """
        self._paused += 1
        try:
            yield self
        finally:
            self._paused -= 1


#: The recorder of each captured embedder, keyed by ``id(pe_embedder)`` — so
#: :func:`coarse_ropes_qwen21` can find the recorder to pause without the
#: caller having to pass it.  One entry per INSTALLED model.
_RECORDERS: dict[int, Qwen21RopeRecorder] = {}


def capture_qwen21_table(pe_embedder) -> tuple:
    """The rope capture of one model: ``(hook, recorder)`` (plan §4).

    ``pe_embedder`` is the model's own embedder
    (``comfy/ldm/qwen_image21/model.py:257`` — a FLUX-layout ``EmbedND`` with
    ``axes_dim = (16, 56, 56)``, so the table is ``[b, 1, n, 64, 2, 2]``); the
    returned ``hook`` is registered on it right here and carries the
    ``RemovableHandle`` it must be torn down by on ``hook.handle``, so the
    wiring (``src/area.py``) registers nothing and removes what it was given.

    The recorder only records while it is ARMED, so an embedder call outside
    the forward of interest (a refiner pass, a chained embedder's own call) is
    ignored.  A foreign output — anything that is not an ``EmbedND`` table — is
    reported once and dropped, so a forward keeps running plain attention
    instead of crashing the sampler.
    """
    recorder = Qwen21RopeRecorder()

    def hook(module, args, output):
        try:
            recorder.record(output)
        except ValueError as exc:
            _warn_once("qwen21-record", "AReA: %s", exc)

    hook.handle = None
    if hasattr(pe_embedder, "register_forward_hook"):
        hook.handle = pe_embedder.register_forward_hook(hook)
    _RECORDERS[id(pe_embedder)] = recorder
    return hook, recorder


# ---------------------------------------------------------------------------
# The coarse tables — the model's own embedder, 2.1's own ids
# ---------------------------------------------------------------------------


def coarse_ropes_qwen21(embedder, layout: Qwen21Layout, f: int, device,
                        recorder: Optional[Qwen21RopeRecorder] = None,
                        target_h: Optional[int] = None,
                        target_w: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The two Phase I rope tables — CONTINUOUS block positions, 2.1's own ids.

    **The two tables have DIFFERENT lengths on 2.1** (the asymmetry the plan
    leaves under-specified): the eligible call has no text queries, so

    * ``rope_q`` covers the ``h*w`` IMAGE QUERIES only, at their own
      compressed centred positions;
    * ``rope_kv`` covers ``txt_len + nh*nw`` rows — the caption at the model's
      own diagonal ids (the caption never moves) and the pooled keys at their
      block's CONTINUOUS centroid in the centred frame (the Krea-2 ripple fix;
      the reference's floor division collapses every ``f x f`` block onto one
      rope position and injects a one-radian square wave into the velocity
      field at every block boundary — the pack's measured Z-Image ripple).

    The block positions are CONTINUOUS (the continuous-position mechanism): every query keeps
    its own position scaled by ``1/f`` and the pooled key sits at its block's
    GEOMETRIC CENTROID — the mean of the block's query positions — so the
    phase step across a boundary equals the one inside the block and the
    staircase is gone; ``f == 1`` degenerates exactly.

    The axis-0 value stays the model's own counter (``txt_len``), so 2.1 needs
    no ``img_axis0`` keyword.

    The tables come from the model's OWN embedder, so a chained DyPE/SEGA
    embedder composes — and the capture is PAUSED around the two calls, so the
    coarse tables are not mistaken for extra tables of the forward (the pause
    guard; see :meth:`Qwen21RopeRecorder.paused`).

    Args:
        embedder: the model's own ``pe_embedder``.
        layout: the image segment's layout of this forward.
        f: the pooling factor of this step.
        device: the device the ids are built on.
        recorder: the rope recorder to pause.  Optional — the recorder that
            :func:`capture_qwen21_table` registered for this embedder is found
            automatically — but passing it explicitly is clearer.
    """
    if embedder is None:
        raise ValueError(
            "AReA: this model has no diffusion_model.pe_embedder, so the "
            "Phase I coarse rope tables cannot be built. Use an unmodified "
            "Qwen-Image-2.1 loader."
        )
    f = int(f)
    if f < 1:
        raise ValueError(
            f"AReA: the pooling factor {f} is not a valid pooling block size, "
            "so the coarse rope grid cannot be built. This forward runs plain "
            "attention."
        )
    if recorder is None:
        recorder = _RECORDERS.get(id(embedder))
    nh, nw = coarse_grid(layout.h, layout.w, f)

    q_ids = qwen21_coarse_query_ids(
        layout.txt_len, layout.h, layout.w, f, target_h, target_w, device, torch.float32
    )
    kv_ids = qwen21_coarse_key_ids(
        layout.txt_len, layout.txt_len, layout.h, layout.w, f,
        target_h, target_w, device, torch.float32,
    )

    if recorder is not None:
        # The pause guard: the coarse tables go through the model's OWN
        # embedder, and the capture hook would otherwise record them as extra
        # tables of this forward.
        with recorder.paused():
            rope_q = embedder(q_ids)
            rope_kv = embedder(kv_ids)
    else:
        rope_q = embedder(q_ids)
        rope_kv = embedder(kv_ids)
    # Head-shared either way, so a chained embedder that returns another
    # layout is handled too.
    return head_shared(rope_q), head_shared(rope_kv)
