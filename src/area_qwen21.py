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
    "qwen21_latent_grid",
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


