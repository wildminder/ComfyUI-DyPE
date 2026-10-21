"""AReA architecture profiles — which backbone a given model actually is.

AReA is a *model-side* attention replacement, so apply time starts with one
question: which diffusion backbone is this?  The engine (``src/area_attn.py``,
``src/area_schedule.py``) is backbone-agnostic — it pools keys, builds windows
and reads rope tables — and every FLUX/Z-Image/Krea-2/Qwen-Image-2.1 difference
lives in the adapter that feeds it.  :class:`_ArchProfile` is that difference in
six fields, resolved once (plan §4, Q7/Q9).

**Three backbones, one node.** FLUX (``Flux``, ``comfy/ldm/flux/model.py``) and
Z-Image (``NextDiT``, ``comfy/ldm/lumina/model.py:442``, bound by
``Lumina2.unet_model`` at ``comfy/model_base.py:1510``) share today's geometry:
the Flux VAE downscales 8x and both patchify 2x2, so one token covers **16 px**
on either backbone (``src/area.py:109-112``) and the node's multiple-of-16
validation covers both.  They differ where the sequence is padded: Z-Image pads
cap AND image tokens to a multiple of 32
(``comfy/ldm/lumina/model.py:418-420``, applied at ``:665`` and ``:725-727``),
FLUX pads to nothing.

**Krea-2** (``SingleStreamDiT``, ``comfy/ldm/krea2/model.py:234``) joins them
in S4 (plan §7/S4).  Its token size is the same **16 px**
(``KREA_PX_PER_PATCH`` — an 8x VAE and a 2x2 patch, ASSUMED: the VAE family is
the checkpoint's, not the source's) and it pads nothing at all
(``seq_pad_multiple == 0``, ``comfy/ldm/krea2/model.py:355``), so its profile
differs from FLUX's only in the backbone name, the pixel factor's provenance and
the trained extent in tokens (``native_tokens``).

**The trained extent in TOKENS.**  FLUX and Z-Image both train on 1024 px over
a 16 px token grid, i.e. 64 tokens (``src/area.py:159``, the module constant
``NATIVE_PATCHES``); the profile carries that number as ``native_tokens`` so a
backbone with a different pixel factor can carry a different one (Krea-2's is
``KREA_NATIVE_TOKENS``, ASSUMED for the same reason).  The field has a default,
so the existing constructions stay valid and their values are unchanged.

**Why the MRO and not just the probes.** :func:`resolve_model_type`
(``src/model_detect.py``) is the single source of truth for which backend a
model belongs to, and its auto probe answers ``zimage`` for anything carrying
``rope_embedder`` (``src/model_detect.py:74-75``).  That is enough to route a
call but not enough to trust: a pin of ``zimage`` on a FLUX checkpoint (or
``flux`` on a Lumina2 one) would otherwise surface as a silently mis-patched
seam.  The MRO is the ground truth, so the resolved profile is always checked
against it — Q7's "pins validated".

**Rejections (Q7).** Three Lumina2-family siblings are *not* the Z-Image
checkpoint, and patching them would corrupt the geometry AReA assumes:

* ``ZImagePixelSpace`` — ``NextDiTPixelSpace`` (``comfy/ldm/lumina/model.py:1005``,
  ``comfy/model_base.py:1573-1576``) deletes ``final_layer`` and runs a
  ``dec_net`` decoder over raw RGB patches, so no 16-px latent patch grid
  exists for AReA to reason about.
* ``MingImage`` (``comfy/model_base.py:1562``) pads the prompt out to
  ``masked_pad_multiple`` (``comfy/ldm/lumina/model.py:684-686``,
  ``model_detection.py:620-622``), which moves the image start past the cap
  length, so the image tokens can no longer be located in the joint stream.
* a requested ``model_type`` that contradicts the model's own backbone.

**Krea-2's own rejections.**  The Krea-2 branch sits BEFORE the Lumina2 one and
has the same two rules: a pin of ``flux``/``zimage`` on a Krea-2 model, and a
pin of ``krea2`` on a model that is not a Krea-2, are both contradictions and
raise (plan Q15).  Krea-2's *runtime* rejections — video latents, reference
latents, the four auxiliary 20-head calls — belong to the forward-time
resolution (``src/area.py``), not to the profile.

**Qwen-Image-2.1** (``QwenImage21Transformer2DModel``,
``comfy/ldm/qwen_image21/model.py:234``) joins them in S5 (plan,
§7/S5).  Its token size is **16 px** — MEASURED, unlike Krea-2's: the latent
format downsamples 16x and 2.1 does not patchify on top of it
(``comfy/latent_formats.py:957-960``, ``model.py:260``) — and it pads nothing
at all (``seq_pad_multiple == 0``, ``model.py:311-339``), so its profile
differs from FLUX's only in the backbone name, the pixel factor's provenance,
the trained extent in tokens and the shape-aware bridge.  Its *runtime*
rejections — the prefix-cache path, the reference latents — belong to the
forward-time resolution (``src/area.py``), not to the profile.

Nunchaku (quantized, fused kernels) is neither: fused attention bypasses the
seam entirely, so it is warned about and returned **unchanged** by the caller
(``src/area.py:605-614``, the HAP precedent at ``src/hap.py:897-903``) instead
of being refused here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .area_krea import KREA_BACKBONE, KREA_NATIVE_TOKENS, KREA_PX_PER_PATCH
from .area_qwen21 import QWEN21_BACKBONE, QWEN21_NATIVE_TOKENS, QWEN21_PX_PER_PATCH
from .model_detect import resolve_model_type

__all__ = [
    "FLUX_BACKBONE",
    "FLUX_PROFILE",
    "KREA_PROFILE",
    "KREA_REQUESTS",
    "NEXT_DIT_BACKBONE",
    "PIXEL_SPACE_BACKBONE",
    "QWEN21_PROFILE",
    "QWEN21_REQUESTS",
    "ZIMAGE_PROFILE",
    "_ArchProfile",
    "resolve_area_arch",
]

# Backbone class names (the MRO markers).
FLUX_BACKBONE = "Flux"
NEXT_DIT_BACKBONE = "NextDiT"
PIXEL_SPACE_BACKBONE = "NextDiTPixelSpace"

#: The Phase I/II sigma switch (``ZIMAGE_SIGMA_STAR`` /
#: :func:`src.area_zimage.sigma_prefix_count`, plan §4) lives with the adapter,
#: because it is expressed on the sampler's sigma table rather than on the
#: backbone.


@dataclass(frozen=True)
class _ArchProfile:
    """Everything about the backbone that is NOT resolved per forward."""

    model_type: str           # "flux" | "zimage" | "krea2" — ``resolve_model_type``'s key
    unet_backbone: str        # class name: "Flux" | "NextDiT" | "SingleStreamDiT"
    px_per_patch: int         # 16 on all three (VAE 8 x patchify 2)
    seq_pad_multiple: int     # 0 (flux, krea2) | 32 (zimage) — see the module docstring
    accepts: str              # human-readable model match string
    native_tokens: int = 64   # the trained extent in TOKENS (1024 px / 16 px)
    #: Whether the Phase II bridge's register count follows the WINDOW'S
    #: ACTUAL SHAPE instead of the window SIDE alone.  The paper's bridge is
    #: ``R = (rho * w)^2`` registers for a SQUARE window of side ``w``
    #: (App. E); a window clipped by a canvas narrower than its side holds
    #: fewer tokens, so the square formula over-provisions the bridge and the
    #: window's own content loses the share of the attention the bridge is
    #: meant to have.  Krea-2 only (the FLUX and Z-Image paths are frozen).
    bridge_shape_aware: bool = False

    def patch_count(self, pixels: int) -> int:
        """Tokens per axis for ``pixels`` — ceil div by ``px_per_patch``.

        Mirrors ``_grid_from_latent``'s ``ceil(H / patch_size)``
        (``src/area.py:206-215``): the patchifier pads a latent that is not a
        whole number of patches, so a fraction rounds UP, never down.
        """
        return -(-int(pixels) // self.px_per_patch)

    def padded_seq_len(self, length: int) -> int:
        """``length`` rounded UP to the architecture's sequence pad multiple.

        The adapter needs the padded joint length to slice the image window at
        the pad boundary — Z-Image's cap and image are each padded to 32
        (``comfy/ldm/lumina/model.py:418-420``), FLUX pads to nothing.
        """
        multiple = self.seq_pad_multiple
        if multiple <= 1:
            return int(length)
        return -(-int(length) // multiple) * multiple


ZIMAGE_PROFILE = _ArchProfile(
    "zimage", NEXT_DIT_BACKBONE, 16, 32, "Lumina2 (NextDiT) diffusion model"
)
FLUX_PROFILE = _ArchProfile(
    "flux", FLUX_BACKBONE, 16, 0, "FLUX.1 (Flux) diffusion model"
)

#: Krea-2 (``comfy/ldm/krea2/model.py:234``) — a single-stream MMDiT with 48
#: query heads over 12 KV heads and a 12-layer text-fusion adapter whose four
#: 20-head calls run BEFORE the main blocks (``model.py:341``).
#:
#: ``seq_pad_multiple`` is 0: the joint stream is ``cat((context, img), dim=1)``
#: (``model.py:355``) — text first, then the image, no padding at all.  The
#: pixel factor and the trained extent come from the adapter, which labels them
#: ASSUMED (no checkpoint in the dev environment — ``src/area_krea.py``).
KREA_PROFILE = _ArchProfile(
    "krea2", KREA_BACKBONE, KREA_PX_PER_PATCH, 0,
    "Krea-2 (SingleStreamDiT) diffusion model",
    native_tokens=KREA_NATIVE_TOKENS,
    # The bridge's register count follows the window's actual shape here: the
    # Krea-2 canvas the user runs (512 x 2000 px -> a 125 x 32 token grid) is
    # narrower than the Phase II window side (64), so the square formula would
    # hand every window half of the attention to the OTHER half's content.
    bridge_shape_aware=True,
)

#: The ``model_type`` requests a Krea-2 backbone accepts.  Everything else is a
#: contradiction and raises: the resolver pins the backbone by CLASS NAME first
#: (``src/model_detect.py:45-49``), so ``resolve_model_type`` cannot report the
#: contradiction for a Krea-2 model — the requested string has to be checked
#: here.
KREA_REQUESTS = ("auto", "krea2")

#: Qwen-Image-2.1 (``comfy/ldm/qwen_image21/model.py:234``) — a single-stream
#: MMDiT with 32 blocks of ONE attention call each and a **block-causal** stream
#: (the caption is a segment of its own, the image another — ``model.py:311-339``).
#:
#: ``seq_pad_multiple`` is 0: 2.1 pads nothing (``model.py:311-339`` builds the
#: joint stream with no padding at all).  The pixel factor is MEASURED — the
#: latent format downsamples 16x and 2.1 does not patchify on top of it
#: (``comfy/latent_formats.py:957-960``, ``model.py:260``) — while the trained
#: extent in tokens is ASSUMED (no checkpoint in the dev environment,
#: ``src/area_qwen21.py``).
#:
#: ``bridge_shape_aware`` is True for the Krea-2 reason: the paper's square
#: ``(rho*w)^2`` bridge count over-provisions on a non-square canvas, and a
#: Qwen-Image-2.1 canvas is whatever the user asked for.
QWEN21_PROFILE = _ArchProfile(
    "qwen21", QWEN21_BACKBONE, QWEN21_PX_PER_PATCH, 0,
    "Qwen-Image-2.1 (QwenImage21Transformer2DModel) diffusion model",
    native_tokens=QWEN21_NATIVE_TOKENS,
    bridge_shape_aware=True,
)

#: The ``model_type`` requests a Qwen-Image-2.1 backbone accepts.  Everything
#: else is a contradiction and raises: the resolver pins the backbone by CLASS
#: NAME first (``src/model_detect.py:50-51``), so ``resolve_model_type`` cannot
#: report the contradiction for a 2.1 model — the requested string has to be
#: checked here.
QWEN21_REQUESTS = ("auto", "qwen21")


def _mro_names(diffusion_model) -> tuple[str, ...]:
    """Class names on the model's MRO — the backbone's identity."""
    return tuple(cls.__name__ for cls in type(diffusion_model).__mro__)


def _is_ming_image(diffusion_model) -> bool:
    """``MingImage`` (``comfy/model_base.py:1562``) by its own two markers.

    ``NextDiT.__init__`` registers an empty ``__ming_image__`` buffer for it
    (``comfy/ldm/lumina/model.py:486``) and detection sets
    ``masked_pad_multiple=32`` (``model_detection.py:620-622``); either one
    means the prompt is masked-padded, so the image no longer starts at the cap
    length.  Plain Z-Image also carries ``pad_tokens_multiple=32`` and is NOT
    matched — that padding is what the profile's ``seq_pad_multiple`` describes.
    """
    return (
        getattr(diffusion_model, "masked_pad_multiple", None) is not None
        or hasattr(diffusion_model, "__ming_image__")
    )


def resolve_area_arch(diffusion_model, requested_type: str = "auto") -> Optional[_ArchProfile]:
    """The AReA architecture profile of ``diffusion_model``.

    Args:
        diffusion_model: the live ``diffusion_model`` the patcher patches — the
            MRO walk and the attribute probes run on it directly.
        requested_type: the node's ``model_type`` knob.  ``"auto"`` probes;
            ``"flux"`` / ``"zimage"`` (or the ``"z_image"`` spelling,
            ``src/model_detect.py:62-63``) / ``"krea2"`` pins.

    Returns:
        The profile to patch, or ``None`` for a Nunchaku model — fused kernels
        bypass the attention seam, so the caller logs the warning and returns
        the clone unchanged (the flow already in place at
        ``src/area.py:605-614``; the HAP precedent ``src/hap.py:897-903``).

    Raises:
        ValueError: when the model is a Lumina2 sibling AReA cannot patch
            (pixel space, Ming-Image), or the requested type contradicts the
            model's own backbone — a ``flux``/``zimage`` pin on a Krea-2 model,
            a ``krea2`` pin on a non-Krea-2 one, or any other contradiction.
            Every message names the offender AND the remedy (house rule,
            ``src/hap.py:876-895``).
    """
    mro = _mro_names(diffusion_model)
    detected_type = resolve_model_type(diffusion_model, requested_type)

    if QWEN21_BACKBONE in mro:
        # A Qwen-Image-2.1 backbone (``comfy/ldm/qwen_image21/model.py:234``).
        # The resolver pins it by CLASS NAME (``src/model_detect.py:50-51``), so
        # a 2.1 model resolves to "qwen21" whatever the node asked for — the
        # contradiction has to be read off the REQUESTED string.
        if requested_type not in QWEN21_REQUESTS:
            raise ValueError(
                f"AReA was asked for {requested_type!r} but this model is a "
                "Qwen-Image-2.1 (QwenImage21Transformer2DModel) diffusion model. "
                "Set the node's model_type to 'qwen21' (or leave it on 'auto')."
            )
        return QWEN21_PROFILE

    if KREA_BACKBONE in mro:
        # A Krea-2 backbone (``comfy/ldm/krea2/model.py:234``).  The resolver
        # pins it by CLASS NAME (``src/model_detect.py:45-49``), so a Krea-2
        # model resolves to "krea2" whatever the node asked for — the
        # contradiction has to be read off the REQUESTED string.
        if requested_type not in KREA_REQUESTS:
            raise ValueError(
                f"AReA was asked for {requested_type!r} but this model is a "
                "Krea-2 (SingleStreamDiT) diffusion model. Set the node's "
                "model_type to 'krea2' (or leave it on 'auto')."
            )
        return KREA_PROFILE

    if NEXT_DIT_BACKBONE in mro:
        # A Lumina2-family backbone.  The pixel-space sibling deletes
        # ``final_layer`` and installs a ``dec_net`` decoder head
        # (comfy/ldm/lumina/model.py:1037-1053), so either marker refuses it.
        if PIXEL_SPACE_BACKBONE in mro or hasattr(diffusion_model, "dec_net"):
            raise ValueError(
                "AReA does not support Z-Image Pixel Space (NextDiTPixelSpace, "
                "comfy/ldm/lumina/model.py:1005-1035): its final_layer is "
                "replaced by a dec_net decoder over raw RGB patches, so it has "
                "no 16 px latent patch grid for AReA to pool. Load the "
                "latent-space Z-Image checkpoint (image_model 'lumina2') and "
                "AReA's geometry applies."
            )
        if _is_ming_image(diffusion_model):
            raise ValueError(
                "AReA does not support Ming-Image (masked_pad_multiple, "
                "comfy/ldm/lumina/model.py:684-686): it pads the prompt out to "
                "the mask multiple, so the image tokens no longer start where "
                "the cap ends and AReA cannot slice them out of the joint "
                "stream. Load a plain Z-Image (Lumina2) checkpoint instead."
            )
        if detected_type != "zimage":
            raise ValueError(
                f"AReA was asked for {requested_type!r} but this model is a "
                "Lumina2 (NextDiT) diffusion model, the Z-Image backbone. Set "
                "the node's model_type to 'zimage' (or leave it on 'auto')."
            )
        return ZIMAGE_PROFILE

    # --- not a Lumina2 backbone -------------------------------------------
    if detected_type == "nunchaku":
        return None
    if detected_type == "krea2":
        # The node asked for Krea-2 (an explicit "krea2" pin short-circuits the
        # resolver, src/model_detect.py:54-55) but the backbone is not a
        # SingleStreamDiT — the same class of contradiction as the Z-Image
        # probe below.
        raise ValueError(
            "AReA was asked for 'krea2' but this model's backbone is not a "
            "SingleStreamDiT (comfy/ldm/krea2/model.py:234), so AReA's Krea-2 "
            "geometry (the joint [text | image] stream, the 16 px token grid) "
            "cannot be trusted. Load a Krea-2 checkpoint, or set the node's "
            "model_type to 'flux' / 'zimage' (or leave it on 'auto')."
        )
    if detected_type == "qwen21":
        # The node asked for Qwen-Image-2.1 (an explicit "qwen21" pin
        # short-circuits the resolver, ``src/model_detect.py:58-59``) but the
        # backbone is not a ``QwenImage21Transformer2DModel`` — the same class
        # of contradiction as the Krea-2 probe below.
        raise ValueError(
            "AReA was asked for 'qwen21' but this model's backbone is not a "
            "QwenImage21Transformer2DModel (comfy/ldm/qwen_image21/model.py:234), "
            "so AReA's Qwen-Image-2.1 geometry (the block-causal joint "
            "[text | image] stream, the 16 px token grid) cannot be trusted. "
            "Load a Qwen-Image-2.1 checkpoint, or set the node's model_type to "
            "'flux' / 'zimage' (or leave it on 'auto')."
        )
    if detected_type == "zimage":
        # The rope_embedder probe claims Z-Image but the backbone disagrees —
        # a pin on a non-Lumina2 model, or an upstream rename of NextDiT.
        raise ValueError(
            "AReA resolved this model as Z-Image (the rope_embedder probe, "
            "src/model_detect.py:74-75) but its backbone is not a NextDiT, so "
            "the AReA geometry (patch grid, sequence padding) cannot be "
            "trusted. Re-point AReA's backbone markers (src/area_arch.py) or "
            "use SPA / DyPE / SEGA for this model."
        )
    if detected_type != "flux":
        raise ValueError(
            "AReA supports FLUX, Z-Image, Krea-2 and Qwen-Image-2.1, but the "
            f"resolved model type is {detected_type!r}. Load an unmodified "
            "(non-quantized) FLUX, Z-Image, Krea-2 or Qwen-Image-2.1 model, or "
            "use SPA / DyPE / SEGA for this architecture."
        )
    return FLUX_PROFILE
