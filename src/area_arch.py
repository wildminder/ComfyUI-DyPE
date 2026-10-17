"""AReA architecture profiles — which backbone a given model actually is.

AReA is a *model-side* attention replacement, so apply time starts with one
question: which diffusion backbone is this?  The engine (``src/area_attn.py``,
``src/area_schedule.py``) is backbone-agnostic — it pools keys, builds windows
and reads rope tables — and every FLUX/Z-Image difference lives in the adapter
that feeds it.  :class:`_ArchProfile` is that difference in five fields,
resolved once (plan §4, Q7/Q9).

**Two backbones, one node.** FLUX (``Flux``, ``comfy/ldm/flux/model.py``) and
Z-Image (``NextDiT``, ``comfy/ldm/lumina/model.py:442``, bound by
``Lumina2.unet_model`` at ``comfy/model_base.py:1510``) share today's geometry:
the Flux VAE downscales 8x and both patchify 2x2, so one token covers **16 px**
on either backbone (``src/area.py:109-112``) and the node's multiple-of-16
validation covers both.  They differ where the sequence is padded: Z-Image pads
cap AND image tokens to a multiple of 32
(``comfy/ldm/lumina/model.py:418-420``, applied at ``:665`` and ``:725-727``),
FLUX pads to nothing.

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

Nunchaku (quantized, fused kernels) is neither: fused attention bypasses the
seam entirely, so it is warned about and returned **unchanged** by the caller
(``src/area.py:605-614``, the HAP precedent at ``src/hap.py:897-903``) instead
of being refused here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .model_detect import resolve_model_type

__all__ = [
    "FLUX_BACKBONE",
    "FLUX_PROFILE",
    "NEXT_DIT_BACKBONE",
    "PIXEL_SPACE_BACKBONE",
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

    model_type: str           # "flux" | "zimage" — ``resolve_model_type``'s key
    unet_backbone: str        # class name: "Flux" | "NextDiT"
    px_per_patch: int         # 16 on both (VAE 8 x patchify 2)
    seq_pad_multiple: int     # 0 (flux) | 32 (zimage) — see the module docstring
    accepts: str              # human-readable model match string

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
            ``src/model_detect.py:62-63``) pins.

    Returns:
        The profile to patch, or ``None`` for a Nunchaku model — fused kernels
        bypass the attention seam, so the caller logs the warning and returns
        the clone unchanged (the flow already in place at
        ``src/area.py:605-614``; the HAP precedent ``src/hap.py:897-903``).

    Raises:
        ValueError: when the model is a Lumina2 sibling AReA cannot patch
            (pixel space, Ming-Image), or the requested type contradicts the
            model's own backbone.  Every message names the offender AND the
            remedy (house rule, ``src/hap.py:876-895``).
    """
    mro = _mro_names(diffusion_model)
    detected_type = resolve_model_type(diffusion_model, requested_type)

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
            "AReA supports FLUX and Z-Image, but the resolved model type is "
            f"{detected_type!r}. Load an unmodified (non-quantized) FLUX model, "
            "or use SPA / DyPE / SEGA for this architecture."
        )
    return FLUX_PROFILE
