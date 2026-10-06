"""VAE image-channel adaptation for the cascade nodes (Qwen-Image 2.1).

Qwen-Image 2.1 ships a 16x RGBA Wan-2.2-layout VAE (comfy/sd.py:841-847):
``downscale_ratio`` is a plain ``16``, ``output_channels`` is read from the
checkpoint tensor ``decoder.head.2.weight`` (so it is *4* for the 2.1 VAE, not
a constant this pack can hardcode), ``pad_channel_value`` is 1.0 ("opaque
alpha for RGB input") and the model is a ``temporal_kernel=1`` WanVAE whose
``latent_dim`` stays 2. ``VAE.decode`` therefore hands back ``[B, H, W, 4]``.

Every cascade node in this pack speaks the 3-channel convention: the layout
probes in ``nodes/pixelrush.py`` / ``nodes/freescale.py`` match on exactly 3,
``_sharpen`` (nodes/hiflow.py) treats channels-last as ``shape[-1] == 3``, and
every downstream consumer (resize, compositing, ``SaveImage``) is RGB. An
RGBA decode therefore breaks the layout probe and hands the rest of the
pipeline a 4-channel image it was never written for.

Two pure helpers normalize the boundary. Both are exact no-ops unless the VAE
reports 4 output channels, so the other six architectures are unchanged.

``pad_to_vae_channels`` deliberately reproduces ComfyUI's own
``vae_encode_crop_pixels`` (comfy/sd.py:1153-1164) — truncate an over-wide
image, pad an under-wide one with ``pad_channel_value``. Being idempotent with
core means the helper can never change what a real ComfyUI VAE receives; it
exists so the three adapters share one decision instead of three copies of it.

Alpha is NOT carried end to end through the cascades: every intermediate
consumer here is RGB, and preserving it would change the meaning of the IMAGE
outputs of three nodes. Dropping it is loud rather than silent — see
:func:`strip_alpha_channel``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("ComfyUI-DyPE")

RGB_CHANNELS = 3
RGBA_CHANNELS = 4

# Latched once per process (the repo's "one warning per run" idiom — see
# src/spa.py:589-594 for why a silent no-op is the failure mode to avoid).
# An upscale decodes the VAE once per guided stage, so an unlatched log would
# repeat dozens of times per generation.
_alpha_drop_logged = False


def vae_image_channels(vae) -> int:
    """Number of image channels the VAE's decoder emits.

    ``getattr`` with the ComfyUI 3-channel default: a mock, a wrapper node or
    an exotic VAE without the attribute must behave like a standard RGB VAE,
    never like an RGBA one.
    """
    return int(getattr(vae, "output_channels", RGB_CHANNELS))


def is_rgba_vae(vae) -> bool:
    """True for a VAE whose decode output carries an alpha channel."""
    return vae_image_channels(vae) == RGBA_CHANNELS


def strip_alpha_channel(
    image: torch.Tensor, vae, node_name: str = "VAE"
) -> torch.Tensor:
    """Drop the alpha channel from a decoded RGBA image (``[..., :3]``).

    Returns ``image`` untouched for every non-RGBA VAE and for images that are
    already 3 channels — the identity for the other six architectures is the
    whole point of gating on ``output_channels`` rather than on shape.

    The one-time INFO names the node that dropped alpha, because a user who
    composites over transparency will otherwise see it vanish without
    explanation (alpha never survives ``vae.encode`` round-trips into these
    cascades either).
    """
    global _alpha_drop_logged
    if not is_rgba_vae(vae) or image.shape[-1] != RGBA_CHANNELS:
        return image
    if not _alpha_drop_logged:
        _alpha_drop_logged = True
        logger.info(
            "%s: this VAE decodes RGBA (output_channels=4) — dropping the "
            "alpha channel and running the cascade in RGB (logged once per "
            "session). PixelRush/FreeScale/HiFlow always return 3-channel "
            "images; alpha is not preserved.",
            node_name,
        )
    return image[..., :RGB_CHANNELS]


def pad_to_vae_channels(image: torch.Tensor, vae) -> torch.Tensor:
    """Pad/truncate ``image`` to the VAE's expected channel count.

    Mirrors ``VAE.vae_encode_crop_pixels`` (comfy/sd.py:1153-1164): a 3-channel
    image going into an RGBA VAE is padded with ``pad_channel_value`` (1.0 =
    opaque alpha, the value ComfyUI itself uses for RGB input), and an
    over-wide image is truncated. A VAE that states no pad value is left alone,
    exactly as core leaves it — the VAE's own conv then raises a shape error
    naming the real problem, instead of this helper inventing a value.
    """
    channels = vae_image_channels(vae)
    have = image.shape[-1]
    if have == channels:
        return image
    if have > channels:
        return image[..., :channels]

    pad_value = getattr(vae, "pad_channel_value", None)
    if pad_value is None:
        return image
    if isinstance(pad_value, str):
        # ComfyUI's own convention (sd.py:1157-1163): a string is a replicate
        # MODE, not a value. Replicating a channel is a cat, not an F.pad —
        # F.pad's replicate mode pads spatial dims, never the channel axis.
        edge = image[..., -1:]
        return torch.cat(
            [image] + [edge] * (channels - have), dim=-1,
        )
    return torch.nn.functional.pad(
        image, (0, channels - have), mode="constant", value=pad_value,
    )
