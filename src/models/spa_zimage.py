"""SPA (Spatial Position Alignment) adapter for Z-Image / Lumina 2 models."""
from ..spa import SPABasePosEmbed
from .zimage import PosEmbedZImage


class PosEmbedSPAZImage(SPABasePosEmbed, PosEmbedZImage):
    """Z-Image RoPE embedder with Spatial Position Alignment enabled.

    Note: SPA computes the base (no-extrapolation) RoPE on bundled coords, so
    Z-Image's native DyPE scale hint is not applied in SPA mode.
    """

    _rope_fmt = "flux"
