"""SPA (Spatial Position Alignment) adapter for standard ComfyUI FLUX models.

Inherits the FLUX rotation-matrix formatting from :class:`PosEmbedFlux` and the
SPA averaging logic from :class:`SPABasePosEmbed`.
"""
from ..spa import SPABasePosEmbed
from .flux import PosEmbedFlux


class PosEmbedSPAFlux(SPABasePosEmbed, PosEmbedFlux):
    """FLUX RoPE embedder with Spatial Position Alignment enabled."""

    _rope_fmt = "flux"
