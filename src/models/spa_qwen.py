"""SPA (Spatial Position Alignment) adapter for Qwen Image models."""
from ..spa import SPABasePosEmbed
from .qwen import PosEmbedQwen


class PosEmbedSPAQwen(SPABasePosEmbed, PosEmbedQwen):
    """Qwen RoPE embedder with Spatial Position Alignment enabled."""

    _rope_fmt = "flux"
