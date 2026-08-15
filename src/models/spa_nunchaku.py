"""SPA (Spatial Position Alignment) adapter for Nunchaku (quantized FLUX) models."""
from ..spa import SPABasePosEmbed
from .nunchaku import PosEmbedNunchaku


class PosEmbedSPANunchaku(SPABasePosEmbed, PosEmbedNunchaku):
    """Nunchaku RoPE embedder with Spatial Position Alignment enabled.

    SPA is unsupported on Nunchaku (fused/quantized kernels bypass the hook), so
    ``apply_spa_to_model`` returns the model unchanged when model_type is nunchaku.
    The format is declared for completeness only.
    """

    _rope_fmt = "nunchaku"
