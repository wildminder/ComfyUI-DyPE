"""SPA (Spatial Position Alignment) adapter for Qwen-Image-2.1.

Subclasses :class:`PosEmbedQwen21` rather than copying it, so 2.1's single
load-bearing difference from 1.0 — the frequency dtype — is stated ONCE per
adapter instead of drifting between the DyPE, SEGA and SPA code paths.
"""
import torch

from ..spa import SPABasePosEmbed
from .qwen21 import PosEmbedQwen21


class PosEmbedSPAQwen21(SPABasePosEmbed, PosEmbedQwen21):
    """Qwen-Image-2.1 RoPE embedder with Spatial Position Alignment enabled.

    ``format_components`` resolves through :class:`PosEmbedQwen21` ->
    :class:`PosEmbedQwen`: 2.1 uses the same ``EmbedND`` position embedder and
    the same ``D/2`` rotation-matrix layout as 1.0, so nothing about the
    formatting changes.  ``SPABasePosEmbed.forward`` wins in the MRO (it returns
    the base RoPE and registers the bundled variants in the
    :class:`~src.spa_context.SPAContext`), and ``total_len`` is recorded at
    registration so the shared unet wrapper's ``causal_prefix`` joint mode can
    recover the block-causal segment bounds.

    Output Format: (B, 1, L, D/2, 2, 2)
    """

    _rope_fmt = "flux"

    def _freqs_dtype(self, pos: torch.Tensor) -> torch.dtype:
        """Always fp32 — 2.1 hands ``pe`` to its own RoPE kernel uncast.

        2.1's call site (``comfy/ldm/qwen_image21/model.py``) does
        ``pe = self.pe_embedder(...).transpose(1, 2).contiguous()`` with **no**
        ``.to(x.dtype)``, unlike 1.0, and feeds the result straight to the fused
        ``comfy.quant_ops.ck.rms_rope`` / ``apply_rope1``.  Building the
        cos/sin frequencies in bfloat16 on CUDA (the base rule) would silently
        change the dtype 2.1's kernel receives, so this adapter pins fp32 on
        every device.
        """
        return torch.float32