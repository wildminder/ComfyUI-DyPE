import torch

from .sega_qwen import SegAPosEmbedQwen


class SegAPosEmbedQwen21(SegAPosEmbedQwen):
    """SEGA implementation for Qwen-Image-2.1.

    The RoPE layout and axis order are identical to 1.0 — ``build_sequence``
    emits ``torch.stack([pos, hh, ww])`` (``comfy/ldm/qwen_image21/model.py``),
    so ``SegAPosEmbed._compute_per_dim_mscale``'s H/W axis mapping applies
    unchanged and ``format_components`` is inherited verbatim.

    Only the frequency dtype differs: 2.1 does not cast the embedder output to
    the activation dtype at its call site (1.0 does), so the fp32 output is
    what its kernel expects.  See :mod:`src.models.qwen21` for the full
    reasoning.

    Output Format: (B, 1, L, D/2, 2, 2)
    """

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.format_components(self.get_components(ids.float(), torch.float32), ids)