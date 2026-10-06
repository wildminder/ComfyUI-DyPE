import torch

from .qwen import PosEmbedQwen


class PosEmbedQwen21(PosEmbedQwen):
    """DyPE implementation for Qwen-Image-2.1.

    Structurally identical to 1.0 — same ``EmbedND`` position embedder, same
    ``D/2`` rotation-matrix layout, so ``format_components`` is inherited
    verbatim.  The one behavioural difference is the frequency dtype.

    1.0 casts the embedder output back to the activation dtype at its call
    site (``comfy/ldm/qwen_image/model.py``: ``self.pe_embedder(ids).to(x.dtype)``),
    while 2.1 does **not** (``comfy/ldm/qwen_image21/model.py``: ``pe =
    self.pe_embedder(...).transpose(1, 2).contiguous()``) — the fp32 embedder
    output is handed straight to ``comfy.quant_ops.ck.rms_rope``.  Downcasting
    to bfloat16 on CUDA (what ``PosEmbedQwen.forward`` does) would therefore
    change the dtype 2.1 feeds its own kernel, so 2.1 keeps fp32 end to end.

    Output Format: (B, 1, L, D/2, 2, 2)
    """

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.format_components(self.get_components(ids.float(), torch.float32), ids)