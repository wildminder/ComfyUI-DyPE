import torch
from ..base import DyPEBasePosEmbed

class PosEmbedNunchaku(DyPEBasePosEmbed):
    """
    DyPE Implementation for Nunchaku Flux Models.
    Output Format: Specific tensor layout for Nunchaku's custom RoPE kernel.
    Shape: (B, M, D//2, 1, 2)
    """
    def _axis_rope_from_cos_sin(self, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """
        Convert cos/sin with shape (..., M, D) into rope shape (B, M, D//2, 1, 2).
        """
        if cos.dim() < 2:
            raise RuntimeError(f"Unexpected cos shape {tuple(cos.shape)}")

        # shape components
        *lead, M, D = list(cos.shape)
        if lead:
            batch = int(torch.prod(torch.tensor(lead, dtype=torch.int64)).item())
            cos_flat = cos.reshape(batch, M, D)
            sin_flat = sin.reshape(batch, M, D)
        else:
            batch = 1
            cos_flat = cos.reshape(1, M, D)
            sin_flat = sin.reshape(1, M, D)

        assert D % 2 == 0, "rotary dimension must be even"
        D_half = D // 2

        # cos/sin were produced with repeat_interleave(2) style: [..., d0, d0, d1, d1, ...]
        # view pairs and take the first of each pair.
        cos_pairs = cos_flat.view(batch, M, D_half, 2)
        sin_pairs = sin_flat.view(batch, M, D_half, 2)
        cos_out = cos_pairs[..., 0]  # (batch, M, D_half)
        sin_out = sin_pairs[..., 0]  # (batch, M, D_half)

        stacked = torch.stack([sin_out, cos_out], dim=-1)  # (batch, M, D_half, 2)
        rope = stacked.view(batch, M, D_half, 1, 2).contiguous()  # (batch, M, D_half, 1, 2)

        if lead:
            restore_shape = (*lead, M, D_half, 1, 2)
            rope = rope.reshape(restore_shape)

        return rope.float()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        added_batch = False
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
            added_batch = True

        pos = ids.float()
        freqs_dtype = torch.float32
        
        components = self.get_components(pos, freqs_dtype)
        
        emb_parts = []
        for cos, sin in components:
            rope_i = self._axis_rope_from_cos_sin(cos, sin)
            emb_parts.append(rope_i)

        # Concatenate along axis dimension: dim = -3
        # shape: (B, M, D_total//2, 1, 2)
        emb = torch.cat(emb_parts, dim=-3)  

        out = emb.unsqueeze(1).to(ids.device)
        return out