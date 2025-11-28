import torch
from ..base import DyPEBasePosEmbed

class PosEmbedQwen(DyPEBasePosEmbed):
    """
    DyPE Implementation for Qwen Image Models.
    
    Qwen's `apply_rope1` expects `D/2` frequency matrices (corresponding to coordinate pairs)
    rather than the `D` repeated values produced by standard RoPE logic.
    
    Output Format: (B, 1, L, D/2, 2, 2)
    This broadcasts correctly against input (B, H, L, D/2, 1, 2).
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32
        
        components = self.get_components(pos, freqs_dtype)
        
        emb_parts = []
        for cos, sin in components:
            # cos: (B, L, D) (interleaved [c0, c0, c1, c1...])
            # Decimate to (B, L, D/2) -> [c0, c1...]
            cos_half = cos[..., ::2]
            sin_half = sin[..., ::2]
            
            # Construct Rotation Matrix Columns for apply_rope1
            # F0: [cos, sin]
            col0 = torch.stack([cos_half, sin_half], dim=-1) # D/2, 2
            # F1: [-sin, cos]
            col1 = torch.stack([-sin_half, cos_half], dim=-1) # D/2, 2
            
            # Stack columns -> (D/2, 2, 2)
            # Last dim is the selector [0, 1] used by apply_rope1
            matrix = torch.stack([col0, col1], dim=-1)
            
            emb_parts.append(matrix)
            
        emb = torch.cat(emb_parts, dim=-3) # Concatenate along Feature Dimension (D/2)
        
        out = emb.unsqueeze(1).to(ids.device) # (B, 1, L, Total_D/2, 2, 2)
        return out