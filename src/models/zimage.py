import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / Lumina 2 Models.
    
    Z-Image uses the 'NextDiT' architecture which shares the same RoPE implementation
    details (comfy.ldm.flux.math.apply_rope) as Standard Flux.
    It expects Rotation Matrices concatenated in the last dimension.
    
    Output Format: (B, 1, L, D) -> Unqueezed to (B, 1, L, D, 2) via formatting
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32
        
        components = self.get_components(pos, freqs_dtype)
        
        emb_parts = []
        for cos, sin in components:
            # Format as Rotation Matrix [[cos, -sin], [sin, cos]]
            # Cos/Sin input shape: (..., D)
            
            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)
            
        emb = torch.cat(emb_parts, dim=-3)
        
        return emb.unsqueeze(1).to(ids.device)