import torch
from ..base import DyPEBasePosEmbed

class PosEmbedAnima(DyPEBasePosEmbed):
    def forward(self, x_B_T_H_W_C, fps=None, device=None, dtype=None):
        B, T, H, W, C = x_B_T_H_W_C.shape

        if device is None:
            device = x_B_T_H_W_C.device
        if dtype is None:
            dtype = x_B_T_H_W_C.dtype
        freqs_dtype = torch.bfloat16 if str(device).startswith('cuda') else torch.float32

        t_grid, h_grid, w_grid = torch.meshgrid(
            torch.arange(T, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij',
        )
        pos = torch.stack([t_grid.flatten(), h_grid.flatten(), w_grid.flatten()], dim=-1)

        components = self.get_components(pos, freqs_dtype)

        emb_parts = []
        for cos, sin in components:
            cos_half = cos[..., ::2]
            sin_half = sin[..., ::2]

            col0 = torch.stack([cos_half, sin_half], dim=-1)
            col1 = torch.stack([-sin_half, cos_half], dim=-1)
            matrix = torch.stack([col0, col1], dim=-1)

            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.to(device=device)
