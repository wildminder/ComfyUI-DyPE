import torch
from ..sega_base import SegAPosEmbed


class SegAPosEmbedAnima(SegAPosEmbed):
    """
    SEGA Implementation for Anima/Cosmos models.

    Cosmos uses per-axis NTK factors (t_ntk_factor, h_ntk_factor, w_ntk_factor)
    which result in per-axis theta values. The base class SegAPosEmbed already
    handles per-axis theta via self.thetas[i], so no get_components() override
    is needed for the NTK base.

    SEGA mscale is applied only to the H and W spatial axes (not T).

    Output Format: (T*H*W, D/2, 2, 2) rotation matrices matching Cosmos RoPE format.
    """

    def forward(self, x_B_T_H_W_C, fps=None, device=None, dtype=None):
        B, T, H, W, C = x_B_T_H_W_C.shape

        if device is None:
            device = x_B_T_H_W_C.device
        if dtype is None:
            dtype = x_B_T_H_W_C.dtype
        freqs_dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

        t_grid, h_grid, w_grid = torch.meshgrid(
            torch.arange(T, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
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
