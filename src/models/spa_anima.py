"""SPA (Spatial Position Alignment) adapter for Anima / Cosmos video models.

Anima uses a 3D RoPE (t, h, w) built from a ``(B, T, H, W, C)`` grid rather than
an ``(B, L, 3)`` id tensor, so its SPA ``forward`` bundles the spatial axes
(``h``, ``w``) while keeping the temporal axis ``t`` untouched.  The native per-axis
NTK factors are preserved because they are baked into ``self.theta`` and
``_spa_components`` applies ``ntk_factor = 1.0`` on top of them.

The legacy ``torch.stack(embs).mean(0)`` path is removed (root-cause bug); the
embedder returns the *base* RoPE and registers the bundled variant RoPEs in the
process-scoped :class:`SPAContext` for the attention hook.
"""
import torch

from ..spa import SPABasePosEmbed, build_bundle_id_variants
from ..spa_attn import compose_rope, inv_rope
from ..spa_context import SPAContext, set_spa_context
from .anima import PosEmbedAnima


class PosEmbedSPAAnima(SPABasePosEmbed, PosEmbedAnima):
    """Anima/Cosmos RoPE embedder with Spatial Position Alignment enabled."""

    _rope_fmt = "anima"

    def _build_pos(self, x_B_T_H_W_C: torch.Tensor) -> torch.Tensor:
        B, T, H, W, C = x_B_T_H_W_C.shape
        device = x_B_T_H_W_C.device
        t_grid, h_grid, w_grid = torch.meshgrid(
            torch.arange(T, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack(
            [t_grid.flatten(), h_grid.flatten(), w_grid.flatten()], dim=-1
        )

    def forward(self, x_B_T_H_W_C, fps=None, device=None, dtype=None):
        B, T, H, W, C = x_B_T_H_W_C.shape

        if device is None:
            device = x_B_T_H_W_C.device
        if dtype is None:
            dtype = x_B_T_H_W_C.dtype
        fdtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32

        pos = self._build_pos(x_B_T_H_W_C)

        if (not self.enable_spa) or self.bundle_size == 1:
            # Identity: base (no-extrapolation) RoPE, no hook effect.
            # (N == 1 is "off"; N == 0 is "auto" and stays active.)
            set_spa_context(None)
            return self.format_components(self._spa_components(pos, fdtype), pos).to(device=device)

        # Return the BASE RoPE; register the variant RoPEs (bundle h,w, keep t).
        base = self.format_components(self._spa_components(pos, fdtype), pos)
        self._register_variants(x_B_T_H_W_C)
        return base.to(device=device)

    def _register_variants(self, x_B_T_H_W_C: torch.Tensor) -> None:
        """Register Anima variant RoPEs (bundle h,w, keep t) for the hook."""
        device = x_B_T_H_W_C.device
        fdtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        pos = self._build_pos(x_B_T_H_W_C)

        base_components = self._spa_components(pos, fdtype)
        base_pe = self.format_components(base_components, pos)

        # build_bundle_id_variants bundles the spatial axes (cols 1,2 = h,w)
        # and leaves the temporal axis (col 0 = t) untouched.
        variants = build_bundle_id_variants(pos, self.bundle_size, self.trained_extent)
        variant_pes = [
            self.format_components(self._spa_components(v, fdtype), v)
            for v in variants
        ]

        # P3 (D5 fix): compose the static delta rotations once per forward so the
        # attention hook consumes them directly instead of recomposing per call.
        inv_base = inv_rope(base_pe, self._rope_fmt)
        variant_deltas = [compose_rope(inv_base, vp, self._rope_fmt) for vp in variant_pes]

        ctx = SPAContext(
            active=True,
            bundle_size=self.bundle_size,
            base_pe=base_pe,
            variant_pes=variant_pes,
            pre_roped=True,
            fmt=self._rope_fmt,
            model_key=id(self),
            variant_deltas=variant_deltas,
        )
        set_spa_context(ctx)
