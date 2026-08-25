"""Tests for the SPA Anima/Cosmos adapter (3D RoPE, bundle h/w, keep t)."""
import pytest
import torch

from src.models.spa_anima import PosEmbedSPAAnima
from src.spa import build_bundle_id_variants, get_spa_context


@pytest.mark.unit
class TestPosEmbedSPAAnima:
    def _make(self, T=1, H=8, W=8, C=128):
        return torch.randn(1, T, H, W, C)

    def _pos(self, T, H, W, device="cpu"):
        t_grid, h_grid, w_grid = torch.meshgrid(
            torch.arange(T, device=device, dtype=torch.float32),
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        return torch.stack([t_grid.flatten(), h_grid.flatten(), w_grid.flatten()], dim=-1)

    def test_output_shape(self):
        emb = PosEmbedSPAAnima(theta=[10000.0, 10000.0, 10000.0], axes_dim=[44, 42, 42],
                               method="vision_yarn")
        x = self._make(T=1, H=8, W=8, C=128)
        out = emb(x)
        assert out.shape == (64, 64, 2, 2)

    def test_finite(self):
        emb = PosEmbedSPAAnima(theta=[10000.0, 10000.0, 10000.0], axes_dim=[44, 42, 42],
                               method="vision_yarn")
        out = emb(self._make(T=2, H=16, W=16, C=128))
        assert torch.isfinite(out).all()

    # T-P2-1: forward returns the BASE RoPE (not the averaged variant RoPE).
    # Uses a 128x128 grid so the trained-extent gate (max_pos > 64) is active.
    def test_forward_returns_base(self):
        emb = PosEmbedSPAAnima(theta=[10000.0, 10000.0, 10000.0], axes_dim=[44, 42, 42],
                               method="vision_yarn", enable_spa=True, bundle_size=3)
        x = self._make(T=1, H=128, W=128, C=128)
        out = emb(x)
        pos = self._pos(1, 128, 128)
        base = emb.format_components(emb._spa_components(pos, torch.float32), pos)
        assert torch.allclose(out, base, atol=1e-6)

    # T-P2-2: forward registers the N variant RoPEs in the active context.
    # Uses a 128x128 grid so the trained-extent gate (max_pos > 64) is active.
    def test_forward_registers_variants(self):
        from src.spa import derive_bundle_s
        emb = PosEmbedSPAAnima(theta=[10000.0, 10000.0, 10000.0], axes_dim=[44, 42, 42],
                               method="vision_yarn", enable_spa=True, bundle_size=3)
        x = self._make(T=1, H=128, W=128, C=128)
        out = emb(x)
        ctx = get_spa_context()
        assert ctx is not None and ctx.active is True
        assert ctx.bundle_size == 3
        # Paper-N semantics: #variants = 2*s - 1, with s = derive_bundle_s(max_pos, N).
        # For 128x128, N=3: max_pos=127 > 64 (trained extent), s = max(3, ceil(127/79)) = 3
        # -> 2*3 - 1 = 5 variants.
        pos = self._pos(1, 128, 128)
        s = derive_bundle_s(int(pos[..., 1:].max()), 3, emb.trained_extent)
        assert len(ctx.variant_pes) == 2 * s - 1
        assert ctx.fmt == "anima"
        assert torch.allclose(ctx.base_pe, out, atol=1e-6)

    # T-P2-3: bundle_size==1 => context inactive.
    def test_bundle_size_one_inactive(self):
        emb = PosEmbedSPAAnima(theta=[10000.0, 10000.0, 10000.0], axes_dim=[44, 42, 42],
                               method="vision_yarn", enable_spa=True, bundle_size=1)
        x = self._make(T=1, H=16, W=16, C=128)
        emb(x)  # output unused; the assertion targets the CONTEXT state
        ctx = get_spa_context()
        assert ctx is None or ctx.active is False

    # T-P2-4: Anima keeps the temporal axis + per-axis NTK; variant pes preserve t.
    # Uses a 128x128 grid so the trained-extent gate (max_pos > 64) is active.
    def test_temporal_axis_unchanged(self):
        emb = PosEmbedSPAAnima(theta=[10000.0, 10000.0, 10000.0], axes_dim=[44, 42, 42],
                               method="vision_yarn", enable_spa=True, bundle_size=3)
        T, H, W = 4, 128, 128
        x = self._make(T=T, H=H, W=W, C=128)
        emb(x)
        ctx = get_spa_context()
        pos = self._pos(T, H, W)
        variants = build_bundle_id_variants(pos, emb.bundle_size, emb.trained_extent)
        for v in variants:
            assert torch.equal(v[..., 0], pos[..., 0])  # temporal axis untouched

        # Temporal RoPE blocks (first axes_dim[0]//2 = 22 blocks) are identical
        # across all variants and equal the base pe's temporal blocks.
        t_blocks = emb.axes_dim[0] // 2
        for vp in ctx.variant_pes:
            assert torch.allclose(vp[..., :t_blocks, :, :], ctx.base_pe[..., :t_blocks, :, :], atol=1e-6)

    def test_off_equals_base(self):
        emb = PosEmbedSPAAnima(theta=[10000.0, 10000.0, 10000.0], axes_dim=[44, 42, 42],
                               method="vision_yarn", enable_spa=False, bundle_size=5)
        x = self._make(T=1, H=8, W=8, C=128)
        out = emb(x)
        pos = self._pos(1, 8, 8)
        base = emb.format_components(emb._spa_components(pos, torch.float32), pos)
        assert torch.allclose(out, base, atol=1e-6)
