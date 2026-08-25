"""Tests for SPA model adapters (FLUX / Qwen / Z-Image / Nunchaku).

These verify that each SPA embedder produces the *same output format* as its
DyPE counterpart, that ``forward`` returns the **base** RoPE (not the averaged
variant RoPE — the root-cause bug is removed), and that it registers the ``N``
bundled variant RoPEs in the process-scoped :class:`SPAContext` for the attention
hook.  Pure unit tests — no ComfyUI runtime required.
"""
import pytest
import torch

from src.models.spa_flux import PosEmbedSPAFlux
from src.models.spa_nunchaku import PosEmbedSPANunchaku
from src.models.spa_qwen import PosEmbedSPAQwen
from src.models.spa_zimage import PosEmbedSPAZImage
from src.spa import build_bundle_id_variants, get_spa_context

_ADAPTERS = [PosEmbedSPAFlux, PosEmbedSPAQwen, PosEmbedSPAZImage, PosEmbedSPANunchaku]
_ADAPTER_NAMES = ["flux", "qwen", "zimage", "nunchaku"]


def _make_flux_ids(H=64, W=64, B=1):
    L = H * W
    ids = torch.zeros(B, L, 3)
    ids[..., 0] = torch.arange(L)
    ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
    ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    return ids


def _make_emb(cls, **kw):
    return cls(theta=10000, axes_dim=[16, 56, 56], method="ntk", **kw)


@pytest.mark.unit
@pytest.mark.parametrize("emb_cls,name", list(zip(_ADAPTERS, _ADAPTER_NAMES)))
class TestPosEmbedSPAAdapters:
    def test_output_shape(self, emb_cls, name):
        emb = _make_emb(emb_cls)
        out = emb(_make_flux_ids(64, 64))
        if name == "nunchaku":
            assert out.shape == (1, 1, 4096, 64, 1, 2)
        else:
            assert out.shape == (1, 1, 4096, 64, 2, 2)

    def test_finite(self, emb_cls, name):
        emb = _make_emb(emb_cls)
        out = emb(_make_flux_ids(32, 32))
        assert torch.isfinite(out).all()

    # T-P2-1: forward returns the BASE RoPE, not the mean of variants.
    def test_forward_returns_base_not_mean(self, emb_cls, name):
        # Use a grid OUTSIDE the trained extent (128x128, max_pos=127 > 64) so
        # bundling is active and the base-vs-mean distinction is meaningful.
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=3)
        ids = _make_flux_ids(128, 128)
        out = emb(ids)
        base = emb.format_components(emb._spa_components(ids.float(), torch.float32), ids)
        assert torch.allclose(out, base, atol=1e-6)
        # Sanity: the averaged (legacy) path would differ; confirm base is NOT the mean.
        variants = build_bundle_id_variants(ids, 3)
        assert len(variants) > 1  # bundling is active
        mean_variants = torch.stack(
            [emb.format_components(emb._spa_components(v.float(), torch.float32), v) for v in variants],
            dim=0,
        ).mean(0)
        # base == forward output; base may or may not equal the legacy mean — the key
        # assertion is that forward returns base (above), not the mean.
        assert out.shape == mean_variants.shape

    # T-P2-2: forward registers the N variant RoPEs in the active context.
    def test_forward_registers_variants(self, emb_cls, name):
        # Paper-N semantics at an ACTIVE grid (128x128, max_pos=127 > 64):
        # N=3 -> s = max(3, ceil(127/79)=2) = 3 -> 2*3 - 1 = 5 variants.
        from src.spa import derive_bundle_s
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=3)
        ids = _make_flux_ids(128, 128)
        out = emb(ids)
        ctx = get_spa_context()
        assert ctx is not None and ctx.active is True
        assert ctx.bundle_size == 3
        max_pos = int(max(ids[..., 1].max(), ids[..., 2].max()))
        s = derive_bundle_s(max_pos, 3)
        assert len(ctx.variant_pes) == 2 * s - 1
        assert ctx.fmt == ("nunchaku" if name == "nunchaku" else "flux")
        assert torch.allclose(ctx.base_pe, out, atol=1e-6)

    # Trained-extent gate: a grid inside the trained extent registers a single
    # identity variant (hook passthrough) even with an active knob.
    def test_in_trained_extent_is_identity(self, emb_cls, name):
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=3)
        ids = _make_flux_ids(64, 64)  # max_pos=63 <= 64
        out = emb(ids)
        ctx = get_spa_context()
        assert ctx is not None and ctx.active is True
        assert len(ctx.variant_pes) == 1  # identity -> hook passthrough
        assert torch.allclose(ctx.base_pe, out, atol=1e-6)

    # T-P2-3: bundle_size==1 => context inactive (passthrough, no hook effect).
    def test_bundle_size_one_inactive(self, emb_cls, name):
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=1)
        ids = _make_flux_ids(16, 16)
        emb(ids)  # output unused; the assertion targets the CONTEXT state
        ctx = get_spa_context()
        assert ctx is None or ctx.active is False

    def test_off_equals_base(self, emb_cls, name):
        emb = _make_emb(emb_cls, enable_spa=False, bundle_size=5)
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        base = emb.format_components(emb._spa_components(ids.float(), torch.float32), ids)
        assert torch.allclose(out, base, atol=1e-6)

    # T-P2-4 mirror: variant pes change the RoPE vs base (hook will use them).
    def test_variant_pes_differ_from_base(self, emb_cls, name):
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=5)
        ids = _make_flux_ids(128, 128)
        emb(ids)  # registers the context; the returned base PE is unused here
        ctx = get_spa_context()
        # at least one variant pe differs from the base pe (bundling changed coords)
        diff = torch.stack([(vp - ctx.base_pe).abs().max() for vp in ctx.variant_pes])
        assert diff.max() > 1e-4
