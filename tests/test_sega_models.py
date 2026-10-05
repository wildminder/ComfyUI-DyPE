"""Tests for SEGA model-specific embedders (Tier 1: pure unit tests)."""
import pytest
import torch

from src.models.sega_anima import SegAPosEmbedAnima
from src.models.sega_flux import SegAPosEmbedFlux
from src.models.sega_nunchaku import SegAPosEmbedNunchaku
from src.models.sega_qwen import SegAPosEmbedQwen
from src.models.sega_qwen21 import SegAPosEmbedQwen21
from src.models.sega_zimage import SegAPosEmbedZImage


def _make_flux_ids(H=64, W=64):
    """Create FLUX-style position IDs: (L, 3) with [seq, h, w]."""
    L = H * W
    ids = torch.zeros(L, 3)
    ids[:, 0] = torch.arange(L)
    ids[:, 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1)
    ids[:, 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1)
    return ids


def _make_anima_input(T=1, H=64, W=64, C=128):
    """Create Anima-style input: (B, T, H, W, C)."""
    return torch.randn(1, T, H, W, C)


# ---------------------------------------------------------------------------
# SegAPosEmbedFlux
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSegAPosEmbedFlux:
    def test_output_shape(self):
        emb = SegAPosEmbedFlux(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        # (L, 1, D/2, 2, 2) — unsqueeze(1) adds dim at position 1
        L = 32 * 32
        assert out.shape[0] == L
        assert out.shape[1] == 1
        assert out.shape[2] == 64  # (16 + 56 + 56) / 2
        assert out.shape[3] == 2
        assert out.shape[4] == 2

    def test_no_spectral_data_works(self):
        """Without spectral data, should still produce valid output."""
        emb = SegAPosEmbedFlux(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        assert not torch.isnan(out).any()

    def test_with_spectral_data(self):
        """With spectral data and extrapolation, output should differ from no-spectral."""
        emb_no_spec = SegAPosEmbedFlux(theta=10000, axes_dim=[16, 56, 56], training_res_pixels=1024)
        emb_with_spec = SegAPosEmbedFlux(theta=10000, axes_dim=[16, 56, 56], training_res_pixels=1024)

        eh = torch.rand(16) * 10 + 1
        ew = torch.rand(16) * 10 + 1
        emb_with_spec.set_spectral_data(eh, ew, 0.8, target_res_h=4096, target_res_w=4096)

        ids = _make_flux_ids(128, 128)  # 2x extrapolation
        out_no = emb_no_spec(ids)
        out_with = emb_with_spec(ids)
        assert not torch.allclose(out_no, out_with, atol=1e-4)

    def test_inherits_sega_base(self):
        from src.sega_base import SegAPosEmbed
        emb = SegAPosEmbedFlux(theta=10000, axes_dim=[16, 56, 56])
        assert isinstance(emb, SegAPosEmbed)


# ---------------------------------------------------------------------------
# SegAPosEmbedQwen
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSegAPosEmbedQwen:
    def test_output_shape(self):
        emb = SegAPosEmbedQwen(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        # (L, 1, D/2, 2, 2) — unsqueeze(1) adds dim at position 1
        L = 32 * 32
        assert out.shape[0] == L
        assert out.shape[1] == 1
        assert out.shape[2] == 64  # (16 + 56 + 56) / 2
        assert out.shape[3] == 2
        assert out.shape[4] == 2

    def test_no_nan(self):
        emb = SegAPosEmbedQwen(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        assert not torch.isnan(out).any()

    def test_with_spectral_data(self):
        emb = SegAPosEmbedQwen(theta=10000, axes_dim=[16, 56, 56], training_res_pixels=1024)
        eh = torch.rand(16) * 10 + 1
        ew = torch.rand(16) * 10 + 1
        emb.set_spectral_data(eh, ew, 0.8, target_res_h=4096, target_res_w=4096)
        ids = _make_flux_ids(128, 128)
        out = emb(ids)
        assert out.shape[0] == 128 * 128


# ---------------------------------------------------------------------------
# SegAPosEmbedQwen21
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSegAPosEmbedQwen21:
    """Qwen-Image-2.1 — 1.0's layout and axis order, fp32 frequencies."""

    def test_output_shape(self):
        emb = SegAPosEmbedQwen21(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        L = 32 * 32
        assert out.shape[0] == L
        assert out.shape[1] == 1
        assert out.shape[2] == 64  # (16 + 56 + 56) / 2
        assert out.shape[3] == 2
        assert out.shape[4] == 2

    def test_no_nan(self):
        emb = SegAPosEmbedQwen21(theta=10000, axes_dim=[16, 56, 56])
        out = emb(_make_flux_ids(32, 32))
        assert not torch.isnan(out).any()

    def test_inherits_qwen_formatting(self):
        """No format duplication: the 1.0 formatter is reused verbatim."""
        from src.sega_base import SegAPosEmbed
        assert issubclass(SegAPosEmbedQwen21, SegAPosEmbedQwen)
        assert isinstance(SegAPosEmbedQwen21(theta=10000, axes_dim=[16, 56, 56]),
                          SegAPosEmbed)
        assert (SegAPosEmbedQwen21.format_components
                is SegAPosEmbedQwen.format_components)

    def test_freqs_dtype_is_fp32(self):
        """freqs_dtype does not survive into the output (rope.py promotes back),
        so the argument itself is the observable."""
        seen = []
        original = SegAPosEmbedQwen21.get_components

        def spy(self, pos, freqs_dtype):
            seen.append(freqs_dtype)
            return original(self, pos, freqs_dtype)

        SegAPosEmbedQwen21.get_components = spy
        try:
            out = SegAPosEmbedQwen21(theta=10000, axes_dim=[16, 56, 56])(_make_flux_ids(8, 8))
        finally:
            SegAPosEmbedQwen21.get_components = original
        assert seen == [torch.float32]
        assert out.dtype is torch.float32

    def test_matches_qwen_1_0_on_cpu(self):
        """On CPU both adapters are fp32, so 2.1 must equal 1.0 numerically."""
        ids = _make_flux_ids(8, 8)
        out10 = SegAPosEmbedQwen(theta=10000, axes_dim=[16, 56, 56])(ids)
        out21 = SegAPosEmbedQwen21(theta=10000, axes_dim=[16, 56, 56])(ids)
        assert torch.allclose(out10, out21, atol=0)

    def test_with_spectral_data(self):
        emb = SegAPosEmbedQwen21(theta=10000, axes_dim=[16, 56, 56],
                                 training_res_pixels=1024)
        emb.set_spectral_data(torch.rand(16) * 10 + 1, torch.rand(16) * 10 + 1,
                              0.8, target_res_h=4096, target_res_w=4096)
        out = emb(_make_flux_ids(32, 32))
        assert out.shape[0] == 32 * 32


# ---------------------------------------------------------------------------
# SegAPosEmbedNunchaku
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSegAPosEmbedNunchaku:
    def test_output_shape(self):
        emb = SegAPosEmbedNunchaku(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        # (B, 1, M, D_total//2, 1, 2)
        assert out.shape[0] == 1
        assert out.shape[1] == 1
        assert out.shape[2] == 32 * 32
        assert out.shape[3] == 64  # (16 + 56 + 56) // 2
        assert out.shape[4] == 1
        assert out.shape[5] == 2

    def test_no_nan(self):
        emb = SegAPosEmbedNunchaku(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        assert not torch.isnan(out).any()

    def test_1d_input_handled(self):
        """Nunchaku handles 1-D input by adding batch dim."""
        emb = SegAPosEmbedNunchaku(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(16, 16).squeeze()  # remove batch
        out = emb(ids)
        assert out.shape[0] == 1


# ---------------------------------------------------------------------------
# SegAPosEmbedZImage
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSegAPosEmbedZImage:
    def test_output_shape(self):
        emb = SegAPosEmbedZImage(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        # (L, 1, D/2, 2, 2) — same layout as Flux
        L = 32 * 32
        assert out.shape[0] == L
        assert out.shape[1] == 1
        assert out.shape[2] == 64  # (16 + 56 + 56) / 2
        assert out.shape[3] == 2
        assert out.shape[4] == 2

    def test_no_nan(self):
        emb = SegAPosEmbedZImage(theta=10000, axes_dim=[16, 56, 56])
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        assert not torch.isnan(out).any()

    def test_scale_hint(self):
        emb = SegAPosEmbedZImage(theta=10000, axes_dim=[16, 56, 56])
        emb.set_scale_hint(2.0)
        assert emb.external_scale_hint == 2.0

    def test_with_spectral_data(self):
        emb = SegAPosEmbedZImage(theta=10000, axes_dim=[16, 56, 56], training_res_pixels=1024)
        emb.set_scale_hint(2.0)
        eh = torch.rand(16) * 10 + 1
        ew = torch.rand(16) * 10 + 1
        emb.set_spectral_data(eh, ew, 0.8, target_res_h=4096, target_res_w=4096)
        ids = _make_flux_ids(64, 64)
        out = emb(ids)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# SegAPosEmbedAnima
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSegAPosEmbedAnima:
    def test_output_shape(self):
        """Anima uses (B, T, H, W, C) input and produces (T*H*W, D/2, 2, 2)."""
        # Anima uses per-axis theta
        theta = [10000.0, 10000.0, 10000.0]
        axes_dim = [16, 56, 56]
        emb = SegAPosEmbedAnima(theta=theta, axes_dim=axes_dim)
        x = _make_anima_input(T=2, H=32, W=32, C=128)
        out = emb(x)
        # (T*H*W, D/2, 2, 2) where D/2 = (16+56+56)/2 = 64
        assert out.shape[0] == 2 * 32 * 32
        assert out.shape[1] == 64
        assert out.shape[2] == 2
        assert out.shape[3] == 2

    def test_no_nan(self):
        theta = [10000.0, 10000.0, 10000.0]
        emb = SegAPosEmbedAnima(theta=theta, axes_dim=[16, 56, 56])
        x = _make_anima_input(T=1, H=16, W=16, C=128)
        out = emb(x)
        assert not torch.isnan(out).any()

    def test_with_spectral_data(self):
        theta = [10000.0, 10000.0, 10000.0]
        emb = SegAPosEmbedAnima(theta=theta, axes_dim=[16, 56, 56], training_res_pixels=1024)
        eh = torch.rand(16) * 10 + 1
        ew = torch.rand(16) * 10 + 1
        emb.set_spectral_data(eh, ew, 0.8, target_res_h=4096, target_res_w=4096)
        x = _make_anima_input(T=1, H=128, W=128, C=128)
        out = emb(x)
        assert out.shape[0] == 1 * 128 * 128

    def test_per_axis_theta(self):
        """Anima should accept per-axis theta (list)."""
        theta = [10000.0, 20000.0, 30000.0]
        emb = SegAPosEmbedAnima(theta=theta, axes_dim=[16, 56, 56])
        assert emb.thetas == [10000.0, 20000.0, 30000.0]

    def test_non_square(self):
        theta = [10000.0, 10000.0, 10000.0]
        emb = SegAPosEmbedAnima(theta=theta, axes_dim=[16, 56, 56])
        x = _make_anima_input(T=1, H=32, W=64, C=128)
        out = emb(x)
        assert out.shape[0] == 1 * 32 * 64
