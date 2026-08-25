"""Tests for calibration loss functions (plan P1: T1.1-T1.4).

Markers: @pytest.mark.unit
Accept (user-run):
    pytest tests/test_hap_calib_loss.py -q
"""

import sys

import pytest
import torch

from src.hap_calib_node import make_calibration_loss

# ---------------------------------------------------------------------------
# T1.1 — output_norm
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOutputNormLoss:
    def test_zero_iff_output_zero(self):
        loss_fn = make_calibration_loss("output_norm")
        zero = torch.zeros(1, 4, 8, 8)
        assert loss_fn(zero).item() == 0.0
        nonzero = torch.ones(1, 4, 8, 8)
        assert loss_fn(nonzero).item() > 0.0

    def test_backward_populates_grad(self):
        loss_fn = make_calibration_loss("output_norm")
        x = torch.randn(1, 4, 8, 8, requires_grad=True)
        loss = loss_fn(x)
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
        # grad of MSE(x, 0) wrt x is 2x/N — nonzero where x is nonzero.
        assert torch.any(x.grad != 0)


# ---------------------------------------------------------------------------
# T1.2 — reference_mse
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestReferenceMseLoss:
    def test_zero_iff_output_equals_reference(self):
        ref = torch.randn(1, 4, 8, 8)
        loss_fn = make_calibration_loss("reference_mse", reference=ref)
        assert loss_fn(ref.clone()).item() == pytest.approx(0.0, abs=1e-12)
        other = torch.randn(1, 4, 8, 8)
        assert loss_fn(other).item() > 0.0

    def test_shape_mismatch_raises(self):
        ref = torch.randn(1, 4, 8, 8)
        loss_fn = make_calibration_loss("reference_mse", reference=ref)
        with pytest.raises(ValueError, match="shape mismatch"):
            loss_fn(torch.randn(1, 4, 16, 16))

    def test_batch_broadcast(self):
        ref = torch.randn(1, 4, 8, 8)
        loss_fn = make_calibration_loss("reference_mse", reference=ref)
        # output B=3, ref B=1 -> broadcast.
        out = ref.expand(3, 4, 8, 8).clone()
        assert loss_fn(out).item() == pytest.approx(0.0, abs=1e-12)

    def test_missing_reference_raises(self):
        with pytest.raises(ValueError, match="requires"):
            make_calibration_loss("reference_mse", reference=None)


# ---------------------------------------------------------------------------
# T1.3 — unknown loss type
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestUnknownLossType:
    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown loss_type"):
            make_calibration_loss("bogus")


# ---------------------------------------------------------------------------
# T1.4 — gradient flows through chunked_attention chunks
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestLossThroughChunkedAttention:
    def test_grad_flows_to_chunk_leaves(self):
        """A toy 2-layer forward through the patched attention + output_norm
        loss yields non-None grads on every chunk leaf."""
        from _hrdit_fixtures import make_toy_dit
        from src.hap_calib import chunked_attention

        attn_mod = sys.modules["comfy.ldm.modules.attention"]
        orig = attn_mod.optimized_attention

        chunks_all = []

        def chunked_attn(q, k, v, heads, *args, **kwargs):
            out, chunks = chunked_attention(q, k, v, scale=1.0, chunk=5)
            chunks_all.extend(chunks)
            return out

        attn_mod.optimized_attention = chunked_attn
        try:
            dit = make_toy_dit(num_layers=2, heads=2, dim=8, text_len=4,
                               img_hw=4, seed=0, dtype=torch.float64)
            out = dit.forward()
            loss_fn = make_calibration_loss("output_norm")
            loss = loss_fn(out)
            loss.backward()
        finally:
            attn_mod.optimized_attention = orig

        assert len(chunks_all) > 0
        for c in chunks_all:
            assert c.grad is not None, "chunk leaf has no gradient"
            assert torch.any(c.grad != 0)
