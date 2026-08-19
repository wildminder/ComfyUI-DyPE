"""Tests for the calibration forward bridge + model-aware collector (plan P2).

Covers:
- T2.1 injected-forward seam (``calibration_forward``);
- T2.2 ``default_calibration_forward`` clear error without the ComfyUI runtime;
- T2.3 ``collect_scope_scores_for_model`` — backend-aware patching, reshape
  conventions, non-square/masked skip, layer grouping, restore semantics;
- T2.4 regression parity with ``src.hap_calib.collect_scope_scores``.

Markers: @pytest.mark.unit
Accept (user-run):
    pytest tests/test_hap_calib_forward.py -q
"""

import logging
import sys

import pytest
import torch

import src.hap_calib_node as hcn
from src.hap_calib_node import (
    CalibrationSpec,
    calibration_forward,
    collect_scope_scores_for_model,
    default_calibration_forward,
)


def _attn_module():
    return sys.modules["comfy.ldm.modules.attention"]


def _toy(seed=5):
    from _hrdit_fixtures import make_toy_dit
    return make_toy_dit(num_layers=2, heads=3, dim=8, text_len=8,
                        img_hw=4, seed=seed, dtype=torch.float64)


def _loss_fn(dit):
    g = torch.Generator().manual_seed(123)
    target = torch.randn(1, dit.seq_len, dit.heads * dit.dim,
                         generator=g, dtype=torch.float64)

    def loss_fn(output):
        return torch.nn.functional.mse_loss(output, target)

    return loss_fn


# ---------------------------------------------------------------------------
# T2.1 — injected-forward seam
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCalibrationForwardSeam:
    def test_injected_forward_called_with_index(self):
        calls = []

        def fake_fwd(model, spec, prompt_index):
            calls.append((model, spec, prompt_index))
            return torch.zeros(1)

        spec = CalibrationSpec(seed=100, prompts=["p"])
        sentinel = object()
        out = calibration_forward(
            model=sentinel, spec=spec, prompt_index=2,
            positive=None, negative=None, forward_fn=fake_fwd,
        )
        assert torch.equal(out, torch.zeros(1))
        assert len(calls) == 1
        assert calls[0][0] is sentinel
        assert calls[0][1] is spec
        assert calls[0][2] == 2

    def test_default_path_uses_seed_offset(self, monkeypatch):
        """Without an injected forward, the default forward receives
        ``seed = spec.seed + prompt_index``."""
        recorded = {}

        def fake_default(**kwargs):
            recorded.update(kwargs)
            return torch.zeros(1)

        monkeypatch.setattr(hcn, "default_calibration_forward", fake_default)
        spec = CalibrationSpec(seed=1000, width=512, height=512,
                               calib_sigma=0.7, prompts=["p"])
        calibration_forward(
            model=object(), spec=spec, prompt_index=3,
            positive="pos", negative="neg", forward_fn=None,
        )
        assert recorded["seed"] == 1003
        assert recorded["sigma"] == 0.7
        assert recorded["width"] == 512
        assert recorded["height"] == 512
        assert recorded["positive"] == "pos"
        assert recorded["negative"] == "neg"


# ---------------------------------------------------------------------------
# T2.2 — default forward without ComfyUI runtime
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDefaultForwardNoRuntime:
    def test_raises_clear_error_without_comfy(self):
        with pytest.raises(RuntimeError, match="ComfyUI runtime"):
            default_calibration_forward(
                model=object(), positive=None, negative=None,
                width=512, height=512, sigma=1.0, seed=0,
            )


# ---------------------------------------------------------------------------
# T2.3 — model-aware collector
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCollectorForModel:
    def test_square_4d_collection_shapes(self):
        """A square 4D toy forward yields (L, H, S) quality/compute tables."""
        dit = _toy()
        quality, compute, seq_len = collect_scope_scores_for_model(
            model=object(), model_type="flux",
            forward_fn=dit.forward, loss_fn=_loss_fn(dit),
            num_scopes=6, text_len=8, chunk=5, scale=1.0,
        )
        assert quality.shape == (2, 3, 6)
        assert compute.shape == (2, 3, 6)
        assert seq_len == dit.seq_len
        # Full-scope column is exactly zero.
        assert torch.all(quality[:, :, -1] == 0.0)

    def test_3d_convention_output_matches_sdpa(self):
        """A 3D-layout call (B, T, H*D) is collected and the returned output
        equals the original SDPA result in the SAME (B, T, H*D) layout."""
        import torch.nn.functional as F

        attn_mod = _attn_module()
        B, H, T, D = 1, 2, 12, 4
        q4 = torch.randn(B, H, T, D, dtype=torch.float64)
        k4 = torch.randn(B, H, T, D, dtype=torch.float64)
        v4 = torch.randn(B, H, T, D, dtype=torch.float64)
        # 3D layout: (B, T, H*D)
        q3 = q4.permute(0, 2, 1, 3).reshape(B, T, H * D)
        k3 = k4.permute(0, 2, 1, 3).reshape(B, T, H * D)
        v3 = v4.permute(0, 2, 1, 3).reshape(B, T, H * D)

        ref = F.scaled_dot_product_attention(q4, k4, v4, scale=1.0)
        ref3 = ref.permute(0, 2, 1, 3).reshape(B, T, H * D)

        captured = {}

        def fwd():
            out = attn_mod.optimized_attention(q3, k3, v3, H)
            captured["out"] = out
            return out

        collect_scope_scores_for_model(
            model=object(), model_type="flux",
            forward_fn=fwd, loss_fn=lambda o: o.sum(),
            num_scopes=4, text_len=0, chunk=5, scale=1.0,
        )
        assert captured["out"].shape == (B, T, H * D)
        assert torch.allclose(captured["out"], ref3, atol=1e-10)

    def test_4d_skip_output_reshape_convention(self):
        """``skip_output_reshape=True`` returns (B, H, T, D)."""
        attn_mod = _attn_module()
        B, H, T, D = 1, 2, 8, 4
        q = torch.randn(B, H, T, D, dtype=torch.float64)
        captured = {}

        def fwd():
            out = attn_mod.optimized_attention(
                q, q, q, H, skip_reshape=True, skip_output_reshape=True,
            )
            captured["out"] = out
            return out

        collect_scope_scores_for_model(
            model=object(), model_type="flux",
            forward_fn=fwd, loss_fn=lambda o: o.sum(),
            num_scopes=3, text_len=0, chunk=4, scale=1.0,
        )
        assert captured["out"].shape == (B, H, T, D)

    def test_nonsquare_call_skipped(self, caplog):
        """One square + one non-square call -> only the square layer is
        collected; the skip logs once at DEBUG."""
        attn_mod = _attn_module()
        B, H, D = 1, 2, 4
        q_sq = torch.randn(B, H, 8, D, dtype=torch.float64)
        q_cross = torch.randn(B, H, 8, D, dtype=torch.float64)
        k_cross = torch.randn(B, H, 5, D, dtype=torch.float64)  # kv_len != q_len

        def fwd():
            out1 = attn_mod.optimized_attention(q_sq, q_sq, q_sq, H)
            out2 = attn_mod.optimized_attention(
                q_cross, k_cross, k_cross, H,
            )
            return out1.sum() + out2.sum()

        with caplog.at_level(logging.DEBUG, logger="ComfyUI-DyPE"):
            quality, _, _ = collect_scope_scores_for_model(
                model=object(), model_type="flux",
                forward_fn=fwd, loss_fn=lambda o: o,
                num_scopes=3, text_len=0, chunk=4, scale=1.0,
            )
        # Only ONE collected layer (the square call).
        assert quality.shape[0] == 1
        skip_msgs = [r for r in caplog.records
                     if "non-square" in r.getMessage()]
        assert len(skip_msgs) == 1

    def test_all_nonsquare_raises(self):
        """A forward with ONLY non-square calls raises the extended error."""
        attn_mod = _attn_module()
        B, H, D = 1, 2, 4
        q = torch.randn(B, H, 8, D)
        k = torch.randn(B, H, 5, D)

        def fwd():
            return attn_mod.optimized_attention(q, k, k, H).sum()

        with pytest.raises(RuntimeError, match="no square"):
            collect_scope_scores_for_model(
                model=object(), model_type="flux",
                forward_fn=fwd, loss_fn=lambda o: o,
                num_scopes=3, text_len=0, chunk=4, scale=1.0,
            )

    def test_masked_call_skipped(self, caplog):
        """Calls carrying an external mask pass through unrecorded."""
        attn_mod = _attn_module()
        B, H, T, D = 1, 2, 8, 4
        q = torch.randn(B, H, T, D, dtype=torch.float64)
        mask = torch.ones(T, T, dtype=torch.bool)

        def fwd():
            out_masked = attn_mod.optimized_attention(
                q, q, q, H, mask=mask, skip_reshape=True,
            )
            out_plain = attn_mod.optimized_attention(
                q, q, q, H, skip_reshape=True,
            )
            return out_masked.sum() + out_plain.sum()

        with caplog.at_level(logging.DEBUG, logger="ComfyUI-DyPE"):
            quality, _, _ = collect_scope_scores_for_model(
                model=object(), model_type="flux",
                forward_fn=fwd, loss_fn=lambda o: o,
                num_scopes=3, text_len=0, chunk=4, scale=1.0,
            )
        assert quality.shape[0] == 1  # only the unmasked call collected
        assert any("masked" in r.getMessage() for r in caplog.records)

    def test_restores_originals_even_on_failure(self):
        attn_mod = _attn_module()
        orig = attn_mod.optimized_attention

        collect_scope_scores_for_model(
            model=object(), model_type="flux",
            forward_fn=_toy().forward, loss_fn=_loss_fn(_toy()),
            num_scopes=4, text_len=8, chunk=6, scale=1.0,
        )
        assert attn_mod.optimized_attention is orig

        def boom():
            raise RuntimeError("calibration forward failed")
        with pytest.raises(RuntimeError):
            collect_scope_scores_for_model(
                model=object(), model_type="flux",
                forward_fn=boom, loss_fn=lambda o: o.sum(),
                num_scopes=4, text_len=0, chunk=4, scale=1.0,
            )
        assert attn_mod.optimized_attention is orig

    def test_layer_grouping_by_hrdit_counter(self):
        """SPA variant passes sharing one wrapper-counter value fold into ONE
        calibrated layer; the next counter value starts the next layer."""
        from src.spa_context import set_hrdit_layer_idx

        attn_mod = _attn_module()
        B, H, T, D = 1, 2, 8, 4
        q = torch.randn(B, H, T, D, dtype=torch.float64)

        def fwd():
            # Layer 0: two variant passes under counter value 1.
            set_hrdit_layer_idx(1)
            o1 = attn_mod.optimized_attention(q, q, q, H, skip_reshape=True)
            o2 = attn_mod.optimized_attention(q, q, q, H, skip_reshape=True)
            # Layer 1: one pass under counter value 2.
            set_hrdit_layer_idx(2)
            o3 = attn_mod.optimized_attention(q, q, q, H, skip_reshape=True)
            return o1.sum() + o2.sum() + o3.sum()

        try:
            quality, _, _ = collect_scope_scores_for_model(
                model=object(), model_type="flux",
                forward_fn=fwd, loss_fn=lambda o: o,
                num_scopes=3, text_len=0, chunk=4, scale=1.0,
            )
        finally:
            set_hrdit_layer_idx(0)  # never leak counter state
        assert quality.shape[0] == 2  # 2 calibrated layers, not 3


# ---------------------------------------------------------------------------
# T2.4 — parity with the original module-level collector
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCollectorParity:
    def test_equals_collect_scope_scores_on_convention_invariant_forward(self):
        """Regression guard: on a forward whose output convention BOTH
        collectors handle identically (4D-in / 4D-out via
        ``skip_output_reshape=True``), the model-aware collector produces
        scores bit-identical to the original ``collect_scope_scores``.

        The two collectors deliberately differ on the toy's default 3D-output
        case (the new one returns the real ComfyUI ``(B, T, H*D)`` layout), so
        strict parity there is neither achievable nor desired — this test pins
        the shared chunked-attention + scoring math instead.
        """
        from src.hap_calib import collect_scope_scores

        attn_mod = _attn_module()
        num_scopes, text_len, chunk = 6, 8, 5
        B, H, T, D = 1, 3, 24, 4

        def _make_fwd(seed):
            g = torch.Generator().manual_seed(seed)
            q = torch.randn(B, H, T, D, generator=g, dtype=torch.float64)
            k = torch.randn(B, H, T, D, generator=g, dtype=torch.float64)
            v = torch.randn(B, H, T, D, generator=g, dtype=torch.float64)
            tg = torch.Generator().manual_seed(999)
            target = torch.randn(B, H, T, D, generator=tg, dtype=torch.float64)

            def fwd():
                out = attn_mod.optimized_attention(
                    q, k, v, H, skip_reshape=True, skip_output_reshape=True,
                )
                return torch.nn.functional.mse_loss(out, target)

            return fwd

        # NOTE: the forward already returns a SCALAR loss, so loss_fn is the
        # identity for both collectors (keeps the autograd graph identical).
        q_new, _, _ = collect_scope_scores_for_model(
            model=object(), model_type="flux",
            forward_fn=_make_fwd(7), loss_fn=lambda o: o,
            num_scopes=num_scopes, text_len=text_len, chunk=chunk, scale=1.0,
        )
        q_old, _ = collect_scope_scores(
            _make_fwd(7), lambda o: o, num_scopes,
            text_len=text_len, chunk=chunk, scale=1.0,
        )
        assert q_new.shape == q_old.shape
        assert torch.allclose(q_new, q_old, atol=1e-12)
