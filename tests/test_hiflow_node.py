"""Tests for src/hiflow_node.py — node adapters (Tier 2).

Step 6 of plan 2026-09-03: flow-model gate + the sampling_function-based
x0 adapter. The conftest mock-comfy does not provide comfy.samplers /
comfy.sampler_helpers / comfy.model_management, so this file registers
minimal fakes in sys.modules (the HAP-calibration test pattern).

Markers: @pytest.mark.unit
"""

import sys
import types

import pytest
import torch

import src.hiflow_node as hfn

# ---------------------------------------------------------------------------
# Fake comfy runtime modules
# ---------------------------------------------------------------------------

def _install_fake_comfy(monkeypatch):
    """Register minimal comfy.sampler_helpers / samplers / model_management
    fakes. Returns handles the tests use to observe adapter behavior."""
    import copy

    fake_helpers = types.ModuleType("comfy.sampler_helpers")
    convert_calls = {"positive": 0, "negative": 0}

    def convert_cond(cond):
        # CONDITIONING public format: list of (tensor, dict) tuples.
        out = []
        for entry in cond:
            tensor, opts = entry
            convert_calls["positive" if opts.get("side") == "positive"
                          else "negative"] += 1
            out.append({"tensor": tensor, "opts": opts})
        return out

    fake_helpers.convert_cond = convert_cond

    fake_samplers = types.ModuleType("comfy.samplers")
    process_calls = {"count": 0}
    sampling_calls = []

    def process_conds(model, noise, conds, device, *args, **kwargs):
        process_calls["count"] += 1
        # process_conds returns conds with model_conds resolved; the fake
        # keeps the dict structure the adapter re-uses.
        return {
            "positive": conds["positive"] if conds["positive"] else [],
            "negative": conds["negative"] if conds["negative"] else [],
        }

    def sampling_function(model, x, timestep, uncond, cond, cond_scale,
                          model_options=None, seed=None):
        sampling_calls.append({
            "x_shape": tuple(x.shape), "timestep": timestep,
            "cond_scale": cond_scale,
        })
        # Deterministic stand-in for calculate_denoised: x0 = 0.5 * x.
        return 0.5 * x

    fake_samplers.process_conds = process_conds
    fake_samplers.sampling_function = sampling_function

    fake_mm = types.ModuleType("comfy.model_management")
    fake_mm.load_models_gpu = lambda models: None

    monkeypatch.setitem(sys.modules, "comfy.sampler_helpers", fake_helpers)
    monkeypatch.setitem(sys.modules, "comfy.samplers", fake_samplers)
    monkeypatch.setitem(sys.modules, "comfy.model_management", fake_mm)
    # The import machinery also reads the attribute off the parent module.
    comfy_mod = sys.modules.get("comfy")
    if comfy_mod is not None:
        monkeypatch.setattr(comfy_mod, "samplers", fake_samplers, raising=False)
        monkeypatch.setattr(
            comfy_mod, "sampler_helpers", fake_helpers, raising=False)
        monkeypatch.setattr(
            comfy_mod, "model_management", fake_mm, raising=False)

    return types.SimpleNamespace(
        convert_calls=convert_calls, process_calls=process_calls,
        sampling_calls=sampling_calls,
        copy=copy,
    )


def _mock_flow_model(latent_dimensions=2, prediction_mixin="CONST"):
    """Mock ModelPatcher with a flow model_sampling and 4D latents.

    ``prediction_mixin`` names a base class that lands in the MRO the same
    way ComfyUI composes ModelSampling(base, CONST) — "CONST" (flow),
    "EPS" (diffusion), "V_PREDICTION", "X0".
    """
    mixins = {
        "CONST": type("CONST", (), {}),
        "EPS": type("EPS", (), {}),
        "V_PREDICTION": type("V_PREDICTION", (), {}),
        "X0": type("X0", (), {}),
    }

    class _Base:
        def timestep(self, sigma):
            return sigma * 1000.0  # DiscreteFlow-style multiplier probe

    ms = type("ModelSampling", (_Base, mixins[prediction_mixin]), {})()

    model = types.SimpleNamespace()
    model.model = types.SimpleNamespace()
    model.model.model_sampling = ms
    model.model.latent_format = types.SimpleNamespace(
        latent_dimensions=latent_dimensions, latent_channels=16)
    model.model.process_latent_in = lambda t: (t - 0.1159) * 0.3611
    model.model.process_latent_out = lambda t: (t / 0.3611) + 0.1159
    model.model_options = {}
    model.load_device = torch.device("cpu")
    model.pre_run = lambda: None
    return model


COND_POS = [(torch.ones(1, 4), {"side": "positive"})]
COND_NEG = [(torch.zeros(1, 4), {"side": "negative"})]


# ---------------------------------------------------------------------------
# Flow gate (D1, D12)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFlowGate:
    def test_flow_gate_accepts_const(self):
        model = _mock_flow_model()
        assert hfn._require_flow_model(model) == "const"

    def test_flow_gate_rejects_eps_with_actionable_message(self):
        model = _mock_flow_model(prediction_mixin="EPS")
        with pytest.raises(ValueError, match="PixelRush"):
            hfn._require_flow_model(model)

    def test_flow_gate_rejects_3d_latents(self):
        model = _mock_flow_model(latent_dimensions=3)
        with pytest.raises(ValueError, match="3D-latent"):
            hfn._require_flow_model(model)


# ---------------------------------------------------------------------------
# predict_x0 adapter (D3)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPredictX0:
    def test_roundtrip_spaces(self, monkeypatch):
        """VAE in -> model space -> sampling_function -> VAE out, with the
        affine latent format cancelling on the way back."""
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        adapter = hfn._make_predict_x0(model, COND_POS, COND_NEG, cfg_scale=3.5)

        torch.manual_seed(0)
        x_vae = torch.randn(1, 4, 8, 8)
        out = adapter(x_vae, sigma=0.6)

        # The fake sampling_function returned 0.5 * x_model; the adapter must
        # hand back process_latent_out(0.5 * process_latent_in(x)).
        x_model = (x_vae - 0.1159) * 0.3611
        expected = (0.5 * x_model / 0.3611) + 0.1159
        assert torch.allclose(out, expected, atol=1e-6)
        assert fx.sampling_calls[-1]["cond_scale"] == 3.5

    def test_timestep_converted_via_model_sampling(self, monkeypatch):
        """sigma -> timestep goes through model_sampling.timestep (x1000
        here), never hand-multiplied."""
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        adapter = hfn._make_predict_x0(model, COND_POS, COND_NEG, cfg_scale=1.0)
        adapter(torch.randn(1, 4, 8, 8), sigma=0.42)
        ts = fx.sampling_calls[-1]["timestep"]
        assert torch.allclose(ts, torch.tensor([420.0])), (
            "adapter must call model_sampling.timestep(sigma)"
        )

    def test_conds_processed_once_per_shape(self, monkeypatch):
        """process_conds is cached per latent shape (stage-stable)."""
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        adapter = hfn._make_predict_x0(model, COND_POS, COND_NEG, cfg_scale=1.0)
        for _ in range(5):
            adapter(torch.randn(1, 4, 8, 8), sigma=0.5)
        assert fx.process_calls["count"] == 1, "shape-stable calls reuse conds"
        adapter(torch.randn(1, 4, 16, 16), sigma=0.5)
        assert fx.process_calls["count"] == 2, "new shape reprocesses conds"

    def test_cfg_scale_forwarded(self, monkeypatch):
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        adapter = hfn._make_predict_x0(model, COND_POS, COND_NEG, cfg_scale=6.0)
        adapter(torch.randn(1, 4, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["cond_scale"] == 6.0

    def test_dtype_and_device_restored(self, monkeypatch):
        _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        adapter = hfn._make_predict_x0(model, COND_POS, COND_NEG, cfg_scale=1.0)
        x = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        out = adapter(x, sigma=0.5)
        assert out.dtype == torch.float16
        assert out.device == x.device

    def test_no_direct_diffusion_model_call(self):
        """The adapter must go through sampling_function, never call
        diffusion_model directly (the D3 seam)."""
        import pathlib
        content = (pathlib.Path(__file__).parent.parent / "src"
                   / "hiflow_node.py").read_text(encoding="utf-8")
        assert "sampling_function" in content
        assert "diffusion_model(" not in content, (
            "hiflow_node must not bypass sampling_function"
        )
