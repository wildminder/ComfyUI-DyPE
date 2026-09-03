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
def _install_fake_pbar_utils(monkeypatch):
    fake_utils = types.ModuleType("comfy.utils")

    class FakeProgressBar:
        def __init__(self, total):
            self.total = total
            self.updates = []

        def update_absolute(self, n):
            self.updates.append(n)

    fake_utils.ProgressBar = FakeProgressBar

    def repeat_to_batch_size(x, count, dim=1):
        # Tile along `dim` until it reaches `count` (mirrors comfy.utils).
        reps = [1] * x.ndim
        reps[dim] = -(-count // x.shape[dim])  # ceil division
        out = x.repeat(reps)
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(0, count)
        return out[tuple(slices)]

    fake_utils.repeat_to_batch_size = repeat_to_batch_size

    monkeypatch.setitem(sys.modules, "comfy.utils", fake_utils)
    comfy_mod = sys.modules.get("comfy")
    if comfy_mod is not None:
        monkeypatch.setattr(comfy_mod, "utils", fake_utils, raising=False)
    return FakeProgressBar


@pytest.mark.unit
class TestVaeAdapters:
    def test_decode_normalizes_layout(self):
        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(1, 8, 8, 3),  # [B,H,W,3]
        )
        dec, _ = hfn._make_vae_adapters(vae, torch.device("cpu"))
        img = dec(torch.randn(1, 4, 8, 8))
        assert img.shape[1] == 3, "decode must return [B,3,H,W]"

    def test_decode_5d_squeezed(self):
        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(1, 3, 1, 8, 8),
        )
        dec, _ = hfn._make_vae_adapters(vae, torch.device("cpu"))
        img = dec(torch.randn(1, 4, 8, 8))
        assert img.ndim == 4

    def test_encode_normalizes_layout(self):
        vae = types.SimpleNamespace(
            encode=lambda im: {"samples": torch.randn(1, 4, 8, 8)},
        )
        _, enc = hfn._make_vae_adapters(vae, torch.device("cpu"))
        lat = enc(torch.randn(1, 8, 8, 3))  # [B,H,W,3] in
        assert lat.shape == (1, 4, 8, 8)

    def test_decode_accepts_dict_latent(self):
        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(1, 3, 8, 8),
        )
        dec, _ = hfn._make_vae_adapters(vae, torch.device("cpu"))
        img = dec({"samples": torch.randn(1, 4, 8, 8)})
        assert img.ndim == 4


@pytest.mark.unit
class TestSharpen:
    def test_constant_image_unchanged_interior(self):
        """Interior: blur of a constant is the constant -> sharpened == input.
        (gaussian_blur_2d zero-pads, so borders darken; the reference
        sharpens interior pixels.)"""
        img = torch.full((1, 3, 8, 8), 0.5)
        out = hfn._sharpen(img, alpha=1.0)
        interior = out[..., 2:-2, 2:-2]
        assert torch.allclose(interior, img[..., 2:-2, 2:-2], atol=1e-4)

    def test_unsharp_formula_on_impulse(self):
        """A delta impulse sharpens toward (alpha+1)*I at the peak (blur
        takes most of the mass away from the peak)."""
        img = torch.zeros(1, 1, 16, 16)
        img[0, 0, 8, 8] = 1.0
        out = hfn._sharpen(img, alpha=1.0)
        assert out[0, 0, 8, 8].item() > 1.0, (
            "the impulse peak must exceed its input (unsharp adds)"
        )
        assert (out[0, 0] < 0).any(), "surroundings must undershoot"


@pytest.mark.unit
class TestBaseSigmas:
    def test_descending_ends_at_zero(self, monkeypatch):
        fake_samplers = types.ModuleType("comfy.samplers")

        def calculate_sigmas(ms, scheduler, steps):
            assert scheduler == "simple"
            interior = torch.linspace(1.0, 0.1, steps)
            return torch.cat([interior, torch.zeros(1)])

        fake_samplers.calculate_sigmas = calculate_sigmas
        monkeypatch.setitem(sys.modules, "comfy.samplers", fake_samplers)
        comfy_mod = sys.modules.get("comfy")
        if comfy_mod is not None:
            monkeypatch.setattr(
                comfy_mod, "samplers", fake_samplers, raising=False)

        model = types.SimpleNamespace()
        model.model = types.SimpleNamespace(
            model_sampling=types.SimpleNamespace())
        sigmas = hfn._base_sigmas(model, steps=5)
        assert sigmas.numel() == 6
        assert float(sigmas[-1]) == 0.0
        assert bool(torch.all(sigmas[:-1] > sigmas[1:]))


@pytest.mark.unit
class TestExecuteWiring:
    def _run_execute(self, monkeypatch, target=512, upsampling="latent",
                     latent=(1, 16, 16, 16), prediction_mixin="CONST"):

        FakePBar = _install_fake_pbar_utils(monkeypatch)
        _install_fake_comfy(monkeypatch)

        model = _mock_flow_model(prediction_mixin=prediction_mixin)
        # model_sampling.sigmas for _base_sigmas via fake calculate_sigmas
        fake_samplers = sys.modules["comfy.samplers"]
        fake_samplers.calculate_sigmas = (
            lambda ms, scheduler, steps:
            torch.cat([torch.linspace(1.0, 0.1, steps), torch.zeros(1)])
        )

        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(1, 3, z.shape[-2] * 8, z.shape[-1] * 8),
            encode=lambda im: {"samples": torch.randn(
                1, 4, im.shape[-2] // 8, im.shape[-1] // 8)},
            downscale_ratio=8,
        )
        z = torch.randn(*latent)

        result = hfn.HiFlowNode.execute(
            model, vae, COND_POS, COND_NEG, {"samples": z},
            cfg=3.5, steps=4, guidance=4.5, steps_per_stage=2,
            tau=0.5, filter_ratio=0.2, alpha_scale=1.0, beta_scale=0.5,
            upsampling=upsampling, target_resolution=target,
        )
        # NodeOutput wraps the payload positionally; unwrap to the dict.
        samples = result[0]["samples"] if not hasattr(result, "shape") \
            else result
        out_dict = {"samples": samples}
        return out_dict, FakePBar

    def test_execute_returns_latent_dict(self, monkeypatch):
        result, _ = self._run_execute(monkeypatch, target=512)
        samples = result["samples"]
        assert samples.ndim == 4
        assert torch.isfinite(samples).all()

    def test_execute_doubles_resolution(self, monkeypatch):
        """16x16 latent (128px) + target 512px -> 64x64 latent out."""
        result, _ = self._run_execute(monkeypatch, target=512)
        assert tuple(result["samples"].shape[-2:]) == (64, 64)

    def test_execute_rejects_non_flow_model(self, monkeypatch):
        with pytest.raises(ValueError, match="PixelRush"):
            self._run_execute(monkeypatch, prediction_mixin="EPS")

    def test_execute_rejects_3d_latents(self, monkeypatch):
        with pytest.raises(ValueError, match="3D \\(video\\) latent"):
            self._run_execute(monkeypatch, latent=(1, 4, 1, 16, 16))

    def test_execute_empty_latent_channels_repeated(self, monkeypatch):
        """An empty 4-channel latent for a 16-channel model is repeated."""

        _install_fake_pbar_utils(monkeypatch)
        _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()  # latent_channels=16
        sys.modules["comfy.samplers"].calculate_sigmas = (
            lambda ms, scheduler, steps:
            torch.cat([torch.linspace(1.0, 0.1, steps), torch.zeros(1)])
        )
        vae = types.SimpleNamespace(downscale_ratio=8)
        z = torch.zeros(1, 4, 16, 16)  # empty, wrong channel count
        result = hfn.HiFlowNode.execute(
            model, vae, COND_POS, COND_NEG, {"samples": z},
            steps=2, steps_per_stage=2, tau=0.5,
            target_resolution=256,
        )
        samples = result[0]["samples"] if not hasattr(result, "shape") \
            else result
        assert tuple(samples.shape[-2:]) == (32, 32)

    def test_execute_pixel_mode_uses_vae(self, monkeypatch):
        """upsampling=pixel must reach the (fake) VAE decode/encode."""

        _install_fake_pbar_utils(monkeypatch)
        _install_fake_comfy(monkeypatch)
        sys.modules["comfy.samplers"].calculate_sigmas = (
            lambda ms, scheduler, steps:
            torch.cat([torch.linspace(1.0, 0.1, steps), torch.zeros(1)])
        )
        model = _mock_flow_model()
        calls = {"n": 0}

        def decode(z):
            calls["n"] += 1
            return torch.randn(1, 3, z.shape[-2] * 8, z.shape[-1] * 8)

        def encode(im):
            return {"samples": torch.randn(
                1, 16, im.shape[-2] // 8, im.shape[-1] // 8)}

        vae = types.SimpleNamespace(decode=decode, encode=encode,
                                    downscale_ratio=8)
        result = hfn.HiFlowNode.execute(
            model, vae, COND_POS, COND_NEG,
            {"samples": torch.randn(1, 16, 16, 16)},
            steps=2, steps_per_stage=2, tau=0.5,
            upsampling="pixel", target_resolution=256,
        )
        assert calls["n"] >= 1, "pixel mode must call vae.decode"
        samples = result[0]["samples"] if not hasattr(result, "shape") \
            else result
        assert samples.shape[-1] == 32


# ---------------------------------------------------------------------------
# Step 8 — schema, registration, docs, version
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestHiFlowNodeSchema:
    def _src(self):
        import pathlib
        return (pathlib.Path(__file__).parent.parent / "src"
                / "hiflow_node.py").read_text(encoding="utf-8")

    def test_schema_inputs_and_paper_defaults(self):
        src = self._src()
        for inp in ["model", "vae", "positive", "negative", "latent_image",
                    "cfg", "steps", "guidance", "steps_per_stage", "tau",
                    "filter_ratio", "alpha_scale", "beta_scale", "upsampling",
                    "target_resolution"]:
            assert f'"{inp}"' in src, f"missing schema input {inp}"
        assert "default=3.5" in src     # cfg (FLUX-dev)
        assert "default=30" in src      # steps (paper)
        assert "default=4.5" in src     # guidance
        assert "default=16" in src      # steps_per_stage (repo)
        assert "default=0.6" in src     # tau (paper 1K->2K)
        assert "default=0.2" in src     # filter_ratio (repo)
        assert 'default="latent"' in src

    def test_schema_execute_signature_matches(self):
        import re
        src = self._src()
        pattern = re.compile(r'io\.\w+\.Input\(\s*"([^"]+)"')
        schema_inputs = set(pattern.findall(src))
        assert schema_inputs, "failed to parse schema inputs"
        sig_start = src.index("def execute(cls,")
        sig_start += len("def execute(cls,")
        sig = src[sig_start:src.index(") -> io.NodeOutput:", sig_start)]
        params = set()
        for chunk in sig.split(","):
            chunk = chunk.strip()
            if "=" in chunk:
                chunk = chunk.split("=")[0].strip()
            if chunk and chunk != "cls":
                params.add(chunk)
        missing = (schema_inputs - params) | (params - schema_inputs)
        assert not missing, (
            f"schema/execute drift: schema-only={schema_inputs - params}, "
            f"exec-only={params - schema_inputs}"
        )

    def test_node_registered_in_extension(self):
        import pathlib
        init = (pathlib.Path(__file__).parent.parent
                / "__init__.py").read_text(encoding="utf-8")
        assert "HiFlowNode" in init, "HiFlowNode must be imported in __init__"
        assert "HiFlowNode" in init.split("get_node_list")[-1], (
            "HiFlowNode must appear in get_node_list()"
        )

    def test_category_matches_cascade_family(self):
        assert 'category="image/upscaling"' in self._src()

    def test_validate_inputs_none_passes(self):
        assert hfn.HiFlowNode.validate_inputs(target_resolution=None) is True

    def test_validate_inputs_small_rejected(self):
        result = hfn.HiFlowNode.validate_inputs(target_resolution=8)
        assert isinstance(result, str)


@pytest.mark.unit
class TestHiFlowDocs:
    def test_readme_documents_hiflow(self):
        import pathlib
        readme = (pathlib.Path(__file__).parent.parent
                  / "README.md").read_text(encoding="utf-8")
        assert "HiFlow" in readme
        assert "user-content-hiflow" in readme

    def test_version_bumped(self):
        import pathlib
        import re
        pyproject = (pathlib.Path(__file__).parent.parent
                     / "pyproject.toml").read_text(encoding="utf-8")
        readme = (pathlib.Path(__file__).parent.parent
                  / "README.md").read_text(encoding="utf-8")
        m = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE)
        assert m and m.group(1) == "2.10.0"
        assert "### v2.10.0" in readme

    def test_workflow_json_parses_and_uses_known_nodes(self):
        import json
        import pathlib
        wf_path = (pathlib.Path(__file__).parent.parent
                   / "example_workflows" / "HiFlow-Flux-workflow.json")
        data = json.loads(wf_path.read_text(encoding="utf-8"))
        types = set()
        for v in data.values():
            if isinstance(v, dict) and "class_type" in v:
                types.add(v["class_type"])
        core = {"UNETLoader", "DualCLIPLoader", "VAELoader", "CLIPTextEncode",
                "EmptySD3LatentImage", "VAEDecode", "SaveImage"}
        unknown = types - core - {"HiFlow"}
        assert not unknown, f"workflow references unknown nodes: {unknown}"
        assert "HiFlow" in types
