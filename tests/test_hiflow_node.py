"""Tests for nodes/hiflow.py — node adapters (Tier 2).

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

import nodes.hiflow as hfn

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
    if latent_dimensions == 3:
        # Wan21-faithful conversions (v2.14.1): [1,C,1,1,1] mean/std
        # stats — a 4D tensor against them BROADCASTS SILENTLY to
        # [B,C,C,H,W] garbage (the real Krea2 T=16 crash), so the mock
        # must replicate the shape hazard exactly, not just the math.
        _mean = torch.zeros(1, 16, 1, 1, 1)
        _std = torch.ones(1, 16, 1, 1, 1)

        def _wan21_in(t):
            assert t.dim() == 5, (
                "process_latent_in must receive the 5D [B,C,1,H,W] tensor"
            )
            return (t - _mean) / _std

        def _wan21_out(t):
            assert t.dim() == 5, (
                "process_latent_out must receive the 5D tensor"
            )
            return t * _std + _mean

        model.model.process_latent_in = _wan21_in
        model.model.process_latent_out = _wan21_out
    else:
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
        assert hfn._require_flow_model(model) == ("const", 2)

    def test_flow_gate_rejects_eps_with_actionable_message(self):
        model = _mock_flow_model(prediction_mixin="EPS")
        with pytest.raises(ValueError, match="PixelRush"):
            hfn._require_flow_model(model)

    def test_flow_gate_accepts_3d_format_image_models(self):
        """Krea2 plan S2: 3D-FORMAT (Wan21: Krea2, Qwen-Image) is an image
        model with a 5D layout — the gate accepts it and reports dims."""
        model = _mock_flow_model(latent_dimensions=3)
        family, dims = hfn._require_flow_model(model)
        assert family == "const"
        assert dims == 3

    def test_flow_gate_rejects_bad_latent_dimensions(self):
        model = _mock_flow_model(latent_dimensions=4)
        with pytest.raises(ValueError, match="latent_dimensions"):
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

    def test_5d_bridge_on_3d_format_models(self, monkeypatch):
        """Krea2 plan S3: with latent_dimensions=3 the adapter unsqueezes
        4D -> 5D before process_latent_in (Wan21 mean/std stats are
        [1,C,1,1,1] — 5D broadcast), the model call sees 5D, and the result
        squeezes back to 4D."""
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model(latent_dimensions=3)

        # Wan21-style process_in: (x - mean)/std with [1,C,1,1,1] views —
        # asserts the 5D input the bridge must provide.
        mean = torch.zeros(1, 16, 1, 1, 1)
        std = torch.ones(1, 16, 1, 1, 1)
        seen_ndim = []

        def process_latent_in(t):
            seen_ndim.append(t.dim())
            assert t.dim() == 5, "Wan21 stats need the 5D tensor"
            return (t - mean) / std

        model.model.process_latent_in = process_latent_in
        model.model.process_latent_out = lambda t: t * std + mean

        adapter = hfn._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=1.0,
            latent_dimensions=3)
        torch.manual_seed(0)
        x_vae = torch.randn(1, 16, 8, 8)
        out = adapter(x_vae, sigma=0.6)

        assert seen_ndim == [5], "process_latent_in must receive 5D"
        assert fx.sampling_calls[-1]["x_shape"] == (1, 16, 1, 8, 8), (
            "the model call must receive the 5D [B,C,1,H,W] tensor"
        )
        assert out.dim() == 4, "the adapter must return the squeezed 4D x0"
        assert out.shape == x_vae.shape
        assert torch.isfinite(out).all()

    def test_4d_models_unaffected_by_bridge(self, monkeypatch):
        """latent_dimensions=2 never unsqueezes — the bridge is inert."""
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        adapter = hfn._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=1.0, latent_dimensions=2)
        torch.manual_seed(0)
        adapter(torch.randn(1, 4, 8, 8), sigma=0.6)
        assert fx.sampling_calls[-1]["x_shape"] == (1, 4, 8, 8)

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
        content = (pathlib.Path(__file__).parent.parent / "nodes"
                   / "hiflow.py").read_text(encoding="utf-8")
        assert "sampling_function" in content
        assert "diffusion_model(" not in content, (
            "hiflow node module must not bypass sampling_function"
        )


@pytest.mark.unit
class TestCfgEmptyNegativeSkip:
    """Z-Image bugfix 2026-09-03 (blurred + over-vibrant output).

    A guidance-free model with an empty-string CLIPTextEncode negative
    produced cond_scale amplification of a meaningless (cond − uncond)
    difference — the "high-CFG look" on a model that has no CFG at all.
    The adapter must force cond_scale=1.0 when the negative carries no
    tokens, so sampling_function's cfg1-skip runs the conditional branch
    only."""

    def _make_conds(self, neg_tokens):
        pos = [(torch.ones(1, 8), {"side": "positive"})]
        if neg_tokens is None:
            neg = []  # no negative entries at all
        elif neg_tokens == 0:
            neg = [(torch.zeros(1, 0), {"side": "negative"})]  # 0 tokens
        else:
            neg = [(torch.zeros(1, neg_tokens), {"side": "negative"})]
        return pos, neg

    def test_empty_negative_forces_cfg_skip(self, monkeypatch):
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        pos, neg = self._make_conds(neg_tokens=None)
        adapter = hfn._make_predict_x0(model, pos, neg, cfg_scale=3.5)
        adapter(torch.randn(1, 4, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["cond_scale"] == 1.0, (
            "empty negative must force cond_scale=1.0 (cfg1 skip)"
        )

    def test_zero_token_negative_forces_cfg_skip(self, monkeypatch):
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        pos, neg = self._make_conds(neg_tokens=0)
        adapter = hfn._make_predict_x0(model, pos, neg, cfg_scale=4.5)
        adapter(torch.randn(1, 4, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["cond_scale"] == 1.0

    def test_real_negative_keeps_cfg(self, monkeypatch):
        """A genuine negative (tokens present) keeps the requested CFG —
        the paper's guidance still applies for CFG-trained flow models."""
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        pos, neg = self._make_conds(neg_tokens=77)
        adapter = hfn._make_predict_x0(model, pos, neg, cfg_scale=3.5)
        adapter(torch.randn(1, 4, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["cond_scale"] == 3.5

    def test_cfg_one_with_real_negative_unchanged(self, monkeypatch):
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        pos, neg = self._make_conds(neg_tokens=77)
        adapter = hfn._make_predict_x0(model, pos, neg, cfg_scale=1.0)
        adapter(torch.randn(1, 4, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["cond_scale"] == 1.0

    def test_list_token_negative_detected(self, monkeypatch):
        """Token lists (batched tokenizations) count as a real negative."""
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        pos = [(torch.ones(1, 8), {"side": "positive"})]
        neg = [([torch.zeros(1, 4)], {"side": "negative"})]
        adapter = hfn._make_predict_x0(model, pos, neg, cfg_scale=3.5)
        adapter(torch.randn(1, 4, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["cond_scale"] == 3.5


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
    """Channels-last VAE boundary (Z-Image bugfix 2026-09-03): ComfyUI
    VAE.decode returns [B,H,W,3] and VAE.encode expects [B,H,W,3] — the
    adapters pass that layout through untouched. Converting to channels-
    first here corrupted spatial dims in encode ("kernel size can't be
    greater than actual input size")."""

    def test_decode_passes_comfyui_channels_last(self):
        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(1, 8, 8, 3),  # ComfyUI layout
        )
        dec, _ = hfn._make_vae_adapters(vae, torch.device("cpu"))
        img = dec(torch.randn(1, 4, 8, 8))
        assert img.shape == (1, 8, 8, 3), (
            "decode output must stay [B,H,W,3] — VAE.encode expects it"
        )

    def test_decode_5d_squeezed(self):
        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(1, 8, 8, 3, 1),
        )
        dec, _ = hfn._make_vae_adapters(vae, torch.device("cpu"))
        img = dec(torch.randn(1, 4, 8, 8))
        assert img.ndim == 4

    def test_encode_accepts_channels_last(self):
        vae = types.SimpleNamespace(
            encode=lambda im: {"samples": torch.randn(1, 4, 8, 8)},
        )
        _, enc = hfn._make_vae_adapters(vae, torch.device("cpu"))
        lat = enc(torch.randn(1, 8, 8, 3))  # [B,H,W,3] in
        assert lat.shape == (1, 4, 8, 8)

    def test_encode_passes_tensor_untouched(self):
        """Regression (the Z-Image crash): the adapter must hand the tensor
        to vae.encode UNTOUCHED — ComfyUI's encode does its own movedim(-1,1)
        on the channels-last input; any pre-conversion corrupts the axes."""
        seen = {}

        def encode(im):
            seen["shape"] = tuple(im.shape)
            # ComfyUI sd.py:1359 — encode does movedim(-1, 1) internally on
            # the channels-last input it receives.
            return {"samples": torch.randn(1, 4, 8, 8)}

        vae = types.SimpleNamespace(encode=encode)
        _, enc = hfn._make_vae_adapters(vae, torch.device("cpu"))
        img = torch.randn(1, 8, 8, 3)  # correct channels-last input
        lat = enc(img)
        assert seen["shape"] == (1, 8, 8, 3), (
            "adapter must hand vae.encode the channels-last tensor untouched"
        )
        assert lat.shape == (1, 4, 8, 8)

    def test_decode_accepts_dict_latent(self):
        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(1, 8, 8, 3),
        )
        dec, _ = hfn._make_vae_adapters(vae, torch.device("cpu"))
        img = dec({"samples": torch.randn(1, 4, 8, 8)})
        assert img.ndim == 4

    @staticmethod
    def _fake_3d_vae(latent_channels=16):
        """Fake Qwen-VAE (latent_dim=3, not_video): decode takes 5D latents
        and returns [B,T,H,W,3] channels-last; encode takes channels-last
        and returns 5D latents [B,C,T,h,w] — the REAL sd.py shapes
        (:1209 decode, :1338 encode with the not_video unsqueeze)."""
        calls = {"decode_shapes": [], "encode_shapes": []}

        def decode(z):
            calls["decode_shapes"].append(tuple(z.shape))
            b, c, t, h, w = z.shape
            assert t == 1, "image models decode T=1 latents"
            # The raw decoder emits channels-FIRST pixels [B,3,T,H,W];
            # VAE.decode's final movedim(1,-1) (sd.py:1283) yields [B,T,H,W,3].
            pixels = torch.randn(b, 3, t, h * 16, w * 16)
            return pixels.movedim(1, -1)

        def encode(im):
            # sd.py:1342 — encode does movedim(-1,1) then, being not_video,
            # unsqueezes the 4D channels-first image to 5D itself.
            calls["encode_shapes"].append(tuple(im.shape))
            assert im.shape[-1] == 3, "channels-last input expected"
            b = im.shape[0]
            lat = torch.randn(b, latent_channels, 1,
                              im.shape[-3] // 16, im.shape[-2] // 16)
            return lat

        vae = types.SimpleNamespace(
            decode=decode, encode=encode, latent_dim=3,
            downscale_ratio=(lambda a: max(0, (a + 15) // 16), 16, 16),
        )
        return vae, calls

    def test_3d_vae_decode_unsqueezes_and_slices_frame(self):
        """Krea2 plan S4: decode unsqueezes the 4D latent to 5D before
        vae.decode and slices the 5D image to its first frame -> [B,H,W,3]."""
        vae, calls = self._fake_3d_vae()
        dec, _ = hfn._make_vae_adapters(vae, torch.device("cpu"))
        img = dec(torch.randn(1, 16, 8, 8))
        assert calls["decode_shapes"] == [(1, 16, 1, 8, 8)], (
            "the raw vae.decode must receive the 5D latent"
        )
        assert img.shape == (1, 128, 128, 3), (
            "first temporal frame of the 5D channels-last image"
        )

    def test_3d_vae_encode_slices_to_4d(self):
        """Krea2 plan S4: encode feeds the channels-last 4D image (the VAE
        itself unsqueezes) and slices the 5D latent to 4D for the core."""
        vae, calls = self._fake_3d_vae()
        _, enc = hfn._make_vae_adapters(vae, torch.device("cpu"))
        lat = enc(torch.randn(1, 128, 128, 3))
        assert calls["encode_shapes"] == [(1, 128, 128, 3)], (
            "encode must receive the channels-last image (the VAE "
            "unsqueezes internally — sd.py:1342-1346)"
        )
        assert lat.shape == (1, 16, 8, 8), "4D latent for the core"

    def test_3d_vae_roundtrip_end_to_end(self):
        """decode -> [B,H,W,3] -> encode -> 4D: the full anchor/pixel path
        works on a 3D-format VAE."""
        vae, calls = self._fake_3d_vae()
        dec, enc = hfn._make_vae_adapters(vae, torch.device("cpu"))
        lat_in = torch.randn(1, 16, 8, 8)
        lat_out = enc(dec(lat_in))
        assert lat_out.shape == lat_in.shape
        assert len(calls["decode_shapes"]) == 1
        assert len(calls["encode_shapes"]) == 1


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

    def test_channels_last_passthrough(self):
        """Z-Image bugfix 2026-09-03: pixel-mode images are channels-last
        [B,H,W,3] (the ComfyUI VAE boundary) — sharpen must return the SAME
        layout (it converts around the channels-first blur internally)."""
        img = torch.zeros(1, 16, 16, 3)
        img[0, 8, 8, 0] = 1.0
        out = hfn._sharpen(img, alpha=1.0)
        assert out.shape == img.shape, "layout must be preserved"
        assert out[0, 8, 8, 0].item() > 1.0, "impulse peak must sharpen"
        # Same values as the channels-first path (layout conversion is
        # exact, not an approximation).
        out_cf = hfn._sharpen(img.movedim(-1, 1), alpha=1.0)
        assert torch.allclose(out, out_cf.movedim(1, -1), atol=1e-6)

    def test_5d_image_sliced_to_first_frame(self):
        """Krea2 plan S4 backstop: a 5D decode output [B,T,H,W,3] slices to
        its first frame — matching the 4D channels-last path exactly."""
        img4 = torch.zeros(1, 16, 16, 3)
        img4[0, 8, 8, 0] = 1.0
        img5 = img4.unsqueeze(1)  # [B, 1, H, W, 3]
        out5 = hfn._sharpen(img5, alpha=1.0)
        assert out5.shape == img4.shape, "must return the 4D frame"
        out4 = hfn._sharpen(img4, alpha=1.0)
        assert torch.allclose(out5, out4, atol=1e-6)

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
    def _run_execute(self, monkeypatch, scale=4.0, upsampling="latent",
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
            # ComfyUI VAE boundary layout: decode -> [B,H,W,3] channels-last,
            # encode <- [B,H,W,3]. (Z-Image bugfix 2026-09-03.) The fake
            # encode returns the MODEL's 16 latent channels so the init
            # anchor matches the stage latent (always-pixel anchor, plan D2).
            decode=lambda z: torch.randn(
                z.shape[0], z.shape[-2] * 8, z.shape[-1] * 8, 3),
            encode=lambda im: {"samples": torch.randn(
                1, 16, im.shape[-3] // 8, im.shape[-2] // 8)},
            downscale_ratio=8,
        )
        z = torch.randn(*latent)

        result = hfn.HiFlowNode.execute(
            model, vae, COND_POS, COND_NEG, {"samples": z},
            cfg=3.5, steps=4, guidance=4.5, steps_per_stage=2,
            tau=0.5, filter_ratio=0.2, alpha_scale=1.0, beta_scale=0.5,
            upsampling=upsampling, scale_factor=scale,
            noise_seed=0, denoise=1.0,
        )
        # NodeOutput wraps the payload positionally; unwrap to the dict.
        samples = result[0]["samples"] if not hasattr(result, "shape") \
            else result
        out_dict = {"samples": samples}
        return out_dict, FakePBar

    def test_execute_returns_latent_dict(self, monkeypatch):
        result, _ = self._run_execute(monkeypatch, scale=4.0)
        samples = result["samples"]
        assert samples.ndim == 4
        assert torch.isfinite(samples).all()

    def test_execute_doubles_resolution(self, monkeypatch):
        """16x16 latent + scale 4.0 -> 64x64 latent out (two stages)."""
        result, _ = self._run_execute(monkeypatch, scale=4.0)
        assert tuple(result["samples"].shape[-2:]) == (64, 64)

    def test_execute_scale_two_one_stage(self, monkeypatch):
        """scale 2.0 on a 16x16 latent -> one doubling stage -> 32x32."""
        result, _ = self._run_execute(monkeypatch, scale=2.0)
        assert tuple(result["samples"].shape[-2:]) == (32, 32)

    def test_execute_rejects_non_flow_model(self, monkeypatch):
        with pytest.raises(ValueError, match="PixelRush"):
            self._run_execute(monkeypatch, prediction_mixin="EPS")

    def test_execute_accepts_5d_t1_latent(self, monkeypatch):
        """Krea2 plan S2: a Wan21 5D [B,C,1,H,W] latent squeezes to 4D on
        entry and the node runs end-to-end."""
        result, _ = self._run_execute(
            monkeypatch, scale=2.0, latent=(1, 16, 1, 16, 16))
        assert tuple(result["samples"].shape[-2:]) == (32, 32)

    def test_execute_rejects_multi_frame_latent(self, monkeypatch):
        """T>1 is a video latent — rejected with the per-frame message."""
        with pytest.raises(ValueError, match="multi-frame"):
            self._run_execute(monkeypatch, latent=(1, 4, 4, 16, 16))

    @staticmethod
    def _mock_3d_vae():
        """3D-format fake VAE (latent_dim=3, tuple downscale_ratio) — see
        TestVaeAdapters._fake_3d_vae for the sd.py-faithful shapes."""
        def decode(z):
            if z.dim() == 4:
                z = z.unsqueeze(2)
            b, c, t, h, w = z.shape
            return torch.randn(b, 3, t, h * 16, w * 16).movedim(1, -1)

        def encode(im):
            assert im.shape[-1] == 3
            b = im.shape[0]
            return torch.randn(
                b, 16, 1, im.shape[-3] // 16, im.shape[-2] // 16)

        return types.SimpleNamespace(
            decode=decode, encode=encode, latent_dim=3,
            downscale_ratio=(lambda a: max(0, (a + 15) // 16), 16, 16))

    def test_execute_end_to_end_krea2_style(self, monkeypatch):
        """Krea2 crown test: a 3D-format model (latent_dimensions=3) + Qwen
        VAE (latent_dim=3, tuple downscale_ratio) + 4D empty latent in ->
        5D [B,C,1,H,W] scaled latent out, everything bridged."""
        FakePBar = _install_fake_pbar_utils(monkeypatch)
        _install_fake_comfy(monkeypatch)
        model = _mock_flow_model(latent_dimensions=3)
        sys.modules["comfy.samplers"].calculate_sigmas = (
            lambda ms, scheduler, steps:
            torch.cat([torch.linspace(1.0, 0.1, steps), torch.zeros(1)])
        )
        vae = self._mock_3d_vae()
        z = torch.zeros(1, 16, 16, 16)   # 4D empty latent (2D generator node)
        result = hfn.HiFlowNode.execute(
            model, vae, COND_POS, COND_NEG, {"samples": z},
            cfg=3.5, steps=4, guidance=4.5, steps_per_stage=2,
            tau=0.5, filter_ratio=0.2, alpha_scale=1.0, beta_scale=0.5,
            upsampling="latent", scale_factor=2.0,
            noise_seed=0, denoise=1.0,
        )
        samples = result[0]["samples"] if not hasattr(result, "shape") \
            else result
        assert tuple(samples.shape) == (1, 16, 1, 32, 32), (
            "3D-format output: 5D [B,C,1,H,W] at the doubled size"
        )
        assert torch.isfinite(samples).all()
        _ = FakePBar

    def test_downscale_ratio_tuple_form(self):
        """Krea2 plan S5: Qwen VAEs report downscale_ratio as
        (callable, 16, 16) — _downscale_ratio takes the h_ratio slot."""
        vae = types.SimpleNamespace(
            downscale_ratio=(lambda a: max(0, (a + 15) // 16), 16, 16))
        assert hfn._downscale_ratio(vae) == 16
        vae2 = types.SimpleNamespace(downscale_ratio=8)
        assert hfn._downscale_ratio(vae2) == 8

    def test_execute_empty_latent_channels_repeated(self, monkeypatch):
        """An empty 4-channel latent for a 16-channel model is repeated."""

        _install_fake_pbar_utils(monkeypatch)
        _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()  # latent_channels=16
        sys.modules["comfy.samplers"].calculate_sigmas = (
            lambda ms, scheduler, steps:
            torch.cat([torch.linspace(1.0, 0.1, steps), torch.zeros(1)])
        )
        vae = types.SimpleNamespace(
            # VAE adapters are ALWAYS wired now (always-pixel anchor, plan
            # D2) — the empty-latent path also needs working fakes.
            decode=lambda z: torch.randn(
                z.shape[0], z.shape[-2] * 8, z.shape[-1] * 8, 3),
            encode=lambda im: {"samples": torch.randn(
                1, 16, im.shape[-3] // 8, im.shape[-2] // 8)},
            downscale_ratio=8,
        )
        z = torch.zeros(1, 4, 16, 16)  # empty, wrong channel count
        result = hfn.HiFlowNode.execute(
            model, vae, COND_POS, COND_NEG, {"samples": z},
            steps=2, steps_per_stage=2, tau=0.5,
            scale_factor=2.0,
            noise_seed=0, denoise=1.0,
        )
        samples = result[0]["samples"] if not hasattr(result, "shape") \
            else result
        assert tuple(samples.shape[-2:]) == (32, 32)

    def test_execute_content_denoise1_warns(self, monkeypatch, caplog):
        """A content latent with denoise=1.0 annihilates the image — the
        node warns (the 'connecting the real latent does nothing' report)."""
        import logging
        _install_fake_pbar_utils(monkeypatch)
        _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        sys.modules["comfy.samplers"].calculate_sigmas = (
            lambda ms, scheduler, steps:
            torch.cat([torch.linspace(1.0, 0.1, steps), torch.zeros(1)])
        )
        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(
                z.shape[0], z.shape[-2] * 8, z.shape[-1] * 8, 3),
            encode=lambda im: {"samples": torch.randn(
                1, 16, im.shape[-3] // 8, im.shape[-2] // 8)},
            downscale_ratio=8,
        )
        z = torch.randn(1, 16, 16, 16)  # NON-empty content latent
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            hfn.HiFlowNode.execute(
                model, vae, COND_POS, COND_NEG, {"samples": z},
                steps=2, steps_per_stage=2, tau=0.5,
                scale_factor=2.0, denoise=1.0,
            )
        assert any("denoise=1.0" in r.message for r in caplog.records)

    def test_execute_img2img_no_warning_on_denoise(self, monkeypatch, caplog):
        """denoise < 1 with a content latent is the intended img2img path —
        no foot-gun warning fires."""
        import logging
        _install_fake_pbar_utils(monkeypatch)
        _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        sys.modules["comfy.samplers"].calculate_sigmas = (
            lambda ms, scheduler, steps:
            torch.cat([torch.linspace(1.0, 0.1, steps), torch.zeros(1)])
        )
        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(
                z.shape[0], z.shape[-2] * 8, z.shape[-1] * 8, 3),
            encode=lambda im: {"samples": torch.randn(
                1, 16, im.shape[-3] // 8, im.shape[-2] // 8)},
            downscale_ratio=8,
        )
        z = torch.randn(1, 16, 16, 16)
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            hfn.HiFlowNode.execute(
                model, vae, COND_POS, COND_NEG, {"samples": z},
                steps=2, steps_per_stage=2, tau=0.5,
                scale_factor=2.0, denoise=0.6,
            )
        assert not any(
            "denoise=1.0" in r.message for r in caplog.records)

    def test_execute_img2img_truncates_entry_sigma(self, monkeypatch):
        """denoise=0.6 with a content latent must lower the first sigma the
        base adapter sees below 1.0 (KSampler truncation, wired through)."""
        _install_fake_pbar_utils(monkeypatch)
        fake = _install_fake_comfy(monkeypatch)
        model = _mock_flow_model()
        sys.modules["comfy.samplers"].calculate_sigmas = (
            lambda ms, scheduler, steps:
            torch.cat([torch.linspace(1.0, 0.1, steps), torch.zeros(1)])
        )
        vae = types.SimpleNamespace(
            decode=lambda z: torch.randn(
                z.shape[0], z.shape[-2] * 8, z.shape[-1] * 8, 3),
            encode=lambda im: {"samples": torch.randn(
                1, 16, im.shape[-3] // 8, im.shape[-2] // 8)},
            downscale_ratio=8,
        )
        z = torch.randn(1, 16, 16, 16)
        hfn.HiFlowNode.execute(
            model, vae, COND_POS, COND_NEG, {"samples": z},
            steps=4, steps_per_stage=2, tau=0.5,
            scale_factor=2.0, denoise=0.5,
        )
        first_t = fake.sampling_calls[0]["timestep"]
        first_sigma = float(first_t) / 1000.0  # the mock's DiscreteFlow probe
        assert first_sigma < 1.0, (
            f"denoise=0.5 must truncate the entry sigma below 1 "
            f"(got {first_sigma})"
        )

    def test_schema_has_denoise_input(self):
        """The V3-schema mock has no introspectable input list — pin the
        input's presence the established way (schema source grep)."""
        import inspect
        import pathlib
        src = pathlib.Path(inspect.getfile(hfn)).read_text(encoding="utf-8")
        assert '"denoise", default=1.0' in src
        assert '"noise_seed", default=0' in src

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
            # ComfyUI decode layout: [B, H, W, 3] channels-last.
            return torch.randn(1, z.shape[-2] * 8, z.shape[-1] * 8, 3)

        def encode(im):
            # ComfyUI encode layout: expects [B, H, W, 3] channels-last.
            assert im.shape[-1] == 3, (
                "encode must receive channels-last [B,H,W,3]"
            )
            return {"samples": torch.randn(
                1, 16, im.shape[-3] // 8, im.shape[-2] // 8)}

        vae = types.SimpleNamespace(decode=decode, encode=encode,
                                    downscale_ratio=8)
        result = hfn.HiFlowNode.execute(
            model, vae, COND_POS, COND_NEG,
            {"samples": torch.randn(1, 16, 16, 16)},
            steps=2, steps_per_stage=2, tau=0.5,
            upsampling="pixel", scale_factor=2.0,
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
        return (pathlib.Path(__file__).parent.parent / "nodes" / "hiflow.py").read_text(encoding="utf-8")

    def test_schema_inputs_and_paper_defaults(self):
        src = self._src()
        for inp in ["model", "vae", "positive", "negative", "latent_image",
                    "cfg", "steps", "guidance", "steps_per_stage", "tau",
                    "filter_ratio", "alpha_scale", "beta_scale", "upsampling",
                    "scale_factor"]:
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
        assert 'category="WMNodes/image"' in self._src()

    def test_validate_inputs_none_passes(self):
        assert hfn.HiFlowNode.validate_inputs(scale_factor=None) is True

    def test_validate_inputs_small_rejected(self):
        result = hfn.HiFlowNode.validate_inputs(scale_factor=16.0)
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
        assert m and m.group(1) == "2.15.0"
        assert "### v2.15.0" in readme

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
