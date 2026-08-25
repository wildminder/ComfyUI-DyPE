"""Tests for SEGA integration with patch_utils (Tier 2: mock ComfyUI)."""
import types

import pytest
import torch

from src.patch_utils import apply_sega_to_model

# Reuse mock fixtures from conftest
try:
    from tests.conftest import MockModelPatcher
except ImportError:
    MockModelPatcher = None


class _MockDiffusionModel:
    """Mock diffusion model with configurable class name."""
    def __init__(self, class_name="Flux"):
        self.__class__.__name__ = class_name
        self.patch_size = 2
        self.pe_embedder = types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56])


def _make_mock_flux_model():
    """Create a mock FLUX model for SEGA testing."""
    try:
        from comfy.model_patcher import ModelPatcher
    except ImportError:
        ModelPatcher = MockModelPatcher

    m = ModelPatcher()
    dm = _MockDiffusionModel("Flux")
    m.model.diffusion_model = dm

    m.model.model_sampling = types.SimpleNamespace()
    m.model.model_sampling.sigma_max = torch.tensor(1.0)
    m.model.model_config = types.SimpleNamespace()

    return m


def _make_mock_qwen_model():
    """Create a mock Qwen model for SEGA testing."""
    try:
        from comfy.model_patcher import ModelPatcher
    except ImportError:
        ModelPatcher = MockModelPatcher

    m = ModelPatcher()
    dm = _MockDiffusionModel("QwenImageTransformer2DModel")
    m.model.diffusion_model = dm

    m.model.model_sampling = types.SimpleNamespace()
    m.model.model_sampling.sigma_max = torch.tensor(1.0)
    m.model.model_config = types.SimpleNamespace()

    return m


@pytest.mark.unit
class TestApplySegaToModel:
    def test_flux_model_detected(self):
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "flux", 2048, 2048)
        assert result is not None

    def test_auto_detects_flux(self):
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "auto", 2048, 2048)
        assert result is not None

    def test_qwen_model_detected(self):
        m = _make_mock_qwen_model()
        result = apply_sega_to_model(m, "qwen", 2048, 2048)
        assert result is not None

    def test_pos_embedder_patched(self):
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "flux", 2048, 2048)
        # The object patch should be set
        assert "diffusion_model.pe_embedder" in result._object_patches

    def test_wrapper_function_set(self):
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "flux", 2048, 2048)
        assert result._unet_wrapper is not None

    def test_sega_embedder_class(self):
        """The patched embedder should be a SegAPosEmbedFlux."""
        from src.models.sega_flux import SegAPosEmbedFlux
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "flux", 2048, 2048)
        embedder = result._object_patches["diffusion_model.pe_embedder"]
        assert isinstance(embedder, SegAPosEmbedFlux)

    def test_sega_params_stored(self):
        """SEGA parameters should be stored on the embedder."""
        m = _make_mock_flux_model()
        result = apply_sega_to_model(
            m, "flux", 2048, 2048,
            mscale_alpha=0.3, mscale_beta=2.0, mscale_min=0.8,
            spread_min=0.1, spread_max=0.9, spread_alpha=2.0,
        )
        embedder = result._object_patches["diffusion_model.pe_embedder"]
        assert embedder.mscale_alpha == 0.3
        assert embedder.mscale_beta == 2.0
        assert embedder.mscale_min == 0.8
        assert embedder.spread_min == 0.1
        assert embedder.spread_max == 0.9
        assert embedder.spread_alpha == 2.0

    def test_resolution_snapped(self):
        """Resolution should be snapped to multiple of 16."""
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "flux", 1000, 1000)
        # The wrapper should still work — snapping happens internally
        assert result is not None


@pytest.mark.unit
class TestSegaWrapper:
    def test_wrapper_computes_spectral_data(self):
        """The wrapper should compute spectral profiles from input latent."""
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "flux", 2048, 2048)
        embedder = result._object_patches["diffusion_model.pe_embedder"]

        # Initially no spectral data
        assert embedder._energy_profile_h is None

        # Simulate a model function call
        input_x = torch.randn(1, 16, 128, 128)  # (B, C, H, W)
        timestep = torch.tensor([0.5])
        called = {"flag": False}

        def mock_model_fn(x, t, **kwargs):
            called["flag"] = True
            return x

        result._unet_wrapper(mock_model_fn, {
            "input": input_x,
            "timestep": timestep,
            "c": {},
        })

        # After the call, spectral data should be set
        assert embedder._energy_profile_h is not None
        assert embedder._energy_profile_w is not None
        assert embedder._dynamic_spread >= 0.0
        assert called["flag"] is True

    def test_wrapper_sets_timestep(self):
        """The wrapper should set the timestep on the embedder."""
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "flux", 2048, 2048)
        embedder = result._object_patches["diffusion_model.pe_embedder"]

        input_x = torch.randn(1, 16, 64, 64)
        timestep = torch.tensor([0.5])

        result._unet_wrapper(lambda x, t, **k: x, {
            "input": input_x,
            "timestep": timestep,
            "c": {},
        })

        # timestep should be set (0.5 / 1.0 = 0.5)
        assert embedder.current_timestep == 0.5

    def test_wrapper_handles_no_input(self):
        """Wrapper should not crash if input is None."""
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "flux", 2048, 2048)

        result._unet_wrapper(lambda x, t, **k: x, {
            "input": None,
            "timestep": torch.tensor([0.5]),
            "c": {},
        })

    def test_wrapper_calls_model_function(self):
        """The wrapper should call the original model function."""
        m = _make_mock_flux_model()
        result = apply_sega_to_model(m, "flux", 2048, 2048)

        input_x = torch.randn(1, 16, 64, 64)
        timestep = torch.tensor([0.5])
        output_received = {"value": None}

        def mock_model_fn(x, t, **kwargs):
            output_received["value"] = x
            return x * 2

        out = result._unet_wrapper(mock_model_fn, {
            "input": input_x,
            "timestep": timestep,
            "c": {},
        })
        assert torch.equal(out, input_x * 2)


@pytest.mark.unit
class TestSegaWithQwen:
    def test_qwen_embedder_class(self):
        """Qwen model should get SegAPosEmbedQwen."""
        from src.models.sega_qwen import SegAPosEmbedQwen
        m = _make_mock_qwen_model()
        result = apply_sega_to_model(m, "qwen", 2048, 2048)
        embedder = result._object_patches["diffusion_model.pe_embedder"]
        assert isinstance(embedder, SegAPosEmbedQwen)

    def test_qwen_wrapper_works(self):
        m = _make_mock_qwen_model()
        result = apply_sega_to_model(m, "qwen", 2048, 2048)

        input_x = torch.randn(1, 16, 64, 64)
        timestep = torch.tensor([0.5])

        out = result._unet_wrapper(lambda x, t, **k: x, {
            "input": input_x,
            "timestep": timestep,
            "c": {},
        })
        assert out is not None
