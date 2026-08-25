"""
Shared fixtures for ComfyUI-DyPE tests.
Provides mock objects that simulate ComfyUI's model structure
without requiring a full ComfyUI installation.
"""
import copy
import os
import sys
import types

import pytest
import torch

# Make the tests/ directory importable so shared helpers such as
# ``_spa_math_helpers`` can be imported as a flat module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# --- Mock ComfyUI modules so tests can run standalone ---

def _create_mock_comfy_modules():
    """Create minimal mock modules for comfy.* imports."""

    # comfy.model_patcher.ModelPatcher
    mock_model_patcher = types.ModuleType("comfy.model_patcher")

    class MockModelPatcher:
        def __init__(self):
            self.model = types.SimpleNamespace()
            self.model.diffusion_model = types.SimpleNamespace()
            self._object_patches = {}
            self._unet_wrapper = None

        def clone(self):
            new = MockModelPatcher()
            new.model = copy.copy(self.model)
            new.model.diffusion_model = copy.copy(self.model.diffusion_model)
            new._object_patches = dict(self._object_patches)
            new._unet_wrapper = self._unet_wrapper
            return new

        def add_object_patch(self, path, obj):
            self._object_patches[path] = obj

        def set_model_unet_function_wrapper(self, fn):
            self._unet_wrapper = fn

    mock_model_patcher.ModelPatcher = MockModelPatcher

    # comfy.model_sampling
    mock_model_sampling = types.ModuleType("comfy.model_sampling")

    class CONST:
        pass

    class ModelSamplingFlux:
        def __init__(self, model_config=None):
            self.sigma_max = torch.tensor(1.0)
            self._shift = 1.0

        def set_parameters(self, shift=1.0):
            self._shift = shift

    mock_model_sampling.CONST = CONST
    mock_model_sampling.ModelSamplingFlux = ModelSamplingFlux

    # comfy (top-level)
    mock_comfy = types.ModuleType("comfy")
    mock_comfy.model_patcher = mock_model_patcher
    mock_comfy.model_sampling = mock_model_sampling

    # Register in sys.modules
    sys.modules.setdefault("comfy", mock_comfy)
    sys.modules.setdefault("comfy.model_patcher", mock_model_patcher)
    sys.modules.setdefault("comfy.model_sampling", mock_model_sampling)

    return MockModelPatcher


# Only mock if comfy is not available (CI/standalone testing).  The import
# itself is the availability probe — hence the noqa on F401.
try:
    import comfy  # noqa: F401

    MockModelPatcher = None
except ImportError:
    MockModelPatcher = _create_mock_comfy_modules()


@pytest.fixture(autouse=True)
def _install_mock_attention_module():
    """Provide ``comfy.ldm.modules.attention`` with a pristine SDPA ``optimized_attention``.

    The SPA hook patches the *module-level* ``comfy.ldm.modules.attention.optimized_attention``.
    Under pytest ``comfy`` is a mock module, so ``comfy.ldm.modules.attention`` does not exist
    by default.  This fixture builds the dotted import chain and resets ``optimized_attention`` to
    a plain scaled-dot-product-attention shim (scale=1.0, matching the HRDiT reference) before
    EVERY test, so the hook installs/uninstalls in isolation and never leaks across tests.
    """
    import torch.nn.functional as F

    # The parent modules are registered only so the dotted import of
    # ``comfy.ldm.modules.attention`` resolves; the attention module itself is
    # what tests patch/read.
    sys.modules.setdefault("comfy.ldm", types.ModuleType("comfy.ldm"))
    sys.modules.setdefault(
        "comfy.ldm.modules", types.ModuleType("comfy.ldm.modules")
    )
    attn_mod = sys.modules.setdefault(
        "comfy.ldm.modules.attention", types.ModuleType("comfy.ldm.modules.attention")
    )

    # REAL ComfyUI signature (comfy/ldm/modules/attention.py::attention_pytorch):
    #   (q, k, v, heads, mask=None, attn_precision=None,
    #    skip_reshape=False, skip_output_reshape=False, **kwargs)
    # The mock MUST mirror it bit-for-bit (positional slots 5-8) — the pre-fix
    # mock inverted ``skip_reshape``/``mask`` and therefore never caught the
    # wrapper's positional mis-forwarding (closed-loop mock-fidelity failure,
    # plan 2026-08-16 G2).  A conformance tripwire lives in
    # tests/test_orig_call_convention.py.
    def _sdpa(q, k, v, heads, mask=None, attn_precision=None,
              skip_reshape=False, skip_output_reshape=False, **kwargs):
        return F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=0.0, is_causal=False)

    attn_mod.optimized_attention = _sdpa
    # Real ComfyUI aliases the masked symbol to the SAME function
    # (attention.py: ``optimized_attention_masked = optimized_attention``).
    # Mirror the alias so masked-backend bindings (Krea-2/Qwen/Z-Image) resolve.
    attn_mod.optimized_attention_masked = _sdpa
    yield attn_mod


@pytest.fixture(autouse=True)
def _reset_hap_layer_ordinal():
    """Reset the HAP plan-layer ordinal around every test (2026-08-19 fix).

    The ordinal advances only for plan-covered attention calls; a test that
    drives the wrapper must not leak a non-zero ordinal into the next test.
    Mirrors the per-file ``set_hrdit_layer_idx(0)`` hygiene for the raw counter.
    """
    from src.spa_context import set_hap_layer_idx

    set_hap_layer_idx(0)
    yield
    set_hap_layer_idx(0)


@pytest.fixture
def sample_pos_1d():
    """1D position tensor: (batch=1, seq_len=64, axes=1)"""
    return torch.arange(64, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)


@pytest.fixture
def sample_pos_3d():
    """3D position tensor: (batch=1, seq_len=4096, axes=3) for 64x64 grid"""
    B, H, W = 1, 64, 64
    L = H * W
    ids = torch.zeros(B, L, 3)
    ids[..., 0] = torch.arange(L)  # text/sequential
    ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()  # height
    ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()  # width
    return ids


@pytest.fixture
def mock_flux_model():
    """Mock a FLUX-like model structure for patch_utils tests."""
    try:
        from comfy.model_patcher import ModelPatcher
    except ImportError:
        ModelPatcher = MockModelPatcher

    m = ModelPatcher()
    dm = m.model.diffusion_model

    # FLUX-like attributes
    dm.__class__.__name__ = "Flux"
    dm.patch_size = 2

    # Mock pe_embedder
    pe = types.SimpleNamespace()
    pe.theta = 10000
    pe.axes_dim = [16, 56, 56]
    dm.pe_embedder = pe

    # Mock model_sampling
    m.model.model_sampling = types.SimpleNamespace()
    m.model.model_sampling.sigma_max = torch.tensor(1.0)
    m.model.model_config = types.SimpleNamespace()

    return m
