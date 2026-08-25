"""
Root conftest.py — installs ComfyUI mocks before any test collection.
This must exist at the project root so pytest loads it before importing
the project's __init__.py (which requires comfy_api).
"""
import copy
import os
import sys
import types

import torch

# Ensure the project root is on sys.path so `from src.xxx import ...` works
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Tell pytest to not try to import the root __init__.py or src/ as test modules
collect_ignore = ["__init__.py", "src"]

# Pre-register the root package in sys.modules so pytest doesn't try to
# import __init__.py (which has relative imports that fail outside ComfyUI).
# This allows `from src.xxx import ...` to work in tests.
import importlib.util

_pkg_name = os.path.basename(_PROJECT_ROOT)
if _pkg_name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        _pkg_name,
        os.path.join(_PROJECT_ROOT, "__init__.py"),
        submodule_search_locations=[_PROJECT_ROOT],
    )
    _mod = importlib.util.module_from_spec(_spec)
    _mod.__path__ = [_PROJECT_ROOT]
    _mod.__package__ = _pkg_name
    sys.modules[_pkg_name] = _mod
    # Do NOT execute the module (it would fail with relative imports)
    # Just register it as a namespace so `src` subpackage is importable


def _install_comfyui_mocks():
    """Install minimal ComfyUI mock modules into sys.modules."""
    if "comfy_api" in sys.modules:
        return  # Already available (real ComfyUI or already mocked)

    # --- comfy_api.latest ---
    mock_comfy_api = types.ModuleType("comfy_api")
    mock_comfy_api_latest = types.ModuleType("comfy_api.latest")

    class ComfyExtension:
        async def get_node_list(self):
            return []
        async def on_load(self):
            pass

    class _Schema:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _Input:
        def __init__(self, name, **kwargs):
            self.name = name
            self.kwargs = kwargs

    class _Output:
        def __init__(self, display_name=None, **kwargs):
            self.display_name = display_name

    class _ComfyNode:
        @classmethod
        def define_schema(cls):
            raise NotImplementedError

    def _node_output(*args, **kwargs):
        """Mock io.NodeOutput — a plain function so ``io.NodeOutput(...)``
        (attribute access on the _IO INSTANCE) does not bind ``self`` as the
        first positional arg (which would corrupt the output tuple)."""
        return args

    class _IO:
        Schema = _Schema
        ComfyNode = _ComfyNode
        NodeOutput = staticmethod(_node_output)

        class Model:
            Input = _Input
            Output = _Output
        class Int:
            Input = _Input
        class Float:
            Input = _Input
        class Boolean:
            Input = _Input
        class Combo:
            Input = _Input
        class String:
            Input = _Input
            Output = _Output
        class Image:
            Input = _Input
            Output = _Output
        class Latent:
            Input = _Input
        class Mask:
            Input = _Input
            Output = _Output
        class Conditioning:
            Input = _Input
            Output = _Output
        class Hidden:
            unique_id = "unique_id"
            prompt = "prompt"

        @staticmethod
        def Custom(name):
            """Mock for io.Custom("TYPE") -> object with .Input/.Output."""
            class _CustomType:
                Input = _Input
                Output = _Output
            _CustomType.type_name = name
            return _CustomType

    class _UI:
        class PreviewImage:
            def __init__(self, *args):
                pass

    mock_comfy_api_latest.ComfyExtension = ComfyExtension
    mock_comfy_api_latest.io = _IO()
    mock_comfy_api_latest.ui = _UI()

    # --- comfy.model_patcher ---
    mock_model_patcher = types.ModuleType("comfy.model_patcher")

    class ModelPatcher:
        def __init__(self, model=None, *args, **kwargs):
            self.model = model or types.SimpleNamespace()
            if not hasattr(self.model, 'diffusion_model'):
                self.model.diffusion_model = types.SimpleNamespace()
            self._object_patches = {}
            self._unet_wrapper = None

        def clone(self):
            new = ModelPatcher()
            new.model = copy.copy(self.model)
            new.model.diffusion_model = copy.copy(self.model.diffusion_model)
            new._object_patches = dict(self._object_patches)
            new._unet_wrapper = self._unet_wrapper
            return new

        def add_object_patch(self, path, obj):
            self._object_patches[path] = obj

        def set_model_unet_function_wrapper(self, fn):
            self._unet_wrapper = fn

    mock_model_patcher.ModelPatcher = ModelPatcher

    # --- comfy.model_sampling ---
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

    # --- comfy (top-level) ---
    mock_comfy = types.ModuleType("comfy")
    mock_comfy.model_patcher = mock_model_patcher
    mock_comfy.model_sampling = mock_model_sampling

    # Register all in sys.modules
    sys.modules["comfy"] = mock_comfy
    sys.modules["comfy.model_patcher"] = mock_model_patcher
    sys.modules["comfy.model_sampling"] = mock_model_sampling
    sys.modules["comfy_api"] = mock_comfy_api
    sys.modules["comfy_api.latest"] = mock_comfy_api_latest


# Install mocks immediately (before pytest collects any test modules)
_install_comfyui_mocks()
