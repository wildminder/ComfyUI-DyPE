import logging
import pathlib
import sys
import types

import pytest
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_pillow_stub() -> None:
    if "PIL" in sys.modules:
        return

    def _missing_pillow(*_args, **_kwargs):
        raise RuntimeError("Pillow is not available in the test environment.")

    pil_module = types.ModuleType("PIL")
    image_module = types.ModuleType("PIL.Image")
    image_draw_module = types.ModuleType("PIL.ImageDraw")
    image_font_module = types.ModuleType("PIL.ImageFont")

    image_module.new = _missing_pillow  # type: ignore[attr-defined]
    image_draw_module.Draw = _missing_pillow  # type: ignore[attr-defined]
    image_font_module.truetype = _missing_pillow  # type: ignore[attr-defined]

    pil_module.Image = image_module  # type: ignore[attr-defined]
    pil_module.ImageDraw = image_draw_module  # type: ignore[attr-defined]
    pil_module.ImageFont = image_font_module  # type: ignore[attr-defined]

    sys.modules["PIL"] = pil_module
    sys.modules["PIL.Image"] = image_module
    sys.modules["PIL.ImageDraw"] = image_draw_module
    sys.modules["PIL.ImageFont"] = image_font_module


_install_pillow_stub()


def _install_psutil_stub() -> None:
    if "psutil" in sys.modules:
        return

    psutil_module = types.ModuleType("psutil")

    def _virtual_memory():
        return types.SimpleNamespace(total=0, available=0)

    psutil_module.virtual_memory = _virtual_memory  # type: ignore[attr-defined]

    sys.modules["psutil"] = psutil_module


_install_psutil_stub()


def _install_av_stub() -> None:
    if "av" in sys.modules:
        return

    av_module = types.ModuleType("av")
    container_module = types.ModuleType("av.container")
    subtitles_module = types.ModuleType("av.subtitles")
    subtitles_stream_module = types.ModuleType("av.subtitles.stream")

    class _PlaceholderInputContainer:  # pragma: no cover - diagnostic stub
        pass

    class _PlaceholderSubtitleStream:  # pragma: no cover - diagnostic stub
        pass

    container_module.InputContainer = _PlaceholderInputContainer  # type: ignore[attr-defined]
    subtitles_stream_module.SubtitleStream = _PlaceholderSubtitleStream  # type: ignore[attr-defined]

    av_module.container = container_module  # type: ignore[attr-defined]
    av_module.subtitles = subtitles_module  # type: ignore[attr-defined]
    subtitles_module.stream = subtitles_stream_module  # type: ignore[attr-defined]

    sys.modules["av"] = av_module
    sys.modules["av.container"] = container_module
    sys.modules["av.subtitles"] = subtitles_module
    sys.modules["av.subtitles.stream"] = subtitles_stream_module


_install_av_stub()


def _install_comfy_stub() -> None:
    if "comfy.text_encoders.llama" in sys.modules:
        return

    llama_module = types.ModuleType("comfy.text_encoders.llama")

    def _precompute_freqs_cis(
        head_dim: int,
        position_ids: torch.Tensor,
        theta: float,
        rope_scale=None,
        rope_dims=None,
        device=None,
    ):
        device_to_use = device if device is not None else position_ids.device
        pos = position_ids.to(device_to_use)
        if pos.ndim == 1:
            pos = pos.unsqueeze(0)
        pos = pos.to(torch.float32)

        dim_range = torch.arange(0, head_dim, 2, device=device_to_use, dtype=torch.float32)
        inv_freq = theta ** (-dim_range / head_dim)
        freqs = torch.einsum("...s,d->...sd", pos, inv_freq)
        cos = torch.cos(freqs).repeat_interleave(2, dim=-1)
        sin = torch.sin(freqs).repeat_interleave(2, dim=-1)
        if rope_dims is None:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return cos.to(torch.float32), sin.to(torch.float32)

    llama_module.precompute_freqs_cis = _precompute_freqs_cis  # type: ignore[attr-defined]

    text_encoders_module = types.ModuleType("comfy.text_encoders")
    text_encoders_module.llama = llama_module  # type: ignore[attr-defined]

    comfy_module = types.ModuleType("comfy")
    comfy_module.text_encoders = text_encoders_module  # type: ignore[attr-defined]
    comfy_module.__path__ = []  # type: ignore[attr-defined]

    model_patcher_module = types.ModuleType("comfy.model_patcher")

    class _StubDiffusionModel:
        def __init__(self):
            self.patch_size = 2
            self.pe_embedder = _StubEmbedder()

    class _StubEmbedder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = 10_000.0
            self.axes_dim = [4, 4, 4]

        def forward(self, ids):
            batch = ids.shape[0]
            tokens = ids.shape[1]
            return torch.zeros(batch, 1, 6, 2, 2)

    class _ModelSamplingFlux:
        def __init__(self):
            self._dype_patched = False
            self.sigma_max = torch.tensor(1.0)

        def sigma(self, timestep):
            return timestep

    class _StubModelWrapper:
        def __init__(self):
            self.model_sampling = _ModelSamplingFlux()
            self.diffusion_model = _StubDiffusionModel()

    class _StubModelPatcher:
        def __init__(self):
            self.model = _StubModelWrapper()
            self._wrapper = None

        def clone(self):
            cloned = _StubModelPatcher()
            cloned.model = self.model
            return cloned

        def add_object_patch(self, path, obj):
            if path == "diffusion_model.pe_embedder":
                self.model.diffusion_model.pe_embedder = obj

        def set_model_unet_function_wrapper(self, wrapper):
            self._wrapper = wrapper

    model_patcher_module.ModelPatcher = _StubModelPatcher  # type: ignore[attr-defined]

    model_sampling_module = types.ModuleType("comfy.model_sampling")

    model_sampling_module.ModelSamplingFlux = _ModelSamplingFlux  # type: ignore[attr-defined]
    model_sampling_module.flux_time_shift = lambda mu, sigma, t: t  # type: ignore[attr-defined]

    comfy_module.model_patcher = model_patcher_module  # type: ignore[attr-defined]
    comfy_module.model_sampling = model_sampling_module  # type: ignore[attr-defined]

    sys.modules["comfy"] = comfy_module
    sys.modules["comfy.text_encoders"] = text_encoders_module
    sys.modules["comfy.text_encoders.llama"] = llama_module
    sys.modules.setdefault("comfy.utils", types.ModuleType("comfy.utils"))
    sys.modules["comfy.model_patcher"] = model_patcher_module
    sys.modules["comfy.model_sampling"] = model_sampling_module


_install_comfy_stub()


def _install_comfy_api_stub() -> None:
    if "comfy_api.latest" in sys.modules:
        return

    comfy_api_module = types.ModuleType("comfy_api")
    comfy_api_module.__path__ = []  # type: ignore[attr-defined]

    latest_module = types.ModuleType("comfy_api.latest")

    class _ComfyExtension:  # pragma: no cover - diagnostic stub
        pass

    class _BaseNode:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)

    class _Placeholder:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class _NodeOutput(_Placeholder):
        pass

    class _FieldFactory:
        def Input(self, *args, **kwargs):
            return _Placeholder(*args, **kwargs)

        def Output(self, *args, **kwargs):
            return _Placeholder(*args, **kwargs)

    io_module = types.ModuleType("comfy_api.latest.io")
    io_module.ComfyNode = _BaseNode  # type: ignore[attr-defined]
    io_module.Schema = _Placeholder  # type: ignore[attr-defined]
    io_module.Model = _FieldFactory()
    io_module.Int = _FieldFactory()
    io_module.Combo = _FieldFactory()
    io_module.Boolean = _FieldFactory()
    io_module.Float = _FieldFactory()
    io_module.Clip = _FieldFactory()
    io_module.NodeOutput = _NodeOutput  # type: ignore[attr-defined]

    latest_module.ComfyExtension = _ComfyExtension  # type: ignore[attr-defined]
    latest_module.io = io_module  # type: ignore[attr-defined]

    sys.modules["comfy_api"] = comfy_api_module
    sys.modules["comfy_api.latest"] = latest_module
    sys.modules["comfy_api.latest.io"] = io_module


_install_comfy_api_stub()

from src import qwen_patch, qwen_spatial  # noqa: E402


def _call_patched_precompute(position_ids, cfg):
    qwen_patch._ensure_llama_patch_installed()  # pylint: disable=protected-access
    token = qwen_patch._QWEN_DYPE_CONTEXT.set(cfg)  # pylint: disable=protected-access
    try:
        return qwen_patch.LLAMA_MODULE.precompute_freqs_cis(  # type: ignore[attr-defined]
            head_dim=128,
            position_ids=position_ids,
            theta=1_000_000.0,
            device=position_ids.device,
        )
    finally:
        qwen_patch._QWEN_DYPE_CONTEXT.reset(token)  # pylint: disable=protected-access


def test_patched_matches_original_before_base_length():
    position_ids = torch.arange(0, 8192, device="cpu").unsqueeze(0)
    cfg = {
        "enable_dype": True,
        "method": "ntk",
        "dype_exponent": 2.0,
        "base_ctx_len": 8192,
        "max_ctx_len": 65536,
    }

    patched_cos, patched_sin = _call_patched_precompute(position_ids, cfg)
    orig_cos, orig_sin = qwen_patch._ORIGINAL_PRECOMPUTE(  # pylint: disable=protected-access
        128, position_ids, 1_000_000.0, None, None, device=position_ids.device
    )

    assert torch.allclose(patched_cos, orig_cos)
    assert torch.allclose(patched_sin, orig_sin)


def test_patched_extends_when_beyond_base_length():
    position_ids = torch.arange(0, 32768, device="cpu").unsqueeze(0)
    cfg = {
        "enable_dype": True,
        "method": "ntk",
        "dype_exponent": 2.0,
        "base_ctx_len": 8192,
        "max_ctx_len": 131072,
    }

    patched_cos, patched_sin = _call_patched_precompute(position_ids, cfg)
    orig_cos, orig_sin = qwen_patch._ORIGINAL_PRECOMPUTE(  # pylint: disable=protected-access
        128, position_ids, 1_000_000.0, None, None, device=position_ids.device
    )

    assert patched_cos.shape == orig_cos.shape
    assert patched_sin.shape == orig_sin.shape
    assert patched_cos.dtype == torch.float32
    assert patched_sin.dtype == torch.float32
    assert not torch.allclose(patched_cos, orig_cos)
    assert not torch.allclose(patched_sin, orig_sin)


class _DummyClip:
    def __init__(self):
        self.cond_stage_model = types.SimpleNamespace(foo="bar")

    def clone(self):
        cloned = _DummyClip()
        cloned.cond_stage_model = self.cond_stage_model
        return cloned


def test_apply_dype_reports_structure_when_missing_transformer(caplog):
    clip = _DummyClip()
    caplog.set_level(logging.ERROR)

    with pytest.raises(ValueError) as exc_info:
        qwen_patch.apply_dype_to_qwen_clip(
            clip=clip,
            method="ntk",
            enable_dype=True,
            dype_exponent=1.0,
            base_ctx_len=1,
            max_ctx_len=2,
        )

    message = str(exc_info.value)
    assert "clip (" in message
    assert "cond_stage_model" in message
    assert "key types: cond_stage_model" in message
    assert "transformer" in message
    assert any(
        "Structure snapshot" in record.message for record in caplog.records
    ), "Expected structure snapshot to be logged when transformer is missing."


class _RecordingEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = 10_000.0
        self.axes_dim = [4, 4, 4]
        self.calls = 0

    def forward(self, ids):
        self.calls += 1
        batch = ids.shape[0]
        return torch.zeros(batch, 1, 6, 2, 2)


def _make_token_ids(expanded: bool) -> torch.Tensor:
    text_tokens = torch.tensor(
        [
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ]
    )
    if expanded:
        image_tokens = torch.tensor(
            [
                [0.0, -2.0, -2.0],
                [0.0, -2.0, 1.0],
                [0.0, 1.0, -2.0],
                [0.0, 1.0, 1.0],
            ]
        )
    else:
        image_tokens = torch.tensor(
            [
                [0.0, -1.0, -1.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        )
    return torch.cat([text_tokens, image_tokens], dim=0).unsqueeze(0)


class _NestedTransformerModel:
    def __init__(self):
        self.config = types.SimpleNamespace(rope_theta=1_000_000.0)
        self.forward_calls = 0

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        return args, kwargs


class _NestedTransformer:
    def __init__(self):
        self.model = _NestedTransformerModel()
        self._modules = {"model": self.model}


class _NestedModuleWrapper:
    def __init__(self, transformer):
        self.transformer = transformer
        self._modules = {}


class _NestedCondStageModel:
    def __init__(self):
        transformer = _NestedTransformer()
        self.transformer = None
        self._modules = {"qwen25_7b": _NestedModuleWrapper(transformer)}


class _NestedClip:
    def __init__(self):
        self.cond_stage_model = _NestedCondStageModel()

    def clone(self):
        cloned = _NestedClip()
        cloned.cond_stage_model = self.cond_stage_model
        return cloned


def test_apply_dype_finds_nested_transformer():
    clip = _NestedClip()

    patched_clip = qwen_patch.apply_dype_to_qwen_clip(
        clip=clip,
        method="yarn",
        enable_dype=True,
        dype_exponent=2.0,
        base_ctx_len=16,
        max_ctx_len=32,
    )

    nested = patched_clip.cond_stage_model._modules["qwen25_7b"].transformer.model
    assert nested._qwen_dype_config["max_ctx_len"] == 32

    result = nested.forward()
    assert result == ((), {})
    assert nested.forward_calls == 1


def test_dype_qwen_clip_execute_emits_info(caplog):
    from __init__ import DyPE_QWEN_CLIP  # type: ignore

    clip = _NestedClip()
    caplog.set_level(logging.INFO, logger="__init__")
    caplog.set_level(logging.INFO, logger="src.qwen_patch")

    result = DyPE_QWEN_CLIP.execute(
        clip=clip,
        method="ntk",
        enable_dype=True,
        dype_exponent=1.5,
        base_ctx_len=16,
        max_ctx_len=64,
    )

    node_messages = [record.message for record in caplog.records if record.name == "__init__"]
    assert any("DyPE_QwenClip: requested patch" in msg for msg in node_messages)

    patch_messages = [record.message for record in caplog.records if record.name == "src.qwen_patch"]
    assert any("patching transformer" in msg for msg in patch_messages)
    assert hasattr(result, "args") and result.args[0].cond_stage_model is clip.cond_stage_model


def test_qwen_spatial_embedder_falls_back_to_backing():
    backing = _RecordingEmbedder()
    embedder = qwen_spatial.QwenSpatialPosEmbed(
        theta=backing.theta,
        axes_dim=backing.axes_dim,
        patch_size=2,
        method="yarn",
        enable_dype=False,
        dype_exponent=2.0,
        base_resolution=(32, 32),
        target_resolution=(32, 32),
        backing_embedder=backing,
    )

    ids = _make_token_ids(expanded=False)
    output = embedder(ids)

    assert backing.calls == 1
    assert output.shape[0] == 1
    assert output.shape[1] == 1


def test_qwen_spatial_embedder_extends_when_grid_grows():
    backing = _RecordingEmbedder()
    embedder = qwen_spatial.QwenSpatialPosEmbed(
        theta=backing.theta,
        axes_dim=backing.axes_dim,
        patch_size=2,
        method="ntk",
        enable_dype=False,
        dype_exponent=1.0,
        base_resolution=(32, 32),
        target_resolution=(64, 64),
        backing_embedder=backing,
    )

    ids = _make_token_ids(expanded=True)
    output = embedder(ids)

    assert backing.calls == 0
    assert output.shape == (1, 1, 6, 2, 2)


def test_apply_dype_to_qwen_image_installs_embedder_and_wrapper():
    patcher = comfy.model_patcher.ModelPatcher()

    patched = qwen_spatial.apply_dype_to_qwen_image(
        model=patcher,
        width=2048,
        height=2048,
        method="ntk",
        enable_dype=True,
        dype_exponent=1.5,
        base_width=1024,
        base_height=1024,
        base_shift=1.15,
        max_shift=1.35,
    )

    embedder = patched.model.diffusion_model.pe_embedder
    assert isinstance(embedder, qwen_spatial.QwenSpatialPosEmbed)
    assert hasattr(patched, "_wrapper") and patched._wrapper is not None
    assert patched.model.model_sampling._dype_patched is True

    args_dict = {
        "input": torch.zeros(1),
        "timestep": torch.tensor([0.5]),
        "c": {},
    }

    def _dummy_model_fn(x, timestep, **kwargs):
        return x, timestep, kwargs

    result = patched._wrapper(_dummy_model_fn, args_dict)
    assert hasattr(embedder, "current_timestep") and embedder.current_timestep == 0.5
    assert isinstance(result, tuple) and result[1] is args_dict["timestep"]
