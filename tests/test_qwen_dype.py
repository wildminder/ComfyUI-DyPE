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

    class _StubModelSampler:
        def __init__(self):
            self._dype_patched = False

    class _StubDiffusionModel:
        def __init__(self):
            self.patch_size = 1
            self.pe_embedder = None

    class _StubModelWrapper:
        def __init__(self):
            self.model_sampling = _StubModelSampler()
            self.diffusion_model = _StubDiffusionModel()

    class _StubModelPatcher:
        def __init__(self):
            self.model = _StubModelWrapper()

        def clone(self):
            return self

    model_patcher_module.ModelPatcher = _StubModelPatcher  # type: ignore[attr-defined]

    model_sampling_module = types.ModuleType("comfy.model_sampling")

    class _ModelSamplingFlux:  # pragma: no cover - diagnostic stub
        pass

    model_sampling_module.ModelSamplingFlux = _ModelSamplingFlux  # type: ignore[attr-defined]

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

from src import qwen_patch  # noqa: E402


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
