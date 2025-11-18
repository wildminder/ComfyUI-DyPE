import logging
import types
from contextvars import ContextVar
from typing import Any, Iterable

import torch

from comfy.text_encoders import llama as llama_module

from .rope import get_1d_rotary_pos_embed


logger = logging.getLogger(__name__)

_QWEN_DYPE_CONTEXT: ContextVar[dict[str, Any] | None] = ContextVar(
    "qwen_dype_context", default=None
)
_ORIGINAL_PRECOMPUTE = llama_module.precompute_freqs_cis
_PATCH_INSTALLED = False


def _safe_attribute_names(obj: Any, *, max_items: int = 20) -> list[str]:
    """
    Return a truncated and sorted list of attribute names for debugging.
    """
    if obj is None:
        return []

    obj_dict = getattr(obj, "__dict__", None)
    if not isinstance(obj_dict, dict):
        return []

    keys = sorted(obj_dict.keys())
    if len(keys) > max_items:
        keys = keys[:max_items] + ["..."]
    return keys


def _describe_clip_structure(clip: Any) -> str:
    """
    Build a human-readable summary of key attributes for debugging.
    """
    parts: list[str] = []
    visited: set[int] = set()

    def _type_string(obj: Any) -> str:
        if obj is None:
            return "None"
        return f"{type(obj).__module__}.{type(obj).__name__}"

    # Attributes we attempt to expand recursively.
    expansion_candidates = [
        "cond_stage_model",
        "clip",
        "clip_model",
        "transformer",
        "model",
        "text_model",
        "language_model",
        "vision_model",
        "encoder",
        "text_encoder",
    ]

    max_depth = 5
    max_module_items = 6

    def _should_expand(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, bytes, int, float, bool)):
            return False
        return True

    def _summarize(path: str, obj: Any, depth: int) -> None:
        if depth > max_depth:
            parts.append(f"{path} <depth limit reached>")
            return
        indent = "  " * depth
        if obj is None:
            parts.append(f"{indent}{path} (None)")
            return

        obj_id = id(obj)
        if obj_id in visited:
            parts.append(f"{indent}{path} ({_type_string(obj)} - already visited)")
            return
        visited.add(obj_id)

        attrs = _safe_attribute_names(obj)
        parts.append(f"{indent}{path} ({_type_string(obj)}): attrs={attrs}")

        candidate_summaries: list[str] = []
        expansions: list[tuple[str, Any]] = []

        for candidate in expansion_candidates:
            try:
                value = getattr(obj, candidate)
            except AttributeError:
                continue
            except Exception as err:  # pragma: no cover - defensive diagnostics
                candidate_summaries.append(f"{candidate}=<error: {err}>")
                continue

            candidate_summaries.append(f"{candidate}={_type_string(value)}")
            if _should_expand(value):
                expansions.append((f"{path}.{candidate}", value))

        modules_info: list[str] = []
        if hasattr(obj, "_modules"):
            try:
                module_items = getattr(obj, "_modules")
            except Exception as err:  # pragma: no cover - defensive diagnostics
                modules_info.append(f"_modules=<error: {err}>")
                module_items = None
            if isinstance(module_items, dict):
                for name, submodule in list(module_items.items())[:max_module_items]:
                    modules_info.append(f"{name}={_type_string(submodule)}")
                    if _should_expand(submodule):
                        expansions.append((f"{path}._modules['{name}']", submodule))
                if len(module_items) > max_module_items:
                    modules_info.append("...")

        if candidate_summaries:
            parts.append(f"{indent}  key types: {', '.join(candidate_summaries)}")
        if modules_info:
            parts.append(f"{indent}  modules: {', '.join(modules_info)}")

        for next_path, value in expansions:
            _summarize(next_path, value, depth + 1)

    _summarize("clip", clip, 0)
    return "\n".join(parts)


def _normalize_method(method: str) -> str:
    method_lower = method.lower()
    if method_lower not in {"yarn", "ntk", "base"}:
        raise ValueError(f"Unsupported DyPE method '{method}'.")
    return method_lower


def _ensure_llama_patch_installed() -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    def _patched_precompute_freqs_cis(
        head_dim: int,
        position_ids: torch.Tensor,
        theta: float | Iterable[float],
        rope_scale=None,
        rope_dims=None,
        device=None,
    ):
        cfg = _QWEN_DYPE_CONTEXT.get()
        if cfg is None or not cfg.get("enable_dype", False):
            return _ORIGINAL_PRECOMPUTE(
                head_dim, position_ids, theta, rope_scale, rope_dims, device=device
            )

        try:
            return _compute_dype_freqs(
                head_dim=head_dim,
                position_ids=position_ids,
                theta=theta,
                rope_scale=rope_scale,
                rope_dims=rope_dims,
                device=device,
                cfg=cfg,
            )
        except Exception:
            logger.exception("Falling back to original RoPE frequencies.")
            return _ORIGINAL_PRECOMPUTE(
                head_dim, position_ids, theta, rope_scale, rope_dims, device=device
            )

    llama_module.precompute_freqs_cis = _patched_precompute_freqs_cis
    _PATCH_INSTALLED = True


def _compute_dype_freqs(
    head_dim: int,
    position_ids: torch.Tensor,
    theta: float | Iterable[float],
    rope_scale,
    rope_dims,
    device,
    cfg: dict[str, Any],
):
    pos = position_ids
    if device is not None:
        pos = pos.to(device)
    pos = pos.to(torch.float32)
    if pos.ndim == 1:
        pos = pos.unsqueeze(0)

    primary_axis = pos[0]
    current_ctx = int(primary_axis.max().item()) + 1
    base_ctx_len = int(cfg["base_ctx_len"])
    max_ctx_len = int(cfg["max_ctx_len"])
    method = cfg["method"]

    if current_ctx <= base_ctx_len or method == "base":
        return _ORIGINAL_PRECOMPUTE(
            head_dim, position_ids, theta, rope_scale, rope_dims, device=device
        )

    target_ctx = min(max_ctx_len, current_ctx)
    span = max(1, max_ctx_len - base_ctx_len)
    progress = min(max(current_ctx - base_ctx_len, 0) / span, 1.0)
    dype_exponent = float(cfg.get("dype_exponent", 1.0))
    enable_dype = bool(cfg.get("enable_dype", True))

    theta_values = theta if isinstance(theta, (list, tuple)) else [theta]
    if isinstance(rope_scale, (list, tuple)):
        rope_scale_values = list(rope_scale)
    else:
        rope_scale_values = [rope_scale] * len(theta_values)

    results = []
    for idx, theta_value in enumerate(theta_values):
        scale_value = rope_scale_values[idx]
        if scale_value is None:
            linear_factor = 1.0
        else:
            linear_factor = scale_value

        common_kwargs = {
            "theta": theta_value,
            "linear_factor": linear_factor,
            "use_real": True,
            "repeat_interleave_real": True,
            "freqs_dtype": torch.float32,
            "dype": enable_dype,
            "current_timestep": progress,
            "dype_exponent": dype_exponent,
        }

        if method == "yarn":
            specific_kwargs = {
                "yarn": True,
                "max_pe_len": torch.tensor(
                    target_ctx, dtype=torch.float32, device=pos.device
                ),
                "ori_max_pe_len": base_ctx_len,
            }
        elif method == "ntk":
            specific_kwargs = {
                "ntk_factor": max(target_ctx / max(base_ctx_len, 1), 1.0),
            }
        else:
            specific_kwargs = {}

        try:
            if rope_dims is not None and pos.shape[0] >= len(rope_dims):
                cos_segments = []
                sin_segments = []
                for axis_idx, axis_dim in enumerate(rope_dims):
                    axis_pos = pos[axis_idx : axis_idx + 1]
                    cos_axis, sin_axis = get_1d_rotary_pos_embed(
                        dim=axis_dim * 2,
                        pos=axis_pos,
                        **common_kwargs,
                        **specific_kwargs,
                    )
                    cos_segments.append(cos_axis.squeeze(0))
                    sin_segments.append(sin_axis.squeeze(0))

                cos = torch.cat(cos_segments, dim=-1).unsqueeze(0)
                sin = torch.cat(sin_segments, dim=-1).unsqueeze(0)
            else:
                cos, sin = get_1d_rotary_pos_embed(
                    dim=head_dim,
                    pos=pos,
                    **common_kwargs,
                    **specific_kwargs,
                )
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
        except RuntimeError as err:
            logger.warning(
                "DyPE RoPE computation failed (%s); using original frequencies.", err
            )
            return _ORIGINAL_PRECOMPUTE(
                head_dim, position_ids, theta, rope_scale, rope_dims, device=device
            )

        results.append((cos.to(pos.device), sin.to(pos.device)))

    if len(results) == 1:
        return results[0]
    return results


def apply_dype_to_qwen_clip(
    clip,
    method: str,
    enable_dype: bool,
    dype_exponent: float,
    base_ctx_len: int,
    max_ctx_len: int,
):
    """
    Clone the provided CLIP/Qwen text encoder and install the DyPE RoPE patch.
    """
    if base_ctx_len <= 0 or max_ctx_len <= base_ctx_len:
        raise ValueError("max_ctx_len must be greater than base_ctx_len.")

    method_normalized = _normalize_method(method)

    clone = clip.clone()
    _ensure_llama_patch_installed()

    transformer = getattr(clone.cond_stage_model, "transformer", None)
    if transformer is None or not hasattr(transformer, "model"):
        clip_structure = _describe_clip_structure(clone)
        logger.error(
            "Provided clip does not expose a transformer model. Structure snapshot:\n%s",
            clip_structure,
        )
        raise ValueError(
            "Provided clip does not expose a transformer model.\n"
            f"Observed structure:\n{clip_structure}"
        )

    core_model = transformer.model
    if not hasattr(core_model, "config") or not hasattr(core_model.config, "rope_theta"):
        raise ValueError("This text encoder is not compatible with DyPE for Qwen.")

    config = {
        "enable_dype": enable_dype,
        "method": method_normalized,
        "dype_exponent": float(dype_exponent),
        "base_ctx_len": int(base_ctx_len),
        "max_ctx_len": int(max_ctx_len),
    }
    core_model._qwen_dype_config = config

    if not hasattr(core_model, "_qwen_dype_wrapped"):
        original_forward = core_model.forward

        def forward_with_dype(self, *args, **kwargs):
            cfg = getattr(self, "_qwen_dype_config", None)
            if cfg is not None and cfg.get("enable_dype", False):
                token = _QWEN_DYPE_CONTEXT.set(cfg)
                try:
                    return original_forward(*args, **kwargs)
                finally:
                    _QWEN_DYPE_CONTEXT.reset(token)
            return original_forward(*args, **kwargs)

        core_model.forward = types.MethodType(forward_with_dype, core_model)
        core_model._qwen_dype_wrapped = True

    return clone


# Expose internals for testing.
LLAMA_MODULE = llama_module
