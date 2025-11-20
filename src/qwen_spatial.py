import logging
import types
from dataclasses import dataclass
from typing import Iterable, Tuple

import torch
import torch.nn as nn

from comfy import model_sampling
from comfy.model_patcher import ModelPatcher

from .rope import get_1d_rotary_pos_embed


logger = logging.getLogger(__name__)
if not logger.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("[DyPE QwenImage] %(message)s")
    )
    logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def _normalize_method(method: str) -> str:
    method_lower = method.lower()
    if method_lower not in {"yarn", "ntk", "base"}:
        raise ValueError(f"Unsupported DyPE method '{method}'.")
    return method_lower


def _ensure_positive(value: int, fallback: int = 1) -> int:
    return value if value > 0 else fallback


def _compute_axis_len(resolution: int, patch_size: int) -> int:
    latent = max(resolution // 8, 1)
    return _ensure_positive((latent + (patch_size // 2)) // patch_size)


def _iter_axes(values: torch.Tensor, mask: torch.Tensor) -> Iterable[torch.Tensor]:
    for batch_idx in range(values.shape[0]):
        batch_mask = mask[batch_idx]
        if not torch.any(batch_mask):
            continue
        yield values[batch_idx][batch_mask]


def _estimate_axis_extent(values: torch.Tensor, mask: torch.Tensor, fallback: int) -> int:
    extents: list[int] = []
    for batch_vals in _iter_axes(values, mask):
        extent = int(torch.round(batch_vals.max() - batch_vals.min()).item()) + 1
        extents.append(_ensure_positive(extent, fallback))
    if not extents:
        return fallback
    return max(extents)


@dataclass(frozen=True)
class _GridConfig:
    base_axes: Tuple[int, int]
    max_axes: Tuple[int, int]

    @property
    def base_ctx_len(self) -> int:
        return self.base_axes[0] * self.base_axes[1]

    @property
    def max_ctx_len(self) -> int:
        return self.max_axes[0] * self.max_axes[1]


class QwenSpatialPosEmbed(nn.Module):
    """
    DyPE-enabled positional embedder for Qwen Image transformer blocks.
    Replicates the structure produced by comfy.ldm.flux.layers.EmbedND while
    allowing dynamic extrapolation of spatial rotary embeddings.
    """

    def __init__(
        self,
        *,
        theta: float,
        axes_dim: Iterable[int],
        patch_size: int,
        method: str,
        enable_dype: bool,
        dype_exponent: float,
        base_resolution: Tuple[int, int],
        target_resolution: Tuple[int, int],
        backing_embedder: nn.Module,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self.patch_size = int(patch_size)
        self.method = _normalize_method(method)
        self.enable_dype = bool(enable_dype)
        self.dype_exponent = float(dype_exponent)
        self.backing_embedder = backing_embedder

        self.current_timestep: float = 1.0

        base_axes = (
            _compute_axis_len(base_resolution[1], self.patch_size),
            _compute_axis_len(base_resolution[0], self.patch_size),
        )
        max_axes = (
            _compute_axis_len(target_resolution[1], self.patch_size),
            _compute_axis_len(target_resolution[0], self.patch_size),
        )
        object.__setattr__(  # dataclass immutability helper
            self,
            "_grid",
            _GridConfig(base_axes=base_axes, max_axes=max_axes),
        )

    @property
    def grid(self) -> _GridConfig:
        return getattr(self, "_grid")

    def set_timestep(self, timestep: float) -> None:
        self.current_timestep = float(timestep)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if not isinstance(ids, torch.Tensor):
            raise TypeError("Expected ids to be a torch.Tensor.")
        if ids.ndim != 3 or ids.shape[-1] != 3:
            raise ValueError(
                "QwenSpatialPosEmbed expects ids shaped [batch, tokens, 3]. "
                f"Received shape {tuple(ids.shape)}."
            )

        if self.method == "base" and not self.enable_dype:
            return self.backing_embedder(ids)

        ids = ids.to(torch.float32)
        axis0 = ids[..., 0]
        axis1 = ids[..., 1]
        axis2 = ids[..., 2]

        diff1 = (axis1 - axis0).abs()
        diff2 = (axis2 - axis0).abs()
        image_mask = torch.logical_or(diff1 > 1e-4, diff2 > 1e-4)

        grid = self.grid
        current_h = _estimate_axis_extent(axis1, image_mask, grid.base_axes[0])
        current_w = _estimate_axis_extent(axis2, image_mask, grid.base_axes[1])
        current_ctx = max(current_h * current_w, 1)

        target_h = min(grid.max_axes[0], current_h)
        target_w = min(grid.max_axes[1], current_w)

        needs_extension = (
            self.method != "base"
            and (target_h > grid.base_axes[0] or target_w > grid.base_axes[1])
        )

        if not needs_extension and not self.enable_dype:
            return self.backing_embedder(ids)

        emb_parts: list[torch.Tensor] = []
        freqs_dtype = torch.float32

        for axis_idx, axis_dim in enumerate(self.axes_dim):
            axis_pos = ids[..., axis_idx]
            axis_name = ["index", "height", "width"][axis_idx] if axis_idx < 3 else f"axis_{axis_idx}"

            common_kwargs = {
                "dim": int(axis_dim),
                "pos": axis_pos,
                "theta": float(self.theta),
                "repeat_interleave_real": True,
                "use_real": True,
                "freqs_dtype": freqs_dtype,
            }

            dype_kwargs = {}
            if self.enable_dype and axis_idx > 0:
                dype_kwargs = {
                    "dype": True,
                    "current_timestep": float(self.current_timestep),
                    "dype_exponent": float(self.dype_exponent),
                }

            if axis_idx == 1:
                base_len = grid.base_axes[0]
                current_len = current_h
                target_len = max(target_h, base_len)
            elif axis_idx == 2:
                base_len = grid.base_axes[1]
                current_len = current_w
                target_len = max(target_w, base_len)
            else:
                base_len = int(ids.shape[1])
                current_len = base_len
                target_len = current_len

            logger.info(
                "axis=%s base_len=%d current_len=%d target_len=%d method=%s enable_dype=%s timestep=%.4f",
                axis_name,
                base_len,
                current_len,
                target_len,
                self.method,
                self.enable_dype,
                self.current_timestep,
            )

            if axis_idx == 0 or self.method == "base":
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)
            else:
                if self.method == "yarn" and target_len > base_len:
                    max_pe_len = torch.tensor(
                        target_len, dtype=freqs_dtype, device=axis_pos.device
                    )
                    cos, sin = get_1d_rotary_pos_embed(
                        **common_kwargs,
                        yarn=True,
                        max_pe_len=max_pe_len,
                        ori_max_pe_len=base_len,
                        **dype_kwargs,
                    )
                elif self.method == "ntk" and target_len > base_len:
                    ntk_factor = max(target_len / max(base_len, 1), 1.0)
                    cos, sin = get_1d_rotary_pos_embed(
                        **common_kwargs,
                        ntk_factor=ntk_factor,
                        **dype_kwargs,
                    )
                else:
                    cos, sin = get_1d_rotary_pos_embed(
                        **common_kwargs,
                        **dype_kwargs,
                    )

            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)


def apply_dype_to_qwen_image(
    model: ModelPatcher,
    width: int,
    height: int,
    method: str,
    enable_dype: bool,
    dype_exponent: float,
    base_width: int,
    base_height: int,
    base_shift: float,
    max_shift: float,
) -> ModelPatcher:
    cloned = model.clone()

    diffusion_model = getattr(cloned.model, "diffusion_model", None)
    if diffusion_model is None:
        raise ValueError("The provided model does not expose a diffusion_model.")
    if not hasattr(diffusion_model, "pe_embedder"):
        raise ValueError("The provided model is missing a positional embedder.")
    if not hasattr(diffusion_model.pe_embedder, "theta"):
        raise ValueError("Unsupported positional embedder: missing theta attribute.")
    if not hasattr(diffusion_model.pe_embedder, "axes_dim"):
        raise ValueError("Unsupported positional embedder: missing axes_dim attribute.")

    patch_size = getattr(diffusion_model, "patch_size", None)
    if patch_size is None:
        raise ValueError("Unsupported diffusion model: missing patch_size attribute.")

    method_normalized = _normalize_method(method)

    backing_embedder = diffusion_model.pe_embedder
    new_embedder = QwenSpatialPosEmbed(
        theta=backing_embedder.theta,
        axes_dim=backing_embedder.axes_dim,
        patch_size=patch_size,
        method=method_normalized,
        enable_dype=enable_dype,
        dype_exponent=dype_exponent,
        base_resolution=(base_width, base_height),
        target_resolution=(width, height),
        backing_embedder=backing_embedder,
    )
    cloned.add_object_patch("diffusion_model.pe_embedder", new_embedder)

    model_sampler = getattr(cloned.model, "model_sampling", None)
    sigma_max = None
    if model_sampler is not None and hasattr(model_sampler, "sigma_max"):
        sigma_max = float(model_sampler.sigma_max.item())

    base_axes = new_embedder.grid.base_axes
    max_axes = new_embedder.grid.max_axes
    base_seq_len = max(base_axes[0] * base_axes[1], 1)
    max_seq_len = max(max_axes[0] * max_axes[1], base_seq_len + 1)

    current_axes = (
        _compute_axis_len(height, patch_size),
        _compute_axis_len(width, patch_size),
    )
    current_seq_len = max(current_axes[0] * current_axes[1], base_seq_len)

    if max_seq_len <= base_seq_len:
        slope = 0.0
    else:
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    dype_shift = current_seq_len * slope + intercept

    if (
        enable_dype
        and model_sampler is not None
        and isinstance(model_sampler, model_sampling.ModelSamplingFlux)
        and not getattr(model_sampler, "_dype_patched", False)
    ):
        def patched_sigma_func(self, timestep):
            return model_sampling.flux_time_shift(dype_shift, 1.0, timestep)

        model_sampler.sigma = types.MethodType(patched_sigma_func, model_sampler)
        model_sampler._dype_patched = True  # type: ignore[attr-defined]

    def dype_wrapper_function(model_function, args_dict):
        if enable_dype:
            timestep_tensor = args_dict.get("timestep")
            if (
                timestep_tensor is not None
                and sigma_max
                and sigma_max > 0
                and torch.is_tensor(timestep_tensor)
                and timestep_tensor.numel() > 0
            ):
                current_sigma = float(timestep_tensor.reshape(-1)[0].item())
                normalized = max(min(current_sigma / sigma_max, 1.0), 0.0)
                new_embedder.set_timestep(normalized)

        input_x = args_dict.get("input")
        timestep = args_dict.get("timestep")
        conditioning = args_dict.get("c", {})
        return model_function(input_x, timestep, **conditioning)

    cloned.set_model_unet_function_wrapper(dype_wrapper_function)

    grid = new_embedder.grid
    logger.info(
        "DyPE_QwenImage: patching positional embedder (method=%s, enable_dype=%s, "
        "dype_exponent=%s, base_axes=%s, target_axes=%s, base_shift=%s, max_shift=%s).",
        method_normalized,
        enable_dype,
        dype_exponent,
        grid.base_axes,
        grid.max_axes,
        base_shift,
        max_shift,
    )

    return cloned


__all__ = [
    "QwenSpatialPosEmbed",
    "apply_dype_to_qwen_image",
]
