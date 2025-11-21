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


def _compute_axis_len(resolution: int, patch_size: int, vae_scale_factor: int) -> int:
    latent = max(resolution // max(vae_scale_factor, 1), 1)
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
    vae_scale_factor: int

    @property
    def base_ctx_len(self) -> int:
        return self.base_axes[0] * self.base_axes[1]

    @property
    def max_ctx_len(self) -> int:
        return self.max_axes[0] * self.max_axes[1]

    @property
    def base_resolution(self) -> Tuple[int, int]:
        h = self.base_axes[0] * self.vae_scale_factor * 2
        w = self.base_axes[1] * self.vae_scale_factor * 2
        return (w, h)


@dataclass(frozen=True)
class _ModelGeometry:
    base_resolution: Tuple[int, int]
    vae_scale_factor: int
    patch_size: int


_EDITING_MODES = {"adaptive", "timestep_aware", "resolution_aware", "minimal", "full"}


def _select_freq_dtype(backing_embedder: nn.Module, device: torch.device) -> torch.dtype:
    default_dtype = getattr(backing_embedder, "freqs_dtype", torch.bfloat16)
    if not isinstance(default_dtype, torch.dtype):
        default_dtype = torch.bfloat16
    device_type = device.type
    if device_type in {"cpu", "mps", "npu"}:
        return torch.float32
    return default_dtype


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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
        vae_scale_factor: int,
        method: str,
        enable_dype: bool,
        dype_exponent: float,
        base_resolution: Tuple[int, int],
        target_resolution: Tuple[int, int],
        backing_embedder: nn.Module,
        editing_strength: float,
        editing_mode: str,
    ):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self.patch_size = int(patch_size)
        self.vae_scale_factor = max(int(vae_scale_factor), 1)
        self.method = _normalize_method(method)
        self.enable_dype = bool(enable_dype)
        self.dype_exponent = float(dype_exponent)
        self.backing_embedder = backing_embedder
        self.editing_strength = _clamp(float(editing_strength), 0.0, 1.0)
        mode_normalized = editing_mode.lower()
        if mode_normalized not in _EDITING_MODES:
            raise ValueError(
                f"Unsupported editing_mode '{editing_mode}'. "
                f"Expected one of {sorted(_EDITING_MODES)}."
            )
        self.editing_mode = mode_normalized

        self.current_timestep: float = 1.0
        self.current_editing: bool = False

        base_axes = (
            _compute_axis_len(base_resolution[1], self.patch_size, self.vae_scale_factor),
            _compute_axis_len(base_resolution[0], self.patch_size, self.vae_scale_factor),
        )
        max_axes = (
            _compute_axis_len(target_resolution[1], self.patch_size, self.vae_scale_factor),
            _compute_axis_len(target_resolution[0], self.patch_size, self.vae_scale_factor),
        )
        object.__setattr__(  # dataclass immutability helper
            self,
            "_grid",
            _GridConfig(
                base_axes=base_axes,
                max_axes=max_axes,
                vae_scale_factor=self.vae_scale_factor,
            ),
        )

    @property
    def grid(self) -> _GridConfig:
        return getattr(self, "_grid")

    def set_timestep(self, timestep: float, *, is_editing: bool = False) -> None:
        self.current_timestep = float(timestep)
        self.current_editing = bool(is_editing)

    def _resolve_editing_strength(self, *, axis_len: int, base_len: int) -> tuple[float, float]:
        if not self.current_editing or self.editing_mode == "full":
            return 1.0, self.dype_exponent
        if self.editing_strength <= 0.0:
            return 1.0, self.dype_exponent

        normalized_t = _clamp(self.current_timestep, 0.0, 1.0)
        effective_strength = 1.0

        if self.editing_mode == "adaptive":
            timestep_factor = 0.3 + (normalized_t * 0.7)
            effective_strength = self.editing_strength * timestep_factor
        elif self.editing_mode == "timestep_aware":
            timestep_factor = 0.2 + (normalized_t * 0.8)
            effective_strength = self.editing_strength * timestep_factor
        elif self.editing_mode == "resolution_aware":
            effective_strength = (
                self.editing_strength if axis_len > base_len else 1.0
            )
        elif self.editing_mode == "minimal":
            effective_strength = self.editing_strength
        else:  # fallback safety
            effective_strength = 1.0

        effective_strength = _clamp(effective_strength, 0.0, 1.0)
        if effective_strength >= 1.0:
            return 1.0, self.dype_exponent

        exponent = self.dype_exponent * effective_strength
        return effective_strength, exponent

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
        freqs_dtype = _select_freq_dtype(self.backing_embedder, ids.device)

        for axis_idx, axis_dim in enumerate(self.axes_dim):
            strength_multiplier = 1.0
            exponent = self.dype_exponent
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
                strength_multiplier, exponent = self._resolve_editing_strength(
                    axis_len=current_h if axis_idx == 1 else current_w,
                    base_len=grid.base_axes[axis_idx - 1],
                )
                dype_kwargs = {
                    "dype": True,
                    "current_timestep": float(self.current_timestep),
                    "dype_exponent": float(exponent),
                }
                ramp_factor = float(self.current_timestep) ** float(exponent)
            else:
                strength_multiplier = 1.0
                ramp_factor = 0.0

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

            if (
                self.current_editing
                and axis_idx > 0
                and strength_multiplier < 1.0
                and target_len > base_len
            ):
                reduced = base_len + int(
                    round((target_len - base_len) * strength_multiplier)
                )
                target_len = max(reduced, base_len)

            mode = "base"
            log_suffix = ""
            if axis_idx == 0 or self.method == "base":
                cos, sin = get_1d_rotary_pos_embed(**common_kwargs)
                mode = "static"
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
                    ratio = target_len / max(base_len, 1)
                    mode = "yarn"
                    log_suffix = f" ratio={ratio:.4f}"
                elif self.method == "ntk" and target_len > base_len:
                    ntk_factor = max(target_len / max(base_len, 1), 1.0)
                    cos, sin = get_1d_rotary_pos_embed(
                        **common_kwargs,
                        ntk_factor=ntk_factor,
                        **dype_kwargs,
                    )
                    mode = "ntk"
                    log_suffix = f" ntk_factor={ntk_factor:.4f}"
                else:
                    cos, sin = get_1d_rotary_pos_embed(
                        **common_kwargs,
                        **dype_kwargs,
                    )
                    mode = "static"

            logger.info(
                "axis=%s base_len=%d current_len=%d target_len=%d method=%s "
                "enable_dype=%s editing=%s editing_strength=%.3f timestep=%.4f "
                "ramp_factor=%.4f mode=%s%s",
                axis_name,
                base_len,
                current_len,
                target_len,
                self.method,
                self.enable_dype,
                self.current_editing,
                strength_multiplier,
                self.current_timestep,
                ramp_factor,
                mode,
                log_suffix,
            )

            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)


def _to_int_tuple(value: Tuple[int, int] | Iterable[int] | int) -> Tuple[int, int]:
    if isinstance(value, int):
        return (int(value), int(value))
    try:
        seq = list(value)  # type: ignore[arg-type]
    except TypeError:
        return (int(value), int(value))  # type: ignore[arg-type]
    if not seq:
        return (0, 0)
    if len(seq) == 1:
        return (int(seq[0]), int(seq[0]))
    return (int(seq[0]), int(seq[1]))


def _detect_model_geometry(diffusion_model: nn.Module) -> _ModelGeometry | None:
    patch_size = getattr(diffusion_model, "patch_size", None)
    vae_scale = getattr(diffusion_model, "vae_scale_factor", None)
    base_resolution: Tuple[int, int] | None = None

    def _extract_from_config(config: object) -> None:
        nonlocal patch_size, vae_scale, base_resolution
        if config is None:
            return
        if patch_size is None and hasattr(config, "patch_size"):
            try:
                patch_size = int(getattr(config, "patch_size"))
            except Exception:
                pass
        if vae_scale is None and hasattr(config, "vae_scale_factor"):
            try:
                vae_scale = int(getattr(config, "vae_scale_factor"))
            except Exception:
                pass
        sample_size = getattr(config, "sample_size", None)
        if sample_size is not None:
            try:
                sample_tuple = _to_int_tuple(sample_size)
                base_resolution = (
                    int(sample_tuple[0]) * max(vae_scale or 8, 1),
                    int(sample_tuple[1]) * max(vae_scale or 8, 1),
                )
            except Exception:
                pass
        image_size = getattr(config, "image_size", None)
        if image_size is not None:
            try:
                base_resolution = _to_int_tuple(image_size)
            except Exception:
                pass
        base_res_attr = getattr(config, "base_resolution", None)
        if base_res_attr is not None and base_resolution is None:
            try:
                base_resolution = _to_int_tuple(base_res_attr)
            except Exception:
                pass

    config_candidates = [
        getattr(diffusion_model, "config", None),
        getattr(getattr(diffusion_model, "transformer", None), "config", None),
        getattr(getattr(diffusion_model, "model", None), "config", None),
    ]
    for cfg in config_candidates:
        _extract_from_config(cfg)

    if base_resolution is None:
        sample_size = getattr(diffusion_model, "sample_size", None)
        if sample_size is not None:
            try:
                latent = _to_int_tuple(sample_size)
                base_resolution = (
                    int(latent[0]) * max(vae_scale or 8, 1),
                    int(latent[1]) * max(vae_scale or 8, 1),
                )
            except Exception:
                base_resolution = None

    if patch_size is None:
        embedder = getattr(diffusion_model, "pe_embedder", None)
        patch_size = getattr(embedder, "patch_size", None)

    if patch_size is None or base_resolution is None:
        return None

    if vae_scale is None:
        vae_scale = 8

    try:
        return _ModelGeometry(
            base_resolution=(int(base_resolution[0]), int(base_resolution[1])),
            vae_scale_factor=max(int(vae_scale), 1),
            patch_size=max(int(patch_size), 1),
        )
    except Exception:
        return None


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
    auto_detect: bool,
    editing_strength: float,
    editing_mode: str,
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

    method_normalized = _normalize_method(method)
    backing_embedder = diffusion_model.pe_embedder

    geometry = None
    if auto_detect:
        geometry = _detect_model_geometry(diffusion_model)
        if geometry is not None:
            logger.info(
                "DyPE_QwenImage: detected geometry base_resolution=%s patch_size=%d "
                "vae_scale_factor=%d",
                geometry.base_resolution,
                geometry.patch_size,
                geometry.vae_scale_factor,
            )
        else:
            logger.warning(
                "DyPE_QwenImage: failed to detect Qwen geometry; falling back to "
                "provided base dimensions (%dx%d).",
                base_width,
                base_height,
            )

    patch_size = getattr(diffusion_model, "patch_size", None)
    if geometry is not None:
        patch_size = geometry.patch_size
        base_width, base_height = geometry.base_resolution
        vae_scale_factor = geometry.vae_scale_factor
    else:
        if patch_size is None:
            raise ValueError("Unsupported diffusion model: missing patch_size attribute.")
        vae_scale_factor = getattr(diffusion_model, "vae_scale_factor", 8)
        try:
            vae_scale_factor = max(int(vae_scale_factor), 1)
        except Exception:
            vae_scale_factor = 8

    new_embedder = QwenSpatialPosEmbed(
        theta=backing_embedder.theta,
        axes_dim=backing_embedder.axes_dim,
        patch_size=patch_size,
        vae_scale_factor=vae_scale_factor,
        method=method_normalized,
        enable_dype=enable_dype,
        dype_exponent=dype_exponent,
        base_resolution=(base_width, base_height),
        target_resolution=(width, height),
        backing_embedder=backing_embedder,
        editing_strength=editing_strength,
        editing_mode=editing_mode,
    )
    cloned.add_object_patch("diffusion_model.pe_embedder", new_embedder)

    model_sampler = getattr(cloned.model, "model_sampling", None)
    sigma_max = None
    if model_sampler is not None and hasattr(model_sampler, "sigma_max"):
        sigma_max_value = getattr(model_sampler, "sigma_max")
        try:
            if hasattr(sigma_max_value, "item"):
                sigma_max = float(sigma_max_value.item())
            else:
                sigma_max = float(sigma_max_value)
        except (TypeError, ValueError):
            sigma_max = None
        if sigma_max is not None and sigma_max <= 0:
            sigma_max = None

    base_axes = new_embedder.grid.base_axes
    max_axes = new_embedder.grid.max_axes
    base_seq_len = max(base_axes[0] * base_axes[1], 1)
    max_seq_len = max(max_axes[0] * max_axes[1], base_seq_len + 1)

    current_axes = (
        _compute_axis_len(height, patch_size, vae_scale_factor),
        _compute_axis_len(width, patch_size, vae_scale_factor),
    )
    current_seq_len = max(current_axes[0] * current_axes[1], base_seq_len)

    if max_seq_len <= base_seq_len:
        slope = 0.0
    else:
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    intercept = base_shift - slope * base_seq_len
    dype_shift = current_seq_len * slope + intercept

    sampler_patched = False
    if enable_dype and model_sampler is not None and not getattr(model_sampler, "_dype_patched", False):
        if isinstance(model_sampler, model_sampling.ModelSamplingFlux):
            def patched_sigma_func(self, timestep):
                return model_sampling.flux_time_shift(dype_shift, 1.0, timestep)

            model_sampler.sigma = types.MethodType(patched_sigma_func, model_sampler)
            sampler_patched = True
        elif hasattr(model_sampler, "sigma") and callable(getattr(model_sampler, "sigma")):
            original_sigma = model_sampler.sigma

            def patched_sigma_func(self, timestep):
                baseline = original_sigma(timestep)
                scale = 1.0 + max(dype_shift, 0.0) * 0.1
                try:
                    return baseline * scale
                except TypeError:
                    baseline_value = float(baseline)
                    return baseline_value * scale

            model_sampler.sigma = types.MethodType(patched_sigma_func, model_sampler)
            sampler_patched = True

        if sampler_patched:
            setattr(model_sampler, "_dype_patched", True)
            logger.info(
                "DyPE_QwenImage: installed sampler shift (fallback=%s, dype_shift=%.4f).",
                not isinstance(model_sampler, model_sampling.ModelSamplingFlux),
                dype_shift,
            )

    def dype_wrapper_function(model_function, args_dict):
        if enable_dype:
            timestep_tensor = args_dict.get("timestep")
            c = args_dict.get("c", {})
            input_x = args_dict.get("input")
            adjust_enabled = editing_mode != "full" and editing_strength < 1.0
            is_editing = False
            if adjust_enabled and isinstance(c, dict):
                editing_keys = {
                    "image",
                    "image_embeds",
                    "image_tokens",
                    "concat_latent_image",
                    "concat_mask",
                    "concat_mask_image",
                }
                for key in editing_keys:
                    if key in c and c[key] is not None:
                        is_editing = True
                        break
            if adjust_enabled and not is_editing and input_x is not None and hasattr(input_x, "abs"):
                try:
                    variance = float(input_x.abs().mean().item())
                    if variance < 0.25:
                        is_editing = True
                except Exception:
                    pass
            current_sigma: float | None = None
            if timestep_tensor is not None and sigma_max and sigma_max > 0:
                if torch.is_tensor(timestep_tensor):
                    if timestep_tensor.numel() > 0:
                        current_sigma = float(timestep_tensor.reshape(-1)[0].item())
                elif isinstance(timestep_tensor, (int, float)):
                    current_sigma = float(timestep_tensor)
            if current_sigma is not None:
                normalized = max(min(current_sigma / sigma_max, 1.0), 0.0)
                new_embedder.set_timestep(normalized, is_editing=is_editing)

        input_x = args_dict.get("input")
        timestep = args_dict.get("timestep")
        conditioning = args_dict.get("c", {})
        return model_function(input_x, timestep, **conditioning)

    cloned.set_model_unet_function_wrapper(dype_wrapper_function)

    grid = new_embedder.grid
    logger.info(
        "DyPE_QwenImage: patching positional embedder (method=%s, enable_dype=%s, "
        "dype_exponent=%s, base_axes=%s, target_axes=%s, base_shift=%s, max_shift=%s, "
        "editing_mode=%s, editing_strength=%.3f).",
        method_normalized,
        enable_dype,
        dype_exponent,
        grid.base_axes,
        grid.max_axes,
        base_shift,
        max_shift,
        editing_mode,
        editing_strength,
    )

    return cloned


__all__ = [
    "QwenSpatialPosEmbed",
    "apply_dype_to_qwen_image",
]
