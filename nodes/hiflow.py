"""
HiFlow ComfyUI node — trajectory-guided flow upscaling.

Runs the HiFlow cascade (plan 2026-09-03): a base-resolution rectified-flow
trajectory is recorded, then each upscale stage reuses it as a time-matched
virtual reference (initialization + direction + acceleration alignment).
Flow models only (FLUX, Qwen-Image, AuraFlow, ...); works in VAE latent
space with the x0 adapter owning the VAE<->model conversions.

Reference: HiFlow paper (arXiv:2504.06232, NeurIPS 2025).
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
from comfy_api.latest import io

try:
    from ..src.freescale import gaussian_blur_2d
    from ..src.hiflow import HiFlowConfig, hiflow_cascade
except ImportError:  # flat repo layout (tests / CLI)
    from src.freescale import gaussian_blur_2d
    from src.hiflow import HiFlowConfig, hiflow_cascade

from .pixelrush import _detect_prediction_type

logger = logging.getLogger("ComfyUI-DyPE")


# Timestep/sigma conversions go through model_sampling.timestep/.sigma only —
# never hand-multiplied. FLUX (ModelSamplingFlux) uses timestep(sigma)==sigma;
# DiscreteFlow models use a x1000 multiplier; the adapter is agnostic either way.

_FLOW_PREDICTIONS = ("const", "img_to_img_flow", "cosmos_rflow")


def _require_flow_model(model) -> tuple[str, int]:
    """Gate HiFlow to rectified-flow models (plan D1, D12; Krea2 plan S2).

    Raises ValueError with an actionable message for non-flow prediction
    types. 3D-FORMAT latent models (Wan21: Krea2, Qwen-Image, Anima) are
    ACCEPTED — they are image models with a 5D [B,C,1,H,W] latent layout;
    the node layer bridges to the 4D core (the PixelRush convention).
    Only actual multi-frame (T>1) latents are rejected — the paper's
    frequency alignment is 2D per-frame.

    Returns (detected flow family, latent_dimensions).
    """
    model_sampling = model.model.model_sampling
    mro_names = [c.__name__ for c in type(model_sampling).__mro__]

    detected = _detect_prediction_type(model_sampling)
    if detected == "const" and "IMG_TO_IMG_FLOW" in mro_names:
        detected = "img_to_img_flow"
    elif detected == "const" and "COSMOS_RFLOW" in mro_names:
        detected = "cosmos_rflow"

    if detected not in _FLOW_PREDICTIONS:
        raise ValueError(
            f"HiFlow needs a rectified-flow model (FLUX, Qwen-Image, "
            f"Krea2, AuraFlow, Z-Image...); this model predicts "
            f"'{detected.upper()}'. For SD/SDXL-style models use the "
            f"PixelRush node instead."
        )

    latent_dimensions = getattr(
        model.model.latent_format, "latent_dimensions", 2)
    if latent_dimensions not in (2, 3):
        raise ValueError(
            f"HiFlow supports 2D or 3D-format image latents; this model "
            f"reports latent_dimensions={latent_dimensions}."
        )

    return detected, int(latent_dimensions)


def _make_predict_x0(
    model,
    positive,
    negative,
    cfg_scale: float,
    latent_dimensions: int = 2,
) -> Callable[[torch.Tensor, float], torch.Tensor]:
    """Create the x0 adapter: (x_vae, sigma) -> x0 in VAE space (plan D3).

    Conditioning is prepared once per latent SHAPE via ComfyUI's canonical
    pipeline (convert_cond -> process_conds — the HAP-calibration precedent).
    Each call runs comfy.samplers.sampling_function, which returns the
    DENOISED x0 (apply_model applies calculate_denoised) with full CFG,
    areas, control nets and hooks. The VAE<->model conversions bracket the
    model call and cancel per call (plan D2).

    ``latent_dimensions == 3`` (Wan21: Krea2, Qwen-Image — Krea2 plan S3):
    the adapter unsqueezes the 4D core tensor to 5D [B,C,1,H,W] BEFORE
    process_latent_in (the Wan21 mean/std stats are [1,C,1,1,1] views —
    they broadcast correctly on 5D only) and the model call, and squeezes
    the result back — the PixelRush predict_eps ``was_4d`` pattern.
    """
    import comfy.model_management
    import comfy.sampler_helpers
    import comfy.samplers

    device = model.load_device if hasattr(model, "load_device") \
        else torch.device("cpu")
    inner_model = model.model
    process_latent_in = getattr(inner_model, "process_latent_in", None)
    process_latent_out = getattr(inner_model, "process_latent_out", None)

    comfy.model_management.load_models_gpu([model])
    model.pre_run()

    _conds_by_shape: dict[tuple, dict] = {}

    def _get_conds(shape: tuple) -> dict:
        key = tuple(shape)
        if key not in _conds_by_shape:
            conds = {
                "positive": comfy.sampler_helpers.convert_cond(positive),
                "negative": comfy.sampler_helpers.convert_cond(negative),
            }
            noise = torch.zeros(shape, device=device)
            _conds_by_shape[key] = comfy.samplers.process_conds(
                inner_model, noise, conds, device,
            )
        return _conds_by_shape[key]

    # CFG guard (Z-Image bugfix 2026-09-03): with a NEGATIVE that carries no
    # tokens (an empty CLIPTextEncode — NOT ConditioningZeroOut), CFG is
    # undefined: the "uncond" branch is a real encoding of the empty string,
    # and cond_scale amplifies a meaningless difference. Guidance-free
    # models (Z-Image, Chroma) always land here when the user leaves the
    # negative empty. Mirror ComfyUI's cfg=1 skip: run the conditional
    # branch only (sampling_function with cond_scale=1.0 sets uncond=None
    # and returns the conditional x0 exactly).
    _has_negative = False
    for _entry in negative or []:
        if isinstance(_entry, (tuple, list)) and len(_entry) == 2:
            _tensor, _opts = _entry
            _tokens = 0
            if torch.is_tensor(_tensor):
                _tokens = int(_tensor.numel())
            elif isinstance(_tensor, (list, tuple)):
                _tokens = sum(
                    int(t.numel()) if torch.is_tensor(t) else len(t)
                    for t in _tensor
                )
            if _tokens > 0:
                _has_negative = True
                break
    if not _has_negative:
        cfg_scale = 1.0
        logger.info(
            "HiFlow: negative conditioning carries no tokens — running "
            "the conditional branch only (CFG skipped, scale forced to 1.0)"
        )

    def predict_x0(x_vae: torch.Tensor, sigma: float) -> torch.Tensor:
        x = x_vae.to(device)
        # 3D-format models (Wan21) need the 5D tensor — the latent-format
        # mean/std stats are [1,C,1,1,1] views and the model was trained
        # on 5D (Krea2 plan S3). Track the bridging to undo it after.
        was_4d = x.dim() == 4
        if was_4d and latent_dimensions == 3:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
        if process_latent_in is not None:
            x = process_latent_in(x)

        conds = _get_conds(tuple(x_vae.shape))
        sigma_t = torch.tensor([float(sigma)], device=device)
        timestep = inner_model.model_sampling.timestep(sigma_t)

        x0 = comfy.samplers.sampling_function(
            inner_model, x, timestep,
            uncond=conds["negative"], cond=conds["positive"],
            cond_scale=cfg_scale,
            model_options=model.model_options,
        )
        if process_latent_out is not None:
            x0 = process_latent_out(x0)
        if was_4d and x0.dim() == 5:
            x0 = x0.squeeze(2)
        return x0.to(x_vae.dtype).to(x_vae.device)

    return predict_x0


# ---------------------------------------------------------------------------
# VAE adapters (image latents; the gate rejects multi-frame T>1 input)
# ---------------------------------------------------------------------------

def _make_vae_adapters(vae, device):
    """Create (vae_decode, vae_encode) callables in VAE latent space.

    HiFlow runs entirely in VAE latent space (plan D2), so these adapters do
    NOT apply process_latent_out/in — that would double-convert an already
    VAE-space tensor (the PixelRush space contract). They only normalize the
    tensor layout around the raw vae.decode/encode calls.

    LAYOUT (Z-Image bugfix 2026-09-03): ComfyUI's VAE boundary is channels-
    LAST — ``VAE.decode`` returns [B, H, W, 3] and ``VAE.encode`` expects
    [B, H, W, 3] (it applies ``movedim(-1, 1)`` internally, sd.py:1359).
    The adapters speak that layout end-to-end: decode passes the decoded
    image through, bicubic/sharpen run on channels-last tensors, and encode
    hands channels-last straight back. Converting to channels-first here
    (the old PixelRush-style convention) made VAE.encode move the WIDTH
    axis into the channel slot — spatial dims corrupted, encoder conv
    crashed ("Kernel size can't be greater than actual input size").

    3D-FORMAT VAEs (Krea2 plan S4, Qwen-VAE ``latent_dim=3``): decode gets
    the 5D [B,C,1,h,w] latent it expects (unsqueeze a 4D input) and returns
    a 5D image [B,T,H,W,3] — take the first temporal frame. encode takes
    the channels-last 4D image and unsqueezes to 5D ITSELF (``not_video``
    branch, sd.py:1342-1346 — the adapters must not assume the video-VAE
    batch trick) and returns a 5D latent [B,C,T,h,w] — take [:, :, 0] for
    the 4D core.
    """

    latent_dim = getattr(vae, "latent_dim", 2)

    def vae_decode(latent: torch.Tensor) -> torch.Tensor:
        """latent [B,C,h,w] (or 5D [B,C,1,h,w]) -> image [B, H, W, 3]."""
        if isinstance(latent, dict):
            latent = latent["samples"]
        latent = latent.to(device)
        if latent_dim == 3 and latent.dim() == 4:
            latent = latent.unsqueeze(2)  # [B, C, 1, h, w]
        decoded = vae.decode(latent)
        if isinstance(decoded, dict):
            decoded = decoded["samples"]
        if decoded.ndim == 5:
            decoded = decoded[:, 0]     # first temporal frame [B, H, W, 3]
        elif decoded.ndim == 3:
            decoded = decoded.unsqueeze(0)
        return decoded  # [B, H, W, 3] channels-last, untouched

    def vae_encode(image: torch.Tensor) -> torch.Tensor:
        """image [B, H, W, 3] -> latent [B, C, h, w] (4D for the core)."""
        image = image.to(device)
        encoded = vae.encode(image)
        if isinstance(encoded, dict):
            encoded = encoded["samples"]
        if latent_dim == 3 and encoded.ndim == 5:
            encoded = encoded[:, :, 0]  # first temporal frame -> 4D
        return encoded

    return vae_decode, vae_encode


def _sharpen(image: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Gaussian unsharp mask: (alpha + 1) * I - alpha * blur(I).

    The reference (utils.gaussian_blur_image_sharpening) sharpens the
    pixel-space upscaled image before re-encoding (pixel mode only). The
    image arrives channels-LAST ([B, H, W, 3] — the ComfyUI VAE boundary);
    gaussian_blur_2d needs channels-first, so convert around the blur. A 5D
    decode output ([B,T,H,W,3], 3D-format VAEs) is sliced to its first
    frame first — the adapters normally hand 4D, this is the defensive
    backstop (Krea2 plan S4).
    """
    if image.dim() == 5:
        image = image[:, 0]
    channels_last = image.dim() == 4 and image.shape[-1] == 3
    if channels_last:
        image = image.movedim(-1, 1)
    blurred = gaussian_blur_2d(image, kernel_size=3, sigma=2.0)
    sharpened = (alpha + 1.0) * image - alpha * blurred
    if channels_last:
        sharpened = sharpened.movedim(1, -1)
    return sharpened


def _base_sigmas(model, steps: int) -> torch.Tensor:
    """The model's own descending sigma schedule for the base stage.

    Uses comfy.samplers.calculate_sigmas with the "simple" scheduler (index
    sampling of the model's sigmas — no spacing resampling), ending at 0.
    """
    import comfy.samplers

    ms = model.model.model_sampling
    sigmas = comfy.samplers.calculate_sigmas(ms, "simple", steps)
    return sigmas.float().cpu()


def _downscale_ratio(vae) -> int:
    ratio = getattr(vae, "downscale_ratio", 8)
    if isinstance(ratio, (tuple, list)):
        ratio = ratio[1]  # (callable, h_ratio, w_ratio) convention
    return int(ratio)


# ---------------------------------------------------------------------------
# HiFlow node (V3 schema)
# ---------------------------------------------------------------------------

class HiFlowNode(io.ComfyNode):
    """HiFlow — trajectory-guided high-resolution upscaling for flow models.

    Records the base-resolution rectified-flow trajectory, then guides each
    upscale stage with it as a time-matched virtual reference: initialization
    alignment (start at noise level tau from the reference prediction),
    direction alignment (low-frequency nudge toward the reference) and
    acceleration alignment (match the reference's velocity rhythm).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HiFlow",
            display_name="HiFlow",
            category="WMNodes/image",
            description=(
                "Training-free high-resolution upscaling for rectified-flow "
                "models (FLUX, Qwen-Image, ...) via flow-aligned guidance. "
                "Works from a base latent; chain DyPE (ntk) before the "
                "loader for RoPE extrapolation at the scaled resolution."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The flow model."),
                io.Vae.Input("vae", tooltip="VAE (used by pixel upsampling)."),
                io.Conditioning.Input(
                    "positive", tooltip="Positive conditioning."),
                io.Conditioning.Input(
                    "negative", tooltip="Negative conditioning."),
                io.Latent.Input(
                    "latent_image",
                    tooltip="Base latent at the model's native resolution "
                            "(e.g. from EmptySD3LatentImage). The cascade "
                            "noises it to the first sigma itself — an empty "
                            "latent + noise_seed reproduces the reference "
                            "pipeline's from-noise start."),
                io.Int.Input(
                    "noise_seed", default=0, min=0, max=2**32 - 1, step=1,
                    tooltip="Seed for the base-stage noise and each stage's "
                            "initialization noise (one shared generator)."),
                io.Float.Input(
                    "denoise", default=1.0, min=0.05, max=1.0, step=0.05,
                    tooltip="Img2img strength for a CONTENT latent (KSampler "
                            "convention): 1.0 regenerates from pure noise; "
                            "lower keeps more of the input image (0.6 "
                            "enters at ~37% content). Ignored for an empty "
                            "latent — that always runs the full schedule."),
                io.Float.Input(
                    "cfg", default=3.5, min=0.0, max=20.0, step=0.1,
                    tooltip="Classifier-free guidance for the BASE stage "
                            "(FLUX-dev default 3.5). Guidance-free models "
                            "(Z-Image, Chroma) or empty negatives: leave at "
                            "1.0 — CFG is auto-skipped when the negative "
                            "carries no tokens."),
                io.Int.Input(
                    "steps", default=30, min=1, max=200, step=1,
                    tooltip="Base-stage sampling steps (paper: 30). The "
                            "per-step clean predictions form the reference "
                            "trajectory."),
                io.Float.Input(
                    "guidance", default=4.5, min=0.0, max=20.0, step=0.1,
                    tooltip="Classifier-free guidance for the guided upscale "
                            "stages (paper uses 4.5-6). Same auto-skip rule "
                            "as cfg."),
                io.Int.Input(
                    "steps_per_stage", default=16, min=1, max=50, step=1,
                    tooltip="Guided-stage sampling steps per cascade stage "
                            "(upper bound; repo default 16/10)."),
                io.Float.Input(
                    "tau", default=0.6, min=0.05, max=0.95, step=0.05,
                    tooltip="Stage-entry noise level (paper cascade: 0.6, "
                            "0.3, 0.3). Lower = stronger content preservation."),
                io.Float.Input(
                    "filter_ratio", default=0.2, min=0.05, max=0.95,
                    step=0.05,
                    tooltip="Normalized Butterworth low-pass cutoff D "
                            "(direction alignment). Paper 0.4, repo default "
                            "0.2."),
                io.Float.Input(
                    "alpha_scale", default=1.0, min=0.0, max=2.0, step=0.05,
                    tooltip="Direction-alignment strength multiplier "
                            "(repo first-stage value 1.0)."),
                io.Float.Input(
                    "beta_scale", default=0.5, min=0.0, max=2.0, step=0.05,
                    tooltip="Acceleration-alignment strength multiplier "
                            "(repo default 0.5)."),
                io.Combo.Input(
                    "upsampling", options=["latent", "pixel"],
                    default="latent",
                    tooltip="Per-step reference upsample: latent bicubic "
                            "(repo default) or pixel decode->sharpen->encode. "
                            "The stage-initialization anchor is always the "
                            "pixel round-trip of the previous final image."),
                io.Float.Input(
                    "scale_factor", default=2.0, min=0.25, max=8.0,
                    step=0.05,
                    tooltip="Output scale relative to the input latent: 2 = "
                            "double resolution per side, 1 = unchanged, 0.5 "
                            "= half. Upscales run 2x doubling stages (so "
                            "scales between 1 and 2 give one 2x stage); "
                            "scales below 1 run a single refinement stage "
                            "at the smaller size."),
            ],
            outputs=[
                io.Latent.Output(display_name="High-Res Latent"),
            ],
        )

    @classmethod
    def validate_inputs(cls, scale_factor):
        # Uninitialized graph state passes through (2026-08-25 fix pattern).
        if scale_factor is None:
            return True
        if not (0.25 <= float(scale_factor) <= 8.0):
            return "scale_factor must be between 0.25 and 8"
        return True

    @classmethod
    def execute(cls, model, vae, positive, negative, latent_image,
                cfg=3.5, steps=30, guidance=4.5, steps_per_stage=16,
                tau=0.6, filter_ratio=0.2, alpha_scale=1.0, beta_scale=0.5,
                upsampling="latent", scale_factor=2.0,
                noise_seed=0, denoise=1.0) -> io.NodeOutput:
        import comfy.utils

        # Gate BEFORE any model calls: flow prediction; 3D-FORMAT image
        # models (Wan21: Krea2, Qwen-Image) pass, multi-frame latents
        # don't (Krea2 plan S2).
        _, latent_dimensions = _require_flow_model(model)

        if isinstance(latent_image, dict):
            initial_latent = latent_image["samples"]
        else:
            initial_latent = latent_image

        if initial_latent.ndim == 5:
            if initial_latent.shape[2] != 1:
                raise ValueError(
                    "HiFlow received a multi-frame (video) latent "
                    f"(T={initial_latent.shape[2]}). It supports single-"
                    "frame image latents only — the frequency alignment "
                    "is 2D per-frame."
                )
            initial_latent = initial_latent.squeeze(2)  # [B, C, H, W]
        elif initial_latent.ndim == 4 and latent_dimensions == 3:
            # A 4D latent on a 3D-format model (EmptySD3LatentImage etc.)
            # carries T=1 implicitly — remember the format for the output.
            pass

        device = model.load_device if hasattr(model, "load_device") \
            else torch.device("cpu")
        initial_latent = initial_latent.to(device)

        # Latent-format conversions for the cascade's img2img noising (the
        # model-space mix, v2.12.1). None for models without them -> the
        # cascade falls back to the plain VAE-space mix.
        # For 3D-format models (Wan21) the conversions are WRAPPED to be
        # NDIM-TRANSPARENT (v2.14.1): the cascade mixes 4D core tensors,
        # but Wan21's mean/std stats are [1,C,1,1,1] — calling the raw
        # conversion on 4D BROADCASTS SILENTLY to [B,C,C,H,W] garbage
        # (reads as T=channels; the real-model Krea2 crash "Expected size
        # 1 but got size 16"), and returning a 5D tensor from the wrapper
        # re-triggers the same broadcast in the cascade's sigma-mix (4D
        # noise + 5D content). The wrapper therefore unsqueezes, converts
        # in true 5D model space, and squeezes back — the cascade's mix
        # runs on 4D tensors with correctly-normalized VALUES (Wan21's
        # per-channel stats commute with the singleton-T squeeze).
        inner_model = model.model
        _raw_in = getattr(inner_model, "process_latent_in", None)
        _raw_out = getattr(inner_model, "process_latent_out", None)
        if latent_dimensions == 3:
            def inner_model_process_latent_in(t):
                was_4d = t.dim() == 4
                if was_4d:
                    t = t.unsqueeze(2)  # [B, C, 1, H, W]
                t = _raw_in(t) if _raw_in is not None else t
                if was_4d and t.dim() == 5:
                    t = t.squeeze(2)
                return t

            def inner_model_process_latent_out(t):
                was_4d = t.dim() == 4
                if was_4d:
                    t = t.unsqueeze(2)  # [B, C, 1, H, W]
                t = _raw_out(t) if _raw_out is not None else t
                if was_4d and t.dim() == 5:
                    t = t.squeeze(2)
                return t
        else:
            inner_model_process_latent_in = _raw_in
            inner_model_process_latent_out = _raw_out

        # Channel handling for empty latents (EmptyLatentImage may produce 4
        # channels for a 16-channel model) — the PixelRush convention.
        model_latent_channels = getattr(
            model.model.latent_format, "latent_channels", None)
        if model_latent_channels is not None and \
                initial_latent.shape[1] != model_latent_channels:
            is_empty = torch.count_nonzero(initial_latent) == 0
            if is_empty:
                logger.info(
                    "HiFlow: empty input latent has %d channels, model "
                    "expects %d — repeating channels",
                    initial_latent.shape[1], model_latent_channels,
                )
                initial_latent = comfy.utils.repeat_to_batch_size(
                    initial_latent, model_latent_channels, dim=1,
                )
            else:
                logger.warning(
                    "HiFlow: non-empty input latent has %d channels, model "
                    "expects %d — results may be unexpected",
                    initial_latent.shape[1], model_latent_channels,
                )

        cfg_obj = HiFlowConfig(
            tau=float(tau), steps=int(steps),
            steps_per_stage=int(steps_per_stage),
            cfg=float(cfg), guidance_high=float(guidance),
            filter_ratio=float(filter_ratio),
            alpha_scale=float(alpha_scale), beta_scale=float(beta_scale),
            upsampling=str(upsampling),
        )

        predict_x0_base = _make_predict_x0(
            model, positive, negative, cfg_scale=float(cfg),
            latent_dimensions=latent_dimensions)
        predict_x0_stage = _make_predict_x0(
            model, positive, negative, cfg_scale=float(guidance),
            latent_dimensions=latent_dimensions)

        # VAE adapters are ALWAYS needed — the stage-initialization anchor
        # is the pixel round-trip of the previous final latent in both
        # upsampling modes (plan D2).
        vae_decode, vae_encode = _make_vae_adapters(vae, device)

        base_sigmas = _base_sigmas(model, int(steps))

        # A content latent with denoise=1.0 is a foot-gun: sigma_start == 1
        # zeroes the content weight entirely (from-noise generation).
        if (torch.count_nonzero(initial_latent) > 0
                and float(denoise) > 0.9999):
            logger.warning(
                "HiFlow: denoise=1.0 with a non-empty latent — the input "
                "image is ignored (the base starts from pure noise). Lower "
                "denoise (e.g. 0.6) to upscale the connected latent."
            )

        # Progress: base transitions + per-stage transitions (upper bound).
        try:
            from ..src.hiflow import _stage_latent_sizes
        except ImportError:
            from src.hiflow import _stage_latent_sizes
        sizes = _stage_latent_sizes(
            initial_latent.shape[-2], initial_latent.shape[-1],
            float(scale_factor), _downscale_ratio(vae),
        )
        total = max(1, len(base_sigmas) - 1 + len(sizes) * int(steps_per_stage))
        pbar = comfy.utils.ProgressBar(total)
        counter = {"n": 0}

        def progress_callback(i, total_steps, stage):
            counter["n"] += 1
            pbar.update_absolute(min(counter["n"], total))

        result = hiflow_cascade(
            initial_latent=initial_latent,
            base_sigmas=base_sigmas,
            predict_x0_base=predict_x0_base,
            predict_x0_stage=predict_x0_stage,
            scale_factor=float(scale_factor),
            cfg=cfg_obj,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            sharpen=_sharpen,
            vae_downscale=_downscale_ratio(vae),
            progress_callback=progress_callback,
            noise_seed=int(noise_seed),
            denoise=float(denoise),
            # Noising runs in MODEL space (ComfyUI samplers.py:1223/993 —
            # process_latent_in on the content BEFORE the sigma mix), so the
            # cascade needs the latent-format conversions (Z-Image round-3
            # fix, v2.12.1).
            process_latent_in=inner_model_process_latent_in,
            process_latent_out=inner_model_process_latent_out,
        )
        pbar.update_absolute(total)

        # 3D-format models expect the 5D [B,C,1,H,W] LATENT on the output
        # (the PixelRush output convention — downstream VAEDecode works on
        # the 5D tensor; Krea2 plan S5).
        if latent_dimensions == 3 and result.dim() == 4:
            result = result.unsqueeze(2)
        return io.NodeOutput({"samples": result})
