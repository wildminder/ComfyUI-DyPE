"""AReA node — training-free two-phase attention replacement (schema + execute).

Node definitions live in ``nodes/``; engines in ``src/`` (pack layout plan
2026-09-08). This module is importable both inside the ComfyUI-loaded pack
package (relative ``..src`` imports) and flat from the repo root (tests).

AReA (arXiv 2610.17227) replaces FLUX / Z-Image / Krea-2 / Qwen-Image-2.1
image self-attention with a time-dependent attention graph: the first
``phase1_steps`` (or, in ``sigma`` mode, the leading steps still above
``phase1_sigma``) steps pool image K/V to one token per ``f x f`` block and run
cross-scale attention (Phase I), the remaining steps run windowed attention
whose windows follow the image's Haar-DWT detail map, each extended by a
randomized Halo ring and a global-KV bridge .  At or below the
native 1024 px extent it is a provable no-op.
"""

import inspect

from comfy_api.latest import io

try:  # loaded as pack package (ComfyUI loader)
    from ..src.area import PHASE1_STEPS, apply_area_to_model
    from ..src.area_zimage import ZIMAGE_SIGMA_STAR
    from ..src.area_attn import POOLING
    from ..src.validation import validate_resolution
except ImportError:  # flat repo layout (tests / CLI)
    from src.area import PHASE1_STEPS, apply_area_to_model
    from src.area_zimage import ZIMAGE_SIGMA_STAR
    from src.area_attn import POOLING
    from src.validation import validate_resolution


def _phase1_switch_kwargs(phase1_mode: str, phase1_sigma: float) -> dict:
    """The Phase I switch, forwarded only where the engine declares it.

    The node's schema is deliberately a SUPERSET of
    ``apply_area_to_model``'s signature: the engine takes ``phase1_mode`` /
    ``phase1_sigma`` from the model-side wiring, and forwarding the two names
    before that would raise a ``TypeError`` on every forward.  The pair is
    therefore filtered against the installed signature, so the knobs start
    flowing the moment the engine declares them (and so does nothing at all
    before it, which is what keeps the FLUX path byte-identical).

    Both default to ``"auto"``, which the engine resolves per backbone
    (``apply_area_to_model``): FLUX gets the reference's ``steps`` mode, where
    ``phase1_steps`` is the switch — so nothing changes for a FLUX graph that
    never touches the knob — and Z-Image gets ``sigma`` mode at
    ``phase1_sigma`` (the schedule-aware handoff).
    """
    accepted = inspect.signature(apply_area_to_model).parameters
    wanted = {"phase1_mode": str(phase1_mode), "phase1_sigma": float(phase1_sigma)}
    return {name: value for name, value in wanted.items() if name in accepted}


class AReA(io.ComfyNode):
    """
    Applies AReA (arXiv 2610.17227) to a model: a training-free, time-dependent
    replacement of FLUX / Z-Image / Krea-2 image self-attention.

    Phase I pools image keys/values to one token per f x f block and runs
    cross-scale attention on a coarse position grid while the queries stay at
    full resolution. Phase II runs windowed attention whose window sizes follow
    the image's Haar-DWT detail map (smooth tiles split into quadrants), with a
    randomized Halo ring and a global-KV bridge of distinctive tokens per
    window. Chaining with DyPE / I-Max positional scaling composes (AReA's
    coarse positions ride on the installed rope embedder); below 1024 px AReA is
    an automatically a no-op.

    ``model_type`` selects the backbone profile (``src.area_arch``): ``auto``
    reads it off the model, and the four supported backbones — FLUX, Z-Image
    (Lumina2 / NextDiT), Krea-2 (SingleStreamDiT) and Qwen-Image-2.1
    (QwenImage21Transformer2DModel) — share the same 16 px
    patch grid, so one width/height pair serves any of them.  ``phase1_mode``
    and ``pooling`` default to ``auto``, which the engine resolves per
    backbone: on Z-Image and Qwen-Image-2.1 the switch is the schedule-aware
    ``sigma`` mode (the leading steps still at or above ``phase1_sigma``),
    while FLUX and Krea-2 keep the reference's ``steps`` constant; pooling
    resolves to the reference's ``subsample`` on FLUX, the norm-preserving
    ``mean`` on Z-Image and Krea-2, and the ``medoid`` on Qwen-Image-2.1 (one
    real token's key AND value per block).  Three knobs tune short (distilled)
    schedules only and are ignored on a long one: ``bridge_floor_steps``
    exempts the leading fine steps from the cost cap (they keep the paper's
    full bridge and halo), ``window_start`` / ``window_hold_steps`` reshape the
    window ramp.  ``phase1_steps`` is authoritative below the short schedule's
    heal cap (``total_steps - 2``); the forward-time log prints the knob, the
    cap and the EFFECTIVE coarse count once per schedule.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AReA",
            display_name="AReA",
            category="WMNodes/image",
            description="AReA (arXiv 2610.17227). Replaces FLUX / Z-Image / Krea-2 / Qwen-Image-2.1 image self-attention with a time-dependent attention graph: coarse cross-scale attention on leading steps, DWT-guided windowed attention (Halo + bridge tokens) on the rest. Training-free; a no-op at or below 1024 px.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The model to patch with AReA. FLUX, Z-Image (Lumina2), Krea-2 and Qwen-Image-2.1 share the same 16 px patch grid, so the same width/height serve any of them.",
                ),
                io.Int.Input(
                    "width",
                    default=2048, min=16, max=16384, step=16,
                    tooltip="Target image width. Must match the width of your empty latent — AReA builds its schedule from this geometry and validates it against the latent at every forward.",
                ),
                io.Int.Input(
                    "height",
                    default=2048, min=16, max=16384, step=16,
                    tooltip="Target image height. Must match the height of your empty latent.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=["auto", "flux", "zimage", "krea2", "qwen21"],
                    default="auto",
                    tooltip="Specify the model architecture. 'auto' usually works. FLUX and Z-Image (Lumina2 / NextDiT) share a 16 px patch grid, so the same width/height serve either; Krea-2 (SingleStreamDiT) uses the same 16 px token grid (ASSUMED — its VAE downscale is unverified, so a 16x VAE would make its token 32 px) and needs no padding; Qwen-Image-2.1 (QwenImage21Transformer2DModel) uses the same 16 px token grid (MEASURED — its latent format downsamples 16x and 2.1 does not patchify) and pads nothing, and AReA switches its prefix K/V cache off on its own clone (the cached path is indistinguishable from the image segment, so with the cache on AReA would engage on no call at all); Nunchaku's fused kernels bypass the attention seam, so it is reported unchanged.",
                ),
                io.Boolean.Input(
                    "enable_area",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Enable or disable AReA. When disabled, the model is returned with nothing patched.",
                ),
                io.Int.Input(
                    "phase1_steps",
                    default=PHASE1_STEPS, min=0, max=100, step=1,
                    tooltip="Number of leading denoising steps that run Phase I (coarse cross-scale attention), clamped to the total step count. 8 = the reference's FLUX constant (of 28). 0 = all Phase II (windowed attention from the first step). Only read when phase1_mode is 'steps'. Z-Image Turbo runs short (8-step) schedules: use 1-3 there (or sigma mode) — 8 of an 8-step run pools every step, which the distilled sampler cannot re-detail. On a SHORT schedule the effective coarse steps are capped at total_steps - 2 (two fine steps must remain, or the pooling footprint never heals): the forward-time log prints the knob, the cap and the effective count once per schedule, so a capped run is readable and the knob stays authoritative below the cap.",
                ),
                io.Combo.Input(
                    "phase1_mode",
                    options=["auto", "steps", "sigma", "bootstrap"],
                    default="auto",
                    tooltip="How the Phase I -> Phase II switch is expressed. 'auto' = resolved per backbone: 'sigma' on Z-Image and Qwen-Image-2.1 (the leading steps still at or above phase1_sigma run Phase I, so the switch follows the schedule) and 'steps' on FLUX and Krea-2. 'steps' = the leading phase1_steps steps run Phase I (the reference's mode). 'sigma' = the leading steps whose noise level is still at or above phase1_sigma run Phase I, so the switch follows the schedule rather than a fixed count (8 of 28 sits at a sigma of roughly 0.88 on a shift-3.0 flow). 'bootstrap' = the paper's Appendix F mode: no Phase I at all, windowed Phase II from the first step; the early steps must have composed the canvas in a separate native-resolution pass whose latent was upscaled and re-noised to phase1_sigma (see the README pattern).",
                ),
                io.Float.Input(
                    "phase1_sigma",
                    default=ZIMAGE_SIGMA_STAR, min=0.0, max=1.0, step=0.01,
                    tooltip="Noise level at which Phase I gives way to Phase II, when phase1_mode is 'sigma'. 0.89 = the paper's value for its first backbone; 0.86 was its Qwen-Image value, so treat this as per-schedule calibration rather than a universal constant.",
                ),
                io.Combo.Input(
                    "pooling",
                    options=["auto"] + list(POOLING),
                    default="auto",
                    tooltip="Phase I pooling: which token of each f x f image block the pooled key/value comes from. 'auto' = resolved per backbone: 'subsample' on FLUX, 'mean' on Z-Image and Krea-2, 'medoid' on Qwen-Image-2.1. 'subsample' = block centre. 'mean' = block average for values, and for keys the block-mean direction rescaled to the mean member norm (the centroid keeps every token contributing while the norm keeps the attention temperature at the un-pooled one). 'medoid' = the token closest to the block-mean key — on Qwen-Image-2.1 it keeps ONE real token's key AND value, so the pooled cell carries a single content instead of a synthetic key over a blended value. 'random' = a token drawn per (seed, step, block). Keys must be pooled BEFORE the position rotation, so all four keep the survivors in the query's frame.",
                ),
                io.Int.Input(
                    "halo",
                    default=8, min=0, max=64, step=1,
                    tooltip="Widest Halo ring a Phase II window may draw (in patches). Each side of each window draws its own fresh width in [0, halo] at every step; AReA narrows it only when the windowed step would otherwise cost more than a full-attention step.",
                ),
                io.Float.Input(
                    "bridge_ratio",
                    default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="Rho of the bridge: each Phase II window also attends to R = round(rho * w)^2 of the canvas's most distinctive unrotated keys, spread over an 8x8 cell grid so no region is starved. 0 = no bridge.",
                ),
                io.Int.Input(
                    "bridge_floor_steps",
                    default=2, min=0, max=16, step=1,
                    tooltip="How many leading Phase II steps keep the paper's full bridge (R = round(rho*w)^2) and the full halo instead of the cost-capped ones. On short schedules (12 total steps or fewer) the cost cap starves the bridge exactly on the composition-critical steps, so those steps are exempted here; a 28-step FLUX run is never floored (long schedules ignore this knob). 0 = the cost cap on every step. The exemption costs about 1.0-1.4x full attention per floored step (measured 1.25x-1.41x on the 2048px Turbo schedule), and warns once naming the multiplier.",
                ),
                io.Int.Input(
                    "window_start",
                    default=0, min=0, max=1024, step=1,
                    tooltip="Window side (patches) Phase II starts its ramp at, instead of the cost-optimal size — smaller gives the early windows more global context. Short schedules only: a long (28-step) schedule ignores this knob. 0 = the cost-optimal start, which is clamped into [w_min, grid]; the ramp still shrinks to w_min.",
                ),
                io.Int.Input(
                    "window_hold_steps",
                    default=0, min=0, max=8, step=1,
                    tooltip="Extra Phase II steps that hold the start window before the ramp shrinks it (a start-heavy ramp for short schedules). Short schedules only: a long (28-step) schedule ignores this knob.",
                ),
                io.Int.Input(
                    "seed",
                    default=0, min=0, max=0xFFFFFFFFFFFFFFFF, step=1,
                    tooltip="Seed of every draw (tile placement, Halo widths and reaches, random pooling). Keyed by (seed, step), so the same seed and step always build the same windows and halos — and cond and uncond attend through identical graphs. No global RNG is read.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The model patched with AReA.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, width, height):
        # 1. Bypass ComfyUI's uninitialized state on load
        if width is None or height is None:
            return True

        # 2. Hard check: a FLUX patch is 16 px (VAE 8 x patch 2), so a latent that
        #    is not a multiple of 16 would silently change the patch grid.
        if int(width) % 16 != 0 or int(height) % 16 != 0:
            return f"Width and height must be multiples of 16 (FLUX: VAE 8 x patch 2). Got {width}x{height}."

        # 3. Pass to the shared validation for any other structural checks
        return validate_resolution(width, height)

    @classmethod
    def execute(cls, model, width: int, height: int, model_type: str, enable_area: bool, phase1_steps: int = PHASE1_STEPS, pooling: str = "auto", halo: int = 8, bridge_ratio: float = 0.5, bridge_floor_steps: int = 2, window_start: int = 0, window_hold_steps: int = 0, seed: int = 0, phase1_mode: str = "auto", phase1_sigma: float = ZIMAGE_SIGMA_STAR) -> io.NodeOutput:
        # Fallback for unlinked/None inputs
        width = 2048 if width is None else int(width)
        height = 2048 if height is None else int(height)

        patched_model = apply_area_to_model(
            model, model_type, width, height,
            enable_area=bool(enable_area),
            phase1_steps=int(phase1_steps),
            pooling=str(pooling),
            halo=int(halo),
            bridge_ratio=float(bridge_ratio),
            bridge_floor_steps=int(bridge_floor_steps),
            window_start=int(window_start),
            window_hold_steps=int(window_hold_steps),
            seed=int(seed),
            **_phase1_switch_kwargs(phase1_mode, phase1_sigma),
        )
        return io.NodeOutput(patched_model)
