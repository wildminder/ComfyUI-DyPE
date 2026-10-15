"""AReA node — training-free two-phase attention replacement (schema + execute).

Node definitions live in ``nodes/``; engines in ``src/`` (pack layout plan
2026-09-08). This module is importable both inside the ComfyUI-loaded pack
package (relative ``..src`` imports) and flat from the repo root (tests).

AReA (arXiv 2610.17227) replaces FLUX image self-attention with a time-dependent
attention graph: the first ``phase1_steps`` steps pool image K/V to one token per
``f x f`` block and run cross-scale attention (Phase I), the remaining steps run
windowed attention whose windows follow the image's Haar-DWT detail map, each
extended by a randomized Halo ring and a global-KV bridge (Phase II).  At or
below the native 1024 px extent it is a provable no-op.
"""

from comfy_api.latest import io

try:  # loaded as pack package (ComfyUI loader)
    from ..src.area import PHASE1_STEPS, apply_area_to_model
    from ..src.area_attn import POOLING
    from ..src.validation import validate_resolution
except ImportError:  # flat repo layout (tests / CLI)
    from src.area import PHASE1_STEPS, apply_area_to_model
    from src.area_attn import POOLING
    from src.validation import validate_resolution


class AReA(io.ComfyNode):
    """
    Applies AReA (arXiv 2610.17227) to a model: a training-free, time-dependent
    replacement of FLUX image self-attention.

    Phase I pools image keys/values to one token per f x f block and runs
    cross-scale attention on a coarse position grid while the queries stay at
    full resolution. Phase II runs windowed attention whose window sizes follow
    the image's Haar-DWT detail map (smooth tiles split into quadrants), with a
    randomized Halo ring and a global-KV bridge of distinctive tokens per
    window. Chaining with DyPE / I-Max positional scaling composes (AReA's
    coarse positions ride on the installed rope embedder); below 1024 px AReA is
    an automatically a no-op.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="AReA",
            display_name="AReA",
            category="WMNodes/image",
            description="AReA (arXiv 2610.17227). Replaces FLUX image self-attention with a time-dependent attention graph: coarse cross-scale attention on leading steps, DWT-guided windowed attention (Halo + bridge tokens) on the rest. Training-free; a no-op at or below 1024 px.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The model to patch with AReA. FLUX only in v1.",
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
                    options=["auto", "flux"],
                    default="auto",
                    tooltip="Specify the model architecture. 'auto' usually works; AReA supports FLUX only in v1 (Nunchaku's fused kernels bypass the attention seam, so it is reported unchanged).",
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
                    tooltip="Number of leading denoising steps that run Phase I (coarse cross-scale attention), clamped to the total step count. 8 = the reference's FLUX constant (of 28). 0 = all Phase II (windowed attention from the first step).",
                ),
                io.Combo.Input(
                    "pooling",
                    options=list(POOLING),
                    default="subsample",
                    tooltip="Phase I pooling: which token of each f x f image block the pooled key/value comes from. 'subsample' = block centre, 'mean' = block average, 'medoid' = the token closest to the block-mean key, 'random' = a token drawn per (seed, step, block). Keys must be pooled BEFORE the position rotation, so all four keep the survivors in the query's frame.",
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
    def execute(cls, model, width: int, height: int, model_type: str, enable_area: bool, phase1_steps: int = PHASE1_STEPS, pooling: str = "subsample", halo: int = 8, bridge_ratio: float = 0.5, seed: int = 0) -> io.NodeOutput:
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
            seed=int(seed),
        )
        return io.NodeOutput(patched_model)
