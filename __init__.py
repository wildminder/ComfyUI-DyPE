import torch
from comfy_api.latest import ComfyExtension, io

try:
    from .src.patch import apply_dype_to_flux
    from .src.qwen_patch import apply_dype_to_qwen_clip
except ImportError:  # pragma: no cover - fallback for direct execution contexts
    from src.patch import apply_dype_to_flux  # type: ignore[no-redef]
    from src.qwen_patch import apply_dype_to_qwen_clip  # type: ignore[no-redef]

class DyPE_FLUX(io.ComfyNode):
    """
    Applies DyPE (Dynamic Position Extrapolation) to a FLUX model.
    This allows generating images at resolutions far beyond the model's training scale
    by dynamically adjusting positional encodings and the noise schedule.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DyPE_FLUX",
            display_name="DyPE for FLUX",
            category="model_patches/unet",
            description="Applies DyPE (Dynamic Position Extrapolation) to a FLUX model for ultra-high-resolution generation.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The FLUX model to patch with DyPE.",
                ),
                io.Int.Input(
                    "width",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image width. Must match the width of your empty latent."
                ),
                io.Int.Input(
                    "height",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image height. Must match the height of your empty latent."
                ),
                io.Combo.Input(
                    "method",
                    options=["yarn", "ntk", "base"],
                    default="yarn",
                    tooltip="Position encoding extrapolation method (YARN recommended).",
                ),
                io.Boolean.Input(
                    "enable_dype",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Enable or disable Dynamic Position Extrapolation for RoPE.",
                ),
                io.Float.Input(
                    "dype_exponent",
                    default=2.0, min=0.0, max=4.0, step=0.1,
                    optional=True,
                    tooltip="Controls DyPE strength over time (λt). 2.0=Exponential (best for 4K+), 1.0=Linear, 0.5=Sub-linear (better for ~2K)."
                ),
                io.Float.Input(
                    "base_shift",
                    default=0.5, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Base shift for the noise schedule (mu). Default is 0.5."
                ),
                io.Float.Input(
                    "max_shift",
                    default=1.15, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Max shift for the noise schedule (mu) at high resolutions. Default is 1.15."
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The FLUX model patched with DyPE.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, width: int, height: int, method: str, enable_dype: bool, dype_exponent: float = 2.0, base_shift: float = 0.5, max_shift: float = 1.15) -> io.NodeOutput:
        """
        Clones the model and applies the DyPE patch for both the noise schedule and positional embeddings.
        """
        if not hasattr(model.model, "diffusion_model") or not hasattr(model.model.diffusion_model, "pe_embedder"):
             raise ValueError("This node is only compatible with FLUX models.")
        
        patched_model = apply_dype_to_flux(model, width, height, method, enable_dype, dype_exponent, base_shift, max_shift)
        return io.NodeOutput(patched_model)

class DyPE_QWEN_CLIP(io.ComfyNode):
    """
    Applies DyPE position extrapolation to a Qwen-based CLIP/text encoder.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DyPE_QwenClip",
            display_name="DyPE for Qwen CLIP",
            category="model_patches/clip",
            description="Extends Qwen text encoder RoPE for longer prompts using DyPE-style extrapolation.",
            inputs=[
                io.Clip.Input("clip"),
                io.Combo.Input(
                    "method",
                    options=["yarn", "ntk", "base"],
                    default="ntk",
                    tooltip="RoPE extrapolation strategy. NTK is a good default for language models.",
                ),
                io.Boolean.Input(
                    "enable_dype",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Toggle Dynamic Position Extrapolation scaling.",
                ),
                io.Float.Input(
                    "dype_exponent",
                    default=2.0, min=0.0, max=4.0, step=0.1,
                    optional=True,
                    tooltip="Controls how aggressively DyPE ramps with context length.",
                ),
                io.Int.Input(
                    "base_ctx_len",
                    default=8192, min=1024, max=512000, step=512,
                    tooltip="Context length the model was trained on. DyPE stays inactive at or below this size.",
                ),
                io.Int.Input(
                    "max_ctx_len",
                    default=262144, min=4096, max=1048576, step=512,
                    tooltip="Target maximum context DyPE should support.",
                ),
            ],
            outputs=[
                io.Clip.Output(
                    display_name="Patched CLIP",
                    tooltip="Qwen text encoder with DyPE RoPE scaling.",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, method: str, enable_dype: bool, dype_exponent: float = 2.0, base_ctx_len: int = 8192, max_ctx_len: int = 262144) -> io.NodeOutput:
        if not hasattr(clip, "cond_stage_model"):
            raise ValueError("This node expects a CLIP/text encoder input.")

        patched_clip = apply_dype_to_qwen_clip(
            clip,
            method=method,
            enable_dype=enable_dype,
            dype_exponent=dype_exponent,
            base_ctx_len=base_ctx_len,
            max_ctx_len=max_ctx_len,
        )
        return io.NodeOutput(patched_clip)

class DyPEExtension(ComfyExtension):
    """Registers the DyPE node."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DyPE_FLUX, DyPE_QWEN_CLIP]

async def comfy_entrypoint() -> DyPEExtension:
    return DyPEExtension()
