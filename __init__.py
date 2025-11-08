import torch
from comfy_api.latest import ComfyExtension, io
from .src.patch import apply_dype_to_flux, apply_dype_to_qwen

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

class DyPE_QWEN(io.ComfyNode):
    """
    Applies DyPE (Dynamic Position Extrapolation) to a Qwen-Image model.
    This allows generating images at resolutions far beyond the model's training scale
    by dynamically adjusting positional encodings and the noise schedule.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DyPE_QWEN",
            display_name="DyPE for Qwen-Image",
            category="model_patches/unet",
            description="Applies DyPE (Dynamic Position Extrapolation) to a Qwen-Image model for ultra-high-resolution generation.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The Qwen-Image model to patch with DyPE.",
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
                    default=3.0, min=0.0, max=10.0, step=0.1,
                    optional=True,
                    tooltip="Controls DyPE strength over time (λt). 3.0=Very aggressive (best for 4K+), 2.0=Exponential, 1.0=Linear, 0.5=Sub-linear (better for ~2K). Higher values (up to 10.0) for extreme high-resolution generation."
                ),
                io.Float.Input(
                    "base_shift",
                    default=0.10, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Base shift for the noise schedule (mu). Default is 0.10."
                ),
                io.Float.Input(
                    "max_shift",
                    default=1.15, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Max shift for the noise schedule (mu) at high resolutions. Default is 1.15."
                ),
                io.Float.Input(
                    "editing_strength",
                    default=0.0, min=0.0, max=1.0, step=0.1,
                    optional=True,
                    tooltip="DyPE strength multiplier for image editing (0.0-1.0). Lower values preserve more original structure. Default 0.0 for maximum preservation. Set to 1.0 for pure generation."
                ),
                io.Combo.Input(
                    "editing_mode",
                    options=["adaptive", "timestep_aware", "resolution_aware", "minimal", "full"],
                    default="adaptive",
                    tooltip="Editing mode strategy: 'adaptive' (recommended) - timestep-aware scaling, 'timestep_aware' - more DyPE early/less late, 'resolution_aware' - only reduce at high res, 'minimal' - minimal DyPE for editing, 'full' - always full DyPE."
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The Qwen-Image model patched with DyPE.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, width: int, height: int, method: str, enable_dype: bool, dype_exponent: float = 3.0, base_shift: float = 0.10, max_shift: float = 1.15, editing_strength: float = 0.0, editing_mode: str = "adaptive") -> io.NodeOutput:
        """
        Clones the model and applies the DyPE patch for both the noise schedule and positional embeddings.
        """
        # Check if this is a Qwen model
        has_transformer = hasattr(model.model, "transformer") or hasattr(model.model, "diffusion_model")
        if not has_transformer:
            raise ValueError("This node is only compatible with Qwen-Image models.")
        
        patched_model = apply_dype_to_qwen(model, width, height, method, enable_dype, dype_exponent, base_shift, max_shift, editing_strength, editing_mode)
        return io.NodeOutput(patched_model)

class DyPEExtension(ComfyExtension):
    """Registers the DyPE nodes for both FLUX and Qwen-Image."""

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DyPE_FLUX, DyPE_QWEN]

async def comfy_entrypoint() -> DyPEExtension:
    return DyPEExtension()