import torch
from comfy_api.latest import ComfyExtension, io
from .src.patch_utils import apply_dype_to_model

class DyPE_FLUX(io.ComfyNode):
    """
    Applies DyPE (Dynamic Position Extrapolation) to a FLUX model.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DyPE_FLUX",
            display_name="DyPE",
            category="model_patches/unet",
            description="Applies DyPE (Dynamic Position Extrapolation) to a models for ultra-high-resolution generation.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The model to patch with DyPE.",
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
                    "model_type",
                    options=["auto", "flux", "nunchaku", "qwen", "zimage"],
                    default="auto",
                    tooltip="Specify the model architecture. 'auto' usually works",
                ),
                io.Combo.Input(
                    "method",
                    options=["vision_yarn", "yarn", "ntk", "base"],
                    default="vision_yarn",
                    tooltip="Position encoding extrapolation method.",
                ),
                io.Boolean.Input(
                    "yarn_alt_scaling",
                    default=False,
                    label_on="Anisotropic (High-Res)",
                    label_off="Isotropic (Stable Default)",
                    tooltip="[YARN Only] Alternate scaling for ultra-high resolutions. Not used for 'vision_yarn'.",
                ),
                io.Boolean.Input(
                    "enable_dype",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Enable or disable DyPE",
                ),
                io.Int.Input(
                    "base_resolution",
                    default=1024, min=256, max=4096, step=16,
                    tooltip="The native training resolution.",
                ),
                io.Float.Input(
                    "dype_start_sigma",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="When to start decaying the scaling effect (1.0 = Start, 0.5 = 50% through generation)."
                ),
                io.Float.Input(
                    "dype_scale",
                    default=2.0, min=0.0, max=8.0, step=0.1,
                    optional=True,
                    tooltip="Controls DyPE magnitude (λs). Default is 2.0."
                ),
                io.Float.Input(
                    "dype_exponent",
                    default=2.0, min=0.0, max=1000.0, step=0.1,
                    optional=True,
                    tooltip="Controls DyPE decay speed (λt). Higher = Faster decay. 2.0=Quadratic."
                ),
                io.Float.Input(
                    "base_shift",
                    default=0.5, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Base shift for the noise schedule (mu)."
                ),
                io.Float.Input(
                    "max_shift",
                    default=1.15, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Max shift for the noise schedule (mu) at high resolutions."
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The model patched with DyPE.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, width: int, height: int, model_type: str, method: str, yarn_alt_scaling: bool, enable_dype: bool, base_resolution: int = 1024, dype_start_sigma: float = 1.0, dype_scale: float = 2.0, dype_exponent: float = 2.0, base_shift: float = 0.5, max_shift: float = 1.15) -> io.NodeOutput:
        patched_model = apply_dype_to_model(model, model_type, width, height, method, yarn_alt_scaling, enable_dype, dype_scale, dype_exponent, base_shift, max_shift, base_resolution, dype_start_sigma)
        return io.NodeOutput(patched_model)

class DyPEExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DyPE_FLUX]

async def comfy_entrypoint() -> DyPEExtension:
    return DyPEExtension()