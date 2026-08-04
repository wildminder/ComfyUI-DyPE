import torch
from comfy_api.latest import ComfyExtension, io
from .src.patch_utils import apply_dype_to_model, apply_sega_to_model

class DyPE_FLUX(io.ComfyNode):
    """
    Applies DyPE (Dynamic Position Extrapolation) to a FLUX model.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="DyPE_FLUX",
            display_name="DyPE",
            category="model_patches/position_encoding",
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
                    options=["auto", "flux", "nunchaku", "qwen", "zimage", "anima"],
                    default="auto",
                    tooltip="Specify the model architecture. 'auto' usually works",
                ),
                io.Combo.Input(
                    "method",
                    options=["vision_yarn", "yarn", "ntk", "pi", "base"],
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


class SEGA(io.ComfyNode):
    """
    Applies SEGA (Spectral-Energy Guided Attention) to a model.
    SEGA computes per-RoPE-dimension mscale from the latent's Fourier
    spectrum at each denoising step for content-aware attention sharpening.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SEGA",
            display_name="SEGA",
            category="model_patches/position_encoding",
            description="Spectral-Energy Guided Attention for ultra-high-resolution generation. Computes per-dimension RoPE mscale from the latent's Fourier spectrum.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The model to patch with SEGA.",
                ),
                io.Int.Input(
                    "width",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image width. Must match the width of your empty latent.",
                ),
                io.Int.Input(
                    "height",
                    default=1024, min=16, max=8192, step=8,
                    tooltip="Target image height. Must match the height of your empty latent.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=["auto", "flux", "nunchaku", "qwen", "zimage", "anima"],
                    default="auto",
                    tooltip="Specify the model architecture. 'auto' usually works.",
                ),
                io.Combo.Input(
                    "method",
                    options=["sega", "ntk"],
                    default="sega",
                    tooltip="SEGA = NTK base + spectral per-dim mscale. NTK = base NTK only (no spectral).",
                ),
                io.Float.Input(
                    "mscale_alpha",
                    default=0.15, min=0.0, max=1.0, step=0.01,
                    tooltip="SEGA amplitude. Controls how much spectral redistribution is applied.",
                ),
                io.Float.Input(
                    "mscale_beta",
                    default=1.5, min=0.0, max=10.0, step=0.1,
                    tooltip="SEGA tanh sharpness. Higher = more binary redistribution.",
                ),
                io.Float.Input(
                    "mscale_min",
                    default=1.0, min=0.1, max=2.0, step=0.05,
                    tooltip="Floor for per-frequency mscale values.",
                ),
                io.Float.Input(
                    "spread_min",
                    default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Minimum spectral spread (early denoising steps).",
                ),
                io.Float.Input(
                    "spread_max",
                    default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Maximum spectral spread (late denoising steps).",
                ),
                io.Float.Input(
                    "spread_alpha",
                    default=1.5, min=0.1, max=5.0, step=0.1,
                    tooltip="Non-linear mapping exponent for spread schedule.",
                ),
                io.Combo.Input(
                    "base_mscale_formula",
                    options=["power_res", "log_res"],
                    default="power_res",
                    tooltip="power_res: m_ref = s^kappa. log_res: m_ref = 1 + kappa*ln(s).",
                ),
                io.Float.Input(
                    "base_mscale_coefficient",
                    default=0.08, min=0.0, max=1.0, step=0.01,
                    tooltip="Kappa coefficient for base mscale. Paper uses 0.08.",
                ),
                io.Int.Input(
                    "base_resolution",
                    default=1024, min=256, max=4096, step=16,
                    tooltip="The native training resolution.",
                ),
                io.Float.Input(
                    "base_shift",
                    default=0.5, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Base shift for the noise schedule (mu).",
                ),
                io.Float.Input(
                    "max_shift",
                    default=1.15, min=0.0, max=10.0, step=0.01,
                    optional=True,
                    tooltip="Advanced: Max shift for the noise schedule (mu) at high resolutions.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The model patched with SEGA.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, width: int, height: int, model_type: str, method: str, mscale_alpha: float, mscale_beta: float, mscale_min: float, spread_min: float, spread_max: float, spread_alpha: float, base_mscale_formula: str, base_mscale_coefficient: float, base_resolution: int = 1024, base_shift: float = 0.5, max_shift: float = 1.15) -> io.NodeOutput:
        patched_model = apply_sega_to_model(
            model, model_type, width, height, method,
            mscale_alpha, mscale_beta, mscale_min,
            spread_min, spread_max, spread_alpha,
            base_mscale_formula, base_mscale_coefficient,
            base_resolution, base_shift, max_shift,
        )
        return io.NodeOutput(patched_model)


class DyPEExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DyPE_FLUX, SEGA]

async def comfy_entrypoint() -> DyPEExtension:
    return DyPEExtension()