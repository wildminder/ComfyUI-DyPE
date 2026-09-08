"""SEGA node — Spectral-Energy Guided Attention (schema + execute wiring).

Node definitions live in ``nodes/``; engines in ``src/`` (pack layout plan
2026-09-08). This module is importable both inside the ComfyUI-loaded pack
package (relative ``..src`` imports) and flat from the repo root (tests,
calibration CLI).
"""

from comfy_api.latest import io

try:  # loaded as pack package (ComfyUI loader)
    from ..src.patch_utils import apply_sega_to_model
    from ..src.validation import validate_resolution
except ImportError:  # flat repo layout (tests / CLI)
    from src.patch_utils import apply_sega_to_model
    from src.validation import validate_resolution


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
            category="WMNodes/image",
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
    def validate_inputs(cls, width, height):
        # 1. Bypass ComfyUI's uninitialized state on load
        if width is None or height is None:
            return True

        # 2. Hard check: Reject if not a multiple of 8
        if width % 8 != 0 or height % 8 != 0:
            return f"Width and height must be multiples of 8. Got {width}x{height}."

        # 3. Pass to your existing validation for any other structural checks
        return validate_resolution(width, height)

    @classmethod
    def execute(cls, model, width: int, height: int, model_type: str, method: str, mscale_alpha: float, mscale_beta: float, mscale_min: float, spread_min: float, spread_max: float, spread_alpha: float, base_mscale_formula: str, base_mscale_coefficient: float, base_resolution: int = 1024, base_shift: float = 0.5, max_shift: float = 1.15) -> io.NodeOutput:
        # Fallback for unlinked/None inputs
        width = 1024 if width is None else int(width)
        height = 1024 if height is None else int(height)

        patched_model = apply_sega_to_model(
            model, model_type, width, height, method,
            mscale_alpha, mscale_beta, mscale_min,
            spread_min, spread_max, spread_alpha,
            base_mscale_formula, base_mscale_coefficient,
            base_resolution, base_shift, max_shift,
        )
        return io.NodeOutput(patched_model)
