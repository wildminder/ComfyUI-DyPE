"""SPA node — Spatial Position Alignment, HRDiT 2608.07003 (schema + execute).

Node definitions live in ``nodes/``; engines in ``src/`` (pack layout plan
2026-09-08). This module is importable both inside the ComfyUI-loaded pack
package (relative ``..src`` imports) and flat from the repo root (tests,
calibration CLI).
"""

from comfy_api.latest import io

try:  # loaded as pack package (ComfyUI loader)
    from ..src.spa import apply_spa_to_model, parse_layer_filter
    from ..src.validation import validate_resolution
except ImportError:  # flat repo layout (tests / CLI)
    from src.spa import apply_spa_to_model, parse_layer_filter
    from src.validation import validate_resolution


class SPA(io.ComfyNode):
    """
    Applies SPA (Spatial Position Alignment, HRDiT 2608.07003) to a model.

    SPA bundles each spatial axis into groups of N tokens (the paper's bundle
    size) before the positions enter the positional embedding, then slides the
    bundle boundary over each axis and averages the resulting attention OUTPUTS.
    This restores spatial distinguishability at ultra-high resolution without
    retraining the model. While the grid is inside the model's trained extent
    (e.g. <= 1024px) SPA is an automatic no-op. Combine with the HAP node
    (attention pruning) for the full HRDiT pipeline — when both are active,
    each of SPA's averaged passes runs through the HAP kernel.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SPA",
            display_name="SPA (HRDiT)",
            category="WMNodes/image",
            description="Spatial Position Alignment (HRDiT). Prevents high-resolution spatial disorder by bundling + averaging RoPE positions. Static, no timestep dependence.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The model to patch with SPA.",
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
                io.Boolean.Input(
                    "enable_spa",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Enable or disable SPA. When disabled, the base RoPE is emitted unchanged.",
                ),
                io.Int.Input(
                    "bundle_size",
                    default=0, min=0, max=256, step=1,
                    optional=True,
                    tooltip="SPA bundle size N (HRDiT paper): tokens per bundle. 0 = auto (minimal compression that keeps every bundled position in-distribution). 1 = off (plain passthrough). 2..8 = explicit (paper recommends 3 at 2K, 5 at 4K). While the grid is inside the model's trained extent (e.g. <= 1024px) SPA is automatically a no-op. Explicit N is floored by the in-distribution minimum so bundled positions never go out of distribution; the averaged-pass count is capped at 15. A single shared bundle size is used for BOTH axes so non-square images keep their aspect ratio (no horizontal squish). Legacy values >= 32 (old group_num semantics) are treated as auto with a warning.",
                ),
                io.Float.Input(
                    "spa_start_sigma",
                    default=1.0, min=0.0, max=1.0, step=0.05,
                    optional=True,
                    tooltip="Optional sigma-threshold gate (AND-combined with spa_steps): SPA runs only while the current sigma is ABOVE this threshold. 1.0 = no sigma gating (default). Lower values make later steps run at baseline speed.",
                ),
                io.Int.Input(
                    "spa_steps",
                    default=3, min=0, max=100, step=1,
                    optional=True,
                    tooltip="Step gating (HRDiT applies SPA only on leading denoising steps): number of LEADING steps on which SPA is active. 3 = HRDiT default (recommended speed/quality tradeoff). 0 = active on every step (backward compatible, slower). A new generation (sigma jump-up) resets the counter. Later steps run plain attention at baseline speed.",
                ),
                io.String.Input(
                    "spa_layer_filter",
                    default="",
                    optional=True,
                    tooltip="Per-layer SPA filter (HRDiT set_spa_filter): restrict the averaged-pass SPA to a subset of transformer layers. Flat layer-index spec: '0-18,38-57' (inclusive ranges, comma-separated) or a single index '3'. Empty = every layer (default). Filtered-out layers run plain attention; the layer counter and HAP are unaffected. Invalid specs raise an error.",
                ),
                io.Boolean.Input(
                    "proportional_attention",
                    default=False,
                    label_on="Enabled",
                    label_off="Disabled",
                    optional=True,
                    tooltip="HRDiT proportional attention scaling: scales the attention logits by sqrt(ln(seq_len)/ln(train_seq_len)) to compensate entropy dilution on long sequences. Exact no-op at/below the trained extent (1024px). Off by default (bit-identical to previous behaviour). Either the SPA or the HAP node may enable it.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="Patched Model",
                    tooltip="The model patched with SPA.",
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
    def execute(cls, model, width: int, height: int, model_type: str, enable_spa: bool, bundle_size: int = 0, spa_start_sigma: float = 1.0, spa_steps: int = 3, spa_layer_filter: str = "", proportional_attention: bool = False) -> io.NodeOutput:
        # Fallback for unlinked/None inputs
        width = 1024 if width is None else int(width)
        height = 1024 if height is None else int(height)

        bs = None if (bundle_size is None or bundle_size <= 0) else int(bundle_size)
        try:
            parsed_filter = parse_layer_filter(spa_layer_filter)
        except ValueError as exc:
            raise ValueError(f"SPA: invalid spa_layer_filter {spa_layer_filter!r}: {exc}") from exc

        patched_model = apply_spa_to_model(
            model, model_type, width, height,
            enable_spa=enable_spa, bundle_size=bs,
            spa_start_sigma=float(spa_start_sigma),
            spa_steps=int(spa_steps),
            spa_layer_filter=parsed_filter,
            proportional_attention=bool(proportional_attention),
        )
        return io.NodeOutput(patched_model)
