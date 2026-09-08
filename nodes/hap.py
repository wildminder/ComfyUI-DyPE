"""HAP node — Head-Adaptive attention Pruning, HRDiT 2608.07003 (schema + execute).

Node definitions live in ``nodes/``; engines in ``src/`` (pack layout plan
2026-09-08). This module is importable both inside the ComfyUI-loaded pack
package (relative ``..src`` imports) and flat from the repo root (tests,
calibration CLI).
"""

import os

from comfy_api.latest import io

try:  # loaded as pack package (ComfyUI loader)
    from ..src.hap import ScopePlan, apply_hap_to_model
except ImportError:  # flat repo layout (tests / CLI)
    from src.hap import ScopePlan, apply_hap_to_model

# Pack root (this file lives in <root>/nodes/) — used to resolve the default
# scope-plan path shipped with the node pack.
_DYPE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class HAP(io.ComfyNode):
    """
    Applies HAP (Head-Adaptive attention Pruning, HRDiT 2608.07003) to a model.

    HAP is the SPEED half of HRDiT: each attention head attends only within its
    calibrated scope (a local band around each query plus text tokens and
    periodic global anchor blocks), pruning the rest of the attention.  The
    per-layer/per-head scopes come from an offline-calibrated scope plan (JSON).
    Combine with the SPA node for the full HRDiT pipeline (SPA fixes quality at
    high resolution; HAP restores speed).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HAP",
            display_name="HAP (HRDiT)",
            category="WMNodes/image",
            description="Head-Adaptive attention Pruning (HRDiT). Block-sparse attention from a calibrated scope plan — restores speed at high resolution. Combine with SPA for full HRDiT.",
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The model to patch with HAP.",
                ),
                io.Custom("SCOPE_PLAN").Input(
                    "scope_plan",
                    optional=True,
                    tooltip="Calibrated scope plan linked from the 'HAP Calibrate' node. When connected, it OVERRIDES scope_plan_path — no file needed.",
                ),
                io.String.Input(
                    "scope_plan_path",
                    default="configs/scope_plan_flux.json",
                    tooltip="Path to the scope-plan JSON (per-layer, per-head alpha/beta). Relative paths resolve against the ComfyUI-DyPE folder. Default ships the reference FLUX plan (57 layers x 24 heads). Generate a plan for your model/resolution with the 'HAP Calibrate' node or calibration/calibrate_hap.py. Ignored when a scope_plan is linked.",
                ),
                io.Combo.Input(
                    "model_type",
                    options=["auto", "flux", "nunchaku", "qwen", "zimage", "anima"],
                    default="auto",
                    tooltip="Specify the model architecture. 'auto' usually works.",
                ),
                io.Int.Input(
                    "anchor_stride",
                    default=32, min=0, max=1024, step=1,
                    optional=True,
                    tooltip="Global anchor blocks: every N-th image block is visible to all queries (keeps global coherence under pruning). 32 = HRDiT default. 0 = off.",
                ),
                io.Int.Input(
                    "text_len",
                    default=512, min=0, max=4096, step=1,
                    optional=True,
                    tooltip="Number of leading text tokens (always fully attended). 512 = FLUX convention. When SPA is also active, the boundary is derived from the position ids and this is only a fallback.",
                ),
                io.Boolean.Input(
                    "enable_hap",
                    default=True,
                    label_on="Enabled",
                    label_off="Disabled",
                    tooltip="Enable or disable HAP. When disabled, the model is returned unchanged.",
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
                    tooltip="The model patched with HAP.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, scope_plan_path: str, model_type: str,
                anchor_stride: int = 32, text_len: int = 512,
                enable_hap: bool = True,
                proportional_attention: bool = False,
                scope_plan=None) -> io.NodeOutput:
        # A linked SCOPE_PLAN object (from the HAP Calibrate node) OVERRIDES
        # the file path — no disk round-trip needed.
        if scope_plan is not None:
            try:
                plan = ScopePlan.from_dict(scope_plan)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"HAP: invalid linked scope_plan: {exc}"
                ) from exc
        else:
            path = scope_plan_path
            if not os.path.isabs(path):
                candidate = os.path.join(_DYPE_ROOT, path)
                if os.path.exists(candidate):
                    path = candidate
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"HAP: scope plan not found: {scope_plan_path!r} (resolved to "
                    f"{path!r}). Provide a path to a scope-plan JSON, link a "
                    f"scope_plan from the 'HAP Calibrate' node, or use the "
                    f"shipped default 'configs/scope_plan_flux.json'."
                )
            try:
                plan = ScopePlan.load(path)
            except ValueError as exc:
                raise ValueError(f"HAP: invalid scope plan {scope_plan_path!r}: {exc}") from exc
        patched_model = apply_hap_to_model(
            model, model_type, plan,
            anchor_stride=int(anchor_stride),
            enable_hap=bool(enable_hap),
            text_len=int(text_len),
            proportional_attention=bool(proportional_attention),
        )
        return io.NodeOutput(patched_model)
