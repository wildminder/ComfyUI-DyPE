"""Empty Qwen-Image 2.1 latent node.

Qwen-Image 2.1 has a 64-channel latent at a 16x spatial downscale and
``latent_dimensions == 2`` (comfy/latent_formats.py:957-960). Core ComfyUI
ships no text-to-image empty-latent node for it: the closest,
``EmptyQwenImageLayeredLatentImage`` (comfy_extras/nodes_qwen.py:213), is the
layered/edit latent — 16 channels, ``//8``, 5-D — wrong on all three axes for
a plain t2i graph, and ``EmptyLatentImage`` is the 4-channel ``//8`` one.
``TextEncodeQwenImage21`` does emit a correctly-shaped empty latent, but only
as a side effect of encoding a prompt, so the latent size cannot be set
independently of the text encoder.

This node is the t2i counterpart: the same shape the model expects, with
width/height as first-class inputs.
"""

from __future__ import annotations

import torch
from comfy_api.latest import io

# Qwen-Image 2.1 latent geometry (comfy/latent_formats.py:957-960 and the
# 16x VAE at comfy/sd.py:841-847).
LATENT_CHANNELS = 64
SPACIAL_DOWNSCALE = 16
# ComfyUI's nodes.MAX_RESOLUTION — imported lazily-free because a pack module
# must not bind ComfyUI's own `nodes` package name at module scope.
MAX_RESOLUTION = 16384


class EmptyQwenImage21LatentImage(io.ComfyNode):
    """Empty latent for Qwen-Image 2.1 text-to-image sampling."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="EmptyQwenImage21LatentImage",
            display_name="Empty Qwen Image 2.1 Latent",
            category="WMNodes/image",
            description=(
                "Empty 64-channel latent at a 16x spatial downscale for "
                "Qwen-Image 2.1 text-to-image sampling. Width and height are "
                "snapped to a multiple of 16 (the VAE downscale). Not for "
                "the layered/edit latent — that one is 16 channels and 5-D."
            ),
            inputs=[
                io.Int.Input(
                    "width", default=1328, min=16, max=MAX_RESOLUTION, step=16,
                    tooltip="Width in pixels (multiple of 16).",
                ),
                io.Int.Input(
                    "height", default=1328, min=16, max=MAX_RESOLUTION, step=16,
                    tooltip="Height in pixels (multiple of 16).",
                ),
                io.Int.Input(
                    "batch_size", default=1, min=1, max=4096,
                    tooltip="Number of latents to generate together.",
                ),
            ],
            outputs=[
                io.Latent.Output(),
            ],
        )

    @classmethod
    def validate_inputs(cls, width, height, batch_size):
        if width is None or height is None:
            return True  # uninitialized graph state
        if width % SPACIAL_DOWNSCALE or height % SPACIAL_DOWNSCALE:
            return (
                f"width and height must be multiples of {SPACIAL_DOWNSCALE} "
                f"for Qwen-Image 2.1 (the VAE's spatial downscale); got "
                f"{width}x{height}."
            )
        return True

    @classmethod
    def execute(cls, width, height, batch_size=1) -> io.NodeOutput:
        import comfy.model_management

        # Snap rather than trust the graph: an API caller can set any int, and
        # a truncated 64-channel latent is a silent shape mismatch at the
        # first conv. (ComfyUI's own empty-latent nodes rely on `//` here.)
        w = max(SPACIAL_DOWNSCALE, int(width) // SPACIAL_DOWNSCALE * SPACIAL_DOWNSCALE)
        h = max(SPACIAL_DOWNSCALE, int(height) // SPACIAL_DOWNSCALE * SPACIAL_DOWNSCALE)
        latent = torch.zeros(
            [int(batch_size), LATENT_CHANNELS, h // SPACIAL_DOWNSCALE,
             w // SPACIAL_DOWNSCALE],
            device=comfy.model_management.intermediate_device(),
        )
        return io.NodeOutput({"samples": latent})
