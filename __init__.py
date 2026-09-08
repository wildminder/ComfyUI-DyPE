"""ComfyUI-DyPE pack entry point.

Node definitions live in ``nodes/``; engines/implementation in ``src/``
(pack layout plan 2026-09-08). This module only registers the extension.
"""

from comfy_api.latest import ComfyExtension, io

from .nodes import (
    DyPE_FLUX,
    FreeScaleNode,
    HAP,
    HAPCalibrate,
    HiFlowNode,
    PixelRushNode,
    SEGA,
    SPA,
)
from .src.qwen2d_vae_patch import install_qwen2d_patch


class DyPEExtension(ComfyExtension):
    async def on_load(self) -> None:
        """Install Qwen2D VAE patch on extension load."""
        install_qwen2d_patch()

    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [DyPE_FLUX, SEGA, SPA, HAP, HAPCalibrate, PixelRushNode, FreeScaleNode, HiFlowNode]


async def comfy_entrypoint() -> DyPEExtension:
    return DyPEExtension()
