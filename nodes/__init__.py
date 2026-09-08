"""Node definitions for ComfyUI-DyPE (pack layout plan 2026-09-08).

Every ComfyUI-facing node class (``io.ComfyNode`` schema + execute wiring)
lives in this package; engines/implementation live in ``src/``. This
``__init__`` re-exports the full node surface so the pack entry point and
tests can import everything from one place.

Transitional note: the four cascade/calibration node classes still live in
``src/*_node.py`` until their move lands (S2/S3); their re-exports below
switch to local ``nodes/`` modules as each move lands.
"""

from .dype import DyPE_FLUX
from .sega import SEGA
from .spa import SPA
from .hap import HAP

# Transitional re-exports (S2/S3 move each module into nodes/):
try:  # loaded as pack package (ComfyUI loader)
    from ..src.hap_calib_node import HAPCalibrate
    from ..src.hiflow_node import HiFlowNode
    from ..src.pixelrush_node import PixelRushNode
    from ..src.freescale_node import FreeScaleNode
except ImportError:  # flat repo layout (tests / CLI)
    from src.hap_calib_node import HAPCalibrate
    from src.hiflow_node import HiFlowNode
    from src.pixelrush_node import PixelRushNode
    from src.freescale_node import FreeScaleNode

__all__ = [
    "DyPE_FLUX",
    "SEGA",
    "SPA",
    "HAP",
    "HAPCalibrate",
    "HiFlowNode",
    "PixelRushNode",
    "FreeScaleNode",
]
