"""W4.2 — Canonical model-type detection (IMP-001 single source of truth).

Every patch family (DyPE, SEGA, SPA, HAP) previously carried its own detector
copy; the copies had drifted (only SPA's knew Krea-2's ``SingleStreamDiT``).
This module is now THE detector; all families delegate to
:func:`resolve_model_type`.

Precedence (mirrors SPA's proven semantics verbatim):

1. CLASS-NAME checks first — Krea-2's ``SingleStreamDiT`` shares the Qwen
   architecture but binds its own attention symbol, so it MUST be detected by
   class name BEFORE any requested-string or attr probe.
2. Explicit ``requested`` overrides ("flux"/"qwen"/"nunchaku"/"zimage"/
   "z_image"/"anima"/"krea2").
3. ``auto``: attribute probes in order — QwenImage/Anima class names,
   ``rope_embedder`` (Z-Image), ``model.pos_embed`` (Nunchaku),
   ``pe_embedder`` (FLUX), ``pos_embedder.dim_spatial_range`` (Anima).

Raises ``ValueError("The provided model is not a compatible model.")`` when
nothing matches.
"""

__all__ = ["resolve_model_type"]


def resolve_model_type(dm, requested: str = "auto") -> str:
    """Resolve the concrete backend key for a diffusion model.

    Args:
        dm: the live diffusion model object (attribute probes run on it).
        requested: user knob.  ``"auto"`` probes attributes; anything else is
            an explicit override (``z_image`` normalizes to ``zimage``).

    Returns one of ``"flux"``, ``"qwen"``, ``"krea2"``, ``"zimage"``,
    ``"nunchaku"``, ``"anima"``.

    Raises:
        ValueError: when ``requested == "auto"`` and no probe matches.
    """
    # 1. Class-name checks FIRST (Krea-2 before everything — see module doc).
    model_class_name = getattr(dm.__class__, "__name__", "")
    if model_class_name == "SingleStreamDiT":
        return "krea2"

    # 2. Explicit override.
    if requested == "krea2":
        return "krea2"
    if requested == "nunchaku":
        return "nunchaku"
    if requested == "qwen":
        return "qwen"
    if requested in ("z_image", "zimage"):
        return "zimage"
    if requested == "anima":
        return "anima"
    if requested == "flux":
        return "flux"

    # 3. auto: attribute probes.
    if "QwenImage" in model_class_name:
        return "qwen"
    if "Anima" in model_class_name or "MiniTrainDIT" in model_class_name:
        return "anima"
    if hasattr(dm, "rope_embedder"):
        return "zimage"
    if hasattr(dm, "model") and hasattr(dm.model, "pos_embed"):
        return "nunchaku"
    if hasattr(dm, "pe_embedder"):
        return "flux"
    if hasattr(dm, "pos_embedder") and hasattr(dm.pos_embedder, "dim_spatial_range"):
        return "anima"
    raise ValueError("The provided model is not a compatible model.")
