"""W5.1 — Shared input validation helpers (IMP-002).

ComfyUI V3 nodes can implement ``validate_inputs`` to reject invalid
configurations at GRAPH-BUILD time (before any GPU work).  The resolution
nodes previously accepted arbitrary width/height and relied on runtime
snapping; this module provides the shared validator so bad resolutions fail
fast with an actionable message.

Importable WITHOUT comfy_api (pure functions) so unit tests need no ComfyUI.
"""

__all__ = ["validate_resolution"]


def validate_resolution(width, height, *, min_px=16, max_px=8192):
    """Validate a resolution pair for the latent-aligned pipeline.

    Rules:
    * both axes must be multiples of 8 (VAE 8x downscale — the hard
      requirement; the runtime additionally snaps to /16 via
      ``_snap_to_multiple``, so widget values like 504 are accepted here and
      snapped at apply time),
    * both axes must lie within ``[min_px, max_px]``.

    Returns:
        True when valid; otherwise an error STRING suitable for returning
        from a node's ``validate_inputs`` classmethod (ComfyUI shows it in
        the graph UI).
    """
    try:
        w = int(width)
        h = int(height)
    except (TypeError, ValueError):
        return f"width/height must be integers; got {width!r}x{height!r}"

    if w % 8 or h % 8:
        return (
            f"width/height must be multiples of 8 (VAE alignment); "
            f"got {w}x{h}"
        )
    if not (min_px <= w <= max_px and min_px <= h <= max_px):
        return (
            f"width/height must be within [{min_px}, {max_px}]; "
            f"got {w}x{h}"
        )
    return True
