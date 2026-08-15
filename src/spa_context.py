"""Process-local SPA activation context.

SPA patches a *module-level* ``optimized_attention`` wrapper, but the activation
state (which N variants are bundled for the current forward, the base pe, the
tensor format) is scoped with a :class:`contextvars.ContextVar` so concurrent or
cross-model forwards never leak one model's bundle into another model's attention
(see remediation decision 2).

The embedder ``forward`` sets the active :class:`SPAContext` for the duration of a
forward; the unet wrapper and the embedder itself clear it so it can never leak
into a subsequent model's forward.
"""
import contextvars
from dataclasses import dataclass, field
from typing import List, Optional

import torch


@dataclass
class SPAContext:
    """Per-forward SPA activation state shared between embedder and attention hook."""

    active: bool = False
    bundle_size: int = 1
    base_pe: Optional[torch.Tensor] = None
    variant_pes: List[torch.Tensor] = field(default_factory=list)
    pre_roped: bool = True
    fmt: str = "flux"
    model_key: int = 0
    # P3 (D5 fix): composed delta rotations ``inv(base) @ variant``, cached once
    # per grid by the embedder (``_cached_variant_deltas``).  When populated and
    # ``pre_roped`` is True, the attention hook consumes these directly instead of
    # recomposing ``inv_rope``/``compose_rope`` on every attention call.  ``None``
    # (or a length mismatch) falls back to per-call composition.
    variant_deltas: Optional[List[torch.Tensor]] = None
    # Z-Image / Lumina multi-group support (see remediation note in spa.py).
    # ``pending`` accumulates one ``(kind, pos_ids)`` entry per ``rope_embedder``
    # call within a forward, where ``kind`` is ``"cap"`` (text, h==w==0) or
    # ``"pos"`` (image/siglip). The attention hook reassembles them in lumina's
    # group-major order to build the FULL-sequence variant PE (the global
    # single-group ``variant_pes`` cannot represent lumina's concatenated seq).
    pending: List = field(default_factory=list)
    uses_pending: bool = False
    embedder: object = None


# Module-global, process-safe activation slot. Default ``None`` == no SPA hook.
_SPA_ACTIVE: "contextvars.ContextVar" = contextvars.ContextVar("spa_active", default=None)

# Step-gating slot (2026-08-15 slowdown fix D2a): the unet wrapper sets this to
# ``False`` for denoising steps whose normalized sigma is below ``spa_start_sigma``
# (HRDiT applies SPA only on the LEADING steps).  The attention wrapper treats a
# closed gate exactly like an inactive context -> plain attention, zero overhead.
# Default ``True`` == gate open (SPA allowed), preserving pre-gating behaviour.
_SPA_STEP_GATE: "contextvars.ContextVar" = contextvars.ContextVar("spa_step_gate", default=True)


def get_spa_context() -> Optional[SPAContext]:
    """Return the active :class:`SPAContext` for the current execution context."""
    return _SPA_ACTIVE.get()


def set_spa_context(ctx: Optional[SPAContext]) -> None:
    """Set (or clear with ``None``) the active :class:`SPAContext`."""
    _SPA_ACTIVE.set(ctx)


def get_spa_step_gate() -> bool:
    """Return whether SPA is allowed to run on the CURRENT denoising step."""
    return _SPA_STEP_GATE.get()


def set_spa_step_gate(open_: bool) -> None:
    """Open (``True``) or close (``False``) the SPA step gate for this forward."""
    _SPA_STEP_GATE.set(bool(open_))
