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
    # HAP integration (plan P3/T3.4): number of leading TEXT tokens derived from
    # the position ids at registration time (tokens with row==col==0).  ``None``
    # = unknown (HAP falls back to ``HapContext.text_len``).
    text_len: Optional[int] = None


# Module-global, process-safe activation slot. Default ``None`` == no SPA hook.
_SPA_ACTIVE: "contextvars.ContextVar" = contextvars.ContextVar("spa_active", default=None)

# Step-gating slot (2026-08-15 slowdown fix D2a): the unet wrapper sets this to
# ``False`` for denoising steps whose normalized sigma is below ``spa_start_sigma``
# (HRDiT applies SPA only on the LEADING steps).  The attention wrapper treats a
# closed gate exactly like an inactive context -> plain attention, zero overhead.
# Default ``True`` == gate open (SPA allowed), preserving pre-gating behaviour.
_SPA_STEP_GATE: "contextvars.ContextVar" = contextvars.ContextVar("spa_step_gate", default=True)

# HAP (HRDiT speed half) activation slot (plan P2/T2.1): holds the active
# ``src.hap.HapContext`` (or ``None`` == HAP off).  Process-scoped exactly like
# ``_SPA_ACTIVE`` so concurrent/cross-model forwards never leak a scope plan.
_HAP_ACTIVE: "contextvars.ContextVar" = contextvars.ContextVar("hap_active", default=None)

# HRDiT per-forward attention-call counter (plan P2/T2.1): our hook patches a
# MODULE-LEVEL ``optimized_attention`` symbol (no per-block processor identity),
# so the layer index is a deterministic counter — reset to 0 by the unet wrapper
# before every forward and incremented by EVERY wrapper call (including gated-off
# early returns, so alignment with the model's block order is preserved).
_HRDIT_LAYER_IDX: "contextvars.ContextVar" = contextvars.ContextVar("hrdit_layer_idx", default=0)

# HAP plan-layer ordinal (2026-08-19, runtime layer-index mismatch fix): a
# SEPARATE per-forward counter that advances ONLY for attention calls the scope
# plan actually covers (square, unmasked, and matching the plan's head count).
# Calibration enumerates plan layers by the dominant-head-only ordinal (its
# heterogeneous-head-count filter drops auxiliary attention with other head
# counts), so the runtime must index the plan by the SAME ordinal — not the raw
# all-call counter.  Krea2 runs 4 auxiliary 20-head projector calls before its 28
# main 48-head blocks; the raw counter consumed indices 0-3 for the aux calls,
# shifting every main block by 4 and pushing the last 4 main blocks past the
# 28-layer plan ("layer 28 exceeds the scope plan").  Reset to 0 by the unet
# wrapper alongside ``_HRDIT_LAYER_IDX``.
_HAP_LAYER_IDX: "contextvars.ContextVar" = contextvars.ContextVar("hap_layer_idx", default=0)

# Proportional attention scaling flag (plan P7/T7.2, G4): the unet wrapper sets
# this from ``m._hrdit_proportional_attention`` for the duration of a forward.
# When True the unified attention wrapper pre-scales ``q`` by
# ``proportional_scale_ratio(seq_len)`` (reference ``attention.py:89-93``) so the
# softmax temperature grows with sequence length at high resolution.  Default
# ``False`` == feature off (bit-identical to the pre-feature baseline).
_HRDIT_PROPORTIONAL: "contextvars.ContextVar" = contextvars.ContextVar("hrdit_proportional", default=False)

# Per-layer SPA filter (plan P8/T8.2, G5): the unet wrapper sets this from
# ``m._spa_layer_filter`` for the duration of a forward.  ``None`` (default) ==
# SPA allowed on EVERY layer (backward compatible); a frozenset of flat layer
# indices (the per-forward attention-call counter) restricts the averaged-pass
# SPA to those layers only.  Filtered-out layers run plain attention — but the
# layer counter still advances (alignment is sacred) and HAP dispatch is NOT
# gated by this filter (reference semantics: the filter affects SPA alone).
_SPA_LAYER_FILTER: "contextvars.ContextVar" = contextvars.ContextVar("spa_layer_filter", default=None)


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


# --- HAP activation (plan P2/T2.1) -----------------------------------------

def get_hap_context():
    """Return the active ``src.hap.HapContext`` (or ``None`` == HAP off)."""
    return _HAP_ACTIVE.get()


def set_hap_context(ctx) -> None:
    """Set (or clear with ``None``) the active HAP context."""
    _HAP_ACTIVE.set(ctx)


# --- HRDiT layer-index counter (plan P2/T2.1) -------------------------------

def get_hrdit_layer_idx() -> int:
    """Return the current per-forward attention-call index (0-based)."""
    return _HRDIT_LAYER_IDX.get()


def set_hrdit_layer_idx(idx: int) -> None:
    """Set the per-forward attention-call index (the unet wrapper resets to 0)."""
    _HRDIT_LAYER_IDX.set(int(idx))


def next_hrdit_layer_idx() -> int:
    """Read the current layer index and advance the counter.

    EVERY unified-wrapper call must use this (including gated-off early
    returns) so the counter stays aligned with the model's block order.
    """
    idx = _HRDIT_LAYER_IDX.get()
    _HRDIT_LAYER_IDX.set(idx + 1)
    return idx


# --- HAP plan-layer ordinal (2026-08-19 layer-index mismatch fix) -----------

def get_hap_layer_idx() -> int:
    """Return the current HAP plan-layer ordinal (0-based).

    Unlike :func:`get_hrdit_layer_idx` (the RAW all-call counter), this ordinal
    advances ONLY for attention calls the scope plan covers (square, unmasked,
    head-count match) — mirroring calibration's dominant-head enumeration.
    """
    return _HAP_LAYER_IDX.get()


def set_hap_layer_idx(idx: int) -> None:
    """Set the HAP plan-layer ordinal (the unet wrapper resets to 0)."""
    _HAP_LAYER_IDX.set(int(idx))


def next_hap_layer_idx() -> int:
    """Read the current HAP plan-layer ordinal and advance the counter.

    Called ONLY by wrapper calls that the scope plan actually covers (the
    HAP dispatch path), so the ordinal stays aligned with calibration's
    dominant-head-only layer enumeration.
    """
    idx = _HAP_LAYER_IDX.get()
    _HAP_LAYER_IDX.set(idx + 1)
    return idx


# --- Proportional attention scaling flag (plan P7/T7.2) ---------------------

def get_hrdit_proportional() -> bool:
    """Return whether proportional attention scaling is active this forward."""
    return _HRDIT_PROPORTIONAL.get()


def set_hrdit_proportional(on: bool) -> None:
    """Enable/disable proportional attention scaling for the current forward."""
    _HRDIT_PROPORTIONAL.set(bool(on))


# --- Per-layer SPA filter (plan P8/T8.2) ------------------------------------

def get_spa_layer_filter():
    """Return the active per-layer SPA filter (frozenset of layer indices or None)."""
    return _SPA_LAYER_FILTER.get()


def set_spa_layer_filter(f) -> None:
    """Set (or clear with ``None``) the per-layer SPA filter for this forward."""
    _SPA_LAYER_FILTER.set(f)
