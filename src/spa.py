"""SPA: Spatial Position Alignment (HRDiT, paper 2608.07003).

SPA bundles each spatial axis into groups of ``N`` tokens (the paper's bundle
size: ``N = 3`` at 2K, ``N = 5`` at 4K) *before* the positions enter the
positional embedding, then slides the bundle boundary over each axis and
averages the resulting attention OUTPUTS.  The user knob ``bundle_size`` IS the
paper's ``N`` (tokens per bundle): ``0 = auto`` (minimal compression that keeps
every bundled position in-distribution, i.e. HRDiT's ``group_num = 80``
ceiling), ``1 = off``, ``2..8`` explicit.  While the grid is inside the model's
trained extent (``max_pos <= trained_extent``, 64 for 1024px-trained DiTs) SPA
is an identity no-op — there is no position extrapolation to fix.  A SINGLE
shared bundle size is used for both axes (derived from the larger axis) so
non-square latents keep their aspect ratio; the independent row/col slides
restore a *unique* per-position signature (paper: SPA spatial-distinguishability),
so spatial structure is preserved at ultra-high resolution without retraining.

This module:
  * implements the bundle/slide math (``_phi``, ``build_bundle_id_variants``),
  * provides ``SPABasePosEmbed`` — a drop-in RoPE embedder that returns the *base*
    (no-extrapolation) RoPE on ``forward`` and registers the ``N`` bundled variant
    RoPEs in a process-scoped :class:`SPAContext`,
  * exposes ``apply_spa_to_model`` to patch any supported ComfyUI model with the
    attention-averaging hook.

Correctness note (fix for the rippled-mosaic bug): SPA no longer averages the
RoPE *rotation matrices* (averaging rotations then a single softmax is NOT equal
to averaging the per-variant attention outputs because softmax is nonlinear, and
``avg_n R_n`` is non-orthogonal).  Instead the attention hook runs ``N`` attention
passes over the bundled RoPE variants and averages the *outputs* — exactly what
HRDiT ``hrdit/attention.py::_spa_attention`` does.  See README "SPA (Spatial
Position Alignment)".
"""
import logging
import math
from typing import List, Optional

import torch

from .base import DyPEBasePosEmbed
from .patch_utils import _snap_to_multiple
from .rope import get_1d_ntk_pos_embed
from .spa_attn import (
    compose_rope,
    inv_rope,
    spa_averaged_attention,
)
from .spa_context import (
    SPAContext,
    get_hap_context,
    get_hrdit_proportional,
    get_spa_context,
    get_spa_layer_filter,
    get_spa_step_gate,
    next_hap_layer_idx,
    next_hrdit_layer_idx,
    set_hap_context,
    set_hap_layer_idx,
    set_hrdit_layer_idx,
    set_hrdit_proportional,
    set_spa_context,
    set_spa_layer_filter,
    set_spa_step_gate,
)

logger = logging.getLogger("ComfyUI-DyPE")

# HRDiT in-distribution ceiling (inference.py --group_num default 80): the maximum
# bundled position index the model saw in training.  ``N = 0`` (auto) derives the
# MINIMAL bundle size that keeps every bundled position ``<= SPA_IN_DIST_MAX``
# (``s_floor = ceil(max_pos / SPA_IN_DIST_MAX)``) — the HRDiT ``group_num = 80``
# behaviour.  Explicit ``N >= 2`` is honoured but never allowed to push bundled
# positions out of distribution (``s = max(N, s_floor)``).
SPA_DEFAULT_GROUP_NUM = 80
SPA_IN_DIST_MAX = SPA_DEFAULT_GROUP_NUM - 1  # 79

# Historical alias for :data:`SPA_DEFAULT_GROUP_NUM`.  Kept for backward
# compatibility with existing tests/imports.
SPA_MIN_GROUP_NUM = SPA_DEFAULT_GROUP_NUM

# Trained spatial extent per axis: the max token index the model saw in training.
# 1024px-trained DiTs (FLUX, Krea-2, Qwen-Image, Z-Image) train on 64x64 latent
# grids (1024 / 8 / 2 = 64).  While ``max_pos <= trained_extent`` the grid is
# IN-distribution and SPA is an identity no-op (there is no extrapolation to fix).
SPA_DEFAULT_TRAINED_EXTENT = 64

# Maximum user knob value with paper semantics (tokens per bundle).  The paper
# reports N = 3 at 2K and N = 5 at 4K; values above 8 are legacy ``group_num``
# semantics and are migrated to ``auto`` with a one-time WARNING (decision M1).
SPA_MAX_N = 8

# Legacy-knob threshold (decision M1): values >= 32 were set under the OLD
# ``group_num`` semantics (target bundles per axis, default 80).  Under the new
# paper-``N`` semantics they would mean absurd 32+-token bundles, so they are
# treated as ``auto`` (minimal in-distribution compression) with a one-time
# WARNING instead of silently changing the output.
SPA_LEGACY_KNOB_THRESHOLD = 32

# COST GUARD: maximum number of averaged attention passes per attention call.
# The pass count is ``2*s - 1`` where ``s`` is the derived bundle size; this cap
# bounds the worst-case slowdown (``s <= 8`` -> 15 passes).  With the in-dist
# floor, resolutions up to 8K stay BELOW this cap naturally (``s <= 7`` -> 13
# passes) while every bundled position stays in-distribution (``<= 79``).
#
# IMPORTANT (2026-08-15 blocky-output fix): the previous "resolution-aware band"
# (``SPA_BUNDLE_S_MIN=8`` / ``SPA_BUNDLE_S_MAX=16``) that FORCED ``s >= 8`` was
# REMOVED.  At low/medium resolutions (e.g. a 1024px latent = 64x64 tokens) SPA
# is an identity no-op (the grid is IN the model's trained distribution and needs
# NO bundling); the band forced ``s = 8``, collapsing 64 positions into ~8 bundles
# so adjacent tokens became positionally indistinguishable -> the "huge blocky
# pixel / JPG-compression" output artifact.  The trained-extent gate + in-dist
# floor + this pass cap is now the single source of truth:
#   * ``s == 1``  -> identity (SPA no-op, zero overhead)  [low/medium res]
#   * ``s > 1``   -> HRDiT bundling, passes capped at ``SPA_MAX_PASSES``  [high res]
SPA_MAX_PASSES = 15


def derive_bundle_s(max_pos: int, N: int,
                    trained_extent: int = SPA_DEFAULT_TRAINED_EXTENT) -> int:
    """Derive the shared per-axis bundle size ``s`` from the paper knob ``N``.

    ``N`` is the user knob with PAPER semantics (tokens per bundle):

    * ``N == 1``  -> off (identity),
    * ``N <= 0``  -> auto: the MINIMAL compression that keeps every bundled
      position in-distribution (``s = ceil(max_pos / SPA_IN_DIST_MAX)``),
    * ``N >= 2``  -> honoured, but never allowed to push bundled positions out
      of distribution (``s = max(N, s_floor)``).

    While ``max_pos <= trained_extent`` the grid is inside the model's trained
    distribution and there is no extrapolation to fix -> identity (``s = 1``).
    The result is capped so the pass count ``2*s - 1`` never exceeds
    :data:`SPA_MAX_PASSES`.

    This is the single source of truth for the bundle size (plan 2026-08-15,
    §2.1); :func:`build_bundle_id_variants` and the embedder both use it.
    """
    if N == 1 or max_pos <= trained_extent:
        return 1  # off / identity: no extrapolation to fix
    s_floor = max(1, math.ceil(max_pos / SPA_IN_DIST_MAX))
    s = s_floor if N <= 0 else max(N, s_floor)
    return min(s, (SPA_MAX_PASSES + 1) // 2)


# ---------------------------------------------------------------------------
# Bundle / slide math
# ---------------------------------------------------------------------------

def _phi(x: torch.Tensor, n1: int, size: int) -> torch.Tensor:
    """Bundle mapping (paper 4.1).

        phi(i) = 0                       if i < n1
               = ceil((i + 1 - n1) / N)  otherwise

    where ``size`` is the middle-bundle size ``N`` and ``n1`` the first-bundle
    size.  Implemented with floor division: ``ceil(a/N) = (a + N - 1) // N``.
    """
    if size <= 0:
        raise ValueError("bundle size must be a positive integer")
    if n1 <= 0:
        raise ValueError("first_bundle_size must be >= 1")
    x = x.long()
    tail = (x + 1 - n1 + size - 1) // size
    return torch.where(x < n1, torch.zeros_like(x), tail)


def bundle_ids_1d(length: int, bundle_size: int, first_bundle_size: int,
                  device=None) -> torch.Tensor:
    """1D bundle ids (paper pseudocode ``MAKE_BUNDLE_IDS``).

    Returns a LongTensor ``[length]`` with ``phi(i)`` for ``i in 0..length-1``.
    """
    if not (1 <= first_bundle_size <= bundle_size):
        raise ValueError("first_bundle_size must be in [1, bundle_size].")
    i = torch.arange(length, device=device)
    tail = torch.div(
        i + 1 - first_bundle_size + bundle_size - 1,
        bundle_size,
        rounding_mode="floor",
    )
    return torch.where(i < first_bundle_size, torch.zeros_like(i), tail)


def build_bundle_id_variants(ids: torch.Tensor, bundle_size: int,
                             trained_extent: int = SPA_DEFAULT_TRAINED_EXTENT,
                             ) -> List[torch.Tensor]:
    """HRDiT-faithful bundle/slide variants of a full position-id tensor.

    ``ids`` is ``(..., 3)`` with columns ``[batch, row, col]`` (FLUX / ComfyUI /
    Qwen / Z-Image / Nunchaku convention).  Text tokens have ``row = col = 0`` and
    are left unchanged because ``phi(0, n1, s) = 0`` for every ``n1 >= 1``.

    ``bundle_size`` is the PAPER's ``N`` (tokens per bundle; paper §4.1: ``N = 3``
    at 2K, ``N = 5`` at 4K):

    * ``N == 1``  -> off (single identity variant),
    * ``N <= 0``  -> auto: minimal compression keeping every bundled position
      in-distribution (HRDiT ``group_num = 80`` ceiling),
    * ``N >= 2``  -> honoured, floored by the in-distribution minimum so bundled
      positions never exceed :data:`SPA_IN_DIST_MAX`.

    The shared per-axis bundle size ``s`` comes from :func:`derive_bundle_s`
    (trained-extent gate + in-dist floor + pass cap).  While the grid is inside
    the model's trained extent (``max_pos <= trained_extent``) SPA is an identity
    no-op — there is no position extrapolation to fix (this is what eliminates
    the big-patch pixelation at 1024px).  The bundle boundary then SLIDES
    independently per axis:

        * 1 base variant        (n1_row=s, n1_col=s)
        * (s - 1) row slides    (n1_row = 1..s-1, n1_col=s)
        * (s - 1) col slides    (n1_row = s,    n1_col = 1..s-1)

    giving ``2*s - 1`` variants whose per-position signature is unique (paper: SPA
    spatial-distinguishability).  A single shared ``s`` is used for BOTH axes
    (derived from the larger axis) so non-square images keep their aspect ratio.

    COST GUARD: the pass count ``2*s - 1`` is capped at :data:`SPA_MAX_PASSES`
    inside :func:`derive_bundle_s`.

    Returns a single identity clone when ``bundle_size == 1`` (off) or when the
    derived ``s == 1`` (grid already in-distribution).
    """
    if bundle_size == 1:
        return [ids.clone()]
    rows = ids[..., 1].long()
    cols = ids[..., 2].long()
    max_pos = int(max(rows.max(), cols.max()))
    # Shared bundle size across BOTH axes (derived from the larger axis) so
    # non-square images keep their aspect ratio -- HRDiT's per-axis s_row != s_col
    # over-compresses the longer axis and squashes the image (the horizontal squish).
    # PAPER-N semantics (2026-08-15): N = tokens per bundle, with the in-dist
    # floor + trained-extent gate + pass cap in derive_bundle_s.
    s = derive_bundle_s(max_pos, bundle_size, trained_extent)
    # Identity: off (N == 1) or the grid is inside the model's trained extent
    # (no extrapolation to fix) -> SPA is a no-op.
    if s <= 1:
        return [ids.clone()]

    def variant(n1_row: int, n1_col: int) -> torch.Tensor:
        v = ids.clone()
        # size = the constant axis bundle size (NOT the slide value) -> in-distribution
        v[..., 1] = _phi(rows, n1_row, s).to(ids.dtype)
        v[..., 2] = _phi(cols, n1_col, s).to(ids.dtype)
        return v

    variants = [variant(s, s)]
    for n in range(1, s):
        variants.append(variant(n, s))
    for m in range(1, s):
        variants.append(variant(s, m))
    return variants


# ---------------------------------------------------------------------------
# Base SPA embedder
# ---------------------------------------------------------------------------

class SPABasePosEmbed(DyPEBasePosEmbed):
    """Base RoPE embedder that applies Spatial Position Alignment.

    Subclasses MUST implement :meth:`format_components` (the model-specific
    rotation-matrix formatting).  ``forward`` returns the *base* (no-extrapolation)
    RoPE for the original ids and registers the ``N`` bundled variant RoPEs in the
    process-scoped :class:`SPAContext` (consumed by the attention hook).  It never
    averages rotation matrices.

    SPA is *static* (no timestep dependence), so the resulting embedder needs
    no noise-schedule patch.
    """

    # Tensor layout of the produced RoPE (decision 5).  Subclasses override.
    _rope_fmt: str = "flux"

    def __init__(self, *args, enable_spa: bool = True, bundle_size: int = 0,
                 trained_extent: int = SPA_DEFAULT_TRAINED_EXTENT, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_spa = bool(enable_spa)
        # ``bundle_size`` is the PAPER's ``N`` (tokens per bundle): 0 = auto
        # (minimal in-distribution compression), 1 = off, 2..8 explicit.  The
        # shared per-axis bundle size ``s`` is derived from ``N`` + the grid
        # extent inside :func:`derive_bundle_s`.
        self.bundle_size = int(bundle_size)
        # Trained spatial extent per axis: while ``max_pos <= trained_extent``
        # the grid is in-distribution and SPA is an identity no-op.
        self.trained_extent = int(trained_extent)

    # -- base (no-extrapolation) RoPE components on bundled coordinates -------
    def _spa_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """Pure base RoPE (``ntk_factor = 1.0``) for every axis.

        This mirrors the reference HRDiT processor, which applies the model's
        "nor" (no-extrapolation) RoPE to each bundled id variant.  Per-axis theta
        is respected so Anima's native NTK factors are preserved.
        """
        n_axes = pos.shape[-1]
        out = []
        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            axis_theta = self.thetas[i] if self.thetas is not None else self.theta
            cos, sin = get_1d_ntk_pos_embed(
                dim=axis_dim,
                pos=axis_pos,
                theta=axis_theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
                ntk_factor=1.0,
            )
            out.append((cos, sin))
        return out

    # -- model-specific formatting -------------------------------------------
    # ``format_components`` is intentionally NOT defined here: concrete adapters
    # (e.g. ``PosEmbedFlux``) supply it, and due to MRO resolution the adapter's
    # implementation wins over this base.  Instantiating ``SPABasePosEmbed``
    # directly (without an adapter) is unsupported.

    # -- forward: base RoPE + register variants (NO tensor averaging) ---------
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if (not self.enable_spa) or self.bundle_size == 1:
            # Identity: base (no-extrapolation) RoPE, no hook effect.
            # (N == 1 is "off"; N == 0 is "auto" and stays active.)
            pos = ids.float()
            fdtype = torch.bfloat16 if pos.device.type == "cuda" else torch.float32
            set_spa_context(None)
            return self.format_components(self._spa_components(pos, fdtype), ids)

        # Return the BASE RoPE; register the variant RoPEs for the attention hook.
        # ``_register_variants`` returns the (cached) base PE so forward does not
        # recompute it (D2b).
        return self._register_variants(ids)

    def _pe_cache_key(self, ids: torch.Tensor):
        """Cache key for the static variant PEs (D2b).

        SPA is static (no timestep dependence), so the base + variant PEs depend
        only on the position-id grid and the bundle size.  The key combines the
        tensor shape, the bundle size, and the per-axis position extents so two
        grids with the same token count but different aspect ratio never collide.
        """
        return (
            tuple(ids.shape),
            self.bundle_size,
            self.trained_extent,
            int(ids[..., 1].max()) if ids.shape[-1] >= 2 else -1,
            int(ids[..., 2].max()) if ids.shape[-1] >= 3 else -1,
        )

    def _cached_variant_pes(self, ids: torch.Tensor):
        """Return ``(base_pe, variant_pes)`` for ``ids``, reusing the static cache.

        D2b: the variant PEs are recomputed on EVERY ``rope_embedder.forward`` call
        (every step, every layer) even though SPA is static.  This caches them on
        the embedder instance keyed by :meth:`_pe_cache_key` so the (potentially
        many) ``_spa_components`` / ``format_components`` calls run only once per
        unique grid.  The attention passes still run every call (q/k change per
        step), but PE construction cost is amortized away.

        P3 (D5 fix): the composed delta rotations ``inv(base) @ variant`` are ALSO
        composed here — ONCE per unique grid — and stored as the third cache slot.
        The attention hook consumes them via :meth:`_cached_variant_deltas` instead
        of recomposing on every attention call (every layer x every step).
        """
        if not hasattr(self, "_pe_cache"):
            self._pe_cache = {}
        key = self._pe_cache_key(ids)
        cached = self._pe_cache.get(key)
        if cached is not None:
            base_pe, variant_pes, _deltas = cached
            return base_pe, variant_pes

        pos = ids.float()
        fdtype = torch.bfloat16 if pos.device.type == "cuda" else torch.float32
        base_pe = self.format_components(self._spa_components(pos, fdtype), ids)
        variants = build_bundle_id_variants(ids, self.bundle_size, self.trained_extent)
        variant_pes = [
            self.format_components(self._spa_components(v.float(), fdtype), v)
            for v in variants
        ]
        # Compose the static delta rotations once per grid (P3 / D5).  The embedder
        # always registers with ``pre_roped=True``, so the hook needs exactly these
        # ``inv(base) @ variant`` deltas; composing them here amortizes the einsums.
        inv_base = inv_rope(base_pe, self._rope_fmt)
        deltas = [compose_rope(inv_base, vp, self._rope_fmt) for vp in variant_pes]
        result = (base_pe, variant_pes, deltas)
        self._pe_cache[key] = result
        return base_pe, variant_pes

    def _cached_variant_deltas(self, ids: torch.Tensor):
        """Return the cached composed delta rotations ``inv(base) @ variant`` (P3).

        Shares the static PE cache with :meth:`_cached_variant_pes`, so the deltas
        are composed exactly once per unique grid and reused on every subsequent
        forward / attention call (the D5 per-call recomputation fix).
        """
        if not hasattr(self, "_pe_cache"):
            self._pe_cache = {}
        key = self._pe_cache_key(ids)
        if key not in self._pe_cache:
            self._cached_variant_pes(ids)  # populate the shared cache
        return self._pe_cache[key][2]

    def _register_variants(self, ids: torch.Tensor) -> torch.Tensor:
        """Populate the active :class:`SPAContext` with the base + variant RoPEs.

        Returns the base PE so :meth:`forward` can reuse the cached tensor (D2b).

        ``ids`` is ``(B, L, 3)`` (FLUX / Qwen / Z-Image / Nunchaku convention).
        The hook consumes ``base_pe`` and ``variant_pes`` (same tensor layout as
        ``forward``'s return) to run ``N`` averaged attention passes.

        For Z-Image / Lumina the ``rope_embedder`` is called once PER token group
        (caption -> siglip -> image) but a single ``JointAttention`` then runs over
        the FULL concatenated sequence.  We therefore (a) REUSE the existing
        :class:`SPAContext` across the per-group calls instead of recreating it
        (recreating would discard the previously accumulated groups) and (b) append
        ``(kind, pos_ids)`` to ``ctx.pending`` so the hook can rebuild the
        FULL-sequence variant PE in lumina's group-major order.  ``kind`` is
        ``"cap"`` for text tokens (h == w == 0; they are unchanged by bundling) and
        ``"pos"`` for image/siglip tokens.
        """
        base_pe, variant_pes = self._cached_variant_pes(ids)
        # P3 (D5 fix): the composed delta rotations are cached once per grid; the
        # attention hook consumes them directly instead of recomposing per call.
        variant_deltas = self._cached_variant_deltas(ids)

        # HAP integration (plan P3/T3.4): derive the text/image boundary from the
        # position ids so the HAP band mask does not rely solely on the node's
        # ``text_len`` default.
        derived_text_len = _spa_derive_text_len(ids)

        ctx = get_spa_context()
        if ctx is None or not ctx.active:
            ctx = SPAContext(
                active=True,
                bundle_size=self.bundle_size,
                base_pe=base_pe,
                variant_pes=variant_pes,
                pre_roped=True,
                fmt=self._rope_fmt,
                model_key=id(self),
                variant_deltas=variant_deltas,
                text_len=derived_text_len,
            )
        else:
            # Reuse the live context (Z-Image multi-group accumulation).
            ctx.base_pe = base_pe
            ctx.variant_pes = variant_pes
            ctx.bundle_size = self.bundle_size
            ctx.fmt = self._rope_fmt
            ctx.model_key = id(self)
            ctx.variant_deltas = variant_deltas
            if derived_text_len is not None:
                ctx.text_len = derived_text_len

        if getattr(self, "_spa_is_zimage", False):
            is_text = (
                ids.shape[-1] >= 3
                and bool((ids[..., 1] == 0).all())
                and bool((ids[..., 2] == 0).all())
            )
            ctx.pending.append(("cap" if is_text else "pos", ids.clone()))
            ctx.uses_pending = True
            ctx.embedder = self

        set_spa_context(ctx)
        return base_pe

    def _compute_fullseq_pe(self, posids: torch.Tensor):
        """Build the base + variant RoPEs for a FULL-sequence ``posids`` tensor.

        Used by the Z-Image hook to compute the per-variant rotation for the entire
        concatenated attention sequence (caption + siglip + image) from the position
        ids reassembled by :func:`_spa_assemble_zimage_posids`.  Returns
        ``(base_pe, variant_pes)`` in this embedder's ``fmt`` layout.  Shares the
        static PE cache (D2b) with :meth:`_register_variants`.
        """
        return self._cached_variant_pes(posids)

    def _compute_fullseq_deltas(self, posids: torch.Tensor):
        """Return the cached composed deltas for a FULL-sequence ``posids`` tensor.

        P3 (D5 fix) for the Z-Image pending path: the full-sequence deltas are
        composed once per unique full grid (shared static cache) and reused on every
        attention call, instead of recomposing ``inv_rope``/``compose_rope`` per call.
        """
        return self._cached_variant_deltas(posids)


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def _get_attr_by_path(obj, path: str):
    """Resolve a dotted attribute path (e.g. ``'diffusion_model.pe_embedder'``)."""
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part)
    return cur


def _spa_ensure_no_incompatible_embedder(orig_embedder) -> None:
    """Reject DyPE/SEGA embedders; allow re-application of an existing SPA embedder.

    Resolves remediation decision 6 (mutual exclusivity of SPA and DyPE/SEGA).

    MESSAGE (2026-08-24): names the offending embedder class and the recovery
    path.  The pre-fix message ("Apply only one") gave no actionable remedy —
    users hit it after switching SEGA->SPA in one session and had to discover
    the reload requirement by trial.
    """
    if isinstance(orig_embedder, SPABasePosEmbed):
        return  # already SPA -> allow (re-apply / idempotent)
    name = type(orig_embedder).__name__
    if name.startswith("SegA") or name in {
        "PosEmbedFlux",
        "PosEmbedNunchaku",
        "PosEmbedQwen",
        "PosEmbedZImage",
        "PosEmbedAnima",
    }:
        kind = "SEGA" if name.startswith("SegA") else "DyPE"
        raise ValueError(
            f"SPA: the model already has {kind} applied "
            f"(embedder class {name!r}).  SPA and DyPE/SEGA are mutually "
            f"exclusive — they both rewrite position encoding and cannot be "
            f"stacked.  To switch methods, reload the model (Load Model node) "
            f"or route this branch from a fresh loader; bypassing the {kind} "
            f"node alone does NOT remove its patch."
        )


def _spa_patch_targets(model_type: str):
    """Backend-specific ``optimized_attention`` symbol(s) the DiT actually calls.

    ComfyUI DiT backends import ``optimized_attention`` / ``optimized_attention_masked``
    as a *module-level name* (e.g. ``comfy/ldm/flux/math.py`` does
    ``from comfy.ldm.modules.attention import optimized_attention``).  The symbol is
    therefore captured into the backend's own namespace at import time.  Patching the
    module attribute ``comfy.ldm.modules.attention.optimized_attention`` is INVISIBLE to
    them — they keep calling their own bound name, so SPA silently becomes a no-op (the
    "node makes no effect" regression).  We must patch the bound name in each backend.

    Returns a list of ``(module_import_path, attr_name, is_masked)`` tuples.
    """
    if model_type == "flux":
        return [("comfy.ldm.flux.math", "optimized_attention", False)]
    if model_type == "qwen":
        return [("comfy.ldm.qwen_image.model", "optimized_attention_masked", True)]
    if model_type == "krea2":
        # Krea-2 (K2) "SingleStreamDiT" binds optimized_attention_masked into its OWN
        # module (comfy.ldm.krea2.model), a *different* symbol than the Qwen backend
        # (comfy.ldm.qwen_image.model) even though it shares the Qwen architecture.
        # Patching the qwen symbol (what the README tells users to select) is a silent
        # no-op — we must patch krea2.model instead.
        return [("comfy.ldm.krea2.model", "optimized_attention_masked", True)]
    if model_type in ("zimage", "z_image"):
        return [("comfy.ldm.lumina.model", "optimized_attention_masked", True)]
    if model_type == "anima":
        # Main DiT (MiniTrainDIT, imported from cosmos.predict2) binds
        # ``optimized_attention``.  A secondary attention path in anima/model.py uses
        # raw ``F.scaled_dot_product_attention`` and cannot be intercepted without a
        # process-global torch patch, so SPA covers the DiT path only in v1.
        return [("comfy.ldm.cosmos.predict2", "optimized_attention", False)]
    # Fallback (auto that didn't resolve to a known DiT type).
    return [("comfy.ldm.modules.attention", "optimized_attention", False)]


def _spa_resolve_type(model_type: str, dm) -> str:
    """Resolve the concrete SPA backend key from the requested ``model_type``
    and the live diffusion model ``dm``.

    W4.3 (2026-08-25): thin adapter over the canonical
    :func:`src.model_detect.resolve_model_type` — SPA no longer owns its own
    detector copy.  Kept under this name/signature because many tests import
    it directly.
    """
    from .model_detect import resolve_model_type

    return resolve_model_type(dm, model_type)


def _spa_run_averaged(q, k, v, ctx, attn_fn):
    """Run ``N`` averaged attention passes over the bundled RoPE variants (HRDiT).

    Emits a one-time-per-forward INFO log (guarded by ``ctx._spa_logged``) so a real
    ComfyUI run can confirm the averaged-attention hook actually fired for the active
    backend AND see the concrete bundle parameters that produced the output.  This is
    the decisive integration discriminator for model-specific SPA artifacts (e.g. the
    Krea-2 "doubled / mosaic-glass" report): if this line is ABSENT, SPA is a silent
    no-op for that model; if it is present with an unexpected variant count, the bug is
    in the variant/slide math, not the model.
    """
    fmt = ctx.fmt
    if ctx.pre_roped:
        # P3 (D5 fix): consume the cached composed deltas when the embedder
        # registered them (composed once per grid).  Fall back to per-call
        # composition when they are absent or inconsistent (e.g. a synthetic
        # SPAContext built directly in tests).
        deltas = getattr(ctx, "variant_deltas", None)
        if deltas is not None and len(deltas) == len(ctx.variant_pes):
            rotations = list(deltas)
        else:
            inv_base = inv_rope(ctx.base_pe, fmt)
            rotations = [compose_rope(inv_base, vp, fmt) for vp in ctx.variant_pes]
    else:
        rotations = list(ctx.variant_pes)
    if not getattr(ctx, "_spa_logged", False):
        ctx._spa_logged = True
        n_variants = len(rotations)
        # s (bundle size in tokens) is recoverable from the variant count: 2*s-1 == n.
        s_est = (n_variants + 1) // 2
        # DEBUG (not INFO): diagnostic detail for developers; common users do not
        # need this line in their console.
        logger.debug(
            "SPA averaged-attention ACTIVE: backend=%s fmt=%s bundle_size=%s -> "
            "%d variant passes (s~=%d tokens/bundle). If this line is missing, SPA is a "
            "no-op for this model; a mosaic/ripple means the variant math, not the model.",
            getattr(ctx, "model_key", "?"), fmt, getattr(ctx, "bundle_size", "?"),
            n_variants, s_est,
        )
    return spa_averaged_attention(q, k, v, None, rotations, attn_fn=attn_fn,
                                 pre_roped=False, fmt=fmt)


def _spa_assemble_zimage_posids(pending):
    """Reassemble the FULL-sequence position ids from lumina's per-group calls.

    ``pending`` is the list of ``(kind, pos_ids)`` entries accumulated by the Z-Image
    embedder (``kind`` in ``{"cap", "pos"}``).  lumina concatenates token groups
    GROUP-MAJOR (all caption tokens, then all siglip, then all image) inside
    ``patchify_and_embed``.  Each ``embed_all`` call emits exactly one ``cap`` group
    followed by its ``pos`` group(s) (``[siglip, image]`` when omni/ref, else just
    ``[image]``), and a ``cap`` call always starts a new ``embed_all`` segment.  We
    therefore split ``pending`` into segments at every ``cap`` entry and, per segment,
    route ``pos[0]`` to siglip and ``pos[-1]`` to image — reproducing lumina's
    group-major order exactly (including ref/siglip cases) so the assembled length
    matches the concatenated attention ``q``.
    """
    segments = []
    cur = None
    for kind, p in pending:
        if kind == "cap":
            if cur is not None:
                segments.append(cur)
            cur = {"cap": p, "pos": []}
        else:
            if cur is None:
                cur = {"cap": None, "pos": []}
            cur["pos"].append(p)
    if cur is not None:
        segments.append(cur)

    caps = [s["cap"] for s in segments if s["cap"] is not None]
    sig_segs = [s for s in segments if len(s["pos"]) >= 2]
    img_segs = [s for s in segments if s["pos"]]
    if not caps and not img_segs:
        return None

    parts = []
    if caps:
        parts.append(torch.cat(caps, dim=1))
    if sig_segs:
        parts.append(torch.cat([s["pos"][0] for s in sig_segs], dim=1))
    if img_segs:
        parts.append(
            torch.cat(
                [(s["pos"][0] if len(s["pos"]) == 1 else s["pos"][1]) for s in img_segs],
                dim=1,
            )
        )
    return torch.cat(parts, dim=1)


def parse_layer_filter(spec: str):
    """Parse a per-layer SPA filter spec into a frozenset of layer indices.

    Plan P8/T8.1 (G5).  The reference HRDiT ``set_spa_filter(double_ids,
    single_ids)`` selects WHICH transformer blocks run SPA; our hook is
    module-level (no per-block identity), so the filter is expressed over the
    FLAT per-forward attention-call counter (the same index HAP uses):

    * ``""`` / ``None``      -> ``None`` (SPA allowed on EVERY layer),
    * ``"3"``                -> ``frozenset({3})``,
    * ``"0-18,38-57"``       -> union of both inclusive ranges,
    * whitespace around tokens is tolerated (``"0-18, 38-57"``).

    Invalid specs (reversed range, non-integer token, negative index, empty
    range part) raise :class:`ValueError` naming the offending token.  The
    result is deduplicated and sorted (frozenset).

    IDEMPOTENT (2026-08-24): an already-parsed ``frozenset`` passes through
    unchanged, so callers may pre-parse (the SPADyPE node parses first to
    scope the "invalid spa_layer_filter" error prefix to filter failures only)
    and :func:`apply_spa_to_model` re-parses safely.
    """
    if spec is None:
        return None
    if isinstance(spec, frozenset):
        return spec
    spec = str(spec).strip()
    if not spec:
        return None
    layers = set()
    for raw_token in spec.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError(
                f"spa_layer_filter: empty token in {spec!r} (dangling comma?)"
            )
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(
                    f"spa_layer_filter: invalid range {token!r} in {spec!r}"
                )
            lo_s, hi_s = parts[0].strip(), parts[1].strip()
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError:
                raise ValueError(
                    f"spa_layer_filter: non-integer range {token!r} in {spec!r}"
                ) from None
            if lo < 0 or hi < 0:
                raise ValueError(
                    f"spa_layer_filter: negative index in range {token!r}"
                )
            if lo > hi:
                raise ValueError(
                    f"spa_layer_filter: reversed range {token!r} (lo > hi)"
                )
            layers.update(range(lo, hi + 1))
        else:
            try:
                idx = int(token)
            except ValueError:
                raise ValueError(
                    f"spa_layer_filter: non-integer token {token!r} in {spec!r}"
                ) from None
            if idx < 0:
                raise ValueError(
                    f"spa_layer_filter: negative index {token!r}"
                )
            layers.add(idx)
    return frozenset(layers)


def _spa_derive_text_len(ids: torch.Tensor):
    """Derive the number of leading TEXT tokens from position ids (plan P3/T3.4).

    FLUX/Qwen/Krea-2 convention: text tokens have ``row == col == 0`` (axes 1 and
    2 of the ``(B, L, 3)`` ids) and come FIRST in the sequence.  We count the
    leading CONTIGUOUS run of such tokens (not the total) so the image pixel at
    grid position (0, 0) — which also has row == col == 0 — is not miscounted as
    text.  Returns the count from the first batch element (uniform across the
    batch), or ``None`` when the ids cannot carry the info (too few axes /
    empty).  Used by HAP to place the text/image boundary of the band mask
    without relying solely on the node-provided default.
    """
    if ids is None:
        return None
    try:
        if ids.dim() < 2 or ids.shape[-1] < 3:
            return None
        flat = ids.reshape(-1, ids.shape[-2], ids.shape[-1]) if ids.dim() > 2 else ids.unsqueeze(0)
        first = flat[0]  # (L, 3)
        is_text = (first[..., 1] == 0) & (first[..., 2] == 0)
        # Length of the leading contiguous run of True.
        run = 0
        for flag in is_text.tolist():
            if not flag:
                break
            run += 1
        return run
    except Exception:  # leak-guard: diagnostic run-length read
        return None


def _spa_dispatch_attention(q, k, v, ctx, _attn, fmt):
    """Run the averaged-attention hook, selecting the Z-Image vs single-group path.

    Z-Image / Lumina accumulates per-group position ids in ``ctx.pending`` (because a
    single ``JointAttention`` runs over the FULL concatenated sequence).  We rebuild the
    FULL-sequence variant PE and verify its length equals ``q``'s seq dim before using
    it; if they disagree (e.g. a caption/siglip-only refinement regime whose ``q`` is
    shorter than the full sequence) we fall back to plain attention so SPA never raises
    on a shape mismatch.  Every other backend uses the single-group ``ctx.variant_pes``.

    DIAGNOSTICS (D3): each path emits a one-time-per-forward log so a real ComfyUI run
    can confirm WHICH path fired.  The pending (Z-Image) path previously emitted
    NOTHING, so a Z-Image run showing only the single-group "SPA averaged-attention
    ACTIVE" line meant the pending path never engaged (or the model was not detected
    as zimage).  These logs make that decisive.
    """
    if getattr(ctx, "uses_pending", False) and ctx.pending:
        full = _spa_assemble_zimage_posids(ctx.pending)
        if full is not None and ctx.embedder is not None and full.shape[1] == q.shape[-2]:
            if not getattr(ctx, "_spa_pending_logged", False):
                ctx._spa_pending_logged = True
                logger.info(
                    "SPA dispatch: PENDING path (zimage multi-group) ACTIVE: "
                    "full_seq_len=%d groups=%d. Full-sequence variant PE assembled.",
                    full.shape[1], len(ctx.pending),
                )
            # Delegate to spa_averaged_attention (which averages the N attention
            # OUTPUTS in spa_attn.py, never a RoPE matrix) so spa.py stays free of any
            # tensor averaging (the no-legacy-averaging guard).
            # P3 (D5 fix): consume the CACHED composed deltas (composed once per
            # unique full grid) and run the passes with pre_roped=False so the
            # per-call inv_rope/compose_rope recomposition is eliminated.
            base_pe, variant_pes = ctx.embedder._compute_fullseq_pe(full)
            deltas = ctx.embedder._compute_fullseq_deltas(full)
            if deltas is not None and len(deltas) == len(variant_pes):
                return spa_averaged_attention(q, k, v, None, deltas,
                                              attn_fn=_attn, pre_roped=False, fmt=fmt)
            return spa_averaged_attention(q, k, v, base_pe, variant_pes,
                                          attn_fn=_attn, pre_roped=True, fmt=fmt)
        # Length mismatch -> SPA cannot apply to this regime; run plain attention.
        if not getattr(ctx, "_spa_mismatch_logged", False):
            ctx._spa_mismatch_logged = True
            logger.debug(
                "SPA dispatch: PENDING path length mismatch (assembled=%s q_seq=%d) "
                "-> plain attention for this regime.",
                None if full is None else full.shape[1], q.shape[-2],
            )
        return _attn(q, k, v)
    logger.debug(
        "SPA dispatch: SINGLE-GROUP path (uses_pending=%s, pending_len=%d).",
        getattr(ctx, "uses_pending", False), len(ctx.pending) if ctx.pending else 0,
    )
    return _spa_run_averaged(q, k, v, ctx, _attn)


def _hrdit_resolve_text_len(hap_ctx, seq_len):
    """Resolve the effective text/image boundary for the HAP mask (plan P3/T3.4).

    Priority: the live SPA context's derived ``text_len`` (from the position
    ids) when available, else the node-provided ``HapContext.text_len``.  The
    result is clamped to ``[0, seq_len]`` so a degenerate/oversized default can
    never index past the sequence.
    """
    spa_ctx = get_spa_context()
    text_len = None
    if spa_ctx is not None and getattr(spa_ctx, "text_len", None) is not None:
        text_len = int(spa_ctx.text_len)
    if text_len is None:
        text_len = int(getattr(hap_ctx, "text_len", 0) or 0)
    return max(0, min(text_len, seq_len))


def _hrdit_hap_dispatch(q, k, v, layer_idx, mask):
    """Route one attention pass through the HAP kernel when HAP is live.

    Returns the masked-attention output, or ``None`` to fall back to the
    original attention.  Plan §2.1 decision matrix (HAP column).  v1 composes
    HAP only when there is NO external attention mask to combine with (the masked
    backend convention passes a ``mask`` argument; HAP's own block-sparse mask
    cannot be composed with it yet, so those calls keep the plain path).
    """
    hap_ctx = get_hap_context()
    if hap_ctx is None or not hap_ctx.active:
        return None
    if mask is not None:
        return None
    from .hap import HapRuntime

    text_len = _hrdit_resolve_text_len(hap_ctx, q.shape[-2])
    return HapRuntime.get().attn(q, k, v, layer_idx, ctx=hap_ctx, text_len=text_len)


def _hrdit_proportional_ratio(q) -> float:
    """Proportional attention scaling ratio for this call (plan P7/T7.2).

    Computed from ``q.shape[-2]`` — the true sequence length INCLUDING text
    tokens.  The reference uses ``key.size(2)`` which is the full concatenated
    seq; identical here because q and k share the sequence dimension.  A scalar
    factor on ``q`` commutes with RoPE and any attention mask, so pre-scaling
    ``q`` once applies uniformly to every downstream pass (SPA variants
    included) without touching backend internals.
    """
    from .hap import proportional_scale_ratio

    return proportional_scale_ratio(int(q.shape[-2]))


def _spa_layer_allowed(layer_idx: int) -> bool:
    """Per-layer SPA filter gate (plan P8/T8.2, G5).

    ``None`` filter (default) == SPA allowed on EVERY layer.  A frozenset of
    flat layer indices restricts the averaged-pass SPA to those layers only;
    filtered-out layers run plain attention.  The filter gates SPA ALONE — the
    layer counter still advances (alignment is sacred) and HAP dispatch is NOT
    affected (reference semantics: ``spa_allowed`` affects SPA only).
    """
    f = get_spa_layer_filter()
    return f is None or layer_idx in f


def _make_hrdit_wrapper(orig, is_masked: bool):
    """Build the unified ``optimized_attention`` / ``optimized_attention_masked`` wrapper.

    Plan P3/T3.1+T3.2, signature fix 2026-08-16 (G1).  The wrapper is a
    **signature-compatible shim** of the real ComfyUI attention function.  Every
    real backend in this build shares ONE signature
    (``comfy/ldm/modules/attention.py::attention_pytorch``)::

        (q, k, v, heads, mask=None, attn_precision=None,
         skip_reshape=False, skip_output_reshape=False, **kwargs)

    and ``optimized_attention_masked = optimized_attention`` (same function).  So
    BOTH the masked and unmasked symbols are wrapped with this single body whose
    positional slots 5-8 are EXACTLY ``mask``, ``attn_precision``, ``skip_reshape``,
    ``skip_output_reshape``.  ``transformer_options`` (and any other extra kwarg
    such as ``enable_gqa`` / ``scale`` / ``low_precision_attention``) rides
    ``**kw`` and is forwarded to ``orig`` untouched.

    The pre-fix wrapper inverted these slots for the unmasked variant (feeding the
    ``skip_reshape`` bool into the real ``mask`` parameter -> ``mask.ndim`` ->
    ``AttributeError`` on Anima) and mis-forwarded them for the masked variant.
    Mirroring the real signature bit-for-bit removes the whole class of bug.

    Behaviour (unchanged from P3):

    - advances the per-forward HRDiT layer counter (:func:`next_hrdit_layer_idx`)
      on EVERY call — including all early-return paths — so the counter stays
      aligned with the model's block order (alignment is sacred), and
    - dispatches each attention pass through the HAP kernel when a HAP context
      is live (plan §2.1).  With SPA active, EVERY variant pass runs through the
      kernel (reference composition); with SPA off, a single kernel pass runs.
    - consults the per-layer SPA filter (plan P8/T8.2): when a filter is set and
      the current layer index is NOT in it, SPA is skipped for that layer (plain
      attention) while the counter and HAP dispatch are unaffected.

    ``is_masked`` is retained for patch-target bookkeeping and tests but no longer
    changes the call convention (the two real symbols are the same function).

    W9.e (2026-08-25): the temporary shape-diagnostic latch from the 2026-08-18
    krea2 investigation is RETIRED — its hypothesis was confirmed and fixed
    (see the output-reshape comment below); behaviour is covered by
    tests/test_anima_crash_rootcause.py + krea2 integration tests.
    """

    def _wrapper(q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kw):
        layer_idx = next_hrdit_layer_idx()
        # HAP PLAN-LAYER ORDINAL (2026-08-19, runtime layer-index mismatch fix).
        # Calibration enumerates scope-plan layers by the DOMINANT-HEAD-ONLY
        # ordinal: its heterogeneous-head-count filter drops auxiliary attention
        # with other head counts (Krea2's 4 x 20-head projector calls) and numbers
        # the remaining dominant-head calls 0..N-1.  The runtime must index the
        # plan by the SAME ordinal — NOT the raw all-call counter.  The pre-fix
        # wrapper fed the raw counter into the plan, so the 4 aux calls consumed
        # indices 0-3, shifting every main block by 4 and pushing the last 4 main
        # blocks past the plan ("layer 28 exceeds the scope plan").
        #
        # Advance the ordinal ONCE per wrapper call that the plan actually covers
        # (square + unmasked + non-GQA + head-count match — exactly calibration's
        # recorded-and-kept set).  Non-covered calls (cross-attention, aux heads,
        # masked, GQA) do NOT consume a plan slot; they decline to plain attention
        # via ``HapRuntime.attn``'s guards.  Computed HERE (not per ``_attn`` pass)
        # so all SPA variant passes of one layer share ONE ordinal — mirroring
        # calibration's grouping of variant passes by layer key.
        _hap_ctx_peek = get_hap_context()
        if (
            _hap_ctx_peek is not None
            and _hap_ctx_peek.active
            and _hap_ctx_peek.plan is not None
            and mask is None
            and q.shape[-2] == k.shape[-2]      # square (self/joint attention)
            and not kw.get("enable_gqa", False)  # calibration skips GQA calls
            and k.shape == q.shape              # non-GQA (calibration skips GQA)
            and q.shape[1] == _hap_ctx_peek.plan.num_heads  # dominant head count
        ):
            hap_plan_layer = next_hap_layer_idx()
        else:
            hap_plan_layer = layer_idx  # non-covered; declines via attn guards
        # Proportional attention scaling (plan P7/T7.2): pre-scale q so the
        # logits gain the factor log(seq_len, train_seq_len).  A scalar on q
        # commutes with RoPE and any mask, so every downstream pass (SPA
        # variants included) sees the scaled q without backend changes.
        if get_hrdit_proportional():
            q = q * _hrdit_proportional_ratio(q)
        ctx = get_spa_context()
        spa_active = (
            get_spa_step_gate()
            and ctx is not None
            and ctx.active
            and len(ctx.variant_pes) > 1
            and _spa_layer_allowed(layer_idx)
        )

        # NON-SQUARE GUARD (2026-08-16, Anima cross-attention crash): the
        # averaged passes apply the registered spatial RoPE rotations to BOTH q
        # and k (``spa_averaged_attention``), which is only valid when q and k
        # span the SAME sequence (self-attention over the spatial grid).  Anima
        # (cosmos.predict2) runs cross-attention — image queries (T*H*W tokens)
        # against text/context keys — through the SAME patched
        # ``optimized_attention`` symbol, so q_len != k_len there and the einsum
        # broadcast crashed (``subscript l has size 512 for operand 1 ...
        # previously seen size 6300``).  Decline SPA for non-square calls and
        # run plain attention — the exact SPA analogue of the HAP non-square
        # guard (``HapRuntime.attn``).  The layer counter already advanced above
        # (alignment is sacred) and HAP dispatch keeps its own guard.  FLUX /
        # Qwen / Krea-2 / Z-Image are unaffected: their attention is joint
        # text+image (always square).
        if spa_active and q.shape[-2] != k.shape[-2]:
            if not getattr(ctx, "_spa_nonsquare_logged", False):
                ctx._spa_nonsquare_logged = True
                logger.debug(
                    "SPA: non-square attention (q_len=%d k_len=%d, e.g. "
                    "cross-attention) cannot use the spatial RoPE variants; "
                    "using plain attention for these calls.",
                    q.shape[-2], k.shape[-2],
                )
            spa_active = False

        def _attn(qq, kk, vv):
            # Use the HAP PLAN-LAYER ORDINAL (dominant-head-only, mirroring
            # calibration) — NOT the raw all-call counter — so auxiliary
            # attention with other head counts never shifts the plan indexing.
            out = _hrdit_hap_dispatch(qq, kk, vv, hap_plan_layer, mask)
            if out is not None:
                # OUTPUT-RESHAPE CONVENTION FIX (2026-08-18, krea2 inference
                # crash ``dim 3: 128 vs 6144``).  ``HapRuntime.attn`` returns
                # head format ``(B, H, T, D)`` (SDPA / FlexAttention layout),
                # but the caller's convention — set by ``skip_output_reshape``
                # — may require FLATTENED ``(B, T, H*D)``.  Krea2 calls
                # ``optimized_attention_masked(..., skip_reshape=True)`` WITHOUT
                # ``skip_output_reshape`` (defaults False) and then does
                # ``out * F.sigmoid(gate)`` where ``gate`` is ``(B, T, H*D)`` —
                # so it NEEDS the flattened layout.  The pre-fix wrapper returned
                # the head-format output as-is, crashing the elementwise multiply.
                #
                # Mirror the calibration path EXACTLY (hap_calib_node.py:919-923):
                #   skip_output_reshape=True  -> head format (B, H, T, D) as-is
                #   skip_output_reshape=False -> flatten to (B, T, H*D)
                # This fixes both the direct HAP path and the SPA-averaged path
                # (which averages ``_attn`` results, so each pass is reshaped
                # before averaging — shape-invariant mean).
                if skip_output_reshape:
                    return out  # (B, H, T, D) head format, as the caller wants
                b, h, t, d = out.shape
                return out.permute(0, 2, 1, 3).reshape(b, t, h * d)
            # Forward to the ORIGINAL attention with the REAL positional
            # convention: slots 5-8 == mask, attn_precision, skip_reshape,
            # skip_output_reshape.  Everything else (transformer_options, ...)
            # rides **kw exactly as the backend sent it.
            return orig(qq, kk, vv, heads, mask, attn_precision, skip_reshape,
                        skip_output_reshape, **kw)

        if not spa_active:
            # SPA off / gated: single pass (still HAP-routed when live).
            return _attn(q, k, v)
        return _spa_dispatch_attention(q, k, v, ctx, _attn, ctx.fmt)

    return _wrapper


# Backward-compatible alias (the pre-P3 name).
_make_spa_wrapper = _make_hrdit_wrapper


# ---------------------------------------------------------------------------
# Clone-state carry-over (plan 2026-08-16 G4)
# ---------------------------------------------------------------------------

#: HRDiT-private patcher attributes that MUST survive ``ModelPatcher.clone()``.
#: The real ``clone()`` copies only its KNOWN fields, so chaining SPA->HAP or
#: HAP->SPA would silently drop the other node's state (making node order decide
#: whether HAP/SPA is live).  :func:`_hrdit_carry_state` re-applies them to the
#: clone right after ``clone()``.  NOTE: ``_hrdit_proportional_attention`` is NOT
#: in this tuple — it has OR-semantics (either node may enable it) and is handled
#: explicitly by the apply functions.
_HRDIT_PATCHER_ATTRS = (
    "_spa_installed",
    "_hrdit_consumers",
    "_spa_orig_optimized_attention",
    "_hap_ctx",
    "_hap_plan",
    "_spa_steps",
    "_spa_start_sigma",
    "_spa_step_counter",
    "_spa_last_sigma",
    "_spa_layer_filter",
    # State indirection (see :func:`_hrdit_carry_state`): a 1-element list
    # pointing at the current AUTHORITATIVE patcher.  The shared unet wrapper's
    # closure captures the install-time patcher, so it resolves live state via
    # this ref; carry-over re-points it to the newest clone.
    "_hrdit_state_ref",
)


def _hrdit_carry_state(src, dst) -> None:
    """Copy every HRDiT-private patcher attr from ``src`` to ``dst`` (plan G4).

    Called immediately after ``model.clone()`` in BOTH
    :func:`apply_spa_to_model` and :func:`apply_hap_to_model` (before any early
    return) so node order never decides whether SPA/HAP state is live:

    * HAP -> SPA: ``_hap_ctx`` / ``_hap_plan`` survive -> HAP stays live under
      the SPA wrapper (the exact scenario that crashed pre-fix).
    * SPA -> HAP: ``_spa_steps`` / ``_spa_layer_filter`` / gates survive -> SPA
      keeps its configured gating; ``_spa_installed`` survives -> the shared
      install fast path fires (consumer added, no re-install attempt).

    Only attrs actually present on ``src`` are copied; bare objects (no HRDiT
    state) are a no-op and never raise.

    STATE REF RE-POINTING: ``_hrdit_state_ref`` is a 1-element list shared by
    reference between ``src`` and ``dst``.  After copying, we re-point its single
    slot to ``dst`` so the shared unet wrapper (whose closure captured the
    install-time patcher) reads/writes state on the NEWEST authoritative clone.
    Without this, a SPA->HAP chain would leave the wrapper reading the old SPA
    patcher's ``_hap_ctx`` (``None``) and HAP would be silently inactive — node
    order would still decide behaviour.  Re-pointing makes the chain fully
    order-independent for linear workflows (the common case).
    """
    for attr in _HRDIT_PATCHER_ATTRS:
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))
    ref = getattr(dst, "_hrdit_state_ref", None)
    if ref is not None:
        ref[0] = dst


def _spa_restore_installed(m) -> None:
    """Undo a previous install recorded on ``m._spa_installed`` (idempotency)."""
    installed = getattr(m, "_spa_installed", None)
    if not installed:
        return
    for mod, attr, orig in installed:
        if getattr(mod, attr, None) is not orig:
            setattr(mod, attr, orig)
    m._spa_installed = None
    m._hrdit_consumers = None


def _hrdit_uninstall_hook(m, consumer: str) -> None:
    """Remove one consumer from the shared hook; restore only when the last leaves.

    Plan P3/T3.3.  SPA and HAP share ONE wrapper; uninstalling one while the
    other is still active must keep the wrapper in place.  Only when the
    consumer set empties do we restore the original attention symbols and drop
    the unet wrapper.
    """
    consumers = getattr(m, "_hrdit_consumers", None)
    if consumers is None:
        # No shared-hook bookkeeping (legacy single-consumer install) -> restore.
        _spa_restore_installed(m)
        if hasattr(m, "set_model_unet_function_wrapper"):
            m.set_model_unet_function_wrapper(None)
        return
    consumers.discard(consumer)
    if consumers:
        return  # another consumer still needs the wrapper
    _spa_restore_installed(m)
    if hasattr(m, "set_model_unet_function_wrapper"):
        m.set_model_unet_function_wrapper(None)


def _spa_install_hook(m, model_type: str) -> None:
    """Install the averaged-attention hook for SPA (backward-compatible entry).

    Delegates to :func:`_hrdit_install_hook` with ``consumer="spa"`` (plan
    P3/T3.3 install-policy generalization).
    """
    _hrdit_install_hook(m, model_type, consumer="spa")


def _hrdit_install_hook(m, model_type: str, consumer: str = "spa") -> None:
    """Install the unified HRDiT attention hook for the active backend.

    Plan P3/T3.3 (ref-counted shared hook): the hook is installed when SPA is
    active (``enable_spa`` and ``bundle_size > 1``) OR HAP is enabled.  Both
    consumers share ONE wrapper: applying SPA then HAP (any order) installs once
    and merely records the second consumer in ``m._hrdit_consumers``; the
    wrapper is restored only when the LAST consumer uninstalls
    (:func:`_hrdit_uninstall_hook`).

    The hook patches the backend's *bound* ``optimized_attention`` symbol (see
    :func:`_spa_patch_targets`) — NOT only the module attribute — so it actually
    fires for FLUX/Qwen/Z-Image/Anima/Krea-2.  It also patches
    ``comfy.ldm.modules.attention`` for classic CrossAttention blocks.  The unet
    wrapper clears the :class:`SPAContext` / HAP context before/after a forward so
    they never leak across models, and resets the HRDiT layer counter.
    """
    import importlib


    # Shared-hook fast path: already installed -> just record the consumer.
    # The unet wrapper reads ``m._hap_ctx`` at CALL time, so HAP state applied
    # after a SPA install is still honoured by the same wrapper.
    if getattr(m, "_spa_installed", None):
        consumers = getattr(m, "_hrdit_consumers", None)
        if consumers is None:
            # Legacy install (pre-T3.3) was always SPA-only.
            consumers = {"spa"}
            m._hrdit_consumers = consumers
        consumers.add(consumer)
        return

    targets = list(_spa_patch_targets(model_type))
    # Always also patch the module-global for classic CrossAttention blocks
    # (SD1.5/SDXL/autoencoders) and any code that does ``attn_mod.optimized_attention(...)``.
    global_target = ("comfy.ldm.modules.attention", "optimized_attention", False)
    if global_target not in targets:
        targets.append(global_target)

    installed = []
    mod_global_orig = None
    for mod_path, attr, is_masked in targets:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as exc:  # backend not importable in this environment  # probe: backend module not importable here
            logger.warning(
                "SPA: cannot import %r to patch attention; SPA may be a no-op for "
                "backend %r. (%s)", mod_path, model_type, exc
            )
            continue
        orig = getattr(mod, attr, None)
        if orig is None or getattr(orig, "_spa_wrapper", False):
            continue
        wrapper = _make_spa_wrapper(orig, is_masked)
        wrapper._spa_wrapper = True
        setattr(mod, attr, wrapper)
        installed.append((mod, attr, orig))
        if mod_path == "comfy.ldm.modules.attention" and attr == "optimized_attention":
            mod_global_orig = orig
        logger.info("SPA: patched %s.%s for backend %r", mod_path, attr, model_type)

    m._spa_installed = installed
    m._spa_orig_optimized_attention = mod_global_orig  # legacy, for T-P3-5
    m._hrdit_consumers = {consumer}  # ref-counted shared-hook consumers (T3.3)
    # STATE REF (plan 2026-08-16 G4): a 1-element list pointing at the current
    # AUTHORITATIVE patcher.  The unet wrapper's closure captures ``m`` (the
    # install-time patcher), but HRDiT state may be carried onto a LATER clone
    # (SPA->HAP / HAP->SPA chaining); :func:`_hrdit_carry_state` re-points this
    # ref so the wrapper always reads/writes the newest clone's state.  A fresh
    # install creates the ref; if one was somehow already carried, re-point it.
    _existing_ref = getattr(m, "_hrdit_state_ref", None)
    if _existing_ref is None:
        m._hrdit_state_ref = [m]
    else:
        _existing_ref[0] = m

    def _spa_unet_wrapper(model_function, args_dict):
        set_spa_context(None)  # clear before forward -> no cross-model leak
        # Reset the per-forward HRDiT layer counter (plan P3/T3.1).  The unified
        # wrapper advances it on EVERY attention call; resetting here keeps the
        # counter aligned with the model's block order on every forward.
        set_hrdit_layer_idx(0)
        # Reset the HAP plan-layer ordinal too (2026-08-19 layer-index fix): it
        # advances only for plan-covered calls, so it must restart per forward
        # exactly like the raw counter.
        set_hap_layer_idx(0)
        # Resolve the AUTHORITATIVE patcher (plan G4): the state ref is
        # re-pointed by _hrdit_carry_state to the newest clone, so a chained
        # SPA->HAP / HAP->SPA workflow reads the combined state, not the stale
        # install-time patcher's.  Fall back to the closure patcher when the ref
        # is absent (legacy installs).
        _ref = getattr(m, "_hrdit_state_ref", None)
        state = _ref[0] if _ref else m
        # Proportional attention scaling (plan P7/T7.2): activate for this forward
        # from the model attr (OR-semantics: either node may enable it).  Read at
        # call time so a later apply_* still takes effect on the shared wrapper.
        set_hrdit_proportional(bool(getattr(state, "_hrdit_proportional_attention", False)))
        # Per-layer SPA filter (plan P8/T8.2): activate for this forward from the
        # model attr (frozenset of flat layer indices, or None == all layers).
        # Read at call time so a later apply_* still takes effect on the shared
        # wrapper.  The filter gates SPA ALONE (counter + HAP unaffected).
        set_spa_layer_filter(getattr(state, "_spa_layer_filter", None))
        # HAP (plan P4/T4.1): activate this model's HapContext for the forward.
        # Read at call time (not install time) so SPA-then-HAP installs share the
        # SAME unet wrapper and HAP state applied later is still honoured.
        hap_ctx = getattr(state, "_hap_ctx", None)
        if hap_ctx is not None:
            set_hap_context(hap_ctx)

        # Read the current sigma ONCE (shared by the step-count gate and the
        # sigma-threshold gate).  NOTE: take the FIRST element, not a reduction --
        # a code-quality guard forbids any tensor averaging in spa.py (the
        # no-legacy-RoPE rule).  The timestep is uniform across the batch, so
        # element 0 is the sigma.
        t = args_dict.get("timestep")
        try:
            sigma = float(t.detach().flatten()[0]) if torch.is_tensor(t) else float(t)
        except Exception:  # degrade: unreadable timestep -> plain attention
            sigma = None  # unreadable timestep

        # STEP-COUNT GATE (P2, HRDiT-faithful fix for D4): SPA is only useful on
        # the LEADING denoising steps (it fixes global position extrapolation
        # established early).  ``state._spa_steps`` is the number of leading steps
        # on which SPA is active; ``0`` = all steps (backward-compatible).  A NEW
        # GENERATION is detected when the incoming sigma jumps UP (or on the first
        # call), which resets the leading-step counter.  This is scheduler-agnostic
        # and deterministic (sigma decreases monotonically within a generation).
        # An unreadable timestep keeps SPA active (safe fallback, no counting).
        # Read/write through ``state`` (the authoritative patcher, plan G4) so a
        # chained SPA->HAP clone keeps its configured gating and step counter.
        spa_steps = int(getattr(state, "_spa_steps", 0) or 0)
        gate_by_steps = True
        if spa_steps > 0 and sigma is not None:
            last_sigma = getattr(state, "_spa_last_sigma", None)
            counter = (
                0
                if (last_sigma is None or sigma > last_sigma)
                else int(getattr(state, "_spa_step_counter", 0))
            )
            gate_by_steps = counter < spa_steps
            state._spa_step_counter = counter + 1
            state._spa_last_sigma = sigma

        # SIGMA-THRESHOLD GATE (D2a, pre-existing): an additional OPTIONAL gate,
        # AND-combined with the step-count gate.  ``spa_start_sigma >= 1.0`` keeps
        # SPA active on every step (backward-compatible default); otherwise SPA
        # runs only while the current sigma is above the threshold.
        start_sigma = float(getattr(state, "_spa_start_sigma", 1.0) or 1.0)
        if start_sigma >= 1.0:
            gate_by_sigma = True
        elif sigma is None:
            gate_by_sigma = True  # unreadable timestep -> keep SPA active
        else:
            gate_by_sigma = sigma > start_sigma

        set_spa_step_gate(gate_by_steps and gate_by_sigma)
        try:
            return model_function(args_dict.get("input"),
                                  args_dict.get("timestep"),
                                  **args_dict.get("c", {}))
        finally:
            set_spa_context(None)  # clear after forward
            set_hap_context(None)  # clear HAP too -> no cross-model leak
            set_hrdit_proportional(False)  # clear proportional flag -> no leak
            set_spa_layer_filter(None)  # clear layer filter -> no cross-model leak
            set_spa_step_gate(True)  # reopen so a non-SPA forward is unaffected
            set_hap_layer_idx(0)  # clear HAP plan ordinal -> no cross-model leak

    m.set_model_unet_function_wrapper(_spa_unet_wrapper)


def restore_spa_attention_hook(m, attn_module=None) -> None:
    """Restore the original ``optimized_attention`` after a SPA un-patch (T-P3-5).

    Consumer-aware (plan P3/T3.3): the hook is SHARED between SPA and HAP.  If
    another consumer (HAP) is still registered, uninstalling SPA merely removes
    the ``"spa"`` consumer and keeps the wrapper; only the LAST consumer triggers
    the full restore.
    """
    consumers = getattr(m, "_hrdit_consumers", None)
    if consumers and "spa" in consumers and len(consumers) > 1:
        consumers.discard("spa")
        return  # HAP (or another consumer) still needs the shared wrapper

    _hrdit_uninstall_hook(m, "spa")
    if attn_module is None:
        try:
            import comfy.ldm.modules.attention as attn_module
        except Exception:  # probe: attention module availability
            attn_module = None
    if attn_module is not None:
        orig = getattr(m, "_spa_orig_optimized_attention", None)
        if orig is not None and getattr(attn_module, "optimized_attention", None) is not orig:
            attn_module.optimized_attention = orig
    m._spa_orig_optimized_attention = None


def apply_spa_to_model(
    model,
    model_type: str,
    width: int,
    height: int,
    method: str = "ntk",
    yarn_alt_scaling: bool = False,
    enable_spa: bool = True,
    bundle_size: Optional[int] = None,
    dype_scale: float = 2.0,
    dype_exponent: float = 2.0,
    base_resolution: int = 1024,
    dype_start_sigma: float = 1.0,
    spa_start_sigma: float = 1.0,
    spa_steps: int = 3,
    proportional_attention: bool = False,
    spa_layer_filter: Optional[str] = None,
):
    """Patch a ComfyUI model with Spatial Position Alignment.

    SPA replaces the model's RoPE embedder with an :class:`SPABasePosEmbed`
    subclass that returns the base (no-extrapolation) RoPE and registers the
    bundled variants, then (when active) installs the averaged-attention hook.

    ``bundle_size`` is the PAPER's ``N`` (tokens per bundle): ``None``/``0`` =
    auto (minimal in-distribution compression), ``1`` = off, ``2..8`` explicit.
    Legacy values ``>= 32`` (old ``group_num`` semantics) are migrated to auto
    with a one-time WARNING (decision M1).

    ``spa_steps`` (P2, HRDiT-faithful speed fix) is the number of LEADING
    denoising steps on which SPA is active; ``0`` = all steps (backward
    compat).  Default ``3`` (HRDiT ``--spa_steps [3, 0]``).  A new generation
    (sigma jump-up) resets the leading-step counter.  It is AND-combined with
    the optional ``spa_start_sigma`` threshold gate.

    ``proportional_attention`` (plan P7/T7.3) enables HRDiT's proportional
    attention scaling: the attention logits gain a factor
    ``sqrt(ln(seq_len) / ln(train_seq_len))`` (pre-scaling ``q``) to
    compensate the entropy dilution of long sequences.  It is an exact no-op
    (ratio ``1.0``) at/below the trained extent (seq 4608 = 1024px FLUX) and
    when disabled (default — bit-identical regression).  OR-semantics: either
    the SPA or the HAP node may enable it; the flag is stored on the patcher
    (``m._hrdit_proportional_attention``) and read by the shared unet wrapper
    at call time.

    ``spa_layer_filter`` (plan P8/T8.3) restricts the averaged-pass SPA to a
    subset of transformer layers, given as a flat layer-index spec string
    (``"0-18,38-57"``; ``""``/``None`` = every layer).  The indices are the
    per-forward attention-call counter values (the same flat index HAP uses).
    Filtered-out layers run plain attention; the counter and HAP dispatch are
    unaffected.  Invalid specs raise :class:`ValueError` (see
    :func:`parse_layer_filter`).

    ``method`` / ``yarn_alt_scaling`` are NO-OPS for SPA: they exist only to
    satisfy the ``DyPEBasePosEmbed`` constructor chain (the SPA embedders
    inherit ``format_components`` from it).  SPA always applies the model's
    native no-extrapolation RoPE (``ntk_factor=1.0``) on the bundled coords
    (HRDiT "nor" RoPE), so the DyPE extrapolation methods never influence the
    output.  The SPA node therefore does not expose a ``method`` input.

    Install policy (decision 3): the averaged-attention hook (the backend-bound
    ``optimized_attention`` symbol + the unet wrapper) is installed ONLY when SPA is
    active (``enable_spa`` and ``bundle_size != 1``).  For ``bundle_size == 1`` (off)
    or ``enable_spa is False`` the embedder is a transparent base-RoPE pass and nothing
    is patched into the attention path.  Any active ``bundle_size`` (including the
    auto/``None`` -> ``0`` default) gets the hook.  The hook patches the DiT
    backend's *bound* ``optimized_attention`` name (FLUX: ``comfy.ldm.flux.math``;
    Qwen/Z-Image: their ``model`` module; Anima: ``comfy.ldm.cosmos.predict2``),
    not merely the ``comfy.ldm.modules.attention`` module attribute, which the
    backends do not look up at call time.

    Nunchaku (decision 4): SPA is unsupported on quantized/fused Nunchaku kernels
    (they bypass ``optimized_attention``), so we log a warning and return the model
    UNCHANGED — never a silent broken path.
    """
    from .models.spa_flux import PosEmbedSPAFlux
    from .models.spa_nunchaku import PosEmbedSPANunchaku
    from .models.spa_qwen import PosEmbedSPAQwen
    from .models.spa_zimage import PosEmbedSPAZImage

    width = _snap_to_multiple(width, 16)
    height = _snap_to_multiple(height, 16)

    # PROPORTIONAL ATTENTION SCALING (plan P7/T7.3): OR-semantics — either the
    # SPA or the HAP node may enable it.  The real ``ModelPatcher.clone()``
    # copies only its KNOWN fields (custom attrs do NOT survive), so the
    # existing flag must be read from the SOURCE patcher BEFORE the clone and
    # re-applied to the clone.  Set right after the clone (before any early
    # return) so the flag survives even when SPA itself is disabled but a hook
    # is (or gets) installed by HAP.  The shared unet wrapper reads it at call
    # time and activates the q pre-scaling for the whole forward.
    prev_proportional = bool(getattr(model, "_hrdit_proportional_attention", False))
    m = model.clone()
    # CLONE-STATE CARRY-OVER (plan 2026-08-16 G4): the real ``ModelPatcher.clone()``
    # drops custom attrs, so a HAP-patched source patcher would lose ``_hap_ctx``
    # here (HAP silently inactive under the SPA wrapper).  Re-apply every
    # HRDiT-private attr from the SOURCE patcher right after the clone, before any
    # early return, so node order (HAP->SPA vs SPA->HAP) never changes behaviour.
    _hrdit_carry_state(model, m)
    m._hrdit_proportional_attention = prev_proportional or bool(proportional_attention)

    dm = m.model.diffusion_model
    detected_type = _spa_resolve_type(model_type, dm)
    is_nunchaku = detected_type == "nunchaku"
    is_qwen = detected_type == "qwen"
    is_z_image = detected_type == "zimage"
    is_anima = detected_type == "anima"
    is_krea2 = detected_type == "krea2"
    logger.info(f"SPA: Detected model type: {detected_type}")

    if bundle_size is None:
        bundle_size = 0  # auto: minimal in-distribution compression
    bundle_size = int(bundle_size)
    # LEGACY KNOB MIGRATION (decision M1, 2026-08-15): values >= 32 were set under
    # the OLD ``group_num`` semantics (target bundles per axis, default 80).  Under
    # the new paper-``N`` semantics (tokens per bundle) they would mean absurd
    # 32+-token bundles (big-patch pixelation), so they are treated as ``auto``
    # with a one-time WARNING instead of silently changing the output.
    if bundle_size >= SPA_LEGACY_KNOB_THRESHOLD:
        if not getattr(apply_spa_to_model, "_spa_legacy_warned", False):
            apply_spa_to_model._spa_legacy_warned = True
            logger.warning(
                "SPA: bundle_size=%d uses the legacy group_num semantics "
                "(target bundles per axis). The knob is now the paper's N "
                "(tokens per bundle: 0=auto, 1=off, 2..8). Treating %d as "
                "'auto' (minimal in-distribution compression).",
                bundle_size, bundle_size,
            )
        bundle_size = 0
    # Semantics (paper §4.1): ``bundle_size`` is N = tokens per bundle.
    #   * N == 1 -> off (passthrough, no hook)
    #   * N == 0 -> auto: minimal compression keeping bundled positions <= 79
    #   * N >= 2 -> honoured, floored by the in-dist minimum (never OOD)
    # While the grid is inside the trained extent, SPA is an identity no-op
    # (derive_bundle_s).  The pass count is capped by SPA_MAX_PASSES.

    # --- resolve target patch path + original embedder ----------------------
    if is_nunchaku:
        target_patch_path = "diffusion_model.model.pos_embed"
    elif is_z_image:
        target_patch_path = "diffusion_model.rope_embedder"
    elif is_krea2:
        target_patch_path = "diffusion_model.pe_embedder"
    elif is_anima:
        target_patch_path = "diffusion_model.pos_embedder"
    else:
        target_patch_path = "diffusion_model.pe_embedder"

    try:
        # NOTE: ``add_object_patch`` resolves ``target_patch_path`` relative to
        # ``m.model`` (see ``ModelPatcher.get_model_object`` -> ``get_attr(self.model, name)``),
        # so the original embedder must be read from ``m.model`` too — NOT ``m`` directly.
        orig_embedder = _get_attr_by_path(m.model, target_patch_path)
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX/Qwen model structure.")

    # --- composition guard (decision 6) -------------------------------------
    _spa_ensure_no_incompatible_embedder(orig_embedder)

    # --- W9.g (NTH-108): genuine double-application warning ------------------
    # A re-apply onto an ALREADY-SPA embedder resets the step counters and
    # double-registers variants.  Distinguish it from the legitimate
    # SPA-after-HAP chain: in that case the SOURCE patcher carried the hook
    # (``_spa_installed``) but the embedder is NOT an SPA embedder (HAP does
    # not replace the embedder).  Only when BOTH the incoming patcher had the
    # hook AND the resolved embedder is already an SPA embedder is this a real
    # double apply.
    if isinstance(orig_embedder, SPABasePosEmbed) and getattr(
        model, "_spa_installed", False
    ):
        logger.warning(
            "SPA applied twice onto the same model; step counters reset. "
            "This usually means two SPA nodes are chained — remove one."
        )

    # --- Nunchaku guard (decision 4): unsupported, return unchanged ----------
    if is_nunchaku:
        logger.warning(
            "SPA disabled: Nunchaku is not supported for SPA in v1 "
            "(fused/quantized attention kernels bypass the SPA hook). "
            "Use a non-quantized FLUX/Qwen loader."
        )
        return m

    # --- read theta / axes_dim from the original embedder -------------------
    # (Z-Image axes_lens probing was removed: the values were never consumed
    # downstream — the Z-Image adapter derives its base grid itself.)

    try:
        if is_anima:
            theta_base = 10000.0
            dm = m.model.diffusion_model
            head_dim = dm.model_channels // dm.num_heads
            dim_h = head_dim // 6 * 2
            dim_t = head_dim - 2 * dim_h
            dim_w = dim_h
            axes_dim = [dim_t, dim_h, dim_w]
            h_extrap = getattr(dm, "rope_h_extrapolation_ratio", 1.0)
            w_extrap = getattr(dm, "rope_w_extrapolation_ratio", 1.0)
            t_extrap = getattr(dm, "rope_t_extrapolation_ratio", 1.0)
            t_ntk = t_extrap ** (dim_t / (dim_t - 2))
            h_ntk = h_extrap ** (dim_h / (dim_h - 2))
            w_ntk = w_extrap ** (dim_w / (dim_w - 2))
            theta = [theta_base * t_ntk, theta_base * h_ntk, theta_base * w_ntk]
        else:
            theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX/Qwen model structure.")

    spa_cls = PosEmbedSPAFlux
    if is_nunchaku:
        spa_cls = PosEmbedSPANunchaku
    elif is_qwen:
        spa_cls = PosEmbedSPAQwen
    elif is_z_image:
        spa_cls = PosEmbedSPAZImage
    elif is_krea2:
        # Krea-2's pe_embedder is a flux-style EmbedND, so its RoPE format and PE
        # layout are identical to FLUX (PosEmbedSPAFlux); only the attention patch
        # target differs (see _spa_patch_targets).
        spa_cls = PosEmbedSPAFlux
    elif is_anima:
        from .models.spa_anima import PosEmbedSPAAnima
        spa_cls = PosEmbedSPAAnima

    # SPA computes the base (no-extrapolation) RoPE on bundled coords, so it
    # does not use a DyPE-style base patch grid / scale hint.
    embedder_base_patches = None

    new_embedder = spa_cls(
        theta, axes_dim, method, yarn_alt_scaling,
        enable_spa,  # 5th positional -> DyPEBasePosEmbed.dype (unused by SPA math)
        dype_scale, dype_exponent, base_resolution, dype_start_sigma,
        embedder_base_patches,
        enable_spa=enable_spa, bundle_size=bundle_size,
    )

    # Z-Image / Lumina calls rope_embedder once per token group but runs a single
    # JointAttention over the full concatenated sequence; mark the embedder so its
    # _register_variants accumulates per-group position ids for the hook (see
    # _spa_assemble_zimage_posids / _spa_dispatch_attention).
    if is_z_image:
        new_embedder._spa_is_zimage = True

    # --- install policy (decision 3) ----------------------------------------
    # STEP GATING (D2a): store the sigma threshold on the patcher; the unet
    # wrapper reads it and closes the SPA gate on late denoising steps
    # (HRDiT applies SPA only on the LEADING steps).  1.0 = always active.
    m._spa_start_sigma = float(spa_start_sigma)
    # STEP-COUNT GATE (P2): number of LEADING denoising steps with SPA active;
    # 0 = all steps (backward compat).  Default 3 (HRDiT).  The unet wrapper
    # counts forwards, resets on a sigma jump-up (new generation), and closes
    # the gate once the counter reaches this value.
    m._spa_steps = max(0, int(spa_steps))
    m._spa_step_counter = 0
    m._spa_last_sigma = None
    # PER-LAYER SPA FILTER (plan P8/T8.3): parse the flat layer-index spec and
    # store the frozenset (or None == every layer) on the patcher; the unet
    # wrapper activates it around each forward and the attention wrapper skips
    # SPA on filtered-out layers (counter + HAP unaffected).  Invalid specs
    # raise ValueError here (clear node error, plan T8.3).
    m._spa_layer_filter = parse_layer_filter(spa_layer_filter)
    if enable_spa and bundle_size != 1:
        _spa_install_hook(m, detected_type)

    m.add_object_patch(target_patch_path, new_embedder)
    return m
