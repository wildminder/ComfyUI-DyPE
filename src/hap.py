"""
HAP (Head-Adaptive attention Pruning) runtime for HRDiT.

Implements the speed half of HRDiT (paper 2608.07003): per-head block-sparse
attention scopes derived from an offline-calibrated scope plan.

Reference implementation: ``.dev/data/HRDit/HRDiT/hrdit/hap.py``.
Theory: ``.dev/data/HRDit/HRDiT-theory-code.md`` sections 6-8.

This module is built incrementally per
``docs/plans/2026-08-15-hrdit-full-hap-implementation.md``:

- P0 (this file, initial): constants + FlexAttention availability probe.
- P1: ``ScopePlan`` + band-mask math + dense mask oracle + cost model.
- P2: ``HapContext`` + dense/flex backends + ``HapRuntime`` facade.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import List, Sequence, Union

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (reference: hrdit/hap.py)
# ---------------------------------------------------------------------------

#: Attention block granularity in tokens (FlexAttention block size).
HAP_BLOCK = 64

#: Default number of leading text tokens (FLUX convention: T5 + guidance).
HAP_DEFAULT_TEXT_LEN = 512

#: Sentinel anchor stride meaning "anchors disabled".
HAP_ANCHOR_OFF = 1 << 30

#: Training sequence length for proportional attention scaling
#: (FLUX: 64*64 image tokens + 512 text tokens).
HAP_TRAIN_SEQ_LEN = 64 * 64 + 512  # 4608


# ---------------------------------------------------------------------------
# Capability probes
# ---------------------------------------------------------------------------

def _torch_version_at_least(major: int, minor: int) -> bool:
    """Return True iff the installed torch version is >= major.minor.

    Parses ``torch.__version__`` defensively (handles suffixes such as
    ``2.5.1+cu124`` or ``2.6.0.dev20241101``). Never raises.
    """
    try:
        version = torch.__version__.split("+", 1)[0]
        parts = version.split(".")
        maj = int(parts[0])
        mnr = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return (maj, mnr) >= (major, minor)
    except Exception:  # pragma: no cover - defensive
        return False


def hap_flex_available() -> bool:
    """Return True iff this environment can run FlexAttention.

    Requirements (all must hold):
    - CUDA is available (FlexAttention is CUDA-only in torch),
    - torch >= 2.5,
    - ``torch.nn.attention.flex_attention`` imports cleanly.

    Never raises: any failure yields False.
    """
    try:
        if not torch.cuda.is_available():
            return False
        if not _torch_version_at_least(2, 5):
            return False
        import torch.nn.attention.flex_attention  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# P7/T7.1 — Proportional attention scaling (reference: attention.py:89-93)
# ---------------------------------------------------------------------------

def proportional_scale_ratio(
    seq_len: int,
    train_seq_len: int = HAP_TRAIN_SEQ_LEN,
) -> float:
    """Attention-scale ratio for sequences longer than the training extent.

    Reference (``hrdit/attention.py`` lines 89-93): at high resolution the
    effective softmax temperature should grow with the sequence length.  The
    reference multiplies the attention scale by

        sqrt( log(seq_len) / log(train_seq_len) )

    so the logits gain the factor ``log(seq_len, train_seq_len)`` on top of
    the model's default ``1/sqrt(head_dim)``.

    Properties (all tested):
    - ``ratio == 1.0`` exactly at ``seq_len == train_seq_len`` (1024px FLUX:
      4096 image + 512 text = 4608) — a strict no-op at the training extent;
    - monotonically increasing in ``seq_len``;
    - ``< 1`` below the training extent (faithful to the formula; the feature
      is only intended for ``seq_len >= train_seq_len``).

    Applied by pre-scaling ``q`` (only) in the wrapper: a scalar factor on
    ``q`` commutes with RoPE and with any attention mask, so no backend
    changes are needed (plan §2.6).
    """
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1, got {seq_len}")
    if train_seq_len < 2:
        raise ValueError(f"train_seq_len must be >= 2, got {train_seq_len}")
    return math.sqrt(math.log(seq_len) / math.log(train_seq_len))


# ---------------------------------------------------------------------------
# P1/T1.1 — Scope plan model (reference-compatible JSON I/O)
# ---------------------------------------------------------------------------

@dataclass
class ScopePlan:
    """Per-layer, per-head HAP scope parameters ``(alpha, beta)``.

    Reference format (``configs/scope_plan_flux.json``)::

        {"alphas": [[..per head..] ..per layer..], "betas": [[...]]}

    ``alpha`` is an absolute band contribution in tokens; ``beta`` a fraction
    of the sequence length (in 64-token blocks).  See :func:`band_blocks`.
    """

    alphas: List[List[float]]
    betas: List[List[float]]

    # -- construction ------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "ScopePlan":
        """Build from a reference-format dict, validating eagerly."""
        if not isinstance(d, dict):
            raise ValueError(f"ScopePlan: expected a dict, got {type(d).__name__}")
        for key in ("alphas", "betas"):
            if key not in d:
                raise ValueError(f"ScopePlan: missing required key {key!r}")
        plan = cls(alphas=d["alphas"], betas=d["betas"])
        plan.validate()
        return plan

    @classmethod
    def load(cls, path: Union[str, "os.PathLike"]) -> "ScopePlan":
        """Load a scope plan from a JSON file (reference format)."""
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def to_dict(self) -> dict:
        """Serialize to the reference JSON shape (round-trip stable)."""
        return {"alphas": self.alphas, "betas": self.betas}

    def save(self, path: Union[str, "os.PathLike"]) -> None:
        """Write the plan as JSON (reference format)."""
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Check structural invariants; raise ``ValueError`` with context.

        Rules: both keys are lists of lists of numbers, rectangular (every
        layer has the same head count, alphas and betas agree), all values
        finite and >= 0, at least one layer and one head.
        """
        for name, table in (("alphas", self.alphas), ("betas", self.betas)):
            if not isinstance(table, (list, tuple)):
                raise ValueError(f"ScopePlan.{name}: expected a list of lists, got {type(table).__name__}")
            if len(table) == 0:
                raise ValueError(f"ScopePlan.{name}: must contain at least one layer")
            width = None
            for li, row in enumerate(table):
                if not isinstance(row, (list, tuple)):
                    raise ValueError(f"ScopePlan.{name}[{li}]: expected a list, got {type(row).__name__}")
                if width is None:
                    width = len(row)
                    if width == 0:
                        raise ValueError(f"ScopePlan.{name}[{li}]: layer has no heads")
                elif len(row) != width:
                    raise ValueError(
                        f"ScopePlan.{name}: ragged plan — layer {li} has {len(row)} "
                        f"heads, expected {width}"
                    )
                for hi, val in enumerate(row):
                    if isinstance(val, bool) or not isinstance(val, (int, float)):
                        raise ValueError(
                            f"ScopePlan.{name}[{li}][{hi}]: expected a number, got {val!r}"
                        )
                    if not math.isfinite(float(val)):
                        raise ValueError(f"ScopePlan.{name}[{li}][{hi}]: value must be finite, got {val!r}")
                    if float(val) < 0:
                        raise ValueError(
                            f"ScopePlan.{name}[{li}][{hi}]: value must be >= 0, got {val!r}"
                        )
        if len(self.alphas) != len(self.betas):
            raise ValueError(
                f"ScopePlan: alphas has {len(self.alphas)} layers but betas has {len(self.betas)}"
            )
        if len(self.alphas[0]) != len(self.betas[0]):
            raise ValueError(
                f"ScopePlan: layer 0 has {len(self.alphas[0])} alpha heads but "
                f"{len(self.betas[0])} beta heads"
            )

    # -- shape accessors ----------------------------------------------------

    @property
    def num_layers(self) -> int:
        return len(self.alphas)

    @property
    def num_heads(self) -> int:
        return len(self.alphas[0])

    def layer_bands(self, layer: int, seq_len: int) -> List[int]:
        """Band width (in 64-token blocks) of every head of ``layer``."""
        return band_blocks(self.alphas[layer], self.betas[layer], seq_len)


# ---------------------------------------------------------------------------
# P1/T1.2 — Band width math (reference formula, exact)
# ---------------------------------------------------------------------------

def band_blocks(
    alphas: Sequence[float],
    betas: Sequence[float],
    seq_len: int,
    block: int = HAP_BLOCK,
) -> List[int]:
    """Scope ``(alpha, beta)`` -> band width in ``block``-token blocks.

    Reference (``hrdit/hap.py`` ``HapRuntime.band_blocks``)::

        nbx = seq_len // BLOCK
        band = max(2 * int(a // BLOCK + b * nbx) - 1, 1)

    NOTE: ``nbx`` is computed from the FULL ``seq_len`` (text tokens included)
    exactly like the reference — replicated bit-for-bit (plan §2.3).
    """
    nbx = seq_len // block
    return [max(2 * int(a // block + b * nbx) - 1, 1) for a, b in zip(alphas, betas)]


def half_blocks(band_widths: Sequence[int]) -> List[int]:
    """Band width -> per-head half-width ``(band - 1) // 2`` (reference)."""
    return [(int(x) - 1) // 2 for x in band_widths]


# ---------------------------------------------------------------------------
# P1/T1.3 — Dense boolean mask builder (FlexAttention mask_mod oracle)
# ---------------------------------------------------------------------------

def build_band_mask(
    seq_len: int,
    text_len: int,
    half_blocks_per_head: Sequence[int],
    anchor_stride: int = 0,
    block: int = HAP_BLOCK,
) -> torch.Tensor:
    """Dense boolean attention mask implementing the reference ``mask_mod``.

    Reference semantics (``hrdit/hap.py`` lines 34-41), per (head, q, k):

    - ``text_q``: ``q < text_len`` — text queries attend everything,
    - ``text_k``: ``k < text_len`` — text keys visible to everyone,
    - ``band``: ``|qb - kb| <= half[h]`` with ``qb = (q - text_len) // block``,
    - ``anchor``: ``kb % anchor_stride == 0`` (every stride-th image block is
      globally visible); ``anchor_stride <= 0`` or ``>= ANCHOR_OFF`` disables.

    Returns a bool tensor of shape ``(H, seq_len, seq_len)``.  Built
    row-block-wise to keep transient memory bounded; for test sizes it is
    fully dense.  This is the ORACLE for the FlexAttention backend and the
    input of the dense SDPA fallback.
    """
    heads = len(half_blocks_per_head)
    stride = anchor_stride if anchor_stride and 0 < anchor_stride < HAP_ANCHOR_OFF else HAP_ANCHOR_OFF

    q_idx = torch.arange(seq_len)
    k_idx = torch.arange(seq_len)
    text_q = q_idx < text_len                       # (Q,)
    text_k = k_idx < text_len                       # (K,)
    qb = (q_idx - text_len) // block                # (Q,)  negative for text rows
    kb = (k_idx - text_len) // block                # (K,)
    # Anchor visibility per key (text keys are already covered by text_k;
    # negative kb for text keys never satisfies kb % stride == 0 for
    # stride >= ANCHOR_OFF, and for small strides Python's floored modulo
    # could mark them — clamp to the reference behaviour: anchors apply to
    # IMAGE blocks only).
    anchor_k = torch.zeros(seq_len, dtype=torch.bool)
    if stride < HAP_ANCHOR_OFF:
        img = k_idx >= text_len
        anchor_k[img] = (kb[img] % stride) == 0

    dist = (qb.unsqueeze(1) - kb.unsqueeze(0)).abs()  # (Q, K)
    mask = torch.empty(heads, seq_len, seq_len, dtype=torch.bool)
    for h, half in enumerate(half_blocks_per_head):
        band = dist <= int(half)
        m = text_q.unsqueeze(1) | text_k.unsqueeze(0) | band | anchor_k.unsqueeze(0)
        # Text rows: qb is negative there, so `band` could spuriously hide or
        # show; text_q already makes the whole row True — nothing to fix.
        mask[h] = m
    return mask


# ---------------------------------------------------------------------------
# P1/T1.4 — Compute-cost model (closed form, no mask materialization)
# ---------------------------------------------------------------------------

def _visible_image_cols_per_row(
    qb: torch.Tensor,
    tokens_per_block: torch.Tensor,
    half: int,
    anchor_stride: int,
) -> torch.Tensor:
    """Number of visible IMAGE key TOKENS for each image query block index ``qb``.

    Visibility is block-granular (the band/anchor conditions depend only on
    block indices), so a visible block contributes ALL its tokens (the last
    block may be partial).  Counts band blocks ``|qb - kb| <= half`` plus
    anchor blocks ``kb % stride == 0`` not already inside the band, using
    prefix sums.  Vectorized over ``qb``; returns an int64 tensor of the same
    shape.
    """
    num_img_blocks = tokens_per_block.shape[0]
    prefix = torch.cat([torch.zeros(1, dtype=torch.int64), tokens_per_block.cumsum(0)])
    # Inclusive band [qb-half, qb+half] as the half-open [lo, hi).
    lo = torch.clamp(qb - half, min=0, max=num_img_blocks)
    hi = torch.clamp(qb + half + 1, min=0, max=num_img_blocks)
    band_cols = prefix[hi] - prefix[lo]

    if anchor_stride and 0 < anchor_stride < HAP_ANCHOR_OFF and num_img_blocks > 0:
        kb_all = torch.arange(num_img_blocks, dtype=torch.int64)
        anchor_tokens = torch.where(kb_all % anchor_stride == 0, tokens_per_block, torch.zeros_like(tokens_per_block))
        aprefix = torch.cat([torch.zeros(1, dtype=torch.int64), anchor_tokens.cumsum(0)])
        total_anchor_cols = aprefix[num_img_blocks]
        anchors_in = aprefix[hi] - aprefix[lo]
        return band_cols + (total_anchor_cols - anchors_in)
    return band_cols


def band_compute_cost(
    seq_len: int,
    text_len: int,
    half_blocks_per_head: Sequence[int],
    anchor_stride: int = 0,
    block: int = HAP_BLOCK,
) -> float:
    """Retained query-key pairs per head (closed form) — the ``I_c`` cost.

    Equals ``build_band_mask(...).sum() / H`` exactly (tested), without
    materializing the ``(H, T, T)`` mask.

    Counting per head:
    - text query rows (``text_len`` of them) see ALL ``seq_len`` keys;
    - image query rows: all ``text_len`` text keys + visible image key tokens
      (band + anchors, counted per block row via prefix sums and weighted by
      the number of query rows in each block row).
    """
    stride = anchor_stride if anchor_stride and 0 < anchor_stride < HAP_ANCHOR_OFF else HAP_ANCHOR_OFF
    num_img_tokens = seq_len - text_len
    if num_img_tokens < 0:
        raise ValueError(f"band_compute_cost: text_len={text_len} exceeds seq_len={seq_len}")
    num_img_blocks = (num_img_tokens + block - 1) // block  # ceil

    # Tokens per image block (last block may be partial) — used both as the
    # query-row weight and as the visible-column weight.
    tokens_per_block = torch.full((num_img_blocks,), block, dtype=torch.int64)
    if num_img_blocks > 0 and num_img_tokens % block:
        tokens_per_block[-1] = num_img_tokens % block
    qb_all = torch.arange(num_img_blocks, dtype=torch.int64)

    total = 0.0
    text_pairs = text_len * seq_len
    for half in half_blocks_per_head:
        vis_cols = _visible_image_cols_per_row(qb_all, tokens_per_block, int(half), stride)
        img_pairs = int((tokens_per_block * (text_len + vis_cols)).sum().item())
        total += text_pairs + img_pairs
    return total / max(len(half_blocks_per_head), 1)


def flops_ratio(
    plan: ScopePlan,
    seq_len: int,
    text_len: int = HAP_DEFAULT_TEXT_LEN,
    anchor_stride: int = 0,
    block: int = HAP_BLOCK,
) -> float:
    """Retained query-key pair fraction of a whole plan vs full attention.

    Averages :func:`band_compute_cost` over all layers and divides by the
    full-attention pair count ``seq_len ** 2``.  Used for logging/telemetry
    (``HapRuntime.flops_ratio`` in the plan).
    """
    full = float(seq_len) * float(seq_len)
    if full <= 0:
        return 1.0
    total = 0.0
    for layer in range(plan.num_layers):
        bands = plan.layer_bands(layer, seq_len)
        halves = half_blocks(bands)
        total += band_compute_cost(seq_len, text_len, halves, anchor_stride, block)
    return (total / plan.num_layers) / full


# ---------------------------------------------------------------------------
# P2/T2.1 — HapContext (process-scoped HAP state)
# ---------------------------------------------------------------------------

@dataclass
class HapContext:
    """Per-forward HAP activation state (like ``SPAContext``).

    Stored in ``src.spa_context._HAP_ACTIVE`` for the duration of a forward.

    Attributes:
        active: master switch (False == plain attention).
        plan: the scope plan (per-layer, per-head alpha/beta).
        text_len: leading text tokens (FLUX convention: 512).
        anchor_stride: globally visible every N-th image block; 0 == off.
        backend: ``"auto"`` (flex if available else dense) | ``"flex"`` |
            ``"dense"`` | ``"off"`` (warn + fall back to plain attention).
    """

    active: bool = False
    plan: ScopePlan = None  # type: ignore[assignment]
    text_len: int = HAP_DEFAULT_TEXT_LEN
    anchor_stride: int = 0
    backend: str = "auto"

    def resolve_backend(self) -> str:
        """Resolve ``"auto"`` to a concrete backend for this environment."""
        if self.backend == "auto":
            return "flex" if hap_flex_available() else "dense"
        return self.backend


# ---------------------------------------------------------------------------
# P2/T2.2 — Dense backend (SDPA + boolean mask; test oracle + fallback)
# ---------------------------------------------------------------------------

def hap_attn_dense(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """Masked attention via SDPA with a boolean mask (True == attend).

    ``q, k, v``: ``(B, H, S, D)``.  ``mask``: bool ``(H, S, S)`` (broadcast
    over the batch).  ``scale`` defaults to ``1/sqrt(D)``.  Works on any
    device/dtype — the correctness oracle for the FlexAttention backend and
    the fallback where FlexAttention is unavailable.
    """
    import torch.nn.functional as F

    if scale is None:
        scale = q.shape[-1] ** -0.5
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False, scale=scale
    )


# ---------------------------------------------------------------------------
# P2/T2.3 — FlexAttention backend adapter (CUDA-gated in production)
# ---------------------------------------------------------------------------

def _make_flex_mask_mod(half_buf: torch.Tensor, anchor_buf: torch.Tensor, text_len: int, block: int):
    """Build the reference ``mask_mod`` closure over device int buffers.

    Mirrors ``hrdit/hap.py`` lines 34-41 exactly (with ``text_len`` as a
    parameter instead of a global):

        text_q | text_k | band | anchor
    """

    def mask_mod(b, h, q, k):
        text_q = q < text_len
        text_k = k < text_len
        qb = (q - text_len) // block
        kb = (k - text_len) // block
        band = (qb - kb).abs() <= half_buf[h]
        anchor = (kb % anchor_buf) == 0
        return text_q | text_k | band | anchor

    return mask_mod


# ---------------------------------------------------------------------------
# P2/T2.4 — HapRuntime facade
# ---------------------------------------------------------------------------

class HapRuntime:
    """One entry point for the unified wrapper: lazy per-layer mask prepare +
    backend dispatch + cache hygiene.

    Singleton via :meth:`get` (mirrors the reference ``HapRuntime.get()``).
    The mask cache is keyed ``(seq_len, text_len, halves, stride)`` so layers
    sharing a scope share a mask; ``prepare_count`` counts actual builds
    (test hook).
    """

    _instance = None

    def __init__(self):
        self.mask_cache = {}
        self.prepare_count = 0
        self._flex_kernel = None
        self._warned_off = False

    @classmethod
    def get(cls) -> "HapRuntime":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Drop the singleton (test hygiene)."""
        cls._instance = None

    # -- mask preparation ----------------------------------------------------

    def prepare_layer(
        self,
        seq_len: int,
        halves: Sequence[int],
        text_len: int,
        anchor_stride: int,
        backend: str,
        device: torch.device = None,
    ):
        """Return the cached mask handle for this configuration, building on miss.

        Dense handle: bool mask ``(H, S, S)``.  Flex handle: a
        ``(block_mask, half_buf, anchor_buf)`` tuple like the reference.
        """
        if backend == "flex":
            return self._prepare_flex(seq_len, halves, text_len, anchor_stride, device)
        return self._prepare_dense(seq_len, halves, text_len, anchor_stride)

    def _prepare_dense(self, seq_len, halves, text_len, anchor_stride):
        key = ("dense", seq_len, text_len, tuple(int(x) for x in halves), int(anchor_stride))
        if key not in self.mask_cache:
            self.mask_cache[key] = build_band_mask(seq_len, text_len, halves, anchor_stride)
            self.prepare_count += 1
        return self.mask_cache[key]

    def _prepare_flex(self, seq_len, halves, text_len, anchor_stride, device):
        from torch.nn.attention.flex_attention import create_block_mask

        stride = anchor_stride if anchor_stride and anchor_stride > 0 else HAP_ANCHOR_OFF
        key = ("flex", seq_len, text_len, tuple(int(x) for x in halves), int(stride))
        if key not in self.mask_cache:
            half_buf = torch.tensor([int(x) for x in halves], device=device, dtype=torch.int32)
            anchor_buf = torch.tensor(stride, device=device, dtype=torch.int32)
            mask_mod = _make_flex_mask_mod(half_buf, anchor_buf, text_len, HAP_BLOCK)
            block_mask = create_block_mask(
                mask_mod, B=None, H=len(halves),
                Q_LEN=seq_len, KV_LEN=seq_len, device=device, _compile=True,
            )
            self.mask_cache[key] = (block_mask, half_buf, anchor_buf)
            self.prepare_count += 1
        return self.mask_cache[key]

    def _flex_kernel_fn(self):
        if self._flex_kernel is None:
            from torch.nn.attention.flex_attention import flex_attention

            self._flex_kernel = torch.compile(flex_attention, dynamic=False)
        return self._flex_kernel

    # -- attention dispatch ----------------------------------------------------

    def attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: int,
        scale: float = None,
        ctx: "HapContext" = None,
        text_len: int = None,
    ):
        """Run HAP-masked attention for ``layer``.

        Returns ``None`` when HAP is inactive or the backend is ``"off"``
        (one-time WARNING) so the caller falls back to plain attention.
        The mask is prepared lazily for the ACTUAL ``q.shape[-2]`` —
        resolution changes mid-run are mismatch-safe.  ``text_len`` overrides
        ``ctx.text_len`` when provided (the wrapper derives it from the live
        SPA context's position ids, plan P3/T3.4).
        """
        from .spa_context import get_hap_context

        if ctx is None:
            ctx = get_hap_context()
        if ctx is None or not ctx.active or ctx.plan is None:
            return None
        if layer >= ctx.plan.num_layers:
            logger.warning(
                "HAP: layer %d exceeds the scope plan (%d layers); using plain attention",
                layer, ctx.plan.num_layers,
            )
            return None

        seq_len = q.shape[-2]
        eff_text_len = ctx.text_len if text_len is None else int(text_len)
        eff_text_len = max(0, min(eff_text_len, seq_len))
        bands = ctx.plan.layer_bands(layer, seq_len)
        halves = half_blocks(bands)
        backend = ctx.resolve_backend()

        if backend == "off":
            if not self._warned_off:
                logger.warning("HAP: backend 'off' requested; falling back to plain attention")
                self._warned_off = True
            return None
        if backend == "flex":
            if not hap_flex_available():
                logger.warning("HAP: FlexAttention unavailable; using the dense SDPA fallback")
                backend = "dense"

        if backend == "flex":
            block_mask, half_buf, anchor_buf = self._prepare_flex(
                seq_len, halves, eff_text_len, ctx.anchor_stride, q.device
            )
            if scale is None:
                scale = q.shape[-1] ** -0.5
            return self._flex_kernel_fn()(q, k, v, block_mask=block_mask, scale=scale)

        mask = self._prepare_dense(seq_len, halves, eff_text_len, ctx.anchor_stride)
        return hap_attn_dense(q, k, v, mask.to(q.device), scale=scale)


# ---------------------------------------------------------------------------
# P4/T4.1 — Model-patching entry point
# ---------------------------------------------------------------------------

def apply_hap_to_model(
    model,
    model_type: str,
    scope_plan,
    anchor_stride: int = 32,
    enable_hap: bool = True,
    text_len: int = HAP_DEFAULT_TEXT_LEN,
    backend: str = "auto",
    proportional_attention: bool = False,
):
    """Patch a ComfyUI model with HAP (Head-Adaptive attention Pruning).

    Mirrors ``apply_spa_to_model`` (plan P4/T4.1):

    - clones the model patcher,
    - resolves the backend type (reusing SPA's detector),
    - Nunchaku guard: fused/quantized kernels bypass ``optimized_attention``,
      so HAP logs a warning and returns the model UNCHANGED (never a silent
      broken path — same policy as SPA decision 4),
    - stores the :class:`HapContext` on ``m._hap_ctx``; the shared unet wrapper
      activates it around every forward (read at call time, so SPA-then-HAP
      installs share one wrapper),
    - installs the shared HRDiT hook as consumer ``"hap"`` (ref-counted with
      SPA, plan P3/T3.3).

    ``scope_plan`` may be a :class:`ScopePlan`, a reference-format dict, or a
    path to a JSON file.  ``enable_hap=False`` returns the clone with no
    wrapper and no context (transparent passthrough).
    """
    from .spa import _hrdit_install_hook, _spa_resolve_type

    # PROPORTIONAL ATTENTION SCALING (plan P7/T7.3): OR-semantics — either the
    # SPA or the HAP node may enable it.  The real ``ModelPatcher.clone()``
    # copies only its KNOWN fields (custom attrs do NOT survive), so the
    # existing flag must be read from the SOURCE patcher BEFORE the clone and
    # re-applied to the clone.  Set right after the clone (before any early
    # return) so the flag survives even when HAP itself is disabled but a hook
    # is (or gets) installed by SPA.  The shared unet wrapper reads it at call
    # time and activates the q pre-scaling for the whole forward.
    prev_proportional = bool(getattr(model, "_hrdit_proportional_attention", False))
    m = model.clone()
    m._hrdit_proportional_attention = prev_proportional or bool(proportional_attention)
    dm = m.model.diffusion_model
    detected_type = _spa_resolve_type(model_type, dm)
    logger.info("HAP: Detected model type: %s", detected_type)

    if detected_type == "nunchaku":
        logger.warning(
            "HAP: unsupported on quantized/fused Nunchaku kernels (they bypass "
            "optimized_attention); returning the model UNCHANGED."
        )
        m._hap_ctx = None
        return m

    if not enable_hap:
        m._hap_ctx = None
        return m

    if isinstance(scope_plan, (str, os.PathLike)):
        scope_plan = ScopePlan.load(scope_plan)
    elif isinstance(scope_plan, dict):
        scope_plan = ScopePlan.from_dict(scope_plan)
    elif not isinstance(scope_plan, ScopePlan):
        raise ValueError(
            f"apply_hap_to_model: scope_plan must be a ScopePlan, dict or path, "
            f"got {type(scope_plan).__name__}"
        )
    scope_plan.validate()

    ctx = HapContext(
        active=True,
        plan=scope_plan,
        text_len=int(text_len),
        anchor_stride=int(anchor_stride),
        backend=backend,
    )
    m._hap_ctx = ctx
    m._hap_plan = scope_plan
    _hrdit_install_hook(m, detected_type, consumer="hap")
    logger.info(
        "HAP: enabled for backend %r: %d layers x %d heads, anchor_stride=%d, "
        "text_len=%d, backend=%s",
        detected_type, scope_plan.num_layers, scope_plan.num_heads,
        ctx.anchor_stride, ctx.text_len, ctx.resolve_backend(),
    )
    return m


def restore_hap_attention_hook(m) -> None:
    """Remove the HAP consumer from the shared hook (plan P4/T4.1).

    The wrapper is restored only when the LAST consumer leaves (ref-counted,
    plan P3/T3.3); clearing ``m._hap_ctx`` deactivates HAP immediately either
    way (the unet wrapper reads it at call time).
    """
    from .spa import _hrdit_uninstall_hook

    m._hap_ctx = None
    _hrdit_uninstall_hook(m, "hap")
