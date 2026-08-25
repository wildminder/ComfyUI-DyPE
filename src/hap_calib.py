"""HAP calibration math (plan phase P5, theory doc §6.2/§6.3/§8).

Pure-torch, CPU-friendly primitives used to derive a per-head scope plan
from one (or a few) full-attention backward passes:

- :func:`taylor_softmax_pruning_score` — the paper's first-order estimate of
  the quality impact of removing the attention entries outside a candidate
  scope, including the softmax renormalization term (theory §6.2, §8).
- :func:`estimate_head_scope_costs` — vectorized scoring of ALL candidate
  scopes per head in one pass (row prefix sums over the nested symmetric
  windows), no per-scope ``T×T`` mask loop (plan T5.2).
- :func:`calibration_cost_table` — the ``I_c[h, s]`` compute-cost table, a
  thin wrapper over :func:`src.hap.band_compute_cost` using the plan §2.4
  scope→beta mapping (plan T5.3).
- :func:`solve_multiple_choice_knapsack` — dependency-free DP solver for the
  "one scope per head under a global compute budget" integer program
  (theory §6.3, §8; plan T5.4).
- :func:`calibrate_scope_plan` — end-to-end pure-math calibration: averaged
  per-prompt score tables → reference-format ``{"alphas", "betas"}`` dict
  (plan T5.5).

The gradient-path collector (chunked differentiable attention) lives here too
but is implemented in phase P6.

All functions are deterministic and side-effect free.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Sequence, Tuple

import torch

from .hap import HAP_BLOCK, band_blocks, band_compute_cost, half_blocks

logger = logging.getLogger("ComfyUI-DyPE")

__all__ = [
    "taylor_softmax_pruning_score",
    "scope_keep_mask",
    "estimate_head_scope_costs",
    "scope_to_beta",
    "calibration_cost_table",
    "solve_multiple_choice_knapsack",
    "calibrate_scope_plan",
    "chunked_attention",
    "collect_scope_scores",
]


# ---------------------------------------------------------------------------
# P5/T5.1 — Taylor-softmax pruning score (theory §6.2 formula)
# ---------------------------------------------------------------------------

def taylor_softmax_pruning_score(
    attn_probs: torch.Tensor,
    dloss_dA: torch.Tensor,
    keep_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """First-order quality impact of removing the entries outside ``keep_mask``.

    Implements theory doc §8 verbatim::

        I_q ≈ Σ_{(u,v) ∈ omit} [ ∂L/∂A(u,v) · (−A(u,v))
              + Σ_{w≠v} ∂L/∂A(u,w) · A(u,w) · A(u,v) / (1 − A(u,v)) ]

    The second term accounts for softmax renormalization: removing one entry
    rescales every kept entry in the same row by ``1 / (1 − A(u,v))``.

    Args:
        attn_probs: post-softmax attention ``A`` of shape ``(T, T)``.
        dloss_dA: gradient of the scalar calibration loss wrt ``A``, ``(T, T)``.
        keep_mask: bool ``(T, T)``; ``True`` = retained pair.
        eps: lower clamp of ``1 − A`` so a saturating row (``A(u,v) → 1``)
            cannot divide by zero.

    Returns:
        Scalar tensor (same dtype as ``attn_probs``).
    """
    if attn_probs.dim() != 2 or attn_probs.shape[0] != attn_probs.shape[1]:
        raise ValueError(
            f"attn_probs must be square (T, T), got {tuple(attn_probs.shape)}"
        )
    if dloss_dA.shape != attn_probs.shape:
        raise ValueError(
            f"dloss_dA shape {tuple(dloss_dA.shape)} != attn_probs shape "
            f"{tuple(attn_probs.shape)}"
        )
    if keep_mask.shape != attn_probs.shape:
        raise ValueError(
            f"keep_mask shape {tuple(keep_mask.shape)} != attn_probs shape "
            f"{tuple(attn_probs.shape)}"
        )

    omitted = ~keep_mask
    A = attn_probs
    G = dloss_dA

    # Direct removal: dL/dA(u,v) * (-A(u,v)).
    direct = -G * A

    # Softmax renormalization contribution:
    #   sum_{w != v} G(u,w) * A(u,w) * A(u,v) / (1 - A(u,v))
    # First compute sum_w G(u,w) A(u,w), then remove the w=v term.
    row_dot = (G * A).sum(dim=-1, keepdim=True)          # (T, 1)
    excluding_v = row_dot - G * A                        # (T, T)
    renorm = excluding_v * A / (1.0 - A).clamp_min(eps)

    return (direct[omitted] + renorm[omitted]).sum()


# ---------------------------------------------------------------------------
# P5/T5.2 — Vectorized nested-band scoring (all scopes in one pass)
# ---------------------------------------------------------------------------

def _scope_radius(seq_len: int, s: int, num_scopes: int) -> int:
    """Token radius of candidate scope ``s ∈ [1, num_scopes]``.

    Mirrors the theory §8 skeleton ``local_window_mask``::

        width  = max(1, ceil((s / num_scopes) * seq_len))
        radius = max(0, width // 2)

    so the kept image window is ``|u - v| <= radius`` and the visible
    fraction is ≈ ``s / num_scopes`` (paper §6.1).  The last candidate
    ``s == num_scopes`` is FULL attention (radius ``seq_len`` keeps every
    column; paper §6.1).
    """
    if s == num_scopes:
        return seq_len
    width = max(1, math.ceil((s / num_scopes) * seq_len))
    return max(0, width // 2)


def scope_keep_mask(
    seq_len: int,
    s: int,
    num_scopes: int,
    text_len: int = 0,
    device=None,
) -> torch.Tensor:
    """Boolean ``(T, T)`` keep mask for candidate scope ``s`` (True = kept).

    Runtime semantics (plan §2.3): text rows attend everything and text
    columns are visible to everyone, so they are always kept; image tokens
    keep the symmetric window ``|u - v| <= radius(s)``.  This is the exact
    window definition used by :func:`estimate_head_scope_costs` — exposed so
    the naive per-scope loop (test oracle) and the vectorized scorer share
    one definition.
    """
    if not (1 <= s <= num_scopes):
        raise ValueError(f"scope s must be in [1, {num_scopes}], got {s}")
    r = _scope_radius(seq_len, s, num_scopes)
    u = torch.arange(seq_len, device=device)
    keep = (u.unsqueeze(1) - u.unsqueeze(0)).abs() <= r
    if text_len > 0:
        text = u < text_len
        keep = keep | text.unsqueeze(1) | text.unsqueeze(0)
    return keep


def estimate_head_scope_costs(
    attn_probs: torch.Tensor,
    dloss_dA: torch.Tensor,
    num_scopes: int = 50,
    text_len: int = 0,
) -> torch.Tensor:
    """Taylor quality cost ``I_q[h, s]`` for every head and candidate scope.

    Vectorized equivalent of looping :func:`taylor_softmax_pruning_score`
    over all ``num_scopes`` nested symmetric windows (plan T5.2):

    1. build the per-entry contribution tensor
       ``C = direct + renorm`` (the theory §8 summand),
    2. zero the text rows AND text-key columns — text queries attend
       everything and text keys are always visible at runtime (plan §2.3),
       so they never contribute to the omitted sum during calibration,
    3. row prefix sums give every nested band sum
       ``kept(u) = Σ_{|u-v| ≤ r(s)} C[u, v]`` in ``O(T)`` per scope,
    4. ``score[s] = Σ_u (row_total − kept(u))`` (the omitted contribution).

    Complexity ``O(H · T² + S · H · T)`` instead of the naive
    ``O(S · H · T²)``; only the (already materialized) ``C`` tensor is held.

    Args:
        attn_probs: post-softmax attention, shape ``(H, T, T)``.
        dloss_dA: gradient wrt the attention matrix, shape ``(H, T, T)``.
        num_scopes: number of candidate scopes ``N_scope`` (paper: 50).
        text_len: leading text tokens that are never omitted.

    Returns:
        Tensor of shape ``(H, num_scopes)`` (same dtype as ``attn_probs``);
        entry ``[h, s-1]`` is the score for scope ``s``.  The last column
        (full scope) is exactly zero.
    """
    if attn_probs.dim() != 3:
        raise ValueError(
            f"attn_probs must be (H, T, T), got {tuple(attn_probs.shape)}"
        )
    heads, seq_len, k_len = attn_probs.shape
    if seq_len != k_len:
        raise ValueError(
            f"attn_probs must be square in the last two dims, got "
            f"{tuple(attn_probs.shape)}"
        )
    if dloss_dA.shape != attn_probs.shape:
        raise ValueError(
            f"dloss_dA shape {tuple(dloss_dA.shape)} != attn_probs shape "
            f"{tuple(attn_probs.shape)}"
        )
    if num_scopes < 1:
        raise ValueError(f"num_scopes must be >= 1, got {num_scopes}")
    if not (0 <= text_len <= seq_len):
        raise ValueError(
            f"text_len must be in [0, seq_len={seq_len}], got {text_len}"
        )

    # Delegate to the shared per-row implementation (row_offset=0, R=T).  The
    # chunked collector (P6) reuses the SAME helper on query-row slices, so
    # chunked accumulation is exactly equal to this dense call by construction.
    return _chunk_row_scores(attn_probs, dloss_dA, num_scopes, text_len, 0)


def _chunk_row_scores(
    A: torch.Tensor,
    G: torch.Tensor,
    num_scopes: int,
    text_len: int,
    row_offset: int,
) -> torch.Tensor:
    """Per-head omitted-sum scores for a SLICE of query rows.

    The workhorse shared by :func:`estimate_head_scope_costs` (dense, one
    slice covering all ``T`` rows) and the P6 chunked collector (many slices).
    Because the Taylor score is a sum over query rows and each row's softmax
    spans the FULL key dimension, slicing rows is exact: summing this over a
    partition of the rows equals the dense result bit-for-bit.

    Args:
        A: post-softmax attention for these rows, shape ``(H, R, T)`` where
            ``R`` is the number of query rows in the slice and ``T`` the full
            key count.
        G: ``dloss/dA`` for the same rows, shape ``(H, R, T)``.
        num_scopes: number of candidate scopes ``N_scope``.
        text_len: leading text tokens (never omitted).
        row_offset: global index of the first query row in this slice (for
            text-row detection and the symmetric window ``|u - v| <= r``).

    Returns:
        ``(H, num_scopes)`` tensor; entry ``[h, s-1]`` is the omitted-sum
        score for scope ``s`` over these rows only.
    """
    heads, rows, seq_len = A.shape

    # Per-entry Taylor summand C = direct + renorm (theory §8).
    direct = -G * A
    row_dot = (G * A).sum(dim=-1, keepdim=True)          # (H, R, 1)
    excluding_v = row_dot - G * A                        # (H, R, T)
    renorm = excluding_v * A / (1.0 - A).clamp_min(1e-8)
    C = direct + renorm

    # Runtime mask semantics (plan §2.3): text queries attend EVERYTHING
    # (``text_q``) and text keys are visible to everyone (``text_k``).  So
    # text rows omit nothing and text columns are never omitted — zero both
    # so they never contribute to the omitted (quality-cost) sum.
    if text_len > 0:
        C = C.clone()
        grow = torch.arange(row_offset, row_offset + rows, device=C.device)
        text_rows = grow < text_len                       # (R,)
        if bool(text_rows.any()):
            C[:, text_rows, :] = 0.0   # text queries omit nothing
        C[:, :, :text_len] = 0.0       # text keys never omitted

    # Row prefix sums: kept(u) = P[u, hi] - P[u, lo] for the band
    # [u - r, u + r] clipped to [0, T).
    P = torch.cat(
        [torch.zeros(heads, rows, 1, dtype=C.dtype, device=C.device),
         torch.cumsum(C, dim=-1)],
        dim=-1,
    )
    row_total = P[:, :, seq_len]                          # (H, R)
    u = torch.arange(row_offset, row_offset + rows, device=C.device)

    scores = torch.zeros(heads, num_scopes, dtype=C.dtype, device=C.device)
    for s in range(1, num_scopes + 1):
        # _scope_radius returns seq_len for the last candidate (full
        # attention ⇒ nothing omitted ⇒ score exactly 0, paper §6.1).
        r = _scope_radius(seq_len, s, num_scopes)
        lo = torch.clamp(u - r, min=0)
        hi = torch.clamp(u + r + 1, max=seq_len)
        # kept(u) = P[u, hi] - P[u, lo]  (prefix-sum range sum over v)
        hi_idx = hi.unsqueeze(0).unsqueeze(-1).expand(heads, rows, 1)
        lo_idx = lo.unsqueeze(0).unsqueeze(-1).expand(heads, rows, 1)
        kept = torch.gather(P, 2, hi_idx).squeeze(-1) - torch.gather(
            P, 2, lo_idx
        ).squeeze(-1)
        scores[:, s - 1] = (row_total - kept).sum(dim=-1)
    return scores


# ---------------------------------------------------------------------------
# P5/T5.3 — Calibration compute-cost table (scope → beta → band cost)
# ---------------------------------------------------------------------------

def scope_to_beta(s: int, num_scopes: int) -> float:
    """Plan §2.4 mapping: candidate scope ``s ∈ [1, num_scopes]`` → beta.

    ``beta = 0.5 · s / N_scope`` with ``alpha = 0`` for ``s < N_scope``; the
    runtime ``band_blocks`` formula consumes these betas unchanged and the
    resulting band width ≈ ``s / N_scope`` of the image blocks (visible
    fraction ≈ ``s / N_scope``, paper §6.1).

    The LAST candidate ``s = N_scope`` maps to ``beta = 1.0`` — NOT ``0.5`` —
    because paper §6.1 requires it to be EXACT full attention.  Under the
    reference band formula ``band = 2·int(beta·nbx) − 1``:

    - ``beta = 0.5`` gives ``band ≈ nbx − 1``; edge query blocks then see
      only ~half the image (e.g. ``nbx = 8`` → ``half = 3`` → query block 0
      covers blocks 0..3 only),
    - ``beta = 1.0`` gives ``band = 2·nbx − 1`` → ``half = nbx − 1``, which
      satisfies ``|qb − kb| <= half`` for every pair — exact full attention.
    """
    if not (1 <= s <= num_scopes):
        raise ValueError(
            f"scope s must be in [1, {num_scopes}], got {s}"
        )
    if s == num_scopes:
        return 1.0
    return 0.5 * s / num_scopes


def calibration_cost_table(
    num_heads: int,
    seq_len: int,
    text_len: int = 0,
    num_scopes: int = 50,
    anchor_stride: int = 0,
    block: int = HAP_BLOCK,
) -> torch.Tensor:
    """Compute-cost table ``I_c[h, s]`` (retained query-key pairs).

    Thin wrapper over :func:`src.hap.band_compute_cost` using the §2.4
    scope→beta mapping.  All heads share the same candidate scopes, so every
    row is identical; the table is shaped ``(num_heads, num_scopes)`` for the
    solver.  The last column is exactly ``seq_len ** 2`` (full attention).
    """
    if num_heads < 1:
        raise ValueError(f"num_heads must be >= 1, got {num_heads}")
    if num_scopes < 1:
        raise ValueError(f"num_scopes must be >= 1, got {num_scopes}")

    costs = torch.zeros(num_heads, num_scopes, dtype=torch.float64)
    for s in range(1, num_scopes + 1):
        beta = scope_to_beta(s, num_scopes)
        betas = [beta] * num_heads
        alphas = [0.0] * num_heads
        bands = band_blocks(alphas, betas, seq_len, block)
        halves = half_blocks(bands)
        c = band_compute_cost(seq_len, text_len, halves, anchor_stride, block)
        costs[:, s - 1] = c
    return costs


# ---------------------------------------------------------------------------
# P5/T5.4 — Multiple-choice knapsack DP (theory §8)
# ---------------------------------------------------------------------------

def solve_multiple_choice_knapsack(
    quality_cost: torch.Tensor,
    compute_cost: torch.Tensor,
    budget_ratio: float = 0.10,
    bins: int = 4000,
) -> List[int]:
    """Solve the HAP scope assignment with multiple-choice knapsack DP.

    Pick exactly one scope per head so that the total compute cost stays
    within ``budget_ratio · full_cost`` while minimizing total quality cost.
    Dependency-free, deterministic replacement for the paper's ILP solver
    (Gurobi) on ceil-discretized costs (theory §8).

    Tie-breaking: scopes are scanned in ascending order with a STRICT
    improvement test, so on equal quality the LOWEST scope index wins —
    the result is fully deterministic.

    Args:
        quality_cost: ``(H, S)`` estimated quality loss, lower is better.
        compute_cost: ``(H, S)`` retained query-key pairs.
        budget_ratio: fraction ``r_c`` of full-attention cost (paper: 0.1).
        bins: discretization resolution for the floating costs.

    Returns:
        Selected scope index per head, each in ``[0, S-1]``.

    Raises:
        RuntimeError: if no feasible assignment exists (budget below the sum
            of per-head minimum costs).
    """
    if quality_cost.dim() != 2:
        raise ValueError(
            f"quality_cost must be (H, S), got {tuple(quality_cost.shape)}"
        )
    if compute_cost.shape != quality_cost.shape:
        raise ValueError(
            f"compute_cost shape {tuple(compute_cost.shape)} != quality_cost "
            f"shape {tuple(quality_cost.shape)}"
        )
    if not (0.0 < budget_ratio <= 1.0):
        raise ValueError(f"budget_ratio must be in (0, 1], got {budget_ratio}")
    if bins < 1:
        raise ValueError(f"bins must be >= 1, got {bins}")

    H, S = quality_cost.shape
    full_cost = float(compute_cost[:, -1].sum().item())
    budget = budget_ratio * full_cost

    # Convert floating costs to small integer bins.
    scale = bins / max(budget, 1e-12)
    cost_int = torch.ceil(compute_cost.double() * scale).to(torch.long)
    budget_int = int(math.floor(budget * scale))

    # W2.7 FIX (2026-08-25, boundary-rounding false infeasibility): ceil
    # discretization adds < 1 bin of rounding PER HEAD, so a truly-feasible
    # assignment can sum to up to ``budget_int + (H - 1)`` bins and was
    # wrongly rejected as infeasible.  Production symptom: any head count H
    # that does not divide the geometry evenly crashed with "No feasible HAP
    # scope assignment found." at high budget ratios (budget_ratio=1.0 with
    # H=3, seq=256: full-everywhere sums to 4002 bins > budget_int=4000).
    #
    # Give the DP an explicit H-bin rounding allowance: every truly-feasible
    # assignment becomes binned-feasible.  Cost: the returned assignment's
    # TRUE cost may exceed the budget by at most ``H * budget / bins`` — the
    # exact slack already documented (and asserted) in
    # tests/test_hap_knapsack.py::TestKnapsack::test_budget_respected.
    capacity = budget_int + H

    inf = float("inf")
    dp = torch.full((capacity + 1,), inf, dtype=torch.float64)
    dp[0] = 0.0

    parents: List[Tuple[torch.Tensor, torch.Tensor]] = []

    for h in range(H):
        next_dp = torch.full_like(dp, inf)
        parent_scope = torch.full((capacity + 1,), -1, dtype=torch.long)
        parent_budget = torch.full((capacity + 1,), -1, dtype=torch.long)

        for s in range(S):
            c = int(cost_int[h, s].item())
            q = float(quality_cost[h, s].item())

            if c > capacity:
                continue

            prior = dp[: capacity + 1 - c]
            candidate = prior + q
            target = next_dp[c:]

            improved = candidate < target
            target[improved] = candidate[improved]

            idx = torch.arange(capacity + 1 - c)[improved]
            parent_scope[c + idx] = s
            parent_budget[c + idx] = idx

        dp = next_dp
        parents.append((parent_scope, parent_budget))

    if bool(torch.isinf(dp).all().item()):
        raise RuntimeError("No feasible HAP scope assignment found.")

    remaining = int(torch.argmin(dp).item())
    choices: List[int] = []

    for h in reversed(range(H)):
        scope, previous = parents[h]
        selected = int(scope[remaining].item())
        if selected < 0:
            raise RuntimeError("No feasible HAP scope assignment found.")

        choices.append(selected)
        remaining = int(previous[remaining].item())

    return list(reversed(choices))


# ---------------------------------------------------------------------------
# P5/T5.5 — End-to-end pure-math calibration
# ---------------------------------------------------------------------------

def calibrate_scope_plan(
    quality_costs: Sequence[Sequence[torch.Tensor]],
    compute_costs: Sequence[torch.Tensor],
    budget_ratio: float = 0.10,
    bins: int = 4000,
) -> Dict[str, List[List[float]]]:
    """Pure-math calibration: per-prompt score tables → scope-plan dict.

    Args:
        quality_costs: ``[layer][prompt]`` tensors of shape
            ``(H, S)`` — Taylor quality costs from
            :func:`estimate_head_scope_costs`, one per calibration prompt.
            Averaged across prompts (paper: 30 prompts).
        compute_costs: ``[layer]`` tensors of shape ``(H, S)`` — from
            :func:`calibration_cost_table` (prompt-independent).
        budget_ratio: attention cost ratio ``r_c`` (paper: 0.1).
        bins: knapsack discretization resolution.

    Returns:
        Reference-format dict ``{"alphas": [[...]], "betas": [[...]]}``
        (per layer, per head) with ``alpha = 0`` and
        ``beta = 0.5 · s / N_scope`` for the solved scope ``s`` — directly
        loadable by :class:`src.hap.ScopePlan` and consumable by the runtime
        with zero conversion (plan §2.4).
    """
    num_layers = len(quality_costs)
    if num_layers == 0:
        raise ValueError("quality_costs must contain at least one layer")
    if len(compute_costs) != num_layers:
        raise ValueError(
            f"compute_costs has {len(compute_costs)} layers, expected "
            f"{num_layers}"
        )

    alphas: List[List[float]] = []
    betas: List[List[float]] = []

    for layer in range(num_layers):
        prompts = quality_costs[layer]
        if len(prompts) == 0:
            raise ValueError(f"layer {layer}: no per-prompt quality costs")

        ref_shape = prompts[0].shape
        for p, table in enumerate(prompts):
            if table.shape != ref_shape:
                raise ValueError(
                    f"layer {layer} prompt {p}: shape {tuple(table.shape)} "
                    f"!= {tuple(ref_shape)}"
                )
        cc = compute_costs[layer]
        if cc.shape != ref_shape:
            raise ValueError(
                f"layer {layer}: compute_costs shape {tuple(cc.shape)} != "
                f"quality_costs shape {tuple(ref_shape)}"
            )

        num_scopes = ref_shape[-1]
        quality_avg = torch.stack(
            [t.detach().to(torch.float64) for t in prompts], dim=0
        ).mean(dim=0)

        # DIAGNOSTIC (false-infeasibility root cause): the knapsack DP treats a
        # NaN/inf quality entry as "never an improvement" (``NaN < x`` is False),
        # so a single all-NaN head leaves the DP all-inf and the solver raises
        # "No feasible HAP scope assignment found" EVEN WHEN THE BUDGET IS AMPLE
        # (validated: cost/budget feasibility is scale-invariant, and finite
        # quality — even 1e30 — is always feasible).  Log NaN/inf counts so a
        # poisoned Taylor table is visible before the solver runs.
        n_nan = int(torch.isnan(quality_avg).sum().item())
        n_inf = int(torch.isinf(quality_avg).sum().item())
        if n_nan or n_inf:
            logger.warning(
                "[HAP calib] layer %d quality table has %d NaN and %d inf "
                "entries (of %d).  NaN/inf Taylor scores poison the knapsack DP "
                "and cause a false 'No feasible HAP scope assignment' even when "
                "the budget is sufficient.  This usually comes from bf16 "
                "attention/gradient overflow feeding the score computation.",
                layer, n_nan, n_inf, quality_avg.numel(),
            )

            # SAFETY NET: replace NaN/inf with a large finite penalty so the
            # solver degrades gracefully instead of crashing.  The penalty is
            # the max finite score + 1 (or 1e30 if all entries are NaN/inf),
            # which marks poisoned entries as worst-quality-but-valid.  The
            # solver will avoid them when possible; if every entry is poisoned
            # the solver still runs (all entries equal penalty) and returns a
            # valid assignment.
            finite_mask = torch.isfinite(quality_avg)
            if bool(finite_mask.any()):
                max_finite = float(quality_avg[finite_mask].max().item())
                penalty = max_finite + 1.0
            else:
                penalty = 1e30
            quality_avg = torch.where(
                finite_mask, quality_avg,
                torch.full_like(quality_avg, penalty),
            )

        chosen = solve_multiple_choice_knapsack(
            quality_avg, cc, budget_ratio=budget_ratio, bins=bins
        )
        alphas.append([0.0] * len(chosen))
        betas.append([scope_to_beta(s + 1, num_scopes) for s in chosen])

    return {"alphas": alphas, "betas": betas}


# ---------------------------------------------------------------------------
# P6/T6.1 — Chunked differentiable attention (gradient path)
# ---------------------------------------------------------------------------

def chunked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float = 1.0,
    chunk: int = 256,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Calibration-only attention that exposes per-chunk ``A.grad``.

    Splits the query rows into chunks of ``chunk``, and for each chunk
    materializes ``A_chunk = softmax(q_c kᵀ · scale)`` as a ``requires_grad``
    LEAF, then computes ``out_chunk = A_chunk @ v``.  The forward output is
    identical to (single-shot) scaled-dot-product attention with the same
    ``scale``; the difference is that each ``A_chunk`` is a leaf, so after a
    scalar ``loss.backward()`` every chunk carries ``A_chunk.grad == dL/dA``
    for its rows — without ever holding a dense ``T×T`` matrix for the whole
    sequence at once (plan §2.5, T6.1).

    Making ``A_chunk`` a leaf (``softmax(...).detach().requires_grad_(True)``)
    deliberately cuts the gradient path to ``q/k/v``: calibration only needs
    ``dL/dA``, and this keeps the autograd graph tiny regardless of how the
    surrounding model produced ``q/k/v``.

    Args:
        q, k, v: ``(B, H, T, D)`` tensors (the ``optimized_attention`` layout).
        scale: attention scale (must match the reference being calibrated
            against; the ComfyUI mock uses ``1.0``).
        chunk: query rows per chunk.  ``chunk >= T`` degenerates to a single
            dense chunk (used as the test oracle).

    Returns:
        ``(out, chunks)`` where ``out`` is ``(B, H, T, D)`` and ``chunks`` is
        the row-ordered list of ``(B, H, C, T)`` attention leaves.
    """
    if q.dim() != 4:
        raise ValueError(f"q must be (B, H, T, D), got {tuple(q.shape)}")
    if k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            f"k/v shape {tuple(k.shape)}/{tuple(v.shape)} != q shape {tuple(q.shape)}"
        )
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")

    # NOTE (2026-08-18, P19 — fp32 logits): the live Krea2 run #8 PROVED the
    # NaN origin: the model runs fp16 with |q|,|k| ~ 600 and head_dim=128, so
    # the fp16 ``q @ kᵀ`` dot product reaches ~128 * 600^2 ~= 4.6e7 — ~700x
    # OVER fp16's max (65504) -> ``inf`` logits -> ``softmax(inf - inf) = NaN``
    # rows -> the forward NaN cascade.  ComfyUI's own ``attention_basic``
    # avoids this by computing ``einsum(q.float(), k.float())`` — the logits
    # are ALWAYS fp32 there (flash/SDPA likewise accumulate in fp32).
    #
    # Fix: compute the logits + softmax in fp32 (up-casting fp16/bf16 only;
    # NEVER down-casting fp32/fp64), then cast ``A`` back to the model dtype
    # for storage and ``A @ v`` (a convex combination of ``v`` — safe in the
    # model dtype, exactly what flash attention does after its fp32 softmax).
    #
    # VRAM: the fp32 logits are TRANSIENT (one chunk at a time, freed each
    # iteration) and the STORED ``A`` leaves stay in the model dtype, so the
    # retained footprint is UNCHANGED (3.6 GiB at seq=1198/H=48).  This is the
    # crucial difference from the reverted P10 upcast, which STORED ``A`` in
    # fp32 (2x retained memory -> OOM).
    _, _, T, _ = q.shape
    compute_dtype = (
        torch.float32
        if q.dtype in (torch.float16, torch.bfloat16)
        else q.dtype
    )
    k_compute = k.to(compute_dtype)  # cast once, reuse across chunks
    outs: List[torch.Tensor] = []
    chunks: List[torch.Tensor] = []
    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        q_c = q[:, :, start:end, :]                              # (B, H, C, D)
        logits = torch.matmul(
            q_c.to(compute_dtype), k_compute.transpose(-1, -2)
        ) * scale                                                # (B, H, C, T) fp32
        A = torch.softmax(logits, dim=-1)
        A = A.to(q.dtype).detach().requires_grad_(True)          # stored in model dtype
        out_c = torch.matmul(A, v)                               # (B, H, C, D)
        outs.append(out_c)
        chunks.append(A)
    out = torch.cat(outs, dim=2)                                 # (B, H, T, D)
    return out, chunks


# ---------------------------------------------------------------------------
# P6/T6.2 — Stats collector over a calibration forward
# ---------------------------------------------------------------------------

def collect_scope_scores(
    model_forward,
    loss_fn,
    num_scopes: int,
    text_len: int = 0,
    chunk: int = 256,
    scale: float = 1.0,
    attn_module=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect per-(layer, head, scope) Taylor scores from ONE backward pass.

    Installs :func:`chunked_attention` as the module-level
    ``optimized_attention`` for the duration of ``model_forward()``, runs
    ``loss_fn(output).backward()``, then accumulates the T5.2 scores from each
    layer's chunk ``(A, A.grad)`` pairs via the shared
    :func:`_chunk_row_scores` (exact row-slice sums — chunking never changes
    the result, only the peak memory).  Finally restores the original
    ``optimized_attention`` (plan T6.2).

    Args:
        model_forward: zero-arg callable running one calibration forward.
        loss_fn: ``output -> scalar loss`` (differentiable).
        num_scopes: candidate scopes ``N_scope``.
        text_len: leading text tokens (never omitted).
        chunk: query rows per chunk (memory knob; result-invariant).
        scale: attention scale matching the model's ``optimized_attention``.
        attn_module: module exposing ``optimized_attention``; defaults to
            ``sys.modules["comfy.ldm.modules.attention"]``.

    Returns:
        ``(quality_cost, compute_cost)`` — both ``(L, H, S)`` fp64 tensors.
        ``quality_cost[l]`` is the per-prompt table for layer ``l`` (this call
        runs one forward ⇒ one prompt); wrap as
        ``[[quality_cost[l]] for l in range(L)]`` to feed
        :func:`calibrate_scope_plan`.  ``compute_cost`` is prompt-independent
        (from :func:`calibration_cost_table`) and identical across layers.
    """
    import sys

    if attn_module is None:
        attn_module = sys.modules["comfy.ldm.modules.attention"]
    if num_scopes < 1:
        raise ValueError(f"num_scopes must be >= 1, got {num_scopes}")

    orig = attn_module.optimized_attention
    records: List[List[torch.Tensor]] = []  # records[layer] = chunk leaves

    def chunked_attn(q, k, v, heads, *args, **kwargs):
        out, chunks = chunked_attention(q, k, v, scale=scale, chunk=chunk)
        records.append(chunks)
        return out

    attn_module.optimized_attention = chunked_attn
    try:
        output = model_forward()
        # W2.7 FIX (2026-08-25): raise the "no calls" error BEFORE backward.
        # The pre-fix order ran ``loss.backward()`` first, so a forward that
        # never called optimized_attention crashed with an unrelated autograd
        # error ("element 0 of tensors does not require grad") instead of the
        # intended actionable message.
        if not records:
            raise RuntimeError(
                "collect_scope_scores: model_forward made no optimized_attention "
                "calls — nothing to calibrate."
            )
        loss = loss_fn(output)
        loss.backward()
    finally:
        attn_module.optimized_attention = orig

    # OBSERVED GEOMETRY + TEXT_LEN CLAMP (text_len>seq_len root cause).  The
    # cost model requires ``text_len <= seq_len`` (seq = text + image).  The
    # node's ``text_len`` knob is a FLUX-ism default (512) that can exceed the
    # observed sequence when calibration runs at a reduced resolution (OOM
    # workaround) or the model's real text length is below the knob.  Clamp to
    # ``[0, seq0]`` — mirroring the HAP runtime's ``max(0, min(text_len,
    # seq_len))`` (src/hap.py HapRuntime.attn) — so a knob mismatch degrades
    # gracefully instead of crashing ``band_compute_cost``.  The clamped value
    # is threaded through BOTH the quality-scoring loop (_chunk_row_scores) and
    # the cost table so the two stay consistent.
    heads0 = records[0][0].shape[1]
    seq0 = records[0][0].shape[3]
    eff_text_len = max(0, min(int(text_len), seq0))
    if eff_text_len != int(text_len):
        logger.warning(
            "[HAP calib] text_len knob (%d) exceeds the observed attention "
            "sequence length (%d) — clamped to %d.  This usually means the "
            "calibration resolution is too small for the configured text_len, "
            "or the model's real text length is below the knob.  Consider "
            "raising width/height or lowering text_len.",
            int(text_len), seq0, eff_text_len,
        )

    quality_layers: List[torch.Tensor] = []
    for li, layer_chunks in enumerate(records):
        heads = layer_chunks[0].shape[1]
        acc = torch.zeros(heads, num_scopes, dtype=torch.float64)
        offset = 0
        for A_chunk in layer_chunks:
            if A_chunk.grad is None:
                raise RuntimeError(
                    f"collect_scope_scores: layer {li} attention chunk has no "
                    "gradient — the loss does not depend on it."
                )
            rows = A_chunk.shape[2]
            # DEVICE-CONSISTENT SCORING: leaves live on the model's device
            # (cuda:0 in ComfyUI) but ``acc`` and the downstream pipeline are
            # CPU-resident.  Move A/G to CPU so the accumulation never mixes
            # devices (``.to(dtype)`` alone preserves device -> cuda:0 vs cpu
            # crash) and the fp64 intermediates are scored off-GPU.
            A = A_chunk[0].detach().to(dtype=torch.float64, device="cpu")        # (H, C, T)
            G = A_chunk.grad[0].detach().to(dtype=torch.float64, device="cpu")   # (H, C, T)
            acc = acc + _chunk_row_scores(A, G, num_scopes, eff_text_len, offset)
            offset += rows
        quality_layers.append(acc)
    quality_cost = torch.stack(quality_layers, dim=0)  # (L, H, S)

    # Prompt-independent compute cost (same for every layer).
    cost = calibration_cost_table(
        heads0, seq0, text_len=eff_text_len, num_scopes=num_scopes
    )  # (H, S)
    compute_cost = cost.unsqueeze(0).expand(quality_cost.shape[0], -1, -1).clone()

    return quality_cost, compute_cost
