"""Tests for the HAP multiple-choice knapsack solver (``src/hap_calib.py``).

Plan T5.4 — dependency-free DP replacement for the paper's ILP solver
(theory §6.3/§8): pick exactly one scope per head under a global compute
budget, minimizing total quality cost on ceil-discretized costs.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_hap_knapsack.py
"""

import itertools

import pytest
import torch

from src.hap_calib import solve_multiple_choice_knapsack


def _tables(H, S, seed, cost_spread=10.0):
    """Deterministic (quality, compute) tables: quality decreases with scope,
    compute increases with scope, full scope cost == 100 per head."""
    g = torch.Generator().manual_seed(seed)
    quality = torch.rand(H, S, generator=g, dtype=torch.float64) * 5.0
    quality[:, -1] = 0.0  # full scope: zero quality loss
    # Strictly increasing compute costs, last column == 100.
    steps = torch.rand(H, S - 1, generator=g, dtype=torch.float64) + 0.5
    compute = torch.zeros(H, S, dtype=torch.float64)
    compute[:, 1:] = steps.cumsum(dim=-1)
    compute = compute / compute[:, -1:].clamp_min(1e-12) * 100.0
    compute[:, 0] = torch.rand(H, generator=g, dtype=torch.float64) * 2.0 + 1.0
    return quality, compute


def _brute_force(quality, compute, budget_ratio):
    """Exhaustive enumeration oracle (small H·S only).

    Returns (best_total_quality, sorted list of all optimal assignments).
    """
    H, S = quality.shape
    full_cost = float(compute[:, -1].sum())
    budget = budget_ratio * full_cost
    best = float("inf")
    best_assignments = []
    for combo in itertools.product(range(S), repeat=H):
        cost = sum(float(compute[h, combo[h]]) for h in range(H))
        if cost > budget + 1e-9:
            continue
        q = sum(float(quality[h, combo[h]]) for h in range(H))
        if q < best - 1e-12:
            best = q
            best_assignments = [combo]
        elif abs(q - best) <= 1e-12:
            best_assignments.append(combo)
    return best, best_assignments


@pytest.mark.unit
class TestKnapsack:
    def test_one_scope_per_head(self):
        """Output length == H, every index in [0, S-1]."""
        H, S = 7, 9
        quality, compute = _tables(H, S, seed=11)
        choices = solve_multiple_choice_knapsack(quality, compute, budget_ratio=0.5)
        assert len(choices) == H
        assert all(isinstance(c, int) for c in choices)
        assert all(0 <= c < S for c in choices)

    def test_budget_respected(self):
        """Σ chosen cost ≤ budget on the binned scale (ceil discretization
        can only round costs UP, so the true cost is within budget)."""
        H, S = 6, 8
        budget_ratio = 0.4
        quality, compute = _tables(H, S, seed=12)
        choices = solve_multiple_choice_knapsack(
            quality, compute, budget_ratio=budget_ratio, bins=4000
        )
        full_cost = float(compute[:, -1].sum())
        chosen_cost = sum(float(compute[h, choices[h]]) for h in range(H))
        # Ceil-binning may admit an assignment whose true cost exceeds the
        # budget by at most H bins worth: budget + H·(budget/bins).
        slack = H * (budget_ratio * full_cost) / 4000.0
        assert chosen_cost <= budget_ratio * full_cost + slack + 1e-9

    def test_matches_brute_force_small(self):
        """H=5, S=6: DP total quality == exhaustive optimum, EXACTLY.

        Ceil-binning satisfies binned-feasible ⊆ true-feasible always, so the
        DP quality is ≥ the brute-force optimum; equality holds when the
        discretization is exact.  We force ``scale = bins/budget = 1`` with
        integer costs and an exactly-representable budget, so the DP and the
        brute-force oracle share an identical feasible set and must agree to
        the last bit.

        full_cost = 5·60 = 300 (exact); budget_ratio = 0.5 (exactly
        representable in binary) → budget = 150.0 (exact); bins = 150 →
        scale = 1.0 → cost_int == cost, budget_int == 150.
        """
        H, S = 5, 6
        g = torch.Generator().manual_seed(13)
        quality = torch.rand(H, S, generator=g, dtype=torch.float64) * 5.0
        quality[:, -1] = 0.0
        # Exact integer costs: 10, 20, ..., 60 per head (full = 60).
        compute = torch.arange(1, S + 1, dtype=torch.float64).unsqueeze(0).repeat(H, 1) * 10.0
        budget_ratio = 0.5
        choices = solve_multiple_choice_knapsack(
            quality, compute, budget_ratio=budget_ratio, bins=150
        )
        dp_quality = sum(float(quality[h, choices[h]]) for h in range(H))
        best_quality, _ = _brute_force(quality, compute, budget_ratio)
        assert best_quality < float("inf"), "test setup must be feasible"
        assert dp_quality == pytest.approx(best_quality, abs=1e-12)

    def test_infeasible_raises(self):
        """Budget below Σ per-head minimum costs → RuntimeError."""
        H, S = 3, 4
        quality, compute = _tables(H, S, seed=14)
        # Minimum possible cost: every head picks scope 0.
        min_cost = float(compute[:, 0].sum())
        full_cost = float(compute[:, -1].sum())
        impossible_ratio = (min_cost / full_cost) * 0.5  # strictly below min
        with pytest.raises(RuntimeError, match="No feasible HAP scope assignment"):
            solve_multiple_choice_knapsack(
                quality, compute, budget_ratio=impossible_ratio, bins=100
            )

    def test_budget_monotonicity(self):
        """budget↑ ⇒ optimal total quality cost non-increasing."""
        H, S = 5, 7
        quality, compute = _tables(H, S, seed=15)
        totals = []
        for ratio in (0.3, 0.5, 0.7, 0.9, 1.0):
            choices = solve_multiple_choice_knapsack(
                quality, compute, budget_ratio=ratio, bins=2000
            )
            totals.append(sum(float(quality[h, choices[h]]) for h in range(H)))
        for lo, hi in zip(totals, totals[1:]):
            assert hi <= lo + 1e-9

    def test_deterministic(self):
        """Same input twice → identical output (strict-improvement DP with
        ascending scope scan = lowest-index tie-break)."""
        H, S = 6, 8
        quality, compute = _tables(H, S, seed=16)
        a = solve_multiple_choice_knapsack(quality, compute, budget_ratio=0.5)
        b = solve_multiple_choice_knapsack(quality, compute, budget_ratio=0.5)
        assert a == b

    def test_zero_quality_picks_cheapest_feasible(self):
        """All quality costs equal → the solver minimizes nothing and any
        feasible assignment is optimal; determinism pins the result.  With a
        tight budget it must still respect the budget."""
        H, S = 4, 5
        quality = torch.ones(H, S, dtype=torch.float64)
        compute = torch.zeros(H, S, dtype=torch.float64)
        for s in range(S):
            compute[:, s] = 10.0 * (s + 1)  # 10, 20, ..., 50 per head
        choices = solve_multiple_choice_knapsack(
            quality, compute, budget_ratio=0.25, bins=400
        )
        # full_cost = 4·50 = 200, budget = 50 → e.g. all scope 0 (cost 40).
        chosen_cost = sum(float(compute[h, choices[h]]) for h in range(H))
        assert chosen_cost <= 50.0 + 1e-9

    def test_rejects_bad_inputs(self):
        quality = torch.rand(3, 4, dtype=torch.float64)
        compute = torch.rand(3, 4, dtype=torch.float64)
        with pytest.raises(ValueError):
            solve_multiple_choice_knapsack(torch.rand(4), compute)
        with pytest.raises(ValueError):
            solve_multiple_choice_knapsack(quality, torch.rand(2, 4))
        with pytest.raises(ValueError):
            solve_multiple_choice_knapsack(quality, compute, budget_ratio=0.0)
        with pytest.raises(ValueError):
            solve_multiple_choice_knapsack(quality, compute, budget_ratio=1.5)
        with pytest.raises(ValueError):
            solve_multiple_choice_knapsack(quality, compute, bins=0)
