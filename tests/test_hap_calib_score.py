"""Tests for the HAP calibration math (``src/hap_calib.py``).

Plan phase P5:
- T5.1 ``taylor_softmax_pruning_score`` (theory §6.2/§8 formula),
- T5.2 ``estimate_head_scope_costs`` (vectorized nested-band scoring),
- T5.3 ``calibration_cost_table`` (scope → beta → band cost),
- T5.5 ``calibrate_scope_plan`` (end-to-end pure-math calibration).

The knapsack solver (T5.4) is tested separately in ``test_hap_knapsack.py``.

Markers: @pytest.mark.unit
Accept (user-run):
    pytest tests/test_hap_calib_score.py -k taylor
    pytest tests/test_hap_calib_score.py -k vectorized
    pytest tests/test_hap_calib_score.py -k cost_table
    pytest tests/test_hap_calib_score.py -k calibrate
"""

import pytest
import torch

from src import hap
from src.hap_calib import (
    calibrate_scope_plan,
    calibration_cost_table,
    estimate_head_scope_costs,
    scope_keep_mask,
    scope_to_beta,
    taylor_softmax_pruning_score,
)


def _softmax_rows(logits: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax (fp64) producing a valid attention matrix."""
    x = logits - logits.max(dim=-1, keepdim=True).values
    e = x.exp()
    return e / e.sum(dim=-1, keepdim=True)


def _entrywise_oracle(A: torch.Tensor, G: torch.Tensor, keep: torch.Tensor,
                      eps: float = 1e-8) -> float:
    """Literal entry-wise evaluation of the theory §8 formula (test oracle).

    score = Σ_{(u,v)∈omit} [ G(u,v)·(−A(u,v))
            + (Σ_{w≠v} G(u,w)·A(u,w)) · A(u,v)/(1−A(u,v)) ]
    """
    T = A.shape[0]
    total = 0.0
    for u in range(T):
        row_dot = sum(float(G[u, w] * A[u, w]) for w in range(T))
        for v in range(T):
            if bool(keep[u, v]):
                continue
            direct = float(G[u, v]) * (-float(A[u, v]))
            excl = row_dot - float(G[u, v] * A[u, v])
            renorm = excl * float(A[u, v]) / max(1.0 - float(A[u, v]), eps)
            total += direct + renorm
    return total


# ---------------------------------------------------------------------------
# T5.1 — taylor_softmax_pruning_score
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTaylorScore:
    def test_score_hand_computed_4x4(self):
        """4×4 A/G with a hand-derived expected value.

        A (rows sum to 1):                G:
            [0.50 0.25 0.15 0.10]            [ 1.0  2.0  3.0  4.0]
            [0.10 0.20 0.30 0.40]            [-1.0  0.5  1.5 -0.5]
            [0.05 0.05 0.80 0.10]            [ 2.0 -2.0  1.0  0.0]
            [0.25 0.25 0.25 0.25]            [ 0.0  1.0 -1.0  2.0]

        keep = all True except row 0 cols {1, 3} and row 2 col {0}.

        Per omitted (u,v):  −G·A + (row_dot − G·A)·A/(1−A),
        row_dot(u) = Σ_w G(u,w)·A(u,w).

        Row 0 (row_dot = 1.0·.50 + 2.0·.25 + 3.0·.15 + 4.0·.10 = 1.85):
          v=1: −2.0·.25 + (1.85−.50)·.25/.75 = −.50 + 1.35·.25/.75 = −.05
          v=3: −4.0·.10 + (1.85−.40)·.10/.90 = −.40 + 1.45·.10/.90
        Row 2 (row_dot = 2.0·.05 − 2.0·.05 + 1.0·.80 + 0 = 0.80):
          v=0: −2.0·.05 + (0.80−.10)·.05/.95 = −.10 + 0.70·.05/.95
        """
        A = torch.tensor(
            [
                [0.50, 0.25, 0.15, 0.10],
                [0.10, 0.20, 0.30, 0.40],
                [0.05, 0.05, 0.80, 0.10],
                [0.25, 0.25, 0.25, 0.25],
            ],
            dtype=torch.float64,
        )
        G = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [-1.0, 0.5, 1.5, -0.5],
                [2.0, -2.0, 1.0, 0.0],
                [0.0, 1.0, -1.0, 2.0],
            ],
            dtype=torch.float64,
        )
        keep = torch.ones(4, 4, dtype=torch.bool)
        keep[0, 1] = False
        keep[0, 3] = False
        keep[2, 0] = False

        score = taylor_softmax_pruning_score(A, G, keep)

        # Exact formula terms (row_dot row0 = 1.85, row_dot row2 = 0.80).
        expected = (
            (-0.50 + 1.35 * 0.25 / 0.75)    # row 0, v=1
            + (-0.40 + 1.45 * 0.10 / 0.90)  # row 0, v=3
            + (-0.10 + 0.70 * 0.05 / 0.95)  # row 2, v=0
        )
        assert float(score) == pytest.approx(expected, abs=1e-12)
        # Cross-check against the literal entry-wise oracle.
        assert float(score) == pytest.approx(_entrywise_oracle(A, G, keep), abs=1e-12)

    def test_score_keep_all_is_zero(self):
        """All-True keep mask → nothing omitted → exactly 0."""
        torch.manual_seed(0)
        A = _softmax_rows(torch.randn(8, 8, dtype=torch.float64))
        G = torch.randn(8, 8, dtype=torch.float64)
        keep = torch.ones(8, 8, dtype=torch.bool)
        score = taylor_softmax_pruning_score(A, G, keep)
        assert float(score) == 0.0

    def test_score_equals_sum_of_single_removals(self):
        """The paper's summand is the EXACT loss change of removing one
        omitted entry (u,v) and renormalizing row u — summed over the omit
        set.  With a linear loss L(A) = Σ G·A this identity is exact (fp64):

            score == Σ_{(u,v)∈omit} [ L(A^{−(u,v)}) − L(A) ]

        where A^{−(u,v)} zeroes A(u,v) and rescales the kept entries of row u
        by 1/(1−A(u,v)).  (Removing entries one-at-a-time is exact; the
        "approximation" in the paper is only that joint removal interactions
        are ignored, which is second-order.)
        """
        torch.manual_seed(1)
        T = 16
        A = _softmax_rows(torch.randn(T, T, dtype=torch.float64))
        G = torch.randn(T, T, dtype=torch.float64)
        keep = scope_keep_mask(T, 6, 10)  # partial scope → non-empty omit set

        score = float(taylor_softmax_pruning_score(A, G, keep))

        total = 0.0
        omitted = (~keep).nonzero()
        assert omitted.numel() > 0, "test needs a non-empty omit set"
        for u, v in omitted.tolist():
            A_single = A.clone()
            kept_mass = 1.0 - float(A_single[u, v])
            A_single[u, :] = A_single[u, :] / kept_mass
            A_single[u, v] = 0.0
            total += float((G * (A_single - A)).sum())

        assert score == pytest.approx(total, abs=1e-10)

    def test_score_first_order_small_omitted_mass(self):
        """When the omitted entries carry tiny mass, the score also matches
        the EXACT change of a single simultaneous prune-and-renormalize —
        the two notions agree to first order in the omitted mass (the paper's
        "approximation" is precisely the ignored second-order interaction).

        Construction: strongly peaked logits (≈ one-hot on the diagonal) so
        every off-diagonal entry has mass ε ≈ e^{-10}; omit all off-diagonal
        entries.  Omitted row mass M_u ≈ (T−1)ε is tiny, so the sum of
        single-removal terms and the simultaneous change differ only by
        O(M_u²) — a relative ~1e-4 effect, asserted at rel 1e-2.
        """
        torch.manual_seed(9)
        T = 8
        logits = 10.0 * torch.eye(T, dtype=torch.float64) + 0.01 * torch.randn(T, T, dtype=torch.float64)
        A = _softmax_rows(logits)
        G = torch.randn(T, T, dtype=torch.float64)
        keep = torch.eye(T, dtype=torch.bool)  # keep only the (dominant) diagonal

        score = float(taylor_softmax_pruning_score(A, G, keep))
        A_pruned = torch.where(keep, A, torch.zeros_like(A))
        A_renorm = A_pruned / A_pruned.sum(dim=-1, keepdim=True)
        simultaneous = float((G * (A_renorm - A)).sum())

        # Both are O(ε); they agree to first order in the omitted mass.
        assert abs(score) > 1e-8, "score must be resolvable at this mass"
        assert score == pytest.approx(simultaneous, rel=1e-2)

    def test_score_eps_guard_no_nan(self):
        """A saturating row (A(u,v) = 1.0) must not divide by zero."""
        A = torch.zeros(4, 4, dtype=torch.float64)
        A[0, 0] = 1.0  # saturated row: 1 − A = 0 → clamped by eps
        A[1] = 0.25
        A[2] = torch.tensor([0.5, 0.2, 0.2, 0.1], dtype=torch.float64)
        A[3] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
        G = torch.ones(4, 4, dtype=torch.float64)
        keep = torch.ones(4, 4, dtype=torch.bool)
        keep[0, 0] = False  # omit the saturated entry

        score = taylor_softmax_pruning_score(A, G, keep)
        assert torch.isfinite(score)
        # direct = −1·1 = −1; renorm = (row_dot − 1)·1/eps = 0 → score = −1.
        assert float(score) == pytest.approx(-1.0, abs=1e-9)

    def test_score_rejects_bad_shapes(self):
        A = torch.rand(4, 4, dtype=torch.float64)
        G = torch.rand(4, 4, dtype=torch.float64)
        keep = torch.ones(4, 4, dtype=torch.bool)
        with pytest.raises(ValueError):
            taylor_softmax_pruning_score(torch.rand(4, 5), G, keep)
        with pytest.raises(ValueError):
            taylor_softmax_pruning_score(A, torch.rand(3, 4), keep)
        with pytest.raises(ValueError):
            taylor_softmax_pruning_score(A, G, torch.ones(3, 4, dtype=torch.bool))


# ---------------------------------------------------------------------------
# T5.2 — vectorized nested-band scoring
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVectorizedScoring:
    def test_vectorized_equals_naive_loop(self):
        """seq=96, H=4, N_scope=10, fp64: vectorized == naive per-scope loop
        over ``taylor_softmax_pruning_score`` with the shared keep mask."""
        torch.manual_seed(2)
        H, T, S = 4, 96, 10
        A = _softmax_rows(torch.randn(H, T, T, dtype=torch.float64))
        G = torch.randn(H, T, T, dtype=torch.float64)

        scores = estimate_head_scope_costs(A, G, num_scopes=S, text_len=0)
        assert scores.shape == (H, S)

        for h in range(H):
            for s in range(1, S + 1):
                keep = scope_keep_mask(T, s, S)
                expected = taylor_softmax_pruning_score(A[h], G[h], keep)
                assert float(scores[h, s - 1]) == pytest.approx(
                    float(expected), abs=1e-10
                )

    def test_vectorized_equals_naive_loop_with_text(self):
        """Text rows/columns never contribute to the omitted sum."""
        torch.manual_seed(3)
        H, T, S, text_len = 3, 80, 8, 16
        A = _softmax_rows(torch.randn(H, T, T, dtype=torch.float64))
        G = torch.randn(H, T, T, dtype=torch.float64)

        scores = estimate_head_scope_costs(A, G, num_scopes=S, text_len=text_len)

        for h in range(H):
            for s in range(1, S + 1):
                keep = scope_keep_mask(T, s, S, text_len=text_len)
                expected = taylor_softmax_pruning_score(A[h], G[h], keep)
                assert float(scores[h, s - 1]) == pytest.approx(
                    float(expected), abs=1e-10
                )

    def test_scores_structural_identity(self):
        """Structural identity: score[h,s] == Σ_{(u,v) omitted} C[u,v] where
        C = direct + renorm is the theory §8 summand.  Also the full scope
        scores exactly 0 and all scores are finite.

        NOTE: the Taylor summand is NOT sign-definite (it satisfies the row
        identity Σ_v C[u,v]·(1−A[u,v]) ≡ 0), so naive score monotonicity in
        the scope is NOT a valid invariant and is deliberately not asserted.
        """
        torch.manual_seed(4)
        H, T, S = 2, 64, 12
        A = _softmax_rows(torch.randn(H, T, T, dtype=torch.float64))
        G = torch.randn(H, T, T, dtype=torch.float64)

        scores = estimate_head_scope_costs(A, G, num_scopes=S, text_len=0)

        # Full scope keeps everything → exactly zero.
        assert torch.all(scores[:, -1] == 0.0)
        assert bool(torch.isfinite(scores).all())

        # Rebuild C explicitly and check the omitted-sum identity.
        direct = -G * A
        row_dot = (G * A).sum(dim=-1, keepdim=True)
        renorm = (row_dot - G * A) * A / (1.0 - A).clamp_min(1e-8)
        C = direct + renorm
        for h in range(H):
            for s in range(1, S + 1):
                keep = scope_keep_mask(T, s, S)
                expected = float(C[h][~keep].sum())
                assert float(scores[h, s - 1]) == pytest.approx(expected, abs=1e-10)

    def test_full_scope_is_zero_with_text(self):
        torch.manual_seed(5)
        H, T, S, text_len = 2, 48, 6, 8
        A = _softmax_rows(torch.randn(H, T, T, dtype=torch.float64))
        G = torch.randn(H, T, T, dtype=torch.float64)
        scores = estimate_head_scope_costs(A, G, num_scopes=S, text_len=text_len)
        assert torch.all(scores[:, -1] == 0.0)

    def test_rejects_bad_inputs(self):
        A = torch.rand(2, 8, 8, dtype=torch.float64)
        G = torch.rand(2, 8, 8, dtype=torch.float64)
        with pytest.raises(ValueError):
            estimate_head_scope_costs(torch.rand(8, 8), G)
        with pytest.raises(ValueError):
            estimate_head_scope_costs(A, torch.rand(2, 8, 7))
        with pytest.raises(ValueError):
            estimate_head_scope_costs(A, G, num_scopes=0)
        with pytest.raises(ValueError):
            estimate_head_scope_costs(A, G, text_len=9)

    def test_scope_keep_mask_structure(self):
        """Keep mask: symmetric band + always-kept text rows/columns."""
        T, S, text_len = 32, 4, 4
        keep = scope_keep_mask(T, 2, S, text_len=text_len)
        assert keep.shape == (T, T)
        assert keep.dtype == torch.bool
        # Text rows/columns fully kept.
        assert bool(keep[:text_len, :].all())
        assert bool(keep[:, :text_len].all())
        # Symmetric.
        assert bool((keep == keep.T).all())
        # Full scope keeps everything.
        assert bool(scope_keep_mask(T, S, S).all())
        with pytest.raises(ValueError):
            scope_keep_mask(T, 0, S)
        with pytest.raises(ValueError):
            scope_keep_mask(T, S + 1, S)


# ---------------------------------------------------------------------------
# T5.3 — calibration compute-cost table
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCostTable:
    def test_calib_cost_table_matches_masks(self):
        """Every table entry equals the mask-sum cost for that scope (exact)."""
        H, T, S, text_len = 3, 256, 8, 64
        table = calibration_cost_table(H, T, text_len=text_len, num_scopes=S)
        assert table.shape == (H, S)

        for s in range(1, S + 1):
            beta = scope_to_beta(s, S)
            bands = hap.band_blocks([0.0] * H, [beta] * H, T)
            halves = hap.half_blocks(bands)
            mask = hap.build_band_mask(T, text_len, halves, 0)
            expected = float(mask.sum().item()) / H
            assert float(table[0, s - 1]) == pytest.approx(expected, rel=0, abs=0.5)

    def test_cost_table_full_scope_is_full_attention(self):
        """Last column == T² (full attention) regardless of text_len."""
        H, T, S = 2, 128, 5
        for text_len in (0, 32):
            table = calibration_cost_table(H, T, text_len=text_len, num_scopes=S)
            assert float(table[0, -1]) == pytest.approx(float(T * T), rel=1e-12)

    def test_cost_table_rows_identical_and_nondecreasing(self):
        H, T, S = 4, 192, 6
        table = calibration_cost_table(H, T, text_len=0, num_scopes=S)
        for h in range(1, H):
            assert bool((table[h] == table[0]).all())
        diffs = table[0, 1:] - table[0, :-1]
        assert bool((diffs >= 0).all())

    def test_cost_table_rejects_bad_inputs(self):
        with pytest.raises(ValueError):
            calibration_cost_table(0, 128)
        with pytest.raises(ValueError):
            calibration_cost_table(2, 128, num_scopes=0)

    def test_scope_to_beta_mapping(self):
        # Full scope maps to beta=1.0 (exact full attention under the
        # reference band formula), NOT 0.5 (see scope_to_beta docstring).
        assert scope_to_beta(50, 50) == pytest.approx(1.0)
        assert scope_to_beta(25, 50) == pytest.approx(0.25)
        assert scope_to_beta(1, 50) == pytest.approx(0.01)
        with pytest.raises(ValueError):
            scope_to_beta(0, 50)
        with pytest.raises(ValueError):
            scope_to_beta(51, 50)

    def test_full_scope_beta_gives_exact_full_attention(self):
        """beta=1.0 ⇒ band = 2·nbx−1 ⇒ half = nbx−1 ⇒ every image block pair
        is visible (exact full attention), matching paper §6.1."""
        H, T = 2, 256
        nbx = T // hap.HAP_BLOCK
        bands = hap.band_blocks([0.0] * H, [1.0] * H, T)
        halves = hap.half_blocks(bands)
        assert all(b == 2 * nbx - 1 for b in bands)
        assert all(h == nbx - 1 for h in halves)
        mask = hap.build_band_mask(T, 0, halves, 0)
        assert bool(mask.all())  # every pair visible


# ---------------------------------------------------------------------------
# T5.5 — end-to-end pure-math calibration
# ---------------------------------------------------------------------------

def _synthetic_stats(num_layers=3, H=4, S=6, prompts=2, seed=7):
    """Deterministic synthetic (quality, compute) tables for the tests."""
    g = torch.Generator().manual_seed(seed)
    quality = []
    compute = []
    for _ in range(num_layers):
        per_prompt = []
        for _ in range(prompts):
            base = torch.rand(H, S, generator=g, dtype=torch.float64)
            # Make quality strictly decrease with scope; full scope == 0.
            decay = torch.linspace(1.0, 0.1, S, dtype=torch.float64)
            table = base * decay.unsqueeze(0)
            table[:, -1] = 0.0
            per_prompt.append(table)
        quality.append(per_prompt)
        compute.append(calibration_cost_table(H, 256, text_len=64, num_scopes=S))
    return quality, compute


@pytest.mark.unit
class TestCalibrateScopePlan:
    # W2.7 re-baseline (2026-08-25): the seq=256/text_len=64 geometry has a
    # TRUE minimum-cost ratio of 0.625 (per-head min scope cost 0.15625 x 4
    # heads), so any budget below that is genuinely infeasible — the pre-fix
    # tests used budgets under the floor and crashed with "No feasible".
    # Budgets here are >= 0.7 (above the floor); the sub-floor infeasibility
    # path itself is covered by test_calibrate_infeasible_budget_raises.
    _FLOOR = 0.625

    def test_calibrate_produces_valid_plan(self):
        """Synthetic 3-layer/4-head stats → ScopePlan validates; the solved
        assignment honours the compute budget (via T1.4 costs)."""
        H, S = 4, 6
        budget_ratio = 0.8
        quality, compute = _synthetic_stats(num_layers=3, H=H, S=S)

        plan_dict = calibrate_scope_plan(quality, compute, budget_ratio=budget_ratio)

        assert set(plan_dict.keys()) == {"alphas", "betas"}
        plan = hap.ScopePlan.from_dict(plan_dict)  # validates eagerly
        assert plan.num_layers == 3
        assert plan.num_heads == H
        assert all(all(a == 0.0 for a in layer) for layer in plan_dict["alphas"])

        # Budget honoured per layer with the chosen scopes.
        # W2.7 re-baseline (2026-08-25): the reference FULL cost must use
        # beta=1.0 (exact full attention under the current scope_to_beta
        # mapping — matching calibration_cost_table's last column).  The
        # pre-fix [0.5]*H reference predates the beta=1.0 full-scope fix and
        # computed a mid-range scope instead.
        full_halves = hap.half_blocks(hap.band_blocks([0.0] * H, [1.0] * H, 256))
        full_cost = hap.band_compute_cost(256, 64, full_halves, 0)
        # Ceil-discretization slack: the solver's H-bin rounding allowance
        # admits assignments whose TRUE cost exceeds the budget by at most
        # H * budget / bins (same bound as TestKnapsack::test_budget_respected).
        slack = H * (budget_ratio * full_cost) / 4000.0
        for layer in range(3):
            chosen_halves = hap.half_blocks(
                hap.band_blocks(plan_dict["alphas"][layer],
                                plan_dict["betas"][layer], 256)
            )
            chosen_cost = hap.band_compute_cost(256, 64, chosen_halves, 0)
            assert chosen_cost <= budget_ratio * full_cost + slack + 1e-6

    def test_calibrate_full_budget_selects_full_scope(self):
        """budget_ratio=1.0 → every head picks the full scope (beta == 1.0):
        it is feasible and its quality cost is exactly 0 (the minimum)."""
        quality, compute = _synthetic_stats(num_layers=2, H=3, S=5)
        plan_dict = calibrate_scope_plan(quality, compute, budget_ratio=1.0)
        for layer_betas in plan_dict["betas"]:
            assert all(b == pytest.approx(1.0) for b in layer_betas)

    def test_calibrate_deterministic(self):
        quality, compute = _synthetic_stats()
        # W2.7 re-baseline: budget raised above the geometry's true
        # feasibility floor (0.625 for seq=256/text_len=64, H=4).
        a = calibrate_scope_plan(quality, compute, budget_ratio=0.8)
        b = calibrate_scope_plan(quality, compute, budget_ratio=0.8)
        assert a == b

    def test_calibrate_infeasible_budget_raises(self):
        """W2.7 regression guard: a budget BELOW the geometry's true minimum
        (sum of per-head min scope costs) must still raise RuntimeError -- the
        H-bin rounding allowance in the solver must not mask genuine
        infeasibility."""
        H, S = 4, 6
        quality, compute = _synthetic_stats(num_layers=1, H=H, S=S)
        with pytest.raises(RuntimeError, match="No feasible"):
            calibrate_scope_plan(quality, compute, budget_ratio=self._FLOOR * 0.5)

    def test_solver_boundary_rounding_feasible(self):
        """W2.7 REGRESSION (production bug): ceil-discretization used to add
        up to 1 bin per head of rounding error, so a truly-feasible boundary
        assignment was falsely rejected ("No feasible") whenever H did not
        divide the geometry evenly.  Exact reproduction: H=3 heads x full-scope
        cost 1334 bins each = 4002 > budget_int=4000 at budget_ratio=1.0.
        The solver must now accept it (H-bin capacity allowance)."""
        from src.hap_calib import solve_multiple_choice_knapsack

        H, S = 3, 5
        # Real cost table for seq=256/text_len=64: full scope == seq^2 per head.
        compute = calibration_cost_table(H, 256, text_len=64, num_scopes=S)
        quality = torch.zeros(H, S, dtype=torch.float64)
        choices = solve_multiple_choice_knapsack(
            quality, compute, budget_ratio=1.0, bins=4000
        )
        assert len(choices) == H
        # True cost sanity: full-everywhere is exactly the budget.
        full_cost = float(compute[:, -1].sum().item())
        chosen = sum(float(compute[h, choices[h]].item()) for h in range(H))
        assert chosen <= full_cost + 1e-9

    def test_calibrate_averages_prompts(self):
        """Averaging over prompts must change the solution vs a single prompt
        when the prompts disagree (sanity that averaging is applied)."""
        H, S = 2, 4
        compute = [calibration_cost_table(H, 128, text_len=0, num_scopes=S)]
        # Prompt 0: scope 0 cheap-and-good; prompt 1: scope 1 cheap-and-good.
        p0 = torch.full((H, S), 10.0, dtype=torch.float64)
        p0[:, 0] = 0.0
        p0[:, -1] = 0.0
        p1 = torch.full((H, S), 10.0, dtype=torch.float64)
        p1[:, 1] = 0.0
        p1[:, -1] = 0.0
        # Averaged: both scopes 0 and 1 tie at 5.0 < 10.0 -> deterministic
        # tie-break picks the LOWEST scope index (0).
        # W2.7 re-baseline: min-scope ratio for this geometry is 0.5, so the
        # pre-fix budget_ratio=0.3 was genuinely infeasible; 0.55 clears it.
        plan_dict = calibrate_scope_plan([[p0, p1]], compute, budget_ratio=0.55)
        assert all(b == scope_to_beta(1, S) for b in plan_dict["betas"][0])

    def test_calibrate_rejects_bad_inputs(self):
        quality, compute = _synthetic_stats(num_layers=2)
        with pytest.raises(ValueError):
            calibrate_scope_plan([], [])
        with pytest.raises(ValueError):
            calibrate_scope_plan(quality, compute[:1])
        with pytest.raises(ValueError):
            calibrate_scope_plan([[]], compute[:1])
        # Shape mismatch between prompts of one layer.
        bad = [quality[0] + [torch.zeros(1, 1, dtype=torch.float64)], quality[1]]
        with pytest.raises(ValueError):
            calibrate_scope_plan(bad, compute)
        # Shape mismatch between quality and compute tables.
        bad_compute = [torch.zeros(1, 1, dtype=torch.float64), compute[1]]
        with pytest.raises(ValueError):
            calibrate_scope_plan(quality, bad_compute)

    # ------------------------------------------------------------------
    # NaN/inf sanitize safety net (bf16 overflow root cause)
    # ------------------------------------------------------------------

    def test_calibrate_sanitizes_all_nan_quality_table(self):
        """SAFETY NET: an all-NaN quality table (the live Krea2 crash — layer 0
        had 960 NaN of 960) must NOT raise 'No feasible HAP scope assignment'.
        The sanitizer replaces NaN with a large finite penalty so the solver
        degrades gracefully and returns a valid assignment.

        budget_ratio=1.0 is used to guarantee cost feasibility for this small
        geometry, isolating the sanitize behaviour from budget feasibility
        (diag #3 proved NaN alone makes the solver raise even with ample
        budget)."""
        H, S = 4, 6
        compute = [calibration_cost_table(H, 256, text_len=64, num_scopes=S)]
        all_nan = torch.full((H, S), float("nan"), dtype=torch.float64)

        # Before the fix this raised RuntimeError("No feasible ...").
        plan_dict = calibrate_scope_plan([[all_nan]], compute, budget_ratio=1.0)

        assert set(plan_dict.keys()) == {"alphas", "betas"}
        plan = hap.ScopePlan.from_dict(plan_dict)  # validates eagerly
        assert plan.num_layers == 1
        assert plan.num_heads == H
        # Every beta is a valid scope_to_beta value.
        for b in plan_dict["betas"][0]:
            assert 0.0 < b <= 1.0

    def test_calibrate_sanitizes_partial_nan_quality_table(self):
        """SAFETY NET: a partially-NaN quality table (some heads poisoned, some
        finite) must NOT raise.  The sanitizer replaces NaN entries with a
        penalty; finite entries are preserved so the solver can still pick
        good scopes for the unpoisoned heads.

        budget_ratio=0.7 is used because the minimum feasible budget for this
        geometry (seq=256, text_len=64) is ~0.63 (validated by
        tmp/diag_partial_nan.py); 0.7 is feasible and picks non-trivial
        scopes, exercising that the solver still differentiates the
        unpoisoned heads."""
        H, S = 4, 6
        compute = [calibration_cost_table(H, 256, text_len=64, num_scopes=S)]
        quality, _ = _synthetic_stats(num_layers=1, H=H, S=S)
        table = quality[0][0].clone()
        # Poison head 0 entirely.
        table[0, :] = float("nan")

        plan_dict = calibrate_scope_plan([[table]], compute, budget_ratio=0.7)

        assert set(plan_dict.keys()) == {"alphas", "betas"}
        plan = hap.ScopePlan.from_dict(plan_dict)
        assert plan.num_heads == H

    def test_calibrate_sanitizes_inf_quality_table(self):
        """SAFETY NET: an all-inf quality table must NOT raise.  The sanitizer
        treats inf the same as NaN (replaces with a finite penalty).
        budget_ratio=1.0 guarantees cost feasibility."""
        H, S = 4, 6
        compute = [calibration_cost_table(H, 256, text_len=64, num_scopes=S)]
        all_inf = torch.full((H, S), float("inf"), dtype=torch.float64)

        plan_dict = calibrate_scope_plan([[all_inf]], compute, budget_ratio=1.0)

        assert set(plan_dict.keys()) == {"alphas", "betas"}
        plan = hap.ScopePlan.from_dict(plan_dict)
        assert plan.num_heads == H

    def test_calibrate_sanitize_warning_fires(self, caplog):
        """The NaN/inf sanitizer must log a WARNING naming the layer and the
        NaN/inf counts, so a poisoned Taylor table is visible in the ComfyUI
        console before the solver runs."""
        import logging

        H, S = 4, 6
        compute = [calibration_cost_table(H, 256, text_len=64, num_scopes=S)]
        all_nan = torch.full((H, S), float("nan"), dtype=torch.float64)

        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            calibrate_scope_plan([[all_nan]], compute, budget_ratio=1.0)

        assert any(
            "quality table has" in rec.message and "NaN" in rec.message
            for rec in caplog.records
        ), "expected the NaN/inf sanitize WARNING to fire"
