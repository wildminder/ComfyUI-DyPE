"""Tests for the calibration orchestrator ``run_hap_calibration`` (plan P3).

Covers:
- T3.1 toy end-to-end (deterministic, valid plan);
- T3.2 budget respected across budget_ratio values;
- T3.3 averaging idempotence;
- T3.4 D11 guard (HAP-live model rejected);
- T3.5 summary contents;
- T3.6 format_summary golden-string check.

Markers: @pytest.mark.unit
Accept (user-run):
    pytest tests/test_hap_calib_orchestrator.py -q
"""

import pathlib
import types

import pytest
import torch

import src.hap_calib_node as hcn
from src.hap import ScopePlan, flops_ratio
from src.hap_calib_node import (
    CalibrationSpec,
    format_summary,
    run_hap_calibration,
)


def _toy(seed=5):
    # img_hw=32 -> seq_len = 8 + 1024 = 1032 >= HAP_BLOCK(64) with nbx=16, so
    # the candidate scopes map to DISTINCT band widths and the knapsack has a
    # feasible assignment at every tested budget_ratio.  (A toy smaller than
    # HAP_BLOCK gives nbx=0, collapsing every scope to band=1 and making any
    # budget_ratio < 1.0 infeasible.)
    from _hrdit_fixtures import make_toy_dit
    return make_toy_dit(num_layers=2, heads=3, dim=8, text_len=8,
                        img_hw=32, seed=seed, dtype=torch.float64)


def _spec(prompts, **overrides):
    kwargs = dict(
        width=512, height=512,
        num_prompts=len(prompts),
        num_scopes=6, budget_ratio=0.5, bins=2000, chunk=5,
        text_len=8, anchor_stride=0, calib_sigma=1.0, seed=100,
        loss_type="output_norm", prompts=prompts,
    )
    kwargs.update(overrides)
    return CalibrationSpec(**kwargs)


def _toy_forward(dit_holder):
    """Build an injected forward that runs the toy DiT (re-seeded per prompt)."""
    def fwd(model, spec, prompt_index):
        dit = _toy(seed=spec.seed + prompt_index)
        dit_holder.append(dit)
        return dit.forward()
    return fwd


# ---------------------------------------------------------------------------
# T3.1 — toy end-to-end
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorEndToEnd:
    def test_toy_e2e_valid_plan(self):
        holder = []
        spec = _spec(["p1", "p2", "p3"])
        plan_dict, summary = run_hap_calibration(
            model=object(), spec=spec, model_type="flux",
            forward_fn=_toy_forward(holder),
        )
        # Valid reference-format plan.
        plan = ScopePlan.from_dict(plan_dict)
        assert plan.num_layers == 2
        assert plan.num_heads == 3
        assert set(plan_dict.keys()) == {"alphas", "betas"}
        # All alphas are 0.
        assert all(all(a == 0.0 for a in row) for row in plan_dict["alphas"])
        # Betas in valid range.
        assert all(0.0 < b <= 1.0 for row in plan_dict["betas"] for b in row)
        # Three prompts were run.
        assert len(holder) == 3

    def test_toy_e2e_deterministic(self):
        holder1, holder2 = [], []
        spec = _spec(["p1", "p2"])
        plan1, _ = run_hap_calibration(
            model=object(), spec=spec, model_type="flux",
            forward_fn=_toy_forward(holder1),
        )
        plan2, _ = run_hap_calibration(
            model=object(), spec=spec, model_type="flux",
            forward_fn=_toy_forward(holder2),
        )
        assert plan1 == plan2


# ---------------------------------------------------------------------------
# T3.2 — budget respected
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorBudget:
    @pytest.mark.parametrize("budget", [0.1, 0.3, 0.8])
    def test_budget_respected(self, budget):
        holder = []
        spec = _spec(["p1", "p2"], budget_ratio=budget)
        plan_dict, summary = run_hap_calibration(
            model=object(), spec=spec, model_type="flux",
            forward_fn=_toy_forward(holder),
        )
        plan = ScopePlan.from_dict(plan_dict)
        seq_len = (512 // 16) * (512 // 16) + 8
        fr = flops_ratio(plan, seq_len, text_len=8, anchor_stride=0)
        # Within knapsack discretization tolerance.
        assert fr <= budget + 0.05


# ---------------------------------------------------------------------------
# T3.3 — averaging idempotence
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorAveraging:
    def test_identical_prompts_equal_single(self):
        """Feeding 3 identical prompts equals feeding 1 prompt (mean is
        idempotent)."""
        spec1 = _spec(["p"])
        spec3 = _spec(["p", "p", "p"])

        # Use a FIXED seed so all prompts produce identical forwards.
        def fixed_fwd(model, spec, prompt_index):
            dit = _toy(seed=42)  # ignore prompt_index -> identical
            return dit.forward()

        plan1, _ = run_hap_calibration(
            model=object(), spec=spec1, model_type="flux", forward_fn=fixed_fwd,
        )
        plan3, _ = run_hap_calibration(
            model=object(), spec=spec3, model_type="flux", forward_fn=fixed_fwd,
        )
        assert plan1 == plan3


# ---------------------------------------------------------------------------
# T3.4 — D11 guard
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorHapGuard:
    def test_hap_live_model_rejected(self):
        model = types.SimpleNamespace()
        model._hap_ctx = types.SimpleNamespace(active=True)
        spec = _spec(["p"])
        with pytest.raises(RuntimeError, match="UNPRUNED"):
            run_hap_calibration(
                model=model, spec=spec, model_type="flux",
                forward_fn=_toy_forward([]),
            )

    def test_hap_inactive_model_allowed(self):
        model = types.SimpleNamespace()
        model._hap_ctx = types.SimpleNamespace(active=False)
        spec = _spec(["p"])
        plan_dict, _ = run_hap_calibration(
            model=model, spec=spec, model_type="flux",
            forward_fn=_toy_forward([]),
        )
        assert "alphas" in plan_dict


# ---------------------------------------------------------------------------
# excluded_head_counts metadata (2026-08-23 head-count warning fix)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorExcludedHeads:
    """The orchestrator persists the collector's non-dominant head counts into
    the plan dict so the runtime can log a friendly INFO (expected auxiliary
    fallback) instead of a scary WARNING (wrong plan)."""

    def _fake_collect(self, excluded):
        """A stand-in collector that returns a tiny valid table and fills meta.

        The compute table must have INCREASING per-scope costs (like the real
        ``calibration_cost_table``) so the knapsack is feasible at the default
        budget_ratio=0.5 — a flat all-ones table makes every scope equally
        expensive and the solver reports "No feasible assignment".
        """
        def fake(model, model_type, forward_fn, loss_fn, num_scopes,
                 text_len, chunk, scale, meta=None):
            L, H, S = 2, 3, num_scopes
            quality = torch.rand(L, H, S, dtype=torch.float64)
            # cost[h][s] increases with scope index (scope 0 cheapest, last=full).
            row = torch.arange(1, S + 1, dtype=torch.float64)
            compute = row.unsqueeze(0).expand(H, -1).clone()
            compute = compute.unsqueeze(0).expand(L, -1, -1).clone()
            if meta is not None:
                meta["excluded_head_counts"] = list(excluded)
            return quality, compute, 1032
        return fake

    def test_excluded_heads_persisted_in_plan_dict(self, monkeypatch):
        monkeypatch.setattr(
            hcn, "collect_scope_scores_for_model", self._fake_collect([20])
        )
        spec = _spec(["p1"])
        plan_dict, _ = run_hap_calibration(
            model=object(), spec=spec, model_type="flux",
        )
        assert plan_dict.get("excluded_head_counts") == [20]
        # The plan still validates and round-trips the field.
        plan = ScopePlan.from_dict(plan_dict)
        assert plan.excluded_head_counts == [20]

    def test_no_excluded_heads_omits_key(self, monkeypatch):
        """A uniform-head model (empty excluded list) keeps the legacy shape."""
        monkeypatch.setattr(
            hcn, "collect_scope_scores_for_model", self._fake_collect([])
        )
        spec = _spec(["p1"])
        plan_dict, _ = run_hap_calibration(
            model=object(), spec=spec, model_type="flux",
        )
        assert "excluded_head_counts" not in plan_dict


# ---------------------------------------------------------------------------
# purge_between_prompts knob + node-level cleanup (plan 2026-08-24 P3/P5)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPurgeBetweenPrompts:
    def test_spec_default_false_and_validates(self):
        """The knob defaults to False and a valid spec passes validation."""
        spec = _spec(["p"])
        assert spec.purge_between_prompts is False
        spec.validate()  # must not raise

    def test_orchestrator_purges_between_prompts_when_enabled(self, monkeypatch):
        """With the knob on: one purge per BETWEEN-prompt gap (+ collector's
        own post-scoring purge is mocked out here)."""
        n = {"purges": 0}
        monkeypatch.setattr(hcn, "_purge_calibration_memory",
                            lambda: n.__setitem__("purges", n["purges"] + 1))
        holder = []
        spec = _spec(["p1", "p2", "p3"], purge_between_prompts=True)
        run_hap_calibration(
            model=object(), spec=spec, model_type="flux",
            forward_fn=_toy_forward(holder),
        )
        # 3 prompts -> 2 between-gaps; plus the final node-level purge happens
        # in execute(), not here.  The collector's post-scoring purge is also
        # counted (it calls the same helper): >= 2 gaps guaranteed.
        assert n["purges"] >= 2

    def test_orchestrator_no_extra_purges_when_disabled(self, monkeypatch):
        """Default (False): only the collector's single post-scoring purge."""
        n = {"purges": 0}
        monkeypatch.setattr(hcn, "_purge_calibration_memory",
                            lambda: n.__setitem__("purges", n["purges"] + 1))
        holder = []
        spec = _spec(["p1", "p2"])
        run_hap_calibration(
            model=object(), spec=spec, model_type="flux",
            forward_fn=_toy_forward(holder),
        )
        assert n["purges"] == 2  # one per prompt (collector), no gap purges

    def test_schema_has_purge_input_with_default_false(self):
        """The node schema exposes purge_between_prompts defaulting to False."""
        src = (
            pathlib.Path(__file__).parent.parent / "src" / "hap_calib_node.py"
        ).read_text(encoding="utf-8")
        assert '"purge_between_prompts"' in src
        assert "purge_between_prompts: bool = False" in src


# ---------------------------------------------------------------------------
# T3.5 — summary contents
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorSummary:
    def test_summary_keys_and_bounds(self):
        holder = []
        spec = _spec(["p1", "p2"])
        _, summary = run_hap_calibration(
            model=object(), spec=spec, model_type="flux",
            forward_fn=_toy_forward(holder),
        )
        for key in ("num_layers", "num_heads", "seq_len", "text_len",
                    "num_prompts", "num_scopes", "budget_ratio",
                    "mean_beta_min", "mean_beta_max", "flops_ratio",
                    "elapsed_seconds"):
            assert key in summary, f"missing summary key {key!r}"
        assert summary["num_layers"] == 2
        assert summary["num_heads"] == 3
        assert summary["num_prompts"] == 2
        assert 0.0 < summary["flops_ratio"] <= 1.0
        assert summary["mean_beta_min"] <= summary["mean_beta_max"]
        assert summary["elapsed_seconds"] >= 0.0


# ---------------------------------------------------------------------------
# T3.6 — format_summary golden string
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFormatSummary:
    def test_golden_format(self):
        summary = {
            "num_layers": 2, "num_heads": 3, "seq_len": 1032,
            "text_len": 8, "num_prompts": 2, "num_scopes": 6,
            "budget_ratio": 0.5, "mean_beta_min": 0.25,
            "mean_beta_max": 0.75, "flops_ratio": 0.42,
            "elapsed_seconds": 1.5,
        }
        text = format_summary(summary)
        assert "HAP Calibration Summary" in text
        assert "layers=2 heads=3 seq_len=1032" in text
        assert "prompts=2 scopes=6 budget=0.50" in text
        assert "min=0.250 max=0.750" in text
        assert "retained flops ratio: 0.420" in text
        assert "elapsed: 1.5s" in text

    def test_flops_ratio_none_omitted(self):
        summary = {
            "num_layers": 1, "num_heads": 1, "seq_len": 100,
            "text_len": 0, "num_prompts": 1, "num_scopes": 4,
            "budget_ratio": 0.1, "mean_beta_min": 0.5,
            "mean_beta_max": 0.5, "flops_ratio": None,
            "elapsed_seconds": 0.1,
        }
        text = format_summary(summary)
        assert "flops ratio" not in text
