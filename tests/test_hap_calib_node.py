"""Tests for the HAPCalibrate node (plan P5: T5.1-T5.5).

The node *wiring* (schema, inputs, registration) is covered by text-checks
that read ``__init__.py`` / ``src/hap_calib_node.py`` directly — the same
pattern as ``tests/test_hap_node.py``.  Functional behaviour is exercised via
``HAPCalibrate.execute`` with an injected forward (monkeypatched
``run_hap_calibration``).

Markers: @pytest.mark.unit
Accept (user-run):
    pytest tests/test_hap_calib_node.py -q
"""

import pathlib

import pytest

import src.hap_calib_node as hcn
from src.hap_calib_node import HAPCalibrate

_ROOT = pathlib.Path(__file__).parent.parent
_INIT = _ROOT / "__init__.py"
_NODE_SRC = _ROOT / "src" / "hap_calib_node.py"


# ---------------------------------------------------------------------------
# T5.1 — schema text-checks (every §3.1 input present)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSchemaInputs:
    @pytest.mark.parametrize(
        "name",
        [
            "model", "positive", "negative", "width", "height", "prompts",
            "prompts_file", "num_prompts", "num_scopes", "budget_ratio",
            "bins", "chunk", "text_len", "anchor_stride", "calib_sigma",
            "seed", "loss_type", "reference_latent", "output_name", "run",
        ],
    )
    def test_input_present(self, name):
        src = _NODE_SRC.read_text(encoding="utf-8")
        assert f'"{name}"' in src, f"input {name!r} missing from schema"

    def test_node_identity(self):
        src = _NODE_SRC.read_text(encoding="utf-8")
        assert 'node_id="HAPCalibrate"' in src
        assert 'display_name="HAP Calibrate (HRDiT)"' in src
        assert 'category="model_patches/position_encoding"' in src

    def test_outputs_present(self):
        src = _NODE_SRC.read_text(encoding="utf-8")
        assert 'io.Custom("SCOPE_PLAN").Output(' in src
        assert 'display_name="plan_path"' in src
        assert 'display_name="summary"' in src

    def test_schema_constructs(self):
        """The schema builds against the (mocked) comfy_api without error."""
        schema = HAPCalibrate.define_schema()
        assert schema.node_id == "HAPCalibrate"


# ---------------------------------------------------------------------------
# T5.2 — execute with injected calibration
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExecute:
    def test_execute_returns_three_outputs(self, monkeypatch, tmp_path):
        plan_dict = {
            "alphas": [[0.0, 0.0]],
            "betas": [[0.5, 1.0]],
        }
        summary = {
            "num_layers": 1, "num_heads": 2, "seq_len": 100,
            "text_len": 0, "num_prompts": 1, "num_scopes": 4,
            "budget_ratio": 0.5, "mean_beta_min": 0.5,
            "mean_beta_max": 0.75, "flops_ratio": 0.5,
            "elapsed_seconds": 0.1,
        }
        monkeypatch.setattr(
            hcn, "run_hap_calibration",
            lambda **kw: (plan_dict, summary),
        )
        monkeypatch.setattr(hcn, "resolve_output_dir", lambda: str(tmp_path))

        out = HAPCalibrate.execute(
            model=object(), positive=None, negative=None,
            width=512, height=512, prompts="a prompt",
            num_prompts=1, num_scopes=4, budget_ratio=0.5,
        )
        plan, path, summary_text = out
        assert plan == plan_dict
        assert path.endswith(".json")
        assert pathlib.Path(path).exists()
        assert "HAP Calibration Summary" in summary_text

    def test_execute_writes_to_dype_hap_subdir(self, monkeypatch, tmp_path):
        plan_dict = {"alphas": [[0.0]], "betas": [[1.0]]}
        summary = {
            "num_layers": 1, "num_heads": 1, "seq_len": 10,
            "text_len": 0, "num_prompts": 1, "num_scopes": 2,
            "budget_ratio": 0.5, "mean_beta_min": 1.0,
            "mean_beta_max": 1.0, "flops_ratio": 1.0,
            "elapsed_seconds": 0.0,
        }
        monkeypatch.setattr(
            hcn, "run_hap_calibration", lambda **kw: (plan_dict, summary)
        )
        monkeypatch.setattr(hcn, "resolve_output_dir", lambda: str(tmp_path))

        _, path, _ = HAPCalibrate.execute(
            model=object(), positive=None, negative=None,
            prompts="p", num_prompts=1,
        )
        assert "dype_hap" in path


# ---------------------------------------------------------------------------
# T5.3 — run=False short-circuit
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRunFalse:
    def test_run_false_skips(self, monkeypatch):
        called = []
        monkeypatch.setattr(
            hcn, "run_hap_calibration",
            lambda **kw: called.append(1) or ({}, {}),
        )
        out = HAPCalibrate.execute(
            model=object(), positive=None, negative=None, run=False,
        )
        plan, path, summary_text = out
        assert plan == {}
        assert path == ""
        assert "skipped" in summary_text
        assert called == []  # forward never ran


# ---------------------------------------------------------------------------
# T5.4 — invalid inputs surface with prefix
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestExecuteErrors:
    def test_invalid_budget_prefixed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hcn, "resolve_output_dir", lambda: str(tmp_path))
        with pytest.raises(ValueError, match="HAP Calibrate:"):
            HAPCalibrate.execute(
                model=object(), positive=None, negative=None,
                prompts="p", num_prompts=1, budget_ratio=0.0,
            )

    def test_reference_mse_without_reference_prefixed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(hcn, "resolve_output_dir", lambda: str(tmp_path))
        with pytest.raises(ValueError, match="HAP Calibrate:"):
            HAPCalibrate.execute(
                model=object(), positive=None, negative=None,
                prompts="p", num_prompts=1, loss_type="reference_mse",
            )

    def test_zero_prompts_after_resolution_prefixed(self, tmp_path, monkeypatch):
        """A prompts_file that exists but is empty of content still falls
        back to defaults, so use an invalid width to trigger the prefix."""
        monkeypatch.setattr(hcn, "resolve_output_dir", lambda: str(tmp_path))
        with pytest.raises(ValueError, match="HAP Calibrate:"):
            HAPCalibrate.execute(
                model=object(), positive=None, negative=None,
                prompts="p", num_prompts=1, width=100,
            )


# ---------------------------------------------------------------------------
# T5.5 — registration text-checks
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRegistration:
    def test_imported_in_init(self):
        src = _INIT.read_text(encoding="utf-8")
        assert "from .src.hap_calib_node import HAPCalibrate" in src

    def test_listed_in_node_list(self):
        src = _INIT.read_text(encoding="utf-8")
        assert "HAPCalibrate" in src.split("get_node_list")[1]

    def test_hap_node_accepts_scope_plan_input(self):
        """The HAP node gained the optional SCOPE_PLAN input (plan P4.3)."""
        src = _INIT.read_text(encoding="utf-8")
        assert 'io.Custom("SCOPE_PLAN").Input(' in src
        assert '"scope_plan"' in src
        # execute prefers the linked plan.
        assert "if scope_plan is not None:" in src
        assert "ScopePlan.from_dict(scope_plan)" in src
