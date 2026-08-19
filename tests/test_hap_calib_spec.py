"""Tests for CalibrationSpec + resolve_prompts (plan P0: T0.1-T0.3).

Markers: @pytest.mark.unit
Accept (user-run):
    pytest tests/test_hap_calib_spec.py -q
"""

import os

import pytest

from src.hap_calib_node import (
    DEFAULT_CALIBRATION_PROMPTS,
    CalibrationSpec,
    resolve_prompts,
)


def _valid_spec(**overrides):
    """Build a valid spec, applying overrides."""
    kwargs = dict(
        width=1024,
        height=1024,
        num_prompts=5,
        num_scopes=50,
        budget_ratio=0.10,
        bins=4000,
        chunk=256,
        text_len=512,
        anchor_stride=32,
        calib_sigma=1.0,
        seed=3407,
        loss_type="output_norm",
        prompts=["a test prompt"],
    )
    kwargs.update(overrides)
    return CalibrationSpec(**kwargs)


# ---------------------------------------------------------------------------
# T0.1 — valid spec passes; every invalid boundary raises ValueError
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCalibrationSpecValidate:
    def test_valid_spec_passes(self):
        _valid_spec().validate()  # no raise

    @pytest.mark.parametrize(
        "field,value",
        [
            ("width", 255),          # < 256
            ("width", 100),          # not multiple of 8
            ("width", 1023),         # not multiple of 8
            ("height", 255),
            ("height", 100),
            ("num_prompts", 0),
            ("num_prompts", -1),
            ("num_scopes", 1),
            ("num_scopes", 0),
            ("budget_ratio", 0.0),
            ("budget_ratio", -0.1),
            ("budget_ratio", 1.5),
            ("bins", 0),
            ("chunk", 0),
            ("text_len", -1),
            ("anchor_stride", -1),
            ("calib_sigma", -0.01),
            ("calib_sigma", 1.01),
        ],
    )
    def test_invalid_field_raises(self, field, value):
        spec = _valid_spec(**{field: value})
        with pytest.raises(ValueError, match=field):
            spec.validate()

    def test_unknown_loss_type_raises(self):
        spec = _valid_spec(loss_type="bogus")
        with pytest.raises(ValueError, match="loss_type"):
            spec.validate()

    def test_reference_mse_without_reference_raises(self):
        spec = _valid_spec(loss_type="reference_mse", reference_latent=None)
        with pytest.raises(ValueError, match="reference_latent"):
            spec.validate()

    def test_reference_mse_with_reference_passes(self):
        import torch
        spec = _valid_spec(
            loss_type="reference_mse",
            reference_latent=torch.zeros(1, 4, 8, 8),
        )
        spec.validate()  # no raise

    def test_empty_prompts_raises(self):
        spec = _valid_spec(prompts=[])
        with pytest.raises(ValueError, match="prompt list is empty"):
            spec.validate()


# ---------------------------------------------------------------------------
# T0.2 — resolve_prompts
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestResolvePrompts:
    def test_multiline_parsing(self):
        text = "prompt one\n\n  prompt two  \n\nprompt three\n"
        result = resolve_prompts(prompts_text=text, num_prompts=10)
        assert result == ["prompt one", "prompt two", "prompt three"]

    def test_blank_only_falls_back_to_defaults(self):
        result = resolve_prompts(prompts_text="\n\n  \n", num_prompts=10)
        assert result == DEFAULT_CALIBRATION_PROMPTS

    def test_empty_falls_back_to_defaults(self):
        result = resolve_prompts(prompts_text="", num_prompts=10)
        assert result == DEFAULT_CALIBRATION_PROMPTS

    def test_num_prompts_truncation(self):
        text = "\n".join(f"prompt {i}" for i in range(10))
        result = resolve_prompts(prompts_text=text, num_prompts=3)
        assert len(result) == 3
        assert result == ["prompt 0", "prompt 1", "prompt 2"]

    def test_file_override(self, tmp_path):
        f = tmp_path / "prompts.txt"
        f.write_text("file prompt A\nfile prompt B\n", encoding="utf-8")
        result = resolve_prompts(
            prompts_text="ignored",
            prompts_file=str(f),
            num_prompts=10,
        )
        assert result == ["file prompt A", "file prompt B"]

    def test_file_relative_resolution(self, tmp_path):
        f = tmp_path / "rel_prompts.txt"
        f.write_text("relative prompt\n", encoding="utf-8")
        result = resolve_prompts(
            prompts_file="rel_prompts.txt",
            num_prompts=5,
            pack_root=str(tmp_path),
        )
        assert result == ["relative prompt"]

    def test_missing_file_raises(self):
        with pytest.raises(ValueError, match="prompts file not found"):
            resolve_prompts(prompts_file="nonexistent_prompts_xyz.txt")

    def test_file_with_blank_lines(self, tmp_path):
        f = tmp_path / "blanks.txt"
        f.write_text("A\n\n\nB\n  \n", encoding="utf-8")
        result = resolve_prompts(prompts_file=str(f), num_prompts=10)
        assert result == ["A", "B"]


# ---------------------------------------------------------------------------
# T0.3 — CLI single-source identity
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCliSingleSource:
    def test_cli_imports_default_prompts(self):
        """The CLI script re-imports DEFAULT_CALIBRATION_PROMPTS from
        src.hap_calib_node (single source, no drift)."""
        import importlib.util

        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "calibration", "calibrate_hap.py")
        spec = importlib.util.spec_from_file_location("calibrate_hap", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.DEFAULT_PROMPTS is DEFAULT_CALIBRATION_PROMPTS
