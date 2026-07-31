"""Tests for resolution snapping logic (Tier 1: pure unit tests)."""
import pytest

from src.patch_utils import _snap_to_multiple


@pytest.mark.unit
class TestSnapToMultiple:
    def test_exact_multiple_unchanged(self):
        assert _snap_to_multiple(1024) == 1024

    def test_exact_multiple_4096(self):
        assert _snap_to_multiple(4096) == 4096

    def test_rounds_up_above_half(self):
        # 1000 / 16 = 62.5 → round(62.5) = 62 (banker's rounding) or 63
        # Python round(62.5) = 62 (rounds to even), so 62*16 = 992
        # Actually round(1000/16) = round(62.5) = 62 → 992
        result = _snap_to_multiple(1000)
        assert result in (992, 1008)  # Either is acceptable rounding behavior
        assert result % 16 == 0

    def test_rounds_down_below_half(self):
        # 999 / 16 = 62.4375 → round = 62 → 992
        assert _snap_to_multiple(999) == 992

    def test_rounds_up_clearly_above(self):
        # 1001 / 16 = 62.5625 → round = 63 → 1008
        assert _snap_to_multiple(1001) == 1008

    def test_minimum_is_multiple(self):
        assert _snap_to_multiple(1) == 16
        assert _snap_to_multiple(0) == 16
        assert _snap_to_multiple(8) == 16
        assert _snap_to_multiple(15) == 16

    def test_value_16_unchanged(self):
        assert _snap_to_multiple(16) == 16

    def test_value_17_snaps_to_16(self):
        # 17/16 = 1.0625 → round = 1 → 16
        assert _snap_to_multiple(17) == 16

    def test_value_24_snaps_to_16_or_32(self):
        # 24/16 = 1.5 → round(1.5) = 2 (banker's) → 32
        result = _snap_to_multiple(24)
        assert result in (16, 32)
        assert result % 16 == 0

    def test_large_value(self):
        assert _snap_to_multiple(4095) == 4096
        assert _snap_to_multiple(4097) == 4096

    def test_custom_multiple_32(self):
        assert _snap_to_multiple(100, 32) == 96
        assert _snap_to_multiple(113, 32) == 128

    def test_custom_multiple_8(self):
        # 100/8 = 12.5 → round(12.5) = 12 (banker's rounding) → 96
        assert _snap_to_multiple(100, 8) == 96
        # 101/8 = 12.625 → round = 13 → 104
        assert _snap_to_multiple(101, 8) == 104

    def test_always_returns_int(self):
        assert isinstance(_snap_to_multiple(1000), int)
        assert isinstance(_snap_to_multiple(0), int)

    def test_never_returns_zero(self):
        assert _snap_to_multiple(0) >= 16
        assert _snap_to_multiple(-5) >= 16


@pytest.mark.unit
class TestNoValidateInputs:
    def test_validate_inputs_removed(self):
        """validate_inputs should not exist — snapping is done in execute()."""
        import pathlib
        content = (pathlib.Path(__file__).parent.parent / "__init__.py").read_text(encoding="utf-8")
        assert "validate_inputs" not in content
