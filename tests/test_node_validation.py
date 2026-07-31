"""Tests for node input validation logic (Tier 1: standalone logic tests)."""
import pytest


def _validate_inputs(**kwargs) -> bool | str:
    """Standalone validation logic matching DyPE_FLUX.validate_inputs()."""
    width = kwargs.get("width", 1024)
    height = kwargs.get("height", 1024)

    if not isinstance(width, int) or not isinstance(height, int):
        return "Width and height must be integers."

    if width < 16 or height < 16:
        return "Width and height must be at least 16 pixels."

    if width % 16 != 0:
        return f"Width ({width}) must be a multiple of 16 for latent space compatibility."

    if height % 16 != 0:
        return f"Height ({height}) must be a multiple of 16 for latent space compatibility."

    base_resolution = kwargs.get("base_resolution", 1024)
    if base_resolution < 256:
        return "base_resolution must be at least 256."

    # Check latent dimensions are even (patch_size=2 compatibility)
    latent_w = width // 8
    latent_h = height // 8
    if latent_w % 2 != 0 or latent_h % 2 != 0:
        return (
            f"Resolution {width}x{height} produces odd latent dimensions "
            f"({latent_w}x{latent_h}). This may cause issues with patch_size=2 models. "
            f"Use dimensions that are multiples of 16."
        )

    return True


@pytest.mark.unit
class TestValidateInputs:
    def test_valid_defaults(self):
        assert _validate_inputs() is True

    def test_valid_4k(self):
        assert _validate_inputs(width=4096, height=4096) is True

    def test_valid_non_square(self):
        assert _validate_inputs(width=2048, height=1024) is True

    def test_valid_minimum(self):
        assert _validate_inputs(width=16, height=16) is True

    def test_valid_1024(self):
        assert _validate_inputs(width=1024, height=1024) is True

    def test_invalid_width_not_multiple_16(self):
        result = _validate_inputs(width=1000, height=1024)
        assert isinstance(result, str)
        assert "multiple of 16" in result

    def test_invalid_height_not_multiple_16(self):
        result = _validate_inputs(width=1024, height=1000)
        assert isinstance(result, str)
        assert "multiple of 16" in result

    def test_invalid_too_small_width(self):
        result = _validate_inputs(width=8, height=1024)
        assert isinstance(result, str)
        assert "at least 16" in result

    def test_invalid_too_small_height(self):
        result = _validate_inputs(width=1024, height=8)
        assert isinstance(result, str)
        assert "at least 16" in result

    def test_invalid_base_resolution(self):
        result = _validate_inputs(base_resolution=128)
        assert isinstance(result, str)
        assert "at least 256" in result

    def test_valid_base_resolution_minimum(self):
        assert _validate_inputs(base_resolution=256) is True

    def test_odd_latent_dimensions_caught(self):
        # 24 is not multiple of 16, so it fails at the multiple-of-16 check first
        result = _validate_inputs(width=24, height=1024)
        assert isinstance(result, str)

    def test_valid_produces_even_latent(self):
        # 1024 // 8 = 128 (even) — valid
        assert _validate_inputs(width=1024, height=1024) is True

    def test_2048_produces_even_latent(self):
        # 2048 // 8 = 256 (even) — valid
        assert _validate_inputs(width=2048, height=2048) is True
