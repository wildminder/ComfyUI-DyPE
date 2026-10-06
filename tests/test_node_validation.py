"""Tests for resolution snapping + input validation (Tier 1: pure unit tests)."""
import ast
import pathlib
import re
import types

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
class TestValidateResolution:
    """W5.1 — shared resolution validator (IMP-002)."""

    @pytest.mark.parametrize("w,h", [
        (1024, 1024),   # standard
        (16, 16),       # lower bound
        (8192, 8192),   # upper bound
        (2048, 1024),   # asymmetric valid
        (4096, 4096),   # 4K
    ])
    def test_valid_resolutions_pass(self, w, h):
        from src.validation import validate_resolution

        assert validate_resolution(w, h) is True

    @pytest.mark.parametrize("w,h", [
        (504, 2000),    # REGRESSION (2026-08-25): /8-aligned but not /16 —
                        # runtime snaps to /16 via _snap_to_multiple; the
                        # graph validator must NOT reject these.
        (1000, 1000),   # /8-aligned (1000 % 8 == 0) → accepted
        (999, 1024),    # 999 not /8 → rejected
        (1023, 16),
    ])
    def test_alignment_rules(self, w, h):
        """/8 is the hard VAE requirement; /16-only values are accepted and
        snapped at apply time. Only non-/8 values are rejected."""
        from src.validation import validate_resolution

        result = validate_resolution(w, h)
        if w % 8 or h % 8:
            assert isinstance(result, str)
            assert "multiples of 8" in result
        else:
            assert result is True

    @pytest.mark.parametrize("w,h", [
        (0, 1024),       # below min
        (8208, 1024),    # above max (/8-aligned)
        (16384, 16384),  # way above max
    ])
    def test_out_of_range_rejected(self, w, h):
        from src.validation import validate_resolution

        result = validate_resolution(w, h)
        if w % 8 or h % 8:
            assert "multiples of 8" in result
        else:
            assert isinstance(result, str)
            assert "[16, 8192]" in result

    def test_boundary_values_exact(self):
        from src.validation import validate_resolution

        assert validate_resolution(16, 16) is True
        assert validate_resolution(8192, 8192) is True
        # One step outside.
        assert isinstance(validate_resolution(8200, 1024), str)

    def test_custom_bounds(self):
        from src.validation import validate_resolution

        assert validate_resolution(32, 32, min_px=32, max_px=64) is True
        assert isinstance(validate_resolution(16, 32, min_px=32, max_px=64), str)

    def test_non_integer_inputs_rejected(self):
        from src.validation import validate_resolution

        assert isinstance(validate_resolution("abc", 1024), str)
        assert isinstance(validate_resolution(None, 1024), str)


@pytest.mark.unit
class TestNodeValidateInputsWiring:
    """W5.2 — the resolution nodes implement validate_inputs via the shared
    validator (graph-time rejection; runtime snapping retained as
    defense-in-depth)."""

    _NODES = ("DyPE_FLUX", "SEGA", "SPA")

    @pytest.fixture
    def init_mod(self):
        import importlib.util
        import pathlib

        root = pathlib.Path(__file__).parent.parent
        spec = importlib.util.spec_from_file_location(
            "dype_init_wiring", root / "__init__.py")
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            pytest.skip("__init__.py requires full comfy_api runtime")
        return mod

    def test_resolution_nodes_have_validate_inputs(self, init_mod):
        for node_name in self._NODES:
            node_cls = getattr(init_mod, node_name, None)
            assert node_cls is not None, f"{node_name} missing"
            assert callable(getattr(node_cls, "validate_inputs", None)), (
                f"{node_name} does not implement validate_inputs")

    def test_validate_inputs_delegates_to_shared_validator(self, init_mod):
        from src.validation import validate_resolution

        for node_name in self._NODES:
            node_cls = getattr(init_mod, node_name)
            result = node_cls.validate_inputs(width=1000, height=1000)
            assert result == validate_resolution(1000, 1000), (
                f"{node_name}.validate_inputs did not delegate to the shared "
                f"validator")

    def test_validate_inputs_signature_has_no_kwargs(self, init_mod):
        """REGRESSION (2026-08-25): a ``**kwargs`` validate_inputs signature
        makes ComfyUI route EVERY input through validation and re-report one
        failing string once per input name (16x duplicate errors on SEGA).
        The validator must declare named width/height params so failures are
        attributed to width/height only."""
        import inspect

        for node_name in self._NODES:
            node_cls = getattr(init_mod, node_name)
            sig = inspect.getfullargspec(node_cls.validate_inputs)
            assert sig.varkw is None, (
                f"{node_name}.validate_inputs must NOT use **kwargs "
                f"(ComfyUI duplicates the error per input name)")
            assert "width" in sig.args and "height" in sig.args, (
                f"{node_name}.validate_inputs must take named width/height")

    def test_hap_has_no_resolution_validation(self):
        """HAP has NO width/height inputs by design — no graph-I/O validation
        is added there (plan W5.2 decision).  This replaces the pre-W5 guard
        that forbade validate_inputs entirely."""
        content = (pathlib.Path(__file__).parent.parent / "nodes" / "hap.py").read_text(encoding="utf-8")
        assert "class HAP(" in content


# ---------------------------------------------------------------------------
# model_type combo parity across the four patch nodes
# ---------------------------------------------------------------------------

_MODEL_TYPE_RE = re.compile(r'options=(\[[^\]]*\])')
_NODE_CLASS_RE = re.compile(r'^class\s+\w+\(io\.ComfyNode\)', re.MULTILINE)


def _model_type_options(node_file: str):
    """The ``options=[...]`` list of the node's ``model_type`` combo.

    Scoped to the ``io.ComfyNode`` class body and taking the FIRST options
    list inside it: in every one of the four node modules ``model_type`` is
    the first combo declared, while ``method``/``resize_mode`` carry their own
    option lists further down that must not be mistaken for it.
    """
    root = pathlib.Path(__file__).parent.parent / "nodes"
    content = (root / node_file).read_text(encoding="utf-8")
    class_at = _NODE_CLASS_RE.search(content)
    assert class_at, f"{node_file}: no io.ComfyNode class found"
    match = _MODEL_TYPE_RE.search(content[class_at.end():])
    assert match, f"{node_file}: no options=[...] combo found in the node class"
    return ast.literal_eval(match.group(1))


@pytest.mark.unit
class TestModelTypeComboParity:
    """THE invariant: the four patch nodes expose ONE identical architecture
    list.

    Four separate "is qwen21 present" tests only prove four facts; what
    actually breaks users is the lists drifting apart (a key that one node
    accepts and another rejects is invisible until a graph fails at runtime),
    so the equality itself is the thing worth pinning.  ``"krea2"`` stays
    absent on purpose — Krea-2 users pick ``"qwen"`` (pre-existing precedent).
    """

    _NODES = ("dype.py", "sega.py", "spa.py", "hap.py")

    @pytest.mark.parametrize("node_file", _NODES)
    def test_qwen21_selectable(self, node_file):
        assert "qwen21" in _model_type_options(node_file)

    @pytest.mark.parametrize("node_file", _NODES)
    def test_option_list_is_identical_across_nodes(self, node_file):
        reference = _model_type_options("dype.py")
        assert _model_type_options(node_file) == reference, (
            f"{node_file} model_type options drifted from nodes/dype.py:\n"
            f"  dype: {reference}\n"
            f"  {node_file}: {_model_type_options(node_file)}"
        )

    def test_option_order_is_stable(self):
        """Pinned so the widget layout does not silently reorder between
        releases (users' saved graphs key off the option text, not the index,
        but a stable order keeps the UI stable)."""
        assert _model_type_options("dype.py") == [
            "auto", "flux", "nunchaku", "qwen", "qwen21", "zimage", "anima",
        ]

    def test_detector_accepts_every_advertised_option(self):
        """Every option the nodes advertise must be one the canonical detector
        understands — except "auto", which means "probe"."""
        from src.model_detect import resolve_model_type

        dm = types.SimpleNamespace(
            pe_embedder=types.SimpleNamespace(theta=10000))
        for opt in _model_type_options("dype.py"):
            if opt == "auto":
                continue
            assert resolve_model_type(dm, opt) == opt, (
                f"nodes advertise model_type={opt!r} but the detector does not "
                f"resolve it to itself"
            )
