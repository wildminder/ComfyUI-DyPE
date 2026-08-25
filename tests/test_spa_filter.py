"""Tests for the per-layer SPA filter (plan phase P8, G5).

- T8.1 ``parse_layer_filter()`` pure parser (``src/spa.py``);
- T8.2 wrapper integration via the layer counter — see ``TestFilterWrapper``;
- T8.3 node knob + plumbing — see ``TestFilterNodeKnobs``.

Reference: HRDiT ``set_spa_filter(double_ids, single_ids)``; plan §2.7.  Our
hook is module-level (no per-block identity), so the filter is expressed over
the FLAT per-forward attention-call counter (the same index HAP uses).

Markers: @pytest.mark.unit / @pytest.mark.mock_integration
Accept (user-run):
    pytest tests/test_spa_filter.py -k parse
    pytest tests/test_spa_filter.py -k wrapper
    pytest tests/test_spa_filter.py tests/test_spa_node.py
"""

import pathlib
import types

import pytest
import torch

from src import hap
from src.spa import (
    _hrdit_install_hook,
    _make_hrdit_wrapper,
    apply_spa_to_model,
    parse_layer_filter,
)
from src.spa_context import (
    SPAContext,
    get_hrdit_layer_idx,
    get_spa_layer_filter,
    set_hap_context,
    set_hrdit_layer_idx,
    set_hrdit_proportional,
    set_spa_context,
    set_spa_layer_filter,
    set_spa_step_gate,
)

_INIT = pathlib.Path(__file__).parent.parent / "__init__.py"


# ---------------------------------------------------------------------------
# T8.1 — parse_layer_filter()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestParseLayerFilter:
    @pytest.mark.parametrize(
        "spec,expected",
        [
            ("", None),
            (None, None),
            ("   ", None),
            ("3", frozenset({3})),
            ("0", frozenset({0})),
            ("2-5", frozenset({2, 3, 4, 5})),
            ("0-18,38-57", frozenset(set(range(0, 19)) | set(range(38, 58)))),
            ("1, 3 , 5", frozenset({1, 3, 5})),          # whitespace tolerated
            (" 0-2 , 7 ", frozenset({0, 1, 2, 7})),      # whitespace + range
            ("7-7", frozenset({7})),                     # degenerate range
        ],
    )
    def test_parse_table(self, spec, expected):
        assert parse_layer_filter(spec) == expected

    def test_parse_deduplicates_and_sorts(self):
        assert parse_layer_filter("3,1,2,2-4") == frozenset({1, 2, 3, 4})
        assert parse_layer_filter("5,5,5") == frozenset({5})

    @pytest.mark.parametrize(
        "spec",
        [
            "5-2",        # reversed range
            "abc",        # non-integer token
            "a-b",        # non-integer range
            "1-a",        # non-integer range hi
            "1,,2",       # empty token (dangling comma inside)
            "1,2,",       # trailing comma
            ",1",         # leading comma
            "-3",         # negative single index (parses as a bad range)
            "0--2",       # malformed negative range
            "1-2-3",      # too many range parts
        ],
    )
    def test_parse_invalid_raises(self, spec):
        with pytest.raises(ValueError):
            parse_layer_filter(spec)

    def test_parse_error_names_offending_token(self):
        with pytest.raises(ValueError, match="5-2"):
            parse_layer_filter("0-18,5-2")
        with pytest.raises(ValueError, match="abc"):
            parse_layer_filter("abc")


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

def _rand_qkv(B=1, H=2, S=64, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g)
    k = torch.randn(B, H, S, D, generator=g)
    v = torch.randn(B, H, S, D, generator=g)
    return q, k, v


def _identity_spa_ctx(num_variants=3, seq_len=64, head_dim=16):
    """Synthetic SPA context with IDENTITY variant rotations.

    ``s = (num_variants + 1) // 2``; identity rotations make every variant pass
    mathematically plain attention, so the ONLY observable effect of the filter
    is the pass count per layer (3 for SPA layers, 1 for filtered-out layers).
    """
    L, P = seq_len, head_dim // 2
    eye = torch.eye(2).expand(1, 1, L, P, 2, 2).clone()
    return SPAContext(
        active=True,
        bundle_size=(num_variants + 1) // 2,
        base_pe=eye.clone(),
        variant_pes=[eye.clone() for _ in range(num_variants)],
        variant_deltas=[eye.clone() for _ in range(num_variants)],
        pre_roped=True,
        fmt="flux",
    )


def _plan(num_layers=4, num_heads=2, alpha=64.0, beta=0.0):
    """A uniform scope plan (alpha=64 -> band=1 -> half=0)."""
    return hap.ScopePlan(
        alphas=[[alpha] * num_heads for _ in range(num_layers)],
        betas=[[beta] * num_heads for _ in range(num_layers)],
    )


def _hap_ctx(num_layers=4, text_len=0, backend="dense"):
    return hap.HapContext(active=True, plan=_plan(num_layers),
                          text_len=text_len, backend=backend)


class _MockModel:
    def __init__(self):
        self._unet_wrapper = None
        self._spa_installed = None
        self._spa_orig_optimized_attention = None
        self._hrdit_consumers = None
        self._hap_ctx = None

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


@pytest.fixture
def mock_attn():
    """The conftest-provided (pristine SDPA) mock attention module."""
    import comfy.ldm.modules.attention as attn_mod

    return attn_mod


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset singletons + contextvars around every test."""
    hap.HapRuntime.reset()
    set_hrdit_layer_idx(0)
    set_spa_layer_filter(None)
    set_hrdit_proportional(False)
    yield
    set_hap_context(None)
    set_spa_context(None)
    set_spa_step_gate(True)
    set_spa_layer_filter(None)
    set_hrdit_layer_idx(0)
    set_hrdit_proportional(False)
    hap.HapRuntime.reset()


# ---------------------------------------------------------------------------
# T8.2 — wrapper integration via the layer counter
# ---------------------------------------------------------------------------

@pytest.mark.mock_integration
class TestFilterWrapper:
    def test_filter_selects_layers(self):
        """Filter {1,2}: layers 1,2 run SPA (3 passes), layers 0,3 plain (1 pass)."""
        calls_per_layer = []

        def recording_orig(q, k, v, heads, mask=None, attn_precision=None,
                           skip_reshape=False, skip_output_reshape=False, **kw):
            calls_per_layer[-1] += 1
            return q

        wrapper = _make_hrdit_wrapper(recording_orig, is_masked=False)
        set_spa_context(_identity_spa_ctx())
        set_spa_layer_filter(frozenset({1, 2}))

        q, k, v = _rand_qkv()
        for _ in range(4):
            calls_per_layer.append(0)
            wrapper(q, k, v, 2)

        assert calls_per_layer == [1, 3, 3, 1]

    def test_filter_counter_alignment(self):
        """After a filtered layer the counter sequence 0,1,2,3 stays unbroken."""
        set_spa_context(_identity_spa_ctx())
        set_spa_layer_filter(frozenset({1}))

        # W2.1 canonical signature: the wrapper forwards all 8 positional
        # slots to orig, so a bare 4-arg lambda breaks (W2 rot fix).
        wrapper = _make_hrdit_wrapper(
            lambda q, k, v, heads, mask=None, attn_precision=None,
            skip_reshape=False, skip_output_reshape=False, **kw: q,
            is_masked=False)
        q, k, v = _rand_qkv()
        for i in range(4):
            wrapper(q, k, v, 2)
            assert get_hrdit_layer_idx() == i + 1

    def test_filter_none_means_all_layers(self):
        """No filter (None): every layer runs SPA (regression)."""
        calls_per_layer = []

        def recording_orig(q, k, v, heads, mask=None, attn_precision=None,
                           skip_reshape=False, skip_output_reshape=False, **kw):
            calls_per_layer[-1] += 1
            return q

        wrapper = _make_hrdit_wrapper(recording_orig, is_masked=False)
        set_spa_context(_identity_spa_ctx())
        # No set_spa_layer_filter -> default None.
        assert get_spa_layer_filter() is None

        q, k, v = _rand_qkv()
        for _ in range(4):
            calls_per_layer.append(0)
            wrapper(q, k, v, 2)

        assert calls_per_layer == [3, 3, 3, 3]

    def test_filter_interacts_with_hap(self):
        """Filter excludes layer 1 -> layer 1 STILL gets HAP (SPA-only gate).

        Layers 0,2,3 run 3 SPA variant passes each THROUGH the kernel; layer 1
        runs a single (plain-SPA) pass through the kernel.  Total kernel calls
        = 3 + 1 + 3 + 3 = 10, one per attention pass, with the correct layer
        index sequence.
        """
        seen = []
        real_attn = hap.HapRuntime.attn

        def spy(self, q, k, v, layer, **kw):
            seen.append(layer)
            return real_attn(self, q, k, v, layer, **kw)

        wrapper = _make_hrdit_wrapper(lambda q, k, v, heads, **kw: q,
                                      is_masked=False)
        set_spa_context(_identity_spa_ctx())
        set_spa_layer_filter(frozenset({0, 2, 3}))
        set_hap_context(_hap_ctx(num_layers=4))

        hap.HapRuntime.attn = spy
        try:
            q, k, v = _rand_qkv()
            for _ in range(4):
                wrapper(q, k, v, 2)
        finally:
            hap.HapRuntime.attn = real_attn

        # Layer 0: 3 variant passes; layer 1: 1 plain pass; layers 2,3: 3 each.
        assert seen == [0, 0, 0, 1, 2, 2, 2, 3, 3, 3]

    def test_filter_empty_set_skips_all_spa(self):
        """An empty frozenset filter skips SPA on every layer (all plain)."""
        calls_per_layer = []

        def recording_orig(q, k, v, heads, mask=None, attn_precision=None,
                           skip_reshape=False, skip_output_reshape=False, **kw):
            calls_per_layer[-1] += 1
            return q

        wrapper = _make_hrdit_wrapper(recording_orig, is_masked=False)
        set_spa_context(_identity_spa_ctx())
        set_spa_layer_filter(frozenset())

        q, k, v = _rand_qkv()
        for _ in range(4):
            calls_per_layer.append(0)
            wrapper(q, k, v, 2)

        assert calls_per_layer == [1, 1, 1, 1]


@pytest.mark.mock_integration
class TestFilterUnetPlumbing:
    def test_unet_wrapper_activates_filter_from_model_attr(self, mock_attn):
        """The unet wrapper sets the contextvar from ``m._spa_layer_filter``."""
        m = _MockModel()
        m._spa_layer_filter = frozenset({1, 2})
        _hrdit_install_hook(m, "flux", consumer="spa")
        assert m._unet_wrapper is not None

        observed = []

        def model_fn(x, t, **c):
            observed.append(get_spa_layer_filter())
            return x

        m._unet_wrapper(model_fn, {"input": torch.zeros(1),
                                   "timestep": torch.tensor(1.0), "c": {}})
        assert observed == [frozenset({1, 2})]
        # Cleared after the forward -> no cross-model leak.
        assert get_spa_layer_filter() is None

    def test_unet_wrapper_filter_default_none(self, mock_attn):
        """No attr on the model -> the filter stays None during the forward."""
        m = _MockModel()
        _hrdit_install_hook(m, "flux", consumer="spa")
        observed = []

        def model_fn(x, t, **c):
            observed.append(get_spa_layer_filter())
            return x

        m._unet_wrapper(model_fn, {"input": torch.zeros(1),
                                   "timestep": torch.tensor(1.0), "c": {}})
        assert observed == [None]
        assert get_spa_layer_filter() is None


# ---------------------------------------------------------------------------
# T8.3 — node knob + plumbing
# ---------------------------------------------------------------------------

class _PatcherMock:
    """Minimal ModelPatcher stand-in (self-contained)."""

    def __init__(self):
        self.model = types.SimpleNamespace()
        self.model.diffusion_model = types.SimpleNamespace()
        self._object_patches = {}
        self._unet_wrapper = None

    def clone(self):
        new = _PatcherMock()
        dst = types.SimpleNamespace()
        for k, v in vars(self.model.diffusion_model).items():
            setattr(dst, k, v)
        new.model.diffusion_model = dst
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def _make_flux_patcher():
    m = _PatcherMock()
    m.model.diffusion_model.pe_embedder = types.SimpleNamespace(
        theta=10000, axes_dim=[16, 56, 56]
    )
    return m


@pytest.mark.unit
class TestFilterNodeKnobs:
    def _content(self):
        return _INIT.read_text(encoding="utf-8")

    def test_schema_has_filter_input(self):
        content = self._content()
        start = content.index("class SPA(io.ComfyNode):")
        end = content.index("class HAP(io.ComfyNode):")
        assert '"spa_layer_filter"' in content[start:end]

    def test_schema_filter_default_empty(self):
        content = self._content()
        start = content.index("class SPA(io.ComfyNode):")
        end = content.index("class HAP(io.ComfyNode):")
        section = content[start:end]
        idx = section.index('"spa_layer_filter"')
        assert 'default=""' in section[idx:idx + 200]

    def test_execute_signature_has_filter(self):
        assert "spa_layer_filter: str = " in self._content()


@pytest.mark.mock_integration
class TestFilterPlumbing:
    def test_execute_flows_filter(self, mock_attn):
        m = apply_spa_to_model(_make_flux_patcher(), "flux", 4096, 4096, "ntk",
                               enable_spa=True, spa_layer_filter="1,2")
        assert m._spa_layer_filter == frozenset({1, 2})

    def test_execute_default_filter_none(self, mock_attn):
        m = apply_spa_to_model(_make_flux_patcher(), "flux", 4096, 4096, "ntk",
                               enable_spa=True)
        assert m._spa_layer_filter is None
        # Explicit empty string is also None.
        m2 = apply_spa_to_model(_make_flux_patcher(), "flux", 4096, 4096, "ntk",
                                enable_spa=True, spa_layer_filter="")
        assert m2._spa_layer_filter is None

    def test_execute_range_filter(self, mock_attn):
        m = apply_spa_to_model(_make_flux_patcher(), "flux", 4096, 4096, "ntk",
                               enable_spa=True, spa_layer_filter="0-18,38-57")
        assert m._spa_layer_filter == frozenset(
            set(range(0, 19)) | set(range(38, 58))
        )

    def test_execute_invalid_filter_raises(self, mock_attn):
        with pytest.raises(ValueError):
            apply_spa_to_model(_make_flux_patcher(), "flux", 4096, 4096, "ntk",
                               enable_spa=True, spa_layer_filter="5-2")
        with pytest.raises(ValueError):
            apply_spa_to_model(_make_flux_patcher(), "flux", 4096, 4096, "ntk",
                               enable_spa=True, spa_layer_filter="abc")
