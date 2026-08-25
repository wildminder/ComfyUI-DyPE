"""W4.1 — Model-detection parity characterization (IMP-001, test-first).

All four patch entry points (``apply_dype_to_model``, ``apply_sega_to_model``,
``apply_spa_to_model``, ``apply_hap_to_model``) must resolve the SAME model
type for the SAME diffusion-model shape.  Historically each family carried its
own detector copy; SPA's copy knows Krea-2 (``SingleStreamDiT`` class-name
check first), while DyPE/SEGA copies do NOT — a user following the README and
passing ``model_type="qwen"`` for Krea-2 gets a silent no-op from DyPE/SEGA.

Mechanism: each family's DETECTION SEAM is monkeypatched with a recorder that
delegates to the real detector and stores the result:

* ``src.spa._spa_resolve_type``          — SPA's wrapper AND HAP (which imports
  it inside ``apply_hap_to_model`` at call time),
* ``src.patch_utils.resolve_model_type`` — the shared detector both DyPE and
  SEGA call after the W4.3 rewire.

The apply call is allowed to FAIL after detection (mock models lack embedder
internals); only the recorded type matters.  Pre-rewire, DyPE/SEGA have no
seam — their parity entries record NOTHING and the test fails, documenting
IMP-001 until W4.3 lands.

Markers: @pytest.mark.unit
"""

import types

import pytest

import src.patch_utils as patch_utils_mod
import src.spa as spa_mod


# ---------------------------------------------------------------------------
# Mock diffusion-model shapes (detection only needs class name + attr probes)
# ---------------------------------------------------------------------------

def _make_dm(cls_name=None):
    """dm whose class is a REAL (mutable) class so class-name detection sees
    it (SimpleNamespace rejects __class__ assignment on Python 3.13+)."""
    if cls_name:
        cls = type(cls_name, (), {
            "patch_size": 2,
            "pe_embedder": types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56]),
        })
        return cls()
    return types.SimpleNamespace(
        patch_size=2,
        pe_embedder=types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56]),
    )


def _zimage_dm():
    dm = types.SimpleNamespace(patch_size=2)
    dm.rope_embedder = types.SimpleNamespace()
    dm.axes_lens = [128, 64, 64]
    return dm


def _anima_dm():
    cls = type("AnimaDIT", (), {
        "patch_spatial": 2,
        "pos_embedder": types.SimpleNamespace(dim_spatial_range=[0, 1, 2]),
    })
    return cls()


def _nunchaku_dm():
    inner = types.SimpleNamespace(
        config=types.SimpleNamespace(patch_size=2),
        pos_embed=types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56]),
    )
    return types.SimpleNamespace(model=inner)


_SHAPES = {
    "flux": lambda: _make_dm(None),
    "qwen": lambda: _make_dm("QwenImageDiT"),
    "zimage": _zimage_dm,
    "anima": _anima_dm,
    "krea2": lambda: _make_dm("SingleStreamDiT"),
    "nunchaku": _nunchaku_dm,
}


class _Patcher:
    """Minimal ModelPatcher stand-in carrying a given dm."""

    def __init__(self, dm):
        # model_sampling lives UNDER ``model`` — apply_dype_to_model reads
        # ``m.model.model_sampling.sigma_max`` (patch_utils.py).
        self.model = types.SimpleNamespace(
            diffusion_model=dm,
            model_sampling=types.SimpleNamespace(
                sigma_max=types.SimpleNamespace(item=lambda: 1.0)),
        )
        self._object_patches = {}
        self._unet_wrapper = None
        self._spa_installed = None
        self._spa_orig_optimized_attention = None
        self._hrdit_consumers = None
        self._hap_ctx = None

    def clone(self):
        new = _Patcher(self.model.diffusion_model)
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


# ---------------------------------------------------------------------------
# Detection recorder
# ---------------------------------------------------------------------------

class _DetectionRecorder:
    """Wrap a family's detection seam; store every resolved type."""

    def __init__(self):
        self.results = []

    def record(self, value):
        self.results.append(value)


@pytest.fixture
def detections(monkeypatch):
    """Install ONE recorder on the canonical seam and return a lookup callable.

    After the W4.3 rewire every family routes detection through
    ``src.model_detect.resolve_model_type`` (DyPE/SEGA via a function-local
    import, SPA via the ``_spa_resolve_type`` adapter, HAP via the same
    adapter imported at call time) — so a single recorder observes them all.

    ``detect(family, shape_name)`` runs the family's apply entry point on the
    mock shape and returns the recorded type (or None when the family never
    reached the canonical seam).
    """
    import src.model_detect as model_detect_mod

    rec = _DetectionRecorder()
    real = model_detect_mod.resolve_model_type

    def rec_resolve(dm, requested="auto"):
        got = real(dm, requested)
        rec.record(got)
        return got

    monkeypatch.setattr(model_detect_mod, "resolve_model_type", rec_resolve)

    def detect(family, shape_name):
        dm = _SHAPES[shape_name]()
        before = len(rec.results)
        try:
            if family == "dype":
                patch_utils_mod.apply_dype_to_model(
                    _Patcher(dm), "auto", 2048, 2048, "ntk", False,
                    enable_dype=False, dype_scale=1.0, dype_exponent=1.0,
                    base_shift=0.5, max_shift=1.15,
                )
            elif family == "sega":
                patch_utils_mod.apply_sega_to_model(
                    _Patcher(dm), "auto", 2048, 2048, method="sega",
                )
            elif family == "spa":
                spa_mod.apply_spa_to_model(
                    _Patcher(dm), "auto", 2048, 2048, "ntk",
                    enable_spa=True, bundle_size=3,
                )
            elif family == "hap":
                from src.hap import ScopePlan, apply_hap_to_model

                plan = ScopePlan(alphas=[[0.0, 0.0]], betas=[[0.5, 0.5]])
                apply_hap_to_model(_Patcher(dm), "auto", plan)
            else:  # pragma: no cover
                raise ValueError(family)
        except Exception:
            # Post-detection mock gaps (missing embedder internals etc.) are
            # irrelevant — the recorded type is what parity requires.
            pass
        results = rec.results
        return results[before] if len(results) > before else None

    return detect


# ---------------------------------------------------------------------------
# Parity table
# ---------------------------------------------------------------------------

_FAMILIES = ["dype", "sega", "spa", "hap"]
_SHAPE_NAMES = ["flux", "qwen", "zimage", "anima", "krea2", "nunchaku"]


@pytest.mark.unit
class TestDetectionParity:
    @pytest.mark.parametrize("shape_name", _SHAPE_NAMES)
    def test_all_families_reach_a_detection_seam(self, detections, shape_name):
        """Precondition: after the W4.3 rewire EVERY family records a type for
        every shape (no silent divergent detectors)."""
        for family in _FAMILIES:
            assert detections(family, shape_name) is not None, (
                f"{family} never reached a detection seam for shape "
                f"{shape_name!r} — the family still uses an inline detector "
                f"copy (IMP-001 not fixed)"
            )

    @pytest.mark.parametrize("shape_name", _SHAPE_NAMES)
    def test_auto_detection_identical_across_families(self, detections, shape_name):
        """THE parity invariant: all four entry points resolve the SAME type."""
        results = {f: detections(f, shape_name) for f in _FAMILIES}
        unique = set(results.values())
        assert len(unique) == 1, (
            f"detection diverges for shape {shape_name!r}: {results}"
        )

    def test_krea2_resolved_as_krea2_everywhere(self, detections):
        """THE IMP-001 case: Krea-2's SingleStreamDiT must resolve to 'krea2'
        everywhere (pre-rewire DyPE/SEGA fall back to 'flux')."""
        results = {f: detections(f, "krea2") for f in _FAMILIES}
        assert all(r == "krea2" for r in results.values()), (
            f"Krea-2 detection diverges across families: {results}"
        )

    def test_canonical_detector_matches_spa_semantics(self):
        """The extracted ``resolve_model_type`` must reproduce SPA's
        precedence table exactly (class-name checks FIRST)."""
        from src.model_detect import resolve_model_type

        for shape_name, factory in _SHAPES.items():
            dm = factory()
            expected = spa_mod._spa_resolve_type("auto", dm)
            got = resolve_model_type(dm, "auto")
            assert got == expected, (
                f"canonical detector returned {got!r} for {shape_name!r}; "
                f"SPA semantics say {expected!r}"
            )

    def test_requested_override_and_unknown(self):
        """Explicit requested types win over probes; unknown raises ValueError."""
        from src.model_detect import resolve_model_type

        dm = _SHAPES["flux"]()
        assert resolve_model_type(dm, "qwen") == "qwen"
        assert resolve_model_type(dm, "nunchaku") == "nunchaku"
        with pytest.raises(ValueError, match="not a compatible"):
            resolve_model_type(types.SimpleNamespace(), "auto")
