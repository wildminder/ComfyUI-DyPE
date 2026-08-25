"""Tests for the SPA ComfyUI node (schema + patching behaviour).

The functional (patching) tests exercise ``apply_spa_to_model`` directly rather
than loading the extension's ``__init__.py``.  Loading ``__init__.py`` transitively
imports the Qwen2D VAE patch / PixelRush / FreeScale nodes, which require the
*real* ``comfy`` package (``comfy.model_management``, ``comfy.ops``,
``comfy.ldm...``) that is not available under pytest's mocked ``comfy_api``.
``apply_spa_to_model`` itself only needs ``comfy_api.latest.io`` (mocked), so it
is the right unit to verify end-to-end patching for every supported model type.
The node *wiring* (schema, inputs, registration) is covered separately by the
schema text-checks below, which read ``__init__.py`` directly.
"""
import pathlib
import re
import types

import pytest

_INIT = pathlib.Path(__file__).parent.parent / "__init__.py"


class _MockModel:
    """Minimal stand-in for comfy.model_patcher.ModelPatcher.

    Self-contained so the SPA node test does not depend on the (real-comfy
    incompatible) ``mock_flux_model`` conftest fixture.
    """

    def __init__(self):
        self.model = types.SimpleNamespace()
        self.model.diffusion_model = types.SimpleNamespace()
        self._object_patches = {}

    def _copy_dm(self, src):
        dst = types.SimpleNamespace()
        for k, v in vars(src).items():
            setattr(dst, k, v)
        return dst

    def clone(self):
        new = _MockModel()
        new.model.diffusion_model = self._copy_dm(self.model.diffusion_model)
        new._object_patches = dict(self._object_patches)
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def _make_flux_mock():
    m = _MockModel()
    m.model.diffusion_model.pe_embedder = types.SimpleNamespace(
        theta=10000, axes_dim=[16, 56, 56]
    )
    return m


def _make_qwen_mock():
    m = _MockModel()
    m.model.diffusion_model.pe_embedder = types.SimpleNamespace(
        theta=10000, axes_dim=[16, 56, 56]
    )
    return m


def _make_zimage_mock():
    m = _MockModel()
    m.model.diffusion_model.rope_embedder = types.SimpleNamespace(
        theta=10000, axes_dim=[16, 56, 56]
    )
    return m


def _make_nunchaku_mock():
    m = _MockModel()
    m.model.diffusion_model.model = types.SimpleNamespace(
        pos_embed=types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56])
    )
    return m


def _make_anima_mock():
    m = _MockModel()
    # apply_spa_to_model computes Anima theta/axes_dim from these model attrs
    # (it does not read them off the original embedder).
    dm = m.model.diffusion_model
    dm.model_channels = 1152
    dm.num_heads = 16
    dm.rope_h_extrapolation_ratio = 1.0
    dm.rope_w_extrapolation_ratio = 1.0
    dm.rope_t_extrapolation_ratio = 1.0
    dm.pos_embedder = types.SimpleNamespace(
        theta=[10000.0, 10000.0, 10000.0], axes_dim=[44, 42, 42]
    )
    return m


@pytest.mark.unit
class TestSpaNodeSchema:
    def test_node_class_defined(self):
        assert "class SPA" in _INIT.read_text(encoding="utf-8")

    def test_node_registered(self):
        content = _INIT.read_text(encoding="utf-8")
        assert "return [DyPE_FLUX, SEGA, SPA" in content

    def test_imports_apply_spa(self):
        assert "apply_spa_to_model" in _INIT.read_text(encoding="utf-8")

    def test_filter_error_scope_is_narrow(self):
        """The 'invalid spa_layer_filter' prefix must wrap ONLY filter-parse
        failures — not every ValueError from apply_spa_to_model (2026-08-24:
        the mutual-exclusion guard surfaced as 'invalid spa_layer_filter'
        because the node re-wrapped all ValueErrors)."""
        content = _INIT.read_text(encoding="utf-8")
        start = content.index("class SPA(io.ComfyNode):")
        end = content.index("class DyPEExtension")
        spa_section = content[start:end]
        # Pre-parses the filter in its own try/except.
        assert "parse_layer_filter(spa_layer_filter)" in spa_section
        assert "invalid spa_layer_filter" in spa_section
        # The apply call itself is NOT inside a blanket try/except ValueError
        # that would re-wrap unrelated errors with the filter message.
        assert "except ValueError as exc:\n            raise type(exc)" not in spa_section

    def test_has_bundle_size_input(self):
        content = _INIT.read_text(encoding="utf-8")
        assert "bundle_size" in content

    def test_has_enable_spa_input(self):
        assert "enable_spa" in _INIT.read_text(encoding="utf-8")

    def test_has_model_type_combo(self):
        content = _INIT.read_text(encoding="utf-8")
        for opt in ('"auto"', '"flux"', '"qwen"', '"anima"'):
            assert opt in content

    def test_category_and_output(self):
        content = _INIT.read_text(encoding="utf-8")
        assert "model_patches/position_encoding" in content
        assert "io.Model.Output" in content

    def test_has_spa_start_sigma_input(self):
        """T2.1: the node schema exposes the step-gating ``spa_start_sigma`` input."""
        content = _INIT.read_text(encoding="utf-8")
        assert '"spa_start_sigma"' in content
        assert "io.Float.Input" in content

    def test_no_method_input_on_spa_node(self):
        """Guard: the SPA node must NOT expose a ``method`` combo.

        SPA always applies the model's native no-extrapolation RoPE
        (``ntk_factor=1.0``) on the bundled coords, so the DyPE extrapolation
        methods (ntk/yarn/vision_yarn/pi) are a no-op for SPA.  The knob was
        removed to stop users A/B-testing a dead input.  Scoped to the SPA
        class section only (the DyPE node legitimately keeps its own
        ``method`` input).
        """
        content = _INIT.read_text(encoding="utf-8")
        start = content.index("class SPA(io.ComfyNode):")
        end = content.index("class DyPEExtension")
        spa_section = content[start:end]
        assert '"method"' not in spa_section, (
            "SPA node still exposes a 'method' input — it is a no-op for SPA "
            "and must stay removed")
        # execute() must not accept a method parameter either.
        assert "def execute(cls, model, width: int, height: int, model_type: str, enable_spa: bool" in spa_section
        assert "method: str" not in spa_section


@pytest.mark.unit
class TestSpaNodePatching:
    """Functional tests that drive ``apply_spa_to_model`` end-to-end.

    These mirror exactly what ``SPA.execute`` does (the node body is thin glue:
    ``bs = None if bundle_size<=0 else int(bundle_size)`` then
    ``apply_spa_to_model(model, model_type, width, height,
    enable_spa=enable_spa, bundle_size=bs)``).  The node no longer exposes a
    ``method`` input (SPA always uses the model's native no-extrapolation
    RoPE); the tests below pass ``"ntk"`` positionally only because
    ``apply_spa_to_model`` keeps the parameter for the DyPE-base constructor
    chain — it is a no-op for the SPA math.
    """

    def test_patches_flux_with_auto_bundle_size(self):
        from src.models.spa_flux import PosEmbedSPAFlux
        from src.spa import apply_spa_to_model

        out = apply_spa_to_model(_make_flux_mock(), "flux", 4096, 4096, "ntk", enable_spa=True)
        embedder = out._object_patches["diffusion_model.pe_embedder"]
        assert isinstance(embedder, PosEmbedSPAFlux)
        # auto -> bundle_size == 0 (minimal in-distribution compression), paper-N
        assert embedder.bundle_size == 0
        assert embedder.enable_spa is True

    def test_auto_default_is_zero(self):
        from src.models.spa_flux import PosEmbedSPAFlux
        from src.spa import apply_spa_to_model

        # SPA is always active when enabled (auto=0 regardless of resolution);
        # only an explicit bundle_size==1 yields a no-op passthrough.
        out = apply_spa_to_model(_make_flux_mock(), "flux", 1024, 1024, "ntk", enable_spa=True)
        embedder = out._object_patches["diffusion_model.pe_embedder"]
        assert isinstance(embedder, PosEmbedSPAFlux)
        assert embedder.bundle_size == 0

    def test_explicit_n_stored_verbatim_and_pass_count(self):
        import torch

        from src.models.spa_flux import PosEmbedSPAFlux
        from src.spa import (
            SPA_MAX_PASSES,
            apply_spa_to_model,
            build_bundle_id_variants,
            derive_bundle_s,
        )

        def _latent_ids(H, W):
            ids = torch.zeros(1, H * W, 3)
            ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
            ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
            return ids

        # Paper-N semantics: an explicit knob (2..8) is stored verbatim.
        out = apply_spa_to_model(_make_flux_mock(), "flux", 1024, 1024, "ntk", enable_spa=True, bundle_size=3)
        embedder = out._object_patches["diffusion_model.pe_embedder"]
        assert isinstance(embedder, PosEmbedSPAFlux)
        assert embedder.bundle_size == 3  # stored verbatim (paper's N)

        # derive_bundle_s is the single source of truth.  At a 128x128 grid
        # (max_pos=127 > trained_extent=64): auto -> s=ceil(127/79)=2 (3 passes);
        # N=3 -> s=max(3,2)=3 (5 passes); N=5 -> s=5 (9 passes).  All in-dist.
        ids = _latent_ids(128, 128)
        assert derive_bundle_s(127, 0) == 2
        assert derive_bundle_s(127, 3) == 3
        assert derive_bundle_s(127, 5) == 5
        assert len(build_bundle_id_variants(ids, 0)) == 3
        assert len(build_bundle_id_variants(ids, 3)) == 5
        assert len(build_bundle_id_variants(ids, 5)) == 9
        # Pass count never exceeds the cap.
        for N in (0, 2, 3, 5, 8):
            assert len(build_bundle_id_variants(ids, N)) <= SPA_MAX_PASSES

    def test_legacy_group_num_values_migrate_to_auto(self, caplog):
        """Decision M1: legacy knob values >= 32 (old group_num semantics) are
        migrated to auto (0) with a one-time WARNING, not stored verbatim."""
        import logging

        from src.models.spa_flux import PosEmbedSPAFlux
        from src.spa import apply_spa_to_model

        # Reset the one-time warning latch so caplog captures it deterministically.
        apply_spa_to_model._spa_legacy_warned = False
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            out = apply_spa_to_model(_make_flux_mock(), "flux", 1024, 1024, "ntk",
                                     enable_spa=True, bundle_size=80)
        embedder = out._object_patches["diffusion_model.pe_embedder"]
        assert isinstance(embedder, PosEmbedSPAFlux)
        # Legacy 80 -> migrated to auto (0), NOT stored as 80.
        assert embedder.bundle_size == 0
        assert any("legacy" in r.message.lower() for r in caplog.records)

    def test_explicit_bundle_size_one_is_off(self):
        from src.models.spa_flux import PosEmbedSPAFlux
        from src.spa import apply_spa_to_model

        # bundle_size == 1 is the explicit OFF knob (true passthrough) and must NOT
        # be clamped (the cost guard only applies to active, non-1 values).
        out = apply_spa_to_model(_make_flux_mock(), "flux", 1024, 1024, "ntk", enable_spa=True, bundle_size=1)
        embedder = out._object_patches["diffusion_model.pe_embedder"]
        assert isinstance(embedder, PosEmbedSPAFlux)
        assert embedder.bundle_size == 1

    def test_disable_spa(self):
        from src.models.spa_flux import PosEmbedSPAFlux
        from src.spa import apply_spa_to_model

        out = apply_spa_to_model(_make_flux_mock(), "flux", 4096, 4096, "ntk", enable_spa=False)
        embedder = out._object_patches["diffusion_model.pe_embedder"]
        assert isinstance(embedder, PosEmbedSPAFlux)
        assert embedder.enable_spa is False

    def test_spa_start_sigma_stored_on_patcher(self):
        """T2.1: ``spa_start_sigma`` flows from apply_spa_to_model onto the patcher."""
        from src.spa import apply_spa_to_model

        # Default: 1.0 (always active, backward compatible).
        out = apply_spa_to_model(_make_flux_mock(), "flux", 4096, 4096, "ntk", enable_spa=True)
        assert getattr(out, "_spa_start_sigma", None) == 1.0

        # Explicit threshold is stored verbatim for the unet wrapper to read.
        out2 = apply_spa_to_model(_make_flux_mock(), "flux", 4096, 4096, "ntk",
                                  enable_spa=True, spa_start_sigma=0.7)
        assert out2._spa_start_sigma == 0.7

    def test_spa_steps_param(self):
        """T2.1: ``spa_steps`` (leading-step count gate) flows onto the patcher.

        Default is 3 (HRDiT ``--spa_steps [3, 0]``); 0 = all steps (backward
        compat); the step counter / last-sigma slots are initialized so the
        unet wrapper's generation-boundary detection starts clean.
        """
        from src.spa import apply_spa_to_model

        # Default: 3 leading steps (HRDiT-faithful speed/quality tradeoff).
        out = apply_spa_to_model(_make_flux_mock(), "flux", 4096, 4096, "ntk", enable_spa=True)
        assert getattr(out, "_spa_steps", None) == 3
        assert getattr(out, "_spa_step_counter", None) == 0
        assert getattr(out, "_spa_last_sigma", "sentinel") is None

        # Explicit value is stored verbatim.
        out5 = apply_spa_to_model(_make_flux_mock(), "flux", 4096, 4096, "ntk",
                                  enable_spa=True, spa_steps=5)
        assert out5._spa_steps == 5

        # 0 = all steps (backward compatible).
        out0 = apply_spa_to_model(_make_flux_mock(), "flux", 4096, 4096, "ntk",
                                  enable_spa=True, spa_steps=0)
        assert out0._spa_steps == 0

        # Negative values are clamped to 0 (all steps), never a broken gate.
        outneg = apply_spa_to_model(_make_flux_mock(), "flux", 4096, 4096, "ntk",
                                    enable_spa=True, spa_steps=-2)
        assert outneg._spa_steps == 0

    def test_qwen_and_zimage_patched(self):
        from src.models.spa_qwen import PosEmbedSPAQwen
        from src.models.spa_zimage import PosEmbedSPAZImage
        from src.spa import apply_spa_to_model

        out_q = apply_spa_to_model(_make_qwen_mock(), "qwen", 4096, 4096, "ntk", enable_spa=True)
        assert isinstance(out_q._object_patches["diffusion_model.pe_embedder"], PosEmbedSPAQwen)

        out_z = apply_spa_to_model(_make_zimage_mock(), "zimage", 4096, 4096, "ntk", enable_spa=True)
        assert isinstance(out_z._object_patches["diffusion_model.rope_embedder"], PosEmbedSPAZImage)

    def test_nunchaku_and_anima_patched(self):
        from src.models.spa_anima import PosEmbedSPAAnima
        from src.spa import apply_spa_to_model

        # Nunchaku is unsupported (decision 4): SPA returns the model UNCHANGED and
        # does NOT install an embedder patch for it.
        out_n = apply_spa_to_model(_make_nunchaku_mock(), "nunchaku", 4096, 4096, "ntk", enable_spa=True)
        assert "diffusion_model.model.pos_embed" not in out_n._object_patches

        out_a = apply_spa_to_model(_make_anima_mock(), "anima", 4096, 4096, "ntk", enable_spa=True)
        embedder = out_a._object_patches["diffusion_model.pos_embedder"]
        assert isinstance(embedder, PosEmbedSPAAnima)
        # Anima: axes_dim derived from model_channels/num_heads.
        # head_dim = 1152//16 = 72; dim_h = (72//6)*2 = 24; dim_t = 72-2*24 = 24
        assert embedder.axes_dim == [24, 24, 24]
        assert embedder.thetas == [10000.0, 10000.0, 10000.0]

    def test_auto_detects_flux_via_pe_embedder(self):
        """auto with a FLUX-like diffusion_model (has pe_embedder) selects flux."""
        from src.models.spa_flux import PosEmbedSPAFlux
        from src.spa import apply_spa_to_model

        out = apply_spa_to_model(_make_flux_mock(), "auto", 4096, 4096, "ntk", enable_spa=True)
        assert isinstance(out._object_patches["diffusion_model.pe_embedder"], PosEmbedSPAFlux)
        assert out._object_patches["diffusion_model.pe_embedder"].bundle_size == 0


@pytest.mark.unit
class TestSpaDoubleApplyWarning:
    """W9.g (NTH-108): a GENUINE SPA re-apply warns; the HAP chain does not."""

    def test_double_apply_warns(self, caplog):
        """Applying SPA onto a patcher that ALREADY has the hook AND an SPA
        embedder emits the double-application WARNING."""
        import logging

        from src.models.spa_flux import PosEmbedSPAFlux
        from src.spa import apply_spa_to_model

        apply_spa_to_model(  # first apply establishes the baseline state
            _make_flux_mock(), "flux", 2048, 2048, "ntk",
            enable_spa=True, bundle_size=3,
        )
        # Simulate a genuine re-apply: the SOURCE patcher carries BOTH the
        # hook flag AND an SPA embedder AS THE LIVE ATTRIBUTE (the mock clone
        # copies dm attributes, mirroring how get_model_object resolves the
        # patched embedder in real ComfyUI).
        second_src = _make_flux_mock()
        second_src._spa_installed = True
        second_src.model.diffusion_model.pe_embedder = PosEmbedSPAFlux(
            theta=10000, axes_dim=[16, 56, 56], method="ntk",
            base_resolution=1024, enable_spa=True, bundle_size=3,
        )
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            apply_spa_to_model(
                second_src, "flux", 2048, 2048, "ntk",
                enable_spa=True, bundle_size=3,
            )
        assert any(
            "applied twice" in r.message for r in caplog.records
        ), "expected the double-application WARNING"

    def test_fresh_apply_no_warning(self, caplog):
        """A fresh apply (no prior hook) must NOT warn."""
        import logging

        from src.spa import apply_spa_to_model

        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            apply_spa_to_model(
                _make_flux_mock(), "flux", 2048, 2048, "ntk",
                enable_spa=True, bundle_size=3,
            )
        assert not any("applied twice" in r.message for r in caplog.records)

    def test_hap_chain_no_false_positive(self, caplog):
        """SPA-after-HAP: the source patcher HAS _spa_installed but its
        embedder is NOT an SPA embedder -> no warning."""
        import logging

        from src.spa import apply_spa_to_model

        src_patcher = _make_flux_mock()
        src_patcher._spa_installed = True  # HAP installed the shared hook
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            apply_spa_to_model(
                src_patcher, "flux", 2048, 2048, "ntk",
                enable_spa=True, bundle_size=3,
            )
        assert not any("applied twice" in r.message for r in caplog.records), (
            "false positive: the legitimate HAP->SPA chain was flagged"
        )


# ---------------------------------------------------------------------------
# P3 — install policy / lifecycle (T-P3-3, T-P3-4)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpaInstallPolicy:
    def test_nunchaku_guard_warns_and_skips_patch(self, caplog):
        """T-P3-3: SPA is unsupported on Nunchaku; log a warning and return m UNCHANGED.

        No object patch is applied and the module-level ``optimized_attention`` hook
        is never installed.
        """
        import logging

        import comfy.ldm.modules.attention as attn_mod

        from src.spa import apply_spa_to_model

        orig_attn = attn_mod.optimized_attention
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            out = apply_spa_to_model(
                _make_nunchaku_mock(), "nunchaku", 4096, 4096, "ntk", enable_spa=True
            )
        # The model is returned unchanged: no embedder patch added.
        assert "diffusion_model.model.pos_embed" not in out._object_patches
        # The optimized_attention hook was NOT installed.
        assert attn_mod.optimized_attention is orig_attn
        # A clear warning was emitted mentioning Nunchaku.
        assert any("Nunchaku" in r.message for r in caplog.records)

    def test_bundle_size_one_installs_no_hook(self):
        """T-P3-4: bundle_size==1 -> no attention hook; module-level optimized_attention untouched."""
        import comfy.ldm.modules.attention as attn_mod

        from src.models.spa_flux import PosEmbedSPAFlux
        from src.spa import apply_spa_to_model

        orig_attn = attn_mod.optimized_attention
        out = apply_spa_to_model(
            _make_flux_mock(), "flux", 4096, 4096, "ntk", enable_spa=True, bundle_size=1
        )
        # Embedder IS replaced (base RoPE passthrough), but the hook must not be installed.
        assert "diffusion_model.pe_embedder" in out._object_patches
        assert isinstance(out._object_patches["diffusion_model.pe_embedder"], PosEmbedSPAFlux)
        assert out._object_patches["diffusion_model.pe_embedder"].bundle_size == 1
        # The module-level optimized_attention remains the original (untouched).
        assert attn_mod.optimized_attention is orig_attn


# ---------------------------------------------------------------------------
# P5 — composition / mutual exclusivity (T-P5-1, T-P5-2, T-P5-3, T-P5-4)
# ---------------------------------------------------------------------------
#
# Remediation decision 6: SPA and DyPE/SEGA are mutually exclusive in v1.  The
# guards live in two places and must BOTH reject:
#   * ``src.spa._spa_ensure_no_incompatible_embedder`` — rejects a DyPE/SEGA
#     embedder when SPA is being applied.
#   * ``src.patch_utils._dype_sega_reject_spa`` — rejects an SPA embedder when
#     DyPE/SEGA is being applied.
#
# The mock's ``add_object_patch`` only records into ``_object_patches`` (it does
# NOT mutate the live embedder attribute), so to exercise the guard we install
# the *incompatible* embedder instance directly onto the live
# ``m.model.diffusion_model.pe_embedder`` — exactly what the guards read.

_SRC = pathlib.Path(__file__).parent.parent / "src"


@pytest.mark.unit
class TestSpaComposition:
    # --- SPA rejects DyPE / SEGA (apply_spa_to_model on an incompatible model) ---
    def test_spa_rejects_dype_embedder(self):
        """T-P5-1: applying SPA onto a DyPE-patched FLUX model raises ValueError."""
        from src.models.flux import PosEmbedFlux
        from src.spa import apply_spa_to_model

        m = _make_flux_mock()
        # Simulate a model already patched by DyPE (live embedder, what the guard reads).
        m.model.diffusion_model.pe_embedder = PosEmbedFlux(10000, [16, 56, 56])
        with pytest.raises(ValueError, match="mutually exclusive"):
            apply_spa_to_model(m, "flux", 4096, 4096, "ntk", enable_spa=True)

    def test_spa_rejects_sega_embedder(self):
        """T-P5-4: applying SPA onto a SEGA-patched FLUX model raises ValueError."""
        from src.models.sega_flux import SegAPosEmbedFlux
        from src.spa import apply_spa_to_model

        m = _make_flux_mock()
        # Simulate a model already patched by SEGA.
        m.model.diffusion_model.pe_embedder = SegAPosEmbedFlux(10000, [16, 56, 56])
        with pytest.raises(ValueError, match="mutually exclusive"):
            apply_spa_to_model(m, "flux", 4096, 4096, "ntk", enable_spa=True)

    # --- DyPE / SEGA reject SPA (apply_dype/apply_sega on an SPA-patched model) ---
    def test_dype_rejects_spa_embedder(self):
        """T-P5-2: applying DyPE onto an SPA-patched FLUX model raises ValueError."""
        from src.models.spa_flux import PosEmbedSPAFlux
        from src.patch_utils import apply_dype_to_model

        m = _make_flux_mock()
        # Simulate a model already patched by SPA (live embedder, what the guard reads).
        m.model.diffusion_model.pe_embedder = PosEmbedSPAFlux(10000, [16, 56, 56])
        with pytest.raises(ValueError, match="mutually exclusive"):
            apply_dype_to_model(m, "flux", 4096, 4096, "ntk", False, False, 2.0, 2.0, 0.5, 1.15)

    def test_sega_rejects_spa_embedder(self):
        """T-P5-4: applying SEGA onto an SPA-patched FLUX model raises ValueError."""
        from src.models.spa_flux import PosEmbedSPAFlux
        from src.patch_utils import apply_sega_to_model

        m = _make_flux_mock()
        m.model.diffusion_model.pe_embedder = PosEmbedSPAFlux(10000, [16, 56, 56])
        with pytest.raises(ValueError, match="mutually exclusive"):
            apply_sega_to_model(m, "flux", 4096, 4096)

    # --- Improved guard message + node error scoping (2026-08-24) ------------

    def test_guard_message_names_embedder_class_and_recovery(self):
        """The SEGA->SPA guard names the offending embedder class, the method
        kind (SEGA/DyPE), and the recovery path (reload)."""
        from src.models.sega_anima import SegAPosEmbedAnima
        from src.spa import _spa_ensure_no_incompatible_embedder

        embedder = SegAPosEmbedAnima(theta=[10000.0, 10000.0, 10000.0],
                                     axes_dim=[16, 56, 56])
        with pytest.raises(ValueError) as exc_info:
            _spa_ensure_no_incompatible_embedder(embedder)
        msg = str(exc_info.value)
        assert "SegAPosEmbedAnima" in msg      # embedder class named
        assert "SEGA" in msg                    # method kind named
        assert "reload" in msg.lower()          # recovery path given
        assert "mutually exclusive" in msg      # original semantics kept

    def test_guard_message_dype_kind(self):
        """A DyPE (non-SEGA) embedder reports kind 'DyPE'."""
        from src.models.flux import PosEmbedFlux
        from src.spa import _spa_ensure_no_incompatible_embedder

        embedder = PosEmbedFlux(10000, [16, 56, 56])
        with pytest.raises(ValueError) as exc_info:
            _spa_ensure_no_incompatible_embedder(embedder)
        msg = str(exc_info.value)
        assert "DyPE" in msg
        assert "PosEmbedFlux" in msg

    def test_parse_layer_filter_idempotent_on_frozenset(self):
        """parse_layer_filter passes an already-parsed frozenset through
        unchanged (the node pre-parses; apply_spa_to_model re-parses)."""
        from src.spa import parse_layer_filter

        fs = frozenset({0, 1, 5})
        assert parse_layer_filter(fs) is fs
        assert parse_layer_filter(None) is None


@pytest.mark.unit
class TestSpaNoLegacyAveraging:
    def test_no_legacy_rope_matrix_averaging(self):
        """T-P5-3: the root-cause bug (average RoPE *matrices* then one softmax) is gone.

        The fix runs N attention passes and averages the *attention outputs*
        (``torch.stack(outs, dim=0).mean(dim=0)`` in ``spa_attn.py``).  Assert that:
          * ``spa.py`` never averages any tensor (the embedder/base path is matrix-free),
          * the only stacked-then-meaned object in the attention path is the *outputs*
            (``outs``), never the RoPE embeddings (``embs``) — the legacy pattern is
            absent from the code that performs attention,
          * every SPA model adapter explicitly documents that the legacy embedding-
            averaging path was removed (a positive signal, not executable code).
        """
        spa_src = _SRC.joinpath("spa.py").read_text(encoding="utf-8")
        # The base embedder path must never reduce tensors via .mean(...)
        assert ".mean(" not in spa_src, "spa.py must not average any tensor (RoPE math)"

        attn_src = _SRC.joinpath("spa_attn.py").read_text(encoding="utf-8")
        # Faithful mechanism: average ATTENTION OUTPUTS.
        assert "torch.stack(outs, dim=0).mean(dim=0)" in attn_src
        # Legacy bug: average RoPE rotation MATRICES before attention — must be absent
        # from the module that actually performs the averaging.
        assert "torch.stack(embs" not in attn_src

        # Every SPA model adapter must document that the legacy embedding-averaging
        # path is removed (the only `torch.stack(embs` mentions live in such docstrings).
        documented_removal = False
        for f in _SRC.glob("models/spa_*.py"):
            src = f.read_text(encoding="utf-8")
            if "torch.stack(embs" in src:
                # It must be framed as "the legacy ... path is removed", not live code.
                assert "removed" in src, f"{f.name}: legacy embedding averaging not documented as removed"
                documented_removal = True
        assert documented_removal, "expected at least one SPA adapter to document legacy-path removal"


# ---------------------------------------------------------------------------
# P6 — documentation assertion (T-P6-1)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpaDocs:
    def test_readme_documents_faithful_mechanism_and_constraints(self):
        """T-P6-1: README documents the output-averaging mechanism, mutual exclusivity,
        and Nunchaku non-support (the three corrections made during remediation)."""
        readme = pathlib.Path(__file__).parent.parent / "README.md"
        content = readme.read_text(encoding="utf-8")
        # Faithful mechanism: average attention OUTPUTS (not RoPE rotation matrices).
        assert "averages the attention outputs" in content
        # SPA and DyPE/SEGA are mutually exclusive in v1.
        assert "mutually exclusive" in content
        # Nunchaku is explicitly unsupported for SPA.
        assert re.search(r"Nunchaku (is not supported|not supported)", content), (
            "README must state that Nunchaku is not supported for SPA")


# ---------------------------------------------------------------------------
# P4 (2026-08-16 fix) — SPA-side carry-over of HRDiT state across clone()
# ---------------------------------------------------------------------------
#
# The real ModelPatcher.clone() drops custom attributes.  apply_spa_to_model
# clones the incoming patcher, so any HRDiT state the source already carries
# (e.g. an HAP runtime installed by an upstream HAP node) must be carried onto
# the clone — otherwise HAP silently dies the moment an SPA node is chained
# after it.  The HAP-side mirror of this test lives in test_hap_node.py
# (TestChainOrderIndependence); this class covers the SPA apply function's
# perspective.

@pytest.mark.unit
class TestSpaCarriesHrditState:
    def test_spa_after_hap_carries_hap_ctx(self):
        """apply_spa_to_model carries an existing _hap_ctx across its clone()."""
        from src.spa import apply_spa_to_model

        m = _make_flux_mock()
        sentinel = object()
        m._hap_ctx = sentinel

        out = apply_spa_to_model(m, "flux", 4096, 4096, "ntk", enable_spa=True)
        assert out is not m
        assert getattr(out, "_hap_ctx", None) is sentinel

    def test_spa_carries_hrdit_consumers_and_state_ref(self):
        """apply_spa_to_model carries _hrdit_consumers and re-points _hrdit_state_ref."""
        from src.spa import apply_spa_to_model

        m = _make_flux_mock()
        m._hrdit_consumers = {"spa"}
        m._hrdit_state_ref = [m]

        out = apply_spa_to_model(m, "flux", 4096, 4096, "ntk", enable_spa=True)
        assert getattr(out, "_hrdit_consumers", None) == {"spa"}
        # The shared state ref must now point at the newest clone so the
        # already-installed unet wrapper reads the authoritative state.
        assert getattr(out, "_hrdit_state_ref", [None])[0] is out

    def test_spa_carry_noop_on_bare_model(self):
        """Carry-over must not raise when the source has no HRDiT attributes."""
        from src.spa import apply_spa_to_model

        out = apply_spa_to_model(
            _make_flux_mock(), "flux", 4096, 4096, "ntk", enable_spa=True
        )
        # No HRDiT attrs were present on the source -> none invented on the clone
        # (the install hook may legitimately add its own attrs afterwards).
        assert not hasattr(out, "_hap_ctx")

