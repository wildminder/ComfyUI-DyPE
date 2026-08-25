"""Regression test for the FLUX name-binding trap (the "node makes no effect" bug).

Real ComfyUI FLUX binds ``optimized_attention`` at import time::

    # comfy/ldm/flux/math.py
    from comfy.ldm.modules.attention import optimized_attention

so ``comfy.ldm.modules.attention.optimized_attention`` (the module *attribute*) and
``comfy.ldm.flux.math.optimized_attention`` (FLUX's *bound name*) are DIFFERENT objects
after import.  Patching only the module attribute is therefore INVISIBLE to FLUX — it
keeps calling its own bound name, and SPA silently becomes a no-op (identical to the
base pipeline).  The 521-passing suite never caught this because its mocks call
``mock_attn.optimized_attention`` (the attribute) directly, which is exactly the path
that does NOT exist in real FLUX.

These tests build a mock ``comfy.ldm.flux.math`` module that captures the symbol as a
bound name (as real FLUX does) and prove the hook now patches that bound name and runs
the averaged attention.

Markers: @pytest.mark.unit
"""
import sys
import types

import pytest
import torch

from src.spa import _spa_install_hook
from src.spa_attn import apply_rope_matrix
from src.spa_context import SPAContext, set_spa_context

try:
    from tests._spa_math_helpers import angles_to_blocks
except ImportError:  # namespace-package import fallback
    from _spa_math_helpers import angles_to_blocks


@pytest.fixture
def flux_math_binding():
    """Simulate FLUX's ``from ... import optimized_attention`` name binding.

    Resolves ``comfy.ldm.flux.math`` (real if ComfyUI is importable, else a mock) and
    captures the current module-global as the value FLUX would have bound at import
    time.  Crucially this is a *separate name* from the module attribute, so patching
    the module attribute alone does not affect it.  The FLUX-bound name is restored on
    teardown so the session stays clean.
    """
    import comfy.ldm.modules.attention as attn_mod
    import torch.nn.functional as F

    def _sdpa(q, k, v, heads, skip_reshape=False, mask=None,
              transformer_options=None, **kw):
        return F.scaled_dot_product_attention(q, k, v, scale=1.0,
                                              dropout_p=0.0, is_causal=False)

    # The value FLUX captures at import time is whatever the module-global points at.
    _sdpa = attn_mod.optimized_attention
    attn_mod.optimized_attention = _sdpa  # make it explicit / stable for this test

    # Resolve (or simulate) the FLUX-bound module.
    created = False
    try:
        import comfy.ldm.flux.math as math_mod
    except Exception:
        flux_mod = types.ModuleType("comfy.ldm.flux")
        math_mod = types.ModuleType("comfy.ldm.flux.math")
        sys.modules["comfy.ldm.flux"] = flux_mod
        sys.modules["comfy.ldm.flux.math"] = math_mod
        created = True

    # Simulate the import-time binding: bound name = current global.
    flux_bound_orig = getattr(math_mod, "optimized_attention", None)
    math_mod.optimized_attention = _sdpa

    yield math_mod, _sdpa

    # Teardown: restore the FLUX-bound name so later tests see the original.
    if created:
        sys.modules.pop("comfy.ldm.flux.math", None)
        sys.modules.pop("comfy.ldm.flux", None)
    else:
        try:
            math_mod.optimized_attention = flux_bound_orig
        except Exception:
            pass


class _MockModel:
    def __init__(self):
        self._unet_wrapper = None
        self._spa_installed = None
        self._spa_orig_optimized_attention = None

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def _active_flux_ctx(L, H, D, N):
    """An active SPA context (pre_roped=True, fmt='flux') with N variant RoPEs."""
    P = D // 2
    g = torch.Generator().manual_seed(0)
    base = torch.randn(L, P, generator=g) * 0.3
    variants = [torch.randn(L, P, generator=g) * 0.3 for _ in range(N)]
    base_R = angles_to_blocks(base)[None, None]
    variant_Rs = [angles_to_blocks(a)[None, None] for a in variants]
    ctx = SPAContext(active=True, bundle_size=N, base_pe=base_R,
                     variant_pes=variant_Rs, pre_roped=True, fmt="flux", model_key=0)
    return ctx, base_R


@pytest.mark.unit
def test_flux_bound_name_is_patched(flux_math_binding):
    """The fix MUST patch ``comfy.ldm.flux.math.optimized_attention`` (the bound name).

    Under the old (broken) code only ``comfy.ldm.modules.attention.optimized_attention``
    was patched, so FLUX's bound name stayed the original -> SPA was a silent no-op.
    """
    math_mod, _sdpa = flux_math_binding
    m = _MockModel()
    _spa_install_hook(m, "flux")
    assert math_mod.optimized_attention is not _sdpa, (
        "FLUX's bound optimized_attention was NOT patched -> SPA is a silent no-op "
        "(the 'node makes no effect' regression)."
    )


@pytest.mark.unit
def test_flux_bound_call_runs_averaging(flux_math_binding):
    """Calling FLUX's bound ``optimized_attention`` with an active SPA context must
    produce a *different* result from the base attention (proves averaging fired).
    """
    math_mod, _sdpa = flux_math_binding
    L, H, D, N = 256, 4, 64, 3
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)

    ctx, base_R = _active_flux_ctx(L, H, D, N)
    set_spa_context(ctx)

    m = _MockModel()
    _spa_install_hook(m, "flux")

    q_base = apply_rope_matrix(q, base_R, "flux")
    k_base = apply_rope_matrix(k, base_R, "flux")

    base_out = _sdpa(q_base, k_base, v, H)
    spa_out = math_mod.optimized_attention(q_base, k_base, v, H)

    assert not torch.allclose(spa_out, base_out, atol=1e-4), (
        "SPA produced identical output to base attention through FLUX's bound call "
        "-> the averaged-attention hook did not actually run."
    )
    # sanity: same shape, finite
    assert spa_out.shape == (1, H, L, D)
    assert torch.isfinite(spa_out).all()


@pytest.mark.unit
def test_flux_inactive_delegates_through_bound_name(flux_math_binding):
    """With an inactive context, the patched FLUX-bound name is a transparent
    passthrough (delegates to the original SDPA, no RoPE/averaging)."""
    math_mod, _sdpa = flux_math_binding
    set_spa_context(None)
    m = _MockModel()
    _spa_install_hook(m, "flux")

    q = torch.randn(1, 4, 64, 64)
    k = torch.randn(1, 4, 64, 64)
    v = torch.randn(1, 4, 64, 64)
    out = math_mod.optimized_attention(q, k, v, 4)
    ref = _sdpa(q, k, v, 4)
    assert torch.allclose(out, ref, atol=1e-6)


@pytest.mark.unit
def test_flux_restore_reverts_bound_name(flux_math_binding):
    """restore_spa_attention_hook must revert the FLUX-bound name to the original."""
    math_mod, _sdpa = flux_math_binding
    from src.spa import restore_spa_attention_hook

    m = _MockModel()
    _spa_install_hook(m, "flux")
    assert math_mod.optimized_attention is not _sdpa
    restore_spa_attention_hook(m)
    assert math_mod.optimized_attention is _sdpa
    assert m._unet_wrapper is None
