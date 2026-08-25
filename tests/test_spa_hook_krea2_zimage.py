"""Regression tests for two SPA backend bugs:

1. Krea-2 (K2) "node makes no effect" — Krea-2's ``SingleStreamDiT`` binds
   ``optimized_attention_masked`` into ``comfy.ldm.krea2.model`` (NOT the Qwen
   backend ``comfy.ldm.qwen_image.model``), and uses a flux-style ``pe_embedder``.
   When a user follows the README and selects ``model_type="qwen"``, the old code
   patched the qwen symbol while Krea-2 kept calling its own -> silent no-op.
   Fix: detect ``SingleStreamDiT`` by class name and patch ``comfy.ldm.krea2.model``.

2. Z-Image / Lumina ``RuntimeError`` in ``apply_rope_matrix`` — lumina calls
   ``rope_embedder`` once PER token group (caption -> siglip -> image) but runs a
   single ``JointAttention`` over the FULL concatenated sequence.  The old code stored
   only the LAST group's variant PE in the process-global ``SPAContext``, so the einsum
   saw a variant PE of length 64 against an attention ``q`` of length 24576 and raised
   ``einsum(): subscript l has size 64 ... 24576``.  Fix: accumulate per-group position
   ids and rebuild the FULL-sequence variant PE in the hook, in lumina's group-major
   order, with a length guard so no regime ever crashes.

Markers: @pytest.mark.unit
"""
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from src.spa import (
    _spa_assemble_zimage_posids,
    _spa_dispatch_attention,
    _spa_install_hook,
    _spa_patch_targets,
    _spa_resolve_type,
)
from src.spa_attn import apply_rope_matrix
from src.spa_context import SPAContext, get_spa_context, set_spa_context

# ---------------------------------------------------------------------------
# Krea-2
# ---------------------------------------------------------------------------

@pytest.fixture
def krea2_binding():
    """Simulate Krea-2's ``from ... import optimized_attention_masked`` name binding."""
    import comfy.ldm.modules.attention as attn_mod

    # Real ComfyUI aliases masked -> unmasked (comfy/ldm/modules/attention.py:883).
    # The mock env may only expose the unmasked name; mirror the alias so the Krea-2
    # binding (which imports optimized_attention_masked) is exercised.
    added_alias = False
    if not hasattr(attn_mod, "optimized_attention_masked"):
        attn_mod.optimized_attention_masked = attn_mod.optimized_attention
        added_alias = True

    orig = attn_mod.optimized_attention_masked

    krea_mod = types.ModuleType("comfy.ldm.krea2")
    krea_model_mod = types.ModuleType("comfy.ldm.krea2.model")
    sys.modules["comfy.ldm.krea2"] = krea_mod
    sys.modules["comfy.ldm.krea2.model"] = krea_model_mod
    krea_model_mod.optimized_attention_masked = orig

    yield krea_model_mod, orig

    for name in ("comfy.ldm.krea2.model", "comfy.ldm.krea2"):
        sys.modules.pop(name, None)
    if added_alias:
        delattr(attn_mod, "optimized_attention_masked")


class _MockModel:
    def __init__(self):
        self._unet_wrapper = None
        self._spa_installed = None
        self._spa_orig_optimized_attention = None

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def test_krea2_patch_target_symbol():
    """_spa_patch_targets("krea2") MUST target comfy.ldm.krea2.model (masked)."""
    targets = _spa_patch_targets("krea2")
    assert targets == [("comfy.ldm.krea2.model", "optimized_attention_masked", True)]


def test_krea2_resolve_from_class_name():
    """Auto-detect and an explicit 'qwen' string for a SingleStreamDiT both -> 'krea2'."""
    class SingleStreamDiT:
        pass

    class QwenImage:
        pass

    dm_krea = SingleStreamDiT()
    dm_qwen = QwenImage()

    assert _spa_resolve_type("auto", dm_krea) == "krea2"
    # The README tells users to pass model_type="qwen" for Krea-2 -> must still resolve
    # to krea2 (otherwise the hook patches the wrong symbol and is a no-op).
    assert _spa_resolve_type("qwen", dm_krea) == "krea2"
    # But a real Qwen model stays qwen.
    assert _spa_resolve_type("qwen", dm_qwen) == "qwen"


def test_krea2_bound_name_is_patched(krea2_binding):
    """The fix MUST patch comfy.ldm.krea2.model.optimized_attention_masked."""
    krea_model_mod, orig = krea2_binding
    m = _MockModel()
    _spa_install_hook(m, "krea2")
    assert krea_model_mod.optimized_attention_masked is not orig, (
        "Krea-2's bound optimized_attention_masked was NOT patched -> SPA is a "
        "silent no-op (the 'node makes no effect' regression)."
    )


def test_krea2_bound_call_runs_averaging(krea2_binding):
    """Calling Krea-2's bound masked attention with an active SPA context must differ
    from base attention (proves the averaged-attention hook fired)."""
    try:
        from tests._spa_math_helpers import angles_to_blocks
    except ImportError:
        from _spa_math_helpers import angles_to_blocks

    krea_model_mod, orig = krea2_binding
    L, H, D, N = 128, 4, 64, 3
    g = torch.Generator().manual_seed(1)
    q = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)

    P = D // 2
    gg = torch.Generator().manual_seed(0)
    base = torch.randn(L, P, generator=gg) * 0.3
    variants = [torch.randn(L, P, generator=gg) * 0.3 for _ in range(N)]
    base_R = angles_to_blocks(base)[None, None]
    variant_Rs = [angles_to_blocks(a)[None, None] for a in variants]
    ctx = SPAContext(active=True, bundle_size=N, base_pe=base_R,
                     variant_pes=variant_Rs, pre_roped=True, fmt="flux", model_key=0)
    set_spa_context(ctx)

    m = _MockModel()
    _spa_install_hook(m, "krea2")

    q_base = apply_rope_matrix(q, base_R, "flux")
    k_base = apply_rope_matrix(k, base_R, "flux")
    base_out = orig(q_base, k_base, v, H, mask=None, skip_reshape=True,
                    transformer_options={})
    spa_out = krea_model_mod.optimized_attention_masked(
        q_base, k_base, v, H, mask=None, skip_reshape=True, transformer_options={})

    assert not torch.allclose(spa_out, base_out, atol=1e-4), (
        "SPA produced identical output to base through Krea-2's bound masked call "
        "-> the averaged-attention hook did not run."
    )
    assert spa_out.shape == (1, H, L, D)
    assert torch.isfinite(spa_out).all()


# ---------------------------------------------------------------------------
# Z-Image / Lumina multi-group
# ---------------------------------------------------------------------------

def _make_zimage_embedder(bundle_size=3):
    from src.models.spa_zimage import PosEmbedSPAZImage

    emb = PosEmbedSPAZImage(
        10000.0, [64, 64, 64], "ntk", False, True, 2.0, 2.0, 1024, 1.0, None,
        enable_spa=True, bundle_size=bundle_size,
    )
    emb._spa_is_zimage = True
    return emb


def _sdpa(q, k, v, heads, skip_reshape=False, mask=None, transformer_options=None, **kw):
    return F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=0.0, is_causal=False)


def _attn_fn(q, k, v, heads):
    """Wrap _sdpa the way the real attention hook binds ``heads`` (so ``_attn(q,k,v)`` works)."""
    return _sdpa(q, k, v, heads)


def test_zimage_assemble_order_group_major():
    """caption / siglip / image position ids reassemble group-major (caps, siglips, imgs)."""
    set_spa_context(None)
    emb = _make_zimage_embedder()
    cap_ids = torch.zeros(1, 5, 3)                     # text: h==w==0
    sig_ids = torch.tensor([[[0, 1.0, 2.0], [0, 3.0, 4.0], [0, 5.0, 6.0]]])  # siglip grid
    img_ids = torch.tensor([[[0, 7.0, 8.0], [0, 9.0, 10.0]]])                 # image grid

    emb.forward(cap_ids)
    emb.forward(sig_ids)
    emb.forward(img_ids)
    ctx = get_spa_context()

    full = _spa_assemble_zimage_posids(ctx.pending)
    assert full is not None
    # 5 + 3 + 2 == 10, in group-major order (cap first, then siglip, then image).
    assert full.shape == (1, 10, 3)
    # First 5 rows are the caption ids (all zeros in h/w).
    assert (full[0, :5, 1:] == 0).all()
    # Rows 5..7 are siglip (h in {1,3,5}), rows 8..9 are image (h in {7,9}).
    assert full[0, 5, 1].item() == 1.0
    assert full[0, 8, 1].item() == 7.0


def test_zimage_multigroup_no_einsum_crash():
    """Full-sequence attention over caption+siglip+image must NOT raise the einsum
    RuntimeError (the original bug).  Lengths must match and output be well-formed."""
    emb = _make_zimage_embedder(bundle_size=3)
    set_spa_context(None)
    Lc, Ls, Li = 7, 9, 100
    cap_ids = torch.zeros(1, Lc, 3)
    sig_ids = torch.stack([
        torch.cat([torch.full((Ls, 1), 0.0),
                   torch.arange(Ls, dtype=torch.float32).reshape(Ls, 1),
                   torch.arange(Ls, dtype=torch.float32).reshape(Ls, 1)], dim=1)
    ], dim=0)
    h = w = int(Li ** 0.5)
    img_ids = torch.stack([
        torch.cat([torch.full((h * w, 1), 0.0),
                   torch.arange(h * w, dtype=torch.float32).reshape(h * w, 1),
                   torch.arange(h * w, dtype=torch.float32).reshape(h * w, 1)], dim=1)
    ], dim=0)  # Li == h*w

    emb.forward(cap_ids)
    emb.forward(sig_ids)
    emb.forward(img_ids)
    ctx = get_spa_context()
    assert ctx.uses_pending and len(ctx.pending) == 3

    L = Lc + Ls + Li
    H, D = 4, 192  # sum(axes_dim) == 192 -> P == 96
    g = torch.Generator().manual_seed(3)
    q = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)

    out = _spa_dispatch_attention(q, k, v, ctx, lambda a, b, c: _attn_fn(a, b, c, H), "flux")
    assert out.shape == (1, H, L, D)
    assert torch.isfinite(out).all()
    # The averaged-attention hook must actually run (variant RoPE != base), so the
    # SPA output differs from a plain single-pass attention over the same q/k/v.
    plain = _attn_fn(q, k, v, H)
    assert not torch.allclose(out, plain, atol=1e-4), (
        "Z-Image SPA produced identical output to plain attention -> the averaged "
        "hook did not fire (would be a silent no-op)."
    )


def test_zimage_caption_only_regime_falls_back():
    """A caption-only refinement regime has q shorter than the full sequence; SPA must
    fall back to plain attention (no crash) rather than mismatching lengths."""
    emb = _make_zimage_embedder(bundle_size=3)
    set_spa_context(None)
    Lc, Li = 6, 50
    cap_ids = torch.zeros(1, Lc, 3)
    img_ids = torch.stack([
        torch.cat([torch.full((Li, 1), 0.0),
                   torch.arange(Li, dtype=torch.float32).reshape(Li, 1),
                   torch.arange(Li, dtype=torch.float32).reshape(Li, 1)], dim=1)
    ], dim=0)
    emb.forward(cap_ids)
    emb.forward(img_ids)
    ctx = get_spa_context()

    # Caption-only attention: q length == Lc, but pending holds the full sequence.
    H, D = 4, 192
    g = torch.Generator().manual_seed(4)
    q = torch.randn(1, H, Lc, D, generator=g)
    k = torch.randn(1, H, Lc, D, generator=g)
    v = torch.randn(1, H, Lc, D, generator=g)

    out = _spa_dispatch_attention(q, k, v, ctx, lambda a, b, c: _attn_fn(a, b, c, H), "flux")
    # Fallback = plain attention, identical to the SDPA reference.
    ref = _sdpa(q, k, v, H)
    assert torch.allclose(out, ref, atol=1e-5)
    assert out.shape == (1, H, Lc, D)


def test_zimage_old_single_group_path_crashes_documentation():
    """Documents the root cause: the OLD single-group path (variant PE from only the last
    group) mismatches the full-sequence q and raises the einsum RuntimeError.  This must
    keep failing so we notice if the multi-group fix regresses."""
    emb = _make_zimage_embedder(bundle_size=3)
    set_spa_context(None)
    Lc, Li = 7, 100
    cap_ids = torch.zeros(1, Lc, 3)
    img_ids = torch.stack([
        torch.cat([torch.full((Li, 1), 0.0),
                   torch.arange(Li, dtype=torch.float32).reshape(Li, 1),
                   torch.arange(Li, dtype=torch.float32).reshape(Li, 1)], dim=1)
    ], dim=0)
    emb.forward(cap_ids)
    emb.forward(img_ids)

    # Reconstruct the OLD broken state: only the image group's variant PE, length Li.
    try:
        from tests._spa_math_helpers import angles_to_blocks
    except ImportError:
        from _spa_math_helpers import angles_to_blocks
    gg = torch.Generator().manual_seed(0)
    P = 96
    base = torch.randn(Li, P, generator=gg) * 0.3
    variants = [torch.randn(Li, P, generator=gg) * 0.3 for _ in range(3)]
    ctx_old = SPAContext(active=True, bundle_size=3,
                         base_pe=angles_to_blocks(base)[None, None],
                         variant_pes=[angles_to_blocks(a)[None, None] for a in variants],
                         pre_roped=True, fmt="flux", model_key=0)

    H, D = 4, 192
    L = Lc + Li
    # Only `q` is needed: apply_rope_matrix must reject the mismatched PE.
    q = torch.randn(1, H, L, D)

    with pytest.raises(RuntimeError):
        apply_rope_matrix(q, ctx_old.variant_pes[0], "flux")


def test_t0_3_zimage_dispatch_uses_pending_path():
    """T0.3 (P0 diagnostic): verify that a Z-Image embedder with multi-group
    registration routes ``_spa_dispatch_attention`` through the PENDING path
    (full-sequence PE assembly), NOT the single-group ``_spa_run_averaged`` path.

    The user's real-run log showed ``SPA averaged-attention ACTIVE`` which is
    emitted ONLY by ``_spa_run_averaged`` (the non-pending path).  For Z-Image
    the dispatch should take the pending branch.  This test confirms whether the
    pending branch engages when ``uses_pending`` is set and ``pending`` is
    populated with a full sequence matching ``q``.
    """
    import src.spa as spa_mod

    emb = _make_zimage_embedder(bundle_size=3)
    set_spa_context(None)
    Lc, Ls, Li = 7, 9, 100
    cap_ids = torch.zeros(1, Lc, 3)
    sig_ids = torch.stack([
        torch.cat([torch.full((Ls, 1), 0.0),
                   torch.arange(Ls, dtype=torch.float32).reshape(Ls, 1),
                   torch.arange(Ls, dtype=torch.float32).reshape(Ls, 1)], dim=1)
    ], dim=0)
    h = w = int(Li ** 0.5)
    img_ids = torch.stack([
        torch.cat([torch.full((h * w, 1), 0.0),
                   torch.arange(h * w, dtype=torch.float32).reshape(h * w, 1),
                   torch.arange(h * w, dtype=torch.float32).reshape(h * w, 1)], dim=1)
    ], dim=0)

    emb.forward(cap_ids)
    emb.forward(sig_ids)
    emb.forward(img_ids)
    ctx = get_spa_context()
    assert ctx is not None and ctx.active
    assert ctx.uses_pending, "Z-Image embedder must set uses_pending=True"
    assert len(ctx.pending) == 3

    L = Lc + Ls + Li
    H, D = 4, 192
    g = torch.Generator().manual_seed(7)
    q = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)

    # Instrument _spa_run_averaged to detect if the single-group path is taken.
    calls = {"run_averaged": 0}
    orig_run_averaged = spa_mod._spa_run_averaged

    def _spy_run_averaged(q_, k_, v_, ctx_, attn_fn):
        calls["run_averaged"] += 1
        return orig_run_averaged(q_, k_, v_, ctx_, attn_fn)

    spa_mod._spa_run_averaged = _spy_run_averaged
    try:
        out = _spa_dispatch_attention(q, k, v, ctx, lambda a, b, c: _attn_fn(a, b, c, H), "flux")
    finally:
        spa_mod._spa_run_averaged = orig_run_averaged

    assert out.shape == (1, H, L, D)
    assert torch.isfinite(out).all()
    assert calls["run_averaged"] == 0, (
        "DIAGNOSTIC: Z-Image dispatch fell through to _spa_run_averaged "
        "(single-group path) instead of the pending path.  This explains the "
        "'SPA averaged-attention ACTIVE' log seen in the user's real run."
    )


def test_t4_1_dispatch_logs_pending_path(caplog):
    """T4.1 (D3 diagnostic): the PENDING path emits a one-time INFO log so a real
    Z-Image run can confirm the full-sequence path fired (it previously emitted
    NOTHING, making the 'SPA averaged-attention ACTIVE' line ambiguous)."""
    import logging

    emb = _make_zimage_embedder(bundle_size=3)
    set_spa_context(None)
    Lc, Ls, Li = 7, 9, 100
    cap_ids = torch.zeros(1, Lc, 3)
    sig_ids = torch.stack([
        torch.cat([torch.full((Ls, 1), 0.0),
                   torch.arange(Ls, dtype=torch.float32).reshape(Ls, 1),
                   torch.arange(Ls, dtype=torch.float32).reshape(Ls, 1)], dim=1)
    ], dim=0)
    h = w = int(Li ** 0.5)
    img_ids = torch.stack([
        torch.cat([torch.full((h * w, 1), 0.0),
                   torch.arange(h * w, dtype=torch.float32).reshape(h * w, 1),
                   torch.arange(h * w, dtype=torch.float32).reshape(h * w, 1)], dim=1)
    ], dim=0)

    emb.forward(cap_ids)
    emb.forward(sig_ids)
    emb.forward(img_ids)
    ctx = get_spa_context()
    assert ctx is not None and ctx.uses_pending

    L = Lc + Ls + Li
    H, D = 4, 192
    g = torch.Generator().manual_seed(7)
    q = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)

    with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
        _spa_dispatch_attention(q, k, v, ctx, lambda a, b, c: _attn_fn(a, b, c, H), "flux")

    pending_logs = [r for r in caplog.records if "PENDING path" in r.getMessage()]
    assert pending_logs, (
        "DIAGNOSTIC: the pending (Z-Image) path emitted no log; a real run cannot "
        "confirm full-sequence SPA engagement.")
    # One-time-per-forward: a second dispatch must NOT re-log.
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
        _spa_dispatch_attention(q, k, v, ctx, lambda a, b, c: _attn_fn(a, b, c, H), "flux")
    assert not [r for r in caplog.records if "PENDING path" in r.getMessage()], (
        "pending-path log must fire once per forward, not per attention call")


def test_apply_rope_matrix_casts_rotation_to_activation_dtype():
    """Regression for the runtime ``RuntimeError: expected scalar type Half but found
    Float``.  SPA builds the rotation blocks ``R`` in fp32/bf16, but Krea-2 / Z-Image
    run attention in fp16 (Half).  ``apply_rope_matrix`` MUST cast ``R`` to the
    activations' dtype before the einsum so the contraction stays in the model's
    native compute precision (and never mismatches dtypes)."""
    try:
        from tests._spa_math_helpers import angles_to_blocks
    except ImportError:
        from _spa_math_helpers import angles_to_blocks

    L, H, D, P = 64, 4, 128, 64
    gg = torch.Generator().manual_seed(0)
    base = torch.randn(L, P, generator=gg) * 0.3
    R_fp32 = angles_to_blocks(base)[None, None]  # float32 rotation blocks (the bug's dtype)

    # --- bfloat16 activations (CPU-safe, exercises the exact cast branch) ---
    gx = torch.Generator().manual_seed(1)
    x_bf16 = torch.randn(1, H, L, D, generator=gx, dtype=torch.bfloat16)
    out_bf16 = apply_rope_matrix(x_bf16, R_fp32, "flux")
    assert out_bf16.dtype == torch.bfloat16, (
        f"apply_rope_matrix must preserve the activation dtype, got {out_bf16.dtype}"
    )
    assert out_bf16.shape == (1, H, L, D)
    assert torch.isfinite(out_bf16.float()).all()

    # --- float16 activations (the real Krea-2/Z-Image dtype) ---
    # Some CPU builds lack an fp16 einsum kernel; skip gracefully rather than red-fail.
    try:
        x_fp16 = torch.randn(1, H, L, D, generator=gx, dtype=torch.float16)
        out_fp16 = apply_rope_matrix(x_fp16, R_fp32, "flux")
        assert out_fp16.dtype == torch.float16
        assert out_fp16.shape == (1, H, L, D)
        assert torch.isfinite(out_fp16.float()).all()
    except RuntimeError as e:
        if "not implemented" in str(e) or "Backend" in str(e):
            pytest.skip(f"CPU fp16 einsum unsupported in this runtime: {e}")
        raise


def test_zimage_delta_cache_separate():
    """T3.2 (P3 cache hygiene): the Z-Image FULL-sequence deltas live in their OWN
    cache entry, distinct from the per-group entries, and are composed once.

    The pending path reassembles the full concatenated sequence and calls
    ``_compute_fullseq_deltas(full)``.  Because the full-sequence ``posids`` have a
    different shape than any single per-group call, ``_pe_cache_key`` yields a
    distinct key, so the full-sequence deltas never collide with (or are invalidated
    by) the per-group entries.  Composing twice returns the SAME cached tensors.
    """
    emb = _make_zimage_embedder(bundle_size=3)
    set_spa_context(None)
    Lc, Ls, Li = 7, 9, 100
    cap_ids = torch.zeros(1, Lc, 3)
    sig_ids = torch.stack([
        torch.cat([torch.full((Ls, 1), 0.0),
                   torch.arange(Ls, dtype=torch.float32).reshape(Ls, 1),
                   torch.arange(Ls, dtype=torch.float32).reshape(Ls, 1)], dim=1)
    ], dim=0)
    h = w = int(Li ** 0.5)
    img_ids = torch.stack([
        torch.cat([torch.full((h * w, 1), 0.0),
                   torch.arange(h * w, dtype=torch.float32).reshape(h * w, 1),
                   torch.arange(h * w, dtype=torch.float32).reshape(h * w, 1)], dim=1)
    ], dim=0)

    emb.forward(cap_ids)
    emb.forward(sig_ids)
    emb.forward(img_ids)
    ctx = get_spa_context()
    assert ctx is not None and ctx.uses_pending

    full = _spa_assemble_zimage_posids(ctx.pending)
    assert full is not None and full.shape[1] == Lc + Ls + Li

    # Full-sequence deltas: composed once, cached, and returned identically on reuse.
    d1 = emb._compute_fullseq_deltas(full)
    d2 = emb._compute_fullseq_deltas(full)
    assert d1[0] is d2[0], "full-sequence deltas must be cached, not recomposed"

    # The full-sequence entry is DISTINCT from every per-group entry (different seq
    # length -> different cache key -> no collision / staleness).
    per_group = emb._cached_variant_deltas(img_ids)
    assert d1[0].shape != per_group[0].shape, (
        "full-sequence deltas must live in a separate cache entry from per-group")
