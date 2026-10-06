"""Qwen-Image-2.1 ``causal_prefix`` joint mode: the attention-hook behaviour.

2.1 breaks every assumption the shared wrapper is built on, and each break is
covered here:

  * its model file binds the UNMASKED ``optimized_attention`` (the 1.0 target
    patches ``optimized_attention_masked`` — a symbol 2.1 never calls, i.e. a
    silent no-op);
  * ``block_causal_attention`` issues ONE call per SEGMENT per block, not one per
    block, so the sacred per-forward layer counter would burn one slot per text
    chunk and desync ``spa_layer_filter`` / the HAP plan ordinal from the block
    order;
  * the image target segment is NON-SQUARE by construction (``q`` is the
    ``[start, end)`` rows of a ``[0, end)`` sequence) — exactly the shape the
    generic non-square guard declines, i.e. the silent no-op this mode exists to
    prevent;
  * ``q``/``k`` arrive head-FLATTENED, so the rotation must be sliced per segment
    and applied per head.

The prefill-cache path (``prefix_cached_attention``) is deliberately NOT special
cased: it emits a single unmasked call with ``k = [cached prefix, target]``, so
``end == total_len`` and it lands on the target branch by arithmetic.  That
"it happens to be compatible" claim is pinned below rather than assumed.

Markers: @pytest.mark.unit
"""

import logging
import sys
import types

import pytest
import torch

from src.spa import (
    _make_hrdit_wrapper,
    _spa_install_hook,
    _spa_patch_targets,
    _spa_resolve_type,
)
from src.spa_attn import _rotate_flat, compose_rope, inv_rope
from src.spa_context import (
    SPAContext,
    get_hrdit_layer_idx,
    get_spa_joint_mode,
    set_hrdit_layer_idx,
    set_spa_context,
    set_spa_joint_mode,
)
from src.models.spa_flux import PosEmbedSPAFlux
from src.spa import _spa_restore_installed

try:
    from tests._hrdit_fixtures import assert_real_signature, make_recording_orig
    from tests._spa_math_helpers import angles_to_blocks
except ImportError:  # namespace-package import fallback
    from _hrdit_fixtures import assert_real_signature, make_recording_orig
    from _spa_math_helpers import angles_to_blocks

# Small-but-real 2.1 geometry: 2 text tokens + a 2x2 image grid = 6 sequence rows,
# 4 heads of dim 8 (head-flattened width 32), rotations over P = D // 2 = 4.
TEXT_LEN = 2
IMG_H = IMG_W = 2
TOTAL_LEN = TEXT_LEN + IMG_H * IMG_W  # 6
HEADS = 4
HEAD_DIM = 8
ROT_P = HEAD_DIM // 2


def _make_orig(record):
    """A 2.1-shaped ``orig``: head-flattened in, ComfyUI's head split inside.

    Built ON TOP of :func:`make_recording_orig` (the mandated factory) so the
    real ComfyUI signature is still asserted at construction time; the shim adds
    only the two conventions 2.1 relies on and the factory does not model — the
    ``not skip_reshape`` head split that ``attention_pytorch`` performs, and the
    flatten-back that matches ``skip_output_reshape=False``.  Incoming q/k are
    recorded BEFORE the split, which is what the rotation assertions inspect.
    """
    base = make_recording_orig()

    def orig(q, k, v, heads, mask=None, attn_precision=None,
             skip_reshape=False, skip_output_reshape=False, **kwargs):
        record.append((q, k, v, heads, mask))
        if not skip_reshape:
            d = q.shape[-1] // heads
            q = q.reshape(q.shape[0], q.shape[1], heads, d).transpose(1, 2)
            k = k.reshape(k.shape[0], k.shape[1], heads, d).transpose(1, 2)
            v = v.reshape(v.shape[0], v.shape[1], heads, d).transpose(1, 2)
        out = base(q, k, v, heads, mask, attn_precision,
                   skip_reshape, skip_output_reshape, **kwargs)
        if not skip_output_reshape:
            b, h, n, d = out.shape
            out = out.permute(0, 2, 1, 3).reshape(b, n, h * d)
        return out

    assert_real_signature(orig)
    return orig


def _rotations(seed=0, n_variants=3, length=TOTAL_LEN):
    """``(base_pe, variant_pes)`` in 2.1's flux layout ``(B, 1, L, P, 2, 2)``."""
    g = torch.Generator().manual_seed(seed)
    base = angles_to_blocks(torch.randn(length, ROT_P, generator=g) * 0.3)[None, None]
    variants = [
        angles_to_blocks(torch.randn(length, ROT_P, generator=g) * 0.3)[None, None]
        for _ in range(n_variants)
    ]
    return base, variants


def _ctx(seed=0, total_len=TOTAL_LEN, n_variants=3):
    base, variants = _rotations(seed=seed, n_variants=n_variants, length=total_len)
    return SPAContext(
        active=True,
        bundle_size=n_variants,
        base_pe=base,
        variant_pes=variants,
        pre_roped=True,
        fmt="flux",
        model_key=0,
        total_len=total_len,
    )


def _deltas(ctx):
    """The composed ``inv(base) @ variant`` rotations the hook actually applies."""
    inv = inv_rope(ctx.base_pe, ctx.fmt)
    return [compose_rope(inv, vp, ctx.fmt) for vp in ctx.variant_pes]


def _qkv(length, seed):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(1, length, HEADS * HEAD_DIM, generator=g),
        torch.randn(1, length, HEADS * HEAD_DIM, generator=g),
        torch.randn(1, length, HEADS * HEAD_DIM, generator=g),
    )


def _causal_mask(n, total):
    """The block-causal boolean mask 2.1 hands its TEXT segments."""
    return torch.ones(n, total, dtype=torch.bool).tril(total - n)


def _fake_qwen21_module():
    """A stand-in for ``comfy.ldm.qwen_image21.model`` with both symbols bound."""
    mod = types.ModuleType("comfy.ldm.qwen_image21.model")

    def bound(*args, **kwargs):
        return torch.zeros(1)

    mod.optimized_attention = bound
    mod.optimized_attention_masked = bound
    mod._test_bound = bound
    return mod


def _grid_ids(H, W):
    ids = torch.zeros(1, H * W, 3)
    ids[..., 0] = torch.arange(H * W)
    ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
    ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    return ids


def _embedder(cls, **kw):
    return cls(theta=10000, axes_dim=[16, 56, 56], method="ntk", **kw)


class Bf16FreqsAdapter(PosEmbedSPAFlux):
    """An adapter that pins the PE dtype — the extension point 2.1 will use."""

    def _freqs_dtype(self, pos):
        return torch.bfloat16


@pytest.fixture(autouse=True)
def _clean_context():
    """No context / joint mode may leak between tests."""
    set_spa_context(None)
    set_spa_joint_mode(None)
    set_hrdit_layer_idx(0)
    yield
    set_spa_context(None)
    set_spa_joint_mode(None)
    set_hrdit_layer_idx(0)


def _wrapper(record):
    set_hrdit_layer_idx(0)
    set_spa_joint_mode("causal_prefix")
    return _make_hrdit_wrapper(_make_orig(record), is_masked=False)


# ---------------------------------------------------------------------------
# Patch target + detection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQwen21PatchTarget:
    def test_targets_the_unmasked_qwen_image21_symbol(self):
        """2.1's own module binds the UNMASKED symbol."""
        assert _spa_patch_targets("qwen21") == [
            ("comfy.ldm.qwen_image21.model", "optimized_attention", False)
        ]

    def test_qwen21_masked_symbol_is_not_patched(self):
        """Negative control: the 1.0 masked symbol would be a silent no-op."""
        targets = _spa_patch_targets("qwen21")
        assert all(attr != "optimized_attention_masked" for _, attr, _ in targets)
        assert all(mod != "comfy.ldm.qwen_image.model" for mod, _, _ in targets)

    def test_install_patches_only_the_bound_symbol(self, monkeypatch):
        """A fake ``comfy.ldm.qwen_image21.model`` module gets its OWN symbol."""
        mod = types.ModuleType("comfy.ldm.qwen_image21.model")
        calls = []

        def bound(*args, **kwargs):  # the module-level name 2.1 imported
            calls.append(1)
            return torch.zeros(1)

        mod.optimized_attention = bound
        mod.optimized_attention_masked = bound
        monkeypatch.setitem(sys.modules, "comfy.ldm.qwen_image21", types.ModuleType("x"))
        monkeypatch.setitem(sys.modules, "comfy.ldm.qwen_image21.model", mod)

        m = _MockModel()
        _spa_install_hook(m, "qwen21")
        try:
            assert mod.optimized_attention is not bound
            assert getattr(mod.optimized_attention, "_spa_wrapper", False)
            # The masked name was never touched.
            assert mod.optimized_attention_masked is bound
        finally:
            _uninstall(m, mod)

    def test_resolves_qwen21_from_the_model_class_name(self):
        """Detection reuses the canonical resolver (no second detector)."""

        class QwenImage21Transformer2DModel:  # noqa: N801 - mirrors ComfyUI's name
            pass

        assert _spa_resolve_type("auto", QwenImage21Transformer2DModel()) == "qwen21"

    def test_qwen_image_1_0_still_resolves_to_qwen(self):
        """Negative control: 1.0 is NOT captured by the 2.1 key."""

        class QwenImageTransformer2DModel:  # noqa: N801
            pass

        assert _spa_resolve_type("auto", QwenImageTransformer2DModel()) == "qwen"

    def test_unet_wrapper_sets_and_clears_the_joint_mode(self, monkeypatch):
        """The mode is scoped to the forward exactly like the step gate."""
        mod = _fake_qwen21_module()
        monkeypatch.setitem(sys.modules, "comfy.ldm.qwen_image21", types.ModuleType("x"))
        monkeypatch.setitem(sys.modules, "comfy.ldm.qwen_image21.model", mod)

        m = _MockModel()
        _spa_install_hook(m, "qwen21")
        m._spa_joint_mode = "causal_prefix"
        try:
            seen = {}

            def model_function(x, t, **kw):
                seen["mode"] = get_spa_joint_mode()
                return torch.zeros(1)

            m._unet_wrapper(
                model_function,
                {"input": None, "timestep": torch.tensor([1.0]), "c": {}},
            )
            assert seen["mode"] == "causal_prefix"
            assert get_spa_joint_mode() == "joint"  # cleared -> no cross-model leak
        finally:
            _uninstall(m, mod)


# ---------------------------------------------------------------------------
# The freqs-dtype hook the 2.1 adapter overrides
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFreqsDtypeHook:
    """One overridable hook instead of a duplicated dtype branch per adapter.

    2.1's call site hands ``pe`` straight to the fused RoPE kernel with no cast,
    so its adapter pins fp32; every other backend keeps the pre-existing
    bf16-on-CUDA / fp32-on-CPU rule.
    """

    def test_base_rule_is_unchanged(self):
        """Negative control: the stock adapter still builds fp32 on CPU."""
        emb = _embedder(PosEmbedSPAFlux, bundle_size=1)
        assert emb._freqs_dtype(torch.zeros(1)) == torch.float32

    @pytest.mark.parametrize("bundle_size", [1, 3])
    def test_override_reaches_both_forward_paths(self, bundle_size):
        """The hook is honoured by the identity path AND the variant path.

        ``bundle_size=1`` hits ``forward``'s early return; ``3`` on a 96x96 grid
        (max_pos 95 > the 64 trained extent) goes through ``_cached_variant_pes``
        and actually bundles.  The observed quantity is the ``freqs_dtype`` both
        sites hand ``_spa_components`` — the finished PE is up-cast by
        ``format_components`` regardless, so its dtype proves nothing.
        """
        emb = _embedder(Bf16FreqsAdapter, bundle_size=bundle_size)
        seen = []
        original = emb._spa_components

        def spy(pos, freqs_dtype):
            seen.append(freqs_dtype)
            return original(pos, freqs_dtype)

        emb._spa_components = spy
        emb(_grid_ids(96, 96))
        assert seen and all(dt == torch.bfloat16 for dt in seen)


# ---------------------------------------------------------------------------
# Segment selection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCausalPrefixSegments:
    def test_target_segment_is_nonsquare_and_runs_spa(self, caplog):
        """The image segment's non-square q/k is the call that must run SPA."""
        ctx = _ctx()
        set_spa_context(ctx)
        record = []
        wrapper = _wrapper(record)
        q, k, v = _qkv(TOTAL_LEN, seed=1)

        caplog.set_level(logging.DEBUG, logger="ComfyUI-DyPE")
        out = wrapper(q[:, TEXT_LEN:], k, v, HEADS)

        assert out.shape == (1, TOTAL_LEN - TEXT_LEN, HEADS * HEAD_DIM)
        assert len(record) == len(ctx.variant_pes)  # N passes
        assert "SPA averaged-attention ACTIVE" in caplog.text
        # Rotated: every pass handed orig a q that differs from the input.
        for rq, _, _, _, _ in record:
            assert not torch.equal(rq, q[:, TEXT_LEN:])

    def test_masked_text_segment_declines_and_does_not_advance(self):
        """A masked call runs plain attention and leaves the counter alone."""
        ctx = _ctx()
        set_spa_context(ctx)
        record = []
        wrapper = _wrapper(record)
        q, k, v = _qkv(TOTAL_LEN, seed=2)

        mask = _causal_mask(TEXT_LEN, TOTAL_LEN)
        out = wrapper(q[:, :TEXT_LEN], k[:, :TEXT_LEN], v[:, :TEXT_LEN], HEADS, mask=mask)

        assert len(record) == 1  # exactly one plain pass, no N averaged passes
        assert torch.equal(record[0][0], q[:, :TEXT_LEN])  # untouched q -> no rotation
        assert out.shape == (1, TEXT_LEN, HEADS * HEAD_DIM)
        assert get_hrdit_layer_idx() == 0  # did NOT advance

    def test_reference_segment_declines_and_does_not_advance(self):
        """A reference-image chunk (``end < total_len``) declines, counter intact."""
        ctx = _ctx()
        set_spa_context(ctx)
        record = []
        wrapper = _wrapper(record)
        q, k, v = _qkv(TOTAL_LEN, seed=3)

        # Reference segment: [0, 2) of a 6-long sequence -> k never reaches end.
        out = wrapper(q[:, :2], k[:, :2], v[:, :2], HEADS)

        assert len(record) == 1
        assert torch.equal(record[0][0], q[:, :2])
        assert out.shape == (1, 2, HEADS * HEAD_DIM)
        assert get_hrdit_layer_idx() == 0

    def test_declined_segment_logs_once_per_forward(self, caplog):
        """The decline is latched, matching the ``_spa_logged`` idiom."""
        ctx = _ctx()
        set_spa_context(ctx)
        wrapper = _wrapper([])
        q, k, v = _qkv(TOTAL_LEN, seed=4)
        mask = _causal_mask(TEXT_LEN, TOTAL_LEN)

        caplog.set_level(logging.DEBUG, logger="ComfyUI-DyPE")
        wrapper(q[:, :TEXT_LEN], k[:, :TEXT_LEN], v[:, :TEXT_LEN], HEADS, mask=mask)
        wrapper(q[:, :TEXT_LEN], k[:, :TEXT_LEN], v[:, :TEXT_LEN], HEADS, mask=mask)
        assert caplog.text.count("declining segment") == 1


# ---------------------------------------------------------------------------
# Counter discipline
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCounterDiscipline:
    def test_one_advance_per_target_segment(self):
        """3 blocks x 2 segments -> the counter ends at 3, not 6."""
        ctx = _ctx()
        set_spa_context(ctx)
        wrapper = _wrapper([])
        q, k, v = _qkv(TOTAL_LEN, seed=5)
        mask = _causal_mask(TEXT_LEN, TOTAL_LEN)

        for _ in range(3):
            wrapper(q[:, :TEXT_LEN], k[:, :TEXT_LEN], v[:, :TEXT_LEN], HEADS, mask=mask)
            wrapper(q[:, TEXT_LEN:], k, v, HEADS)

        assert get_hrdit_layer_idx() == 3

    def test_target_segment_sees_its_own_block_index(self, caplog):
        """Block *n*'s image segment logs/peeks the counter at exactly *n*."""
        ctx = _ctx()
        set_spa_context(ctx)
        wrapper = _wrapper([])
        q, k, v = _qkv(TOTAL_LEN, seed=6)
        mask = _causal_mask(TEXT_LEN, TOTAL_LEN)

        seen = []
        for block in range(4):
            wrapper(q[:, :TEXT_LEN], k[:, :TEXT_LEN], v[:, :TEXT_LEN], HEADS, mask=mask)
            seen.append(get_hrdit_layer_idx())  # peeked by the text segment
            wrapper(q[:, TEXT_LEN:], k, v, HEADS)

        assert seen == [0, 1, 2, 3]
        assert get_hrdit_layer_idx() == 4

    def test_layer_filter_indexes_blocks_not_segments(self):
        """``spa_layer_filter`` selects BLOCKS: {1} runs only block 1's image call."""
        from src.spa_context import set_spa_layer_filter

        ctx = _ctx()
        set_spa_context(ctx)
        record = []
        wrapper = _wrapper(record)
        q, k, v = _qkv(TOTAL_LEN, seed=7)
        mask = _causal_mask(TEXT_LEN, TOTAL_LEN)
        n_variants = len(ctx.variant_pes)
        set_spa_layer_filter(frozenset({1}))
        try:
            passes_per_block = []
            for _ in range(3):
                wrapper(q[:, :TEXT_LEN], k[:, :TEXT_LEN], v[:, :TEXT_LEN], HEADS, mask=mask)
                before = len(record)
                wrapper(q[:, TEXT_LEN:], k, v, HEADS)
                passes_per_block.append(len(record) - before)
        finally:
            set_spa_layer_filter(None)
        # Only block 1 is filtered in; blocks 0 and 2 run a single plain pass.
        assert passes_per_block == [1, n_variants, 1]
        assert get_hrdit_layer_idx() == 3  # the counter still ran per block

    def test_three_segment_reference_graph_advances_once(self):
        """ref + text + image: only the image segment advances the counter."""
        ctx = _ctx()
        set_spa_context(ctx)
        wrapper = _wrapper([])
        q, k, v = _qkv(TOTAL_LEN, seed=8)

        wrapper(q[:, :2], k[:, :2], v[:, :2], HEADS)  # reference
        wrapper(q[:, 2:4], k[:, :4], v[:, :4], HEADS,  # text (masked, end < total)
                mask=_causal_mask(2, 4))
        wrapper(q[:, 4:], k, v, HEADS)  # image target
        assert get_hrdit_layer_idx() == 1


# ---------------------------------------------------------------------------
# Slice correctness
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSegmentSlices:
    def test_rotation_uses_the_segment_rows_of_the_full_rotation(self):
        """q is rotated with ``rot[start:end]``, k with ``rot[:end]``."""
        ctx = _ctx()
        set_spa_context(ctx)
        record = []
        wrapper = _wrapper(record)
        q, k, v = _qkv(TOTAL_LEN, seed=9)
        start, end = TEXT_LEN, TOTAL_LEN

        wrapper(q[:, start:], k, v, HEADS)

        deltas = _deltas(ctx)
        for (rq, rk, _, _, _), rot in zip(record, deltas):
            assert torch.allclose(
                rq, _rotate_flat(q[:, start:], rot[..., start:end, :, :, :], "flux", HEADS),
                atol=1e-6,
            )
            assert torch.allclose(
                rk, _rotate_flat(k, rot[..., :end, :, :, :], "flux", HEADS),
                atol=1e-6,
            )

    def test_prefill_cache_path_lands_on_the_target_branch(self):
        """Cached step: ``k = [prefix, target]``, ``q = target`` -> start == prefix."""
        prefix_len = TEXT_LEN
        ctx = _ctx()
        set_spa_context(ctx)
        record = []
        wrapper = _wrapper(record)
        q, k, v = _qkv(TOTAL_LEN, seed=10)

        # prefix_cached_attention: target rows only, k = [cached prefix, target].
        out = wrapper(q[:, prefix_len:], k, v, HEADS)

        assert len(record) == len(ctx.variant_pes)  # ran SPA, did NOT decline
        deltas = _deltas(ctx)
        assert torch.allclose(
            record[0][0],
            _rotate_flat(q[:, prefix_len:], deltas[0][..., prefix_len:, :, :, :],
                         "flux", HEADS),
            atol=1e-6,
        )
        assert out.shape == (1, TOTAL_LEN - prefix_len, HEADS * HEAD_DIM)

    def test_qwen21_spa_active_log_present(self, caplog):
        """The canonical "SPA is NOT a silent no-op" signal must appear.

        ``src/spa.py`` logs ``SPA averaged-attention ACTIVE`` once per forward
        when the averaged passes actually run; its ABSENCE is this project's
        documented indicator that the patched symbol was never called (the
        Krea-2 failure mode, and the trap 2.1's own module/symbol split sets).
        Asserting on the log rather than on the pass count is deliberate: the
        count can be non-zero while the mode still declines the call that
        matters.
        """
        ctx = _ctx()
        set_spa_context(ctx)
        wrapper = _wrapper([])
        q, k, v = _qkv(TOTAL_LEN, seed=12)

        caplog.set_level(logging.DEBUG, logger="ComfyUI-DyPE")
        # Prefill-cache shape: one unmasked call, k = [cached prefix, target].
        wrapper(q[:, TEXT_LEN:], k, v, HEADS)

        assert "SPA averaged-attention ACTIVE" in caplog.text

    def test_non_square_target_runs_spa_and_other_nonsquare_declines(self):
        """The non-square TARGET segment runs SPA; a non-square call that does
        not reach the end of the sequence still declines.

        The target is identified by ``end == total_len``, not by squareness — a
        call whose ``k`` reaches the end IS the target by construction, so only
        the reaching definition is testable (and is what the code keys on).
        """
        ctx = _ctx()
        set_spa_context(ctx)
        record = []
        wrapper = _wrapper(record)
        q, k, v = _qkv(TOTAL_LEN, seed=11)

        # Non-square by construction: q is the [start, end) suffix of [0, end).
        wrapper(q[:, TEXT_LEN:], k, v, HEADS)
        assert len(record) == len(ctx.variant_pes)

        # Non-square AND short of the end -> a reference-style segment -> declines.
        before = len(record)
        wrapper(q[:, :3], k[:, :4], v[:, :4], HEADS)
        assert len(record) - before == 1  # single plain pass
        assert get_hrdit_layer_idx() == 1  # the target advanced, this one peeked


# ---------------------------------------------------------------------------
# Default mode must not change any other backend
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDefaultModeUnchanged:
    def test_default_mode_is_joint(self):
        """The contextvar default is ``"joint"`` and ``None`` normalises to it."""
        set_spa_joint_mode(None)
        assert get_spa_joint_mode() == "joint"

    def test_joint_mode_still_declines_non_square_and_advances_every_call(self):
        """Negative control: with the default mode the legacy rules are intact."""
        ctx = _ctx()
        set_spa_context(ctx)
        record = []
        set_hrdit_layer_idx(0)
        set_spa_joint_mode(None)
        wrapper = _make_hrdit_wrapper(_make_orig(record), is_masked=False)
        q, k, v = _qkv(TOTAL_LEN, seed=12)

        wrapper(q[:, :3], k, v, HEADS)  # non-square -> declines
        assert len(record) == 1
        assert torch.equal(record[0][0], q[:, :3])
        assert get_hrdit_layer_idx() == 1  # every call advances, as before

    def test_causal_prefix_without_a_registered_context_advances_every_call(self):
        """No ``total_len`` to key on -> the unconditional advance is kept."""
        set_spa_context(None)
        record = []
        wrapper = _wrapper(record)
        q, k, v = _qkv(TOTAL_LEN, seed=13)

        wrapper(q[:, :3], k[:, :3], v[:, :3], HEADS)
        wrapper(q[:, :3], k[:, :3], v[:, :3], HEADS)
        assert get_hrdit_layer_idx() == 2


class _MockModel:
    """Minimal ModelPatcher stand-in for ``_spa_install_hook``."""

    def __init__(self):
        self._object_patches = {}
        self._unet_wrapper = None
        self._spa_orig_optimized_attention = None

    def clone(self):
        new = _MockModel()
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        new._spa_orig_optimized_attention = self._spa_orig_optimized_attention
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def _uninstall(m, mod):
    """Restore every symbol ``_spa_install_hook`` recorded on ``m``."""
    _spa_restore_installed(m)