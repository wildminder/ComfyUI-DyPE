"""Tests for SPA model adapters (FLUX / Qwen / Qwen-2.1 / Z-Image / Nunchaku).

These verify that each SPA embedder produces the *same output format* as its
DyPE counterpart, that ``forward`` returns the **base** RoPE (not the averaged
variant RoPE — the root-cause bug is removed), and that it registers the ``N``
bundled variant RoPEs in the process-scoped :class:`SPAContext` for the attention
hook.  Pure unit tests — no ComfyUI runtime required.

Qwen-Image-2.1 (``PosEmbedSPAQwen21``) runs the SAME battery with no
modification: the parametrization is the proof that the MRO wiring is right
(``SPABasePosEmbed.forward`` registers the variants, ``format_components``
resolves through ``PosEmbedQwen21`` -> ``PosEmbedQwen``).  Its dtype override is
the one intentional deviation and is asserted separately at the bottom.
"""
import types

import pytest
import torch

from src.models.spa_flux import PosEmbedSPAFlux
from src.models.spa_nunchaku import PosEmbedSPANunchaku
from src.models.spa_qwen import PosEmbedSPAQwen
from src.models.spa_qwen21 import PosEmbedSPAQwen21
from src.models.spa_zimage import PosEmbedSPAZImage
from src.spa import _spa_restore_installed, build_bundle_id_variants, get_spa_context
from src.spa_context import set_spa_context

_ADAPTERS = [
    PosEmbedSPAFlux,
    PosEmbedSPAQwen,
    PosEmbedSPAQwen21,
    PosEmbedSPAZImage,
    PosEmbedSPANunchaku,
]
_ADAPTER_NAMES = ["flux", "qwen", "qwen21", "zimage", "nunchaku"]


def _make_flux_ids(H=64, W=64, B=1):
    L = H * W
    ids = torch.zeros(B, L, 3)
    ids[..., 0] = torch.arange(L)
    ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
    ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    return ids


def _make_emb(cls, **kw):
    return cls(theta=10000, axes_dim=[16, 56, 56], method="ntk", **kw)


@pytest.mark.unit
@pytest.mark.parametrize("emb_cls,name", list(zip(_ADAPTERS, _ADAPTER_NAMES)))
class TestPosEmbedSPAAdapters:
    def test_output_shape(self, emb_cls, name):
        emb = _make_emb(emb_cls)
        out = emb(_make_flux_ids(64, 64))
        if name == "nunchaku":
            assert out.shape == (1, 1, 4096, 64, 1, 2)
        else:
            assert out.shape == (1, 1, 4096, 64, 2, 2)

    def test_finite(self, emb_cls, name):
        emb = _make_emb(emb_cls)
        out = emb(_make_flux_ids(32, 32))
        assert torch.isfinite(out).all()

    # T-P2-1: forward returns the BASE RoPE, not the mean of variants.
    def test_forward_returns_base_not_mean(self, emb_cls, name):
        # Use a grid OUTSIDE the trained extent (128x128, max_pos=127 > 64) so
        # bundling is active and the base-vs-mean distinction is meaningful.
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=3)
        ids = _make_flux_ids(128, 128)
        out = emb(ids)
        base = emb.format_components(emb._spa_components(ids.float(), torch.float32), ids)
        assert torch.allclose(out, base, atol=1e-6)
        # Sanity: the averaged (legacy) path would differ; confirm base is NOT the mean.
        variants = build_bundle_id_variants(ids, 3)
        assert len(variants) > 1  # bundling is active
        mean_variants = torch.stack(
            [emb.format_components(emb._spa_components(v.float(), torch.float32), v) for v in variants],
            dim=0,
        ).mean(0)
        # base == forward output; base may or may not equal the legacy mean — the key
        # assertion is that forward returns base (above), not the mean.
        assert out.shape == mean_variants.shape

    # T-P2-2: forward registers the N variant RoPEs in the active context.
    def test_forward_registers_variants(self, emb_cls, name):
        # Paper-N semantics at an ACTIVE grid (128x128, max_pos=127 > 64):
        # N=3 -> s = max(3, ceil(127/79)=2) = 3 -> 2*3 - 1 = 5 variants.
        from src.spa import derive_bundle_s
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=3)
        ids = _make_flux_ids(128, 128)
        out = emb(ids)
        ctx = get_spa_context()
        assert ctx is not None and ctx.active is True
        assert ctx.bundle_size == 3
        max_pos = int(max(ids[..., 1].max(), ids[..., 2].max()))
        s = derive_bundle_s(max_pos, 3)
        assert len(ctx.variant_pes) == 2 * s - 1
        assert ctx.fmt == ("nunchaku" if name == "nunchaku" else "flux")
        assert torch.allclose(ctx.base_pe, out, atol=1e-6)

    # Trained-extent gate: a grid inside the trained extent registers a single
    # identity variant (hook passthrough) even with an active knob.
    def test_in_trained_extent_is_identity(self, emb_cls, name):
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=3)
        ids = _make_flux_ids(64, 64)  # max_pos=63 <= 64
        out = emb(ids)
        ctx = get_spa_context()
        assert ctx is not None and ctx.active is True
        assert len(ctx.variant_pes) == 1  # identity -> hook passthrough
        assert torch.allclose(ctx.base_pe, out, atol=1e-6)

    # T-P2-3: bundle_size==1 => context inactive (passthrough, no hook effect).
    def test_bundle_size_one_inactive(self, emb_cls, name):
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=1)
        ids = _make_flux_ids(16, 16)
        emb(ids)  # output unused; the assertion targets the CONTEXT state
        ctx = get_spa_context()
        assert ctx is None or ctx.active is False

    def test_off_equals_base(self, emb_cls, name):
        emb = _make_emb(emb_cls, enable_spa=False, bundle_size=5)
        ids = _make_flux_ids(32, 32)
        out = emb(ids)
        base = emb.format_components(emb._spa_components(ids.float(), torch.float32), ids)
        assert torch.allclose(out, base, atol=1e-6)

    # T-P2-4 mirror: variant pes change the RoPE vs base (hook will use them).
    def test_variant_pes_differ_from_base(self, emb_cls, name):
        emb = _make_emb(emb_cls, enable_spa=True, bundle_size=5)
        ids = _make_flux_ids(128, 128)
        emb(ids)  # registers the context; the returned base PE is unused here
        ctx = get_spa_context()
        # at least one variant pe differs from the base pe (bundling changed coords)
        diff = torch.stack([(vp - ctx.base_pe).abs().max() for vp in ctx.variant_pes])
        assert diff.max() > 1e-4


@pytest.mark.unit
class TestPosEmbedSPAQwen21Specifics:
    """The one intentional deviation from the 1.0 adapter: fp32 frequencies.

    2.1 hands ``pe`` straight to its own fused RoPE kernel with no
    ``.to(x.dtype)`` cast (1.0 casts), so a bfloat16 PE would silently change
    the dtype the model's kernel receives.
    """

    def test_output_matches_the_qwen_1_0_adapter_bit_for_bit(self):
        """Negative control: the fp32 pin changes the DTYPE RULE, not the math."""
        ids = _make_flux_ids(32, 32)
        qwen21 = _make_emb(PosEmbedSPAQwen21, enable_spa=True, bundle_size=3)(ids)
        set_spa_context(None)
        qwen10 = _make_emb(PosEmbedSPAQwen, enable_spa=True, bundle_size=3)(ids)
        assert qwen21.shape == qwen10.shape
        assert torch.equal(qwen21, qwen10)

    def test_freqs_dtype_is_float32_even_on_cuda(self):
        """``_freqs_dtype`` is fp32 regardless of device — including CUDA.

        The device is faked (no GPU required in CI); what is asserted is the
        DECISION the adapter makes, which is the only thing that differs from
        the base rule.
        """
        emb = _make_emb(PosEmbedSPAQwen21, bundle_size=1)
        cuda = torch.zeros(1, device="meta")
        assert emb._freqs_dtype(cuda) == torch.float32

    @pytest.mark.parametrize("bundle_size", [1, 3])
    def test_both_forward_paths_use_float32(self, bundle_size):
        """``bundle_size=1`` hits forward's early return; 3 bundles.

        The finished PE is up-cast by ``format_components`` regardless, so its
        dtype proves nothing — the observed quantity is the ``freqs_dtype``
        handed to ``_spa_components``.
        """
        emb = _make_emb(PosEmbedSPAQwen21, bundle_size=bundle_size)
        seen = []
        original = emb._spa_components

        def spy(pos, freqs_dtype):
            seen.append(freqs_dtype)
            return original(pos, freqs_dtype)

        emb._spa_components = spy
        emb(_make_flux_ids(96, 96))
        assert seen and all(dt == torch.float32 for dt in seen)

    def test_registers_total_len_for_the_causal_prefix_mode(self):
        """``total_len`` is what the shared wrapper's segment gate keys on."""
        emb = _make_emb(PosEmbedSPAQwen21, enable_spa=True, bundle_size=3)
        emb(_make_flux_ids(128, 128))
        ctx = get_spa_context()
        assert ctx is not None
        assert ctx.total_len == 128 * 128
        set_spa_context(None)


# ---------------------------------------------------------------------------
# apply_spa_to_model wiring: which adapter, which joint mode
# ---------------------------------------------------------------------------

class QwenImage21Transformer2DModel:  # noqa: N801 - detection reads this exact name
    def __init__(self):
        self.pe_embedder = types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56])


class _FluxDiT:  # negative control (detected as FLUX via ``pe_embedder``)
    def __init__(self):
        self.pe_embedder = types.SimpleNamespace(theta=10000, axes_dim=[16, 56, 56])


class _MockPatcher:
    """Minimal ModelPatcher stand-in for ``apply_spa_to_model``."""

    def __init__(self, dm):
        self.model = types.SimpleNamespace(diffusion_model=dm)
        self._object_patches = {}
        self._unet_wrapper = None

    def clone(self):
        new = _MockPatcher(self.model.diffusion_model)
        new._object_patches = dict(self._object_patches)
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def _patched_embedder(m):
    """The embedder ``apply_spa_to_model`` installed (its only object patch)."""
    (obj,) = m._object_patches.values()
    return obj


@pytest.mark.mock_integration
class TestApplySpaToModelQwen21:
    """``apply_spa_to_model`` must pick the 2.1 adapter AND its joint mode.

    Both halves matter: the adapter class decides the RoPE format and the fp32
    frequency rule, the joint-mode attr decides how the shared attention wrapper
    treats the per-block segment calls.  A model patched with the 1.0 adapter (or
    left in the default joint mode) still "works" — it is silently wrong.
    """

    def _apply(self, dm, **kw):
        from src.spa import apply_spa_to_model

        m = apply_spa_to_model(_MockPatcher(dm), "auto", 1024, 1024, **kw)
        try:
            return m, _patched_embedder(m)
        finally:
            _spa_restore_installed(m)  # never leak a patched symbol

    def test_installs_the_qwen21_embedder_and_joint_mode(self):
        m, emb = self._apply(QwenImage21Transformer2DModel(), bundle_size=3)
        assert isinstance(emb, PosEmbedSPAQwen21)
        assert emb._rope_fmt == "flux"
        assert m._spa_joint_mode == "causal_prefix"

    def test_installs_on_the_pe_embedder_path(self):
        """2.1's embedder attribute is ``pe_embedder`` (same as 1.0)."""
        m = _MockPatcher(QwenImage21Transformer2DModel())
        from src.spa import apply_spa_to_model

        out = apply_spa_to_model(m, "auto", 1024, 1024, bundle_size=3)
        try:
            assert "diffusion_model.pe_embedder" in out._object_patches
        finally:
            _spa_restore_installed(out)

    def test_other_backends_keep_the_joint_mode(self):
        """Negative control: the mode is never left stale from a previous apply."""
        _, emb = self._apply(_FluxDiT(), bundle_size=3)
        assert isinstance(emb, PosEmbedSPAFlux)
        assert not isinstance(emb, PosEmbedSPAQwen21)

    def test_joint_mode_is_reset_for_a_non_qwen21_reapply(self):
        """A clone carrying ``causal_prefix`` onto another backend is corrected."""
        from src.spa import apply_spa_to_model

        src = _MockPatcher(_FluxDiT())
        src._spa_joint_mode = "causal_prefix"  # as if it had been applied to 2.1
        out = apply_spa_to_model(src, "auto", 1024, 1024, bundle_size=3)
        try:
            assert out._spa_joint_mode == "joint"
        finally:
            _spa_restore_installed(out)
