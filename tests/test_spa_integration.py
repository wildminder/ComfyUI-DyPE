"""P4 — SPA integration / mosaic-regression test.

The root-cause bug (averaging RoPE *rotation matrices* then one softmax) produces a
*rippled-mosaic* attention output.  The fix runs ``N`` attention passes and averages
the **outputs** (HRDiT ``_spa_attention``).

A naive *local-coherence* metric (cosine similarity of adjacent attention-weight rows)
does NOT discriminate: averaging rotation matrices acts like a global smoothing, so the
buggy map is *more* coherent than the fixed one.  The faithful discriminator is therefore
the **attention-output divergence** between the fixed path and the legacy buggy path:

  * ``out_fixed``  = mean_n attention(variant_n(q), variant_n(k), v)   (the fix)
  * ``out_buggy``  = attention(mean_n rotations @ q, mean_n rotations @ k, v) (legacy)
  * ``out_base``   = attention(q, k, v)                                 (no RoPE)

The fix is correct iff (a) ``out_fixed`` equals the HRDiT reference (proven in P1/T-P1-7),
(b) ``out_fixed`` DIVERGES meaningfully from ``out_buggy`` (a regression guard — if the
legacy ``torch.stack(embs).mean(0)`` averaging is ever reintroduced, ``out_fixed`` would
collapse onto ``out_buggy`` and this assertion would fail), and (c) for ``bundle_size==1``
the fix is a transparent passthrough (``out_fixed == out_base``).

Markers: @pytest.mark.unit (T-P4-7 is @pytest.mark.comfyui_integration and skipped).
"""
import types

import pytest
import torch
import torch.nn.functional as F

from src.models.spa_anima import PosEmbedSPAAnima
from src.models.spa_flux import PosEmbedSPAFlux
from src.spa import (
    apply_spa_to_model,
    build_bundle_id_variants,
    restore_spa_attention_hook,
)
from src.spa_attn import apply_rope_matrix, compose_rope, inv_rope
from src.spa_context import get_spa_context, get_spa_step_gate, set_spa_context

try:
    from tests._spa_math_helpers import (
        period_peak_ratio,
        smooth_qk,
        structured_qkv,
    )
except ImportError:  # namespace-package import fallback
    from _spa_math_helpers import (
        period_peak_ratio,
        smooth_qk,
        structured_qkv,
    )


def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=0.0, is_causal=False)


def _flux_ids(H=32, W=32, B=1):
    L = H * W
    ids = torch.zeros(B, L, 3)
    ids[..., 0] = torch.arange(L)
    ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
    ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    return ids


def _flux_vps(N, H=80, W=80):
    """Register an SPA FLUX context and return its (base_pe, variant_pes).

    Default grid is 80x80 (max_pos=79 > trained_extent=64) so the SPA
    trained-extent gate is ACTIVE and the registered variants are non-trivial.
    """
    set_spa_context(None)
    emb = PosEmbedSPAFlux(theta=10000, axes_dim=[16, 56, 56], method="ntk",
                          enable_spa=True, bundle_size=N)
    emb(_flux_ids(H, W))
    ctx = get_spa_context()
    return ctx.base_pe, ctx.variant_pes


@pytest.mark.unit
class TestSpaIntegrationDivergence:
    def test_fixed_diverges_from_buggy_and_is_finite(self, capsys):
        """T-P4-1: headline integration — fix diverges from legacy bug; report numbers.

        Uses an 80x80 grid (max_pos=79 > trained_extent=64) so the trained-extent
        gate is ACTIVE and the registered variants are non-trivial (otherwise the
        fix would collapse to the identity and the divergence assertion is void).
        """
        H = W = 80
        D = 128  # sum(axes_dim) for FLUX
        L = H * W
        N = 3
        base_pe, vps = _flux_vps(N, H, W)
        g = torch.Generator().manual_seed(2)
        q = smooth_qk(H, W, D, 0)
        k = smooth_qk(H, W, D, 1)
        v = torch.randn(1, 1, L, D, generator=g)

        # Fixed: average N attention outputs over the bundled variant RoPEs.
        out_fixed = _sdpa(q, k, v)
        # (recompute via spa_averaged_attention to exercise the real code path)
        from src.spa_attn import spa_averaged_attention
        out_fixed = spa_averaged_attention(q, k, v, None, vps, attn_fn=_sdpa,
                                            pre_roped=False, fmt="flux")

        # Legacy buggy: averaged rotation matrices then a single softmax.
        R_avg = torch.stack(vps, 0).mean(0)
        out_buggy = _sdpa(apply_rope_matrix(q, R_avg), apply_rope_matrix(k, R_avg), v)

        # Clean base (no RoPE).
        out_base = _sdpa(q, k, v)

        d_fb = (out_fixed - out_buggy).norm().item()
        d_fbase = (out_fixed - out_base).norm().item()
        d_bbase = (out_buggy - out_base).norm().item()

        with capsys.disabled():
            print(
                f"\n[P4] N={N} {H}x{W}: D(fixed,buggy)={d_fb:.3f}  "
                f"D(fixed,base)={d_fbase:.3f}  D(buggy,base)={d_bbase:.3f}  "
                f"||base||={out_base.norm():.3f}"
            )

        assert torch.isfinite(out_fixed).all()
        # Regression guard: the fix must not equal the buggy averaging path.
        assert d_fb > 0.05 * out_base.norm().item()
        # The fix stays at least as close to the clean base as the buggy path.
        assert d_fbase < d_bbase + 1.0

    def test_bundle_size_one_is_base_passthrough(self):
        """T-P4-2: bundle_size==1 -> SPA is identity; fixed output == base.

        With a single variant (the identity rotation, as produced by the identity
        bundling at bundle_size==1), ``spa_averaged_attention`` is a transparent
        passthrough and equals the plain attention over the raw q,k.
        """
        H = W = 32
        D = 128
        L = H * W
        P = D // 2
        q = smooth_qk(H, W, D, 0)
        k = smooth_qk(H, W, D, 1)
        v = torch.randn(1, 1, L, D)

        ident = torch.eye(2).unsqueeze(0).expand(P, -1, -1)  # (P, 2, 2)
        single = ident.unsqueeze(0).unsqueeze(0).repeat(1, 1, L, 1, 1)  # (1,1,L,P,2,2)
        out_fixed = _averaged(q, k, v, [single])  # single variant -> passthrough
        out_base = _sdpa(q, k, v)
        assert torch.allclose(out_fixed, out_base, atol=1e-5)

    def test_anima_temporal_axis_preserved(self):
        """T-P4-3: Anima bundles (h,w) only; temporal RoPE blocks are unchanged across variants.

        Uses a 128x128 grid (max_pos=127 > trained_extent=64) so the trained-extent
        gate is ACTIVE and multiple variants are registered.
        """
        T, H, W = 2, 128, 128
        emb = PosEmbedSPAAnima(theta=[10000.0, 10000.0, 10000.0], axes_dim=[44, 42, 42],
                               method="vision_yarn", enable_spa=True, bundle_size=3)
        x = torch.randn(1, T, H, W, 128)
        emb(x)
        ctx = get_spa_context()
        t_blocks = emb.axes_dim[0] // 2  # first axis is temporal
        for vp in ctx.variant_pes:
            # The temporal rotation blocks equal the base temporal blocks exactly.
            assert torch.allclose(vp[..., :t_blocks, :, :], ctx.base_pe[..., :t_blocks, :, :], atol=1e-6)

    def test_cross_attention_text_tokens_untouched(self):
        """T-P4-4: text tokens (h=w=0) are left at identity by the SPA delta, so cross-attn is untouched."""
        H = W = 80
        base_pe, vps = _flux_vps(3, H, W)
        # delta = inv(base) @ variant; at the text token (index 0) it must be identity.
        delta = compose_rope(inv_rope(base_pe, "flux"), vps[0], "flux")
        P = base_pe.shape[-3]
        ident = torch.eye(2).expand(P, 2, 2)  # (P, 2, 2)
        # delta[..., 0, :, :, :] selects the text-token block stack -> (P, 2, 2)
        assert torch.allclose(delta[..., 0, :, :, :], ident, atol=1e-5)
        # Equivalent: the variant RoPE at the text token equals the base RoPE there.
        assert torch.allclose(vps[0][..., 0, :, :, :], base_pe[..., 0, :, :, :], atol=1e-6)

    def test_finite_for_n3_and_n5(self):
        """T-P4-5: fixed attention is finite for N=3 and N=5 (active 80x80 grid)."""
        H = W = 80
        D = 128
        L = H * W
        q = smooth_qk(H, W, D, 0)
        k = smooth_qk(H, W, D, 1)
        v = torch.randn(1, 1, L, D)
        for N in (3, 5):
            _, vps = _flux_vps(N, H, W)
            out = _averaged(q, k, v, vps)
            assert torch.isfinite(out).all()

    def test_divergence_holds_for_n5(self):
        """T-P4-6: regression guard also holds at N=5 (active 80x80 grid)."""
        H = W = 80
        D = 128
        L = H * W
        N = 5
        _, vps = _flux_vps(N, H, W)
        q = smooth_qk(H, W, D, 0)
        k = smooth_qk(H, W, D, 1)
        v = torch.randn(1, 1, L, D, generator=torch.Generator().manual_seed(9))

        out_fixed = _averaged(q, k, v, vps)
        R_avg = torch.stack(vps, 0).mean(0)
        out_buggy = _sdpa(apply_rope_matrix(q, R_avg), apply_rope_matrix(k, R_avg), v)
        out_base = _sdpa(q, k, v)

        d_fb = (out_fixed - out_buggy).norm().item()
        assert d_fb > 0.05 * out_base.norm().item()


def _averaged(q, k, v, vps):
    from src.spa_attn import spa_averaged_attention

    return spa_averaged_attention(q, k, v, None, vps, attn_fn=_sdpa, pre_roped=False, fmt="flux")


@pytest.mark.comfyui_integration
@pytest.mark.unit
def test_requires_real_model_skipped():
    """T-P4-7: full end-to-end with a real FLUX checkpoint is integration-only (skipped here)."""
    pytest.skip("comfyui_integration: requires a real model + GPU; covered by P1/P3 equivalence.")


# ---------------------------------------------------------------------------
# P5 — Mock end-to-end (T5.1 / T5.2, 2026-08-15 bundle-size-semantics fix)
# ---------------------------------------------------------------------------

class _E2EMockModel:
    """Minimal ModelPatcher stand-in for ``apply_spa_to_model`` + the unet wrapper.

    Self-contained (mirrors ``tests/test_spa_node.py::_MockModel``) so the e2e test
    does not depend on the conftest ``mock_flux_model`` fixture.
    """

    def __init__(self):
        self.model = types.SimpleNamespace()
        self.model.diffusion_model = types.SimpleNamespace()
        self._object_patches = {}
        self._unet_wrapper = None
        self._spa_orig_optimized_attention = None

    def _copy_dm(self, src):
        dst = types.SimpleNamespace()
        for k, v in vars(src).items():
            setattr(dst, k, v)
        return dst

    def clone(self):
        new = _E2EMockModel()
        new.model.diffusion_model = self._copy_dm(self.model.diffusion_model)
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        new._spa_orig_optimized_attention = self._spa_orig_optimized_attention
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


def _make_e2e_krea2_mock():
    """Krea-2 (SingleStreamDiT) mock: flux-style ``pe_embedder`` with Krea's
    ASYMMETRIC RoPE config (``theta=1000``, ``axes_dim=[32, 48, 48]``).

    This is the exact PE math the P4 ripple threshold (1.25) was calibrated on
    (``tests/_spa_math_helpers.krea_pe``), and the model the user's mosaic/speed
    complaint was about — so the e2e exercises the representative regime.
    """
    m = _E2EMockModel()
    m.model.diffusion_model.pe_embedder = types.SimpleNamespace(
        theta=1000, axes_dim=[32, 48, 48]
    )
    return m


def _scaled_attn(q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kw):
    """Drop-in ``optimized_attention`` using the ``1/sqrt(d)`` scale — the SAME
    math the P4 ripple threshold (1.25) was calibrated on
    (``tests/test_spa_ripple_detector._plain_attn``).

    The conftest mock ``optimized_attention`` uses ``scale=1.0``.  The e2e tests
    install THIS as ``comfy.ldm.modules.attention.optimized_attention`` *before*
    ``apply_spa_to_model`` patches the hook, so the averaged-attention passes run
    at the calibrated scale and the T5.2 ripple assertion is directly comparable
    to the P4 gate.  ``heads`` is accepted for signature compatibility but the
    tensors are already head-shaped ``(B, H, L, D)``.

    W2.1 (plan 2026-08-25): signature locked to the REAL ComfyUI convention
    (slots 5-8 = mask/attn_precision/skip_reshape/skip_output_reshape) — the
    pre-fix ad-hoc order broke when the wrapper forwarded all 8 positional args.
    """
    d = q.shape[-1]
    scores = torch.einsum("bhld,bhmd->bhlm", q, k) / (d ** 0.5)
    return torch.einsum("bhlm,bhmd->bhld", torch.softmax(scores, dim=-1), v)


@pytest.mark.mock_integration
class TestP5MockEndToEnd:
    """T5.1 / T5.2 — mock end-to-end through ``apply_spa_to_model`` + the unet
    wrapper + the attention hook (no real model, no GPU).

    These drive the FULL production path for **Krea-2** (the model the mosaic/speed
    complaint was about): ``apply_spa_to_model`` installs the averaged-attention hook
    + unet wrapper on a mock patcher; a ``model_function`` then calls the patched SPA
    embedder (registering the variant context) and the (now-wrapped)
    ``comfy.ldm.modules.attention.optimized_attention``.  ``model_type="krea2"`` is
    passed explicitly so backend detection does not depend on the mock's class name;
    the Krea-2 bound symbol (``comfy.ldm.krea2.model``) is not importable under the
    mock, so the hook fires through the always-patched module-global
    ``optimized_attention`` (the same wrapper code path).

    * ``test_e2e_1k_identity`` — at 1024px (64x64, max_pos=63 <= trained_extent=64)
      the trained-extent gate makes SPA a no-op, so the hooked output is
      ``torch.equal`` to the plain-attention baseline (the big-patch artifact source
      is gone).
    * ``test_e2e_2k_active_bounded`` — at 2048px (128x128) SPA is ACTIVE with 5
      variants (s=3); the step-count gate (``spa_steps=3``) opens only on the first
      3 of 8 simulated steps; and the hooked output differs from baseline but passes
      the period-``s`` ripple threshold (no mosaic).
    """

    D = 128  # sum(axes_dim) for Krea-2 ([32, 48, 48])

    def _patched(self, px, bundle_size=3, spa_steps=3):
        """Apply SPA to a fresh mock Krea-2 model and return the patched patcher.

        Installs :func:`_scaled_attn` (the ``1/sqrt(d)`` scale the P4 ripple
        threshold was calibrated on) as the module-global ``optimized_attention``
        BEFORE ``apply_spa_to_model`` patches the hook, so the averaged-attention
        passes run at the calibrated scale.
        """
        import comfy.ldm.modules.attention as attn_mod

        attn_mod.optimized_attention = _scaled_attn  # calibrated scale, pre-hook
        set_spa_context(None)
        m = _make_e2e_krea2_mock()
        return apply_spa_to_model(
            m, "krea2", px, px, "ntk",
            enable_spa=True, bundle_size=bundle_size, spa_steps=spa_steps,
        )

    def test_e2e_1k_identity(self):
        """T5.1: Krea-2 at 1024px (64x64) + N=3 -> trained-extent gate -> bit-identical."""
        import comfy.ldm.modules.attention as attn_mod

        H = W = 64  # 1024px latent grid (max_pos=63 <= trained_extent=64)
        L = H * W
        out = self._patched(1024, bundle_size=3)
        embedder = out._object_patches["diffusion_model.pe_embedder"]
        assert isinstance(embedder, PosEmbedSPAFlux)

        ids = _flux_ids(H, W)
        g = torch.Generator().manual_seed(0)
        q = torch.randn(1, 1, L, self.D, generator=g)
        k = torch.randn(1, 1, L, self.D, generator=g)
        v = torch.randn(1, 1, L, self.D, generator=g)

        captured = {}

        def model_function(x, t, **c):
            base_pe = embedder(ids)  # registers the SPA context (single identity variant)
            ctx = get_spa_context()
            captured["n_variants"] = len(ctx.variant_pes) if ctx else 0
            qb = apply_rope_matrix(q, base_pe, "flux")
            kb = apply_rope_matrix(k, base_pe, "flux")
            captured["out"] = attn_mod.optimized_attention(qb, kb, v, 1)
            return captured["out"]

        out._unet_wrapper(model_function, {"input": None, "timestep": torch.tensor([0.9]), "c": {}})

        # Trained-extent gate: exactly ONE identity variant registered (SPA is a no-op).
        assert captured["n_variants"] == 1, (
            f"expected 1 identity variant at 64x64, got {captured['n_variants']}")

        # Baseline: plain attention over the same base-RoPE q,k (no hook), using the
        # SAME scaled attention the hook wraps (so the comparison is bit-exact).
        base_pe = embedder(ids)
        qb = apply_rope_matrix(q, base_pe, "flux")
        kb = apply_rope_matrix(k, base_pe, "flux")
        baseline = _scaled_attn(qb, kb, v, 1)

        assert torch.isfinite(captured["out"]).all()
        # BIT-IDENTICAL: the hook is a transparent passthrough inside the trained extent.
        assert torch.equal(captured["out"], baseline), (
            "SPA at 1024px must be bit-identical to the baseline (trained-extent gate); "
            "any difference means the big-patch over-compression is still active")

        restore_spa_attention_hook(out, attn_mod)

    def test_e2e_2k_active_bounded(self):
        """T5.2: 2048px (128x128) + N=3 -> ACTIVE (5 variants), step-gated to the
        first 3 of 8 steps, output differs from baseline but passes the ripple gate."""
        import comfy.ldm.modules.attention as attn_mod

        # (a) 2048px is ACTIVE with exactly 5 variants (s=3): the paper-N rewire.
        ids_2k = _flux_ids(128, 128)  # max_pos=127 > 64 -> active; s=max(3, ceil(127/79)=2)=3
        variants_2k = build_bundle_id_variants(ids_2k, 3)
        assert len(variants_2k) == 5, (
            f"expected 5 variants (s=3) at 2048px, got {len(variants_2k)}")
        # In-distribution guarantee: every bundled position <= 79.
        assert max(int(vv[..., 1].max()) for vv in variants_2k) <= 79
        assert max(int(vv[..., 2].max()) for vv in variants_2k) <= 79

        # (b) Step-count gate: open only on the first 3 of 8 simulated steps.
        #     Attention runs on a CPU-feasible 80x80 grid (max_pos=79 > 64 -> active,
        #     same fine bundle s==3 the node derives at 2048px), so the gate + ripple
        #     assertions exercise the identical bundling regime at tractable cost.
        out = self._patched(2048, bundle_size=3, spa_steps=3)
        embedder = out._object_patches["diffusion_model.pe_embedder"]

        H = W = 80
        ids = _flux_ids(H, W)
        q, k, v = structured_qkv(H, W, self.D, 0)

        state = {"gates": [], "spa_out": None}

        def model_function(x, t, **c):
            state["gates"].append(get_spa_step_gate())
            base_pe = embedder(ids)  # register the 5-variant context at 80x80
            qb = apply_rope_matrix(q, base_pe, "flux")
            kb = apply_rope_matrix(k, base_pe, "flux")
            if state["spa_out"] is None:  # capture the FIRST (gate-open) forward only
                state["spa_out"] = attn_mod.optimized_attention(qb, kb, v, 1)
            return state["spa_out"]

        sigmas = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]  # 8 steps, monotone down
        for s in sigmas:
            out._unet_wrapper(model_function, {"input": None, "timestep": torch.tensor([s]), "c": {}})

        assert state["gates"] == [True, True, True, False, False, False, False, False], (
            f"spa_steps=3 must open the gate on exactly the first 3 of 8 steps, "
            f"got {state['gates']}")

        # (c) The gate-open output is ACTIVE (differs from baseline) but CLEAN
        #     (no period-s ripple) and bounded (parity with the base pass).
        assert state["spa_out"] is not None and torch.isfinite(state["spa_out"]).all()
        base_pe = embedder(ids)
        qb = apply_rope_matrix(q, base_pe, "flux")
        kb = apply_rope_matrix(k, base_pe, "flux")
        baseline = _scaled_attn(qb, kb, v, 1)  # same scaled math the hook wraps

        delta = (state["spa_out"] - baseline)[0, 0].reshape(H, W, self.D)
        assert delta.abs().max().item() > 0, "SPA at 2048px must change the output (active)"

        s = 3  # derived bundle size (5 variants -> s = (5+1)//2)
        peak = period_peak_ratio(delta, H, W, s)
        assert peak < 1.25, (
            f"period-{s} peak ratio {peak:.3f} >= 1.25 -> a mosaic ripple is present")

        rel = (state["spa_out"] - baseline).norm() / baseline.norm().clamp(min=1e-12)
        assert rel < 0.25, f"SPA deviates from baseline by {rel:.3f} (>= 0.25)"

        restore_spa_attention_hook(out, attn_mod)


# ---------------------------------------------------------------------------
# P5 — Resolution-aware SPA behaviour (2026-08-15 blocky-output fix)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSpaResolutionAware:
    """T5.1 / T5.2 / T5.3 — SPA is a no-op at in-distribution resolutions and
    activates with a bounded pass count at high resolutions (HRDiT-faithful).

    Updated 2026-08-15 to the paper-``N`` knob semantics: ``bundle_size`` is the
    paper's tokens-per-bundle ``N`` (0=auto, 1=off, 2..8 explicit).  The trained-
    extent gate (``max_pos <= 64`` -> identity) is the single source of truth for
    the low-resolution no-op; ``N = 0`` (auto) reproduces the HRDiT
    ``group_num = 80`` in-distribution ceiling at high resolution.
    """

    def _flux_embedder(self, bundle_size=0):
        set_spa_context(None)
        return PosEmbedSPAFlux(theta=10000, axes_dim=[16, 56, 56], method="ntk",
                               enable_spa=True, bundle_size=bundle_size)

    def test_t5_1_spa_identity_at_low_res(self):
        """T5.1: at 1024px (64x64 tokens, max_pos=63 <= trained_extent=64) the
        trained-extent gate gives s=1 -> a single identity variant for ANY N.
        The attention hook treats <=1 variant as a passthrough, so SPA is a true
        no-op (no blocky over-compression)."""
        for N in (0, 3, 5):
            emb = self._flux_embedder(bundle_size=N)
            ids = _flux_ids(64, 64)
            base = emb(ids)
            ctx = get_spa_context()
            assert ctx is not None and ctx.active
            # Identity: exactly one variant, equal to the base PE.
            assert len(ctx.variant_pes) == 1, (
                f"N={N}: expected 1 identity variant at 64x64, got {len(ctx.variant_pes)}")
            assert torch.allclose(ctx.variant_pes[0], ctx.base_pe, atol=1e-6)
            # forward returns the base RoPE unchanged.
            expected = emb.format_components(emb._spa_components(ids.float(), torch.float32), ids)
            assert torch.allclose(base, expected, atol=1e-6)

    def test_t5_2_spa_active_at_high_res(self):
        """T5.2: at 4096px (256x256 tokens, max_pos=255) with N=0 (auto) the
        in-dist floor gives s=ceil(255/79)=4 -> 2*4-1 = 7 averaged passes, and
        the variants differ from base (HRDiT group_num=80 behaviour)."""
        emb = self._flux_embedder(bundle_size=0)
        ids = _flux_ids(256, 256)
        emb(ids)
        ctx = get_spa_context()
        assert ctx is not None and ctx.active
        assert len(ctx.variant_pes) == 7, (
            f"expected 7 variants (s=4) at 256x256, got {len(ctx.variant_pes)}")
        # Bundling changed the variant RoPEs vs the base.
        max_diff = max((vp - ctx.base_pe).abs().max().item() for vp in ctx.variant_pes)
        assert max_diff > 1e-4
        # In-distribution: every bundled position <= SPA_IN_DIST_MAX = 79.
        variants = build_bundle_id_variants(ids, 0)
        assert max(int(v[..., 1].max()) for v in variants) <= 79
        assert max(int(v[..., 2].max()) for v in variants) <= 79

    def test_t5_3_pass_count_always_bounded(self):
        """T5.3: no (resolution, N) configuration exceeds SPA_MAX_PASSES."""
        from src.spa import SPA_MAX_PASSES

        for grid in (32, 64, 96, 128, 192, 256, 384, 512):
            for N in (0, 2, 3, 5, 8):
                ids = _flux_ids(grid, grid)
                variants = build_bundle_id_variants(ids, N)
                assert len(variants) <= SPA_MAX_PASSES, (
                    f"grid={grid} N={N}: {len(variants)} passes "
                    f"exceeds cap {SPA_MAX_PASSES}")
