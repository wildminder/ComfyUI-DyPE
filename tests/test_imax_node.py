"""Tests for nodes/imax.py — I-Max inference toolkit (Tier 2).

Phases 4-5 of plan 2026-10-05:
- the NTK-aware RoPE embedder (D4) and its omega formula, pinned against a
  local mirror of comfy's EmbedND rope (ldm/flux/math.py:20-31 +
  ldm/flux/layers.py:15-30) and the reference omega
  (I-Max/transformer_flux.py:40-58);
- the proportional-attention q pre-scale (D8, base-4608 log per
  attention_processor.py:1773-1777);
- the text-duplication post_input patch (D9, transformer_flux.py:364-375);
- the per-pass model_options builder (D5).

Markers: @pytest.mark.unit. No ComfyUI required — nodes/imax.py imports
comfy only lazily, and the oracles below are self-contained mirrors.
"""

import logging
import math
import types

import pytest
import torch

import nodes.imax as imx


# ---------------------------------------------------------------------------
# Oracles / fixtures
# ---------------------------------------------------------------------------

def _reference_rope(pos: torch.Tensor, dim: int, theta) -> torch.Tensor:
    """Bit-exact mirror of comfy/ldm/flux/math.py:20-31 (EmbedND rope)."""
    assert dim % 2 == 0
    scale = torch.linspace(
        0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64,
    )
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega)
    out = torch.stack(
        [out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1,
    )
    return out.reshape(*out.shape[:-1], 2, 2).to(dtype=torch.float32)


def _reference_embednd(ids, theta, axes_dim) -> torch.Tensor:
    """Bit-exact mirror of comfy/ldm/flux/layers.py:15-30 (EmbedND)."""
    embs = [
        _reference_rope(ids[..., i], axes_dim[i], theta)
        for i in range(ids.shape[-1])
    ]
    return torch.cat(embs, dim=-3).unsqueeze(1)


def _flux_ids(n_txt: int = 512, grid_h: int = 64, grid_w: int = 64):
    """Flux-like joint position ids: 512 zero text rows + an h x w grid."""
    txt = torch.zeros(1, n_txt, 3)
    img = torch.zeros(1, grid_h * grid_w, 3)
    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    img[..., 1] = rows.repeat_interleave(grid_w)
    img[..., 2] = cols.repeat(grid_h)
    return torch.cat([txt, img], dim=1)


class _EmbedND(torch.nn.Module):
    """Plain comfy EmbedND stand-in: native theta/axes_dim, no dynamics."""

    def __init__(self, theta=10000, axes_dim=(16, 56, 56)):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids):
        return _reference_embednd(ids, self.theta, self.axes_dim)


class _PosEmbedFluxStandIn(torch.nn.Module):
    """DyPE-family stand-in (src/base.py DyPEBasePosEmbed contract):
    theta/axes_dim, optional per-axis ``thetas`` and ``set_timestep``."""

    def __init__(self, theta=10000, axes_dim=(16, 56, 56)):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)
        self.thetas = None
        self.current_timestep = 1.0

    def set_timestep(self, timestep: float) -> None:
        self.current_timestep = timestep

    def forward(self, ids):
        return _reference_embednd(ids, self.theta, self.axes_dim)


def _existing_attn_patch(q, k, v, **kwargs):
    return {"q": q}


def _existing_post_input_patch(state):
    return state


_WRAPPER_SENTINEL = object()


def _patched_model():
    """ModelPatcher stand-in AFTER clone+wrapper (the P6 execute shape):
    model_options carries the D4 wrapper plus pre-existing SPA/HAP-style
    patches the builder must preserve."""
    return types.SimpleNamespace(model_options={
        "model_function_wrapper": _WRAPPER_SENTINEL,
        "transformer_options": {
            "patches": {
                "attn1_patch": [_existing_attn_patch],
                "post_input": [_existing_post_input_patch],
            },
            "samplers": {"erg": 1},
        },
    })


def _post_input_state(grid_h, grid_w, n_txt=7, batch=1, channels=16):
    img = torch.randn(batch, grid_h * grid_w, channels)
    txt = torch.randn(batch, n_txt, channels)
    img_ids = torch.zeros(batch, grid_h * grid_w, 3)
    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    img_ids[..., 1] = rows.repeat_interleave(grid_w)
    img_ids[..., 2] = cols.repeat(grid_h)
    txt_ids = torch.zeros(batch, n_txt, 3)
    return {
        "img": img, "txt": txt, "img_ids": img_ids, "txt_ids": txt_ids,
        "transformer_options": {},
    }


# ---------------------------------------------------------------------------
# NTK RoPE embedder (plan D4, phase P4)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIMaxNTKEmbedder:
    def test_ntk_factor_one_is_identity_to_inner(self):
        """ntk_factor=1 at the native N=4608: bitwise equal to the plain
        EmbedND output (omega grid, einsum and layout all mirrored)."""
        ids = _flux_ids()
        inner = _EmbedND()
        emb = imx.IMaxNTKEmbedder(inner, ntk_factor=1.0)
        assert torch.equal(emb(ids), inner(ids))

    def test_theta_is_scaled_by_ntk_factor(self):
        """omega == (theta*ntk_factor)^-s when the clip floor sits below the
        NTK branch (g = sqrt((N-512)/4096) = 10 > 10^0.875)."""
        omega = imx.ntk_rope_omega(10000, 16, 10.0, 512 + 100 * 4096)
        dim = 16
        scale = torch.linspace(
            0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64,
        )
        assert torch.equal(omega, 1.0 / ((10000.0 * 10.0) ** scale))

    def test_ntk_clip_floor_is_applied_above_native(self):
        """At 2048x2048 (N=16896, g=2): omega is the max of the NTK branch
        and theta^-s/2 — the floor wins for s > log(g)/log(f)."""
        theta, dim, f, n = 10000, 16, 10.0, 16896
        omega = imx.ntk_rope_omega(theta, dim, f, n)
        scale = torch.linspace(
            0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64,
        )
        ntk = 1.0 / ((theta * f) ** scale)
        floor = 1.0 / (theta ** scale) / math.sqrt((n - 512) / 4096)
        assert torch.equal(omega, torch.maximum(ntk, floor))
        # both branches bind somewhere (the max is doing real work)
        assert not torch.equal(omega, ntk)
        assert not torch.equal(omega, floor)
        # hand check: s=0.125 (f^s=1.33 < g=2 -> NTK binds), s=0.375
        # (f^s=2.37 > g=2 -> floor binds)
        assert omega[1] == pytest.approx((10000.0 * 10.0) ** -0.125)
        assert omega[3] == pytest.approx(10000.0 ** -0.375 / 2.0)

    def test_ntk_clip_floor_is_inactive_at_native(self):
        """At the native N=4608 the floor equals the native omega, so the
        result is the PLAIN embedding even at ntk_factor=10 — a
        native-resolution pass cannot be moved by the embedder."""
        dim = 56
        omega = imx.ntk_rope_omega(10000, dim, 10.0, 4608)
        scale = torch.linspace(
            0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64,
        )
        assert torch.equal(omega, 1.0 / (10000 ** scale))

    def test_output_shape_matches_inner(self):
        ids = _flux_ids().repeat(2, 1, 1)  # batch 2
        inner = _EmbedND()
        emb = imx.IMaxNTKEmbedder(inner, ntk_factor=10.0)
        out = emb(ids)
        assert out.shape == inner(ids).shape
        assert out.shape == (2, 1, ids.shape[1], 64, 2, 2)

    def test_reads_theta_from_chained_dype_embedder(self):
        """theta is read off the INSTALLED embedder (a DyPE-style wrapper
        with a modified base), not assumed to be 10000."""
        ids = _flux_ids()
        inner = _PosEmbedFluxStandIn(theta=20000)
        emb = imx.IMaxNTKEmbedder(inner, ntk_factor=1.0)
        assert emb.theta == 20000
        assert torch.equal(emb(ids), _reference_embednd(ids, 20000, [16, 56, 56]))

    def test_reads_per_axis_thetas_from_dype_embedder(self):
        """DyPEBasePosEmbed allows per-axis thetas (src/base.py:19-26) —
        each axis must use its own base."""
        ids = _flux_ids()
        inner = _PosEmbedFluxStandIn(theta=10000)
        inner.thetas = [9000, 11000, 11000]
        emb = imx.IMaxNTKEmbedder(inner, ntk_factor=1.0)
        assert emb.thetas == [9000, 11000, 11000]
        expected = torch.cat(
            [_reference_rope(ids[..., i], inner.axes_dim[i], inner.thetas[i])
             for i in range(3)],
            dim=-3,
        ).unsqueeze(1)
        assert torch.equal(emb(ids), expected)

    def test_warns_when_foreign_positional_patch_present(self, caplog):
        inner = _PosEmbedFluxStandIn()
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            imx.IMaxNTKEmbedder(inner, ntk_factor=10.0)
        assert "positional embedder" in caplog.text

    def test_embedder_does_not_mutate_inner(self):
        inner = _PosEmbedFluxStandIn(theta=12345)
        inner.thetas = [9000, 11000, 11000]
        before = (inner.theta, list(inner.axes_dim), list(inner.thetas))
        emb = imx.IMaxNTKEmbedder(inner, ntk_factor=10.0)
        emb(_flux_ids())
        after = (inner.theta, list(inner.axes_dim), list(inner.thetas))
        assert before == after
        assert emb.inner is inner

    def test_missing_theta_or_axes_dim_raises(self):
        with pytest.raises(ValueError, match="theta"):
            imx.IMaxNTKEmbedder(types.SimpleNamespace())

    def test_ntk_clip_skips_below_text_token_floor(self):
        """Below 512 joint tokens the reference's floor formula is undefined
        (math domain error); the clip degrades to the NTK branch instead."""
        dim = 16
        omega = imx.ntk_rope_omega(10000, dim, 10.0, 100)
        scale = torch.linspace(
            0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64,
        )
        assert torch.equal(omega, 1.0 / ((10000.0 * 10.0) ** scale))

    def test_module_imports_no_comfy(self):
        """P4 constraint: torch-only at module scope — every comfy import
        in nodes/imax.py is lazy (inside a function)."""
        assert not [k for k in vars(imx) if k.startswith("comfy")]


# ---------------------------------------------------------------------------
# Proportional attention patch (plan D8, phase P5)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestProportionalAttentionPatch:
    def test_ratio_is_one_at_native_seq_len(self):
        assert imx.proportional_attention_factor(4608) == 1.0

    def test_ratio_matches_reference_formula(self):
        """The reference is log BASE 4608 (attention_processor.py:1777,
        math.log(key.size(2), train_seq_len)); head_dim cancels in the
        q pre-scale ratio. This also pins the base against the natural-log
        misread of plan D8's written form."""
        for n in (9216, 16896, 262144 + 512):
            expected = math.sqrt(math.log(n) / math.log(4608))
            assert imx.proportional_attention_factor(n) == pytest.approx(expected)
        # the misread form (natural log of the ratio) would give 0.83 —
        # a q scale BELOW 1 at 2x, i.e. sharpened attention: not ours.
        assert imx.proportional_attention_factor(9216) != pytest.approx(
            math.sqrt(math.log(9216 / 4608)))

    def test_ratio_is_one_below_native_seq_len(self):
        """D8 clamp: at/below native the patch is a no-op (the reference
        would scale q DOWN below 4608 tokens)."""
        for n in (1, 512, 2048, 4607):
            assert imx.proportional_attention_factor(n) == 1.0

    def test_ratio_is_monotonic_above_native(self):
        seqs = [4608, 5120, 9216, 16896, 65536 + 512, 262144 + 512]
        factors = [imx.proportional_attention_factor(n) for n in seqs]
        assert factors == sorted(factors)
        assert len(set(factors)) == len(factors)  # strictly increasing

    def test_patch_scales_q_only(self):
        n = 16896  # 2048x2048 target
        q = torch.randn(1, 2, n, 8)
        k = torch.randn(1, 2, n, 8)
        v = torch.randn(1, 2, n, 8)
        patch = imx.make_proportional_attention_patch()
        out = patch(q, k, v, pe=None, attn_mask=None, extra_options={})
        expected = math.sqrt(math.log(n) / math.log(4608))
        assert torch.equal(out["q"], q * expected)
        # k/v are returned ABSENT — comfy's out.get("k", k) loop keeps the
        # caller's originals (and the inputs are never mutated in place).
        assert "k" not in out and "v" not in out
        assert torch.equal(k, k.clone())
        assert torch.equal(v, v.clone())

    def test_patch_preserves_flux_attn1_signature(self):
        """Called exactly as comfy calls it (ldm/flux/layers.py:237):
        p(q, k, v, pe=..., attn_mask=..., extra_options=...) -> dict with q;
        absent keys fall back to the originals in comfy's out.get() loop."""
        q = torch.randn(1, 1, 9216, 4)
        patch = imx.make_proportional_attention_patch()
        out = patch(q, q, q, pe="pe-sentinel", attn_mask=None, extra_options={})
        assert isinstance(out, dict)
        assert "q" in out
        assert out["q"] is not q  # actually scaled at 2x

    def test_patch_leaves_k_and_v_untouched(self):
        n = 9216
        q = torch.randn(1, 1, n, 4)
        k = torch.randn(1, 1, n, 4)
        v = torch.randn(1, 1, n, 4)
        out = imx.make_proportional_attention_patch()(q, k, v)
        assert "k" not in out and "v" not in out
        assert out["q"] is not q

    @pytest.mark.parametrize("heads,head_dim,n", [(2, 128, 9216), (1, 64, 5120)])
    def test_patch_accepts_both_double_and_single_block_shapes(
        self, heads, head_dim, n,
    ):
        """Double and single blocks hand the patch the same [b, heads, N, d]
        joint-stream layout (ldm/flux/layers.py:227-229, :338-340)."""
        q = torch.randn(1, heads, n, head_dim)
        k = torch.randn(1, heads, n, head_dim)
        v = torch.randn(1, heads, n, head_dim)
        out = imx.make_proportional_attention_patch()(q, k, v)
        expected = math.sqrt(math.log(n) / math.log(4608))
        assert torch.equal(out["q"], q * expected)

    def test_patch_is_noop_at_native_res(self):
        """At/below the native joint length the SAME tensor is returned —
        no per-block copy of q."""
        q = torch.randn(1, 1, 4608, 8)
        out = imx.make_proportional_attention_patch()(q, q, q)
        assert out["q"] is q


# ---------------------------------------------------------------------------
# Text duplication patch (plan D9, phase P5)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTextDuplicationPatch:
    def test_native_resolution_is_a_noop(self):
        state = _post_input_state(64, 64)
        out = imx.make_text_duplication_patch()(state)
        assert out is state

    def test_doubles_text_tokens_at_2x(self):
        """2048x1024 target: latent grid 128x64 -> nh=2, nw=1."""
        state = _post_input_state(128, 64, n_txt=7)
        out = imx.make_text_duplication_patch()(state)
        assert out["txt"].shape[1] == 14
        assert torch.equal(out["txt"][:, :7], state["txt"])
        assert torch.equal(out["txt"][:, 7:], state["txt"])

    def test_quadruples_text_tokens_at_4x(self):
        """2048x2048 target: latent grid 128x128 -> 4 copies."""
        state = _post_input_state(128, 128, n_txt=5)
        out = imx.make_text_duplication_patch()(state)
        assert out["txt"].shape[1] == 20

    def test_txt_ids_grid_is_offset(self):
        """Copy (i, j) sits at position-grid offset (i*64, j*64) — the
        reference loop order i-major, j-minor (transformer_flux.py:370-374)."""
        state = _post_input_state(128, 128, n_txt=3)
        out = imx.make_text_duplication_patch()(state)
        blocks = out["txt_ids"][0].chunk(4, dim=0)
        for (want_h, want_w), block in zip(
            [(0, 0), (0, 64), (64, 0), (64, 64)], blocks,
        ):
            assert block[..., 1].max().item() == want_h
            assert block[..., 2].max().item() == want_w

    def test_txt_ids_batch_axis_untouched(self):
        state = _post_input_state(128, 64, n_txt=4, batch=2)
        out = imx.make_text_duplication_patch()(state)
        assert out["txt_ids"].shape[0] == 2
        assert torch.count_nonzero(out["txt_ids"][..., 0]) == 0

    def test_returns_all_four_keys(self):
        state = _post_input_state(128, 128)
        out = imx.make_text_duplication_patch()(state)
        for key in ("img", "txt", "img_ids", "txt_ids"):
            assert key in out
        assert out["transformer_options"] is state["transformer_options"]

    def test_img_and_img_ids_untouched(self):
        state = _post_input_state(128, 64)
        out = imx.make_text_duplication_patch()(state)
        assert out["img"] is state["img"]
        assert out["img_ids"] is state["img_ids"]

    def test_ceil_covers_partial_native_tiles(self):
        """D9 deviation (documented): an 80-patch-wide grid (1280 px) spans
        TWO native tiles — ceil duplicates, the reference's floor((max+1)
        //64) would leave a single text copy for two tiles of image."""
        state = _post_input_state(80, 64, n_txt=6)
        out = imx.make_text_duplication_patch()(state)
        assert out["txt"].shape[1] == 12

    def test_none_img_ids_returns_state_unchanged(self):
        """post_input also fires when img_ids is None (pe=None models) —
        the patch must pass the state through, not crash on .max()."""
        state = _post_input_state(64, 64)
        state["img_ids"] = None
        out = imx.make_text_duplication_patch()(state)
        assert out is state


# ---------------------------------------------------------------------------
# Per-pass model options (plan D5, phase P5)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPassModelOptions:
    def test_disabled_options_have_no_imax_patches(self):
        model = _patched_model()
        opts = imx.build_pass_model_options(model, enabled=False)
        patches = opts["transformer_options"]["patches"]
        assert patches["attn1_patch"] == [_existing_attn_patch]
        assert patches["post_input"] == [_existing_post_input_patch]
        # a clone, not the source dict
        assert opts is not model.model_options
        assert opts == model.model_options

    def test_enabled_options_contain_both_patches(self):
        model = _patched_model()
        opts = imx.build_pass_model_options(model, enabled=True)
        patches = opts["transformer_options"]["patches"]
        assert len(patches["attn1_patch"]) == 2
        assert len(patches["post_input"]) == 2
        # the appended callables are the real toolkit patches, live
        q = torch.randn(1, 1, 9216, 4)
        assert patches["attn1_patch"][-1](q, q, q)["q"] is not q
        doubled = patches["post_input"][-1](_post_input_state(128, 64, n_txt=3))
        assert doubled["txt"].shape[1] == 6

    def test_existing_patches_are_preserved(self):
        model = _patched_model()
        opts = imx.build_pass_model_options(model, enabled=True)
        patches = opts["transformer_options"]["patches"]
        assert patches["attn1_patch"][0] is _existing_attn_patch
        assert patches["post_input"][0] is _existing_post_input_patch

    def test_source_model_options_are_not_mutated(self):
        model = _patched_model()
        source_patches = model.model_options["transformer_options"]["patches"]
        source_attn_list = source_patches["attn1_patch"]
        imx.build_pass_model_options(model, enabled=True)
        imx.build_pass_model_options(model, enabled=False)
        assert model.model_options["transformer_options"]["patches"] \
            is source_patches
        assert source_patches["attn1_patch"] is source_attn_list
        assert len(source_patches["attn1_patch"]) == 1
        assert len(source_patches["post_input"]) == 1

    def test_two_passes_get_independent_dicts(self):
        model = _patched_model()
        opts_a = imx.build_pass_model_options(model, enabled=True)
        opts_b = imx.build_pass_model_options(model, enabled=False)
        assert opts_a is not opts_b
        assert opts_a["transformer_options"] is not opts_b["transformer_options"]
        assert opts_a["transformer_options"]["patches"] \
            is not opts_b["transformer_options"]["patches"]
        # mutating pass A must not reach pass B or the source
        opts_a["transformer_options"]["patches"]["attn1_patch"].append("x")
        opts_a["transformer_options"]["samplers"]["erg"] = 99
        assert "x" not in opts_b["transformer_options"]["patches"]["attn1_patch"]
        assert opts_b["transformer_options"]["samplers"]["erg"] == 1
        assert model.model_options["transformer_options"]["patches"][
            "attn1_patch"] == [_existing_attn_patch]
        assert model.model_options["transformer_options"]["samplers"]["erg"] == 1

    def test_both_pass_dicts_carry_model_function_wrapper(self):
        """The D4 unet wrapper rides INSIDE model_options
        (model_patcher.py:656-657) — so BOTH per-pass dicts must carry it:
        the wrapper is what swaps pe_embedder per forward (P6)."""
        model = _patched_model()
        opts_a = imx.build_pass_model_options(model, enabled=False)
        opts_b = imx.build_pass_model_options(model, enabled=True)
        assert opts_a["model_function_wrapper"] is _WRAPPER_SENTINEL
        assert opts_b["model_function_wrapper"] is _WRAPPER_SENTINEL

    def test_toggles_disable_individual_patches(self):
        model = _patched_model()
        opts = imx.build_pass_model_options(
            model, enabled=True, proportional_attention=False,
        )
        patches = opts["transformer_options"]["patches"]
        assert len(patches["attn1_patch"]) == 1
        assert len(patches["post_input"]) == 2

    def test_non_patch_options_are_cloned_too(self):
        model = _patched_model()
        opts = imx.build_pass_model_options(model, enabled=True)
        samplers = opts["transformer_options"]["samplers"]
        assert samplers == {"erg": 1}
        assert samplers is not model.model_options["transformer_options"]["samplers"]
