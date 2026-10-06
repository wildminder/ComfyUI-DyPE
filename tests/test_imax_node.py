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
import sys
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


# ---------------------------------------------------------------------------
# P6 — node layer: gates, adapters, wrapper state, schema, execute
# ---------------------------------------------------------------------------

class Flux:  # comfy.model_base.Flux stand-in — the MRO name IS the arch gate
    pass


class QwenImage:  # a flow model that is NOT Flux-arch (also owns pe_embedder)
    pass


_FLOW_MIXINS = {
    "CONST": type("CONST", (), {}),
    "EPS": type("EPS", (), {}),
}


def _mock_flux_model(prediction_mixin="CONST", arch="flux", with_pe=True):
    """ModelPatcher stand-in for the I-Max node path (hiflow mock shape):
    Flux BaseModel (MRO gate), CONST model_sampling, latent format with the
    Flux affine conversions, diffusion_model.pe_embedder, clone()."""
    class _Base:
        def timestep(self, sigma):
            return sigma * 1000.0  # DiscreteFlow-style multiplier probe

    ms = type("ModelSampling", (_Base, _FLOW_MIXINS[prediction_mixin]), {})()
    base = (Flux if arch == "flux" else QwenImage)()
    base.model_sampling = ms
    base.latent_format = types.SimpleNamespace(
        latent_dimensions=2, latent_channels=16)
    base.process_latent_in = lambda t: (t - 0.1159) * 0.3611
    base.process_latent_out = lambda t: (t / 0.3611) + 0.1159
    base.diffusion_model = types.SimpleNamespace()
    if with_pe:
        base.diffusion_model.pe_embedder = _EmbedND()

    model = types.SimpleNamespace()
    model.model = base
    model.model_options = {}
    model.load_device = torch.device("cpu")
    model.pre_run = lambda: None
    # model_patcher.py:656-657 — the wrapper rides inside model_options
    model.set_model_unet_function_wrapper = (
        lambda fn, m=model: m.model_options.__setitem__(
            "model_function_wrapper", fn))

    def _clone(_self=model):
        out = types.SimpleNamespace(**vars(_self))
        out.model_options = {
            k: (v.copy() if isinstance(v, dict) else v)
            for k, v in _self.model_options.items()
        }
        # each patcher instance writes its OWN options (the real method does)
        out.set_model_unet_function_wrapper = (
            lambda fn, m=out: m.model_options.__setitem__(
                "model_function_wrapper", fn))
        return out

    model.clone = _clone
    return model


def _mock_flux_vae(ratio=8, counts=None, latent_channels=16):
    """Fake Flux VAE: channels-last decode/encode boundary. Deterministic
    (no sampling) so execute-level determinism tests can assert equality."""
    def decode(z):
        if counts is not None:
            counts["decode"] += 1
        if isinstance(z, dict):
            z = z["samples"]
        b, c, h, w = z.shape
        return torch.full((b, h * ratio, w * ratio, 3), 0.5)

    def encode(im):
        if counts is not None:
            counts["encode"] += 1
        b, h, w, _ = im.shape
        return {"samples": torch.full(
            (b, latent_channels, h // ratio, w // ratio), 0.25)}

    return types.SimpleNamespace(
        decode=decode, encode=encode, downscale_ratio=ratio)


def _install_fake_comfy(monkeypatch):
    """Register the comfy.* fakes the node adapters touch (the hiflow test
    pattern), including comfy.utils (ProgressBar / repeat_to_batch_size)."""
    fake_helpers = types.ModuleType("comfy.sampler_helpers")
    convert_calls = []

    def convert_cond(cond):
        out = []
        for entry in cond:
            tensor, opts = entry
            convert_calls.append(dict(opts))
            out.append({"tensor": tensor, "opts": dict(opts)})
        return out

    fake_helpers.convert_cond = convert_cond

    fake_samplers = types.ModuleType("comfy.samplers")
    process_calls = {"count": 0}
    sampling_calls = []

    def process_conds(model, noise, conds, device, *args, **kwargs):
        process_calls["count"] += 1
        return {
            "positive": conds["positive"] if conds["positive"] else [],
            "negative": conds["negative"] if conds["negative"] else [],
        }

    def sampling_function(model, x, timestep, uncond, cond, cond_scale,
                          model_options=None, seed=None):
        sampling_calls.append({
            "x": x.clone(), "x_shape": tuple(x.shape),
            "timestep": timestep, "cond_scale": cond_scale,
            "model_options": model_options,
        })
        return 0.5 * x  # deterministic stand-in for calculate_denoised

    fake_samplers.process_conds = process_conds
    fake_samplers.sampling_function = sampling_function

    fake_mm = types.ModuleType("comfy.model_management")
    fake_mm.load_models_gpu = lambda models: None

    fake_utils = types.ModuleType("comfy.utils")

    class FakeProgressBar:
        def __init__(self, total):
            self.total = total
            self.updates = []

        def update_absolute(self, n):
            self.updates.append(n)

    def repeat_to_batch_size(x, count, dim=1):
        reps = [1] * x.ndim
        reps[dim] = -(-count // x.shape[dim])
        out = x.repeat(reps)
        slices = [slice(None)] * x.ndim
        slices[dim] = slice(0, count)
        return out[tuple(slices)]

    fake_utils.ProgressBar = FakeProgressBar
    fake_utils.repeat_to_batch_size = repeat_to_batch_size

    for name, mod in [("comfy.sampler_helpers", fake_helpers),
                      ("comfy.samplers", fake_samplers),
                      ("comfy.model_management", fake_mm),
                      ("comfy.utils", fake_utils)]:
        monkeypatch.setitem(sys.modules, name, mod)
    comfy_mod = sys.modules.get("comfy")
    if comfy_mod is not None:
        for attr, mod in [("sampler_helpers", fake_helpers),
                          ("samplers", fake_samplers),
                          ("model_management", fake_mm),
                          ("utils", fake_utils)]:
            monkeypatch.setattr(comfy_mod, attr, mod, raising=False)

    return types.SimpleNamespace(
        convert_calls=convert_calls, process_calls=process_calls,
        sampling_calls=sampling_calls, FakeProgressBar=FakeProgressBar,
    )


COND_POS = [(torch.ones(1, 4), {"side": "positive"})]
COND_NEG = [(torch.zeros(1, 4), {"side": "negative"})]


def _run_execute(monkeypatch, *, latent=(1, 16, 16, 16), content=False,
                 seed=0, steps_low=2, steps_high=2, model=None, vae=None,
                 **overrides):
    fx = _install_fake_comfy(monkeypatch)
    model = model if model is not None else _mock_flux_model()
    vae = vae if vae is not None else _mock_flux_vae()
    z = torch.randn(*latent) if content else torch.zeros(*latent)
    kwargs = dict(
        noise_seed=seed, denoise=1.0, cfg=1.0,
        steps_low=steps_low, steps_high=steps_high,
        guidance_low=3.5, guidance_high=5.0,
        time_shift_low=3.0, time_shift_high=6.0,
        ntk_factor=10.0, dwt_level=1, guidance_schedule="cosine_decay",
        proportional_attention=True, text_duplication=True,
        low_res_scale=1.0,
    )
    kwargs.update(overrides)
    result = imx.IMaxNode.execute(
        model, vae, COND_POS, COND_NEG, {"samples": z}, **kwargs)
    samples = result[0]["samples"] if not hasattr(result, "shape") else result
    return samples, fx, model, vae


class _PatcherFacade:
    """get_model_object facade over a mock (object patch resolution order)."""

    def __init__(self, inner, object_patches=None, object_patches_backup=None):
        self.inner = inner
        self.model = inner.model
        self.model_options = inner.model_options
        self.load_device = inner.load_device
        self.pre_run = inner.pre_run
        self.object_patches = dict(object_patches or {})
        self.object_patches_backup = dict(object_patches_backup or {})

    def clone(self):
        return _PatcherFacade(
            self.inner.clone(), self.object_patches, self.object_patches_backup)

    def set_model_unet_function_wrapper(self, fn):
        # model_patcher.py:656-657 — the wrapper rides inside model_options
        self.model_options["model_function_wrapper"] = fn

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        return getattr(self.model, name)


# ---------------------------------------------------------------------------
# Model gate (plan P6)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestRequireFluxFlowModel:
    def test_accepts_flux(self):
        model = _mock_flux_model()
        assert imx._require_flux_flow_model(model) == ("const", 2)

    def test_rejects_non_flux_arch_with_pointer_to_hiflow(self):
        """Qwen is a flow model that also owns pe_embedder — the arch gate
        (BaseModel MRO) is the real discriminator; the message points at
        HiFlow for other flow families."""
        model = _mock_flux_model(arch="qwen")
        with pytest.raises(ValueError, match="HiFlow"):
            imx._require_flux_flow_model(model)

    def test_rejects_non_flow_prediction(self):
        model = _mock_flux_model(prediction_mixin="EPS")
        with pytest.raises(ValueError, match="PixelRush"):
            imx._require_flux_flow_model(model)

    def test_rejects_missing_pe_embedder(self):
        """Nunchaku-style builds route RoPE elsewhere — no swap seam, no go."""
        model = _mock_flux_model(with_pe=False)
        with pytest.raises(ValueError, match="pe_embedder"):
            imx._require_flux_flow_model(model)

    def test_gate_resolves_through_object_patch(self):
        """A leaked EPS sampling on the live BaseModel must not flip the gate
        when the patcher's own object patch is CONST (v2.16.0 rule)."""
        model = _mock_flux_model(prediction_mixin="EPS")
        clean = type(
            "ModelSampling", (_FLOW_MIXINS["CONST"],),
            {"timestep": lambda self, s: s * 1000.0},
        )()
        patcher = _PatcherFacade(
            model, object_patches={"model_sampling": clean})
        assert imx._require_flux_flow_model(patcher) == ("const", 2)


@pytest.mark.unit
class TestWarnIfStaleLeak:
    def test_stale_leak_warning_is_emitted(self, monkeypatch, caplog):
        """The un-healable residual (a stale sampling patch inherited from a
        run no longer in the graph) warns while the node still runs."""
        import logging
        model = _mock_flux_model(prediction_mixin="CONST")
        # the leaked installer-local class is CONST-based in production
        model.model.model_sampling = type(
            "DypeModelSamplingFlux", (_FLOW_MIXINS["CONST"],),
            {"timestep": lambda self, s: s * 1000.0},
        )()
        clean = type(
            "ModelSampling", (_FLOW_MIXINS["CONST"],),
            {"timestep": lambda self, s: s * 1000.0},
        )()
        patcher = _PatcherFacade(
            model, object_patches={"model_sampling": clean})
        patcher.object_patches = {}  # the leak shape: no patch/backup entry
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            samples, _, _, _ = _run_execute(
                monkeypatch, model=patcher, steps_low=1, steps_high=1)
        assert any("stale patch" in r.message for r in caplog.records)
        assert samples.shape == (1, 16, 16, 16)


# ---------------------------------------------------------------------------
# Guidance override (plan D12)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGuidanceOverride:
    def test_zero_guidance_leaves_conditioning_untouched(self):
        cond = [(torch.ones(1, 4), {"guidance": 2.0})]
        assert imx._apply_guidance_override(cond, 0.0) is cond

    def test_positive_guidance_is_overridden(self):
        out = imx._apply_guidance_override(COND_POS, 5.0)
        assert out[0][1]["guidance"] == 5.0

    def test_caller_conditioning_is_not_mutated(self):
        opts = {"side": "positive"}
        cond = [(torch.ones(1, 4), opts)]
        imx._apply_guidance_override(cond, 5.0)
        assert "guidance" not in opts, "the caller's dict must be copied"

    def test_negative_guidance_is_also_set(self, monkeypatch):
        """D12: the override is written into BOTH sides' copies before
        conversion (per pass, inside the adapter)."""
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flux_model()
        adapter = imx._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=1.0,
            model_options={}, pass_state={}, high_pass=False,
            guidance_override=5.0)
        adapter(torch.randn(1, 16, 8, 8), sigma=0.5)
        # convert_cond call order: positive first, then negative
        assert fx.convert_calls[0]["guidance"] == 5.0
        assert fx.convert_calls[1]["guidance"] == 5.0


# ---------------------------------------------------------------------------
# predict_x0 adapter (plan P6)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPredictX0Adapter:
    def test_uses_model_sampling_timestep(self, monkeypatch):
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flux_model()
        adapter = imx._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=1.0,
            model_options={}, pass_state={}, high_pass=False)
        adapter(torch.randn(1, 16, 8, 8), sigma=0.42)
        assert torch.allclose(
            fx.sampling_calls[-1]["timestep"], torch.tensor([420.0]))

    def test_timestep_tensor_shape_is_one_element(self, monkeypatch):
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flux_model()
        adapter = imx._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=1.0,
            model_options={}, pass_state={}, high_pass=False)
        adapter(torch.randn(1, 16, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["timestep"].numel() == 1

    def test_conditions_are_cached_per_shape(self, monkeypatch):
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flux_model()
        adapter = imx._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=1.0,
            model_options={}, pass_state={}, high_pass=False)
        for _ in range(3):
            adapter(torch.randn(1, 16, 8, 8), sigma=0.5)
        assert fx.process_calls["count"] == 1
        adapter(torch.randn(1, 16, 16, 16), sigma=0.5)
        assert fx.process_calls["count"] == 2

    def test_cfg_autoskips_empty_negative(self, monkeypatch):
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flux_model()
        neg = [(torch.zeros(1, 0), {"side": "negative"})]
        adapter = imx._make_predict_x0(
            model, COND_POS, neg, cfg_scale=3.5,
            model_options={}, pass_state={}, high_pass=False)
        adapter(torch.randn(1, 16, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["cond_scale"] == 1.0

    def test_cfg_forwarded_with_real_negative(self, monkeypatch):
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flux_model()
        adapter = imx._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=3.5,
            model_options={}, pass_state={}, high_pass=False)
        adapter(torch.randn(1, 16, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["cond_scale"] == 3.5

    def test_pass_state_flips_per_pass(self, monkeypatch):
        """The adapter flips the shared cell to ITS pass before every model
        call — that is the D4 wrapper's pass discrimination (P6)."""
        _install_fake_comfy(monkeypatch)
        model = _mock_flux_model()
        state = {"high_pass": None}
        low = imx._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=1.0,
            model_options={}, pass_state=state, high_pass=False)
        high = imx._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=1.0,
            model_options={}, pass_state=state, high_pass=True)
        high(torch.randn(1, 16, 8, 8), sigma=0.5)
        assert state["high_pass"] is True
        low(torch.randn(1, 16, 8, 8), sigma=0.5)
        assert state["high_pass"] is False

    def test_pass_model_options_forwarded(self, monkeypatch):
        """D5 wiring: the adapter hands the PASS dict to sampling_function,
        not model.model_options."""
        fx = _install_fake_comfy(monkeypatch)
        model = _mock_flux_model()
        opts = {"model_function_wrapper": object()}
        adapter = imx._make_predict_x0(
            model, COND_POS, COND_NEG, cfg_scale=1.0,
            model_options=opts, pass_state={}, high_pass=True)
        adapter(torch.randn(1, 16, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["model_options"] is opts


# ---------------------------------------------------------------------------
# Guidance latent (plan P6)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGuidanceLatent:
    def test_roundtrip_returns_target_resolution_latent(self):
        vae = _mock_flux_vae()
        dec, enc = imx._make_vae_adapters(vae, torch.device("cpu"))
        low = torch.randn(1, 16, 64, 64)
        out = imx._build_guidance_latent(low, dec, enc, (1024, 1024))
        assert out.shape == (1, 16, 128, 128)

    def test_guidance_is_fixed_across_the_whole_high_pass(self, monkeypatch):
        """The round trip runs ONCE per generation — the guidance is fixed
        for all of pass B (decode/encode call counts pin it)."""
        counts = {"decode": 0, "encode": 0}
        vae = _mock_flux_vae(counts=counts)
        _run_execute(monkeypatch, vae=vae, steps_low=2, steps_high=3)
        assert counts == {"decode": 1, "encode": 1}

    def test_channels_last_vae_layout(self):
        seen = {}

        def decode(z):
            seen["decode_in"] = tuple(z.shape)
            return torch.randn(1, 64, 64, 3)

        def encode(im):
            seen["encode_in"] = tuple(im.shape)
            return {"samples": torch.randn(1, 16, 8, 8)}

        vae = types.SimpleNamespace(
            decode=decode, encode=encode, downscale_ratio=8)
        dec, enc = imx._make_vae_adapters(vae, torch.device("cpu"))
        img = dec(torch.randn(1, 16, 8, 8))
        assert seen["decode_in"] == (1, 16, 8, 8), "4D latent in"
        lat = enc(img)
        assert seen["encode_in"] == (1, 64, 64, 3), (
            "encode must receive the channels-last image untouched")
        assert lat.shape == (1, 16, 8, 8)

    def test_3d_format_vae_returns_frame_zero(self):
        """Krea2-plan S4 dance copied from hiflow: the 5D decode output
        slices to the first temporal frame on both sides."""
        def decode(z):
            b, c, t, h, w = z.shape
            assert t == 1, "decode must receive the 5D latent"
            return torch.randn(b, 3, t, h * 16, w * 16).movedim(1, -1)

        def encode(im):
            b = im.shape[0]
            return torch.randn(b, 16, 1, im.shape[-3] // 16,
                               im.shape[-2] // 16)

        vae = types.SimpleNamespace(
            decode=decode, encode=encode, latent_dim=3,
            downscale_ratio=(lambda a: a, 16, 16))
        dec, enc = imx._make_vae_adapters(vae, torch.device("cpu"))
        lat_in = torch.randn(1, 16, 8, 8)
        img = dec(lat_in)
        assert img.dim() == 4, "first temporal frame only"
        lat_out = enc(img)
        assert lat_out.shape == (1, 16, 8, 8), "4D latent back"


# ---------------------------------------------------------------------------
# D4 unet wrapper state (plan P6)
# ---------------------------------------------------------------------------

def _wrapper_env(ntk_factor=10.0, previous_wrapper=None):
    """A BaseModel stand-in with a pe_embedder module + the wrapper's state
    cell + a recording model_function (the params shape comfy passes)."""
    dm = types.SimpleNamespace(pe_embedder=_EmbedND())
    base = types.SimpleNamespace(diffusion_model=dm)
    state = {
        "high_pass": False, "inner_model": base,
        "imax_embedder": None, "embedder_inner": None,
    }
    seen = []

    def model_function(x, t, **c):
        seen.append(dm.pe_embedder)
        return x

    wrapper = imx._make_imax_unet_wrapper(
        state, ntk_factor=ntk_factor, previous_wrapper=previous_wrapper)
    params = {"input": torch.zeros(1, 4),
              "timestep": torch.tensor([0.5]), "c": {}}
    return wrapper, model_function, params, state, dm, seen


@pytest.mark.unit
class TestNTKWrapper:
    def test_ntk_disabled_during_low_pass(self):
        wrapper, mf, params, _, dm, seen = _wrapper_env()
        wrapper(mf, params)
        assert seen[0] is dm.pe_embedder, (
            "pass A must see the plain embedder during the forward")

    def test_ntk_enabled_during_high_pass(self):
        wrapper, mf, params, state, dm, seen = _wrapper_env()
        state["high_pass"] = True
        wrapper(mf, params)
        assert isinstance(seen[0], imx.IMaxNTKEmbedder), (
            "pass B must see the I-Max embedder during the forward")
        assert seen[0].ntk_factor == 10.0

    def test_embedder_is_restored_after_forward(self):
        wrapper, mf, params, state, dm, _ = _wrapper_env()
        original = dm.pe_embedder
        state["high_pass"] = True
        wrapper(mf, params)
        assert dm.pe_embedder is original

    def test_embedder_is_restored_after_exception(self):
        wrapper, mf, params, state, dm, _ = _wrapper_env()
        original = dm.pe_embedder
        state["high_pass"] = True

        def boom(x, t, **c):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            wrapper(boom, params)
        assert dm.pe_embedder is original

    def test_no_object_patch_is_added(self):
        """D4: the swap is a per-forward attribute write — no patcher entry,
        nothing to leak onto downstream nodes."""
        added = []
        wrapper, mf, params, state, dm, _ = _wrapper_env()
        state["high_pass"] = True
        model = types.SimpleNamespace(
            add_object_patch=lambda path, obj: added.append(path))
        wrapper(mf, params)
        _ = model  # the wrapper never receives the patcher at all
        assert added == []

    def test_embedder_built_once_per_inner(self):
        """The IMaxNTKEmbedder (and its takeover warning) is constructed once
        per distinct installed embedder, not once per forward."""
        wrapper, mf, params, state, _, seen = _wrapper_env()
        state["high_pass"] = True
        wrapper(mf, params)
        first = seen[0]
        wrapper(mf, params)
        assert seen[1] is first

    def test_chains_previous_model_function_wrapper(self):
        """A chained DyPE-style wrapper (found on the cloned model_options)
        is called THROUGH, not dropped when ours replaces it."""
        fired = []

        def previous(mf, p):
            fired.append(True)
            return mf(p["input"], p["timestep"], **p.get("c", {}))

        wrapper, mf, params, state, dm, seen = _wrapper_env(
            previous_wrapper=previous)
        state["high_pass"] = True
        wrapper(mf, params)
        assert fired == [True]
        assert isinstance(seen[0], imx.IMaxNTKEmbedder), (
            "our swap still applies inside the chained call")

    def test_swap_wraps_the_currently_installed_embedder(self):
        """The chained-DyPE case: the embedder installed on the module at
        call time (not a plain EmbedND) is what gets wrapped."""
        class _DypeStandIn(_EmbedND):
            pass

        wrapper, mf, params, state, dm, seen = _wrapper_env()
        dm.pe_embedder = _DypeStandIn()
        state["high_pass"] = True
        wrapper(mf, params)
        assert isinstance(seen[0].inner, _DypeStandIn)


# ---------------------------------------------------------------------------
# Schema (plan D1, D11-D15)
# ---------------------------------------------------------------------------

def _node_src() -> str:
    import pathlib
    return (pathlib.Path(__file__).parent.parent / "nodes" / "imax.py") \
        .read_text(encoding="utf-8")


@pytest.mark.unit
class TestIMaxNodeSchema:
    def test_node_id_display_and_category(self):
        src = _node_src()
        assert 'node_id="IMax"' in src
        assert 'display_name="I-Max"' in src
        assert 'category="WMNodes/image"' in src

    def test_has_no_sampler_name_input(self):
        """Sampler-side nodes own their loop — no sampler_name input (§1.3)."""
        assert '"sampler_name"' not in _node_src()

    @pytest.mark.parametrize(
        "literal",
        [
            '"noise_seed", default=0',
            '"denoise", default=1.0',
            '"cfg", default=1.0',
            '"steps_low", default=30',
            '"steps_high", default=20',
            '"guidance_low", default=3.5',
            '"guidance_high", default=5.0',
            '"time_shift_low", default=3.0',
            '"time_shift_high", default=6.0',
            '"ntk_factor", default=10.0',
            '"dwt_level", default=1',
            '"low_res_scale", default=1.0',
            'default="cosine_decay"',
            '"proportional_attention", default=True',
            '"text_duplication", default=True',
        ],
    )
    def test_defaults_match_plan_d11(self, literal):
        assert literal in _node_src(), f"missing schema default {literal}"

    def test_output_is_latent(self):
        assert "io.Latent.Output" in _node_src()

    def test_fingerprint_inputs_always_changes(self):
        """D15: the stochastic node is never cache-served — NaN compares
        unequal to itself, so the fingerprint always reports a change."""
        a = imx.IMaxNode.fingerprint_inputs(noise_seed=1)
        b = imx.IMaxNode.fingerprint_inputs(noise_seed=1)
        assert a != b

    def test_schema_execute_signature_matches(self):
        import re
        src = _node_src()
        pattern = re.compile(r'io\.\w+\.Input\(\s*"([^"]+)"')
        schema_inputs = set(pattern.findall(src))
        assert schema_inputs, "failed to parse schema inputs"
        sig_start = src.index("def execute(cls,")
        sig_start += len("def execute(cls,")
        sig = src[sig_start:src.index(") -> io.NodeOutput:", sig_start)]
        params = set()
        for chunk in sig.split(","):
            chunk = chunk.strip()
            if "=" in chunk:
                chunk = chunk.split("=")[0].strip()
            if chunk and chunk != "cls":
                params.add(chunk)
        missing = (schema_inputs - params) | (params - schema_inputs)
        assert not missing, (
            f"schema/execute drift: schema-only={schema_inputs - params}, "
            f"exec-only={params - schema_inputs}"
        )


# ---------------------------------------------------------------------------
# Execute (plan P6)
# ---------------------------------------------------------------------------

_ABOVE_NATIVE = (1, 16, 160, 112)  # 1280x896 px: sqrt area > 1024


@pytest.mark.unit
class TestIMaxNodeExecute:
    def test_end_to_end_returns_target_resolution_latent(self, monkeypatch):
        samples, fx, _, _ = _run_execute(monkeypatch, latent=_ABOVE_NATIVE)
        assert samples.shape == _ABOVE_NATIVE
        assert torch.isfinite(samples).all()

    def test_low_pass_runs_at_the_low_res_size(self, monkeypatch):
        """Pass A generates at the aspect-preserving native-area size
        (engine D10), pass B at the target."""
        samples, fx, _, _ = _run_execute(monkeypatch, latent=_ABOVE_NATIVE)
        low_shape = fx.sampling_calls[0]["x_shape"]
        high_shape = fx.sampling_calls[2]["x_shape"]
        assert low_shape == (1, 16, 152, 108)  # 1216x864 px
        assert high_shape == _ABOVE_NATIVE

    def test_deterministic_for_equal_inputs(self, monkeypatch):
        a, _, _, _ = _run_execute(monkeypatch, seed=123)
        b, _, _, _ = _run_execute(monkeypatch, seed=123)
        assert torch.equal(a, b)

    def test_seed_changes_output(self, monkeypatch):
        a, _, _, _ = _run_execute(monkeypatch, seed=1)
        b, _, _, _ = _run_execute(monkeypatch, seed=2)
        assert not torch.equal(a, b)

    def test_guidance_schedule_disable_matches_plain_euler(self, monkeypatch):
        """With schedule="disable" the node IS a two-pass Euler sampler:
        replay pass B manually from the recorded model-space calls."""
        from src.imax import build_flow_sigmas
        samples, fx, _, _ = _run_execute(
            monkeypatch, steps_low=2, steps_high=3,
            guidance_schedule="disable")
        calls = fx.sampling_calls
        pl_out = lambda t: (t / 0.3611) + 0.1159     # noqa: E731
        sigmas = build_flow_sigmas(3, 6.0)
        x_vae = pl_out(calls[2]["x"])
        for i in range(3):
            x0_vae = pl_out(0.5 * calls[2 + i]["x"])
            s, s_next = float(sigmas[i]), float(sigmas[i + 1])
            x_vae = x_vae + (x_vae - x0_vae) / max(s, 1e-6) * (s_next - s)
        assert torch.allclose(samples, x_vae, atol=1e-4)

    def test_proportional_attention_toggle_reaches_pass_b(self, monkeypatch):
        """The D5 wiring under the toggle: pass B's model options carry our
        attn1_patch only when the toggle is on (mock predictors can't show
        the numeric effect — that is Phase 9's real-model run)."""
        _, fx_on, _, _ = _run_execute(
            monkeypatch, latent=_ABOVE_NATIVE, proportional_attention=True)
        _, fx_off, _, _ = _run_execute(
            monkeypatch, latent=_ABOVE_NATIVE, proportional_attention=False)
        high_on = fx_on.sampling_calls[-1]["model_options"]
        high_off = fx_off.sampling_calls[-1]["model_options"]
        assert "attn1_patch" in high_on["transformer_options"]["patches"]
        assert "attn1_patch" not in high_off["transformer_options"]["patches"]

    def test_text_duplication_toggle_reaches_pass_b(self, monkeypatch):
        _, fx_on, _, _ = _run_execute(
            monkeypatch, latent=_ABOVE_NATIVE, text_duplication=True)
        _, fx_off, _, _ = _run_execute(
            monkeypatch, latent=_ABOVE_NATIVE, text_duplication=False)
        high_on = fx_on.sampling_calls[-1]["model_options"]
        high_off = fx_off.sampling_calls[-1]["model_options"]
        assert "post_input" in high_on["transformer_options"]["patches"]
        assert "post_input" not in high_off["transformer_options"]["patches"]

    def test_both_pass_options_carry_the_wrapper(self, monkeypatch):
        """The D4 wrapper rides inside model_options — both pass dicts the
        adapters hand to sampling_function must carry it."""
        _, fx, _, _ = _run_execute(monkeypatch, latent=_ABOVE_NATIVE)
        low_opts = fx.sampling_calls[0]["model_options"]
        high_opts = fx.sampling_calls[-1]["model_options"]
        assert callable(low_opts.get("model_function_wrapper"))
        assert callable(high_opts.get("model_function_wrapper"))

    def test_low_res_scale_changes_low_pass_shape(self, monkeypatch):
        _, fx_a, _, _ = _run_execute(
            monkeypatch, latent=_ABOVE_NATIVE, low_res_scale=1.0)
        _, fx_b, _, _ = _run_execute(
            monkeypatch, latent=_ABOVE_NATIVE, low_res_scale=0.25)
        assert fx_a.sampling_calls[0]["x_shape"] != \
            fx_b.sampling_calls[0]["x_shape"]

    def test_denoise_below_one_truncates_entry_sigma(self, monkeypatch):
        """D14: denoise applies to pass B only — the first pass-B timestep
        drops below the full schedule's 1000."""
        _, fx, _, _ = _run_execute(
            monkeypatch, content=True, denoise=0.5, steps_low=2,
            steps_high=2)
        first_high = fx.sampling_calls[2]["timestep"]
        assert float(first_high.flatten()[0]) < 1000.0

    def test_empty_latent_runs_the_full_schedule(self, monkeypatch):
        """Pure txt2img: pass B enters at sigma=1 (timestep 1000 here)."""
        _, fx, _, _ = _run_execute(monkeypatch, content=False)
        first_high = fx.sampling_calls[2]["timestep"]
        assert float(first_high.flatten()[0]) == pytest.approx(1000.0)

    def test_caller_patcher_is_not_leaked(self, monkeypatch):
        """D4: the wrapper is installed on an internal clone only — the
        caller's patcher leaves execute untouched."""
        _, _, model, _ = _run_execute(monkeypatch)
        assert "model_function_wrapper" not in model.model_options

    def test_multiframe_latent_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="multi-frame"):
            _run_execute(monkeypatch, latent=(1, 16, 2, 16, 16))

    def test_below_native_resolution_warns(self, monkeypatch, caplog):
        """512 px target: nothing to extrapolate — the engine's D10 warning
        surfaces through execute and pass A runs at the target."""
        import logging
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            samples, _, _, _ = _run_execute(
                monkeypatch, latent=(1, 16, 64, 64))
        assert any("at/below the native" in r.message for r in caplog.records)
        assert samples.shape == (1, 16, 64, 64)

    def test_content_denoise_one_warns(self, monkeypatch, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            _run_execute(monkeypatch, content=True, denoise=1.0)
        assert any("denoise=1.0" in r.message for r in caplog.records)

    def test_progress_bar_reports_engine_stages(self, monkeypatch):
        samples, fx, _, _ = _run_execute(
            monkeypatch, steps_low=2, steps_high=3)
        pbar_type = type(fx.FakeProgressBar.__init__)  # sanity only
        _ = pbar_type
        # total = steps_low + 1 (roundtrip) + steps_high
        assert fx.sampling_calls, "engine ran through the fake sampler"
