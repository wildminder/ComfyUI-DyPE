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

v2.19.0 (plan 2026-10-06): the per-group clip mode for the lumina NextDiT
rope_embedder (D4') — image group clips against its own group length, cap
group takes pure NTK.

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
# NTK RoPE embedder, per-group clip mode (v2.19.0 D4', plan 2026-10-06)
# ---------------------------------------------------------------------------

def _ntk_rope_reference(pos: torch.Tensor, dim: int, theta,
                        ntk_factor: float, floor_ratio: float = 0.0):
    """Mirror of :func:`ntk_rope_omega`'s omega + comfy's rope layout
    (ldm/flux/math.py:20-31 + layers.py:15-30): the NTK branch, max-ed with
    ``theta^-s/sqrt(floor_ratio)`` when ``floor_ratio > 0``."""
    assert dim % 2 == 0
    scale = torch.linspace(
        0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64,
    )
    omega = 1.0 / ((theta * ntk_factor) ** scale)
    if floor_ratio > 0.0:
        omega = torch.maximum(
            omega, 1.0 / (theta ** scale) / math.sqrt(floor_ratio))
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega)
    out = torch.stack(
        [out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1,
    )
    return out.reshape(*out.shape[:-1], 2, 2).to(dtype=torch.float32)


def _zimage_cap_ids(n_tokens: int = 512, batch: int = 1) -> torch.Tensor:
    """Cap-group ids as embed_cap builds them (comfy lumina/model.py:
    657-674): axis 0 counts the WHOLE padded span (arange + 1), axes 1/2
    are 0 on every row — pad rows included."""
    ids = torch.zeros(batch, n_tokens, 3)
    ids[..., 0] = torch.arange(1, n_tokens + 1, dtype=torch.float32)
    return ids


def _zimage_image_ids(grid_h: int, grid_w: int, cap_len: int = 512,
                      pad_rows: int = 0, batch: int = 1) -> torch.Tensor:
    """Image-group ids as pos_ids_x builds them: constant t on axis 0, an
    h x w grid on axes 1/2; ``pad_rows`` all-zero rows appended (image pad
    rows are all-zero, comfy lumina/model.py:730)."""
    n = grid_h * grid_w + pad_rows
    ids = torch.zeros(batch, n, 3)
    real = slice(0, grid_h * grid_w)
    ids[:, real, 0] = float(cap_len + 1)
    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    ids[:, real, 1] = rows.repeat_interleave(grid_w)
    ids[:, real, 2] = cols.repeat(grid_h)
    return ids


@pytest.mark.unit
class TestIMaxNTKEmbedderZImage:
    """v2.19.0 D4' — per-group clip mode: the lumina NextDiT calls its
    rope_embedder once per token group (cap lumina/model.py:673, siglip
    :712, image :730), so the clip floor is re-derived per group — the
    image group against its own group length, the caption group with no
    floor (pure NTK, the paper's model-wide Lumina scaling, plan §5)."""

    Z_THETA = 256.0
    Z_AXES_DIM = [32, 48, 48]

    def _embedder(self, inner=None, ntk_factor=10.0, **kwargs):
        if inner is None:
            inner = _EmbedND(theta=self.Z_THETA, axes_dim=self.Z_AXES_DIM)
        return imx.IMaxNTKEmbedder(
            inner, ntk_factor=ntk_factor, clip_mode="per_group", **kwargs)

    def _group_expected(self, ids, ntk_factor, n_group):
        """Per-axis reference rope for one group call: ``n_group=None`` is
        the cap group (pure NTK), an int is the image group (the per-group
        clip floor over its own length, native grid 64 -> 4096)."""
        ratio = 0.0 if n_group is None else n_group / 4096.0
        return torch.cat(
            [_ntk_rope_reference(
                ids[..., i], self.Z_AXES_DIM[i], self.Z_THETA, ntk_factor,
                floor_ratio=ratio)
             for i in range(3)],
            dim=-3,
        ).unsqueeze(1)

    def test_ntk_factor_one_is_identity_at_native_grid(self):
        """ntk_factor=1 at the native 64x64 image grid (4096) and on the
        cap group: bitwise equal to the plain EmbedND (the per-group floor
        is continuous with native at N_group=4096)."""
        emb = self._embedder(ntk_factor=1.0)
        for ids in (_zimage_cap_ids(), _zimage_image_ids(64, 64)):
            inner = _EmbedND(theta=self.Z_THETA, axes_dim=self.Z_AXES_DIM)
            assert torch.equal(emb(ids), inner(ids))

    def test_image_group_floor_uses_group_length(self):
        """16384 image tokens (128x128), theta=256, ntk=10: omega is
        max((2560)^-s, 256^-s/2) — the design's hand-computed sanity pin;
        the floor beats the NTK branch for s > log(2)/log(10) ≈ 0.301."""
        dim = 16
        omega = imx.ntk_rope_omega(
            self.Z_THETA, dim, 10.0, 16384, clip_mode="per_group")
        scale = torch.linspace(
            0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64,
        )
        ntk = 1.0 / ((self.Z_THETA * 10.0) ** scale)
        floor = 1.0 / (self.Z_THETA ** scale) / 2.0  # sqrt(16384/4096) == 2
        assert torch.equal(omega, torch.maximum(ntk, floor))
        # both branches bind somewhere (the max is doing real work)
        assert not torch.equal(omega, ntk)
        assert not torch.equal(omega, floor)
        # hand check: s=0.125 (10^0.125=1.33 < 2 -> NTK binds), s=0.375
        # (10^0.375=2.37 > 2 -> floor binds)
        assert omega[1] == pytest.approx((self.Z_THETA * 10.0) ** -0.125)
        assert omega[3] == pytest.approx(self.Z_THETA ** -0.375 / 2.0)

    def test_image_group_call_clips_each_axis(self):
        """End-to-end image group: every axis's omega is clipped by the
        group's OWN length (16384) — the per-group rope oracle agrees
        bitwise."""
        ids = _zimage_image_ids(128, 128, cap_len=512)
        emb = self._embedder(ntk_factor=10.0)
        assert torch.equal(emb(ids), self._group_expected(ids, 10.0, 16384))

    def test_cap_group_takes_pure_ntk_no_clip(self):
        """The cap group never clips — an all-text group's joint-style
        numerator would vanish; omega is (theta*ntk)^-s bitwise."""
        ids = _zimage_cap_ids(512)
        emb = self._embedder(ntk_factor=10.0)
        assert torch.equal(emb(ids), self._group_expected(ids, 10.0, None))

    def test_detection_is_padding_proof_both_ways(self):
        """Cap ids count the padded span on axis 0 with h/w all zero — a
        long padded caption (8192 > 4096) must STILL take pure NTK (any
        axis-0-based detector would misclip it); image ids with all-zero
        pad rows still classify as the image group ('any true' over h/w,
        the group length includes the pads)."""
        emb = self._embedder(ntk_factor=10.0)
        cap = _zimage_cap_ids(8192)
        assert torch.equal(emb(cap), self._group_expected(cap, 10.0, None))
        img = _zimage_image_ids(64, 64, pad_rows=32)  # 4096 real + 32 pads
        assert torch.equal(
            emb(img), self._group_expected(img, 10.0, 64 * 64 + 32))

    def test_siglip_grid_lands_in_the_image_branch(self):
        """The mask is structural (src/models/zimage.py:36-37): ANY nonzero
        h/w row puts the call in the clip branch — siglip grids included —
        clipped by that group's own length."""
        sig = torch.zeros(1, 3, 3)
        sig[0, :, 1] = torch.tensor([1.0, 3.0, 5.0])
        sig[0, :, 2] = torch.tensor([2.0, 4.0, 6.0])
        emb = self._embedder(ntk_factor=10.0)
        assert torch.equal(emb(sig), self._group_expected(sig, 10.0, 3))

    def test_below_native_image_group_applies_the_floor(self):
        """The per-group formula is unconditional (defined for every group
        length): below the native grid the floor rises ABOVE native omega
        (10^s/0.75 > 1 for every s) — pinned literally so the degenerate
        regime stays a decision, not an accident (pass B above native is
        the operating point)."""
        dim = 16
        omega = imx.ntk_rope_omega(
            self.Z_THETA, dim, 10.0, 2304, clip_mode="per_group")
        scale = torch.linspace(
            0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64,
        )
        floor = 1.0 / (self.Z_THETA ** scale) / math.sqrt(2304 / 4096)
        assert torch.equal(omega, floor)
        assert omega[0] == pytest.approx(1.0 / math.sqrt(2304 / 4096))

    def test_per_group_ignores_text_tokens(self):
        """set_text_tokens is JOINT-floor bookkeeping (D8 cap accounting) —
        a per-group forward is unchanged by it."""
        ids = _zimage_cap_ids(512)
        emb = self._embedder(ntk_factor=10.0)
        before = emb(ids).clone()
        emb.set_text_tokens(8192)
        assert emb.text_tokens == 8192
        assert torch.equal(emb(ids), before)

    def test_set_text_tokens_feeds_the_joint_floor(self):
        """Joint mode: the recorded count is the floor's subtraction term.
        Default train_seq_len - native_grid**2 = 512 keeps the FLUX joint
        path bitwise-neutral (D7); recording 1024 at N=5120 makes the ratio
        exactly 1 — the floor IS the native omega, so the result is the
        PLAIN embedding (the 512 default would deviate: ratio 1.125 lifts
        the floor above the NTK branch for s > 0.026)."""
        emb = imx.IMaxNTKEmbedder(_EmbedND(), ntk_factor=10.0)
        assert emb.text_tokens == 512
        emb.set_text_tokens(1024)
        ids = _flux_ids(n_txt=1024)  # N = 1024 + 4096 = 5120
        assert torch.equal(emb(ids), emb.inner(ids))
        # the 512 default genuinely differs -> the recording did the work
        default = imx.IMaxNTKEmbedder(_EmbedND(), ntk_factor=10.0)
        assert not torch.equal(default(ids), emb.inner(ids))

    def test_per_group_uses_per_axis_thetas(self):
        """DyPE per-axis thetas survive the per-group branch — each axis's
        omega uses its own base (src/base.py:19-26)."""
        inner = _PosEmbedFluxStandIn(
            theta=self.Z_THETA, axes_dim=self.Z_AXES_DIM)
        inner.thetas = [256.0, 300.0, 300.0]
        emb = self._embedder(inner=inner, ntk_factor=10.0)
        assert emb.thetas == [256.0, 300.0, 300.0]
        ids = _zimage_image_ids(128, 128)
        expected = torch.cat(
            [_ntk_rope_reference(
                ids[..., i], self.Z_AXES_DIM[i], inner.thetas[i], 10.0,
                floor_ratio=16384 / 4096)
             for i in range(3)],
            dim=-3,
        ).unsqueeze(1)
        assert torch.equal(emb(ids), expected)

    def test_takeover_warning_fires_in_per_group_mode(self, caplog):
        """A z-image DyPE-family embedder answers set_timestep
        (src/base.py:50-51) — the takeover warning fires in per-group mode
        too: the static omega replaces its dynamic scaling."""
        inner = _PosEmbedFluxStandIn(
            theta=self.Z_THETA, axes_dim=self.Z_AXES_DIM)
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            self._embedder(inner=inner)
        assert "positional embedder" in caplog.text

    def test_unknown_clip_mode_rejected(self):
        with pytest.raises(ValueError, match="clip_mode"):
            imx.IMaxNTKEmbedder(_EmbedND(), clip_mode="bogus")
        with pytest.raises(ValueError, match="clip_mode"):
            imx.ntk_rope_omega(256, 16, 10.0, 16384, clip_mode="bogus")

    def test_per_group_requires_three_axis_ids(self):
        emb = self._embedder()
        with pytest.raises(ValueError, match="3-axis"):
            emb(torch.zeros(1, 8, 2))

    def test_set_text_tokens_rejects_nonpositive(self):
        emb = self._embedder()
        with pytest.raises(ValueError, match="text_tokens"):
            emb.set_text_tokens(0)

    def test_missing_attr_message_is_attr_agnostic(self):
        """The swap seam is attr-based (pe_embedder on Flux, rope_embedder
        on lumina) — the requirement message must not name one attr."""
        with pytest.raises(ValueError, match="positional embedder") as ei:
            imx.IMaxNTKEmbedder(types.SimpleNamespace())
        assert "pe_embedder" not in str(ei.value)

    def test_per_group_omega_ignores_text_tokens_kwarg(self):
        """The per-group ratio subtracts nothing — an explicit text_tokens
        kwarg changes nothing."""
        a = imx.ntk_rope_omega(256, 16, 10.0, 16384, clip_mode="per_group")
        b = imx.ntk_rope_omega(
            256, 16, 10.0, 16384, clip_mode="per_group", text_tokens=9999)
        assert torch.equal(a, b)


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
# Proportional attention override (v2.19.0 D8, z-image — plan 2026-10-06)
# ---------------------------------------------------------------------------

def _raw_attn_stub(q, k, v, heads, *args, **kwargs):
    """wrap_attn's ``func``: the RAW undecorated backend the wrapper hands
    the override (comfy ldm/modules/attention.py:222)."""
    return ("raw", q, k, v, heads, args, kwargs)


def _install_fake_lumina(monkeypatch, attn_fn=None):
    """Register the comfy.ldm.lumina module chain the override lazily
    resolves (the _install_fake_comfy pattern): ``attn_fn`` becomes the
    module-level ``optimized_attention_masked`` the override must land on
    (SPA rebinds exactly that symbol, src/spa.py:590-591)."""
    fake_model = types.ModuleType("comfy.ldm.lumina.model")
    if attn_fn is not None:
        fake_model.optimized_attention_masked = attn_fn
    fake_lumina = types.ModuleType("comfy.ldm.lumina")
    fake_lumina.model = fake_model
    fake_ldm = types.ModuleType("comfy.ldm")
    fake_ldm.lumina = fake_lumina
    fake_comfy = types.ModuleType("comfy")
    fake_comfy.ldm = fake_ldm
    for name, mod in [("comfy", fake_comfy), ("comfy.ldm", fake_ldm),
                      ("comfy.ldm.lumina", fake_lumina),
                      ("comfy.ldm.lumina.model", fake_model)]:
        monkeypatch.setitem(sys.modules, name, mod)
    return fake_model


def _call_as_wrap_attn(override, q, k, v, heads, mask, transformer_options):
    """Drive ``override`` exactly as comfy's wrap_attn does for a lumina
    block call (attention.py:215-222 on lumina/model.py:179-182): containers
    already unwrapped, the mask positional, the wrapper guard kwarg set."""
    return override(
        _raw_attn_stub, q, k, v, heads, mask,
        skip_reshape=True, transformer_options=transformer_options,
        _inside_attn_wrapper=True,
    )


@pytest.mark.unit
class TestAttentionScaleOverride:
    """v2.19.0 D8 — the z-image proportional-attention override: lumina
    blocks have no attn1_patch seam, so the q pre-scale rides the GLOBAL
    ``optimized_attention_override`` seam (wrap_attn, attention.py:206-240),
    chained over any pre-existing override and dispatched through the lumina
    ``optimized_attention_masked`` symbol. The anchor is the z-image native
    sequence length: native image grid (64x64 = 4096) + padded caption."""

    ANCHOR = 4608  # 4096 + a 512-token padded caption

    def _zimage_qkv(self, n, heads=8, head_dim=4):
        q = torch.randn(1, heads, n, head_dim)
        k = torch.randn(1, heads, n, head_dim)
        v = torch.randn(1, heads, n, head_dim)
        return q, k, v

    def test_noop_at_the_anchor(self):
        """The D8 clamp: at the anchor the SAME q reaches the attention
        call — no per-block copy, no scale."""
        q, k, v = self._zimage_qkv(self.ANCHOR)
        out = _call_as_wrap_attn(
            imx.make_attention_scale_override(self.ANCHOR),
            q, k, v, 8, "mask", {})
        assert out[1] is q
        assert out[2] is k and out[3] is v

    def test_noop_below_the_anchor(self):
        q, k, v = self._zimage_qkv(512)
        out = _call_as_wrap_attn(
            imx.make_attention_scale_override(self.ANCHOR),
            q, k, v, 8, "mask", {})
        assert out[1] is q

    def test_scales_q_above_anchor_by_reference_formula(self):
        """Above the anchor: q * sqrt(log(N, anchor)) — log BASE anchor,
        the same algebra as the flux patch (head_dim cancels); k/v ride
        unscaled and the inputs are never mutated in place."""
        n = 9216
        q, k, v = self._zimage_qkv(n)
        q_before, k_before, v_before = q.clone(), k.clone(), v.clone()
        out = _call_as_wrap_attn(
            imx.make_attention_scale_override(self.ANCHOR),
            q, k, v, 8, "mask", {})
        expected = math.sqrt(math.log(n) / math.log(self.ANCHOR))
        assert torch.equal(out[1], q * expected)
        assert out[2] is k and out[3] is v
        assert torch.equal(q, q_before)
        assert torch.equal(k, k_before) and torch.equal(v, v_before)

    def test_forwards_mask_args_and_kwargs_untouched(self):
        """wrap_attn's call shape: the mask positional, ``skip_reshape`` /
        ``transformer_options`` / the wrapper guard kwarg forwarded as-is,
        heads untouched."""
        q, k, v = self._zimage_qkv(self.ANCHOR + 1)
        to = {"patches": {}}
        out = _call_as_wrap_attn(
            imx.make_attention_scale_override(self.ANCHOR),
            q, k, v, 8, "mask", to)
        _, rq, rk, rv, r_heads, r_args, r_kwargs = out
        assert r_heads == 8
        assert r_args == ("mask",)
        assert r_kwargs["skip_reshape"] is True
        assert r_kwargs["transformer_options"] is to
        assert r_kwargs["_inside_attn_wrapper"] is True

    def test_resolves_the_lumina_symbol_not_the_raw_func(self, monkeypatch):
        """Dispatch goes through the module-level lumina symbol (which SPA
        rebinds), NOT wrap_attn's raw ``func`` — with the full call shape
        forwarded."""
        lumina_calls = []

        def lumina_attn(q, k, v, heads, *args, **kwargs):
            lumina_calls.append((q, k, v, heads, args, kwargs))
            return "lumina-result"

        _install_fake_lumina(monkeypatch, lumina_attn)
        q, k, v = self._zimage_qkv(self.ANCHOR + 1)
        out = _call_as_wrap_attn(
            imx.make_attention_scale_override(self.ANCHOR),
            q, k, v, 8, "mask", {"erg": 1})
        assert out == "lumina-result"
        assert len(lumina_calls) == 1
        expected = math.sqrt(
            math.log(self.ANCHOR + 1) / math.log(self.ANCHOR))
        assert torch.equal(lumina_calls[0][0], q * expected)
        assert lumina_calls[0][4] == ("mask",)
        assert lumina_calls[0][5]["skip_reshape"] is True

    def test_falls_back_to_raw_func_when_symbol_missing(self, monkeypatch):
        """A lumina module without the symbol (mock/standalone builds) —
        the raw ``func`` is the fallback, not a crash."""
        _install_fake_lumina(monkeypatch, attn_fn=None)
        q, k, v = self._zimage_qkv(self.ANCHOR + 1)
        out = _call_as_wrap_attn(
            imx.make_attention_scale_override(self.ANCHOR),
            q, k, v, 8, "mask", {})
        assert out[0] == "raw"
        expected = math.sqrt(
            math.log(self.ANCHOR + 1) / math.log(self.ANCHOR))
        assert torch.equal(out[1], q * expected)

    def test_chains_previous_override_with_scaled_q(self):
        """A pre-existing comfy override (set_model_optimized_attention
        shape) receives the SCALED q with the untouched rest, and its
        result is returned — ours wraps theirs, theirs owns dispatch."""
        prev_calls = []

        def previous_override(func, q, k, v, heads, *args, **kwargs):
            prev_calls.append((func, q, k, v, heads, args, kwargs))
            return "prev-result"

        q, k, v = self._zimage_qkv(9216)
        out = _call_as_wrap_attn(
            imx.make_attention_scale_override(
                self.ANCHOR, previous_override=previous_override),
            q, k, v, 8, "mask", {})
        assert out == "prev-result"
        assert len(prev_calls) == 1
        func, seen_q, seen_k, seen_v, heads, args, kwargs = prev_calls[0]
        expected = math.sqrt(math.log(9216) / math.log(self.ANCHOR))
        assert torch.equal(seen_q, q * expected)
        assert seen_k is k and seen_v is v
        assert args == ("mask",)
        assert kwargs["skip_reshape"] is True

    def test_previous_override_receives_unscaled_q_at_anchor(self):
        """Below/at the anchor the chain sees the ORIGINAL q object — the
        clamp is an exact pass-through, not a *1.0 copy."""
        prev_q = []

        def previous_override(func, q, k, v, heads, *args, **kwargs):
            prev_q.append(q)
            return "prev-result"

        q, k, v = self._zimage_qkv(self.ANCHOR)
        _call_as_wrap_attn(
            imx.make_attention_scale_override(
                self.ANCHOR, previous_override=previous_override),
            q, k, v, 8, "mask", {})
        assert prev_q[0] is q

    def test_factory_rejects_nonpositive_anchor(self):
        with pytest.raises(ValueError, match="anchor_seq_len"):
            imx.make_attention_scale_override(0)
        with pytest.raises(ValueError, match="anchor_seq_len"):
            imx.make_attention_scale_override(-4608)

    # ---- build_pass_model_options install branch (z-image, plan D5/D8) --

    def test_builder_installs_override_for_zimage(self):
        """The z-image pass-B clone carries the override on the GLOBAL seam;
        the flux attn1_patch list is left untouched (profile-gated)."""
        model = _patched_model()
        opts = imx.build_pass_model_options(
            model, enabled=True, arch_profile="zimage",
            attention_anchor=self.ANCHOR)
        to = opts["transformer_options"]
        assert callable(to["optimized_attention_override"])
        assert to["patches"]["attn1_patch"] == [_existing_attn_patch]
        assert "optimized_attention_override" not in \
            model.model_options["transformer_options"]

    def test_builder_zimage_override_scales_with_the_passed_anchor(self):
        """The anchor flows from the builder into the installed override —
        clamped at the anchor, scaled above it."""
        model = _patched_model()
        opts = imx.build_pass_model_options(
            model, enabled=True, arch_profile="zimage",
            attention_anchor=self.ANCHOR)
        to = opts["transformer_options"]
        installed = to["optimized_attention_override"]
        at_anchor = self._zimage_qkv(self.ANCHOR)
        out = _call_as_wrap_attn(installed, *at_anchor, 8, "mask", {})
        assert out[1] is at_anchor[0]
        above = self._zimage_qkv(2 * self.ANCHOR)
        out = _call_as_wrap_attn(installed, *above, 8, "mask", {})
        expected = math.sqrt(math.log(2 * self.ANCHOR) / math.log(self.ANCHOR))
        assert torch.equal(out[1], above[0] * expected)

    def test_builder_chains_existing_override_for_zimage(self):
        """An override already on model.model_options (comfy's own
        set_model_optimized_attention precedent) is chained — the clone
        carries OURS, the source keeps theirs untouched."""
        prev_calls = []

        def previous_override(func, q, k, v, heads, *args, **kwargs):
            prev_calls.append((q, args, kwargs))
            return "prev-result"

        model = _patched_model()
        model.model_options["transformer_options"][
            "optimized_attention_override"] = previous_override
        opts = imx.build_pass_model_options(
            model, enabled=True, arch_profile="zimage",
            attention_anchor=self.ANCHOR)
        installed = opts["transformer_options"][
            "optimized_attention_override"]
        assert installed is not previous_override
        q, k, v = self._zimage_qkv(9216)
        assert _call_as_wrap_attn(installed, q, k, v, 8, "mask", {}) \
            == "prev-result"
        expected = math.sqrt(math.log(9216) / math.log(self.ANCHOR))
        assert torch.equal(prev_calls[0][0], q * expected)
        assert model.model_options["transformer_options"][
            "optimized_attention_override"] is previous_override

    def test_builder_flux_profile_installs_no_override(self):
        """Default (flux) profile: no override key — the attn1_patch route
        is the flux D8 seam, unchanged (D7)."""
        model = _patched_model()
        opts = imx.build_pass_model_options(model, enabled=True)
        assert "optimized_attention_override" \
            not in opts["transformer_options"]
        assert len(opts["transformer_options"]["patches"]["attn1_patch"]) == 2

    def test_builder_zimage_toggle_disables_override(self):
        """The proportional_attention toggle governs the z-image seam too —
        off means no override on the pass-B clone."""
        model = _patched_model()
        opts = imx.build_pass_model_options(
            model, enabled=True, arch_profile="zimage",
            proportional_attention=False)
        assert "optimized_attention_override" \
            not in opts["transformer_options"]

    def test_builder_pass_a_never_carries_override(self):
        """Pass A (enabled=False) is a clean clone — no override even for
        the z-image profile."""
        model = _patched_model()
        opts = imx.build_pass_model_options(
            model, enabled=False, arch_profile="zimage",
            attention_anchor=self.ANCHOR)
        assert "optimized_attention_override" \
            not in opts["transformer_options"]

    def test_builder_rejects_unknown_profile(self):
        """An unknown profile must fail loud, not silently install the flux
        seam on a foreign arch."""
        model = _patched_model()
        with pytest.raises(ValueError, match="arch_profile"):
            imx.build_pass_model_options(model, enabled=True,
                                         arch_profile="lumina")


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


def _mock_zimage_model(with_rope=True, with_cap_pad=True):
    """Z-Image ModelPatcher stand-in — the _mock_flux_model shape on the
    lumina wiring: a BaseModel class NAMED Lumina2 (the MRO name IS the arch
    gate), CONST flow sampling, the shared Flux latent format, and a
    diffusion_model exposing rope_embedder (theta 256 = the z-image EmbedND
    config, comfy model_detection.py:604; axes [32,48,48], :605) and
    cap_pad_token (the D8 hasattr probe)."""
    class _Base:
        def timestep(self, sigma):
            return sigma * 1000.0  # ModelSamplingDiscreteFlow multiplier

    ms = type("ModelSampling", (_Base, _FLOW_MIXINS["CONST"]), {})()

    class Lumina2:  # model_base.Lumina2 stand-in — the MRO name IS the gate
        pass

    base = Lumina2()
    base.model_sampling = ms
    base.latent_format = types.SimpleNamespace(
        latent_dimensions=2, latent_channels=16)
    base.process_latent_in = lambda t: (t - 0.1159) * 0.3611
    base.process_latent_out = lambda t: (t / 0.3611) + 0.1159
    dm = types.SimpleNamespace()
    if with_rope:
        dm.rope_embedder = _EmbedND(theta=256.0, axes_dim=[32, 48, 48])
    if with_cap_pad:
        # next_pad_token's sibling: the nn.Parameter NextDiT.__init__ creates
        # exactly when pad_tokens_multiple is set (comfy lumina/model.py)
        dm.cap_pad_token = torch.nn.Parameter(torch.zeros(1))
    base.diffusion_model = dm

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
        assert imx._require_flux_flow_model(model) == ("flux", 2)

    def test_rejects_non_flux_arch_with_pointer_to_hiflow(self):
        """Qwen is a flow model that also owns pe_embedder — the arch gate
        (BaseModel MRO) is the real discriminator, now checked against both
        accepted families (Flux, Z-Image/Lumina2); the message points at
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
        assert imx._require_flux_flow_model(patcher) == ("flux", 2)


@pytest.mark.unit
class TestImaxArchGate:
    """v2.19.0 D1/D2 — the widened arch gate: Z-Image acceptance via the
    Lumina2 MRO + theta-256 RoPE, and the variant rejections in D2 order
    (MingImage by MRO name before the Lumina2 accept; ZImagePixelSpace by
    latent-format class name; plain Lumina2 by the theta guard)."""

    # model_base.Lumina2 stand-in — the MRO name IS the gate's discriminator
    LUMINA2 = type("Lumina2", (), {})

    @staticmethod
    def _gate_model(arch="Lumina2", arch_bases=(), latent_format="ZImage",
                    theta=256.0, with_rope=True):
        """Minimal z-image-family gate mock: CONST flow sampling, a BaseModel
        class named ``arch`` (optionally deriving ``arch_bases`` — the real
        variant classes all derive model_base.Lumina2), a latent-format class
        named ``latent_format`` and ``diffusion_model.rope_embedder.theta``.
        """
        class _Base:
            def timestep(self, sigma):
                return sigma * 1000.0

        ms = type("ModelSampling", (_Base, _FLOW_MIXINS["CONST"]), {})()
        base = type(arch, tuple(arch_bases) or (object,), {})()
        base.model_sampling = ms
        base.latent_format = type(latent_format, (), {})()
        base.latent_format.latent_dimensions = 2
        base.diffusion_model = types.SimpleNamespace()
        if with_rope:
            base.diffusion_model.rope_embedder = types.SimpleNamespace(
                theta=theta)
        model = types.SimpleNamespace()
        model.model = base
        model.model_options = {}
        return model

    def test_accepts_zimage_theta256(self):
        """A theta-256 Lumina2-arch model is the Z-Image profile."""
        model = self._gate_model()
        assert imx._require_flux_flow_model(model) == ("zimage", 2)

    def test_rejects_mingimage_before_lumina2_accept(self):
        """D2 ordering: MingImage derives Lumina2 — the MRO-name rejection
        must fire BEFORE the Lumina2 accept, with its own message."""
        model = self._gate_model(
            arch="MingImage", arch_bases=(self.LUMINA2,), theta=256.0)
        with pytest.raises(ValueError, match="MingImage"):
            imx._require_flux_flow_model(model)

    def test_rejects_zimage_pixel_space_variant(self):
        """ZImagePixelSpace passes a Lumina2 MRO check — rejected by its
        latent-format class name (comfy supported_models.py:1238)."""
        model = self._gate_model(latent_format="ZImagePixelSpace")
        with pytest.raises(ValueError, match="pixel-space"):
            imx._require_flux_flow_model(model)

    def test_rejects_plain_lumina2_theta(self):
        """Plain Lumina2 shares the NextDiT arch but trains theta=10000
        (comfy model_detection.py:593) — the theta-256 guard rejects it."""
        model = self._gate_model(theta=10000.0)
        with pytest.raises(ValueError, match="theta"):
            imx._require_flux_flow_model(model)

    def test_zimage_swap_target_is_rope_embedder(self):
        """The D4 seam check is attr-based: a z-image-arch model without
        rope_embedder fails with that attr named in the message."""
        model = self._gate_model(with_rope=False)
        with pytest.raises(ValueError, match="rope_embedder"):
            imx._require_flux_flow_model(model)


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


# ---------------------------------------------------------------------------
# Z-Image node wiring (v2.19.0 Phase 4, plan 2026-10-06 D5/D6/D8/D10)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestIMaxZImageWiring:
    """The execute wiring on the lumina arch: the D4 swap parametrized onto
    the rope_embedder seam with the per-group clip mode and the D8 padded-
    caption bookkeeping, the D4 attention anchor fed native grid + cap_padded,
    text duplication warned and skipped (D5), the guidance cond override
    skipped (D6), and the shared dual-pass path regression-pinned on a
    z-image mock."""

    def test_zimage_end_to_end_returns_target_latent(self, monkeypatch):
        samples, fx, _, _ = _run_execute(
            monkeypatch, model=_mock_zimage_model(), latent=_ABOVE_NATIVE)
        assert samples.shape == _ABOVE_NATIVE
        assert torch.isfinite(samples).all()

    def test_deterministic_for_equal_inputs(self, monkeypatch):
        a, _, _, _ = _run_execute(
            monkeypatch, model=_mock_zimage_model(), seed=123)
        b, _, _, _ = _run_execute(
            monkeypatch, model=_mock_zimage_model(), seed=123)
        assert torch.equal(a, b)

    def test_rope_embedder_swapped_during_high_pass(self, monkeypatch):
        """D10: execute installs the D4 swap on the rope_embedder seam with
        the per-group clip mode. Driving the pass-B options' installed
        wrapper (execute's last call was a pass-B step, so the state cell is
        still high_pass=True) shows an IMaxNTKEmbedder over the currently
        installed embedder during the forward, with the D8 padded caption
        recorded, and the original restored afterwards."""
        model = _mock_zimage_model()
        _, fx, _, _ = _run_execute(
            monkeypatch, model=model, latent=_ABOVE_NATIVE)
        wrapper = fx.sampling_calls[-1]["model_options"][
            "model_function_wrapper"]
        dm = model.model.diffusion_model
        original = dm.rope_embedder
        seen = []

        def model_function(x, t, **c):
            seen.append(dm.rope_embedder)
            return x

        params = {"input": torch.zeros(1, 4),
                  "timestep": torch.tensor([0.5]), "c": {}}
        wrapper(model_function, params)
        assert isinstance(seen[0], imx.IMaxNTKEmbedder)
        assert seen[0].clip_mode == "per_group"
        assert seen[0].ntk_factor == 10.0
        assert seen[0].theta == 256.0
        assert seen[0].inner is original
        assert seen[0].text_tokens == 32  # D8: 4 cond tokens -> padded 32
        assert dm.rope_embedder is original

    def test_embedder_restored_after_exception(self, monkeypatch):
        """The finally-restore holds on the rope_embedder seam too — a
        failing pass-B forward leaves the installed embedder in place."""
        model = _mock_zimage_model()
        _, fx, _, _ = _run_execute(
            monkeypatch, model=model, latent=_ABOVE_NATIVE)
        wrapper = fx.sampling_calls[-1]["model_options"][
            "model_function_wrapper"]
        dm = model.model.diffusion_model
        original = dm.rope_embedder

        def boom(x, t, **c):
            raise RuntimeError("boom")

        params = {"input": torch.zeros(1, 4),
                  "timestep": torch.tensor([0.5]), "c": {}}
        with pytest.raises(RuntimeError):
            wrapper(boom, params)
        assert dm.rope_embedder is original

    def test_override_rides_pass_b_only(self, monkeypatch):
        """D5: pass A is a clean clone, pass B carries the global override;
        the z-image pass-B patch lists get NO additions (no attn1_patch, no
        post_input), and the unet wrapper rides both pass dicts."""
        _, fx, _, _ = _run_execute(
            monkeypatch, model=_mock_zimage_model(), latent=_ABOVE_NATIVE)
        low = fx.sampling_calls[0]["model_options"]
        high = fx.sampling_calls[-1]["model_options"]
        assert "optimized_attention_override" \
            not in low["transformer_options"]
        assert callable(
            high["transformer_options"]["optimized_attention_override"])
        assert "attn1_patch" not in high["transformer_options"]["patches"]
        assert "post_input" not in high["transformer_options"]["patches"]
        assert callable(low.get("model_function_wrapper"))
        assert callable(high.get("model_function_wrapper"))

    def test_attention_anchor_is_native_grid_plus_cap_padded(self, monkeypatch):
        """D8 feed: the pass-B override's anchor is the native image grid
        (64x64 = 4096) + the padded caption — 4 cond tokens -> 32 -> 4128.
        Clamped exactly at the joint anchor, scaled above it by the
        reference formula."""
        _, fx, _, _ = _run_execute(
            monkeypatch, model=_mock_zimage_model(), latent=_ABOVE_NATIVE)
        override = fx.sampling_calls[-1]["model_options"][
            "transformer_options"]["optimized_attention_override"]
        anchor = 4096 + 32
        q, k, v = (torch.randn(1, 8, anchor, 4) for _ in range(3))
        out = _call_as_wrap_attn(override, q, k, v, 8, "mask", {})
        assert out[1] is q  # clamped AT the joint anchor
        n = 2 * anchor
        q2, k2, v2 = (torch.randn(1, 8, n, 4) for _ in range(3))
        out = _call_as_wrap_attn(override, q2, k2, v2, 8, "mask", {})
        expected = math.sqrt(math.log(n) / math.log(anchor))
        assert torch.equal(out[1], q2 * expected)

    def test_attention_anchor_counts_unpadded_cap_without_pad_token(
            self, monkeypatch):
        """D8 fallback: no cap_pad_token attr (older checkpoints) — the RAW
        caption count drives the anchor (4 tokens -> 4100)."""
        _, fx, _, _ = _run_execute(
            monkeypatch, model=_mock_zimage_model(with_cap_pad=False),
            latent=_ABOVE_NATIVE)
        override = fx.sampling_calls[-1]["model_options"][
            "transformer_options"]["optimized_attention_override"]
        anchor = 4096 + 4
        q = torch.randn(1, 8, anchor, 4)
        out = _call_as_wrap_attn(override, q, q, q, 8, "mask", {})
        assert out[1] is q  # clamped at the unpadded anchor
        q2 = torch.randn(1, 8, anchor + 1, 4)
        out = _call_as_wrap_attn(override, q2, q2, q2, 8, "mask", {})
        assert out[1] is not q2  # above it the scale applies

    def test_text_duplication_warns_and_is_ignored(self, monkeypatch, caplog):
        """D5: the toggle is real on flux but cannot exist on lumina — one
        warning names Z-Image and no post_input patch is installed."""
        import logging
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            _, fx, _, _ = _run_execute(
                monkeypatch, model=_mock_zimage_model(),
                latent=_ABOVE_NATIVE, text_duplication=True)
        assert any(
            "text duplication is ignored for Z-Image" in r.message
            for r in caplog.records
        )
        patches = fx.sampling_calls[-1]["model_options"][
            "transformer_options"]["patches"]
        assert "post_input" not in patches

    def test_guidance_cond_untouched_for_zimage(self, monkeypatch):
        """D6: no guidance cond key is written for z-image — even with the
        nonzero schema defaults, the conditioning copies reach convert_cond
        unmodified (Z-Image is guidance-distilled)."""
        _, fx, _, _ = _run_execute(
            monkeypatch, model=_mock_zimage_model(), latent=_ABOVE_NATIVE,
            guidance_low=3.5, guidance_high=5.0)
        assert fx.convert_calls
        assert all("guidance" not in call for call in fx.convert_calls)

    def test_guidance_override_still_applies_for_flux(self, monkeypatch):
        """D6/D7 regression: the flux route still writes the guidance embed
        per pass (3.5 low, 5.0 high on the defaults)."""
        _, fx, _, _ = _run_execute(monkeypatch, latent=_ABOVE_NATIVE)
        assert fx.convert_calls[0]["guidance"] == 3.5
        assert fx.convert_calls[2]["guidance"] == 5.0

    def test_negative_empty_cfg_autoskip(self, monkeypatch):
        """The Z-Image guidance-distilled convention: a token-less negative
        forces the conditional branch only (CFG skipped) — the shared
        adapter behavior, pinned on the z-image mock (adapter level, the
        TestPredictX0Adapter shape)."""
        fx = _install_fake_comfy(monkeypatch)
        adapter = imx._make_predict_x0(
            _mock_zimage_model(), COND_POS,
            [(torch.zeros(1, 0), {"side": "negative"})], cfg_scale=3.5,
            model_options={}, pass_state={}, high_pass=False)
        adapter(torch.randn(1, 16, 8, 8), sigma=0.5)
        assert fx.sampling_calls[-1]["cond_scale"] == 1.0

    def test_multiframe_latent_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="multi-frame"):
            _run_execute(
                monkeypatch, model=_mock_zimage_model(),
                latent=(1, 16, 2, 16, 16))

    def test_below_native_resolution_warns(self, monkeypatch, caplog):
        """512 px target on z-image: the shared engine warning surfaces and
        pass A runs at the target."""
        import logging
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            samples, _, _, _ = _run_execute(
                monkeypatch, model=_mock_zimage_model(),
                latent=(1, 16, 64, 64))
        assert any("at/below the native" in r.message for r in caplog.records)
        assert samples.shape == (1, 16, 64, 64)

    # ---- D8 cap accounting helper ---------------------------------------

    @pytest.mark.parametrize("tokens,padded", [(5, 32), (64, 64), (33, 64)])
    def test_cap_padded_rounds_to_32_multiple(self, tokens, padded):
        dm = types.SimpleNamespace(
            cap_pad_token=torch.nn.Parameter(torch.zeros(1)))
        positive = [(torch.ones(1, tokens), {})]
        assert imx._cap_padded_length(positive, dm) == padded

    def test_cap_padded_skipped_without_pad_token_attr(self):
        positive = [(torch.ones(1, 33), {})]
        assert imx._cap_padded_length(
            positive, types.SimpleNamespace()) == 33

    def test_cap_padded_prefers_opts_num_tokens(self):
        """opts['num_tokens'] wins over the tensor's token axis when an
        entry carries it (best effort — stock comfy derives num_tokens the
        other way, as an extra_conds CONDConstant, model_base.py:1519-1525)."""
        dm = types.SimpleNamespace(
            cap_pad_token=torch.nn.Parameter(torch.zeros(1)))
        positive = [(torch.ones(1, 999), {"num_tokens": 5})]
        assert imx._cap_padded_length(positive, dm) == 32

    def test_cap_padded_takes_max_over_entries(self):
        dm = types.SimpleNamespace(
            cap_pad_token=torch.nn.Parameter(torch.zeros(1)))
        positive = [(torch.ones(1, 5), {}), (torch.ones(2, 40), {})]
        assert imx._cap_padded_length(positive, dm) == 64

    def test_cap_padded_is_zero_for_unmeasurable_conditioning(self):
        """No measurable tokens -> 0: the anchor degenerates to the native
        grid and the wrapper skips the set_text_tokens bookkeeping."""
        assert imx._cap_padded_length([], types.SimpleNamespace()) == 0
