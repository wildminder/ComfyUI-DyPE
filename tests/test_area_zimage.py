"""Z-Image support — arch profile and node inputs (plan 2026-10-16, Phase 1).

``docs/2026-10-16-area-zimage.md`` §4/§7-t2.  Everything here is mock-based and
GPU-free, per project convention.  What this file pins:

* the arch resolver (``src/area_arch.py``) — MRO-first: a plain Lumina2 / NextDiT
  model is the Z-Image profile, its Pixel-Space and Ming-Image siblings refuse
  with an actionable message, and a requested ``model_type`` that contradicts
  the model's own backbone refuses too;
* the two profiles' geometry — the shared 16 px patch grid (ceil div) and
  Z-Image's 32-token sequence padding (FLUX pads to nothing);
* the Phase I switch — ``sigma_prefix_count`` counts the leading steps still at
  or above ``sigma_star`` (theory.md:240-246), and at shift 3.0 that count is
  the reference's ``phase1_steps=8``;
* the node schema — ``model_type`` offers ``zimage`` and the ``phase1_mode`` /
  ``phase1_sigma`` switch is present with the reference's defaults, with
  ``validate_inputs`` (multiple-of-16) untouched.

Phase 2 (the z-image adapter tensors, ``src/area_zimage.py``) is appended under
the ``# --- Phase 2`` marker below; the sections before it are Phase 1.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_zimage.py tests/test_area_node.py
"""
from __future__ import annotations

import pytest
import torch
from nodes.area import AReA

import src.area as area
from src.area import PHASE1_STEPS
from src.area_arch import (
    FLUX_PROFILE,
    ZIMAGE_PROFILE,
    _ArchProfile,
    resolve_area_arch,
)
from src.area_attn import apply_rope1, pool_keys_values
from src.area_zimage import (
    ZIMAGE_AXES,
    ZIMAGE_SIGMA_STAR,
    ZImageRopeRecorder,
    coarse_ropes_zimage,
    sigma_prefix_count,
    stitch_fine_zeros,
    unrotate1,
    unrotate_table,
    zimage_layout,
)

# ---------------------------------------------------------------------------
# Mocks (comfy-faithful, GPU-free)
# ---------------------------------------------------------------------------


class NextDiT(torch.nn.Module):
    """A stock plain Z-Image diffusion model.

    Named after the real backbone on purpose: ``resolve_area_arch`` identifies
    the backbone from the model's MRO class names, and comfy names the class
    this way (``comfy/ldm/lumina/model.py:442``, bound as ``Lumina2``'s
    ``unet_model`` at ``comfy/model_base.py:1510``) — the same class-name
    discipline ``src/model_detect.py:48-50`` already relies on.

    The attributes an AReA apply actually touches, per
    ``comfy/ldm/lumina/model.py:442-500``: ``patch_size`` 2 over 16-channel
    Flux-format latents (``comfy/supported_models.py:1190``,
    ``model_detection.py:582-583``), the ``rope_embedder`` the
    ``src/model_detect`` auto probe keys on (``src/model_detect.py:74-75``),
    the 32-token ``pad_tokens_multiple`` (``model.py:484,665,725-727``), and no
    ``masked_pad_multiple`` — that one is the Ming-Image marker.

    ``pixel`` / ``ming`` build the two refused siblings.
    """

    def __init__(self, pixel: bool = False, ming: bool = False):
        super().__init__()
        self.patch_size = 2
        self.pad_tokens_multiple = 32
        self.masked_pad_multiple = 32 if ming else None
        self.rope_embedder = torch.nn.Linear(4, 4)
        if ming:
            # model.py:486 — NextDiT registers this buffer for ming_image.
            self.register_buffer("__ming_image__", torch.empty(0))
        if pixel:
            # model.py:1005-1053 — the pixel path deletes final_layer and
            # installs a dec_net decoder head.
            self.dec_net = torch.nn.Linear(4, 4)
        else:
            self.final_layer = torch.nn.Linear(4, 4)


class NextDiTPixelSpace(NextDiT):
    """``comfy/ldm/lumina/model.py:1005`` — the pixel-space sibling."""

    def __init__(self):
        super().__init__(pixel=True)


class MockFluxTransformer(torch.nn.Module):
    """A stock FLUX diffusion model: the ``pe_embedder`` probe
    (``src/model_detect.py:78-79``) is what identifies it.  The class name is
    deliberately generic so it is not claimed by a class-name probe."""

    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.pe_embedder = torch.nn.Linear(4, 4)
        self.img_in = torch.nn.Linear(4, 4)


def _flow_sigmas(steps: int, shift: float) -> torch.Tensor:
    """``ModelSamplingDiscreteFlow``'s table in denoising order.

    ``comfy/model_sampling.py:289-292`` (``time_snr_shift``) applied to
    ``t = (1..steps)/steps`` as ``:308-312`` builds it, then reversed so
    sigma_max leads — the order ``transformer_options["sample_sigmas"]``
    carries (``comfy/samplers.py:1229``), and the order ``_area_locate`` reads.
    """
    times = torch.arange(1, steps + 1, dtype=torch.float32) / steps
    sigmas = (shift * times) / (1.0 + (shift - 1.0) * times)
    return sigmas.flip(0)


# ---------------------------------------------------------------------------
# The arch resolver (Q1/Q7)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArchProfileResolution:
    def test_resolve_arch_plain_lumina2_returns_zimage_profile(self):
        profile = resolve_area_arch(NextDiT(), "auto")
        assert profile is ZIMAGE_PROFILE          # one canonical profile each
        assert profile.model_type == "zimage"      # the resolve_model_type key
        assert profile.unet_backbone == "NextDiT"  # the class the MRO found
        assert profile.px_per_patch == 16
        # the profile's pad multiple IS the model's own knob (model.py:484)
        assert NextDiT().pad_tokens_multiple == 32
        assert profile.seq_pad_multiple == 32
        assert "Lumina2" in profile.accepts
        # ``src/model_detect.py:62-63`` normalizes both pin spellings and the
        # profile agrees with each — an explicit pin is not a mismatch
        assert resolve_area_arch(NextDiT(), "zimage") is ZIMAGE_PROFILE
        assert resolve_area_arch(NextDiT(), "z_image") is ZIMAGE_PROFILE

    def test_resolve_arch_rejects_zimage_pixel_space_latent_input(self):
        with pytest.raises(ValueError, match="Pixel Space") as excinfo:
            resolve_area_arch(NextDiTPixelSpace(), "auto")
        # the message names the remedy, not just the offender (house rule)
        assert "latent-space Z-Image" in str(excinfo.value)

    def test_resolve_arch_rejects_ming_image_prompt_masking(self):
        with pytest.raises(ValueError, match="Ming-Image") as excinfo:
            resolve_area_arch(NextDiT(ming=True), "auto")
        assert "masked_pad_multiple" in str(excinfo.value)

    def test_resolve_arch_rejects_mismatched_pinned_model_type(self):
        # a FLUX pin on a Lumina2 model: the backbone and the request disagree
        with pytest.raises(ValueError, match="Lumina2 \\(NextDiT\\)"):
            resolve_area_arch(NextDiT(), "flux")
        # ... and the mirror case, a zimage pin on a FLUX model
        with pytest.raises(ValueError, match="not a NextDiT"):
            resolve_area_arch(MockFluxTransformer(), "zimage")

    def test_resolve_arch_flux_profile_unchanged(self):
        """The FLUX path is byte-identical: today's constants, today's probe."""
        for requested in ("auto", "flux"):
            profile = resolve_area_arch(MockFluxTransformer(), requested)
            assert profile is FLUX_PROFILE
            assert profile.model_type == "flux"
            assert profile.unet_backbone == "Flux"
            assert profile.px_per_patch == 16
            assert profile.seq_pad_multiple == 0     # nothing is padded
            assert profile.accepts == "FLUX.1 (Flux) diffusion model"
            assert isinstance(profile, _ArchProfile)


# ---------------------------------------------------------------------------
# Geometry (shared grid, z-image padding)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestArchGeometry:
    def test_zimage_grid_from_pixels_is_ceil_divide_16(self):
        # the VAE downscale (8) x the patchify (2) is 16 px per token on both
        # backbones, so one node geometry serves either (src/area.py:109-112)
        assert FLUX_PROFILE.px_per_patch == ZIMAGE_PROFILE.px_per_patch == 16
        assert ZIMAGE_PROFILE.patch_count(2048) == 128
        assert FLUX_PROFILE.patch_count(2048) == 128
        # a fraction rounds UP, as the patchifier pads a short latent
        # (src/area.py:206-215) — never down
        assert ZIMAGE_PROFILE.patch_count(2072) == 130
        assert ZIMAGE_PROFILE.patch_count(2064) == 129

    def test_zimage_sequence_padded_to_multiple_of_32(self):
        # cap AND image tokens are padded to 32 (comfy/ldm/lumina/model.py:
        # 418-420, applied at :665 and :725-727)
        assert ZIMAGE_PROFILE.padded_seq_len(4) == 32
        assert ZIMAGE_PROFILE.padded_seq_len(32) == 32
        assert ZIMAGE_PROFILE.padded_seq_len(33) == 64
        assert ZIMAGE_PROFILE.padded_seq_len(0) == 0
        # FLUX pads to nothing — the same call is the identity there
        assert FLUX_PROFILE.padded_seq_len(33) == 33


# ---------------------------------------------------------------------------
# The Phase I switch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhase1Switch:
    def test_phase1_sigma_prefix_count_shift3(self):
        """At the Z-Image shift of 3.0 the sigma rule reproduces the count."""
        table = _flow_sigmas(28, shift=3.0)
        assert table[0] == 1.0                      # sigma_max leads
        count = sigma_prefix_count(table, ZIMAGE_SIGMA_STAR)
        # 8 leading steps at or above 0.89 == the reference's FLUX constant
        # (PHASE1_STEPS), i.e. sigma mode agrees with the default step mode
        assert count == 8 == PHASE1_STEPS
        # and the boundary is where the paper's switch sits: the 8th entry is
        # still in Phase I (sigma ~0.9), the 9th is not (~0.88)
        assert table[count - 1] >= ZIMAGE_SIGMA_STAR
        assert table[count] < ZIMAGE_SIGMA_STAR
        # it is a leading-RUN count: the degenerate thresholds bound it (every
        # step, only sigma_max) and it stops at the first sub-threshold step
        # rather than skipping past it
        assert sigma_prefix_count(table, 0.0) == len(table)
        assert sigma_prefix_count(table, 1.0) == 1
        assert sigma_prefix_count([], 0.89) == 0
        assert sigma_prefix_count([1.0, 0.2, 0.95], 0.89) == 1
        assert sigma_prefix_count((0.5, 0.6), 0.4) == 2


# ---------------------------------------------------------------------------
# The node schema (Q9)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestNodeSchema:
    def test_node_model_type_combo_offers_zimage(self):
        inputs = {i.name: i for i in AReA.define_schema().inputs}
        assert inputs["model_type"].kwargs["options"] == ["auto", "flux", "zimage"]
        assert inputs["model_type"].kwargs["default"] == "auto"

    def test_node_phase1_mode_input_present_with_steps_default(self):
        inputs = {i.name: i for i in AReA.define_schema().inputs}
        mode = inputs["phase1_mode"]
        assert mode.kwargs["options"] == ["steps", "sigma"]
        assert mode.kwargs["default"] == "steps"          # the reference's mode
        # 0.89 = the paper's first-backbone value (theory.md:494), one
        # definition shared with the engine-facing constant
        assert inputs["phase1_sigma"].kwargs["default"] == ZIMAGE_SIGMA_STAR == 0.89
        assert inputs["phase1_steps"].kwargs["default"] == PHASE1_STEPS
        # the multiple-of-16 validation is untouched by the new inputs
        # (VAE 8 x patchify 2 — both backbones share it)
        assert AReA.validate_inputs(1280, 1280) is True
        assert "multiples of 16" in AReA.validate_inputs(1208, 1280)


# --- Phase 2 -----------------------------------------------------------------
# The z-image adapter tensors (``src/area_zimage.py``): un-rotating the seam's
# already-roped q/k, rebuilding the Phase I coarse tables from lumina's own ids,
# capturing the model's per-group rope tables, and the 32-token pad policy.

# ---------------------------------------------------------------------------
# Mocks (comfy-faithful, GPU-free)
# ---------------------------------------------------------------------------


def _mock_rope(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """The real ``rope`` (``comfy/ldm/flux/math.py:20-32``) — no comfy import.

    comfy computes the table in float64 (falling back to CPU only when the
    device lacks fp64, which a CPU test device supports) and returns float32;
    the ``(i j) -> i j`` rearrange is the row-major reshape of the stacked
    ``[cos, -sin, sin, cos]`` vector, so the block is ``[[cos, -sin], [sin, cos]]``.
    """
    assert dim % 2 == 0
    device = pos.device
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=device)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32, device=device), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).to(dtype=torch.float32, device=device)


class MockEmbedND(torch.nn.Module):
    """The real ``EmbedND`` (``comfy/ldm/flux/layers.py:15-29``).

    ``ids`` ``[b, n, 3]`` -> ``[b, 1, n, head_dim // 2, 2, 2]``: one rope per
    axis (``axes_dim``) concatenated on the frequency axis, then the head dim
    unsqueezed.  ``theta``/``axes_dim`` are constructor arguments on purpose —
    the adapter must read them off the installed embedder, never assume them.
    """

    def __init__(self, theta: float = 10000.0, axes_dim=(16, 56, 56)):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    @property
    def head_dim(self) -> int:
        return sum(self.axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat(
            [_mock_rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(ids.shape[-1])],
            dim=-3,
        )
        return emb.unsqueeze(1)


#: A canvas that exercises the pad policy: 20 prompt tokens pad to a 32-token
#: caption block, and a 6x8 (48-token) image pads to 64 — so both pad spans are
#: non-empty and neither is a multiple of the other.  The grid is 2-divisible so
#: the pooling factor really tiles it (the ramp only ever supplies grid
#: divisors, ``src/area_schedule.py:173-175``).
TXT_LEN, H, W, F = 20, 6, 8, 2
CAP_LEN, IMG_TOKENS, IMG_PAD = 32, H * W, 64 - H * W
SEQ_LEN = CAP_LEN + IMG_TOKENS + IMG_PAD          # 96

#: The one t-axis value every image token of the frame carries
#: (``pos_ids_x(start_t=cap_len + 1 + f, ...)``, comfy/ldm/lumina/model.py:722).
IMG_T = float(CAP_LEN + 1)


def _filled_stream(b: int, heads: int, dim: int) -> torch.Tensor:
    """A joint stream whose every token carries its own stream index.

    ``x[..., i, :] == i`` bounds every token, so a slice can be checked for both
    content and single-use: a pad token that leaked into the image window, or an
    image token that vanished, is visible as a wrong index rather than a wrong
    number.
    """
    idx = torch.arange(SEQ_LEN, dtype=torch.float32).view(1, 1, SEQ_LEN, 1)
    return idx.expand(b, heads, SEQ_LEN, dim).contiguous()


# ---------------------------------------------------------------------------
# Un-rotating the seam's tensors
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestUnrotate:
    def test_unrotate_table_inverts_comfy_rope(self):
        """Bit-level: unrotate1 undoes comfy's rope, and the block form is as claimed."""
        embedder = MockEmbedND()                      # theta 10000, axes (16,56,56)
        n, heads, dim = 13, 3, embedder.head_dim
        # (t, row, col) per token, all distinct so every frequency is exercised
        ids = torch.zeros(1, n, ZIMAGE_AXES)
        ids[0, :, 0] = torch.arange(1, n + 1, dtype=torch.float32)
        ids[0, :, 1] = torch.arange(n, dtype=torch.float32) % 5
        ids[0, :, 2] = torch.arange(n, dtype=torch.float32).flip(0)
        table = embedder(ids)

        # the table really is [[cos, -sin], [sin, cos]] (math.rope, math.py:30)
        assert torch.allclose(table[..., 0, 0], table[..., 1, 1])
        assert torch.allclose(table[..., 0, 1], -table[..., 1, 0])

        torch.manual_seed(0)
        x = torch.randn(1, heads, n, dim, dtype=torch.float64)
        table64 = table.double()
        rotated = apply_rope1(x, table64)
        # NOT vacuous: a nonzero position really moves the tensor
        assert not torch.allclose(rotated, x)
        # unrotate1 is the exact inverse of the engine's rotation (1e-6 on fp64)
        assert torch.allclose(unrotate1(rotated, table64), x, atol=1e-6, rtol=0.0)
        # ... and unrotating a rotated tensor restores the un-rotated one
        assert torch.allclose(unrotate1(x, table64), apply_rope1(x, unrotate_table(table64)),
                              atol=1e-6, rtol=0.0)

        # the inverse block is the transpose of a rotation matrix, block by block
        inv = unrotate_table(table64)
        assert torch.allclose(inv, table64.transpose(-1, -2))
        eye = inv @ table64
        assert torch.allclose(eye, torch.eye(2, dtype=torch.float64).expand_as(eye).clone(),
                              atol=1e-6, rtol=0.0)
        # the caller's table is never mutated (the seam re-uses it for the q side)
        untouched = table64.clone()
        unrotate_table(table64)
        assert torch.allclose(table64, untouched)


# ---------------------------------------------------------------------------
# The coarse rope tables
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCoarseRopes:
    def test_coarse_rope_builder_q_and_kv_grids(self):
        """The two Phase I tables: q on the full stream at BLOCK positions, kv on the coarse grid."""
        embedder = MockEmbedND()
        layout = zimage_layout(TXT_LEN, CAP_LEN, H, W, SEQ_LEN)
        assert layout.img_pad == IMG_PAD                 # the pad policy is live here

        rope_q, rope_kv = coarse_ropes_zimage(embedder, layout, F, "cpu")

        # ``q`` answers for EVERY stream slot (Phase I needs no stitching); the
        # pad slots ride the model's own zero ids — position (0, 0, 0) rotates by
        # 0, so the identity matrix: the pad tokens' rope is a no-op
        assert tuple(rope_q.shape) == (1, 1, SEQ_LEN, embedder.head_dim // 2, 2, 2)
        identity = torch.eye(2).expand_as(rope_q[0, 0, CAP_LEN + IMG_TOKENS:]).clone()
        assert torch.allclose(rope_q[0, 0, CAP_LEN + IMG_TOKENS:], identity)
        # the caption half is the model's own t-axis ids (1..cap_len, row/col zero)
        cap_ids = torch.zeros(1, CAP_LEN, ZIMAGE_AXES)
        cap_ids[0, :, 0] = torch.arange(1, CAP_LEN + 1, dtype=torch.float32)
        assert torch.allclose(rope_q[0, 0, :CAP_LEN], embedder(cap_ids)[0, 0])

        # ``kv`` is in the engine's key order: caption, pad (unchanged), coarse grid
        assert rope_kv.shape[2] == layout.pass_through_len + (H // F) * (W // F)
        assert torch.allclose(rope_kv[0, 0, CAP_LEN:CAP_LEN + IMG_PAD],
                              torch.eye(2).expand_as(
                                  rope_kv[0, 0, CAP_LEN:CAP_LEN + IMG_PAD]).clone())

        # an image query rides its BLOCK's coarse position, not its own: the
        # token at (row 3, col 5) rides (1, 2) at f=2 — built by hand from the
        # fixture's theta so a hardcoded theta/table cannot pass
        q_index = CAP_LEN + 3 * W + 5
        block_ids = torch.zeros(1, 1, ZIMAGE_AXES)
        block_ids[0, 0, 0] = IMG_T
        block_ids[0, 0, 1] = 1.0
        block_ids[0, 0, 2] = 2.0
        assert torch.allclose(rope_q[0, 0, q_index], embedder(block_ids)[0, 0, 0])
        own_ids = block_ids.clone()
        own_ids[0, 0, 1], own_ids[0, 0, 2] = 3.0, 5.0
        assert not torch.allclose(rope_q[0, 0, q_index], embedder(own_ids)[0, 0, 0])

        # the pooled keys are the coarse grid in raster order, all on one t value
        for k in range((H // F) * (W // F)):
            kid = torch.zeros(1, 1, ZIMAGE_AXES)
            kid[0, 0, 0] = IMG_T
            kid[0, 0, 1] = float(k // (W // F))
            kid[0, 0, 2] = float(k % (W // F))
            assert torch.allclose(rope_kv[0, 0, CAP_LEN + IMG_PAD + k], embedder(kid)[0, 0, 0])

        # theta comes from the installed embedder, not a literal in the adapter
        other = MockEmbedND(theta=500.0)
        other_kv = coarse_ropes_zimage(other, layout, F, "cpu")[1]
        assert not torch.allclose(other_kv, rope_kv)


# ---------------------------------------------------------------------------
# Capture — the model's own rope tables, in call order
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRopeCapture:
    def test_group_capture_reads_cap_and_image_lengths(self):
        """Groups are kept in call order; cap_len is group #1's length, not a guess."""
        embedder = MockEmbedND()
        # lumina's two groups for this canvas: the padded caption block, then
        # the padded image (model.py:663-668, 725-727)
        cap_ids = torch.zeros(1, CAP_LEN, ZIMAGE_AXES)
        cap_ids[0, :, 0] = torch.arange(1, CAP_LEN + 1, dtype=torch.float32)
        cap_table = embedder(cap_ids)
        img_ids = torch.zeros(1, IMG_TOKENS + IMG_PAD, ZIMAGE_AXES)
        img_ids[0, :, 0] = IMG_T
        image_table = embedder(img_ids)

        rec = ZImageRopeRecorder()
        rec.record(cap_table)
        rec.record(image_table)

        assert rec.cap_len == CAP_LEN == 32          # group #1's sequence length
        assert rec.image_table.shape[2] == IMG_TOKENS + IMG_PAD
        assert len(rec.groups) == 2
        full = rec.full_table()
        assert full.shape[2] == CAP_LEN + IMG_TOKENS + IMG_PAD == SEQ_LEN
        assert torch.allclose(full, torch.cat((cap_table, image_table), dim=2))

        # lumina's own ``movedim(1, 2)`` view of the same tables records identically
        rec_view = ZImageRopeRecorder()
        rec_view.record(cap_table.movedim(1, 2))
        rec_view.record(image_table.movedim(1, 2))
        assert rec_view.cap_len == CAP_LEN
        assert torch.allclose(rec_view.full_table(), full)

        # a fresh forward shares nothing (the SPA discipline)
        rec.reset()
        assert rec.groups == ()
        with pytest.raises(ValueError, match="no group"):
            rec.full_table()
        rec.record(embedder(cap_ids))          # only the caption group recorded
        with pytest.raises(ValueError, match="image"):
            rec.image_table


# ---------------------------------------------------------------------------
# Pad policy
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPadPolicy:
    def test_pad_tokens_excluded_from_engine_inputs(self):
        """The engine's image input is exactly h*w tokens — never a pad token (Q6)."""
        layout = zimage_layout(TXT_LEN, CAP_LEN, H, W, SEQ_LEN)
        k = torch.arange(SEQ_LEN, dtype=torch.float32).view(1, 1, SEQ_LEN, 1)

        image = layout.image(k)
        assert image.shape[2] == IMG_TOKENS == H * W
        # the window is the stream's own [cap_len : cap_len + h*w) and no token
        # of it is a pad token (every value is a real image-token index)
        assert torch.equal(image[0, 0, :, 0],
                           torch.arange(CAP_LEN, CAP_LEN + IMG_TOKENS, dtype=torch.float32))
        pad = layout.img_end
        assert not ((image[0, 0, :, 0] >= pad).any())          # no image-pad token
        assert not ((image[0, 0, :, 0] < CAP_LEN).any())      # no caption token

        # the pass-through side is everything else, cap block first then image pad
        passthrough = layout.pass_through(k)
        assert passthrough.shape[2] == layout.pass_through_len == SEQ_LEN - IMG_TOKENS
        assert torch.equal(
            passthrough[0, 0, :, 0],
            torch.cat((torch.arange(CAP_LEN, dtype=torch.float32),
                       torch.arange(pad, SEQ_LEN, dtype=torch.float32))),
        )
        # the engine order is that side followed by the image window
        assert torch.equal(layout.engine_stream(k)[0, 0, :, 0],
                           torch.cat((passthrough[0, 0, :, 0], image[0, 0, :, 0])))

        # a stream that does not fit the policy is refused, not silently re-sliced
        with pytest.raises(ValueError, match="padded caption block|pad to a multiple"):
            zimage_layout(TXT_LEN, 40, H, W, SEQ_LEN)                 # cap not 32-multiple of 20
        with pytest.raises(ValueError, match="joint stream is"):
            zimage_layout(TXT_LEN, CAP_LEN, H, W, SEQ_LEN - 1)        # one token short

    def test_phase1_pad_kv_passed_through_unchanged(self):
        """Phase I: the pad key/value tokens survive pooling untouched."""
        layout = zimage_layout(TXT_LEN, CAP_LEN, H, W, SEQ_LEN)
        torch.manual_seed(1)
        heads, dim = 2, 4
        k = _filled_stream(1, heads, dim) * 0.5 + 0.25
        v = _filled_stream(1, heads, dim) * -0.75

        k_eng, v_eng = layout.engine_stream(k), layout.engine_stream(v)
        k_pooled, v_pooled = pool_keys_values(
            k_eng, v_eng, layout.pass_through_len, H, W, F, "subsample",
        )

        # one pooled key per f x f block behind the never-pooled pass-through side
        assert k_pooled.shape[2] == layout.pass_through_len + (H // F) * (W // F)
        assert v_pooled.shape[2] == k_pooled.shape[2]
        # the pass-through rows — caption block AND image pad — are the stream's
        # own tokens, bit for bit (text is never pooled, src/area_attn.py:175-190)
        assert torch.equal(k_pooled[:, :, :layout.pass_through_len],
                           k_eng[:, :, :layout.pass_through_len])
        assert torch.equal(v_pooled[:, :, :layout.pass_through_len],
                           v_eng[:, :, :layout.pass_through_len])
        # the pooled image keys come from the REAL image tokens only (the block
        # centre of each f x f block), so no pad token ever enters the key set
        expected_centre = (F // 2) + W * (F // 2)
        assert torch.equal(k_pooled[0, 0, -((H // F) * (W // F)), 0],
                           k[0, 0, CAP_LEN + expected_centre, 0])

    def test_phase2_pad_positions_filled_after_stitch(self):
        """Phase II: the answers go back in stream order and the pad slots read zero."""
        layout = zimage_layout(TXT_LEN, CAP_LEN, H, W, SEQ_LEN)
        assert (layout.cap_pad, layout.img_pad) == (CAP_LEN - TXT_LEN, IMG_PAD)
        heads, dim = 2, 4
        torch.manual_seed(2)
        out_text = torch.randn(1, heads, TXT_LEN, dim)
        out_image = torch.randn(1, heads, IMG_TOKENS, dim)

        out = stitch_fine_zeros(out_text, out_image, layout)

        assert out.shape == (1, heads, SEQ_LEN, dim)
        assert torch.equal(out[:, :, :TXT_LEN], out_text)
        assert torch.equal(out[:, :, CAP_LEN:CAP_LEN + IMG_TOKENS], out_image)
        # both discarded pad spans are exactly zero, and nothing else is
        assert torch.equal(out[:, :, TXT_LEN:CAP_LEN], torch.zeros_like(out[:, :, TXT_LEN:CAP_LEN]))
        assert torch.equal(out[:, :, CAP_LEN + IMG_TOKENS:],
                           torch.zeros_like(out[:, :, CAP_LEN + IMG_TOKENS:]))
        # the zero slots are exactly the pad spans — caption pad plus image pad
        assert layout.cap_pad == CAP_LEN - TXT_LEN
        assert layout.img_pad == IMG_PAD
        assert int((out[0, 0, :, 0] == 0).sum()) == layout.cap_pad + layout.img_pad
        # a mismatched output pair is refused rather than mis-placed
        with pytest.raises(ValueError, match="do not match the Z-Image layout"):
            stitch_fine_zeros(out_text[:, :, :-1], out_image, layout)


# ---------------------------------------------------------------------------
# Phase-regime advisory (docs/2026-10-17-area-zimage-ripple-diagnosis.md)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhaseRegimeAdvisory:
    """``_warn_regime``: warn once when Phase I floods a short distilled schedule.

    The node default ``phase1_steps=8`` is the reference's FLUX constant *of 28*
    (``src/area.py:144``).  Measured on Z-Image Turbo's 8-step schedule it makes
    every step Phase I — pooled K/V everywhere, no fine windowed steps to restore
    local detail, periodic ripple/mosaic + swirls (the I-Max Turbo failure class,
    ``.dev/docs/2026-10-08-imax-zimage-guidance-bug.md``).  Advisory only: a
    28-step FLUX schedule never trips it, and a large phase-1 share is the user's
    choice, kept by the explicit knob (same discipline as I-Max's advisory).
    """

    def _warns(self):
        return "area-regime" in area._warned

    def test_short_schedule_all_coarse_warns_once(self):
        area._warned.clear()
        try:
            area._warn_regime(8, 8, "coarse")
            assert self._warns()
            # warn-once: a second forward (same key set) adds nothing
            area._warn_regime(8, 8, "coarse")
            assert len(area._warned) == 1
        finally:
            area._warned.clear()

    def test_short_schedule_phase1_2_does_not_warn(self):
        area._warned.clear()
        try:
            area._warn_regime(8, 2, "coarse")
            assert not self._warns()
        finally:
            area._warned.clear()

    def test_long_schedule_default_does_not_warn(self):
        area._warned.clear()
        try:
            # FLUX parity: the paper's 8-of-28 share on a long schedule is fine
            area._warn_regime(28, 8, "coarse")
            assert not self._warns()
        finally:
            area._warned.clear()

    def test_plain_or_fine_phase_does_not_warn(self):
        area._warned.clear()
        try:
            area._warn_regime(8, 8, "fine")
            area._warn_regime(8, 8, None)
            assert not self._warns()
        finally:
            area._warned.clear()

    def test_warning_names_the_fix(self, caplog):
        area._warned.clear()
        try:
            with caplog.at_level("WARNING", logger="ComfyUI-DyPE"):
                area._warn_regime(8, 8, "coarse")
            message = caplog.text
            assert "8 of 8" in message and "short schedule" in message
            assert "phase1_steps" in message and "sigma" in message
        finally:
            area._warned.clear()

    def test_sigma_mode_short_schedule_below_threshold(self):
        # sigma mode auto-selects a fine majority on the same 8-step table, so
        # the advisory stays silent on the schedule-safe switch (theory.md:240-246)
        area._warned.clear()
        try:
            sigmas = torch.tensor([1.0, 0.9474, 0.8824, 0.8, 0.6923, 0.5455, 0.3333])
            count = sigma_prefix_count(sigmas, ZIMAGE_SIGMA_STAR)
            assert count == 2
            area._warn_regime(8, count, "coarse")
            assert not self._warns()
        finally:
            area._warned.clear()

    def test_resolve_forward_warns_on_short_schedule(self):
        # Integration: the advisory fires from the real per-forward resolver, not
        # only when called directly (pins the ``_area_resolve_forward`` wiring).
        area._warned.clear()
        rt = area._AreaRuntime(area.AreaConfig(phase1_steps=8), NextDiT(),
                               ZIMAGE_PROFILE)
        params = {
            "input": torch.zeros(1, 16, 256, 256),       # 128x128 patch grid
            "timestep": torch.tensor([1.0]),
            "c": {
                "c_crossattn": torch.zeros(1, 64, 64),
                "transformer_options": {
                    "sample_sigmas": torch.tensor(
                        [1.0, 0.9474, 0.8824, 0.8, 0.6923, 0.5455, 0.3333, 0.0]),
                    "sigmas": torch.tensor([1.0]),
                },
            },
        }
        try:
            ctx = area._area_resolve_forward(rt, params)
            assert ctx.phase == "coarse"
            assert self._warns()
        finally:
            area._warned.clear()
            area._AREA_RUNTIME = None
            area._ZRUNTIME = None
