"""AReA × Krea-2 — the geometry, the arch profile and the adapter (plan §7/S2/S4/S5).

Krea-2's latent arrives 5-D — its latent format is ``Wan21``
(``comfy/supported_models.py:2009-2035``), whose ``latent_dimensions == 3``
(``comfy/latent_formats.py:731``) makes comfy unsqueeze the sampler's 4-D
latent to ``[B, 16, T, H, W]`` (``comfy/sample.py:58-59``) — and AReA's own
guard (``x.ndim != 4``, ``src/area.py:1215``) would make it a SILENT NO-OP on
every Krea-2 image run.  These tests pin the two pure helpers that fix that,
and the geometry constants they depend on.

The constants are **ASSUMED, not measured** (no Krea-2 checkpoint exists in the
dev environment), so they carry a provenance label and the tests read it: a
silent constant fails the build.

**S4 — the arch profile** (``src/area_arch.py``): Krea-2 resolves to
``KREA_PROFILE``, the FLUX/Z-Image profiles keep every field, and a
``model_type`` that contradicts the backbone raises.

**S5 — the adapter** (``src/area_krea.py``): the single-table rope recorder
(with its pause guard), the joint layout, the FLUX-shaped coarse tables built
through the model's own embedder, and the head-count + ``img_slice`` + length
eligibility guard.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_krea.py
"""
from __future__ import annotations

import pytest
import torch

from _krea_fixtures import MockEmbedND, SingleStreamDiT
from src.area_arch import (
    FLUX_PROFILE,
    KREA_PROFILE,
    ZIMAGE_PROFILE,
    _ArchProfile,
    resolve_area_arch,
)
from src.area_krea import (
    KREA_GEOMETRY_NOTE,
    KREA_GEOMETRY_PROVENANCE,
    KREA_NATIVE_TOKENS,
    KREA_PATCH,
    KREA_PX_PER_PATCH,
    KREA_TRAINED_RESOLUTION_PX,
    KREA_VAE_DOWNSCALE,
    KreaRopeRecorder,
    capture_krea_table,
    coarse_ropes_krea,
    krea_latent_grid,
    krea_layout,
    krea_stream_eligible,
    krea_squeeze_latent,
)

#: The model's own patch factor (``comfy/ldm/krea2/model.py:242``).
PATCH = KREA_PATCH


def _latent(b=1, c=16, t=1, h=8, w=8):
    """A Krea-2 latent ``[B, C, T, H, W]`` (the shape comfy hands the model)."""
    return torch.zeros(b, c, t, h, w)


@pytest.mark.unit
class TestKreaLatent:
    def test_squeeze_drops_the_singleton_temporal_axis(self):
        """``[B, 16, 1, H, W]`` -> ``[B, 16, H, W]`` — the shape AReA can read."""
        squeezed = krea_squeeze_latent(_latent(b=2, h=8, w=12))
        assert squeezed.shape == (2, 16, 8, 12)
        # not a copy of the wrong axes: the CONTENT is the temporal slice
        latent = torch.arange(2 * 16 * 1 * 8 * 12, dtype=torch.float32).reshape(2, 16, 1, 8, 12)
        assert torch.equal(krea_squeeze_latent(latent), latent[:, :, 0])

    def test_squeeze_leaves_a_4d_latent_untouched(self):
        """A 4-D latent is already what AReA wants — it passes through."""
        latent = torch.zeros(1, 16, 8, 12)
        squeezed = krea_squeeze_latent(latent)
        assert squeezed is latent

    def test_squeeze_refuses_more_than_one_frame(self):
        """A video latent is REFUSED — the frames are not one rectangular canvas."""
        with pytest.raises(ValueError) as excinfo:
            krea_squeeze_latent(_latent(t=4))
        message = str(excinfo.value)
        # names the offender AND the remedy (house rule, src/hap.py:876-895)
        assert "4-frame" in message
        assert "plain attention" in message

    def test_squeeze_refuses_a_non_tensor(self):
        with pytest.raises(ValueError, match="must be a tensor"):
            krea_squeeze_latent([1, 2, 3, 4])


@pytest.mark.unit
class TestKreaGrid:
    def test_grid_is_ceil_div_by_patch(self):
        """A latent that is not a whole number of patches rounds UP, never down."""
        assert krea_latent_grid(_latent(h=8, w=8), PATCH) == (4, 4)
        assert krea_latent_grid(_latent(h=9, w=7), PATCH) == (5, 4)
        assert krea_latent_grid(_latent(h=1, w=1), PATCH) == (1, 1)

    def test_grid_matches_the_models_process_img_padding(self):
        """The grid is the model's OWN ``h_, w_`` after ``pad_to_patch_size``.

        ``comfy/ldm/krea2/model.py:287-288`` pads the latent up to a whole
        number of patches and then divides — so the token grid is the ceil
        division, and a partial edge row still owns a token.
        """
        for height, width in ((8, 8), (9, 7), (1, 1), (16, 32)):
            latent = _latent(h=height, w=width)
            padded_h = -(-height // PATCH) * PATCH
            padded_w = -(-width // PATCH) * PATCH
            assert krea_latent_grid(latent, PATCH) == (padded_h // PATCH, padded_w // PATCH)
            # ... and the padded latent's own division agrees
            assert (padded_h // PATCH) * (padded_w // PATCH) == \
                krea_latent_grid(latent, PATCH)[0] * krea_latent_grid(latent, PATCH)[1]

    def test_grid_accepts_the_five_dimensional_latent(self):
        """The 5-D latent (the NORMAL Krea-2 case) resolves to the same grid."""
        assert krea_latent_grid(_latent(t=1, h=8, w=8), PATCH) == (4, 4)
        with pytest.raises(ValueError, match="video latent"):
            krea_latent_grid(_latent(t=3), PATCH)


@pytest.mark.unit
class TestKreaGeometryConstants:
    def test_px_per_patch_is_vae_downscale_times_patch(self):
        """One token covers ``vae_downscale * patch`` pixels — no magic number.

        The VAE downscale is decided by the CHECKPOINT's weights
        (``comfy/sd.py:641-667``), so a hard-coded 16 that ignores the measured
        factor would be wrong the moment the checkpoint says 16x.
        """
        assert KREA_PX_PER_PATCH == KREA_VAE_DOWNSCALE * KREA_PATCH

    def test_native_tokens_times_px_per_patch_is_the_trained_resolution(self):
        """The trained extent in tokens and the pixel factor must AGREE."""
        assert KREA_NATIVE_TOKENS * KREA_PX_PER_PATCH == KREA_TRAINED_RESOLUTION_PX
        # the pack's own hypothesis: 1024 px over the 16 px token grid
        assert KREA_NATIVE_TOKENS == 64
        assert KREA_TRAINED_RESOLUTION_PX == 1024

    def test_geometry_constants_are_documented_as_measured_or_assumed(self):
        """An unlabelled constant fails the build."""
        assert KREA_GEOMETRY_PROVENANCE in ("MEASURED", "ASSUMED")
        assert KREA_GEOMETRY_PROVENANCE in KREA_GEOMETRY_NOTE
        # the note says what is NOT known, in the plan's wording
        assert "ASSUMED" in KREA_GEOMETRY_NOTE
        assert "not measured" in KREA_GEOMETRY_NOTE
        assert "VAE downscale" in KREA_GEOMETRY_NOTE
        # the module's own docstring repeats the label
        import src.area_krea as krea

        assert KREA_GEOMETRY_PROVENANCE in (krea.__doc__ or "")


# ---------------------------------------------------------------------------
# S4 — the arch profile
# ---------------------------------------------------------------------------


class Flux(torch.nn.Module):
    """A FLUX-shaped backbone — the detectors read the CLASS NAME."""

    def __init__(self):
        super().__init__()
        self.pe_embedder = torch.nn.Linear(3, 3)


class NextDiT(torch.nn.Module):
    """A Lumina2-shaped backbone (``rope_embedder`` is the Z-Image probe)."""

    def __init__(self):
        super().__init__()
        self.rope_embedder = torch.nn.Linear(3, 3)


class _Nunchaku(torch.nn.Module):
    """A quantized backbone — the ``model.pos_embed`` probe claims it."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.pos_embed = torch.nn.Linear(3, 3)


@pytest.fixture(scope="module")
def krea_model():
    """The comfy-faithful Krea-2 mock backbone (``tests/_krea_fixtures.py``)."""
    return SingleStreamDiT()


@pytest.mark.unit
class TestKreaArchProfile:
    def test_resolve_arch_returns_krea_profile_for_single_stream_dit(self, krea_model):
        """A Krea-2 model resolves — otherwise the node would refuse everything."""
        profile = resolve_area_arch(krea_model, "auto")
        assert profile is KREA_PROFILE
        assert profile.model_type == "krea2"
        assert profile.unet_backbone == "SingleStreamDiT"

    def test_krea_profile_geometry_fields(self):
        """The profile carries the S2 constants and the NO-PAD stream policy."""
        assert KREA_PROFILE.px_per_patch == KREA_PX_PER_PATCH
        assert KREA_PROFILE.seq_pad_multiple == 0      # Krea-2 pads nothing
        assert KREA_PROFILE.native_tokens == KREA_NATIVE_TOKENS
        assert KREA_PROFILE.accepts

    def test_resolve_arch_rejects_flux_pin_on_krea_model(self, krea_model):
        with pytest.raises(ValueError) as excinfo:
            resolve_area_arch(krea_model, "flux")
        message = str(excinfo.value)
        assert "flux" in message          # names the offender
        assert "krea2" in message         # names the remedy

    def test_resolve_arch_rejects_zimage_pin_on_krea_model(self, krea_model):
        for spelling in ("zimage", "z_image"):
            with pytest.raises(ValueError) as excinfo:
                resolve_area_arch(krea_model, spelling)
            message = str(excinfo.value)
            assert spelling in message
            assert "krea2" in message

    def test_resolve_arch_rejects_krea_pin_on_flux_model(self):
        with pytest.raises(ValueError) as excinfo:
            resolve_area_arch(Flux(), "krea2")
        message = str(excinfo.value)
        assert "krea2" in message
        assert "SingleStreamDiT" in message

    def test_resolve_arch_flux_profile_unchanged(self):
        """The FLUX profile keeps EVERY field — the byte-identity requirement."""
        profile = resolve_area_arch(Flux(), "auto")
        assert profile is FLUX_PROFILE
        assert (profile.model_type, profile.unet_backbone) == ("flux", "Flux")
        assert (profile.px_per_patch, profile.seq_pad_multiple) == (16, 0)
        assert profile.native_tokens == 64

    def test_resolve_arch_zimage_profile_unchanged(self):
        profile = resolve_area_arch(NextDiT(), "auto")
        assert profile is ZIMAGE_PROFILE
        assert (profile.model_type, profile.unet_backbone) == ("zimage", "NextDiT")
        assert (profile.px_per_patch, profile.seq_pad_multiple) == (16, 32)
        assert profile.native_tokens == 64

    def test_resolve_arch_nunchaku_returns_none(self):
        """The nunchaku path (``None``) survives the new branch."""
        assert resolve_area_arch(_Nunchaku(), "auto") is None

    def test_arch_profile_positional_construction_still_valid(self):
        """The new field has a default — the existing constructions stay valid."""
        profile = _ArchProfile("flux", "Flux", 16, 0, "FLUX.1 (Flux) diffusion model")
        assert profile.native_tokens == 64
        assert profile.padded_seq_len(70) == 70


# ---------------------------------------------------------------------------
# S5 — the rope recorder
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKreaRopeRecorder:
    def test_recorder_keeps_the_single_table_of_a_forward(self):
        """The FIRST table wins; a second one is counted, never overwriting."""
        recorder = KreaRopeRecorder()
        recorder.armed = True
        first = torch.zeros(1, 1, 4, 3, 2, 2)
        second = torch.ones(1, 1, 4, 3, 2, 2)
        recorder.record(first)
        recorder.record(second)
        assert recorder.table is first
        assert recorder.extra_tables == 1
        assert recorder.ambiguous is True

    def test_recorder_ignores_calls_while_disarmed(self):
        """Unarmed, the recorder sees nothing — the arming-window discipline."""
        recorder = KreaRopeRecorder()
        recorder.record(torch.zeros(1, 1, 4, 3, 2, 2))
        assert recorder.table is None
        assert recorder.extra_tables == 0

    def test_recorder_reset_clears_the_forward_state(self):
        """No table leaks across steps (the SPA discipline)."""
        recorder = KreaRopeRecorder()
        recorder.armed = True
        recorder.record(torch.zeros(1, 1, 4, 3, 2, 2))
        recorder.reset()
        assert recorder.table is None
        assert recorder.extra_tables == 0

    def test_capture_registers_the_hook_on_the_embedder(self):
        """The hook really fires on the model's own embedder — and can be removed."""
        embedder = MockEmbedND()
        hook, recorder = capture_krea_table(embedder)
        try:
            assert hook.handle is not None
            recorder.armed = True
            embedder(torch.zeros(1, 6, 3))
            assert recorder.table is not None
            assert int(recorder.table.shape[2]) == 6
        finally:
            hook.handle.remove()
        recorder.reset()
        embedder(torch.zeros(1, 6, 3))
        assert recorder.table is None

    def test_coarse_build_does_not_pollute_the_recorded_table(self):
        """The pause guard: the coarse tables go through the SAME embedder."""
        embedder = MockEmbedND()
        hook, recorder = capture_krea_table(embedder)
        try:
            recorder.armed = True
            model_table = embedder(torch.zeros(1, 6, 3))
            assert recorder.table is model_table
            layout = krea_layout(txt_len=2, h=2, w=2, stream_len=6)
            coarse_ropes_krea(embedder, layout, 2, torch.device("cpu"))
            # the model's own table survives, and nothing extra was recorded
            assert recorder.table is model_table
            assert recorder.extra_tables == 0
            assert recorder.ambiguous is False
        finally:
            hook.handle.remove()

    def test_the_pause_is_what_suppresses_the_extra_table(self):
        """Without the pause, an embedder call WOULD be recorded as an extra."""
        embedder = MockEmbedND()
        hook, recorder = capture_krea_table(embedder)
        try:
            recorder.armed = True
            embedder(torch.zeros(1, 6, 3))
            with recorder.paused():
                embedder(torch.zeros(1, 8, 3))
            assert recorder.extra_tables == 0
            embedder(torch.zeros(1, 8, 3))
            assert recorder.extra_tables == 1
        finally:
            hook.handle.remove()


# ---------------------------------------------------------------------------
# S5 — the joint layout
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKreaJointLayout:
    def test_layout_rejects_a_stream_that_is_not_txt_plus_hw(self):
        """A foreign stream is refused, not sliced (the zimage_layout discipline)."""
        with pytest.raises(ValueError) as excinfo:
            krea_layout(txt_len=8, h=4, w=4, stream_len=99)
        message = str(excinfo.value)
        assert "plain attention" in message
        assert "24" in message            # 8 caption + 16 image tokens

    def test_layout_spans_are_txt_len_and_h_times_w(self):
        """The image starts where the caption ends — the text block is never pooled."""
        layout = krea_layout(txt_len=8, h=4, w=4, stream_len=24)
        assert layout.img_start == 8
        assert layout.img_end == 24
        assert layout.img_tokens == 16
        stream = torch.arange(24, dtype=torch.float32).reshape(1, 1, 24, 1)
        assert layout.text(stream).shape[2] == 8
        assert layout.image(stream).shape[2] == 16
        assert float(layout.image(stream).flatten()[0]) == 8.0


# ---------------------------------------------------------------------------
# S5 — the coarse tables
# ---------------------------------------------------------------------------


class _SpyEmbedder(torch.nn.Module):
    """An embedder that RECORDS the ids it was handed and returns a marker."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, ids):
        self.calls.append(ids)
        return torch.full((1, 1, ids.shape[1], 3, 2, 2), float(len(self.calls)))


@pytest.mark.unit
class TestKreaCoarseTables:
    def test_coarse_tables_use_the_models_own_embedder(self):
        """The tables come from the model's embedder — a chained one composes."""
        spy = _SpyEmbedder()
        layout = krea_layout(txt_len=8, h=4, w=4, stream_len=24)
        rope_q, rope_kv = coarse_ropes_krea(spy, layout, 2, torch.device("cpu"))
        assert len(spy.calls) == 2
        assert torch.equal(rope_q, torch.full((1, 1, 24, 3, 2, 2), 1.0))
        assert torch.equal(rope_kv, torch.full((1, 1, 12, 3, 2, 2), 2.0))

    def test_coarse_query_table_covers_text_plus_full_grid(self):
        """The query table keeps the caption AND every image token.

        The block positions are CONTINUOUS (the continuous-position mechanism): every token
        keeps its own position scaled by ``1/f`` instead of collapsing onto
        its block's integer coordinate (the floor division, plan Q9), so the
        phase does not jump by a whole unit at every block boundary.
        """
        spy = _SpyEmbedder()
        layout = krea_layout(txt_len=8, h=4, w=4, stream_len=24)
        coarse_ropes_krea(spy, layout, 2, torch.device("cpu"))
        q_ids = spy.calls[0]
        assert q_ids.shape == (1, 8 + 16, 3)
        # the caption rides at the origin (FLUX's zero text ids)
        assert torch.equal(q_ids[0, :8], torch.zeros(8, 3))
        # the image rides on its own continuous position (row / f)
        assert torch.equal(q_ids[0, 8:, 1], torch.tensor([0.] * 4 + [0.5] * 4
                                                          + [1.] * 4 + [1.5] * 4))
        assert torch.equal(q_ids[0, 8:, 2], torch.tensor([0., 0.5, 1., 1.5] * 4))

    def test_coarse_key_table_sits_on_the_ceil_based_coarse_grid(self):
        """The pooled keys sit on the ceil-based grid (a partial edge block counts).

        They sit at their block's GEOMETRIC CENTROID — ``R + (f - 1) / (2f)``,
        the mean of the block's query positions — so the phase step across a
        block boundary equals the one inside the block.
        """
        spy = _SpyEmbedder()
        layout = krea_layout(txt_len=8, h=4, w=4, stream_len=24)
        coarse_ropes_krea(spy, layout, 2, torch.device("cpu"))
        kv_ids = spy.calls[1]
        assert kv_ids.shape == (1, 8 + 4, 3)      # ceil(4/2) x ceil(4/2)
        assert torch.equal(kv_ids[0, 8:, 1], torch.tensor([0.25, 0.25, 1.25, 1.25]))
        assert torch.equal(kv_ids[0, 8:, 2], torch.tensor([0.25, 1.25, 0.25, 1.25]))

    def test_coarse_key_grid_is_ceil_based_on_a_partial_axis(self):
        """A grid the blocks do not tile pools onto a ceil-sized canvas."""
        spy = _SpyEmbedder()
        layout = krea_layout(txt_len=3, h=5, w=4, stream_len=3 + 20)
        coarse_ropes_krea(spy, layout, 2, torch.device("cpu"))
        kv_ids = spy.calls[1]
        assert kv_ids.shape == (1, 3 + 6, 3)      # ceil(5/2) x ceil(4/2)

    def test_coarse_tables_are_zero_on_axis_zero(self):
        """Axis 0 is the FRAME index — a single-frame image carries 0."""
        spy = _SpyEmbedder()
        layout = krea_layout(txt_len=8, h=4, w=4, stream_len=24)
        coarse_ropes_krea(spy, layout, 2, torch.device("cpu"))
        for ids in spy.calls:
            assert torch.equal(ids[0, :, 0], torch.zeros(ids.shape[1]))


# ---------------------------------------------------------------------------
# S5 — the eligibility guard
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestKreaEligibility:
    def test_eligibility_accepts_the_main_calls(self):
        assert krea_stream_eligible(
            heads=48, dm_heads=48, stream_len=24, txt_len=8, h=4, w=4,
            img_slice=[8, 24],
        ) is True

    def test_eligibility_rejects_the_20_head_aux_calls(self):
        """The four text-fusion calls: 20 heads, and no ``img_slice`` yet.

        Their stream length is the PROMPT length — a PREFIX of the joint
        stream's — so the head count and the missing ``img_slice`` are what
        keep them out of the engine.
        """
        assert krea_stream_eligible(
            heads=20, dm_heads=48, stream_len=8, txt_len=8, h=4, w=4,
            img_slice=[8, 24],
        ) is False
        assert krea_stream_eligible(
            heads=20, dm_heads=48, stream_len=8, txt_len=8, h=4, w=4,
            img_slice=None,
        ) is False
        # ... and the head count alone is enough: even a call whose geometry
        # matches EXACTLY is refused when its head count is not the block's.
        assert krea_stream_eligible(
            heads=20, dm_heads=48, stream_len=24, txt_len=8, h=4, w=4,
            img_slice=[8, 24],
        ) is False

    def test_eligibility_rejects_a_foreign_stream_length(self):
        assert krea_stream_eligible(
            heads=48, dm_heads=48, stream_len=25, txt_len=8, h=4, w=4,
            img_slice=[8, 25],
        ) is False

    def test_eligibility_rejects_a_missing_img_slice(self):
        assert krea_stream_eligible(
            heads=48, dm_heads=48, stream_len=24, txt_len=8, h=4, w=4,
            img_slice=None,
        ) is False

    def test_eligibility_never_raises_on_garbage(self):
        """A malformed call is ineligible, not an exception."""
        assert krea_stream_eligible(
            heads=None, dm_heads=48, stream_len=24, txt_len=8, h=4, w=4,
            img_slice="nonsense",
        ) is False
