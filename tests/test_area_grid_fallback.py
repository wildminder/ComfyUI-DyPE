"""AReA on a patch grid with no common divisor >= 2 — the 512x2000 canvas.

The reference's Eq. 5 refuses a grid whose patch counts share no factor >= 2:
``initial_factor`` still raises (it is the reference's contract, kept verbatim
and pinned by ``tests/test_area_schedule.py``).  The ENGINE never refuses: the
factor comes from the extent rule instead of the divisibility rule, the last
pooling block on a partial axis is a *partial* block, and the user is told
once, by name, what changed.  The canvas, the token stream, the latent and the
output resolution are untouched — only the block partition moves.

This file pins that contract behaviourally:

* the 2000x512 px canvas (a 125x32 patch grid, ``gcd == 1``) runs on BOTH seams:
  the apply raises nothing, a forward completes, and the seam answers with the
  pristine attention's stream length;
* every image token belongs to exactly one pooling block, and the pooled
  canvas is the ceil-based one (16x63 = 1008 tokens);
* exactly ONE warning per adjusted grid, and none at all for a grid that
  already tiles (nor for a disabled AReA);
* a grid that DOES tile behaves exactly as before — the byte-identity goldens
  below were captured on the pre-change tree (HEAD ``56f90a3``) with the CI
  interpreter (``torch 2.11.0+cu130``) and must not move.

The seam mocks are the repo's own (imported from the two seam test files), so
the goldens are the ones those files' harness produces.

Markers: @pytest.mark.unit
"""
from __future__ import annotations

import hashlib
import logging
import sys
import types

import pytest
import torch

import src.area as area

# The repo's seam doubles, reused rather than forked: the byte-identity
# goldens were captured with exactly these mocks (the FLUX ones reproduce the
# values the design note records), so a second copy here would move them.
import test_area_node as flux
import test_area_zimage_seam as zimage
from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_attn import apply_rope1, pool_keys_values
from src.area_schedule import build_schedule, coarse_grid, initial_factor, resolve_factor
from src.area_zimage import ZImageJointLayout, coarse_ropes_zimage

#: The user's canvas: 2000x512 px -> a 125x32 patch grid, ``gcd(125, 32) == 1``.
WIDTH, HEIGHT = 2000, 512
GRID_W, GRID_H = WIDTH // 16, HEIGHT // 16          # 125 x 32 patches
TXT_LEN = 32
HEADS, HEAD_DIM = 2, 128
STEPS, PHASE1 = 28, 8

#: The golden canvas: 1280 px -> an 80x80 patch grid (the FLUX default's twin).
GOLD_SIDE = 1280
GOLD_GRID = GOLD_SIDE // 16                          # 80
#: A multi-rung grid: 2560 px -> 160x160 patches (f0 = 4 -> 2 mid-phase).
BIG_SIDE = 2560
BIG_GRID = BIG_SIDE // 16                            # 160

#: The pooled canvas of the adjusted grid: ``ceil(32/2) x ceil(125/2)``.
POOLED_H, POOLED_W = 16, 63
POOLED_TOKENS = POOLED_H * POOLED_W                  # 1008

#: Byte-identity goldens, captured on the pre-change tree (see the module
#: docstring).  The recipe is :func:`_hash` — the sha256 of the tensor's
#: contiguous float32 bytes, truncated to 32 hex characters.
GOLDENS = {
    # FLUX: the seam's own answer on the 80x80 canvas, coarse and fine step.
    "flux80_coarse": "dca3a04b5655b81b6a40049a010c0d94",
    "flux80_fine": "1c5d27e285761eef13ec76dc10638022",
    # FLUX: the 160x160 canvas, coarse step.
    "flux160_coarse": "52ed0106c10bd149ea66b16728efd973",
    # Z-Image: the seam file's own 65x65 canvas, coarse and fine step.
    "zimage65_coarse": "394b3e8a2529bb35436fbaea4618d808",
    "zimage65_fine": "694ecab825562441a447a4426ac996be",
    # The pooled keys / values of the 80x80 grid, per pooling mode.
    "pool_subsample_k": "93fb5f62bc218d3d2a6deab3124eff6f",
    "pool_subsample_v": "a5392ef83c215b60d9c89862258107c9",
    "pool_mean_k": "8c6db78b59148a21bfca9a17b3333a53",
    "pool_mean_v": "b7fc0cc00fb961d62cfaa75bd7b3be25",
    "pool_medoid_k": "8d27a65069c7f1fb774b2c60c778ac6a",
    "pool_medoid_v": "9731af70c28bb4a9097e898dbeae38de",
    "pool_random_k": "780ac6ff520fc8bce97a0ca408284f6d",
    "pool_random_v": "951c32eae697798dc595c41f015c25b7",
    # The FLUX coarse rope ids of the 80x80 grid (query side / key side).
    "rope_q_ids": "b1b7b8ada72c8841fea5c88fad54463d",
    "rope_kv_ids": "6cc6716177cb8b2ec132fd89dd0ac7d6",
}


def _hash(tensor: torch.Tensor) -> str:
    """The golden recipe: sha256 of the tensor's contiguous float32 bytes."""
    data = tensor.detach().cpu().contiguous().float().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()[:32]


def _rect_ids(txt_len: int, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """``[1, txt_len + n, 3]`` position ids of a RECTANGULAR canvas."""
    ids = torch.zeros(1, txt_len + int(rows.numel()), 3, dtype=torch.float32)
    ids[0, txt_len:, 1] = rows.float()
    ids[0, txt_len:, 2] = cols.float()
    return ids


# ---------------------------------------------------------------------------
# Mocks / fixtures (the repo's seam doubles, installed per test)
# ---------------------------------------------------------------------------


@pytest.fixture
def flux_layers():
    """A mock ``comfy.ldm.flux.layers`` with a pristine ``attention``."""
    saved = sys.modules.get("comfy.ldm.flux.layers")
    saved_parent = getattr(sys.modules.get("comfy.ldm.flux"), "layers", None)
    parent = sys.modules.setdefault("comfy.ldm.flux", types.ModuleType("comfy.ldm.flux"))
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.attention = flux.plain_flux_attention
    sys.modules["comfy.ldm.flux.layers"] = layers
    parent.layers = layers
    try:
        yield layers
    finally:
        sys.modules["comfy.ldm.flux.layers"] = saved
        parent.layers = saved_parent
        if saved is None:
            sys.modules.pop("comfy.ldm.flux.layers", None)


@pytest.fixture
def lumina_model():
    """A mock ``comfy.ldm.lumina.model`` with a pristine bound symbol."""
    saved = sys.modules.get("comfy.ldm.lumina.model")
    saved_parent = getattr(sys.modules.get("comfy.ldm.lumina"), "model", None)
    parent = sys.modules.setdefault("comfy.ldm.lumina", types.ModuleType("comfy.ldm.lumina"))
    mod = types.ModuleType("comfy.ldm.lumina.model")
    mod.optimized_attention_masked = zimage.base_lumina_attention
    sys.modules["comfy.ldm.lumina.model"] = mod
    parent.model = mod
    try:
        yield mod
    finally:
        sys.modules["comfy.ldm.lumina.model"] = saved
        parent.model = saved_parent
        if saved is None:
            sys.modules.pop("comfy.ldm.lumina.model", None)


@pytest.fixture(autouse=True)
def _clean_area_state():
    """AReA's seam state and its one-time warning keys are process-global."""
    area._warned.clear()
    area._SEAM.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    area._ZSEAM.update(module=None, attr=None, orig=None, hook=None, recorder=None,
                       consumers=0)
    area._AREA_RUNTIME = None
    area._ZRUNTIME = None
    yield
    area._warned.clear()
    area._SEAM.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    area._ZSEAM.update(module=None, attr=None, orig=None, hook=None, recorder=None,
                       consumers=0)
    area._AREA_RUNTIME = None
    area._ZRUNTIME = None


# ---------------------------------------------------------------------------
# The rectangular harness: the 2000x512 canvas
# ---------------------------------------------------------------------------


def _flux_stream(h: int, w: int, txt_len: int = TXT_LEN, seed: int = 0):
    """The joint q/k/v of a rectangular canvas, plus the model's own rope table."""
    n = txt_len + h * w
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, HEADS, n, HEAD_DIM, generator=g)
    k = torch.randn(1, HEADS, n, HEAD_DIM, generator=g)
    v = torch.randn(1, HEADS, n, HEAD_DIM, generator=g)
    pos = torch.arange(h * w)
    pe = flux.MockPeEmbedder()(_rect_ids(txt_len, pos // w, pos % w))
    return q, k, v, pe


def _flux_forward(m, h: int, w: int, step: int, txt_len: int = TXT_LEN, seed: int = 0):
    """One sampled forward of a rectangular canvas, through the installed wrapper.

    The wrapper is what resolves the forward's context (the grid comes from the
    LATENT, not from the widget), so the forward has to go through it — a bare
    seam call would find no context and delegate to plain attention.
    """
    seam = sys.modules["comfy.ldm.flux.layers"].attention
    q, k, v, pe = _flux_stream(h, w, txt_len, seed)
    sigmas = torch.linspace(1.0, 0.0, STEPS + 1)
    opts = {"sample_sigmas": sigmas, "sigmas": sigmas[step]}

    def model_function(x, timestep, **c):
        # img_in is called for real, so the Phase II detail map has its source
        # (``ldm/flux/model.py:169``).
        img_in = getattr(m.model.diffusion_model, "img_in", None)
        if callable(img_in):
            img_in(torch.randn(x.shape[0], h * w, 2 * 2 * 16))
        return seam(flux._Container(q), flux._Container(k), flux._Container(v), pe=pe,
                    mask=None, transformer_options=c["transformer_options"],
                    preferred_attention=None)

    params = {"input": torch.randn(1, 16, h * 2, w * 2),
              "timestep": torch.tensor([float(opts["sigmas"])]),
              "c": {"c_crossattn": torch.randn(1, txt_len, 77), "transformer_options": opts}}
    return flux._unet_wrapper(m)(model_function, params)


def _zimage_stream(m, h: int, w: int, txt_len: int = zimage.TXT_LEN, seed: int = 0):
    """The joint (rotated) q/k/v of a rectangular Z-Image canvas."""
    cap, img = zimage.CAP_LEN, h * w
    dm = m.model.diffusion_model
    cap_ids = torch.zeros(1, cap, 3, dtype=torch.float32)
    cap_ids[0, :, 0] = torch.arange(1, cap + 1, dtype=torch.float32)
    img_ids = torch.zeros(1, img, 3, dtype=torch.float32)
    img_ids[0, :, 0] = float(cap + 1)
    pos = torch.arange(img)
    img_ids[0, :, 1] = (pos // w).float()
    img_ids[0, :, 2] = (pos % w).float()
    full = torch.cat((dm.rope_embedder(cap_ids), dm.rope_embedder(img_ids)), dim=2)
    g = torch.Generator().manual_seed(seed)
    shape = (1, zimage.HEADS, cap + img, zimage.HEAD_DIM)
    q = torch.randn(shape, generator=g)
    k = torch.randn(shape, generator=g)
    v = torch.randn(shape, generator=g)
    return apply_rope1(q, full), apply_rope1(k, full), v


def _zimage_forward(m, h: int, w: int, step: int, txt_len: int = zimage.TXT_LEN,
                    seed: int = 0):
    """One sampled forward of a rectangular Z-Image canvas."""
    seam = sys.modules["comfy.ldm.lumina.model"].optimized_attention_masked
    q, k, v = _zimage_stream(m, h, w, txt_len, seed)
    sigmas = torch.linspace(1.0, 0.0, STEPS + 1)
    opts = {"sample_sigmas": sigmas, "sigmas": sigmas[step]}

    def model_function(x, timestep, **c):
        # the model's own two rope groups, recorded by the seam's recorder
        dm = m.model.diffusion_model
        cap, img = zimage.CAP_LEN, h * w
        cap_ids = torch.zeros(1, cap, 3, dtype=torch.float32)
        cap_ids[0, :, 0] = torch.arange(1, cap + 1, dtype=torch.float32)
        img_ids = torch.zeros(1, img, 3, dtype=torch.float32)
        img_ids[0, :, 0] = float(cap + 1)
        pos = torch.arange(img)
        img_ids[0, :, 1] = (pos // w).float()
        img_ids[0, :, 2] = (pos % w).float()
        dm.rope_embedder(cap_ids)
        dm.rope_embedder(img_ids)
        for layer in dm.noise_refiner:
            layer(torch.randn(1, img, zimage.HIDDEN))
        return seam(zimage._Container(q), zimage._Container(k), zimage._Container(v),
                    zimage.HEADS, None, skip_reshape=True,
                    transformer_options=c["transformer_options"],
                    preferred_attention=None)

    params = {"input": torch.randn(1, 16, h * 2, w * 2),
              "timestep": torch.tensor([float(opts["sigmas"])]),
              "c": {"c_crossattn": torch.randn(1, txt_len, 77), "transformer_options": opts}}
    return flux._unet_wrapper(m)(model_function, params)


# ---------------------------------------------------------------------------
# The fallback itself
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAdjustedGrid:
    """A grid with no common divisor >= 2 is ADJUSTED, never refused."""

    def test_the_apply_raises_nothing_on_either_seam(self, flux_layers, lumina_model):
        """AC-1: the engine accepts the canvas on both backbones."""
        for model, kind in ((flux.mock_flux_model(), "flux"),
                            (zimage.mock_zimage_model(), "zimage")):
            m = apply_area_to_model(model, kind, WIDTH, HEIGHT,
                                    phase1_steps=PHASE1, seed=7)
            try:
                assert m._area_config is not None
            finally:
                restore_area_attention_hook(m)

    def test_the_warning_names_the_geometry(self, flux_layers, caplog):
        """AC-5: the advisory names the grid, the pixels, the factor, the canvas."""
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            m = apply_area_to_model(flux.mock_flux_model(), "flux", WIDTH, HEIGHT,
                                    phase1_steps=PHASE1, seed=7)
        try:
            text = caplog.text
            assert "no common divisor >= 2" in text
            assert "32x125" in text                 # the patch grid (rows x cols)
            assert "2000x512" in text               # the requested pixels
            assert "2x2 blocks" in text             # the working factor
            assert "16x63" in text                  # the pooled grid
            assert "output resolution are unchanged" in text
        finally:
            restore_area_attention_hook(m)

    def test_a_flux_forward_completes_with_the_untouched_stream(self, flux_layers):
        """AC-2: the forward runs and the stream length is the pristine one."""
        m = apply_area_to_model(flux.mock_flux_model(), "flux", WIDTH, HEIGHT,
                                phase1_steps=PHASE1, seed=7)
        try:
            out = _flux_forward(m, GRID_H, GRID_W, step=0)
        finally:
            restore_area_attention_hook(m)
        assert out.shape == (1, TXT_LEN + GRID_H * GRID_W, HEADS * HEAD_DIM)
        # AReA really replaced the attention (the plain answer would differ)
        q, k, v, pe = _flux_stream(GRID_H, GRID_W)
        plain = flux.plain_flux_attention(flux._Container(q), flux._Container(k),
                                          flux._Container(v), pe=pe, mask=None,
                                          transformer_options=None,
                                          preferred_attention=None)
        assert not torch.allclose(out, plain)

    def test_a_zimage_forward_completes_with_the_untouched_stream(self, lumina_model):
        """AC-2: the Z-Image seam runs the same canvas."""
        m = apply_area_to_model(zimage.mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                phase1_steps=PHASE1, seed=3)
        # the pristine answer of the same stream, drawn BEFORE the forward (the
        # seam's rope recorder is armed during one)
        q, k, v = _zimage_stream(m, GRID_H, GRID_W)
        plain = zimage.base_lumina_attention(zimage._Container(q), zimage._Container(k),
                                             zimage._Container(v), zimage.HEADS, None,
                                             skip_reshape=True)
        try:
            out = _zimage_forward(m, GRID_H, GRID_W, step=0)
        finally:
            restore_area_attention_hook(m)
        assert out.shape == (1, zimage.CAP_LEN + GRID_H * GRID_W,
                             zimage.HEADS * zimage.HEAD_DIM)
        assert not torch.allclose(out, plain)

    def test_the_pooled_canvas_is_the_ceil_of_the_factor(self, flux_layers,
                                                          lumina_model, monkeypatch):
        """AC-2: 1008 pooled tokens, and every image token in exactly one block."""
        seen = []
        real = area.pool_keys_values

        def spy(key, value, txt_len, h, w, f, pooling, **kw):
            out = real(key, value, txt_len, h, w, f, pooling, **kw)
            seen.append({"h": h, "w": w, "f": f, "k_len": out[0].shape[2]})
            return out

        monkeypatch.setattr(area, "pool_keys_values", spy)
        m = apply_area_to_model(flux.mock_flux_model(), "flux", WIDTH, HEIGHT,
                                phase1_steps=PHASE1, seed=7)
        try:
            _flux_forward(m, GRID_H, GRID_W, step=0)
        finally:
            restore_area_attention_hook(m)
        assert seen, "Phase I never pooled"
        for entry in seen:
            assert (entry["h"], entry["w"], entry["f"]) == (GRID_H, GRID_W, 2)
            assert entry["k_len"] == TXT_LEN + POOLED_TOKENS
        assert coarse_grid(GRID_H, GRID_W, 2) == (POOLED_H, POOLED_W)
        # the partition: every row and every column of the canvas is owned once
        rows = [i for r in range(POOLED_H) for i in range(r * 2, min(r * 2 + 2, GRID_H))]
        cols = [i for c in range(POOLED_W) for i in range(c * 2, min(c * 2 + 2, GRID_W))]
        assert rows == list(range(GRID_H))
        assert cols == list(range(GRID_W))

    def test_the_zimage_partial_block_keeps_its_own_centroid(self):
        """The Z-Image pooled key of a PARTIAL block sits at ITS members' mean.

        A full ``f`` block's key sits at ``(f - 1) / (2f)`` — the mean of its
        members' query positions.  The last block of the partial column axis
        owns a single column, so its key must sit at that column's own mean
        (not a quarter of a block away).
        """
        captured = []

        class SpyEmbedder:
            """A stand-in for the model's own ``rope_embedder``."""

            def __call__(self, ids):
                captured.append(ids.clone())
                return torch.zeros(1, 1, ids.shape[1], 4)

        layout = ZImageJointLayout(txt_len=zimage.TXT_LEN, cap_len=zimage.CAP_LEN,
                                   h=GRID_H, w=GRID_W,
                                   seq_len=zimage.CAP_LEN + GRID_H * GRID_W)
        coarse_ropes_zimage(SpyEmbedder(), layout, 2, "cpu", None)
        kv = captured[-1][0, -POOLED_TOKENS:]        # the coarse grid's key ids
        # the last block: rows 30-31 (a full pair) and column 124 (one member)
        members = [(r / 2.0, c / 2.0) for r in range(30, 32) for c in range(124, 125)]
        assert float(kv[-1, 1]) == pytest.approx(sum(m[0] for m in members) / len(members))
        assert float(kv[-1, 2]) == pytest.approx(sum(m[1] for m in members) / len(members))
        assert float(kv[-1, 2]) != pytest.approx(62 + 0.25)   # not the full-block offset
        # a full block keeps the reference's offset, unchanged
        assert float(kv[0, 1]) == pytest.approx(0.25)
        assert float(kv[0, 2]) == pytest.approx(0.25)

    def test_one_warning_per_adjusted_grid(self, flux_layers, caplog):
        """AC-5: the advisory is one-shot, however often the grid is resolved."""
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            m = apply_area_to_model(flux.mock_flux_model(), "flux", WIDTH, HEIGHT,
                                    phase1_steps=PHASE1, seed=7)
            try:
                _flux_forward(m, GRID_H, GRID_W, step=0)
                _flux_forward(m, GRID_H, GRID_W, step=0)
            finally:
                restore_area_attention_hook(m)
        assert caplog.text.count("no common divisor >= 2") == 1
        assert "area-grid-factor:32x125" in area._warned

    def test_another_adjusted_grid_warns_again(self, flux_layers, caplog):
        """AC-5: the key is per grid — a different resolution is not silenced."""
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            for width, height in ((WIDTH, HEIGHT), (2064, 2048)):
                m = apply_area_to_model(flux.mock_flux_model(), "flux", width, height,
                                        phase1_steps=PHASE1, seed=7)
                restore_area_attention_hook(m)
        assert caplog.text.count("no common divisor >= 2") == 2
        assert "area-grid-factor:32x125" in area._warned
        assert "area-grid-factor:128x129" in area._warned

    def test_a_tiling_grid_never_warns(self, flux_layers, caplog):
        """AC-5: a grid that already tiles is never announced."""
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            m = apply_area_to_model(flux.mock_flux_model(), "flux", GOLD_SIDE, GOLD_SIDE,
                                    phase1_steps=PHASE1, seed=7)
            try:
                _flux_forward(m, GOLD_GRID, GOLD_GRID, step=0)
            finally:
                restore_area_attention_hook(m)
        assert "no common divisor" not in caplog.text
        assert not [k for k in area._warned if k.startswith("area-grid-factor")]

    def test_a_disabled_area_is_silent(self, flux_layers, caplog):
        """AC-6: ``enable_area=False`` touches nothing and says nothing."""
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            m = apply_area_to_model(flux.mock_flux_model(), "flux", WIDTH, HEIGHT,
                                    enable_area=False)
        assert m._area_config is not None
        assert flux._unet_wrapper(m) is None
        assert sys.modules["comfy.ldm.flux.layers"].attention is flux.plain_flux_attention
        assert "no common divisor" not in caplog.text
        assert not [k for k in area._warned if k.startswith("area-grid-factor")]

    def test_two_forwards_on_the_adjusted_grid_are_bit_identical(self, flux_layers):
        """AC-8: the fallback is as deterministic as the tiled path."""
        outs = []
        for _ in range(2):
            m = apply_area_to_model(flux.mock_flux_model(), "flux", WIDTH, HEIGHT,
                                    phase1_steps=PHASE1, seed=7)
            try:
                outs.append(_flux_forward(m, GRID_H, GRID_W, step=0))
            finally:
                restore_area_attention_hook(m)
        assert torch.equal(outs[0], outs[1])

    def test_the_schedule_cache_separates_the_grids(self):
        """AC-8: the cached schedule is per canvas, so grids never share one."""
        rt = area._AreaRuntime(area.AreaConfig(phase1_steps=PHASE1), None)
        adjusted = area._area_schedule(rt, GRID_H, GRID_W, STEPS, PHASE1, TXT_LEN)
        tiled = area._area_schedule(rt, GOLD_GRID, GOLD_GRID, STEPS, PHASE1, TXT_LEN)
        assert len(rt.schedules) == 2
        assert adjusted.factors == tiled.factors == (2,) * PHASE1
        assert adjusted.grid == (GRID_H, GRID_W)
        assert tiled.grid == (GOLD_GRID, GOLD_GRID)

    def test_the_runtime_schedule_builds_on_the_adjusted_grid(self):
        """AC-3: the runtime path builds instead of raising."""
        schedule = build_schedule(125, 32, STEPS, PHASE1, area.NATIVE_PATCHES)
        assert schedule.factors == (2,) * PHASE1
        assert len(schedule.windows) == STEPS - PHASE1
        # the resolver is the engine's entry point and never raises
        assert resolve_factor(125, 32, area.NATIVE_PATCHES) == (2, True)
        assert resolve_factor(32, 125, area.NATIVE_PATCHES) == (2, True)
        assert resolve_factor(80, 80, area.NATIVE_PATCHES) == (2, False)


# ---------------------------------------------------------------------------
# The tiling path must not have moved (byte identity)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestTilingGridUnchanged:
    """A grid that tiles behaves EXACTLY as before the fallback.

    The hashes below were captured on the pre-change tree (HEAD ``56f90a3``)
    with the CI interpreter; a change in the pooling, the rope builders or the
    schedule would move one of them.
    """

    def test_the_flux_seam_answers_are_byte_identical(self, flux_layers):
        for tag, side, step in (("flux80_coarse", GOLD_SIDE, 0),
                                ("flux80_fine", GOLD_SIDE, STEPS - 1),
                                ("flux160_coarse", BIG_SIDE, 0)):
            m = apply_area_to_model(flux.mock_flux_model(), "flux", side, side,
                                    phase1_steps=PHASE1, seed=7)
            try:
                sim, _out = flux._run_forward(m, step=step, side=side)
            finally:
                restore_area_attention_hook(m)
            assert _hash(sim.calls[0]) == GOLDENS[tag], tag

    def test_the_zimage_seam_answers_are_byte_identical(self, lumina_model):
        for tag, step in (("zimage65_coarse", 0), ("zimage65_fine", zimage.STEPS - 1)):
            m = apply_area_to_model(zimage.mock_zimage_model(), "zimage", zimage.WIDTH,
                                    zimage.HEIGHT, phase1_steps=zimage.PHASE1, seed=3)
            try:
                _sim, out = zimage._run_forward(m, step=step)
            finally:
                restore_area_attention_hook(m)
            assert _hash(out) == GOLDENS[tag], tag

    def test_the_pooled_keys_and_values_are_byte_identical(self):
        """Every pooling mode pools the tiled grid exactly as before."""
        g = torch.Generator().manual_seed(0)
        key = torch.randn(1, 2, 32 + GOLD_GRID * GOLD_GRID, 64, generator=g)
        value = torch.randn(1, 2, 32 + GOLD_GRID * GOLD_GRID, 64, generator=g)
        for pooling in ("subsample", "mean", "medoid", "random"):
            k, v = pool_keys_values(key, value, 32, GOLD_GRID, GOLD_GRID, 2, pooling,
                                    seed=7, step=0, block=0)
            assert _hash(k) == GOLDENS[f"pool_{pooling}_k"], pooling
            assert _hash(v) == GOLDENS[f"pool_{pooling}_v"], pooling

    def test_the_coarse_rope_ids_are_byte_identical(self):
        """The FLUX coarse tables' position ids did not move."""
        n = GOLD_GRID * GOLD_GRID
        q_ids = flux._ids(32, torch.arange(n) // GOLD_GRID // 2,
                          torch.arange(n) % GOLD_GRID // 2)
        kv_ids = flux._ids(32, torch.arange(n // 4) // (GOLD_GRID // 2),
                           torch.arange(n // 4) % (GOLD_GRID // 2))
        assert _hash(q_ids) == GOLDENS["rope_q_ids"]
        assert _hash(kv_ids) == GOLDENS["rope_kv_ids"]

    def test_the_schedules_of_tiling_grids_are_unchanged(self):
        """The reference's Eq. 5 and the ramp's own numbers, unchanged."""
        native = area.NATIVE_PATCHES
        assert initial_factor(128, 128, native) == 2
        assert initial_factor(130, 130, native) == 5
        assert initial_factor(99, 99, native) == 3
        assert build_schedule(80, 80, STEPS, PHASE1, native).factors == (2,) * PHASE1
        assert build_schedule(96, 144, STEPS, PHASE1, native).factors == \
            (3, 3, 3, 3, 2, 2, 2, 2)
        assert build_schedule(130, 130, STEPS, PHASE1, native).factors == \
            (5, 5, 5, 5, 2, 2, 2, 2)
