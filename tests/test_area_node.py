"""AReA node + FLUX attention-seam conformance (Phase 3, ``src/area.py``).

Everything here is mock-based and GPU-free, per project convention.  The seam
conformance tests replicate how ComfyUI actually calls the seam —
``comfy/ldm/flux/layers.py:244`` / ``:349``:

    q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
    attn = attention(q, k, v, pe=pe, mask=attn_mask,
                     transformer_options=transformer_options,
                     preferred_attention=self.comfy_attention)

so the mock carries its own ``peek``/``take`` container (exactly
``ldm/modules/attention.py:193-209``) and a ``math.attention`` port whose output
shape is comfy's ``[b, N, heads*dim]`` (``ldm/modules/attention.py:303-314``).
Real-GPU / visual validation is out of scope.

Markers: @pytest.mark.unit
"""
from __future__ import annotations

import logging
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from nodes.area import AReA

import src.area as area
from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_attn import apply_rope1
from src.area_schedule import initial_factor, pooling_ramp

try:
    from comfy.model_patcher import ModelPatcher
except ImportError:  # pragma: no cover - mock env always provides it
    from tests.conftest import MockModelPatcher as ModelPatcher

# A 1280 px canvas: an 80x80 patch grid, just past the native 64-patch extent,
# small enough that a whole sampled forward runs in milliseconds on CPU.
WIDTH, HEIGHT = 1280, 1280
GRID_H = GRID_W = 80
TXT_LEN = 32
HEADS, HEAD_DIM = 2, 128
HIDDEN = HEADS * HEAD_DIM          # 256 — pe_embedder's dim (axes_dim sums to 128)
STEPS = 28                          # the reference's FLUX total
PHASE1 = 8
# A 2560 px canvas: 160x160 patches, the first grid whose Phase I ramp is
# multi-rung (f0 = 4 -> 2 mid-phase).  The 80x80 ramp is constant at 2, so a
# rung change is only observable through the seam on the big canvas.
BIG_WIDTH, BIG_HEIGHT = 2560, 2560
BIG_GRID = 160                       # 2560 px / 16 px per patch


# ---------------------------------------------------------------------------
# Mocks (comfy-faithful, GPU-free)
# ---------------------------------------------------------------------------


class _Container:
    """``comfy.ldm.modules.attention.AttentionTensorContainer``
    (``ldm/modules/attention.py:193-209``): single-owner, consumed by ``take``."""

    __slots__ = ("tensor",)

    def __init__(self, tensor):
        self.tensor = tensor

    def peek(self):
        if self.tensor is None:
            raise RuntimeError("attention tensor container has already been consumed")
        return self.tensor

    def take(self):
        tensor = self.peek()
        self.tensor = None
        return tensor


def plain_flux_attention(q, k, v, pe=None, mask=None, transformer_options=None,
                         preferred_attention=None):
    """``comfy.ldm.flux.math.attention`` — the function AReA wraps.

    Takes the containers once, applies ``pe`` to q/k, runs plain attention and
    returns comfy's ``[b, N, heads*dim]`` layout.
    """
    if hasattr(q, "take"):
        q, k, v = q.take(), k.take(), v.take()
    if pe is not None:
        q, k = apply_rope1(q, pe), apply_rope1(k, pe)
    heads = q.shape[1]
    out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], heads * out.shape[3])


def _rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """``comfy.ldm.flux.math.rope`` — the table builder of a stock ``EmbedND``."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2)


class MockPeEmbedder(torch.nn.Module):
    """A stock FLUX ``EmbedND`` (``ldm/flux/layers.py:17-27``): head-shared table."""

    def __init__(self, axes_dim=(16, 56, 56), theta=10000):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([_rope(ids[..., i], self.axes_dim[i], self.theta)
                         for i in range(ids.shape[-1])], dim=-3)
        return emb.unsqueeze(1)


class MockFluxDiT(torch.nn.Module):
    """The FLUX pieces AReA touches: ``pe_embedder``, ``img_in``, ``patch_size``.

    The class name is deliberately generic so ``src.model_detect`` resolves it as
    ``flux`` (``pe_embedder`` probe), not as a Krea-2 / Qwen-2.1 / Nunchaku model.
    """

    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.pe_embedder = MockPeEmbedder()
        self.img_in = torch.nn.Linear(2 * 2 * 16, HIDDEN)


def mock_flux_model() -> ModelPatcher:
    m = ModelPatcher()
    m.model.diffusion_model = MockFluxDiT()
    m.model.model_sampling = types.SimpleNamespace()
    return m


def _ids(txt_len: int, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """``[1, txt_len + n, 3]`` position ids: zero text ids, image row/col (as
    ``ldm/flux/model.py:338-342`` builds them)."""
    ids = torch.zeros(1, txt_len + int(rows.numel()), 3, dtype=torch.float32)
    ids[0, txt_len:, 1] = rows.float()
    ids[0, txt_len:, 2] = cols.float()
    return ids


def _grid(side: int) -> int:
    """Patch tokens per axis: the latent holds 2 latent pixels per patch token."""
    return side // 16


def _joint(b=1, seed=0, side=WIDTH):
    """Unrotated joint q/k/v ``[b, heads, T + h*w, d]`` plus the seam's ``pe``."""
    g = torch.Generator().manual_seed(seed)
    n = TXT_LEN + _grid(side) ** 2
    shape = (b, HEADS, n, HEAD_DIM)
    q = torch.randn(shape, generator=g)
    k = torch.randn(shape, generator=g)
    v = torch.randn(shape, generator=g)
    pos = torch.arange(_grid(side) ** 2)
    pe = MockPeEmbedder()(_ids(TXT_LEN, pos // _grid(side), pos % _grid(side)))
    return q, k, v, pe


def _opts(step: int, steps: int = STEPS) -> dict:
    """``transformer_options`` of one sampled forward (``samplers.py:316,1229``)."""
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": sigmas, "sigmas": sigmas[step]}


def _latent(b=1, side=WIDTH, channels=16):
    """The FLUX latent ``[b, c, H, W]`` (2 latent pixels per patch token)."""
    g = _grid(side)
    gen = torch.Generator().manual_seed(1234)
    return torch.randn(b, channels, g * 2, g * 2, generator=gen)


def _unet_wrapper(m):
    """The wrapper the patcher installed (mock and real ModelPatcher)."""
    try:
        return m.model_options["model_function_wrapper"]
    except (AttributeError, KeyError, TypeError):
        return getattr(m, "_unet_wrapper", None)


class SimulatedFlux:
    """One Forward of a FLUX-like transformer: ``img_in`` then a couple of blocks.

    ``img_in`` is called for real, so the AReA forward hook on
    ``diffusion_model.img_in`` fires exactly as it does in
    ``ldm/flux/model.py:169``; the attention calls replicate ``layers.py:244``.
    """

    def __init__(self, dm, blocks: int = 2, side: int = WIDTH):
        self.dm = dm
        self.blocks = blocks
        self.side = side
        self.seam = sys.modules["comfy.ldm.flux.layers"].attention
        self.calls: list = []
        self.layouts: list = []

    def __call__(self, q, k, v, pe, opts):
        out = self.seam(_Container(q), _Container(k), _Container(v), pe=pe, mask=None,
                        transformer_options=opts, preferred_attention=None)
        self.calls.append(out)
        return out

    def model_function(self, x, timestep, **c):
        opts = c["transformer_options"]
        # img_in(x) — the hook that supplies the Phase II detail map.  A model
        # whose img_in is not a Module has nothing to hook and nothing to call.
        img_in = getattr(self.dm, "img_in", None)
        if callable(img_in):
            self.img_out = img_in(torch.randn(x.shape[0], _grid(self.side) ** 2, 2 * 2 * 16))
        q, k, v, pe = _joint(b=x.shape[0], seed=self.seed, side=self.side)
        out = None
        for _ in range(self.blocks):
            out = self(q, k, v, pe, opts)
        return out

    seed = 0


def _run_forward(m, step, blocks=2, b=1, seed=0, sigmas=True, side=WIDTH):
    """Drive one sampled forward through the installed unet wrapper."""
    dm = m.model.diffusion_model
    sim = SimulatedFlux(dm, blocks, side)
    sim.seed = seed
    opts = _opts(step) if sigmas else {"sigmas": torch.tensor([1.0])}
    latent = _latent(b=b, side=side)
    cross = torch.randn(b, TXT_LEN, 77)
    params = {"input": latent, "timestep": torch.tensor([float(opts["sigmas"])]),
              "c": {"c_crossattn": cross, "transformer_options": opts}}
    out = _unet_wrapper(m)(sim.model_function, params)
    return sim, out


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flux_layers():
    """Install a mock ``comfy.ldm.flux.layers`` with a pristine ``attention``.

    AReA patches the *bound* symbol (``from .math import attention``), so the
    module must hold the original function as a plain attribute — the same shape
    as the real backend module.
    """
    saved = sys.modules.get("comfy.ldm.flux.layers")
    saved_parent_layers = getattr(sys.modules.get("comfy.ldm.flux"), "layers", None)
    parent = sys.modules.setdefault("comfy.ldm.flux", types.ModuleType("comfy.ldm.flux"))
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.attention = plain_flux_attention
    sys.modules["comfy.ldm.flux.layers"] = layers
    parent.layers = layers
    try:
        yield layers
    finally:
        sys.modules["comfy.ldm.flux.layers"] = saved
        parent.layers = saved_parent_layers
        if saved is None:
            sys.modules.pop("comfy.ldm.flux.layers", None)


@pytest.fixture(autouse=True)
def _clean_area_state():
    """AReA's module state is process-global (one-time warnings, seam refcount).

    Reset it around every test so the seam symbol and the warning keys never
    leak into the next one.
    """
    area._warned.clear()
    area._SEAM.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    yield
    area._warned.clear()
    area._SEAM.update(module=None, attr=None, orig=None, hook=None, consumers=0)
    area._AREA_RUNTIME = None


@pytest.fixture
def patched(flux_layers):
    """An AReA-patched FLUX model on a 1280 px canvas."""
    m = mock_flux_model()
    return apply_area_to_model(m, "flux", WIDTH, HEIGHT, phase1_steps=PHASE1, seed=7)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestAReASchema:
    def test_node_id_display_and_category(self):
        schema = AReA.define_schema()
        assert schema.node_id == "AReA"
        assert schema.display_name == "AReA"
        assert schema.category == "WMNodes/image"

    def test_inputs_match_the_reference_options_table(self):
        schema = AReA.define_schema()
        inputs = {i.name: i for i in schema.inputs}
        assert list(inputs) == ["model", "width", "height", "model_type", "enable_area",
                                "phase1_steps", "pooling", "halo", "bridge_ratio", "seed"]
        assert inputs["model_type"].kwargs["options"] == ["auto", "flux"]
        assert inputs["phase1_steps"].kwargs["default"] == PHASE1      # backbones/flux.py:16
        assert inputs["pooling"].kwargs["options"] == ["subsample", "mean", "medoid", "random"]
        assert inputs["pooling"].kwargs["default"] == "subsample"
        assert inputs["halo"].kwargs["default"] == 8
        assert inputs["bridge_ratio"].kwargs["default"] == 0.5
        assert inputs["enable_area"].kwargs["default"] is True

    def test_outputs_patched_model(self):
        schema = AReA.define_schema()
        assert [o.display_name for o in schema.outputs] == ["Patched Model"]

    def test_validate_inputs_rejects_non_multiple_of_16(self):
        message = AReA.validate_inputs(1208, 1280)
        assert isinstance(message, str)
        assert "multiples of 16" in message                # VAE 8 x patch 2
        assert "1208x1280" in message

    def test_validate_inputs_accepts_multiple_of_16(self):
        assert AReA.validate_inputs(1280, 1280) is True

    def test_execute_returns_the_patched_model(self, flux_layers):
        out = AReA.execute(mock_flux_model(), WIDTH, HEIGHT, "auto", True)
        model = out[0]
        assert model._area_config is not None
        assert model._area_config.phase1_steps == PHASE1
        restore_area_attention_hook(model)


# ---------------------------------------------------------------------------
# Apply gates
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestApplyGates:
    def test_non_flux_raises_actionable_error(self, flux_layers):
        class MockQwen21(MockFluxDiT):
            pass

        MockQwen21.__name__ = "QwenImage21Transformer2DModel"
        m = mock_flux_model()
        m.model.diffusion_model = MockQwen21()
        with pytest.raises(ValueError, match="FLUX only"):
            apply_area_to_model(m, "auto", WIDTH, HEIGHT)
        # the message names the offender and the remedy (house rule)
        with pytest.raises(ValueError, match="qwen21"):
            apply_area_to_model(m, "auto", WIDTH, HEIGHT)
        with pytest.raises(ValueError, match="SPA / DyPE / SEGA"):
            apply_area_to_model(m, "auto", WIDTH, HEIGHT)

    def test_nunchaku_warns_and_returns_the_clone_unchanged(self, flux_layers, caplog):
        source = mock_flux_model()
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            m = apply_area_to_model(source, "nunchaku", WIDTH, HEIGHT)
        assert m is not source                       # still a clone ...
        assert m._area_config is None                # ... but unpatched
        assert sys.modules["comfy.ldm.flux.layers"].attention is plain_flux_attention
        assert _unet_wrapper(m) is None
        assert any("Nunchaku" in rec.message for rec in caplog.records)

    def test_disabled_enable_flag_patches_nothing(self, flux_layers):
        m = apply_area_to_model(mock_flux_model(), "flux", WIDTH, HEIGHT, enable_area=False)
        assert m._area_config is not None
        assert sys.modules["comfy.ldm.flux.layers"].attention is plain_flux_attention
        assert _unet_wrapper(m) is None

    def test_below_native_installs_no_seam(self, flux_layers, caplog):
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            m = apply_area_to_model(mock_flux_model(), "flux", 1024, 1024)
        # 64x64 grid = the native extent: attention would be replaced by itself
        assert sys.modules["comfy.ldm.flux.layers"].attention is plain_flux_attention
        assert _unet_wrapper(m) is None
        assert any("provable no-op" in rec.message for rec in caplog.records)

    def test_geometry_without_a_common_divisor_raises_at_apply(self, flux_layers):
        # 128 x 129 patches (2048 x 2064 px) share no divisor >= 2, so no
        # pooling factor tiles the grid exactly (schedule.py:39-48).
        with pytest.raises(ValueError, match="no common divisor >= 2"):
            apply_area_to_model(mock_flux_model(), "flux", 2048, 2064)

    def test_phase1_steps_and_pooling_are_clamped(self, patched):
        assert patched._area_config.phase1_steps == PHASE1
        assert patched._area_config.seed == 7
        # an unknown pooling silently falls back to the reference default
        weird = apply_area_to_model(mock_flux_model(), "flux", WIDTH, HEIGHT, pooling="nope")
        assert weird._area_config.pooling == "subsample"
        restore_area_attention_hook(weird)

    def test_carry_state_copies_the_config_to_a_clone(self, patched):
        """``ModelPatcher.clone()`` drops custom attrs; the carry puts them back."""
        bare = patched.clone()
        assert not hasattr(bare, "_area_config")      # the clone() trap
        area._area_carry_state(patched, bare)
        assert bare._area_config == patched._area_config
        restore_area_attention_hook(patched)

    def test_double_apply_shares_one_seam(self, patched, flux_layers):
        """A second AReA apply re-uses the seam (ref-counted), never wraps ours."""
        first_symbol = flux_layers.attention
        second = apply_area_to_model(patched, "flux", WIDTH, HEIGHT, seed=99)
        assert flux_layers.attention is first_symbol            # not re-wrapped
        assert area._SEAM["consumers"] == 2
        assert second._area_config.seed == 99                    # newest apply wins
        restore_area_attention_hook(second)
        assert area._SEAM["consumers"] == 1
        assert flux_layers.attention is first_symbol
        restore_area_attention_hook(patched)
        assert flux_layers.attention is plain_flux_attention

    def test_restore_removes_the_seam_and_the_hook(self, patched, flux_layers):
        dm = patched.model.diffusion_model
        assert area._SEAM["consumers"] == 1
        assert area._SEAM["hook"] is not None
        restore_area_attention_hook(patched)
        assert flux_layers.attention is plain_flux_attention
        assert area._SEAM == {"module": None, "attr": None, "orig": None,
                              "hook": None, "consumers": 0}
        assert dm._forward_hooks == {}
        assert patched._area_config is None
        # idempotent
        restore_area_attention_hook(patched)


# ---------------------------------------------------------------------------
# Seam conformance (layers.py:244)
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeamConformance:
    def test_plain_attention_delegates_bit_exact(self, patched, flux_layers):
        """No forward of ours -> the wrapper is transparent, to the last bit."""
        q, k, v, pe = _joint()
        opts = _opts(0)
        wrapped = flux_layers.attention(_Container(q), _Container(k), _Container(v), pe=pe,
                                        mask=None, transformer_options=opts, preferred_attention=None)
        expected = plain_flux_attention(_Container(q.clone()), _Container(k.clone()),
                                        _Container(v.clone()), pe=pe, mask=None,
                                        transformer_options=opts, preferred_attention=None)
        assert torch.equal(wrapped, expected)
        assert wrapped.shape == (1, TXT_LEN + GRID_H * GRID_W, HEADS * HEAD_DIM)

    def test_plain_step_delegates_and_uses_the_models_own_attention(self, patched, caplog):
        """No ``sample_sigmas`` (exotic sampler) -> plain attention, not a guess."""
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            sim, out = _run_forward(patched, step=0, sigmas=False)
        q, k, v, pe = _joint(seed=0)
        expected = plain_flux_attention(_Container(q), _Container(k), _Container(v), pe=pe,
                                        mask=None, transformer_options=None,
                                        preferred_attention=None)
        assert torch.equal(sim.calls[0], expected)
        assert any("sample_sigmas" in rec.message for rec in caplog.records)
        assert area._AREA_RUNTIME is None        # context cleared after the forward

    def test_containers_are_consumed_exactly_once(self, patched, flux_layers):
        """The v2.19.1 crash class: a taken container is never re-wrapped."""
        q, k, v, pe = _joint()
        qc, kc, vc = _Container(q), _Container(k), _Container(v)
        flux_layers.attention(qc, kc, vc, pe=pe, mask=None,
                              transformer_options=_opts(0), preferred_attention=None)
        for container in (qc, kc, vc):
            assert container.tensor is None
            with pytest.raises(RuntimeError, match="already been consumed"):
                container.take()

    def test_coarse_step_pools_keys_before_rope(self, patched, monkeypatch):
        """Phase I: K/V shrink to T + (h/f)(w/f) while the output stays full."""
        seen = {}
        real = area.pool_keys_values

        def spy(key, value, txt_len, h, w, f, pooling, **kw):
            out = real(key, value, txt_len, h, w, f, pooling, **kw)
            seen["k_len"] = out[0].shape[2]
            seen["f"] = f
            seen["txt_len"] = txt_len
            seen.setdefault("blocks", []).append(kw.get("block"))
            return out

        monkeypatch.setattr(area, "pool_keys_values", spy)
        sim, out = _run_forward(patched, step=0)          # step 0 -> Phase I
        assert seen["f"] == 2
        assert seen["txt_len"] == TXT_LEN
        assert seen["k_len"] == TXT_LEN + (GRID_H // 2) * (GRID_W // 2)
        assert seen["blocks"] == [0, 1]                   # per-forward block ordinal
        # queries keep full resolution, and the result is comfy-shaped
        assert out.shape == (1, TXT_LEN + GRID_H * GRID_W, HEADS * HEAD_DIM)
        assert sim.calls[0].shape == out.shape

    def test_coarse_ramp_rungs_reach_the_seam(self, flux_layers, monkeypatch):
        """Phase I's ramp is multi-rung at 2560 px (f0 = 4 -> 2 mid-phase), and
        the node must hand each step its own factor through the seam.  The
        80x80 canvas' ramp is constant at 2, so a rung change is only
        observable through the seam on the big grid."""
        seen = []
        real = area.pool_keys_values

        def spy(key, value, txt_len, h, w, f, pooling, **kw):
            out = real(key, value, txt_len, h, w, f, pooling, **kw)
            seen.append({"step": kw.get("step"), "block": kw.get("block"), "f": f,
                         "k_len": out[0].shape[2]})
            return out

        monkeypatch.setattr(area, "pool_keys_values", spy)
        m = apply_area_to_model(mock_flux_model(), "flux", BIG_WIDTH, BIG_HEIGHT,
                                phase1_steps=PHASE1, seed=7)
        try:
            for step in (0, 3, 4, PHASE1 - 1):     # early, mid, post-rung, last
                _run_forward(m, step, blocks=1, side=BIG_WIDTH)
        finally:
            restore_area_attention_hook(m)
        # one pooling per forward (blocks=1): every step drew its own rung
        assert [e["step"] for e in seen] == [0, 3, 4, PHASE1 - 1]
        ramp = pooling_ramp(initial_factor(BIG_GRID, BIG_GRID, area.NATIVE_PATCHES),
                            PHASE1, BIG_GRID, BIG_GRID)
        assert len(set(ramp)) >= 2                        # the ramp really rungs
        assert [e["f"] for e in seen] == [ramp[s] for s in (0, 3, 4, PHASE1 - 1)]
        assert len({e["f"] for e in seen}) >= 2           # ... and the seam saw both
        for entry in seen:                                # each rung pools to g/f per axis
            assert entry["k_len"] == TXT_LEN + (BIG_GRID // entry["f"]) ** 2

    def test_coarse_step_advances_the_block_ordinal(self, patched, monkeypatch):
        """Every attention call of a forward draws its own random-pooling stream."""
        blocks = []
        real = area.pool_keys_values

        def spy(key, value, txt_len, h, w, f, pooling, **kw):
            blocks.append(kw.get("block"))
            return real(key, value, txt_len, h, w, f, pooling, **kw)

        monkeypatch.setattr(area, "pool_keys_values", spy)
        _run_forward(patched, step=0, blocks=3)
        assert blocks == [0, 1, 2]

    def test_fine_step_runs_the_windowed_path(self, patched, monkeypatch):
        """Phase II: windowed attention, merged over the whole canvas."""
        seen = {}
        real = area.windowed_attention

        def spy(q_txt, k_txt, v_txt, q_img, k_img, v_img, k_unrot, layout):
            seen["txt"] = q_txt.shape[2]
            seen["img"] = q_img.shape[2]
            seen["k_unrot"] = k_unrot.shape
            seen["registers"] = layout.registers
            seen["windows"] = len(layout.windows)
            return real(q_txt, k_txt, v_txt, q_img, k_img, v_img, k_unrot, layout)

        monkeypatch.setattr(area, "windowed_attention", spy)
        sim, out = _run_forward(patched, step=STEPS - 1)   # last step -> Phase II
        assert seen["txt"] == TXT_LEN
        assert seen["img"] == GRID_H * GRID_W
        assert seen["k_unrot"] == (1, HEADS, GRID_H * GRID_W, HEAD_DIM)
        assert seen["windows"] >= 1
        assert seen["registers"] >= 0
        assert out.shape == (1, TXT_LEN + GRID_H * GRID_W, HEADS * HEAD_DIM)

    def test_fine_step_reads_the_img_in_detail_map(self, patched, monkeypatch):
        """The Phase II detail map comes from the img_in hook, not from the keys."""
        captured = {}
        real = area.detail_map

        def spy(hidden_states, h, w):
            captured["shape"] = tuple(hidden_states.shape)
            return real(hidden_states, h, w)

        monkeypatch.setattr(area, "detail_map", spy)
        _run_forward(patched, step=STEPS - 1)
        assert captured["shape"] == (1, GRID_H * GRID_W, HIDDEN)

    def test_fine_step_falls_back_to_the_projected_keys(self, flux_layers, caplog):
        """Without an img_in hook the detail map falls back, with one warning."""

        class MockNoImgIn(MockFluxDiT):
            """A model whose img_in is not a Module (nothing to hook)."""

            def __init__(self):
                super().__init__()
                del self._modules["img_in"]
                self.img_in = types.SimpleNamespace()

        m = mock_flux_model()
        m.model.diffusion_model = MockNoImgIn()
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            out = apply_area_to_model(m, "flux", WIDTH, HEIGHT)
        assert area._SEAM["hook"] is None
        assert any("img_in could not be hooked" in rec.message for rec in caplog.records)
        patched = out
        sim, out = _run_forward(patched, step=STEPS - 1)
        assert out.shape == (1, TXT_LEN + GRID_H * GRID_W, HEADS * HEAD_DIM)
        restore_area_attention_hook(patched)

    def test_unknown_geometry_delegates_instead_of_failing(self, patched, flux_layers):
        """A reference-token path (more image tokens than the grid) must degrade."""
        dm = patched.model.diffusion_model
        sim = SimulatedFlux(dm)
        opts = _opts(0)
        params = {"input": _latent(), "timestep": torch.tensor([float(opts["sigmas"])]),
                  "c": {"c_crossattn": torch.randn(1, TXT_LEN, 77), "transformer_options": opts}}
        # an extra image token: the stream no longer matches the latent's grid
        big_q, big_k, big_v, pe = _joint()
        extra = torch.randn(1, HEADS, 1, HEAD_DIM)
        q = torch.cat([big_q, extra], dim=2)
        k = torch.cat([big_k, extra], dim=2)
        v = torch.cat([big_v, extra], dim=2)
        # the model's own rope table grows with the stream (ldm/flux/model.py:194)
        rows = torch.arange(GRID_H * GRID_W + 1)
        pe = MockPeEmbedder()(_ids(TXT_LEN, rows // GRID_W, rows % GRID_W))

        def model_function(x, timestep, **c):
            return sim(q, k, v, pe, c["transformer_options"])

        _unet_wrapper(patched)(model_function, params)
        expected = plain_flux_attention(_Container(q.clone()), _Container(k.clone()),
                                        _Container(v.clone()), pe=pe, mask=None,
                                        transformer_options=opts, preferred_attention=None)
        assert torch.equal(sim.calls[0], expected)
        assert sim.calls[0].shape == (1, TXT_LEN + GRID_H * GRID_W + 1, HEADS * HEAD_DIM)
