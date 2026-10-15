"""AReA determinism: CFG layout sharing, seed and step streams, no global RNG.

Phase 3 conformance for ``src/area.py``.  ComfyUI runs cond and uncond through
ONE batched forward (``samplers.py:309``), so everything AReA derives per
forward — the Phase II tiling, the Halo rings, the bridge registers, the random
pooling of Phase I — is drawn once from a keyed stream
(``step_rng(seed, step, salt)``) and shared by both halves.  These tests pin
that down with the same mock seam the node tests use (GPU-free, unit-marked).

Two canvases appear: 1280 px (80x80 patches, just past the native extent, where
the four even windows carry no Halo and no bridge) and 2560 px, the first size
whose tiling actually consults the step's random stream.  The harness is
deliberately self-contained: importing the node module's mocks would couple two
files that CI must be able to run independently.
"""
from __future__ import annotations

import random
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import src.area as area
from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_attn import apply_rope1

try:
    from comfy.model_patcher import ModelPatcher
except ImportError:  # pragma: no cover - mock env always provides it
    from tests.conftest import MockModelPatcher as ModelPatcher

WIDTH, HEIGHT = 1280, 1280
BIG_WIDTH, BIG_HEIGHT = 2560, 2560
TXT_LEN = 32
HEADS, HEAD_DIM = 2, 128
HIDDEN = HEADS * HEAD_DIM
STEPS = 28                          # the reference's FLUX total
PHASE1 = 8
SEED = 7


def _grid(side: int) -> int:
    """Patch tokens per axis: the latent holds 2 latent pixels per patch token."""
    return side // 16


# ---------------------------------------------------------------------------
# Mocks (same shape as the node tests; see tests/test_area_node.py)
# ---------------------------------------------------------------------------


class _Container:
    """``AttentionTensorContainer`` (``ldm/modules/attention.py:193-209``)."""

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
    """``comfy.ldm.flux.math.attention`` — what AReA wraps."""
    if hasattr(q, "take"):
        q, k, v = q.take(), k.take(), v.take()
    if pe is not None:
        q, k = apply_rope1(q, pe), apply_rope1(k, pe)
    heads = q.shape[1]
    out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], heads * out.shape[3])


def _rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """``comfy.ldm.flux.math.rope`` — a stock ``EmbedND``'s table builder."""
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2)


class MockPeEmbedder(torch.nn.Module):
    """A stock FLUX ``EmbedND`` (``ldm/flux/layers.py:17-27``)."""

    def __init__(self, axes_dim=(16, 56, 56), theta=10000):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat([_rope(ids[..., i], self.axes_dim[i], self.theta)
                         for i in range(ids.shape[-1])], dim=-3)
        return emb.unsqueeze(1)


class MockFluxDiT(torch.nn.Module):
    """The FLUX pieces AReA touches: ``pe_embedder``, ``img_in``, ``patch_size``."""

    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.pe_embedder = MockPeEmbedder()
        self.img_in = torch.nn.Linear(2 * 2 * 16, HIDDEN)


def mock_flux_model(weight_seed: int = 1234) -> ModelPatcher:
    """A mock FLUX patcher.  ``weight_seed`` fixes ``img_in``'s init draw, so two
    models built for a comparison start from the same weights."""
    torch.manual_seed(weight_seed)
    m = ModelPatcher()
    m.model.diffusion_model = MockFluxDiT()
    m.model.model_sampling = types.SimpleNamespace()
    return m


def _ids(txt_len: int, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """``[1, txt_len + n, 3]`` position ids (``ldm/flux/model.py:338-342``)."""
    ids = torch.zeros(1, txt_len + int(rows.numel()), 3, dtype=torch.float32)
    ids[0, txt_len:, 1] = rows.float()
    ids[0, txt_len:, 2] = cols.float()
    return ids


def _joint(b=1, seed=0, side=WIDTH):
    """Unrotated joint q/k/v ``[b, heads, T + h*w, d]`` plus the seam's ``pe``.

    Every draw goes through a local generator, so a repeat run feeds the forward
    byte-identical inputs — the precondition for a bit-exactness claim.
    """
    g = torch.Generator().manual_seed(seed)
    rows = cols = _grid(side)
    n = TXT_LEN + rows * cols
    shape = (b, HEADS, n, HEAD_DIM)
    q = torch.randn(shape, generator=g)
    k = torch.randn(shape, generator=g)
    v = torch.randn(shape, generator=g)
    pos = torch.arange(rows * cols)
    pe = MockPeEmbedder()(_ids(TXT_LEN, pos // cols, pos % cols))
    return q, k, v, pe


def _opts(step: int, steps: int = STEPS) -> dict:
    """``transformer_options`` of one sampled forward (``samplers.py:316,1229``)."""
    sigmas = torch.linspace(1.0, 0.0, steps + 1)
    return {"sample_sigmas": sigmas, "sigmas": sigmas[step]}


def _latent(b=1, side=WIDTH):
    g = _grid(side)
    gen = torch.Generator().manual_seed(1234)
    return torch.randn(b, 16, g * 2, g * 2, generator=gen)


def _unet_wrapper(m):
    try:
        return m.model_options["model_function_wrapper"]
    except (AttributeError, KeyError, TypeError):
        return getattr(m, "_unet_wrapper", None)


class SimulatedFlux:
    """One forward of a FLUX-like transformer, everything generator-seeded.

    ``img_in`` runs for real so the AReA forward hook
    (``src/area.py:_area_img_in_hook``) fires exactly as in
    ``ldm/flux/model.py:169``; each block calls the seam the way
    ``ldm/flux/layers.py:244`` does.
    """

    def __init__(self, dm, blocks: int = 2, side: int = WIDTH):
        self.dm = dm
        self.blocks = blocks
        self.side = side
        self.seam = sys.modules["comfy.ldm.flux.layers"].attention
        self.calls: list = []
        self.seed = 0

    def __call__(self, q, k, v, pe, opts):
        out = self.seam(_Container(q), _Container(k), _Container(v), pe=pe, mask=None,
                        transformer_options=opts, preferred_attention=None)
        self.calls.append(out)
        return out

    def model_function(self, x, timestep, **c):
        opts = c["transformer_options"]
        g = torch.Generator().manual_seed(self.seed)
        grid = _grid(self.side)
        self.img_out = self.dm.img_in(torch.randn(x.shape[0], grid * grid, 2 * 2 * 16,
                                                  generator=g))
        q, k, v, pe = _joint(b=x.shape[0], seed=self.seed, side=self.side)
        out = None
        for _ in range(self.blocks):
            out = self(q, k, v, pe, opts)
        return out


def _run(m, step, blocks=2, b=1, seed=SEED, side=WIDTH):
    """Drive one sampled forward through the installed unet wrapper."""
    sim = SimulatedFlux(m.model.diffusion_model, blocks, side)
    sim.seed = seed
    opts = _opts(step)
    cross = torch.randn(b, TXT_LEN, 77, generator=torch.Generator().manual_seed(4242))
    params = {"input": _latent(b=b, side=side), "timestep": torch.tensor([float(opts["sigmas"])]),
              "c": {"c_crossattn": cross, "transformer_options": opts}}
    out = _unet_wrapper(m)(sim.model_function, params)
    return sim, out


def _fine_signature(layout) -> tuple:
    """Everything that identifies one step's Phase II layout.

    ``Window`` is a plain (unhashable) dataclass, so the comparison is by
    sequence, never by set.
    """
    return (layout.h, layout.w, tuple(layout.windows), int(layout.registers))


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
    """AReA's module state is process-global: reset it around every test."""
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
    return apply_area_to_model(m, "flux", WIDTH, HEIGHT, phase1_steps=PHASE1, seed=SEED)


def _record_fine(monkeypatch):
    """Spy on every Phase II layout build; returns the list of signatures."""
    seen: list = []
    real = area.build_fine_layout

    def spy(windows, h, w, halo, registers, seed, step, dtype, device):
        layout = real(windows, h, w, halo, registers, seed, step, dtype, device)
        seen.append({"signature": _fine_signature(layout), "seed": seed, "step": step,
                     "halo": halo, "registers": registers})
        return layout

    monkeypatch.setattr(area, "build_fine_layout", spy)
    return seen


def _record_pooling(monkeypatch):
    """Spy on every Phase I pooling; returns the recorded pooled keys."""
    seen: list = []
    real = area.pool_keys_values

    def spy(key, value, txt_len, h, w, f, pooling, **kw):
        out = real(key, value, txt_len, h, w, f, pooling, **kw)
        seen.append({"pooling": pooling, "seed": kw.get("seed"), "step": kw.get("step"),
                     "block": kw.get("block"), "k": out[0].clone(), "k_len": int(out[0].shape[2])})
        return out

    monkeypatch.setattr(area, "pool_keys_values", spy)
    return seen


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestDeterminism:
    def test_a_repeat_forward_is_bit_identical(self, patched):
        """Same model, same inputs, same step -> the same bits, phase I and II."""
        for step in (0, STEPS - 1):                      # coarse, then fine
            sim_a, out_a = _run(patched, step)
            sim_b, out_b = _run(patched, step)
            assert out_a.shape == out_b.shape
            assert torch.equal(out_a, out_b)
            assert len(sim_a.calls) == len(sim_b.calls)
            for call_a, call_b in zip(sim_a.calls, sim_b.calls):
                assert torch.equal(call_a, call_b)

    def test_cond_and_uncond_share_one_detail_map_and_layout(self, patched, monkeypatch):
        """One batched forward (``samplers.py:309``): the map and the layout are
        drawn ONCE for both halves, never once per half."""
        maps: list = []
        real_map = area.detail_map

        def map_spy(hidden_states, h, w):
            maps.append(tuple(hidden_states.shape))
            return real_map(hidden_states, h, w)

        monkeypatch.setattr(area, "detail_map", map_spy)
        layouts = _record_fine(monkeypatch)
        _run(patched, STEPS - 1, b=2)                       # cond + uncond in one batch
        assert maps == [(2, _grid(WIDTH) ** 2, HIDDEN)]     # batch-reduced once
        assert layouts                                     # one layout, both halves
        assert all(entry["signature"] == layouts[0]["signature"] for entry in layouts)

    def test_the_layout_is_rebuilt_once_per_forward(self, patched, monkeypatch):
        """The layout is per-forward state: one build every forward, and the
        previous forward's layout never leaks into this one."""
        layouts = _record_fine(monkeypatch)
        _run(patched, STEPS - 1)
        _run(patched, STEPS - 1)
        assert [entry["step"] for entry in layouts] == [STEPS - 1, STEPS - 1]
        assert layouts[0]["signature"] == layouts[1]["signature"]
        assert layouts[0]["seed"] == SEED

    def test_a_bigger_canvas_consults_the_step_stream(self, flux_layers, monkeypatch):
        """2560 px is the first size whose tiling draws from the step's stream,
        so a new step gets a new layout (and the seed a new one again)."""
        layouts = _record_fine(monkeypatch)
        for step in (STEPS - 2, STEPS - 1):
            m = apply_area_to_model(mock_flux_model(), "flux", BIG_WIDTH, BIG_HEIGHT,
                                    phase1_steps=PHASE1, seed=SEED)
            _run(m, step, side=BIG_WIDTH)
            restore_area_attention_hook(m)
        assert len(layouts) == 2
        assert layouts[0]["signature"] != layouts[1]["signature"]

        seeded = _record_fine(monkeypatch)
        for seed in (SEED, SEED + 1):
            m = apply_area_to_model(mock_flux_model(), "flux", BIG_WIDTH, BIG_HEIGHT,
                                    phase1_steps=PHASE1, seed=seed)
            _run(m, STEPS - 1, side=BIG_WIDTH)
            restore_area_attention_hook(m)
        assert len(seeded) == 2
        assert seeded[0]["signature"] != seeded[1]["signature"]

    def test_subsample_pooling_and_the_energy_map_ignore_the_seed(self, flux_layers, monkeypatch):
        """The seed must not move anything the canvas itself decides: the
        detail map and the default (subsample) pooling."""
        maps, pooled = {}, {}
        for seed in (SEED, SEED + 1):
            seen_maps: list = []
            real_map = area.detail_map

            def map_spy(hidden_states, h, w, _real=real_map, _out=seen_maps):
                energy = _real(hidden_states, h, w)
                _out.append((h, w, float(energy.sum())))
                return energy

            monkeypatch.setattr(area, "detail_map", map_spy)
            pooling = _record_pooling(monkeypatch)
            m = apply_area_to_model(mock_flux_model(), "flux", WIDTH, HEIGHT,
                                    phase1_steps=PHASE1, seed=seed)
            _run(m, 0)                                    # Phase I
            _run(m, STEPS - 1)                            # Phase II (the detail map)
            maps[seed] = seen_maps[0]
            pooled[seed] = pooling[0]["k"]
            restore_area_attention_hook(m)
        assert maps[SEED] == maps[SEED + 1]
        assert torch.equal(pooled[SEED], pooled[SEED + 1])

    def test_the_seed_moves_a_selecting_pooling(self, flux_layers, monkeypatch):
        """``random`` pooling is seed-keyed: a different seed represents each
        block by a different token (same seed, same choice)."""
        pooled = {}
        for seed in (SEED, SEED + 1):
            seen = _record_pooling(monkeypatch)
            m = apply_area_to_model(mock_flux_model(), "flux", WIDTH, HEIGHT,
                                    phase1_steps=PHASE1, pooling="random", seed=seed)
            _run(m, 0, blocks=1)
            assert [entry["block"] for entry in seen] == [0]
            pooled[seed] = seen[0]["k"]
            restore_area_attention_hook(m)
        assert not torch.equal(pooled[SEED], pooled[SEED + 1])

    def test_each_block_draws_its_own_pooling_stream(self, flux_layers, monkeypatch):
        """Within one step every block draws its own stream (same seed would
        otherwise collapse the blocks onto one choice)."""
        seen = _record_pooling(monkeypatch)
        m = apply_area_to_model(mock_flux_model(), "flux", WIDTH, HEIGHT,
                                phase1_steps=PHASE1, pooling="random", seed=SEED)
        _run(m, 0, blocks=3)
        assert [entry["block"] for entry in seen] == [0, 1, 2]
        assert [entry["k_len"] for entry in seen] == [TXT_LEN + (_grid(WIDTH) // 2) ** 2] * 3
        for a, b in zip(seen, seen[1:]):
            assert not torch.equal(a["k"], b["k"])
        restore_area_attention_hook(m)

    def test_block_ordinals_reset_every_forward(self, patched, monkeypatch):
        """Each forward restarts the block ordinal, so a block's stream repeats
        across forwards and differs within one."""
        seen = _record_pooling(monkeypatch)
        _run(patched, 0)
        _run(patched, 0)
        assert [entry["block"] for entry in seen] == [0, 1, 0, 1]
        assert [entry["step"] for entry in seen] == [0, 0, 0, 0]

    def test_no_global_rng_is_read(self, patched, flux_layers, monkeypatch):
        """AReA draws only from keyed streams — the process RNG is never touched,
        so a sampler's own seeding stays reproducible.  A draw that names its
        own generator/seed is allowed; anything else is a bug.  Both phases run
        on both canvases: 2560 px is the first size whose Phase II tiling draws
        from the step's stream at all (per-side Halo widths and row/column
        reaches), so only the big canvas puts the halo draw inside the guard —
        the 80x80 layout carries no Halo and no bridge."""
        def guard(name, original, needs_source: bool):
            def check(*args, **kwargs):
                if needs_source and not ("generator" in kwargs or "seed" in kwargs):
                    raise AssertionError(f"AReA read the global RNG: {name}")
                if not needs_source:
                    raise AssertionError(f"AReA read the global RNG: {name}")
                return original(*args, **kwargs)

            return check

        for name in ("randn", "rand", "randint", "randperm", "normal"):
            monkeypatch.setattr(torch, name, guard(f"torch.{name}", getattr(torch, name), True),
                                raising=False)
        for name in ("random", "rand", "randint", "choice", "uniform", "randn"):
            monkeypatch.setattr(np.random, name, guard(f"np.random.{name}", None, False),
                                raising=False)
        for name in ("random", "randint", "choice", "uniform", "randrange"):
            monkeypatch.setattr(random, name, guard(f"random.{name}", None, False), raising=False)

        sim, out = _run(patched, 0)                         # Phase I (selecting poolings)
        assert sim.calls and out.shape[-1] == HIDDEN
        sim, out = _run(patched, STEPS - 1)                 # Phase II (no Halo yet)
        assert sim.calls and out.shape[-1] == HIDDEN

        # 2560 px: Phase I pools at f=4, and Phase II really draws — Halo rings,
        # their per-side widths/reaches and the jittered window placement all
        # come from the keyed step stream.
        big = apply_area_to_model(mock_flux_model(), "flux", BIG_WIDTH, BIG_HEIGHT,
                                  phase1_steps=PHASE1, seed=SEED)
        try:
            sim, out = _run(big, 0, side=BIG_WIDTH)         # Phase I on the big grid
            assert sim.calls and out.shape[-1] == HIDDEN
            sim, out = _run(big, STEPS - 1, side=BIG_WIDTH)  # Phase II + Halo draws
            assert sim.calls and out.shape[-1] == HIDDEN
        finally:
            restore_area_attention_hook(big)

    def test_runtime_is_cleared_after_every_forward(self, patched):
        """The per-forward context never leaks (SPA discipline, spa.py:1385-1476)."""
        assert area._AREA_RUNTIME is None
        _run(patched, 0)
        assert area._AREA_RUNTIME is None
        _run(patched, STEPS - 1)
        assert area._AREA_RUNTIME is None

        class Boom(RuntimeError):
            pass

        def exploding(x, timestep, **c):
            raise Boom("a backend that fails mid-forward")

        params = {"input": _latent(), "timestep": torch.tensor([1.0]),
                  "c": {"c_crossattn": torch.randn(1, TXT_LEN, 77), "transformer_options": _opts(0)}}
        with pytest.raises(Boom):
            _unet_wrapper(patched)(exploding, params)
        assert area._AREA_RUNTIME is None                   # the finally clause

    def test_plain_steps_stay_bit_exact_to_the_original(self, patched):
        """A step with no resolvable sigma must not perturb the sampler's output."""
        q, k, v, pe = _joint(seed=SEED)
        opts = {"sigmas": torch.tensor([1.0])}             # no sample_sigmas
        wrapped = sys.modules["comfy.ldm.flux.layers"].attention(
            _Container(q), _Container(k), _Container(v), pe=pe, mask=None,
            transformer_options=opts, preferred_attention=None)
        expected = plain_flux_attention(_Container(q), _Container(k), _Container(v), pe=pe,
                                        mask=None, transformer_options=opts,
                                        preferred_attention=None)
        assert torch.equal(wrapped, expected)
