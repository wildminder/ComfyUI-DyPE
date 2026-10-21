"""AReA × Z-Image — the seam wiring (plan §4/§7-t4).

Everything here is mock-based and GPU-free, per project convention.  The mock
mirrors the real call convention exactly as the rest of the suite does:
``comfy/ldm/lumina/model.py:14`` binds ``optimized_attention_masked`` into the
backend module's own namespace and ``:182`` calls it

    output = optimized_attention_masked(xq, xk, xv, self.n_local_heads, x_mask,
                                        skip_reshape=True,
                                        transformer_options=transformer_options,
                                        preferred_attention=self.comfy_attention)

with q/k/v as ``AttentionTensorContainer``s **already rotated** by the model's own
rope (``:159-173``) and NO rope table, and ``self.out(output)`` at ``:184``
reading a sequence-first ``[b, n, heads*d]`` answer.  The mock therefore lives in
a ``comfy.ldm.lumina.model`` module registered in ``sys.modules`` (the precedent
at ``tests/test_spa_hook_krea2_zimage.py:57-66``), carries its own
``peek``/``take`` container (``tests/test_area_node.py:61-78``), and pins the
positional slots — a wrapper that mis-slots ``heads`` or ``mask`` fails silently,
which is what these tests exist to catch.

What this file pins:

* the install rebinds the *bound* symbol (not the defining module), ref-counts,
  and collides safely with SPA
* a plain-attention step (no locatable sigma table) delegates to the PRISTINE
  symbol, bit-exact, consuming each container once and never calling the engine; — the one symbol both packs patch for Z-Image;
* restore returns the pristine symbol on the last release and is idempotent;
* the Phase I switch: ``steps`` mode follows ``phase1_steps``, ``sigma`` mode
  follows ``sigma_prefix_count`` on the sampler's own table;
* the per-forward hooks: the rope recorder captures the model's two groups in
  call order and the ``noise_refiner`` hook feeds the detail map the real image
  tokens only, never the pad — registered on every refiner LAYER, because lumina
  iterates the ModuleList (``model.py:816``) instead of calling it;
* the computation: comfy-shaped output, RNG keyed by ``(seed, step)``, ONE
  layout shared by cond and uncond of a batched forward, and the Phase II text
  block spanning the whole padded caption block (``cap_len``, the prompt tokens
  plus the model's learned caption pad) with only the image pad zeroed;
* the FLUX seam is untouched by a Z-Image install, and vice versa.

Markers: @pytest.mark.unit
Accept (user-run): pytest tests/test_area_zimage_seam.py tests/test_area_node.py
                   tests/test_spa_hook_krea2_zimage.py
"""
from __future__ import annotations

import sys
import types

import pytest
import torch
import torch.nn.functional as F

import src.area as area
from src.area import apply_area_to_model, restore_area_attention_hook
from src.area_attn import apply_rope1, bridge_registers, windowed_step_cost
from src.area_zimage import sigma_prefix_count

try:
    from comfy.model_patcher import ModelPatcher
except ImportError:  # pragma: no cover - mock env always provides it
    from tests.conftest import MockModelPatcher as ModelPatcher

# A 1040 px canvas: a 65x65 patch grid, just past the native 64-patch extent, small
# enough that a whole sampled forward runs in well under a second on CPU.
WIDTH, HEIGHT = 1040, 1040
GRID = 65
# How many layers the refiner ModuleList holds — ``n_refiner_layers``
# (``comfy/ldm/lumina/model.py:453``, the ModuleList at ``:497-514``).
REFINER_LAYERS = 2
# A canvas that exercises the pad policy: 20 prompt tokens pad to a 32-token
# caption block, and a 65x65 (4225-token) image pads to 4256 — so both pad spans
# are non-empty (the same geometry ``tests/test_area_zimage.py`` uses).
TXT_LEN, CAP_LEN = 20, 32
IMG_TOKENS, IMG_PAD = GRID * GRID, 32 - (GRID * GRID) % 32      # 4225, 31
SEQ_LEN = CAP_LEN + IMG_TOKENS + IMG_PAD                        # 4288
# The one t-axis value every image token of the frame carries
# (``pos_ids_x(start_t=cap_len + 1 + f, ...)``, comfy/ldm/lumina/model.py:722).
IMG_T = float(CAP_LEN + 1)
HEADS, HEAD_DIM = 2, 32
HIDDEN = 64                        # the noise_refiner output's model dim
STEPS, PHASE1 = 28, 8              # the reference's FLUX total / Phase I count
F0 = 5                             # initial_factor(65, 65, 64) — see src/area_schedule.py:178


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


def base_lumina_attention(q, k, v, heads, mask=None, attn_precision=None,
                          skip_reshape=False, skip_output_reshape=False, **kw):
    """The pristine ``optimized_attention_masked`` — the symbol AReA wraps.

    ``attention_pytorch``'s convention (``ldm/modules/attention.py``): with
    ``skip_reshape=True`` the tensors are already heads-first and the output is
    reshaped back to ``[b, n, heads*dim]`` unless ``skip_output_reshape``.  The
    real backend receives q/k *rotated* and applies no rope of its own — the same
    convention the mock keeps.
    """
    if hasattr(q, "take"):
        q, k, v = q.take(), k.take(), v.take()
    out = F.scaled_dot_product_attention(q, k, v)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(out.shape[0], -1, heads * out.shape[3])
    return out


def _mock_rope(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """``comfy.ldm.flux.math.rope`` (``math.py:20-32``) — no comfy import.

    The ``(i j) -> i j`` rearrange is the row-major reshape of the stacked
    ``[cos, -sin, sin, cos]`` vector, so the 2x2 block is
    ``[[cos, -sin], [sin, cos]]`` — the block :func:`unrotate_table` inverts.
    """
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).to(dtype=torch.float32)


class MockEmbedND(torch.nn.Module):
    """A stock lumina ``EmbedND`` (``model.py:634`` over ``flux/layers.py:15-29``).

    ``ids`` ``[b, n, 3]`` -> ``[b, 1, n, head_dim // 2, 2, 2]``: one rope per axis
    (``axes_dim``) concatenated on the frequency axis.  ``theta``/``axes_dim`` are
    constructor arguments on purpose — the adapter reads them off the installed
    embedder and never assumes them (plan Q7: any ``rope_theta`` is supported).
    """

    def __init__(self, theta: float = 256.0, axes_dim=(8, 12, 12)):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    @property
    def head_dim(self) -> int:
        return sum(self.axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat(
            [_mock_rope(ids[..., i], self.axes_dim[i], self.theta)
             for i in range(ids.shape[-1])], dim=-3,
        )
        return emb.unsqueeze(1)


class MockNoiseRefiner(torch.nn.Module):
    """One ``NextDiT.noise_refiner`` block (``model.py:497-514``): image-only.

    Each refiner block runs attention over the IMAGE BLOCK ALONE through the very
    symbol AReA patches, so this mock calls the bound symbol once with a short
    image-only stream (cheap, and enough to prove the wrapper delegates it —
    the refiner stream is never the joint stream).  The block is TRANSFORMED
    deterministically — every token shifted by this layer's own ``shift``, a
    different constant per layer — so input != output and a layer's input is
    distinguishable from its output: the hook reads the layer's INPUT for the
    Phase II detail map, and a mock that returned the input unchanged could
    not tell the two sources apart.
    """

    def __init__(self, seam_getter, shift=1.0):
        super().__init__()
        self._seam = seam_getter
        self.shift = shift
        self.calls = 0

    def forward(self, x):
        b = x.shape[0]
        g = torch.Generator().manual_seed(11)
        q = torch.randn(b, HEADS, 8, HEAD_DIM, generator=g)
        self._seam()(_Container(q), _Container(q.clone()), _Container(q.clone()),
                     HEADS, None, skip_reshape=True, transformer_options={},
                     preferred_attention=None)
        self.calls += 1
        return x + self.shift


class NextDiT(torch.nn.Module):
    """A stock plain Z-Image diffusion model (``comfy/ldm/lumina/model.py:442``).

    The class name is load-bearing: ``src/model_detect``'s class-name discipline
    and ``resolve_area_arch``'s MRO walk both read it (``src/area_arch.py:142``),
    exactly as comfy's ``Lumina2.unet_model`` (``model_base.py:1510``) binds it.
    ``rope_embedder`` is the auto probe (``src/model_detect.py:74-75``);
    ``pad_tokens_multiple`` is the model's own pad knob (``model.py:484``); no
    ``masked_pad_multiple`` — that one is the Ming-Image marker.
    """

    def __init__(self):
        super().__init__()
        self.patch_size = 2
        self.pad_tokens_multiple = 32
        self.masked_pad_multiple = None
        self.rope_embedder = MockEmbedND()
        # An nn.ModuleList, like the real one (``model.py:497-514``), and for the
        # same reason the mock forward LOOPS over it: a forward hook on the
        # container never fires when the model iterates its layers
        # (``model.py:816``).  A plain module here would hide that bug.  Each
        # layer shifts its input by its own constant, so the input and the
        # output of the pass are distinguishable in the tests.
        self.noise_refiner = torch.nn.ModuleList([
            MockNoiseRefiner(
                lambda: sys.modules["comfy.ldm.lumina.model"].optimized_attention_masked,
                shift=1.0 + idx)
            for idx in range(REFINER_LAYERS)
        ])


def mock_zimage_model() -> ModelPatcher:
    m = ModelPatcher()
    m.model.diffusion_model = NextDiT()
    m.model.model_sampling = types.SimpleNamespace()
    return m


def _flow_sigmas(steps: int, shift: float) -> torch.Tensor:
    """``ModelSamplingDiscreteFlow``'s table in denoising order (``model_sampling.py:289-312``),
    flipped so ``sigma_max`` leads — the order ``sample_sigmas`` carries
    (``comfy/samplers.py:1229``) and ``_area_locate`` reads."""
    times = torch.arange(1, steps + 1, dtype=torch.float32) / steps
    sigmas = (shift * times) / (1.0 + (shift - 1.0) * times)
    return sigmas.flip(0)


def _cap_ids() -> torch.Tensor:
    """The caption group's position ids: ``t = 1..cap_len``, row/col zero
    (``embed_cap``, ``comfy/ldm/lumina/model.py:664-667``)."""
    ids = torch.zeros(1, CAP_LEN, 3, dtype=torch.float32)
    ids[0, :, 0] = torch.arange(1, CAP_LEN + 1, dtype=torch.float32)
    return ids


def _img_ids() -> torch.Tensor:
    """The image group's position ids, pad rows included (``pos_ids_x`` ``:421-437``
    plus the ``F.pad`` zero rows of ``embed_all`` ``:725-727``)."""
    ids = torch.zeros(1, IMG_TOKENS + IMG_PAD, 3, dtype=torch.float32)
    ids[0, :, 0] = IMG_T
    pos = torch.arange(IMG_TOKENS)
    ids[0, :IMG_TOKENS, 1] = (pos // GRID).float()
    ids[0, :IMG_TOKENS, 2] = (pos % GRID).float()
    return ids


def _latent(b=1):
    """The Z-Image latent ``[b, c, H, W]`` (16 channels, 2 latent px per patch)."""
    g = torch.Generator().manual_seed(1234)
    return torch.randn(b, 16, GRID * 2, GRID * 2, generator=g)


def _joint(b=1, seed=0):
    """The joint stream q/k/v ``[b, heads, SEQ_LEN, d]`` plus the model's own table.

    One deterministic draw per token so cond and uncond differ, exactly as comfy's
    one batched forward carries both branches.
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(b, HEADS, SEQ_LEN, HEAD_DIM, generator=g)
    k = torch.randn(b, HEADS, SEQ_LEN, HEAD_DIM, generator=g)
    v = torch.randn(b, HEADS, SEQ_LEN, HEAD_DIM, generator=g)
    return q, k, v


class SimulatedZImage:
    """One forward of a Z-Image-like transformer, faithful to ``NextDiT._forward``.

    Order is the real one (``comfy/ldm/lumina/model.py:842-892``): ``embed_cap``'s
    rope call, ``embed_all``'s rope call, the ``noise_refiner`` pass over the image
    block alone, then the joint layer loop's attention calls over the concatenated
    stream.  q/k reach the seam ROTATED by the model's own full-res table — the
    concatenation of the two recorded groups — and with no rope argument, exactly
    as ``JointAttention.forward`` leaves them (``:159-182``).
    """

    def __init__(self, dm, blocks: int = 1, b: int = 1, seed: int = 0):
        self.dm = dm
        self.blocks = blocks
        self.b = b
        self.seed = seed
        self.refiner_calls = 0

    @property
    def seam(self):
        return sys.modules["comfy.ldm.lumina.model"].optimized_attention_masked

    def model_function(self, x, timestep, **c):
        opts = c["transformer_options"]
        dm = self.dm
        # 1. embed_cap: rope group #1, the padded caption block (model.py:663-668).
        cap_table = dm.rope_embedder(_cap_ids())
        # 2. embed_all: rope group #2, the padded image (model.py:722-727).
        img_table = dm.rope_embedder(_img_ids())
        full = torch.cat((cap_table, img_table), dim=2)      # model.py:731/:805
        # 3. noise_refiner over the image block alone, layer by layer — the real
        # forward LOOPS the ModuleList (``model.py:816``) rather than calling it,
        # so only the layers' own hooks fire.  The FIRST layer's INPUT is the
        # detail map's input (the reference's ``x_embedder`` analogue); the
        # pre-refiner block is kept for the tests to compare, and each layer
        # shifts its input so no later layer's input or output can be mistaken
        # for it.
        self.img_block = torch.randn(self.b, IMG_TOKENS + IMG_PAD, HIDDEN)
        x = self.img_block
        for layer in dm.noise_refiner:
            x = layer(x)
        self.refiner_calls = sum(layer.calls for layer in dm.noise_refiner)
        # 4. the joint layer loop: the seam sees the whole padded stream, rotated.
        q, k, v = _joint(self.b, seed=self.seed)
        q, k = apply_rope1(q, full), apply_rope1(k, full)
        out = None
        for _ in range(self.blocks):
            out = self.seam(_Container(q), _Container(k), _Container(v), HEADS, None,
                            skip_reshape=True, transformer_options=opts,
                            preferred_attention=None)
        return out


def _run_forward(m, step, blocks=1, b=1, seed=0, sigmas=True, steps=STEPS):
    """Drive one sampled forward through the installed unet wrapper.

    ``steps`` selects the schedule length (the default is the reference's FLUX
    28; the Turbo table is 8), so a short-schedule test can drive the very
    shift-3.0 table the adaptation research measured on.
    """
    dm = m.model.diffusion_model
    sim = SimulatedZImage(dm, blocks, b, seed)
    opts = {"sample_sigmas": _flow_sigmas(steps, 3.0), "sigmas": _flow_sigmas(steps, 3.0)[step]}
    if not sigmas:
        opts = {"sigmas": torch.tensor([1.0])}
    cross = torch.randn(b, TXT_LEN, 77)
    params = {"input": _latent(b), "timestep": torch.tensor([float(opts["sigmas"])]),
              "c": {"c_crossattn": cross, "transformer_options": opts}}
    return sim, _unet_wrapper(m)(sim.model_function, params)


def _turbo_sigmas() -> torch.Tensor:
    """Turbo's 8-step shift-3.0 table in denoising order, terminal zero included.

    The same table ``docs/2026-10-18-area-zimage-adaptation-research.md`` §2
    lists, and the one ``_area_locate`` reads (it drops the terminal zero).
    """
    return torch.cat([_flow_sigmas(8, 3.0), torch.zeros(1)])


def _unet_wrapper(m):
    """The wrapper the patcher installed (mock and real ModelPatcher)."""
    try:
        return m.model_options["model_function_wrapper"]
    except (AttributeError, KeyError, TypeError):
        return getattr(m, "_unet_wrapper", None)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def lumina_model():
    """Install a mock ``comfy.ldm.lumina.model`` holding a pristine bound symbol.

    AReA patches the bound name the blocks actually call (``model.py:14`` binds
    it, ``:182`` calls it), so the mock module must hold the original function as
    a plain attribute — the same shape the real backend module leaves behind.
    """
    saved = sys.modules.get("comfy.ldm.lumina.model")
    saved_parent = getattr(sys.modules.get("comfy.ldm.lumina"), "model", None)
    parent = sys.modules.setdefault("comfy.ldm.lumina", types.ModuleType("comfy.ldm.lumina"))
    mod = types.ModuleType("comfy.ldm.lumina.model")
    mod.optimized_attention_masked = base_lumina_attention
    sys.modules["comfy.ldm.lumina.model"] = mod
    parent.model = mod
    try:
        yield mod
    finally:
        sys.modules["comfy.ldm.lumina.model"] = saved
        parent.model = saved_parent
        if saved is None:
            sys.modules.pop("comfy.ldm.lumina.model", None)


@pytest.fixture
def flux_layers():
    """A mock ``comfy.ldm.flux.layers`` with a pristine ``attention`` — the FLUX
    seam a Z-Image install must leave alone."""
    saved = sys.modules.get("comfy.ldm.flux.layers")
    saved_parent_layers = getattr(sys.modules.get("comfy.ldm.flux"), "layers", None)
    parent = sys.modules.setdefault("comfy.ldm.flux", types.ModuleType("comfy.ldm.flux"))
    layers = types.ModuleType("comfy.ldm.flux.layers")
    layers.attention = base_flux_attention
    sys.modules["comfy.ldm.flux.layers"] = layers
    parent.layers = layers
    try:
        yield layers
    finally:
        sys.modules["comfy.ldm.flux.layers"] = saved
        parent.layers = saved_parent_layers
        if saved is None:
            sys.modules.pop("comfy.ldm.flux.layers", None)


def base_flux_attention(q, k, v, pe=None, mask=None, transformer_options=None,
                        preferred_attention=None):
    """A pristine FLUX ``attention`` — only its identity is asserted, never its maths."""
    if hasattr(q, "take"):
        q, k, v = q.take(), k.take(), v.take()
    if pe is not None:
        q, k = apply_rope1(q, pe), apply_rope1(k, pe)
    out = F.scaled_dot_product_attention(q, k, v)
    return out.permute(0, 2, 1, 3).reshape(out.shape[0], out.shape[2], HEADS * out.shape[3])


@pytest.fixture(autouse=True)
def _clean_area_state():
    """AReA's seam state is process-global (both ref-counts, both runtimes)."""
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
# Install / restore / ref-count / collision
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSeamInstall:
    def test_install_rebinds_bound_symbol_in_lumina_model(self, lumina_model):
        m = mock_zimage_model()
        patched = apply_area_to_model(m, "zimage", WIDTH, HEIGHT,
                                      phase1_steps=PHASE1, seed=3)
        try:
            assert lumina_model.optimized_attention_masked is not base_lumina_attention
            assert getattr(lumina_model.optimized_attention_masked, "_area_wrapper", False)
            assert lumina_model.optimized_attention_masked._area_orig is base_lumina_attention
            assert area._ZSEAM["module"] is lumina_model
            assert area._ZSEAM["attr"] == "optimized_attention_masked"
            assert area._ZSEAM["orig"] is base_lumina_attention
            assert area._ZSEAM["consumers"] == 1
            # the FLUX twin seam was never touched
            assert area._SEAM["consumers"] == 0
            # the rope hook sits on the model's own embedder; the refiner hooks
            # on every LAYER — the ModuleList is iterated, never called
            # (``model.py:816``), so a hook on the container itself never fires
            dm = patched.model.diffusion_model
            assert len(dm.rope_embedder._forward_hooks) == 1
            assert dm.noise_refiner._forward_hooks == {}
            for layer in dm.noise_refiner:
                assert len(layer._forward_hooks) == 1
        finally:
            restore_area_attention_hook(patched)

    def test_plain_step_delegates_to_pristine_base(self, lumina_model, monkeypatch):
        """A plain step delegates to the PRISTINE symbol — bit-exact, engine untouched.

        ``_area_locate`` finds no step index when the sampler publishes no full
        ``sample_sigmas`` (an exotic sampler, or a plain-attention step), so
        ``ctx.phase`` stays ``None`` and the wrapper's delegating branch hands the
        job back to the symbol AReA wrapped.  Nothing in between may compute: the
        engine is monkeypatched to raise, and the answer is the pristine base's
        own, to the last bit.
        """
        # wrap the pristine base in a spy, so the install wraps the spy and every
        # delegation is visible — the same discipline as the FLUX seam tests.
        calls = []
        real_base = base_lumina_attention

        def pristine(q, k, v, heads, mask=None, attn_precision=None,
                     skip_reshape=False, skip_output_reshape=False, **kw):
            # record BEFORE the call: the pristine base consumes the containers,
            # so the shape must be read while they are still owned.
            seq = q.peek().shape[2]
            calls.append({"containers": (q, k, v), "seq": seq, "heads": heads,
                          "mask": mask, "skip_reshape": skip_reshape, "kw": kw})
            return real_base(q, k, v, heads, mask, attn_precision,
                             skip_reshape, skip_output_reshape, **kw)

        monkeypatch.setattr(lumina_model, "optimized_attention_masked", pristine)

        def _engine(*a, **kw):        # pragma: no cover - must never run
            raise AssertionError("the engine computed on a plain step")

        monkeypatch.setattr(area, "pool_keys_values", _engine)
        monkeypatch.setattr(area, "windowed_attention", _engine)

        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=PHASE1, seed=3)
        try:
            sim, out = _run_forward(patched, step=0, sigmas=False)
        finally:
            restore_area_attention_hook(patched)

        assert area._ZRUNTIME is None            # context cleared after the forward
        # the refiner pass delegated too (its stream is never the joint stream)
        assert sim.refiner_calls == REFINER_LAYERS
        # every delegation carried the real positional slots and kw untouched
        joint = [c for c in calls if c["seq"] == SEQ_LEN]
        assert joint, "the joint stream never reached the seam"
        for call in joint:
            assert call["heads"] == HEADS
            assert call["mask"] is None
            assert call["skip_reshape"] is True
            assert call["kw"]["preferred_attention"] is None
        # the containers were consumed exactly once each (no double-take)
        for call in joint:
            for container in call["containers"]:
                assert container.tensor is None
                with pytest.raises(RuntimeError, match="already been consumed"):
                    container.take()
        # bit-exact: the pristine base's own answer on the same rotated stream
        dm = patched.model.diffusion_model
        full = torch.cat((dm.rope_embedder(_cap_ids()), dm.rope_embedder(_img_ids())), dim=2)
        q, k, v = _joint(1, seed=0)
        expected = base_lumina_attention(_Container(apply_rope1(q, full)),
                                         _Container(apply_rope1(k, full)),
                                         _Container(v), HEADS, None, skip_reshape=True)
        assert torch.equal(out, expected)

    def test_restore_returns_pristine_symbol(self, lumina_model):
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT)
        dm = patched.model.diffusion_model
        assert area._ZSEAM["hook"]                        # handles were registered
        restore_area_attention_hook(patched)
        assert lumina_model.optimized_attention_masked is base_lumina_attention
        assert area._ZSEAM == {"module": None, "attr": None, "orig": None,
                              "hook": None, "recorder": None, "consumers": 0}
        assert dm.rope_embedder._forward_hooks == {}
        assert dm.noise_refiner._forward_hooks == {}
        for layer in dm.noise_refiner:
            assert layer._forward_hooks == {}
        assert patched._area_config is None
        # idempotent — a second restore is a no-op, not a crash
        restore_area_attention_hook(patched)

    def test_ref_counted_install_needs_matching_restores(self, lumina_model):
        first = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT, seed=1)
        wrapped = lumina_model.optimized_attention_masked
        second = apply_area_to_model(first, "zimage", WIDTH, HEIGHT, seed=2)
        try:
            # a second apply re-uses the wrapper instead of wrapping our own wrapper
            assert lumina_model.optimized_attention_masked is wrapped
            assert area._ZSEAM["consumers"] == 2
            assert second._area_config.seed == 2          # newest apply wins
            restore_area_attention_hook(second)
            assert area._ZSEAM["consumers"] == 1
            assert lumina_model.optimized_attention_masked is wrapped
            restore_area_attention_hook(first)
            assert area._ZSEAM["consumers"] == 0
            assert lumina_model.optimized_attention_masked is base_lumina_attention
        finally:
            restore_area_attention_hook(first)
            restore_area_attention_hook(second)

    def test_collision_with_foreign_patch_raises_actionable_error(self, lumina_model):
        # SPA's z-image target is the very same bound symbol (src/spa.py:590-591)
        foreign = lambda *a, **kw: None
        foreign._spa_wrapper = True
        lumina_model.optimized_attention_masked = foreign
        with pytest.raises(ValueError, match="already wrapped by SPA") as excinfo:
            apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT)
        message = str(excinfo.value)
        # names the offender AND the remedy (house rule, src/hap.py:876-895)
        assert "comfy.ldm.lumina.model" in message
        assert "src/spa.py:590-591" in message
        assert lumina_model.optimized_attention_masked is foreign   # nothing rebound
        assert area._ZSEAM["consumers"] == 0                        # no state leaked


# ---------------------------------------------------------------------------
# The Phase I switch
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPhaseSwitch:
    def _phase_of(self, m, step, monkeypatch):
        """Which phase a forward at ``step`` actually ran, seen through the engine."""
        seen = {}
        real_pool, real_window = area.pool_keys_values, area.windowed_attention

        def pool(*args, **kw):
            seen["phase"] = "coarse"
            return real_pool(*args, **kw)

        def window(*args, **kw):
            seen["phase"] = "fine"
            return real_window(*args, **kw)

        monkeypatch.setattr(area, "pool_keys_values", pool)
        monkeypatch.setattr(area, "windowed_attention", window)
        _run_forward(m, step)
        return seen.get("phase")

    def test_phase_switch_steps_mode_matches_schedule(self, lumina_model, monkeypatch):
        """The default mode: the leading ``phase1_steps`` steps are Phase I."""
        table = _flow_sigmas(STEPS, 3.0)
        # the Z-Image shift reproduces the reference's constant in sigma space
        # (tests/test_area_zimage.py pins the same identity)
        assert sigma_prefix_count(table, 0.89) == PHASE1
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=PHASE1, seed=3)
        try:
            assert self._phase_of(patched, 0, monkeypatch) == "coarse"
            assert self._phase_of(patched, PHASE1 - 1, monkeypatch) == "coarse"
            assert self._phase_of(patched, PHASE1, monkeypatch) == "fine"
        finally:
            restore_area_attention_hook(patched)

    def test_phase_switch_sigma_mode_uses_threshold(self, lumina_model, monkeypatch):
        """``sigma`` mode: the switch follows the schedule, not a fixed count."""
        table = _flow_sigmas(STEPS, 3.0)
        expected = sigma_prefix_count(table, 0.5)
        assert expected != PHASE1                      # a different boundary, on purpose
        # σ(t) = 3t/(1+2t) ≥ 0.5 ⟺ t ≥ 1/4, so steps i = 7..28 of 28 are Phase I
        assert expected == STEPS - 6
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=PHASE1, seed=3,
                                      phase1_mode="sigma", phase1_sigma=0.5)
        try:
            assert self._phase_of(patched, expected - 1, monkeypatch) == "coarse"
            assert self._phase_of(patched, expected, monkeypatch) == "fine"
        finally:
            restore_area_attention_hook(patched)


# ---------------------------------------------------------------------------
# The per-forward hooks
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPerForwardHooks:
    def test_noise_refiner_hook_layout_excludes_padding(self, lumina_model, monkeypatch):
        """The detail map sees the real image tokens only, and one layout serves the forward."""
        captured = {}
        real_detail = area.detail_map

        def spy(hidden_states, h, w):
            captured["shape"] = tuple(hidden_states.shape)
            captured["batch"] = hidden_states.shape[0]
            captured["hidden"] = hidden_states.detach().clone()
            return real_detail(hidden_states, h, w)

        monkeypatch.setattr(area, "detail_map", spy)
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=4, seed=5)
        try:
            sim, out = _run_forward(patched, step=STEPS - 1)     # a fine step
            # the refiner pass fired per layer, and its own seam calls delegated
            assert sim.refiner_calls == REFINER_LAYERS
            # the detail map got exactly h*w image tokens — no caption, no pad
            assert captured["shape"] == (1, IMG_TOKENS, HIDDEN)
            # it was the FIRST refiner layer's INPUT — the image embedding that
            # enters refinement (the reference's ``x_embedder`` analogue) — not
            # the projected-keys fallback (which would have warned "no-img-in"
            # and re-derived the map from the rotated keys) and not any later
            # layer's transformed stream (every layer shifts by ``layer.shift``)
            assert torch.equal(captured["hidden"], sim.img_block[:, :IMG_TOKENS])
            last_out = sim.img_block + sum(
                layer.shift for layer in sim.dm.noise_refiner)
            assert not torch.equal(captured["hidden"], last_out[:, :IMG_TOKENS])
            assert "no-img-in" not in area._warned
            # ... and the recorder holds the model's own two groups in call order
            rt = area._ZRUNTIME
            assert rt is None                     # cleared after the forward
        finally:
            restore_area_attention_hook(patched)
        assert out.shape == (1, SEQ_LEN, HEADS * HEAD_DIM)


@pytest.mark.unit
class TestRefinerHookRegistration:
    """The refiner hook sits on the LAYERS, not the ModuleList.

    ``comfy/ldm/lumina/model.py:816`` runs ``for layer in self.noise_refiner`` —
    it iterates, it does not call the container — so a forward hook on the
    ModuleList never fires.  That was a real defect, found by the GPU repro of
    ``docs/2026-10-17-area-zimage-ripple-diagnosis.md``: the arming window stayed
    open, the seam's own coarse-rope calls were recorded as extra groups (the
    table stopped matching the stream) and the first coarse forward's blocks fell
    back to plain attention, while Phase II always used the projected-keys
    fallback instead of the detail map.  These tests pin the per-layer
    registration the fix uses, against a mock that iterates like the real model.
    """

    def test_hooks_sit_on_every_layer_not_the_container(self, lumina_model):
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=PHASE1, seed=3)
        dm = patched.model.diffusion_model
        try:
            assert len(dm.noise_refiner) == REFINER_LAYERS
            assert dm.noise_refiner._forward_hooks == {}      # never fires
            for layer in dm.noise_refiner:
                assert len(layer._forward_hooks) == 1
            # one removable handle per layer, next to the rope embedder's
            assert len(area._ZSEAM["hook"]) == REFINER_LAYERS + 1
        finally:
            restore_area_attention_hook(patched)
        for layer in dm.noise_refiner:                       # all removed
            assert layer._forward_hooks == {}

    def test_detail_map_comes_from_the_first_refiner_input(self, lumina_model,
                                                           monkeypatch):
        """End to end: the iterating pass feeds the map the FIRST layer's input."""
        seen = {}
        real_detail = area.detail_map

        def spy(hidden_states, h, w):
            seen["hidden"] = hidden_states.detach().clone()
            return real_detail(hidden_states, h, w)

        monkeypatch.setattr(area, "detail_map", spy)
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=4, seed=5)
        try:
            sim, out = _run_forward(patched, step=STEPS - 1)
        finally:
            restore_area_attention_hook(patched)
        # the PRE-refiner input of the first layer (``img_block`` before the
        # loop), never a transformed layer output — each layer shifts its input
        assert torch.equal(seen["hidden"], sim.img_block[:, :IMG_TOKENS])
        assert "no-img-in" not in area._warned

    def test_second_refiner_layer_never_overwrites_the_detail_map_input(
            self, lumina_model, monkeypatch):
        """Guard: every layer's hook fires, yet the map keeps the FIRST input.

        Each layer's hook closes the arming window on every call (unchanged),
        but only the module tagged ``_area_refiner_entry`` at install captures
        the detail map's input — the transformed inputs and outputs of the later
        layers must never reach it.
        """
        seen = []
        real_detail = area.detail_map

        def spy(hidden_states, h, w):
            seen.append(hidden_states.detach().clone())
            return real_detail(hidden_states, h, w)

        monkeypatch.setattr(area, "detail_map", spy)
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=4, seed=5)
        try:
            sim, out = _run_forward(patched, step=STEPS - 1)     # a fine step
        finally:
            restore_area_attention_hook(patched)
        # every refiner layer's hook fired ...
        assert sim.refiner_calls == REFINER_LAYERS
        # ... yet ONE capture served the whole forward: the FIRST layer's input
        assert len(seen) == 1
        assert seen[0].shape == (1, IMG_TOKENS, HIDDEN)
        assert torch.equal(seen[0], sim.img_block[:, :IMG_TOKENS])
        # not the second layer's input (the first layer's shifted output) ...
        second_in = sim.img_block + sim.dm.noise_refiner[0].shift
        assert not torch.equal(seen[0], second_in[:, :IMG_TOKENS])
        # ... and not the last layer's shifted output either
        last_out = sim.img_block + sum(
            layer.shift for layer in sim.dm.noise_refiner)
        assert not torch.equal(seen[0], last_out[:, :IMG_TOKENS])
        assert "no-img-in" not in area._warned

    def test_coarse_builders_ropes_are_never_recorded(self, lumina_model, monkeypatch):
        """The arming window closes before the seam's own coarse builder runs.

        With the window leak, the coarse builder's re-roping was captured as
        extra groups, so ``full_table()`` no longer matched the joint stream and
        every block after the first fell back to plain attention — a silent
        quality loss, only visible as a missing Phase I computation per block.
        """
        pooled = []
        real_pool = area.pool_keys_values

        def spy(*a, **kw):
            pooled.append(1)
            return real_pool(*a, **kw)

        monkeypatch.setattr(area, "pool_keys_values", spy)
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=PHASE1, seed=7)
        try:
            _run_forward(patched, step=0, blocks=3)      # a coarse step
        finally:
            restore_area_attention_hook(patched)
        assert len(pooled) == 3                          # every block pooled
        assert "zimage-table" not in area._warned        # the table still matched
        assert "no-img-in" not in area._warned

# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestZImageComputation:
    def test_zimage_attention_output_shape_matches_plain(self, lumina_model):
        """Both phases answer comfy's sequence-first ``[b, n, heads*d]`` shape."""
        embedder = MockEmbedND()
        full = torch.cat((embedder(_cap_ids()), embedder(_img_ids()), ), dim=2)
        q, k, v = _joint(1, seed=0)
        # the shape the pristine attention answers for the same stream
        plain = base_lumina_attention(_Container(apply_rope1(q, full)),
                                      _Container(apply_rope1(k, full)),
                                      _Container(v), HEADS, None, skip_reshape=True)
        assert plain.shape == (1, SEQ_LEN, HEADS * HEAD_DIM)
        for step in (0, STEPS - 1):                     # coarse, then fine
            patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                          phase1_steps=PHASE1, seed=9)
            try:
                sim, out = _run_forward(patched, step=step)
            finally:
                restore_area_attention_hook(patched)
            assert out.shape == plain.shape
            # AReA actually replaced attention: not the plain attention's answer
            assert not torch.allclose(out, plain)

    def test_rng_streams_keyed_by_seed_and_step(self, lumina_model):
        """The same seed and step rebuild the same graph; a different seed does not."""
        def run(seed, step):
            patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                          phase1_steps=PHASE1, pooling="random", seed=seed)
            try:
                return _run_forward(patched, step=step, seed=seed)[1]
            finally:
                restore_area_attention_hook(patched)

        coarse_a, coarse_b = run(0, 0), run(0, 0)
        coarse_c = run(1, 0)
        assert torch.equal(coarse_a, coarse_b)          # identical seed and step
        assert not torch.allclose(coarse_a, coarse_c)   # a different key draws different blocks
        fine_a, fine_c = run(0, STEPS - 1), run(1, STEPS - 1)
        assert not torch.allclose(fine_a, fine_c)       # windows / halos follow the seed too

    def test_cond_uncond_share_one_layout(self, lumina_model, monkeypatch):
        """One batched forward carries cond and uncond; ONE layout serves both."""
        built = []
        real_build = area.build_fine_layout

        def spy(windows, h, w, halo, registers, seed, step, dtype, device):
            layout = real_build(windows, h, w, halo, registers, seed, step, dtype, device)
            built.append([(win.r0, win.c0, win.r1, win.c1) for win in layout.windows])
            return layout

        monkeypatch.setattr(area, "build_fine_layout", spy)
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=4, seed=6)
        try:
            sim, out = _run_forward(patched, step=STEPS - 1, b=2)
        finally:
            restore_area_attention_hook(patched)
        assert len(built) == 1                          # one layout for the whole forward
        assert out.shape == (2, SEQ_LEN, HEADS * HEAD_DIM)
        # both branches attend through the same windows but answer for their own q
        assert not torch.allclose(out[0], out[1])

    def test_phase2_attends_the_whole_caption_block(self, lumina_model, monkeypatch):
        """Phase II hands the engine the FULL padded caption block (``cap_len``).

        The prompt here is 20 tokens inside a 32-token caption block, so the
        caption pad span is non-empty: the seam must pass ``cap_len`` text
        queries/keys/values to ``windowed_attention`` — the prompt tokens plus the
        model's own learned pad (``cap_pad_token``,
        ``comfy/ldm/lumina/model.py:620-623``), the same span Phase I passes
        through — and stitch their answer back, leaving ONLY the image pad span
        zero (``stitch_fine_zeros``).
        """
        seen: dict = {}
        real_window = area.windowed_attention
        real_stitch = area.stitch_fine_zeros

        def window(*args, **kw):
            seen["txt_len"], seen["kv_len"], seen["val_len"] = (int(args[0].shape[2]),
                                                                int(args[1].shape[2]),
                                                                int(args[2].shape[2]))
            return real_window(*args, **kw)

        def stitch(out_text, out_image, layout):
            seen["text_out"] = int(out_text.shape[2])
            stitched = real_stitch(out_text, out_image, layout)
            seen["stitched"] = stitched.clone()
            return stitched

        monkeypatch.setattr(area, "windowed_attention", window)
        monkeypatch.setattr(area, "stitch_fine_zeros", stitch)
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=PHASE1, seed=9)
        try:
            sim, out = _run_forward(patched, step=STEPS - 1)      # a fine step
        finally:
            restore_area_attention_hook(patched)

        # the engine's text block is the whole padded caption block, not the
        # prompt alone: TXT_LEN=20 prompt tokens + CAP_LEN - TXT_LEN learned pad
        assert (seen["txt_len"], seen["kv_len"], seen["val_len"]) == (CAP_LEN,) * 3
        assert seen["text_out"] == CAP_LEN
        # the caption pad span now carries the model's own attention answer
        stitched = seen["stitched"]
        assert stitched[:, :, TXT_LEN:CAP_LEN].abs().sum() > 0
        # ONLY the image pad span stays zero — and still zero through the
        # seam's own flatten of the stitched answer
        assert torch.equal(stitched[:, :, CAP_LEN + IMG_TOKENS:],
                           torch.zeros_like(stitched[:, :, CAP_LEN + IMG_TOKENS:]))
        assert torch.equal(out[:, CAP_LEN + IMG_TOKENS:, :],
                           torch.zeros_like(out[:, CAP_LEN + IMG_TOKENS:, :]))


# ---------------------------------------------------------------------------
# The FLUX path is untouched
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestFluxSeparation:
    def test_flux_seam_untouched_by_zimage_install(self, lumina_model, flux_layers):
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT)
        try:
            # the z-image symbol is wrapped; the FLUX one is pristine
            assert getattr(lumina_model.optimized_attention_masked, "_area_wrapper", False)
            assert flux_layers.attention is base_flux_attention
            assert area._SEAM["consumers"] == 0
            assert area._ZSEAM["consumers"] == 1
        finally:
            restore_area_attention_hook(patched)
        assert flux_layers.attention is base_flux_attention
        assert area._SEAM["consumers"] == 0

# ---------------------------------------------------------------------------
# The short-schedule bridge floor (A1) through the shared call site
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBridgeFloorSeam:
    """The Z-Image seam consumes the floor through the SAME call site FLUX does.

    Both seams funnel their fine steps into ``src.area._area_fine_layout``
    (FLUX: ``_area_attention``'s fine branch; Z-Image:
    ``_zimage_attention``'s fine branch), so one spy on ``build_fine_layout``
    pins the decision for both backbones.  Geometry: Turbo's 8-step shift-3.0
    table with ``phase1_steps=3`` — 3 coarse, then 5 fine — on the 65x65 grid.
    """

    def _run(self, monkeypatch, step, floor, steps=8, phase1=3, halo=8,
             bridge_ratio=0.5):
        """One sampled forward; the (registers, halo) the layout builder got."""
        seen = {}
        real_build = area.build_fine_layout

        def spy(windows, h, w, halo, registers, seed, st, dtype, device):
            layout = real_build(windows, h, w, halo, registers, seed, st, dtype, device)
            seen.update(registers=registers, halo=halo, windows=list(windows), h=h, w=w)
            return layout

        monkeypatch.setattr(area, "build_fine_layout", spy)
        cfg = dict(phase1_steps=phase1, seed=4, halo=halo,
                   bridge_ratio=bridge_ratio, bridge_floor_steps=floor)
        if steps != 8:
            # the 28-step reference table for this mocked schedule
            cfg["phase1_mode"] = "steps"
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT, **cfg)
        try:
            sim, out = _run_forward(patched, step, blocks=1, steps=steps)
        finally:
            restore_area_attention_hook(patched)
        assert out.shape == (1, SEQ_LEN, HEADS * HEAD_DIM)
        return seen

    def _cost(self, seen):
        return windowed_step_cost(seen["windows"], seen["h"], seen["w"], TXT_LEN,
                                  seen["registers"], seen["halo"])

    def test_floored_steps_keep_the_bridge_and_halo(self, lumina_model, monkeypatch):
        """The first two fine steps run the paper's bridge and the full halo."""
        # the 65x65 grid hands over a 64x64 start window on every fine step, so
        # R = round(0.5 * 64)^2 — asserted through the shared helper, not literals
        expected = bridge_registers(64, 0.5)
        for step in (3, 4):                                # fine steps 0 and 1
            seen = self._run(monkeypatch, step, floor=2)
            assert seen["registers"] == expected
            assert seen["halo"] == 8                       # cfg.halo, uncut
            # the cost cap is deliberately waived: this is observable
            assert self._cost(seen) > 1.0

    def test_steps_past_the_floor_are_trimmed_as_today(self, lumina_model, monkeypatch):
        for step in (5, 6, 7):                             # fine steps 2..4
            floored = self._run(monkeypatch, step, floor=2)
            plain = self._run(monkeypatch, step, floor=0)
            assert (floored["registers"], floored["halo"]) == \
                   (plain["registers"], plain["halo"])

    def test_coarse_steps_run_phase_i(self, lumina_model, monkeypatch):
        """A floored step is a fine step: coarse steps are untouched by the floor."""
        pooled = []
        real_pool = area.pool_keys_values

        def pool(*a, **kw):
            pooled.append(1)
            return real_pool(*a, **kw)

        monkeypatch.setattr(area, "pool_keys_values", pool)
        patched = apply_area_to_model(mock_zimage_model(), "zimage", WIDTH, HEIGHT,
                                      phase1_steps=3, seed=4, bridge_floor_steps=2)
        try:
            _run_forward(patched, 2, blocks=1, steps=8)     # the last coarse step
        finally:
            restore_area_attention_hook(patched)
        assert len(pooled) == 1                            # Phase I ran, no layout

    def test_a_long_schedule_is_never_floored(self, lumina_model, monkeypatch):
        floored = self._run(monkeypatch, STEPS - 1, floor=2, steps=STEPS)
        plain = self._run(monkeypatch, STEPS - 1, floor=0, steps=STEPS)
        assert (floored["registers"], floored["halo"]) == (plain["registers"], plain["halo"])

    def test_the_warning_names_the_cost_multiplier(self, lumina_model, monkeypatch,
                                                   caplog):
        area._warned.clear()
        try:
            with caplog.at_level("WARNING", logger="ComfyUI-DyPE"):
                seen = self._run(monkeypatch, 3, floor=2)
            message = caplog.text
            assert "area-bridge-floor" in area._warned
            assert "1." in message and "x a" in message        # the multiplier
            assert f"{seen['registers']}" in message           # and the R it keeps
            # warn-once: the next floored step adds nothing
            self._run(monkeypatch, 4, floor=2)
            assert caplog.text.count("1.0x") == 1
        finally:
            area._warned.clear()
