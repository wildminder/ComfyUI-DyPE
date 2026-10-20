"""A comfy-faithful Krea-2 mock harness for the AReA seam (plan 2026-10-20, §7/S1).

Everything here is mock-based and GPU-free, per project convention.  The mock
mirrors ``comfy/ldm/krea2/model.py`` (393 lines) closely enough that the seam
probe can prove what the real backend does:

* the backend binds ``optimized_attention_masked`` into its OWN namespace
  (``from comfy.ldm.modules.attention import ... optimized_attention_masked``,
  ``model.py:21``) and calls it at ``model.py:99-100`` with FOUR POSITIONAL
  arguments — ``(q, k, v, heads)`` — and the kwargs ``mask``,
  ``skip_reshape=True``, ``preferred_attention`` and ``transformer_options``.
  ``attn_precision`` and ``skip_output_reshape`` are NOT passed;
* the q/k that reach the seam are ROTATED — the model applies rope itself
  (``model.py:93``) and hands the seam no table;
* the four text-fusion calls run with **20 heads** over streams of length
  ``txtlayers = 12`` (the layerwise blocks, ``model.py:159-161``) and of length
  the PROMPT length (the refiner blocks, ``model.py:162-165``) — and they run
  BEFORE ``transformer_options["img_slice"]`` is published (``model.py:369``),
  so they carry no ``img_slice`` and no ``block_index``;
* the 28 main blocks run with **48 heads** over the joint stream
  ``txt_len + h*w`` and DO carry ``img_slice`` and ``block_index``;
* the answer is the FLATTENED ``[b, n, heads*dim]`` — the model's
  ``out * F.sigmoid(gate)`` multiplies a ``[b, n, dim]`` gate
  (``model.py:106``).

**Why this module IS the backend module.**  The mock's attention class resolves
``optimized_attention_masked`` through ITS OWN module's globals, exactly as
comfy's does.  A class defined in the test file would resolve the symbol in the
TEST's globals and the seam's rebind would never reach the model — the very
silent no-op SPA documents (``src/spa.py:558-567``).  So the fixture below
registers THIS module under ``comfy.ldm.krea2.model``: the registered module
and the class's globals are then the same dictionary, and a rebind of the
bound name is seen by the model.

The class name ``SingleStreamDiT`` is load-bearing, exactly like the Z-Image
harness's ``NextDiT`` (``tests/test_area_zimage_seam.py:200``): the backbone
detectors read it (``src/model_detect.py:47-49``, ``src/area_arch.py:116-118``).
``MockSingleStreamDiT`` is an alias of it.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import torch

from src.area_attn import apply_rope1

#: The seam's identity — the module that BINDS the symbol and the symbol
#: itself (``comfy/ldm/krea2/model.py:21`` and ``:99-100``).
KREA_SEAM_MODULE = "comfy.ldm.krea2.model"
KREA_SEAM_SYMBOL = "optimized_attention_masked"

#: The real constructor's defaults (``comfy/ldm/krea2/model.py:235-238``) —
#: they are what the seam's eligibility guard reads, so the mock keeps them.
HEADS = 48
KV_HEADS = 12
TXT_HEADS = 20
TXT_KV_HEADS = 20
LAYERS = 28
PATCH = 2
CHANNELS = 16
THETA = 1000
TXT_LAYERS = 12

#: The probe's prompt length — the joint stream is ``[prompt | image]``, and the
#: refiner calls' stream length IS this number (a prefix of the joint one).
SEQ = 8

#: The mock's shrunk feature dims (the real ones are 6144 / 2560): the head
#: COUNTS stay the real ones, the widths do not — a whole 28-block forward has
#: to run on CPU in milliseconds.
HEAD_DIM = 6
FEATURES = HEADS * HEAD_DIM
TXT_DIM = TXT_HEADS * HEAD_DIM
#: ``axes_dim`` of the mock's embedder — three axes (t, row, col) summing to
#: the head dim, as the real one does (``model.py:249-252``).
AXES_DIM = (2, 2, 2)


# ---------------------------------------------------------------------------
# The seam's own side: the container and the pristine symbol
# ---------------------------------------------------------------------------


class AttentionTensorContainer:
    """``comfy.ldm.modules.attention.AttentionTensorContainer`` (``:196-209``).

    Single-owner: ``peek()`` reads it, ``take()`` consumes it once.  Krea-2
    wraps its q/k/v in these before the seam call (``model.py:98``).
    """

    __slots__ = ("tensor",)

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def peek(self) -> torch.Tensor:
        if self.tensor is None:
            raise RuntimeError("attention tensor container has already been consumed")
        return self.tensor

    def take(self) -> torch.Tensor:
        tensor = self.peek()
        self.tensor = None
        return tensor


def optimized_attention_masked(q, k, v, heads, mask=None, attn_precision=None,
                                skip_reshape=False, skip_output_reshape=False, **kwargs):
    """The pristine ``optimized_attention_masked`` — the symbol AReA wraps.

    comfy's contract (``comfy/ldm/modules/attention.py:255-256``): with
    ``skip_reshape=True`` the tensors arrive heads-first, and with
    ``skip_output_reshape=False`` — the default, and Krea-2 does NOT pass it
    (``model.py:99-100``) — the answer is the FLATTENED ``[b, n, heads*dim]``
    (``attention.py:300-314``), the layout Krea-2's ``out * F.sigmoid(gate)``
    multiplies (``model.py:106``).

    The real backend applies no rope of its own: the model rotated q/k before
    the call (``model.py:93``).
    """
    if isinstance(q, AttentionTensorContainer):
        q, k, v = q.take(), k.take(), v.take()
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    if not skip_output_reshape:
        out = out.transpose(1, 2).reshape(out.shape[0], -1, heads * out.shape[3])
    return out


# ---------------------------------------------------------------------------
# The rope embedder — a stock FLUX-layout ``EmbedND``
# ---------------------------------------------------------------------------


def _mock_rope(pos: torch.Tensor, dim: int, theta: float) -> torch.Tensor:
    """``comfy.ldm.flux.math.rope`` (``math.py:20-32``) — no comfy import.

    The ``(i j) -> i j`` rearrange is the row-major reshape of the stacked
    ``[cos, -sin, sin, cos]`` vector, so the 2x2 block is
    ``[[cos, -sin], [sin, cos]]`` — the block :func:`src.area_rope.unrotate_table`
    inverts.
    """
    scale = torch.linspace(0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64)
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos.to(dtype=torch.float32), omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return out.reshape(*out.shape[:-1], 2, 2).to(dtype=torch.float32)


class MockEmbedND(torch.nn.Module):
    """A stock FLUX-layout ``EmbedND`` (``comfy/ldm/flux/layers.py:14-29``).

    Krea-2's rope embedder IS that class (``comfy/ldm/krea2/model.py:252``), so
    the mock's table has the real layout: ``ids [b, n, 3]`` ->
    ``[b, 1, n, sum(axes_dim)//2, 2, 2]`` — one rope per axis, concatenated on
    the frequency axis.
    """

    def __init__(self, theta: float = THETA, axes_dim=AXES_DIM):
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


def _pad_to_patch_size(img: torch.Tensor, patch_size=(2, 2)) -> torch.Tensor:
    """``comfy.ldm.common_dit.pad_to_patch_size`` (``common_dit.py:5-13``).

    The latent is padded UP to a whole number of patches — the model's own
    ``h, w`` are the PADDED ones (``model.py:287-288``), so the token grid is
    the CEIL division.
    """
    pad = ()
    for i in range(img.ndim - 2):
        pad = (0, (patch_size[i] - img.shape[i + 2] % patch_size[i]) % patch_size[i]) + pad
    return torch.nn.functional.pad(img, pad, mode="circular")


def _seed_linears(module: torch.nn.Module, seed: int) -> None:
    """Give the mock's weights a FIXED, reproducible draw.

    torch's global RNG would make a forward depend on the test order, and the
    seam probe compares the tensors that reach the seam bit for bit.
    """
    gen = torch.Generator().manual_seed(int(seed))
    with torch.no_grad():
        for _name, param in module.named_parameters():
            if param.dim() == 2:
                param.copy_(torch.randn(param.shape, generator=gen) * 0.05)
            else:
                param.zero_()


# ---------------------------------------------------------------------------
# The blocks
# ---------------------------------------------------------------------------


class MockAttention(torch.nn.Module):
    """``comfy/ldm/krea2/model.py:62-106`` — the seam call site, in shape.

    ``forward`` resolves ``optimized_attention_masked`` through THIS module's
    globals — the module the seam install rebinds — so the probe can prove the
    model really calls the BOUND symbol and that the bound name is the only one
    that reaches it.

    The order inside ``forward`` is the real one: project, normalise, rotate
    with the model's own table (``model.py:93``), resolve the GQA repeat
    (``model.py:94-97``), wrap the tensors in containers (``model.py:98``), call
    the seam (``model.py:99-100``) and gate the answer (``model.py:106``).
    """

    def __init__(self, dim: int, heads: int, kvheads=None, seed: int = 0):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads
        self.wq = torch.nn.Linear(dim, self.headdim * self.heads)
        self.wk = torch.nn.Linear(dim, self.headdim * self.kvheads)
        self.wv = torch.nn.Linear(dim, self.headdim * self.kvheads)
        self.gate = torch.nn.Linear(dim, dim)
        self.wo = torch.nn.Linear(dim, dim)
        _seed_linears(self, seed)
        #: The pre-rotation q/k and the table of the LAST call — the probe
        #: compares the seam's tensors against the model's own rotation.
        self.pre_rope = None
        self.freqs = None

    def forward(self, x, freqs=None, mask=None, transformer_options={}):
        b, n, _ = x.shape
        q = self.wq(x).reshape(b, n, self.heads, self.headdim).transpose(1, 2)
        k = self.wk(x).reshape(b, n, self.kvheads, self.headdim).transpose(1, 2)
        v = self.wv(x).reshape(b, n, self.kvheads, self.headdim).transpose(1, 2)
        self.pre_rope = (q, k)
        self.freqs = freqs
        if freqs is not None:
            # the MODEL rotates q/k, before the seam ever sees them (model.py:93)
            q = apply_rope1(q, freqs)
            k = apply_rope1(k, freqs)
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        out = optimized_attention_masked(q, k, v, self.heads, mask=mask, skip_reshape=True,
                                         preferred_attention=None,
                                         transformer_options=transformer_options)
        gate = self.gate(x)
        return self.wo(out * torch.sigmoid(gate))


class MockTextFusion(torch.nn.Module):
    """``comfy/ldm/krea2/model.py:144-166`` — the FOUR 20-head calls.

    The two ``layerwise_blocks`` run on ``(b*seq, txtlayers, txtdim)`` — the
    attention stream is the 12-layer Qwen3-VL stack — and the two
    ``refiner_blocks`` on ``(b, seq, txtdim)``, so a refiner call's stream
    length is the PROMPT length, a PREFIX of the joint stream's length.  That
    is exactly why the seam's eligibility guard cannot be a length check
    alone: the head count is the exact discriminator (20 vs 48), because the
    GQA repeat is resolved BEFORE the seam (``model.py:94-97``).
    """

    def __init__(self, txtlayers: int, txtdim: int, heads: int, kvheads=None, seed: int = 0):
        super().__init__()
        self.layerwise_blocks = torch.nn.ModuleList([
            MockAttention(txtdim, heads, kvheads, seed + 1),
            MockAttention(txtdim, heads, kvheads, seed + 2),
        ])
        self.projector = torch.nn.Linear(txtlayers, 1)
        self.refiner_blocks = torch.nn.ModuleList([
            MockAttention(txtdim, heads, kvheads, seed + 3),
            MockAttention(txtdim, heads, kvheads, seed + 4),
        ])
        _seed_linears(self, seed)

    def forward(self, x, mask=None, transformer_options={}):
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x, None, transformer_options=transformer_options)
        x = x.reshape(b, l, n, d).permute(0, 1, 3, 2)      # "(b l) n d -> b l d n"
        x = self.projector(x).squeeze(-1)                   # (b, l, d)
        for block in self.refiner_blocks:
            x = block(x, mask, transformer_options=transformer_options)
        return x


class MockSingleStreamBlock(torch.nn.Module):
    """``comfy/ldm/krea2/model.py:169-218`` — one main block."""

    def __init__(self, dim: int, heads: int, kvheads=None, seed: int = 0):
        super().__init__()
        self.attn = MockAttention(dim, heads, kvheads, seed)
        self.mlp = torch.nn.Linear(dim, dim)
        _seed_linears(self, seed + 100)

    def forward(self, x, freqs, mask=None, transformer_options={}):
        attended = self.attn(x, freqs, mask, transformer_options=transformer_options)
        return x + attended + self.mlp(x)


class SingleStreamDiT(torch.nn.Module):
    """``comfy/ldm/krea2/model.py:234-382`` — the Krea-2 backbone, mocked.

    The class name is load-bearing (the backbone detectors read it, exactly as
    the Z-Image harness's ``NextDiT`` does).  The constructor's defaults are the
    REAL ones — heads 48 / kvheads 12 / txtheads 20 / 28 layers / patch 2 /
    16 channels / theta 1e3 / 12 txt layers (``model.py:235-238``) — because
    they are what the seam's eligibility guard reads; only the feature widths
    are shrunk.

    The forward reproduces the model's ORDER exactly:

    1. the text fusion — FOUR 20-head calls (``model.py:341``);
    2. the joint stream ``[text | image]``, no padding (``model.py:355``);
    3. ONE rope call for the whole joint stream (``model.py:364``);
    4. ``total_blocks`` / ``block_type`` / ``img_slice`` published
       (``model.py:367-369``) — AFTER the text fusion, so the four auxiliary
       calls carry no ``img_slice``;
    5. the 28 main blocks, each publishing its ``block_index``
       (``model.py:370-372``).
    """

    def __init__(self, features: int = FEATURES, tdim: int = 64, txtdim: int = TXT_DIM,
                 heads: int = HEADS, kvheads: int = KV_HEADS, layers: int = LAYERS,
                 patch: int = PATCH, channels: int = CHANNELS, theta: float = THETA,
                 txtlayers: int = TXT_LAYERS, txtheads: int = TXT_HEADS,
                 txtkvheads: int = TXT_KV_HEADS, seed: int = 0):
        super().__init__()
        self.patch = patch
        self.channels = channels
        self.heads = heads
        self.tdim = tdim
        self.txtdim = txtdim
        self.txtlayers = txtlayers
        self.pe_embedder = MockEmbedND(theta=theta, axes_dim=AXES_DIM)
        self.first = torch.nn.Linear(channels * patch ** 2, features)
        self.blocks = torch.nn.ModuleList([
            MockSingleStreamBlock(features, heads, kvheads, seed + 10 * (index + 1))
            for index in range(layers)
        ])
        self.txtfusion = MockTextFusion(txtlayers, txtdim, txtheads, txtkvheads, seed)
        self.txtmlp = torch.nn.Linear(txtdim, features)
        self.last = torch.nn.Linear(features, patch * patch * channels)
        _seed_linears(self, seed)

    # --- the model's own geometry ------------------------------------------

    def process_img(self, x, index=0):
        """``comfy/ldm/krea2/model.py:285-295`` — patchify + the image's ids.

        The ids are ``(index, row, col)``: axis 0 is the FRAME index, and a
        single-frame image carries frame 0.
        """
        patch = self.patch
        x = _pad_to_patch_size(x, (patch, patch))
        h, w = x.shape[-2] // patch, x.shape[-1] // patch
        img = (x.reshape(x.shape[0], x.shape[1], h, patch, w, patch)
                .permute(0, 2, 4, 1, 3, 5).reshape(x.shape[0], h * w, -1))
        img_ids = torch.zeros(h, w, 3, device=x.device, dtype=torch.float32)
        img_ids[..., 0] = index
        img_ids[..., 1] = torch.arange(h, device=x.device, dtype=torch.float32)[:, None]
        img_ids[..., 2] = torch.arange(w, device=x.device, dtype=torch.float32)[None, :]
        return img, img_ids.reshape(1, h * w, 3).repeat(x.shape[0], 1, 1), h, w

    def _unpack_context(self, context):
        """``comfy/ldm/krea2/model.py:384-393`` — the Qwen3-VL stack."""
        b, seq, fused = context.shape
        return context.reshape(b, seq, self.txtlayers, self.txtdim)

    # --- the forward -------------------------------------------------------

    def forward(self, x, timesteps, context, attention_mask=None, ref_latents=None,
                transformer_options={}, **kwargs):
        transformer_options = transformer_options.copy()
        bs, _, h_orig, w_orig = x.shape
        patch = self.patch

        context = self._unpack_context(context)
        img, imgpos, h_, w_ = self.process_img(x)
        img_tokens = img.shape[1]
        img = self.first(img)

        # 1. the text fusion — FOUR 20-head calls, BEFORE img_slice exists
        context = self.txtfusion(context, mask=None, transformer_options=transformer_options)
        context = self.txtmlp(context)

        txtlen = context.shape[1]
        device = context.device
        txtpos = torch.zeros(bs, txtlen, 3, device=device, dtype=torch.float32)

        # 2. the joint stream: text FIRST, then the image, no padding
        combined = torch.cat((context, img), dim=1)

        # 3. ONE rope call for the whole joint stream
        pos = torch.cat((txtpos, imgpos), dim=1)
        freqs = self.pe_embedder(pos)

        # 4. the published geometry — after the text fusion, before the blocks
        transformer_options["total_blocks"] = len(self.blocks)
        transformer_options["block_type"] = "single"
        transformer_options["img_slice"] = [txtlen, combined.shape[1]]

        # 5. the main blocks
        for index, block in enumerate(self.blocks):
            transformer_options["block_index"] = index
            combined = block(combined, freqs, None, transformer_options=transformer_options)

        final = self.last(combined)
        out = final[:, txtlen:txtlen + img_tokens, :]
        out = (out.reshape(out.shape[0], h_, w_, self.channels, patch, patch)
                .permute(0, 3, 1, 4, 2, 5).reshape(out.shape[0], self.channels, h_ * patch,
                                                    w_ * patch))
        return out[:, :, :h_orig, :w_orig]


#: The plan's name for the mock backbone (``§7/S1``) — an alias, because the
#: class name itself is load-bearing.
MockSingleStreamDiT = SingleStreamDiT


# ---------------------------------------------------------------------------
# Recording the seam's calls
# ---------------------------------------------------------------------------


class SeamCall:
    """One recorded call of the bound seam symbol.

    Everything is snapshotted AT CALL TIME: the pristine symbol consumes the
    containers, and ``transformer_options`` is MUTATED later in the forward
    (``block_index`` is set per block, ``model.py:371``), so a late read would
    see the wrong state.
    """

    __slots__ = ("n_positional", "heads", "kwargs", "q", "k", "v", "stream_len",
                 "img_slice", "block_index")

    def __init__(self, args, kwargs):
        self.n_positional = len(args)
        self.heads = int(args[3])
        self.kwargs = dict(kwargs)
        self.q = args[0].peek()
        self.k = args[1].peek()
        self.v = args[2].peek()
        self.stream_len = int(self.q.shape[2])
        options = kwargs.get("transformer_options") or {}
        self.img_slice = options.get("img_slice")
        self.block_index = options.get("block_index")


class SeamRecorder:
    """A spy on the bound seam symbol: records every call, delegates to the pristine one."""

    def __init__(self, pristine):
        self.pristine = pristine
        self.calls: list[SeamCall] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(SeamCall(args, kwargs))
        return self.pristine(*args, **kwargs)

    def by_heads(self, heads: int) -> list[SeamCall]:
        """Every recorded call with ``heads`` — the aux/main discriminator."""
        return [call for call in self.calls if call.heads == heads]


def install_recorder():
    """Install a recorder on the bound symbol; return ``(recorder, restore)``."""
    backend = sys.modules[KREA_SEAM_MODULE]
    pristine = getattr(backend, KREA_SEAM_SYMBOL)
    recorder = SeamRecorder(pristine)
    setattr(backend, KREA_SEAM_SYMBOL, recorder)

    def restore() -> None:
        setattr(backend, KREA_SEAM_SYMBOL, pristine)

    return recorder, restore


# ---------------------------------------------------------------------------
# Running a forward
# ---------------------------------------------------------------------------


def run_krea_forward(dm, batch: int = 1, seq: int = SEQ, height: int = 8, width: int = 8,
                     transformer_options=None):
    """One Krea-2-shaped forward through the mock backbone.

    ``height``/``width`` are the LATENT's cells (not pixels): the model
    patchifies them by ``patch`` (``model.py:287-288``), so a 8x8 latent is a
    4x4 token grid.
    """
    options = {} if transformer_options is None else dict(transformer_options)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(batch, dm.channels, height, width, generator=gen)
    context = torch.randn(batch, seq, dm.txtlayers * dm.txtdim, generator=gen)
    timesteps = torch.randn(batch, generator=gen)
    return dm(x, timesteps, context, transformer_options=options)


# ---------------------------------------------------------------------------
# The fixture: this module IS comfy's Krea-2 backend module
# ---------------------------------------------------------------------------


@contextmanager
def krea_backend_module():
    """Register this module as ``comfy.ldm.krea2.model`` and restore it after.

    The registered module and the mock classes' globals are the SAME
    dictionary, so a rebind of the bound name
    (``comfy.ldm.krea2.model.optimized_attention_masked``) is seen by the
    model — which is the whole point of the seam and the whole point of this
    harness.

    A context manager, not a pytest fixture: the test file that uses it
    declares its own ``krea_backend`` fixture (pytest only sees the fixtures
    of the module it collects), and an imported name that a test argument
    shadows would trip the linter's F811.
    """
    saved = sys.modules.get(KREA_SEAM_MODULE)
    parent = sys.modules.setdefault("comfy.ldm.krea2", types.ModuleType("comfy.ldm.krea2"))
    saved_parent = getattr(parent, "model", None)
    pristine = optimized_attention_masked
    backend = sys.modules[__name__]
    sys.modules[KREA_SEAM_MODULE] = backend
    parent.model = backend
    try:
        yield backend
    finally:
        globals()[KREA_SEAM_SYMBOL] = pristine
        if saved is None:
            sys.modules.pop(KREA_SEAM_MODULE, None)
        else:
            sys.modules[KREA_SEAM_MODULE] = saved
        parent.model = saved_parent
