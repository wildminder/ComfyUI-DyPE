"""A comfy-faithful Qwen-Image-2.1 mock harness for the AReA seam (plan 2026-10-21, §7/S1).

Everything here is mock-based and GPU-free, per project convention.  The mock
mirrors ``comfy/ldm/qwen_image21/model.py`` (418 lines) closely enough that the
seam probe can prove what the real backend does:

* the backend binds the **unmasked** ``optimized_attention`` into its OWN
  namespace (``from comfy.ldm.modules.attention import ... optimized_attention``,
  ``model.py:15``) and calls the BOUND name at ``model.py:211`` (the per-segment
  call) and ``model.py:222`` (the prefix-cache call);
* the per-segment call passes FOUR POSITIONAL arguments — ``(q, k, v, heads)`` —
  and the kwargs ``mask``, ``transformer_options`` and ``preferred_attention``;
  the prefix-cache call passes NO ``mask`` AT ALL;
* the per-segment q/k/v are **3-D** ``[b, n, heads*dim]`` — the model flattens
  ``(B, n, H, D)`` itself (``model.py:206-208``) and calls the seam WITHOUT
  ``skip_reshape``, so comfy reads the flat index head-major
  (``comfy/ldm/modules/attention.py:147-158``);
* the q/k that reach the seam are ROTATED — the model applies its own rope
  before the call (``model.py:99-100``);
* the caption segment carries a CAUSAL BOOL mask (``model.py:324``) and the
  image segment carries ``mask=None`` (``model.py:333``);
* the answer is the FLATTENED ``[b, n, heads*dim]`` (``model.py:212``
  concatenates the per-segment answers on dim 1).

**Why this module IS the backend module.**  The mock's attention resolves
``optimized_attention`` through ITS OWN module's globals, exactly as comfy's
does.  A class defined in the test file would resolve the symbol in the TEST's
globals and the seam's rebind would never reach the model — the very silent
no-op SPA documents (``src/spa.py:558-567``).  So the fixture below registers
THIS module under ``comfy.ldm.qwen_image21.model``: the registered module and
the class's globals are then the same dictionary, and a rebind of the bound
name is seen by the model.

The class name ``QwenImage21Transformer2DModel`` is load-bearing, exactly like
the Krea-2 harness's ``SingleStreamDiT`` (``tests/_krea_fixtures.py:35-38``):
the backbone detectors read it (``src/model_detect.py:50-51``,
``src/area_arch.py``).
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import torch

class _NoMask:
    """The ABSENT-mask sentinel — 2.1 passes no ``mask`` on the cached path."""

    def __repr__(self) -> str:  # pragma: no cover - a debugging aid
        return "<qwen21-no-mask>"


QWEN21_NO_MASK = _NoMask()

#: The seam's identity — the module that BINDS the symbol and the symbol
#: itself (``comfy/ldm/qwen_image21/model.py:15`` and ``:211``).
QWEN21_SEAM_MODULE = "comfy.ldm.qwen_image21.model"
QWEN21_SEAM_SYMBOL = "optimized_attention"

#: The real constructor's defaults (``comfy/ldm/qwen_image21/model.py:236-250``)
#: — the head count and the rope axes are what the seam's geometry rests on.
HEADS = 32
HEAD_DIM = 128
LAYERS = 32
CHANNELS = 64
THETA = 10000
AXES_DIM = (16, 56, 56)

#: The mock's shrunk feature dims (the real ones are 4096): the head COUNT and
#: the rope AXES stay the real ones, the widths do not — a whole 32-block
#: forward has to run on CPU in milliseconds.
MOCK_HEADS = 4
MOCK_HEAD_DIM = 8
#: The rope axes — they sum to the head dim, so the table's frequency axis is
#: ``head_dim // 2`` (``comfy/ldm/flux/layers.py:14-29``); the real ones are
#: ``(16, 56, 56)`` for a head dim of 128.
MOCK_AXES_DIM = (2, 2, 4)
MOCK_CHANNELS = 8
MOCK_CONTEXT_DIM = 16


# ---------------------------------------------------------------------------
# The seam's own side: the container and the pristine symbol
# ---------------------------------------------------------------------------


class AttentionTensorContainer:
    """``comfy.ldm.modules.attention.AttentionTensorContainer`` (``:196-209``).

    Single-owner: ``peek()`` reads it, ``take()`` consumes it once.  2.1 wraps
    its q/k/v in these before the seam call (``model.py:107``) and builds FRESH
    per-segment containers (``model.py:206-208``).
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


def optimized_attention(q, k, v, heads, mask=None, attn_precision=None,
                        skip_reshape=False, skip_output_reshape=False, **kwargs):
    """The pristine ``optimized_attention`` — the symbol AReA wraps.

    comfy's contract (``comfy/ldm/modules/attention.py:255-314``): WITHOUT
    ``skip_reshape`` the tensors arrive as ``[b, n, heads*dim]`` and are read
    HEAD-MAJOR (``_reshape_qkv_to_heads``, ``attention.py:147-158`` — the flat
    index ``j`` of a row is ``head * dim_head + d``); with
    ``skip_output_reshape=False`` — the default, and 2.1 does NOT pass it
    (``model.py:211``) — the answer is the FLATTENED ``[b, n, heads*dim]``.

    The real backend applies no rope of its own: the model rotated q/k before
    the call (``model.py:99-100``).
    """
    if isinstance(q, AttentionTensorContainer):
        q, k, v = q.take(), k.take(), v.take()
    b = q.shape[0]
    if skip_reshape:
        dim_head = q.shape[-1]
        q, k, v = (t.unsqueeze(2) for t in (q, k, v))
    else:
        dim_head = q.shape[-1] // heads
        # comfy's _reshape_qkv_to_heads — HEAD-MAJOR, then heads-first
        q = q.unsqueeze(3).reshape(b, -1, heads, dim_head).transpose(1, 2)
        k = k.unsqueeze(3).reshape(k.shape[0], -1, heads, dim_head).transpose(1, 2)
        v = v.unsqueeze(3).reshape(v.shape[0], -1, heads, dim_head).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
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

    2.1's rope embedder IS that class (``comfy/ldm/qwen_image21/model.py:257``),
    so the mock's table has the real layout: ``ids [b, n, 3]`` ->
    ``[b, 1, n, sum(axes_dim)//2, 2, 2]``.
    """

    def __init__(self, theta: float = THETA, axes_dim=AXES_DIM):
        super().__init__()
        self.theta = theta
        self.axes_dim = list(axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        emb = torch.cat(
            [_mock_rope(ids[..., i], self.axes_dim[i], self.theta)
             for i in range(ids.shape[-1])], dim=-3,
        )
        return emb.unsqueeze(1)


def _apply_rope_bnd(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """``comfy.ldm.flux.math._apply_rope1`` for the ``(B, N, H, D)`` layout.

    2.1 keeps the head axis THIRD (``model.py:83``), so the model's own rope
    call is the (B, N, H, D) variant of comfy's rotation: the table is
    ``(1, N, 1, A, 2, 2)`` and the query is ``(B, N, H, D)``.
    """
    x_ = x.to(dtype=freqs_cis.dtype).reshape(*x.shape[:-1], -1, 1, 2)
    x_out = freqs_cis[..., 0] * x_[..., 0]
    x_out = x_out + freqs_cis[..., 1] * x_[..., 1]
    return x_out.reshape(*x.shape).type_as(x)


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
# The attention call sites
# ---------------------------------------------------------------------------


def block_causal_attention(segments, transformer_options={}, cache=None,
                           block_index=0, prefix_len=0):
    """``comfy/ldm/qwen_image21/model.py:197-213`` — the plain path.

    The outer containers are taken ONCE (``model.py:200``) and FRESH
    per-segment containers are built for every segment (``model.py:206-208``) —
    the seam therefore never sees the outer ones.
    """
    def attn(q, k, v, heads, preferred_attention=None):
        q, k, v = q.take(), k.take(), v.take()
        outs = []
        for i, (start, end, mask) in enumerate(segments):
            query = AttentionTensorContainer(q[:, start:end].flatten(2))
            key = AttentionTensorContainer(k[:, :end].flatten(2))
            value = AttentionTensorContainer(v[:, :end].flatten(2))
            outs.append(optimized_attention(query, key, value, heads, mask=mask,
                                            transformer_options=transformer_options,
                                            preferred_attention=preferred_attention))
        return torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
    return attn


def prefix_cached_attention(prefix_k, prefix_v, transformer_options={}):
    """``comfy/ldm/qwen_image21/model.py:216-223`` — the cached path.

    The seam is called with NO ``mask`` keyword at all (``model.py:222``) and
    the key is ``cat([prefix_k, k_target])`` — the SAME length signature as the
    plain image segment, which is why only the keyword's presence separates the
    two paths.
    """
    def attn(q, k, v, heads, preferred_attention=None):
        q = AttentionTensorContainer(q.take().flatten(2))
        k = AttentionTensorContainer(torch.cat([prefix_k, k.take()], dim=1).flatten(2))
        v = AttentionTensorContainer(torch.cat([prefix_v, v.take()], dim=1).flatten(2))
        return optimized_attention(q, k, v, heads, transformer_options=transformer_options,
                                   preferred_attention=preferred_attention)
    return attn


# ---------------------------------------------------------------------------
# The blocks
# ---------------------------------------------------------------------------


class MockAttention(torch.nn.Module):
    """``comfy/ldm/qwen_image21/model.py:69-108`` — the seam call site, in shape.

    ``forward`` resolves ``optimized_attention`` through THIS module's globals
    — the module the seam install rebinds — so the probe can prove the model
    really calls the BOUND symbol and that the bound name is the only one that
    reaches it.
    """

    def __init__(self, dim: int, heads: int, head_dim: int, seed: int = 0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.to_q = torch.nn.Linear(dim, heads * head_dim, bias=False)
        self.to_k = torch.nn.Linear(dim, heads * head_dim, bias=False)
        self.to_v = torch.nn.Linear(dim, heads * head_dim, bias=False)
        self.to_out = torch.nn.Linear(heads * head_dim, dim, bias=False)
        _seed_linears(self, seed)
        #: The pre-rotation q/k of the LAST call — the probe compares the
        #: seam's tensors against the model's own rotation.
        self.pre_rope = None

    def forward(self, x, pe, attn_fn, prefix_len=0, transformer_options={}):
        b, n, _ = x.shape
        q = self.to_q(x).view(b, n, self.heads, -1)
        k = self.to_k(x).view(b, n, self.heads, -1)
        v = self.to_v(x).view(b, n, self.heads, -1)
        self.pre_rope = (q, k)
        # the MODEL rotates q/k, before the seam ever sees them (model.py:99-100)
        q = _apply_rope_bnd(q, pe)
        k = _apply_rope_bnd(k, pe)
        q, k, v = AttentionTensorContainer(q), AttentionTensorContainer(k), AttentionTensorContainer(v)
        # model.py:108 — the model passes the heads and the preferred
        # attention ONLY; the options travel inside the attn closure.
        return self.to_out(attn_fn(q, k, v, self.heads, preferred_attention=None))


class QwenImage21Transformer2DModel(torch.nn.Module):
    """``comfy/ldm/qwen_image21/model.py:234-417`` — the 2.1 backbone, mocked.

    The class name is load-bearing (the backbone detectors read it, exactly as
    the Krea-2 harness's ``SingleStreamDiT`` does).  The forward reproduces the
    model's ORDER exactly:

    1. ``build_sequence`` — the caption's DIAGONAL ids and the image's CENTRED
       ids with the parity term (``model.py:317-335``);
    2. ONE rope call for the whole joint stream (``model.py:338``);
    3. the per-segment seam calls, the caption first (``model.py:205-211``).
    """

    def __init__(self, inner_dim: int = MOCK_HEADS * MOCK_HEAD_DIM,
                 heads: int = MOCK_HEADS, head_dim: int = MOCK_HEAD_DIM,
                 layers: int = 2, channels: int = MOCK_CHANNELS,
                 context_dim: int = MOCK_CONTEXT_DIM, axes_dim=MOCK_AXES_DIM,
                 theta: float = THETA, seed: int = 0):
        super().__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.channels = channels
        self.out_channels = channels
        self.pe_embedder = MockEmbedND(theta=theta, axes_dim=axes_dim)
        self.txt_in = torch.nn.Linear(context_dim, inner_dim, bias=False)
        self.img_in = torch.nn.Linear(channels, inner_dim, bias=False)
        self.transformer_blocks = torch.nn.ModuleList([
            MockAttention(inner_dim, heads, head_dim, seed + 10 * (index + 1))
            for index in range(layers)
        ])
        self.proj_out = torch.nn.Linear(inner_dim, channels, bias=False)
        _seed_linears(self, seed)

    # --- the model's own geometry ------------------------------------------

    def build_sequence(self, x, context, ref_latents=None):
        """``comfy/ldm/qwen_image21/model.py:311-339``.

        The caption's ids are the DIAGONAL ``(pos+j, pos+j, pos+j)`` with ``pos``
        a running counter (``model.py:323``); the image's ids are
        ``(pos, hh(r), ww(c))`` with the spatial axes CENTRED on the canvas and
        a half-token parity term (``model.py:330-332``).
        """
        ref_latents = list(ref_latents or [])
        txt = self.txt_in(context)
        bounds = [0] + [txt.shape[1]] * len(ref_latents) + [txt.shape[1]]
        parts, ids, segments = [], [], []
        pos, length = 0, 0
        for (start, end), img in zip(zip(bounds[:-1], bounds[1:]), ref_latents + [x]):
            n = end - start
            if n > 0:
                parts.append(txt[:, start:end])
                ids.append(torch.arange(pos, pos + n, device=x.device, dtype=torch.float32)
                           .unsqueeze(1).expand(n, 3))
                segments.append((length, length + n,
                                 torch.ones((n, length + n), dtype=torch.bool, device=x.device).tril(length)))
                pos += n
                length += n
            h, w = img.shape[-2:]
            parts.append(self.img_in(img.flatten(2).transpose(1, 2)))
            hh = torch.arange(h, device=x.device, dtype=torch.float32) - (h - h // 2) \
                + 0.5 * (h % 2 - x.shape[-2] % 2)
            ww = torch.arange(w, device=x.device, dtype=torch.float32) - (w - w // 2) \
                + 0.5 * (w % 2 - x.shape[-1] % 2)
            ids.append(torch.stack(
                [torch.full((h, w), pos, device=x.device, dtype=torch.float32),
                 hh[:, None].expand(h, w), ww[None, :].expand(h, w)], dim=-1).flatten(0, 1))
            segments.append((length, length + h * w, None))
            pos += max(h, w)
            length += h * w
        pe = self.pe_embedder(torch.cat(ids, dim=0).unsqueeze(0)).transpose(1, 2).contiguous()
        #: The model's OWN ids of the last ``build_sequence`` — the adapter's
        #: id builders are pinned against them.
        self.last_ids = torch.cat(ids, dim=0)
        return torch.cat(parts, dim=1), pe, segments

    # --- the forward -------------------------------------------------------

    def forward(self, x, timesteps, context, transformer_options={},
                ref_latents=None, cached=False):
        transformer_options = transformer_options.copy()
        hidden, pe, segments = self.build_sequence(x, context, ref_latents)
        prefix_len = hidden.shape[1] - x.shape[-2] * x.shape[-1]
        transformer_options["total_blocks"] = len(self.transformer_blocks)
        transformer_options["block_type"] = "single"
        prefix_k = prefix_v = None
        if cached:
            b, n, _ = hidden.shape
            prefix_k = self.transformer_blocks[0].to_k(hidden[:, :prefix_len]) \
                .view(b, prefix_len, self.heads, -1)
            prefix_v = self.transformer_blocks[0].to_v(hidden[:, :prefix_len]) \
                .view(b, prefix_len, self.heads, -1)
        if cached:
            # model.py:376 - a cached step runs the TARGET rows only, so
            # the seam's queries are the image rows and its keys are
            # [cached caption K | target K].
            hidden, pe = hidden[:, prefix_len:], pe[:, prefix_len:]
            prefix_len = 0
        for index, block in enumerate(self.transformer_blocks):
            transformer_options["block_index"] = index
            if cached:
                attn_fn = prefix_cached_attention(prefix_k, prefix_v, transformer_options)
            else:
                attn_fn = block_causal_attention(segments, transformer_options, None,
                                                 index, prefix_len)
            hidden = block(hidden, pe, attn_fn, prefix_len, transformer_options)
        out = self.proj_out(hidden[:, prefix_len:])
        return out.transpose(1, 2).reshape(x.shape[0], self.out_channels,
                                           x.shape[-2], x.shape[-1])


#: The plan's name for the mock backbone — an alias, because the class name
#: itself is load-bearing.
MockQwenImage21 = QwenImage21Transformer2DModel


# ---------------------------------------------------------------------------
# Recording the seam's calls
# ---------------------------------------------------------------------------


class SeamCall:
    """One recorded call of the bound seam symbol.

    Everything is snapshotted AT CALL TIME: the pristine symbol consumes the
    containers, and ``transformer_options`` is MUTATED later in the forward
    (``block_index`` is set per block, ``model.py:388``), so a late read would
    see the wrong state.

    ``mask_present`` is the DECISIVE fact: the plain path always passes the
    keyword (``model.py:211``) and the prefix-cache path never does
    (``model.py:222``), and the two are length-indistinguishable.
    """

    __slots__ = ("n_positional", "heads", "kwargs", "q", "k", "v", "mask",
                 "mask_present", "q_len", "k_len", "block_index")

    def __init__(self, args, kwargs):
        self.n_positional = len(args)
        self.heads = int(args[3])
        self.kwargs = dict(kwargs)
        self.q = args[0].peek()
        self.k = args[1].peek()
        self.v = args[2].peek()
        self.mask_present = "mask" in kwargs
        self.mask = kwargs["mask"] if self.mask_present else QWEN21_NO_MASK
        self.q_len = int(self.q.shape[1])
        self.k_len = int(self.k.shape[1])
        options = kwargs.get("transformer_options") or {}
        self.block_index = options.get("block_index")


class SeamRecorder:
    """A spy on the bound seam symbol: records every call, delegates to the pristine one."""

    def __init__(self, pristine):
        self.pristine = pristine
        self.calls: list[SeamCall] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(SeamCall(args, kwargs))
        return self.pristine(*args, **kwargs)

    def image_calls(self) -> list[SeamCall]:
        """Every recorded call that carries NO mask — the image segment."""
        return [call for call in self.calls if call.mask is None]

    def caption_calls(self) -> list[SeamCall]:
        """Every recorded call that carries a causal mask — the caption segment."""
        return [call for call in self.calls if call.mask is not None
                and call.mask is not QWEN21_NO_MASK]

    def cached_calls(self) -> list[SeamCall]:
        """Every recorded call with NO mask keyword at all — the cached path."""
        return [call for call in self.calls if not call.mask_present]


def install_recorder():
    """Install a recorder on the bound symbol; return ``(recorder, restore)``."""
    backend = sys.modules[QWEN21_SEAM_MODULE]
    pristine = getattr(backend, QWEN21_SEAM_SYMBOL)
    recorder = SeamRecorder(pristine)
    setattr(backend, QWEN21_SEAM_SYMBOL, recorder)

    def restore() -> None:
        setattr(backend, QWEN21_SEAM_SYMBOL, pristine)

    return recorder, restore


# ---------------------------------------------------------------------------
# Running a forward
# ---------------------------------------------------------------------------


def run_qwen21_forward(dm, batch: int = 1, seq: int = 4, height: int = 4,
                       width: int = 4, transformer_options=None, cached=False):
    """One Qwen-Image-2.1-shaped forward through the mock backbone.

    ``height``/``width`` are the LATENT's cells (not pixels): 2.1 does not
    patchify, so the token grid IS the latent's own ``(H, W)``
    (``model.py:327``).
    """
    options = {} if transformer_options is None else dict(transformer_options)
    gen = torch.Generator().manual_seed(7)
    x = torch.randn(batch, dm.channels, height, width, generator=gen)
    context = torch.randn(batch, seq, dm.txt_in.in_features, generator=gen)
    timesteps = torch.randn(batch, generator=gen)
    return dm(x, timesteps, context, transformer_options=options, cached=cached)


# ---------------------------------------------------------------------------
# The fixture: this module IS comfy's Qwen-Image-2.1 backend module
# ---------------------------------------------------------------------------


@contextmanager
def qwen21_backend_module():
    """Register this module as ``comfy.ldm.qwen_image21.model`` and restore it after.

    The registered module and the mock classes' globals are the SAME
    dictionary, so a rebind of the bound name
    (``comfy.ldm.qwen_image21.model.optimized_attention``) is seen by the
    model — which is the whole point of the seam and the whole point of this
    harness.

    A context manager, not a pytest fixture: the test file that uses it
    declares its own ``qwen21_backend`` fixture (pytest only sees the fixtures
    of the module it collects), and an imported name that a test argument
    shadows would trip the linter's F811.
    """
    saved = sys.modules.get(QWEN21_SEAM_MODULE)
    parent = sys.modules.setdefault("comfy.ldm", types.ModuleType("comfy.ldm"))
    saved_parent = getattr(parent, "qwen_image21", None)
    pristine = optimized_attention
    backend = sys.modules[__name__]
    sys.modules[QWEN21_SEAM_MODULE] = backend
    parent.qwen_image21 = backend
    try:
        yield backend
    finally:
        globals()[QWEN21_SEAM_SYMBOL] = pristine
        if saved is None:
            sys.modules.pop(QWEN21_SEAM_MODULE, None)
        else:
            sys.modules[QWEN21_SEAM_MODULE] = saved
        parent.qwen_image21 = saved_parent
