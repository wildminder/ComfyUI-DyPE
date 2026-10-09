"""
I-Max ComfyUI node layer — inference toolkit + node (plan 2026-10-05, P4-P6).

The inference-time compensations of I-Max (arXiv 2410.07536 §2.3) as ComfyUI
transformer patches, the per-pass ``model_options`` builder that installs them
on the high-resolution pass only (D5), and the :class:`IMaxNode` dual-pass
node itself — gates, x0/VAE adapters, guidance-latent builder, schema and
execute.

Everything here is torch-only at MODULE scope — the single exception is the
V3 schema base (``comfy_api.latest.io``, which the root conftest mocks): the
comfy.* modules are imported LAZILY inside the functions that need them (the
``nodes/hiflow.py`` discipline one level stricter), so this layer imports and
unit-tests without a ComfyUI installation.

Toolkit items and their decisions:

- D4 — NTK-aware scaled RoPE. :class:`IMaxNTKEmbedder` reimplements ComfyUI's
  ``EmbedND`` (ldm/flux/layers.py:15-30) with the reference's omega
  (transformer_flux.py:40-58):

      omega(s) = max((theta * ntk_factor)^-s, theta^-s / sqrt((N - 512)/64^2))

  with ``s = arange(0, dim, 2)/dim`` and N the JOINT sequence length (text +
  image tokens). The clip floor restores native rotation toward the
  extrapolation budget: at the native ``N = 4608`` the embedder is BITWISE
  identical to the plain EmbedND (so a native-resolution pass is unaffected),
  and ``ntk_factor=1`` is an identity at every N where the clip formula is
  defined. ``theta``/``axes_dim`` are read off the CURRENTLY installed
  embedder — chained DyPE/SEGA/SPA wrappers included (they all expose
  ``.theta``/``.axes_dim``, src/base.py:18-19; per-axis thetas via
  ``.thetas``). A DyPE-family embedder (anything answering ``set_timestep``,
  src/base.py:43) triggers a takeover warning: I-Max's static NTK omega
  replaces the dynamic scaling for the duration of the pass. From v2.19.0
  a ``clip_mode="per_group"`` serves the lumina ``NextDiT`` wiring
  (Z-Image), which calls its embedder once per token group: the image
  group clips against its own group length, the caption group takes the
  paper's model-wide NTK scaling with no floor (plan 2026-10-06 D4').
- D8 — proportional self-attention. The reference overrides the SDPA scale
  with ``sqrt(log(N_joint, 4608) / head_dim)`` (attention_processor.py:1773-
  1777 AND :1889-1893 — log BASE 4608, not the natural log of the ratio a
  casual read of plan D8's ``sqrt(log(N/4608))`` would suggest). As a q
  pre-scale in ``attn1_patch`` this is exact: the patch fires before RoPE and
  a scalar on q commutes with the rotation, so head_dim cancels and the patch
  is ``q *= sqrt(log(N, 4608))`` — clamped to exactly 1.0 at/below the native
  sequence length (the plan D8 clamp; at ``N = 4608`` the formula itself is
  continuously 1.0). From v2.19.0 the z-image profile installs the SAME
  algebra as a global ``transformer_options["optimized_attention_override"]``
  instead — lumina blocks have no ``attn1_patch`` seam (their only attention
  call is the module-level ``optimized_attention_masked``, ldm/lumina/
  model.py:179-182) — with the anchor the z-image native sequence length
  (native image grid + padded caption) and the call forwarded through the
  lumina ``optimized_attention_masked`` symbol so chained SPA rebindings
  compose (plan 2026-10-06).
- D9 — text duplication via ``post_input``. The reference tiles the text
  stream ``nh*nw`` times (transformer_flux.py:364-375), offsetting each
  copy's position grid by ``(i*64, j*64)`` so every text copy shares a RoPE
  frame with one image tile. D9 keeps ``ceil`` tiles per side instead of the
  reference's ``floor`` (intentional: floor under-covers targets extending
  past a whole number of native tiles; the two agree at every power-of-two
  target). Downstream slicing stays correct because the Flux forward cuts
  with the POST-patch txt length (ldm/flux/model.py:306,407). At/below the
  native grid the patch is a provable no-op (nh = nw = 1). From v2.19.0 the
  patch is FLUX-only — the paper applies text duplication to MMDiT Flux
  (Lumina-Next uses cross-attention) and lumina blocks have no
  ``post_input`` seam — so the z-image wiring warns and installs nothing
  (plan 2026-10-06 D5).
- D5 — per-pass model options. :func:`build_pass_model_options` nested-clones
  ``model.model_options`` (comfy's own ``create_model_options_clone`` when
  importable, a local mirror otherwise) and APPENDS the toolkit patches —
  never replaces (comfy ``set_model_patch`` semantics, model_patcher.py:662-
  666) — only for the enabled pass. Pass A gets a clone with NO I-Max
  patches; ``m.model_options`` is never mutated. The D4 unet function wrapper
  rides INSIDE ``model_options`` (``model_options["model_function_wrapper"]``,
  model_patcher.py:656-657, consumed at samplers.py:332-333), so BOTH pass
  dicts carry it — pass discrimination is a state cell closed over by the
  wrapper itself (P6).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import torch
from comfy_api.latest import io
from torch import Tensor

try:
    from ..src.effective_sampling import effective_model_sampling, warn_if_stale_leak
    from ..src.imax import IMaxConfig, build_flow_sigmas, imax_dual_pass
    from ..src.prefix_cache import disable_prefix_kv_cache
except ImportError:  # flat repo layout (tests / CLI)
    from src.effective_sampling import effective_model_sampling, warn_if_stale_leak
    from src.imax import IMaxConfig, build_flow_sigmas, imax_dual_pass
    from src.prefix_cache import disable_prefix_kv_cache

from .pixelrush import _detect_prediction_type

logger = logging.getLogger("ComfyUI-DyPE")

# Flux native training geometry: 64x64 latent patches (1024 px) + 512 text
# tokens = 4608 joint positions. The reference hardcodes 512 and 64**2
# (transformer_flux.py:52, attention_processor.py:1772); both derive from
# these two constants.
_TRAIN_SEQ_LEN = 4608
_NATIVE_GRID = 64
# D4' clip-floor variants (plan 2026-10-06): "joint" — the FLUX reference
# port (N is the whole stream, text_tokens subtracted); "per_group" — the
# lumina NextDiT wiring (one token group per call, no subtraction).
_CLIP_MODES = ("joint", "per_group")


# ---------------------------------------------------------------------------
# NTK-aware RoPE omega (D4) — reference transformer_flux.py:40-58
# ---------------------------------------------------------------------------

def ntk_rope_omega(
    theta: float,
    dim: int,
    ntk_factor: float,
    seq_len: int,
    ntk_clip: bool = True,
    train_seq_len: int = _TRAIN_SEQ_LEN,
    native_grid: int = _NATIVE_GRID,
    device: torch.device | None = None,
    clip_mode: str = "joint",
    text_tokens: int | None = None,
) -> Tensor:
    """NTK-aware RoPE frequencies for one axis: ``max(NTK, clip floor)``.

    Literal port of the reference (transformer_flux.py:42-52) on comfy's
    grid: ``scale = linspace(0, (dim-2)/dim, dim//2)`` in fp64 — numerically
    identical to the reference's ``arange(0, dim, 2)/dim`` AND bitwise
    identical to comfy's own rope grid (ldm/flux/math.py:27), which is what
    makes :meth:`IMaxNTKEmbedder.forward` bitwise-equal to the plain
    ``EmbedND`` at ``ntk_factor=1`` and native N.

        omega     = (theta * ntk_factor)^-scale     (the NTK base scaling)
        omega_int = theta^-scale / sqrt((N - text_tokens)/native_grid^2)
        omega     = max(omega, omega_int)           (the clip floor)

    The floor restores rotation frequency the NTK scaling removed, relative
    to the extrapolation budget: at the native ``N = train_seq_len`` the
    ratio is exactly 1 (native omega wins for any ``ntk_factor > 1``), above
    it the floor decays as ``1/sqrt(N)``. The reference computes the floor
    unconditionally and CRASHES with a math domain error once
    ``N <= text_tokens`` (512); here the clip is skipped in that degenerate
    regime instead (documented deviation, plan §1.1 defect policy).

    ``clip_mode`` selects the floor's ratio (plan 2026-10-06 D4'):

    - ``"joint"`` (default, FLUX): ``N`` is the JOINT stream length and the
      ratio subtracts ``text_tokens`` — the literal port above.
    - ``"per_group"`` (Z-Image / lumina NextDiT): the embedder is called
      once per token group, so ``N`` is ONE group's length and the ratio
      subtracts nothing — ``sqrt(N/native_grid^2)``. The image group clips
      against its own grid extent; the caption group is all text (a
      joint-style numerator would vanish), so the caller passes it with
      ``ntk_clip=False`` — pure NTK, the paper's model-wide Lumina scaling
      (plan §5). Without the subtraction the ratio is positive for every
      ``seq_len >= 1``: the joint degenerate regime is unreachable in
      per-group mode.

    ``text_tokens`` overrides the joint ratio's subtracted count (derived
    from ``train_seq_len - native_grid**2`` when None); per-group mode
    never reads it.

    Returns fp64 omega of length ``dim // 2`` on ``device`` (CPU default).
    """
    if dim % 2 != 0:
        raise ValueError(f"ntk_rope_omega needs an even dim; got {dim!r}")
    if ntk_factor <= 0.0:
        raise ValueError(f"ntk_factor must be > 0; got {ntk_factor!r}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be >= 1; got {seq_len!r}")
    if native_grid < 1:
        raise ValueError(f"native_grid must be >= 1; got {native_grid!r}")
    if clip_mode not in _CLIP_MODES:
        raise ValueError(
            f"clip_mode must be 'joint' or 'per_group'; got {clip_mode!r}")

    dev = torch.device("cpu") if device is None else torch.device(device)
    if text_tokens is None:
        text_tokens = train_seq_len - native_grid * native_grid
    else:
        text_tokens = int(text_tokens)
    # comfy's own grid form (ldm/flux/math.py:27) — bitwise parity with
    # EmbedND; NOT src/rope.py's arange form (DyPE layout, unusable here).
    scale = torch.linspace(
        0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=dev,
    )
    omega = 1.0 / ((theta * ntk_factor) ** scale)
    if ntk_clip:
        if clip_mode == "per_group":
            ratio = seq_len / (native_grid * native_grid)
        else:
            ratio = (seq_len - text_tokens) / (native_grid * native_grid)
        if ratio > 0.0:  # joint: N <= text_tokens is the degenerate regime
            omega_inter = 1.0 / (theta ** scale) / math.sqrt(ratio)
            omega = torch.max(omega, omega_inter)
    return omega


def _rope_compute_device(device: torch.device) -> torch.device:
    """Comfy's rope() fp64 rule (ldm/flux/math.py:22-25): the fp64 omega
    grid computes on CPU when the target device has no fp64 support.

    Lazy import keeps this module importable without comfy (mocked in
    tests, where the import fails and the device is used as-is).
    """
    try:
        import comfy.model_management as model_management
    except (ImportError, AttributeError):
        return device
    supports_fp64 = getattr(model_management, "supports_fp64", None)
    if supports_fp64 is None or supports_fp64(device):
        return device
    return torch.device("cpu")


class IMaxNTKEmbedder(torch.nn.Module):
    """I-Max NTK-aware scaled RoPE: b' = b * ntk_factor (+ clip).

    ``inner`` is the currently-installed embedder — a plain comfy ``EmbedND``
    or a chained DyPE/SEGA/SPA wrapper; ``theta``/``axes_dim`` (and the
    DyPE-family per-axis ``thetas``) are read off it at construction and the
    embedder is never mutated. :meth:`forward` reimplements ``EmbedND``
    (ldm/flux/layers.py:15-30) with :func:`ntk_rope_omega` frequencies,
    reproducing comfy's exact output layout: per axis
    ``stack([cos, -sin, sin, cos]) -> [b, n, dim/2, 2, 2]``, concatenated on
    the frequency axis and ``unsqueeze(1)`` — the shape the Flux blocks feed
    to ``apply_rope``. The wrapper (P6) swaps this in for the duration of a
    forward pass and restores ``inner`` afterwards.

    ``clip_mode`` (plan 2026-10-06 D4'): ``"joint"`` — the FLUX wiring, one
    call with the whole stream, the clip floor subtracts ``text_tokens``
    (default ``train_seq_len - native_grid**2`` = 512, bitwise-neutral;
    :meth:`set_text_tokens` re-records it). ``"per_group"`` — the lumina
    ``NextDiT`` wiring (Z-Image), which calls its ``rope_embedder`` once per
    token group (cap lumina/model.py:673, siglip :712, image :730): the
    image group clips against its own group length over the native grid,
    the caption group takes pure NTK with no floor.
    """

    def __init__(
        self,
        inner,
        ntk_factor: float = 10.0,
        ntk_clip: bool = True,
        train_seq_len: int = _TRAIN_SEQ_LEN,
        native_grid: int = _NATIVE_GRID,
        clip_mode: str = "joint",
    ) -> None:
        super().__init__()
        if clip_mode not in _CLIP_MODES:
            raise ValueError(
                f"clip_mode must be 'joint' or 'per_group'; got {clip_mode!r}")
        theta = getattr(inner, "theta", None)
        axes_dim = getattr(inner, "axes_dim", None)
        if theta is None or not axes_dim:
            raise ValueError(
                "IMaxNTKEmbedder needs the installed positional embedder to "
                "expose .theta and .axes_dim (comfy's EmbedND and the DyPE "
                f"family all do); got {type(inner).__name__}."
            )
        self.inner = inner
        self.theta = theta
        self.thetas = getattr(inner, "thetas", None)  # DyPE per-axis thetas
        self.axes_dim = list(axes_dim)
        self.ntk_factor = float(ntk_factor)
        self.ntk_clip = bool(ntk_clip)
        self.train_seq_len = int(train_seq_len)
        self.native_grid = int(native_grid)
        self.clip_mode = clip_mode
        # D8 cap accounting: the JOINT floor's subtracted text-token count
        # (512 on FLUX — the bitwise-neutral default); the wiring records
        # the real count via set_text_tokens. Per-group mode never reads it.
        self.text_tokens = int(train_seq_len) - int(native_grid) ** 2
        # Takeover warning (plan Phase 4): a DyPE-family embedder carries
        # dynamic, timestep-dependent scaling that this static omega replaces
        # for the whole pass. The family marker is structural —
        # DyPEBasePosEmbed.set_timestep (src/base.py:43) — so the check also
        # fires for cross-pack wrappers, not just this repo's classes.
        if hasattr(inner, "set_timestep"):
            logger.warning(
                "I-Max: the installed positional embedder is a dynamic "
                "scaling wrapper (%s) — its per-step scaling is REPLACED by "
                "the I-Max NTK omega for the high-resolution pass. Chain "
                "only one of DyPE / I-Max positional scaling.",
                type(inner).__name__,
            )

    def set_text_tokens(self, count: int) -> None:
        """Record the text-token count the JOINT clip floor subtracts (D8).

        Bookkeeping for the wiring: the z-image path computes the padded
        caption length and records it here alongside the attention anchor;
        per-group :meth:`forward` never reads it (the group length carries
        the accounting). The default ``train_seq_len - native_grid**2``
        (512 on FLUX) keeps the joint path bitwise-neutral (D7).
        """
        count = int(count)
        if count < 1:
            raise ValueError(f"text_tokens must be >= 1; got {count!r}")
        self.text_tokens = count

    def _is_image_group(self, ids: Tensor) -> bool:
        """Per-group discrimination — the pack's proven mask
        (src/models/zimage.py:36-37): a call is the IMAGE group iff any row
        carries a nonzero h/w id (axes 1/2). Caption ids are all-zero on
        axes 1/2 across the WHOLE padded span (embed_cap writes the token
        count on axis 0 only, comfy lumina/model.py:657-674), so the test
        cannot fire on pad rows; image pad rows are all-zero too, but the
        group's real grid rows fire it. Siglip grids (nonzero h/w) land in
        the image branch.
        """
        if ids.shape[-1] < 3:
            raise ValueError(
                "per-group clip mode needs 3-axis position ids [b, n, 3]; "
                f"got shape {tuple(ids.shape)}."
            )
        return bool(((ids[..., 1] != 0) | (ids[..., 2] != 0)).any())

    def forward(self, ids: Tensor) -> Tensor:
        """Position ids ``[b, n, axes]`` -> rope table ``[b, 1, n, D, 2, 2]``.

        Joint mode: N is the JOINT sequence length (text + image tokens) —
        the clip floor is a function of the whole stream, exactly as in the
        reference where each axis's rope call sees the full position tensor.
        Per-group mode (z-image): ONE group per call — the image group clips
        against its own length, the caption group takes pure NTK.
        """
        n_axes = int(ids.shape[-1])
        seq_len = int(ids.shape[1])
        device = _rope_compute_device(ids.device)
        if self.clip_mode == "per_group":
            clip = self._is_image_group(ids) and self.ntk_clip
            clip_mode = "per_group"
        else:
            clip = self.ntk_clip
            clip_mode = "joint"
        embs = []
        for axis in range(n_axes):
            theta = (
                self.thetas[axis] if self.thetas is not None else self.theta
            )
            omega = ntk_rope_omega(
                theta, self.axes_dim[axis], self.ntk_factor, seq_len,
                ntk_clip=clip,
                train_seq_len=self.train_seq_len,
                native_grid=self.native_grid,
                device=device,
                clip_mode=clip_mode,
                text_tokens=self.text_tokens,
            )
            pos = ids[..., axis].to(dtype=torch.float32, device=device)
            out = torch.einsum("...n,d->...nd", pos, omega)
            out = torch.stack(
                [out.cos(), -out.sin(), out.sin(), out.cos()], dim=-1,
            )
            embs.append(
                out.reshape(*out.shape[:-1], 2, 2).to(
                    dtype=torch.float32, device=ids.device,
                )
            )
        return torch.cat(embs, dim=-3).unsqueeze(1)


# ---------------------------------------------------------------------------
# Proportional self-attention (D8) — reference attention_processor.py:1773
# ---------------------------------------------------------------------------

def proportional_attention_factor(
    n_tokens: int, train_seq_len: int = _TRAIN_SEQ_LEN,
) -> float:
    """Q pre-scale that reproduces the reference's SDPA scale override.

    The reference runs SDPA with ``scale = sqrt(log(N, 4608)/head_dim)``
    instead of the default ``sqrt(1/head_dim)`` (attention_processor.py:
    1773-1777, single :1889-1893 — log BASE 4608). Pre-scaling q by the
    RATIO of the two scales is equivalent, and head_dim cancels:

        factor = sqrt(log(N, 4608))

    Clamped to exactly 1.0 at/below ``train_seq_len``: below native the
    reference would scale q DOWN (uniform-attention drift at 1024 px — the
    plan §1.1 defect), and pass B at/below native must behave natively. At
    ``N = 4608`` the formula itself is continuously 1.0.
    """
    if n_tokens < 1:
        raise ValueError(f"n_tokens must be >= 1; got {n_tokens!r}")
    if train_seq_len < 1:
        raise ValueError(f"train_seq_len must be >= 1; got {train_seq_len!r}")
    if n_tokens <= train_seq_len:
        return 1.0
    return math.sqrt(math.log(n_tokens, train_seq_len))


def make_proportional_attention_patch(
    train_seq_len: int = _TRAIN_SEQ_LEN,
) -> Callable:
    """``attn1_patch`` factory: scale q by :func:`proportional_attention_factor`.

    The patch fires BEFORE RoPE (ldm/flux/layers.py:234-238 double,
    :341-345 single) with the FLUX signature
    ``p(q, k, v, pe=, attn_mask=, extra_options=)``; q is ``[b, heads, N, d]``
    with N the joint stream in both block kinds. A scalar on q commutes with
    the RoPE rotation, so pre-scaling here is EXACTLY the reference's SDPA
    ``scale`` override (which is applied after RoPE in the reference).
    Only ``q`` is returned — comfy restores k/v/pe/attn_mask from the
    unmodified originals for any absent key (``out.get(...)``).
    """

    def patch(q, k, v, pe=None, attn_mask=None, extra_options=None):
        factor = proportional_attention_factor(int(q.shape[2]), train_seq_len)
        if factor == 1.0:
            return {"q": q}
        return {"q": q * factor}

    return patch


def make_attention_scale_override(
    anchor_seq_len: int = _NATIVE_GRID * _NATIVE_GRID,
    previous_override: Callable | None = None,
) -> Callable:
    """``optimized_attention_override`` factory (D8, z-image profile).

    Lumina blocks have no ``attn1_patch`` seam: their ONLY attention call is
    the module-level ``optimized_attention_masked`` (comfy ldm/lumina/
    model.py:179-182), intercepted through the global
    ``transformer_options["optimized_attention_override"]`` seam — wrap_attn
    (ldm/modules/attention.py:206-240) unwraps the AttentionTensorContainers
    and calls this override as ``override(func, q, k, v, heads, *args,
    **kwargs)``: ``func`` is the RAW undecorated backend, q/k/v are already
    ``[b, heads, N, d]`` (the lumina call passes ``skip_reshape=True``), the
    mask rides positionally in ``args`` and ``_inside_attn_wrapper=True`` in
    kwargs (comfy's own installer precedent: set_model_optimized_attention,
    model_patcher.py:688-695).

    The override pre-scales q by :func:`proportional_attention_factor` of
    the CALL's N against ``anchor_seq_len`` — the same algebra as the flux
    ``attn1_patch`` (a scalar on q commutes with RoPE, head_dim cancels),
    with the anchor the z-image native sequence length: the native image
    grid (64x64 = 4096) plus the padded caption length, which the wiring
    passes. Clamped to exactly 1.0 at/below the anchor — an exact no-op
    pass-through. APPROXIMATION: the override sees the per-call N only — an
    image-only sub-stream call (no caption rows) is scaled against the JOINT
    anchor, a <=1% ratio error (plan 2026-10-06 §5).

    Dispatch after the scale deliberately does NOT call ``func``: ``func``
    is wrap_attn's raw backend and calling it would bypass a chained SPA
    rebinding. The override resolves the lumina module symbol
    ``optimized_attention_masked`` (``comfy.ldm.lumina.model``) lazily
    INSIDE the call (import discipline) and forwards q/k/v/heads/args/
    kwargs untouched — SPA rebinds exactly that symbol (src/spa.py:590-591),
    so the call composes with it, and the still-set ``_inside_attn_wrapper``
    makes the wrapped symbol skip the override branch (no recursion;
    wrap_attn already popped ``preferred_attention``). When the comfy import
    fails (standalone/mock use) the raw ``func`` is the fallback.

    ``previous_override`` chains a pre-existing comfy override read off
    ``model_options`` when the builder installs ours: it receives the SCALED
    q with the untouched rest.
    """
    if anchor_seq_len < 1:
        raise ValueError(
            f"anchor_seq_len must be >= 1; got {anchor_seq_len!r}")

    def override(func, q, k, v, heads, *args, **kwargs):
        n_tokens = int(q.shape[-2])
        factor = proportional_attention_factor(n_tokens, anchor_seq_len)
        if factor != 1.0:
            q = q * factor
        if previous_override is not None:
            return previous_override(func, q, k, v, heads, *args, **kwargs)
        try:
            from comfy.ldm.lumina import model as lumina_model
            attn_fn = lumina_model.optimized_attention_masked
        except (ImportError, AttributeError):
            attn_fn = func
        return attn_fn(q, k, v, heads, *args, **kwargs)

    return override


# ---------------------------------------------------------------------------
# Text duplication (D9) — reference transformer_flux.py:364-375
# ---------------------------------------------------------------------------

def make_text_duplication_patch(native_grid: int = _NATIVE_GRID) -> Callable:
    """``post_input`` factory: tile the text stream over the image grid.

    The reference duplicates ``txt`` and ``txt_ids`` ``nh*nw`` times with
    ``nh = max(1, floor(grid_h/64))`` per side, offsetting copy ``(i, j)``'s
    position grid by ``(i*64, j*64)`` so each text copy lands in the RoPE
    frame of one native image tile (transformer_flux.py:364-375). D9 uses
    CEIL per side — floor under-covers targets that extend past a whole
    number of native tiles (e.g. an 80-patch-wide grid would get a single
    text copy for two tiles of image); identical at power-of-two targets.

    Contract (ldm/flux/model.py:184-196): receives
    ``{img, txt, img_ids, txt_ids, transformer_options}`` before
    ``ids = cat((txt_ids, img_ids))`` and returns at least the four tensors;
    img/img_ids pass through untouched, and the native case (``nh = nw = 1``)
    returns the state dict itself — a provable no-op.
    """

    def patch(state: dict) -> dict:
        img_ids = state.get("img_ids")
        txt_ids = state.get("txt_ids")
        if img_ids is None or txt_ids is None:
            return state
        grid_h = int(img_ids[..., 1].max().item()) + 1
        grid_w = int(img_ids[..., 2].max().item()) + 1
        nh = max(1, math.ceil(grid_h / native_grid))
        nw = max(1, math.ceil(grid_w / native_grid))
        if nh == 1 and nw == 1:
            return state
        txt = state["txt"]
        txt_copies = []
        ids_copies = []
        for i in range(nh):
            for j in range(nw):
                ids_ij = txt_ids.clone()
                ids_ij[:, :, 1] = ids_ij[:, :, 1] + i * native_grid
                ids_ij[:, :, 2] = ids_ij[:, :, 2] + j * native_grid
                txt_copies.append(txt)
                ids_copies.append(ids_ij)
        out = dict(state)
        out["txt"] = torch.cat(txt_copies, dim=1)
        out["txt_ids"] = torch.cat(ids_copies, dim=1)
        return out

    return patch


# ---------------------------------------------------------------------------
# Per-pass model options (D5)
# ---------------------------------------------------------------------------

def _copy_nested_dicts(input_dict: dict) -> dict:
    """Local mirror of comfy.patcher_extension.copy_nested_dicts
    (patcher_extension.py:137-144) for standalone/mock imports."""
    new_dict = input_dict.copy()
    for key, value in input_dict.items():
        if isinstance(value, dict):
            new_dict[key] = _copy_nested_dicts(value)
        elif isinstance(value, list):
            new_dict[key] = value.copy()
    return new_dict


def _clone_model_options(model_options: dict) -> dict:
    """Nested-clone ``model_options`` — comfy's own helper when importable.

    ``create_model_options_clone`` (model_patcher.py:126-127) is comfy's
    canonical cloner; the local mirror only serves mock/standalone imports.
    Dicts are cloned recursively, lists one level (shared patch callables —
    appending to a cloned list can never touch the source list).
    """
    try:
        from comfy.model_patcher import create_model_options_clone
    except (ImportError, AttributeError):
        return _copy_nested_dicts(model_options)
    return create_model_options_clone(model_options)


def build_pass_model_options(
    model,
    enabled: bool,
    *,
    proportional_attention: bool = True,
    text_duplication: bool = True,
    arch_profile: str = "flux",
    attention_anchor: int = _NATIVE_GRID * _NATIVE_GRID,
) -> dict:
    """Model options for ONE pass of the dual-pass loop (plan D5).

    Clones ``model.model_options`` nested and, when ``enabled``, APPENDS the
    toolkit patches to ``transformer_options["patches"]["attn1_patch"]`` /
    ``["post_input"]`` — ``patches.get(name, []) + [patch]``, comfy's own
    ``set_model_patch`` append semantics (model_patcher.py:662-666): chained
    SPA/HAP patches are preserved, no existing list is mutated, and
    ``m.model_options`` is never touched. Pass A builds with ``enabled=False``
    and gets a clone with no I-Max patches; pass B builds with ``True``.

    ``arch_profile`` selects the D8 seam (plan 2026-10-06): ``"flux"`` keeps
    the ``attn1_patch`` q pre-scale; ``"zimage"`` installs
    :func:`make_attention_scale_override` on the GLOBAL
    ``transformer_options["optimized_attention_override"]`` instead — lumina
    blocks have no ``attn1_patch`` seam — chaining any pre-existing override
    found on ``model.model_options`` (comfy's own
    ``set_model_optimized_attention`` writes the same key,
    model_patcher.py:688-695). ``attention_anchor`` is that override's
    native sequence length: the wiring passes the native image grid plus
    the padded caption length (D8 cap accounting).

    The D4 unet function wrapper is NOT installed here — it rides inside
    ``model_options["model_function_wrapper"]`` on the clone (set once via
    ``set_model_unet_function_wrapper``, model_patcher.py:656-657) and is
    therefore carried by BOTH pass dicts; pass discrimination is a state
    cell closed over by the wrapper itself (P6).
    """
    if arch_profile not in ("flux", "zimage"):
        raise ValueError(
            f"arch_profile must be 'flux' or 'zimage'; got {arch_profile!r}")
    options = _clone_model_options(model.model_options)
    if not enabled:
        return options
    transformer_options = options.setdefault("transformer_options", {})
    patches = transformer_options.setdefault("patches", {})
    if proportional_attention:
        if arch_profile == "zimage":
            previous_override = (
                model.model_options.get("transformer_options") or {}
            ).get("optimized_attention_override")
            transformer_options["optimized_attention_override"] = (
                make_attention_scale_override(
                    anchor_seq_len=attention_anchor,
                    previous_override=previous_override,
                )
            )
        else:
            patches["attn1_patch"] = patches.get("attn1_patch", []) + [
                make_proportional_attention_patch(),
            ]
    if text_duplication:
        patches["post_input"] = patches.get("post_input", []) + [
            make_text_duplication_patch(),
        ]
    return options


# ---------------------------------------------------------------------------
# Model gate (plan P6 / D1 — FLUX-family flow models; Z-Image from v2.19.0)
# ---------------------------------------------------------------------------

def _resolve_diffusion_model(model):
    """Resolve ``diffusion_model`` the pack's patcher-aware way.

    ``ModelPatcher.get_model_object`` order (model_patcher.py:758-768):
    object patch -> backup -> live attribute. Plain mocks without the method
    fall back to the live attribute (the effective_model_sampling pattern).
    """
    get_model_object = getattr(model, "get_model_object", None)
    if callable(get_model_object):
        try:
            return get_model_object("diffusion_model")
        except AttributeError:
            pass  # degrade: path missing on the patcher's BaseModel
    return getattr(getattr(model, "model", None), "diffusion_model", None)


@dataclass(frozen=True)
class _ArchProfile:
    """The static per-arch facts the gate resolves once (plan D1/D2).

    ``name`` is the arch id the gate returns (``"flux"`` | ``"zimage"``);
    ``embedder_attr`` is the RoPE seam I-Max swaps on the diffusion model —
    Flux MMDiT: ``pe_embedder``; Lumina2 ``NextDiT``: ``rope_embedder``
    (comfy ldm/lumina/model.py:634, called per token group, never joint).
    """

    name: str
    embedder_attr: str


_FLUX_PROFILE = _ArchProfile(name="flux", embedder_attr="pe_embedder")
_ZIMAGE_PROFILE = _ArchProfile(name="zimage", embedder_attr="rope_embedder")

# Z-Image's RoPE base (comfy model_detection.py:604). Plain Lumina2 shares
# the NextDiT arch but trains with theta=10000 (model_detection.py:593) and
# different axes_lens — the theta on the installed embedder is what
# discriminates a Z-Image checkpoint from the Lumina2 arch it derives from.
_Z_IMAGE_ROPE_THETA = 256.0


def _resolve_arch_profile(model, diffusion_model) -> _ArchProfile:
    """Resolve the arch profile from the BaseModel MRO (plan D1/D2).

    Order is the order a user can act on:

    1. Flux MRO — the v1 profile, accepted outright;
    2. ``MingImage`` (the multi-frame Z-Image variant, comfy
       supported_models.py:1246) — rejected BY MRO NAME before the Lumina2
       accept, with its own message;
    3. Lumina2 MRO — the Z-Image family, behind two guards: the pixel-space
       variant (``ZImagePixelSpace`` latent format — rejected by that class
       name since it passes a Lumina2 MRO check) and the rope theta (plain
       Lumina2 passes MRO but trains theta=10000). A theta-256 embedder is
       Z-Image; a missing ``rope_embedder`` is left to the seam check so it
       gets the swap-target message instead;
    4. anything else — the rejection pointing at HiFlow.
    """
    base = getattr(model, "model", None)
    base_mro = [c.__name__ for c in type(base).__mro__]
    if "Flux" in base_mro:
        return _FLUX_PROFILE

    if "MingImage" in base_mro:
        raise ValueError(
            "I-Max does not support MingImage (the multi-frame Z-Image "
            "variant): its ref_frames conditioning and pad geometry differ "
            "from the Z-Image base arch the z-image profile is calibrated "
            "against."
        )

    if "Lumina2" in base_mro:
        latent_format_name = type(
            getattr(base, "latent_format", None)).__name__
        if latent_format_name == "ZImagePixelSpace":
            raise ValueError(
                "I-Max does not support the Z-Image pixel-space variant "
                "(ZImagePixelSpace operates on raw RGB patches, no VAE "
                "latents): the dual-pass x0 path decodes and re-encodes "
                "VAE latents."
            )
        rope_embedder = getattr(diffusion_model, "rope_embedder", None)
        if rope_embedder is not None:
            found_theta = getattr(rope_embedder, "theta", None)
            try:
                theta = float(found_theta)
            except (TypeError, ValueError):
                theta = None
            if theta != _Z_IMAGE_ROPE_THETA:
                raise ValueError(
                    "I-Max's z-image profile is calibrated to Z-Image's "
                    f"RoPE base theta={_Z_IMAGE_ROPE_THETA} (comfy "
                    "model_detection.py:604); this Lumina2-arch model's "
                    f"rope_embedder reports theta={found_theta!r} (plain "
                    "Lumina2 trains at theta=10000 with different "
                    "axes_lens)."
                )
        return _ZIMAGE_PROFILE

    raise ValueError(
        "I-Max supports the FLUX and Z-Image families (its NTK RoPE, text "
        "duplication and guidance math are written against their MMDiT "
        "wiring); this model's arch is "
        f"{type(getattr(model, 'model', None)).__name__}. For other "
        f"flow models (Qwen-Image, AuraFlow, ...) use the HiFlow node."
    )


def _require_flux_flow_model(model) -> tuple[str, int]:
    """Gate I-Max to the supported rectified-flow arches (plan D1 scope).

    I-Max is written against two MMDiT wirings — the Flux family (the
    ``pe_embedder`` RoPE seam, the ``attn1_patch``/``post_input`` patch
    contracts and the 3-axis ``txt_ids`` grid) and, from v2.19.0, Z-Image
    (the Lumina2 ``rope_embedder`` seam, per-group RoPE ids, plan
    2026-10-06). Checks, in the order a user can act on:

    1. flow prediction (rectified flow: CONST / img_to_img_flow /
       cosmos_rflow — the HiFlow gate, resolved through the patcher);
    2. arch profile (:func:`_resolve_arch_profile` — Flux MRO, or Lumina2
       MRO behind the ZImagePixelSpace / MingImage / theta-256 guards);
    3. the D4 swap target exists:
       ``diffusion_model.<profile.embedder_attr>`` (nunchaku builds route
       RoPE through ``model.pos_embed`` instead).

    3D-FORMAT latent models (Wan21) would pass 1 but die at 2 — the v1
    answer for them is HiFlow. Returns ``(profile.name,
    latent_dimensions)``.
    """
    # Patch-resolved (KSampler semantics): a schedule leaked by a previous
    # run's patch node must not flip this gate.
    model_sampling = effective_model_sampling(model)
    detected = _detect_prediction_type(model_sampling)
    mro_names = [c.__name__ for c in type(model_sampling).__mro__]
    if detected == "const" and "IMG_TO_IMG_FLOW" in mro_names:
        detected = "img_to_img_flow"
    elif detected == "const" and "COSMOS_RFLOW" in mro_names:
        detected = "cosmos_rflow"

    if detected not in ("const", "img_to_img_flow", "cosmos_rflow"):
        raise ValueError(
            f"I-Max needs a rectified-flow model (FLUX family); this model "
            f"predicts '{detected.upper()}'. For SD/SDXL-style models use "
            f"the PixelRush node instead."
        )

    diffusion_model = _resolve_diffusion_model(model)
    profile = _resolve_arch_profile(model, diffusion_model)

    if diffusion_model is None or not hasattr(
            diffusion_model, profile.embedder_attr):
        raise ValueError(
            f"I-Max could not find diffusion_model.{profile.embedder_attr} "
            "on this model (nunchaku builds route RoPE through "
            "model.pos_embed). The D4 positional scaling has no seam to "
            "install on."
        )

    latent_dimensions = getattr(
        getattr(model, "model", None).latent_format, "latent_dimensions", 2)
    if latent_dimensions not in (2, 3):
        raise ValueError(
            f"I-Max supports 2D or 3D-format image latents; this model "
            f"reports latent_dimensions={latent_dimensions}."
        )
    return profile.name, int(latent_dimensions)


def _apply_guidance_override(conditioning, guidance_value) -> list:
    """D12: write the FLUX guidance embed into a COPY of a conditioning list.

    0.0 (or None) = passthrough — whatever a chained FluxGuidance set wins
    (node_helpers.conditioning_set_values semantics, node_helpers.py:9-22:
    every entry's option dict is copied, the caller's objects are never
    mutated). The key is ``guidance`` (comfy model_base.py:1046-1048).
    """
    if guidance_value is None or float(guidance_value) <= 0.0:
        return conditioning
    out = []
    for entry in conditioning or []:
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            tensor, opts = entry
            opts = dict(opts) if isinstance(opts, dict) else opts
            opts["guidance"] = float(guidance_value)
            out.append((tensor, opts))
        else:
            out.append(entry)
    return out


# ---------------------------------------------------------------------------
# D4 unet function wrapper — the per-pass positional-embedder swap (D10)
# ---------------------------------------------------------------------------

def _make_imax_unet_wrapper(
    pass_state: dict,
    ntk_factor: float = 10.0,
    ntk_clip: bool = True,
    previous_wrapper: Callable | None = None,
    embedder_attr: str = "pe_embedder",
    clip_mode: str = "joint",
    text_tokens: int | None = None,
) -> Callable:
    """Build the ``model_function_wrapper`` that swaps the RoPE embedder.

    comfy consumes ``model_options["model_function_wrapper"]`` per batch as
    ``wrapper(model.apply_model, {"input","timestep","c","cond_or_uncond"})``
    (samplers.py:332-335) — so the wrapper receives the BaseModel (via the
    bound method's ``__self__``; the ``pass_state["inner_model"]`` fallback
    covers plain test callables) and can swap the positional embedder around
    each forward. The seam is arch-dependent (D10): ``embedder_attr`` is
    ``pe_embedder`` on the Flux MMDiT, ``rope_embedder`` on the lumina
    ``NextDiT`` (comfy ldm/lumina/model.py:634).

    - pass A (``pass_state["high_pass"] is False``): call through unchanged —
      the model runs exactly as installed (D4: pass A is unmodified);
    - pass B: resolve the CURRENTLY installed embedder per forward (so a
      chained DyPE/SEGA object-patched embedder is seen at call time), wrap
      it in :class:`IMaxNTKEmbedder` (constructed once per distinct inner —
      that is where the takeover warning fires, once per run), swap it in and
      restore the original in ``finally``. No ``add_object_patch``, no
      patch/unpatch window, nothing to leak (D4).

    ``clip_mode`` feeds the embedder's constructor (``"per_group"`` serves the
    lumina per-token-group wiring, plan 2026-10-06 D4'); ``text_tokens``
    records the D8 padded caption length on the constructed embedder
    (``None`` — the default — keeps the embedder's flux default, D7).
    ``previous_wrapper`` (a chained DyPE wrapper found on the cloned
    model_options) is called THROUGH: our swap happens first, then the
    previous wrapper keeps its per-forward state updates and performs the
    forward — replacing it outright would silently drop its behavior.
    """

    def _call_through(model_function: Callable, params: dict):
        if previous_wrapper is not None:
            return previous_wrapper(model_function, params)
        return model_function(
            params["input"], params["timestep"], **params.get("c", {}))

    def wrapper(model_function: Callable, params: dict) -> Tensor:
        if not pass_state.get("high_pass"):
            return _call_through(model_function, params)
        inner_model = getattr(model_function, "__self__", None) \
            or pass_state.get("inner_model")
        diffusion_model = getattr(inner_model, "diffusion_model", None)
        if diffusion_model is None or not hasattr(
                diffusion_model, embedder_attr):
            raise ValueError(
                f"I-Max could not find diffusion_model.{embedder_attr} on "
                "the model for the high-resolution pass."
            )
        installed = getattr(diffusion_model, embedder_attr)
        imax_embedder = pass_state.get("imax_embedder")
        if imax_embedder is None or pass_state.get("embedder_inner") \
                is not installed:
            imax_embedder = IMaxNTKEmbedder(
                installed, ntk_factor=ntk_factor, ntk_clip=ntk_clip,
                clip_mode=clip_mode)
            if text_tokens is not None:
                imax_embedder.set_text_tokens(text_tokens)
            pass_state["imax_embedder"] = imax_embedder
            pass_state["embedder_inner"] = installed
        setattr(diffusion_model, embedder_attr, imax_embedder)
        try:
            return _call_through(model_function, params)
        finally:
            setattr(diffusion_model, embedder_attr, installed)

    return wrapper


# ---------------------------------------------------------------------------
# x0 adapter (plan P6 — the nodes/hiflow.py:89-200 pattern)
# ---------------------------------------------------------------------------

def _make_predict_x0(
    model,
    positive,
    negative,
    cfg_scale: float,
    model_options: dict,
    pass_state: dict,
    high_pass: bool,
    guidance_override: float = 0.0,
    latent_dimensions: int = 2,
) -> Callable[[Tensor, float], Tensor]:
    """Create a pass-bound x0 adapter: ``(x_vae, sigma) -> x0_vae``.

    Mirrors ``nodes/hiflow.py:89-200`` — conditioning prepared once per
    latent shape via convert_cond -> process_conds, per-call
    ``comfy.samplers.sampling_function`` (full CFG, areas, control nets),
    VAE<->model conversions bracketing the model call, CFG auto-skip for
    token-less negatives — with three I-Max differences:

    - ``model_options`` is the PASS dict (D5): the pass-A clone carries no
      I-Max patches, the pass-B clone carries the toolkit patches; both
      carry the D4 unet wrapper (it rides inside model_options);
    - the adapter flips ``pass_state["high_pass"]`` to its own pass before
      every model call — that state cell is what the D4 wrapper reads to
      decide whether to swap the RoPE embedder (P6);
    - ``guidance_override`` (D12) rewrites the conditioning's ``guidance``
      embed on copies before conversion (0.0 = passthrough).
    """
    import comfy.model_management
    import comfy.sampler_helpers
    import comfy.samplers

    device = model.load_device if hasattr(model, "load_device") \
        else torch.device("cpu")
    inner_model = model.model
    model_sampling = effective_model_sampling(model)
    process_latent_in = getattr(inner_model, "process_latent_in", None)
    process_latent_out = getattr(inner_model, "process_latent_out", None)

    comfy.model_management.load_models_gpu([model])
    model.pre_run()

    positive = _apply_guidance_override(positive, guidance_override)
    negative = _apply_guidance_override(negative, guidance_override)

    _conds_by_shape: dict[tuple, dict] = {}

    def _get_conds(shape: tuple) -> dict:
        key = tuple(shape)
        if key not in _conds_by_shape:
            conds = {
                "positive": comfy.sampler_helpers.convert_cond(positive),
                "negative": comfy.sampler_helpers.convert_cond(negative),
            }
            noise = torch.zeros(shape, device=device)
            _conds_by_shape[key] = comfy.samplers.process_conds(
                inner_model, noise, conds, device,
            )
        return _conds_by_shape[key]

    # CFG guard (the Z-Image bugfix pattern): with a NEGATIVE that carries no
    # tokens, CFG amplifies a meaningless difference. Mirror ComfyUI's cfg=1
    # skip: run the conditional branch only.
    _has_negative = False
    for _entry in negative or []:
        if isinstance(_entry, (tuple, list)) and len(_entry) == 2:
            _tensor, _opts = _entry
            _tokens = 0
            if torch.is_tensor(_tensor):
                _tokens = int(_tensor.numel())
            elif isinstance(_tensor, (list, tuple)):
                _tokens = sum(
                    int(t.numel()) if torch.is_tensor(t) else len(t)
                    for t in _tensor
                )
            if _tokens > 0:
                _has_negative = True
                break
    if not _has_negative:
        cfg_scale = 1.0
        logger.info(
            "I-Max: negative conditioning carries no tokens — running "
            "the conditional branch only (CFG skipped, scale forced to 1.0)"
        )

    def predict_x0(x_vae: Tensor, sigma: float) -> Tensor:
        # Pass discrimination for the D4 wrapper (read inside this call's
        # forward; see _make_imax_unet_wrapper).
        pass_state["high_pass"] = high_pass
        x = x_vae.to(device)
        # 3D-format models need the 5D tensor (the Wan21 stats contract) —
        # inert for FLUX (latent_dimensions == 2).
        was_4d = x.dim() == 4
        if was_4d and latent_dimensions == 3:
            x = x.unsqueeze(2)  # [B, C, 1, H, W]
        if process_latent_in is not None:
            x = process_latent_in(x)

        conds = _get_conds(tuple(x_vae.shape))
        sigma_t = torch.tensor([float(sigma)], device=device)
        timestep = model_sampling.timestep(sigma_t)

        x0 = comfy.samplers.sampling_function(
            inner_model, x, timestep,
            uncond=conds["negative"], cond=conds["positive"],
            cond_scale=cfg_scale,
            model_options=model_options,
        )
        if process_latent_out is not None:
            x0 = process_latent_out(x0)
        if was_4d and x0.dim() == 5:
            x0 = x0.squeeze(2)
        return x0.to(x_vae.dtype).to(x_vae.device)

    return predict_x0


# ---------------------------------------------------------------------------
# VAE adapters + guidance latent (plan P6; nodes/hiflow.py:207-269 pattern)
# ---------------------------------------------------------------------------

def _downscale_ratio(vae) -> int:
    """The VAE's latent->pixel downscale ratio (tuple-tolerant, Qwen style)."""
    ratio = getattr(vae, "downscale_ratio", 8)
    if isinstance(ratio, (tuple, list)):
        ratio = ratio[1]  # (callable, h_ratio, w_ratio) convention
    return int(ratio)


def _upsample_guidance_latent(
    low_latent: Tensor,
    target_hw: tuple[int, int],
) -> Tensor:
    """Pass-A result -> the FIXED target-geometry guidance latent.

    Latent-space bicubic upsample of pass A's clean prediction — the
    reference (pipeline_flux_imax.py:711-718) upsamples in IMAGE space
    (VAE decode -> bicubic -> re-encode), but re-encoding does not preserve
    latent statistics: on z_image_turbo the re-encoded G carried ~1.8x the
    natural latent std and ~35x the high-frequency energy (the encoder
    re-normalizes contrast and its sampling adds broadband noise), and the
    schedules injected that off-manifold signal into every step — the
    "noisy / muted / needs more denoise steps" failure at 1024 (round trip
    at identity size, fixed first) and at extrapolated targets alike
    (measured 2026-10-08: latent-space G keeps std within ~0.5% of pass A's
    and the final gains 2.7x the edge energy at 1536 px).

    Runs ONCE per generation — the result is fixed for the whole pass B,
    and the engine derives P(G) from it exactly once more (haar_lowpass).
    Fully deterministic (no VAE sampling draw); fingerprint_inputs still
    keeps the node uncached for ComfyUI-cache safety.

    ``target_hw`` is the TARGET LATENT grid ``(H, W)`` (pass B runs there);
    interpolation is per-channel on the [B, C, h, w] latent.
    """
    up = torch.nn.functional.interpolate(
        low_latent, size=tuple(target_hw), mode="bicubic",
        align_corners=False,
    )
    return up.to(dtype=low_latent.dtype)    # [B, C, h_t, w_t]


def _wrap_final_x0_call(
    predict_x0: Callable[[Tensor, float], Tensor],
    total_steps: int,
    on_final: Callable[[Tensor], None],
) -> Callable[[Tensor, float], Tensor]:
    """Fire ``on_final(x0)`` on the adapter's LAST scheduled call.

    The bridge that lets the node own the guidance build while the engine
    owns the loop: the engine's final Euler step of pass A lands on σ=0,
    and x + (x − x̂₁)/σ · (0 − σ) = x̂₁ analytically — the model's last clean
    prediction IS the final low-res latent (fp32 deviation ~1e-7, far below
    any later processing). ``on_final`` therefore runs inside the last
    pass-A step, BEFORE the engine computes P(G) from the guidance buffer
    the callback fills (see IMaxNode.execute).
    """
    calls = {"n": 0}

    def wrapped(x: Tensor, sigma: float) -> Tensor:
        x0 = predict_x0(x, sigma)
        calls["n"] += 1
        if calls["n"] == total_steps:
            on_final(x0)
        return x0

    return wrapped


# ---------------------------------------------------------------------------
# D8 cap-token accounting (plan 2026-10-06 — the z-image caption padding)
# ---------------------------------------------------------------------------

# Z-Image's caption pad multiple: comfy sets pad_tokens_multiple=32 exactly
# when cap_pad_token is in the state dict (model_detection.py:613-614) and
# embed_cap applies it via pad_zimage (ldm/lumina/model.py:418-420, 663-665).
# The value is not stored as a runtime attribute, so it is pinned here.
_CAP_PAD_MULTIPLE = 32


def _cap_padded_length(positive, diffusion_model) -> int:
    """D8 cap accounting: the caption length the model will actually see.

    Z-Image pads its text tokens to :data:`_CAP_PAD_MULTIPLE` (``pad_zimage``
    appends ``(-len) % multiple`` pad rows, comfy ldm/lumina/model.py:418-420)
    and the PADDED length is what drives ``cap_pos_ids``, ``cap_size`` and the
    image start-t (``embed_cap``, model.py:663-673) — so it is the length the
    per-group RoPE groups and the D4 attention anchor must count. The padding
    applies iff the diffusion model carries the ``cap_pad_token`` nn.Parameter
    (NextDiT.__init__ creates it exactly when ``pad_tokens_multiple`` is set);
    without it (older checkpoints / plain-Lumina2 shape) the raw count is used.

    ``context_len`` is the positive conditioning's token count — max over the
    entries: ``opts["num_tokens"]`` when an entry carries it (best effort:
    stock comfy derives num_tokens the other way, as an extra_conds
    CONDConstant, model_base.py:1519-1525), else the text tensor's token axis
    (dim 1).
    """
    context_len = 0
    for entry in positive or []:
        if not (isinstance(entry, (tuple, list)) and len(entry) == 2):
            continue
        tensor, opts = entry
        count = None
        if isinstance(opts, dict) and opts.get("num_tokens") is not None:
            count = int(opts["num_tokens"])
        elif torch.is_tensor(tensor) and tensor.ndim >= 2:
            count = int(tensor.shape[1])
        if count is not None:
            context_len = max(context_len, count)
    if context_len < 1 or not hasattr(diffusion_model, "cap_pad_token"):
        return context_len
    return context_len + (-context_len % _CAP_PAD_MULTIPLE)


# ---------------------------------------------------------------------------
# I-Max node (plan D1, D11-D15)
# ---------------------------------------------------------------------------

class IMaxNode(io.ComfyNode):
    """I-Max — tuning-free resolution extrapolation for FLUX (arXiv 2410.07536).

    Pass A generates at the native-area low resolution unpatched; its final
    clean prediction becomes a FIXED guidance latent at the target geometry
    (used directly at the target geometry, latent-space bicubic upsampled
    below it — no VAE round trip: re-encoding distorts latent statistics,
    measured on z_image_turbo 2026-10-08); pass B at the target resolution
    runs with the NTK RoPE embedder + proportional attention + text
    duplication active and its clean predictions pulled toward the low pass
    of the guidance (Projected Flow, paper §2.2).
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="IMax",
            display_name="I-Max",
            category="WMNodes/image",
            description=(
                "Tuning-free resolution extrapolation for FLUX (arXiv "
                "2410.07536): a low-resolution pass builds a fixed guidance "
                "latent; the target-resolution pass is pulled toward its "
                "low pass while NTK RoPE, proportional attention and text "
                "duplication compensate the resolution gap. Feed a latent "
                "at the TARGET size (EmptySD3LatentImage)."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The FLUX-family flow model."),
                io.Vae.Input(
                    "vae",
                    tooltip="VAE — used for the latent/pixel geometry "
                            "(downscale ratio); the guidance latent itself "
                            "is built in latent space."),
                io.Conditioning.Input(
                    "positive", tooltip="Positive conditioning."),
                io.Conditioning.Input(
                    "negative", tooltip="Negative conditioning."),
                io.Latent.Input(
                    "latent_image",
                    tooltip="Latent at the TARGET resolution (e.g. "
                            "EmptySD3LatentImage at the desired output "
                            "size). An empty latent is pure txt2img; a "
                            "content latent + denoise < 1 is img2img."),
                io.Int.Input(
                    "noise_seed", default=0, min=0, max=2**32 - 1, step=1,
                    tooltip="One shared generator: pass A's start noise, "
                            "then pass B's img2img noise — same seed, same "
                            "result (modulo the VAE encode sampling)."),
                io.Float.Input(
                    "denoise", default=1.0, min=0.05, max=1.0, step=0.05,
                    tooltip="Img2img strength for PASS B only (KSampler "
                            "convention); pass A always starts from pure "
                            "noise. Ignored for an empty latent — that "
                            "always runs the full schedule."),
                io.Float.Input(
                    "cfg", default=1.0, min=0.0, max=20.0, step=0.1,
                    tooltip="Classifier-free guidance for BOTH passes. "
                            "FLUX-dev: leave at 1.0 — the guidance embeds "
                            "below do the work. CFG is auto-skipped when "
                            "the negative carries no tokens."),
                io.Int.Input(
                    "steps_low", default=30, min=1, max=200, step=1,
                    tooltip="Pass A (low-resolution guidance) steps — "
                            "paper: 30."),
                io.Int.Input(
                    "steps_high", default=20, min=1, max=200, step=1,
                    tooltip="Pass B (target resolution) steps — paper: 20."),
                io.Float.Input(
                    "guidance_low", default=3.5, min=0.0, max=20.0, step=0.1,
                    tooltip="FLUX guidance embed written into a copy of the "
                            "conditioning for pass A (paper/gradio 3.5). "
                            "0.0 keeps whatever a chained FluxGuidance set."),
                io.Float.Input(
                    "guidance_high", default=5.0, min=0.0, max=20.0,
                    step=0.1,
                    tooltip="FLUX guidance embed for pass B (gradio 5.0). "
                            "0.0 keeps whatever a chained FluxGuidance set."),
                io.Float.Input(
                    "time_shift_low", default=3.0, min=0.01, max=10.0,
                    step=0.05,
                    tooltip="Static flow shift of pass A's sigma schedule "
                            "(paper: 3.0) — I-Max owns both schedules, no "
                            "model_sampling patch is involved."),
                io.Float.Input(
                    "time_shift_high", default=6.0, min=0.01, max=10.0,
                    step=0.05,
                    tooltip="Static flow shift of pass B's schedule (paper: "
                            "6.0) — the SNR re-balance between the passes."),
                io.Float.Input(
                    "ntk_factor", default=10.0, min=1.0, max=100.0, step=0.5,
                    tooltip="NTK-aware RoPE base multiplier for the "
                            "high-resolution pass (paper: 10 for "
                            "Flux.1-dev). 1.0 = plain RoPE."),
                io.Int.Input(
                    "dwt_level", default=1, min=1, max=8, step=1,
                    tooltip="Haar low-pass level of the guidance projection "
                            "(paper: 1). Higher = coarser guidance detail."),
                io.Combo.Input(
                    "guidance_schedule",
                    options=["disable", "cosine_decay", "cosine_shift",
                             "constant"],
                    default="cosine_decay",
                    tooltip="Projected-Flow guidance schedule (paper §2.2): "
                            "cosine_decay is the README/gradio default; "
                            "disable runs pass B as plain Euler."),
                io.Float.Input(
                    "guidance_strength", default=1.0, min=0.0, max=2.0,
                    step=0.05,
                    tooltip="Linear scale on the Projected-Flow correction "
                            "(1.0 = the paper's unscaled pull; 0 = disable). "
                            "Distilled checkpoints (Z-Image Turbo) hallucinate "
                            "swirls/noise at extrapolated targets at full "
                            "strength — use 0.25-0.5 there."),
                io.Boolean.Input(
                    "proportional_attention", default=True,
                    tooltip="Scale the attention temperature with the joint "
                            "sequence length (paper §2.3). Clamped to a "
                            "no-op at/below the native 1024 px."),
                io.Boolean.Input(
                    "text_duplication", default=True,
                    tooltip="Duplicate the text tokens per native 1024 px "
                            "tile so the image/text token ratio stays in "
                            "distribution (paper §2.3). No-op at/below "
                            "1024 px."),
                io.Float.Input(
                    "low_res_scale", default=1.0, min=0.25, max=2.0,
                    step=0.05,
                    tooltip="Scales the LOW-RES pass AREA (1.0 = the "
                            "paper's native-area guidance; 0.5 halves it)."),
            ],
            outputs=[
                io.Latent.Output(display_name="High-Res Latent"),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """Never serve this node from the cache (D15).

        The guidance latent is deterministic (no VAE sampling since the
        latent-space upsample), but the node stays always-rerun for
        ComfyUI-cache safety — model/option patches can change results
        without changing visible inputs. NaN compares unequal to itself —
        the canonical always-rerun fingerprint.
        """
        return float("nan")

    @classmethod
    def execute(cls, model, vae, positive, negative, latent_image,
                noise_seed=0, denoise=1.0, cfg=1.0,
                steps_low=30, steps_high=20, guidance_low=3.5,
                guidance_high=5.0, time_shift_low=3.0, time_shift_high=6.0,
                ntk_factor=10.0, dwt_level=1,
                guidance_schedule="cosine_decay", guidance_strength=1.0,
                proportional_attention=True, text_duplication=True,
                low_res_scale=1.0) -> io.NodeOutput:
        import comfy.utils

        # Gate BEFORE any model calls: FLUX/Z-Image rectified flow only (D1).
        _, latent_dimensions = _require_flux_flow_model(model)
        warn_if_stale_leak(model, "I-Max")

        # ---- Arch profile branch (plan 2026-10-06 D5/D6/D8/D10). ----------
        # The gate just accepted this model, so re-resolving the profile is
        # pure (the same MRO/attr reads — no guard can fire twice).
        diffusion_model = _resolve_diffusion_model(model)
        profile = _resolve_arch_profile(model, diffusion_model)
        is_zimage = profile.name == "zimage"
        # D8 cap accounting: the PADDED caption length z-image's embed_cap
        # produces drives the D4 attention anchor (native image grid 64x64 +
        # cap) and the embedder's text-token bookkeeping; the flux joint
        # route reads neither.
        cap_padded = (
            _cap_padded_length(positive, diffusion_model)
            if is_zimage else 0
        )
        attention_anchor = _NATIVE_GRID * _NATIVE_GRID + cap_padded
        # D5: text duplication is FLUX-only — the paper applies it to MMDiT
        # Flux (Lumina-Next uses cross-attention) and lumina blocks have no
        # post_input seam. Warn once per run and install nothing.
        duplication_enabled = bool(text_duplication)
        if duplication_enabled and is_zimage:
            logger.warning(
                "I-Max: text duplication is ignored for Z-Image — the paper "
                "applies text duplication to MMDiT Flux only (Lumina-Next "
                "uses cross-attention) and the lumina blocks have no "
                "post_input seam."
            )
            duplication_enabled = False
        # Distilled z-image (Turbo) re-sharpens far less than Flux after
        # each projected-flow pull, and full-strength corrections
        # hallucinate swirls at extrapolated targets (z_image_turbo
        # measurements, 2026-10-08: |δ|/|x0| up to 0.74 per early step;
        # strength 0.25-0.5 keeps the anchoring at control-level noise).
        if is_zimage and guidance_schedule != "disable":
            logger.warning(
                "I-Max: on Z-Image, guidance schedules trade some contrast "
                "for low-pass structure-following (distilled checkpoints "
                "re-sharpen less than Flux after each pull, and full-"
                "strength corrections hallucinate swirls at extrapolated "
                "targets). Set guidance_strength 0.25-0.5 for clean "
                "structure-following, or guidance_schedule=disabled for the "
                "model's native look; on Turbo, steps_low 8-12 with "
                "time_shift_low=3.0 keeps the low pass sharp."
            )

        # Clone FIRST: the unet wrapper and per-pass options are run-scoped
        # and must never outlive this node onto the caller's patcher (D4).
        # The prefix-cache disable rides the same clone (Qwen-2.1 no-op here).
        model = disable_prefix_kv_cache(model)

        if isinstance(latent_image, dict):
            content_latent = latent_image["samples"]
        else:
            content_latent = latent_image
        if content_latent.ndim == 5:
            if content_latent.shape[2] != 1:
                raise ValueError(
                    "I-Max received a multi-frame (video) latent "
                    f"(T={content_latent.shape[2]}). It supports single-"
                    "frame image latents only — the dual pass is 2D per "
                    "frame."
                )
            content_latent = content_latent.squeeze(2)  # [B, C, H, W]

        device = model.load_device if hasattr(model, "load_device") \
            else torch.device("cpu")
        content_latent = content_latent.to(device)

        # Channel handling for empty latents (EmptyLatentImage may produce 4
        # channels for a 16-channel model) — the PixelRush convention.
        model_latent_channels = getattr(
            model.model.latent_format, "latent_channels", None)
        if model_latent_channels is not None and \
                content_latent.shape[1] != model_latent_channels:
            if torch.count_nonzero(content_latent) == 0:
                logger.info(
                    "I-Max: empty input latent has %d channels, model "
                    "expects %d — repeating channels",
                    content_latent.shape[1], model_latent_channels,
                )
                content_latent = comfy.utils.repeat_to_batch_size(
                    content_latent, model_latent_channels, dim=1,
                )
            else:
                logger.warning(
                    "I-Max: non-empty input latent has %d channels, model "
                    "expects %d — results may be unexpected",
                    content_latent.shape[1], model_latent_channels,
                )

        h_t, w_t = int(content_latent.shape[-2]), int(content_latent.shape[-1])
        vae_ratio = _downscale_ratio(vae)

        # A content latent with denoise=1.0 is a foot-gun (HiFlow warning).
        if torch.count_nonzero(content_latent) > 0 and \
                float(denoise) > 0.9999:
            logger.warning(
                "I-Max: denoise=1.0 with a non-empty latent — the input "
                "image is ignored (pass B starts from pure noise). Lower "
                "denoise (e.g. 0.6) to condition on the connected latent."
            )

        # ---- D4 wrapper + D5 per-pass model options. -----------------------
        pass_state = {
            "high_pass": False,          # the pass-discrimination cell (P6)
            "inner_model": model.model,  # wrapper fallback for plain fns
            "imax_embedder": None,
            "embedder_inner": None,
        }
        previous_wrapper = model.model_options.get("model_function_wrapper")
        model.set_model_unet_function_wrapper(_make_imax_unet_wrapper(
            pass_state, ntk_factor=float(ntk_factor),
            previous_wrapper=previous_wrapper,
            embedder_attr=profile.embedder_attr,
            clip_mode="per_group" if is_zimage else "joint",
            # D8 bookkeeping: record the padded caption length on the
            # constructed embedder (None keeps the flux default, D7).
            text_tokens=cap_padded if is_zimage and cap_padded >= 1 else None,
        ))
        # Build the pass clones AFTER installing the wrapper: it rides inside
        # model_options, so BOTH pass dicts must carry it (D5/P6).
        options_low = build_pass_model_options(model, enabled=False)
        options_high = build_pass_model_options(
            model, enabled=True,
            proportional_attention=bool(proportional_attention),
            text_duplication=duplication_enabled,
            arch_profile=profile.name,
            attention_anchor=attention_anchor,
        )

        # ---- Pass-bound x0 adapters (D5, D12). -----------------------------
        # D6: the guidance cond key is FLUX-only — lumina extra_conds has no
        # guidance entry (Z-Image is guidance-distilled); 0.0 = passthrough.
        guidance_low_override = float(guidance_low) if not is_zimage else 0.0
        guidance_high_override = float(guidance_high) if not is_zimage else 0.0
        predict_x0_low = _make_predict_x0(
            model, positive, negative, cfg_scale=float(cfg),
            model_options=options_low, pass_state=pass_state,
            high_pass=False, guidance_override=guidance_low_override,
            latent_dimensions=latent_dimensions,
        )
        predict_x0_high = _make_predict_x0(
            model, positive, negative, cfg_scale=float(cfg),
            model_options=options_high, pass_state=pass_state,
            high_pass=True, guidance_override=guidance_high_override,
            latent_dimensions=latent_dimensions,
        )

        # ---- The guidance latent: fixed for all of pass B. -----------------
        # The engine takes the guidance as a static argument, but it is a
        # function of pass A's output — so the buffer below is filled IN the
        # last pass-A step (see _wrap_final_x0_call), before the engine
        # computes P(G) from it. Built exactly once per generation, entirely
        # in latent space (see _upsample_guidance_latent).
        guidance_buffer = torch.zeros_like(content_latent)
        target_hw = (h_t, w_t)
        target_shape = tuple(content_latent.shape)

        def _fill_guidance(x0_low: Tensor) -> None:
            # Pass A at the target latent geometry: nothing to upsample —
            # the guidance latent IS the pass-A output (no interpolation).
            if tuple(x0_low.shape) == target_shape:
                logger.info(
                    "I-Max: pass A ran at the target geometry — the "
                    "guidance latent IS the pass-A output."
                )
                guidance_buffer.copy_(x0_low.to(
                    device=guidance_buffer.device,
                    dtype=guidance_buffer.dtype))
                return
            guidance_buffer.copy_(_upsample_guidance_latent(
                x0_low, target_hw,
            ).to(device=guidance_buffer.device, dtype=guidance_buffer.dtype))

        predict_x0_low = _wrap_final_x0_call(
            predict_x0_low, int(steps_low), _fill_guidance)

        # ---- I-Max owns both sigma schedules (D3). -------------------------
        cfg_obj = IMaxConfig(
            steps_low=int(steps_low), steps_high=int(steps_high),
            time_shift_low=float(time_shift_low),
            time_shift_high=float(time_shift_high),
            dwt_level=int(dwt_level),
            guidance_schedule=str(guidance_schedule),
            guidance_strength=float(guidance_strength),
            denoise=float(denoise), low_res_scale=float(low_res_scale),
            pixels_per_latent=vae_ratio,
        )
        sigmas_low = build_flow_sigmas(int(steps_low), float(time_shift_low))
        sigmas_high = build_flow_sigmas(
            int(steps_high), float(time_shift_high))

        total = int(steps_low) + 1 + int(steps_high)
        pbar = comfy.utils.ProgressBar(total)
        counter = {"n": 0}

        def progress_callback(i, total_steps, stage):
            counter["n"] += 1
            pbar.update_absolute(min(counter["n"], total))

        # Model-space noising conversions (the v2.12.1 HiFlow fix); None on
        # format-less models falls back to the engine's VAE-space mix.
        inner_model = model.model
        result = imax_dual_pass(
            predict_x0_low, predict_x0_high, sigmas_low, sigmas_high,
            content_latent, guidance_buffer, int(noise_seed), cfg_obj,
            progress_callback,
            process_latent_in=getattr(inner_model, "process_latent_in", None),
            process_latent_out=getattr(
                inner_model, "process_latent_out", None),
        )
        pbar.update_absolute(total)
        return io.NodeOutput({"samples": result})
