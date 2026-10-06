"""
I-Max ComfyUI node layer — inference toolkit (plan 2026-10-05, P4-P5).

The inference-time compensations of I-Max (arXiv 2410.07536 §2.3) as ComfyUI
transformer patches, plus the per-pass ``model_options`` builder that installs
them on the high-resolution pass only (D5). The node itself — gates, adapters,
schema, execute — lands with P6 on top of this module.

Everything here is torch-only at MODULE scope: comfy modules are imported
LAZILY inside the functions that need them (the ``nodes/hiflow.py`` discipline
one level stricter — not even ``comfy_api`` at import time), so this layer
imports and unit-tests without a ComfyUI installation.

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
  replaces the dynamic scaling for the duration of the pass.
- D8 — proportional self-attention. The reference overrides the SDPA scale
  with ``sqrt(log(N_joint, 4608) / head_dim)`` (attention_processor.py:1773-
  1777 AND :1889-1893 — log BASE 4608, not the natural log of the ratio a
  casual read of plan D8's ``sqrt(log(N/4608))`` would suggest). As a q
  pre-scale in ``attn1_patch`` this is exact: the patch fires before RoPE and
  a scalar on q commutes with the rotation, so head_dim cancels and the patch
  is ``q *= sqrt(log(N, 4608))`` — clamped to exactly 1.0 at/below the native
  sequence length (the plan D8 clamp; at ``N = 4608`` the formula itself is
  continuously 1.0).
- D9 — text duplication via ``post_input``. The reference tiles the text
  stream ``nh*nw`` times (transformer_flux.py:364-375), offsetting each
  copy's position grid by ``(i*64, j*64)`` so every text copy shares a RoPE
  frame with one image tile. D9 keeps ``ceil`` tiles per side instead of the
  reference's ``floor`` (intentional: floor under-covers targets extending
  past a whole number of native tiles; the two agree at every power-of-two
  target). Downstream slicing stays correct because the Flux forward cuts
  with the POST-patch txt length (ldm/flux/model.py:306,407). At/below the
  native grid the patch is a provable no-op (nh = nw = 1).
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
from typing import Callable

import torch
from torch import Tensor

logger = logging.getLogger("ComfyUI-DyPE")

# Flux native training geometry: 64x64 latent patches (1024 px) + 512 text
# tokens = 4608 joint positions. The reference hardcodes 512 and 64**2
# (transformer_flux.py:52, attention_processor.py:1772); both derive from
# these two constants.
_TRAIN_SEQ_LEN = 4608
_NATIVE_GRID = 64


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

    dev = torch.device("cpu") if device is None else torch.device(device)
    text_tokens = train_seq_len - native_grid * native_grid
    # comfy's own grid form (ldm/flux/math.py:27) — bitwise parity with
    # EmbedND; NOT src/rope.py's arange form (DyPE layout, unusable here).
    scale = torch.linspace(
        0, (dim - 2) / dim, steps=dim // 2, dtype=torch.float64, device=dev,
    )
    omega = 1.0 / ((theta * ntk_factor) ** scale)
    if ntk_clip and seq_len > text_tokens:
        ratio = (seq_len - text_tokens) / (native_grid * native_grid)
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
    """I-Max NTK-aware scaled RoPE for FLUX: b' = b * ntk_factor (+ clip).

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
    """

    def __init__(
        self,
        inner,
        ntk_factor: float = 10.0,
        ntk_clip: bool = True,
        train_seq_len: int = _TRAIN_SEQ_LEN,
        native_grid: int = _NATIVE_GRID,
    ) -> None:
        super().__init__()
        theta = getattr(inner, "theta", None)
        axes_dim = getattr(inner, "axes_dim", None)
        if theta is None or not axes_dim:
            raise ValueError(
                "IMaxNTKEmbedder needs the installed pe_embedder to expose "
                ".theta and .axes_dim (comfy's EmbedND and the DyPE family "
                f"all do); got {type(inner).__name__}."
            )
        self.inner = inner
        self.theta = theta
        self.thetas = getattr(inner, "thetas", None)  # DyPE per-axis thetas
        self.axes_dim = list(axes_dim)
        self.ntk_factor = float(ntk_factor)
        self.ntk_clip = bool(ntk_clip)
        self.train_seq_len = int(train_seq_len)
        self.native_grid = int(native_grid)
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

    def forward(self, ids: Tensor) -> Tensor:
        """Position ids ``[b, n, axes]`` -> rope table ``[b, 1, n, D, 2, 2]``.

        N is the JOINT sequence length (text + image tokens) — the clip
        floor is a function of the whole stream, exactly as in the reference
        where each axis's rope call sees the full position tensor.
        """
        n_axes = int(ids.shape[-1])
        seq_len = int(ids.shape[1])
        device = _rope_compute_device(ids.device)
        embs = []
        for axis in range(n_axes):
            theta = (
                self.thetas[axis] if self.thetas is not None else self.theta
            )
            omega = ntk_rope_omega(
                theta, self.axes_dim[axis], self.ntk_factor, seq_len,
                ntk_clip=self.ntk_clip,
                train_seq_len=self.train_seq_len,
                native_grid=self.native_grid,
                device=device,
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
) -> dict:
    """Model options for ONE pass of the dual-pass loop (plan D5).

    Clones ``model.model_options`` nested and, when ``enabled``, APPENDS the
    toolkit patches to ``transformer_options["patches"]["attn1_patch"]`` /
    ``["post_input"]`` — ``patches.get(name, []) + [patch]``, comfy's own
    ``set_model_patch`` append semantics (model_patcher.py:662-666): chained
    SPA/HAP patches are preserved, no existing list is mutated, and
    ``m.model_options`` is never touched. Pass A builds with ``enabled=False``
    and gets a clone with no I-Max patches; pass B builds with ``True``.

    The D4 unet function wrapper is NOT installed here — it rides inside
    ``model_options["model_function_wrapper"]`` on the clone (set once via
    ``set_model_unet_function_wrapper``, model_patcher.py:656-657) and is
    therefore carried by BOTH pass dicts; pass discrimination is a state
    cell closed over by the wrapper itself (P6).
    """
    options = _clone_model_options(model.model_options)
    if not enabled:
        return options
    transformer_options = options.setdefault("transformer_options", {})
    patches = transformer_options.setdefault("patches", {})
    if proportional_attention:
        patches["attn1_patch"] = patches.get("attn1_patch", []) + [
            make_proportional_attention_patch(),
        ]
    if text_duplication:
        patches["post_input"] = patches.get("post_input", []) + [
            make_text_duplication_patch(),
        ]
    return options
