"""Krea-2 adapter — the geometry (plan 2026-10-20, §5.2/S2).

Krea-2 is a single-stream MMDiT (``comfy/ldm/krea2/model.py``): 28
``SingleStreamBlock``s, 48 query heads over 12 KV heads, a 12-layer Qwen3-VL
text-fusion adapter whose **four 20-head attention calls run before the main
blocks** (``model.py:341``), a FLUX-layout ``EmbedND`` rope embedder called
ONCE per forward (``model.py:364``), a joint ``[text | image]`` stream with no
padding (``model.py:355``), rope applied by the model itself BEFORE the seam
(``model.py:93``) and a ``[b, n, heads*d]`` output contract (``model.py:99-100``,
``skip_output_reshape`` is not passed).

**Geometry — ASSUMED, not measured.**  The two constants below decide how many
pixels one token covers and how large the model's trained canvas is.  The VAE
downscale is decided by the CHECKPOINT's VAE weights, not by the ComfyUI source
(``comfy/sd.py:641-667``), and no Krea-2 checkpoint exists in the dev
environment, so the values are the pack's own working hypothesis
(``src/spa.py:86-91``: "1024 / 8 / 2 = 64" tokens at 1024 px) and they are
LABELLED as such:

    ASSUMED — not measured; VAE downscale family unverified (no checkpoint in
    the dev environment)

The label is not decoration: :data:`KREA_GEOMETRY_PROVENANCE` and
:data:`KREA_GEOMETRY_NOTE` are read by the tests, so an unlabelled constant
fails the build.  A wrong pixel constant cannot corrupt an output — the
attention maths depends only on the TOKEN grid, which comes from the latent —
it can only mis-size a warning, which is why the constants are safe to ship
unmeasured (plan Q7).

**What a measured value changes.**  If the checkpoint's VAE turns out to be the
16x family (the Qwen-style ``decoder.conv_in.weight`` 32-channel branch,
``comfy/sd.py:655``), the token covers 32 px, so the trained extent in TOKENS
halves: :data:`KREA_VAE_DOWNSCALE` becomes 16, :data:`KREA_PX_PER_PATCH`
becomes 32 and :data:`KREA_NATIVE_TOKENS` becomes 32 — three one-line changes
in this file and nothing else.

**No comfy import here.**  The adapter names the seam's identity and owns the
tensors; the install lives in ``src/area.py``.  This module imports torch (and
the pack's own rope helpers) only, like every other adapter in the pack.

**The S5 parts — the tensors the seam needs.**

* **The rope recorder.**  The model builds its rope table ONCE per forward
  (``self.pe_embedder(pos)``, ``comfy/ldm/krea2/model.py:364``) and applies it
  itself BEFORE the seam (``model.py:93``), so the seam sees rotated q/k and no
  table.  :class:`KreaRopeRecorder` captures that one table through a forward
  hook on the model's own embedder, so a chained DyPE/SEGA embedder composes
  instead of being reconstructed.  The recorder keeps the **FIRST** table of a
  forward and never overwrites it; the coarse tables are built through the very
  same embedder, so :meth:`KreaRopeRecorder.paused` suspends the capture around
  them (the pause guard — without it the recorder would see the coarse tables
  as extra tables of the forward).
* **The joint layout.**  Krea-2's stream is ``cat((context, img), dim=1)``
  (``model.py:355``) — text FIRST, then the image, with NO padding — so the
  image tokens start exactly where the caption ends.  :class:`KreaJointLayout`
  is that geometry, and :func:`krea_layout` refuses a stream that is not
  ``txt_len + h*w`` long.
* **The coarse tables.**  Krea-2's rope ids are FLUX-shaped — the caption sits
  at the origin and every image token carries ``(0, row, col)`` with axis 0
  the FRAME index (``model.py:291-294``, ``:346``) — so the Phase I tables are
  FLUX's own recipe (floor-divided block coordinates, plan Q9) built through
  the model's OWN embedder: :func:`coarse_ropes_krea`.
* **The eligibility guard.**  The four text-fusion calls run with 20 heads and
  their stream length is a PREFIX of the joint stream's, so a length check
  alone cannot tell them apart.  :func:`krea_stream_eligible` is the exact
  guard: the head count, the published ``img_slice`` and the stream length
  together.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch

from .area_rope import flux_style_ids, head_shared
from .area_schedule import coarse_grid

logger = logging.getLogger("ComfyUI-DyPE")

__all__ = [
    "KREA_BACKBONE",
    "KREA_GEOMETRY_NOTE",
    "KREA_GEOMETRY_PROVENANCE",
    "KREA_NATIVE_TOKENS",
    "KREA_PATCH",
    "KREA_PX_PER_PATCH",
    "KREA_SEAM_MODULE",
    "KREA_SEAM_SYMBOL",
    "KREA_TRAINED_RESOLUTION_PX",
    "KREA_VAE_DOWNSCALE",
    "KreaJointLayout",
    "KreaRopeRecorder",
    "capture_krea_table",
    "coarse_ropes_krea",
    "krea_latent_grid",
    "krea_layout",
    "krea_stream_eligible",
    "krea_squeeze_latent",
]

#: One-time warnings (module scoped, like ``src/area_zimage.py``'s).
_warned: set = set()


def _warn_once(key: str, message: str, *args) -> None:
    """Emit ``message`` at WARNING level the first time ``key`` is seen."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


#: The seam's identity — the module that BINDS the symbol and the symbol
#: itself.  Krea-2 imports its own copy of ``optimized_attention_masked``
#: (``from comfy.ldm.modules.attention import ... optimized_attention_masked``,
#: ``comfy/ldm/krea2/model.py:21``) and calls the BOUND name at ``:99-100``,
#: so patching the defining module (``comfy.ldm.modules.attention``, shared
#: with FLUX/Qwen/Z-Image) would move every backend and patching the Krea-2
#: module moves only Krea-2.  Named here (plan §4) so the adapter's interface
#: has one definition; ``src/area.py`` does the import and the rebind.
KREA_SEAM_MODULE = "comfy.ldm.krea2.model"
KREA_SEAM_SYMBOL = "optimized_attention_masked"

#: The backbone's class name — the MRO marker the arch profile reads
#: (``src/area_arch.py``).
KREA_BACKBONE = "SingleStreamDiT"

#: The provenance of the geometry constants below: ``"ASSUMED"`` until a
#: Krea-2 checkpoint is measured (plan S2), ``"MEASURED"`` after.  The tests
#: read it, so a silent constant fails the build.
KREA_GEOMETRY_PROVENANCE = "ASSUMED"

#: The label the constants carry, verbatim — the plan's wording (S2 decision
#: rule): the VAE downscale is unverified because no checkpoint is available in
#: the dev environment.
KREA_GEOMETRY_NOTE = (
    "ASSUMED — not measured; VAE downscale family unverified "
    "(no checkpoint in the dev environment)"
)

#: The model's own patch factor — MEASURED from the source: ``self.patch =
#: patch`` with ``patch=2`` in the constructor's defaults
#: (``comfy/ldm/krea2/model.py:242``, ``comfy/model_detection.py:1043``).
KREA_PATCH = 2

#: The VAE's spatial downscale — ASSUMED.  comfy decides it from the
#: checkpoint's VAE weights, not from the model: the 32-channel 5-D
#: ``decoder.conv_in.weight`` branch is the 16x Qwen-style VAE, the default
#: SD-style branch is 8x (``comfy/sd.py:641-667``).  The pack's own hypothesis
#: is 8x (``src/spa.py:86-91``).
KREA_VAE_DOWNSCALE = 8

#: Pixels per token — ASSUMED (it is the product of the two constants above:
#: ``vae_downscale * patch``).  8x VAE and a 2x2 patch give 16 px per token,
#: the same token size as FLUX and Z-Image.
KREA_PX_PER_PATCH = KREA_VAE_DOWNSCALE * KREA_PATCH

#: The model's TRAINED extent, in TOKENS — ASSUMED.  1024 px over the 16 px
#: token grid is 64 tokens, the same number the pack uses for FLUX, Z-Image and
#: (assumed) Qwen-Image (``src/spa.py:86-91``).  Every rope distance the model
#: ever saw is inside it, so a canvas wider than this is compressed back into
#: it (the GTCN's canvas extent ratio, area-fix3).
KREA_NATIVE_TOKENS = 64

#: The trained canvas in PIXELS — ASSUMED (1024 px, the pack's hypothesis).
#: It is the cross-check of the two constants above: a trained extent in tokens
#: and a pixel factor that disagree would mis-size every advisory, so the
#: tests assert ``native_tokens * px_per_patch == trained resolution``.
KREA_TRAINED_RESOLUTION_PX = 1024


def krea_squeeze_latent(latent: torch.Tensor) -> torch.Tensor:
    """A Krea-2 latent without its singleton temporal axis.

    Krea-2's latent format is ``Wan21`` (``comfy/supported_models.py:2009-2035``),
    whose ``latent_dimensions == 3`` (``comfy/latent_formats.py:731``) makes
    comfy unsqueeze the sampler's 4-D latent to ``[B, 16, T, H, W]``
    (``comfy/sample.py:58-59``).  So the 5-D shape is the NORMAL case for an
    image, not an exotic one — and AReA's own guard
    (``if not torch.is_tensor(x) or x.ndim != 4``, ``src/area.py:1215``) would
    make AReA a **silent no-op on every Krea-2 image run**.  This helper undoes
    the unsqueeze for the ``T == 1`` case.

    A 4-D latent passes through unchanged (a loader that hands one over is fine).

    Raises:
        ValueError: when the latent carries more than one frame.  A ``T > 1``
            latent is a VIDEO: the frames share their positions
            (``comfy/ldm/krea2/model.py:299-303``) and do not form one
            rectangular canvas, so AReA's windowed attention — a latent-IMAGE
            construction — cannot be applied to it (plan Q13).
    """
    if not torch.is_tensor(latent):
        raise ValueError(
            "AReA: the Krea-2 latent must be a tensor, but got %s. This forward "
            "runs plain attention." % type(latent).__name__
        )
    if latent.dim() == 4:
        return latent
    if latent.dim() != 5:
        raise ValueError(
            f"AReA: a Krea-2 latent is 4-D [B, C, H, W] or 5-D [B, C, T, H, W] "
            f"(comfy/ldm/krea2/model.py:299-303), but this one is {latent.dim()}-D "
            f"with shape {tuple(latent.shape)}. This forward runs plain attention."
        )
    frames = int(latent.shape[2])
    if frames != 1:
        raise ValueError(
            f"AReA: this Krea-2 forward carries a {frames}-frame video latent "
            "[B, C, T, H, W], and AReA's windowed attention is a latent-IMAGE "
            "construction: the frames share their positions "
            "(comfy/ldm/krea2/model.py:299-303) and do not form one rectangular "
            "canvas. This forward runs plain attention — render single images "
            "for AReA."
        )
    return latent[:, :, 0]


def krea_latent_grid(latent: torch.Tensor, patch: int = KREA_PATCH) -> tuple[int, int]:
    """The ``(h, w)`` TOKEN grid of a Krea-2 latent.

    The model patchifies the latent with ``pad_to_patch_size(x, (patch, patch))``
    (``comfy/ldm/krea2/model.py:287``, ``comfy/ldm/common_dit.py:5-13``) and
    then reads ``h, w = x.shape[-2] // patch, x.shape[-1] // patch``
    (``model.py:288``) — so a latent that is not a whole number of patches is
    PADDED UP and the grid is the CEIL division, never the floor.  This helper
    is that arithmetic, and it agrees with the engine's own
    ``_grid_from_latent`` (``src/area.py:355-363``).

    The latent may be the squeezed 4-D one or a 5-D one with a single frame;
    anything else is refused (the same refusal as
    :func:`krea_squeeze_latent`).
    """
    if not torch.is_tensor(latent):
        raise ValueError(
            "AReA: the Krea-2 latent must be a tensor, but got %s. This forward "
            "runs plain attention." % type(latent).__name__
        )
    if latent.dim() == 5:
        latent = krea_squeeze_latent(latent)
    if latent.dim() != 4:
        raise ValueError(
            f"AReA: a Krea-2 latent is 4-D [B, C, H, W] or 5-D [B, C, T, H, W] "
            f"(comfy/ldm/krea2/model.py:299-303), but this one is {latent.dim()}-D "
            f"with shape {tuple(latent.shape)}. This forward runs plain attention."
        )
    patch = int(patch)
    if patch < 1:
        raise ValueError(
            f"AReA: the Krea-2 patch factor {patch} is not a valid patch size, "
            "so the token grid cannot be resolved. This forward runs plain "
            "attention."
        )
    height, width = int(latent.shape[-2]), int(latent.shape[-1])
    return -(-height // patch), -(-width // patch)


# ---------------------------------------------------------------------------
# The rope capture — the model's own table, once per forward
# ---------------------------------------------------------------------------


class KreaRopeRecorder:
    """Per-forward capture of the ONE rope table the model's own embedder built.

    Krea-2 builds its table ONCE per forward — ``self.pe_embedder(pos)`` over
    the whole joint stream (``comfy/ldm/krea2/model.py:364``) — and applies it
    itself BEFORE the seam (``model.py:93``), so the seam receives rotated q/k
    and NO table.  Phase I has to pool *unrotated* keys, which means the
    model's own table has to be inverted
    (:func:`src.area_rope.unrotate_table`).

    **The first table wins, and it is never overwritten.**  The coarse Phase I
    tables are built through the very SAME embedder (so a chained DyPE/SEGA
    embedder composes), which means the capture hook would see them too.  Two
    mechanisms keep that from polluting the forward:

    * :meth:`paused` suspends the capture around a coarse build — the pause
      guard;
    * a table that arrives anyway (an embedder call the wiring did not know
      about) is IGNORED: the first table stays, and the extra one is only
      counted (:attr:`extra_tables`, :attr:`ambiguous`).

    The recorder stores tensors only; it never rebinds a module symbol.  It is
    armed and reset per forward (the SPA discipline,
    ``src/spa.py:1385-1476``), so no table leaks across models or steps.
    """

    __slots__ = ("_extra", "_paused", "_table", "armed")

    def __init__(self) -> None:
        self._table: Optional[torch.Tensor] = None
        self._extra = 0
        self._paused = 0
        # The caller's arming window: the recorder only records while the
        # wiring has armed it, so an embedder call outside the forward of
        # interest (a refiner pass, a chained embedder's own call) is ignored.
        self.armed = False

    def reset(self) -> None:
        """Forget this forward's table — called at the start of each forward."""
        self._table = None
        self._extra = 0

    def record(self, table: torch.Tensor) -> None:
        """Capture one table, if this forward's table has not been seen yet.

        ``table`` is what the embedder's forward hook sees: an ``EmbedND``
        result ``[b, 1, n, axes, 2, 2]``.  It is normalised to the head-shared
        layout on the way in, so every accessor reads the sequence length off
        axis 2.

        The FIRST table of the forward is kept; any later one is counted and
        dropped — it never replaces the model's own table (the recorder is
        paused around the coarse builds, so a second table means the wiring
        missed a pause).
        """
        if not self.armed or self._paused:
            return
        if not torch.is_tensor(table) or table.dim() < 3:
            raise ValueError(
                "AReA: the Krea-2 rope recorder was handed %s, which is not an "
                "EmbedND table ([b, 1, n, axes, 2, 2]). A pe_embedder forward "
                "hook must record the embedder's output." % type(table).__name__
            )
        if self._table is None:
            self._table = head_shared(table)
        else:
            self._extra += 1

    @property
    def table(self) -> Optional[torch.Tensor]:
        """This forward's table — the model's OWN, or ``None`` if none was seen."""
        return self._table

    @property
    def extra_tables(self) -> int:
        """How many tables arrived AFTER the first one and were dropped."""
        return self._extra

    @property
    def ambiguous(self) -> bool:
        """Whether an extra table arrived (informational — it is NOT a refusal).

        The recorder never refuses and never overwrites: the model's own table
        is the first one and it stays.  This flag only says that the forward
        produced more than one table, which the wiring may log.
        """
        return self._extra > 0

    @contextmanager
    def paused(self):
        """Suspend the capture for the duration of the block.

        The coarse Phase I tables are built through the model's OWN embedder
        (``src/area.py``'s FLUX recipe, reused as-is for Krea-2's FLUX-shaped
        ids) — so without this guard the capture would record them as extra
        tables of the forward.  The pause is re-entrant and always unwinds.
        """
        self._paused += 1
        try:
            yield self
        finally:
            self._paused -= 1


#: The recorder of each captured embedder, keyed by ``id(pe_embedder)`` — so
#: :func:`coarse_ropes_krea` can find the recorder to pause without the
#: caller having to pass it.  One entry per INSTALLED model.
_RECORDERS: dict[int, KreaRopeRecorder] = {}


def capture_krea_table(pe_embedder) -> tuple:
    """The rope capture of one model: ``(hook, recorder)`` (plan §4).

    ``pe_embedder`` is the model's own embedder (``comfy/ldm/krea2/model.py:252``
    — a FLUX-layout ``EmbedND``); the returned ``hook`` is registered on it
    right here and carries the ``RemovableHandle`` it must be torn down by on
    ``hook.handle``, so the wiring (``src/area.py``) registers nothing and
    removes what it was given.

    The recorder only records while it is ARMED, so an embedder call outside
    the forward of interest (a refiner pass, a chained embedder's own call) is
    ignored.  A foreign output — anything that is not an ``EmbedND`` table — is
    reported once and dropped, so a forward keeps running plain attention
    instead of crashing the sampler.
    """
    recorder = KreaRopeRecorder()

    def hook(module, args, output):
        try:
            recorder.record(output)
        except ValueError as exc:
            _warn_once("krea-record", "AReA: %s", exc)

    hook.handle = None
    if hasattr(pe_embedder, "register_forward_hook"):
        hook.handle = pe_embedder.register_forward_hook(hook)
    _RECORDERS[id(pe_embedder)] = recorder
    return hook, recorder


# ---------------------------------------------------------------------------
# The joint layout — text first, then the image, no padding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KreaJointLayout:
    """The geometry of one Krea-2 joint stream.

    Krea-2 concatenates the caption and the image tokens with no padding at
    all (``combined = torch.cat((context, img), dim=1)``,
    ``comfy/ldm/krea2/model.py:355``), so the image tokens start EXACTLY where
    the caption ends — the engine's own text-first convention applies verbatim,
    with no adapter slicing and no pad policy.

    ``txt_len``  the caption's length (``context.shape[1]``, ``model.py:344``).
    ``h``, ``w``  the patch grid of the canvas (from the latent).
    ``seq_len``  the stream length the seam sees (``q.shape[2]``).
    """

    txt_len: int
    h: int
    w: int
    seq_len: int

    # --- the spans ----------------------------------------------------------

    @property
    def img_start(self) -> int:
        """``txt_len`` under its slicing name: the first image token."""
        return self.txt_len

    @property
    def img_end(self) -> int:
        """One past the last image token."""
        return self.txt_len + self.h * self.w

    @property
    def img_tokens(self) -> int:
        """``h * w`` — how many image tokens the canvas owns."""
        return self.h * self.w

    # --- the captures -------------------------------------------------------

    def text(self, x: torch.Tensor) -> torch.Tensor:
        """The caption tokens ``[0, txt_len)`` — never pooled (text is the text side)."""
        return x[:, :, : self.txt_len]

    def image(self, x: torch.Tensor) -> torch.Tensor:
        """The ``h*w`` image tokens ``[txt_len, txt_len + h*w)``."""
        return x[:, :, self.img_start : self.img_end]


def krea_layout(txt_len: int, h: int, w: int, stream_len: int) -> KreaJointLayout:
    """Resolve — and check — the geometry of one Krea-2 joint stream.

    Raises:
        ValueError: when the stream does not fit the joint geometry — the
            caption is empty, the grid is empty, or the stream is not
            ``txt_len + h*w`` long.  The message names the offender and the
            remedy (house rule, ``src/hap.py:876-895``), so a foreign stream is
            refused instead of silently pooling the wrong tokens.
    """
    txt_len, h, w, seq_len = int(txt_len), int(h), int(w), int(stream_len)
    if min(txt_len, h, w) < 1:
        raise ValueError(
            f"AReA: a Krea-2 joint stream needs at least one caption token and "
            f"a non-empty patch grid, but got txt_len={txt_len} on a {h}x{w} "
            "grid. This forward runs plain attention."
        )
    expected = txt_len + h * w
    if seq_len != expected:
        raise ValueError(
            f"AReA: the joint stream is {seq_len} tokens, but a {h}x{w} patch "
            f"grid behind a {txt_len}-token caption needs {expected} "
            "(comfy/ldm/krea2/model.py:355). The latent and the prompt do not "
            "belong to one forward, so the image tokens cannot be located. "
            "This forward runs plain attention."
        )
    return KreaJointLayout(txt_len, h, w, seq_len)


# ---------------------------------------------------------------------------
# The coarse tables — FLUX's recipe, through the model's OWN embedder
# ---------------------------------------------------------------------------


def coarse_ropes_krea(
    embedder,
    layout: KreaJointLayout,
    f: int,
    device,
    recorder: Optional[KreaRopeRecorder] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two Phase I rope tables — FLUX's recipe, Krea-2's ids.

    Queries keep full resolution and ride on their block's coarse position;
    the pooled keys sit on the pooled grid (``ceil(h/f) x ceil(w/f)`` — the
    ceil only differs from ``(h/f) x (w/f)`` on a grid the blocks do not tile,
    where the last block on a partial axis is a partial block), so both are
    measured in the same units.  This is FLUX's own recipe
    (``_area_coarse_ropes``, ``src/area.py:482-512``) reused as-is, because
    Krea-2's rope ids ARE FLUX-shaped: the caption sits at the origin and every
    image token carries ``(0, row, col)`` with axis 0 the FRAME index
    (``comfy/ldm/krea2/model.py:291-294``, ``:346``, ``:361``).

    The block coordinates are FLUX's **floor division** (``row // f``), not
    Z-Image's continuous refinement (plan Q9): Z-Image's refinement exists
    because its theta of 256 makes every band fast, while Krea-2's theta of
    1000 keeps the low bands slow — the FLUX recipe is the reference-faithful
    choice there.

    The tables come from the model's OWN embedder, so a chained DyPE/SEGA
    embedder composes — and the capture is PAUSED around the two calls, so the
    coarse tables are not mistaken for extra tables of the forward (the pause
    guard; see :meth:`KreaRopeRecorder.paused`).

    Args:
        embedder: the model's own ``pe_embedder``.
        layout: the joint layout of this forward.
        f: the pooling factor of this step.
        device: the device the ids are built on.
        recorder: the rope recorder to pause.  Optional — the recorder that
            :func:`capture_krea_table` registered for this embedder is found
            automatically — but passing it explicitly is clearer.
    """
    if embedder is None:
        raise ValueError(
            "AReA: this model has no diffusion_model.pe_embedder, so the "
            "Phase I coarse rope tables cannot be built. Use an unmodified "
            "Krea-2 loader."
        )
    f = int(f)
    if f < 1:
        raise ValueError(
            f"AReA: the pooling factor {f} is not a valid pooling block size, "
            "so the coarse rope grid cannot be built. This forward runs plain "
            "attention."
        )
    if recorder is None:
        recorder = _RECORDERS.get(id(embedder))
    nh, nw = coarse_grid(layout.h, layout.w, f)
    pos = torch.arange(layout.h * layout.w, device=device)
    coarse = torch.arange(nh * nw, device=device)
    # FLUX's floor-divided block coordinates (plan Q9): the query of a token
    # rides on its block's integer coordinate, the pooled key sits on its
    # block's own coordinate.
    q_ids = flux_style_ids(
        layout.txt_len, pos // layout.w // f, pos % layout.w // f, device, torch.float32
    )
    kv_ids = flux_style_ids(
        layout.txt_len, coarse // nw, coarse % nw, device, torch.float32
    )
    if recorder is not None:
        # The pause guard: the coarse tables go through the model's OWN
        # embedder, and the capture hook would otherwise record them as extra
        # tables of this forward.
        with recorder.paused():
            rope_q = embedder(q_ids)
            rope_kv = embedder(kv_ids)
    else:
        rope_q = embedder(q_ids)
        rope_kv = embedder(kv_ids)
    # Head-shared either way, so a chained embedder that returns another
    # layout is handled too.
    return head_shared(rope_q), head_shared(rope_kv)


# ---------------------------------------------------------------------------
# The eligibility guard — the main blocks only
# ---------------------------------------------------------------------------


def krea_stream_eligible(heads, dm_heads, stream_len, txt_len, h, w, img_slice) -> bool:
    """Is this seam call one of Krea-2's MAIN blocks? (plan Q2)

    Three conditions, all necessary:

    * ``heads == dm_heads`` — the four text-fusion calls run with 20 heads over
      48-head blocks (``comfy/ldm/krea2/model.py:237``, ``:235``), and the GQA
      repeat is resolved BEFORE the seam (``model.py:94-97``), so the head count
      at the seam is exact.  A length check alone cannot do this: a refiner
      call's stream length is the PROMPT length, a PREFIX of the joint stream's.
    * ``img_slice == [txt_len, stream_len]`` — the model publishes
      ``img_slice`` only for the main blocks (``model.py:369``), AFTER the text
      fusion, so the four auxiliary calls carry none.
    * ``stream_len == txt_len + h * w`` — the stream is the caption plus the
      canvas, with no padding (``model.py:355``).  A foreign stream is
      refused here.

    Returns ``False`` — never raises — so a call that fails any of the three
    is simply DELEGATED UNCHANGED by the seam.
    """
    try:
        if int(heads) != int(dm_heads):
            return False
        if img_slice is None:
            return False
        span = list(img_slice)
        if len(span) != 2:
            return False
        if int(span[0]) != int(txt_len) or int(span[1]) != int(stream_len):
            return False
        if int(stream_len) != int(txt_len) + int(h) * int(w):
            return False
    except (TypeError, ValueError):
        return False
    return True
