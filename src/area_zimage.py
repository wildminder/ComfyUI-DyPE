"""Z-Image adapter tensors — ropes, capture, pad policy (plan 2026-10-16, §4/§7-t3).

The engine (``src/area_attn.py``, ``src/area_schedule.py``) is backbone-agnostic
and imports no comfy module; everything Z-Image-specific about *tensors* lives
here, exactly as FLUX's lives in ``src/area.py:264-316``.  Three parts, one per
difference between the two backbones:

**RoPE — the seam's q/k arrive ROTATED.**  ``JointAttention.forward`` applies
rope itself (``comfy/ldm/lumina/model.py:159-169`` through
``comfy.quant_ops.ck.rms_rope``, or the ``apply_rope`` fallback at ``:173``) and
then calls the module-level symbol ``optimized_attention_masked`` (bound at
``model.py:14``, called at ``:182``) with NO rope argument, so the seam sees
rotated q/k and no table.  Phase I must pool *unrotated* keys
(``src/area_attn.py:180-184``: pooling rotated keys would mix keys rotated by
different angles inside one block), so the model's own table has to be inverted
— :func:`unrotate_table` / :func:`unrotate1` — and the coarse tables rebuilt
(:func:`coarse_ropes_zimage`).  lumina's ids are ``[b, N, 3]`` = ``(t, row,
col)`` (``pos_ids_x``, ``model.py:421-437``) and its caption tokens carry real
t-axis values (``cap_pos_ids[:, :, 0] = arange(len) + 1``, ``model.py:664-667``),
so FLUX's all-zero text ids (``src/area.py:264-277``) are NOT copied.
The three helpers that undo the rotation — :func:`unrotate_table`,
:func:`unrotate1` and :func:`_head_shared` — are shared with the Krea-2
adapter and live in ``src/area_rope.py``; the names here are re-exports of the
very same objects (plan 2026-10-20, §5.1/S3).

**Capture — the model's own tables, in call order.**  lumina calls
``rope_embedder`` once per token GROUP (caption, then image — ``embed_all``,
``model.py:663-668`` and ``:725-727``) while a single ``JointAttention`` runs
over the whole concatenated sequence.  SPA learned this the hard way
(``tests/test_spa_hook_krea2_zimage.py:20-32``): one group's table against a
full-sequence attention stream is a crash.  :class:`ZImageRopeRecorder` keeps
every group of a forward in call order, so the seam gets the table the model
*itself* applied — rope options and a chained positional-scaling embedder
included — instead of a reconstruction that could silently disagree.

**Pad policy — the stream is padded.**  ``pad_zimage``
(``model.py:418-420``, applied at ``:665`` for the caption and ``:725-727`` for
the image) pads both to a multiple of 32, so the stream the seam sees is
``[caption block][image h*w][image pad]`` and the image starts at the *padded*
caption length, not at the prompt's.  :class:`ZImageJointLayout` is that
geometry, validated against the pad multiple so a Ming-Image-style prompt mask
or a foreign stream is refused instead of pooling the wrong tokens.  The engine
is handed exactly ``h*w`` image tokens (plan Q6); the pad key/value tokens pass
through Phase I unchanged, and Phase II stitches its answer over the whole
padded caption block (the prompt tokens plus the model's learned caption pad,
``cap_pad_token``, ``comfy/ldm/lumina/model.py:620-623``) while only the image
pad slots stay zero (they are discarded at unpatchify).

No comfy import and no module rebinding here: this module owns tensors only, the
seam install is t4 (``src/area.py``), and the arch profile is t2
(``src/area_arch.py``).

**Names, not rebinding.**  The seam's own identity — the module that binds it and
the symbol the blocks call — is only *named* here
(:data:`ZIMAGE_SEAM_MODULE` / :data:`ZIMAGE_SEAM_SYMBOL`, plan §4), as is the
advisory Phase I threshold (:data:`ZIMAGE_SIGMA_STAR` /
:func:`sigma_prefix_count`) and the rope capture the wiring registers on the
model's own embedder (:func:`capture_tables`), so plan §4's whole interface is
one import for the installer.  Nothing here binds or patches a comfy symbol.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from .area_rope import head_shared as _head_shared
from .area_rope import unrotate1, unrotate_table
from .area_schedule import coarse_grid

logger = logging.getLogger("ComfyUI-DyPE")

# One-time warnings (module scoped, like src/area.py's).
_warned: set = set()


def _warn_once(key: str, message: str, *args) -> None:
    """Emit ``message`` at WARNING level the first time ``key`` is seen."""
    if key in _warned:
        return
    _warned.add(key)
    logger.warning(message, *args)


__all__ = [
    "ZIMAGE_AXES",
    "ZIMAGE_GROUP_ORDER",
    "ZIMAGE_PAD_MULTIPLE",
    "ZIMAGE_SEAM_MODULE",
    "ZIMAGE_SEAM_SYMBOL",
    "ZIMAGE_SIGMA_STAR",
    "ZIMAGE_TRAINED_TOKENS",
    "ZImageJointLayout",
    "ZImageRopeRecorder",
    "capture_tables",
    "coarse_ropes_zimage",
    "sigma_prefix_count",
    "stitch_fine_zeros",
    "unrotate1",
    "unrotate_table",
    "zimage_layout",
]

#: The Z-Image seam: the module that binds the symbol (``from ... import
#: optimized_attention_masked``, ``comfy/ldm/lumina/model.py:14``) and the symbol
#: itself, called at ``:182``.  Naming them here (plan §4) keeps the adapter's
#: interface in one place; ``src/area.py`` rebinds the *bound* name, because
#: patching the defining module would be a silent no-op.
ZIMAGE_SEAM_MODULE = "comfy.ldm.lumina.model"
ZIMAGE_SEAM_SYMBOL = "optimized_attention_masked"

#: Phase I/II switch of the reference (theory.md:240-246): the leading steps
#: whose noise level is still at or above ``sigma_star`` run Phase I.  The
#: paper's value for its first backbone — advisory for the distilled Z-Image
#: schedule (theory.md:494: 0.89 for FLUX, 0.86 for Qwen; configuration
#: specific), which is why the step count stays the default mode.
ZIMAGE_SIGMA_STAR = 0.89

#: lumina rope ids carry three axes — ``(t, row, col)`` (``pos_ids_x``,
#: ``comfy/ldm/lumina/model.py:421-437``); FLUX's are ``(zero, row, col)``.
ZIMAGE_AXES = 3

#: The model's TRAINED extent, in tokens — 1024 px over the 16 px patch grid
#: (``src/area.py``'s ``NATIVE_PATCHES``, the same 64).  Every rope distance
#: the model ever saw is inside it, so a canvas wider than this is compressed
#: back into it: the GTCN's canvas extent ratio (area-fix3,
#: ``.dev/area-debug/area-fix3.txt``).
ZIMAGE_TRAINED_TOKENS = 64

#: Z-Image pads the caption AND the image token count to a multiple of 32
#: (``pad_zimage``, ``comfy/ldm/lumina/model.py:418-420``).  The arch profile
#: carries it as ``seq_pad_multiple`` (``src/area_arch.py:112-114``).
ZIMAGE_PAD_MULTIPLE = 32

#: A plain Z-Image forward's rope groups, in the model's own call order
#: (``embed_cap`` then ``embed_all``, comfy/ldm/lumina/model.py:663-668,
#: :725-727) — the ``forward_order`` :meth:`ZImageRopeRecorder.reads` expects.
ZIMAGE_GROUP_ORDER = ("caption", "image")


def sigma_prefix_count(sigmas, sigma_star: float) -> int:
    """Number of LEADING steps of ``sigmas`` still at or above ``sigma_star``.

    ``theory.md:240-246`` — ``Sc = count_prefix(sigmas, lambda s: s >=
    sigma_star)`` splits a run into a Phase I prefix (global composition) and a
    Phase II remainder (local refinement).  ``sigmas`` is anything whose items
    coerce to ``float``: the ``transformer_options["sample_sigmas"]`` table is a
    tensor (``comfy/samplers.py:1229``) and reads *descending*, so the leading run
    is the noisy prefix.

    The count is capped by the table's length, and an empty table yields 0 (the
    caller clamps Phase I to the step count anyway).
    """
    threshold = float(sigma_star)
    count = 0
    for sigma in sigmas:
        if float(sigma) < threshold:
            break
        count += 1
    return count


# ---------------------------------------------------------------------------
# Un-rotating the seam's tensors
# ---------------------------------------------------------------------------
#
# ``unrotate_table``, ``unrotate1`` and ``_head_shared`` live in
# ``src/area_rope.py`` — they are backbone-agnostic (Krea-2's seam is rotated
# the same way, ``comfy/ldm/krea2/model.py:93``), so they are shared rather than
# duplicated (plan 2026-10-20, §5.1/S3).  The names below are RE-EXPORTS of the
# very same objects, so ``src/area.py`` and the existing tests keep importing
# them from here unchanged.


# ---------------------------------------------------------------------------
# Capture — the model's own rope tables, in call order
# ---------------------------------------------------------------------------


class ZImageRopeRecorder:
    """Per-forward capture of the tables the model's own ``rope_embedder`` built.

    lumina calls the embedder once per token GROUP and concatenates the groups
    group-major — caption first, then image (``embed_all``,
    ``comfy/ldm/lumina/model.py:663-668`` and ``:725-727``) — and a single
    ``JointAttention`` then runs over the whole concatenated sequence
    (``model.py:831``).  The recorder keeps every group of the forward in call
    order and exposes:

    * :attr:`cap_len` — the sequence length of group #1, the *padded* caption
      block, which is also the joint-stream index where the real image tokens
      start (plan Q6);
    * :attr:`image_table` — group #2, the image group's own table;
    * :meth:`full_table` — every group concatenated in call order: the table the
      model itself applied to the joint stream, so :func:`unrotate1` undoes the
      seam's rotation exactly, rope options and a chained positional-scaling
      embedder included.

    State is per forward (the SPA discipline, ``src/spa.py:1385-1476``): the
    per-forward hook calls :meth:`reset` before the model runs, so no group may
    leak across models or steps, and the caller decides WHEN recording is armed
    (:attr:`armed`, set by :func:`capture_tables`' hook) — the refiner groups of
    a plain Z-Image forward are excluded by arming only between the layout and
    the layer loop (``model.py:880-881``, t4).  The recorder stores tensors only;
    it never rebinds a module symbol.
    """

    __slots__ = ("_groups", "armed")

    def __init__(self) -> None:
        self._groups: list[torch.Tensor] = []
        # The caller's arming window (plan Q2 / fact 12): a plain Z-Image
        # forward calls ``rope_embedder`` outside the joint groups too — the
        # refiners run their own passes — so a capture stays OFF until the wiring
        # arms it and is disarmed again once the joint stream is identified.
        # The recorder never arms itself; :func:`capture_tables` only feeds it
        # while ``armed`` is true.
        self.armed = False

    def reset(self) -> None:
        """Forget every group — called at the start of each recorded forward."""
        self._groups = []

    def record(self, table: torch.Tensor) -> None:
        """Append one rope group's table, in call order.

        ``table`` is what a ``rope_embedder`` forward hook sees: an ``EmbedND``
        result ``[b, 1, n, axes, 2, 2]`` (or lumina's own ``[b, n, 1, ...]`` view
        of it).  It is normalised to the head-shared layout on the way in, so
        every accessor below can read the sequence length off axis 2.
        """
        if not torch.is_tensor(table) or table.dim() < 3:
            raise ValueError(
                "AReA: the Z-Image rope recorder was handed %s, which is not an "
                "EmbedND table ([b, 1, n, axes, 2, 2]). A rope_embedder forward "
                "hook must record the embedder's output." % type(table).__name__
            )
        self._groups.append(_head_shared(table))

    @property
    def groups(self) -> tuple[torch.Tensor, ...]:
        """Every recorded group, in call order."""
        return tuple(self._groups)

    def _group(self, index: int, what: str) -> torch.Tensor:
        """Group ``index``, or an actionable error naming the expected call order."""
        try:
            return self._groups[index]
        except IndexError as exc:
            raise ValueError(
                f"AReA: the Z-Image rope recorder holds {len(self._groups)} "
                f"group(s), so the {what} group (#{index + 1}) is missing. A "
                "plain Z-Image forward calls rope_embedder twice — caption, "
                "then image (comfy/ldm/lumina/model.py:663-668,725-727); an "
                "armed recorder must see both. This forward runs plain "
                "attention."
            ) from exc

    @property
    def cap_table(self) -> torch.Tensor:
        """Group #1 — the caption block's table ``[b, 1, cap_len, axes, 2, 2]``."""
        return self._group(0, "caption")

    @property
    def image_table(self) -> torch.Tensor:
        """Group #2 — the image group's table ``[b, 1, h*w + pad, axes, 2, 2]``."""
        return self._group(1, "image")

    @property
    def cap_len(self) -> int:
        """The padded caption length — where the real image tokens start."""
        return int(self.cap_table.shape[2])

    def full_table(self) -> torch.Tensor:
        """Every group concatenated in call order — the model's own full table.

        ``[b, 1, seq_len, axes, 2, 2]`` in the head-shared layout the embedder
        returns; lumina itself views it as ``[b, seq_len, 1, axes, 2, 2]`` after
        ``movedim(1, 2)`` (``comfy/ldm/lumina/model.py:666``), which
        :func:`unrotate1` accepts as well.
        """
        groups = self.groups
        if not groups:
            raise ValueError(
                "AReA: the Z-Image rope recorder holds no group, so the model's "
                "own rope table is unavailable. A plain Z-Image forward calls "
                "rope_embedder twice (comfy/ldm/lumina/model.py:663-668,"
                "725-727). This forward runs plain attention."
            )
        if len(groups) == 1:
            return groups[0]
        return torch.cat(groups, dim=2)

    def reads(self, forward_order=ZIMAGE_GROUP_ORDER):
        """``(cap_len, img_len, full_table, group_tables)`` of this forward.

        ``forward_order`` names the groups the model made, in call order — a
        plain Z-Image forward's caption block first, then its image
        (:attr:`ZIMAGE_GROUP_ORDER`, ``embed_cap`` / ``embed_all``,
        comfy/ldm/lumina/model.py:663-668,725-727).  The recorder holds the
        tables in exactly that order, so the read is a direct map: ``cap_len``
        is the caption block's length (where the real image tokens start),
        ``img_len`` the image group's own length (pad rows included — the block
        the model built), ``full_table`` the concatenation the model itself
        applied to the joint stream and ``group_tables`` every group in call
        order.

        A forward that made a different number of groups — a refiner pass, or a
        rope call the arming window kept out — has no caption/image split to
        offer, so the read raises the actionable error instead of handing back a
        wrong table (the same discipline as :meth:`_group`).
        """
        expected = len(forward_order)
        if len(self._groups) != expected:
            raise ValueError(
                f"AReA: the Z-Image rope recorder holds {len(self._groups)} "
                f"group(s), but {expected} ({', '.join(forward_order)}) is the "
                "model's own call order for a plain Z-Image forward "
                "(comfy/ldm/lumina/model.py:663-668,725-727), so the joint "
                "stream cannot be located. This forward runs plain attention."
            )
        return self.cap_len, int(self.image_table.shape[2]), self.full_table(), self.groups


def capture_tables(rope_embedder):
    """The rope capture of one model: ``(hook, recorder)`` (plan §4).

    ``rope_embedder`` is the model's own embedder (``comfy/ldm/lumina/model.py:634``);
    the returned ``hook`` is registered on it right here and carries the
    ``RemovableHandle`` it must be torn down by on ``hook.handle``, so the wiring
    (``src/area.py``) registers nothing and removes what it was given.  ``recorder``
    is the :class:`ZImageRopeRecorder` the hook feeds.

    Recording is gated by :attr:`ZImageRopeRecorder.armed`: the model calls
    ``rope_embedder`` outside the joint groups too (the refiners run their own
    passes, ``model.py:497-531``), so the caller arms the recorder once a
    forward's geometry is resolved and disarms it again when the joint stream is
    identified (plan Q2 / fact 12).  A foreign output — anything that is not an
    ``EmbedND`` table — is reported once and the group is dropped, so a forward
    keeps running plain attention instead of crashing the sampler.
    """
    recorder = ZImageRopeRecorder()

    def hook(module, args, output):
        if not recorder.armed:
            return
        try:
            recorder.record(output)
        except ValueError as exc:
            _warn_once("zimage-record", "AReA: %s", exc)

    hook.handle = None
    if hasattr(rope_embedder, "register_forward_hook"):
        hook.handle = rope_embedder.register_forward_hook(hook)
    return hook, recorder


# ---------------------------------------------------------------------------
# Pad policy — where the real tokens sit in the padded joint stream
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZImageJointLayout:
    """The pad geometry of one padded Z-Image joint stream.

    ``patchify_and_embed`` returns the stream as ``padded_full_embed`` —
    ``[caption block][image h*w][image pad]`` — and publishes only
    ``l_effective_cap_len = total - img_len`` (``comfy/ldm/lumina/model.py:831``),
    which is the *padded* caption length, not the prompt's.  The adapter needs
    the whole picture, resolved once per forward:

    ``txt_len``  the prompt's own caption length (``c_crossattn``).
    ``cap_len``  the caption block the model concatenated — where the image starts.
    ``h``, ``w`` the patch grid of the canvas.
    ``seq_len``  the stream length the seam sees (``q.shape[2]``).

    :func:`zimage_layout` checks those against the pad multiple, and the
    slicing methods below are the policy itself: the engine is handed exactly
    ``h*w`` image tokens (:meth:`image`, plan Q6) plus the never-pooled
    non-image tokens (:meth:`pass_through`), so the pad key/value tokens ride
    through Phase I unchanged and :func:`stitch_fine_zeros` writes the Phase II
    answer over the whole padded caption block, leaving only the image pad slots
    zero.
    """

    txt_len: int
    cap_len: int
    h: int
    w: int
    seq_len: int

    # --- the spans ----------------------------------------------------------

    @property
    def img_start(self) -> int:
        """``cap_len`` under its slicing name: the first real image token."""
        return self.cap_len

    @property
    def img_end(self) -> int:
        """One past the last real image token — where the image pad starts."""
        return self.cap_len + self.h * self.w

    @property
    def cap_pad(self) -> int:
        """Caption pad tokens (``cap_len - txt_len``)."""
        return self.cap_len - self.txt_len

    @property
    def img_pad(self) -> int:
        """Image pad tokens (``seq_len - img_end``)."""
        return self.seq_len - self.img_end

    @property
    def pass_through_len(self) -> int:
        """Tokens that are never pooled: the caption block plus the image pad."""
        return self.seq_len - self.h * self.w

    # --- the captures -------------------------------------------------------
    # Every method slices axis 2 only, so it works for joint tensors
    # (``[b, heads, N, d]``) and rope tables (``[b, 1, N, axes, 2, 2]``) alike.

    def text(self, x: torch.Tensor) -> torch.Tensor:
        """The prompt's own caption tokens ``[0, txt_len)``."""
        return x[:, :, : self.txt_len]

    def image(self, x: torch.Tensor) -> torch.Tensor:
        """The ``h*w`` real image tokens ``[cap_len, cap_len + h*w)`` — never a pad token."""
        return x[:, :, self.img_start : self.img_end]

    def pass_through(self, x: torch.Tensor) -> torch.Tensor:
        """Every non-image token, in stream order: the caption block, then the image pad.

        Phase I hands these to the engine as its un-pooled text side — text is
        never pooled (``src/area_attn.py:175-190``) — so the pad key/value
        tokens pass through unchanged, the phase's pad policy (plan Q6).
        """
        return torch.cat((x[:, :, : self.cap_len], x[:, :, self.img_end :]), dim=2)

    def engine_stream(self, x: torch.Tensor) -> torch.Tensor:
        """One joint tensor in the engine's Phase I key order: pass-through side, then image."""
        return torch.cat((self.pass_through(x), self.image(x)), dim=2)


def zimage_layout(
    txt_len: int,
    cap_len: int,
    h: int,
    w: int,
    seq_len: int,
    multiple: int = ZIMAGE_PAD_MULTIPLE,
) -> ZImageJointLayout:
    """Resolve — and check — the pad geometry of one joint stream.

    Args:
        txt_len: the prompt's own caption length.
        cap_len: the caption block the model concatenated — the rope recorder's
            :attr:`cap_len <ZImageRopeRecorder.cap_len>` (group #1's length).
        h, w: the patch grid of the canvas (``_ArchProfile.patch_count``).
        seq_len: the joint stream length the seam sees.
        multiple: the architecture's sequence pad multiple — 32 for Z-Image
            (``src/area_arch.py``).  A value of 0 or 1 means "nothing is padded"
            (FLUX), which degenerates to the identity geometry.

    Raises:
        ValueError: when the stream does not fit the pad policy — the caption
            block is not ``ceil(txt_len / multiple) * multiple``, or the stream
            is not that block plus the padded image.  Both messages name the
            offender and the remedy (house rule, ``src/hap.py:876-895``), so a
            Ming-Image-style prompt mask or a non-Z-Image stream is refused
            instead of silently pooling the wrong tokens.
    """
    txt_len, cap_len, seq_len = int(txt_len), int(cap_len), int(seq_len)
    if min(txt_len, h, w) < 1:
        raise ValueError(
            f"AReA: a Z-Image joint stream needs at least one caption token and "
            f"a non-empty patch grid, but got txt_len={txt_len} on a {h}x{w} "
            "grid. This forward runs plain attention."
        )
    if multiple > 1:
        expected_cap = -(-txt_len // multiple) * multiple
        if cap_len != expected_cap:
            raise ValueError(
                f"AReA: the recorded caption block is {cap_len} tokens, but "
                f"{txt_len} prompt tokens pad to a multiple of {multiple} into "
                f"{expected_cap} (pad_zimage, comfy/ldm/lumina/model.py:418-420). "
                "The prompt was padded by something other than Z-Image's own "
                "32-token rule (Ming-Image masked padding, comfy/ldm/lumina/"
                "model.py:684-686), so the image tokens no longer start where "
                "the caption ends. This forward runs plain attention."
            )
        expected_seq = expected_cap + -(-(h * w) // multiple) * multiple
    else:
        expected_seq = cap_len + h * w
    if seq_len != expected_seq:
        raise ValueError(
            f"AReA: the joint stream is {seq_len} tokens, but a {h}x{w} patch "
            f"grid behind a {cap_len}-token caption block needs {expected_seq} "
            "(comfy/ldm/lumina/model.py:825-832). The latent and the prompt do "
            "not belong to one forward, so the image tokens cannot be located. "
            "This forward runs plain attention."
        )
    return ZImageJointLayout(txt_len, cap_len, h, w, seq_len)


def _axis_scales(rope_options: Optional[dict]) -> tuple[float, float, float, float]:
    """``(h_scale, w_scale, h_start, w_start)`` from ``transformer_options``.

    ``pos_ids_x`` (``comfy/ldm/lumina/model.py:421-437``) reads exactly these
    four keys of ``rope_options`` and applies them to the image axes only (the
    t axis is never scaled), so the rebuilt coarse tables match the model's own
    table when another node has installed a positional scaling.
    """
    if not rope_options:
        return 1.0, 1.0, 0.0, 0.0
    return (
        float(rope_options.get("scale_y", 1.0)),
        float(rope_options.get("scale_x", 1.0)),
        float(rope_options.get("shift_y", 0.0)),
        float(rope_options.get("shift_x", 0.0)),
    )


def _ids(t: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor, device,
         dtype=torch.float32) -> torch.Tensor:
    """``[1, n, 3]`` position ids ``(t, row, col)`` from per-token axis values."""
    ids = torch.zeros(1, int(t.numel()), ZIMAGE_AXES, device=device, dtype=dtype)
    ids[0, :, 0] = t.to(device=device, dtype=dtype)
    ids[0, :, 1] = rows.to(device=device, dtype=dtype)
    ids[0, :, 2] = cols.to(device=device, dtype=dtype)
    return ids


def _zeros(n: int, device, dtype=torch.float32) -> torch.Tensor:
    """``[1, n, 3]`` zero ids — the model's own image-pad positions.

    ``embed_all`` pads the image position ids with ``F.pad``
    (``comfy/ldm/lumina/model.py:727``), so an image-pad token rides position
    (0, 0, 0) and its rotation is the identity: the pad tokens' rope is a
    no-op, which is what "the pad k/v pass through unchanged" means in tensors.
    """
    return torch.zeros(1, int(n), ZIMAGE_AXES, device=device, dtype=dtype)


def _block_offsets(n_blocks: int, size: int, f: int, uniform: float, device) -> torch.Tensor:
    """The per-block key offset on one axis — the block's centroid inside its block.

    A full ``f``-cell block's centroid sits at ``(f - 1) / (2f)`` (``uniform``),
    the mean of its members' positions.  The LAST block of a partial axis owns
    fewer than ``f`` cells (the grid does not tile), and its centroid is the
    mean of *its* members, ``(members - 1) / (2f)`` — otherwise the pooled key
    would sit a quarter of a block away from the tokens it stands for.  Every
    other block keeps the full-block offset, so a grid that tiles exactly is
    unchanged, bit for bit.
    """
    off = torch.full((int(n_blocks),), float(uniform), dtype=torch.float32, device=device)
    if int(size) % int(f):
        members = int(size) - (int(n_blocks) - 1) * int(f)
        off[-1] = (members - 1) / (2.0 * int(f))
    return off


def coarse_ropes_zimage(
    embedder,
    layout: ZImageJointLayout,
    f: int,
    device,
    rope_options: Optional[dict] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The two Phase I rope tables — the reference's ``_coarse_ropes`` for lumina.

    Queries keep full resolution while the pooled keys sit on the pooled grid
    (``ceil(h/f) x ceil(w/f)`` — the ceil only differs from ``(h/f) x (w/f)``
    on a grid the blocks do not tile, where the last block is partial), so both
    are measured in the same units — CONTINUOUSLY
    (AReA-Z mechanism 1, ``.dev/area-debug/area-fix.txt`` Flaw 2): the reference
    floor-divided the query's position into its block's integer coordinate, so
    every ``f x f`` block shared one rope position and the phase jumped by a
    whole unit at every block boundary — at Z-Image's low rope theta (256, vs
    FLUX's 10 000) an ``f = 2`` boundary jump is about 1 rad (~57 degrees), a
    32 px square wave injected into the velocity field.  Here each query keeps
    its own position scaled by ``1/f`` and the pooled key sits at its block's
    geometric centroid, ``R + (f - 1) / (2f)`` — the mean of its block's query
    positions — so the phase jump across a boundary equals the one inside the
    block and the staircase is gone; ``f == 1`` degenerates exactly (offset 0,
    positions unchanged).  The tables come from the model's own
    ``rope_embedder`` (so a chained positional-scaling embedder composes,
    ``src/area.py:279-316``) but from lumina's ids: the caption block rides the
    t axis (``arange(cap_len) + 1``, ``model.py:664-667`` — FLUX's zero text ids
    are NOT copied) and every image token of the frame carries one t value,
    ``cap_len + 1``, with its row/col on the other two axes (``pos_ids_x``,
    ``model.py:421-437``).

    **GTCN — the canvas extent ratio (area-fix3, ``.dev/area-debug/area-fix3.txt``).**
    Lumina2 puts EVERY caption token at the spatial origin (``(t, 0, 0)``,
    ``model.py:664-667``), so the rope distance between an image token and the
    prompt is the token's own distance from the origin — and at Z-Image's theta
    (256, one rad per token) a distance beyond the trained 64 tokens wraps the
    rotation several times and the text cross-attention collapses into noise.
    The image coordinates are therefore compressed by the canvas extent ratio
    ``s = max(1, h / 64)`` per axis, so the canvas always fits the trained
    domain; the CAPTION stays at the origin, which is what bounds the distance
    to it, and the t axis is never touched.

    The block units this table already measures in ARE a compression by the
    pooling factor ``f``, and the factor rule (Eq. 5) keeps ``f >= h / 64`` for
    the first ramp rungs — so the units already hold the canvas inside the
    trained domain there.  The scaling is applied only where it is NEEDED: the
    compression factor is ``max(f, s)`` per axis and the coordinates are scaled
    by ``f / max(f, s)`` — exactly ``1.0`` (a no-op, bit for bit) when the block
    units already fit, and the spec's ``r / s`` when they do not (the ramp's
    later rungs, whose factor has fallen below the canvas ratio — exactly the
    rungs that used to leave the canvas out of the trained domain).

    ``rope_q`` covers the whole padded stream in STREAM order — caption block,
    image at continuous coarse positions, image pad on the model's own zero ids
    — and ``rope_kv`` the engine's Phase I key order: the caption block (never
    pooled), the image pad (unchanged), then the coarse grid.  That is why the
    two tables differ while the caption half is shared.
    """
    if embedder is None:
        raise ValueError(
            "AReA: this model has no diffusion_model.rope_embedder, so the "
            "Phase I coarse rope tables cannot be built. Use an unmodified "
            "Z-Image (Lumina2) loader."
        )
    f = int(f)
    if f < 1:
        # The only real precondition: the block size must be a positive
        # integer.  The grid does NOT have to tile — a patch grid with no
        # common divisor >= 2 is pooled with a *partial* edge block
        # (``src/area_schedule.py``'s resolver), and the pooled grid is then the
        # ceil-based one.
        raise ValueError(
            f"AReA: the pooling factor {f} is not a valid pooling block size, "
            "so the coarse rope grid cannot be built. This forward runs plain "
            "attention."
        )
    nh, nw = coarse_grid(layout.h, layout.w, f)
    ff = float(f)
    h_scale, w_scale, h_start, w_start = _axis_scales(rope_options)

    # GTCN — the canvas extent ratio, per axis: the trained extent is
    # ``ZIMAGE_TRAINED_TOKENS`` tokens, so a canvas of ``h`` tokens is
    # compressed by ``h / 64`` to stay inside it.  The coordinates below are
    # already block units (a compression by the pooling factor ``f``), so the
    # factor that actually compresses the canvas is ``max(f, s)`` and the
    # multiplier applied to the block units is ``f / max(f, s)`` — ``1.0``
    # (bit-for-bit no-op) wherever the block units already hold the canvas
    # inside the trained domain, and the spec's ``r / s`` where they do not.
    s_y = max(1.0, float(layout.h) / float(ZIMAGE_TRAINED_TOKENS))
    s_x = max(1.0, float(layout.w) / float(ZIMAGE_TRAINED_TOKENS))
    norm_y = ff / max(ff, s_y)
    norm_x = ff / max(ff, s_x)

    # The caption block: t = 1..cap_len, row/col zero (model.py:664-667).
    cap_len = layout.cap_len
    cap_ids = _ids(
        torch.arange(1, cap_len + 1, device=device),
        torch.zeros(cap_len, device=device),
        torch.zeros(cap_len, device=device),
        device,
    )
    # The image pad: position (0, 0, 0), an identity rotation (model.py:727).
    pad_ids = _zeros(layout.img_pad, device)

    # Queries: one entry per image token at its OWN position on the coarse
    # grid (``r / f``, then the GTCN multiplier) — no floor division, so the
    # f^2 tokens of a block stay distinct instead of collapsing onto one
    # coordinate with a phase jump at every boundary (mechanism 1).
    pos = torch.arange(layout.h * layout.w, device=device)
    q_rows = (pos // layout.w).to(torch.float32) / ff * norm_y * h_scale + h_start
    q_cols = (pos % layout.w).to(torch.float32) / ff * norm_x * w_scale + w_start
    q_image_ids = _ids(
        torch.full((layout.h * layout.w,), float(cap_len + 1), device=device),
        q_rows, q_cols, device,
    )
    # Keys: one entry per pooled block, at its block's GEOMETRIC CENTROID
    # ``R + (f - 1) / (2f)`` — the mean of the block's query positions, so the
    # phase step across a block boundary equals the one inside the block — and
    # scaled by the same GTCN multiplier, so the keys stay the mean of their
    # block's query positions.  A PARTIAL block (the last one on an axis the
    # grid does not tile) has fewer members, so its centroid is the mean of
    # *its* members instead (``_block_offsets``).
    coarse = torch.arange(nh * nw, device=device)
    kv_offset = (ff - 1.0) / (2.0 * ff)
    row_off = _block_offsets(nh, layout.h, f, kv_offset, device)
    col_off = _block_offsets(nw, layout.w, f, kv_offset, device)
    block_rows = coarse // nw
    block_cols = coarse % nw
    kv_rows = (block_rows.to(torch.float32) + row_off[block_rows]) * norm_y * h_scale + h_start
    kv_cols = (block_cols.to(torch.float32) + col_off[block_cols]) * norm_x * w_scale + w_start
    kv_image_ids = _ids(
        torch.full((nh * nw,), float(cap_len + 1), device=device),
        kv_rows, kv_cols, device,
    )

    rope_q = embedder(torch.cat((cap_ids, q_image_ids, pad_ids), dim=1))
    rope_kv = embedder(torch.cat((cap_ids, pad_ids, kv_image_ids), dim=1))
    # Head-shared either way (``_head_shared``), so a chained embedder that
    # returns lumina's own ``[b, N, 1, ...]`` view is handled too.
    return _head_shared(rope_q), _head_shared(rope_kv)


def stitch_fine_zeros(
    out_text: torch.Tensor, out_image: torch.Tensor, layout: ZImageJointLayout
) -> torch.Tensor:
    """A Phase II output in stream order, zeros only in the discarded pad.

    :func:`src.area_attn.windowed_attention` answers for the text and image
    tokens only — the image pad slots are neither text nor windows — so the seam
    has to put the answers back where the stream expects them: the text answer
    spans the whole PADDED caption block ``[0, cap_len)`` (the prompt's tokens
    plus the model's own learned caption pad, ``cap_pad_token``,
    ``comfy/ldm/lumina/model.py:620-623``), the image tokens at
    ``[cap_len, cap_len + h*w)``, and ONLY the image pad span stays zero (it is
    discarded at unpatchify; ``model.py:725-727``).
    """
    if out_text.shape[2] != layout.cap_len or out_image.shape[2] != layout.h * layout.w:
        raise ValueError(
            f"AReA: the Phase II outputs ({out_text.shape[2]} text, "
            f"{out_image.shape[2]} image tokens) do not match the Z-Image "
            f"layout (cap_len={layout.cap_len}, h*w={layout.h * layout.w}), so "
            "they cannot be stitched back into the padded stream. This call "
            "runs plain attention."
        )
    b, heads, _, dim = out_image.shape
    out = out_image.new_zeros(b, heads, layout.seq_len, dim)
    out[:, :, : layout.cap_len] = out_text
    out[:, :, layout.img_start : layout.img_end] = out_image
    return out
