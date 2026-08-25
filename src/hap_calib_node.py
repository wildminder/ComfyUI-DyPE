"""HAP calibration node (plan 2026-08-16-hap-calibration-node).

Provides the ``HAP Calibrate`` ComfyUI node that runs the full HAP scope-plan
calibration pipeline in-graph:

    collector (chunked differentiable attention) → Taylor scores →
    knapsack solver → scope-plan JSON → SCOPE_PLAN output

All heavy math is delegated to the existing, fully-tested primitives in
:mod:`src.hap_calib` and :mod:`src.hap`.  This module adds only thin glue:

- :class:`CalibrationSpec` — validated parameter bundle (P0)
- :func:`resolve_prompts` — prompt list resolution (P0)
- :func:`make_calibration_loss` — differentiable loss factory (P1)
- :func:`default_calibration_forward` — one-step model forward (P2)
- :func:`run_hap_calibration` — multi-prompt orchestrator (P3)
- :func:`write_scope_plan` / :func:`resolve_output_dir` — persistence (P4)
- :class:`HAPCalibrate` — the ComfyUI node (P5)

ComfyUI imports are LAZY (inside functions) so the module is importable in
pytest without the real ``comfy`` package.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

# ``comfy_api`` is only needed for the node class + schema.  The pure-math
# helpers (CalibrationSpec, resolve_prompts, losses, collector, orchestrator,
# persistence) must stay importable WITHOUT ComfyUI — the CLI dry-run and any
# standalone tooling import them directly.  So the import is optional; the node
# class below falls back to a plain ``object`` base when ``io`` is unavailable
# (its methods are only ever invoked inside ComfyUI / the pytest mock).
try:
    from comfy_api.latest import io
except ImportError:  # pragma: no cover - standalone CLI without ComfyUI
    io = None

logger = logging.getLogger("ComfyUI-DyPE")

# Pack root (this file lives in src/).
_PACK_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# P0 — Default calibration prompts (single source; CLI re-imports these)
# ---------------------------------------------------------------------------

DEFAULT_CALIBRATION_PROMPTS: List[str] = [
    "a photograph of a mountain landscape at sunrise",
    "a detailed portrait of an elderly sailor",
    "a bustling city street at night in the rain",
    "a macro shot of a dewdrop on a leaf",
    "an oil painting of a sailing ship in a storm",
]

LOSS_TYPES = ("output_norm", "reference_mse")


# ---------------------------------------------------------------------------
# P0/T0.1 — CalibrationSpec (validated parameter bundle)
# ---------------------------------------------------------------------------

@dataclass
class CalibrationSpec:
    """Validated bundle of every HAP calibration parameter.

    All fields mirror the node inputs (§3.1 of the plan).  Call
    :meth:`validate` before use; it raises ``ValueError`` naming the offending
    field.
    """

    width: int = 1024
    height: int = 1024
    num_prompts: int = 5
    num_scopes: int = 50
    budget_ratio: float = 0.10
    bins: int = 4000
    chunk: int = 256
    text_len: int = 512
    anchor_stride: int = 32
    calib_sigma: float = 1.0
    seed: int = 3407
    loss_type: str = "output_norm"
    prompts: List[str] = field(default_factory=list)
    reference_latent: Optional[torch.Tensor] = None
    # PURGE-BETWEEN-PROMPTS KNOB (plan 2026-08-24 P5): opt-in gc + allocator
    # purge between calibration prompts.  Helps low-VRAM cards where one
    # prompt's residual cache would otherwise overlap the next forward.
    # Default False keeps current speed; purging has NO numeric effect on
    # results.
    purge_between_prompts: bool = False

    def validate(self) -> None:
        """Raise ``ValueError`` with context for every invalid field."""
        if self.width < 256 or self.width % 8 != 0:
            raise ValueError(
                f"CalibrationSpec: width must be a multiple of 8 and >= 256, "
                f"got {self.width}"
            )
        if self.height < 256 or self.height % 8 != 0:
            raise ValueError(
                f"CalibrationSpec: height must be a multiple of 8 and >= 256, "
                f"got {self.height}"
            )
        if self.num_prompts < 1:
            raise ValueError(
                f"CalibrationSpec: num_prompts must be >= 1, got {self.num_prompts}"
            )
        if self.num_scopes < 2:
            raise ValueError(
                f"CalibrationSpec: num_scopes must be >= 2, got {self.num_scopes}"
            )
        if not (0.0 < self.budget_ratio <= 1.0):
            raise ValueError(
                f"CalibrationSpec: budget_ratio must be in (0, 1], "
                f"got {self.budget_ratio}"
            )
        if self.bins < 1:
            raise ValueError(
                f"CalibrationSpec: bins must be >= 1, got {self.bins}"
            )
        if self.chunk < 1:
            raise ValueError(
                f"CalibrationSpec: chunk must be >= 1, got {self.chunk}"
            )
        if self.text_len < 0:
            raise ValueError(
                f"CalibrationSpec: text_len must be >= 0, got {self.text_len}"
            )
        if self.anchor_stride < 0:
            raise ValueError(
                f"CalibrationSpec: anchor_stride must be >= 0, "
                f"got {self.anchor_stride}"
            )
        if not (0.0 <= self.calib_sigma <= 1.0):
            raise ValueError(
                f"CalibrationSpec: calib_sigma must be in [0, 1], "
                f"got {self.calib_sigma}"
            )
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(
                f"CalibrationSpec: loss_type must be one of {LOSS_TYPES}, "
                f"got {self.loss_type!r}"
            )
        if self.loss_type == "reference_mse" and self.reference_latent is None:
            raise ValueError(
                "CalibrationSpec: loss_type='reference_mse' requires a "
                "reference_latent input"
            )
        if len(self.prompts) == 0:
            raise ValueError(
                "CalibrationSpec: prompt list is empty after resolution"
            )


# ---------------------------------------------------------------------------
# P0/T0.3 — Prompt resolution
# ---------------------------------------------------------------------------

def resolve_prompts(
    prompts_text: str = "",
    prompts_file: str = "",
    num_prompts: int = 5,
    pack_root: str = "",
) -> List[str]:
    """Resolve the calibration prompt list from node inputs.

    Priority: ``prompts_file`` (if non-empty) > ``prompts_text`` (multiline) >
    :data:`DEFAULT_CALIBRATION_PROMPTS`.  Result is truncated to
    ``num_prompts``.

    Args:
        prompts_text: multiline string, one prompt per line.
        prompts_file: optional path to a text file (one prompt per line).
            Relative paths resolve against ``pack_root`` (or the pack root).
        num_prompts: cap on the number of prompts returned.
        pack_root: root for relative path resolution (defaults to pack root).

    Returns:
        Non-empty list of prompt strings (length <= num_prompts).

    Raises:
        ValueError: if ``prompts_file`` is given but cannot be read.
    """
    root = pack_root or _PACK_ROOT
    lines: List[str] = []

    if prompts_file and prompts_file.strip():
        path = prompts_file.strip()
        if not os.path.isabs(path):
            path = os.path.join(root, path)
        if not os.path.isfile(path):
            raise ValueError(
                f"resolve_prompts: prompts file not found: {prompts_file!r} "
                f"(resolved to {path!r})"
            )
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    elif prompts_text and prompts_text.strip():
        lines = prompts_text.splitlines()
    else:
        lines = list(DEFAULT_CALIBRATION_PROMPTS)

    # Strip, drop blanks.
    prompts = [ln.strip() for ln in lines if ln.strip()]
    if not prompts:
        prompts = list(DEFAULT_CALIBRATION_PROMPTS)

    return prompts[: max(1, int(num_prompts))]


# ---------------------------------------------------------------------------
# P1 — Calibration loss functions
# ---------------------------------------------------------------------------

def make_calibration_loss(
    loss_type: str,
    reference: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a differentiable calibration loss function.

    Args:
        loss_type: ``"output_norm"`` or ``"reference_mse"``.
        reference: target tensor for ``reference_mse`` (must be provided).

    Returns:
        ``loss_fn(output) -> scalar tensor`` (differentiable wrt output).

    Raises:
        ValueError: unknown loss_type or missing reference.
    """
    if loss_type == "output_norm":
        def _output_norm(output: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.mse_loss(
                output, torch.zeros_like(output)
            )
        return _output_norm

    if loss_type == "reference_mse":
        if reference is None:
            raise ValueError(
                "make_calibration_loss: loss_type='reference_mse' requires "
                "a reference tensor"
            )
        ref = reference

        def _reference_mse(output: torch.Tensor) -> torch.Tensor:
            r = ref
            # Broadcast batch dim if reference has B=1 and output has B>1.
            if r.shape[0] == 1 and output.shape[0] > 1:
                r = r.expand(output.shape[0], *r.shape[1:])
            if r.shape != output.shape:
                raise ValueError(
                    f"reference_mse: shape mismatch — output "
                    f"{tuple(output.shape)} vs reference {tuple(r.shape)}"
                )
            return torch.nn.functional.mse_loss(output, r)

        return _reference_mse

    raise ValueError(
        f"make_calibration_loss: unknown loss_type {loss_type!r}; "
        f"expected one of {LOSS_TYPES}"
    )


# ---------------------------------------------------------------------------
# P2 — Calibration forward (ComfyUI bridge)
# ---------------------------------------------------------------------------

def default_calibration_forward(
    model,
    positive,
    negative,
    width: int,
    height: int,
    sigma: float,
    seed: int,
    device=None,
) -> torch.Tensor:
    """Run ONE denoising-step model forward at ``sigma``; return the output.

    This is the DEFAULT forward function used by :func:`run_hap_calibration`
    when no injected ``forward_fn`` is provided.  It calls the model's
    ``apply_model`` (or the installed unet wrapper) directly under
    ``torch.enable_grad()`` so the chunked-attention leaves receive gradients.

    IMPORTANT: we do NOT use ``comfy.sample.sample`` because all k_diffusion
    samplers are decorated ``@torch.no_grad()`` which would prevent gradient
    flow to the attention leaves.  Instead we replicate the single-step
    forward that the sampler would perform.

    Args:
        model: ComfyUI ``ModelPatcher``.
        positive: CONDITIONING list (positive).
        negative: CONDITIONING list (negative).
        width, height: target pixel resolution.
        sigma: the noise level for this calibration step.
        seed: noise seed.
        device: target device (defaults to model's load device).

    Returns:
        The model's output tensor (denoised prediction), shape depends on
        model but is a differentiable function of the attention outputs.

    Raises:
        RuntimeError: if the ComfyUI runtime is not available.
    """
    try:
        import comfy.model_management
        import comfy.sample
        import comfy.sampler_helpers
        import comfy.samplers
    except ImportError as exc:
        raise RuntimeError(
            "HAP calibration requires the ComfyUI runtime.  Run this node "
            "inside ComfyUI (or use the CLI script in the ComfyUI venv).  "
            f"Import failed: {exc!r}"
        ) from exc

    if device is None:
        device = model.load_device if hasattr(model, "load_device") else torch.device("cpu")

    # Ensure model is on GPU and finalized for a forward.
    #
    # MODEL REUSE (plan 2026-08-24 P4): we deliberately load the SAME
    # ModelPatcher object that flows through the graph — never a clone — so it
    # stays registered with ComfyUI's memory manager and post-calibration
    # inference reuses it directly.  Symmetrically, HAPCalibrate.execute must
    # NEVER detach/unload this patcher after calibration: unloading would
    # force exactly the reload-into-RAM we are trying to avoid.  The correct
    # post-run action is only gc + cache purge (see _purge_calibration_memory),
    # which lets ComfyUI re-pin the weights into VRAM on its own.
    comfy.model_management.load_models_gpu([model])
    if hasattr(model, "pre_run"):
        try:
            model.pre_run()
        except Exception:  # pragma: no cover - defensive  # leak-guard: pre_run is best-effort
            pass

    # Build empty latent at target resolution.
    latent_format = model.get_model_object("latent_format")
    channels = getattr(latent_format, "latent_channels", 4)
    latent_dims = getattr(latent_format, "latent_dimensions", 2)
    downscale = getattr(latent_format, "spacial_downscale_ratio", 8)

    lh = height // downscale
    lw = width // downscale
    gen = torch.Generator(device="cpu").manual_seed(seed)
    latent = torch.randn(
        1, channels, lh, lw, generator=gen, dtype=torch.float32
    )
    if latent_dims == 3:
        latent = latent.unsqueeze(2)  # (1, C, 1, H, W) for video models

    # Scale into model's internal latent space.
    diffusion_model = model.model
    latent = diffusion_model.process_latent_in(latent.to(device))

    # Compute timestep from sigma.
    model_sampling = diffusion_model.model_sampling
    timestep = model_sampling.timestep(torch.tensor(sigma, dtype=torch.float32))
    timestep = timestep.unsqueeze(0).to(device)

    # Prepare conditioning using ComfyUI's internal machinery.
    # We use sampling_function with cond_scale=1.0 which skips the uncond
    # pass entirely (cfg optimization), giving us a single conditioned forward.
    #
    # IMPORTANT: sampling_function -> calc_cond_batch -> get_area_and_mult
    # expects the INTERNAL cond format (a list of dicts, each with a
    # `model_conds` key holding CondBase objects).  The public CONDITIONING
    # type is a list of (tensor, dict) tuples, so we must convert it first —
    # otherwise get_area_and_mult does `conds["model_conds"]` on a list and
    # raises `TypeError: list indices must be integers or slices, not str`.
    # This mirrors CFGGuider (comfy/samplers.py): convert_cond + process_conds,
    # then sampling_function on the INNER BaseModel (model.model).
    inner_model = model.model
    conds = {
        "positive": comfy.sampler_helpers.convert_cond(positive),
        "negative": comfy.sampler_helpers.convert_cond(negative),
    }
    # Resolve areas/masks/timesteps and wrap cross_attn into CondBase objects
    # via the model's extra_conds (same as CFGGuider.inner_sample).
    conds = comfy.samplers.process_conds(inner_model, latent, conds, device)

    model_options = model.model_options

    # Exit ComfyUI's inference_mode AND enable grad so the chunked-attention
    # leaves track gradients (see collect_scope_scores_for_model for why
    # enable_grad alone is insufficient under inference_mode).
    with torch.inference_mode(mode=False), torch.enable_grad():
        output = comfy.samplers.sampling_function(
            inner_model, latent, timestep,
            uncond=conds["negative"],
            cond=conds["positive"],
            cond_scale=1.0,
            model_options=model_options,
            seed=seed,
        )

    return output


def calibration_forward(
    model,
    spec: "CalibrationSpec",
    prompt_index: int,
    positive,
    negative,
    forward_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """Seam wrapper: run one calibration forward for prompt ``prompt_index``.

    All orchestrator code calls ONLY this function so tests can inject a toy
    ``forward_fn``.

    Args:
        model: ComfyUI ModelPatcher (or toy).
        spec: validated CalibrationSpec.
        prompt_index: 0-based index (used for seed offset).
        positive: CONDITIONING.
        negative: CONDITIONING.
        forward_fn: injectable forward (defaults to
            :func:`default_calibration_forward`).

    Returns:
        Model output tensor.
    """
    if forward_fn is not None:
        return forward_fn(model, spec, prompt_index)

    return default_calibration_forward(
        model=model,
        positive=positive,
        negative=negative,
        width=spec.width,
        height=spec.height,
        sigma=spec.calib_sigma,
        seed=spec.seed + prompt_index,
    )


# ---------------------------------------------------------------------------
# P2/T2.3 — Collector with backend-aware patching + non-square skip
# ---------------------------------------------------------------------------

def _gpu_mem_str(device=None) -> str:
    """Return a short human-readable GPU memory usage string (or 'n/a').

    Diagnostic helper: lets the collector log how much VRAM is in use as
    attention chunks accumulate, so OOM root-causes (retained differentiable
    attention vs. weights vs. activations) can be validated.
    """
    try:
        if not torch.cuda.is_available():
            return "n/a (no CUDA)"
        idx = device.index if (device is not None and device.type == "cuda") else torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(idx) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(idx) / (1024 ** 3)
        total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 3)
        return f"alloc={alloc:.2f}GiB reserved={reserved:.2f}GiB total={total:.2f}GiB"
    except Exception:  # pragma: no cover - diagnostic only  # leak-guard: diagnostic string only
        return "n/a"


def _chunk_bytes(chunks) -> int:
    """Total bytes held by a list of attention chunk leaves."""
    return sum(c.numel() * c.element_size() for c in chunks)


def _free_scored_chunk(chunk: torch.Tensor) -> None:
    """Release a scored attention chunk's GPU memory (plan 2026-08-24 P1).

    After a chunk's ``A``/``G`` have been copied to CPU and scored, the GPU
    leaf is dead weight: nothing downstream reads it again (``records`` is
    consumed ONLY by the scoring loop).  Dropping the ``.grad`` reference and
    the tensor itself lets the CUDA caching allocator reuse the segment while
    scoring continues, instead of holding every layer's leaves resident until
    function exit (~3.6 GiB on Krea2).

    Safe to call on an already-freed (None) slot; never raises.
    """
    if chunk is None:
        return
    try:
        chunk.grad = None
    except Exception:  # pragma: no cover - defensive  # leak-guard: grad release is best-effort
        pass


def _purge_calibration_memory() -> None:
    """Release cached allocator blocks after calibration transients die.

    Public-API only (checklist rule #6): prefers ComfyUI's
    ``model_management.soft_empty_cache()`` so the memory manager stays
    consistent; falls back to raw ``torch.cuda.empty_cache()`` when the
    ComfyUI runtime is unavailable (tests / standalone CLI).  Best-effort —
    never raises.

    Without this purge the CUDA caching allocator keeps calibration's stale
    reserved segments mapped, so the next ``load_models_gpu`` sees
    insufficient free VRAM and aimdo keeps the model weights in RAM (the
    "model reloads into RAM after calibration" symptom).
    """
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
        return
    except Exception:  # leak-guard: allocator flush best-effort
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive  # leak-guard: allocator flush best-effort
        pass


# ---------------------------------------------------------------------------
# Memory-bounded calibration: per-block gradient checkpointing
# ---------------------------------------------------------------------------
#
# Root cause of the calibration OOM (see .dev/docs/oom.md): a single end-to-end
# backward retains O(layers) of autograd ACTIVATION memory (~0.81 GiB/layer on
# Krea2 → ~22 GiB for 28 layers > 16 GiB device).  The attention chunk leaves
# themselves are negligible (~0.3 GiB total).
#
# Fix: wrap each transformer block in ``torch.utils.checkpoint`` with
# ``use_reentrant=False``.  The NON-REENTRANT variant runs the forward WITH
# grad enabled and re-runs each block's forward during backward (also with
# grad).  (The reentrant variant runs forward under no_grad and would break
# .grad capture — do NOT use it.)
#
# RECORDING (empirically verified, tmp/dbg_ckpt2.py): because the forward runs
# WITH grad, the chunked-attention leaves created during the FORWARD are
# directly in the autograd graph (``A @ v = out``) and receive ``.grad`` from
# ``loss.backward()`` WITHOUT needing the backward recompute.  The recompute
# only bounds the NON-attention activation memory (the actual OOM cause).  So
# the collector records leaves during the forward exactly as in the
# non-checkpointed path — no backward-gating or key replay is needed.

# Attribute names probed (in FORWARD order) for transformer block lists.  A
# model may expose SEVERAL lists (e.g. Flux: ``double_blocks`` then
# ``single_blocks``); all non-empty matches are concatenated in this order and
# deduped by identity so the resulting flat index matches the model's forward
# (and hence the HRDiT attention-call counter used at inference).
_BLOCK_LIST_ATTRS = (
    "blocks",
    "layers",
    "transformer_blocks",
    "double_blocks",
    "single_blocks",
)


def _find_block_list(diffusion_model) -> List[torch.nn.Module]:
    """Locate the transformer blocks on a diffusion model as a FLAT list.

    Generic detection (plan option (a)): probes common attribute names for
    ``ModuleList``/``Module`` sequences whose children are the per-layer
    transformer blocks.  ALL non-empty matches are concatenated in the probe
    order (which matches the model's forward order) and deduped by identity, so
    multi-list models (e.g. Flux ``double_blocks`` + ``single_blocks``) yield a
    single flat, forward-ordered list.  Returns an empty list if no suitable
    list is found (the collector then falls back to the un-checkpointed forward).

    Args:
        diffusion_model: the inner ``diffusion_model`` (e.g. Krea2, Flux).

    Returns:
        Flat forward-ordered list of block ``Module``s (may be empty).
    """
    if diffusion_model is None:
        return []
    flat: List[torch.nn.Module] = []
    seen_ids = set()
    for attr in _BLOCK_LIST_ATTRS:
        candidate = getattr(diffusion_model, attr, None)
        if candidate is None:
            continue
        # Accept ModuleList or a plain list/tuple of Modules.
        items = None
        if isinstance(candidate, torch.nn.ModuleList) and len(candidate) > 0:
            items = list(candidate)
        elif isinstance(candidate, (list, tuple)) and len(candidate) > 0 and all(
            isinstance(m, torch.nn.Module) for m in candidate
        ):
            items = list(candidate)
        if items is None:
            continue
        for m in items:
            if id(m) not in seen_ids:
                seen_ids.add(id(m))
                flat.append(m)
    return flat


def _install_block_checkpointing(diffusion_model) -> List[Tuple]:
    """Wrap each transformer block's ``forward`` in gradient checkpointing.

    Uses ``torch.utils.checkpoint(..., use_reentrant=False)`` so the backward
    recompute runs with grad enabled (required for .grad capture on the
    chunked-attention leaves).

    Args:
        diffusion_model: the inner diffusion model.

    Returns:
        List of ``(block, original_forward)`` tuples for restoration via
        :func:`_uninstall_block_checkpointing`.  Empty list if no block list
        was found (caller should proceed without checkpointing).
    """
    from torch.utils.checkpoint import checkpoint as _torch_checkpoint

    block_list = _find_block_list(diffusion_model)
    if not block_list:
        return []

    installed: List[Tuple] = []
    for block in block_list:
        orig_forward = block.forward

        def _make_ckpt_forward(orig):
            def _ckpt_forward(*args, **kwargs):
                # preserve_rng_state=False (W2.7, 2026-08-25): calibration
                # forwards are DETERMINISTIC (no dropout / RNG consumers), so
                # stashing RNG state is unnecessary — and torch >= 2.6 hard-
                # rejects checkpointed forwards where the accelerator module
                # initializes mid-forward (CPU-only test environments hit this
                # on every block).  Disabling RNG preservation removes the
                # device-state stash entirely and keeps backward recompute
                # bit-identical for our deterministic graphs.
                return _torch_checkpoint(
                    orig, *args, use_reentrant=False,
                    preserve_rng_state=False, **kwargs
                )
            return _ckpt_forward

        block.forward = _make_ckpt_forward(orig_forward)
        installed.append((block, orig_forward))

    return installed


def _uninstall_block_checkpointing(installed: List[Tuple]) -> None:
    """Restore original block forwards installed by
    :func:`_install_block_checkpointing`."""
    for block, orig_forward in installed:
        block.forward = orig_forward


def _flush_gpu_allocator() -> None:
    """Best-effort GPU allocator flush before a calibration run.

    Mitigates the aimdo fragmentation caveat (.dev/docs/oom.md): a long-lived
    ComfyUI session accumulates reserved-but-freed blocks that can fragment
    the large contiguous allocations calibration needs.  ``empty_cache``
    returns unused cached memory to the driver.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - defensive  # leak-guard: allocator flush best-effort
        pass


def collect_scope_scores_for_model(
    model,
    model_type: str,
    forward_fn: Callable[[], torch.Tensor],
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    num_scopes: int,
    text_len: int = 0,
    chunk: int = 256,
    scale: Optional[float] = None,
    meta: Optional[Dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Collect per-(layer, head, scope) Taylor scores from ONE backward pass.

    Unlike :func:`src.hap_calib.collect_scope_scores` (which patches only the
    module-level ``optimized_attention``), this function patches the
    BACKEND-SPECIFIC bound symbols (same targets as
    :func:`src.spa._spa_patch_targets`) so it fires for real ComfyUI DiT
    backends that captured the symbol at import time.

    Non-square attention calls (cross-attention, e.g. Anima) are SKIPPED —
    they pass through to the original attention unrecorded.  Only square
    self/joint attention calls are collected.

    The installed chunked attention handles BOTH reshape conventions:
    - ``skip_reshape=True``: q is ``(B, H, T, D)`` — used directly.
    - ``skip_reshape=False``: q is ``(B, T, H*D)`` — reshaped to heads.

    Args:
        model: ModelPatcher (used to detect backend if model_type == "auto").
        model_type: backend key (``"flux"``, ``"qwen"``, etc.) or ``"auto"``.
        forward_fn: zero-arg callable running one calibration forward.
        loss_fn: ``output -> scalar loss`` (differentiable).
        num_scopes: candidate scopes N_scope.
        text_len: leading text tokens (never omitted).
        chunk: query rows per chunk (memory knob; result-invariant).
        scale: attention scale override.  ``None`` (default) mirrors the
            ComfyUI convention exactly: a per-call ``kwargs["scale"]`` wins,
            else ``dim_head ** -0.5`` (see ``attention_basic``).  Pass an
            explicit float (e.g. ``1.0``) only for tests whose toy attention
            uses that scale.

    Returns:
        ``(quality_cost, compute_cost, seq_len)`` — the first two are
        ``(L, H, S)`` fp64 tensors; ``seq_len`` is the OBSERVED attention
        sequence length (authoritative for cost/summary accounting).
    """
    import importlib

    from .hap_calib import (
        _chunk_row_scores,
        calibration_cost_table,
        chunked_attention,
    )

    # Determine patch targets.
    from .spa import _spa_patch_targets, _spa_resolve_type

    if model_type == "auto":
        try:
            dm = model.model.diffusion_model
            model_type = _spa_resolve_type("auto", dm)
        except Exception:  # probe: auto type detection fallback
            model_type = "flux"  # safe fallback

    targets = list(_spa_patch_targets(model_type))
    # Always include the module-global for classic CrossAttention blocks.
    global_target = ("comfy.ldm.modules.attention", "optimized_attention", False)
    if global_target not in targets:
        targets.append(global_target)

    # records: (layer_key, chunk_leaves) per COLLECTED call.  ``layer_key``
    # groups SPA variant passes of the same layer (see below).
    records: List[Tuple[int, List[torch.Tensor]]] = []
    call_counter = [0]
    skipped_nonsquare = [False]  # one-time log latch
    skipped_masked = [False]     # one-time log latch
    mag_diag = [0]               # magnitude-diagnostic call budget (case-b NaN)
    installed: List[Tuple] = []  # (module, attr, orig)

    # DIAGNOSTIC (grad-missing root cause): which autograd phase created each
    # record.  ``use_reentrant=False`` checkpointing RE-RUNS each block's forward
    # during ``loss.backward()``; if the still-patched attention fires during that
    # RECOMPUTE it appends extra records whose leaves never receive ``.grad``.
    # Tagging the phase lets us tell "backward-recompute records" (hypothesis #1)
    # apart from "forward records that got no grad" (hypothesis #2).
    phase = ["forward"]
    record_phases: List[str] = []  # parallel to ``records``: phase at creation

    # MEMORY-BOUNDED CALIBRATION — RECORDING (see module header).  Empirically
    # verified (tmp/dbg_ckpt2.py): with ``use_reentrant=False`` the FORWARD pass
    # runs WITH grad enabled, so the chunked-attention leaves created during the
    # forward are directly in the autograd graph (``A @ v = out``) and receive
    # ``.grad`` from ``loss.backward()`` WITHOUT needing the backward recompute.
    # The recompute (which checkpoint triggers) only bounds the NON-attention
    # activation memory — the actual OOM cause.  Therefore we record leaves
    # during the FORWARD exactly as in the non-checkpointed path; no
    # backward-gating or key replay is needed.
    #
    # LAYER KEY: the canonical call-counter-based key (matches the HAP runtime's
    # scope-plan indexing).  The forward runs in forward order whether or not
    # checkpointing is active, so the sequential counter is always correct.
    def _canonical_key() -> int:
        """The call-counter-based layer key (matches the HAP runtime indexing).

        Mirrors the pre-checkpointing convention: HRDiT wrapper counter minus one
        when a wrapper is live, else the sequential call index.
        """
        from .spa_context import get_hrdit_layer_idx
        cur = get_hrdit_layer_idx()
        return (cur - 1) if cur > 0 else call_counter[0]

    def _make_chunked_attn(orig_fn):
        """Build a convention-aware chunked attention replacement.

        Output convention mirrors the real ComfyUI attention functions: the
        output is ``(B, T, H*D)`` unless ``skip_output_reshape=True`` (then
        ``(B, H, T, D)``) — exactly what ``attention_pytorch`` /
        ``attention_basic`` return for either input layout.
        """

        def chunked_attn(q, k, v, heads, mask=None, attn_precision=None,
                         skip_reshape=False, skip_output_reshape=False, **kwargs):
            def _passthrough():
                return orig_fn(q, k, v, heads, mask, attn_precision,
                               skip_reshape, skip_output_reshape, **kwargs)

            # NON-SQUARE GUARD: cross-attention calls pass through unrecorded
            # (plan D13).  4D layout: (B, H, T, D) -> T at index 2; 3D layout:
            # (B, T, H*D) -> T at index 1.
            q_len = q.shape[2] if q.dim() == 4 else q.shape[1]
            k_len = k.shape[2] if k.dim() == 4 else k.shape[1]
            if q_len != k_len:
                if not skipped_nonsquare[0]:
                    skipped_nonsquare[0] = True
                    logger.debug(
                        "HAP calib: skipping non-square attention call "
                        "(q_len=%d, k_len=%d) — cross-attention is not "
                        "calibrated.", q_len, k_len,
                    )
                return _passthrough()

            # MASK GUARD (v1): calls carrying an external attention mask are not
            # calibrated (the scope-mask semantics assume unmasked joint
            # attention).  Pass through unrecorded.
            if mask is not None:
                if not skipped_masked[0]:
                    skipped_masked[0] = True
                    logger.debug(
                        "HAP calib: skipping masked attention call — masked "
                        "backends are not calibrated in v1."
                    )
                return _passthrough()

            # GQA GUARD: k/v with fewer heads than q cannot be chunked
            # directly; pass through unrecorded.
            if kwargs.get("enable_gqa", False) or k.shape != q.shape:
                return _passthrough()

            # Normalize to (B, H, T, D).
            if q.dim() == 3:
                # (B, T, H*D) -> (B, H, T, D)
                b, t, hd = q.shape
                dim_head = hd // heads
                q4 = q.reshape(b, t, heads, dim_head).permute(0, 2, 1, 3)
                k4 = k.reshape(b, k.shape[1], heads, dim_head).permute(0, 2, 1, 3)
                v4 = v.reshape(b, v.shape[1], heads, dim_head).permute(0, 2, 1, 3)
            else:
                q4, k4, v4 = q, k, v

            # SCALE CONVENTION (2026-08-18, P19 — mirrors ComfyUI's
            # ``attention_basic``: ``scale = kwargs.get("scale", dim_head **
            # -0.5)``).  The collector previously hardcoded ``scale=1.0``,
            # which made the calibration forward run with logits
            # ``sqrt(dim_head)``x too large (~11.3x at head_dim=128) —
            # calibrating against a WRONG attention distribution AND
            # contributing to the fp16 logit overflow.  Resolution order:
            #   1. ``kwargs["scale"]`` — the model's explicit per-call scale
            #      (exactly what the real attention function would use).
            #   2. the outer ``scale`` parameter (explicit override; tests).
            #   3. ``dim_head ** -0.5`` — ComfyUI's default.
            dim_head = q4.shape[-1]
            if kwargs.get("scale", None) is not None:
                eff_scale = float(kwargs["scale"])
            elif scale is not None:
                eff_scale = float(scale)
            else:
                eff_scale = dim_head ** -0.5

            # NaN ORIGIN DIAGNOSTIC (2026-08-18, forward-cascade root cause):
            # the live run showed layer 0's attention ``A`` partially NaN and
            # every later layer fully NaN — a forward-propagating cascade whose
            # ORIGIN is at/before layer 0's attention.  To distinguish
            #   (a) NaN ALREADY in q/k/v (=> upstream: calibration input latent,
            #       conditioning, or a layer-0 projection weight), from
            #   (b) q/k/v clean but ``A`` NaN (=> our bf16 ``q@kᵀ``/softmax
            #       produces it; flash/SDPA compute in fp32 internally so normal
            #       inference is unaffected),
            # count NaN/inf in q/k/v of EVERY forward call.  Cheap (no large
            # allocation) and only logged when non-zero.
            if phase[0] == "forward":
                # no_grad: these probes must NOT build autograd nodes / save
                # tensors, or the forward and the checkpoint backward-recompute
                # would save a DIFFERENT number of tensors (CheckpointError).
                with torch.no_grad():
                    q_bad = int(torch.isnan(q4).sum().item()) + int(torch.isinf(q4).sum().item())
                    k_bad = int(torch.isnan(k4).sum().item()) + int(torch.isinf(k4).sum().item())
                    v_bad = int(torch.isnan(v4).sum().item()) + int(torch.isinf(v4).sum().item())
                if q_bad or k_bad or v_bad:
                    logger.warning(
                        "[HAP calib][qkv-nan] call=%d layer_key=%d: q has %d, "
                        "k has %d, v has %d NaN/inf entries.  %s",
                        call_counter[0], _canonical_key(), q_bad, k_bad, v_bad,
                        ("NaN in q/k/v => UPSTREAM source (calibration input "
                         "latent / conditioning / a projection weight), NOT the "
                         "attention math.")
                        if (q_bad or k_bad or v_bad) else "",
                    )

            # MAGNITUDE DIAGNOSTIC (2026-08-18, case-b root cause): the qkv-nan
            # check above only tests isnan/isinf, so it MISSES huge-but-FINITE
            # q/k that overflow the bf16 ``q @ kᵀ`` matmul.  The live run showed
            # the first long-sequence main block (layer_key=4, T=1198) receives
            # FINITE q/k/v yet produces ~25.6% NaN attention — so the NaN is born
            # in our explicit bf16 matmul/softmax (flash/SDPA accumulate in fp32
            # internally and are unaffected).  To confirm, log the q/k/v magnitude
            # and the chunk-0 logits (min/max/|.|max/inf/NaN) for the first few
            # finite-q/k large-T calls.
            if (
                phase[0] == "forward"
                and mag_diag[0] < 3
                and q4.shape[2] >= 512
                and not (q_bad or k_bad or v_bad)
            ):
                mag_diag[0] += 1
                # no_grad: the magnitude/logit probes are DIAGNOSTIC-ONLY.  They
                # run during the FORWARD but are skipped during the checkpoint
                # BACKWARD-RECOMPUTE (phase-gated).  If they built autograd nodes
                # they would save tensors in the forward that the recompute never
                # saves -> ``CheckpointError: A different number of tensors was
                # saved during the original forward and recomputation`` (the live
                # 2026-08-18 crash: 298 vs 280).  Under no_grad they save nothing,
                # so forward and recompute stay identical regardless of whether
                # this block fires.
                with torch.no_grad():
                    q_max = float(q4.abs().max().item())
                    k_max = float(k4.abs().max().item())
                    v_max = float(v4.abs().max().item())
                    # Chunk-0 logits only (one extra matmul, cheap).
                    c0 = min(chunk, q4.shape[2])
                    logits0 = torch.matmul(
                        q4[:, :, :c0, :], k4.transpose(-1, -2)
                    ) * eff_scale
                    lg_bad = int(torch.isnan(logits0).sum().item()) + int(
                        torch.isinf(logits0).sum().item()
                    )
                    lg_min = float(logits0.min().item())
                    lg_max = float(logits0.max().item())
                    lg_absmax = float(logits0.abs().max().item())
                logger.warning(
                    "[HAP calib][mag] call=%d layer_key=%d T=%d H=%d "
                    "scale=%.6g dtype=%s: |q|max=%.4g |k|max=%.4g "
                    "|v|max=%.4g  logits[chunk0] min=%.4g max=%.4g "
                    "|.|max=%.4g NaN/inf=%d.  %s",
                    call_counter[0], _canonical_key(), q4.shape[2],
                    q4.shape[1], eff_scale, q4.dtype, q_max, k_max, v_max,
                    lg_min, lg_max, lg_absmax, lg_bad,
                    ("FINITE q/k but inf/NaN logits => bf16 matmul overflow "
                     "(fix: compute logits+softmax in fp32, keep stored A in "
                     "the model dtype — no retained-VRAM cost).")
                    if lg_bad else
                    ("finite logits => NaN must come from a softmax edge case "
                     "or the scale; inspect the values above."),
                )

            out4, chunks = chunked_attention(q4, k4, v4, scale=eff_scale, chunk=chunk)

            def _record(key: int) -> None:
                """Append ``(key, chunks)`` and emit the diagnostic VRAM log."""
                records.append((key, chunks))
                record_phases.append(phase[0])
                retained = sum(_chunk_bytes(ch) for _, ch in records)
                leaf0 = chunks[0]
                logger.info(
                    "[HAP calib][mem] call=%d layer_key=%d phase=%s T=%d H=%d "
                    "req_grad=%s grad_fn=%s retained_attn=%.2fGiB gpu=%s",
                    call_counter[0], key, phase[0], q4.shape[2], q4.shape[1],
                    leaf0.requires_grad, leaf0.grad_fn is not None,
                    retained / (1024 ** 3), _gpu_mem_str(q4.device),
                )

            # RECORDING (memory-bounded calibration).  Empirically verified
            # (tmp/dbg_ckpt2.py): with ``use_reentrant=False`` the forward runs
            # WITH grad, so the forward leaves are directly in the graph and
            # receive ``.grad`` from ``loss.backward()`` — no backward-gating or
            # key replay needed.  Record during the forward with the canonical
            # (call-counter) key, exactly as in the non-checkpointed path.  The
            # forward runs in forward order whether or not checkpointing is
            # active, so the sequential counter is always correct.
            #
            # BACKWARD-RECOMPUTE RECORDING GATE (plan 2026-08-17-hap-calib-
            # backward-recompute-no-grad-fix).  ``use_reentrant=False``
            # checkpointing RE-RUNS each block's forward during
            # ``loss.backward()``; the still-patched ``chunked_attn`` fires
            # again and would append spurious ``phase='backward'`` records whose
            # orphaned leaves never receive ``.grad`` (the crash at layer 32).
            # We must NOT swap the computation (checkpoint requires the recompute
            # to run the EXACT same ops/shapes as the forward — a passthrough to
            # the pristine attention raises ``CheckpointError: Recomputed values
            # ... different metadata``).  So we keep running ``chunked_attention``
            # (identical ops) but SKIP the recording + counter bump during the
            # recompute.  The forward-phase leaves remain the authoritative
            # grad-carriers.
            if phase[0] == "forward":
                key = _canonical_key()
                call_counter[0] += 1
                _record(key)

            # Reshape output to the caller's convention (see docstring).
            if skip_output_reshape:
                return out4  # (B, H, T, D)
            b, h, t, d = out4.shape
            return out4.permute(0, 2, 1, 3).reshape(b, t, h * d)

        return chunked_attn

    # Install on all targets.
    for mod_path, attr, _is_masked in targets:
        try:
            mod = importlib.import_module(mod_path)
        except Exception:  # probe: target module may be absent
            continue
        orig = getattr(mod, attr, None)
        if orig is None:
            continue
        replacement = _make_chunked_attn(orig)
        setattr(mod, attr, replacement)
        installed.append((mod, attr, orig))

    if not installed:
        raise RuntimeError(
            "collect_scope_scores_for_model: no attention symbols could be "
            f"patched for model_type={model_type!r}. Ensure the ComfyUI "
            "runtime is available."
        )

    # MEMORY-BOUNDED CALIBRATION: install per-block gradient checkpointing on
    # the diffusion model (if a block list is found).  This bounds peak
    # NON-ATTENTION activation memory to ~one block instead of O(layers) — the
    # root cause of the calibration OOM (.dev/docs/oom.md).  ``use_reentrant=False``
    # so the forward runs WITH grad (the forward attention leaves stay in the
    # graph and receive ``.grad``) and the backward recompute is grad-enabled.
    # If no block list is found (toy models / unusual architectures) we simply
    # run the un-checkpointed forward — recording is identical either way.
    diffusion_model = None
    try:
        diffusion_model = model.model.diffusion_model
    except Exception:  # probe: dm shape probe
        diffusion_model = None
    ckpt_installed = _install_block_checkpointing(diffusion_model)
    if ckpt_installed:
        logger.info(
            "[HAP calib] gradient checkpointing active on %d blocks "
            "(memory-bounded calibration).", len(ckpt_installed),
        )
    # Best-effort allocator flush to mitigate aimdo fragmentation before the
    # large contiguous allocations calibration needs (.dev/docs/oom.md).
    _flush_gpu_allocator()

    try:
        # ComfyUI's execution engine wraps the whole prompt in
        # ``torch.inference_mode()`` (execution.py).  Inference mode is STRONGER
        # than no_grad: tensors created inside become *inference tensors* and
        # ``torch.enable_grad()`` alone CANNOT escape it — only
        # ``torch.inference_mode(mode=False)`` can.  Without this escape the
        # chunked-attention leaves are inference tensors with no grad_fn and
        # ``loss.backward()`` raises "element 0 of tensors does not require
        # grad".  Exit inference mode AND re-enable grad for the whole
        # forward+loss+backward so the A_chunk leaves track gradients.
        with torch.inference_mode(mode=False), torch.enable_grad():
            output = forward_fn()
            if not records:
                raise RuntimeError(
                    "collect_scope_scores_for_model: the forward made no square "
                    "optimized_attention calls — nothing to calibrate. (Non-square "
                    "cross-attention and masked calls are skipped by design.)"
                )
            loss = loss_fn(output)
            n_fwd_records = len(records)
            # DIAGNOSTIC: switch phase so any attention calls made during the
            # backward RECOMPUTE (checkpoint re-running block forwards) are tagged
            # ``phase=backward`` in the [mem] log.
            phase[0] = "backward"
            # ``loss.backward()`` is fully synchronous; the forward leaves are
            # already in the graph (use_reentrant=False runs the forward with
            # grad), so they receive ``.grad`` here.  Checkpointing bounds the
            # non-attention activation memory via the backward recompute.
            loss.backward()
            # DIAGNOSTIC: after backward, report how many records were created in
            # each phase and how many chunk leaves actually received ``.grad``.
            # This distinguishes:
            #   - hypothesis #1: extra ``phase=backward`` records exist (the
            #     recompute fired the patched attention) -> those have grad=None.
            #   - hypothesis #2: all records are ``phase=forward`` but some chunks
            #     have grad=None (forward leaves did NOT receive grad).
            total_chunks = 0
            chunks_with_grad = 0
            for _, ch in records:
                for leaf in ch:
                    total_chunks += 1
                    if leaf.grad is not None:
                        chunks_with_grad += 1
            logger.info(
                "[HAP calib][diag] after backward: total_records=%d "
                "(forward=%d, backward_recompute=%d) total_chunks=%d "
                "chunks_with_grad=%d chunks_missing_grad=%d",
                len(records), n_fwd_records, len(records) - n_fwd_records,
                total_chunks, chunks_with_grad, total_chunks - chunks_with_grad,
            )
    finally:
        # Restore ALL originals (attention symbols + block forwards).
        for mod, attr, orig in installed:
            setattr(mod, attr, orig)
        _uninstall_block_checkpointing(ckpt_installed)

    # GROUP BY LAYER KEY: key -> list of CALLS; each call is the list of chunk
    # leaves covering that call's full query-row range.  SPA variant passes of
    # one layer share a key.  Each variant is a FULL attention over the same
    # tokens (not a continuation of the row range), so we score each call
    # independently (row offset resets per call) and SUM the per-call tables.
    # Summing is exact: ``spa_averaged_attention`` averages the variant
    # OUTPUTS, so each variant's ``A.grad`` already carries the ``1/N`` factor
    # and the total Taylor cost is the sum of the per-variant costs.  Keys are
    # sorted so the layer order matches block order.
    grouped: Dict[int, List[Tuple[List[torch.Tensor], str]]] = {}
    for (key, chunks), rphase in zip(records, record_phases):
        grouped.setdefault(key, []).append((chunks, rphase))
    sorted_keys = sorted(grouped.keys())

    # HETEROGENEOUS-HEAD-COUNT FILTER (Krea2 live crash: ``stack expects each
    # tensor to be equal size, but got [20, 50] at entry 0 and [48, 50] at entry
    # 4``).  A real DiT forward can route SEVERAL square attention calls through
    # the patched symbol with DIFFERENT head counts: the main transformer blocks
    # (Krea2: 48 heads) plus auxiliary modules (e.g. context/projector attention:
    # 20 heads).  ``torch.stack`` below requires a uniform ``(H, S)`` shape per
    # layer, so heterogeneous head counts crash.
    #
    # The HAP runtime ALREADY guards head-count mismatches by falling back to
    # plain attention (``src/hap.py``: ``if q.shape[1] != ctx.plan.num_heads``),
    # so the scope plan must cover exactly the DOMINANT head count — the main
    # transformer blocks HAP actually prunes.  Auxiliary modules with a different
    # head count are not HAP targets and are excluded here (logged once).
    #
    # Dominant = the head count shared by the MOST collected layers; ties break to
    # the head count that appears FIRST in forward (sorted-key) order, which is
    # deterministic.
    def _heads_of(key: int) -> int:
        # grouped[key][0] is a (chunks, phase) tuple; [0][0] is the chunks list;
        # [0][0][0] is the first chunk tensor of shape (B, H, C, T).
        return grouped[key][0][0][0].shape[1]

    head_counts: Dict[int, int] = {}
    for key in sorted_keys:
        h = _heads_of(key)
        head_counts[h] = head_counts.get(h, 0) + 1
    if len(head_counts) > 1:
        # Mode (max count); ties -> smallest first-appearance index in sorted_keys.
        first_index: Dict[int, int] = {}
        for i, key in enumerate(sorted_keys):
            h = _heads_of(key)
            first_index.setdefault(h, i)
        dominant = max(
            head_counts.keys(),
            key=lambda h: (head_counts[h], -first_index[h]),
        )
        excluded = [k for k in sorted_keys if _heads_of(k) != dominant]
        logger.info(
            "[HAP calib] heterogeneous head counts detected: %s.  Calibrating "
            "the dominant head count (%d heads, %d layers); excluding %d "
            "auxiliary attention layer(s) with other head counts (keys=%s) — "
            "the HAP runtime falls back to plain attention for them.",
            dict(sorted(head_counts.items())), dominant, head_counts[dominant],
            len(excluded), excluded,
        )
        sorted_keys = [k for k in sorted_keys if _heads_of(k) == dominant]
        # EXCLUDED-HEAD-COUNT METADATA (2026-08-23 head-count warning fix):
        # record the NON-dominant head counts so the runtime can log a friendly
        # INFO ("expected auxiliary fallback") instead of a scary WARNING when
        # those calls decline to plain attention.  Threaded to the caller via
        # the optional ``meta`` container (backward compatible).
        if meta is not None:
            meta["excluded_head_counts"] = sorted(
                h for h in head_counts if h != dominant
            )
    else:
        if meta is not None:
            meta["excluded_head_counts"] = []

    # OBSERVED GEOMETRY + TEXT_LEN CLAMP (text_len>seq_len root cause).
    # ``seq0`` is the OBSERVED attention sequence length — the authoritative
    # value for the cost table and the orchestrator's summary/flops accounting.
    # grouped[key][0] is a (chunks, phase) tuple; [0][0] is the chunks list;
    # [0][0][0] is the first chunk tensor.
    heads0 = grouped[sorted_keys[0]][0][0][0].shape[1]
    seq0 = grouped[sorted_keys[0]][0][0][0].shape[3]
    # The cost model requires ``text_len <= seq_len`` (seq = text + image).
    # The node's ``text_len`` knob is a FLUX-ism default (512) that can exceed
    # the observed sequence when calibration runs at a reduced resolution (OOM
    # workaround) or the model's real text length is below the knob.  Clamp to
    # ``[0, seq0]`` — mirroring the HAP runtime's ``max(0, min(text_len,
    # seq_len))`` (src/hap.py HapRuntime.attn) — so a knob mismatch degrades
    # gracefully instead of crashing ``band_compute_cost``.  The clamped value
    # is threaded through BOTH the quality-scoring loop (_chunk_row_scores) and
    # the cost table so the two stay consistent.
    eff_text_len = max(0, min(int(text_len), seq0))
    logger.info(
        "[HAP calib][geom] observed seq_len=%d heads=%d knob text_len=%d "
        "effective text_len=%d implied image_len=%d",
        seq0, heads0, int(text_len), eff_text_len, seq0 - eff_text_len,
    )
    if eff_text_len != int(text_len):
        logger.warning(
            "[HAP calib] text_len knob (%d) exceeds the observed attention "
            "sequence length (%d) — clamped to %d.  This usually means the "
            "calibration resolution is too small for the configured text_len, "
            "or the model's real text length is below the knob.  Consider "
            "raising width/height or lowering text_len.",
            int(text_len), seq0, eff_text_len,
        )

    quality_layers: List[torch.Tensor] = []
    for li, key in enumerate(sorted_keys):
        calls = grouped[key]
        heads = calls[0][0][0].shape[1]
        per_call_tables: List[torch.Tensor] = []
        for call_chunks, call_phase in calls:
            acc = torch.zeros(heads, num_scopes, dtype=torch.float64)
            offset = 0
            # EARLY GPU-LEAF RELEASE (plan 2026-08-24 P1): iterate by index so
            # each consumed chunk slot can be cleared as soon as its A/G have
            # been copied to CPU.  Nothing downstream reads the chunks again —
            # ``records`` feeds ONLY this loop — so freeing here returns the
            # leaf VRAM to the allocator while scoring continues instead of
            # holding all layers' leaves resident until function exit.
            for ci in range(len(call_chunks)):
                A_chunk = call_chunks[ci]
                if A_chunk is None:
                    continue  # already freed (defensive)
                if A_chunk.grad is None:
                    raise RuntimeError(
                        f"collect_scope_scores_for_model: layer {li} attention "
                        f"chunk has no gradient (created during phase="
                        f"{call_phase!r}) — the loss does not depend on "
                        "it. Ensure the calibration loss is differentiable "
                        "and the forward runs under torch.enable_grad()."
                    )
                rows = A_chunk.shape[2]
                # DEVICE-CONSISTENT SCORING: the attention leaves live on the
                # model's device (cuda:0 in ComfyUI) but the ``acc`` accumulator
                # and the whole downstream pipeline (stack -> calibrate_scope_plan
                # -> knapsack -> calibration_cost_table) are CPU-resident.  Move
                # A/G to CPU here so ``acc + _chunk_row_scores(...)`` never mixes
                # devices, and so the large fp64 (H, C, T) intermediates are
                # scored off-GPU (frees VRAM during scoring).  ``.to(dtype)``
                # alone PRESERVES device — that was the crash (cuda:0 vs cpu).
                A = A_chunk[0].detach().to(dtype=torch.float64, device="cpu")        # (H, C, T)
                G = A_chunk.grad[0].detach().to(dtype=torch.float64, device="cpu")   # (H, C, T)
                # Copy complete -> release the GPU leaf NOW (P1).  The local
                # ``A_chunk`` name still references the tensor until the next
                # iteration rebinds it, so also drop the record slot.
                _free_scored_chunk(A_chunk)
                call_chunks[ci] = None

                # NaN-SOURCE DIAGNOSTIC (2026-08-18): the live all-NaN quality
                # table could come from EITHER the attention values ``A`` (the
                # model's own bf16 FORWARD produced NaN q/k -> NaN softmax) OR
                # the gradient ``G`` (the model's bf16 BACKWARD produced NaN
                # dL/dout -> NaN dL/dA).  De-risking (tmp/diag_oom_redesign.py,
                # Q1) proved the attention-leaf backward ``grad_out @ vᵀ`` is
                # clean in bf16, so a NaN ``G`` means the NaN is UPSTREAM of the
                # attention (in the model), which up-casting ``A`` cannot fix.
                # Count NaN/inf in A and G SEPARATELY per layer so the next live
                # run pinpoints the true origin.
                a_bad = int(torch.isnan(A).sum().item()) + int(torch.isinf(A).sum().item())
                g_bad = int(torch.isnan(G).sum().item()) + int(torch.isinf(G).sum().item())
                if a_bad or g_bad:
                    logger.warning(
                        "[HAP calib][nan-src] layer %d chunk rows=%d: A has %d "
                        "and G has %d NaN/inf entries (of %d).  %s",
                        li, rows, a_bad, g_bad, A.numel(),
                        ("NaN in A => the model's FORWARD produced NaN attention "
                         "(check q/k / the conditioning / the input latent).")
                        if a_bad else
                        ("NaN only in G => the model's BACKWARD produced NaN "
                         "gradients (check the loss magnitude / bf16 grad "
                         "overflow upstream of the attention)."),
                    )

                acc = acc + _chunk_row_scores(A, G, num_scopes, eff_text_len, offset)
                offset += rows
            per_call_tables.append(acc)
        quality_layers.append(
            torch.stack(per_call_tables, dim=0).sum(dim=0)
        )
    quality_cost = torch.stack(quality_layers, dim=0)  # (L, H, S)

    # Prompt-independent compute cost (same for every layer).
    cost = calibration_cost_table(
        heads0, seq0, text_len=eff_text_len, num_scopes=num_scopes
    )  # (H, S)
    compute_cost = cost.unsqueeze(0).expand(
        quality_cost.shape[0], -1, -1
    ).clone()

    # CHUNKS-FREED FLAG (plan 2026-08-24 P1): observable for tests — every
    # recorded chunk slot was cleared during scoring.
    if meta is not None:
        meta["chunks_freed"] = True

    # POST-SCORING PURGE (plan 2026-08-24 P2): all chunk leaves were freed
    # above; return their cached segments to the driver so subsequent
    # allocations (next prompt / inference) see real free VRAM.
    _purge_calibration_memory()

    return quality_cost, compute_cost, seq0


# ---------------------------------------------------------------------------
# P3 — Orchestrator
# ---------------------------------------------------------------------------

def run_hap_calibration(
    model,
    spec: CalibrationSpec,
    model_type: str = "auto",
    positive=None,
    negative=None,
    forward_fn: Optional[Callable] = None,
) -> Tuple[Dict[str, List[List[float]]], Dict]:
    """Run the full HAP calibration pipeline.

    Args:
        model: ComfyUI ModelPatcher (or toy for tests).
        spec: validated CalibrationSpec (call ``spec.validate()`` first).
        model_type: backend key or "auto".
        positive: CONDITIONING (passed to forward_fn if using default).
        negative: CONDITIONING (passed to forward_fn if using default).
        forward_fn: injectable forward for tests.  Signature:
            ``forward_fn(model, spec, prompt_index) -> output_tensor``.

    Returns:
        ``(plan_dict, summary_dict)`` where ``plan_dict`` is the
        reference-format ``{"alphas", "betas"}`` and ``summary_dict`` has
        metadata for the summary output.

    Raises:
        RuntimeError: if HAP is already active on the model (D11 guard).
    """
    from .hap import ScopePlan, flops_ratio
    from .hap_calib import calibrate_scope_plan

    spec.validate()

    # D11 guard: calibrating through pruned attention measures the pruning,
    # not the model.
    hap_ctx = getattr(model, "_hap_ctx", None)
    if hap_ctx is not None and getattr(hap_ctx, "active", False):
        raise RuntimeError(
            "HAP calibration: the model already has HAP active.  Calibrate "
            "on an UNPRUNED model (remove or disable the HAP node upstream)."
        )

    prompts = spec.prompts
    num_prompts = len(prompts)
    quality_per_layer: Optional[List[List[torch.Tensor]]] = None
    compute_costs: Optional[List[torch.Tensor]] = None
    observed_seq_len: Optional[int] = None
    # EXCLUDED-HEAD-COUNT METADATA (2026-08-23): filled by the collector when
    # the model has heterogeneous head counts (auxiliary attention).  Identical
    # across prompts (same model), so one shared container suffices.
    calib_meta: Dict = {}
    t0 = time.time()

    for pi in range(num_prompts):
        logger.info(
            "[HAP calib] prompt %d/%d: %r", pi + 1, num_prompts,
            prompts[pi][:60],
        )

        def _fwd():
            return calibration_forward(
                model, spec, pi, positive, negative, forward_fn
            )

        loss_fn = make_calibration_loss(spec.loss_type, spec.reference_latent)

        quality, compute, observed_seq = collect_scope_scores_for_model(
            model=model,
            model_type=model_type,
            forward_fn=_fwd,
            loss_fn=loss_fn,
            num_scopes=spec.num_scopes,
            text_len=spec.text_len,
            chunk=spec.chunk,
            # None => ComfyUI convention: kwargs["scale"] wins, else
            # dim_head ** -0.5 (P19 scale fix — the old hardcoded 1.0 made
            # logits sqrt(dim_head)x too large during calibration).
            scale=None,
            meta=calib_meta,
        )

        # BETWEEN-PROMPT PURGE (plan 2026-08-24 P5, opt-in): release the just-
        # scored prompt's cached segments before the next forward so low-VRAM
        # cards don't stack two prompts' worth of reserved cache.  No numeric
        # effect — purge only returns freed segments to the driver.
        if spec.purge_between_prompts and pi < num_prompts - 1:
            import gc as _gc

            _gc.collect()
            _purge_calibration_memory()

        num_layers = quality.shape[0]
        if quality_per_layer is None:
            quality_per_layer = [[] for _ in range(num_layers)]
            compute_costs = [compute[l] for l in range(num_layers)]
            observed_seq_len = observed_seq
        for l in range(num_layers):
            quality_per_layer[l].append(quality[l])

    elapsed = time.time() - t0

    # Solve.
    plan_dict = calibrate_scope_plan(
        quality_per_layer, compute_costs,
        budget_ratio=spec.budget_ratio, bins=spec.bins,
    )

    # EXCLUDED-HEAD-COUNT METADATA (2026-08-23): persist the non-dominant head
    # counts into the plan so the runtime can distinguish an EXPECTED auxiliary
    # fallback (INFO) from a genuinely wrong plan (WARNING).  Only emitted when
    # non-empty, so single-head-count plans keep the exact legacy JSON shape.
    excluded_heads = calib_meta.get("excluded_head_counts") or []
    if excluded_heads:
        plan_dict["excluded_head_counts"] = list(excluded_heads)

    # Validate round-trip.
    plan = ScopePlan.from_dict(plan_dict)

    # Build summary.  ``seq_len`` is the OBSERVED attention sequence length
    # (authoritative); the width/height estimate is only a fallback.
    seq_len = observed_seq_len if observed_seq_len is not None else (
        (spec.width // 16) * (spec.height // 16) + spec.text_len
    )
    try:
        fr = flops_ratio(
            plan, seq_len, text_len=spec.text_len,
            anchor_stride=spec.anchor_stride,
        )
    except Exception:  # degrade: flops ratio is informational
        fr = None

    mean_betas = []
    for l in range(plan.num_layers):
        row = plan_dict["betas"][l]
        mean_betas.append(sum(row) / max(len(row), 1))

    summary = {
        "num_layers": plan.num_layers,
        "num_heads": plan.num_heads,
        "seq_len": seq_len,
        "text_len": spec.text_len,
        "num_prompts": num_prompts,
        "num_scopes": spec.num_scopes,
        "budget_ratio": spec.budget_ratio,
        "mean_beta_min": min(mean_betas) if mean_betas else 0.0,
        "mean_beta_max": max(mean_betas) if mean_betas else 0.0,
        "flops_ratio": fr,
        "elapsed_seconds": elapsed,
    }

    logger.info(
        "[HAP calib] done: %d layers x %d heads, %d prompts, %.1fs",
        plan.num_layers, plan.num_heads, num_prompts, elapsed,
    )

    return plan_dict, summary


def format_summary(summary: Dict) -> str:
    """Format the summary dict as a human-readable string."""
    lines = [
        "HAP Calibration Summary",
        f"  layers={summary['num_layers']} heads={summary['num_heads']} "
        f"seq_len={summary['seq_len']}",
        f"  prompts={summary['num_prompts']} scopes={summary['num_scopes']} "
        f"budget={summary['budget_ratio']:.2f}",
        f"  per-layer mean beta: min={summary['mean_beta_min']:.3f} "
        f"max={summary['mean_beta_max']:.3f}",
    ]
    if summary.get("flops_ratio") is not None:
        lines.append(
            f"  expected retained flops ratio: {summary['flops_ratio']:.3f}"
        )
    lines.append(f"  elapsed: {summary['elapsed_seconds']:.1f}s")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# P4 — Plan persistence
# ---------------------------------------------------------------------------

def resolve_output_dir() -> str:
    """Resolve the output directory for calibration artifacts.

    Uses ComfyUI's ``folder_paths.get_output_directory()`` when available;
    falls back to ``<pack_root>/tmp`` with a warning.
    """
    try:
        import folder_paths
        return folder_paths.get_output_directory()
    except (ImportError, Exception):  # degrade: output-dir fallback
        fallback = os.path.join(_PACK_ROOT, "tmp")
        logger.warning(
            "HAP calib: folder_paths unavailable; writing to fallback %r",
            fallback,
        )
        return fallback


def write_scope_plan(
    plan_dict: Dict[str, List[List[float]]],
    output_dir: str,
    name: str = "scope_plan_calibrated.json",
) -> str:
    """Write the scope plan JSON and validate the round-trip.

    Args:
        plan_dict: reference-format ``{"alphas", "betas"}`` dict.
        output_dir: directory to write into (created if needed).
        name: file name.

    Returns:
        Absolute path of the written file.

    Raises:
        ValueError: if ``name`` contains path separators or the round-trip
            validation fails.
    """
    from .hap import ScopePlan

    if os.sep in name or "/" in name or "\\" in name:
        raise ValueError(
            f"write_scope_plan: output_name must not contain path "
            f"separators, got {name!r}"
        )
    if not name.endswith(".json"):
        name += ".json"

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(plan_dict, fh)

    # Validate round-trip.
    ScopePlan.load(path)
    return os.path.abspath(path)


# ---------------------------------------------------------------------------
# P5 — The HAPCalibrate node
# ---------------------------------------------------------------------------

def _define_hap_calibrate_schema():
    """Build the node schema (separated for testability)."""
    return io.Schema(
        node_id="HAPCalibrate",
        display_name="HAP Calibrate (HRDiT)",
        category="model_patches/position_encoding",
        description=(
            "Calibrates a per-head HAP scope plan for this model + resolution. "
            "Runs one denoising step per calibration prompt, collects attention "
            "gradients, and solves the scope-selection knapsack. Output links "
            "directly into the HAP node. Calibrate once per model+resolution; "
            "reuse the plan for all subsequent generations."
        ),
        inputs=[
            io.Model.Input(
                "model",
                tooltip=(
                    "The model to calibrate. Must be the SAME model + "
                    "resolution you will run HAP on. Do NOT connect a "
                    "HAP-patched model (calibrate unpruned)."
                ),
            ),
            io.Conditioning.Input(
                "positive",
                tooltip=(
                    "Positive conditioning (from a CLIP Text Encode node). "
                    "Keep it representative of your typical prompts. The "
                    "same conditioning is used for every calibration prompt; "
                    "each prompt index varies the noise seed."
                ),
            ),
            io.Conditioning.Input(
                "negative",
                tooltip="Negative conditioning (from a CLIP Text Encode node).",
            ),
            io.Int.Input(
                "width",
                default=1024, min=256, max=8192, step=8,
                tooltip=(
                    "Calibration resolution width. Calibrate at <= 2K "
                    "(memory); reuse the plan at higher resolutions."
                ),
            ),
            io.Int.Input(
                "height",
                default=1024, min=256, max=8192, step=8,
                tooltip=(
                    "Calibration resolution height. Calibrate at <= 2K "
                    "(memory); reuse the plan at higher resolutions."
                ),
            ),
            io.String.Input(
                "prompts",
                default="",
                multiline=True,
                optional=True,
                tooltip=(
                    "Calibration prompts, one per line. Empty = built-in "
                    "default list (5 prompts). Paper uses 30."
                ),
            ),
            io.String.Input(
                "prompts_file",
                default="",
                optional=True,
                tooltip=(
                    "Optional text file with one prompt per line (overrides "
                    "the 'prompts' input). Relative paths resolve against "
                    "the ComfyUI-DyPE folder."
                ),
            ),
            io.Int.Input(
                "num_prompts",
                default=5, min=1, max=64, step=1,
                tooltip=(
                    "Number of calibration prompts to actually run (first N "
                    "of the list). Paper: 30. More prompts = better averaging "
                    "but longer calibration."
                ),
            ),
            io.Int.Input(
                "num_scopes",
                default=50, min=2, max=200, step=1,
                tooltip=(
                    "Candidate scopes N_scope. Paper: 50. More scopes = "
                    "finer granularity but slower solver."
                ),
            ),
            io.Float.Input(
                "budget_ratio",
                default=0.10, min=0.01, max=1.0, step=0.01,
                tooltip=(
                    "Attention cost ratio r_c (fraction of full attention "
                    "compute retained). Paper: 0.1. Lower = faster but more "
                    "pruning."
                ),
            ),
            io.Int.Input(
                "bins",
                default=4000, min=100, max=20000, step=100,
                tooltip="Knapsack discretization resolution.",
            ),
            io.Int.Input(
                "chunk",
                default=256, min=1, max=4096, step=1,
                tooltip=(
                    "Query rows per calibration chunk (memory knob; "
                    "result-invariant). Lower = less VRAM but slower."
                ),
            ),
            io.Int.Input(
                "text_len",
                default=512, min=0, max=4096, step=1,
                tooltip=(
                    "Number of leading text tokens (never pruned). "
                    "512 = FLUX convention."
                ),
            ),
            io.Int.Input(
                "anchor_stride",
                default=32, min=0, max=1024, step=1,
                tooltip=(
                    "Global anchor blocks in the cost model. 32 = HRDiT "
                    "default. 0 = off."
                ),
            ),
            io.Float.Input(
                "calib_sigma",
                default=1.0, min=0.0, max=1.0, step=0.01,
                tooltip=(
                    "Denoising sigma at which the single calibration step "
                    "runs. 1.0 = first step (max noise). Lower values probe "
                    "mid-trajectory behaviour."
                ),
            ),
            io.Int.Input(
                "seed",
                default=3407, min=0, max=2**31, step=1,
                tooltip="Noise seed base (prompt i uses seed + i).",
            ),
            io.Combo.Input(
                "loss_type",
                options=list(LOSS_TYPES),
                default="output_norm",
                tooltip=(
                    "Calibration loss. 'output_norm' = MSE of the denoised "
                    "prediction vs zero (no external data needed). "
                    "'reference_mse' = MSE vs a reference latent (connect "
                    "the reference_latent input)."
                ),
            ),
            io.Latent.Input(
                "reference_latent",
                optional=True,
                tooltip=(
                    "Target latent for 'reference_mse' loss (e.g. an encoded "
                    "real image). Only needed when loss_type='reference_mse'."
                ),
            ),
            io.String.Input(
                "output_name",
                default="scope_plan_calibrated.json",
                optional=True,
                tooltip=(
                    "JSON file name inside <output>/dype_hap/. The written "
                    "file is also loadable by the HAP node's scope_plan_path."
                ),
            ),
            io.Boolean.Input(
                "purge_between_prompts",
                default=False,
                optional=True,
                tooltip=(
                    "Run gc + VRAM cache purge between calibration prompts "
                    "(low-VRAM cards). Slower; results identical."
                ),
            ),
            io.Boolean.Input(
                "run",
                default=True,
                label_on="Run Calibration",
                label_off="Skip (dry wiring)",
                tooltip=(
                    "Master switch. False = return empty plan without "
                    "running (for safe graph wiring)."
                ),
            ),
        ],
        outputs=[
            io.Custom("SCOPE_PLAN").Output(
                display_name="scope_plan",
                tooltip=(
                    "Validated scope plan dict — link into the HAP node's "
                    "scope_plan input."
                ),
            ),
            io.String.Output(
                display_name="plan_path",
                tooltip=(
                    "Absolute path of the written JSON (also usable in "
                    "HAP's scope_plan_path)."
                ),
            ),
            io.String.Output(
                display_name="summary",
                tooltip="Human-readable calibration report.",
            ),
        ],
    )


# Base class is conditional so the module imports even without ``comfy_api``
# (standalone CLI).  Method bodies reference ``io`` only at call time, which
# only happens inside ComfyUI / the pytest mock where ``io`` is available.
_ComfyNodeBase = io.ComfyNode if io is not None else object


class HAPCalibrate(_ComfyNodeBase):
    """HAP Calibrate (HRDiT) — in-graph scope-plan calibration node."""

    @classmethod
    def define_schema(cls):
        return _define_hap_calibrate_schema()

    @classmethod
    def execute(
        cls,
        model,
        positive,
        negative,
        width: int = 1024,
        height: int = 1024,
        prompts: str = "",
        prompts_file: str = "",
        num_prompts: int = 5,
        num_scopes: int = 50,
        budget_ratio: float = 0.10,
        bins: int = 4000,
        chunk: int = 256,
        text_len: int = 512,
        anchor_stride: int = 32,
        calib_sigma: float = 1.0,
        seed: int = 3407,
        loss_type: str = "output_norm",
        reference_latent=None,
        output_name: str = "scope_plan_calibrated.json",
        purge_between_prompts: bool = False,
        run: bool = True,
    ):
        # Master switch.
        if not run:
            return io.NodeOutput({}, "", "calibration skipped (run=False)")

        # Resolve prompts.
        prompt_list = resolve_prompts(prompts, prompts_file, num_prompts)

        # Extract reference tensor if provided.
        ref_tensor = None
        if reference_latent is not None:
            if isinstance(reference_latent, dict):
                ref_tensor = reference_latent.get("samples")
            else:
                ref_tensor = reference_latent

        # Build spec.
        spec = CalibrationSpec(
            width=int(width),
            height=int(height),
            num_prompts=int(num_prompts),
            num_scopes=int(num_scopes),
            budget_ratio=float(budget_ratio),
            bins=int(bins),
            chunk=int(chunk),
            text_len=int(text_len),
            anchor_stride=int(anchor_stride),
            calib_sigma=float(calib_sigma),
            seed=int(seed),
            loss_type=str(loss_type),
            prompts=prompt_list,
            reference_latent=ref_tensor,
            purge_between_prompts=bool(purge_between_prompts),
        )

        try:
            plan_dict, summary = run_hap_calibration(
                model=model,
                spec=spec,
                model_type="auto",
                positive=positive,
                negative=negative,
            )
        except (ValueError, RuntimeError) as exc:
            raise type(exc)(f"HAP Calibrate: {exc}") from exc
        finally:
            # NODE-LEVEL MEMORY CLEANUP (plan 2026-08-24 P3): runs on success
            # AND failure.  gc.collect() first breaks autograd reference cycles
            # so the transient tensors actually die; the purge then returns the
            # cached segments to the driver via ComfyUI's public API.  Without
            # this, stale reserved VRAM makes aimdo keep the model weights in
            # RAM for the next workflow run (the "model reloads into RAM"
            # symptom).  We deliberately do NOT detach/unload the patcher —
            # the same ModelPatcher flows on to inference and must stay
            # registered with ComfyUI's manager.
            import gc as _gc

            _gc.collect()
            _purge_calibration_memory()
            if torch.cuda.is_available():
                logger.info(
                    "HAP calib: memory purged (cache returned to driver)"
                )

        # Write JSON.
        out_dir = os.path.join(resolve_output_dir(), "dype_hap")
        name = output_name if output_name else "scope_plan_calibrated.json"
        path = write_scope_plan(plan_dict, out_dir, name)

        return io.NodeOutput(plan_dict, path, format_summary(summary))
