#!/usr/bin/env python
"""HAP scope-plan calibration script (plan T6.3).

One-command generation of a per-head HAP scope plan for a target model +
resolution.  Runs the P6 gradient-path collector (T6.2) over a set of
calibration prompts, solves the multiple-choice knapsack (T5.5), and writes a
reference-format ``{"alphas", "betas"}`` JSON consumable by the ``HAP`` node
with zero conversion.

Two modes:

``--dry_run``
    CI-safe end-to-end pipeline on a synthetic toy DiT (no GPU model, no real
    ComfyUI weights).  Validates collector → solver → JSON round-trip.  This is
    the only path exercised by automated tests (``-k dry_run``).

real model (default)
    Loads a checkpoint through ComfyUI and calibrates at ONE representative
    denoising step per prompt (paper: impact estimation on attention probs).
    This path requires the user's ComfyUI venv and a GPU; it is a documented
    manual step (plan §6 acceptance, checklist A5) and is NOT run in CI.

Usage (dry run)::

    python calibration/calibrate_hap.py --dry_run --out tmp/scope_plan_dry.json

Usage (real model, in the ComfyUI venv)::

    python calibration/calibrate_hap.py \
        --model_path /path/to/flux.safetensors --model_type flux \
        --width 4096 --height 4096 --num_prompts 30 --out configs/scope_plan_flux_4k.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the project root importable when run as a script.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch

from src.hap import ScopePlan, flops_ratio
from src.hap_calib import calibrate_scope_plan, collect_scope_scores

# Single source for default prompts (plan 2026-08-16 P6.1): the node module
# owns the list; the CLI re-imports it so the two never drift.
# Re-exported on purpose — tests/test_hap_calib_spec.py::TestCliSingleSource
# asserts ``mod.DEFAULT_PROMPTS is DEFAULT_CALIBRATION_PROMPTS``.
from src.hap_calib_node import DEFAULT_CALIBRATION_PROMPTS as DEFAULT_PROMPTS  # noqa: F401


def _ensure_mock_attention_module():
    """Install a minimal ``comfy.ldm.modules.attention`` mock for ``--dry_run``.

    The P6 collector patches the module-level ``optimized_attention``.  In the
    user's ComfyUI venv the real module exists; in CI / standalone dry-run it
    does not, so we provide a plain SDPA shim (scale=1.0, matching the HRDiT
    reference and the test conftest) to keep the dry-run fully self-contained.
    """
    import types

    if "comfy.ldm.modules.attention" in sys.modules:
        return sys.modules["comfy.ldm.modules.attention"]

    import torch.nn.functional as F

    for name in ("comfy", "comfy.ldm", "comfy.ldm.modules"):
        sys.modules.setdefault(name, types.ModuleType(name))
    attn_mod = types.ModuleType("comfy.ldm.modules.attention")

    # REAL ComfyUI signature (attention.py::attention_pytorch) — positional slots
    # 5-8 == mask, attn_precision, skip_reshape, skip_output_reshape.  Must mirror
    # the test conftest mock (plan 2026-08-16 G2 mock-fidelity fix).
    def _sdpa(q, k, v, heads, mask=None, attn_precision=None,
              skip_reshape=False, skip_output_reshape=False, **kwargs):
        return F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=0.0, is_causal=False)

    attn_mod.optimized_attention = _sdpa
    attn_mod.optimized_attention_masked = _sdpa  # real alias (attention.py:883)
    sys.modules["comfy.ldm.modules.attention"] = attn_mod
    sys.modules["comfy.ldm.modules"].attention = attn_mod
    return attn_mod


# ---------------------------------------------------------------------------
# Dry-run toy model (self-contained; mirrors tests/_hrdit_fixtures.ToyDiT)
# ---------------------------------------------------------------------------

class _DryRunToyDiT:
    """Minimal multi-layer attention-only transformer for ``--dry_run``.

    Calls the module-level ``optimized_attention`` once per layer so the P6
    collector can observe a deterministic per-forward call sequence.  Uses fp64
    for clean calibration gradients.
    """

    def __init__(self, num_layers=4, heads=4, dim=16, text_len=16, img_hw=8, seed=0):
        self.num_layers = num_layers
        self.heads = heads
        self.dim = dim
        self.text_len = text_len
        self.img_hw = img_hw
        self.seed = seed
        self.seq_len = text_len + img_hw * img_hw

    def forward(self):
        attn_mod = sys.modules["comfy.ldm.modules.attention"]
        x = torch.zeros(1, self.seq_len, self.heads * self.dim, dtype=torch.float64)
        h = x
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(self.num_layers):
            q = torch.randn(1, self.heads, self.seq_len, self.dim, generator=g, dtype=torch.float64)
            k = torch.randn(1, self.heads, self.seq_len, self.dim, generator=g, dtype=torch.float64)
            v = torch.randn(1, self.heads, self.seq_len, self.dim, generator=g, dtype=torch.float64)
            out = attn_mod.optimized_attention(q, k, v, self.heads)
            h = h + out.reshape(1, self.seq_len, self.heads * self.dim)
        return h


def _dry_run_loss_fn(dit):
    g = torch.Generator().manual_seed(999)
    target = torch.randn(1, dit.seq_len, dit.heads * dit.dim, generator=g, dtype=torch.float64)

    def loss_fn(output):
        return torch.nn.functional.mse_loss(output, target)

    return loss_fn


def run_dry_run(args) -> dict:
    """End-to-end pipeline on the toy model; returns the plan dict."""
    _ensure_mock_attention_module()
    # img_hw=16 -> 256 image tokens = 4 blocks of 64 (nbx=4), so the candidate
    # scopes map to genuinely different band widths and the knapsack does real
    # work.  Still tiny enough to run on CPU in well under a second.
    dit = _DryRunToyDiT(
        num_layers=args.dry_layers, heads=args.dry_heads, dim=16,
        text_len=16, img_hw=16, seed=0,
    )
    num_scopes = args.num_scopes
    text_len = dit.text_len

    # One calibration "prompt" per seed (toy: re-seed the toy per prompt).
    quality_per_layer = [[] for _ in range(dit.num_layers)]
    compute = None
    for p in range(args.num_prompts):
        toy = _DryRunToyDiT(
            num_layers=args.dry_layers, heads=args.dry_heads, dim=16,
            text_len=16, img_hw=16, seed=p,
        )
        quality, comp = collect_scope_scores(
            toy.forward, _dry_run_loss_fn(toy), num_scopes,
            text_len=text_len, chunk=args.chunk, scale=1.0,
        )
        for l in range(dit.num_layers):
            quality_per_layer[l].append(quality[l])
        if compute is None:
            compute = comp[0]  # prompt-independent (H, S)

    plan_dict = calibrate_scope_plan(
        quality_per_layer, [compute] * dit.num_layers,
        budget_ratio=args.budget_ratio, bins=args.bins,
    )
    return plan_dict


# ---------------------------------------------------------------------------
# Real-model calibration (user's ComfyUI venv; manual step, checklist A5)
# ---------------------------------------------------------------------------

def _encode_conditioning(clip, prompt: str):
    """Build a ComfyUI CONDITIONING list for ``prompt`` using ``clip``.

    Mirrors the CLIP Text Encode node: tokenize, encode, and wrap the pooled
    output so ``comfy.samplers.sampling_function`` receives a valid cond.
    """
    tokens = clip.tokenize(prompt)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {"pooled_output": pooled}]]


def run_real(args) -> dict:  # pragma: no cover - requires ComfyUI venv + GPU
    """Calibrate a real checkpoint.  Requires the ComfyUI runtime + a GPU.

    This is a documented manual step (plan §6, checklist A5) and is not run in
    CI.  It loads the checkpoint via ComfyUI's standard loader, builds
    conditioning from the calibration prompts, then delegates to the SAME
    orchestrator the in-graph node uses (:func:`src.hap_calib_node.
    run_hap_calibration`) so the CLI and the node can never drift.
    """
    try:
        import comfy.sd
        import comfy.utils  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            "Real-model calibration requires the ComfyUI runtime.  Run this "
            "script inside your ComfyUI venv (checklist A5), or use --dry_run "
            "to validate the pipeline without a model.  Import failed: "
            f"{exc!r}"
        ) from exc

    from src.hap_calib_node import (
        CalibrationSpec,
        resolve_prompts,
        run_hap_calibration,
    )

    if not args.model_path:
        raise ValueError(
            "run_real: --model_path is required for real-model calibration "
            "(use --dry_run to validate the pipeline without a checkpoint)."
        )

    # Load the checkpoint (model + clip) through ComfyUI's canonical loader.
    model, clip, _vae, *_rest = comfy.sd.load_checkpoint_guess_config(
        args.model_path,
        output_vae=False,
        output_clip=True,
        output_clipvision=False,
        output_model=True,
    )

    # Resolve the calibration prompt list (file > defaults).
    prompts = resolve_prompts(
        prompts_text="",
        prompts_file=args.prompts_file or "",
        num_prompts=args.num_prompts,
        pack_root=_PROJECT_ROOT,
    )

    # Condition on the FIRST prompt; each prompt index varies the noise seed
    # inside the orchestrator (matches the node's behaviour).
    positive = _encode_conditioning(clip, prompts[0])
    negative = _encode_conditioning(clip, "")

    spec = CalibrationSpec(
        width=int(args.width),
        height=int(args.height),
        num_prompts=int(args.num_prompts),
        num_scopes=int(args.num_scopes),
        budget_ratio=float(args.budget_ratio),
        bins=int(args.bins),
        chunk=int(args.chunk),
        text_len=512,
        anchor_stride=int(args.anchor_stride),
        calib_sigma=1.0,
        seed=3407,
        loss_type="output_norm",
        prompts=prompts,
    )
    spec.validate()

    plan_dict, _summary = run_hap_calibration(
        model=model,
        spec=spec,
        model_type=args.model_type,
        positive=positive,
        negative=negative,
    )
    return plan_dict


# ---------------------------------------------------------------------------
# Summary + CLI
# ---------------------------------------------------------------------------

def summarize(plan_dict: dict, seq_len: int, text_len: int, anchor_stride: int) -> str:
    """Human-readable summary: per-layer mean beta + expected flops ratio."""
    plan = ScopePlan.from_dict(plan_dict)
    lines = [f"layers={plan.num_layers} heads={plan.num_heads} seq_len={seq_len}"]
    mean_betas = []
    for l in range(plan.num_layers):
        mb = sum(plan_dict["betas"][l]) / max(len(plan_dict["betas"][l]), 1)
        mean_betas.append(mb)
    lines.append(f"per-layer mean beta: min={min(mean_betas):.3f} max={max(mean_betas):.3f}")
    try:
        fr = flops_ratio(plan, seq_len, text_len=text_len, anchor_stride=anchor_stride)
        lines.append(f"expected retained flops ratio: {fr:.3f}")
    except Exception as exc:  # pragma: no cover - defensive
        lines.append(f"flops ratio unavailable: {exc!r}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Calibrate an HAP scope plan.")
    p.add_argument("--model_path", type=str, default=None, help="Checkpoint path (real mode).")
    p.add_argument("--model_type", type=str, default="flux",
                   choices=["flux", "qwen", "zimage", "anima", "nunchaku"])
    p.add_argument("--width", type=int, default=4096)
    p.add_argument("--height", type=int, default=4096)
    p.add_argument("--num_prompts", type=int, default=30)
    p.add_argument("--num_scopes", type=int, default=50)
    p.add_argument("--budget_ratio", type=float, default=0.1)
    p.add_argument("--anchor_stride", type=int, default=0)
    p.add_argument("--bins", type=int, default=4000)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--prompts_file", type=str, default=None,
                   help="Optional text file with one prompt per line.")
    p.add_argument("--out", type=str, default="configs/scope_plan_calibrated.json")
    p.add_argument("--dry_run", action="store_true",
                   help="Run the pipeline on a synthetic toy model (CI-safe).")
    p.add_argument("--dry_layers", type=int, default=4)
    p.add_argument("--dry_heads", type=int, default=4)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.dry_run:
        plan_dict = run_dry_run(args)
        toy_seq = 16 + 16 * 16  # matches _DryRunToyDiT dry-run defaults
        print(summarize(plan_dict, seq_len=toy_seq, text_len=16, anchor_stride=0))
    else:
        plan_dict = run_real(args)
        seq_len = (args.width // 16) * (args.height // 16) + 512
        print(summarize(plan_dict, seq_len=seq_len, text_len=512,
                        anchor_stride=args.anchor_stride))

    out_path = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(plan_dict, fh)
    # Validate the written plan round-trips.
    ScopePlan.load(out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
