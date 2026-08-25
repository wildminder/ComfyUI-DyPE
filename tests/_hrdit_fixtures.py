"""Shared fixtures for HRDiT (HAP) tests.

Plan T0.2: a synthetic multi-layer DiT that calls the patched
``comfy.ldm.modules.attention.optimized_attention`` once per layer in a fixed
order, so the per-forward attention-call counter (layer index) is observable
and deterministic.

The toy model mirrors the FLUX-family call pattern: one attention call per
transformer block, double blocks first then single blocks (our flat layer
counter order == reference ``iter_blocks`` order).

Plan 2026-08-25 W2.1 (tracker CRIT-002 / IMP-108): this module is now ALSO
the single source of truth for hand-rolled ``orig`` attention mocks.
``make_recording_orig`` builds a fake with the EXACT real ComfyUI signature
(locked by ``assert_real_signature`` at construction time) — per-test local
mocks with ad-hoc signatures caused three rounds of stale-mock rot (the
v2.7.1 Anima crash class). New tests MUST use this factory instead of
defining their own fakes.
"""

from __future__ import annotations

import inspect
import sys
from typing import List, Optional

import torch
import torch.nn.functional as F

# Canonical REAL signature of comfy/ldm/modules/attention.py::attention_pytorch
# (positional slots 1-8, locked by tests/test_orig_call_convention.py).
CANONICAL_ATTN_PARAMS = (
    "q", "k", "v", "heads", "mask", "attn_precision",
    "skip_reshape", "skip_output_reshape",
)


def assert_real_signature(fn) -> None:
    """Raise AssertionError unless ``fn``'s first 8 params match the canonical
    real ``optimized_attention`` order exactly.

    Called inside :func:`make_recording_orig` so signature drift fails at
    FIXTURE CONSTRUCTION, not mid-test with a confusing TypeError.
    """
    sig = inspect.signature(fn)
    names = tuple(sig.parameters)[0:8]
    if names != CANONICAL_ATTN_PARAMS:
        raise AssertionError(
            "Mock attention signature drifted from the real ComfyUI "
            f"convention.\n  expected: {CANONICAL_ATTN_PARAMS}\n"
            f"  got     : {names}\nUse make_recording_orig() from "
            "_hrdit_fixtures instead of hand-rolled fakes."
        )


def make_recording_orig(scale: float = 1.0, record: Optional[List[tuple]] = None):
    """Build an ``orig`` attention fake with the EXACT real signature.

    Mirrors ``attention_pytorch``: touches ``mask.ndim`` when a mask is given,
    runs plain SDPA at ``scale``, and optionally records every call's first 8
    positional slots into ``record`` for wrapper-forwarding assertions.

    Usage::

        calls = []
        orig = make_recording_orig(record=calls)
        ... drive the wrapper ...
        assert calls[0][4] is my_mask          # slot 5 == mask
        assert calls[0][7] is False            # slot 8 == skip_output_reshape
    """
    def recording_orig(q, k, v, heads, mask=None, attn_precision=None,
                       skip_reshape=False, skip_output_reshape=False, **kwargs):
        if mask is not None:
            _ = mask.ndim  # the real fn reads mask.ndim; keep parity
        if record is not None:
            record.append((q, k, v, heads, mask, attn_precision,
                           skip_reshape, skip_output_reshape))
        return F.scaled_dot_product_attention(q, k, v, scale=scale)

    assert_real_signature(recording_orig)
    return recording_orig


def _attention_module():
    """Return the (mock) ``comfy.ldm.modules.attention`` module."""
    return sys.modules["comfy.ldm.modules.attention"]


class ToyDiT:
    """Deterministic multi-layer attention-only transformer.

    Each layer performs ONE call to the patched ``optimized_attention`` with
    fresh (seeded) q/k/v tensors of shape ``(1, heads, seq_len, dim)`` and adds
    the result to the running hidden state, so every layer's call is observable
    and the forward is a pure function of the seed.

    Attributes:
        num_layers: number of attention calls per forward.
        heads: attention heads per call.
        dim: head dimension.
        text_len: leading text tokens (token layout: text then image).
        img_hw: image grid side; image tokens = img_hw * img_hw.
        call_log: list of per-call records (dicts) from the last forward.
    """

    def __init__(
        self,
        num_layers: int = 4,
        heads: int = 2,
        dim: int = 16,
        text_len: int = 8,
        img_hw: int = 4,
        seed: int = 0,
        dtype: Optional[torch.dtype] = None,
    ):
        self.num_layers = num_layers
        self.heads = heads
        self.dim = dim
        self.text_len = text_len
        self.img_hw = img_hw
        self.seed = seed
        self.dtype = dtype  # None -> torch default (fp32); set fp64 for calib tests
        self.seq_len = text_len + img_hw * img_hw
        self.call_log: List[dict] = []

    def forward(self, x: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Run ``num_layers`` attention calls; returns the final hidden state.

        ``x`` is optional; when omitted a seeded zero-state of shape
        ``(1, seq_len, heads * dim)`` is used. The return value is the
        accumulated hidden state (same shape).
        """
        attn_mod = _attention_module()
        self.call_log = []
        if x is None:
            x = torch.zeros(1, self.seq_len, self.heads * self.dim, dtype=self.dtype)
        h = x
        g = torch.Generator().manual_seed(self.seed)
        for layer in range(self.num_layers):
            q = torch.randn(1, self.heads, self.seq_len, self.dim, generator=g, dtype=self.dtype)
            k = torch.randn(1, self.heads, self.seq_len, self.dim, generator=g, dtype=self.dtype)
            v = torch.randn(1, self.heads, self.seq_len, self.dim, generator=g, dtype=self.dtype)
            out = attn_mod.optimized_attention(q, k, v, self.heads)
            self.call_log.append(
                {
                    "layer": layer,
                    "q_shape": tuple(q.shape),
                    "seq_len": self.seq_len,
                    "text_len": self.text_len,
                }
            )
            # Fold the attention output back so every layer influences the
            # result (keeps the forward non-trivial without parameters).
            h = h + out.reshape(1, self.seq_len, self.heads * self.dim)
        return h

    __call__ = forward


def make_toy_dit(
    num_layers: int = 4,
    heads: int = 2,
    dim: int = 16,
    text_len: int = 8,
    img_hw: int = 4,
    seed: int = 0,
    dtype: Optional[torch.dtype] = None,
) -> ToyDiT:
    """Factory for :class:`ToyDiT` (plan T0.2 signature).

    ``dtype`` (optional) sets the activation dtype; ``None`` keeps the torch
    default (fp32).  Calibration collector tests pass ``torch.float64``.
    """
    return ToyDiT(
        num_layers=num_layers,
        heads=heads,
        dim=dim,
        text_len=text_len,
        img_hw=img_hw,
        seed=seed,
        dtype=dtype,
    )


class CallRecorder:
    """Wrap the patched ``optimized_attention`` to record every call.

    Usage::

        rec = CallRecorder.install()   # wraps current optimized_attention
        ... run model ...
        rec.calls                      # list of (q, k, v, heads) arg tuples
        rec.uninstall()
    """

    def __init__(self):
        self.calls: List[tuple] = []
        self._orig = None

    def install(self) -> "CallRecorder":
        attn_mod = _attention_module()
        self._orig = attn_mod.optimized_attention
        orig = self._orig
        calls = self.calls

        def recording(q, k, v, heads, *args, **kwargs):
            calls.append((q, k, v, heads))
            return orig(q, k, v, heads, *args, **kwargs)

        attn_mod.optimized_attention = recording
        return self

    def uninstall(self):
        if self._orig is not None:
            _attention_module().optimized_attention = self._orig
            self._orig = None
