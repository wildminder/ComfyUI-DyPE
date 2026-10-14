"""Legacy tensor and single-owner attention container regression tests."""
import sys

import pytest
import torch
import torch.nn.functional as F

from nodes.hap_calibrate import collect_scope_scores_for_model
from src.hap import HapContext, HapRuntime, ScopePlan
from src.hap_calib import collect_scope_scores
from src.spa import _make_hrdit_wrapper
from src.spa_context import (
    SPAContext, set_hap_context, set_hap_layer_idx, set_hrdit_layer_idx, set_spa_context,
    set_spa_step_gate,
)
from _spa_math_helpers import random_variants


class AttentionTensorContainer:
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


@pytest.fixture(autouse=True)
def reset_contexts():
    set_hrdit_layer_idx(0)
    HapRuntime.reset()
    yield
    set_spa_context(None)
    set_hap_context(None)
    set_spa_step_gate(True)
    set_hrdit_layer_idx(0)
    HapRuntime.reset()


@pytest.mark.parametrize("mode", ["plain", "masked", "cross", "gated", "spa", "hap", "spa_hap"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("flatten_output", [False, True])
def test_container_attention_matches_tensors(mode, dtype, flatten_output):
    generator = torch.Generator().manual_seed(21)
    qkv = [torch.randn(1, 2, 16, 8, generator=generator, dtype=dtype) for _ in range(3)]
    if mode == "cross":
        qkv[1:] = [torch.randn(1, 2, 20, 8, generator=generator, dtype=dtype) for _ in range(2)]
    passes = []
    preferred = object()
    options = {"test": True}

    def original(q, k, v, heads, mask, precision, skip_reshape, skip_output_reshape, **kwargs):
        passes.append(isinstance(q, AttentionTensorContainer))
        if isinstance(q, AttentionTensorContainer):
            q, k, v = q.take(), k.take(), v.take()
        assert kwargs["preferred_attention"] is preferred
        assert kwargs["transformer_options"] is options
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return out if skip_output_reshape else out.transpose(1, 2).flatten(2)

    if "spa" in mode or mode == "gated":
        variants = random_variants(16, 8, 3, 5)
        set_spa_context(SPAContext(active=True, base_pe=variants[0], variant_pes=variants))
    if "hap" in mode:
        plan = ScopePlan(alphas=[[64.0, 64.0]], betas=[[0.0, 0.0]])
        set_hap_context(HapContext(active=True, plan=plan, text_len=0, backend="dense"))
    set_spa_step_gate(mode != "gated")
    wrapper = _make_hrdit_wrapper(original, False)
    mask = torch.ones(16, 16, dtype=torch.bool) if mode == "masked" else None
    kwargs = dict(mask=mask, skip_reshape=True, skip_output_reshape=not flatten_output,
                  preferred_attention=preferred, transformer_options=options)
    expected = wrapper(*qkv, 2, **kwargs)
    legacy_passes = list(passes)
    passes.clear()
    set_hrdit_layer_idx(0)
    set_hap_layer_idx(0)
    containers = [AttentionTensorContainer(tensor) for tensor in qkv]
    actual = wrapper(*containers, 2, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.dtype == dtype
    assert actual.device == qkv[0].device
    assert all(container.tensor is None for container in containers)
    assert passes == [True] * len(legacy_passes)
    assert len(passes) == ({"plain": 1, "masked": 1, "cross": 1, "gated": 1, "spa": 3}.get(mode, 0))


@pytest.mark.parametrize("collector,flatten_input", [("core", False), ("node", False), ("node", True)])
def test_container_calibration_matches_tensors(collector, flatten_input):
    attention = sys.modules["comfy.ldm.modules.attention"]
    original = attention.optimized_attention
    generator = torch.Generator().manual_seed(12)
    qkv = [torch.randn(1, 2, 16, 8, generator=generator) for _ in range(3)]
    if flatten_input:
        qkv = [tensor.transpose(1, 2).flatten(2) for tensor in qkv]
    consumed = []

    def collect(use_containers):
        def forward():
            inputs = qkv
            if use_containers:
                inputs = [AttentionTensorContainer(tensor) for tensor in qkv]
                consumed.extend(inputs)
            return attention.optimized_attention(*inputs, 2, skip_reshape=not flatten_input)
        if collector == "core":
            return collect_scope_scores(forward, lambda out: out.square().mean(),
                                        num_scopes=3, chunk=5, attn_module=attention)
        return collect_scope_scores_for_model(None, "auto", forward,
                                             lambda out: out.square().mean(),
                                             num_scopes=3, chunk=5, scale=1.0)

    expected = collect(False)
    actual = collect(True)
    for result, reference in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
        assert torch.isfinite(result).all()
    assert all(container.tensor is None for container in consumed)
    assert attention.optimized_attention is original
