"""P0 — Root-cause characterization & numerical lock-in.

Proves (pure torch, no ComfyUI) that averaging RoPE *rotation matrices* then a
single softmax is mathematically wrong, while averaging the per-variant attention
outputs is correct.  This is the permanent regression anchor for the fix.

Markers: @pytest.mark.unit
"""
import pytest
import torch

from src.spa import SPA_MAX_PASSES, build_bundle_id_variants
from src.spa_attn import apply_rope_matrix

try:
    from tests._spa_math_helpers import random_variants
except ImportError:  # namespace-package import fallback
    from _spa_math_helpers import random_variants


def _qkv(L=256, H=4, D=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, H, L, D, generator=g)
    k = torch.randn(1, H, L, D, generator=g)
    v = torch.randn(1, H, L, D, generator=g)
    return q, k, v


def _buggy_vs_correct(q, k, v, rotations):
    """buggy = single softmax on averaged-R; correct = avg of per-variant softmaxs."""
    D = q.shape[-1]
    scale = D ** 0.5
    R_avg = torch.stack(rotations, 0).mean(0)
    q_avg = apply_rope_matrix(q, R_avg)
    k_avg = apply_rope_matrix(k, R_avg)
    buggy = torch.softmax((q_avg @ k_avg.transpose(-1, -2)) / scale, dim=-1) @ v

    outs = []
    for R in rotations:
        qn = apply_rope_matrix(q, R)
        kn = apply_rope_matrix(k, R)
        wn = torch.softmax((qn @ kn.transpose(-1, -2)) / scale, dim=-1)
        outs.append(wn @ v)
    correct = torch.stack(outs, 0).mean(0)
    return buggy, correct


@pytest.mark.unit
class TestP0RootCause:
    def test_averaged_rotation_not_orthogonal(self):
        for seed in (0, 1, 2, 7):
            for N in (3, 5):
                R = random_variants(256, 64, N, seed)
                R_avg = torch.stack(R, 0).mean(0)
                RtR = R_avg.transpose(-1, -2) @ R_avg
                I = torch.eye(2, dtype=R_avg.dtype, device=R_avg.device)
                frob = (RtR - I).norm("fro")
                assert frob > 0.05, f"seed={seed} N={N} frob={frob.item():.4f}"

    def test_avg_rotation_softmax_differs_from_avg_of_softmaxs(self):
        q, k, v = _qkv(seed=0)
        R = random_variants(256, 64, 5, 0)
        buggy, correct = _buggy_vs_correct(q, k, v, R)
        diff = (buggy - correct).norm()
        assert diff > 1e-3, f"abs diff {diff.item():.4e}"
        assert diff / correct.norm() > 1e-2, f"rel diff {(diff / correct.norm()).item():.4e}"

    def test_identity_bundle_size_one_has_no_gap(self):
        q, k, v = _qkv(seed=1)
        R = random_variants(256, 64, 1, 1)  # single identity-ish variant
        buggy, correct = _buggy_vs_correct(q, k, v, R)
        rel = (buggy - correct).norm() / correct.norm()
        assert rel <= 1e-6, f"rel gap {rel.item():.2e}"

    def test_determinism(self):
        Ra = random_variants(256, 64, 5, 0)
        Rb = random_variants(256, 64, 5, 0)
        for a, b in zip(Ra, Rb):
            assert torch.allclose(a, b, atol=0)
        q, k, v = _qkv(seed=2)
        ba, ca = _buggy_vs_correct(q, k, v, Ra)
        bb, cb = _buggy_vs_correct(q, k, v, Rb)
        assert torch.allclose(ba, bb, atol=0)
        assert torch.allclose(ca, cb, atol=0)

    def test_variant_signature_uniqueness_and_text_untouched(self):
        # Build FLUX-style ids (text token has h=w=0).  Use a grid OUTSIDE the
        # trained extent (128x128, max_pos=127 > 64) so bundling is ACTIVE and
        # the uniqueness property is exercised under real compression.
        H = W = 128
        L = H * W
        ids = torch.zeros(1, L, 3)
        ids[..., 0] = torch.arange(L)
        ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()

        for N in (0, 3, 5):
            variants = build_bundle_id_variants(ids, N)
            nv = len(variants)
            stacked = torch.stack([v[0] for v in variants], dim=0)  # (N, L, 3)
            s_est = (nv + 1) // 2  # inverse of 2*s-1 variants

            # (a) text token untouched across every variant
            for v in variants:
                assert v[0, 0, 1] == 0 and v[0, 0, 2] == 0

            # (b) COST GUARD: the effective bundle size s is capped so the
            # averaged-pass count never exceeds SPA_MAX_PASSES.
            assert s_est <= (SPA_MAX_PASSES + 1) // 2, (
                f"N={N} s={s_est} exceeds pass cap "
                f"(SPA_MAX_PASSES={SPA_MAX_PASSES})"
            )
            # Bundling is active at 128x128 for every knob.
            assert nv > 1, f"N={N} at 128x128 must be active"

            # (c) per-position uniqueness: each of the L positions has a distinct
            # (N, 3) signature across the variant stack (paper SPA
            # spatial-distinguishability property).
            sig = stacked.permute(1, 0, 2).reshape(L, nv * 3)
            assert torch.unique(sig, dim=0).shape[0] == L

    def test_in_trained_extent_is_identity(self):
        # Trained-extent gate: 64x64 (max_pos=63 <= 64) is in-distribution ->
        # identity for every knob (no SPA needed, no big-patch collapse).
        H = W = 64
        L = H * W
        ids = torch.zeros(1, L, 3)
        ids[..., 0] = torch.arange(L)
        ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
        ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
        for N in (0, 3, 5, 8):
            variants = build_bundle_id_variants(ids, N)
            assert len(variants) == 1, f"N={N} at 64x64 must be identity"
            assert torch.equal(variants[0], ids)
