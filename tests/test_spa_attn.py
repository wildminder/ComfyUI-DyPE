"""P1 — SPAContext-independent core math: delta-RoPE + averaged attention.

Pure torch, no ComfyUI.  Verifies ``inv_rope`` / ``compose_rope`` / ``apply_rope_matrix``
for the FLUX / Anima / Nunchaku formats and that ``spa_averaged_attention`` equals
HRDiT's ``reference_spa_attention`` (the equivalence required by the brief).
"""
import pytest
import torch
import torch.nn.functional as F

from src.spa_attn import apply_rope_matrix, compose_rope, inv_rope, spa_averaged_attention

try:
    from tests._spa_math_helpers import (
        angles_to_blocks,
        angles_to_cos_sin,
        random_variants,
        reference_spa_attention,
    )
except ImportError:  # namespace-package import fallback
    from _spa_math_helpers import (
        angles_to_blocks,
        angles_to_cos_sin,
        random_variants,
        reference_spa_attention,
    )


def _attn3(q, k, v):
    # Match HRDiT ``reference_spa_attention`` which uses ``scale=1.0`` (so the
    # P1 equivalence check is exact up to float precision, not a scale mismatch).
    return F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=0.0, is_causal=False)


def _angles(L, P, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(L, P, generator=g) * 0.4


@pytest.mark.unit
class TestP1InvCompose:
    def test_inv_rope_is_inverse_flux(self):
        x = torch.randn(1, 4, 128, 64)
        R = random_variants(128, 64, 3, 0)[0]
        recon = apply_rope_matrix(apply_rope_matrix(x, R), inv_rope(R, "flux"), "flux")
        assert torch.allclose(recon, x, atol=1e-5)

    def test_compose_recovers_variant_flux(self):
        x = torch.randn(1, 4, 128, 64)
        base, var = random_variants(128, 64, 3, 0)[:2]
        via_delta = apply_rope_matrix(apply_rope_matrix(x, base), compose_rope(inv_rope(base, "flux"), var, "flux"), "flux")
        direct = apply_rope_matrix(x, var)
        assert torch.allclose(via_delta, direct, atol=1e-5)

    def test_inv_rope_is_inverse_anima(self):
        x = torch.randn(1, 4, 128, 64)
        R = random_variants(128, 64, 3, 0)[0]  # (128, 32, 2, 2)
        recon = apply_rope_matrix(apply_rope_matrix(x, R, "anima"), inv_rope(R, "anima"), "anima")
        assert torch.allclose(recon, x, atol=1e-5)

    def test_compose_recovers_variant_anima(self):
        x = torch.randn(1, 4, 128, 64)
        base, var = random_variants(128, 64, 3, 0)[:2]
        via_delta = apply_rope_matrix(
            apply_rope_matrix(x, base, "anima"),
            compose_rope(inv_rope(base, "anima"), var, "anima"),
            "anima",
        )
        direct = apply_rope_matrix(x, var, "anima")
        assert torch.allclose(via_delta, direct, atol=1e-5)

    def test_inv_rope_is_inverse_nunchaku(self):
        x = torch.randn(1, 4, 128, 64)
        R = random_variants(128, 64, 3, 0)[0]  # (128, 32, 2, 2)
        pe = _blocks_to_nunchaku(R).unsqueeze(0).unsqueeze(0)  # (1,1,128,32,1,2)
        inv_pe = inv_rope(pe, "nunchaku")
        recon = apply_rope_matrix(apply_rope_matrix(x, pe, "nunchaku"), inv_pe, "nunchaku")
        assert torch.allclose(recon, x, atol=1e-5)

    def test_compose_recovers_variant_nunchaku(self):
        x = torch.randn(1, 4, 128, 64)
        base, var = random_variants(128, 64, 3, 0)[:2]
        base_pe = _blocks_to_nunchaku(base).unsqueeze(0).unsqueeze(0)
        var_pe = _blocks_to_nunchaku(var).unsqueeze(0).unsqueeze(0)
        via_delta = apply_rope_matrix(
            apply_rope_matrix(x, base_pe, "nunchaku"),
            compose_rope(inv_rope(base_pe, "nunchaku"), var_pe, "nunchaku"),
            "nunchaku",
        )
        direct = apply_rope_matrix(x, var_pe, "nunchaku")
        assert torch.allclose(via_delta, direct, atol=1e-5)


def _blocks_to_nunchaku(blocks):
    s = blocks[..., 1, 0]
    c = blocks[..., 0, 0]
    pe = torch.stack([s, c], dim=-1).unsqueeze(-2)
    return pe


@pytest.mark.unit
class TestP1AveragedAttention:
    def test_single_variant_is_identity(self):
        q = torch.randn(1, 4, 64, 64)
        k = torch.randn(1, 4, 64, 64)
        v = torch.randn(1, 4, 64, 64)
        base = random_variants(64, 64, 1, 0)[0]
        single = [random_variants(64, 64, 1, 1)[0]]
        out = spa_averaged_attention(q, k, v, base, single, attn_fn=_attn3, pre_roped=True, fmt="flux")
        assert torch.allclose(out, _attn3(q, k, v), atol=1e-6)

    def test_equivalence_to_hrdit(self):
        for seed in (0, 1, 2, 7):
            for D in (64, 128):
                for L in (64, 256, 1024):
                    for N in (3, 5):
                        H = 4
                        P = D // 2
                        g = torch.Generator().manual_seed(seed)
                        q = torch.randn(1, H, L, D, generator=g)
                        k = torch.randn(1, H, L, D, generator=g)
                        v = torch.randn(1, H, L, D, generator=g)

                        base_angles = _angles(L, P, seed)
                        variant_angles = [_angles(L, P, seed * 10 + n + 1) for n in range(N)]

                        base_R = angles_to_blocks(base_angles)[None, None]  # (1,1,L,P,2,2)
                        variant_Rs = [angles_to_blocks(a)[None, None] for a in variant_angles]
                        variant_cos_sins = [angles_to_cos_sin(a) for a in variant_angles]

                        q_base = apply_rope_matrix(q, base_R, "flux")
                        k_base = apply_rope_matrix(k, base_R, "flux")

                        ours = spa_averaged_attention(
                            q_base, k_base, v, base_R, variant_Rs,
                            attn_fn=_attn3, pre_roped=True, fmt="flux",
                        )
                        ref = reference_spa_attention(q, k, v, variant_cos_sins, attention_scale=1.0)
                        assert torch.allclose(ours, ref, atol=1e-5, rtol=1e-4), (
                            f"seed={seed} D={D} L={L} N={N} "
                            f"max={ (ours - ref).abs().max().item():.2e}"
                        )

    def test_bridge_cos_sin_is_exact(self):
        # T-P1-7: ours (base-roped q,k + deltas) == reference (raw q,k + variant cos/sin)
        seed, D, L, N, H = 0, 64, 256, 3, 4
        P = D // 2
        g = torch.Generator().manual_seed(seed)
        q = torch.randn(1, H, L, D, generator=g)
        k = torch.randn(1, H, L, D, generator=g)
        v = torch.randn(1, H, L, D, generator=g)

        base_angles = _angles(L, P, seed)
        variant_angles = [_angles(L, P, seed * 10 + n + 1) for n in range(N)]
        base_R = angles_to_blocks(base_angles)[None, None]
        variant_Rs = [angles_to_blocks(a)[None, None] for a in variant_angles]
        variant_cos_sins = [angles_to_cos_sin(a) for a in variant_angles]

        q_base = apply_rope_matrix(q, base_R, "flux")
        k_base = apply_rope_matrix(k, base_R, "flux")

        ours = spa_averaged_attention(
            q_base, k_base, v, base_R, variant_Rs,
            attn_fn=_attn3, pre_roped=True, fmt="flux",
        )
        ref = reference_spa_attention(q, k, v, variant_cos_sins, attention_scale=1.0)
        assert torch.allclose(ours, ref, atol=1e-4)
