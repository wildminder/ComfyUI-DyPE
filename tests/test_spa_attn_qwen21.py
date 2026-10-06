"""Qwen-Image-2.1 segment math in the SPA averaged-attention core (pure torch).

2.1's block-causal attention hands ``optimized_attention`` a PREFIX of the
sequence per segment (``q = q[:, start:end]``, ``k = k[:, :end]``) with both
already collapsed to ``(B, N, H*D)`` and WITHOUT ``skip_reshape``.  Two facts
follow, and both are load-bearing:

  * :func:`src.spa_attn.apply_rope_matrix` derives ``P`` from the LAST dim, so
    rotating ``(B, N, H*D)`` directly would use ``P = H*D/2`` against a rotation
    built for ``P = D/2`` — the einsum raises.  The unflatten/rotate/reflatten
    round trip is what makes it work at all, and it is EXACT because the
    rotation is head-shared (``R``'s head dim is a singleton).
  * the rotations cover the FULL sequence, so they must be sliced per segment
    before rotation (``rot[..., start:end]`` / ``rot[..., :end]``).

These tests are pure torch (no ComfyUI import, no patcher), which is why they can
run on CPU in CI.  Markers: @pytest.mark.unit
"""

import inspect

import pytest
import torch

from src.spa_attn import apply_rope_matrix, spa_averaged_attention
from src.spa_attn import _rotate_flat

try:
    from tests._spa_math_helpers import angles_to_blocks
except ImportError:  # namespace-package import fallback
    from _spa_math_helpers import angles_to_blocks


def _rot_flux(B, L, P, seed):
    """A reproducible ``(B, 1, L, P, 2, 2)`` head-shared rotation (flux layout)."""
    g = torch.Generator().manual_seed(seed)
    return angles_to_blocks(torch.randn(L, P, generator=g) * 0.3)[None, None]


def _sdpa_flat(q, k, v, heads):
    """SDPA over a head-FLATTENED ``(B, N, H*D)`` triple, as ComfyUI does it."""
    b, n, hd = q.shape
    d = hd // heads
    q4 = q.reshape(b, n, heads, d).transpose(1, 2)
    k4 = k.reshape(b, k.shape[1], heads, d).transpose(1, 2)
    v4 = v.reshape(b, v.shape[1], heads, d).transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(q4, k4, v4, scale=1.0)
    return out.transpose(1, 2).reshape(b, n, heads * d)


@pytest.mark.unit
class TestFlattenUnflatten:
    """The reshape that turns a crash into the backend's own head split."""

    def test_rotate_flat_equals_per_head_rotation(self):
        """Rotating (B,H,N,D) per head == rotating (B,N,H*D) with head-broadcast R.

        The reference is the layout ``attention_pytorch`` itself builds before
        SDPA — ``apply_rope_matrix`` reads its leading dims as broadcastable, so
        the head dim lives there.
        """
        B, N, H, D = 2, 6, 4, 8
        g = torch.Generator().manual_seed(0)
        x = torch.randn(B, N, H * D, generator=g)
        R = _rot_flux(B, N, D // 2, seed=1)

        flat = _rotate_flat(x, R, "flux", H)
        per_head = apply_rope_matrix(
            x.reshape(B, N, H, D).transpose(1, 2), R, "flux"
        )
        assert flat.shape == (B, N, H * D)
        ref = per_head.transpose(1, 2).reshape(B, N, H * D)
        assert torch.allclose(flat, ref, atol=1e-6)

    def test_rotation_is_shared_across_heads(self):
        """The SAME rotation is applied to every head (that is why it broadcasts)."""
        B, N, H, D = 1, 5, 4, 8
        g = torch.Generator().manual_seed(2)
        head_slice = torch.randn(B, N, D, generator=g)
        # Identical content per head -> identical output per head iff the
        # rotation really is head-shared.
        x = head_slice.unsqueeze(2).expand(B, N, H, D).reshape(B, N, H * D)
        R = _rot_flux(B, N, D // 2, seed=3)
        out = _rotate_flat(x, R, "flux", H).reshape(B, N, H, D)
        for h in range(1, H):
            assert torch.equal(out[:, :, 0], out[:, :, h])

    def test_flattened_input_raises_without_flatten_heads(self):
        """Negative control: the unflatten is required, not cosmetic.

        Rotating the flattened tensor with ``apply_rope_matrix`` derives
        ``P = H*D/2`` from the last dim against a ``P = D/2`` rotation.
        """
        B, N, H, D = 1, 5, 4, 8
        x = torch.randn(B, N, H * D)
        R = _rot_flux(B, N, D // 2, seed=4)
        with pytest.raises(RuntimeError):
            apply_rope_matrix(x, R, "flux")


@pytest.mark.unit
class TestSegmentSlicing:
    """``rot[..., start:end]`` selects the rows that match the segment tensors."""

    def test_sliced_rotation_matches_per_head_slice(self):
        """A (B,Nq,H*D) segment == the same rows of the full sequence, rotated."""
        B, L, H, D = 1, 9, 4, 8
        start, end = 4, 9
        g = torch.Generator().manual_seed(5)
        x_full = torch.randn(B, L, H * D, generator=g)
        R = _rot_flux(B, L, D // 2, seed=6)

        seg = _rotate_flat(x_full[:, start:end], R[..., start:end, :, :, :], "flux", H)
        ref = apply_rope_matrix(
            x_full[:, start:end].reshape(B, end - start, H, D).transpose(1, 2),
            R[..., start:end, :, :, :],
            "flux",
        ).transpose(1, 2).reshape(B, end - start, H * D)
        assert seg.shape == (B, end - start, H * D)
        assert torch.allclose(seg, ref, atol=1e-6)

    def test_slice_is_ellipsis_first_for_the_anima_layout(self):
        """The 4-D ``(L, P, 2, 2)`` anima layout slices with the same expression."""
        B, L, D = 1, 9, 8
        start, end = 4, 9
        g = torch.Generator().manual_seed(7)
        x = torch.randn(B, 1, L, D, generator=g)
        R = angles_to_blocks(torch.randn(L, D // 2, generator=g) * 0.3)
        seen = {}

        def attn(qq, kk, vv):
            seen.setdefault("q", qq)
            return qq

        spa_averaged_attention(
            x[:, :, start:end], x, x, None, [R, R],
            attn_fn=attn, pre_roped=False, fmt="anima",
            q_slice=slice(start, end), k_slice=slice(0, end),
        )
        ref = apply_rope_matrix(x[:, :, start:end], R[start:end], "anima")
        assert torch.allclose(seen["q"], ref, atol=1e-6)


@pytest.mark.unit
class TestAveragedAttentionKwargs:
    """The three new kwargs are additive; the default path is untouched."""

    def test_defaults_are_the_legacy_path(self):
        """With no new kwargs the output equals the explicit pre-existing loop.

        Run in the ordinary head format ``(B, H, L, D)`` every other backend uses,
        so this is a real lock on "the other six backends are unchanged" rather
        than a restatement of the 2.1 branch.
        """
        B, N, H, D = 1, 7, 4, 8
        g = torch.Generator().manual_seed(8)
        q = torch.randn(B, H, N, D, generator=g)
        k = torch.randn(B, H, N, D, generator=g)
        v = torch.randn(B, H, N, D, generator=g)
        rots = [_rot_flux(B, N, D // 2, seed=10 + i) for i in range(3)]

        def attn(qq, kk, vv):
            return torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, scale=1.0
            )

        out = spa_averaged_attention(
            q, k, v, None, rots, attn_fn=attn, pre_roped=False, fmt="flux"
        )
        ref = torch.stack(
            [
                attn(apply_rope_matrix(q, rot, "flux"), apply_rope_matrix(k, rot, "flux"), v)
                for rot in rots
            ],
            dim=0,
        ).mean(dim=0)
        assert torch.allclose(out, ref, atol=1e-6)

    def test_new_kwargs_default_to_none(self):
        """Lock the additive contract: the 2.1 knobs are opt-in, not required."""
        params = inspect.signature(spa_averaged_attention).parameters
        for name in ("flatten_heads", "q_slice", "k_slice"):
            assert params[name].default is None

    def test_flatten_heads_path_averages_n_outputs(self):
        """``flatten_heads`` runs N passes over the unflattened views and averages."""
        B, N, H, D = 1, 7, 4, 8
        g = torch.Generator().manual_seed(9)
        q = torch.randn(B, N, H * D, generator=g)
        k = torch.randn(B, N, H * D, generator=g)
        v = torch.randn(B, N, H * D, generator=g)
        rots = [_rot_flux(B, N, D // 2, seed=20 + i) for i in range(3)]

        seen = []

        def attn(qq, kk, vv):
            seen.append(qq.shape)
            return _sdpa_flat(qq, kk, vv, H)

        out = spa_averaged_attention(
            q, k, v, None, rots, attn_fn=attn, pre_roped=False, fmt="flux",
            flatten_heads=H,
        )
        # One call per variant, each still in the caller's flattened layout.
        assert seen == [(B, N, H * D)] * 3
        assert out.shape == (B, N, H * D)
        assert torch.isfinite(out).all()

    def test_single_variant_is_a_passthrough(self):
        """<= 1 variant short-circuits BEFORE any reshape/slice logic."""
        B, N, H, D = 1, 7, 4, 8
        q = torch.randn(B, N, H * D)
        v = torch.randn(B, N, H * D)
        sentinel = torch.zeros(B, N, H * D)
        calls = []

        def attn(qq, kk, vv):
            calls.append((qq, kk))
            return sentinel

        out = spa_averaged_attention(
            q, q, v, None, [_rot_flux(B, N, D // 2, seed=30)],
            attn_fn=attn, pre_roped=False, fmt="flux",
            flatten_heads=H, q_slice=slice(2, 5), k_slice=slice(0, 5),
        )
        assert calls == [(q, q)]  # untouched inputs
        assert torch.equal(out, sentinel)