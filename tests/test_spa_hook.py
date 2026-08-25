"""P3 — SPA attention hook behaviour (module-level optimized_attention wrapper).

These tests isolate ``_spa_install_hook`` / ``restore_spa_attention_hook`` from the
real ComfyUI runtime by injecting a *mock* ``comfy.ldm.modules.attention`` module
whose ``optimized_attention`` is a plain scaled-dot-product-attention shim with
``scale=1.0`` (matching the HRDiT reference).  They verify:

  * T-P3-1: when the active :class:`SPAContext` is inactive, the wrapper delegates
    straight to ``optimized_attention`` (no RoPE, no averaging).
  * T-P3-2: when active with ``N=3`` bundled variants, the hooked attention equals
    HRDiT's ``reference_spa_attention`` exactly (the faithful fix).
  * T-P3-5: ``restore_spa_attention_hook`` restores the original ``optimized_attention``
    and clears the unet wrapper.
  * T-P3-6: ``pre_roped`` / ``fmt`` are wired through to ``spa_averaged_attention``
    (``pre_roped=False`` applies the full variant RoPE, not the delta).

Markers: @pytest.mark.unit
"""

import pytest
import torch
import torch.nn.functional as F

from src.spa import _spa_install_hook, restore_spa_attention_hook
from src.spa_attn import apply_rope_matrix
from src.spa_context import SPAContext, set_spa_context

try:
    from tests._spa_math_helpers import (
        angles_to_blocks,
        angles_to_cos_sin,
        reference_spa_attention,
    )
except ImportError:  # namespace-package import fallback
    from _spa_math_helpers import (
        angles_to_blocks,
        angles_to_cos_sin,
        reference_spa_attention,
    )


@pytest.fixture
def mock_attn():
    """The conftest-provided (pristine SDPA) mock ``comfy.ldm.modules.attention`` module."""
    import comfy.ldm.modules.attention as attn_mod

    return attn_mod


class _MockModel:
    """Minimal ModelPatcher stand-in for ``_spa_install_hook`` / ``restore``."""

    def __init__(self):
        self._object_patches = {}
        self._unet_wrapper = None
        self._spa_orig_optimized_attention = None

    def clone(self):
        new = _MockModel()
        new._object_patches = dict(self._object_patches)
        new._unet_wrapper = self._unet_wrapper
        new._spa_orig_optimized_attention = self._spa_orig_optimized_attention
        return new

    def add_object_patch(self, path, obj):
        self._object_patches[path] = obj

    def set_model_unet_function_wrapper(self, fn):
        self._unet_wrapper = fn


@pytest.mark.unit
class TestSpaHookPassthrough:
    def test_inactive_delegates_to_original(self, mock_attn):
        """T-P3-1: inactive context -> wrapper is a transparent passthrough."""
        set_spa_context(None)  # no active SPA
        m = _MockModel()
        _spa_install_hook(m, "flux")

        q = torch.randn(1, 4, 64, 64)
        k = torch.randn(1, 4, 64, 64)
        v = torch.randn(1, 4, 64, 64)
        out = mock_attn.optimized_attention(q, k, v, 4)
        ref = F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=0.0, is_causal=False)
        assert torch.allclose(out, ref, atol=1e-6)


@pytest.mark.unit
class TestSpaHookAveraged:
    def test_hooked_equals_hrdit_n3(self, mock_attn):
        """T-P3-2: active (N=3, pre_roped=True) hooked output == HRDiT reference."""
        L, H, D, N = 256, 4, 64, 3
        P = D // 2
        g = torch.Generator().manual_seed(0)
        q = torch.randn(1, H, L, D, generator=g)
        k = torch.randn(1, H, L, D, generator=g)
        v = torch.randn(1, H, L, D, generator=g)

        base_angles = torch.randn(L, P, generator=g) * 0.3
        variant_angles = [torch.randn(L, P, generator=g) * 0.3 for _ in range(N)]
        base_R = angles_to_blocks(base_angles)[None, None]            # (1,1,L,P,2,2)
        variant_Rs = [angles_to_blocks(a)[None, None] for a in variant_angles]

        ctx = SPAContext(
            active=True, bundle_size=N, base_pe=base_R, variant_pes=variant_Rs,
            pre_roped=True, fmt="flux", model_key=0,
        )
        set_spa_context(ctx)

        m = _MockModel()
        _spa_install_hook(m, "flux")

        q_base = apply_rope_matrix(q, base_R, "flux")
        k_base = apply_rope_matrix(k, base_R, "flux")
        out = mock_attn.optimized_attention(q_base, k_base, v, H)
        ref = reference_spa_attention(
            q, k, v, [angles_to_cos_sin(a) for a in variant_angles], attention_scale=1.0
        )
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


@pytest.mark.unit
class TestSpaHookRestore:
    def test_restore_reinstalls_original_and_clears_wrapper(self, mock_attn):
        """T-P3-5: restore returns the original optimized_attention and clears the wrapper."""
        real_orig = mock_attn.optimized_attention
        m = _MockModel()
        _spa_install_hook(m, "flux")

        # Hook installed: module-level fn replaced, original captured on the model.
        assert mock_attn.optimized_attention is not real_orig
        assert m._spa_orig_optimized_attention is real_orig
        assert m._unet_wrapper is not None

        restore_spa_attention_hook(m, mock_attn)
        assert mock_attn.optimized_attention is real_orig
        assert m._unet_wrapper is None
        assert m._spa_orig_optimized_attention is None


@pytest.mark.unit
class TestSpaHookWiring:
    def test_pre_roped_false_applies_full_variant(self, mock_attn):
        """T-P3-6: ``pre_roped=False`` applies the full variant RoPE (no inv(base)@variant)."""
        L, H, D, N = 256, 4, 64, 3
        P = D // 2
        g = torch.Generator().manual_seed(3)
        q = torch.randn(1, H, L, D, generator=g)
        k = torch.randn(1, H, L, D, generator=g)
        v = torch.randn(1, H, L, D, generator=g)

        variant_angles = [torch.randn(L, P, generator=g) * 0.3 for _ in range(N)]
        variant_Rs = [angles_to_blocks(a)[None, None] for a in variant_angles]

        # pre_roped=False: the hook applies each full variant RoPE directly.
        ctx = SPAContext(
            active=True, bundle_size=N, base_pe=None, variant_pes=variant_Rs,
            pre_roped=False, fmt="flux", model_key=0,
        )
        set_spa_context(ctx)

        m = _MockModel()
        _spa_install_hook(m, "flux")
        out = mock_attn.optimized_attention(q, k, v, H)
        ref = reference_spa_attention(
            q, k, v, [angles_to_cos_sin(a) for a in variant_angles], attention_scale=1.0
        )
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)

    def test_fmt_anima_runs_finite(self, mock_attn):
        """T-P3-6: ``fmt='anima'`` is honoured by the hook (3D RoPE layout)."""
        L, H, D, N = 128, 2, 48, 3
        P = D // 2
        g = torch.Generator().manual_seed(5)
        q = torch.randn(1, H, L, D, generator=g)
        k = torch.randn(1, H, L, D, generator=g)
        v = torch.randn(1, H, L, D, generator=g)

        variant_angles = [torch.randn(L, P, generator=g) * 0.3 for _ in range(N)]
        variant_Rs = [angles_to_blocks(a)[None, None] for a in variant_angles]

        ctx = SPAContext(
            active=True, bundle_size=N, base_pe=None, variant_pes=variant_Rs,
            pre_roped=False, fmt="anima", model_key=0,
        )
        set_spa_context(ctx)

        m = _MockModel()
        _spa_install_hook(m, "anima")
        out = mock_attn.optimized_attention(q, k, v, H)
        assert torch.isfinite(out).all()
        # The result is a proper average of N attention outputs (finite, non-trivial).
        assert out.shape == (1, H, L, D)


@pytest.mark.unit
class TestSpaStepGating:
    """T2.2 — sigma-based step gating (D2a, HRDiT-faithful leading-steps-only).

    The unet wrapper closes the SPA step gate when the current sigma is below
    ``m._spa_start_sigma``; the attention wrapper then runs plain attention at
    baseline speed.  ``spa_start_sigma >= 1.0`` keeps SPA active on every step
    (backward-compatible default).
    """

    def _active_ctx(self, L=256, D=64, N=3, seed=0):
        P = D // 2
        g = torch.Generator().manual_seed(seed)
        base_angles = torch.randn(L, P, generator=g) * 0.3
        variant_angles = [torch.randn(L, P, generator=g) * 0.3 for _ in range(N)]
        base_R = angles_to_blocks(base_angles)[None, None]
        variant_Rs = [angles_to_blocks(a)[None, None] for a in variant_angles]
        return SPAContext(
            active=True, bundle_size=N, base_pe=base_R, variant_pes=variant_Rs,
            pre_roped=True, fmt="flux", model_key=0,
        ), base_R, variant_angles

    def test_gate_closed_below_threshold_runs_plain_attention(self, mock_attn):
        """Low sigma (< spa_start_sigma) -> gate closed -> plain attention."""
        L, H, D = 256, 4, 64
        g = torch.Generator().manual_seed(1)
        q = torch.randn(1, H, L, D, generator=g)
        k = torch.randn(1, H, L, D, generator=g)
        v = torch.randn(1, H, L, D, generator=g)
        ctx, base_R, _ = self._active_ctx(L, D)

        m = _MockModel()
        m._spa_start_sigma = 0.5
        _spa_install_hook(m, "flux")

        captured = {}

        def model_function(x, t, **c):
            set_spa_context(ctx)  # simulate the embedder forward
            captured["out"] = mock_attn.optimized_attention(q, k, v, H)
            return captured["out"]

        # sigma = 0.3 < 0.5 -> gate CLOSED -> plain attention (no SPA averaging)
        m._unet_wrapper(model_function, {"input": None, "timestep": torch.tensor([0.3]), "c": {}})
        ref = F.scaled_dot_product_attention(q, k, v, scale=1.0, dropout_p=0.0, is_causal=False)
        assert torch.allclose(captured["out"], ref, atol=1e-6), (
            "gate closed but output differs from plain attention")

    def test_gate_open_above_threshold_runs_spa(self, mock_attn):
        """High sigma (> spa_start_sigma) -> gate open -> SPA averaged attention."""
        L, H, D = 256, 4, 64
        g = torch.Generator().manual_seed(1)
        q = torch.randn(1, H, L, D, generator=g)
        k = torch.randn(1, H, L, D, generator=g)
        v = torch.randn(1, H, L, D, generator=g)
        ctx, base_R, variant_angles = self._active_ctx(L, D)

        m = _MockModel()
        m._spa_start_sigma = 0.5
        _spa_install_hook(m, "flux")

        captured = {}

        def model_function(x, t, **c):
            set_spa_context(ctx)
            q_base = apply_rope_matrix(q, base_R, "flux")
            k_base = apply_rope_matrix(k, base_R, "flux")
            captured["out"] = mock_attn.optimized_attention(q_base, k_base, v, H)
            return captured["out"]

        # sigma = 0.8 > 0.5 -> gate OPEN -> SPA averaged attention
        m._unet_wrapper(model_function, {"input": None, "timestep": torch.tensor([0.8]), "c": {}})
        ref = reference_spa_attention(
            q, k, v, [angles_to_cos_sin(a) for a in variant_angles], attention_scale=1.0
        )
        assert torch.allclose(captured["out"], ref, atol=1e-5, rtol=1e-4), (
            "gate open but output is not the HRDiT averaged attention")

    def test_default_threshold_always_active(self, mock_attn):
        """spa_start_sigma=1.0 (default) -> SPA active on every step."""
        L, H, D = 256, 4, 64
        g = torch.Generator().manual_seed(2)
        q = torch.randn(1, H, L, D, generator=g)
        k = torch.randn(1, H, L, D, generator=g)
        v = torch.randn(1, H, L, D, generator=g)
        ctx, base_R, variant_angles = self._active_ctx(L, D, seed=2)

        m = _MockModel()
        m._spa_start_sigma = 1.0  # default: always active
        _spa_install_hook(m, "flux")

        captured = {}

        def model_function(x, t, **c):
            set_spa_context(ctx)
            q_base = apply_rope_matrix(q, base_R, "flux")
            k_base = apply_rope_matrix(k, base_R, "flux")
            captured["out"] = mock_attn.optimized_attention(q_base, k_base, v, H)
            return captured["out"]

        # Even a very low sigma keeps SPA active with the default threshold.
        m._unet_wrapper(model_function, {"input": None, "timestep": torch.tensor([0.01]), "c": {}})
        ref = reference_spa_attention(
            q, k, v, [angles_to_cos_sin(a) for a in variant_angles], attention_scale=1.0
        )
        assert torch.allclose(captured["out"], ref, atol=1e-5, rtol=1e-4)

    def test_gate_reopened_after_forward(self, mock_attn):
        """The gate is reopened after the forward so a later non-SPA forward is unaffected."""
        from src.spa_context import get_spa_step_gate

        m = _MockModel()
        m._spa_start_sigma = 0.5
        _spa_install_hook(m, "flux")

        def model_function(x, t, **c):
            return None

        m._unet_wrapper(model_function, {"input": None, "timestep": torch.tensor([0.1]), "c": {}})
        assert get_spa_step_gate() is True, "gate must be reopened after the forward"


@pytest.mark.unit
class TestP0Turn3HookMultiplier:
    """T0.3 — quantify the slowdown source on the CURRENT (defective) code.

    With the knob driven as the paper's N but implemented as group_num, a 64x64
    grid at knob=3 yields 15 variants (s=8 cap).  Every hooked attention call
    then invokes ``orig`` 15x, on EVERY step (default spa_start_sigma=1.0 keeps
    the gate open).  This test documents the steps x layers x 15 multiplier.
    It is inverted in Phase P2 (T2.3: total-work bound with spa_steps gating).
    """

    def test_t0_3_hook_multiplier(self, mock_attn):
        """S steps x 2 layers x 15 variants -> orig called S*2*15 times."""
        L, H, D, N = 256, 4, 64, 15  # 15 variants == s=8 cap (knob=3 at 64x64)
        P = D // 2
        g = torch.Generator().manual_seed(11)
        q = torch.randn(1, H, L, D, generator=g)
        k = torch.randn(1, H, L, D, generator=g)
        v = torch.randn(1, H, L, D, generator=g)

        base_angles = torch.randn(L, P, generator=g) * 0.3
        variant_angles = [torch.randn(L, P, generator=g) * 0.3 for _ in range(N)]
        base_R = angles_to_blocks(base_angles)[None, None]
        variant_Rs = [angles_to_blocks(a)[None, None] for a in variant_angles]
        ctx = SPAContext(
            active=True, bundle_size=N, base_pe=base_R, variant_pes=variant_Rs,
            pre_roped=True, fmt="flux", model_key=0,
        )

        # Counting orig: wrap the pristine SDPA shim.
        calls = {"n": 0}
        real_orig = mock_attn.optimized_attention

        def counting_orig(*args, **kwargs):
            calls["n"] += 1
            return real_orig(*args, **kwargs)

        mock_attn.optimized_attention = counting_orig

        m = _MockModel()
        m._spa_start_sigma = 1.0  # default: gate open on every step
        _spa_install_hook(m, "flux")

        S, LAYERS = 5, 2

        def model_function(x, t, **c):
            set_spa_context(ctx)  # simulate the embedder forward
            for _ in range(LAYERS):
                q_base = apply_rope_matrix(q, base_R, "flux")
                k_base = apply_rope_matrix(k, base_R, "flux")
                mock_attn.optimized_attention(q_base, k_base, v, H)
            return None

        for step in range(S):
            m._unet_wrapper(
                model_function,
                {"input": None, "timestep": torch.tensor([1.0 - 0.1 * step]), "c": {}},
            )

        # Documents the defect: 15x per-call multiplier, no step gating.
        assert calls["n"] == S * LAYERS * N == 5 * 2 * 15, (
            f"expected {S * LAYERS * N} orig calls, got {calls['n']}")


@pytest.mark.unit
class TestP2StepCountGate:
    """T2.2 — HRDiT step gating by LEADING-STEP COUNT (the D4 speed fix).

    ``m._spa_steps`` is the number of leading denoising steps on which SPA is
    active; ``0`` = all steps (backward compat).  A NEW GENERATION is detected
    when the incoming sigma jumps UP (or on the first call), which resets the
    leading-step counter.  Sigma decreases monotonically within a generation, so
    this boundary detection is scheduler-agnostic and deterministic.

    The gate state is observed from INSIDE the model function via
    ``get_spa_step_gate()`` (the wrapper sets it before the forward and reopens
    it in ``finally``).
    """

    def _run_sigmas(self, sigmas, spa_steps, start_sigma=1.0):
        """Install the hook, drive the wrapper over ``sigmas``, and record the
        gate state seen inside the model function at each step."""
        from src.spa_context import get_spa_step_gate

        m = _MockModel()
        m._spa_steps = spa_steps
        m._spa_start_sigma = start_sigma
        _spa_install_hook(m, "flux")

        gates = []

        def model_function(x, t, **c):
            gates.append(get_spa_step_gate())
            return None

        for s in sigmas:
            m._unet_wrapper(
                model_function,
                {"input": None, "timestep": torch.tensor([s]), "c": {}},
            )
        return gates

    def test_step_gate_opens_for_leading_steps(self):
        """T2.2a: sigma [1.0,.9,.8,.7], spa_steps=2 -> open,open,closed,closed."""
        gates = self._run_sigmas([1.0, 0.9, 0.8, 0.7], spa_steps=2)
        assert gates == [True, True, False, False], (
            f"expected [open,open,closed,closed], got {gates}")

    def test_step_gate_resets_on_new_generation(self):
        """T2.2b: sigma jump-up resets the counter -> the gate reopens.

        [1.0,.9,.8, 1.0,.9,.8] with spa_steps=2 -> open,open,closed, open,open,closed.
        """
        gates = self._run_sigmas([1.0, 0.9, 0.8, 1.0, 0.9, 0.8], spa_steps=2)
        assert gates == [True, True, False, True, True, False], (
            f"expected the counter to reset on the sigma jump-up, got {gates}")

    def test_spa_steps_zero_means_all_steps(self):
        """T2.2c: spa_steps=0 (backward compat) -> gate open on every step."""
        gates = self._run_sigmas([1.0, 0.9, 0.8, 0.7, 0.6], spa_steps=0)
        assert gates == [True] * 5, f"spa_steps=0 must keep the gate open, got {gates}"

    def test_step_gate_and_sigma_gate_combine(self):
        """The step-count gate is AND-combined with the sigma-threshold gate.

        spa_steps=3 keeps the count gate open for 3 leading steps, but
        spa_start_sigma=0.85 closes the sigma gate once sigma <= 0.85.
        sigma [1.0,.9,.8,.7]: count gate open,open,open,open; sigma gate
        open(1.0>.85), open(.9>.85), closed(.8), closed(.7) -> AND gives
        open,open,closed,closed.
        """
        gates = self._run_sigmas([1.0, 0.9, 0.8, 0.7], spa_steps=3, start_sigma=0.85)
        assert gates == [True, True, False, False], (
            f"AND of count+sigma gates expected [open,open,closed,closed], got {gates}")

    def test_unreadable_timestep_keeps_spa_active(self):
        """A missing/unreadable timestep is a safe fallback: SPA stays active."""
        from src.spa_context import get_spa_step_gate

        m = _MockModel()
        m._spa_steps = 2
        m._spa_start_sigma = 1.0
        _spa_install_hook(m, "flux")

        gates = []

        def model_function(x, t, **c):
            gates.append(get_spa_step_gate())
            return None

        # No timestep key -> sigma is None -> both gates fall back to open.
        for _ in range(4):
            m._unet_wrapper(model_function, {"input": None, "c": {}})
        assert gates == [True] * 4, (
            f"unreadable timestep must keep SPA active, got {gates}")


@pytest.mark.unit
class TestP2TotalWorkBound:
    """T2.3 — end-to-end multiplier proof (the 10x -> ~1.6x fix).

    20 simulated steps, 2 layers, N=3 (5 averaged passes), spa_steps=3:
      * steps 0..2 (gate open):  2 layers x 3 steps x 5 passes = 30 orig calls
      * steps 3..19 (gate closed): 2 layers x 17 steps x 1 pass = 34 orig calls
      * total = 64 vs baseline (no SPA) 2 x 20 x 1 = 40  ->  1.6x.
    """

    def test_total_work_bounded(self, mock_attn):
        L, H, D, N = 256, 4, 64, 5  # N=3 -> 2*s-1 = 5 passes
        P = D // 2
        g = torch.Generator().manual_seed(21)
        q = torch.randn(1, H, L, D, generator=g)
        k = torch.randn(1, H, L, D, generator=g)
        v = torch.randn(1, H, L, D, generator=g)

        base_angles = torch.randn(L, P, generator=g) * 0.3
        variant_angles = [torch.randn(L, P, generator=g) * 0.3 for _ in range(N)]
        base_R = angles_to_blocks(base_angles)[None, None]
        variant_Rs = [angles_to_blocks(a)[None, None] for a in variant_angles]
        ctx = SPAContext(
            active=True, bundle_size=N, base_pe=base_R, variant_pes=variant_Rs,
            pre_roped=True, fmt="flux", model_key=0,
        )

        calls = {"n": 0}
        real_orig = mock_attn.optimized_attention

        def counting_orig(*args, **kwargs):
            calls["n"] += 1
            return real_orig(*args, **kwargs)

        mock_attn.optimized_attention = counting_orig

        m = _MockModel()
        m._spa_steps = 3          # HRDiT default: SPA on the 3 leading steps
        m._spa_start_sigma = 1.0   # no sigma gating
        _spa_install_hook(m, "flux")

        S, LAYERS = 20, 2

        def model_function(x, t, **c):
            set_spa_context(ctx)
            for _ in range(LAYERS):
                q_base = apply_rope_matrix(q, base_R, "flux")
                k_base = apply_rope_matrix(k, base_R, "flux")
                mock_attn.optimized_attention(q_base, k_base, v, H)
            return None

        # Monotonically decreasing sigma (one generation, 20 steps).
        for step in range(S):
            m._unet_wrapper(
                model_function,
                {"input": None, "timestep": torch.tensor([1.0 - 0.045 * step]), "c": {}},
            )

        # 3 gated-open steps x 2 layers x 5 passes + 17 closed steps x 2 layers x 1.
        expected = LAYERS * (3 * N + (S - 3) * 1)
        assert calls["n"] == expected == 64, (
            f"expected {expected} orig calls (1.6x baseline), got {calls['n']}")
        baseline = LAYERS * S * 1
        assert calls["n"] / baseline == pytest.approx(1.6), (
            f"total work must be ~1.6x baseline, got {calls['n'] / baseline:.2f}x")
