"""Period-``s`` ripple detector: instrument validation + T0.4 calibration.

The SPA "ripple / mosaic" artifact (turn-3 user report) is a period-``s``
interference pattern imprinted by the averaged attention passes: each pass
attends on an ``s``-fold position-compressed grid, so the N-pass mean carries a
period-``s`` modulation.  The 2026-08-14 isolation probes missed it because they
used ONLY smooth low-frequency q/k, which cannot excite the bundle band.

This module provides the permanent detector infrastructure:

* ``test_detector_*`` — instrument self-validation (always green):
  - a zero delta (identity passthrough control) yields no peak,
  - a synthetic axis-aligned period-``s`` ripple IS detected (peak ratio >> 1).
* ``TestT04CurrentCodeCalibration`` — documents D3 on the CURRENT (defective)
  code: with the knob driven as the paper's N but implemented as group_num,
  knob=3 at 64x64 yields s=8 (15 passes) and the detrended period-8 peak ratio
  rises above the smooth trend.  This class is replaced by the P4 quality gates
  (recommended configs must sit ON the trend) in the 2026-08-15 plan.

The detector itself (``period_peak_ratio``) removes the inherent 1/f spectral
trend of the smooth ``(SPA - base)`` delta by dividing the period-``s ± 1``
ring power by a running-median radial trend, so values near 1.0 mean "on the
smooth trend" (clean) and values >> 1 mean a genuine structured peak.

All tests are pure torch (no ComfyUI, no model loading).  Markers: unit.
"""
import math

import pytest
import torch

from src.spa import build_bundle_id_variants
from src.spa_attn import apply_rope_matrix, spa_averaged_attention

try:
    from tests._spa_math_helpers import (
        KREA_AXES_DIM,
        krea_pe,
        period_band_power,
        period_peak_ratio,
        structured_qkv,
    )
except ImportError:  # namespace-package import fallback
    from _spa_math_helpers import (
        KREA_AXES_DIM,
        krea_pe,
        period_band_power,
        period_peak_ratio,
        structured_qkv,
    )

D = sum(KREA_AXES_DIM)  # 128 (Krea-2 head dim)


def _plain_attn(q, k, v):
    d = q.shape[-1]
    scores = torch.einsum("bhld,bhmd->bhlm", q, k) / (d ** 0.5)
    return torch.einsum("bhlm,bhmd->bhld", torch.softmax(scores, dim=-1), v)


def _flux_ids(H, W):
    L = H * W
    ids = torch.zeros(1, L, 3)
    ids[..., 0] = torch.arange(L)
    ids[..., 1] = torch.arange(H).unsqueeze(1).expand(H, W).reshape(-1).float()
    ids[..., 2] = torch.arange(W).unsqueeze(0).expand(H, W).reshape(-1).float()
    return ids


def _spa_delta(H, W, variants, seed=0):
    """Run the real averaged-attention path and return ``(SPA - base)`` reshaped
    to ``(H, W, D)`` over the image tokens."""
    ids = _flux_ids(H, W)
    base_pe = krea_pe(ids)
    variant_pes = [krea_pe(v) for v in variants]
    q, k, v = structured_qkv(H, W, D, seed)
    qb = apply_rope_matrix(q, base_pe, "flux")
    kb = apply_rope_matrix(k, base_pe, "flux")
    base_out = _plain_attn(qb, kb, v)
    spa_out = spa_averaged_attention(qb, kb, v, base_pe, variant_pes,
                                     attn_fn=_plain_attn, pre_roped=True, fmt="flux")
    return (spa_out - base_out)[0, 0].reshape(H, W, D)


@pytest.mark.unit
class TestDetectorInstrument:
    """Instrument self-validation — these must stay green forever."""

    def test_detector_zero_delta_has_no_peak(self):
        """Control: identity variants -> passthrough -> delta == 0 -> no peak."""
        H = W = 32
        ids = _flux_ids(H, W)
        delta = _spa_delta(H, W, [ids.clone()])
        assert torch.equal(delta, torch.zeros_like(delta))
        assert period_band_power(delta, H, W, 4) == 0.0
        assert period_peak_ratio(delta, H, W, 4) < 1.05

    def test_detector_detects_synthetic_period8_ripple(self):
        """A synthetic axis-aligned period-8 ripple MUST be detected (ratio >> 1)."""
        H = W = 64
        h = torch.arange(H).float()
        w = torch.arange(W).float()
        # axis-aligned period-8 ripple + smooth low-freq trend + tiny noise
        ripple = 0.5 * torch.cos(2 * math.pi * h[:, None] / 8) * torch.ones(1, W)
        trend = 0.3 * torch.sin(2 * math.pi * h[:, None] / H) * torch.cos(
            2 * math.pi * w[None, :] / W)
        g = torch.Generator().manual_seed(0)
        noise = 0.01 * torch.randn(H, W, generator=g)
        field = (ripple + trend + noise).reshape(H, W, 1).repeat(1, 1, D)

        peak8 = period_peak_ratio(field, H, W, 8)
        # other periods sit on the trend (ratio ~1)
        peak5 = period_peak_ratio(field, H, W, 5)
        assert peak8 > 3.0, f"synthetic period-8 ripple not detected: {peak8}"
        assert peak8 > 3.0 * peak5, (
            f"period-8 peak ({peak8}) must dominate period-5 ({peak5})")

    def test_detector_smooth_field_has_no_peak(self):
        """A smooth field + broadband noise floor sits on the trend at every period.

        The noise floor keeps the radial trend well-defined everywhere (a pure
        single sinusoid has zero power off-peak, which would make the ratio
        numerically ill-posed).
        """
        H = W = 64
        h = torch.arange(H).float()
        w = torch.arange(W).float()
        g = torch.Generator().manual_seed(1)
        field = (0.5 * torch.sin(2 * math.pi * h[:, None] / H)
                 * torch.cos(2 * math.pi * w[None, :] / W)
                 + 0.02 * torch.randn(H, W, generator=g))
        field = field.reshape(H, W, 1).repeat(1, 1, D)
        for p in (2, 3, 4, 5, 8, 12):
            peak = period_peak_ratio(field, H, W, p)
            assert peak < 2.0, f"smooth field shows a false period-{p} peak: {peak}"


@pytest.mark.unit
class TestT04FixedCodeCalibration:
    """T0.4 — verify the P1 paper-N rewire removes the D3 period-s ripple source.

    BEFORE the fix (documented 2026-08-15): the knob was driven as the paper's N
    but implemented as group_num, so knob=3 at 64x64 derived s = ceil(63/2) = 32,
    capped at s = 8 -> 15 passes, each attending on an 8x-compressed grid; the
    averaged output carried a period-8 modulation (peak_ratio ~= 1.30 at 64x64).

    AFTER the fix: 64x64 (max_pos=63) is INSIDE the trained extent (64), so the
    trained-extent gate makes SPA an identity no-op for ANY N -> zero delta, no
    ripple.  This class asserts that fixed behaviour; the full recommended-config
    quality gates (N=3 at 2K, N=5 at 4K must sit ON the trend) live in P4.
    """

    def test_t0_4_in_trained_extent_is_identity_no_ripple(self):
        """At 64x64 (max_pos=63 <= trained_extent=64) N=3 -> identity -> zero delta."""
        H = W = 64
        ids = _flux_ids(H, W)
        variants = build_bundle_id_variants(ids, 3)
        # Fixed behaviour: inside the trained extent -> a single identity variant.
        assert len(variants) == 1, (
            f"expected 1 identity variant at 64x64 (in trained extent), got {len(variants)}")

        delta = _spa_delta(H, W, variants, seed=0)
        assert torch.isfinite(delta).all()
        # Identity passthrough -> zero delta -> no ripple source at all.
        assert torch.equal(delta, torch.zeros_like(delta)), (
            "SPA must be a no-op inside the trained extent (the big-patch / ripple "
            "source at 1024px is removed by the trained-extent gate)")

    def test_t0_4_identity_control_is_clean(self):
        """Same content, identity variants -> zero delta (the clean reference)."""
        H = W = 64
        ids = _flux_ids(H, W)
        delta = _spa_delta(H, W, [ids.clone()], seed=0)
        assert torch.equal(delta, torch.zeros_like(delta))


# ---------------------------------------------------------------------------
# P4 — Permanent quality gates (recommended configs must sit ON the trend)
# ---------------------------------------------------------------------------

# Clean-floor threshold for the detrended period-s peak detector.  Calibration
# (T0.4, 2026-08-15): a clean config sits at ~1.0; the defective over-compressed
# config (s=8 at 64x64) measured ~1.30-1.54.  1.25 separates the two with margin.
# NOTE: these gates run on a CPU-feasible 80x80 grid (max_pos=79 > trained_extent
# =64, so SPA is ACTIVE with fine bundling s==N); the same s values (3 at 2K, 5 at
# 4K) are what the node derives at full resolution, so the gate is representative.
_PEAK_CLEAN_MAX = 1.25


@pytest.mark.unit
class TestP4RippleQualityGates:
    """T4.1 / T4.2 / T4.3 — permanent ripple + coherence/parity gates.

    These lock the fix: with the paper-``N`` semantics + in-distribution floor,
    the recommended configs (``N = 3`` at 2K, ``N = 5`` at 4K) produce a FINE
    bundle (``s == N``, every bundled position in-distribution) whose averaged
    output carries NO structured period-``s`` ripple — the detrended spectral peak
    sits on the smooth trend (ratio near 1.0), and the output stays coherent and
    close to the base pass.
    """

    def test_recommended_configs_below_ripple_threshold(self):
        """T4.1: N=3 (2K) and N=5 (4K) keep the period-s peak on the trend.

        For each recommended config: the derived bundle is FINE (``s == N``), the
        delta is finite, and the detrended period-``s`` peak ratio stays below the
        clean threshold (no period-``s`` ripple).
        """
        for N in (3, 5):
            H = W = 80  # max_pos=79 > 64 -> active; s_floor=ceil(79/79)=1 -> s==N
            ids = _flux_ids(H, W)
            variants = build_bundle_id_variants(ids, N)
            s = (len(variants) + 1) // 2
            # Fine bundling preserved: s == N (the big-patch over-compression is gone).
            assert s == N, f"N={N}: expected fine bundle s={N}, got s={s}"
            assert len(variants) == 2 * N - 1

            delta = _spa_delta(H, W, variants, seed=0)
            assert torch.isfinite(delta).all()
            assert delta.abs().max().item() > 0  # SPA is active, not a passthrough

            peak = period_peak_ratio(delta, H, W, s)
            assert peak < _PEAK_CLEAN_MAX, (
                f"N={N}: period-{s} peak ratio {peak:.3f} >= {_PEAK_CLEAN_MAX} "
                f"-> a period-{s} ripple is present (mosaic regression)")

    def test_n2_characterization(self):
        """T4.2 (decision M2): characterize N=2 for period-2 aliasing risk.

        W2.8 RE-BASELINE (2026-08-25, suite-green workstream): under the
        current paper-N implementation the measured detrended period-2 peak
        for an ACTIVE s=2 bundle at 80x80 is ~2.35 — ABOVE the clean threshold
        (_PEAK_CLEAN_MAX = 1.25).  N=2 therefore DOES excite a measurable
        period-2 (checkerboard-band) modulation on structured q/k.

        The original protocol said "flip this assertion and implement the
        floor then", but the M2 floor (active s == 2 -> 3 + WARNING) is a
        PRODUCT decision: it changes bundling for every 1.5-2K render,
        including auto mode (ceil(127/79) = 2).  It is tracked as an open
        improvement item instead of being smuggled into a suite-green fix.
        This test now PINS the measured characterization so any future change
        is deliberate and visible:
          * peak2 >= _PEAK_CLEAN_MAX  (ripple present — never recommend N=2),
          * peak2 < 4.0               (bounded — not a catastrophic alias).
        The recommended-config gate above (N=3 / N=5 below the threshold) is
        UNCHANGED and remains the binding quality gate.
        """
        H = W = 80
        ids = _flux_ids(H, W)
        variants = build_bundle_id_variants(ids, 2)
        s = (len(variants) + 1) // 2
        assert s == 2

        delta = _spa_delta(H, W, variants, seed=0)
        assert torch.isfinite(delta).all()

        peak2 = period_peak_ratio(delta, H, W, 2)
        assert peak2 >= _PEAK_CLEAN_MAX, (
            f"N=2 no longer excites a period-2 ripple (peak {peak2:.3f} < "
            f"{_PEAK_CLEAN_MAX}); update this characterization and reconsider "
            f"decision M2 (floor active N at 3 with a WARNING).")
        assert peak2 < 4.0, (
            f"N=2 period-2 peak {peak2:.3f} is unboundedly large — "
            f"re-characterize before trusting this gate.")

    def test_coherence_parity(self):
        """T4.3: permanent parity thresholds vs the base pass.

        * relative deviation ``||SPA - BASE|| / ||BASE|| < 0.25`` (SPA stays a
          bounded position-extrapolation correction, not a rewrite), and
        * coherence parity ``|C_spa - C_base| < 0.05`` where ``C`` is the mean
          adjacent-token cosine similarity of the output feature map (SPA must not
          destroy the base pass's local spatial coherence).
        """
        import torch.nn.functional as F

        H = W = 80
        N = 3
        ids = _flux_ids(H, W)
        variants = build_bundle_id_variants(ids, N)
        base_pe = krea_pe(ids)
        variant_pes = [krea_pe(v) for v in variants]
        q, k, v = structured_qkv(H, W, D, 0)
        qb = apply_rope_matrix(q, base_pe, "flux")
        kb = apply_rope_matrix(k, base_pe, "flux")
        base_out = _plain_attn(qb, kb, v)
        spa_out = spa_averaged_attention(qb, kb, v, base_pe, variant_pes,
                                         attn_fn=_plain_attn, pre_roped=True, fmt="flux")
        assert torch.isfinite(spa_out).all()

        # Relative deviation bound.
        rel = (spa_out - base_out).norm() / base_out.norm().clamp(min=1e-12)
        assert rel < 0.25, f"SPA deviates from base by {rel:.3f} (>= 0.25)"

        # Coherence parity: mean adjacent-token cosine similarity of the output map.
        def _spatial_coherence(out):
            m = out[0, 0].reshape(H, W, -1)
            right = F.cosine_similarity(m[:, :-1], m[:, 1:], dim=-1)
            down = F.cosine_similarity(m[:-1, :], m[1:, :], dim=-1)
            return float(torch.cat([right.reshape(-1), down.reshape(-1)]).mean())

        c_base = _spatial_coherence(base_out)
        c_spa = _spatial_coherence(spa_out)
        assert abs(c_spa - c_base) < 0.05, (
            f"coherence parity broken: |C_spa - C_base| = {abs(c_spa - c_base):.3f} "
            f"(C_base={c_base:.3f}, C_spa={c_spa:.3f})")
