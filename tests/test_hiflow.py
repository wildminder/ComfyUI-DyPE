"""Tests for src/hiflow.py — HiFlow core algorithm (Tier 1: pure unit tests).

Steps 1-2 of plan 2026-09-03:
- Butterworth low-pass mask + FFT frequency split (the loop-parity test is
  the fidelity tripwire against the authors' reference implementation,
  .dev/data/HiFlow/HiFlow/utils.py);
- stage sigma slicing + alignment scales (paper alpha_t = beta_t = t/tau).
"""

import time

import pytest
import torch

from src.hiflow import (
    alignment_scales,
    build_stage_sigmas,
    butterworth_low_pass_filter_2d,
    split_frequency_components_fft,
)


def flux_like_sigmas(steps: int = 30, shift: float = 1.15) -> torch.Tensor:
    """FLUX-style shifted flow schedule (descending, ends at 0).

    Flow time descends 1 -> 1/steps across the `steps` interior sigmas
    (sigma at t=0 is 0, appended last).
    """
    def time_snr_shift(alpha, t):
        return alpha * t / (1 + (alpha - 1) * t)

    flow_times = torch.linspace(1.0, 1.0 / steps, steps)
    sigmas = [time_snr_shift(shift, float(t)) for t in flow_times]
    return torch.tensor(sigmas + [0.0])


# ---------------------------------------------------------------------------
# Butterworth low-pass mask
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestButterworthFilter:
    def test_mask_shape_and_range(self):
        for ratio in (0.2, 0.4, 0.95):
            mask = butterworth_low_pass_filter_2d((4, 16, 16), ratio=ratio)
            assert mask.shape == (16, 16)
            assert mask.min() >= 0.0
            assert mask.max() <= 1.0

    def test_ratio_zero_returns_zero_mask(self):
        """Reference behavior: ratio=0 -> all-zero (nothing passes)."""
        mask = butterworth_low_pass_filter_2d((16, 16), ratio=0.0)
        assert mask.shape == (16, 16)
        assert torch.count_nonzero(mask) == 0

    def test_high_ratio_nearly_passes_everything(self):
        mask = butterworth_low_pass_filter_2d((16, 16), ratio=0.95)
        assert mask.max().item() >= 0.9

    def test_center_passes_high_freq_blocked(self):
        """DC center (fftshifted layout) == 1; corners (highest |d|) < 0.05."""
        h = w = 16
        mask = butterworth_low_pass_filter_2d((h, w), ratio=0.2)
        assert mask[h // 2, w // 2].item() == pytest.approx(1.0, abs=1e-5)
        for corner in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            assert mask[corner].item() < 0.05, (
                f"corner {corner} must be strongly attenuated at ratio 0.2"
            )

    def test_matches_loop_reference(self):
        """Vectorized mask == the authors' Python double-loop formula verbatim.

        Loop body copied from .dev/data/HiFlow/HiFlow/utils.py
        (butterworth_low_pass_filter_2d): d_square uses (2h/H - 1) terms with
        h the row index and H the full height.
        """
        for ratio in (0.2, 0.4):
            for h, w in ((16, 16), (8, 24)):
                mask_vec = butterworth_low_pass_filter_2d((h, w), ratio=ratio)

                loop = torch.zeros((h, w))
                for row in range(h):
                    for col in range(w):
                        d_square = ((2 * row / h - 1) ** 2 + (2 * col / w - 1) ** 2)
                        loop[row, col] = 1 / (1 + (d_square / ratio ** 2) ** 4)

                assert torch.allclose(mask_vec, loop, atol=1e-5), (
                    f"vectorized mask diverges from the reference loop "
                    f"(h={h}, w={w}, ratio={ratio})"
                )

    def test_vectorized_build_is_fast(self):
        """64x64 must build in < 50 ms on CPU (guards against regression to a
        Python loop; generous bound)."""
        t0 = time.perf_counter()
        butterworth_low_pass_filter_2d((64, 64), ratio=0.2)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.050, f"mask build took {elapsed * 1000:.1f} ms"

    def test_invalid_shape_rejected(self):
        with pytest.raises(ValueError, match="positive H, W"):
            butterworth_low_pass_filter_2d((0, 8))

    def test_symmetric_mask(self):
        """The mask is mirror-symmetric about the axis through the DC row pair.

        The reference grid g[h] = 2h/H - 1 satisfies g[H-h] = -g[h], so
        mask[h, :] == mask[H-h, :] for h in 1..H-1 (row 0's mirror would be
        row H, which does not exist — the grid is edge-asymmetric, NOT
        flip-symmetric, for even sizes).
        """
        h = w = 16
        mask = butterworth_low_pass_filter_2d((h, w), ratio=0.3)
        for row in range(1, h):
            assert torch.allclose(mask[row], mask[h - row], atol=1e-7), (
                f"row {row} must mirror row {h - row}"
            )
        for col in range(1, w):
            assert torch.allclose(mask[:, col], mask[:, w - col], atol=1e-7), (
                f"col {col} must mirror col {w - col}"
            )


# ---------------------------------------------------------------------------
# FFT frequency split
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestFFTSplit:
    def test_low_plus_high_equals_x(self):
        torch.manual_seed(0)
        x = torch.randn(2, 4, 16, 16)
        f = butterworth_low_pass_filter_2d((16, 16), ratio=0.2)
        low = split_frequency_components_fft(x, f, is_low=True)
        high = split_frequency_components_fft(x, f, is_low=False)
        assert torch.allclose(low + high, x, atol=1e-4), (
            "low + high must reconstruct the input exactly (mask + (1-mask) = 1)"
        )

    def test_split_low_is_smooth(self):
        """A checkerboard is pure high frequency: its low component must carry
        little energy at a small cutoff."""
        x = torch.zeros(1, 1, 16, 16)
        x[0, 0, ::2, ::2] = 1.0
        x[0, 0, 1::2, 1::2] = 1.0
        f = butterworth_low_pass_filter_2d((16, 16), ratio=0.2)
        low = split_frequency_components_fft(x, f, is_low=True)
        assert low.std().item() < 0.2 * x.std().item(), (
            "low-pass component of a checkerboard should be near zero"
        )

    def test_zero_mask_kills_low_passes_everything(self):
        torch.manual_seed(1)
        x = torch.randn(1, 3, 8, 8)
        low = split_frequency_components_fft(x, torch.zeros(8, 8), is_low=True)
        high = split_frequency_components_fft(x, torch.zeros(8, 8), is_low=False)
        assert torch.count_nonzero(low) == 0
        assert torch.allclose(high, x, atol=1e-4)

    def test_ones_mask_passes_low_untouched(self):
        """Filter of ones: LPF(x) == x (up to FFT round trip)."""
        torch.manual_seed(2)
        x = torch.randn(1, 3, 8, 8)
        low = split_frequency_components_fft(x, torch.ones(8, 8), is_low=True)
        assert torch.allclose(low, x, atol=1e-4)

    def test_fp16_input_no_nan(self):
        """fp16 latents must not produce NaN (fp32 upcast inside the split)."""
        torch.manual_seed(3)
        x = torch.randn(1, 4, 16, 16, dtype=torch.float16)
        f = butterworth_low_pass_filter_2d((16, 16), ratio=0.2, dtype=torch.float32)
        out = split_frequency_components_fft(x, f)
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all()

    def test_dc_only_input_survives_low_pass(self):
        """A constant image is pure DC: the low-pass component is the image."""
        x = torch.full((1, 2, 8, 8), 0.7)
        f = butterworth_low_pass_filter_2d((8, 8), ratio=0.2)
        low = split_frequency_components_fft(x, f, is_low=True)
        assert torch.allclose(low, x, atol=1e-4)

    def test_parity_isotropic_norms(self):
        """Energy sanity: ||low||^2 + ||high||^2 == ||x||^2 for a real mask
        (Parseval-adjacent, exact only for 0/1 masks)."""
        torch.manual_seed(4)
        x = torch.randn(1, 2, 16, 16)
        h, w = 16, 16
        # Binary mask (ideal low-pass) keeps the energy split exact.
        ys = torch.linspace(-1.0, 1.0 - 2.0 / h, h)
        xs = torch.linspace(-1.0, 1.0 - 2.0 / w, w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        f = ((yy.square() + xx.square()) <= 0.2 ** 2).float()
        low = split_frequency_components_fft(x, f, is_low=True)
        high = split_frequency_components_fft(x, f, is_low=False)
        e_total = x.square().sum().item()
        e_split = low.square().sum().item() + high.square().sum().item()
        assert e_split == pytest.approx(e_total, rel=1e-3, abs=1e-3), (
            f"energy split {e_split} != total {e_total}"
        )

    def test_output_finite_random_shapes(self):
        torch.manual_seed(5)
        for h, w in ((8, 8), (12, 20), (32, 32)):
            x = torch.randn(2, 3, h, w)
            f = butterworth_low_pass_filter_2d((h, w), ratio=0.4)
            out = split_frequency_components_fft(x, f)
            assert out.shape == x.shape
            assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Stage sigma schedule
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildStageSigmas:
    SIGMAS = flux_like_sigmas(30, 1.15)

    @pytest.mark.parametrize("tau", [0.6, 0.3, 0.95, 0.1])
    def test_stage_starts_at_or_below_tau(self, tau):
        stage = build_stage_sigmas(self.SIGMAS, tau, 16)
        assert float(stage[0]) <= tau + 1e-6, (
            f"entry sigma {float(stage[0])} must be <= tau {tau}"
        )

    @pytest.mark.parametrize("tau", [0.6, 0.3, 0.95])
    def test_stage_ends_at_zero_and_is_descending(self, tau):
        stage = build_stage_sigmas(self.SIGMAS, tau, 16)
        assert float(stage[-1]) == 0.0
        # strictly descending over the interior (trailing 0 is the floor)
        interior = stage[:-1]
        assert torch.all(interior[:-1] > interior[1:]), (
            f"interior sigmas must strictly decrease: {interior.tolist()}"
        )
        assert not torch.isnan(stage).any()

    def test_stage_length_matches_steps(self):
        stage = build_stage_sigmas(self.SIGMAS, tau=0.6, steps=16)
        assert stage.numel() == 17, "16 transitions -> len == steps + 1"

    def test_stage_uses_schedule_spacing(self):
        """Tail sigmas must be a suffix slice of the model's own schedule
        (the reference's dlfg_timesteps semantics). steps=16 gives an entry
        plus 15 schedule sigmas, then 0."""
        stage = build_stage_sigmas(self.SIGMAS, tau=0.6, steps=16)
        interior = self.SIGMAS[self.SIGMAS > 0]
        tail = stage[1:-1]
        assert tail.numel() == 15
        expected = interior[-15:]
        assert torch.allclose(tail, expected, atol=1e-6)

    def test_stage_small_tau_only_low_sigmas(self):
        """tau below all walkable sigmas still yields a valid short stage."""
        stage = build_stage_sigmas(self.SIGMAS, tau=0.05, steps=16)
        assert float(stage[0]) <= 0.05 + 1e-6
        assert stage.numel() >= 2

    def test_tau_above_schedule_clamps(self):
        """tau above some schedule sigmas but below sigma_max enters at the
        largest sigma <= tau (tau=0.99 clamps below sigma_max=1.0)."""
        stage = build_stage_sigmas(self.SIGMAS, tau=0.99, steps=16)
        interior = self.SIGMAS[self.SIGMAS > 0]
        below = interior[interior <= 0.99 + 1e-9]
        assert float(stage[0]) == pytest.approx(float(below.max()), abs=1e-6)

    def test_tau_at_sigma_max_uses_full_schedule(self):
        stage = build_stage_sigmas(self.SIGMAS, tau=1.0, steps=16)
        assert float(stage[0]) == pytest.approx(float(self.SIGMAS[0]), abs=1e-6)

    def test_tau_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            build_stage_sigmas(self.SIGMAS, tau=0.0, steps=16)

    def test_steps_below_one_raises(self):
        with pytest.raises(ValueError, match="steps"):
            build_stage_sigmas(self.SIGMAS, tau=0.6, steps=0)

    def test_non_descending_schedule_rejected(self):
        with pytest.raises(ValueError, match="non-increasing"):
            build_stage_sigmas(torch.tensor([0.1, 0.5, 0.0]), tau=0.3, steps=4)

    def test_too_short_schedule_rejected(self):
        with pytest.raises(ValueError, match=">= 2"):
            build_stage_sigmas(torch.tensor([1.0]), tau=0.5, steps=2)


@pytest.mark.unit
class TestAlignmentScales:
    def test_first_is_one(self):
        stage = build_stage_sigmas(flux_like_sigmas(), tau=0.6, steps=16)
        alpha, beta = alignment_scales(stage)
        assert alpha[0].item() == pytest.approx(1.0)
        assert beta[0].item() == pytest.approx(1.0)

    def test_decay_monotone_to_zero(self):
        stage = build_stage_sigmas(flux_like_sigmas(), tau=0.6, steps=16)
        alpha, _ = alignment_scales(stage)
        assert torch.all(alpha[:-1] >= alpha[1:]), "scales must decay"
        assert alpha[-1].item() == pytest.approx(0.0, abs=1e-6)

    def test_formula_sigma_over_entry(self):
        """alpha[i] == sigma_i / sigma_entry for every i (D5)."""
        stage = build_stage_sigmas(flux_like_sigmas(), tau=0.6, steps=16)
        alpha, beta = alignment_scales(stage)
        expected = torch.clamp(stage / stage[0], min=0.0)
        assert torch.allclose(alpha, expected, atol=1e-6)
        assert torch.allclose(beta, expected, atol=1e-6)

    def test_alpha_equals_beta(self):
        """The paper uses the same t/tau schedule for both alignments."""
        stage = build_stage_sigmas(flux_like_sigmas(), tau=0.3, steps=8)
        alpha, beta = alignment_scales(stage)
        assert torch.allclose(alpha, beta)

    def test_zero_entry_raises(self):
        with pytest.raises(ValueError, match="positive"):
            alignment_scales(torch.tensor([0.0, 0.0]))


# ---------------------------------------------------------------------------
# Step 3 — TrajectoryDict + base_trajectory
# ---------------------------------------------------------------------------

from src.hiflow import HiFlowConfig, TrajectoryDict, base_trajectory  # noqa: E402


@pytest.mark.unit
class TestTrajectoryDict:
    def test_put_get_roundtrip(self):
        traj = TrajectoryDict()
        x0 = torch.randn(1, 4, 8, 8)
        traj.put(0.5679, x0)
        out = traj.get(0.5679)
        assert torch.equal(out, x0)

    def test_stored_on_cpu(self):
        """D8: entries park on CPU so a long 4K trajectory stays off-GPU."""
        traj = TrajectoryDict()
        traj.put(0.5, torch.randn(1, 2, 4, 4))
        stored = traj.as_dict()[0.5]
        assert stored.device.type == "cpu"

    def test_get_missing_key_message(self):
        traj = TrajectoryDict()
        traj.put(0.5, torch.zeros(1))
        with pytest.raises(KeyError, match="no trajectory entry"):
            traj.get(0.123)

    def test_nearest_exact(self):
        traj = TrajectoryDict()
        x0 = torch.randn(1, 2, 4, 4)
        traj.put(0.4321, x0)
        key, out = traj.nearest(0.4321)
        assert key == 0.4321
        assert torch.equal(out, x0)

    def test_nearest_off_key_within_tol(self):
        traj = TrajectoryDict()
        traj.put(0.4, torch.zeros(1))
        traj.put(0.2, torch.ones(1))
        # 0.0004 from 0.4 — within the 1e-3 tolerance, far from 0.2.
        key, out = traj.nearest(0.4004)
        assert key == 0.4

    def test_nearest_tol_exceeded_raises(self):
        traj = TrajectoryDict()
        traj.put(0.4, torch.zeros(1))
        with pytest.raises(ValueError, match="within"):
            traj.nearest(0.6, tol=1e-3)

    def test_nearest_empty_dict_raises(self):
        traj = TrajectoryDict()
        with pytest.raises(ValueError, match="reference trajectory"):
            traj.nearest(0.5)

    def test_nearest_moves_to_device(self):
        traj = TrajectoryDict()
        traj.put(0.5, torch.zeros(1))
        _, out = traj.nearest(0.5, device=torch.device("cpu"))
        assert out.device.type == "cpu"

    def test_sigma_rounding(self):
        """Keys are 6-decimal rounded: 0.499999999 vs 0.5 share one slot."""
        traj = TrajectoryDict()
        traj.put(0.499999999, torch.zeros(1))
        assert torch.equal(traj.get(0.5), torch.zeros(1))
        assert len(traj) == 1

    def test_sigmas_sorted(self):
        traj = TrajectoryDict()
        for s in (0.3, 0.9, 0.1):
            traj.put(s, torch.zeros(1))
        assert traj.sigmas() == [0.1, 0.3, 0.9]


@pytest.mark.unit
class TestBaseTrajectory:
    CFG = HiFlowConfig(steps=5)

    def _sigmas(self):
        return torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.0])

    def test_records_every_sigma_plus_endpoint(self):
        sigmas = self._sigmas()
        traj_out = {}

        def predict(x, s):
            traj_out[s] = True
            return torch.zeros_like(x)

        _, traj = base_trajectory(
            torch.randn(1, 4, 8, 8), sigmas, predict, self.CFG)
        for s in sigmas[:-1].tolist():
            assert traj.get(s) is not None, f"missing recorded sigma {s}"
        assert traj.get(0.0) is not None, "endpoint at sigma 0 must be stored"
        assert len(traj) == 6

    def test_final_matches_reference_euler(self):
        """Euler fidelity tripwire: analytic model, hand-computed walk.

        predict_x0(x, s) = 0.5 * x  =>  v = (x - 0.5x)/s = x/(2s), so
        x_next = x + (s_next - s) * x/(2s)  — deterministic closed form.
        """
        sigmas = self._sigmas()
        x0 = torch.randn(1, 2, 4, 4)

        x = x0.clone()
        for i in range(len(sigmas) - 1):
            s = float(sigmas[i])
            v = x / (2.0 * s)
            x = x + v * (float(sigmas[i + 1]) - s)

        _, traj = base_trajectory(
            x0.clone(), sigmas, lambda x, s: 0.5 * x, self.CFG)
        final = traj.get(0.0)
        assert torch.allclose(final, x, atol=1e-5), (
            "base_trajectory must reproduce the hand-rolled Euler walk"
        )

    def test_calls_predict_in_sigma_order(self):
        calls = []

        def predict(x, s):
            calls.append(s)
            return torch.zeros_like(x)

        base_trajectory(
            torch.randn(1, 2, 4, 4), self._sigmas(), predict, self.CFG)
        # float32 sigmas round-trip with tiny artifacts — compare 6-decimal.
        assert [round(s, 6) for s in calls] == [1.0, 0.8, 0.6, 0.4, 0.2]

    def test_records_time_matched_x0(self):
        """Each stored entry must be the x0 predicted AT that sigma, not a
        later state (time matching is the whole point of the reference)."""
        seen = {}

        def predict(x, s):
            x0 = x + s  # sigma-identifiable values
            seen[s] = x0.clone()
            return x0

        x_start = torch.zeros(1, 2, 2, 2)
        _, traj = base_trajectory(x_start, self._sigmas(), predict, self.CFG)
        for s, x0_at in seen.items():
            assert torch.equal(traj.get(s), x0_at)

    def test_progress_callback_events(self):
        events = []
        base_trajectory(
            torch.randn(1, 2, 4, 4), self._sigmas(),
            lambda x, s: torch.zeros_like(x), self.CFG,
            progress_callback=lambda i, total, stage: events.append((i, total, stage)),
        )
        assert events == [
            (0, 5, -1), (1, 5, -1), (2, 5, -1), (3, 5, -1), (4, 5, -1),
        ]

    def test_no_nan_fp16_input(self):
        x = torch.randn(1, 2, 4, 4, dtype=torch.float16)
        out, traj = base_trajectory(
            x, self._sigmas(), lambda x_, s: 0.5 * x_, self.CFG)
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all()
        assert torch.isfinite(traj.get(0.0)).all()

    def test_output_shape_preserved(self):
        x = torch.randn(2, 3, 16, 12)
        out, _ = base_trajectory(
            x, self._sigmas(), lambda x_, s: 0.5 * x_, self.CFG)
        assert out.shape == x.shape
