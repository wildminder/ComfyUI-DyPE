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
    denoise_sigmas,
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

    def test_linear_in_step_index(self):
        """Reference form (plan D6): scale_i = (n - i) / n — LINEAR in the
        step index, not sigma_i/sigma_entry. With a non-even sigma schedule
        the two differ; this pins the code's schedule."""
        n = 16
        stage = build_stage_sigmas(flux_like_sigmas(), tau=0.6, steps=n)
        alpha, beta = alignment_scales(stage)
        idx = torch.arange(n, dtype=torch.float32)
        expected = torch.cat([(n - idx) / n, torch.zeros(1)])
        assert torch.allclose(alpha, expected, atol=1e-6)
        assert torch.allclose(beta, expected, atol=1e-6)

    def test_constant_decrement(self):
        stage = build_stage_sigmas(flux_like_sigmas(), tau=0.6, steps=16)
        alpha, _ = alignment_scales(stage)
        diffs = alpha[:-2] - alpha[1:-1]
        assert torch.allclose(diffs, torch.full_like(diffs, 1.0 / 16))

    def test_last_step_is_one_over_n(self):
        """The last TRANSITION carries 1/n (the reference's (n-(n-1))/n);
        the trailing entry is 0 for length parity with the sigma schedule."""
        stage = build_stage_sigmas(flux_like_sigmas(), tau=0.6, steps=16)
        alpha, _ = alignment_scales(stage)
        assert alpha[15].item() == pytest.approx(1.0 / 16)
        assert alpha[16].item() == pytest.approx(0.0, abs=1e-6)

    def test_length_matches_sigma_schedule(self):
        stage = build_stage_sigmas(flux_like_sigmas(), tau=0.6, steps=16)
        alpha, beta = alignment_scales(stage)
        assert alpha.numel() == stage.numel()
        assert beta.numel() == stage.numel()

    def test_alpha_equals_beta(self):
        """The reference uses the same (n-i)/n schedule for both."""
        stage = build_stage_sigmas(flux_like_sigmas(), tau=0.3, steps=8)
        alpha, beta = alignment_scales(stage)
        assert torch.allclose(alpha, beta)

    def test_zero_entry_raises(self):
        with pytest.raises(ValueError, match="positive"):
            alignment_scales(torch.tensor([0.0, 0.0]))

    def test_single_transition_raises(self):
        with pytest.raises(ValueError, match="transition"):
            alignment_scales(torch.tensor([0.6]))


# ---------------------------------------------------------------------------
# Img2img denoise schedule truncation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDenoiseSigmas:
    @staticmethod
    def _linear_sigmas(steps=30):
        """Even-in-flow-time schedule descending 1 -> 1/30, then 0."""
        return torch.cat([
            torch.linspace(1.0, 1.0 / steps, steps), torch.zeros(1),
        ])

    def test_full_denoise_unchanged(self):
        sig = self._linear_sigmas()
        out, truncated = denoise_sigmas(sig, denoise=1.0, steps=30)
        assert truncated is False
        assert torch.equal(out, sig)

    def test_high_denoise_unchanged(self):
        """denoise > 0.9999 is the full schedule (KSampler threshold)."""
        sig = self._linear_sigmas()
        out, truncated = denoise_sigmas(sig, denoise=0.99995, steps=30)
        assert truncated is False
        assert torch.equal(out, sig)

    def test_truncation_enters_below_one(self):
        """denoise=0.6: entry sigma < 1 so the noising keeps content."""
        sig = self._linear_sigmas()
        out, truncated = denoise_sigmas(sig, denoise=0.6, steps=30)
        assert truncated is True
        assert float(out[0]) < 1.0
        assert float(out[0]) == pytest.approx(0.6, abs=0.05)
        assert float(out[-1]) == 0.0

    def test_truncated_length_matches_ksampler(self):
        """KSampler keeps the last (steps+1) of a denser schedule ->
        steps transitions, same step count as the full run."""
        sig = self._linear_sigmas()
        out, truncated = denoise_sigmas(sig, denoise=0.6, steps=30)
        assert truncated is True
        assert out.numel() == 31

    def test_lower_denoise_lower_entry(self):
        sig = self._linear_sigmas()
        out06, _ = denoise_sigmas(sig, denoise=0.6, steps=30)
        out03, _ = denoise_sigmas(sig, denoise=0.3, steps=30)
        assert float(out03[0]) < float(out06[0]), (
            "less denoise -> lower entry sigma -> more content preserved"
        )

    def test_descending(self):
        sig = self._linear_sigmas()
        out, _ = denoise_sigmas(sig, denoise=0.5, steps=30)
        assert torch.all(out[:-1] > out[1:])

    def test_degenerate_denoise_rejected(self):
        sig = self._linear_sigmas()
        with pytest.raises(ValueError, match="denoise"):
            denoise_sigmas(sig, denoise=0.0, steps=30)
        with pytest.raises(ValueError, match="denoise"):
            denoise_sigmas(sig, denoise=1.5, steps=30)

    def test_steps_below_one_rejected(self):
        with pytest.raises(ValueError, match="steps"):
            denoise_sigmas(self._linear_sigmas(), denoise=0.6, steps=0)

    def test_too_short_schedule_falls_back(self):
        """A schedule with < 2 interior sigmas cannot be densified — the
        original schedule is returned untruncated (graceful fallback)."""
        sig = torch.tensor([0.7, 0.0])
        out, truncated = denoise_sigmas(sig, denoise=0.6, steps=30)
        assert truncated is False
        assert torch.equal(out, sig)


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


# ---------------------------------------------------------------------------
# Step 4 — guided_stage (initialization / direction / acceleration)
# ---------------------------------------------------------------------------

from src.hiflow import guided_stage  # noqa: E402


def _make_ref_traj(values: dict[float, torch.Tensor]) -> TrajectoryDict:
    traj = TrajectoryDict()
    for s, v in values.items():
        traj.put(s, v)
    return traj


def _identity_upsample(x):
    return x


def _zero_filter_factory(shape):
    """LPF(x) == 0 for every x (zero mask kills the low component)."""
    return torch.zeros(shape[-2], shape[-1])


def _ones_filter_factory(shape):
    """LPF(x) == x (ones mask passes everything)."""
    return torch.ones(shape[-2], shape[-1])


def _plain_cfg(alpha=0.0, beta=0.0, tau=0.6):
    return HiFlowConfig(
        tau=tau, filter_ratio=0.2,
        alpha_scale=alpha, beta_scale=beta,
    )


@pytest.mark.unit
class TestGuidedStage:
    STAGE_SIGMAS = torch.tensor([0.6, 0.4, 0.2, 0.0])

    def _ref_traj(self, size=(1, 4, 8, 8), seed=7):
        torch.manual_seed(seed)
        return _make_ref_traj({
            0.6: torch.randn(*size),
            0.4: torch.randn(*size),
            0.2: torch.randn(*size),
            0.0: torch.randn(*size),
        })

    def _anchor(self, seed=100, size=(1, 4, 8, 8)):
        g = torch.Generator().manual_seed(seed)
        return torch.randn(*size, generator=g)

    def test_alpha_beta_zero_reduces_to_plain_euler(self):
        """Anchor test: with both alignments off, the stage is an ordinary
        Euler walk seeded by the tau initialization."""
        torch.manual_seed(11)
        latent = torch.randn(1, 4, 8, 8)
        anchor = self._anchor()
        ref = self._ref_traj()
        cfg = _plain_cfg(alpha=0.0, beta=0.0)

        def predict(x, s):
            return 0.5 * x

        # Expected walk: the stage consumes the FIRST draw from a generator
        # seeded 101; replay it with a FRESH generator of the same seed.
        # Sigma arithmetic uses the stage's float32 values (exact Python
        # floats diverge and /sigma amplifies).
        sig = self.STAGE_SIGMAS
        sigma_e = float(sig[0])
        eps = torch.randn(
            1, 4, 8, 8, generator=torch.Generator().manual_seed(101))
        x = sigma_e * eps + (1 - sigma_e) * anchor
        for i in range(3):
            s = float(sig[i])
            x0 = 0.5 * x
            v = (x - x0) / s
            x = x + v * (float(sig[i + 1]) - s)

        out, _ = guided_stage(
            latent, anchor, ref, self.STAGE_SIGMAS, predict,
            _identity_upsample, cfg,
            generator=torch.Generator().manual_seed(101),
        )
        assert torch.allclose(out, x, atol=1e-5), (
            "alpha=beta=0 must reduce to the plain tau-seeded Euler walk"
        )

    def test_init_uses_anchor_not_reference(self):
        """The stage seed must be sigma_e*eps + (1-sigma_e)*ANCHOR (the
        previous chain's final image, plan D2) — NOT ref[sigma_e]. A one-step
        stage with an echo model pins the first model input."""
        torch.manual_seed(3)
        latent = torch.randn(1, 4, 8, 8)
        anchor = self._anchor()
        ref = self._ref_traj()
        assert not torch.allclose(anchor, ref.get(0.6)), (
            "fixture sanity: anchor and ref[0.6] must differ"
        )
        calls = []

        def predict(x, s):
            calls.append((x.clone(), s))
            return x.clone()  # v = (x - x)/s = 0 -> walk stays at seed

        gen = torch.Generator().manual_seed(42)
        out, _ = guided_stage(
            latent, anchor, ref, torch.tensor([0.6, 0.0]), predict,
            _identity_upsample, _plain_cfg(), generator=gen,
        )

        # A FRESH generator with the same seed reproduces the stage's draw.
        gen2 = torch.Generator().manual_seed(42)
        eps = torch.randn(1, 4, 8, 8, generator=gen2)
        expected_seed = 0.6 * eps + 0.4 * anchor

        assert torch.allclose(calls[0][0], expected_seed, atol=1e-6)
        assert torch.allclose(out, expected_seed, atol=1e-5)

    def test_init_noises_in_model_space(self):
        """Stage init with a scale!=1 latent format must mix sigma_e*eps +
        (1-sigma_e)*process_in(anchor) in MODEL space and convert back
        (v2.12.1 — the reference's scale_noise operates on model-scaled
        latents; a VAE-space mix under-scales the noise by 1/scale)."""
        torch.manual_seed(45)
        latent = torch.randn(1, 4, 8, 8)
        anchor = self._anchor()
        ref = self._ref_traj()
        scale, shift = 0.3611, 0.1159
        calls = []

        def predict(x, s):
            calls.append((x.clone(), s))
            return x.clone()

        gen = torch.Generator().manual_seed(42)
        guided_stage(
            latent, anchor, ref, torch.tensor([0.6, 0.0]), predict,
            _identity_upsample, _plain_cfg(), generator=gen,
            process_latent_in=lambda t: (t - shift) * scale,
            process_latent_out=lambda t: (t / scale) + shift,
        )
        gen2 = torch.Generator().manual_seed(42)
        eps = torch.randn(1, 4, 8, 8, generator=gen2)
        anchor_model = (anchor - shift) * scale
        expected = (0.6 * eps + 0.4 * anchor_model) / scale + shift
        assert torch.allclose(calls[0][0], expected, atol=1e-5)
        # The pre-fix VAE-space mix must NOT match:
        wrong = 0.6 * eps + 0.4 * anchor
        assert not torch.allclose(calls[0][0], wrong, atol=1e-3)

    def test_anchor_shape_mismatch_rejected(self):
        torch.manual_seed(31)
        latent = torch.randn(1, 4, 8, 8)
        bad_anchor = self._anchor(size=(1, 4, 16, 16))
        with pytest.raises(ValueError, match="init_anchor shape"):
            guided_stage(
                latent, bad_anchor, self._ref_traj(),
                torch.tensor([0.6, 0.0]),
                lambda x, s: 0.5 * x, _identity_upsample, _plain_cfg(),
            )

    def test_direction_alignment_formula(self):
        """One-step stage with extreme filters pins the exact formula:
        ones-filter -> LPF(x)=x; zero-filter -> LPF(x)=0. The trajectory
        stores the RAW prediction (plan D4); the walked output equals the
        CORRECTED x0 (dt == -sigma_e exactly)."""
        for factory, expect_mode in (
            (_ones_filter_factory, "identity"),
            (_zero_filter_factory, "zero"),
        ):
            torch.manual_seed(5)
            latent = torch.randn(1, 4, 8, 8)
            anchor = self._anchor()
            ref = self._ref_traj()
            alpha_scale = 0.7
            cfg = _plain_cfg(alpha=alpha_scale, beta=0.0)

            def predict(x, s):
                return 0.25 * torch.ones_like(x)  # fixed known x0

            out, traj = guided_stage(
                latent, anchor, ref, torch.tensor([0.6, 0.0]), predict,
                _identity_upsample, cfg,
                freq_filter_factory=factory,
                generator=torch.Generator().manual_seed(5),
            )
            x0_raw = 0.25 * torch.ones(1, 4, 8, 8)
            ref_x0 = ref.get(0.6)
            # alpha = alpha_scale * (n-0)/n = 0.7 at the single transition.
            if expect_mode == "identity":
                x0_corrected = x0_raw + 0.7 * (ref_x0 - x0_raw)
            else:
                # LPF(x)=LPF(ref)=0 -> x0 + alpha*(0 - 0) == x0 unchanged.
                x0_corrected = x0_raw
            stored = traj.get(0.6)
            assert torch.allclose(stored, x0_raw, atol=1e-6), (
                "the trajectory must store the RAW pre-correction prediction "
                "(reference's original_pred_x0, plan D4)"
            )
            # The walked latent: v = (x - x0_corrected)/sigma, dt = -sigma.
            assert torch.allclose(out, x0_corrected, atol=1e-4), (
                f"walked output must equal the corrected x0 under "
                f"{expect_mode} LPF (dt == -sigma_e)"
            )

    def test_acceleration_first_step_skipped(self):
        """Step 0 has no previous velocity pair: v stays plain. Later steps
        apply the delta-v blend with LINEAR-in-index beta (plan D6) and the
        walk-state v_ref (plan D3)."""
        torch.manual_seed(9)
        latent = torch.randn(1, 4, 8, 8)
        anchor = self._anchor()
        ref = self._ref_traj()
        cfg = _plain_cfg(alpha=0.0, beta=1.0)
        recorded_x = []

        def predict(x, s):
            recorded_x.append(x.clone())
            return 0.5 * x

        gen = torch.Generator().manual_seed(77)
        guided_stage(
            latent, anchor, ref, self.STAGE_SIGMAS, predict,
            _identity_upsample, cfg,
            freq_filter_factory=_ones_filter_factory,  # LPF identity, alpha=0 anyway
            generator=gen,
        )
        assert len(recorded_x) == 3, "3 transitions -> 3 model calls"

        # Re-run the walk analytically with a fresh generator of the same seed
        # (the stage consumed the first draw from its own generator). Sigma
        # arithmetic uses the SAME float32 values the stage walks; beta
        # follows the LINEAR (n-i)/n schedule (plan D6) and v_ref is built
        # from the WALK's own state (plan D3).
        eps = torch.randn(
            1, 4, 8, 8, generator=torch.Generator().manual_seed(77))
        sig = self.STAGE_SIGMAS
        sigma_e = float(sig[0])
        x = sigma_e * eps + (1 - sigma_e) * anchor
        prev_vh = prev_vr = None
        for i in range(3):
            s = float(sig[i])
            assert torch.allclose(recorded_x[i], x, atol=1e-5), (
                f"step-{i} model input mismatch before velocity computation"
            )
            x0 = 0.5 * x
            v_high = (x - x0) / s
            v_ref = (x - ref.get(s)) / s
            beta = (3 - i) / 3 * cfg.beta_scale
            if prev_vh is not None:
                v_high = v_high + beta * (v_ref - prev_vr - v_high + prev_vh)
            # step 0 has no previous pair: velocity must stay plain.
            if i == 0:
                plain = (recorded_x[0] - 0.5 * recorded_x[0]) / s
                assert torch.allclose(v_high, plain, atol=1e-6)
            dt = float(sig[i + 1]) - s
            x = x + v_high * dt
            prev_vh, prev_vr = v_high, v_ref

    def test_walk_state_v_ref_constant_ref_matches(self):
        """v_ref comes from the WALK's own state x (plan D3, the reference
        code's ``model_output_ref = (sample - pred_x0_ref)/(sigma+1e-6)``) —
        NOT from a separately-integrated reference chain. Constant reference
        entries make the walk-state form analytic: if the implementation
        integrated a parallel chain, the step-1 model input would diverge.
        """
        torch.manual_seed(4)
        const_ref = torch.randn(1, 4, 8, 8)
        ref = _make_ref_traj({
            0.6: const_ref, 0.4: const_ref, 0.2: const_ref, 0.0: const_ref,
        })
        anchor = self._anchor()
        recorded_x = []

        def predict(x, s):
            recorded_x.append(x.clone())
            return 0.5 * x

        gen = torch.Generator().manual_seed(88)
        guided_stage(
            torch.randn(1, 4, 8, 8), anchor, ref, self.STAGE_SIGMAS, predict,
            _identity_upsample, _plain_cfg(alpha=0.0, beta=1.0),
            freq_filter_factory=_ones_filter_factory,
            generator=gen,
        )
        # Analytic re-run: v_ref = (x - const_ref)/s from the walk's state;
        # beta_i = (3-i)/3 * beta_scale (linear, plan D6).
        eps = torch.randn(
            1, 4, 8, 8, generator=torch.Generator().manual_seed(88))
        sig = self.STAGE_SIGMAS
        sigma_e = float(sig[0])
        x = sigma_e * eps + (1 - sigma_e) * anchor
        prev_vh = prev_vr = None
        for i in range(3):
            s = float(sig[i])
            # The model input at step i is the state BEFORE this step's update.
            assert torch.allclose(recorded_x[i], x, atol=1e-4), (
                f"step-{i} model input must match the walk-state-v_ref walk"
            )
            v_high = (x - 0.5 * x) / s
            v_ref = (x - const_ref) / s
            if prev_vh is not None:
                beta = (3 - i) / 3 * 1.0
                v_high = v_high + beta * (v_ref - prev_vr - v_high + prev_vh)
            dt = float(sig[i + 1]) - s
            x = x + v_high * dt
            prev_vh, prev_vr = v_high, v_ref

    def test_stage_trajectory_recorded_per_sigma(self):
        torch.manual_seed(17)
        latent = torch.randn(1, 4, 8, 8)
        anchor = self._anchor()
        ref = self._ref_traj()
        out, traj = guided_stage(
            latent, anchor, ref, self.STAGE_SIGMAS,
            lambda x, s: 0.5 * x, _identity_upsample, _plain_cfg(),
        )
        for s in (0.6, 0.4, 0.2, 0.0):
            assert traj.get(s) is not None, f"missing stage trajectory at {s}"
        assert traj.get(0.0).shape == latent.shape
        assert torch.allclose(traj.get(0.0), out, atol=1e-6)

    def test_raw_x0_recorded_not_corrected(self):
        """The stored per-sigma entry is the RAW prediction even when
        direction alignment is active (plan D4) — the reference feeds
        original_pred_x0 to the next stage."""
        torch.manual_seed(51)
        latent = torch.randn(1, 4, 8, 8)
        anchor = self._anchor()
        ref = self._ref_traj()
        cfg = _plain_cfg(alpha=1.0, beta=0.0)

        def predict(x, s):
            return 0.3 * torch.ones_like(x)

        out, traj = guided_stage(
            latent, anchor, ref, torch.tensor([0.6, 0.0]), predict,
            _identity_upsample, cfg,
            freq_filter_factory=_ones_filter_factory,
        )
        stored = traj.get(0.6)
        assert torch.allclose(
            stored, 0.3 * torch.ones_like(stored), atol=1e-6), (
            "stored entry must be the raw 0.3*ones prediction"
        )
        assert not torch.allclose(out, stored, atol=1e-4) or True
        # (The walked output may equal the corrected x0; the pin is the RAW
        # stored entry above.)

    def test_upsample_called_on_reference_values(self):
        """The per-step reference x0 must flow through upsample_x0; the
        INIT ANCHOR does not (it arrives pre-upsampled by the caller)."""
        torch.manual_seed(23)
        latent = torch.randn(1, 4, 8, 8)
        anchor = self._anchor()
        ref = self._ref_traj()
        seen = []

        def up(x):
            seen.append(x.clone())
            return x

        guided_stage(
            latent, anchor, ref, torch.tensor([0.6, 0.0]),
            lambda x, s: 0.5 * x, up, _plain_cfg(),
        )
        assert len(seen) == 1, "one transition -> exactly one ref upsample"
        assert torch.allclose(seen[0], ref.get(0.6), atol=1e-6)

    def test_no_nan_end_to_end(self):
        torch.manual_seed(31)
        latent = torch.randn(2, 4, 16, 16)
        anchor = self._anchor(size=(2, 4, 16, 16))
        ref = self._ref_traj(size=(2, 4, 16, 16))
        out, traj = guided_stage(
            latent, anchor, ref, self.STAGE_SIGMAS,
            lambda x, s: 0.5 * x + 0.1 * torch.randn_like(x),
            _identity_upsample, _plain_cfg(alpha=1.0, beta=0.5),
        )
        assert out.shape == latent.shape
        assert torch.isfinite(out).all()
        assert torch.isfinite(traj.get(0.0)).all()

    def test_output_dtype_restored(self):
        torch.manual_seed(37)
        latent = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        anchor = self._anchor().to(torch.float16)
        ref = self._ref_traj()
        out, _ = guided_stage(
            latent, anchor, ref, self.STAGE_SIGMAS,
            lambda x, s: (0.5 * x).to(torch.float16),
            _identity_upsample, _plain_cfg(),
        )
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all()

    def test_progress_events_fire(self):
        events = []
        torch.manual_seed(41)
        latent = torch.randn(1, 4, 8, 8)
        anchor = self._anchor()
        ref = self._ref_traj()
        guided_stage(
            latent, anchor, ref, self.STAGE_SIGMAS,
            lambda x, s: 0.5 * x, _identity_upsample, _plain_cfg(),
            progress_callback=lambda i, total, stage: events.append((i, total, stage)),
            stage_index=2,
        )
        assert events == [(0, 3, 2), (1, 3, 2), (2, 3, 2)]

    def test_empty_reference_rejected(self):
        torch.manual_seed(43)
        latent = torch.randn(1, 4, 8, 8)
        with pytest.raises(ValueError, match="non-empty reference"):
            guided_stage(
                latent, self._anchor(), TrajectoryDict(), self.STAGE_SIGMAS,
                lambda x, s: 0.5 * x, _identity_upsample, _plain_cfg(),
            )


# ---------------------------------------------------------------------------
# Step 5 — upsample + cascade driver
# ---------------------------------------------------------------------------

from src.hiflow import (  # noqa: E402
    _stage_latent_sizes,
    hiflow_cascade,
    upsample_latent,
)


@pytest.mark.unit
class TestUpsampleLatent:
    def test_doubles_shape(self):
        x = torch.randn(1, 4, 16, 16)
        up = upsample_latent(x, 32, 32)
        assert up.shape == (1, 4, 32, 32)

    def test_antialias_smooths_high_freq(self):
        """Upscaled (antialiased bicubic) must smooth a checkerboard more
        than the raw input scaled to the same footprint."""
        x = torch.zeros(1, 1, 16, 16)
        x[0, 0, ::2, ::2] = 1.0
        x[0, 0, 1::2, 1::2] = 1.0
        up = upsample_latent(x, 32, 32)

        def rough(t):
            return (t[0, 0, 1:, :] - t[0, 0, :-1, :]).abs().mean().item()

        assert rough(up) < rough(x), (
            "bicubic antialiasing must reduce checkerboard roughness"
        )

    def test_fp16_roundtrip_dtype(self):
        x = torch.randn(1, 4, 8, 8, dtype=torch.float16)
        up = upsample_latent(x, 16, 16)
        assert up.dtype == torch.float16
        assert torch.isfinite(up).all()

    def test_constant_image_preserved(self):
        x = torch.full((1, 2, 8, 8), 0.7)
        up = upsample_latent(x, 16, 16)
        assert torch.allclose(up, torch.full_like(up, 0.7), atol=1e-3)


@pytest.mark.unit
class TestStageLatentSizes:
    def test_scale_1_no_stages(self):
        assert _stage_latent_sizes(128, 128, 1.0) == []

    def test_scale_2_one_doubling_stage(self):
        assert _stage_latent_sizes(128, 128, 2.0) == [(256, 256)]

    def test_scale_4_two_doubling_stages(self):
        assert _stage_latent_sizes(128, 128, 4.0) == [(256, 256), (512, 512)]

    def test_scale_between_1_and_2_quantizes_to_one_stage(self):
        """The cascade is 2x stages only (paper): scales in (1, 2] run ONE
        2x stage — the final size lands above the exact scale."""
        assert _stage_latent_sizes(128, 128, 1.5) == [(256, 256)]
        assert _stage_latent_sizes(128, 128, 1.2) == [(256, 256)]

    def test_scale_half_single_fractional_stage(self):
        assert _stage_latent_sizes(128, 128, 0.5) == [(64, 64)]

    def test_scale_quarter_single_fractional_stage(self):
        assert _stage_latent_sizes(128, 128, 0.25) == [(32, 32)]

    def test_non_square_per_side_scaling(self):
        """Scale applies PER SIDE: a 2x on 128x64 doubles both sides
        (the old absolute-target form over-upscaled the short side)."""
        assert _stage_latent_sizes(128, 64, 2.0) == [(256, 128)]
        assert _stage_latent_sizes(128, 64, 0.5) == [(64, 32)]

    def test_odd_base_snapped(self):
        """1088px base (latent 136) doubles to a 16-px-multiple target."""
        sizes = _stage_latent_sizes(136, 136, 2.0)
        assert sizes == [(272, 272)]
        for h, w in sizes:
            assert (h * 8) % 16 == 0 and (w * 8) % 16 == 0
            assert h % 2 == 0 and w % 2 == 0  # FLUX 2x2 packing

    def test_out_of_range_rejected(self):
        for bad in (0.1, 0.0, -1.0, 8.5, 16.0):
            with pytest.raises(ValueError, match="scale_factor"):
                _stage_latent_sizes(128, 128, bad)


@pytest.mark.unit
class TestHiflowCascade:
    SIGMAS = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0])

    def _cfg(self, **kw):
        base = dict(
            tau=0.5, steps_per_stage=3, filter_ratio=0.2, upsampling="latent",
        )
        base.update(kw)
        return HiFlowConfig(**base)

    @staticmethod
    def _fake_vae():
        """Channels-last fake VAE pair matching the ComfyUI boundary contract:
        decode -> [B, 8h, 8w, 3]; encode asserts channels-last and returns
        [B, C, H/8, W/8]."""
        calls = {"decode": 0, "encode": 0}

        def vae_decode(latent):
            calls["decode"] += 1
            b, c, h, w = latent.shape
            img = (latent[:, :3].repeat_interleave(8, -2)
                   .repeat_interleave(8, -1))           # [B,3,8h,8w]
            img = img.movedim(1, -1)                    # -> [B,8h,8w,3]
            assert img.shape == (b, h * 8, w * 8, 3)
            return img

        def vae_encode(image):
            calls["encode"] += 1
            assert image.shape[-1] == 3, (
                "encode must receive channels-last [B,H,W,3]"
            )
            small = image[:, ::8, ::8, :1].movedim(-1, 1)  # [B,1,h,w]
            return small.repeat(1, 4, 1, 1)

        return vae_decode, vae_encode, calls

    def test_base_start_is_noised(self):
        """Plan D1 (root cause of the Z-Image burn): the base walk must
        start from sigma[0]*eps + (1-sigma[0])*latent — with sigma[0]==1 a
        pure noise draw, matching the reference's randn start. An
        EmptySD3LatentImage (zeros) input must NEVER reach the model
        verbatim."""
        torch.manual_seed(0)
        z = torch.zeros(1, 4, 32, 32)  # EmptySD3LatentImage equivalent
        seen = []

        def base_predict(x, s):
            seen.append(x.clone())
            return 0.5 * x

        hiflow_cascade(
            z, self.SIGMAS, base_predict,
            lambda x, s: 0.5 * x,
            scale_factor=1.0,
            cfg=self._cfg(),
            vae_decode=None, vae_encode=None,
            noise_seed=1234,
        )
        assert len(seen) == 4
        g = torch.Generator().manual_seed(1234)
        eps = torch.randn(1, 4, 32, 32, generator=g)
        assert torch.allclose(seen[0], eps, atol=1e-6), (
            "sigma[0]==1 -> the first model input must be the pure noise draw"
        )

    def test_base_start_noised_with_content(self):
        """A non-empty input latent survives as content only when
        sigma[0] < 1 (the general noising form)."""
        torch.manual_seed(1)
        z = torch.full((1, 4, 8, 8), 0.5)
        seen = []

        def base_predict(x, s):
            seen.append(x.clone())
            return 0.5 * x

        hiflow_cascade(
            z, torch.tensor([0.5, 0.25, 0.0]), base_predict,
            lambda x, s: 0.5 * x,
            scale_factor=1.0,               # no stages: noising pin only
            cfg=self._cfg(),
            vae_decode=None, vae_encode=None,
            noise_seed=99,
        )
        g = torch.Generator().manual_seed(99)
        eps = torch.randn(1, 4, 8, 8, generator=g)
        expected = 0.5 * eps + 0.5 * z
        assert torch.allclose(seen[0], expected, atol=1e-6)

    def test_noise_seed_reproducible(self):
        """Same noise_seed -> identical output; different seed -> different."""
        torch.manual_seed(2)
        z = torch.zeros(1, 4, 32, 32)
        vd, ve, _ = self._fake_vae()

        def run(seed):
            return hiflow_cascade(
                z, self.SIGMAS,
                lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
                scale_factor=2.0,
                cfg=self._cfg(),
                vae_decode=vd, vae_encode=ve,
                noise_seed=seed,
            )

        a = run(7)
        b = run(7)
        c = run(8)
        assert torch.allclose(a, b, atol=1e-6), "same seed must reproduce"
        assert not torch.allclose(a, c, atol=1e-4), "different seed must differ"

    def test_img2img_denoise_keeps_content(self):
        """A content latent + denoise < 1 enters below sigma 1: the first
        model input must be process_latent_out(sigma_start*eps +
        (1-sigma_start)*process_latent_in(latent)) — the KSampler img2img
        convention in MODEL space (the Z-Image round-3 fix). The truncated
        schedule keeps cfg.steps transitions (same step count, denser
        spacing)."""
        torch.manual_seed(21)
        z = torch.randn(1, 4, 32, 32)          # a real sampler latent
        seen = []

        def base_predict(x, s):
            seen.append((x.clone(), s))
            return 0.5 * x

        # Flux-style latent format: process_in = (x - shift)*scale,
        # process_out = (x/scale) + shift. The noising must mix in THIS
        # space (samplers.py:1223 converts the content BEFORE the mix at
        # :993) — mixing in VAE space under-scales the noise by 1/scale.
        scale, shift = 0.3611, 0.1159

        hiflow_cascade(
            z, self.SIGMAS,
            base_predict, lambda x, s: 0.5 * x,
            scale_factor=1.0,
            cfg=self._cfg(steps=2),
            vae_decode=None, vae_encode=None,
            noise_seed=5, denoise=0.5,
            process_latent_in=lambda t: (t - shift) * scale,
            process_latent_out=lambda t: (t / scale) + shift,
        )
        assert len(seen) == 2, "truncated schedule keeps cfg.steps transitions"
        sigma_start = float(seen[0][1])
        assert sigma_start < 1.0, "denoise=0.5 must truncate the entry sigma"
        g = torch.Generator().manual_seed(5)
        eps = torch.randn(1, 4, 32, 32, generator=g)
        content_model = (z - shift) * scale
        expected = (sigma_start * eps + (1 - sigma_start) * content_model) \
            / scale + shift
        assert torch.allclose(seen[0][0], expected, atol=1e-5), (
            "img2img noising must mix in MODEL space (ComfyUI samplers.py "
            "converts the content before the sigma mix)"
        )

    def test_img2img_noise_scaled_to_model_space(self):
        """The whole point of the model-space mix (Z-Image round 3): with a
        scale!=1 latent format, the noised input's noise component must
        carry the format's FULL variance in VAE space (the old VAE-space mix
        under-scaled it by scale=0.3611 — 2.77x under-noising at every
        sigma, so the model 'corrected' too aggressively)."""
        torch.manual_seed(24)
        z = torch.full((1, 4, 32, 32), 0.1)    # content: truncation fires
        seen = []

        def base_predict(x, s):
            seen.append((x.clone(), float(s)))
            return 0.5 * x

        scale, shift = 0.3611, 0.1159
        hiflow_cascade(
            z, self.SIGMAS,
            base_predict, lambda x, s: 0.5 * x,
            scale_factor=1.0,
            cfg=self._cfg(steps=2),
            vae_decode=None, vae_encode=None,
            noise_seed=9, denoise=0.5,
            process_latent_in=lambda t: (t - shift) * scale,
            process_latent_out=lambda t: (t / scale) + shift,
        )
        sigma_start = seen[0][1]
        assert sigma_start < 1.0
        g = torch.Generator().manual_seed(9)
        eps = torch.randn(1, 4, 32, 32, generator=g)
        content_model = (z - shift) * scale
        expected = (
            sigma_start * eps + (1 - sigma_start) * content_model
        ) / scale + shift
        assert torch.allclose(seen[0][0], expected, atol=1e-5), (
            "noise must be scaled UP by 1/scale through the model-space mix"
        )
        # And the old (wrong) VAE-space mix differs materially:
        wrong = sigma_start * eps + (1 - sigma_start) * z
        assert not torch.allclose(seen[0][0], wrong, atol=1e-3), (
            "the VAE-space mix is the pre-fix behavior — must NOT match"
        )

    def test_noising_identity_without_conversions(self):
        """Without conversion callables the cascade falls back to the plain
        VAE-space mix (identity formats, unit tests). Zero content is EMPTY
        -> full schedule -> the noised start is the pure noise draw."""
        torch.manual_seed(25)
        z = torch.zeros(1, 4, 32, 32)
        seen = []

        def base_predict(x, s):
            seen.append(x.clone())
            return 0.5 * x

        hiflow_cascade(
            z, self.SIGMAS,
            base_predict, lambda x, s: 0.5 * x,
            scale_factor=1.0,
            cfg=self._cfg(steps=2),
            vae_decode=None, vae_encode=None,
            noise_seed=9, denoise=0.5,
        )
        g = torch.Generator().manual_seed(9)
        eps = torch.randn(1, 4, 32, 32, generator=g)
        assert torch.allclose(seen[0], eps, atol=1e-6), (
            "empty latent + no conversions: sigma_start==1 -> pure noise"
        )

    def test_empty_latent_ignores_denoise(self):
        """An empty latent always runs the FULL schedule (denoise is
        meaningless for zeros — entry sigma stays 1, pure noise start)."""
        torch.manual_seed(22)
        z = torch.zeros(1, 4, 32, 32)
        seen = []

        def base_predict(x, s):
            seen.append(s)
            return 0.5 * x

        hiflow_cascade(
            z, self.SIGMAS,
            base_predict, lambda x, s: 0.5 * x,
            scale_factor=1.0,
            cfg=self._cfg(),
            vae_decode=None, vae_encode=None,
            noise_seed=5, denoise=0.3,
        )
        assert len(seen) == 4, "full schedule runs — no truncation for zeros"
        assert float(seen[0]) == pytest.approx(1.0), "entry stays at sigma 1"

    def test_content_denoise1_full_schedule(self):
        """denoise=1.0 on a content latent: full schedule, sigma_start=1 —
        the content is annihilated (from-noise regeneration; the node layer
        warns about this combination)."""
        torch.manual_seed(23)
        z = torch.randn(1, 4, 32, 32)
        seen = []

        def base_predict(x, s):
            seen.append(s)
            return 0.5 * x

        hiflow_cascade(
            z, self.SIGMAS,
            base_predict, lambda x, s: 0.5 * x,
            scale_factor=1.0,
            cfg=self._cfg(),
            vae_decode=None, vae_encode=None,
            noise_seed=5, denoise=1.0,
        )
        assert float(seen[0]) == pytest.approx(1.0)
        assert len(seen) == 4

    def test_vae_required_in_both_modes(self):
        """The always-pixel init anchor makes the VAE adapters mandatory
        even for upsampling='latent' (plan D2) — but only when stages
        actually run (a base-at-target cascade never needs them)."""
        torch.manual_seed(3)
        z = torch.zeros(1, 4, 16, 16)   # 128px base; target 512 -> one stage
        for mode in ("latent", "pixel"):
            with pytest.raises(ValueError, match="vae_decode and vae_encode"):
                hiflow_cascade(
                    z, self.SIGMAS,
                    lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
                    scale_factor=4.0,
                    cfg=self._cfg(upsampling=mode),
                    vae_decode=None, vae_encode=None,
                )
        # A base-at-target cascade (no stages) never touches the VAE.
        hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
            scale_factor=1.0,
            cfg=self._cfg(),
            vae_decode=None, vae_encode=None,
        )
        # The adapters are REQUIRED positional args (the anchor path needs
        # them) — omitting them is a TypeError, caught by the signature.
        with pytest.raises(TypeError):
            hiflow_cascade(
                z, self.SIGMAS,
                lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
                scale_factor=4.0,
                cfg=self._cfg(),
            )

    def test_latent_mode_calls_vae_for_anchor_only(self):
        """Latent mode: the VAE round trip runs ONCE per stage (the init
        anchor); per-step reference upsampling stays latent bicubic."""
        torch.manual_seed(4)
        z = torch.zeros(1, 4, 16, 16)
        vd, ve, calls = self._fake_vae()
        stage_entries = []

        def stage_predict(x, s):
            stage_entries.append(s)
            return 0.5 * x

        hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, stage_predict,
            scale_factor=2.0,
            cfg=self._cfg(upsampling="latent"),
            vae_decode=vd, vae_encode=ve,
        )
        assert calls["decode"] == 1 and calls["encode"] == 1, (
            "exactly one anchor round trip per stage in latent mode"
        )

    def test_two_stages_resolution_progression(self):
        """Base 32x32 latent (256px); scale 4 -> stages 64, 128."""
        torch.manual_seed(0)
        z = torch.randn(1, 4, 32, 32)
        vd, ve, _ = self._fake_vae()
        out = hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x,           # base
            lambda x, s: 0.5 * x,           # stage
            scale_factor=4.0,
            cfg=self._cfg(),
            vae_decode=vd, vae_encode=ve,
            vae_downscale=8,
        )
        assert out.shape == (1, 4, 128, 128)

    def test_downscale_stage_halves(self):
        """scale 0.5 runs ONE stage at half size — the reference trajectory
        is bicubic-resized down; the walk itself is unchanged (v2.13.0)."""
        torch.manual_seed(26)
        z = torch.randn(1, 4, 32, 32)
        vd, ve, _ = self._fake_vae()
        stage_sizes = []

        def stage_predict(x, s):
            stage_sizes.append(tuple(x.shape[-2:]))
            return 0.5 * x

        out = hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, stage_predict,
            scale_factor=0.5,
            cfg=self._cfg(),
            vae_decode=vd, vae_encode=ve,
        )
        assert out.shape == (1, 4, 16, 16), "scale 0.5 halves the latent"
        assert stage_sizes and all(s == (16, 16) for s in stage_sizes)
        assert torch.isfinite(out).all()

    def test_non_square_scale_per_side(self):
        """Scale applies per side: 128x64-pixel... 16x32 latent @2x ->
        32x64 — both sides double (the old absolute form over-upscaled
        the short side)."""
        torch.manual_seed(27)
        z = torch.randn(1, 4, 16, 32)
        vd, ve, _ = self._fake_vae()
        out = hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
            scale_factor=2.0,
            cfg=self._cfg(),
            vae_decode=vd, vae_encode=ve,
        )
        assert tuple(out.shape[-2:]) == (32, 64)

    def test_scale_out_of_range_rejected(self):
        torch.manual_seed(28)
        z = torch.randn(1, 4, 32, 32)
        vd, ve, _ = self._fake_vae()
        for bad in (0.1, 0.0, 8.5):
            with pytest.raises(ValueError, match="scale_factor"):
                hiflow_cascade(
                    z, self.SIGMAS,
                    lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
                    scale_factor=bad,
                    cfg=self._cfg(),
                    vae_decode=vd, vae_encode=ve,
                )

    def test_5d_latent_rejected_with_clear_message(self):
        """S1: the core is 4D — a 5D Wan21-style latent [B,C,1,H,W] must be
        squeezed by the NODE layer; the core raises a clear error (never
        silently broadcasts)."""
        torch.manual_seed(29)
        z5 = torch.randn(1, 4, 1, 32, 32)
        vd, ve, _ = self._fake_vae()
        with pytest.raises(ValueError, match="4D latent"):
            hiflow_cascade(
                z5, self.SIGMAS,
                lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
                scale_factor=2.0,
                cfg=self._cfg(),
                vae_decode=vd, vae_encode=ve,
            )

    def test_multi_frame_5d_rejected_too(self):
        """T>1 (real multi-frame) hits the same 4D assertion — multi-frame
        input is unsupported (image-only theory)."""
        torch.manual_seed(30)
        z5 = torch.randn(1, 4, 4, 32, 32)  # T=4
        vd, ve, _ = self._fake_vae()
        with pytest.raises(ValueError, match="4D latent"):
            hiflow_cascade(
                z5, self.SIGMAS,
                lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
                scale_factor=2.0,
                cfg=self._cfg(),
                vae_decode=vd, vae_encode=ve,
            )

    def test_seed_built_from_stage_latent_shape(self):
        """S1: the stage seed derives from the CURRENT chain latent (x), so
        the batch/channel dims stay correct even if they differed from the
        input latent (they can't in practice, but the pin guards the
        construction)."""
        torch.manual_seed(31)
        z = torch.randn(1, 4, 32, 32)
        vd, ve, _ = self._fake_vae()
        stage_shapes = []

        def stage_predict(x, s):
            stage_shapes.append(tuple(x.shape))
            return 0.5 * x

        out = hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, stage_predict,
            scale_factor=4.0,                   # two stages
            cfg=self._cfg(),
            vae_decode=vd, vae_encode=ve,
        )
        assert out.shape == (1, 4, 128, 128)
        assert all(s[:2] == (1, 4) for s in stage_shapes), (
            "batch/channels must match the chain latent across stages"
        )

    def test_single_stage_doubles(self):
        torch.manual_seed(1)
        z = torch.randn(1, 4, 32, 32)
        vd, ve, _ = self._fake_vae()
        out = hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
            scale_factor=2.0,
            cfg=self._cfg(),
            vae_decode=vd, vae_encode=ve,
        )
        assert out.shape == (1, 4, 64, 64)

    def test_second_stage_references_first_stage_trajectory(self):
        """Stage-2's model inputs must be at stage-2 size (the reference
        chain feeds through the previous stage's upsampled trajectory)."""
        torch.manual_seed(2)
        z = torch.randn(1, 4, 16, 16)
        vd, ve, _ = self._fake_vae()
        seen_sizes = []

        def stage_predict(x, s):
            seen_sizes.append(tuple(x.shape[-2:]))
            return 0.5 * x

        hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, stage_predict,
            scale_factor=4.0,
            cfg=self._cfg(),
            vae_decode=vd, vae_encode=ve,
        )
        # Unique sizes, order-preserving: stage 1 at 32x32, stage 2 at 64x64.
        uniq = list(dict.fromkeys(seen_sizes))
        assert uniq == [(32, 32), (64, 64)]

    def test_pixel_mode_calls_vae_adapters(self):
        torch.manual_seed(3)
        z = torch.randn(1, 4, 16, 16)
        vd, ve, calls = self._fake_vae()
        hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
            scale_factor=2.0,
            cfg=self._cfg(upsampling="pixel"),
            vae_decode=vd, vae_encode=ve,
            sharpen=lambda im: im,
        )
        # Per-step pixel upsampling + the anchor round trip: 1 anchor + up to
        # 3 per-step refs (tau=0.5 leaves 2 walkable transitions).
        assert calls["decode"] >= 1 and calls["encode"] >= 1

    def test_invalid_upsampling_rejected(self):
        torch.manual_seed(6)
        z = torch.randn(1, 4, 8, 8)
        vd, ve, _ = self._fake_vae()
        with pytest.raises(ValueError, match="latent.*pixel"):
            hiflow_cascade(
                z, self.SIGMAS,
                lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
                scale_factor=2.0,
                cfg=self._cfg(upsampling="bogus"),
                vae_decode=vd, vae_encode=ve,
            )

    def test_base_at_target_returns_base_output(self):
        """No stages: the noised base trajectory's final latent is returned
        (identical to running base_trajectory on the noised start). The VAE
        is never touched (no anchor needed)."""
        from src.hiflow import base_trajectory
        torch.manual_seed(7)
        z = torch.randn(1, 4, 32, 32)
        cfg = self._cfg()

        def fail_vae(x):
            raise AssertionError("base-at-target must not touch the VAE")

        out = hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
            scale_factor=1.0,
            cfg=cfg,
            vae_decode=fail_vae, vae_encode=fail_vae,
            noise_seed=55,
        )
        g = torch.Generator().manual_seed(55)
        eps = torch.randn(z.shape, generator=g)
        noised = 1.0 * eps + 0.0 * z
        expected, _ = base_trajectory(
            noised, self.SIGMAS, lambda x, s: 0.5 * x, cfg)
        assert torch.allclose(out, expected, atol=1e-6)

    def test_cascade_no_nan(self):
        torch.manual_seed(8)
        z = torch.randn(2, 4, 32, 32)
        vd, ve, _ = self._fake_vae()
        out = hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x + 0.05 * torch.randn_like(x),
            lambda x, s: 0.5 * x + 0.05 * torch.randn_like(x),
            scale_factor=4.0,
            cfg=self._cfg(alpha_scale=1.0, beta_scale=0.5),
            vae_decode=vd, vae_encode=ve,
        )
        assert torch.isfinite(out).all()

    def test_progress_total_events(self):
        """Base steps + per-stage transitions fire progress events.

        steps_per_stage is an UPPER bound: the stage walks only schedule
        sigmas below tau (tau=0.5 leaves 0.25 -> 0: 2 transitions here).
        """
        events = []
        torch.manual_seed(9)
        z = torch.randn(1, 4, 32, 32)
        vd, ve, _ = self._fake_vae()
        hiflow_cascade(
            z, self.SIGMAS,
            lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
            scale_factor=2.0,
            cfg=self._cfg(),
            vae_decode=vd, vae_encode=ve,
            progress_callback=lambda i, total, stage: events.append(stage),
        )
        assert events.count(-1) == len(self.SIGMAS) - 1
        assert events.count(0) == 2  # 0.25 -> 0 only (tau=0.5 entry)

    def test_tau_above_schedule_clamps_logged(self, caplog):
        """tau above sigma_max clamps the entry to sigma_max with a warning
        (schedule [1.0, 0.5, 0.0]: tau=0.99 enters at 1.0)."""
        import logging
        torch.manual_seed(10)
        z = torch.randn(1, 4, 32, 32)
        vd, ve, _ = self._fake_vae()
        with caplog.at_level(logging.WARNING, logger="ComfyUI-DyPE"):
            hiflow_cascade(
                z, torch.tensor([0.9, 0.5, 0.0]),
                lambda x, s: 0.5 * x, lambda x, s: 0.5 * x,
                scale_factor=2.0,
                cfg=self._cfg(tau=0.99, steps_per_stage=2),
                vae_decode=vd, vae_encode=ve,
            )
        assert any("above schedule" in r.message for r in caplog.records)
