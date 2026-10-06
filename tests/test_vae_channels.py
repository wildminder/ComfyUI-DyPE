"""Tests for src/vae_channels.py — the RGBA VAE channel helpers (v2.17.0).

Qwen-Image 2.1's VAE (comfy/sd.py:841-847) is a 16x Wan-2.2-layout VAE with a
checkpoint-derived ``output_channels`` (4) and ``pad_channel_value`` 1.0. The
cascade nodes speak RGB, so the boundary is normalized here.

These tests are pure-torch: no ComfyUI import, no patcher, CPU only.

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import src.vae_channels as vc  # noqa: E402


class FakeVAE:
    """Minimal VAE stand-in carrying only the channel attributes we read."""

    def __init__(self, output_channels=3, pad_channel_value=None):
        self.output_channels = output_channels
        self.pad_channel_value = pad_channel_value


class NoAttrVAE:
    """A VAE object without the channel attributes at all (mock/wrapper)."""


RGBA_VAE = FakeVAE(output_channels=4, pad_channel_value=1.0)
RGB_VAE = FakeVAE(output_channels=3, pad_channel_value=None)


@pytest.fixture(autouse=True)
def _reset_alpha_log(monkeypatch):
    """The one-time INFO latch is module state — reset it per test."""
    monkeypatch.setattr(vc, "_alpha_drop_logged", False)


def _rgba_image(b=1, h=2, w=3):
    """[B,H,W,4] with the alpha channel = 1.0 everywhere (easy to assert on)."""
    img = torch.zeros(b, h, w, 4)
    img[..., 3] = 1.0
    img[..., 0] = 0.5
    return img


# ---------------------------------------------------------------------------
# Channel introspection
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVaeImageChannels:
    def test_default_is_three_without_the_attribute(self):
        assert vc.vae_image_channels(NoAttrVAE()) == 3

    def test_reads_output_channels(self):
        assert vc.vae_image_channels(RGBA_VAE) == 4

    def test_is_rgba_true_only_for_four(self):
        assert vc.is_rgba_vae(RGBA_VAE) is True
        assert vc.is_rgba_vae(RGB_VAE) is False
        # A 2-channel VAE (SD-inpainting style) is NOT the RGBA case: the
        # helpers must leave its 2-channel convention alone.
        assert vc.is_rgba_vae(FakeVAE(output_channels=2)) is False


# ---------------------------------------------------------------------------
# Decode side: alpha is dropped, loudly
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestStripAlphaChannel:
    def test_slices_to_three_for_rgba_vae(self):
        out = vc.strip_alpha_channel(_rgba_image(), RGBA_VAE)
        assert out.shape == (1, 2, 3, 3)
        torch.testing.assert_close(out[..., 0], torch.full((1, 2, 3), 0.5))

    def test_keeps_rgb_values_untouched(self):
        img = torch.rand(1, 4, 4, 4)
        out = vc.strip_alpha_channel(img, RGBA_VAE)
        torch.testing.assert_close(out, img[..., :3])

    def test_identity_for_three_channel_vae(self):
        img = torch.rand(1, 4, 4, 3)
        assert vc.strip_alpha_channel(img, RGB_VAE) is img

    def test_identity_for_vae_without_the_attribute(self):
        img = torch.rand(1, 4, 4, 4)
        assert vc.strip_alpha_channel(img, NoAttrVAE()) is img

    def test_identity_when_image_already_rgb(self):
        img = torch.rand(1, 4, 4, 3)
        assert vc.strip_alpha_channel(img, RGBA_VAE) is img

    def test_alpha_not_silently_dropped_when_user_expects_it(self, caplog):
        """The drop must be announced — once — naming the node responsible."""
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            vc.strip_alpha_channel(_rgba_image(), RGBA_VAE, "HiFlow")
        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1, f"expected exactly one INFO, got {infos}"
        message = infos[0].getMessage()
        assert "HiFlow" in message
        assert "alpha" in message.lower()

    def test_log_is_latched_once_per_process(self, caplog):
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            for _ in range(5):
                vc.strip_alpha_channel(_rgba_image(), RGBA_VAE)
        assert len([r for r in caplog.records if r.levelno == logging.INFO]) == 1

    def test_no_log_for_rgb_vae(self, caplog):
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            vc.strip_alpha_channel(torch.rand(1, 2, 2, 3), RGB_VAE)
        assert caplog.records == []


# ---------------------------------------------------------------------------
# Encode side: what the VAE asked for
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPadToVaeChannels:
    def test_pads_rgb_for_rgba_vae_with_pad_channel_value(self):
        img = torch.rand(1, 2, 2, 3)
        out = vc.pad_to_vae_channels(img, RGBA_VAE)
        assert out.shape == (1, 2, 2, 4)
        torch.testing.assert_close(out[..., 3], torch.ones(1, 2, 2))

    def test_pad_value_is_not_hardcoded_to_one(self):
        vae = FakeVAE(output_channels=4, pad_channel_value=0.5)
        out = vc.pad_to_vae_channels(torch.zeros(1, 2, 2, 3), vae)
        torch.testing.assert_close(out[..., 3], torch.full((1, 2, 2), 0.5))

    def test_no_pad_value_means_no_padding_just_like_core(self):
        """sd.py:1156 — pad_channel_value is None -> core does not pad either."""
        vae = FakeVAE(output_channels=4)  # pad_channel_value not set
        img = torch.rand(1, 2, 2, 3)
        assert vc.pad_to_vae_channels(img, vae) is img

    def test_replicate_mode_repeats_the_edge_channel(self):
        vae = FakeVAE(output_channels=4, pad_channel_value="replicate")
        img = torch.rand(1, 2, 2, 3)
        out = vc.pad_to_vae_channels(img, vae)
        torch.testing.assert_close(out[..., 3], img[..., -1])

    def test_truncates_an_over_wide_image(self):
        img = torch.rand(1, 2, 2, 5)
        out = vc.pad_to_vae_channels(img, FakeVAE(output_channels=4))
        torch.testing.assert_close(out, img[..., :4])

    def test_identity_for_three_channel_vae(self):
        img = torch.rand(1, 2, 2, 3)
        assert vc.pad_to_vae_channels(img, RGB_VAE) is img

    def test_identity_for_vae_without_the_attribute(self):
        img = torch.rand(1, 2, 2, 3)
        assert vc.pad_to_vae_channels(img, NoAttrVAE()) is img

    def test_does_not_mutate_the_input(self):
        img = torch.zeros(1, 2, 2, 3)
        vc.pad_to_vae_channels(img, RGBA_VAE)
        assert img.shape == (1, 2, 2, 3)

    def test_two_channel_vae_pads_down_to_its_own_width(self):
        """Mirrors core's vae_encode_crop_pixels for a 2-channel VAE."""
        vae = FakeVAE(output_channels=2, pad_channel_value=1.0)
        out = vc.pad_to_vae_channels(torch.rand(1, 2, 2, 3), vae)
        assert out.shape == (1, 2, 2, 2)


@pytest.mark.unit
class TestRoundTrip:
    def test_decode_then_encode_returns_the_original_pixels(self):
        """The pair must be a no-op on RGB content — no value drift."""
        img = torch.rand(1, 2, 2, 4)
        img[..., 3] = 1.0  # an "opaque alpha" decode
        back = vc.pad_to_vae_channels(vc.strip_alpha_channel(img, RGBA_VAE),
                                      RGBA_VAE)
        torch.testing.assert_close(back, img)
