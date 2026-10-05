"""Qwen-Image 2.1 cascade support — VAE channels + the 2.1 empty latent node.

Two independent facts about Qwen-Image 2.1 drive this file, both read from
the ComfyUI source of the gate build (ComfyUI_t211c130p313):

1. ``comfy/sd.py:841-847`` — the 2.1 VAE is a Wan-2.2-layout VAE with
   ``downscale_ratio = 16``, ``output_channels`` read from the checkpoint
   tensor ``decoder.head.2.weight`` (4 for 2.1) and ``pad_channel_value``
   1.0. Its ``latent_dim`` stays **2**, so the ``latent_dim == 3`` unsqueeze
   bridge in the cascade adapters is correctly skipped — this file pins that.
2. ``comfy/latent_formats.py:957-960`` — ``QwenImage21`` is 64 channels,
   ``latent_dimensions = 2``, ``spacial_downscale_ratio = 16``. Core ships no
   text-to-image empty latent for it, so the pack registers one.

Markers: @pytest.mark.unit
"""

from __future__ import annotations

import json
import logging
import os
import sys

from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

import nodes.freescale as fsn  # noqa: E402
import nodes.hiflow as hfn  # noqa: E402
import nodes.pixelrush as prn  # noqa: E402
import nodes.qwen21_latent as q21  # noqa: E402
import src.vae_channels as vc  # noqa: E402


# ---------------------------------------------------------------------------
# Fake VAEs
# ---------------------------------------------------------------------------

class FakeVAE:
    """Records what decode/encode actually received.

    ``downscale`` mimics the VAE's spatial ratio so shape assertions stay
    honest about the //16 (2.1) vs //8 (everything else) split.
    """

    def __init__(self, output_channels=3, pad_channel_value=None,
                 latent_dim=2, downscale=16, latent_channels=64):
        self.output_channels = output_channels
        self.pad_channel_value = pad_channel_value
        self.latent_dim = latent_dim
        self.downscale = downscale
        self.downscale_ratio = downscale
        self.latent_channels = latent_channels
        self.encode_inputs = []
        self.decode_inputs = []

    def decode(self, latent):
        self.decode_inputs.append(latent)
        h, w = latent.shape[-2] * self.downscale, latent.shape[-1] * self.downscale
        if self.latent_dim == 3:
            return torch.rand(latent.shape[0], 1, h, w, self.output_channels)
        return torch.rand(latent.shape[0], h, w, self.output_channels)

    def encode(self, pixels):
        self.encode_inputs.append(pixels)
        c = pixels.shape[-1] * 2  # fake mapping; only the width matters here
        h = pixels.shape[-3] // self.downscale
        w = pixels.shape[-2] // self.downscale
        return torch.zeros(pixels.shape[0], c, h, w)


def rgba_vae():
    """The Qwen-Image 2.1 VAE: 16x, 4 output channels, opaque-alpha pad."""
    return FakeVAE(output_channels=4, pad_channel_value=1.0,
                   latent_dim=2, downscale=16)


def rgb_vae(downscale=8, latent_channels=16):
    return FakeVAE(output_channels=3, pad_channel_value=None,
                   latent_dim=2, downscale=downscale,
                   latent_channels=latent_channels)


@pytest.fixture(autouse=True)
def _reset_alpha_log(monkeypatch):
    monkeypatch.setattr(vc, "_alpha_drop_logged", False)


# (node name, adapter factory) — the three cascade nodes with their own
# _make_vae_adapters pair.
ADAPTERS = [
    ("HiFlow", lambda vae: hfn._make_vae_adapters(vae, torch.device("cpu"))),
    ("PixelRush", lambda vae: prn._make_vae_adapters(vae, torch.device("cpu"))),
    ("FreeScale", lambda vae: fsn._make_vae_adapters(vae, torch.device("cpu"))),
]
# The two nodes whose decode layout is channels-FIRST.
CHANNELS_FIRST = [a for a in ADAPTERS if a[0] != "HiFlow"]

IDS = [a[0] for a in ADAPTERS]
FIRST_IDS = [a[0] for a in CHANNELS_FIRST]


# ---------------------------------------------------------------------------
# Decode: the RGBA VAE must not leak a 4-channel image downstream
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCascadeDecodeWithRgbaVae:
    @pytest.mark.parametrize("name,make", ADAPTERS, ids=IDS)
    def test_decoded_image_is_three_channel(self, name, make):
        vae = rgba_vae()
        vae_decode, _ = make(vae)
        latent = torch.zeros(1, 64, 4, 4)
        image = vae_decode(latent)
        assert image.shape[-1] == 3 or image.shape[1] == 3, (
            f"{name}: alpha survived the decode adapter — {tuple(image.shape)}"
        )

    def test_hiflow_decode_stays_channels_last(self):
        vae_decode, _ = hfn._make_vae_adapters(rgba_vae(), torch.device("cpu"))
        image = vae_decode(torch.zeros(1, 64, 4, 4))
        assert image.shape == (1, 64, 64, 3)   # channels-LAST, RGB

    @pytest.mark.parametrize("name,make", CHANNELS_FIRST, ids=FIRST_IDS)
    def test_pixelrush_freescale_decode_is_channels_first(self, name, make):
        """Their layout probe matches on exactly 3 channels — an RGBA decode
        that skipped the strip would slip through it channels-LAST."""
        vae_decode, _ = make(rgba_vae())
        image = vae_decode(torch.zeros(1, 64, 4, 4))
        assert image.shape == (1, 3, 64, 64), (
            f"{name}: expected channels-FIRST RGB, got {tuple(image.shape)}"
        )

    def test_alpha_drop_is_logged_with_the_node_name(self, caplog):
        vae_decode, _ = hfn._make_vae_adapters(rgba_vae(), torch.device("cpu"))
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            vae_decode(torch.zeros(1, 64, 2, 2))
        infos = [r.getMessage() for r in caplog.records
                 if r.levelno == logging.INFO]
        assert any("HiFlow" in m and "alpha" in m.lower() for m in infos)

    def test_hiflow_sharpen_still_treats_the_decoded_image_as_channels_last(self):
        """_sharpen probes shape[-1] == 3 — the reason decode strips alpha
        BEFORE handing the image over, rather than at the very end."""
        vae_decode, _ = hfn._make_vae_adapters(rgba_vae(), torch.device("cpu"))
        image = vae_decode(torch.zeros(1, 64, 2, 2))
        assert hfn._sharpen(image, alpha=1.0).shape == image.shape


# ---------------------------------------------------------------------------
# Encode: the VAE must receive the channel count it asked for
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCascadeEncodeWithRgbaVae:
    def test_hiflow_hands_the_vae_four_channels(self):
        vae = rgba_vae()
        _, vae_encode = hfn._make_vae_adapters(vae, torch.device("cpu"))
        vae_encode(torch.rand(1, 64, 64, 3))
        assert vae.encode_inputs[0].shape[-1] == 4

    def test_hiflow_pads_with_the_vae_opaque_alpha_value(self):
        vae = rgba_vae()
        _, vae_encode = hfn._make_vae_adapters(vae, torch.device("cpu"))
        vae_encode(torch.rand(1, 8, 8, 3))
        torch.testing.assert_close(vae.encode_inputs[0][..., 3],
                                   torch.ones(1, 8, 8))

    @pytest.mark.parametrize("name,make", ADAPTERS, ids=IDS)
    def test_every_adapter_encodes_rgba(self, name, make):
        vae = rgba_vae()
        _, vae_encode = make(vae)
        vae_encode(torch.rand(1, 3, 64, 64))
        assert vae.encode_inputs[0].shape[-1] == 4, (
            f"{name}: the VAE got {vae.encode_inputs[0].shape[-1]} channels"
        )

    def test_pixelrush_hiflow_round_trip_is_pixel_stable(self):
        """decode -> encode must hand back the same RGB values (plus opaque
        alpha): the helper pair may not scale, shift or reorder anything."""
        vae = rgba_vae()
        vae_decode, vae_encode = hfn._make_vae_adapters(vae, torch.device("cpu"))
        image = vae_decode(torch.zeros(1, 64, 4, 4))
        vae_encode(image)
        torch.testing.assert_close(vae.encode_inputs[0][..., :3], image)


# ---------------------------------------------------------------------------
# Negative controls: every other architecture is untouched
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestThreeChannelVaePathUnchanged:
    def test_hiflow_decode_still_channels_last_rgb(self):
        vae_decode, _ = hfn._make_vae_adapters(rgb_vae(), torch.device("cpu"))
        image = vae_decode(torch.zeros(1, 16, 4, 4))
        assert image.shape == (1, 32, 32, 3)

    @pytest.mark.parametrize("name,make", ADAPTERS, ids=IDS)
    def test_no_prompt_pad_for_an_rgb_vae(self, name, make):
        vae = rgb_vae()
        _, vae_encode = make(vae)
        vae_encode(torch.rand(1, 3, 32, 32))
        assert vae.encode_inputs[0].shape[-1] == 3

    @pytest.mark.parametrize("name,make", ADAPTERS, ids=IDS)
    def test_three_channel_vae_path_unchanged(self, name, make):
        """The whole negative control: a 3-channel VAE crosses both cascade
        boundaries BIT-FOR-BIT unchanged.

        The RGBA handling is two helpers (``src/vae_channels.py``), each gated
        on ``output_channels == 4``.  This asserts the gate holds for every
        adapter in both directions: decode hands back exactly what the VAE
        emitted, and encode hands the VAE exactly the tensor the node passed
        (no pad, no truncate).  Only the pre-existing LAYOUT difference between
        the nodes is accounted for — HiFlow speaks channels-last, PixelRush
        and FreeScale channels-first (see ``CHANNELS_FIRST`` above); no pixel
        and no channel is touched.
        """
        channels_first = name != "HiFlow"
        vae = rgb_vae()
        vae_decode, vae_encode = make(vae)
        latent = torch.zeros(1, 16, 4, 4)

        torch.manual_seed(0)
        decoded = vae_decode(latent)
        torch.manual_seed(0)          # same draw -> the raw decode, untouched
        raw = vae.decode(latent)
        assert torch.equal(decoded, raw.movedim(-1, 1) if channels_first else raw)

        image = torch.rand(1, 32, 32, 3)
        # HiFlow's encode boundary is channels-last, the other two take
        # channels-first and permute at the call; either way the VAE receives
        # the same pixels in the same channel order.
        node_image = image.movedim(-1, 1) if channels_first else image
        vae_encode(node_image)
        assert torch.equal(
            vae.encode_inputs[0],
            node_image.movedim(1, -1) if channels_first else node_image,
        )

        # The helpers themselves are the identity, not merely equivalent.
        assert vc.strip_alpha_channel(raw, vae) is raw
        assert vc.pad_to_vae_channels(image, vae) is image

    @pytest.mark.parametrize("name,make", ADAPTERS, ids=IDS)
    def test_no_alpha_log_for_an_rgb_vae(self, name, make, caplog):
        vae_decode, _ = make(rgb_vae())
        with caplog.at_level(logging.INFO, logger="ComfyUI-DyPE"):
            vae_decode(torch.zeros(1, 16, 4, 4))
        assert not [r for r in caplog.records if r.levelno == logging.INFO]


# ---------------------------------------------------------------------------
# T7.1 — latent_dim stays 2 for 2.1, so no unsqueeze bridge
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestQwen21LatentDimIsTwo:
    def test_decode_receives_a_four_dimensional_latent(self):
        vae = rgba_vae()
        vae_decode, _ = hfn._make_vae_adapters(vae, torch.device("cpu"))
        vae_decode(torch.zeros(1, 64, 4, 4))
        assert vae.decode_inputs[0].dim() == 4, (
            "the 16x 2.1 VAE has latent_dim == 2 — no T dimension must be "
            "invented (a 5D input here would be the Krea2 path)"
        )

    def test_encode_returns_a_four_dimensional_latent(self):
        vae = rgba_vae()
        _, vae_encode = hfn._make_vae_adapters(vae, torch.device("cpu"))
        assert vae_encode(torch.rand(1, 64, 64, 3)).dim() == 4


# ---------------------------------------------------------------------------
# The ComfyUI-source facts this file's design rests on, pinned against a real
# build when one is available (skipped otherwise).
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestComfySourceFacts:
    def test_qwen21_vae_branch_is_16x_rgba(self):
        source = _comfy_source("sd.py")
        if source is None:
            pytest.skip("COMFYUI_ROOT not set — VAE facts unpinned")
        assert "# Qwen Image 2.1 VAE" in source
        branch = (source.split("# Qwen Image 2.1 VAE")[1]
                  .split("elif wan22_layout")[0])
        assert "self.downscale_ratio = 16" in branch
        assert "self.pad_channel_value = 1.0" in branch
        assert "vae2_2.WanVAE" in branch
        assert "self.output_channels = sd[" in branch, (
            "output_channels must stay checkpoint-derived; a hardcoded 4 "
            "here would make the helper's gate wrong for other VAEs"
        )
        assert "self.latent_dim" not in branch, (
            "the 2.1 VAE must keep latent_dim == 2 — the cascade adapters' "
            "5-D unsqueeze bridge is for Wan21 image models, not this one"
        )

    def test_vae_encode_crop_pixels_still_drives_the_pad_helper(self):
        """pad_to_vae_channels mirrors this function; if ComfyUI changes it,
        the mirror is what has to follow."""
        source = _comfy_source("sd.py")
        if source is None:
            pytest.skip("COMFYUI_ROOT not set — VAE facts unpinned")
        body = source.split("def vae_encode_crop_pixels")[1].split("\n    def ")[0]
        assert "self.output_channels" in body
        assert "self.pad_channel_value" in body


# ---------------------------------------------------------------------------
# The 2.1 empty latent node
# ---------------------------------------------------------------------------

def _comfy_source(relpath):
    """Return the text of a ComfyUI module from $COMFYUI_ROOT, or None.

    Read as TEXT, never imported: the root conftest pre-registers a mock
    ``comfy`` package, so an import would read the mock instead of the real
    source, and re-pointing ``sys.modules`` at the real tree would mutate
    global state shared with every other test in the session.
    """
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        return None
    path = Path(root) / "comfy" / relpath
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _comfy_class_body(source, class_name):
    """The source lines of ``class <class_name>:`` up to the next top-level
    ``class``/``def`` at column 0."""
    lines = source.splitlines()
    start = next(
        i for i, line in enumerate(lines)
        if line.startswith(f"class {class_name}")
    )
    end = next(
        (i for i in range(start + 1, len(lines))
         if lines[i] and not lines[i][0].isspace()),
        len(lines),
    )
    return "\n".join(lines[start:end])


@pytest.fixture
def fake_intermediate_device(monkeypatch):
    """The root conftest's comfy.model_management mock has no
    ``intermediate_device``. ``import comfy.model_management`` inside execute
    resolves the ATTRIBUTE on the ``comfy`` package object, so the attribute —
    not the sys.modules entry — is what must be patched."""
    import comfy.model_management as fake_mm

    monkeypatch.setattr(fake_mm, "intermediate_device",
                        lambda: torch.device("cpu"), raising=False)


@pytest.mark.unit
class TestEmptyQwenImage21LatentImage:
    def test_node_id_is_registered(self, monkeypatch):
        # The root conftest's io mock has io.Latent.Input but no
        # io.Latent.Output, so supply one locally rather than editing the
        # shared mock (conftest.py is not this task's file).
        from comfy_api.latest import io
        monkeypatch.setattr(io.Latent, "Output", io.Image.Output, raising=False)
        schema = q21.EmptyQwenImage21LatentImage.define_schema()
        assert schema.node_id == "EmptyQwenImage21LatentImage"

    def test_exported_from_the_nodes_package(self):
        import nodes as nodes_pkg
        assert "EmptyQwenImage21LatentImage" in nodes_pkg.__all__
        assert nodes_pkg.EmptyQwenImage21LatentImage is \
            q21.EmptyQwenImage21LatentImage

    @pytest.mark.parametrize("width,height,batch,expected", [
        (1328, 1328, 1, (1, 64, 83, 83)),
        (2560, 1440, 2, (2, 64, 90, 160)),
        (16, 16, 1, (1, 64, 1, 1)),
    ])
    def test_shape_is_64ch_at_sixteen_x(self, width, height, batch, expected,
                                        fake_intermediate_device):
        out = q21.EmptyQwenImage21LatentImage.execute(width, height, batch)
        latent = out[0]["samples"]
        assert tuple(latent.shape) == expected

    def test_matches_the_comfy_latent_format(self):
        """Pin the node's constants against ComfyUI's own QwenImage21 format.

        Read from the real source (see _comfy_source), not imported — a real
        import would be served the conftest's mock ``comfy`` package.
        Skipped when COMFYUI_ROOT is unset.
        """
        source = _comfy_source("latent_formats.py")
        if source is None:
            pytest.skip("COMFYUI_ROOT not set — geometry constants unpinned")
        body = _comfy_class_body(source, "QwenImage21")
        assert f"latent_channels = {q21.LATENT_CHANNELS}" in body
        assert f"spacial_downscale_ratio = {q21.SPACIAL_DOWNSCALE}" in body
        assert "latent_dimensions = 2" in body, (
            "this node emits a 4-D latent; a format change here would turn it "
            "into a layered/edit latent"
        )

    def test_snaps_a_non_multiple_of_16(self, fake_intermediate_device):
        latent = q21.EmptyQwenImage21LatentImage.execute(1000, 1000, 1)[0]["samples"]
        assert latent.shape[-2] == latent.shape[-1] == 1000 // 16

    def test_never_produces_a_zero_latent(self, fake_intermediate_device):
        latent = q21.EmptyQwenImage21LatentImage.execute(1, 1, 1)[0]["samples"]
        assert latent.shape[-2] == latent.shape[-1] == 1

    def test_is_all_zeros(self, fake_intermediate_device):
        latent = q21.EmptyQwenImage21LatentImage.execute(512, 512, 1)[0]["samples"]
        assert torch.count_nonzero(latent) == 0

    def test_validate_inputs_accepts_multiples_of_16(self):
        assert q21.EmptyQwenImage21LatentImage.validate_inputs(
            1328, 1328, 1) is True

    def test_validate_inputs_rejects_odd_widths_with_an_actionable_message(self):
        result = q21.EmptyQwenImage21LatentImage.validate_inputs(1000, 1000, 1)
        assert isinstance(result, str)
        assert "16" in result and "1000x1000" in result

    def test_validate_inputs_passes_uninitialized_state(self):
        assert q21.EmptyQwenImage21LatentImage.validate_inputs(
            None, None, None) is True

    def test_is_not_the_layered_edit_latent(self):
        """The core node it must not be confused with is 16 channels, //8 and
        5-D (comfy_extras/nodes_qwen.py:213)."""
        layered = torch.zeros(1, 16, 4, 80, 80)   # [B, 16, layers+1, h//8, w//8]
        assert layered.shape[1] != q21.LATENT_CHANNELS
        assert layered.shape[-1] * 8 != 1328


# ---------------------------------------------------------------------------
# Docs guards for the shipped 2.1 assets
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent


@pytest.mark.unit
class TestQwen21AssetsAreDocumented:
    def test_example_workflow_exists(self):
        assert (_PROJECT_ROOT / "example_workflows" /
                "DyPE-Qwen21-workflow.json").exists()

    def test_example_workflow_selects_the_qwen21_model_type(self):
        data = json.loads(
            (_PROJECT_ROOT / "example_workflows" /
             "DyPE-Qwen21-workflow.json").read_text(encoding="utf-8"))
        dype = [n for n in data["nodes"] if n["type"] == "DyPE_FLUX"]
        assert len(dype) == 1
        # DyPE widget order: width, height, model_type, method, ...
        assert dype[0]["widgets_values"][2] == "qwen21", (
            "the shipped 2.1 workflow must pin model_type='qwen21' (auto works "
            "too, but an explicit graph is the documentation)"
        )

    def test_example_workflow_resolutions_are_multiples_of_16(self):
        data = json.loads(
            (_PROJECT_ROOT / "example_workflows" /
             "DyPE-Qwen21-workflow.json").read_text(encoding="utf-8"))
        latent = [n for n in data["nodes"]
                  if n["type"] == "EmptyQwenImage21LatentImage"]
        assert len(latent) == 1
        width, height = latent[0]["widgets_values"][:2]
        assert width % 16 == 0 and height % 16 == 0

        dype = [n for n in data["nodes"] if n["type"] == "DyPE_FLUX"][0]
        assert dype["widgets_values"][0] == width
        assert dype["widgets_values"][1] == height, (
            "DyPE's width/height must match the latent (nodes/dype.py tooltip)"
        )

    def test_readme_mentions_the_new_node(self):
        text = (_PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "EmptyQwenImage21LatentImage" in text
