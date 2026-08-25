"""
Qwen2D VAE patching module — patches ComfyUI's VAE loading to detect
and use the Qwen2D VAE (2D version of WanVAE) for image generation.

When a user loads the Qwen2D VAE checkpoint (from HuggingFace: Anzhc/Qwen2D-VAE),
this patch intercepts the VAE initialization and uses the 2D Qwen2DVAE architecture
instead of the 3D WanVAE. This enables FreeScale/PixelRush to work with
Krea2/Qwen/Anima models by providing a VAE that accepts 16-channel latents
(matching the model's latent format) without the temporal dimension overhead.

Source: https://github.com/anzhc/qwen2d-comfyui
"""

import logging
import types

import comfy.model_management as model_management
import comfy.model_patcher
import comfy.sd as comfy_sd
import comfy.utils
import torch

from .qwen2d_vae import Qwen2DVAE

_PATCH_INSTALLED_ATTR = "_anzhc_qwen2d_patch_installed"
_ORIGINAL_INIT_ATTR = "_anzhc_qwen2d_original_init"

logger = logging.getLogger("ComfyUI-DyPE")


def _is_qwen2d_state_dict(sd):
    """Check if a state dict is a Qwen2D VAE checkpoint."""
    if sd is None:
        return False
    if "decoder.mid_block.attentions.0.norm.gamma" not in sd:
        return False
    if "quant_conv.weight" not in sd or "post_quant_conv.weight" not in sd:
        return False
    if "decoder.conv_in.weight" not in sd:
        return False
    return sd["decoder.conv_in.weight"].ndim == 4


def _maybe_convert_diffusers_vae_state_dict(sd):
    """Convert diffusers-format VAE state dict to ComfyUI format."""
    if sd is not None and "decoder.up_blocks.0.resnets.0.norm1.weight" in sd:
        return comfy_sd.diffusers_convert.convert_vae_state_dict(sd)
    return sd


def _init_common_vae_defaults(vae):
    """Set common VAE defaults for Qwen2D VAE."""
    if model_management.is_amd():
        vae_kl_mem_ratio = 2.73
    else:
        vae_kl_mem_ratio = 1.0

    vae.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * model_management.dtype_size(dtype) * vae_kl_mem_ratio
    vae.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * model_management.dtype_size(dtype) * vae_kl_mem_ratio
    vae.downscale_ratio = 8
    vae.upscale_ratio = 8
    vae.latent_channels = 4
    vae.latent_dim = 2
    vae.output_channels = 3
    vae.pad_channel_value = None
    vae.process_input = lambda image: image * 2.0 - 1.0
    vae.process_output = lambda image: torch.clamp((image + 1.0) / 2.0, min=0.0, max=1.0)
    vae.working_dtypes = [torch.bfloat16, torch.float32]
    vae.disable_offload = False
    vae.not_video = False
    vae.size = None
    vae.downscale_index_formula = None
    vae.upscale_index_formula = None
    vae.extra_1d_channel = None
    vae.crop_input = True
    vae.audio_sample_rate = 44100


def _strip_singleton_temporal(tensor):
    """Remove singleton temporal dimension from 5D tensor."""
    if tensor.ndim == 5 and tensor.shape[2] == 1:
        return tensor[:, :, 0], True
    return tensor, False


def _flatten_temporal_batch(tensor):
    """Flatten 5D [B, C, T, H, W] to 4D [B*T, C, H, W]."""
    if tensor.ndim != 5:
        return tensor, None
    batch, channels, frames, height, width = tensor.shape
    flat = tensor.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    return flat, (batch, frames)


def _restore_temporal_batch(tensor, frame_info):
    """Restore 4D [B*T, C, H, W] to 5D [B, C, T, H, W]."""
    if frame_info is None:
        return tensor
    batch, frames = frame_info
    channels, height, width = tensor.shape[1:]
    return tensor.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def _qwen2d_decode_tiled_2d(self, samples, tile_x=64, tile_y=64, overlap=16):
    """Tiled decode for Qwen2D VAE."""
    steps = samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
    steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
    steps += samples.shape[0] * comfy.utils.get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)
    pbar = comfy.utils.ProgressBar(steps)
    upscale_amount = self.spacial_compression_decode()

    decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
    output = self.process_output(
        (
            comfy.utils.tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=upscale_amount, output_device=self.output_device, pbar=pbar)
            + comfy.utils.tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=upscale_amount, output_device=self.output_device, pbar=pbar)
            + comfy.utils.tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=upscale_amount, output_device=self.output_device, pbar=pbar)
        )
        / 3.0
    )
    return output


def _qwen2d_encode_tiled_2d(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
    """Tiled encode for Qwen2D VAE."""
    steps = pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
    steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
    steps += pixel_samples.shape[0] * comfy.utils.get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)
    pbar = comfy.utils.ProgressBar(steps)
    upscale_amount = 1 / self.spacial_compression_encode()

    encode_fn = lambda a: self.first_stage_model.encode(self.process_input(a).to(self.vae_dtype).to(self.device)).float()
    samples = comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount=upscale_amount, out_channels=self.latent_channels, output_device=self.output_device, pbar=pbar)
    samples = samples + comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=upscale_amount, out_channels=self.latent_channels, output_device=self.output_device, pbar=pbar)
    samples = samples + comfy.utils.tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=upscale_amount, out_channels=self.latent_channels, output_device=self.output_device, pbar=pbar)
    samples /= 3.0
    return samples


def _qwen2d_decode(self, samples_in, vae_options={}):
    """Decode method for Qwen2D VAE."""
    self.throw_exception_if_invalid()
    pixel_samples = None
    do_tile = False
    samples_in, squeezed_temporal = _strip_singleton_temporal(samples_in)
    samples_in, frame_info = _flatten_temporal_batch(samples_in)
    try:
        memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
        model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
        free_memory = self.patcher.get_free_memory(self.device)
        batch_number = max(1, int(free_memory / max(1, memory_used)))

        for x in range(0, samples_in.shape[0], batch_number):
            samples = samples_in[x : x + batch_number].to(self.vae_dtype).to(self.device)
            out = self.process_output(self.first_stage_model.decode(samples, **vae_options).to(self.output_device).float())
            if pixel_samples is None:
                pixel_samples = torch.empty((samples_in.shape[0],) + tuple(out.shape[1:]), device=self.output_device)
            pixel_samples[x : x + batch_number] = out
    except model_management.OOM_EXCEPTION:
        logging.warning("Warning: Ran out of memory when regular Qwen2D VAE decoding, retrying with tiled VAE decoding.")
        do_tile = True

    if do_tile:
        pixel_samples = self.decode_tiled_(samples_in)

    pixel_samples = _restore_temporal_batch(pixel_samples, frame_info)
    if squeezed_temporal and pixel_samples.ndim == 5 and pixel_samples.shape[2] == 1:
        pixel_samples = pixel_samples[:, :, 0]
    return pixel_samples.to(self.output_device).movedim(1, -1)


def _qwen2d_decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
    """Tiled decode method for Qwen2D VAE."""
    self.throw_exception_if_invalid()
    samples, squeezed_temporal = _strip_singleton_temporal(samples)
    samples, frame_info = _flatten_temporal_batch(samples)

    memory_used = self.memory_used_decode(samples.shape, self.vae_dtype)
    model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)

    args = {}
    if tile_x is not None:
        args["tile_x"] = tile_x
    if tile_y is not None:
        args["tile_y"] = tile_y
    if overlap is not None:
        args["overlap"] = overlap

    output = self.decode_tiled_(samples, **args)
    output = _restore_temporal_batch(output, frame_info)
    if squeezed_temporal and output.ndim == 5 and output.shape[2] == 1:
        output = output[:, :, 0]
    return output.movedim(1, -1)


def _qwen2d_encode(self, pixel_samples):
    """Encode method for Qwen2D VAE."""
    self.throw_exception_if_invalid()
    pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
    pixel_samples = pixel_samples.movedim(-1, 1)
    if pixel_samples.ndim == 4:
        pixel_samples = pixel_samples.unsqueeze(2)

    do_tile = False
    pixel_samples, squeezed_temporal = _strip_singleton_temporal(pixel_samples)
    pixel_samples, frame_info = _flatten_temporal_batch(pixel_samples)
    try:
        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
        model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)
        free_memory = self.patcher.get_free_memory(self.device)
        batch_number = max(1, int(free_memory / max(1, memory_used)))
        samples = None
        for x in range(0, pixel_samples.shape[0], batch_number):
            pixels_in = self.process_input(pixel_samples[x : x + batch_number]).to(self.vae_dtype).to(self.device)
            out = self.first_stage_model.encode(pixels_in).to(self.output_device).float()
            if samples is None:
                samples = torch.empty((pixel_samples.shape[0],) + tuple(out.shape[1:]), device=self.output_device)
            samples[x : x + batch_number] = out
    except model_management.OOM_EXCEPTION:
        logging.warning("Warning: Ran out of memory when regular Qwen2D VAE encoding, retrying with tiled VAE encoding.")
        do_tile = True

    if do_tile:
        samples = self.encode_tiled_(pixel_samples)

    samples = _restore_temporal_batch(samples, frame_info)
    if squeezed_temporal:
        samples = samples.unsqueeze(2)
    return samples


def _qwen2d_encode_tiled(self, pixel_samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
    """Tiled encode method for Qwen2D VAE."""
    self.throw_exception_if_invalid()
    pixel_samples = self.vae_encode_crop_pixels(pixel_samples)
    pixel_samples = pixel_samples.movedim(-1, 1)
    if pixel_samples.ndim == 4:
        pixel_samples = pixel_samples.unsqueeze(2)

    pixel_samples, squeezed_temporal = _strip_singleton_temporal(pixel_samples)
    pixel_samples, frame_info = _flatten_temporal_batch(pixel_samples)

    memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
    model_management.load_models_gpu([self.patcher], memory_required=memory_used, force_full_load=self.disable_offload)

    args = {}
    if tile_x is not None:
        args["tile_x"] = tile_x
    if tile_y is not None:
        args["tile_y"] = tile_y
    if overlap is not None:
        args["overlap"] = overlap

    samples = self.encode_tiled_(pixel_samples, **args)
    samples = _restore_temporal_batch(samples, frame_info)
    if squeezed_temporal:
        samples = samples.unsqueeze(2)
    return samples


def _bind_qwen2d_methods(vae):
    """Bind Qwen2D VAE methods to the VAE object."""
    vae.decode = types.MethodType(_qwen2d_decode, vae)
    vae.decode_tiled = types.MethodType(_qwen2d_decode_tiled, vae)
    vae.decode_tiled_ = types.MethodType(_qwen2d_decode_tiled_2d, vae)
    vae.encode = types.MethodType(_qwen2d_encode, vae)
    vae.encode_tiled = types.MethodType(_qwen2d_encode_tiled, vae)
    vae.encode_tiled_ = types.MethodType(_qwen2d_encode_tiled_2d, vae)


def _init_qwen2d_vae(vae, sd, device=None, dtype=None):
    """Initialize a Qwen2D VAE from a state dict."""
    _init_common_vae_defaults(vae)

    vae.upscale_ratio = (lambda a: a, 8, 8)
    vae.upscale_index_formula = (1, 8, 8)
    vae.downscale_ratio = (lambda a: a, 8, 8)
    vae.downscale_index_formula = (1, 8, 8)
    vae.latent_dim = 3
    vae.latent_channels = sd["decoder.conv_in.weight"].shape[1]
    vae.output_channels = sd["decoder.conv_out.weight"].shape[0]
    vae.not_video = True
    vae.working_dtypes = [torch.bfloat16, torch.float16, torch.float32]
    vae.memory_used_encode = lambda shape, dtype: (1400 * (shape[2] if len(shape) == 5 else 1) * shape[-2] * shape[-1]) * model_management.dtype_size(dtype)
    vae.memory_used_decode = lambda shape, dtype: (1800 * (shape[2] if len(shape) == 5 else 1) * shape[-2] * shape[-1] * 64) * model_management.dtype_size(dtype)

    ddconfig = {
        "base_dim": sd["decoder.norm_out.gamma"].shape[0],
        "z_dim": vae.latent_channels,
        "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "attn_scales": [],
        "temperal_downsample": [False, True, True],
        "image_channels": vae.output_channels,
        "dropout": 0.0,
    }

    vae.first_stage_model = Qwen2DVAE(**ddconfig).eval()
    _bind_qwen2d_methods(vae)

    if device is None:
        device = model_management.vae_device()
    vae.device = device
    offload_device = model_management.vae_offload_device()
    if dtype is None:
        dtype = model_management.vae_dtype(vae.device, vae.working_dtypes)
    vae.vae_dtype = dtype
    vae.first_stage_model.to(vae.vae_dtype)
    model_management.archive_model_dtypes(vae.first_stage_model)
    vae.output_device = model_management.intermediate_device()

    patcher_cls = comfy.model_patcher.CoreModelPatcher
    if vae.disable_offload:
        patcher_cls = comfy.model_patcher.ModelPatcher
    vae.patcher = patcher_cls(vae.first_stage_model, load_device=vae.device, offload_device=offload_device)

    missing, leftover = vae.first_stage_model.load_state_dict(sd, strict=False, assign=vae.patcher.is_dynamic())
    if len(missing) > 0:
        logging.warning("Missing VAE keys %s", missing)
    if len(leftover) > 0:
        logging.debug("Leftover VAE keys %s", leftover)

    logging.info("VAE load device: %s, offload device: %s, dtype: %s", vae.device, offload_device, vae.vae_dtype)
    vae.model_size()


def install_qwen2d_patch():
    """Patch ComfyUI's VAE.__init__ to detect and use Qwen2D VAE.

    This patches comfy.sd.VAE.__init__ to intercept Qwen2D VAE state dicts
    and initialize them with the 2D Qwen2DVAE architecture instead of the
    3D WanVAE. This enables FreeScale/PixelRush to work with Krea2/Qwen/Anima
    models by providing a VAE that accepts 16-channel latents.
    """
    if getattr(comfy_sd.VAE, _PATCH_INSTALLED_ATTR, False):
        return

    original_init = comfy_sd.VAE.__init__

    def patched_init(self, sd=None, device=None, config=None, dtype=None, metadata=None):
        sd = _maybe_convert_diffusers_vae_state_dict(sd)
        if config is None and _is_qwen2d_state_dict(sd):
            _init_qwen2d_vae(self, sd, device=device, dtype=dtype)
            return
        return original_init(self, sd=sd, device=device, config=config, dtype=dtype, metadata=metadata)

    comfy_sd.VAE.__init__ = patched_init
    setattr(comfy_sd.VAE, _ORIGINAL_INIT_ATTR, original_init)
    setattr(comfy_sd.VAE, _PATCH_INSTALLED_ATTR, True)
    logger.info("Installed Qwen2D VAE patch for FreeScale/PixelRush compatibility.")
