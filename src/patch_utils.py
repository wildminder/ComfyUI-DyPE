import math
import types
import torch
import torch.nn.functional as F
import comfy
from comfy.model_patcher import ModelPatcher
from comfy import model_sampling

from .models.flux import PosEmbedFlux
from .models.nunchaku import PosEmbedNunchaku
from .models.qwen import PosEmbedQwen
from .models.z_image import PosEmbedZImage


def apply_dype_to_model(model: ModelPatcher, model_type: str, width: int, height: int, method: str, yarn_alt_scaling: bool, enable_dype: bool, dype_scale: float, dype_exponent: float, base_shift: float, max_shift: float, base_resolution: int = 1024, dype_start_sigma: float = 1.0) -> ModelPatcher:
    m = model.clone()

    is_nunchaku = False
    is_qwen = False
    is_zimage = False

    normalized_model_type = model_type.replace("_", "").lower()

    if normalized_model_type == "nunchaku":
        is_nunchaku = True
    elif normalized_model_type == "qwen":
        is_qwen = True
    elif normalized_model_type == "zimage":
        is_zimage = True
    elif model_type == "flux":
        pass # defaults false
    else: # auto
        if hasattr(m.model, "diffusion_model"):
            dm = m.model.diffusion_model
            model_class_name = dm.__class__.__name__

            if "QwenImage" in model_class_name:
                is_qwen = True
            elif "NextDiT" in model_class_name or hasattr(dm, "rope_embedder"):
                is_zimage = True
            elif hasattr(dm, "model") and hasattr(dm.model, "pos_embed"):
                is_nunchaku = True
            elif hasattr(dm, "pe_embedder"):
                pass
            else:
                pass
        else:
            raise ValueError("The provided model is not a compatible model.")

    new_dype_params = (width, height, base_shift, max_shift, method, yarn_alt_scaling, base_resolution, dype_start_sigma, is_nunchaku, is_qwen, is_zimage)

    should_patch_schedule = True
    if hasattr(m.model, "_dype_params"):
        if m.model._dype_params == new_dype_params:
            should_patch_schedule = False
        else:
            pass

    base_patch_h_tokens = None
    base_patch_w_tokens = None
    default_base_patches = (base_resolution // 8) // 2
    default_base_seq_len = default_base_patches * default_base_patches

    if is_zimage:
        axes_lens = getattr(m.model.diffusion_model, "axes_lens", None)
        if isinstance(axes_lens, (list, tuple)) and len(axes_lens) >= 3:
            base_patch_h_tokens = int(axes_lens[1])
            base_patch_w_tokens = int(axes_lens[2])

    patch_size = 2 # Default Flux/Qwen
    try:
        if is_nunchaku:
            patch_size = m.model.diffusion_model.model.config.patch_size
        else:
            patch_size = m.model.diffusion_model.patch_size
    except:
        pass

    if base_patch_h_tokens is not None and base_patch_w_tokens is not None:
        derived_base_patches = max(base_patch_h_tokens, base_patch_w_tokens)
        derived_base_seq_len = base_patch_h_tokens * base_patch_w_tokens
    else:
        derived_base_patches = default_base_patches
        derived_base_seq_len = default_base_seq_len

    if enable_dype and should_patch_schedule:
        try:
            if isinstance(m.model.model_sampling, model_sampling.ModelSamplingFlux) or is_qwen or is_zimage:
                latent_h, latent_w = height // 8, width // 8
                padded_h, padded_w = math.ceil(latent_h / patch_size) * patch_size, math.ceil(latent_w / patch_size) * patch_size
                image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)

                base_seq_len = derived_base_seq_len
                max_seq_len = image_seq_len

                if max_seq_len <= base_seq_len:
                    dype_shift = base_shift
                else:
                    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                    intercept = base_shift - slope * base_seq_len
                    dype_shift = image_seq_len * slope + intercept

                dype_shift = max(0.0, dype_shift)

                class DypeModelSamplingFlux(model_sampling.ModelSamplingFlux, model_sampling.CONST):
                    pass

                new_model_sampler = DypeModelSamplingFlux(m.model.model_config)
                new_model_sampler.set_parameters(shift=dype_shift)

                m.add_object_patch("model_sampling", new_model_sampler)
                m.model._dype_params = new_dype_params
        except:
            pass

    elif not enable_dype:
        if hasattr(m.model, "_dype_params"):
            class DefaultModelSamplingFlux(model_sampling.ModelSamplingFlux, model_sampling.CONST): pass
            default_sampler = DefaultModelSamplingFlux(m.model.model_config)
            m.add_object_patch("model_sampling", default_sampler)
            del m.model._dype_params

    try:
        if is_nunchaku:
            orig_embedder = m.model.diffusion_model.model.pos_embed
            target_patch_path = "diffusion_model.model.pos_embed"
        elif is_zimage:
            orig_embedder = m.model.diffusion_model.rope_embedder
            target_patch_path = "diffusion_model.rope_embedder"
        else:
            orig_embedder = m.model.diffusion_model.pe_embedder
            target_patch_path = "diffusion_model.pe_embedder"

        theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX/Qwen/Z-Image model structure.")

    embedder_cls = PosEmbedFlux
    if is_nunchaku:
        embedder_cls = PosEmbedNunchaku
    elif is_qwen:
        embedder_cls = PosEmbedQwen
    elif is_zimage:
        embedder_cls = PosEmbedZImage

    embedder_base_patches = derived_base_patches if is_zimage else None

    new_pe_embedder = embedder_cls(
        theta, axes_dim, method, yarn_alt_scaling, enable_dype,
        dype_scale, dype_exponent, base_resolution, dype_start_sigma, embedder_base_patches
    )

    m.add_object_patch(target_patch_path, new_pe_embedder)

    if is_zimage:
        original_patchify_and_embed = getattr(m.model.diffusion_model, "patchify_and_embed", None)
        if original_patchify_and_embed is not None:
            m.model.diffusion_model._dype_original_patchify_and_embed = original_patchify_and_embed

        base_hw_override = None
        if base_patch_h_tokens is not None and base_patch_w_tokens is not None:
            base_hw_override = (base_patch_h_tokens, base_patch_w_tokens)
        elif derived_base_patches is not None:
            base_hw_override = (derived_base_patches, derived_base_patches)

        if base_hw_override is not None:
            m.model.diffusion_model._dype_base_hw = base_hw_override

        def dype_patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens, transformer_options={}):
            pH = pW = self.patch_size
            x_tensor = x if isinstance(x, torch.Tensor) else torch.stack(list(x), dim=0)
            device = x_tensor.device

            B, C, H, W = x_tensor.shape
            if (H % pH != 0) or (W % pW != 0):
                x_tensor = comfy.ldm.common_dit.pad_to_patch_size(x_tensor, (pH, pW))
                B, C, H, W = x_tensor.shape

            bsz = x_tensor.shape[0]

            if self.pad_tokens_multiple is not None:
                pad_extra = (-cap_feats.shape[1]) % self.pad_tokens_multiple
                if pad_extra > 0:
                    cap_pad = self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True).unsqueeze(0).repeat(cap_feats.shape[0], pad_extra, 1)
                    cap_feats = torch.cat((cap_feats, cap_pad), dim=1)

            cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
            cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0

            x_emb = self.x_embedder(x_tensor.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))

            rope_options = transformer_options.get("rope_options", None)
            base_hw = getattr(self, "_dype_base_hw", None)
            default_h_scale = 1.0
            default_w_scale = 1.0

            rope_embedder = getattr(self, "rope_embedder", None)
            dype_blend_factor = None
            if rope_embedder is not None:
                dype_start_sigma = getattr(rope_embedder, "dype_start_sigma", None)
                dype_exponent = getattr(rope_embedder, "dype_exponent", None)
                current_timestep = getattr(rope_embedder, "current_timestep", None)

                if all(value is not None for value in (dype_start_sigma, dype_exponent, current_timestep)) and dype_start_sigma > 0:
                    if current_timestep > dype_start_sigma:
                        t_norm = 1.0
                    else:
                        t_norm = current_timestep / dype_start_sigma

                    dype_blend_factor = math.pow(t_norm, dype_exponent)

            H_tokens, W_tokens = H // pH, W // pW
            if base_hw is not None and len(base_hw) == 2 and base_hw[0] > 0 and base_hw[1] > 0:
                default_h_scale = H_tokens / base_hw[0]
                default_w_scale = W_tokens / base_hw[1]

            def _blend_scale(default_scale: float) -> float:
                if dype_blend_factor is None:
                    return default_scale
                return 1.0 + (default_scale - 1.0) * dype_blend_factor

            h_scale = rope_options.get("scale_y", _blend_scale(default_h_scale)) if rope_options is not None else _blend_scale(default_h_scale)
            w_scale = rope_options.get("scale_x", _blend_scale(default_w_scale)) if rope_options is not None else _blend_scale(default_w_scale)

            h_start = rope_options.get("shift_y", 0.0) if rope_options is not None else 0.0
            w_start = rope_options.get("shift_x", 0.0) if rope_options is not None else 0.0

            x_pos_ids = torch.zeros((bsz, x_emb.shape[1], 3), dtype=torch.float32, device=device)
            x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
            x_pos_ids[:, :, 1] = (torch.arange(H_tokens, dtype=torch.float32, device=device) * h_scale + h_start).view(-1, 1).repeat(1, W_tokens).flatten()
            x_pos_ids[:, :, 2] = (torch.arange(W_tokens, dtype=torch.float32, device=device) * w_scale + w_start).view(1, -1).repeat(H_tokens, 1).flatten()

            if self.pad_tokens_multiple is not None:
                pad_extra = (-x_emb.shape[1]) % self.pad_tokens_multiple
                if pad_extra > 0:
                    pad_token = self.x_pad_token.to(device=x_emb.device, dtype=x_emb.dtype, copy=True).unsqueeze(0).repeat(x_emb.shape[0], pad_extra, 1)
                    x_emb = torch.cat((x_emb, pad_token), dim=1)
                    x_pos_ids = F.pad(x_pos_ids, (0, 0, 0, pad_extra))

            freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2)

            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, :cap_pos_ids.shape[1]], transformer_options=transformer_options)

            padded_img_mask = None
            for layer in self.noise_refiner:
                x_emb = layer(x_emb, padded_img_mask, freqs_cis[:, cap_pos_ids.shape[1]:], t, transformer_options=transformer_options)

            padded_full_embed = torch.cat((cap_feats, x_emb), dim=1)
            mask = None
            img_sizes = [(H, W)] * bsz
            l_effective_cap_len = [cap_feats.shape[1]] * bsz
            return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

        m.add_object_patch("diffusion_model.patchify_and_embed", types.MethodType(dype_patchify_and_embed, m.model.diffusion_model))

        if original_patchify_and_embed is not None:
            m.model._dype_zimage_override_active = True
            m.model._dype_zimage_step_count = 0

    sigma_max = m.model.model_sampling.sigma_max.item()

    def dype_wrapper_function(model_function, args_dict):
        current_sigma = None
        timestep_tensor = args_dict.get("timestep")
        if timestep_tensor is not None and timestep_tensor.numel() > 0:
            current_sigma = timestep_tensor.flatten()[0].item()

            if sigma_max > 0:
                normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                new_pe_embedder.set_timestep(normalized_timestep)

        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        output = model_function(input_x, args_dict.get("timestep"), **c)

        if getattr(m.model, "_dype_zimage_override_active", False):
            current_step = getattr(m.model, "_dype_zimage_step_count", 0) + 1
            m.model._dype_zimage_step_count = current_step

            if current_sigma is not None and current_sigma <= dype_start_sigma:
                original_fn = getattr(m.model.diffusion_model, "_dype_original_patchify_and_embed", None)
                if original_fn is not None:
                    m.model.diffusion_model.patchify_and_embed = original_fn

                if hasattr(m.model.diffusion_model, "_dype_base_hw"):
                    delattr(m.model.diffusion_model, "_dype_base_hw")

                new_pe_embedder.base_patches = default_base_patches

                m.model._dype_zimage_override_active = False

        return output

    m.set_model_unet_function_wrapper(dype_wrapper_function)

    return m
