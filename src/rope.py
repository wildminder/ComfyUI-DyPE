import torch
import numpy as np
import math


def find_correction_factor(num_rotations: float, dim: int, base: float, max_position_embeddings: int) -> float:
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))


def find_correction_range(low_ratio: float, high_ratio: float, dim: int, base: float, ori_max_pe_len: int) -> tuple[float, float]:
    low = np.floor(find_correction_factor(low_ratio, dim, base, ori_max_pe_len))
    high = np.ceil(find_correction_factor(high_ratio, dim, base, ori_max_pe_len))
    return max(low, 0), min(high, dim-1)


def linear_ramp_mask(min_val: float, max_val: float, dim: int) -> torch.Tensor:
    if min_val == max_val:
        max_val += 0.001
    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def find_newbase_ntk(dim: int, base: float, scale: float) -> float:
    return base * (scale ** (dim / (dim - 2)))


# Vision YaRN with DyPE dynamic scheduling
def get_1d_dype_yarn_pos_embed(
        dim: int,
        pos: torch.Tensor,
        theta: float,
        use_real: bool,
        repeat_interleave_real: bool,
        freqs_dtype: torch.dtype,
        linear_scale: float,
        ntk_scale: float,
        ori_max_pe_len: int,
        dype: bool,
        current_timestep: float,
        dype_scale: float,
        dype_exponent: float,
        override_mscale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pos.device
    linear_scale = max(linear_scale, 1.0)
    ntk_scale = max(ntk_scale, 1.0)

    # YaRN ramp thresholds (Peng et al., 2023b "YaRN", Section 3.3):
    # beta_0, beta_1 define the NTK-by-parts lower ramp boundary.
    # gamma_0, gamma_1 define the upper ramp boundary for base-frequency blending.
    # These are the default values from the YaRN paper, empirically determined.
    # In DyPE, they are modulated by k_t = dype_scale * t^dype_exponent
    # to dynamically shift frequency band allocation over the sampling trajectory.
    # See DyPE paper (Issachar et al.) Appendix D for ablation of these values.
    beta_0, beta_1 = 1.25, 0.75
    gamma_0, gamma_1 = 16, 2

    if dype:
        k_t = dype_scale * (current_timestep ** dype_exponent)
        beta_0 = beta_0 ** k_t
        beta_1 = beta_1 ** k_t
        gamma_0 = gamma_0 ** k_t
        gamma_1 = gamma_1 ** k_t

    freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
    freqs_linear = freqs_base / linear_scale

    new_base = find_newbase_ntk(dim, theta, ntk_scale)
    if isinstance(new_base, torch.Tensor) and new_base.dim() > 0:
        new_base = new_base.view(-1, 1)
    freqs_ntk = 1.0 / torch.pow(new_base, (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
    if freqs_ntk.dim() > 1: freqs_ntk = freqs_ntk.squeeze()

    low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    mask_beta = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
    freqs = freqs_linear * (1 - mask_beta) + freqs_ntk * mask_beta

    low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    mask_gamma = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
    freqs = freqs * (1 - mask_gamma) + freqs_base * mask_gamma
    
    freqs = torch.einsum("...s,d->...sd", pos, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1).float()
        
        if override_mscale is not None:
            mscale = torch.tensor(override_mscale, dtype=freqs_dtype, device=device)
        else:
            mscale_val = 1.0 + 0.1 * math.log(ntk_scale) / math.sqrt(ntk_scale)
            mscale = torch.tensor(mscale_val, dtype=freqs_dtype, device=device)
        
        return freqs_cos * mscale, freqs_sin * mscale
    elif use_real:
        return freqs.cos().float(), freqs.sin().float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)

# YaRN
def get_1d_yarn_pos_embed(
        dim: int,
        pos: torch.Tensor,
        theta: float,
        use_real: bool,
        repeat_interleave_real: bool,
        freqs_dtype: torch.dtype,
        max_pe_len: torch.Tensor,
        ori_max_pe_len: int,
        dype: bool,
        current_timestep: float,
        dype_scale: float,
        dype_exponent: float,
        use_aggressive_mscale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pos.device
    scale = torch.clamp_min(max_pe_len / ori_max_pe_len, 1.0)

    beta_0, beta_1 = 1.25, 0.75
    gamma_0, gamma_1 = 16, 2

    freqs_base = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
    freqs_linear = 1.0 / torch.einsum('..., f -> ... f', scale, (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim)))

    new_base = find_newbase_ntk(dim, theta, scale)
    if new_base.dim() > 0: new_base = new_base.view(-1, 1)
    freqs_ntk = 1.0 / torch.pow(new_base, (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
    if freqs_ntk.dim() > 1: freqs_ntk = freqs_ntk.squeeze()

    if dype:
        k_t = dype_scale * (current_timestep ** dype_exponent)
        beta_0 = beta_0 ** k_t
        beta_1 = beta_1 ** k_t

    low, high = find_correction_range(beta_0, beta_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
    freqs = freqs_linear * (1 - freqs_mask) + freqs_ntk * freqs_mask

    if dype:
        k_t = dype_scale * (current_timestep ** dype_exponent)
        gamma_0 = gamma_0 ** k_t
        gamma_1 = gamma_1 ** k_t

    low, high = find_correction_range(gamma_0, gamma_1, dim, theta, ori_max_pe_len)
    low, high = max(0, low), min(dim // 2, high)
    freqs_mask = (1 - linear_ramp_mask(low, high, dim // 2).to(device).to(freqs_dtype))
    freqs = freqs * (1 - freqs_mask) + freqs_base * freqs_mask
    
    freqs = torch.einsum("...s,d->...sd", pos, freqs)

    if use_real and repeat_interleave_real:
        freqs_cos = freqs.cos().repeat_interleave(2, dim=-1).float()
        freqs_sin = freqs.sin().repeat_interleave(2, dim=-1).float()
        
        mscale = None
        if use_aggressive_mscale:
            mscale = torch.where(scale <= 1., torch.tensor(1.0), 0.1 * torch.log(scale) + 1.0).to(scale)
        else:
            mscale = torch.where(scale <= 1., torch.tensor(1.0), 1.0 + 0.1 * torch.log(scale) / torch.sqrt(scale)).to(scale)
        
        return freqs_cos * mscale, freqs_sin * mscale
    elif use_real:
        return freqs.cos().float(), freqs.sin().float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)

# NTK / Base
def get_1d_ntk_pos_embed(
        dim: int,
        pos: torch.Tensor,
        theta: float,
        use_real: bool,
        repeat_interleave_real: bool,
        freqs_dtype: torch.dtype,
        ntk_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pos.device
    theta_ntk = theta * ntk_factor
    freqs = 1.0 / (theta_ntk ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=device) / dim))
    freqs = torch.einsum("...s,d->...sd", pos, freqs)

    if use_real and repeat_interleave_real:
        return freqs.cos().repeat_interleave(2, dim=-1).float(), freqs.sin().repeat_interleave(2, dim=-1).float()
    elif use_real:
        return freqs.cos().float(), freqs.sin().float()
    else:
        return torch.polar(torch.ones_like(freqs), freqs)