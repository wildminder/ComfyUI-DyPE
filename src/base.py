import torch
import torch.nn as nn
import math
from .rope import get_1d_dype_yarn_pos_embed, get_1d_yarn_pos_embed, get_1d_ntk_pos_embed

class DyPEBasePosEmbed(nn.Module):
    """
    Base class for Dynamic Position Extrapolation.
    Handles the calculation of DyPE scaling factors and raw (cos, sin) components.
    Subclasses must implement `forward` to format the output for specific model architectures.
    """
    def __init__(self, theta: int | list[float], axes_dim: list[int], method: str = 'yarn', yarn_alt_scaling: bool = False, dype: bool = True, dype_scale: float = 2.0, dype_exponent: float = 2.0, base_resolution: int = 1024, dype_start_sigma: float = 1.0, base_patch_grid: tuple[int, int] | int | None = None) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        # per-axis thetas: if list/tuple, use theta[i]; otherwise use scalar theta for all
        if isinstance(theta, (list, tuple)):
            self.thetas = list(theta)
        else:
            self.thetas = None
        self.method = method
        self.yarn_alt_scaling = yarn_alt_scaling
        self.dype = True if method == 'vision_yarn' else (dype if method != 'base' else False)
        self.dype_scale = dype_scale
        self.dype_exponent = dype_exponent
        self.base_resolution = base_resolution
        self.dype_start_sigma = max(0.001, min(1.0, dype_start_sigma)) # Clamp 0.001-1.0
        
        self.current_timestep = 1.0
        self._span_cache: dict[tuple, float] = {}

        # Determine Base Patch Grid and Max Patches
        if base_patch_grid is None:
            # Default heuristic: 1024px -> 128 latent -> 64 patches (assuming patch_size=2)
            val = (self.base_resolution // 8) // 2
            self.base_patch_grid = (val, val)
        elif isinstance(base_patch_grid, int):
             self.base_patch_grid = (base_patch_grid, base_patch_grid)
        else:
            self.base_patch_grid = base_patch_grid
            
        self.base_patches = max(self.base_patch_grid)

    def set_timestep(self, timestep: float) -> None:
        self.current_timestep = timestep

    def _axis_token_span(self, axis_pos: torch.Tensor) -> float:
        cache_key = (axis_pos.shape, axis_pos.device, axis_pos.dtype)
        if cache_key in self._span_cache:
            return self._span_cache[cache_key]

        flat = axis_pos.float().reshape(-1)

        if flat.numel() <= 1:
            self._span_cache[cache_key] = 1.0
            return 1.0

        min_val, max_val = flat.min(), flat.max()
        span = max_val - min_val

        if span <= 0:
            self._span_cache[cache_key] = 1.0
            return 1.0

        unique_vals = torch.unique(flat)

        if unique_vals.numel() <= 1:
            self._span_cache[cache_key] = 1.0
            return 1.0

        step = torch.diff(unique_vals).min().item()

        if step <= 1e-6:
            result = float(flat.numel())
        else:
            result = float((span / step) + 1.0)

        self._span_cache[cache_key] = result
        return result

    def _get_mscale(self, scale_global: float) -> float:
        mscale_start = 0.1 * math.log(scale_global) + 1.0
        mscale_end = 1.0
        t_effective = self.current_timestep
        t_norm = 1.0 if t_effective > self.dype_start_sigma else (t_effective / self.dype_start_sigma)
        return mscale_end + (mscale_start - mscale_end) * math.pow(t_norm, self.dype_exponent)

    def _calc_vision_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n_axes = pos.shape[-1]
        components = []
        
        if n_axes >= 3:
            h_span = self._axis_token_span(pos[..., 1])
            w_span = self._axis_token_span(pos[..., 2])
            scale_global = max(1.0, max(h_span/self.base_patch_grid[0], w_span/self.base_patch_grid[1]))
        else:
            max_current_patches = self._axis_token_span(pos)
            scale_global = max(1.0, max_current_patches / self.base_patches)
            
        current_mscale = self._get_mscale(scale_global)

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            current_patches = self._axis_token_span(axis_pos)
            
            axis_theta = self.thetas[i] if self.thetas is not None else self.theta
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': axis_theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent, 'ntk_scale': scale_global, 'override_mscale': current_mscale}

            if i > 0:
                base_axis_len = self.base_patch_grid[i-1] if (n_axes >=3 and i-1 < len(self.base_patch_grid)) else self.base_patches
                
                scale_local = max(1.0, current_patches / base_axis_len)
                dype_kwargs['linear_scale'] = scale_local 
                
                if scale_global > 1.0:
                    cos, sin = get_1d_dype_yarn_pos_embed(**common_kwargs, ori_max_pe_len=base_axis_len, **dype_kwargs)
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
            else:
                cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

            components.append((cos, sin))
            
        return components

    def _calc_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n_axes = pos.shape[-1]
        components = []
        
        if n_axes >= 3:
            h_span = self._axis_token_span(pos[..., 1])
            w_span = self._axis_token_span(pos[..., 2])
            max_current_patches = max(h_span, w_span)
        else:
            max_current_patches = self._axis_token_span(pos)

        needs_extrapolation = (max_current_patches > self.base_patches)

        if needs_extrapolation and self.yarn_alt_scaling:
            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                axis_theta = self.thetas[i] if self.thetas is not None else self.theta
                common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': axis_theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                current_patches = self._axis_token_span(axis_pos)
                base_axis_len = self.base_patch_grid[i-1] if (n_axes >=3 and i > 0 and i-1 < len(self.base_patch_grid)) else self.base_patches

                if i > 0 and current_patches > base_axis_len:
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_yarn_pos_embed(**common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=base_axis_len, **dype_kwargs, use_aggressive_mscale=True)
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
                
                components.append((cos, sin))
        else:
            cos_full_spatial, sin_full_spatial = None, None
            if needs_extrapolation:
                spatial_axis_dim = self.axes_dim[1]
                spatial_theta = self.thetas[1] if self.thetas is not None else self.theta
                square_pos = torch.arange(0, max_current_patches, device=pos.device).float()
                max_pe_len = torch.tensor(max_current_patches, dtype=freqs_dtype, device=pos.device)
                
                common_kwargs_spatial = {'dim': spatial_axis_dim, 'theta': spatial_theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                cos_full_spatial, sin_full_spatial = get_1d_yarn_pos_embed(
                    **common_kwargs_spatial, pos=square_pos, max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches, **dype_kwargs, use_aggressive_mscale=False
                )

            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                axis_theta = self.thetas[i] if self.thetas is not None else self.theta
                
                if i > 0 and needs_extrapolation:
                    offset_indices = axis_pos.long() - axis_pos.long().min()
                    pos_indices = offset_indices.view(-1)
                    pos_indices = torch.clamp(pos_indices, max=cos_full_spatial.shape[0]-1)
                    
                    cos = cos_full_spatial[pos_indices].view(*axis_pos.shape, -1)
                    sin = sin_full_spatial[pos_indices].view(*axis_pos.shape, -1)
                else:
                    common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': axis_theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

                components.append((cos, sin))
            
        return components

    def _calc_ntk_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
        n_axes = pos.shape[-1]
        components = []
        
        if n_axes >= 3:
            h_span = self._axis_token_span(pos[..., 1])
            w_span = self._axis_token_span(pos[..., 2])
            scale_global = max(1.0, max(h_span/self.base_patch_grid[0], w_span/self.base_patch_grid[1]))
        else:
            max_current_patches = self._axis_token_span(pos)
            scale_global = max(1.0, max_current_patches / self.base_patches)

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            axis_theta = self.thetas[i] if self.thetas is not None else self.theta
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': axis_theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            
            ntk_factor = 1.0
            if i > 0 and scale_global > 1.0:
                base_ntk = scale_global ** (axis_dim / (axis_dim - 2))
                if self.dype:
                    k_t = self.dype_scale * (self.current_timestep ** self.dype_exponent)
                    ntk_factor = base_ntk ** k_t
                else:
                    ntk_factor = base_ntk
                ntk_factor = max(1.0, ntk_factor)
            
            cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=ntk_factor)
            components.append((cos, sin))
        return components

    def _calc_pi_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        DY-PI: Position Interpolation with time-dependent scaling.
        PI scales positions uniformly: pos_effective = pos / s^kappa(t)
        """
        n_axes = pos.shape[-1]
        components = []

        if n_axes >= 3:
            h_span = self._axis_token_span(pos[..., 1])
            w_span = self._axis_token_span(pos[..., 2])
            scale_global = max(1.0, max(h_span / self.base_patch_grid[0], w_span / self.base_patch_grid[1]))
        else:
            max_current_patches = self._axis_token_span(pos)
            scale_global = max(1.0, max_current_patches / self.base_patches)

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            axis_theta = self.thetas[i] if self.thetas is not None else self.theta

            if i > 0 and scale_global > 1.0:
                if self.dype:
                    k_t = self.dype_scale * (self.current_timestep ** self.dype_exponent)
                    effective_scale = scale_global ** k_t
                else:
                    effective_scale = scale_global
                effective_scale = max(1.0, effective_scale)
                scaled_pos = axis_pos / effective_scale
            else:
                scaled_pos = axis_pos

            common_kwargs = {'dim': axis_dim, 'pos': scaled_pos, 'theta': axis_theta,
                            'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
            components.append((cos, sin))

        return components

    # Public Interface
    def get_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.method == 'vision_yarn':
            return self._calc_vision_yarn_components(pos, freqs_dtype)
        elif self.method == 'yarn':
            return self._calc_yarn_components(pos, freqs_dtype)
        elif self.method == 'pi':
            return self._calc_pi_components(pos, freqs_dtype)
        else:
            return self._calc_ntk_components(pos, freqs_dtype)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class does not implement forward. Use a specific model subclass.")
