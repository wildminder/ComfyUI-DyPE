"""
PixelRush — training-free cascade-based high-resolution generation.

Turns high-resolution generation into a sequence of coarse-to-fine cascade
refinements: generate a native-resolution image, upscale it, then use a
single partial DDIM inversion + single denoising step per overlapping
latent patch to add detail rather than regenerate the whole image from noise.

Reference: PixelRush paper (arXiv:2602.12769).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Iterator, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor

logger = logging.getLogger("ComfyUI-DyPE")


@dataclass
class PixelRushConfig:
    """Configuration for PixelRush cascade refinement.

    Parameters
    ----------
    patch_h, patch_w : int
        Latent-space patch dimensions.  For SDXL (native 1024px, VAE
        downscale 8×): 128×128.  For SD1.5 (native 512px): 64×64.
    overlap : float
        Fractional overlap along H and W (0.0–0.75).  Paper default: 0.50.
    k_timestep : int
        Partial-inversion timestep.  Must correspond to a valid timestep
        in the model's schedule.  Paper default: 249.
    noise_lambda : float
        Noise injection coefficient.  λ weights the REFINER'S PREDICTION in
        the injection: ``slerp(eps_random, eps_refined, λ)`` — at the
        paper default 0.95 the injected eps is 95% the model's prediction
        plus 5% random noise ("noise injection strength" is the informal
        reading; the ablation only makes sense with λ as prediction weight).
    noise_injection : str
        Injection mode.  ``"slerp"`` (paper, default):
        ``slerp(eps_random, eps_refined, noise_lambda)``.  ``"additive"``
        (legacy 2026-08-13 formula, same λ convention):
        ``eps_refined + (1 - noise_lambda) * eps_random``.
    gaussian_sigma : float
        Gaussian feathering sigma for the analytic patch weight mask.
        Paper default: 24.0. Rule of thumb: sigma ~ patch_size / 5.
    eps : float
        Numerical stability epsilon.

    Notes
    -----
    The core algorithm runs in VAE latent space (the ComfyUI LATENT
    convention); adapters injected by the node own the VAE<->model
    conversions internally (plan 2026-09-02).
    """

    patch_h: int
    patch_w: int
    overlap: float = 0.50
    k_timestep: int = 249
    noise_lambda: float = 0.95
    noise_injection: str = "slerp"
    gaussian_sigma: float = 24.0
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Spherical interpolation
# ---------------------------------------------------------------------------

def slerp(a: Tensor, b: Tensor, t: float, eps: float = 1e-7) -> Tensor:
    """Standard vector SLERP between tensors ``a`` and ``b`` (t=0 → a, t=1 → b).

    Treats each sample's complete latent tensor as one vector. Raw-vector
    form (paper/corrected-theory convention): the magnitudes are carried by
    the slerp coefficients themselves, not interpolated separately. Falls
    back to linear interpolation when the vectors are nearly collinear
    (sin(omega) < 1e-4), where SLERP is numerically unstable.

    Parameters
    ----------
    a, b : Tensor
        Shape ``[B, C, H, W]`` (or any shape; flattened to ``[B, N]``).
    t : float
        Interpolation coefficient in ``[0, 1]``.  ``t=0`` → ``a``,
        ``t=1`` → ``b``.

    Returns
    -------
    Tensor
        Same shape as ``a``.
    """
    assert a.shape == b.shape

    a_flat = a.flatten(start_dim=1)
    b_flat = b.flatten(start_dim=1)

    a_norm = a_flat.norm(dim=1, keepdim=True).clamp_min(eps)
    b_norm = b_flat.norm(dim=1, keepdim=True).clamp_min(eps)

    cos_omega = (a_flat * b_flat).sum(dim=1, keepdim=True) / (a_norm * b_norm)
    cos_omega = cos_omega.clamp(-1.0 + eps, 1.0 - eps)

    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)

    t_tensor = torch.full_like(omega, float(t))

    # Standard vector SLERP. No separate magnitude multiplication.
    slerp_flat = (
        torch.sin((1.0 - t_tensor) * omega) / sin_omega * a_flat
        + torch.sin(t_tensor * omega) / sin_omega * b_flat
    )

    # If vectors are almost collinear, SLERP becomes unstable.
    lerp_flat = (1.0 - t_tensor) * a_flat + t_tensor * b_flat
    use_lerp = sin_omega.abs() < 1e-4

    return torch.where(use_lerp, lerp_flat, slerp_flat).view_as(a)


# Backward-compatibility alias (previous name).
spherical_lerp = slerp


# ---------------------------------------------------------------------------
# Gaussian feathering
# ---------------------------------------------------------------------------

def gaussian_feather_mask(
    height: int,
    width: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """Create a ``[1, 1, H, W]`` analytic Gaussian weight mask, peak = 1.

    Corrected-theory form: ``exp(-(xx^2 + yy^2) / (2 sigma^2))`` centered on
    the patch and normalized so the peak is exactly 1 — an explicit encoding
    of the paper's Gaussian-filtered patch mask, not a blurred all-ones
    approximation. The mask is highest at the patch center and smoothly
    decreases toward the boundaries.
    """
    y = torch.arange(height, device=device, dtype=dtype) - (height - 1) / 2.0
    x = torch.arange(width, device=device, dtype=dtype) - (width - 1) / 2.0
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    mask = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma ** 2))
    return (mask / mask.max().clamp_min(1e-8))[None, None]


# ---------------------------------------------------------------------------
# Patch positions
# ---------------------------------------------------------------------------

def patch_positions(
    full_h: int,
    full_w: int,
    patch_h: int,
    patch_w: int,
    overlap: float,
) -> Iterator[Tuple[int, int]]:
    """Yield ``(y, x)`` top-left coordinates covering a latent completely.

    The final patch in each dimension is shifted so it touches the edge.
    """
    assert 0.0 <= overlap < 1.0

    stride_h = max(1, int(round(patch_h * (1.0 - overlap))))
    stride_w = max(1, int(round(patch_w * (1.0 - overlap))))

    def starts(full_size: int, patch_size: int, stride: int) -> list[int]:
        if full_size <= patch_size:
            return [0]
        values = list(range(0, full_size - patch_size + 1, stride))
        last = full_size - patch_size
        if values[-1] != last:
            values.append(last)
        return values

    ys = starts(full_h, patch_h, stride_h)
    xs = starts(full_w, patch_w, stride_w)

    for y in ys:
        for x in xs:
            yield y, x


# ---------------------------------------------------------------------------
# DDIM inversion / denoising
# ---------------------------------------------------------------------------

def predict_x0_from_epsilon(
    x_t: Tensor,
    epsilon: Tensor,
    alpha_bar_t: Tensor | float,
    eps: float = 1e-8,
) -> Tensor:
    """Recover x_0 from a noised latent and its epsilon.

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon, so:
        x_0 = (x_t - sqrt(1 - alpha_bar_t) * epsilon) / sqrt(alpha_bar_t)
    """
    alpha_bar_t = torch.as_tensor(
        alpha_bar_t, device=x_t.device, dtype=x_t.dtype
    )

    sqrt_alpha = alpha_bar_t.sqrt().clamp_min(eps)
    sqrt_one_minus_alpha = (1.0 - alpha_bar_t).clamp_min(0.0).sqrt()

    return (x_t - sqrt_one_minus_alpha * epsilon) / sqrt_alpha


def ddim_deterministic_step(
    x_from: Tensor,
    epsilon_from: Tensor,
    alpha_bar_from: Tensor | float,
    alpha_bar_to: Tensor | float,
) -> Tensor:
    """Deterministic DDIM (eta=0) transition between ARBITRARY timesteps.

    Recovers x_hat_0 from the source timestep via ``predict_x0_from_epsilon``,
    then re-noises it to the destination timestep, keeping the SAME epsilon:

        x_to = sqrt(alpha_bar_to) * x_hat_0 + sqrt(1 - alpha_bar_to) * epsilon

    Works in either direction:
      - inversion: source 0 -> destination K
      - denoising: source K -> destination 0
    """
    x0_pred = predict_x0_from_epsilon(
        x_t=x_from,
        epsilon=epsilon_from,
        alpha_bar_t=alpha_bar_from,
    )

    alpha_bar_to = torch.as_tensor(
        alpha_bar_to, device=x_from.device, dtype=x_from.dtype
    )

    return (
        alpha_bar_to.sqrt() * x0_pred
        + (1.0 - alpha_bar_to).clamp_min(0.0).sqrt() * epsilon_from
    )


def ddim_forward_one_step(
    z0: Tensor,
    eps0: Tensor,
    alpha_bar_k: Tensor | float,
) -> Tensor:
    """Deterministic DDIM inversion from timestep 0 to K.

    Since ``alpha_bar_0 = 1``::

        z_K = sqrt(alpha_bar_K) * z_0
              + sqrt(1 - alpha_bar_K) * eps(z_0, 0)
    """
    return ddim_deterministic_step(z0, eps0, 1.0, alpha_bar_k)


def ddim_reverse_one_step_to_zero(
    z_k: Tensor,
    eps_k: Tensor,
    alpha_bar_k: Tensor | float,
) -> Tensor:
    """Deterministic DDIM denoising from K to 0.

    Since ``alpha_bar_0 = 1``::

        z_0_hat = (z_K - sqrt(1-alpha_bar_K) * eps_K) / sqrt(alpha_bar_K)
    """
    return ddim_deterministic_step(z_k, eps_k, alpha_bar_k, 1.0)


# ---------------------------------------------------------------------------
# Single cascade stage
# ---------------------------------------------------------------------------

@torch.no_grad()
def refine_latent_once(
    coarse_latent: Tensor,
    inversion_eps: Callable[[Tensor, int], Tensor],
    refiner_eps: Callable[[Tensor, int], Tensor],
    alpha_bar_at: Callable[[int], Tensor | float],
    cfg: PixelRushConfig,
    progress_callback: Callable[[int, int], None] | None = None,
    forward_step: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None,
    reverse_step: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None,
    sigma_at: Callable[[int], float] | None = None,
) -> Tensor:
    """Apply one PixelRush refinement stage to a coarse latent.

    Parameters
    ----------
    coarse_latent : Tensor
        ``[B, C, H, W]`` latent obtained by pixel-space upsampling and
        VAE encoding.
    inversion_eps : callable
        ``inversion_eps(latent, timestep) -> [B, C, H, W]`` epsilon
        prediction from the BASE generator, used to drive the partial
        DDIM inversion (should already include CFG). Distinct from
        ``refiner_eps`` per the corrected theory — the paper uses a
        different (distilled one-step) model for refinement.
    refiner_eps : callable
        ``refiner_eps(latent, timestep) -> [B, C, H, W]`` epsilon
        prediction from the REFINER model at timestep K (should already
        include CFG).
    alpha_bar_at : callable
        ``alpha_bar_at(K) -> alpha_cumprod[K]``.  Used only when ``forward_step``
        / ``reverse_step`` are not provided (EPS-only fallback).
    cfg : PixelRushConfig
        Hyperparameters.
    progress_callback : callable, optional
        ``progress_callback(patch_idx, total_patches)`` called after each
        patch is refined.  Used for ComfyUI progress bar integration.
    forward_step : callable, optional
        ``forward_step(x_0, eps, sigma) -> x_K``.  Uses the model's own
        ``noise_scaling`` so the forward (0→K) step is correct for all
        prediction types (EPS, CONST/flow, V_PREDICTION, ...).
    reverse_step : callable, optional
        ``reverse_step(x_K, eps_injected, sigma) -> x_0_hat``.  Uses the
        inverse of ``noise_scaling`` so the reverse (K→0) step is correct for
        all prediction types.
    sigma_at : callable, optional
        ``sigma_at(timestep) -> sigma float``.  Used to get the sigma at
        timestep K for the forward/reverse adapters.

    Returns
    -------
    Tensor
        Refined latent ``[B, C, H, W]``.
    """
    b, c, full_h, full_w = coarse_latent.shape
    assert cfg.patch_h <= full_h and cfg.patch_w <= full_w, (
        "Patch dimensions must not exceed the latent."
    )

    # Schedule values at timestep K. sigma_k feeds the forward/reverse
    # adapters; alpha_k feeds the EPS-only DDIM fallback transitions. alpha_k
    # is computed whenever ANY fallback branch can execute (either adapter
    # missing), so partially-provided adapters can never hit an undefined
    # name — and alpha_bar_at is never called when both adapters are given.
    if sigma_at is not None:
        sigma_k = sigma_at(cfg.k_timestep)
        sigma_k_tensor = torch.tensor(
            [sigma_k], device=coarse_latent.device, dtype=coarse_latent.dtype
        )
    else:
        sigma_k = None
        sigma_k_tensor = None
    alpha_k = None
    if forward_step is None or reverse_step is None:
        alpha_k = alpha_bar_at(cfg.k_timestep)

    # Overlap-add buffers
    output_sum = torch.zeros_like(coarse_latent)
    weight_sum = torch.zeros_like(coarse_latent)

    feather = gaussian_feather_mask(
        cfg.patch_h,
        cfg.patch_w,
        sigma=cfg.gaussian_sigma,
        device=coarse_latent.device,
        dtype=coarse_latent.dtype,
    )  # [1, 1, patch_h, patch_w]

    # Collect all patch positions for progress reporting
    positions = list(patch_positions(
        full_h=full_h,
        full_w=full_w,
        patch_h=cfg.patch_h,
        patch_w=cfg.patch_w,
        overlap=cfg.overlap,
    ))
    total_patches = len(positions)
    logger.info(
        "PixelRush: refining %dx%d latent with %dx%d patches, %d patches total (overlap=%.0f%%)",
        full_h, full_w, cfg.patch_h, cfg.patch_w, total_patches, cfg.overlap * 100,
    )

    for idx, (y, x) in enumerate(positions):
        if idx % 4 == 0:
            logger.info("PixelRush: patch %d/%d", idx + 1, total_patches)
        patch_0 = coarse_latent[:, :, y:y + cfg.patch_h, x:x + cfg.patch_w]

        # 1. Partial inversion: 0 -> K (driven by the BASE model's eps)
        eps_for_inversion = inversion_eps(patch_0, timestep=0)
        # Ensure eps is on the same device as the patch
        eps_for_inversion = eps_for_inversion.to(patch_0.device)
        if forward_step is not None:
            patch_k = forward_step(patch_0, eps_for_inversion, sigma_k_tensor)
        else:
            patch_k = ddim_forward_one_step(patch_0, eps_for_inversion, alpha_k)

        # 2. One-step denoise: K -> 0 (driven by the REFINER model's eps)
        eps_refined = refiner_eps(patch_k, timestep=cfg.k_timestep)
        eps_refined = eps_refined.to(patch_k.device)

        # 3. PixelRush noise injection. lambda weights the REFINER'S
        # PREDICTION: slerp(eps_random, eps_refined, lambda) — at the
        # paper's lambda=0.95 the injected eps is 95% the model's
        # prediction with 5% random. (The corrected doc's reference code
        # used the opposite argument order, which makes lambda=0.95 mean
        # 95% PURE RANDOM noise — at real scales that is per-pixel noise
        # std ~1.2 vs signal std ~1, matching the reported "structure
        # visible but completely noisy in soft patches" artifact. The
        # doc itself flags this as the one detail to check against the
        # authors' implementation; the ablation only makes sense with
        # lambda as prediction weight.) The "additive" mode preserves the
        # 2026-08-13 legacy formula with the same convention: 5% random.
        eps_random = torch.randn_like(eps_refined)
        if cfg.noise_injection == "additive":
            # Legacy 2026-08-13 mode: eps_pred + (1 - lambda) * eps_rand.
            eps_injected = eps_refined + (1.0 - cfg.noise_lambda) * eps_random
        elif cfg.noise_injection == "slerp":
            eps_injected = slerp(eps_random, eps_refined, cfg.noise_lambda)
        else:
            raise ValueError(
                f"Unknown noise_injection mode: {cfg.noise_injection!r} "
                "(expected 'slerp' or 'additive')"
            )

        # 4. Reverse step: K -> 0
        if reverse_step is not None:
            refined_patch = reverse_step(patch_k, eps_injected, sigma_k_tensor)
        else:
            refined_patch = ddim_reverse_one_step_to_zero(patch_k, eps_injected, alpha_k)

        # 5. Gaussian-feather overlap-add
        output_sum[:, :, y:y + cfg.patch_h, x:x + cfg.patch_w] += (
            refined_patch * feather
        )
        weight_sum[:, :, y:y + cfg.patch_h, x:x + cfg.patch_w] += feather

        # 6. Progress callback
        if progress_callback is not None:
            progress_callback(idx + 1, total_patches)

    return output_sum / weight_sum.clamp_min(cfg.eps)


# ---------------------------------------------------------------------------
# Full cascade
# ---------------------------------------------------------------------------

@torch.no_grad()
def pixelrush_cascade(
    initial_latent: Tensor,
    num_cascade_stages: int,
    vae_decode: Callable[[Tensor], Tensor],
    vae_encode: Callable[[Tensor], Tensor],
    inversion_eps: Callable[[Tensor, int], Tensor],
    refiner_eps: Callable[[Tensor, int], Tensor],
    alpha_bar_at: Callable[[int], Tensor | float],
    cfg: PixelRushConfig,
    progress_callback: Callable[[int, int, int, int], None] | None = None,
    forward_step: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None,
    reverse_step: Callable[[Tensor, Tensor, Tensor], Tensor] | None = None,
    sigma_at: Callable[[int], float] | None = None,
) -> Tensor:
    """PixelRush cascade: repeatedly upscale and refine.

    Parameters
    ----------
    initial_latent : Tensor
        Native-resolution base latent ``[B, C, H, W]``.
    num_cascade_stages : int
        1: native → 2×, 2: native → 4×, 3: native → 8×.
    vae_decode : callable
        ``vae_decode(latent) -> image`` (B, C_img, H_img, W_img).
    vae_encode : callable
        ``vae_encode(image) -> latent`` (B, C, H, W).
    inversion_eps : callable
        ``inversion_eps(latent, timestep) -> eps`` from the BASE generator
        (with CFG). Drives the partial DDIM inversion.
    refiner_eps : callable
        ``refiner_eps(latent, timestep) -> eps`` from the REFINER model
        (with CFG). Drives the one-step refinement at timestep K.
    alpha_bar_at : callable
        ``alpha_bar_at(timestep) -> alpha_bar``.  Used only when the
        forward/reverse adapters are not provided (EPS-only fallback).
    cfg : PixelRushConfig
    progress_callback : callable, optional
        ``progress_callback(patch_idx, total_patches, stage, num_stages)``
        called after each patch is refined.  Used for ComfyUI progress
        bar integration.
    forward_step : callable, optional
        ``forward_step(x_0, eps, sigma) -> x_K`` (model's noise_scaling).
    reverse_step : callable, optional
        ``reverse_step(x_K, eps_injected, sigma) -> x_0_hat`` (inverse of
        noise_scaling).
    sigma_at : callable, optional
        ``sigma_at(timestep) -> sigma float``.

    Returns
    -------
    Tensor
        Final refined latent at target resolution.
    """
    z = initial_latent

    for stage in range(num_cascade_stages):
        logger.info(
            "PixelRush: cascade stage %d/%d — latent shape %s",
            stage + 1, num_cascade_stages, tuple(z.shape),
        )
        # Pixel-space cascade upsample: latent → RGB → 2× bicubic → latent
        image = vae_decode(z)
        # Cast to float32 because antialiased bicubic is not implemented for Half
        orig_dtype = image.dtype
        image_up = F.interpolate(
            image.float(),
            scale_factor=2.0,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        ).to(orig_dtype)

        coarse_latent = vae_encode(image_up)
        # Ensure coarse_latent is on the same device as the model output
        # (VAE may return on CPU even if input was on GPU)
        coarse_latent = coarse_latent.to(image_up.device)
        # For 3D latent models, vae_encode may return 5D [B,C,T,H,W].
        # Squeeze temporal dim for the 4D spatial core algorithm.
        # predict_eps will unsqueeze back to 5D before calling apply_model.
        if coarse_latent.ndim == 5:
            coarse_latent = coarse_latent.squeeze(2)  # [B, C, H, W]
        logger.info(
            "PixelRush: upscaled to %s, starting patch refinement",
            tuple(coarse_latent.shape),
        )

        # Patch-based refinement
        if progress_callback is not None:
            def stage_callback(patch_idx, total_patches):
                progress_callback(patch_idx, total_patches, stage, num_cascade_stages)
        else:
            stage_callback = None

        z = refine_latent_once(
            coarse_latent=coarse_latent,
            inversion_eps=inversion_eps,
            refiner_eps=refiner_eps,
            alpha_bar_at=alpha_bar_at,
            cfg=cfg,
            progress_callback=stage_callback,
            forward_step=forward_step,
            reverse_step=reverse_step,
            sigma_at=sigma_at,
        )
        logger.info("PixelRush: stage %d complete", stage + 1)

    return z
