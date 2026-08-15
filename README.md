<div id="readme-top" align="center">
  <h1 align="center">ComfyUI-DyPE</h1>

<img src="https://github.com/user-attachments/assets/4f11966b-86f7-4bdb-acd4-ada6135db2f8" alt="ComfyUI-DyPE Banner" width="70%">

  
  <p align="center">
    A ComfyUI custom node that implements <strong>DyPE (Dynamic Position Extrapolation)</strong>, <strong>SEGA (Spectral-Energy Guided Attention)</strong>, and <strong>SPA (Spatial Position Alignment, HRDiT)</strong>, enabling Diffusion Transformers (like <strong>FLUX</strong>, <strong>Qwen Image</strong>, <strong>Z-Image</strong>, <strong>Anima/Cosmos</strong>, and <strong>Krea-2</strong>) to generate ultra-high-resolution images (4K and beyond) with exceptional coherence and detail.
    <br />
    <br />
    <a href="https://github.com/wildminder/ComfyUI-DyPE/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/wildminder/ComfyUI-DyPE/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]

</div>

<br>

## About The Project

DyPE is a training-free method that allows pre-trained DiT models to generate images at resolutions far beyond their training data, with no additional sampling cost.

It works by taking advantage of the spectral progression inherent to the diffusion process. By dynamically adjusting the model's positional encodings at each step, DyPE matches their frequency spectrum with the current stage of the generative process—focusing on low-frequency structures early on and resolving high-frequency details in later steps. This prevents the repeating artifacts and structural degradation typically seen when pushing models beyond their native resolution.

<div align="center">

  <img alt="ComfyUI-DyPE example workflow" width="70%" src="https://github.com/user-attachments/assets/31f5d254-68a7-435b-8e1f-c4e636d4f3c2" />
      <p><sub><i>A simple, single-node integration to patch your model for high-resolution generation.</i></sub></p>
  </div>


  
This node provides a seamless, "plug-and-play" integration of DyPE into your workflow.

**✨ Key Features:**
*   **Multi-Architecture Support:** Supports **FLUX** (Standard), **Nunchaku** (Quantized Flux), **Qwen Image**, **Z-Image** (Lumina 2), **Anima/Cosmos**, and **Krea-2**.
*   **SPA (HRDiT):** Spatial Position Alignment — a static, training-free RoPE patch that fixes high-resolution *spatial disorder* by bundling + sliding positions and averaging the `2s − 1` **attention outputs**. Resolution-aware (automatic no-op ≤ 1024px) and step-gated (`spa_steps = 3` default → ~1.3–1.8× overhead at 2K/4K), no timestep coupling. **Mutually exclusive** with DyPE/SEGA (apply only one).
*   **High-Resolution Generation:** Push models to 4096x4096 and beyond.
*   **Single-Node Integration:** Simply place the `DyPE for FLUX` node after your model loader to patch the model. No complex workflow changes required.
*   **Full Compatibility:** Works seamlessly with your existing ComfyUI workflows, samplers, schedulers, and other optimization nodes.
*   **Fine-Grained Control:** Exposes key DyPE hyperparameters, allowing you to tune the algorithm's strength and behavior for optimal results at different target resolutions.
*   **Zero Inference Overhead:** DyPE's adjustments happen on-the-fly with negligible performance impact.

<div align="center">
<img alt="Node" width="70%" src="https://github.com/user-attachments/assets/ef900ba2-e019-496a-89f6-abd92c857029" />
</div>

## Example output

<div align="center">
<img alt="Example dype" src="https://github.com/user-attachments/assets/f85861fd-4d2f-4b57-8058-26881600b7ca" />
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## SEGA Node

**SEGA** (Spectral-Energy Guided Attention) — content-aware per-dimension RoPE mscale from the latent's FFT spectrum. Use as an alternative to DyPE for FLUX/Qwen. For Anima, use DyPE `vision_yarn` instead.

### Usage

1. Add the **SEGA** node after your model loader
2. Set width/height to match your latent
3. Use `method: sega` (default) or `method: ntk` (NTK only, no spectral)
4. Tune `mscale_alpha` (amplitude) and `spread_min`/`spread_max` (spectral gate range)

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `method` | sega | `sega` = NTK + spectral mscale, `ntk` = NTK only |
| `mscale_alpha` | 0.15 | Spectral redistribution amplitude |
| `mscale_beta` | 1.5 | tanh sharpness |
| `mscale_min` | 1.0 | Floor for per-frequency mscale |
| `spread_min` | 0.0 | Min spectral spread (early steps) |
| `spread_max` | 1.0 | Max spectral spread (late steps) |
| `spread_alpha` | 1.5 | Spread schedule non-linearity |
| `base_mscale_formula` | power_res | `power_res`: s^κ, `log_res`: 1+κ·ln(s) |
| `base_mscale_coefficient` | 0.08 | κ (paper default) |

> **Note:** SEGA uses NTK as its base extrapolation. It refines NTK with per-dimension spectral mscale. If NTK doesn't work for your model (e.g. Anima), SEGA won't either — use DyPE `vision_yarn` instead.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## SPA Node (HRDiT)

**SPA** (Spatial Position Alignment, from **HRDiT** — arXiv 2608.07003) is a static, training-free positional-encoding patch for ultra-high-resolution generation. It is **mutually exclusive** with DyPE/SEGA — apply only one.

*   **Why:** Pushing a DiT beyond its native resolution makes the RoPE token indices grow out of the model's training distribution, causing *spatial disorder* — repeated structures and positional collisions.
*   **How:** SPA compresses out-of-distribution token indices into bundles of `N` tokens *before* they enter the positional embedding, then slides the bundle boundary independently along each axis. This yields `2s − 1` variants (1 base + `(s−1)` row slides + `(s−1)` column slides, where `s` is the derived per-axis bundle size). For each variant it builds that variant's (no-extrapolation) RoPE and runs a **full attention pass**; SPA runs the `2s − 1` passes and **averages the attention outputs** — exactly HRDiT `_spa_attention`. Averaging attention *outputs* (not the RoPE rotation matrices) is essential: softmax is nonlinear, so `meanₙ softmax(Rₙ)·V ≠ softmax(meanₙ Rₙ)·V`, and averaging the rotations yields a non-orthogonal matrix — the root cause of the old *rippled-mosaic* bug. The paper proves each original position keeps a unique signature across slides (`Σₙ φ⁽ⁿ⁾(i) = i`), so spatial distinguishability is restored without retraining.
*   **Static:** SPA has **no timestep dependence** — it does not patch the noise schedule. When active (`enable_spa` and `bundle_size != 1`) it replaces the model's RoPE embedder (which returns the base RoPE and registers the `2s − 1` variant RoPEs) and installs an attention hook that runs the averaged attention passes. `bundle_size == 1` (off) or `enable_spa = False` installs nothing and is a transparent base-RoPE pass. `bundle_size == 0` (auto) stays active but is an automatic **no-op** while the grid is inside the model's trained extent.
*   **Resolution-aware (trained-extent gate):** While the token grid is inside the model's trained distribution (`max_pos ≤ 64`, i.e. ≤ 1024px for 1024px-trained DiTs) there is no position extrapolation to fix, so SPA is an **identity no-op** for any `N` — zero overhead and no artifacts. Above that, the shared per-axis bundle size `s` is derived from the grid and the knob (see `bundle_size` below), and every bundled position is kept in-distribution (`≤ 79`, HRDiT's `group_num = 80` ceiling).

### Usage

1. Add the **SPA (HRDiT)** node after your model loader (under `model_patches/position_encoding`).
2. Set `width`/`height` to match your latent.
3. Leave `model_type: auto` (or force it). SPA auto-detects the architecture and reads `theta` / `axes_dim` from the model.
4. Set `bundle_size` to the paper's `N` (tokens per bundle): `0` = auto, `1` = off, `2..8` explicit. **Recommended: `3` at 2K, `5` at 4K** (paper §4.1). `0` (auto) derives the minimal compression that keeps every bundled position in-distribution (HRDiT `group_num = 80` ceiling). SPA is automatically a **no-op** while the grid is inside the model's trained extent (≤ 1024px). The averaged-pass count is `2s − 1`, capped at 15.
5. Leave `spa_steps` at `3` (HRDiT default): SPA runs only on the first 3 denoising steps of each generation — later steps run at baseline speed. Set `0` to run SPA on every step. Optionally combine with `spa_start_sigma < 1.0` for an additional sigma-threshold gate.
6. Connect the patched `MODEL` to your KSampler.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_type` | auto | Same detection as DyPE (`flux` / `nunchaku` / `qwen` / `zimage` / `anima`). Reads `theta` & `axes_dim` from the model. |
| `method` | ntk | Base RoPE method on the bundled coords. SPA always uses `ntk_factor = 1.0` (no extrapolation) on the bundled positions; this only selects which base RoPE the bundle is built from. |
| `enable_spa` | True | Disable to emit the model's base RoPE unchanged. |
| `bundle_size` | 0 (auto) | The paper's `N` = **tokens per bundle** (paper §4.1). `0` = auto (minimal compression keeping every bundled position ≤ 79, i.e. HRDiT `group_num = 80`). `1` = off (passthrough). `2..8` = explicit; **recommended `3` at 2K, `5` at 4K**. While the grid is inside the trained extent (`max_pos ≤ 64`, e.g. ≤ 1024px) SPA is automatically a no-op for any `N`. Explicit `N` is floored by the in-distribution minimum (never out-of-distribution). Legacy values `≥ 32` (old `group_num` semantics) are migrated to auto with a one-time warning. The averaged-pass count is `2s − 1`, capped at 15. |
| `spa_steps` | 3 | Step-count gating (HRDiT `--spa_steps`): SPA runs only on the first `spa_steps` denoising steps of each generation; later steps run at baseline speed. `0` = all steps (backward compatible). A new generation (sigma jump-up) resets the counter. |
| `spa_start_sigma` | 1.0 | Optional sigma-threshold gate (AND-combined with `spa_steps`): SPA runs only while the current sigma is **above** this threshold. `1.0` = no sigma gating (default). |

> **Performance:** With the defaults (`spa_steps = 3`, `N = 3` at 2K / `N = 5` at 4K) expect **zero overhead at ≤ 1024px** (trained-extent no-op) and roughly **1.3–1.8×** total inference time at 2K/4K (SPA's `2s − 1` averaged passes run only on the first 3 steps; the variant RoPEs and delta rotations are cached per grid). Setting `spa_steps = 0` runs SPA on every step and raises the cost to ~`2s − 1`× while active.

> **Note:** SPA supports **FLUX, Qwen/Krea-2, Z-Image, and Anima/Cosmos**. **Nunchaku is not supported** in v1: its fused/quantized attention kernels bypass the SPA hook, so applying SPA to a Nunchaku model logs a warning and returns the model unchanged. For Anima, the temporal RoPE axis is left untouched and per-axis NTK factors are preserved; only the spatial (h, w) axes are bundled.

> [!WARNING]
> **SPA and DyPE/SEGA are mutually exclusive** in v1. Apply only one — stacking them raises `ValueError("SPA and DyPE/SEGA are mutually exclusive in v1. Apply only one.")`. Use SPA for static spatial-disorder correction, or DyPE/SEGA for dynamic spectral/scale extrapolation.

### When to use SPA vs DyPE/SEGA

*   **SPA alone:** fix high-res *spatial disorder* with a small, bounded sampling overhead (~1.3–1.8× with the `spa_steps = 3` default) and no timestep coupling.
*   **DyPE/SEGA:** full dynamic extrapolation (spectral/scale progression) for resolutions far beyond native.
*   **Not both:** SPA and DyPE/SEGA cannot be combined — they are mutually exclusive in v1.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

The easiest way to install is via **ComfyUI Manager**. Search for `ComfyUI-DyPE` and click "Install".

Alternatively, to install manually:

1.  **Clone the Repository:**

    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```sh
    git clone https://github.com/wildminder/ComfyUI-DyPE.git
    ```
2. **Start/Restart ComfyUI:**
   Launch ComfyUI. No further dependency installation is required.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🛠️ Usage

Using the node is straightforward and designed for minimal workflow disruption.

1.  **Load Your Model:** Use your preferred loader (e.g., `Load Checkpoint` for Flux, `Nunchaku Flux DiT Loader`, `ZImage` loader, or the Anima/Krea-2 UNET loaders).
2.  **Add the DyPE Node:** Add the `DyPE for FLUX` node to your graph (found under `model_patches/unet`).
3.  **Connect the Model:** Connect the `MODEL` output from your loader to the `model` input of the DyPE node.
4.  **Set Resolution:** Set the `width` and `height` on the DyPE node to match the resolution of your `Empty Latent Image`.
5.  **Connect to KSampler:** Use the `MODEL` output from the DyPE node as the input for your `KSampler`.
6.  **Generate!** That's it. Your workflow is now DyPE-enabled.

> [!NOTE]
> This node specifically patches the **diffusion model (UNet)** positional embeddings. It does not modify the CLIP or VAE models.

### Node Inputs

#### 1. Model Configuration
*   **`model_type`**:
    *   **`auto`**: Attempts to automatically detect the model architecture. Recommended.
    *   **`flux`**: Forces Standard Flux logic.
    *   **`nunchaku`**: Forces Nunchaku (Quantized Flux) logic.
    *   **`qwen`**: Forces Qwen Image logic (also used for **Krea-2**, which shares the Qwen architecture).
    *   **`zimage`**: Forces Z-Image (Lumina 2) logic.
    *   **`anima`**: Forces Anima/Cosmos logic.
*   **`base_resolution`**: The native resolution the model was trained on.
    *   Flux / Z-Image: `1024`
    *   Qwen / Krea-2: `1328` (Recommended setting for Qwen-family models)
    *   Anima/Cosmos: `1920` (native `max_img` is 240 latent = 1920px; auto-detected from the model)

#### 2. Method Selection
*   **`method`**:
    *   **`vision_yarn`:** A novel variant designed specifically for aspect-ratio robustness. It decouples structure from texture: low frequencies (shapes) are scaled to fit your canvas aspect ratio, while high frequencies (details) are scaled uniformly. It uses a dynamic attention schedule to ensure sharpness.
    *   **`yarn`:** The standard YaRN method. Good general performance but can struggle with extreme aspect ratios.
    *   **`ntk`:** Neural Tangent Kernel scaling. Very stable but tends to be softer/blurrier at high resolutions.
    *   **`pi`:** Position Interpolation. Scales positions uniformly (`pos / s^κ(t)`) with a time-dependent exponent. Preserves local structure well; a good alternative when `ntk` over-smooths.
    *   **`base`:** No positional interpolation (standard behavior).

##### Scaling Options
*   **`yarn_alt_scaling`** (Only affects `yarn` method):
    *   **Anisotropic (High-Res):** Scales Height and Width independently. Can cause geometric stretching if the aspect ratio differs significantly from the training data.
    *   **Isotropic (Stable Default):** Scales both dimensions based on the largest axis. .
    *   *Note: `vision_yarn` automatically handles this balance internally, so this switch is ignored when `vision_yarn` is selected.*

> [!TIP]
> **Z-Image (Lumina 2) Specifics:** 
> *   Z-Image models use a very low RoPE base frequency (`theta=256`).
> *   **Geometric Stretching:** To prevent vertical stretching, the node automatically enforces **Isotropic Scaling** for Z-Image, regardless of user settings.
> *   **Method Choice:** recommend **`vision_yarn`** or **`ntk`**. Standard `yarn` may produce artifacts.

> [!TIP]
> **Anima/Cosmos Specifics:**
> *   The native patch grid and per-axis NTK factors are read from the model, so DyPE only extrapolates beyond the native 1920px training resolution.
> *   **Method Choice:** **`vision_yarn`** is recommended. Other methods may produce speckle noise at ultra-high resolutions (>2K).

#### 3. Dynamic Control
*   **`enable_dype`**: Enables or disables the **dynamic, time-aware** component of DyPE.
    *   **Enabled (True):** Both the noise schedule and RoPE will be dynamically adjusted throughout sampling. This is the full DyPE algorithm.
    *   **Disabled (False):** The node will only apply the dynamic noise schedule shift. The RoPE will use static extrapolation.
*   **`dype_scale`**: (λs) Controls the "magnitude" of the DyPE modulation. Default is `2.0`.
*   **`dype_exponent`**: (λt) Controls the "strength" of the dynamic effect over time.
    *   `2.0`: Recommended for **4K+** resolutions. Aggressive schedule that transitions quickly to clean up artifacts.
    *   `1.0`: Good starting point for **~2K-3K** resolutions.
    *   `0.5`: Gentler schedule for resolutions just above native.

#### 4. Advanced Noise Scheduling
*   **`base_shift` / `max_shift`**: These parameters control the Noise Schedule Shift (`mu`). In this implementation, `max_shift` (Default 1.15) acts as the target shift for any resolution larger than the base.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🚀 PixelRush Node

**PixelRush** is a training-free, cascade-based high-resolution generation node. It turns
high-resolution generation into a sequence of coarse-to-fine cascade refinements: generate a
native-resolution image, upscale it, then use a single partial DDIM inversion + single
denoising step per overlapping latent patch to add detail rather than regenerate the whole
image from noise. Works with any ComfyUI model (SDXL, SD1.5, FLUX, Qwen, etc.).

### VAE-space operation (important)

PixelRush **operates entirely in VAE latent space** (latent std ≈ 1). The injected model
adapters convert to model space internally (via `process_latent_in`) only when running the
diffusion model, and the predicted epsilon is returned at std ≈ 1 (it is **not** scaled back
by `process_latent_out`).

This is required for models whose `process_latent_in` scales the latent down — most notably
**SDXL** (`scale_factor = 0.13025`). If the algorithm ran in model space, the latent would
have std ≈ 0.13 while the fixed-magnitude noise injection has std ≈ 0.95, so noise would
dominate the signal ~6× and the output would look "totally noisy". Running in VAE space keeps
the noise injection (std ≈ 0.95) balanced against the signal (std ≈ 1), exactly as the
reference PixelRush implementation expects.

The behavior is controlled by the `operate_in_vae_space` flag on `PixelRushConfig`
(**default: `True`**). Setting it to `False` restores the legacy model-space path (only as a
fallback; the VAE-space path is the recommended default).

### Usage

1. Load your model (e.g. `Load Checkpoint` for SDXL, `Flux` loader, `Qwen Image` loader).
2. Generate a base latent at native resolution (e.g. `Empty Latent Image` at 1024×1024 for SDXL).
3. Add the **PixelRush** node and connect `model`, `vae`, `positive`, `negative`, and the base `latent_image`.
4. Set `num_cascade_stages` (1 = 2× upscale, 2 = 4×, 3 = 8×) and tune `noise_lambda` / `overlap` / `patch_h` / `patch_w`.
5. The node outputs a refined latent — connect it to a `VAE Decode` node.

> [!NOTE]
> PixelRush calls the diffusion model directly (not through ComfyUI's sampler/guider), so it
> performs its own CFG and prediction-type conversion (EPS, CONST/flow, V_PREDICTION, X0).

## Changelog

#### v2.6.1 — SPA bundle-size semantics & speed fix (2026-08-15)
*   **`bundle_size` is now the paper's `N` (tokens per bundle):** `0` = auto, `1` = off, `2..8` explicit (recommended `3` @ 2K, `5` @ 4K). The knob was previously implemented as HRDiT's `group_num` (target bundles per axis), which over-compressed the grid into big patches — the source of the *pixelated / mosaic* output at `bundle_size > 2`. Legacy values `≥ 32` are migrated to auto with a one-time warning.
*   **Trained-extent gate:** SPA is an automatic **identity no-op** while the grid is inside the model's trained extent (`max_pos ≤ 64`, i.e. ≤ 1024px) — no big-patch artifacts, zero overhead.
*   **`spa_steps` (new, default `3`):** HRDiT-faithful leading-step gating — SPA runs only on the first 3 denoising steps of each generation (a sigma jump-up resets the counter). This cuts the `bundle_size > 2` slowdown from ~10× to ~1.3–1.8×. `0` = all steps.
*   **Delta-rotation cache:** the `inv(base) @ variant` rotations are composed once per grid (not per attention call), removing the per-call overhead.
*   **HAP (Head-adaptive Attention Pruning)** — the paper's per-head sparse-attention speed-up — is tracked as future work (not in this release).

#### v2.6.0 — SPA (HRDiT) Node
*   **SPA Node:** Added **SPA** (Spatial Position Alignment, HRDiT arXiv 2608.07003) — a static, training-free RoPE patch that fixes high-resolution *spatial disorder* by bundling token indices into a few bundles, sliding the bundle boundaries `N` times, and **averaging the `N` attention outputs** (faithful to HRDiT `_spa_attention`). Supports FLUX, Qwen/Krea-2, Z-Image, and Anima/Cosmos; **Nunchaku is unsupported** (fused kernels bypass the hook — logs a warning, returns the model unchanged). Anima's temporal axis and per-axis NTK factors are preserved.
*   **Auto bundle size:** `N = 5` at ≥4K, `N = 3` at ≥2K, `1` otherwise (no-op). Configurable via `bundle_size`.
*   **Composable:** SPA is **mutually exclusive** with DyPE/SEGA in v1 (apply only one).
*   **Example workflow:** added `example_workflows/SPA_basic.json` (2048×2048 FLUX + SPA).

#### PixelRush — SDXL noise-dominance fix
*   **VAE-space operation:** PixelRush now runs entirely in VAE latent space (std ≈ 1) and converts to model space only inside the `predict_eps` adapter. This fixes the SDXL "totally noisy" output caused by `process_latent_in` scaling the latent down to std ≈ 0.13 (noise injection std ≈ 0.95 then dominated ~6×).
*   **`operate_in_vae_space` flag:** added to `PixelRushConfig` (default `True`). `False` restores the legacy model-space path as a fallback.
*   **Regression tests:** added `TestPixelRushCascadeVAESpace` guarding `out.std()/z0.std() < 2.0` for a realistic SDXL mock (was > 6 before the fix).

#### v2.5.0
*   **SEGA Node:** Added **SEGA** (Spectral-Energy Guided Attention) — a new node that computes per-RoPE-dimension mscale from the latent's Fourier spectrum at each denoising step. Content-aware attention sharpening for FLUX/Qwen. Uses NTK as base extrapolation with per-dim spectral refinement.
*   **5D Latent Support:** SEGA wrapper handles both 4D `(B,C,H,W)` and 5D `(B,C,T,H,W)` latents for video models.
*   **Native Patch Grid:** SEGA reads Anima's native `max_img_h`/`patch_spatial` for correct scale computation.

#### v2.4.0
*   **Anima/Cosmos Support:** Added support for **Anima/Cosmos** models. Reads the model's native per-axis NTK factors and patch grid (`max_img_h/w`, `patch_spatial`) so DyPE only extrapolates beyond native resolution. Recommended method: `vision_yarn`.
*   **Krea-2 Support:** Added support for **Krea-2** (Qwen-family architecture, auto-detected).
*   **State Pollution Fix:** Patch parameters are now cached on the `ModelPatcher` to avoid re-patching and state leakage across runs.
*   **Example Workflows:** Added Anima and Krea-2 example workflows.

#### v2.3.0
*   **Z-Image Overhaul:** Fixed geometric stretching artifacts
*   **Method Fixes**

#### v2.2.0
*   **Z-Image Support:** Added experimental support for **Z-Image (Lumina 2)** architecture.

#### v2.1.0
*   **New Architecture Support:** Added support for **Qwen Image** and **Nunchaku** (Quantized Flux) models.
*   **Modular Architecture:** Refactored codebase into a modular adapter pattern (`src/models/`) to ensure stability and easier updates for future models.
*   **UI Updates:** Added `model_type` selector for explicit model definition.

#### v2.0.0
*   **Vision-YaRN:** Introduced the `vision_yarn` method for decoupled aspect-ratio handling.
*   **Dynamic Attention:** Implemented quadratic decay schedule for `mscale` to balance sharpness and artifacts.
*   **Start Sigma:** Added `dype_start_sigma` control.

#### v1.0.0
*   **Initial Release:** Core DyPE implementation for Standard Flux models.
*   **Basic Modes:** Support for `yarn` (Isotropic/Anisotropic) and `ntk`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ❗ Important Notes & Best Practices

> [!IMPORTANT]
> **Limitations at Extreme Resolutions (4K)**
> While DyPE significantly extends the capabilities of DiT models, generating perfectly clean 4096x4096 images is still a limitation of the base model itself. Even with DyPE, you are pushing a model trained on ~1 megapixel to generate 16 megapixels. You may still encounter minor artifacts at these extreme scales.

> [!TIP]
> **Dealing with Speckle Noise**
> At extreme resolutions (4K+), you may notice high-frequency "speckle" noise in focused areas (e.g., hair, eyes). This is a side effect of scaling the model's attention mechanism beyond its training limits.
> 
> **How to fix:**
> 1.  **Increase `dype_exponent`:** Try raising this to `3.0` or `4.0` or any other higher values.
> 2.  **Use LoRAs:** Smoothing or "Detailer" LoRAs can help suppress high-frequency artifacts.

> [!TIP]
> **Experimentation is Required**
> There is no single "magic setting" that works for every prompt and every resolution. To achieve the best results:
> *   **Test different Methods:** Start with `vision_yarn`, but try `yarn` if you encounter issues.
> *   **Adjust `dype_exponent`:** This is your main knob for balancing sharpness vs. artifacts.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

*   **Noam Issachar, Guy Yariv, and the co-authors** for their groundbreaking research and for open-sourcing the [DyPE](https://github.com/guyyariv/DyPE) project.
*   **The ComfyUI team** for creating such a powerful and extensible platform for diffusion model research and creativity.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-DyPE.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-DyPE/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-DyPE.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-DyPE/issues
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-DyPE.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-DyPE/network/members
