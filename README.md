<div id="readme-top" align="center">
<h1 align="center">ComfyUI-DyPE</h1>

<img src="https://github.com/user-attachments/assets/4f11966b-86f7-4bdb-acd4-ada6135db2f8" alt="ComfyUI-DyPE Banner" width="70%">

<p align="center">
ComfyUI custom node pack for <strong>ultra-high-resolution generation</strong> (4K and beyond) with Diffusion Transformers — <strong>FLUX</strong>, <strong>Qwen Image</strong>, <strong>Z-Image</strong>, <strong>Anima/Cosmos</strong>, <strong>Krea-2</strong>.
<br />

[![Report Bug][bug-shield]][bug-url] [![Request Feature][feature-shield]][feature-url]

</p>
</div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]

</div>

<br>

## ▷ About

Training-free methods that push pre-trained DiT models far beyond their native resolution — no retraining, no workflow changes. Patch the model once after your loader and generate at 2K, 4K and above.

<div align="center">
<img alt="ComfyUI-DyPE example workflow" width="70%" src="https://github.com/user-attachments/assets/31f5d254-68a7-435b-8e1f-c4e636d4f3c2" />
<p><sub><i>A simple, single-node integration to patch your model for high-resolution generation.</i></sub></p>
</div>

### ❖ Highlights

* **Multi-Architecture** — FLUX, Nunchaku, Qwen Image, Krea-2, Z-Image, Anima/Cosmos
* **High-Resolution Generation** — 4096×4096 and beyond
* **Single-Node Integration** — place after your model loader, done
* **Full Compatibility** — works with existing workflows, samplers and optimization nodes
* **Zero Overhead** — adjustments happen on-the-fly with negligible performance impact

<div align="center">
<img alt="Node" width="70%" src="https://github.com/user-attachments/assets/f85861fd-4d2f-4b57-8058-26881600b7ca" />
</div>

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Nodes

| Node | What it does |
|:---|:---|
| **❖ [DyPE](#user-content-dype)** | Dynamic Position Extrapolation — the core high-res method. |
| **❖ [SEGA](#user-content-sega)** | Content-aware spectral sharpening as an alternative to DyPE. |
| **❖ [SPA (HRDiT)](#user-content-spa-hrdit)** | Fixes spatial disorder (repeated/collapsed structures) at high res. |
| **❖ [HAP (HRDiT)](#user-content-hap-hrdit)** | Sparse-attention acceleration — the speed half of HRDiT. |
| **❖ [PixelRush](#user-content-pixelrush)** | Cascade patch refinement of an existing base image. |
| **❖ [FreeScale](#user-content-freescale)** | Tuning-free self-cascade upscaling. |

<a id="user-content-dype"></a>
### ❖ DyPE

Dynamic Position Extrapolation ([paper](https://arxiv.org/abs/2411.17087), [code](https://github.com/guyyariv/DyPE)). Adjusts positional encodings at each denoising step to match the current stage of generation — low-frequency structure early, fine detail later. Training-free, no additional sampling cost.

**Usage:** Load model → add `DyPE for FLUX` (under `model_patches/unet`) → connect `MODEL` → set `width`/`height` to match your latent → connect to KSampler.

<details>
<summary><b>Inputs & Parameters</b></summary>

#### Model Configuration
* **`model_type`**
    * **`auto`** — auto-detects the architecture. Recommended.
    * **`flux`** — Standard Flux.
    * **`nunchaku`** — Quantized Flux.
    * **`qwen`** — Qwen Image (also used for Krea-2).
    * **`zimage`** — Z-Image (Lumina 2).
    * **`anima`** — Anima/Cosmos.
* **`base_resolution`** — native training resolution of the model.
    * Flux / Z-Image: `1024`
    * Qwen / Krea-2: `1328`
    * Anima/Cosmos: `1920` (auto-detected)

#### Method Selection (`method`)
* **`vision_yarn`** — decouples structure from texture; best aspect-ratio robustness. Recommended default.
* **`yarn`** — standard YaRN; good general performance.
* **`ntk`** — very stable, but softer at high resolutions.
* **`pi`** — Position Interpolation; preserves local structure well.
* **`base`** — no interpolation.

##### Scaling Options
* **`yarn_alt_scaling`** (only affects `yarn`): Anisotropic scales H/W independently (may stretch); Isotropic (default) is stable. Ignored by `vision_yarn`.

#### Dynamic Control
* **`enable_dype`** — full dynamic algorithm (on), or schedule shift only (off).
* **`dype_scale`** — magnitude of the modulation (default `2.0`).
* **`dype_exponent`** — strength over time: `2.0` for 4K+, `1.0` for ~2K–3K, `0.5` just above native.

#### Advanced Noise Scheduling
* **`base_shift` / `max_shift`** — noise-schedule shift control (`max_shift` default `1.15`).

</details>

> [!TIP]
> **Z-Image:** isotropic scaling is enforced automatically. Prefer `vision_yarn` or `ntk`.
> **Anima/Cosmos:** prefer `vision_yarn`; other methods may produce speckle noise above 2K.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="user-content-sega"></a>
### ❖ SEGA

Spectral-Energy Guided Attention ([code](https://github.com/rajabi2001/sega)). Content-aware RoPE sharpening derived from the latent's frequency spectrum. Use as an alternative to DyPE on FLUX/Qwen.

**Usage:** Add the `SEGA` node after your model loader → set `width`/`height` to match your latent → tune `mscale_alpha` and `spread_min`/`spread_max`.

<div align="center">
<img alt="Example sega" src="https://github.com/user-attachments/assets/c9d812c8-a88b-4e8d-bb84-0f4bd5ef18ef" />
</div>

<details>
<summary><b>Inputs & Parameters</b></summary>

| Parameter | Default | Description |
|:---|:---:|:---|
| `method` | sega | `sega` = NTK + spectral mscale, `ntk` = NTK only |
| `mscale_alpha` | 0.15 | Spectral redistribution amplitude |
| `mscale_beta` | 1.5 | tanh sharpness |
| `mscale_min` | 1.0 | Floor for per-frequency mscale |
| `spread_min` | 0.0 | Min spectral spread (early steps) |
| `spread_max` | 1.0 | Max spectral spread (late steps) |
| `spread_alpha` | 1.5 | Spread schedule non-linearity |
| `base_mscale_formula` | power_res | `power_res` or `log_res` |
| `base_mscale_coefficient` | 0.08 | κ (paper default) |

</details>

> [!NOTE]
> SEGA builds on NTK. If NTK doesn't work for your model (e.g. Anima), use DyPE `vision_yarn` instead.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="user-content-spa-hrdit"></a>
### ❖ SPA (HRDiT)

Spatial Position Alignment, from the **HRDiT** paper ([arXiv 2608.07003](https://arxiv.org/abs/2608.07003)). A static, training-free patch that fixes high-resolution **spatial disorder** — repeated structures and positional collisions when pushing past native resolution. Resolution-aware (automatic no-op ≤ 1024px) with bounded overhead at 2K/4K. Mechanism: bundles token positions into groups of `N`, slides the bundle boundary per axis (`2s − 1` variants), and **averages the attention outputs** across variants — never the RoPE matrices themselves.

**Usage:** Add the `SPA (HRDiT)` node after your model loader → set `width`/`height` → leave `model_type: auto` → connect to KSampler. Recommended `bundle_size`: `3` at 2K, `5` at 4K (`0` = auto).

<details>
<summary><b>Inputs & Parameters</b></summary>

| Parameter | Default | Description |
|:---|:---:|:---|
| `model_type` | auto | Same detection as DyPE. Reads `theta` & `axes_dim` from the model. |
| `enable_spa` | True | Disable to pass the model through unchanged. |
| `bundle_size` | 0 (auto) | Tokens per bundle (paper's `N`). `0` = auto, `1` = off, `2..8` explicit. Auto no-op inside the model's trained extent (≤ 1024px). |
| `spa_steps` | 3 | SPA runs only on the first 3 denoising steps; later steps run at baseline speed. `0` = all steps. |
| `spa_start_sigma` | 1.0 | Optional sigma-threshold gate (combined AND with `spa_steps`). |
| `spa_layer_filter` | "" | Restrict SPA to a subset of layers, e.g. `"0-18,38-57"`. Empty = every layer. |
| `proportional_attention` | False | HRDiT proportional attention scaling for long sequences. No-op at/below 1024px. |

> **Performance:** ~zero overhead at ≤ 1024px; roughly **1.3–1.8×** total inference time at 2K/4K with defaults.

> **Model support:** FLUX, Qwen/Krea-2, Z-Image, Anima/Cosmos. **Nunchaku not supported** (logs a warning, returns the model unchanged).

</details>

> [!WARNING]
> **SPA and DyPE/SEGA are mutually exclusive** — apply only one.
> * **SPA** — fix spatial disorder with small, bounded overhead.
> * **DyPE/SEGA** — full dynamic extrapolation far beyond native resolution.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="user-content-hap-hrdit"></a>
### ❖ HAP (HRDiT)

Head-Adaptive attention Pruning, from the same **HRDiT** paper — the **speed** half complementing SPA (the quality half). Each attention head only sees the keys it actually needs, via a pre-calibrated scope plan executed through block-sparse attention. Composable with SPA in any order.

A ready-to-use FLUX scope plan ships at `configs/scope_plan_flux.json`.

**Usage:** Add the `HAP (HRDiT)` node after your model loader → point `scope_plan_path` at a plan JSON → connect to KSampler (optionally through an SPA node first).

<details>
<summary><b>Inputs & Parameters</b></summary>

| Parameter | Default | Description |
|:---|:---:|:---|
| `scope_plan_path` | `configs/scope_plan_flux.json` | Path to the scope-plan JSON. Relative paths resolve against the repo root. Also accepts a linked `scope_plan` input. |
| `model_type` | auto | Architecture detection. Nunchaku unsupported. |
| `anchor_stride` | 0 | Every Nth image key block stays globally visible. `0` = off. |
| `text_len` | 512 | Leading text tokens always kept visible. |
| `enable_hap` | True | Disable to pass the model through unchanged. |
| `proportional_attention` | False | See SPA. Either node may enable it. |

> **Backends:** fast path needs CUDA + PyTorch ≥ 2.5; otherwise falls back automatically to a correct dense-mask backend.

</details>

<details>
<summary><b>Calibration</b></summary>

Scope plans are model-specific. Calibrate a custom plan with the **HAP Calibrate (HRDiT)** node in-graph, or via the [`calibration/calibrate_hap.py`](calibration/calibrate_hap.py) CLI:

```sh
# Self-contained dry run (no GPU needed):
python calibration/calibrate_hap.py --dry_run --out tmp/scope_plan_toy.json

# Real-model calibration:
python calibration/calibrate_hap.py --model_path /path/to/flux.safetensors \
    --model_type flux --width 4096 --height 4096 --num_prompts 30 \
    --out configs/scope_plan_flux_4k.json
```

Calibrate once per model, then reuse the plan across resolutions and prompts.

From the paper (FLUX, budget 0.1): ~**2.9×** faster attention at 2K, ~**5.5×** at 4K.

</details>

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="user-content-pixelrush"></a>
### ❖ PixelRush

Cascade-based refinement node. Generates at native resolution first, then progressively adds detail through coarse-to-fine cascade refinements — producing crisp 4K output without regenerating the whole image from noise. Works with any ComfyUI model (SDXL, SD1.5, FLUX, Qwen, …).

**Usage:** Generate a base latent at native resolution → connect `model`, `vae`, `positive`, `negative` and the base `latent_image` → set `num_cascade_stages` (1 = 2× upscale, 2 = 4×, 3 = 8×) → decode the output latent.

<details>
<summary><b>Inputs & Parameters</b></summary>

| Parameter | Description |
|:---|:---|
| `num_cascade_stages` | Number of cascade stages — each doubles the resolution. |
| `noise_lambda` | Noise injection strength per cascade stage. |
| `overlap` | Overlap between adjacent patches (blends seams). |
| `patch_h` / `patch_w` | Latent patch size (~native spatial size keeps VRAM flat). |

> [!NOTE]
> PixelRush calls the diffusion model directly (not through ComfyUI's sampler), performing its own CFG and prediction-type handling for EPS, flow, V-prediction and X0 models.

</details>

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<a id="user-content-freescale"></a>
### ❖ FreeScale

Tuning-free higher-resolution generation via scale-fused attention and self-cascade upscaling ([paper](https://arxiv.org/abs/2412.09626), [code](https://github.com/ali-vilab/FreeScale)). Supports FLUX-family DiTs (auto-detected); base-resolution inputs pass through untouched.

<details>
<summary><b>Inputs & Parameters</b></summary>

| Input | Default | Notes |
|:---|:---:|:---|
| `width` / `height` | 2048 | Target resolution (snapped to multiples of 16). |
| `steps` | 20 | Sampler steps per cascade stage. |
| `cfg` | 1.0 | Classifier-free guidance scale. |
| `cascade_stages` | 1 | Number of self-cascade stages (each doubles resolution). |

</details>

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Node Reference

All nodes registered by this pack (V3 schema ids):

| Node id | Display name | Purpose |
|:---|:---|:---|
| `DyPE_FLUX` | DyPE | Dynamic Position Extrapolation for ultra-high-res generation. |
| `SEGA` | SEGA | Spectral-Energy Guided Attention (content-aware sharpening). |
| `SPA` | SPA (HRDiT) | Spatial Position Alignment — fixes spatial disorder. |
| `HAP` | HAP (HRDiT) | Head-Adaptive attention Pruning — the speed half. |
| `HAPCalibrate` | HAP Calibrate (HRDiT) | In-graph scope-plan calibration for HAP. |
| `PixelRushNode` | PixelRush | Cascade refinement for existing latents. |
| `FreeScaleNode` | FreeScale | Tuning-free scale-fusion + self-cascade upscaling. |

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Getting Started

**Via ComfyUI Manager:** Search `ComfyUI-DyPE` → Install.

**Manual install:**

```sh
cd ComfyUI/custom_nodes/
git clone https://github.com/wildminder/ComfyUI-DyPE.git
```

Restart ComfyUI. No further dependency installation is required.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Tips & Best Practices

> [!IMPORTANT]
> **Limitations at Extreme Resolutions (4K):** you are pushing a model trained on ~1 megapixel toward 16 megapixels — minor artifacts can still appear even with these methods.

> [!TIP]
> **Speckle noise at 4K+:** increase `dype_exponent` (e.g. `3.0`–`4.0`) or apply smoothing / detailer LoRAs.

> [!TIP]
> **Experiment:** there is no single magic setting — try different methods and adjust `dype_exponent` for the best sharpness/artifact balance.

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Changelog

### v2.8.1 — 2026-08-25
- Fixed valid resolutions being rejected at graph build
- Validation errors are now reported once, for the right input

### v2.8.0 — 2026-08-16
- New **HAP Calibrate** node: calibrate HAP directly in-graph
- HAP accepts calibrated plans either by file or by direct connection
- CLI calibration tooling completed

### v2.7.1 — 2026-08-16
- Fixed crashes on Anima/Cosmos models
- Safer automatic fallbacks instead of hard errors
- SPA and HAP nodes now work in any order

### v2.7.0 — 2026-08-15
- New **HAP** node: sparse-attention acceleration (up to ~5× faster attention at 4K)
- One-click scope-plan calibration pipeline (in-graph + CLI)
- New optional attention scaling and per-layer filtering controls
- SPA and HAP can be composed together

### v2.6.1 — 2026-08-15
- Reworked SPA bundle-size control to match the paper
- Much faster SPA runs (up to ~10× less overhead at strong settings)
- Automatic no-op at/below native resolution

### v2.6.0 — 2026-08-15
- New **SPA** node (HRDiT)

### PixelRush update
- Fixed "totally noisy" output on SDXL models

### v2.5.0
- New **SEGA** node
- Video-model latent support

### v2.4.0
- Anima/Cosmos support
- Krea-2 support
- Stability fixes and new example workflows

### v2.3.0
- Z-Image quality improvements

### v2.2.0
- Experimental Z-Image support

### v2.1.0
- Qwen Image and Nunchaku support
- Modular codebase refactor for easier future model support

### v2.0.0
- New `vision_yarn` method for better aspect-ratio handling
- Sharper results with fewer artifacts
- New start-sigma control

### v1.0.0
- Initial release: core DyPE for FLUX with `yarn` and `ntk` methods

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

## ▓ Acknowledgments

* **Noam Issachar, Guy Yariv and co-authors** — [DyPE](https://github.com/guyyariv/DyPE) ([paper](https://arxiv.org/abs/2411.17087))
* **The SEGA authors** — [SEGA](https://github.com/rajabi2001/sega)
* **The HRDiT team** — [HRDiT](https://arxiv.org/abs/2608.07003) ([code](https://github.com/zylwithxy/HRDiT-HAP)) — basis for SPA & HAP
* **The PixelRush authors** — [PixelRush](https://arxiv.org/abs/2602.12769)
* **Yanhong Zeng et al.** — [FreeScale](https://github.com/ali-vilab/FreeScale) ([paper](https://arxiv.org/abs/2412.09626))
* **The ComfyUI team** — for the platform

<p align="right"><a href="#readme-top" title="back to top">⟔ ▲ ⟓</a></p>

<p align="center">══════════════════════════════════</p>

<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-DyPE.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-DyPE/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-DyPE.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-DyPE/issues
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-DyPE.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-DyPE/network/members
[bug-shield]: https://img.shields.io/badge/Report-Bug-red?style=flat-square&logo=github
[bug-url]: https://github.com/wildminder/ComfyUI-DyPE/issues/new?labels=bug&template=bug-report---.md
[feature-shield]: https://img.shields.io/badge/Request-Feature-blue?style=flat-square&logo=github
[feature-url]: https://github.com/wildminder/ComfyUI-DyPE/issues/new?labels=enhancement&template=feature-request---.md
