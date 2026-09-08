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
| **❖ [HiFlow](#user-content-hiflow)** | Trajectory-guided flow upscaling for rectified-flow models (FLUX, Qwen-Image, Krea2, Z-Image, …). |

### Which method when?

Two families: **model patches** alter how your own KSampler run attends (no image input) — best for *native* high-res generation; **cascades** consume an existing latent/image and refine it.

| Method | Models | Mechanism | Takes your image | Output character |
|:---|:---|:---|:---:|:---|
| **DyPE** | FLUX, Nunchaku, Qwen/Krea-2, Z-Image, Anima | Dynamic position-encoding extrapolation | ✗ | Native high-res generation |
| **SEGA** | FLUX, Nunchaku, Qwen/Krea-2, Z-Image, Anima | Spectral-energy RoPE sharpening | ✗ | Native high-res generation |
| **SPA** | FLUX, Qwen/Krea-2, Z-Image, Anima | Position-bundle attention alignment | ✗ | Native high-res generation |
| **HAP** | FLUX, Qwen/Krea-2, Z-Image, Anima | Calibrated sparse attention (speed) | ✗ | Native high-res generation |
| **PixelRush** | Any (SDXL, SD1.5, FLUX, Qwen, …) | Patch-wise low-denoise img2img cascade | ✓ | Faithful upscale + refinement |
| **FreeScale** | FLUX-family DiTs | Scale-fused attention + self-cascade | ✓ | Regenerative hi-res, mostly new content |
| **HiFlow** | Flow models (FLUX, Qwen-Image, Krea2, Z-Image, …) | Time-matched reference trajectory guidance | ✓ | Structure-faithful flow upscale |

> [!TIP]
> **Quick picker:** starting from noise → DyPE (or SEGA), add SPA if you see repeated/collapsed structures, add HAP for speed. Starting from an existing image → PixelRush to keep it faithful, FreeScale to re-imagine it at high res (lower its `noise_timestep` for more fidelity), HiFlow for FLUX-family flow models — it reuses the whole base-resolution denoising trajectory as guidance, so structure survives while detail is re-synthesized.

<a id="user-content-dype"></a>

### ❖ DyPE

Dynamic Position Extrapolation ([paper](https://arxiv.org/abs/2411.17087), [code](https://github.com/guyyariv/DyPE)). Adjusts positional encodings at each denoising step to match the current stage of generation — low-frequency structure early, fine detail later. Training-free, no additional sampling cost.

**Usage:** Load model → add `DyPE` (under `WMNodes/image`) → connect `MODEL` → set `width`/`height` to match your latent → connect to KSampler.

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
| `refiner_model` | **Optional** separate refiner model (paper setup: SDXL base + SDXL-Turbo). When not connected, the base model refines too. |
| `noise_lambda` | Noise injection coefficient — the weight of the model's prediction (paper default 0.95 = 95% prediction + 5% random noise). |
| `noise_injection` | `slerp` (paper default) or `additive` (legacy pre-2.9 behavior, kept for workflows tuned against it). |
| `overlap` | Overlap between adjacent patches (blends seams). |
| `gaussian_sigma` | Analytic Gaussian feather sigma (paper default 24; rule of thumb: σ ≈ patch_size / 5). |
| `patch_h` / `patch_w` | Latent patch size (~native spatial size keeps VRAM flat). |

> [!NOTE]
> PixelRush calls the diffusion model directly (not through ComfyUI's sampler), performing its own CFG and prediction-type handling for EPS, flow, V-prediction and X0 models.

> [!IMPORTANT]
> **2.9 migration notes:** the noise injection now uses the paper's SLERP with λ weighting the model's prediction (set `noise_injection` to `additive` for the legacy formula); `gaussian_sigma` default moved 8 → 24 and its range extends to 128; the `gaussian_kernel_size` input was removed (the mask is now the paper's analytic Gaussian — old workflows simply ignore the stale value).

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

<a id="user-content-hiflow"></a>
### ❖ HiFlow

Training-free high-resolution upscaling for **rectified-flow models** (FLUX, Qwen-Image, Krea2, Z-Image, …) via flow-aligned guidance ([paper](https://arxiv.org/abs/2504.06232), NeurIPS 2025). The base-resolution sampling runs once, recording every per-step clean prediction; each upscale stage then reuses that **time-matched trajectory** as a virtual reference — initialization alignment seeds the stage from it, direction alignment keeps low frequencies true to it, acceleration alignment matches its detail-generation rhythm. Structure survives; high-res detail is synthesized fresh.

**Usage:** connect `model` (flow models only), `vae`, `positive`, `negative` and a base latent at native resolution (e.g. `EmptySD3LatentImage`) → set `noise_seed` + `scale_factor` → decode. The cascade noises the latent to the first sigma itself — an empty latent + seed reproduces the reference pipeline's from-noise start. Chain `DyPE (ntk)` before the loader for RoPE extrapolation at the scaled resolution.

<details>
<summary><b>Inputs & Parameters</b></summary>

| Parameter | Default | Description |
|:---|:---:|:---|
| `cfg` | 3.5 | Base-stage CFG (FLUX-dev default). Guidance-free models (Z-Image, Chroma) or empty negatives: leave at 1.0 — CFG is auto-skipped when the negative carries no tokens. |
| `steps` | 30 | Base-stage steps; their clean predictions form the reference trajectory. |
| `guidance` | 4.5 | Guided-stage CFG (paper uses 4.5–6). Same auto-skip rule as `cfg`. |
| `steps_per_stage` | 16 | Guided steps per cascade stage (upper bound — the stage walks schedule sigmas below `tau`). |
| `noise_seed` | 0 | Seed for the base noise and each stage's initialization noise. |
| `denoise` | 1.0 | Img2img strength for a content latent (KSampler convention): 1.0 regenerates from pure noise; lower keeps more of the input (ignored for an empty latent). |
| `tau` | 0.6 | Stage-entry noise level (paper cascade: 0.6, 0.3, 0.3). Lower = stronger content preservation. |
| `filter_ratio` | 0.2 | Butterworth low-pass cutoff D for direction alignment (paper 0.4, repo 0.2). |
| `alpha_scale` / `beta_scale` | 1.0 / 0.5 | Direction / acceleration strength multipliers. |
| `upsampling` | latent | Per-step reference upsample: `latent` bicubic (repo default) or `pixel` decode→sharpen→encode. The stage anchor is always the pixel round-trip. |
| `scale_factor` | 2.0 | Output scale relative to the input latent: 2 = double each side, 1 = unchanged, 0.5 = half. Upscales run 2× doubling stages (scales between 1 and 2 give one 2× stage); below 1 runs one refinement stage at the smaller size. |

</details>

> [!TIP]
> **HiFlow inherits the reference's structure** — including its mistakes. Generate a good base first; `tau` lower keeps more of it, higher re-imagines. 3D-latent image models (Krea2, Qwen-Image — Wan21 format, Qwen VAE) work as single-frame (T=1) latents; actual multi-frame/video input is rejected.
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
| `HiFlowNode` | HiFlow | Trajectory-guided flow upscaling (initialization + direction + acceleration alignment). |

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

### v2.15.0 — 2026-09-08
- **Restructured the pack layout + unified the node category.** All node definitions now live in a dedicated `nodes/` folder (`nodes/dype.py`, `sega.py`, `spa.py`, `hap.py`, `hap_calibrate.py`, `freescale.py`, `pixelrush.py`, `hiflow.py`); `src/` holds engines/implementation only and the pack `__init__.py` just registers the extension. All 8 nodes moved to the single **`WMNodes/image`** menu category (previously split across two menu paths). No node ids, inputs, defaults, or behavior changed — workflows keep loading. Also merges PR #41 (FreeScale fp16 antialiased-bicubic crash fix).

### v2.14.1 — 2026-09-07
- **Fixed HiFlow Krea2/Qwen-Image noising crash** (user-reported `torch.cat` size mismatch, "Expected size 1 but got size 16"): the v2.12.1 model-space noising called the model's `process_latent_in` on the 4D core tensor, but Wan21's per-channel mean/std stats are shaped `[1,C,1,1,1]` — a 4D tensor against 5D stats **broadcasts silently to `[B,C,C,H,W]` garbage** (the model reads T=16=channels). The node now wraps the noising conversions ndim-transparently: unsqueeze → convert in true 5D model space → squeeze back, so the cascade's σ-mix runs on 4D tensors with correctly-normalized values. The node-test mock now uses Wan21-faithful stats (replicating the broadcast hazard — the earlier affine mock masked the bug class).

### v2.14.0 — 2026-09-07
- **HiFlow: 3D-latent image model support — Krea2 and Qwen-Image work now** (plan 2026-09-07, user-reported Krea2 "does not support 3D-latent (video) models" rejection). These models are *image* models with a 5D Wan21-style latent layout `[B,C,1,H,W]` (Qwen VAE) — the old gate conflated 5D tensors with video. The gate now accepts `latent_dimensions=3` image models and rejects only actual multi-frame (T>1) input; the node bridges 5D↔4D around the 4D core (the PixelRush convention): latents squeeze on entry and re-expand on output, the model-call adapter unsqueezes before `process_latent_in` (Wan21's per-channel mean/std stats broadcast on 5D only), and the VAE adapters speak the Qwen-VAE `latent_dim=3` boundary (decode frame-slices the `[B,T,H,W,3]` image; encode lets the VAE do its own `not_video` unsqueeze). Qwen-Image gains real (previously gate-blocked) support from the same fix; Anima inherits it, untested on real runs.

### v2.13.0 — 2026-09-04
- **HiFlow: `target_resolution` replaced by `scale_factor`** (user request — the absolute pixel target was unintuitive). `scale_factor` is relative to the input latent: 2 doubles each side, 1 returns the base unchanged, 0.5 halves it via a single refinement stage. Scales now apply per side (the absolute form over-upscaled the short side of non-square images), upscales keep the paper's 2×-stage quantization (a 1.5 scale runs one 2× stage), and downscale scales (0.25–1) run one guided stage at the smaller size. Example workflow updated.

### v2.12.1 — 2026-09-03
- **Fixed HiFlow img2img noising space** (user-reported "drastic changes at any usable denoise; only 0.05 looks right"): the σ-mix `σ·ε + (1−σ)·content` now runs in MODEL space (convert the content with `process_latent_in` first, convert the mix back), matching ComfyUI's KSampler pipeline (samplers.py converts the content before the σ-mix). Mixing in VAE space scaled the noise by the latent format's `scale_factor` (Flux/Z-Image: 0.3611 — **2.77× under-noised**) and added spurious shift offsets, so the model aggressively "corrected" every img2img input. The guided-stage initialization σ-mix got the same fix. The sampler itself (rectified-flow Euler) and scheduler spacing (model-table "simple") were already faithful — the defect was the space mix, not the routine.

### v2.12.0 — 2026-09-03
- **HiFlow img2img: `denoise` parameter** (user-reported "connecting the real latent does nothing"): with the full flow schedule the base start σ=1 zeroes the content weight, so a sampler latent connected to the node was silently ignored. The KSampler convention now applies — `denoise` < 1 truncates the base schedule so the walk enters below σ=1 and keeps `(1−σ_start)` of the input latent (an empty latent always runs the full schedule; the node warns when a content latent meets `denoise=1.0`).

### v2.11.0 — 2026-09-03
- **HiFlow realigned with the authors' implementation** (plan 2026-09-03-realignment, user-reported Z-Image "burned and blurred" output identical in both upsampling modes): the base stage now starts from noised latent instead of the raw input (an `EmptySD3LatentImage` was being sampled verbatim as all-zeros "noise" — the root cause); stage initialization anchors on the previous chain's final image (always pixel round-tripped) instead of the time-matched reference; the reference velocity derives from the walk's own state; trajectories store the raw (uncorrected) x0 so guidance doesn't compound across stages; α/β follow the code's linear-in-index schedule, not the paper's σ/σ_entry (which over-locks low frequencies late on shifted schedules). New `noise_seed` input drives the base and per-stage init noise reproducibly.

### v2.10.0 — 2026-09-03
- **New HiFlow node** (plan 2026-09-03): training-free high-resolution upscaling for rectified-flow models (FLUX, Qwen-Image, …) via flow-aligned guidance (arXiv:2504.06232). The base-resolution trajectory is recorded per-step and guides each upscale stage through initialization, direction and acceleration alignment. Non-flow and video models are rejected with a pointer to PixelRush.

### v2.9.1 — 2026-09-02
- **Fixed the PixelRush noise-injection λ convention** (user-reported "structure visible but completely noisy, soft blurred patches"). The injection now uses `slerp(eps_random, eps_refined, λ)` — λ weights the **model's prediction** (0.95 = 95% prediction + 5% noise). The previous order (`slerp(eps_pred, eps_random, λ)`) made λ=0.95 mean 99.6% pure random noise: at real scales per-pixel noise std ≈ 1.17 vs signal ≈ 1.0, which rendered through the Gaussian feather as the reported soft-patch noise. The `additive` legacy mode uses the same convention (`eps_refined + (1−λ)·eps_random`). This was exactly the argument-order caveat `pixelrush-correct.txt` flagged for verification against the authors' implementation.

### v2.9.0 — 2026-09-02
- **PixelRush realigned with the corrected theory** (plan 2026-09-02): standard raw-vector SLERP (with collinear lerp fallback) for the noise injection — the paper's `slerp(eps_pred, eps_random, λ)` is now the default, with the 2026-08-13 additive injection kept as an opt-in (`noise_injection`).
- **Fixed the VAE/model space mixing** in the forward/reverse steps: adapters now convert via `process_latent_in/out`, so the model sees noise at the scale its timestep claims. For SDXL the previous code under-noised 7.7× — the root cause behind the "compressed look" that the additive hack had papered over.
- Generic DDIM transitions (`ddim_deterministic_step` between arbitrary timesteps, `predict_x0_from_epsilon`); analytic Gaussian feather mask (σ default 24, `gaussian_kernel_size` input removed).
- **Optional `refiner_model` input** — use a separate distilled refiner (e.g. SDXL-Turbo) as in the paper; the base model drives the partial inversion.
- **Bug fixes:** empty-negative conditioning no longer amplifies eps by `cfg_scale` (CFG is skipped); `alpha_k` NameError with partially-provided adapters; empty positive now raises a clear error.

### v2.8.3 — 2026-08-31
- **Qwen2D VAE support disabled by default.** User reports showed that with the Qwen2D VAE interception installed, loading certain non-Qwen2D (video-style) VAE checkpoints crashed with a size-mismatch error whose traceback passed through this pack's delegation frame — breaking workflows that never used the Qwen2D VAE. The patch now installs only when the environment variable `DYPE_ENABLE_QWEN2D_VAE=1` is set. If you relied on the Qwen2D VAE (Anzhc/Qwen2D-VAE checkpoint with FreeScale/PixelRush on Krea-2/Qwen/Anima), set that variable in your ComfyUI environment to restore the previous behavior.

### v2.8.2 — 2026-08-31
- Fixed graph-build and execution crashes when resolution inputs are `None` (validate_inputs now passes through uninitialized state; execute falls back to 1024)
- Fixed PixelRush crash on float16: antialiased bicubic upsample casts to float32 and restores the original dtype

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
* **The HiFlow authors** — [HiFlow](https://arxiv.org/abs/2504.06232) ([code](https://github.com/Bujiazi/HiFlow))
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
