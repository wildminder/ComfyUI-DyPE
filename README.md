<a id="readme-top"></a>

<div align="center">
  <h1 align="center">ComfyUI-DyPE</h1>

<img src="https://github.com/user-attachments/assets/4f11966b-86f7-4bdb-acd4-ada6135db2f8" alt="ComfyUI-DyPE Banner" width="70%">

  
  <p align="center">
    A ComfyUI custom node that implements <strong>DyPE (Dynamic Position Extrapolation)</strong>, enabling FLUX-based models to generate ultra-high-resolution images (4K and beyond) with exceptional coherence and detail.
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

DyPE is a novel, training-free method that allows pre-trained diffusion transformers like FLUX (and now **Qwen Image**) to generate images at resolutions far beyond their training data, with no additional sampling cost.

It works by taking advantage of the spectral progression inherent to the diffusion process. By dynamically adjusting the model's positional encodings at each step, DyPE matches their frequency spectrum with the current stage of the generative process—focusing on low-frequency structures early on and resolving high-frequency details in later steps. This prevents the repeating artifacts and structural degradation typically seen when pushing models beyond their native resolution.

<div align="center">

  <img alt="ComfyUI-DyPE example workflow" width="70%" src="https://github.com/user-attachments/assets/e5c1d202-b2c4-474b-b52f-9691ab44c47a" />
      <p><sub><i>A simple, single-node integration to patch your FLUX model for high-resolution generation.</i></sub></p>
  </div>
  
This node provides a seamless, "plug-and-play" integration of DyPE into any FLUX-based workflow.

**✨ Key Features:**
*   **True High-Resolution Generation:** Push FLUX models to 4096x4096 and beyond while maintaining global coherence and fine detail.
*   **Single-Node Integration:** Simply place the `DyPE for FLUX` or `DyPE for Qwen Image` node after your model loader to patch the model. No complex workflow changes required.
*   **Full Compatibility:** Works seamlessly with your existing ComfyUI workflows, samplers, schedulers, and other optimization nodes like Self-Attention or quantization.
*   **Fine-Grained Control:** Exposes key DyPE hyperparameters, allowing you to tune the algorithm's strength and behavior for optimal results at different target resolutions.
*   **Model-Aware Qwen Support:** Automatically infers Qwen patch geometry, adds editing-aware DyPE tapers, and gracefully patches non-FLUX samplers.
*   **Zero Inference Overhead:** DyPE's adjustments happen on-the-fly with negligible performance impact.

<div align="center">
<img alt="Node" width="70%" src="https://github.com/user-attachments/assets/3ef232d2-6268-4e3d-8522-b704dade03ac" />
</div>

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🚀 Getting Started

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

### FLUX Workflows

1.  **Load Your FLUX Model:** Use a standard `Load Checkpoint` node to load your FLUX model (e.g., `FLUX.1-Krea-dev`).
2.  **Add the DyPE Node:** Add the `DyPE for FLUX` node to your graph (found under `model_patches/unet`).
3.  **Connect the Model:** Connect the `MODEL` output from your loader to the `model` input of the DyPE node.
4.  **Set Resolution:** Set the `width` and `height` on the DyPE node to match the resolution of your `Empty Latent Image`.
5.  **Connect to KSampler:** Use the `MODEL` output from the DyPE node as the input for your `KSampler`.
6.  **Generate!** That's it. Your workflow is now DyPE-enabled.

> [!NOTE]
> This node specifically patches the **diffusion model (UNet)**. It does not modify the CLIP or VAE models. It is designed exclusively for **FLUX-based** architectures.

### Qwen Image Workflows

1.  **Load Your Qwen Image Model:** Use the usual `Load Checkpoint` node for `QwenImage`.
2.  **Add the DyPE Node:** Drop in the `DyPE for Qwen Image` node (under `model_patches/unet`).
3.  **Set Width/Height:** Match the values to your target latent/image resolution (the same numbers you use in `Empty Latent Image`).
4.  **Auto-Detect Geometry:** Leave `auto_detect` enabled to let the node read the Qwen patch size and base grid directly from the checkpoint. Disable it only if you need to override the base dimensions for custom fine-tunes.
5.  **Dial In Editing:** Lower `editing_strength` (and pick an `editing_mode`) when you are working on inpainting or image-to-image tasks so DyPE eases off as it preserves source structure.
6.  **Choose Method:** `yarn` is recommended for aggressive extrapolation; switch to `ntk` if you prefer a smoother scaling curve.
7.  **Run the KSampler:** Route the patched model output into your sampler as usual.

> [!TIP]
> `base_shift`/`max_shift` let you blend the flow-matching schedule as you scale to extremely large canvases. Keeping them at `1.15`/`1.35` mirrors the defaults we found stable in early tests—feel free to tune if you observe over-smoothing or excess repetition.

#### How DyPE for Qwen Image Works

The native Qwen Image transformer was trained on a 1024×1024 latent grid, so every attention layer expects RoPE caches sized for 58×104 spatial tokens (plus text tokens). When you push beyond that window, the model reuses frequencies and starts repeating structures.

The DyPE node swaps the stock `EmbedND` for `QwenSpatialPosEmbed`, a drop-in replacement that:

* Clones the original positional embedder so the node can be removed without side-effects.
* Recomputes the rotary cache using YaRN or NTK scaling for the height/width axes while leaving the text index axis untouched.
* Tracks the sampler’s normalized timestep (via a lightweight wrapper) and applies the DyPE power ramp (`t^λ`) to blend from the base grid to the expanded grid over the course of sampling. Early steps stay close to the training spectrum; late steps receive the extra high-frequency coverage that keeps 4K images coherent.
* Interpolates the FLUX-style flow shift between `base_shift` and `max_shift` according to the requested canvas size so the noise schedule stays in sync with the wider attention field.
* Emits INFO logs (`[DyPE QwenImage] axis=…`) showing the current grid lengths, ramp strength, and YaRN/NTK factors. These diagnostics make it easy to correlate visual artifacts with the positional scaling parameters.

Because the embedder is swapped via `ModelPatcher`, you can chain other ComfyUI optimizations after the DyPE node, and disabling the node returns you to the stock Qwen behaviour instantly.

### Node Inputs

*   **`model`**: The FLUX model to be patched.
*   **`width` / `height`**: The target image resolution. **This must match the resolution set in your `Empty Latent Image` node.**
*   **`method`**: The core position encoding extrapolation method. `yarn` is the recommended default, as it forms the basis of the paper's best-performing "DY-YaRN" variant.
*   **`enable_dype`**: Enables or disables the **dynamic, time-aware** component of DyPE.
    *   **Enabled (True):** Both the noise schedule and RoPE will be dynamically adjusted throughout sampling. This is the full DyPE algorithm.
    *   **Disabled (False):** The node will only apply the dynamic noise schedule shift. The RoPE will use a static extrapolation method (e.g., standard YARN). This can be useful for comparison or if you find it works better at certain moderate resolutions.
*   **`dype_exponent`**: (λt) Controls the "strength" of the dynamic effect over time. This is the most important tuning parameter.
    *   `2.0` (Exponential): Recommended for **4K+** resolutions. It's an aggressive schedule that transitions quickly.
    *   `1.0` (Linear): A good starting point for **~2K-3K** resolutions.
    *   `0.5` (Sub-linear): A gentler schedule that may work best for resolutions just above the model's native 1K.
*   **`base_shift` / `max_shift`** (Advanced): These parameters control the interpolation of the dynamic noise schedule shift (`mu`). The default values (`0.5`, `1.15`) are taken directly from the FLUX architecture and are generally optimal. Adjust only if you are an advanced user experimenting with the noise schedule.
*   **`auto_detect`**: When enabled (default), the node inspects the loaded Qwen checkpoint to recover its training grid and patch size. Disable it if you need to supply `base_width`/`base_height` manually.
*   **`base_width` / `base_height`**: Manual override for the training canvas; only consulted when `auto_detect` is turned off.
*   **`editing_strength` & `editing_mode`**: Let you taper DyPE during edits. Reduce the strength (e.g., 0.5) and pick a mode like `adaptive` to keep structure intact during image-to-image or inpainting workflows.

> [!WARNING]
> It seems the width/height parameters in the node are buggy. Keep the values below 1024x1024; doing so won’t affect your output.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<p align="center">══════════════════════════════════</p>

Beyond the code, I believe in the power of community and continuous learning. I invite you to join the 'TokenDiff AI News' and 'TokenDiff Community Hub'

<table border="0" align="center" cellspacing="10" cellpadding="0">
  <tr>
    <td align="center" valign="top">
      <h4>TokenDiff AI News</h4>
      <a href="https://t.me/TokenDiff">
        <img width="40%" alt="tokendiff-tg-qw" src="https://github.com/user-attachments/assets/e29f6b3c-52e5-4150-8088-12163a2e1e78" />
      </a>
      <p><sub>🗞️ AI for every home, creativity for every mind!</sub></p>
    </td>
    <td align="center" valign="top">
      <h4>TokenDiff Community Hub</h4>
      <a href="https://t.me/TokenDiff_hub">
        <img width="40%" alt="token_hub-tg-qr" src="https://github.com/user-attachments/assets/da544121-5f5b-4e3d-a3ef-02272535929e" />
      </a>
      <p><sub>💬 questions, help, and thoughtful discussion.</sub> </p>
    </td>
  </tr>
</table>

<p align="center">══════════════════════════════════</p>

## ⚠️ Known Issues and Limitations
*   **FLUX Only:** This implementation is highly specific to the architecture of the FLUX model and will not work on standard U-Net models (like SD 1.5/SDXL) or other Diffusion Transformers.
*   **Parameter Tuning:** The optimal `dype_exponent` can vary based on your target resolution. Experimentation is key to finding the best setting for your use case. The default of `2.0` is optimized for 4K.
*   **Qwen CLIP Diagnostics:** When a supplied CLIP encoder is missing the expected `transformer.model`, the extension now recursively searches typical attachment points (including nested module dictionaries) and, if still unresolved, raises an error that includes a structured snapshot in both the logs and exception text to speed up debugging. Both DyPE nodes also emit INFO-level logs summarizing the requested patch parameters whenever they run.
*   **Qwen Spatial Scaling:** Extremely aggressive aspect ratios (>3:1) may still require manual tuning of `max_shift` or method selection to maintain coherence. Start with `yarn` and step the exponent down (e.g. to `1.0`) if the model oversharpens.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License
The original DyPE project is patent pending. For commercial use or licensing inquiries regarding the underlying method, please contact the [original authors](mailto:noam.issachar@mail.huji.ac.il).

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
