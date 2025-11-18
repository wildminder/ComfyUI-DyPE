# ComfyUI-DyPE Learnings

- Added structured diagnostics for Qwen CLIP encoders that do not expose a `transformer.model`. The exception now surfaces the object's key attributes and emits the same snapshot via logging, which helps track down alternative attachment points when integrating unfamiliar CLIP implementations.
