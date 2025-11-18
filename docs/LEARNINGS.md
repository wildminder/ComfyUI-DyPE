# ComfyUI-DyPE Learnings

- Added structured diagnostics for Qwen CLIP encoders that do not expose a `transformer.model`. The exception now surfaces the object's key attributes, walks nested candidates (e.g., `clip.cond_stage_model.clip`), and emits the same snapshot via logging to help track down alternative attachment points when integrating unfamiliar CLIP implementations.
