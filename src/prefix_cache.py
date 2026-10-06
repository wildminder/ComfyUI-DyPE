"""Qwen-Image-2.1 prefix K/V cache control for the cascade pipelines.

Qwen-Image-2.1 caches the step-independent text/reference K/V per sampling run
and keys the cache on the target shape (``prefix_cache_key`` includes the
latent H/W).  A cascade changes latent shape between stages, so each stage
allocates another slot in the shared ``PoseBranchCache``.

Once two or more slots exist, upstream evicts the touched one with
``self.slots.remove(s)`` on a list of dicts whose values are tensors
(``comfy/ldm/wan/model_animate2.py``).  ``list.remove`` compares non-identical
elements with ``==``, which for those dicts reduces to a multi-element tensor
comparison, so the eviction raises::

    RuntimeError: Boolean value of Tensor with more than one value is ambiguous

A single slot escapes it through CPython's identity fast path in
``list.remove``, which is why an ordinary one-prompt sampler run never trips
this and a cascade does.

The cache is also worth nothing to a cascade: the key carries the target shape,
so the upscaled stage invalidates everything the first stage cached.  Disable it
via the same switch ComfyUI's own ``QwenImage21Cache`` node exposes for
``device="off"``, which makes ``select_prefix_cache`` return before touching the
cache at all.

This is a no-op for every model except Qwen-Image-2.1, the only reader of the
key.
"""

__all__ = ["disable_prefix_kv_cache"]


def disable_prefix_kv_cache(model):
    """Return a clone of ``model`` with 2.1's prefix K/V cache switched off.

    Clones first (``ModelPatcher.clone`` deep-copies ``model_options``) so the
    caller's patcher and any sibling branch are left untouched.
    """
    m = model.clone()
    transformer_options = m.model_options.get("transformer_options")
    if transformer_options is None:
        transformer_options = {}
        m.model_options["transformer_options"] = transformer_options
    transformer_options["qwen_image21_cache"] = {"device": "off"}
    return m