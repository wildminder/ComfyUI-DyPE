"""Effective model_sampling resolution through a ModelPatcher (v2.16.0).

The node layer must never read ``model.model.model_sampling`` (the LIVE
attribute on the shared BaseModel) to derive sigma schedules, timestep
conversions or prediction types. ComfyUI's object-patch lifecycle makes that
attribute history-dependent:

- object patches are applied to the shared BaseModel at load and stay applied
  between runs (model_patcher.py patch_model / partially_load);
- ``load_models_gpu`` detaches stale same-base patchers with
  ``detach(unpatch_all=False)`` (model_management.py:962) — their patches LEAK;
- restore-to-original only happens through ``object_patches_backup``, which
  every ``unpatch_model`` CLEARS (model_patcher.py:1165-1169);
- ``partially_unload`` moves weights only — patches and backup untouched.

Whichever patcher (a DyPE/SEGA clone, a stock ModelSamplingFlux node clone,
...) was loaded last therefore decides the schedule a live-attr reader sees.
ComfyUI's own sampler resolves through the patcher instead —
``comfy/samplers.py:1425``: ``calculate_sigmas(self.model.get_model_object(
"model_sampling"), ...)``. :func:`effective_model_sampling` mirrors exactly
that semantics (object_patches -> object_patches_backup -> live attr), making
the schedule a deterministic function of the graph's own patch chain.

:func:`is_stale_dype_leak` detects the residual case this cannot repair: the
patcher carries no ``model_sampling`` patch/backup entry while the live attr is
one of this pack's function-local leak classes — i.e. a stale patch inherited
from a PREVIOUS run's patch node that is no longer in the graph.
"""

from __future__ import annotations

import logging

# Function-local classes installed by apply_dype_to_model / apply_sega_to_model
# (src/patch_utils.py). Matched by __name__: the classes are defined inside the
# installer functions, so identity comparison across imports is impossible.
STALE_LEAK_CLASS_NAMES = (
    "DypeModelSamplingFlux",
    "SegaModelSamplingFlux",
    "DefaultModelSamplingFlux",
)


def effective_model_sampling(model):
    """Resolve the model_sampling object the way ComfyUI's sampler does.

    ``model`` is a ModelPatcher (or a test mock). Resolution order mirrors
    ``ModelPatcher.get_model_object`` (model_patcher.py:758-768): the patcher's
    own object patch, then its object_patches_backup (the original captured at
    patch time), then the live BaseModel attribute. Plain objects without the
    method fall back to the live attribute (mock/test safety).
    """
    get_model_object = getattr(model, "get_model_object", None)
    if callable(get_model_object):
        return get_model_object("model_sampling")
    return getattr(getattr(model, "model", None), "model_sampling", None)


def is_stale_dype_leak(model) -> bool:
    """True when the live sampling is a stale patch from a previous run.

    Stale means: the patcher carries NO ``model_sampling`` object patch and NO
    backup entry (so the resolution falls through to the live attribute), and
    the live attribute's class is one of this pack's installer-local sampling
    patches. Only then is the schedule inherited from a run that is no longer
    in the graph — the user-fixable-by-cache-clear drift this pack warns about.
    """
    if not callable(getattr(model, "get_model_object", None)):
        return False
    if "model_sampling" in (getattr(model, "object_patches", None) or {}):
        return False
    if "model_sampling" in (getattr(model, "object_patches_backup", None) or {}):
        return False
    live = getattr(getattr(model, "model", None), "model_sampling", None)
    return type(live).__name__ in STALE_LEAK_CLASS_NAMES


def warn_if_stale_leak(model, node_name: str) -> None:
    """User-facing signal for the un-healable residual (plan S6).

    Fired by the direct-sampling nodes when the resolved schedule is a stale
    patch inherited from a run that is no longer in the graph — the exact
    "identical params, different results until caches are cleared" report.
    """
    if is_stale_dype_leak(model):
        live = getattr(getattr(model, "model", None), "model_sampling", None)
        logging.getLogger("ComfyUI-DyPE").warning(
            "%s: model_sampling is a stale patch from a previous run (%s) — "
            "schedules may not match this workflow. Reload the models (clear "
            "cache) or add the patch node to this graph.",
            node_name, type(live).__name__,
        )
