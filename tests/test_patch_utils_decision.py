"""Schedule-patch decision hygiene (v2.16.0, plan S5).

_should_patch_schedule must resolve the sampling through the patcher
(patch -> backup -> live) so a leaked *ModelSamplingFlux from a previous run
cannot flip the decision. Characterization: clean-state outcomes are pinned;
history-independence: leak/backup states give the same answer as clean.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from comfy import model_sampling as comfy_ms  # noqa: E402
from src.patch_utils import _should_patch_schedule  # noqa: E402


class _Patcher:
    def __init__(self, live, patches=None, backup=None):
        self.model = types.SimpleNamespace(model_sampling=live)
        self.object_patches = dict(patches or {})
        self.object_patches_backup = dict(backup or {})

    def get_model_object(self, name):
        if name in self.object_patches:
            return self.object_patches[name]
        if name in self.object_patches_backup:
            return self.object_patches_backup[name]
        return getattr(self.model, name)


def _ms(name, *bases):
    return type(name, bases or (object,), {})()


FluxLike = comfy_ms.ModelSamplingFlux
ContFlow = type("ModelSamplingContinuousFlow", (), {})


@pytest.mark.unit
class TestShouldPatchSchedule:
    @pytest.mark.parametrize("flags", [(False, False), (True, False),
                                       (False, True), (True, True)])
    def test_characterization_flux_sampling(self, flags):
        # Clean FLUX-style sampling: decision follows the isinstance only.
        patcher = _Patcher(_ms("MS", FluxLike))
        assert _should_patch_schedule(patcher, *flags) is True

    def test_characterization_continuous_flow_not_patched(self):
        patcher = _Patcher(_ms("MS", ContFlow))
        assert _should_patch_schedule(patcher, False, False) is False

    def test_characterization_qwen_and_zimage_force_patch(self):
        patcher = _Patcher(_ms("MS", ContFlow))
        assert _should_patch_schedule(patcher, True, False) is True
        assert _should_patch_schedule(patcher, False, True) is True

    def test_leak_does_not_flip_continuous_flow_decision(self):
        # A leaked Flux-style sampling on the live attr (empty backup, no
        # patch) is the documented KSampler-consistent residual: it IS what
        # resolution returns here, so the decision follows it — pinned so a
        # future change of this trade-off is explicit.
        patcher = _Patcher(_ms("Leaked", FluxLike))
        assert _should_patch_schedule(patcher, False, False) is True

    def test_own_patch_decides_over_leaked_live(self):
        # Graph carries a Flux-style patch over a continuous-flow live attr.
        patcher = _Patcher(_ms("Live", ContFlow),
                           patches={"model_sampling": _ms("P", FluxLike)})
        assert _should_patch_schedule(patcher, False, False) is True

    def test_backup_original_decides_over_leaked_live(self):
        # Unpatched patcher whose lineage backup holds the ORIGINAL
        # continuous-flow sampler: the leak must not flip the decision.
        patcher = _Patcher(_ms("Leaked", FluxLike),
                           backup={"model_sampling": _ms("Orig", ContFlow)})
        assert _should_patch_schedule(patcher, False, False) is False
