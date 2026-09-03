# SPDX-License-Identifier: Apache-2.0
"""No-workarounds gate: lifecycle safety is native abort/release.

Watchers, force-unload timeouts, emergency process killers, and vision-token
caps used as OOM substitutes are not a fix. Cancellation must release the
same owned resources as rejection, and processor defaults must stay native.
"""

from __future__ import annotations

from types import SimpleNamespace
import inspect

from omlx.engine import vlm as vlm_engine
from omlx.engine_pool import EnginePool
from omlx.process_memory_enforcer import ProcessMemoryEnforcer
from omlx.scheduler import Scheduler


def test_abort_uses_rejection_release_not_retain_for_reuse():
    source = inspect.getsource(Scheduler._do_abort_request)
    assert "self._release_paged_cache_for_request(request_id)" in source
    assert "clear_request_entry(" not in source
    assert "release_for_eviction(" not in source


def test_engine_pool_has_no_force_unload_timeout():
    assert not hasattr(EnginePool, "_PENDING_UNLOAD_FORCE_S")
    source = inspect.getsource(EnginePool._wait_for_pending_unload)
    assert "force" not in source.lower()
    assert "_unload_engine" not in source


def test_guard_off_does_not_install_emergency_unloader():
    source = inspect.getsource(ProcessMemoryEnforcer._check_and_enforce)
    assert "if ceiling <= 0:" in source
    enforce_after_zero = source.split("if ceiling <= 0:", 1)[1]
    early_return = enforce_after_zero.split("current = self._current_usage_bytes()", 1)[0]
    assert "return" in early_return
    assert "_unload_engine" not in early_return


def test_fix_processor_does_not_rewrite_checkpoint_pixel_budget():
    ip = SimpleNamespace(
        patch_size=16,
        merge_size=2,
        min_pixels=65536,
        max_pixels=16777216,
        size={"longest_edge": 16777216, "shortest_edge": 65536},
    )
    vlm_engine._fix_processor_none_pixels(SimpleNamespace(image_processor=ip))
    assert ip.max_pixels == 16777216
    assert ip.size["longest_edge"] == 16777216
    assert ip.min_pixels == 65536
