# SPDX-License-Identifier: Apache-2.0
"""The paged cache block size follows MiMo's wider prefill floor.

With the prefix cache on, every prefill chunk is clamped to the next block
boundary; a block smaller than the prefill floor would silently split the
wider chunks back down.
"""

from types import SimpleNamespace

import pytest

from omlx.scheduler import Scheduler


def _scheduler(*, floor: int, window: int = 128, mimo: bool = True) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s.config = SimpleNamespace(paged_ssd_cache_dir="/tmp/ssd", paged_cache_block_size=256)
    s._qwen35_prefill_floor = floor
    s._detect_rotating_window_sizes = lambda: {window}
    s._detect_pooling_cache = lambda: False
    s._is_mimo_hybrid = lambda: mimo
    return s


def test_block_size_reaches_the_wide_prefill_floor():
    s = _scheduler(floor=4096)
    s._align_block_size_with_rotating_window()
    assert s.config.paged_cache_block_size == 4096


def test_block_size_keeps_the_pooling_default_without_a_floor():
    s = _scheduler(floor=0)
    s._align_block_size_with_rotating_window()
    assert s.config.paged_cache_block_size == Scheduler._POOLING_ROTATING_BLOCK_SIZE


def test_block_size_ignores_a_floor_off_the_window_grid():
    s = _scheduler(floor=4000)
    s._align_block_size_with_rotating_window()
    assert s.config.paged_cache_block_size == Scheduler._POOLING_ROTATING_BLOCK_SIZE


def test_non_mimo_rotating_models_are_unchanged():
    s = _scheduler(floor=4096, mimo=False)
    s._align_block_size_with_rotating_window()
    # window 128 -> smallest multiple in [512, 1024]
    assert s.config.paged_cache_block_size == 512


def test_wide_mimo_chunk_requires_fused_full_attention(monkeypatch):
    """Without the fused 192/128 attention the wider chunk is not used."""
    import sys
    from types import SimpleNamespace as NS

    import omlx.utils
    from omlx import scheduler

    def use(module):
        # ``from .utils import fast_attention`` reads the package attribute
        # first, then sys.modules; None in sys.modules makes it ImportError.
        monkeypatch.setitem(sys.modules, "omlx.utils.fast_attention", module)
        if module is None:
            monkeypatch.delattr(omlx.utils, "fast_attention", raising=False)
        else:
            monkeypatch.setattr(omlx.utils, "fast_attention", module, raising=False)

    use(None)
    assert scheduler._mimo_fused_full_attention() is False

    use(NS(_native_mixed_dims_supported=lambda qk, v: False, _nax_available=lambda: True))
    assert scheduler._mimo_fused_full_attention() is True

    use(NS(_native_mixed_dims_supported=lambda qk, v: True, _nax_available=lambda: False))
    assert scheduler._mimo_fused_full_attention() is True

    use(NS(_native_mixed_dims_supported=lambda qk, v: False, _nax_available=lambda: False))
    assert scheduler._mimo_fused_full_attention() is False


def test_block_size_follows_the_nax_floor():
    s = _scheduler(floor=8192)
    s._align_block_size_with_rotating_window()
    assert s.config.paged_cache_block_size == 8192


def _floor_probe(
    monkeypatch, *, memory_gb: int, nax: bool, fused=True, gather=True, model_type="mimo_v2"
):
    import omlx.custom_kernels.nax as nax_mod
    import omlx.scheduler as scheduler_mod
    import omlx.settings as settings

    monkeypatch.setattr(settings, "get_system_memory", lambda: memory_gb * 1024**3)
    monkeypatch.setattr(nax_mod, "is_nax_available", lambda: nax)
    monkeypatch.setattr(scheduler_mod, "_mimo_fused_full_attention", lambda: fused)
    monkeypatch.setattr(scheduler_mod, "_oversized_sorted_gather_ok", lambda: gather)
    s = Scheduler.__new__(Scheduler)
    s.model = SimpleNamespace(model_type=model_type)
    return s._detect_qwen35_prefill_floor()


@pytest.mark.parametrize(
    "memory_gb,nax,fused,gather,expected",
    [
        (256, True, True, True, 8192),  # NAX hosts: ~256 rows per expert per chunk
        (128, True, True, True, 8192),
        (256, True, True, False, 4096),  # >32768-row gathers would be split
        (256, False, True, True, 4096),  # other large hosts keep the measured 4096
        (192, False, True, True, 4096),
        (256, True, False, True, 0),  # unfused 192/128 attention: default step
        (96, True, True, True, 0),  # small hosts keep the configured step
        (64, False, True, True, 0),
    ],
)
def test_mimo_prefill_floor(monkeypatch, memory_gb, nax, fused, gather, expected):
    got = _floor_probe(monkeypatch, memory_gb=memory_gb, nax=nax, fused=fused, gather=gather)
    assert got == expected


def test_non_mimo_models_get_no_mimo_floor(monkeypatch):
    assert _floor_probe(monkeypatch, memory_gb=256, nax=True, model_type="llama") == 0
