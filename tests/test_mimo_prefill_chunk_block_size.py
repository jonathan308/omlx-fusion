# SPDX-License-Identifier: Apache-2.0
"""The paged cache block size follows MiMo's wider prefill floor.

With the prefix cache on, every prefill chunk is clamped to the next block
boundary; a block smaller than the prefill floor would silently split the
wider chunks back down.
"""

from types import SimpleNamespace

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
