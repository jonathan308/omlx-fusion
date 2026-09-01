# SPDX-License-Identifier: Apache-2.0
"""Regression tests for mixed Qwen4 QSA text/MRoPE cache positions."""

import mlx.core as mx
import pytest

from omlx.cache.type_handlers import Qwen4QSAKVCacheHandler
from omlx.cache.prefix_cache import BlockAwarePrefixCache


def _state(positions):
    return {"states": (None, None, None, positions)}


def _text(value, length, batch=1):
    return mx.full((batch, 1, length), value, dtype=mx.int32)


def _mrope(values, length, batch=1):
    return mx.stack(
        [mx.full((batch, length), value, dtype=mx.int32) for value in values],
        axis=1,
    )


def _positions(*segments):
    result = Qwen4QSAKVCacheHandler().concatenate_states(
        [_state(segment) for segment in segments]
    )["index_position_ids"]
    mx.eval(result)
    return result


@pytest.mark.parametrize(
    ("segments", "expected"),
    [
        (
            (_text(7, 2), _mrope((10, 20, 30), 1)),
            [[[7, 7, 10], [7, 7, 20], [7, 7, 30]]],
        ),
        (
            (_mrope((10, 20, 30), 1), _text(7, 2)),
            [[[10, 7, 7], [20, 7, 7], [30, 7, 7]]],
        ),
        (
            (
                _text(1, 1),
                _mrope((2, 3, 4), 1),
                _text(5, 1),
                _mrope((6, 7, 8), 1),
            ),
            [[[1, 2, 5, 6], [1, 3, 5, 7], [1, 4, 5, 8]]],
        ),
    ],
)
def test_mixed_text_positions_promote_to_mrope(segments, expected):
    result = _positions(*segments)

    assert result.shape == (1, 3, len(expected[0][0]))
    assert result.tolist() == expected


def test_all_text_positions_keep_one_channel():
    result = _positions(_text(1, 2), _text(2, 3))

    assert result.shape == (1, 1, 5)
    assert result.tolist() == [[[1, 1, 2, 2, 2]]]


def test_all_mrope_positions_keep_three_channels():
    result = _positions(_mrope((1, 2, 3), 2), _mrope((4, 5, 6), 1))

    assert result.shape == (1, 3, 3)
    assert result.tolist() == [[[1, 1, 4], [2, 2, 5], [3, 3, 6]]]


@pytest.mark.parametrize(
    "segments",
    [
        (mx.zeros((1, 2, 4), dtype=mx.int32),),
        (_text(1, 2), mx.zeros((1, 2, 4), dtype=mx.int32)),
    ],
)
def test_unsupported_channel_count_is_rejected_in_every_state(segments):
    with pytest.raises(ValueError, match="require 1 text channel or 3 MRoPE channels"):
        _positions(*segments)


def test_incompatible_batch_shape_is_rejected():
    with pytest.raises(ValueError, match="consistent batch dimension"):
        _positions(_text(1, 2, batch=1), _mrope((2, 3, 4), 2, batch=2))


@pytest.mark.parametrize(
    "segments",
    [
        (mx.zeros((1, 4), dtype=mx.int32),),
        (_mrope((1, 2, 3), 1), mx.zeros((1, 4), dtype=mx.int32)),
    ],
)
def test_non_serialized_position_shape_is_rejected_in_every_state(segments):
    with pytest.raises(ValueError, match=r"must be \[B, C, S\]"):
        _positions(*segments)


def test_card_572_fourteen_block_topology_reconstructs_exact_prefix():
    block_size = 2048
    segments = [_text(block, block_size) for block in range(14)]
    segments[9] = _mrope((90, 91, 92), block_size)

    result = _positions(*segments)

    assert result.shape == (1, 3, 28_672)
    # Text blocks on either side are duplicated into every MRoPE coordinate.
    expected_text_prefix = [
        block for block in range(9) for _ in range(block_size)
    ]
    for channel in range(3):
        assert result[0, channel, : 9 * block_size].tolist() == expected_text_prefix
        assert result[0, channel, 10 * block_size : 11 * block_size].tolist() == (
            [10] * block_size
        )
    for channel, value in enumerate((90, 91, 92)):
        assert result[0, channel, 9 * block_size : 10 * block_size].tolist() == (
            [value] * block_size
        )


def _qsa_block(positions):
    length = int(positions.shape[-1])
    elements = (
        mx.zeros((1, 2, length, 4)),
        mx.zeros((1, 2, length, 4)),
        mx.zeros((1, length, 8)),
        positions,
    )
    return [("__nstate__", "QSAKVCache", elements)]


def test_equal_plane_qsa_block_requires_unit_position_timeline():
    positions = mx.broadcast_to(mx.arange(16)[None, None], (1, 3, 16))

    valid, reason = (
        BlockAwarePrefixCache._validate_qwen4_qsa_block_position_timeline(
            _qsa_block(positions),
            ["QSAKVCache"],
        )
    )

    assert valid
    assert reason == "ok"


def test_equal_plane_qsa_block_rejects_speculative_position_reset():
    # Reproduces the physical failure found in an old 4K SSD block:
    # 16,384..20,281, then 0..8, then 20,291...
    positions = mx.concatenate(
        [
            mx.arange(16_384, 20_282),
            mx.arange(9),
            mx.arange(20_291, 20_480),
        ]
    )[None, None]
    positions = mx.broadcast_to(positions, (1, 3, 4096))

    valid, reason = (
        BlockAwarePrefixCache._validate_qwen4_qsa_block_position_timeline(
            _qsa_block(positions),
            ["QSAKVCache"],
        )
    )

    assert not valid
    assert reason == "equal-plane text positions reset within the block"


def test_genuine_mrope_qsa_block_is_not_forced_to_text_timeline():
    positions = mx.stack(
        [
            mx.array([[0, 1, 2, 3]]),
            mx.array([[0, 0, 1, 1]]),
            mx.array([[0, 1, 0, 1]]),
        ],
        axis=1,
    )

    valid, reason = (
        BlockAwarePrefixCache._validate_qwen4_qsa_block_position_timeline(
            _qsa_block(positions),
            ["QSAKVCache"],
        )
    )

    assert valid
    assert reason == "ok"


def test_qsa_block_position_validation_checks_all_qsa_layers():
    good = mx.arange(8)[None, None]
    bad = mx.array([0, 1, 2, 0, 1, 2, 3, 4])[None, None]
    cache_data = _qsa_block(good) + _qsa_block(bad)

    valid, reason = (
        BlockAwarePrefixCache._validate_qwen4_qsa_block_position_timeline(
            cache_data,
            ["QSAKVCache", "QSAQuantizedKVCache"],
        )
    )

    assert not valid
    assert reason == "equal-plane text positions reset within the block"
