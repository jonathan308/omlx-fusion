# SPDX-License-Identifier: Apache-2.0
"""Host-only gates for the exact Qwen4 MTP M=6 compact-stage route.

These tests never evaluate a real MLX graph.  Native calls and ``mx.eval`` are
replaced with fakes so route, capability, validation, fallback, and ordering
can be qualified without dispatching Metal work.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np

from omlx.custom_kernels.glm_moe_dsa import fast
from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


compat.apply_mlx_vlm_qwen4_exp_compat_patch()
from mlx_vlm.models.qwen4_exp import qsa_fast  # noqa: E402


@dataclass
class _FakeArray:
    shape: tuple[int, ...]
    dtype: object
    value: int | None = None
    evaluated: bool = False

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def astype(self, dtype):
        return _FakeArray(self.shape, dtype, self.value, self.evaluated)

    def __getitem__(self, index):
        # Production uses only ``selected_blocks[:, None]`` in this seam.
        if self.shape == (1, 6, 512):
            return _FakeArray((1, 1, 6, 512), self.dtype)
        raise AssertionError(f"unexpected fake index {index!r} for {self.shape}")

    def transpose(self, *axes):
        return _FakeArray(tuple(self.shape[axis] for axis in axes), self.dtype)

    def item(self) -> int:
        if not self.evaluated:
            raise AssertionError("validation status was read before staged eval")
        assert self.value is not None
        return self.value


def _geometry(key_tokens: int, query_tokens: int = 6):
    queries = _FakeArray((1, 24, query_tokens, 256), mx.bfloat16)
    keys = _FakeArray((1, 2, key_tokens, 256), mx.bfloat16)
    values = _FakeArray(keys.shape, keys.dtype)
    selected = _FakeArray((1, query_tokens, 512), mx.uint32)
    return queries, keys, values, selected


def test_compact_stage_activates_only_at_the_40960_token_boundary():
    for key_tokens, expected in ((40_959, False), (40_960, True)):
        queries, keys, values, selected = _geometry(key_tokens)
        assert (
            qsa_fast._compact_stage_mtp_m6_geometry(
                queries,
                keys,
                values,
                selected,
                q_offset=key_tokens - 6,
                mtp_m6_target_verify=True,
            )
            is expected
        )


def test_compact_stage_geometry_preserves_mtp_off_m1_and_non_m6():
    queries, keys, values, selected = _geometry(40_960)
    assert not qsa_fast._compact_stage_mtp_m6_geometry(
        queries,
        keys,
        values,
        selected,
        q_offset=40_954,
        mtp_m6_target_verify=False,
    )

    for width in (1, 2, 5, 7, 9):
        queries, keys, values, selected = _geometry(40_960, width)
        assert not qsa_fast._compact_stage_mtp_m6_geometry(
            queries,
            keys,
            values,
            selected,
            q_offset=40_960 - width,
            mtp_m6_target_verify=True,
        )

    queries, keys, values, selected = _geometry(40_960)
    assert not qsa_fast._compact_stage_mtp_m6_geometry(
        queries,
        keys,
        values,
        selected,
        q_offset=40_953,
        mtp_m6_target_verify=True,
    )

    queries, keys, values, selected = _geometry(0x80000000)
    assert not qsa_fast._compact_stage_mtp_m6_geometry(
        queries,
        keys,
        values,
        selected,
        q_offset=0x80000000 - 6,
        mtp_m6_target_verify=True,
    )

    queries, keys, values, _selected = _geometry(40_960)
    float_selector = _FakeArray((1, 6, 512), mx.float32)
    assert not qsa_fast._compact_stage_mtp_m6_geometry(
        queries,
        keys,
        values,
        float_selector,
        q_offset=40_954,
        mtp_m6_target_verify=True,
    )


def test_missing_compact_capability_fails_closed_before_native_calls(monkeypatch):
    queries, keys, values, selected = _geometry(40_960)
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_COMPACT_STAGE_DISABLED", False)
    monkeypatch.setattr(fast, "is_native_available", lambda: True)
    monkeypatch.setattr(
        fast,
        "has_symbol",
        lambda name: name != "qwen4_qsa_compact_stage_gather",
    )
    monkeypatch.setattr(
        fast,
        "qwen4_qsa_compact_stage_plan",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("mapper must not run without every native symbol")
        ),
    )

    assert (
        qsa_fast._native_compact_stage_mtp_m6_attention(
            queries,
            keys,
            values,
            selected,
            q_offset=40_954,
            mtp_m6_target_verify=True,
        )
        is None
    )
    assert qsa_fast._NATIVE_QSA_COMPACT_STAGE_DISABLED


def test_compact_stage_symbols_are_part_of_the_extension_abi():
    for symbol in qsa_fast._QSA_COMPACT_STAGE_SYMBOLS:
        assert symbol in fast.NATIVE_SYMBOLS


def _install_native_stage_fakes(monkeypatch, *, validation_status: int):
    calls: list[str] = []
    status = _FakeArray((1,), mx.uint32, value=validation_status)
    staged_output = _FakeArray((1, 24, 6, 256), mx.bfloat16)

    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_COMPACT_STAGE_DISABLED", False)
    monkeypatch.setattr(fast, "is_native_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(mx, "contiguous", lambda value: value)

    def plan(selected, q_offset, key_tokens):
        calls.append("plan")
        assert selected.shape == (1, 1, 6, 512)
        assert q_offset == key_tokens - 6
        return (
            _FakeArray((3072,), mx.uint32),
            _FakeArray((1,), mx.uint32),
            _FakeArray((1, 1, 6, 2051), mx.uint32),
            _FakeArray((12306,), mx.uint32),
            status,
        )

    def gather(keys, values, sources):
        calls.append("gather")
        assert keys.shape == values.shape == (1, 2, 40_960, 256)
        assert sources.shape == (12306,)
        return (
            _FakeArray((1, 2, 12306, 256), mx.bfloat16),
            _FakeArray((1, 2, 12306, 256), mx.bfloat16),
        )

    def attention(queries, keys, values, slots, scale, **kwargs):
        calls.append("attention")
        assert queries.shape == (1, 24, 6, 256)
        assert keys.shape == values.shape == (1, 2, 12306, 256)
        assert slots.shape == (1, 1, 6, 2051)
        assert scale == 256**-0.5
        assert kwargs == {"key_tile": 64, "dimension_tile": 64}
        return staged_output

    def evaluate(*arrays):
        calls.append("eval")
        assert arrays == (staged_output, status)
        for array in arrays:
            array.evaluated = True

    monkeypatch.setattr(fast, "qwen4_qsa_compact_stage_plan", plan)
    monkeypatch.setattr(fast, "qwen4_qsa_compact_stage_gather", gather)
    monkeypatch.setattr(fast, "qwen4_qsa_sparse_gqa_attention_tokens", attention)
    monkeypatch.setattr(mx, "eval", evaluate)
    return calls, staged_output, status


def test_zero_status_returns_realized_stage_and_uses_one_local_bank(monkeypatch):
    calls, staged_output, status = _install_native_stage_fakes(
        monkeypatch, validation_status=0
    )
    queries, keys, values, selected = _geometry(40_960)
    actual = qsa_fast._native_compact_stage_mtp_m6_attention(
        queries,
        keys,
        values,
        selected,
        q_offset=40_954,
        mtp_m6_target_verify=True,
    )

    assert calls == ["plan", "gather", "attention", "eval"]
    assert staged_output.evaluated and status.evaluated
    assert actual is not None and actual.shape == (1, 6, 24, 256)


def test_nonzero_status_discards_staged_output_and_returns_direct(monkeypatch):
    calls, staged_output, status = _install_native_stage_fakes(
        monkeypatch, validation_status=4
    )
    queries, keys, values, selected = _geometry(40_960)
    direct = object()
    monkeypatch.setattr(
        qsa_fast, "_native_sparse_gqa_attention", lambda *a, **k: direct
    )

    actual = qsa_fast._native_routed_sparse_gqa_attention(
        queries,
        keys,
        values,
        selected,
        q_offset=40_954,
        mtp_m6_target_verify=True,
    )

    assert calls == ["plan", "gather", "attention", "eval"]
    assert staged_output.evaluated and status.evaluated
    assert actual is direct
    assert qsa_fast._NATIVE_QSA_COMPACT_STAGE_DISABLED


def test_mtp_off_skips_compact_probe_and_uses_direct(monkeypatch):
    queries, keys, values, selected = _geometry(40_960)
    direct = object()
    monkeypatch.setattr(
        qsa_fast,
        "_native_compact_stage_mtp_m6_attention",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("MTP-off route must not probe compact staging")
        ),
    )
    monkeypatch.setattr(
        qsa_fast, "_native_sparse_gqa_attention", lambda *a, **k: direct
    )

    assert (
        qsa_fast._native_routed_sparse_gqa_attention(
            queries,
            keys,
            values,
            selected,
            q_offset=40_954,
            mtp_m6_target_verify=False,
        )
        is direct
    )


def test_native_stage_exception_latches_and_returns_direct(monkeypatch):
    queries, keys, values, selected = _geometry(40_960)
    direct = object()
    monkeypatch.setattr(qsa_fast, "_NATIVE_QSA_COMPACT_STAGE_DISABLED", False)
    monkeypatch.setattr(fast, "is_native_available", lambda: True)
    monkeypatch.setattr(fast, "has_symbol", lambda name: True)
    monkeypatch.setattr(mx, "contiguous", lambda value: value)
    monkeypatch.setattr(
        fast,
        "qwen4_qsa_compact_stage_plan",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stale native ABI")),
    )
    monkeypatch.setattr(
        qsa_fast, "_native_sparse_gqa_attention", lambda *a, **k: direct
    )

    actual = qsa_fast._native_routed_sparse_gqa_attention(
        queries,
        keys,
        values,
        selected,
        q_offset=40_954,
        mtp_m6_target_verify=True,
    )
    assert actual is direct
    assert qsa_fast._NATIVE_QSA_COMPACT_STAGE_DISABLED


def _host_fixed_stage(selected: np.ndarray, q_offset: int):
    """Independent host oracle for the fixed Metal mapper's output ordering."""

    invalid = np.iinfo(np.uint32).max
    union = np.unique(selected)
    union_slots = {int(block): slot for slot, block in enumerate(union)}
    sources = np.full(12_306, invalid, dtype=np.uint32)
    row_slots = np.full((6, 2051), invalid, dtype=np.uint32)
    for block, union_slot in union_slots.items():
        sources[union_slot * 4 : union_slot * 4 + 4] = np.arange(
            block * 4, block * 4 + 4, dtype=np.uint32
        )
    for row in range(6):
        for column, block in enumerate(selected[row]):
            stage_start = union_slots[int(block)] * 4
            row_slots[row, column * 4 : column * 4 + 4] = np.arange(
                stage_start, stage_start + 4, dtype=np.uint32
            )
        query = q_offset + row
        complete = (query + 1) // 4
        tail_count = query + 1 - complete * 4
        tail_base = 12_288 + row * 3
        sources[tail_base : tail_base + tail_count] = np.arange(
            complete * 4, query + 1, dtype=np.uint32
        )
        row_slots[row, 2048 : 2048 + tail_count] = np.arange(
            tail_base, tail_base + tail_count, dtype=np.uint32
        )
    return sources, row_slots


def test_host_ordering_preserves_every_block_token_then_each_causal_tail():
    rng = np.random.default_rng(20260831)
    selected = np.stack(
        [
            np.sort(rng.choice(10_000, size=512, replace=False)).astype(np.uint32)
            for _ in range(6)
        ]
    )
    q_offset = 40_954
    sources, row_slots = _host_fixed_stage(selected, q_offset)
    invalid = np.iinfo(np.uint32).max

    for row in range(6):
        slots = row_slots[row]
        reconstructed = sources[slots[slots != invalid]]
        block_tokens = (
            selected[row, :, None] * np.uint32(4)
            + np.arange(4, dtype=np.uint32)
        ).reshape(-1)
        query = q_offset + row
        complete = (query + 1) // 4
        tail = np.arange(complete * 4, query + 1, dtype=np.uint32)
        assert np.array_equal(reconstructed, np.concatenate((block_tokens, tail)))


def test_native_source_keeps_overflow_status_and_capacity_strides():
    root = Path(__file__).parents[1]
    csrc = root / "omlx/custom_kernels/glm_moe_dsa/csrc"
    cpp = (csrc / "qwen4_qsa_compact_stage.cpp").read_text()
    metal = (csrc / "qwen4_qsa_compact_stage.metal").read_text()

    assert "source_last > ulong(kInvalid)" in metal
    assert "ulong(raw_block) * ulong(4)" in metal
    assert "device uint *validation_status [[buffer(5)]]" in metal
    assert "lane != 0 || checked_status != 0" in metal
    assert "raw_block <= selected[index - 1]" in metal
    assert "source_last >= ulong(params.key_tokens)" in metal
    assert "source_tokens[count * 4 + token] = uint(source)" in metal
    assert "count * 4 + token" in metal
    assert "kBlockTokenCapacity + row * kTailPerRow + tail" in metal
    assert "kBlockTokensPerRow + tail" in metal
    assert "keys.flags().row_contiguous" not in cpp
    assert "static_cast<uint64_t>(keys.strides(1))" in cpp
    assert "static_cast<uint64_t>(keys.strides(2))" in cpp
    assert "static_cast<uint64_t>(values.strides(1))" in cpp
    assert "static_cast<uint64_t>(values.strides(2))" in cpp
    sparse_cpp = (csrc / "qwen4_qsa_sparse_gqa.cpp").read_text()
    assert "k.shape(2) != kQwen4StageTokenCapacity" in sparse_cpp


def test_language_threads_target_verify_not_six_token_prefill_shape():
    root = Path(__file__).parents[1]
    language = (
        root
        / "omlx/patches/mlx_vlm_qwen4_exp_compat/vendor/mlx_vlm/models/"
        / "qwen4_exp/language.py"
    ).read_text()
    assert "mtp_m6_target_verify=bool(target_verify and length == 6)" in language
