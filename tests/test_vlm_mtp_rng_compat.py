# SPDX-License-Identifier: Apache-2.0
"""Compat shim for mlx-vlm's speculative RNG restore across mlx RNG APIs.

mlx commit ce3073389 (PR #3828) replaced the plain-list ``mx.random.state``
with a process-global ``_RandomState`` sentinel (no ``__setitem__``), which
breaks mlx-vlm's ``_restore_rng_state`` (``mx.random.state[i] = value``).
``omlx.speculative.vlm_mtp`` patches that function at import time; these
tests pin the shim against both API shapes (via monkeypatching) and verify
a bit-exact capture/restore round-trip on the installed mlx.
"""

from __future__ import annotations

import mlx.core as mx
import mlx_vlm.speculative.common as vlm_spec_common

from omlx.speculative import vlm_mtp


class _FakeSentinel:
    """Mimic the new ``mx.random.state`` API: read-only, no ``__setitem__``."""

    def __init__(self, key: mx.array) -> None:
        self._key = key

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> mx.array:
        if index not in (0, -1):
            raise IndexError(index)
        return self._key

    def __iter__(self):
        return iter([self._key])


def _restore(state):
    vlm_spec_common._restore_rng_state(state)


class TestShimApplication:
    def test_import_applies_patch(self):
        assert (
            vlm_spec_common._restore_rng_state.__name__
            == "_restore_rng_state_compat"
        )

    def test_patch_is_idempotent(self):
        vlm_mtp._patch_mlx_vlm_rng_state_restore()
        vlm_mtp._patch_mlx_vlm_rng_state_restore()
        assert (
            vlm_spec_common._restore_rng_state.__name__
            == "_restore_rng_state_compat"
        )

    def test_empty_state_is_noop(self, monkeypatch):
        monkeypatch.setattr(
            mx.random, "state", _FakeSentinel(mx.array([1, 2], dtype=mx.uint32))
        )
        _restore([])  # must not touch the (unassignable) state at all


class TestRestoreBothApiShapes:
    def test_old_list_api_uses_item_assignment(self, monkeypatch):
        original = mx.array([1, 2], dtype=mx.uint32)
        fake_state = [original]
        monkeypatch.setattr(mx.random, "state", fake_state)

        new_key = mx.array([3, 4], dtype=mx.uint32)
        _restore([new_key])

        assert fake_state[0] is new_key

    def test_new_sentinel_api_reseeds_from_key_words(self, monkeypatch):
        sentinel = _FakeSentinel(mx.array([0, 0], dtype=mx.uint32))
        monkeypatch.setattr(mx.random, "state", sentinel)
        recorded = []
        monkeypatch.setattr(mx.random, "seed", recorded.append)

        hi, lo = 0xDEADBEEF, 0x12345678
        _restore([mx.array([hi, lo], dtype=mx.uint32)])

        # A setitem attempt would raise TypeError on the sentinel; reaching
        # this assert proves the reseed branch was taken.
        assert recorded == [(hi << 32) | lo]


class TestRealMlxRoundtrip:
    """Capture/restore against the installed mlx, whichever API it exposes."""

    def test_capture_restore_is_bit_exact(self):
        mx.random.seed(1234)
        saved = vlm_spec_common._copy_rng_state()
        expected = mx.random.uniform(shape=(8,))

        mx.random.seed(99)  # disturb the stream
        mx.random.uniform(shape=(3,))

        _restore(saved)
        actual = mx.random.uniform(shape=(8,))
        assert bool(mx.array_equal(expected, actual))

    def test_draft_tokens_preserves_target_stream(self):
        def draft_fn(n):
            return mx.random.uniform(shape=(n,)).tolist()

        mx.random.seed(7)
        rng = vlm_spec_common._SpeculativeSamplerRNG(object(), enabled=True)
        t0 = mx.random.uniform(shape=(4,)).tolist()
        rng.target_sampled()
        d1 = rng.draft_tokens(draft_fn, 4)
        d2 = rng.draft_tokens(draft_fn, 4)
        t1 = mx.random.uniform(shape=(4,)).tolist()

        # Reference target stream with no draft interference.
        mx.random.seed(7)
        assert t0 == mx.random.uniform(shape=(4,)).tolist()
        assert t1 == mx.random.uniform(shape=(4,)).tolist()

        # Reference draft stream: the draft state was captured right after
        # seed(7), so consecutive draft calls continue that stream.
        mx.random.seed(7)
        assert d1 == mx.random.uniform(shape=(4,)).tolist()
        assert d2 == mx.random.uniform(shape=(4,)).tolist()
