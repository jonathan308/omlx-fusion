# SPDX-License-Identifier: MIT
"""Exact host-side ``ngram-mod`` lookup drafter.

This is a bounded Python port of llama.cpp's ``common_ngram_mod`` algorithm
(MIT): a fixed-size modulo hash table maps an exact N-token suffix to the next
token observed in the request history.  It proposes only long spans; the target
model must still verify every token, so this module never changes output by
itself.

The serving integration is deliberately separate.  Qwen4-Exp needs its GDN,
QSA, PLE, and KV caches rolled back together after a partial accept; oMLX's
existing verify-window machinery owns that contract.
"""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from typing import Final, Sequence

EMPTY: Final = -1
_HASH_MULTIPLIER: Final = 6_364_136_223_846_793_005
_U64_MASK: Final = (1 << 64) - 1


@dataclass(slots=True)
class NGramModSequence:
    """Per-request cursor and acceptance state."""

    i_last: int = 0
    n_draft_last: int = 0
    low_accept_streak: int = 0


class NGramMod:
    """Fixed-memory exact suffix lookup table compatible with llama.cpp."""

    def __init__(
        self,
        *,
        n_match: int = 24,
        n_min: int = 48,
        n_max: int = 64,
        table_size: int = 4 * 1024 * 1024,
    ) -> None:
        if n_match <= 0:
            raise ValueError("n_match must be positive")
        if not 0 < n_min <= n_max:
            raise ValueError("expected 0 < n_min <= n_max")
        if table_size <= 0:
            raise ValueError("table_size must be positive")
        self.n_match = int(n_match)
        self.n_min = int(n_min)
        self.n_max = int(n_max)
        self.table_size = int(table_size)
        self._entries = array("i", [EMPTY]) * self.table_size
        self.used = 0

    @property
    def occupancy(self) -> float:
        return self.used / self.table_size

    def reset(self) -> None:
        self._entries = array("i", [EMPTY]) * self.table_size
        self.used = 0

    def index(self, tokens: Sequence[int], start: int = 0) -> int:
        if start < 0 or start + self.n_match > len(tokens):
            raise ValueError("ngram window is outside the token sequence")
        result = 0
        for pos in range(start, start + self.n_match):
            result = (
                result * _HASH_MULTIPLIER + int(tokens[pos])
            ) & _U64_MASK
        return result % self.table_size

    def add(self, tokens: Sequence[int], start: int) -> None:
        if start < 0 or start + self.n_match >= len(tokens):
            raise ValueError("ngram add requires a following token")
        idx = self.index(tokens, start)
        if self._entries[idx] == EMPTY:
            self.used += 1
        self._entries[idx] = int(tokens[start + self.n_match])

    def begin(self, prompt: Sequence[int]) -> NGramModSequence:
        state = NGramModSequence()
        if len(prompt) < self.n_match:
            return state
        for start in range(len(prompt) - self.n_match):
            self.add(prompt, start)
        state.i_last = len(prompt) - self.n_match
        if self.occupancy > 0.25:
            self.reset()
            state.i_last = 0
        return state

    def draft(
        self,
        state: NGramModSequence,
        history: Sequence[int],
    ) -> list[int]:
        """Return 48--64 deterministic proposals, or no proposal.

        ``history`` includes the most recently committed token.  The update
        cadence and minimum-span rejection match llama.cpp: newly committed
        ngrams are indexed in chunks after the cursor falls 32 tokens behind,
        and a chain shorter than ``n_min`` is discarded entirely.
        """

        state.n_draft_last = 0
        current = len(history)
        if current < self.n_match:
            return []
        if state.i_last + 32 < current:
            for start in range(state.i_last, current - self.n_match):
                self.add(history, start)
            state.i_last = current - self.n_match

        window = [int(token) for token in history[-self.n_match :]]
        proposed: list[int] = []
        for _ in range(self.n_max):
            idx = self.index(window, len(window) - self.n_match)
            token = int(self._entries[idx])
            if token == EMPTY:
                if len(proposed) < self.n_min:
                    return []
                break
            proposed.append(token)
            window.append(token)
        state.n_draft_last = len(proposed)
        return proposed

    def accept(self, state: NGramModSequence, accepted: int) -> None:
        """Apply llama.cpp's low-acceptance reset policy."""

        if state.n_draft_last <= 0:
            return
        fraction = max(0, int(accepted)) / state.n_draft_last
        if fraction < 0.25:
            state.low_accept_streak += 1
            if state.low_accept_streak >= 5:
                self.reset()
                state.low_accept_streak = 0
                state.i_last = 0
        else:
            state.low_accept_streak = 0


__all__ = ["EMPTY", "NGramMod", "NGramModSequence"]
