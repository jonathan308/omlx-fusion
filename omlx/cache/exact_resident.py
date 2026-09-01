# SPDX-License-Identifier: Apache-2.0
"""Bounded, exact-token resident prompt-cache handoff.

The paged prefix cache is the durable/general fallback.  This small tier keeps
the most recently detached *live* cache object so a following chat turn can
take ownership without serializing, loading, or concatenating its blocks.

Correctness is deliberately simple: an entry is reusable only when every
stored token is an exact prefix of the new scheduler-owned prompt.  Entries
are removed on acquisition, so a mutable cache object is never shared by two
requests.  Callers remain responsible for validating model/cache offsets and
for excluding media-keyed requests.
"""

from __future__ import annotations

import threading
from array import array
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ExactResidentHit:
    """Exclusive ownership transfer returned by :meth:`acquire_prefix`."""

    cache: list[Any]
    cached_tokens: int
    cache_nbytes: int
    durable_tokens: int


@dataclass
class _ExactResidentEntry:
    tokens: array
    cache: list[Any]
    cache_nbytes: int
    durable_tokens: int


class ExactResidentPrefixCache:
    """A tiny LRU of exact terminal cache objects.

    The scheduler never hands a retained entry to the asynchronous durable
    writer.  It is therefore claimable immediately after the response, with
    no shared-array reader race. ``durable_tokens`` records the independently
    published paged/SSD prompt boundary that remains the crash, eviction, and
    concurrent-claim fallback for that mutable terminal state.
    """

    def __init__(
        self,
        max_entries: int = 1,
        max_bytes: int = 8 * 1024**3,
    ) -> None:
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self._entries: OrderedDict[int, _ExactResidentEntry] = OrderedDict()
        self._size_bytes = 0
        self._next_id = 0
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.oversize_rejections = 0
        self.protected_rejections = 0

    @staticmethod
    def _tokens_equal_prefix(stored: array, prompt: list[int]) -> bool:
        if len(stored) >= len(prompt):
            # A non-empty suffix is required.  Generation from cache state at
            # exactly N tokens needs an N-1 trim/kickoff, which recurrent cache
            # families cannot perform generically.
            return False
        return all(saved == current for saved, current in zip(stored, prompt))

    @staticmethod
    def _stored_extends_candidate(stored: array, candidate: array) -> bool:
        return bool(
            len(stored) > len(candidate)
            and all(saved == current for saved, current in zip(stored, candidate))
        )

    def can_fit_protected_candidate(
        self,
        tokens: Iterable[int],
        *,
        estimated_cache_nbytes: int,
    ) -> bool:
        """Preflight a fallback without sacrificing a longer terminal entry.

        Unrelated LRU entries are evictable.  A longer entry that extends this
        exact candidate is protected, and its measured bytes also calibrate a
        conservative same-cache-family estimate for the shorter candidate.
        """

        try:
            token_array = array("I", [int(token) for token in tokens])
        except (OverflowError, TypeError, ValueError):
            return False
        if not token_array or token_array.itemsize != 4:
            return False
        estimate = max(0, int(estimated_cache_nbytes))
        with self._lock:
            protected = [
                entry
                for entry in self._entries.values()
                if self._stored_extends_candidate(entry.tokens, token_array)
            ]
            if not protected:
                return True
            for entry in protected:
                if len(entry.tokens) > 0:
                    estimate = max(
                        estimate,
                        (
                            entry.cache_nbytes * len(token_array)
                            + len(entry.tokens)
                            - 1
                        )
                        // len(entry.tokens),
                    )
            return bool(
                len(protected) + 1 <= self.max_entries
                and sum(entry.cache_nbytes for entry in protected) + estimate
                <= self.max_bytes
            )

    def put(
        self,
        tokens: Iterable[int],
        cache: list[Any],
        *,
        cache_nbytes: int = 0,
        durable_tokens: int = 0,
        protect_longer_prefix: bool = False,
    ) -> bool:
        """Retain one detached cache, evicting oldest entries as needed."""

        if self.max_entries <= 0 or not isinstance(cache, list) or not cache:
            return False
        token_values = [int(token) for token in tokens]
        if any(token < 0 or token > 0xFFFFFFFF for token in token_values):
            return False
        token_array = array("I", token_values)
        if token_array.itemsize != 4:
            # The durable/on-wire scheduler token contract is uint32.  Refuse
            # a platform whose native unsigned-int array is not 32 bits.
            return False
        if not token_array:
            return False
        cache_nbytes = max(0, int(cache_nbytes))
        durable_tokens = int(durable_tokens)
        if durable_tokens < 0 or durable_tokens > len(token_array):
            return False
        if self.max_bytes <= 0 or cache_nbytes > self.max_bytes:
            with self._lock:
                self.oversize_rejections += 1
            return False
        entry = _ExactResidentEntry(
            tokens=token_array,
            cache=cache,
            cache_nbytes=cache_nbytes,
            durable_tokens=durable_tokens,
        )
        with self._lock:
            if protect_longer_prefix:
                protected_ids = {
                    entry_id
                    for entry_id, existing in self._entries.items()
                    if self._stored_extends_candidate(
                        existing.tokens,
                        token_array,
                    )
                }
                if protected_ids:
                    prospective_count = len(self._entries) + 1
                    prospective_bytes = self._size_bytes + cache_nbytes
                    victims: list[int] = []
                    for entry_id, existing in self._entries.items():
                        if (
                            prospective_count <= self.max_entries
                            and prospective_bytes <= self.max_bytes
                        ):
                            break
                        if entry_id in protected_ids:
                            continue
                        victims.append(entry_id)
                        prospective_count -= 1
                        prospective_bytes -= existing.cache_nbytes
                    if (
                        prospective_count > self.max_entries
                        or prospective_bytes > self.max_bytes
                    ):
                        # Prompt-tail prewarm is a fallback for transcript
                        # divergence, never a reason to evict a longer terminal
                        # state that may give the next turn a better exact hit.
                        self.protected_rejections += 1
                        return False
                    for entry_id in victims:
                        evicted = self._entries.pop(entry_id)
                        self._size_bytes -= evicted.cache_nbytes
                        self.evictions += 1
            self._next_id += 1
            self._entries[self._next_id] = entry
            self._size_bytes += cache_nbytes
            while (
                len(self._entries) > self.max_entries
                or self._size_bytes > self.max_bytes
            ):
                _, evicted = self._entries.popitem(last=False)
                self._size_bytes -= evicted.cache_nbytes
                self.evictions += 1
        return True

    def acquire_prefix(self, prompt_tokens: list[int]) -> ExactResidentHit | None:
        """Pop the longest ready entry that exactly prefixes ``prompt_tokens``."""

        if self.max_entries <= 0 or not prompt_tokens:
            self.misses += 1
            return None
        with self._lock:
            best_id = None
            best_len = -1
            for entry_id, entry in reversed(self._entries.items()):
                if len(entry.tokens) <= best_len:
                    continue
                if self._tokens_equal_prefix(entry.tokens, prompt_tokens):
                    best_id = entry_id
                    best_len = len(entry.tokens)

            if best_id is None:
                self.misses += 1
                return None

            entry = self._entries.pop(best_id)
            self._size_bytes -= entry.cache_nbytes
            self.hits += 1
            return ExactResidentHit(
                cache=entry.cache,
                cached_tokens=len(entry.tokens),
                cache_nbytes=entry.cache_nbytes,
                durable_tokens=entry.durable_tokens,
            )

    def clear(self) -> int:
        """Drop resident references without touching the durable cache tier."""

        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            self._size_bytes = 0
            return count

    def resize(self, max_entries: int) -> int:
        """Apply a live entry limit and evict oldest entries to fit."""

        max_entries = max(0, int(max_entries))
        evicted_count = 0
        with self._lock:
            self.max_entries = max_entries
            while len(self._entries) > self.max_entries:
                _, evicted = self._entries.popitem(last=False)
                self._size_bytes -= evicted.cache_nbytes
                self.evictions += 1
                evicted_count += 1
        return evicted_count

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "size_bytes": self._size_bytes,
                "max_token_count": max(
                    (len(entry.tokens) for entry in self._entries.values()),
                    default=0,
                ),
                "max_durable_token_count": max(
                    (entry.durable_tokens for entry in self._entries.values()),
                    default=0,
                ),
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes if self.max_entries > 0 else 0,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "oversize_rejections": self.oversize_rejections,
                "protected_rejections": self.protected_rejections,
            }
