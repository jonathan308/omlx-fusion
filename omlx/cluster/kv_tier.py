# SPDX-License-Identifier: Apache-2.0
"""Durable, rank-local SSD tier for the distributed prompt cache.

oMLX's distributed path pins mlx-lm's prompt cache to a single resident slot
because per-rank eviction used to be unsafe: ranks retaining different
prefixes start the next request at different token offsets and block forever
in the first unmatched collective. This module ports ThunderMLX's answer
(docs/PERSISTENT_CACHE.md + the 2026-07-20 rank-local restore verdict): the
reuse ladder

    live holder -> resident slot -> validated SSD artifact -> cold recompute

where the durable rung is *rank-local* — every rank keeps its own artifacts,
its own disk budget, and its own eviction — and cross-rank coherence comes
from a single min-prefix vote taken between requests, never from shared
storage state.

Load-bearing invariants (ThunderMLX lab, 2026-07-12/19/20):

* COLLECTIVE-FREE ACTIONS: save, restore, and eviction perform no collectives
  and no cross-rank messaging. The one collective is ``agree_restore_tokens``,
  a between-requests vote in the request preflight — the same discipline as
  ``prefill_guard``'s deficit vote. Because the agreed length is the minimum
  of every rank's offer, and every saved stack is trimmable, each rank can
  always serve exactly the agreed prefix; a rank with nothing forces one cold
  rebuild whose completion autosave rewrites aligned artifacts everywhere.
* NEVER change cache growth geometry. Restore recreates spare capacity only
  by zero-padding the token axis to a bounded append reserve
  (``logical + reserve``, capped by ``max_kv_size`` and any rotating-cache
  window). Propagating a larger growth step into the allocator is the
  documented IOGPUFamily ``completeMemory() prepare count underflow`` kernel
  panic trigger — the 256-token native cadence is not touched here.
* STORAGE TUNING STAYS RANK-LOCAL: disk budget, directory, and append reserve
  resolve from per-rank environment (``..._RANK<n>`` overrides, ThunderMLX's
  rank-aware pattern). They never enter the plan: one Mac's disk budget says
  nothing about another's, and a shared budget sized for the big rank paged
  the small one into a crawl in production.
* Schema discipline (ThunderMLX schema v3): persist logical contents only —
  spare capacity is cheap to recreate and expensive to serialize — while
  recording logical length and physical layout separately, and validate
  before restore (schema version, model/cache fingerprint, token hash, cache
  classes, array geometry). Any mismatch is a safe miss, never an error.

mlx and mlx-lm imports are deliberately function-local: the inference worker
imports this module before it installs its torch stub and imports mlx.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import tempfile
import threading
import time
from array import array
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Artifact format version. Bump when the layout changes; older artifacts miss
# safely at validation and are rebuilt by the next autosave.
KV_TIER_SCHEMA_VERSION = 1

# Keep the enable env name in sync with performance.KV_TIER_ENV, which reads
# it once at plan time; this module reads it per rank as the local killswitch.
KV_TIER_ENV = "OMLX_CLUSTER_KV_TIER"
KV_TIER_DIR_ENV = "OMLX_CLUSTER_KV_TIER_DIR"
KV_TIER_BYTES_ENV = "OMLX_CLUSTER_KV_TIER_BYTES"
KV_TIER_MIN_TOKENS_ENV = "OMLX_CLUSTER_KV_TIER_MIN_TOKENS"
KV_TIER_APPEND_RESERVE_ENV = "OMLX_CLUSTER_KV_TIER_APPEND_RESERVE"

# Defaults. The disk budget is deliberately modest next to ThunderMLX's
# 300 GiB production rank0: the tier is opt-in and the floor protects
# daily-driver Macs; dedicated clusters raise it per rank.
_DEFAULT_TIER_BYTES = 64 * 1024**3
# ThunderMLX's SSD_MIN_TOKENS: smaller prompts are cheap to recompute and must
# not pay artifact I/O.
_DEFAULT_MIN_TOKENS = 8192
# ThunderMLX production append reserve (rank1 value; its rank0 used 8192).
_DEFAULT_APPEND_RESERVE_TOKENS = 4096

# At most one save waits behind the in-flight one. Artifacts supersede each
# other (a newer save is a longer prefix of the same conversation), so a full
# queue drops the oldest *pending* save, never blocks the serving thread.
_MAX_PENDING_SAVES = 1

# Cache classes whose every state array carries the token axis at ``shape[-2]``
# and whose spare capacity may be recreated by zero-padding that axis.
# ``MiniMaxM3KVCache`` (the cluster's sparse-index stack) qualifies: its
# nested ``(kv_state, index_state)`` leaves are all token-axis arrays, and its
# offset setter reseats both the KV and the index offsets. Any other class
# (MSA/conv/recurrent state, padded batch caches) is restored at exactly its
# logical length and is never cropped or padded.
_PAD_SAFE_CLASSES = frozenset(
    {
        "KVCache",
        "QuantizedKVCache",
        "ConcatenateKVCache",
        "RotatingKVCache",
        "MiniMaxM3KVCache",
    }
)

_INDEX_FILE = "_index.json"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int_rank_aware(name: str, rank: int, default: int) -> int:
    """A rank-local integer knob: ``name_RANK<rank>`` wins over ``name``.

    ThunderMLX's ``_env_int_rank_aware``: the two Macs have very different
    headroom, and a shared value sized for one paged the other into a crawl.
    """

    for key in (f"{name}_RANK{rank}", name):
        raw = os.environ.get(key, "").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                logger.warning("ignoring unparseable %s=%r", key, raw)
    return default


def _token_hash(tokens: list[int]) -> str:
    """Content hash of a token prefix, for artifact keys and restore validation.

    Token ids fit uint32. The byte order is the host's native one; every rank
    in an oMLX cluster is a little-endian Apple-silicon Mac, and the artifact
    is only ever read by the rank that wrote it.
    """

    data = array("I", tokens)
    return hashlib.blake2b(data.tobytes(), digest_size=16).hexdigest()


def cache_fingerprint(
    *,
    model_path: str,
    cache_classes: list[str],
    start_layer: int,
    end_layer: int,
    world_size: int,
    tensor_parallel_size: int,
    max_kv_size: int | None,
) -> str:
    """The identity an artifact is validated against before restore.

    Covers the model, this rank's pipeline slice, the world shape, and the
    cache classes the loaded model produces — ThunderMLX's runtime fingerprint
    reduced to what decides whether a serialized cache can stand in for the
    live one. Any change (different model, different split, different cache
    stack) makes every stored artifact a safe miss.
    """

    payload = json.dumps(
        {
            "model_path": model_path,
            "cache_classes": cache_classes,
            "start_layer": int(start_layer),
            "end_layer": int(end_layer),
            "world_size": int(world_size),
            "tensor_parallel_size": int(tensor_parallel_size),
            "max_kv_size": int(max_kv_size) if max_kv_size else None,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:32]


@dataclass(frozen=True)
class RankKVTierConfig:
    """One rank's storage knobs. Every field here is rank-local by design."""

    local_enabled: bool
    directory: Path
    max_bytes: int
    min_tokens: int
    append_reserve_tokens: int

    @classmethod
    def from_env(cls, *, rank: int) -> RankKVTierConfig:
        enabled = os.environ.get(KV_TIER_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        directory = Path(
            os.environ.get(KV_TIER_DIR_ENV, "~/.omlx/cluster/kv-tier")
        ).expanduser()
        return cls(
            local_enabled=enabled,
            directory=directory / f"rank-{int(rank)}",
            max_bytes=max(
                0, _env_int_rank_aware(KV_TIER_BYTES_ENV, rank, _DEFAULT_TIER_BYTES)
            ),
            min_tokens=max(
                1, _env_int(KV_TIER_MIN_TOKENS_ENV, _DEFAULT_MIN_TOKENS)
            ),
            append_reserve_tokens=max(
                0,
                _env_int_rank_aware(
                    KV_TIER_APPEND_RESERVE_ENV,
                    rank,
                    _DEFAULT_APPEND_RESERVE_TOKENS,
                ),
            ),
        )


class RankLocalKVTier:
    """SSD persistence for one rank's prompt-cache artifacts.

    All methods are fail-open: a corrupt index, an unreadable artifact, or a
    failed save degrades to a cache miss, never a request failure. Nothing
    here issues a collective except ``agree_restore_tokens``, which callers
    invoke between requests only.
    """

    def __init__(
        self,
        config: RankKVTierConfig,
        *,
        rank: int,
        world_size: int,
        model_fingerprint: str,
        cache_factory: Any,
        max_kv_size: int | None = None,
        eval_stream: Any | None = None,
    ) -> None:
        self._config = config
        self._rank = int(rank)
        self._world_size = int(world_size)
        self._fingerprint = model_fingerprint
        self._cache_factory = cache_factory
        self._max_kv_size = (
            int(max_kv_size) if max_kv_size and max_kv_size > 0 else None
        )
        # The worker's cross-thread generation stream. Cache state is created
        # lazily on it, and an MLX stream only exists on the thread that made
        # it — the saver thread must adopt the same stream to materialize
        # those arrays, which also serializes save I/O after model work.
        self._eval_stream = eval_stream
        self._lock = threading.Lock()
        self._artifacts: dict[str, dict[str, Any]] = {}
        self._save_queue: queue.Queue | None = None
        self._save_thread: threading.Thread | None = None
        self._closed = False
        if config.local_enabled:
            try:
                self._open()
            except Exception as exc:  # a tier that cannot open is a cold tier
                logger.warning(
                    "KV tier on rank %d cannot open %s (%s); running without it",
                    self._rank,
                    config.directory,
                    exc,
                )
                self._config = replace(config, local_enabled=False)

    # -- lifecycle -----------------------------------------------------------

    def _open(self) -> None:
        self._config.directory.mkdir(parents=True, exist_ok=True)
        self._artifacts = self._read_index()
        # Sweep files the index does not reference (interrupted writes) and
        # index entries whose file vanished (manual cleanup).
        referenced = {record["file"] for record in self._artifacts.values()}
        for record in list(self._artifacts.values()):
            if not (self._config.directory / record["file"]).is_file():
                del self._artifacts[record["artifact_id"]]
        for path in self._config.directory.glob("*.safetensors"):
            if path.name not in referenced:
                with suppress(OSError):
                    path.unlink()
        self._save_queue = queue.Queue(maxsize=_MAX_PENDING_SAVES)
        self._save_thread = threading.Thread(
            target=self._saver,
            name=f"omlx-kv-tier-rank-{self._rank}",
            daemon=True,
        )
        self._save_thread.start()

    def flush(self, timeout: float = 10.0) -> bool:
        """Wait for queued saves to finish (tests, shutdown). False on timeout."""

        deadline = time.monotonic() + max(0.0, timeout)
        save_queue = self._save_queue
        while save_queue is not None and save_queue.unfinished_tasks:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.01)
        return True

    def close(self, timeout: float = 5.0) -> None:
        self._closed = True
        save_queue = self._save_queue
        thread = self._save_thread
        if save_queue is None or thread is None:
            return
        self.flush(timeout)
        with suppress(queue.Full):
            save_queue.put_nowait(None)
        thread.join(timeout=max(0.0, timeout))
        self._save_queue = None
        self._save_thread = None

    # -- index persistence ---------------------------------------------------

    def _read_index(self) -> dict[str, dict[str, Any]]:
        path = self._config.directory / _INDEX_FILE
        try:
            payload = json.loads(path.read_text())
            records = payload.get("artifacts", [])
        except (OSError, ValueError):
            return {}
        artifacts: dict[str, dict[str, Any]] = {}
        for record in records if isinstance(records, list) else ():
            try:
                artifact = {
                    "artifact_id": str(record["artifact_id"]),
                    "file": str(record["file"]),
                    "token_count": int(record["token_count"]),
                    "token_hash": str(record["token_hash"]),
                    "model_fingerprint": str(record["model_fingerprint"]),
                    "byte_size": int(record["byte_size"]),
                    "created_at": float(record["created_at"]),
                    "last_access_at": float(record["last_access_at"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            artifacts[artifact["artifact_id"]] = artifact
        return artifacts

    def _write_index_locked(self) -> None:
        """Rewrite the index atomically; the caller holds ``self._lock``."""

        payload = {
            "schema_version": KV_TIER_SCHEMA_VERSION,
            "artifacts": sorted(
                self._artifacts.values(), key=lambda record: record["artifact_id"]
            ),
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".index.",
            suffix=".tmp",
            dir=self._config.directory,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._config.directory / _INDEX_FILE)
        finally:
            with suppress(OSError):
                os.unlink(temporary)

    # -- inspection ----------------------------------------------------------

    @property
    def total_bytes(self) -> int:
        with self._lock:
            return sum(record["byte_size"] for record in self._artifacts.values())

    @property
    def artifact_count(self) -> int:
        with self._lock:
            return len(self._artifacts)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._config.local_enabled,
                "artifacts": len(self._artifacts),
                "total_bytes": sum(
                    record["byte_size"] for record in self._artifacts.values()
                ),
                "budget_bytes": self._config.max_bytes,
                "directory": str(self._config.directory),
            }

    # -- save ----------------------------------------------------------------

    def save(self, tokens: Any, caches: list[Any]) -> bool:
        """Persist one finished conversation's caches. Never raises."""

        if not self._config.local_enabled or self._closed:
            return False
        try:
            return self._save_guarded(list(tokens), caches)
        except Exception as exc:  # a failed save is a future cache miss only
            logger.warning("KV tier save failed on rank %d: %s", self._rank, exc)
            return False

    def save_async(self, tokens: Any, caches: list[Any]) -> None:
        """Queue a save on the tier's daemon thread; never blocks serving.

        A full queue drops the oldest pending save: the newest queued artifact
        is a longer prefix of the same conversation and supersedes it.
        """

        if not self._config.local_enabled or self._closed:
            return
        try:
            token_list = list(tokens)
        except TypeError:
            return
        if len(token_list) < self._config.min_tokens:
            return
        save_queue = self._save_queue
        if save_queue is None:
            return
        try:
            save_queue.put_nowait((token_list, caches))
        except queue.Full:
            try:
                save_queue.get_nowait()
                save_queue.task_done()
            except queue.Empty:
                pass
            with suppress(queue.Full):
                save_queue.put_nowait((token_list, caches))

    def _saver(self) -> None:
        save_queue = self._save_queue
        if save_queue is None:
            return
        # A fresh thread has no MLX default stream; adopt the generation
        # stream the cache graphs were created on so save-time evaluation
        # works and stays serialized with model work.
        import mlx.core as mx

        if self._eval_stream is not None:
            mx.set_default_stream(self._eval_stream)
        while True:
            item = save_queue.get()
            try:
                if item is None:
                    return
                tokens, caches = item
                self.save(tokens, caches)
            finally:
                save_queue.task_done()

    def _save_guarded(self, tokens: list[int], caches: list[Any]) -> bool:
        import mlx.core as mx

        token_count = len(tokens)
        if token_count < self._config.min_tokens:
            return False
        validated = self._validate_saveable(tokens, caches, mx)
        if validated is None:
            return False
        arrays, classes, meta_states, shapes, dtypes, none_keys = validated

        token_hash = _token_hash(tokens)
        artifact_id = f"{token_count:012d}-{token_hash[:16]}"
        metadata = {
            "schema_version": str(KV_TIER_SCHEMA_VERSION),
            "token_count": str(token_count),
            "token_hash": token_hash,
            "model_fingerprint": self._fingerprint,
            "classes": json.dumps(classes),
            "meta_states": json.dumps(meta_states),
            "shapes": json.dumps(shapes),
            "dtypes": json.dumps(dtypes),
            "none_keys": json.dumps(none_keys),
        }
        # Write to a sibling temporary and rename: a rank that dies mid-save
        # leaves a file the open-time sweep reclaims, never a torn artifact.
        # The .safetensors suffix is load-bearing — mx.save_safetensors
        # appends it to any name that lacks it, which would strand the write.
        descriptor, temporary = tempfile.mkstemp(
            prefix=".artifact-",
            suffix=".safetensors",
            dir=self._config.directory,
        )
        os.close(descriptor)
        final = self._config.directory / f"{artifact_id}.safetensors"
        try:
            mx.save_safetensors(temporary, arrays, metadata)
            os.replace(temporary, final)
        finally:
            with suppress(OSError):
                os.unlink(temporary)
        now = time.time()
        with self._lock:
            self._artifacts[artifact_id] = {
                "artifact_id": artifact_id,
                "file": final.name,
                "token_count": token_count,
                "token_hash": token_hash,
                "model_fingerprint": self._fingerprint,
                "byte_size": final.stat().st_size,
                "created_at": now,
                "last_access_at": now,
            }
            self._write_index_locked()
        evicted = self._enforce_budget()
        if artifact_id in evicted:
            logger.warning(
                "KV tier artifact of %d tokens exceeds the rank %d budget alone; "
                "dropped it again",
                token_count,
                self._rank,
            )
            return False
        logger.info(
            "KV tier rank %d saved %d tokens (%.2f GiB, %d artifacts, %.2f/%.2f GiB)",
            self._rank,
            token_count,
            final.stat().st_size / 1024**3,
            self.artifact_count,
            self.total_bytes / 1024**3,
            self._config.max_bytes / 1024**3,
        )
        return True

    def _validate_saveable(
        self,
        tokens: list[int],
        caches: list[Any],
        mx: Any,
    ) -> (
        tuple[
            dict[str, Any],
            list[str],
            list[Any],
            list[Any],
            list[Any],
            list[list[str]],
        ]
        | None
    ):
        """Refuse a stack this tier cannot honestly persist.

        The stack must be trimmable (the min-prefix vote can ask any rank to
        serve a shorter prefix than its artifact), every cache's logical
        length must equal the token count (otherwise the artifact is not *this
        prefix* — a rotated window is not), and every state leaf must be an
        array or a recorded None so the restore is faithful.
        """

        from mlx.utils import tree_flatten
        from mlx_lm.models.cache import can_trim_prompt_cache

        token_count = len(tokens)
        if not caches or not can_trim_prompt_cache(caches):
            return None
        arrays: dict[str, Any] = {}
        classes: list[str] = []
        meta_states: list[list[str]] = []
        shapes: list[Any] = []
        dtypes: list[Any] = []
        none_keys: list[list[str]] = []
        for index, cache in enumerate(caches):
            offset = getattr(cache, "offset", None)
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or offset != token_count
            ):
                return None
            flat = dict(tree_flatten(cache.state))
            if not flat:
                return None
            layer_shapes: dict[str, Any] = {}
            layer_dtypes: dict[str, Any] = {}
            # Optional leaves (e.g. the sparse index of a MiniMax M3 cache) are
            # recorded by key and rebuilt as None on restore — safetensors
            # stores arrays only.
            layer_none: list[str] = []
            for key, leaf in flat.items():
                if leaf is None:
                    layer_none.append(key)
                    continue
                if not isinstance(leaf, mx.array):
                    return None
                arrays[f"{index}.{key}"] = leaf
                layer_shapes[key] = list(leaf.shape)
                layer_dtypes[key] = str(leaf.dtype)
            if not layer_shapes:
                return None
            classes.append(type(cache).__name__)
            # meta_state is a tuple of strings for the mlx-lm caches but a
            # plain string for MiniMaxM3KVCache — preserve the distinction or
            # the restore-side setter gets the wrong shape.
            meta = cache.meta_state
            meta_states.append(
                [str(part) for part in meta]
                if isinstance(meta, (tuple, list))
                else ("" if not meta else str(meta))
            )
            shapes.append(layer_shapes)
            dtypes.append(layer_dtypes)
            none_keys.append(layer_none)
        return arrays, classes, meta_states, shapes, dtypes, none_keys

    # -- restore -------------------------------------------------------------

    def match(self, tokens: list[int]) -> dict[str, Any] | None:
        """The longest stored artifact verified to be a prefix of ``tokens``.

        Index lookup plus a prefix content hash — no array I/O. A match here
        is only an *offer*; ``restore_prompt_cache`` still reconciles offers
        across ranks before anything is materialized for serving.
        """

        if not self._config.local_enabled:
            return None
        length = len(tokens)
        with self._lock:
            candidates = sorted(
                (
                    record
                    for record in self._artifacts.values()
                    if record["token_count"] <= length
                    and record["model_fingerprint"] == self._fingerprint
                ),
                key=lambda record: record["token_count"],
                reverse=True,
            )
        for record in candidates:
            if _token_hash(tokens[: record["token_count"]]) == record["token_hash"]:
                with self._lock:
                    current = self._artifacts.get(record["artifact_id"])
                    if current is not None:
                        current["last_access_at"] = time.time()
                return record
        return None

    def restore_prompt_cache(
        self,
        tokens: list[int],
        *,
        mx_module: Any,
    ) -> tuple[list[Any], list[int]] | None:
        """The durable rung of the reuse ladder: match, materialize, vote, trim.

        Called from the request preflight on every rank at the same point (a
        full RAM miss with the tier plan-enabled — both rank-uniform), so the
        vote below executes exactly once per request per rank. Everything else
        is rank-local. The artifact is materialized BEFORE the vote: a rank
        whose disk read fails must offer zero rather than promise a prefix it
        cannot serve and desync the group. The agreed length is the minimum
        offer, which every rank can then serve by trimming.
        """

        local_offer = 0
        prepared: list[Any] | None = None
        record: dict[str, Any] | None = None
        try:
            record = self.match(tokens)
            if record is not None:
                prepared = self._load_artifact(record)
                if prepared is not None:
                    local_offer = int(record["token_count"])
        except Exception as exc:  # fail-open: offer nothing, still vote
            logger.warning(
                "KV tier restore probe failed on rank %d: %s", self._rank, exc
            )
            local_offer = 0
            prepared = None

        agreed = self.agree_restore_tokens(local_offer, mx_module=mx_module)
        if agreed <= 0 or prepared is None:
            if local_offer > 0 and agreed <= 0:
                logger.info(
                    "KV tier rank %d drops a %d-token restore: a peer offered "
                    "nothing, so the cluster rebuilds cold and realigns",
                    self._rank,
                    local_offer,
                )
            return None
        # agreed <= local_offer by min(): trim this rank's longer artifact
        # down to the agreed prefix. trim() on the validated plain cache
        # classes is offset arithmetic — infallible, which matters because a
        # post-vote failure on one rank would desync the group.
        trim = local_offer - agreed
        if trim:
            from mlx_lm.models.cache import trim_prompt_cache

            trim_prompt_cache(prepared, trim)
        self._pad_caches(prepared, agreed)
        logger.info(
            "KV tier rank %d restored %d/%d tokens from SSD",
            self._rank,
            agreed,
            len(tokens),
        )
        return prepared, tokens[agreed:]

    def agree_restore_tokens(self, local_offer: int, *, mx_module: Any) -> int:
        """The ladder's one collective: the minimum restorable prefix, cluster-wide.

        One-hot-per-rank offer vector reduced with ``all_sum`` — the same
        shape as ``prefill_guard``'s deficit vote. A rank that cannot serve a
        prefix must block its use everywhere, so the agreement is the minimum:
        unequal disk budgets mean unequal artifact sets, and a restore only
        one rank can serve would start the pipeline at different token
        offsets. Between requests only; never from the decode/pipeline path.
        """

        local_offer = max(0, int(local_offer))
        if self._world_size <= 1:
            return local_offer
        offers = [0] * self._world_size
        offers[self._rank] = local_offer
        agreed = mx_module.distributed.all_sum(mx_module.array(offers)).tolist()
        return int(min(agreed))

    def _load_artifact(self, record: dict[str, Any]) -> list[Any] | None:
        """Materialize one artifact into fresh caches, or None on any mismatch.

        Validation order follows ThunderMLX's load-side miss reasons: schema,
        fingerprint, token count/hash, cache classes, then per-array geometry.
        Layers materialize one at a time so the transient footprint stays
        bounded (retaining every source tensor until one final eval doubles
        the KV footprint at long context).
        """

        import mlx.core as mx
        from mlx.utils import tree_unflatten

        if record["model_fingerprint"] != self._fingerprint:
            return None
        path = self._config.directory / record["file"]
        try:
            arrays, metadata = mx.load(str(path), return_metadata=True)
        except Exception:
            return None
        if metadata.get("schema_version") != str(KV_TIER_SCHEMA_VERSION):
            return None
        if metadata.get("token_hash") != record["token_hash"]:
            return None
        try:
            token_count = int(metadata.get("token_count", ""))
        except ValueError:
            return None
        if token_count != record["token_count"]:
            return None
        try:
            classes = json.loads(metadata["classes"])
            meta_states = json.loads(metadata["meta_states"])
            shapes = json.loads(metadata["shapes"])
            dtypes = json.loads(metadata["dtypes"])
            none_keys = json.loads(metadata.get("none_keys", "[]"))
        except (KeyError, ValueError):
            return None
        try:
            fresh = list(self._cache_factory())
        except Exception:
            return None
        if len(fresh) != len(classes) or [type(c).__name__ for c in fresh] != classes:
            return None

        for index, cache in enumerate(fresh):
            prefix = f"{index}."
            leaves = {
                key[len(prefix) :]: value
                for key, value in arrays.items()
                if key.startswith(prefix)
            }
            if not leaves:
                return None
            # Crop any oversized stored backing to the logical contents; the
            # token axis is shape[-2] for every pad-safe class's arrays.
            class_name = classes[index]
            cropped: dict[str, Any] = {}
            for key, leaf in leaves.items():
                if (
                    class_name in _PAD_SAFE_CLASSES
                    and getattr(leaf, "ndim", 0) >= 2
                    and leaf.shape[-2] != token_count
                ):
                    if leaf.shape[-2] < token_count:
                        return None
                    leaf = leaf[..., :token_count, :]
                cropped[key] = leaf
            recorded_shapes = shapes[index]
            recorded_dtypes = dtypes[index]
            if sorted(cropped) != sorted(recorded_shapes):
                return None
            for key, leaf in cropped.items():
                if list(leaf.shape) != list(recorded_shapes[key]):
                    return None
                if str(leaf.dtype) != str(recorded_dtypes[key]):
                    return None
            try:
                layer_none = none_keys[index] if index < len(none_keys) else []
                state = tree_unflatten(
                    list(cropped.items()) + [(key, None) for key in layer_none]
                )
                cache.state = state
                meta = meta_states[index]
                if isinstance(meta, str):
                    if meta:
                        cache.meta_state = meta
                elif meta:
                    cache.meta_state = tuple(meta)
            except Exception:
                return None
            offset = getattr(cache, "offset", None)
            if isinstance(offset, int) and offset != token_count:
                cache.offset = token_count
            try:
                # Layer-at-a-time: materialize this layer, then drop the loaded
                # source references before touching the next.
                mx.eval(cache.state)
            except Exception:
                return None
            del leaves, cropped
        return fresh

    def _pad_caches(self, caches: list[Any], logical_tokens: int) -> None:
        """Recreate a bounded append reserve after a restore.

        ThunderMLX's bounded rank-aware reserve: a restored cache has zero
        spare slots, so every 256-token allocator crossing would realloc-copy
        the whole KV on every layer. Pad the token axis to
        ``logical + reserve``, capped by ``max_kv_size`` and by any rotating
        window. This NEVER changes the growth step — reserving by padding is
        the safe side of the IOGPUFamily underflow panic line.
        """

        reserve = self._config.append_reserve_tokens
        if reserve <= 0:
            return
        target = logical_tokens + reserve
        if self._max_kv_size is not None:
            target = min(target, self._max_kv_size)
        for cache in caches:
            self._pad_cache(cache, target)

    def _pad_cache(self, cache: Any, target: int) -> None:
        import mlx.core as mx
        from mlx.utils import tree_map

        if type(cache).__name__ not in _PAD_SAFE_CLASSES:
            return
        max_size = getattr(cache, "max_size", None)
        if isinstance(max_size, int) and max_size > 0:
            target = min(target, max_size)
        offset = getattr(cache, "offset", None)
        if not isinstance(offset, int) or isinstance(offset, bool):
            return
        if target <= offset:
            return

        def pad_leaf(leaf: Any) -> Any:
            if getattr(leaf, "ndim", 0) >= 2 and leaf.shape[-2] == offset:
                zeros = mx.zeros(
                    (*leaf.shape[:-2], target - offset, leaf.shape[-1]),
                    leaf.dtype,
                )
                return mx.concatenate([leaf, zeros], axis=-2)
            return leaf

        try:
            cache.state = tree_map(pad_leaf, cache.state)
        except Exception:
            return
        # KVCache/ConcatenateKVCache derive the offset from the (now padded)
        # backing shape in their state setter; the logical length is unchanged.
        cache.offset = offset
        meta = cache.meta_state
        if isinstance(meta, tuple) and meta:
            cache.meta_state = tuple(str(part) for part in meta)

    # -- capacity management -------------------------------------------------

    def _enforce_budget(self) -> list[str]:
        """Evict least-recently-accessed artifacts until within budget.

        Pure rank-local file maintenance: no mlx, no collectives, no cross-rank
        messaging — an eviction on one rank can only shrink its future restore
        offers, and the min-prefix vote reconciles that with its peers. Keep
        this function free of array machinery; the acceptance test asserts it.
        """

        evicted: list[str] = []
        with self._lock:
            total = sum(record["byte_size"] for record in self._artifacts.values())
            if total <= self._config.max_bytes:
                return evicted
            for record in sorted(
                self._artifacts.values(),
                key=lambda item: (item["last_access_at"], item["artifact_id"]),
            ):
                if total <= self._config.max_bytes:
                    break
                with suppress(OSError):
                    (self._config.directory / record["file"]).unlink()
                total -= record["byte_size"]
                evicted.append(record["artifact_id"])
                del self._artifacts[record["artifact_id"]]
            if evicted:
                self._write_index_locked()
        if evicted:
            logger.info(
                "KV tier rank %d evicted %d artifact(s) to stay within its %.2f GiB budget",
                self._rank,
                len(evicted),
                self._config.max_bytes / 1024**3,
            )
        return evicted


def build_rank_kv_tier(
    *,
    plan_enabled: bool,
    rank: int,
    world_size: int,
    model_path: str,
    model: Any,
    start_layer: int,
    end_layer: int,
    tensor_parallel_size: int,
    max_kv_size: int | None,
    eval_stream: Any | None = None,
) -> RankLocalKVTier | None:
    """Construct the worker's tier, or None when the plan did not enable it.

    When the plan enables the tier this returns an object even if the local
    probes fail (env killswitch, unreadable directory, unprobeable model): the
    object is what makes the restore vote run on every rank at the same
    requests — a rank that simply had no object would skip a collective its
    peers are waiting on. A degraded tier just offers zero, which is a cold
    rebuild, never a hang.
    """

    if not plan_enabled:
        return None
    config = RankKVTierConfig.from_env(rank=rank)
    fingerprint = ""
    factory: Any = None
    if config.local_enabled:
        try:
            from mlx_lm.models.cache import make_prompt_cache

            probe = make_prompt_cache(model)
            fingerprint = cache_fingerprint(
                model_path=model_path,
                cache_classes=[type(cache).__name__ for cache in probe],
                start_layer=start_layer,
                end_layer=end_layer,
                world_size=world_size,
                tensor_parallel_size=tensor_parallel_size,
                max_kv_size=max_kv_size,
            )

            def probe_factory() -> Any:
                return make_prompt_cache(model)

            factory = probe_factory
        except Exception as exc:
            logger.warning(
                "KV tier disabled on rank %d: cannot probe the model's caches: %s",
                rank,
                exc,
            )
    if factory is None:
        config = replace(config, local_enabled=False)

        def broken_factory() -> Any:
            raise RuntimeError("KV tier has no usable cache factory")

        factory = broken_factory

    return RankLocalKVTier(
        config,
        rank=rank,
        world_size=world_size,
        model_fingerprint=fingerprint,
        cache_factory=factory,
        max_kv_size=max_kv_size,
        eval_stream=eval_stream,
    )
