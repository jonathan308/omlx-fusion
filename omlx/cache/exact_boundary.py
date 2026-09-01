# SPDX-License-Identifier: Apache-2.0
"""Fail-closed detached exact-boundary cache providers.

The first generic provider is intentionally narrow: an ordinary, unwrapped
``mlx_lm.models.cache.KVCache`` graph with batch size one.  Planning validates
the entire graph without allocating.  Materialization then copies only the
logical prefix into independently owned arrays and revalidates the source
before returning it to the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(frozen=True)
class _PlainKVLayerPlan:
    cache: Any
    keys: mx.array
    values: mx.array
    offset: int
    key_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    key_dtype: Any
    value_dtype: Any


@dataclass(frozen=True)
class PlainKVBoundaryPlan:
    """Allocation-free proof for one exact plain-KV prompt boundary."""

    source_tokens: int
    target_tokens: int
    estimated_nbytes: int
    layers: tuple[_PlainKVLayerPlan, ...]


@dataclass(frozen=True)
class DetachedPlainKVBoundary:
    """Independently owned, evaluated plain-KV boundary."""

    cache: list[Any]
    arrays: tuple[mx.array, ...]
    nbytes: int
    token_count: int


def _logical_prefix_nbytes(array: mx.array, target_tokens: int) -> int:
    shape = tuple(int(dim) for dim in array.shape)
    elements = 1
    for axis, dim in enumerate(shape):
        elements *= target_tokens if axis == 2 else dim
    return elements * int(array.itemsize)


def plan_plain_kv_boundary(
    cache_list: Any,
    *,
    source_tokens: int,
    target_tokens: int,
) -> PlainKVBoundaryPlan | None:
    """Validate a whole plain-KV graph without allocating any MLX arrays."""

    try:
        from mlx_lm.models.cache import KVCache
    except ImportError:
        return None

    if (
        not isinstance(cache_list, list)
        or not cache_list
        or type(source_tokens) is not int
        or type(target_tokens) is not int
        or source_tokens < 2
        or target_tokens != source_tokens - 1
    ):
        return None

    layers: list[_PlainKVLayerPlan] = []
    source_array_ids: set[int] = set()
    estimated_nbytes = 0
    for cache in cache_list:
        # Exact type is load-bearing. Subclasses may change layout, rotation,
        # quantization, or mutation semantics while retaining familiar fields.
        if type(cache) is not KVCache:
            return None
        keys = getattr(cache, "keys", None)
        values = getattr(cache, "values", None)
        offset = getattr(cache, "offset", None)
        if (
            not isinstance(keys, mx.array)
            or not isinstance(values, mx.array)
            or type(offset) is not int
            or offset != target_tokens
            or keys.ndim != 4
            or values.ndim != 4
        ):
            return None
        key_shape = tuple(int(dim) for dim in keys.shape)
        value_shape = tuple(int(dim) for dim in values.shape)
        if (
            key_shape[0] != 1
            or value_shape[0] != 1
            or key_shape[:3] != value_shape[:3]
            or key_shape[2] < target_tokens
            or value_shape[2] < target_tokens
            or cache.size() != target_tokens
            or id(keys) in source_array_ids
            or id(values) in source_array_ids
            or keys is values
        ):
            return None
        source_array_ids.update((id(keys), id(values)))
        estimated_nbytes += _logical_prefix_nbytes(keys, target_tokens)
        estimated_nbytes += _logical_prefix_nbytes(values, target_tokens)
        layers.append(
            _PlainKVLayerPlan(
                cache=cache,
                keys=keys,
                values=values,
                offset=offset,
                key_shape=key_shape,
                value_shape=value_shape,
                key_dtype=keys.dtype,
                value_dtype=values.dtype,
            )
        )

    if not layers or estimated_nbytes <= 0:
        return None
    return PlainKVBoundaryPlan(
        source_tokens=source_tokens,
        target_tokens=target_tokens,
        estimated_nbytes=estimated_nbytes,
        layers=tuple(layers),
    )


def _copy_prefix_array(array: mx.array, target_tokens: int) -> mx.array:
    """Build a distinct lazy output for the logical sequence prefix."""

    prefix = array[:, :, :target_tokens, :]
    try:
        detached = mx.copy(prefix)
    except AttributeError:
        # MLX before mx.copy: keep the established oMLX compatibility path.
        detached = prefix + mx.zeros((), dtype=prefix.dtype)
    # A plain slice may retain a much larger source allocation. Force the
    # detached logical boundary into its own compact row-contiguous buffer.
    return mx.contiguous(detached)


def materialize_plain_kv_boundary(
    plan: PlainKVBoundaryPlan,
    *,
    stream: Any,
) -> DetachedPlainKVBoundary | None:
    """Copy and evaluate a previously proved plain-KV boundary.

    Any source mutation between planning and materialization rejects the whole
    graph. Partial clones remain private locals and are never published.
    """

    try:
        from mlx_lm.models.cache import KVCache
    except ImportError:
        return None
    if not isinstance(plan, PlainKVBoundaryPlan) or not plan.layers:
        return None

    # Phase two starts with a no-allocation revalidation of every source leaf.
    for layer in plan.layers:
        cache = layer.cache
        if (
            type(cache) is not KVCache
            or getattr(cache, "keys", None) is not layer.keys
            or getattr(cache, "values", None) is not layer.values
            or getattr(cache, "offset", None) != layer.offset
            or tuple(int(dim) for dim in layer.keys.shape) != layer.key_shape
            or tuple(int(dim) for dim in layer.values.shape) != layer.value_shape
            or layer.keys.dtype != layer.key_dtype
            or layer.values.dtype != layer.value_dtype
            or cache.size() != plan.target_tokens
        ):
            return None

    source_ids = {
        identity
        for layer in plan.layers
        for identity in (id(layer.keys), id(layer.values))
    }
    cloned: list[Any] = []
    arrays: list[mx.array] = []
    try:
        with mx.stream(stream):
            for layer in plan.layers:
                keys = _copy_prefix_array(layer.keys, plan.target_tokens)
                values = _copy_prefix_array(layer.values, plan.target_tokens)
                clone = KVCache()
                clone.keys = keys
                clone.values = values
                clone.offset = plan.target_tokens
                cloned.append(clone)
                arrays.extend((keys, values))
            if source_ids.intersection(id(array) for array in arrays):
                return None
            mx.eval(*arrays)
    except Exception:  # noqa: BLE001 - provider contract is fail closed
        return None

    # Prove the source graph did not change while the detached graph evaluated.
    for layer in plan.layers:
        if (
            layer.cache.keys is not layer.keys
            or layer.cache.values is not layer.values
            or layer.cache.offset != layer.offset
            or tuple(int(dim) for dim in layer.keys.shape) != layer.key_shape
            or tuple(int(dim) for dim in layer.values.shape) != layer.value_shape
        ):
            return None
    if any(
        type(cache) is not KVCache
        or cache.offset != plan.target_tokens
        or cache.keys.shape[2] != plan.target_tokens
        or cache.values.shape[2] != plan.target_tokens
        for cache in cloned
    ):
        return None
    nbytes = sum(int(array.nbytes) for array in arrays)
    if nbytes != plan.estimated_nbytes:
        return None
    return DetachedPlainKVBoundary(
        cache=cloned,
        arrays=tuple(arrays),
        nbytes=nbytes,
        token_count=plan.target_tokens,
    )


__all__ = [
    "DetachedPlainKVBoundary",
    "PlainKVBoundaryPlan",
    "materialize_plain_kv_boundary",
    "plan_plain_kv_boundary",
]
