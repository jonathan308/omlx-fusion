# SPDX-License-Identifier: Apache-2.0
"""Prefill attention fast paths that MLX's fused SDPA does not cover.

Two gaps in ``mx.fast.scaled_dot_product_attention`` (mlx 0.32) cost a lot of
prefill time on large GPUs:

* Sliding-window attention is expressed as an array mask over the full
  ``[L, S]`` score matrix, so every query still scores every key and the
  window only masks the result. For a 128-token window at 4K context that is
  ~97% wasted work (and ~4 GB of scores per layer on the unfused path).
  ``blocked_sliding_window_attention`` tiles the queries into blocks that only
  see their ``block + window`` key span.
* Prefill with different query/key and value head dims (e.g. 192/128) has no
  fused kernel, so MLX materialises the full score matrix. Zero-padding V to
  the query head dim makes the fused kernel applicable; the extra output
  columns are exactly zero and sliced away. MLX also routes head dim 192/256
  prefill to the unfused path by default, which measures slower on NAX (M5)
  GPUs, so the fused kernel is requested explicitly there.

Both helpers compute the same attention as the masked full computation (up to
floating-point summation order).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

import mlx.core as mx

# Kill switch for A/B comparisons: OMLX_FAST_ATTENTION=0 keeps MLX's default
# SDPA routing everywhere.
_ENABLED = os.environ.get("OMLX_FAST_ATTENTION", "1").strip().lower() not in {
    "0",
    "false",
    "off",
}


@lru_cache(maxsize=1)
def _nax_available() -> bool:
    try:
        from omlx.custom_kernels.nax import is_nax_available

        return bool(is_nax_available())
    except Exception:  # noqa: BLE001
        return False


def mixed_head_dim_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    mask,
    sinks: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Fused SDPA for prefill with ``qk_dim > v_dim``; None when not applicable."""
    qk_dim, v_dim = queries.shape[-1], values.shape[-1]
    if (
        not _ENABLED
        or qk_dim <= v_dim
        or queries.shape[2] <= 8
        or not _nax_available()
        or qk_dim not in (64, 72, 80, 96, 128, 192, 256)
        or not (mask is None or isinstance(mask, str) or mask.dtype == mx.bool_)
    ):
        return None
    pad = [(0, 0)] * (values.ndim - 1) + [(0, qk_dim - v_dim)]
    out = mx.fast.scaled_dot_product_attention(
        queries,
        keys,
        mx.pad(values, pad),
        scale=scale,
        mask=mask,
        sinks=sinks,
        force_fused=True,
    )
    return out[..., :v_dim]


def blocked_sliding_window_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    window: int,
    sinks: Optional[mx.array] = None,
    mask=None,
    block: int = 128,
) -> Optional[mx.array]:
    """Causal sliding-window attention computed per query block.

    ``keys``/``values`` hold ``P >= 0`` prefix positions immediately preceding
    the ``L`` queries followed by the queries' own positions (the layout a
    KVCache / RotatingKVCache returns during prefill). Query ``i`` attends to
    keys at positions ``(i - window, i]``, matching mlx-lm's
    ``create_causal_mask(window_size=window)``. A boolean ``mask`` of shape
    ``[..., L, S]`` (e.g. one carrying batch-cache left padding) is honoured
    by slicing it per block; it must not allow keys outside the window.
    Returns None when the inputs do not fit this layout (batched inputs,
    short prompts, uneven blocks, additive masks).
    """
    B, H, L, D = queries.shape
    S = keys.shape[2]
    prefix = S - L
    if (
        not _ENABLED
        or B != 1
        or window <= 0
        or prefix < 0
        or L < 2 * block
        or keys.shape[2] != values.shape[2]
    ):
        return None
    user_mask = None
    if isinstance(mask, mx.array):
        if mask.dtype != mx.bool_ or mask.shape[-2:] != (L, S) or mask.size != L * S:
            return None
        user_mask = mask.reshape(L, S)
    elif mask is not None and mask != "causal":
        return None
    # Prompt chunks are rarely a multiple of the block (the scheduler keeps the
    # last prompt token for generation, so 4095 is typical): pad the queries
    # and the corresponding key/value positions and drop the padded rows at
    # the end. Padded keys sit after every real query position, so the causal
    # window never lets a real query see them.
    L_real = L
    pad_q = (-L) % block
    if pad_q:
        queries = mx.pad(queries, [(0, 0), (0, 0), (0, pad_q), (0, 0)])
        keys = mx.pad(keys, [(0, 0), (0, 0), (0, pad_q), (0, 0)])
        values = mx.pad(values, [(0, 0), (0, 0), (0, pad_q), (0, 0)])
        if user_mask is not None:
            user_mask = mx.pad(user_mask, [(0, pad_q), (0, pad_q)])
        L += pad_q
        S += pad_q
    Hk = keys.shape[1]
    v_dim = values.shape[-1]
    nb = L // block
    used = min(prefix, window)
    lead = window - used  # zero-padded (masked) positions before the prefix
    k = keys[:, :, S - L - used :, :]
    v = values[:, :, S - L - used :, :]
    if lead:
        k = mx.pad(k, [(0, 0), (0, 0), (lead, 0), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, 0), (lead, 0), (0, 0)])
    span = block + window
    idx = (mx.arange(nb) * block)[:, None] + mx.arange(span)[None, :]
    kb = k[:, :, idx, :].transpose(0, 2, 1, 3, 4).reshape(nb, Hk, span, D)
    vb = v[:, :, idx, :].transpose(0, 2, 1, 3, 4).reshape(nb, Hk, span, v_dim)
    qb = queries.reshape(B, H, nb, block, D).transpose(0, 2, 1, 3, 4)
    qb = qb.reshape(nb, H, block, D)

    # In block coordinates query r sits at key index r + window; it sees keys
    # j with r < j <= r + window. Keys in the zero padding (global index below
    # `lead`) are masked; only the first blocks can contain padding.
    r = mx.arange(block)[:, None]
    j = mx.arange(span)[None, :]
    base = (j > r) & (j <= r + window)
    starts = (mx.arange(nb) * block)[:, None, None]
    block_mask = base[None] & ((starts + j[None]) >= lead)
    if user_mask is not None:
        # Re-index the caller's key columns to the padded block layout.
        cols = user_mask[:, S - L - used :]
        if lead:
            cols = mx.pad(cols, [(0, 0), (lead, 0)])
        rows = cols.reshape(nb, block, lead + used + L)
        user_blocks = mx.take_along_axis(
            rows, mx.broadcast_to(idx[:, None, :], (nb, block, span)), axis=2
        )
        block_mask = block_mask & user_blocks
    block_mask = block_mask[:, None]  # broadcast over heads

    out = mx.fast.scaled_dot_product_attention(
        qb, kb, vb, scale=scale, mask=block_mask, sinks=sinks
    )
    out = out.reshape(B, nb, H, block, v_dim).transpose(0, 2, 1, 3, 4)
    return out.reshape(B, H, L, v_dim)[:, :, :L_real]
