# SPDX-License-Identifier: Apache-2.0
"""Native, fail-closed GLM-5.3-Flash vision execution for MLX.

Math and media ABI are pinned to Hugging Face Transformers commit
``eb4d9e2a64a013bec12289288b85d0b1210ba0aa``. MLX is imported only by
runtime factories so checkpoint inspection cannot allocate the vision tower.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache, partial
from typing import Any, Final, NamedTuple

VISION_PREFIX: Final = "model.visual."
TRANSFORMERS_REVISION: Final = "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"
TRANSFORMERS_SOURCE: Final = "src/transformers/models/glm5_next/modeling_glm5_next.py"
IMAGE_TOKEN_ID: Final = 154_854
VIDEO_START_TOKEN_ID: Final = 154_832
VIDEO_END_TOKEN_ID: Final = 154_833
_PATCH_ELEMENTS: Final = 3 * 2 * 14 * 14
GLM5_NEXT_VISION_RUNTIME_READY: Final = True
OFFICIAL_PROCESSOR: Final = "transformers.AutoProcessor"
OFFICIAL_PROCESSOR_REQUIRED_OUTPUTS: Final = (
    "input_ids",
    "attention_mask",
    "mm_token_type_ids",
)
OFFICIAL_IMAGE_PROCESSOR_OUTPUTS: Final = ("pixel_values", "image_grid_thw")
OFFICIAL_VIDEO_PROCESSOR_OUTPUTS: Final = (
    "pixel_values_videos",
    "video_grid_thw",
)


class Glm5NextVisionUnsupportedError(RuntimeError):
    """The supplied media does not satisfy the pinned native vision ABI."""


class MediaKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    IMAGE_AND_VIDEO = "image_and_video"


@dataclass(frozen=True)
class VisionMediaInput:
    pixel_values: Any
    grid_thw: Any
    encoder_grid_thw: tuple[tuple[int, int, int], ...]
    split_sizes: tuple[int, ...]


@dataclass(frozen=True)
class PreparedVisionMedia:
    kind: MediaKind
    image: VisionMediaInput | None = None
    video: VisionMediaInput | None = None


class VisionModelOutput(NamedTuple):
    last_hidden_state: Any
    pooler_output: Any


@dataclass(frozen=True)
class VisionRuntimeIntegration:
    """Pure outer-model binding; constructing it imports or allocates no MLX arrays."""

    model_constructor: type
    configure_for_converted_weights: Callable[[Any], Any]
    prepare_media: Callable[..., PreparedVisionMedia]
    load_processor: Callable[..., Any]
    install_processor_namespace: Callable[[], bool]
    processor_class: str
    processor_revision: str
    processor_required_outputs: tuple[str, ...]
    image_processor_outputs: tuple[str, ...]
    video_processor_outputs: tuple[str, ...]


def _get(config: Any, name: str, default: Any = None) -> Any:
    return (
        config.get(name, default)
        if isinstance(config, Mapping)
        else getattr(config, name, default)
    )


def _vision_config(config: Any) -> Any:
    nested = _get(config, "vision_config")
    return config if nested is None else nested


def _shape(value: Any) -> tuple[int, ...]:
    return tuple(getattr(value, "shape", ()))


def _rows(value: Any, *, label: str) -> tuple[tuple[int, int, int], ...]:
    shape = _shape(value)
    if (
        not shape
        and isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
    ):
        shape = (
            (len(value), 3)
            if all(
                isinstance(row, Sequence)
                and not isinstance(row, (str, bytes))
                and len(row) == 3
                for row in value
            )
            else ()
        )
    if len(shape) != 2 or shape[1] != 3 or shape[0] < 1:
        raise Glm5NextVisionUnsupportedError(
            f"GLM5-Next {label} must have shape (num_items, 3); found {shape}"
        )
    try:
        raw = value.tolist()
    except AttributeError:
        raw = value
    try:
        rows = tuple(tuple(int(cell) for cell in row) for row in raw)
    except (TypeError, ValueError) as error:
        raise Glm5NextVisionUnsupportedError(
            f"GLM5-Next {label} must contain integer (t, h, w) rows"
        ) from error
    if len(rows) != shape[0] or any(len(row) != 3 for row in rows):
        raise Glm5NextVisionUnsupportedError(
            f"GLM5-Next {label} contents do not match shape {shape}"
        )
    if any(
        cell != converted
        for row, converted_row in zip(raw, rows)
        for cell, converted in zip(row, converted_row)
    ):
        raise Glm5NextVisionUnsupportedError(
            f"GLM5-Next {label} must contain exact integer (t, h, w) values"
        )
    for t, height, width in rows:
        if t < 1 or height < 1 or width < 1:
            raise Glm5NextVisionUnsupportedError(
                f"GLM5-Next {label} entries must be positive; found {(t, height, width)}"
            )
        if height % 2 or width % 2:
            raise Glm5NextVisionUnsupportedError(
                f"GLM5-Next {label} height/width must be divisible by spatial_merge_size=2; "
                f"found {(t, height, width)}"
            )
    return rows


def validate_vision_config(config: Any) -> None:
    config = _vision_config(config)
    expected = {
        "model_type": "glm5_next_vision",
        "depth": 24,
        "hidden_size": 1_024,
        "intermediate_size": 4_096,
        "num_heads": 16,
        "image_size": 448,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 4_096,
        "projection_intermediate_size": 10_240,
        "in_channels": 3,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-5,
    }
    errors = [
        f"{name}={_get(config, name)!r} (expected {wanted!r})"
        for name, wanted in expected.items()
        if _get(config, name) != wanted
    ]
    if errors:
        raise ValueError("Unsupported GLM5-Next vision config: " + "; ".join(errors))


def _required_vision_shapes(prefix: str) -> dict[str, tuple[int, ...]]:
    shapes = {
        f"{prefix}patch_embed.proj.weight": (1_024, 3, 2, 14, 14),
        f"{prefix}patch_embed.proj.bias": (1_024,),
        f"{prefix}post_layernorm.weight": (1_024,),
        f"{prefix}downsample.weight": (4_096, 1_024, 2, 2),
        f"{prefix}downsample.bias": (4_096,),
        f"{prefix}merger.proj.weight": (4_096, 4_096),
        f"{prefix}merger.post_projection_norm.weight": (4_096,),
        f"{prefix}merger.post_projection_norm.bias": (4_096,),
        f"{prefix}merger.gate_proj.weight": (10_240, 4_096),
        f"{prefix}merger.up_proj.weight": (10_240, 4_096),
        f"{prefix}merger.down_proj.weight": (4_096, 10_240),
    }
    block = {
        "attn.qkv.weight": (3_072, 1_024),
        "attn.qkv.bias": (3_072,),
        "attn.q_norm.weight": (64,),
        "attn.k_norm.weight": (64,),
        "attn.proj.weight": (1_024, 1_024),
        "attn.proj.bias": (1_024,),
        "mlp.gate_proj.weight": (4_096, 1_024),
        "mlp.gate_proj.bias": (4_096,),
        "mlp.up_proj.weight": (4_096, 1_024),
        "mlp.up_proj.bias": (4_096,),
        "mlp.down_proj.weight": (1_024, 4_096),
        "mlp.down_proj.bias": (1_024,),
        "norm1.weight": (1_024,),
        "norm2.weight": (1_024,),
    }
    for layer in range(24):
        for suffix, shape in block.items():
            shapes[f"{prefix}blocks.{layer}.{suffix}"] = shape
    return shapes


def _validate_affine_quantization(quantization: Mapping[str, Any]) -> tuple[int, int]:
    bits = quantization.get("bits")
    group_size = quantization.get("group_size")
    mode = quantization.get("mode", "affine")
    if bits not in {4, 8}:
        raise ValueError(f"GLM5-Next vision affine bits must be 4 or 8; found {bits!r}")
    if not isinstance(group_size, int) or group_size <= 0 or group_size % 32:
        raise ValueError(
            "GLM5-Next vision affine group_size must be a positive multiple of 32; "
            f"found {group_size!r}"
        )
    if mode != "affine":
        raise ValueError(
            f"GLM5-Next vision quantization mode must be 'affine'; found {mode!r}"
        )
    return bits, group_size


def _converted_affine_eligible(
    name: str, shape: tuple[int, ...], group_size: int
) -> bool:
    """Mirror the pinned converter's dense/affine decision for vision tensors."""

    lower = name.lower()
    must_remain_dense = (
        "norm" in lower
        or lower.endswith(".bias")
        or "a_log" in lower
        or "dt_bias" in lower
        or "conv1d" in lower
        or ".indexer." in lower
        or ".hc_" in lower
        or ".mlp.gate." in lower
        or "index_kpool" in lower
    )
    return (
        name.endswith(".weight")
        and len(shape) >= 2
        and shape[-1] % group_size == 0
        and not must_remain_dense
    )


def converted_vision_parameter_shapes(
    *, bits: int, group_size: int = 64, prefix: str = VISION_PREFIX
) -> dict[str, tuple[int, ...]]:
    """Return every converted Q8/Q4 vision parameter name and packed shape.

    This is an allocation-free projection of the converter ABI. Affine layers
    become ``weight/scales/biases``; patch embedding, downsample, norms, and
    learned biases retain their official dense names.
    """

    _validate_affine_quantization(
        {"bits": bits, "group_size": group_size, "mode": "affine"}
    )
    converted: dict[str, tuple[int, ...]] = {}
    for name, shape in _required_vision_shapes(prefix).items():
        if not _converted_affine_eligible(name, shape, group_size):
            converted[name] = shape
            continue
        base = name.removesuffix(".weight")
        converted[name] = (*shape[:-1], shape[-1] * bits // 32)
        group_shape = (*shape[:-1], shape[-1] // group_size)
        converted[base + ".scales"] = group_shape
        converted[base + ".biases"] = group_shape
    return converted


def validate_converted_vision_weight_layout(
    weights_or_names: Mapping[str, Any] | Iterable[str],
    *,
    bits: int,
    group_size: int = 64,
    prefix: str = VISION_PREFIX,
) -> None:
    """Require the exact converted affine family, optionally checking shapes."""

    expected = converted_vision_parameter_shapes(
        bits=bits, group_size=group_size, prefix=prefix
    )
    if isinstance(weights_or_names, Mapping):
        names = {name for name in weights_or_names if name.startswith(prefix)}
    else:
        names = {name for name in weights_or_names if name.startswith(prefix)}
    missing, extra = set(expected) - names, names - set(expected)
    if missing or extra:
        raise ValueError(
            "GLM5-Next converted vision parameter names changed: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if isinstance(weights_or_names, Mapping):
        for name, wanted in expected.items():
            actual = _shape(weights_or_names[name])
            if (
                not actual
                and isinstance(weights_or_names[name], tuple)
                and all(
                    isinstance(dimension, int) for dimension in weights_or_names[name]
                )
            ):
                actual = weights_or_names[name]
            # Index maps use shard-name strings rather than tensor headers.
            if actual and actual != wanted:
                raise ValueError(
                    f"Invalid converted GLM5-Next vision tensor {name}: "
                    f"found {actual}, expected {wanted}"
                )


def validate_vision_weight_layout(
    weights: Mapping[str, Any], *, prefix: str = VISION_PREFIX
) -> None:
    expected = _required_vision_shapes(prefix)
    for key, wanted in expected.items():
        if key not in weights:
            raise ValueError(
                f"GLM5-Next vision tower is missing checkpoint tensor: {key}"
            )
        actual = _shape(weights[key])
        if actual != wanted:
            raise ValueError(
                f"Invalid GLM5-Next vision tensor {key}: found {actual}, expected {wanted}"
            )
    unexpected = sorted(
        {key for key in weights if key.startswith(prefix)} - set(expected)
    )
    if unexpected:
        raise ValueError(
            f"Unexpected GLM5-Next vision checkpoint tensor: {unexpected[0]}"
        )
    marker = prefix + "blocks."
    block_ids = {
        int(key[len(marker) :].split(".", 1)[0])
        for key in weights
        if key.startswith(marker) and key[len(marker) :].split(".", 1)[0].isdigit()
    }
    if block_ids != set(range(24)):
        raise ValueError("GLM5-Next vision weights must contain exactly blocks 0..23")


def sanitize_vision_weights(weights: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and retain channels-first tensors; runtime contracts them directly."""
    sanitized = dict(weights)
    if any(key.startswith(VISION_PREFIX) for key in sanitized):
        validate_vision_weight_layout(sanitized)
    return sanitized


_IMAGE_KEYS = frozenset(("pixel_values", "image_grid_thw", "images", "image"))
_VIDEO_KEYS = frozenset(
    ("pixel_values_videos", "video_grid_thw", "videos", "video", "second_per_grid_ts")
)


def classify_media_inputs(
    inputs: Mapping[str, Any] | None = None, /, **kwargs: Any
) -> MediaKind:
    values = dict(inputs or {})
    values.update(kwargs)
    image = any(values.get(key) is not None for key in _IMAGE_KEYS)
    video = any(values.get(key) is not None for key in _VIDEO_KEYS)
    if image and video:
        return MediaKind.IMAGE_AND_VIDEO
    if image:
        return MediaKind.IMAGE
    if video:
        return MediaKind.VIDEO
    return MediaKind.TEXT


def _prepare_one(
    values: Mapping[str, Any], *, pixel_key: str, grid_key: str, label: str
) -> VisionMediaInput | None:
    pixels, grid = values.get(pixel_key), values.get(grid_key)
    if pixels is None and grid is None:
        return None
    if pixels is None or grid is None:
        raise Glm5NextVisionUnsupportedError(
            f"GLM5-Next {label} input requires both {pixel_key} and {grid_key}"
        )
    pixel_shape = _shape(pixels)
    if len(pixel_shape) != 2 or pixel_shape[1] != _PATCH_ELEMENTS:
        raise Glm5NextVisionUnsupportedError(
            f"GLM5-Next {pixel_key} must have processor shape (patches, {_PATCH_ELEMENTS}); found {pixel_shape}"
        )
    rows = _rows(grid, label=grid_key)
    patch_count = sum(t * height * width for t, height, width in rows)
    if pixel_shape[0] != patch_count:
        raise Glm5NextVisionUnsupportedError(
            f"GLM5-Next {pixel_key} patch rows ({pixel_shape[0]}) do not match {grid_key} ({patch_count})"
        )
    split_sizes = tuple(t * height * width // 4 for t, height, width in rows)
    encoder_rows = (
        tuple((1, height, width) for t, height, width in rows for _ in range(t))
        if label == "video"
        else rows
    )
    return VisionMediaInput(pixels, grid, encoder_rows, split_sizes)


def prepare_media_inputs(
    inputs: Mapping[str, Any] | None = None, /, **kwargs: Any
) -> PreparedVisionMedia:
    """Validate flattened ``(N,1176)`` outputs from the pinned processor."""
    values = dict(inputs or {})
    values.update(kwargs)
    kind = classify_media_inputs(values)
    if kind is MediaKind.TEXT:
        return PreparedVisionMedia(kind)
    aliases = [
        key
        for key in ("images", "image", "videos", "video")
        if values.get(key) is not None
    ]
    if aliases:
        raise Glm5NextVisionUnsupportedError(
            "GLM5-Next native vision accepts processor outputs, not raw media aliases: "
            + ", ".join(aliases)
        )
    image = _prepare_one(
        values, pixel_key="pixel_values", grid_key="image_grid_thw", label="image"
    )
    video = _prepare_one(
        values,
        pixel_key="pixel_values_videos",
        grid_key="video_grid_thw",
        label="video",
    )
    if values.get("second_per_grid_ts") is not None and video is None:
        raise Glm5NextVisionUnsupportedError(
            "GLM5-Next second_per_grid_ts requires pixel_values_videos and video_grid_thw"
        )
    actual_kind = (
        MediaKind.IMAGE_AND_VIDEO
        if image is not None and video is not None
        else MediaKind.IMAGE
        if image is not None
        else MediaKind.VIDEO
    )
    return PreparedVisionMedia(actual_kind, image=image, video=video)


def reject_unsupported_media(
    inputs: Mapping[str, Any] | None = None, /, **kwargs: Any
) -> None:
    """Compatibility guard: allow text and exact processor outputs only."""
    kind = classify_media_inputs(inputs, **kwargs)
    if kind is MediaKind.TEXT:
        return
    try:
        prepare_media_inputs(inputs, **kwargs)
    except Glm5NextVisionUnsupportedError as error:
        raise Glm5NextVisionUnsupportedError(
            f"GLM5-Next {kind.value} input was detected, but it is not an exact supported processor output: {error}"
        ) from error


def vision_position_ids(
    grid_thw: Any, spatial_merge_size: int = 2
) -> tuple[tuple[int, int], ...]:
    """Official block-major 2-D rotary positions, without importing MLX."""
    if spatial_merge_size != 2:
        raise Glm5NextVisionUnsupportedError(
            "GLM5-Next spatial_merge_size must remain 2"
        )
    output: list[tuple[int, int]] = []
    for t, height, width in _rows(grid_thw, label="grid_thw"):
        frame = [
            (bh * 2 + ih, bw * 2 + iw)
            for bh in range(height // 2)
            for bw in range(width // 2)
            for ih in range(2)
            for iw in range(2)
        ]
        output.extend(frame * t)
    return tuple(output)


def vision_cu_seqlens(grid_thw: Any) -> tuple[int, ...]:
    """Official packed-attention boundaries, one segment per frame."""
    cumulative = [0]
    for t, height, width in _rows(grid_thw, label="grid_thw"):
        for _ in range(t):
            cumulative.append(cumulative[-1] + height * width)
    return tuple(cumulative)


def _split_features(features: Any, sizes: Sequence[int]) -> tuple[Any, ...]:
    output, start = [], 0
    for size in sizes:
        output.append(features[start : start + size])
        start += size
    return tuple(output)


@lru_cache(maxsize=1)
def make_vision_component_classes() -> Mapping[str, type]:
    """Resolve small MLX components for tests and runtime composition."""
    import mlx.core as mx
    import mlx.nn as nn

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-5):
            super().__init__()
            self.weight, self.eps = mx.ones((dim,)), eps

        def __call__(self, x):
            dtype = x.dtype
            value = x.astype(mx.float32)
            value *= mx.rsqrt(
                mx.mean(mx.square(value), axis=-1, keepdims=True) + self.eps
            )
            return self.weight * value.astype(dtype)

    class RawPatchProjection(nn.Module):
        def __init__(self, embed_dim: int, in_channels: int, temporal: int, patch: int):
            super().__init__()
            self.weight = mx.zeros((embed_dim, in_channels, temporal, patch, patch))
            self.bias = mx.zeros((embed_dim,))

        def __call__(self, x):
            x = x.astype(self.weight.dtype)
            return x @ self.weight.reshape(self.weight.shape[0], -1).T + self.bias

    class PatchEmbed(nn.Module):
        def __init__(
            self, embed_dim=1024, in_channels=3, temporal_patch_size=2, patch_size=14
        ):
            super().__init__()
            self.patch_elements = (
                in_channels * temporal_patch_size * patch_size * patch_size
            )
            self.proj = RawPatchProjection(
                embed_dim, in_channels, temporal_patch_size, patch_size
            )

        def __call__(self, hidden_states):
            return self.proj(hidden_states.reshape(-1, self.patch_elements))

    class Attention(nn.Module):
        def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5):
            super().__init__()
            if hidden_size % num_heads:
                raise ValueError("vision hidden_size must be divisible by num_heads")
            self.dim, self.num_heads = hidden_size, num_heads
            self.head_dim, self.scaling = (
                hidden_size // num_heads,
                (hidden_size // num_heads) ** -0.5,
            )
            self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
            self.q_norm, self.k_norm = (
                RMSNorm(self.head_dim, eps),
                RMSNorm(self.head_dim, eps),
            )
            self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        @staticmethod
        def _rotate_half(x):
            half = x.shape[-1] // 2
            return mx.concatenate((-x[..., half:], x[..., :half]), axis=-1)

        def __call__(self, hidden_states, cu_seqlens, position_embeddings):
            length = hidden_states.shape[0]
            qkv = self.qkv(hidden_states).reshape(
                length, 3, self.num_heads, self.head_dim
            )
            query, key, value = [qkv[:, index] for index in range(3)]
            query, key = self.q_norm(query), self.k_norm(key)
            cos, sin = position_embeddings
            cos, sin = (
                cos[:, None, :].astype(mx.float32),
                sin[:, None, :].astype(mx.float32),
            )
            q_dtype, k_dtype = query.dtype, key.dtype
            qf, kf = query.astype(mx.float32), key.astype(mx.float32)
            query = (qf * cos + self._rotate_half(qf) * sin).astype(q_dtype)
            key = (kf * cos + self._rotate_half(kf) * sin).astype(k_dtype)
            boundaries, chunks = tuple(int(v) for v in cu_seqlens), []
            for start, stop in zip(boundaries, boundaries[1:]):
                q = query[start:stop].transpose(1, 0, 2)[None]
                k = key[start:stop].transpose(1, 0, 2)[None]
                v = value[start:stop].transpose(1, 0, 2)[None]
                chunk = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=self.scaling, mask=None
                )
                chunks.append(chunk[0].transpose(1, 0, 2))
            return self.proj(mx.concatenate(chunks, axis=0).reshape(length, self.dim))

    class MLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=True)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=True)

        def __call__(self, x):
            gate = mx.minimum(self.gate_proj(x), mx.array(10.0, dtype=x.dtype))
            up = mx.clip(self.up_proj(x), -10.0, 10.0)
            return self.down_proj(nn.silu(gate) * up)

    class Block(nn.Module):
        def __init__(self, hidden_size, intermediate_size, num_heads, eps):
            super().__init__()
            self.norm1, self.norm2 = (
                RMSNorm(hidden_size, eps),
                RMSNorm(hidden_size, eps),
            )
            self.attn, self.mlp = (
                Attention(hidden_size, num_heads, eps),
                MLP(hidden_size, intermediate_size),
            )

        def __call__(self, hidden_states, cu_seqlens, position_embeddings):
            hidden_states += self.attn(
                self.norm1(hidden_states), cu_seqlens, position_embeddings
            )
            return hidden_states + self.mlp(self.norm2(hidden_states))

    class Downsample(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, merge_size=2):
            super().__init__()
            self.merge_size = merge_size
            self.weight = mx.zeros((out_channels, in_channels, merge_size, merge_size))
            self.bias = mx.zeros((out_channels,))

        def __call__(self, hidden_states):
            blocks = hidden_states.reshape(
                -1, self.merge_size, self.merge_size, self.weight.shape[1]
            )
            kernel = self.weight.transpose(0, 2, 3, 1).reshape(self.weight.shape[0], -1)
            return blocks.reshape(blocks.shape[0], -1) @ kernel.T + self.bias

    class PatchMerger(nn.Module):
        def __init__(self, dim: int, intermediate_size: int):
            super().__init__()
            self.proj = nn.Linear(dim, dim, bias=False)
            self.post_projection_norm = nn.LayerNorm(dim, eps=1e-5)
            self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
            self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

        def __call__(self, hidden_states):
            hidden_states = nn.gelu(self.post_projection_norm(self.proj(hidden_states)))
            gate = mx.minimum(
                self.gate_proj(hidden_states), mx.array(10.0, dtype=hidden_states.dtype)
            )
            up = mx.clip(self.up_proj(hidden_states), -10.0, 10.0)
            return self.down_proj(nn.silu(gate) * up)

    return {
        "RMSNorm": RMSNorm,
        "PatchEmbed": PatchEmbed,
        "Attention": Attention,
        "MLP": MLP,
        "Block": Block,
        "Downsample": Downsample,
        "PatchMerger": PatchMerger,
    }


@lru_cache(maxsize=1)
def make_vision_model_class() -> type:
    """Return the exact 24-block MLX model class without instantiating it."""
    import mlx.core as mx
    import mlx.nn as nn

    c = make_vision_component_classes()

    class _Glm5NextVisionModel(nn.Module):
        def __init__(self, config: Any):
            super().__init__()
            validate_vision_config(config)
            self.config = _vision_config(config)
            self.spatial_merge_size, self.patch_size = 2, 14
            self.patch_embed = c["PatchEmbed"]()
            self.blocks = [c["Block"](1024, 4096, 16, 1e-5) for _ in range(24)]
            self.post_layernorm = c["RMSNorm"](1024, 1e-5)
            self.downsample = c["Downsample"](1024, 4096, 2)
            self.merger = c["PatchMerger"](4096, 10240)

        @property
        def dtype(self):
            return self.patch_embed.proj.weight.dtype

        def __call__(self, hidden_states, grid_thw):
            prepared = _prepare_one(
                {"pixel_values": hidden_states, "image_grid_thw": grid_thw},
                pixel_key="pixel_values",
                grid_key="image_grid_thw",
                label="image",
            )
            assert prepared is not None
            positions = vision_position_ids(prepared.encoder_grid_thw)
            cu_seqlens = vision_cu_seqlens(prepared.encoder_grid_thw)
            hidden_states = self.patch_embed(hidden_states)
            position_ids = mx.array(positions, dtype=mx.float32)
            # Transformers registers inv_freq as a non-persistent buffer. Keep
            # it local so MLX does not expose a spurious checkpoint parameter.
            inv_freq = 1.0 / (10_000.0 ** (mx.arange(0, 32, 2, dtype=mx.float32) / 32))
            rotary = (position_ids[..., None] * inv_freq).reshape(
                position_ids.shape[0], -1
            )
            embedding = mx.concatenate((rotary, rotary), axis=-1)
            position_embeddings = (mx.cos(embedding), mx.sin(embedding))
            for block in self.blocks:
                hidden_states = block(hidden_states, cu_seqlens, position_embeddings)
            hidden_states = self.post_layernorm(hidden_states)
            downsampled = self.downsample(hidden_states)
            return VisionModelOutput(downsampled, self.merger(downsampled))

        def encode_image(self, pixel_values, image_grid_thw):
            prepared = _prepare_one(
                {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
                pixel_key="pixel_values",
                grid_key="image_grid_thw",
                label="image",
            )
            assert prepared is not None
            return _split_features(
                self(pixel_values, image_grid_thw).pooler_output, prepared.split_sizes
            )

        def encode_video(self, pixel_values_videos, video_grid_thw):
            prepared = _prepare_one(
                {
                    "pixel_values_videos": pixel_values_videos,
                    "video_grid_thw": video_grid_thw,
                },
                pixel_key="pixel_values_videos",
                grid_key="video_grid_thw",
                label="video",
            )
            assert prepared is not None
            output = self(
                pixel_values_videos, mx.array(prepared.encoder_grid_thw, dtype=mx.int32)
            )
            return _split_features(output.pooler_output, prepared.split_sizes)

        @staticmethod
        def placeholder_masks(input_ids):
            special = input_ids == IMAGE_TOKEN_ID
            in_video = mx.cumsum(
                input_ids == VIDEO_START_TOKEN_ID, axis=-1
            ) > mx.cumsum(input_ids == VIDEO_END_TOKEN_ID, axis=-1)
            return (special & ~in_video)[..., None], (special & in_video)[..., None]

        @staticmethod
        def _inject_one(inputs_embeds, mask, features, label):
            flat, flat_mask = (
                inputs_embeds.reshape(-1, inputs_embeds.shape[-1]),
                mask.reshape(-1),
            )
            expected = int(mx.sum(flat_mask).item())
            if _shape(features) != (expected, flat.shape[-1]):
                raise Glm5NextVisionUnsupportedError(
                    f"GLM5-Next {label} features and placeholder tokens do not match: "
                    f"tokens={expected}, features={_shape(features)}"
                )
            if expected == 0:
                return inputs_embeds
            ranks = mx.clip(
                mx.cumsum(flat_mask.astype(mx.int32), axis=0) - 1, 0, expected - 1
            )
            flat = mx.where(flat_mask[:, None], features[ranks], flat)
            return flat.reshape(inputs_embeds.shape)

        @classmethod
        def inject_media_features(
            cls, inputs_embeds, input_ids, image_features=None, video_features=None
        ):
            image_mask, video_mask = cls.placeholder_masks(input_ids)
            if image_features is not None:
                inputs_embeds = cls._inject_one(
                    inputs_embeds, image_mask, image_features, "image"
                )
            if video_features is not None:
                inputs_embeds = cls._inject_one(
                    inputs_embeds, video_mask, video_features, "video"
                )
            return inputs_embeds

    _Glm5NextVisionModel.__name__ = _Glm5NextVisionModel.__qualname__ = (
        "Glm5NextVisionModel"
    )
    _Glm5NextVisionModel.__module__ = __name__
    return _Glm5NextVisionModel


class Glm5NextVisionModel:
    """Lazy constructor for the exact native MLX vision tower."""

    def __new__(cls, config: Any):
        return make_vision_model_class()(config)


def _affine_module_paths() -> frozenset[str]:
    block_modules = (
        "attn.qkv",
        "attn.proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    return frozenset(
        [f"blocks.{layer}.{module}" for layer in range(24) for module in block_modules]
        + [
            "merger.proj",
            "merger.gate_proj",
            "merger.up_proj",
            "merger.down_proj",
        ]
    )


def configure_vision_for_converted_weights(
    model: Any, quantization: Mapping[str, Any]
) -> Any:
    """Quantize an allocated vision skeleton so converted names bind exactly.

    The outer model calls this *after* constructing ``model.visual`` and before
    ``load_weights``. The function intentionally accepts only the vision model,
    not the whole language model.
    """

    bits, group_size = _validate_affine_quantization(quantization)
    import mlx.nn as nn

    paths = _affine_module_paths()

    def predicate(path: str, module: Any) -> bool:
        return path in paths and hasattr(module, "to_quantized")

    nn.quantize(
        model,
        bits=bits,
        group_size=group_size,
        mode="affine",
        class_predicate=predicate,
    )
    return model


def vision_runtime_factory(
    config: Any,
    *,
    quantization: Mapping[str, Any] | None = None,
    converted_weights: Mapping[str, Any] | Iterable[str] | None = None,
) -> VisionRuntimeIntegration:
    """Return the allocation-free contract consumed by an outer GLM5 model."""

    validate_vision_config(config)
    selected = (
        quantization if quantization is not None else _get(config, "quantization")
    )
    if not isinstance(selected, Mapping):
        raise ValueError("GLM5-Next converted vision requires a quantization object")
    bits, group_size = _validate_affine_quantization(selected)
    if converted_weights is not None:
        validate_converted_vision_weight_layout(
            converted_weights, bits=bits, group_size=group_size
        )
    frozen_quantization = {
        "bits": bits,
        "group_size": group_size,
        "mode": "affine",
    }
    from .processor import (
        install_glm5_next_processor_namespace,
        load_glm5_next_processor,
    )

    return VisionRuntimeIntegration(
        model_constructor=Glm5NextVisionModel,
        configure_for_converted_weights=partial(
            configure_vision_for_converted_weights,
            quantization=frozen_quantization,
        ),
        prepare_media=prepare_media_inputs,
        load_processor=load_glm5_next_processor,
        install_processor_namespace=install_glm5_next_processor_namespace,
        processor_class=OFFICIAL_PROCESSOR,
        processor_revision=TRANSFORMERS_REVISION,
        processor_required_outputs=OFFICIAL_PROCESSOR_REQUIRED_OUTPUTS,
        image_processor_outputs=OFFICIAL_IMAGE_PROCESSOR_OUTPUTS,
        video_processor_outputs=OFFICIAL_VIDEO_PROCESSOR_OUTPUTS,
    )


def vision_runtime_gaps() -> list[str]:
    """Return structural runtime gaps; empty only for the exact pinned ABI."""

    return []


__all__ = [
    "GLM5_NEXT_VISION_RUNTIME_READY",
    "Glm5NextVisionModel",
    "Glm5NextVisionUnsupportedError",
    "IMAGE_TOKEN_ID",
    "MediaKind",
    "OFFICIAL_IMAGE_PROCESSOR_OUTPUTS",
    "OFFICIAL_PROCESSOR",
    "OFFICIAL_PROCESSOR_REQUIRED_OUTPUTS",
    "OFFICIAL_VIDEO_PROCESSOR_OUTPUTS",
    "PreparedVisionMedia",
    "TRANSFORMERS_REVISION",
    "TRANSFORMERS_SOURCE",
    "VIDEO_END_TOKEN_ID",
    "VIDEO_START_TOKEN_ID",
    "VISION_PREFIX",
    "VisionMediaInput",
    "VisionModelOutput",
    "VisionRuntimeIntegration",
    "classify_media_inputs",
    "configure_vision_for_converted_weights",
    "converted_vision_parameter_shapes",
    "make_vision_component_classes",
    "make_vision_model_class",
    "prepare_media_inputs",
    "reject_unsupported_media",
    "sanitize_vision_weights",
    "validate_converted_vision_weight_layout",
    "validate_vision_config",
    "validate_vision_weight_layout",
    "vision_cu_seqlens",
    "vision_position_ids",
    "vision_runtime_factory",
    "vision_runtime_gaps",
]
