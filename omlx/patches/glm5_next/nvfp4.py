# SPDX-License-Identifier: Apache-2.0
"""Exact two-level ModelOpt NVFP4 support for GLM-5.3-Flash.

The numerical definition follows NVIDIA Model Optimizer's Apache-2.0
``NVFP4QTensor`` implementation (E2M1 values, per-16 E4M3FN scale, and a
per-tensor FP32 scale):

https://github.com/NVIDIA/Model-Optimizer/blob/main/modelopt/torch/quantization/qtensor/nvfp4_tensor.py

Model Optimizer is Apache-2.0 licensed; its repository license is at:

https://github.com/NVIDIA/Model-Optimizer/blob/main/LICENSE

This is an independent NumPy implementation, not a vendored source copy.  It
keeps ModelOpt's second-level FP32 scale as a separate parameter.  MLX's
native ``mode="nvfp4"`` kernels consume the packed E2M1 codes and E4M3 group
scales; the small wrappers below apply the retained FP32 scale after the
native matmul.  Folding the outer scale into E4M3 would re-round every group
scale and would not be the exact ModelOpt representation.

The module has no MLX import or allocation at import time.  Runtime classes
are created lazily when a converted model is instantiated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

import numpy as np

NVFP4_GROUP_SIZE: Final = 16
NVFP4_BITS: Final = 4
NVFP4_E2M1_MAX: Final = 6.0
NVFP4_E4M3_MAX: Final = 448.0
NVFP4_E4M3_MIN_NORMALIZED_SCALE: Final = 2.0**-9
NVFP4_LAYOUT: Final = "glm5-next-modelopt-nvfp4-v1"

# The ABI was checked against MLX QuantizedLinear/QuantizedSwitchLinear:
# uint32 E2M1 carriers, uint8 E4M3 scales, group_size=16, bits=4,
# mode="nvfp4".  The only ModelOpt field MLX does not consume is scale_2,
# which these wrappers apply explicitly.
GLM5_NEXT_NVFP4_RUNTIME_READY: Final = True

_E2M1_VALUES: Final = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32
)
_E2M1_BOUNDS: Final = np.array(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32
)
_E2M1_ODD_TIE_BOUNDS: Final = np.array([0.75, 1.75, 3.5], dtype=np.float32)


class Glm5NextNVFP4Error(ValueError):
    """The requested tensor/configuration cannot use the exact NVFP4 ABI."""


@dataclass(frozen=True, slots=True)
class NVFP4Tensor:
    """Serialized ModelOpt NVFP4 fields in MLX-compatible carrier shapes."""

    weight: np.ndarray
    scales: np.ndarray
    global_scale: np.ndarray
    logical_shape: tuple[int, ...]

    @property
    def serialized_bytes(self) -> int:
        return int(self.weight.nbytes + self.scales.nbytes + self.global_scale.nbytes)


@dataclass(frozen=True, slots=True)
class NVFP4AdapterResult:
    """Summary returned by :func:`configure_glm5_next_nvfp4`."""

    configured: bool
    module_count: int


def _require_float_matrix(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim < 2:
        raise Glm5NextNVFP4Error("NVFP4 weights must have rank >= 2")
    if array.shape[-1] % NVFP4_GROUP_SIZE:
        raise Glm5NextNVFP4Error(
            f"NVFP4 input width must be divisible by {NVFP4_GROUP_SIZE}: "
            f"shape={array.shape}"
        )
    if array.shape[-1] % 8:
        raise Glm5NextNVFP4Error("NVFP4 uint32 packing requires a width divisible by 8")
    if not np.issubdtype(array.dtype, np.floating):
        raise Glm5NextNVFP4Error(f"NVFP4 source must be floating, found {array.dtype}")
    converted = np.asarray(array, dtype=np.float32)
    if not np.all(np.isfinite(converted)):
        raise Glm5NextNVFP4Error("NVFP4 source contains NaN or infinity")
    return converted


def _encode_e4m3fn(value: np.ndarray) -> np.ndarray:
    """Clamp and round positive scales exactly through E4M3FN storage."""

    try:
        import ml_dtypes
    except ImportError as exc:  # pragma: no cover - required converter dep
        raise RuntimeError("ml_dtypes is required for E4M3FN NVFP4 scales") from exc
    clipped = np.clip(
        np.asarray(value, dtype=np.float32),
        NVFP4_E4M3_MIN_NORMALIZED_SCALE,
        NVFP4_E4M3_MAX,
    )
    return clipped.astype(ml_dtypes.float8_e4m3fn).view(np.uint8)


def decode_e4m3fn(codes: np.ndarray) -> np.ndarray:
    """Decode an E4M3FN uint8 carrier to float32."""

    if np.asarray(codes).dtype != np.uint8:
        raise Glm5NextNVFP4Error("E4M3FN scale carriers must be uint8")
    try:
        import ml_dtypes
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("ml_dtypes is required for E4M3FN NVFP4 scales") from exc
    return np.asarray(codes).view(ml_dtypes.float8_e4m3fn).astype(np.float32)


def encode_e2m1(values: np.ndarray) -> np.ndarray:
    """Return ModelOpt E2M1 nibbles, including its ties-to-even rule.

    ``searchsorted(..., side="left")`` selects the lower value at a midpoint.
    At the three midpoints whose lower code is odd, ModelOpt increments the
    code so the even endpoint wins.  The remaining midpoints already select
    an even lower code.
    """

    array = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise Glm5NextNVFP4Error("E2M1 input contains NaN or infinity")
    magnitude = np.abs(array)
    ordinal = np.searchsorted(_E2M1_BOUNDS, magnitude, side="left").astype(np.uint8)
    odd_tie = np.any(magnitude[..., None] == _E2M1_ODD_TIE_BOUNDS, axis=-1)
    ordinal = ordinal + odd_tie.astype(np.uint8)
    sign = (array < 0).astype(np.uint8) << np.uint8(3)
    return sign | ordinal


def decode_e2m1(codes: np.ndarray) -> np.ndarray:
    """Decode unpacked ModelOpt E2M1 nibbles to float32."""

    nibble = np.asarray(codes, dtype=np.uint8)
    magnitude = _E2M1_VALUES[nibble & np.uint8(7)]
    return np.where((nibble & np.uint8(8)) != 0, -magnitude, magnitude)


def pack_e2m1(nibbles: np.ndarray) -> np.ndarray:
    """Pack adjacent ModelOpt nibbles and reinterpret them as MLX uint32."""

    codes = np.asarray(nibbles, dtype=np.uint8)
    if codes.shape[-1] % 8:
        raise Glm5NextNVFP4Error("E2M1 packing width must be divisible by 8")
    packed_u8 = (codes[..., 1::2] << np.uint8(4)) | codes[..., 0::2]
    return np.ascontiguousarray(packed_u8).view(np.dtype("<u4"))


def unpack_e2m1(weight: np.ndarray, logical_width: int | None = None) -> np.ndarray:
    """Unpack an MLX uint32 carrier into ModelOpt nibbles."""

    packed = np.asarray(weight)
    if packed.dtype != np.dtype("<u4") and packed.dtype != np.uint32:
        raise Glm5NextNVFP4Error(
            f"packed NVFP4 weights must be uint32, found {packed.dtype}"
        )
    bytes_ = np.ascontiguousarray(packed).view(np.uint8)
    shape = (*packed.shape[:-1], packed.shape[-1] * 8)
    output = np.empty(shape, dtype=np.uint8)
    output[..., 0::2] = bytes_ & np.uint8(0x0F)
    output[..., 1::2] = bytes_ >> np.uint8(4)
    if logical_width is not None:
        if logical_width < 0 or logical_width > output.shape[-1]:
            raise Glm5NextNVFP4Error("invalid logical NVFP4 width")
        output = output[..., :logical_width]
    return output


def quantize_modelopt_nvfp4(value: np.ndarray) -> NVFP4Tensor:
    """Quantize with ModelOpt's dynamic/max NVFP4 weight recipe.

    The returned fields reconstruct as ``E2M1 * E4M3(group) * FP32(tensor)``.
    Zero tensors use a global scale of one and E4M3 group scales of one; this
    deterministic extension avoids ModelOpt's otherwise undefined 0/0 global
    scale while reconstructing exactly zero.
    """

    array = _require_float_matrix(value)
    global_amax = float(np.max(np.abs(array), initial=0.0))
    global_scale = np.float32(
        global_amax / (NVFP4_E2M1_MAX * NVFP4_E4M3_MAX) if global_amax else 1.0
    )
    blocked = array.reshape(*array.shape[:-1], -1, NVFP4_GROUP_SIZE)
    block_amax = np.max(np.abs(blocked), axis=-1)
    raw_group_scale = block_amax / (np.float32(NVFP4_E2M1_MAX) * global_scale)
    raw_group_scale = np.where(block_amax == 0, np.float32(1.0), raw_group_scale)
    scale_codes = _encode_e4m3fn(raw_group_scale)
    decoded_scales = decode_e4m3fn(scale_codes)
    divisor = decoded_scales[..., None] * global_scale
    scaled = blocked / divisor
    nibbles = encode_e2m1(scaled.reshape(array.shape))
    packed = pack_e2m1(nibbles)
    return NVFP4Tensor(
        weight=packed,
        scales=np.ascontiguousarray(scale_codes),
        global_scale=np.asarray(global_scale, dtype=np.float32),
        logical_shape=tuple(array.shape),
    )


def dequantize_modelopt_nvfp4(value: NVFP4Tensor) -> np.ndarray:
    """Independent-friendly dequantization of the serialized representation."""

    if value.logical_shape[-1] % NVFP4_GROUP_SIZE:
        raise Glm5NextNVFP4Error("invalid logical shape in NVFP4 tensor")
    nibbles = unpack_e2m1(value.weight, value.logical_shape[-1])
    decoded = decode_e2m1(nibbles).reshape(
        *value.logical_shape[:-1], -1, NVFP4_GROUP_SIZE
    )
    expected_scales = (*value.logical_shape[:-1], value.logical_shape[-1] // 16)
    if tuple(value.scales.shape) != expected_scales:
        raise Glm5NextNVFP4Error(
            f"invalid NVFP4 scale shape: {value.scales.shape} != {expected_scales}"
        )
    result = decoded * decode_e4m3fn(value.scales)[..., None]
    result = result * np.asarray(value.global_scale, dtype=np.float32)
    return result.reshape(value.logical_shape)


def is_glm5_next_nvfp4_config(config: Any) -> bool:
    """Affirmatively recognize only this converter's exact runtime contract."""

    if not isinstance(config, Mapping):
        return False
    common = (
        config.get("bits") == NVFP4_BITS
        and config.get("group_size") == NVFP4_GROUP_SIZE
        and config.get("mode") == "nvfp4"
        and config.get("layout") == NVFP4_LAYOUT
        and config.get("modelopt_global_scale") is True
    )
    scope = config.get("scope")
    return common and (
        scope == "glm5_next_mlp"
        or (
            scope == "glm5_next_routed_experts"
            and config.get("source_layout") == "modelopt-0.45-per-expert"
        )
    )


def require_glm5_next_nvfp4_config(config: Any) -> None:
    if not is_glm5_next_nvfp4_config(config):
        raise Glm5NextNVFP4Error(
            "unsupported GLM5-Next NVFP4 configuration; exact layout, group-16, "
            "4-bit, scope, and ModelOpt global-scale markers are required"
        )


@lru_cache(maxsize=1)
def _runtime_classes():
    import mlx.core as mx
    import mlx.nn as nn

    class ScaledNVFP4Linear(nn.Module):
        """Native MLX NVFP4 matmul followed by ModelOpt's FP32 scale_2."""

        def __init__(self, input_dims: int, output_dims: int, *, bias: bool = False):
            super().__init__()
            if input_dims % NVFP4_GROUP_SIZE:
                raise Glm5NextNVFP4Error("NVFP4 Linear width is not group-16 aligned")
            self.weight = mx.zeros((output_dims, input_dims // 8), dtype=mx.uint32)
            self.scales = mx.zeros(
                (output_dims, input_dims // NVFP4_GROUP_SIZE), dtype=mx.uint8
            )
            self.global_scale = mx.ones((), dtype=mx.float32)
            self.group_size = NVFP4_GROUP_SIZE
            self.bits = NVFP4_BITS
            self.mode = "nvfp4"
            if bias:
                self.bias = mx.zeros((output_dims,))
            self.freeze()

        @property
        def input_dims(self):
            return self.scales.shape[-1] * NVFP4_GROUP_SIZE

        @property
        def output_dims(self):
            return self.weight.shape[-2]

        def __call__(self, x):
            output = mx.quantized_matmul(
                x,
                self["weight"],
                self["scales"],
                transpose=True,
                group_size=NVFP4_GROUP_SIZE,
                bits=NVFP4_BITS,
                mode="nvfp4",
            )
            output = output * self["global_scale"].astype(output.dtype)
            if "bias" in self:
                output = output + self["bias"]
            return output

    class ScaledNVFP4SwitchLinear(nn.Module):
        """Native gather-QMM with one retained ModelOpt scale per expert."""

        def __init__(
            self,
            input_dims: int,
            output_dims: int,
            num_experts: int,
            *,
            bias: bool = False,
        ):
            super().__init__()
            if input_dims % NVFP4_GROUP_SIZE:
                raise Glm5NextNVFP4Error("NVFP4 SwitchLinear width is not aligned")
            self.weight = mx.zeros(
                (num_experts, output_dims, input_dims // 8), dtype=mx.uint32
            )
            self.scales = mx.zeros(
                (num_experts, output_dims, input_dims // NVFP4_GROUP_SIZE),
                dtype=mx.uint8,
            )
            self.global_scale = mx.ones((num_experts,), dtype=mx.float32)
            self.group_size = NVFP4_GROUP_SIZE
            self.bits = NVFP4_BITS
            self.mode = "nvfp4"
            if bias:
                self.bias = mx.zeros((num_experts, output_dims))
            self.freeze()

        @property
        def input_dims(self):
            return self.scales.shape[-1] * NVFP4_GROUP_SIZE

        @property
        def output_dims(self):
            return self.weight.shape[-2]

        @property
        def num_experts(self):
            return self.weight.shape[0]

        def __call__(self, x, indices, sorted_indices=False):
            output = mx.gather_qmm(
                x,
                self["weight"],
                self["scales"],
                rhs_indices=indices,
                transpose=True,
                group_size=NVFP4_GROUP_SIZE,
                bits=NVFP4_BITS,
                mode="nvfp4",
                sorted_indices=sorted_indices,
            )
            selected = self["global_scale"][indices].astype(output.dtype)
            output = output * selected[..., None, None]
            if "bias" in self:
                output = output + mx.expand_dims(self["bias"][indices], -2)
            return output

    return ScaledNVFP4Linear, ScaledNVFP4SwitchLinear


def make_scaled_nvfp4_linear(input_dims: int, output_dims: int, *, bias: bool = False):
    """Construct the exact dense runtime carrier lazily."""

    return _runtime_classes()[0](input_dims, output_dims, bias=bias)


def make_scaled_nvfp4_switch_linear(
    input_dims: int,
    output_dims: int,
    num_experts: int,
    *,
    bias: bool = False,
):
    """Construct the exact routed-expert runtime carrier lazily."""

    return _runtime_classes()[1](input_dims, output_dims, num_experts, bias=bias)


def _replace_dense(module: Any, *, input_dims: int, output_dims: int) -> Any:
    if type(module).__name__ == "ScaledNVFP4Linear":
        return module
    shape = tuple(getattr(getattr(module, "weight", None), "shape", ()))
    if shape != (output_dims, input_dims):
        raise Glm5NextNVFP4Error(
            f"dense module geometry changed: found {shape}, expected "
            f"{(output_dims, input_dims)}"
        )
    return make_scaled_nvfp4_linear(
        input_dims,
        output_dims,
        bias=getattr(module, "get", lambda *_: None)("bias") is not None,
    )


def _replace_switch(
    module: Any, *, input_dims: int, output_dims: int, experts: int
) -> Any:
    if type(module).__name__ == "ScaledNVFP4SwitchLinear":
        return module
    shape = tuple(getattr(getattr(module, "weight", None), "shape", ()))
    expected = (experts, output_dims, input_dims)
    if shape != expected:
        raise Glm5NextNVFP4Error(
            f"switch module geometry changed: found {shape}, expected {expected}"
        )
    return make_scaled_nvfp4_switch_linear(
        input_dims,
        output_dims,
        experts,
        bias=getattr(module, "get", lambda *_: None)("bias") is not None,
    )


def configure_glm5_next_nvfp4(
    model: Any, quantization: Mapping[str, Any] | None
) -> NVFP4AdapterResult:
    """Bind converted GLM-5.3 MLP modules before ``load_weights``.

    Only the three dense MLPs, 42 routed switch banks, and their 42 shared
    experts are replaced.  Attention/KDA/DSA, mHC, routers, norms, convolutions,
    embeddings, LM head, vision, and MTP remain untouched.  Official geometry
    is required; an unexpected module shape fails closed.
    """

    if quantization is None:
        return NVFP4AdapterResult(False, 0)
    require_glm5_next_nvfp4_config(quantization)
    routed_only = quantization.get("scope") == "glm5_next_routed_experts"
    text = getattr(model, "language_model", model)
    backbone = getattr(text, "model", None)
    layers = getattr(backbone, "layers", None)
    if not isinstance(layers, list) or len(layers) != 45:
        raise Glm5NextNVFP4Error("GLM5-Next NVFP4 requires exactly 45 main layers")

    count = 0
    for index, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is None:
            raise Glm5NextNVFP4Error(f"layer {index} has no MLP")
        if index < 3:
            if routed_only:
                continue
            for projection, dims in (
                ("gate_proj", (4096, 12288)),
                ("up_proj", (4096, 12288)),
                ("down_proj", (12288, 4096)),
            ):
                current = getattr(mlp, projection)
                replacement = _replace_dense(
                    current, input_dims=dims[0], output_dims=dims[1]
                )
                setattr(mlp, projection, replacement)
                count += int(replacement is not current)
            continue

        switch = getattr(mlp, "switch_mlp", None)
        shared = getattr(mlp, "shared_experts", None)
        if switch is None or shared is None:
            raise Glm5NextNVFP4Error(f"layer {index} sparse MLP ABI changed")
        for projection, dims in (
            ("gate_proj", (4096, 2048)),
            ("up_proj", (4096, 2048)),
            ("down_proj", (2048, 4096)),
        ):
            current = getattr(switch, projection)
            replacement = _replace_switch(
                current, input_dims=dims[0], output_dims=dims[1], experts=288
            )
            setattr(switch, projection, replacement)
            count += int(replacement is not current)

            if not routed_only:
                current_shared = getattr(shared, projection)
                replacement_shared = _replace_dense(
                    current_shared, input_dims=dims[0], output_dims=dims[1]
                )
                setattr(shared, projection, replacement_shared)
                count += int(replacement_shared is not current_shared)

    if routed_only:
        heads = getattr(text, "mtp", None)
        if not isinstance(heads, list) or len(heads) != 1:
            raise Glm5NextNVFP4Error("routed-only NVFP4 requires one MTP head")
        switch = getattr(getattr(heads[0], "block", None), "mlp", None)
        switch = getattr(switch, "switch_mlp", None)
        if switch is None:
            raise Glm5NextNVFP4Error("routed-only NVFP4 MTP SwitchGLU is missing")
        for projection, dims in (
            ("gate_proj", (4096, 2048)),
            ("up_proj", (4096, 2048)),
            ("down_proj", (2048, 4096)),
        ):
            current = getattr(switch, projection)
            replacement = _replace_switch(
                current, input_dims=dims[0], output_dims=dims[1], experts=288
            )
            setattr(switch, projection, replacement)
            count += int(replacement is not current)
    return NVFP4AdapterResult(True, count)


__all__ = [
    "GLM5_NEXT_NVFP4_RUNTIME_READY",
    "Glm5NextNVFP4Error",
    "NVFP4AdapterResult",
    "NVFP4Tensor",
    "NVFP4_BITS",
    "NVFP4_GROUP_SIZE",
    "NVFP4_LAYOUT",
    "configure_glm5_next_nvfp4",
    "decode_e2m1",
    "decode_e4m3fn",
    "dequantize_modelopt_nvfp4",
    "encode_e2m1",
    "is_glm5_next_nvfp4_config",
    "make_scaled_nvfp4_linear",
    "make_scaled_nvfp4_switch_linear",
    "pack_e2m1",
    "quantize_modelopt_nvfp4",
    "require_glm5_next_nvfp4_config",
    "unpack_e2m1",
]
