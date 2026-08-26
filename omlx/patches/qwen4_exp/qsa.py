# SPDX-License-Identifier: Apache-2.0
"""Fail-closed execution boundary for Qwen3.8-Flash-Next QSA.

Qwen sparse attention (QSA) selects *four-token micro-blocks*.  It is not
DeepSeek token-level sparse attention and it cannot be represented by a dense
attention call plus a large materialized mask without defeating the memory and
latency contract of the model.  This module therefore contains no fallback
attention implementation.  It validates the published Qwen4-Exp geometry and
dispatches only to an explicitly installed micro-block sparse backend.

The boundary is intentionally independent of MLX.  A future native MLX kernel
can implement :class:`Qwen4ExpQSASparseBackend` without changing model loading,
weight validation, or the fail-closed behavior exercised here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class Qwen4ExpQSAContractError(ValueError):
    """The model config does not describe the published Qwen4-Exp QSA."""


class Qwen4ExpQSAWeightError(ValueError):
    """The QSA checkpoint weights do not match the published layout."""


class Qwen4ExpQSAInputError(ValueError):
    """A projected QSA runtime tensor has an invalid shape."""


class Qwen4ExpQSABackendUnavailableError(RuntimeError):
    """No exact micro-block sparse QSA execution backend is available."""


_MISSING = object()


def _get(source: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(source, Mapping):
        value = source.get(name, _MISSING)
    else:
        value = getattr(source, name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise Qwen4ExpQSAContractError(
                f"Qwen4-Exp QSA config is missing required field {name!r}"
            )
        return default
    return value


def _require_exact(source: Any, name: str, expected: Any) -> None:
    value = _get(source, name)
    if isinstance(expected, bool):
        matches = type(value) is bool and value is expected
    elif isinstance(expected, int):
        matches = type(value) is int and value == expected
    elif isinstance(expected, float):
        matches = type(value) in (int, float) and float(value) == expected
    else:
        matches = value == expected
    if not matches:
        raise Qwen4ExpQSAContractError(
            f"Qwen4-Exp QSA requires {name}={expected!r}, got {value!r}"
        )


@dataclass(frozen=True)
class Qwen4ExpQSASelectionPlan:
    """Exact complete-block and visible-tail selection bounds for one row."""

    visible_tokens: int
    complete_blocks: int
    selected_blocks: int
    tail_tokens: int
    selected_token_capacity: int


@dataclass(frozen=True)
class Qwen4ExpQSAContract:
    """Published Qwen3.8-Flash-Next QSA geometry.

    This is deliberately a checkpoint fingerprint rather than a collection of
    permissive defaults.  Unknown Qwen4-Exp variants must establish and test a
    separate contract before they can reach generation.
    """

    model_type: str = "qwen4_exp_text"
    hidden_size: int = 2560
    num_query_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    indexer_query_heads: int = 4
    indexer_key_heads: int = 1
    indexer_head_dim: int = 128
    compress_ratio: int = 4
    token_budget: int = 2048
    num_hidden_layers: int = 48
    full_attention_interval: int = 4

    def __post_init__(self) -> None:
        expected = {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_query_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rotary_dim": 64,
            "indexer_query_heads": 4,
            "indexer_key_heads": 1,
            "indexer_head_dim": 128,
            "compress_ratio": 4,
            "token_budget": 2048,
            "num_hidden_layers": 48,
            "full_attention_interval": 4,
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise Qwen4ExpQSAContractError(
                    f"published Qwen4-Exp QSA requires {name}={value!r}, "
                    f"got {getattr(self, name)!r}"
                )

    @property
    def block_budget(self) -> int:
        """Maximum selected complete micro-blocks (2048 / 4 = 512)."""

        return self.token_budget // self.compress_ratio

    @property
    def max_selected_tokens_with_tail(self) -> int:
        """Maximum selected tokens including the always-visible partial tail."""

        return self.token_budget + self.compress_ratio - 1

    @property
    def qsa_layer_indices(self) -> tuple[int, ...]:
        """Zero-indexed QSA layers: 3, 7, ..., 47."""

        return tuple(
            range(
                self.full_attention_interval - 1,
                self.num_hidden_layers,
                self.full_attention_interval,
            )
        )

    def selection_plan(self, visible_tokens: int) -> Qwen4ExpQSASelectionPlan:
        """Return the exact block budget for one causally visible token row.

        Complete four-token blocks compete for 512 slots.  A final incomplete
        block is not scored and its zero-to-three visible tokens are retained.
        """

        if type(visible_tokens) is not int or visible_tokens < 0:
            raise Qwen4ExpQSAInputError("visible_tokens must be a non-negative integer")
        complete_blocks, tail_tokens = divmod(visible_tokens, self.compress_ratio)
        selected_blocks = min(complete_blocks, self.block_budget)
        return Qwen4ExpQSASelectionPlan(
            visible_tokens=visible_tokens,
            complete_blocks=complete_blocks,
            selected_blocks=selected_blocks,
            tail_tokens=tail_tokens,
            selected_token_capacity=(
                selected_blocks * self.compress_ratio + tail_tokens
            ),
        )

    @classmethod
    def from_config(cls, config: Any) -> Qwen4ExpQSAContract:
        """Validate and bind either the official root or text config."""

        text_config = _get(config, "text_config", None)
        if text_config is not None:
            _require_exact(config, "model_type", "qwen4_exp")
        else:
            text_config = config

        expected_fields = {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_head_dim": 128,
            "indexer_budget": 2048,
            "indexer_compress_ratio": 4,
            "num_hidden_layers": 48,
            "full_attention_interval": 4,
            "attention_bias": False,
            "output_gate_type": "sigmoid",
        }
        for name, expected in expected_fields.items():
            _require_exact(text_config, name, expected)

        partial_rotary_factor = _get(text_config, "partial_rotary_factor")
        if type(partial_rotary_factor) not in (int, float):
            raise Qwen4ExpQSAContractError("partial_rotary_factor must be numeric")
        rotary_dim = int(256 * float(partial_rotary_factor))
        if float(partial_rotary_factor) != 0.25 or rotary_dim != 64:
            raise Qwen4ExpQSAContractError(
                "Qwen4-Exp QSA requires partial_rotary_factor=0.25 "
                f"(64 rotary dimensions), got {partial_rotary_factor!r}"
            )

        rope_parameters = _get(text_config, "rope_parameters", None)
        if rope_parameters is not None:
            rope_factor = _get(
                rope_parameters, "partial_rotary_factor", partial_rotary_factor
            )
            if type(rope_factor) not in (int, float) or float(rope_factor) != 0.25:
                raise Qwen4ExpQSAContractError(
                    "rope_parameters.partial_rotary_factor must equal 0.25"
                )

        layer_types = _get(text_config, "layer_types")
        if not isinstance(layer_types, (list, tuple)) or len(layer_types) != 48:
            raise Qwen4ExpQSAContractError(
                "Qwen4-Exp QSA requires exactly 48 layer_types entries"
            )
        expected_qsa_layers = set(range(3, 48, 4))
        for layer_idx, layer_type in enumerate(layer_types):
            expected = (
                "full_attention"
                if layer_idx in expected_qsa_layers
                else "linear_attention"
            )
            # Transformers normalizes full_attention to
            # qwen_sparse_attention after config loading; accept that exact
            # semantic spelling only on the same twelve layers.
            accepted = {expected}
            if expected == "full_attention":
                accepted.add("qwen_sparse_attention")
            if layer_type not in accepted:
                raise Qwen4ExpQSAContractError(
                    f"layer_types[{layer_idx}] must be {sorted(accepted)!r}, "
                    f"got {layer_type!r}"
                )

        contract = cls()
        if contract.block_budget != 512:
            # Defensive assertion: generation must never continue if future
            # edits silently change the 2048-token / four-token-block meaning.
            raise Qwen4ExpQSAContractError("Qwen4-Exp QSA block budget must equal 512")
        return contract


_QSA_WEIGHT_SHAPES: dict[str, tuple[int, ...]] = {
    # q_proj is the official fused query + sigmoid-gate projection.
    "q_proj.weight": (24 * 256 * 2, 2560),
    "k_proj.weight": (2 * 256, 2560),
    "v_proj.weight": (2 * 256, 2560),
    "o_proj.weight": (2560, 24 * 256),
    "q_norm.weight": (256,),
    "k_norm.weight": (256,),
    # Index queries and the single index key are one fused projection.
    "indexer.index_qk_proj.weight": ((4 + 1) * 128, 2560),
    "indexer.q_layernorm.weight": (128,),
    "indexer.k_layernorm.weight": (128,),
}

_FORBIDDEN_SPLIT_WEIGHTS = (
    "gate_proj.weight",
    "indexer.q_proj.weight",
    "indexer.k_proj.weight",
    "indexer.wq.weight",
    "indexer.wk.weight",
)

_FORBIDDEN_BIASES = (
    "q_proj.bias",
    "k_proj.bias",
    "v_proj.bias",
    "o_proj.bias",
    "indexer.index_qk_proj.bias",
)


def _prefixed(prefix: str, suffix: str) -> str:
    normalized = prefix.rstrip(".")
    return f"{normalized}.{suffix}" if normalized else suffix


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", _MISSING)
    if shape is _MISSING and isinstance(value, Mapping):
        shape = value.get("shape", _MISSING)
    if shape is _MISSING and isinstance(value, (tuple, list)):
        shape = value
    if shape is _MISSING:
        return None
    try:
        result = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError):
        return None
    if any(dim < 0 for dim in result):
        return None
    return result


def validate_qsa_weights(weights: Mapping[str, Any], *, prefix: str = "") -> None:
    """Validate the canonical published BF16 QSA tensor layout by shape.

    Values may be tensors or shape-only objects.  This makes validation usable
    before allocating the 125B checkpoint.  Quantization must validate these
    logical shapes before replacing tensors with packed representations.
    """

    if not isinstance(weights, Mapping):
        raise Qwen4ExpQSAWeightError("QSA weights must be a mapping")

    for suffix, expected_shape in _QSA_WEIGHT_SHAPES.items():
        key = _prefixed(prefix, suffix)
        if key not in weights:
            raise Qwen4ExpQSAWeightError(
                f"Qwen4-Exp QSA is missing required tensor {key!r}"
            )
        actual_shape = _shape_of(weights[key])
        if actual_shape != expected_shape:
            raise Qwen4ExpQSAWeightError(
                f"Qwen4-Exp QSA tensor {key!r} must have logical shape "
                f"{expected_shape}, got {actual_shape}"
            )

    for suffix in _FORBIDDEN_SPLIT_WEIGHTS:
        key = _prefixed(prefix, suffix)
        if key in weights:
            raise Qwen4ExpQSAWeightError(
                f"Qwen4-Exp QSA forbids alternate split tensor {key!r}; "
                "the published checkpoint uses fused projections"
            )
    for suffix in _FORBIDDEN_BIASES:
        key = _prefixed(prefix, suffix)
        if key in weights and weights[key] is not None:
            raise Qwen4ExpQSAWeightError(
                f"Qwen4-Exp QSA attention_bias=false forbids tensor {key!r}"
            )


@dataclass(frozen=True)
class Qwen4ExpQSARequest:
    """Projected state passed to an exact sparse backend.

    Query/key tensors have already passed the published q/k RMSNorms but have
    not had partial RoPE applied.  ``index_queries`` has passed the indexer
    q-layernorm.  ``index_keys`` remains per-token because the backend must
    average each complete four-token block *before* applying the indexer
    k-layernorm and RoPE.  Keeping this sequence explicit prevents a backend
    from accidentally implementing token-level DSA semantics.
    """

    queries: Any
    keys: Any
    values: Any
    index_queries: Any
    index_keys: Any
    position_cos: Any
    position_sin: Any
    attention_mask: Any
    cache: Any = None

    def validate(self, contract: Qwen4ExpQSAContract) -> None:
        q_shape = _shape_of(self.queries)
        if q_shape is None or len(q_shape) != 4:
            raise Qwen4ExpQSAInputError(
                "queries must have shape (batch, 24, query_tokens, 256)"
            )
        batch, q_heads, query_tokens, head_dim = q_shape
        if (
            batch <= 0
            or query_tokens <= 0
            or q_heads != contract.num_query_heads
            or head_dim != contract.head_dim
        ):
            raise Qwen4ExpQSAInputError(
                f"queries must have shape (batch, 24, query_tokens, 256), got {q_shape}"
            )

        key_shape = _shape_of(self.keys)
        value_shape = _shape_of(self.values)
        if (
            key_shape is None
            or len(key_shape) != 4
            or key_shape[0] != batch
            or key_shape[1] != contract.num_key_value_heads
            or key_shape[2] <= 0
            or key_shape[3] != contract.head_dim
        ):
            raise Qwen4ExpQSAInputError(
                f"keys must have shape (batch, 2, key_tokens, 256), got {key_shape}"
            )
        if value_shape != key_shape:
            raise Qwen4ExpQSAInputError(
                f"values must match keys shape {key_shape}, got {value_shape}"
            )
        key_tokens = key_shape[2]

        expected_shapes = {
            "index_queries": (
                batch,
                query_tokens,
                contract.indexer_query_heads,
                contract.indexer_head_dim,
            ),
            # The one index-key head is squeezed in the official layout.
            "index_keys": (batch, key_tokens, contract.indexer_head_dim),
            "position_cos": (batch, key_tokens, contract.rotary_dim),
            "position_sin": (batch, key_tokens, contract.rotary_dim),
            "attention_mask": (batch, 1, query_tokens, key_tokens),
        }
        for name, expected in expected_shapes.items():
            actual = _shape_of(getattr(self, name))
            if actual != expected:
                raise Qwen4ExpQSAInputError(
                    f"{name} must have shape {expected}, got {actual}"
                )


@runtime_checkable
class Qwen4ExpQSASparseBackend(Protocol):
    """Protocol for a true Qwen4-Exp micro-block sparse MLX backend."""

    @property
    def name(self) -> str: ...

    def supports(self, contract: Qwen4ExpQSAContract) -> bool: ...

    def execute(
        self,
        request: Qwen4ExpQSARequest,
        *,
        contract: Qwen4ExpQSAContract,
    ) -> Any: ...


class Qwen4ExpQSAExecutor:
    """Validate QSA requests and dispatch to an installed sparse backend."""

    def __init__(
        self,
        config: Any,
        *,
        backend: Qwen4ExpQSASparseBackend | None = None,
    ) -> None:
        self.contract = Qwen4ExpQSAContract.from_config(config)
        self.backend = backend

    def __call__(self, request: Qwen4ExpQSARequest) -> Any:
        request.validate(self.contract)
        backend = self.backend
        if backend is None:
            raise Qwen4ExpQSABackendUnavailableError(
                "Qwen4-Exp QSA generation is disabled: no true four-token "
                "micro-block sparse MLX backend is installed. Dense SDPA and "
                "token-level DSA fallbacks are intentionally forbidden."
            )
        supports = getattr(backend, "supports", None)
        execute = getattr(backend, "execute", None)
        if not callable(supports) or not callable(execute):
            raise Qwen4ExpQSABackendUnavailableError(
                "Qwen4-Exp QSA backend does not implement the sparse backend protocol"
            )
        if not supports(self.contract):
            backend_name = getattr(backend, "name", type(backend).__name__)
            raise Qwen4ExpQSABackendUnavailableError(
                f"Qwen4-Exp QSA backend {backend_name!r} does not support the "
                "published four-token micro-block contract"
            )
        return execute(request, contract=self.contract)


__all__ = [
    "Qwen4ExpQSABackendUnavailableError",
    "Qwen4ExpQSAContract",
    "Qwen4ExpQSAContractError",
    "Qwen4ExpQSAExecutor",
    "Qwen4ExpQSAInputError",
    "Qwen4ExpQSARequest",
    "Qwen4ExpQSASelectionPlan",
    "Qwen4ExpQSASparseBackend",
    "Qwen4ExpQSAWeightError",
    "validate_qsa_weights",
]
