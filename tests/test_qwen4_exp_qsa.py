from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from omlx.patches.qwen4_exp.qsa import (
    Qwen4ExpQSABackendUnavailableError,
    Qwen4ExpQSAContract,
    Qwen4ExpQSAContractError,
    Qwen4ExpQSAExecutor,
    Qwen4ExpQSAInputError,
    Qwen4ExpQSARequest,
    Qwen4ExpQSAWeightError,
    validate_qsa_weights,
)


@dataclass(frozen=True)
class ShapeOnly:
    shape: tuple[int, ...]


def _official_text_config() -> dict[str, object]:
    return {
        "model_type": "qwen4_exp_text",
        "hidden_size": 2560,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"partial_rotary_factor": 0.25},
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "num_hidden_layers": 48,
        "full_attention_interval": 4,
        "attention_bias": False,
        "output_gate_type": "sigmoid",
        "layer_types": [
            "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
            for layer in range(48)
        ],
    }


def _official_root_config() -> dict[str, object]:
    return {"model_type": "qwen4_exp", "text_config": _official_text_config()}


def _weights(prefix: str = "") -> dict[str, ShapeOnly]:
    base = prefix.rstrip(".")

    def key(suffix: str) -> str:
        return f"{base}.{suffix}" if base else suffix

    return {
        key("q_proj.weight"): ShapeOnly((12288, 2560)),
        key("k_proj.weight"): ShapeOnly((512, 2560)),
        key("v_proj.weight"): ShapeOnly((512, 2560)),
        key("o_proj.weight"): ShapeOnly((2560, 6144)),
        key("q_norm.weight"): ShapeOnly((256,)),
        key("k_norm.weight"): ShapeOnly((256,)),
        key("indexer.index_qk_proj.weight"): ShapeOnly((640, 2560)),
        key("indexer.q_layernorm.weight"): ShapeOnly((128,)),
        key("indexer.k_layernorm.weight"): ShapeOnly((128,)),
    }


def _request(*, batch: int = 2, query_tokens: int = 3, key_tokens: int = 11):
    return Qwen4ExpQSARequest(
        queries=ShapeOnly((batch, 24, query_tokens, 256)),
        keys=ShapeOnly((batch, 2, key_tokens, 256)),
        values=ShapeOnly((batch, 2, key_tokens, 256)),
        index_queries=ShapeOnly((batch, query_tokens, 4, 128)),
        index_keys=ShapeOnly((batch, key_tokens, 128)),
        position_cos=ShapeOnly((batch, key_tokens, 64)),
        position_sin=ShapeOnly((batch, key_tokens, 64)),
        attention_mask=ShapeOnly((batch, 1, query_tokens, key_tokens)),
    )


def test_official_contract_binds_root_config_and_exact_qsa_layers():
    contract = Qwen4ExpQSAContract.from_config(_official_root_config())

    assert contract.num_query_heads == 24
    assert contract.num_key_value_heads == 2
    assert contract.head_dim == 256
    assert contract.rotary_dim == 64
    assert contract.indexer_query_heads == 4
    assert contract.indexer_key_heads == 1
    assert contract.indexer_head_dim == 128
    assert contract.qsa_layer_indices == tuple(range(3, 48, 4))


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("num_attention_heads", 16),
        ("num_key_value_heads", 4),
        ("head_dim", 128),
        ("partial_rotary_factor", 0.5),
        ("indexer_n_heads", 8),
        ("indexer_kv_heads", 2),
        ("indexer_head_dim", 64),
        ("indexer_budget", 1024),
        ("indexer_compress_ratio", 8),
        ("attention_bias", True),
        ("attention_bias", 0),
        ("output_gate_type", "silu"),
    ],
)
def test_config_geometry_is_strict_and_does_not_alias_other_attention(field, wrong):
    config = _official_text_config()
    config[field] = wrong

    with pytest.raises(Qwen4ExpQSAContractError, match=field):
        Qwen4ExpQSAContract.from_config(config)


def test_layer_layout_must_remain_linear_linear_linear_qsa():
    config = _official_text_config()
    config["layer_types"] = ["linear_attention"] * 48

    with pytest.raises(Qwen4ExpQSAContractError, match=r"layer_types\[3\]"):
        Qwen4ExpQSAContract.from_config(config)


def test_block_budget_is_512_and_preserves_only_the_partial_tail():
    contract = Qwen4ExpQSAContract.from_config(_official_text_config())

    assert contract.block_budget == 512
    assert contract.max_selected_tokens_with_tail == 2051
    assert contract.selection_plan(2048).selected_blocks == 512
    assert contract.selection_plan(2048).selected_token_capacity == 2048
    with_tail = contract.selection_plan(2051)
    assert with_tail.complete_blocks == 512
    assert with_tail.tail_tokens == 3
    assert with_tail.selected_token_capacity == 2051
    over_budget = contract.selection_plan(2052)
    assert over_budget.complete_blocks == 513
    assert over_budget.selected_blocks == 512
    assert over_budget.tail_tokens == 0
    assert over_budget.selected_token_capacity == 2048


def test_weight_validation_pins_fused_projection_and_norm_shapes():
    prefix = "model.language_model.layers.3.self_attn"
    validate_qsa_weights(_weights(prefix), prefix=prefix)

    wrong = _weights(prefix)
    wrong[f"{prefix}.indexer.index_qk_proj.weight"] = ShapeOnly((512, 2560))
    with pytest.raises(Qwen4ExpQSAWeightError, match="index_qk_proj"):
        validate_qsa_weights(wrong, prefix=prefix)

    split = _weights(prefix)
    split[f"{prefix}.indexer.q_proj.weight"] = ShapeOnly((512, 2560))
    with pytest.raises(Qwen4ExpQSAWeightError, match="alternate split tensor"):
        validate_qsa_weights(split, prefix=prefix)


def test_weight_validation_rejects_missing_norm_and_attention_bias():
    missing = _weights()
    del missing["q_norm.weight"]
    with pytest.raises(Qwen4ExpQSAWeightError, match="q_norm.weight"):
        validate_qsa_weights(missing)

    biased = _weights()
    biased["q_proj.bias"] = ShapeOnly((12288,))
    with pytest.raises(Qwen4ExpQSAWeightError, match="attention_bias=false"):
        validate_qsa_weights(biased)


def test_generation_refuses_by_default_without_a_true_sparse_backend():
    executor = Qwen4ExpQSAExecutor(_official_text_config())

    with pytest.raises(
        Qwen4ExpQSABackendUnavailableError,
        match="no true four-token micro-block sparse MLX backend",
    ):
        executor(_request())


def test_request_shape_validation_happens_before_backend_dispatch():
    class MustNotRun:
        name = "must-not-run"

        def supports(self, _contract):
            raise AssertionError("invalid input reached backend capability check")

        def execute(self, _request, *, contract):
            raise AssertionError("invalid input reached backend")

    request = replace(_request(), index_keys=ShapeOnly((2, 11, 4, 128)))
    executor = Qwen4ExpQSAExecutor(_official_text_config(), backend=MustNotRun())

    with pytest.raises(Qwen4ExpQSAInputError, match="index_keys"):
        executor(request)


def test_injected_micro_block_backend_receives_validated_contract_and_request():
    sentinel = object()

    class RecordingBackend:
        name = "test-native-qsa"

        def __init__(self):
            self.supported_contract = None
            self.execution = None

        def supports(self, contract):
            self.supported_contract = contract
            return contract.block_budget == 512

        def execute(self, request, *, contract):
            self.execution = (request, contract)
            return sentinel

    backend = RecordingBackend()
    request = _request()
    executor = Qwen4ExpQSAExecutor(_official_root_config(), backend=backend)

    assert executor(request) is sentinel
    assert backend.supported_contract is executor.contract
    assert backend.execution == (request, executor.contract)


def test_backend_that_cannot_prove_exact_support_fails_closed():
    class WrongBackend:
        name = "token-dsa"

        def supports(self, _contract):
            return False

        def execute(self, _request, *, contract):
            raise AssertionError("unsupported backend was dispatched")

    executor = Qwen4ExpQSAExecutor(_official_text_config(), backend=WrongBackend())

    with pytest.raises(Qwen4ExpQSABackendUnavailableError, match="does not support"):
        executor(_request())
