# SPDX-License-Identifier: Apache-2.0
"""Bounded contracts for the GLM-5.3 mlx-lm pipeline adapter."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import mlx.core as mx


def _adapter_model():
    from omlx.patches.mlx_lm_glm5_next import apply_mlx_lm_glm5_next_patch
    from tests.test_mlx_vlm_glm5_next_compat import _tiny_config

    assert apply_mlx_lm_glm5_next_patch()
    from mlx_lm.models.glm5_next import Model

    mx.random.seed(7)
    return Model(_tiny_config())


class _Group:
    def __init__(self, rank: int, size: int = 2) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def test_adapter_registers_explicit_pipeline_contract():
    model = _adapter_model()
    from mlx_lm.models import glm5_next

    assert glm5_next.SUPPORTS_PIPELINE is True
    assert glm5_next.HONORS_PIPELINE_ASSIGNMENT is True
    assert model._omlx_supports_rank_zero_logits is True
    assert model._omlx_output_vocab_size == 128
    assert "skip_logits" in inspect.signature(type(model).__call__).parameters


def test_single_node_adapter_is_logit_exact():
    model = _adapter_model()
    from mlx_vlm.models.glm5_next.linear import linear_forward

    from omlx.patches.mlx_lm_glm5_next import pipeline_patch

    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)
    actual = model(inputs, cache=model.make_cache())
    assert pipeline_patch._ORIGINAL_MODEL_CALL is not None
    assert pipeline_patch._ORIGINAL_MAKE_CACHE is not None
    original_cache = pipeline_patch._ORIGINAL_MAKE_CACHE(model.language_model)
    expected_hidden = pipeline_patch._ORIGINAL_MODEL_CALL(
        model.model,
        inputs,
        cache=original_cache,
    )
    expected = linear_forward(model.language_model.lm_head, expected_hidden)
    mx.eval(actual, expected)
    assert mx.array_equal(actual, expected).item()


def test_pipeline_stages_build_only_local_typed_caches():
    from mlx_lm.models.cache import ArraysCache, CacheList

    early = _adapter_model()
    early.model.pipeline(_Group(rank=1))
    assert (early.model.start_idx, early.model.end_idx) == (0, 1)
    assert len(early.layers) == 1
    early_cache = early.make_cache()
    assert len(early_cache) == 1
    assert isinstance(early_cache[0], ArraysCache)

    late = _adapter_model()
    late.model.pipeline(_Group(rank=0))
    assert (late.model.start_idx, late.model.end_idx) == (1, 2)
    assert len(late.layers) == 1
    late_cache = late.make_cache()
    assert len(late_cache) == 1
    assert isinstance(late_cache[0], CacheList)


def test_unequal_assignment_is_consumed_by_pipeline_hook():
    from omlx.cluster.planner import PipelineAssignment, install_unequal_pipeline_plan

    model = _adapter_model()
    assignments = (
        PipelineAssignment(
            node_id="rank-zero",
            rank=0,
            start_layer=1,
            end_layer=2,
            fixed_weight_bytes=0,
            layer_weight_bytes=1,
            capacity_bytes=2,
            reserve_bytes=0,
        ),
        PipelineAssignment(
            node_id="rank-one",
            rank=1,
            start_layer=0,
            end_layer=1,
            fixed_weight_bytes=0,
            layer_weight_bytes=1,
            capacity_bytes=2,
            reserve_bytes=0,
        ),
    )
    with install_unequal_pipeline_plan(assignments):
        model.model.pipeline(_Group(rank=1))
    assert (model.model.start_idx, model.model.end_idx) == (0, 1)


def test_sanitize_keeps_direct_language_root_and_drops_vision(monkeypatch):
    model = _adapter_model()
    seen = {}

    def sanitize(weights):
        seen.update(weights)
        return weights

    monkeypatch.setattr(model.language_model, "sanitize", sanitize)
    weights = {
        "language_model.model.layers.0.weight": mx.ones((1,)),
        "model.language_model.layers.1.weight": mx.ones((1,)),
        "vision_model.block.weight": mx.ones((1,)),
    }
    result = model.sanitize(weights)
    assert set(result) == {
        "language_model.model.layers.0.weight",
        "language_model.model.layers.1.weight",
    }
    assert set(seen) == {"model.layers.0.weight", "model.layers.1.weight"}


def test_planner_prices_all_mhc_streams(tmp_path: Path):
    from omlx.cluster.planner import _activation_bytes_per_token

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": {"hidden_size": 4096, "hc_mult": 4},
            }
        )
    )
    assert _activation_bytes_per_token(tmp_path) == 4096 * 4 * 2


def test_planner_accepts_registered_text_pipeline_despite_vision_config():
    from omlx.cluster.planner import _supports_pipeline

    _adapter_model()
    assert _supports_pipeline(
        {
            "model_type": "glm5_next",
            "text_config": {"model_type": "glm5_next_text"},
            "vision_config": {"model_type": "glm5_next_vision"},
        }
    )


def test_worker_recognizes_unequal_assignment_contract(tmp_path: Path):
    from omlx.cluster.pipeline_compat import pipeline_assignment_is_honored
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm5_next",
                "text_config": {"model_type": "glm5_next_text"},
            }
        )
    )
    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=False)
    assert pipeline_assignment_is_honored(tmp_path)


def test_runtime_accepts_rank_zero_logits_contract():
    from omlx.cluster.runtime_optimizations import _supports_rank_zero_logits

    supported, vocab_size, reason = _supports_rank_zero_logits(_adapter_model())
    assert supported is True
    assert vocab_size == 128
    assert "skip" in reason
