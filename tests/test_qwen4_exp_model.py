from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from omlx.model_discovery import detect_model_type
from omlx.patches.qwen4_exp.model import Model


def test_registration_installs_native_module_without_qwen35_alias(tmp_path: Path):
    code = f"""
from omlx.patches.qwen4_exp import apply_qwen4_exp_patch
assert apply_qwen4_exp_patch({str(tmp_path)!r}) is True
from mlx_lm.models import qwen4_exp
assert qwen4_exp.QWEN4_EXP_STRICT_QSA is True
assert qwen4_exp.ModelArgs.__module__ == 'omlx.patches.qwen4_exp.config'
assert qwen4_exp.ModelArgs.__name__ == 'ModelArgs'
assert qwen4_exp.TextModelArgs.__name__ == 'TextModelArgs'
print(qwen4_exp.__name__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "mlx_lm.models.qwen4_exp"


def test_split_compute_artifact_binds_sibling_ssd_ple(tmp_path: Path):
    from omlx.patches.qwen4_exp import get_model_dir, set_model_dir

    compute = tmp_path / "compute-q8"
    ple = tmp_path / "ple-bf16"
    compute.mkdir()
    ple.mkdir()
    (compute / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "qwen4_exp_artifact": {"ple_artifact": "../ple-bf16"},
            }
        )
    )

    set_model_dir(compute)
    assert get_model_dir() == ple.resolve()

    (compute / "config.json").write_text(
        json.dumps({"qwen4_exp_artifact": {"ple_artifact": "../../escape"}})
    )
    with pytest.raises(ValueError, match="escapes"):
        set_model_dir(compute)


def test_multimodal_outer_config_routes_to_explicit_text_backbone(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "architectures": ["Qwen4ExpForConditionalGeneration"],
                "vision_config": {"model_type": "qwen4_exp"},
                "text_config": {"model_type": "qwen4_exp_text"},
            }
        )
    )

    assert detect_model_type(tmp_path) == "llm"


def test_outer_sanitize_keeps_mtp_and_excludes_ssd_ple_table():
    class IdentityLanguageModel:
        model = SimpleNamespace(
            layers=[
                None,
                SimpleNamespace(ple=SimpleNamespace(_get_pool=lambda: object())),
            ]
        )

        @staticmethod
        def sanitize(weights):
            return weights

    instance = Model.__new__(Model)
    instance._quantized_checkpoint = True
    instance.language_model = IdentityLanguageModel()
    weights = {
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": object(),
        "model.language_model.layers.1.ple.key_proj.weight": "ple-projection",
        "mtp.fc_embedding.weight": "mtp",
        "model.visual.blocks.0.attn.weight": "vision",
    }

    sanitized = Model.sanitize(instance, weights)

    assert not any("ngram_embedding" in key for key in sanitized)
    assert (
        sanitized["language_model.model.layers.1.ple.key_proj.weight"]
        == "ple-projection"
    )
    assert sanitized["language_model.mtp.fc_embedding.weight"] == "mtp"
    assert not any("visual" in key for key in sanitized)


def test_mtp_factory_surface_is_depth_one_qsa_without_ple():
    # Do not instantiate the 125B graph here. The isolated MTP patch consumes
    # this exact module-level factory and verifies its fail-closed QSA edge.
    from omlx.patches.qwen4_exp import model as qwen4_model

    assert callable(qwen4_model.build_mtp_decoder_layer)
    assert qwen4_model.Qwen4ExpTextDecoderLayer is qwen4_model.DecoderLayer
    assert qwen4_model.QWEN4_EXP_STRICT_QSA is True


def test_initial_hyper_streams_repeat_vectors_not_elements():
    import mlx.core as mx

    from omlx.patches.qwen4_exp.model import expand_hyper_streams

    hidden = mx.array([[[1.0, 2.0, 3.0]]])
    expanded = expand_hyper_streams(hidden, 4)

    assert expanded.tolist() == [
        [[1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0]]
    ]


def test_qsa_cache_is_accepted_by_mlx_lm_continuous_batching():
    from mlx_lm.generate import _make_cache
    from mlx_lm.models.cache import BatchKVCache, CacheList, KVCache

    from omlx.patches.qwen4_exp.model import make_qsa_cache

    cache = make_qsa_cache()
    assert isinstance(cache, CacheList)
    assert all(isinstance(item, KVCache) for item in cache.caches)

    class ModelWithQSA:
        @staticmethod
        def make_cache():
            return [make_qsa_cache()]

    batched = _make_cache(ModelWithQSA(), left_padding=[0, 3], max_kv_size=None)
    assert len(batched) == 1
    assert isinstance(batched[0], CacheList)
    assert all(isinstance(item, BatchKVCache) for item in batched[0].caches)
    mask = batched[0].caches[0].make_mask(2, return_array=True, window_size=None)
    assert mask.shape == (2, 1, 2, 2)


def test_depth_one_reject_replays_gdn_and_ple_then_trims_qsa():
    from omlx.patches.qwen4_exp.model import TextModel

    class FakeArrayCache:
        def __init__(self):
            self.values = ["conv-full", "ssm-full", "ple-full", "ctx-full"]
            self.rollback_state = ("conv-before", "ssm-before")
            self._mtp_draft_stash = (
                np.zeros((1, 2, 3)),
                np.zeros((1, 2, 1)),
                np.zeros((1, 2, 1)),
            )
            self._qwen4_ple_rollback_state = ("ple-before", "ctx-before")
            self._qwen4_ple_draft_stash = (
                np.zeros((1, 2, 4)),
                np.asarray([[10, 11]]),
            )

        def __getitem__(self, index):
            return self.values[index]

        def __setitem__(self, index, value):
            self.values[index] = value

    class FakeGDN:
        @staticmethod
        def _process_chunk(qkv, a, b, conv, recurrent, mask):
            assert qkv.shape[1] == a.shape[1] == b.shape[1] == 1
            assert (conv, recurrent, mask) == ("conv-before", "ssm-before", None)
            return None, "conv-kept", "ssm-kept"

    class FakePLE:
        def __call__(self, hidden, tokens, cache=None):
            assert hidden.shape[1] == tokens.shape[1] == 1
            assert cache[2:] == ["ple-before", "ctx-before"]
            cache[2], cache[3] = "ple-kept", "ctx-kept"

    class FakeQSA:
        def __init__(self):
            self.trimmed = 0

        def is_trimmable(self):
            return True

        def trim(self, count):
            self.trimmed += count
            return count

    linear = SimpleNamespace(
        is_linear=True,
        linear_attn=FakeGDN(),
        ple=FakePLE(),
    )
    qsa = SimpleNamespace(is_linear=False, ple=None)
    owner = SimpleNamespace(model=SimpleNamespace(pipeline_layers=[linear, qsa]))
    linear_cache = FakeArrayCache()
    qsa_cache = FakeQSA()

    assert TextModel.mtp_partial_rollback(
        owner, [linear_cache, qsa_cache], accepted=0, num_drafts=1
    )
    assert linear_cache.values == [
        "conv-kept",
        "ssm-kept",
        "ple-kept",
        "ctx-kept",
    ]
    assert linear_cache.rollback_state is None
    assert linear_cache._mtp_draft_stash is None
    assert linear_cache._qwen4_ple_rollback_state is None
    assert linear_cache._qwen4_ple_draft_stash is None
    assert qsa_cache.trimmed == 1
