from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from omlx.patches.glm5_next.vision import (
    GLM5_NEXT_VISION_RUNTIME_READY,
    IMAGE_TOKEN_ID,
    VIDEO_END_TOKEN_ID,
    VIDEO_START_TOKEN_ID,
    Glm5NextVisionUnsupportedError,
    MediaKind,
    configure_vision_for_converted_weights,
    converted_vision_parameter_shapes,
    make_vision_component_classes,
    make_vision_model_class,
    prepare_media_inputs,
    reject_unsupported_media,
    validate_converted_vision_weight_layout,
    vision_cu_seqlens,
    vision_position_ids,
    vision_runtime_factory,
    vision_runtime_gaps,
)


class _Array:
    def __init__(self, shape, values=None):
        self.shape, self._values = shape, values

    def tolist(self):
        if self._values is None:
            raise AssertionError("tolist was not expected")
        return self._values


def _official_vision_config():
    return {
        "model_type": "glm5_next_vision",
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "image_size": 448,
        "patch_size": 14,
        "temporal_patch_size": 2,
        "spatial_merge_size": 2,
        "out_hidden_size": 4096,
        "projection_intermediate_size": 10240,
        "in_channels": 3,
        "attention_bias": True,
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "swiglu_limit": 10.0,
        "rms_norm_eps": 1e-5,
    }


def test_module_and_factory_are_lazy_about_mlx():
    code = """
import sys
import omlx.patches.glm5_next.vision as vision
assert 'mlx.core' not in sys.modules
assert vision.make_vision_model_class.cache_info().currsize == 0
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("bits", [8, 4])
def test_converted_affine_names_match_lazy_tower_binding(bits):
    shapes = converted_vision_parameter_shapes(bits=bits, group_size=64)
    qkv = "model.visual.blocks.0.attn.qkv"
    assert shapes[qkv + ".weight"] == (3072, 1024 * bits // 32)
    assert shapes[qkv + ".scales"] == (3072, 16)
    assert shapes[qkv + ".biases"] == (3072, 16)
    assert shapes[qkv + ".bias"] == (3072,)
    assert shapes["model.visual.merger.down_proj.scales"] == (4096, 160)
    assert "model.visual.patch_embed.proj.scales" not in shapes
    assert shapes["model.visual.patch_embed.proj.weight"] == (1024, 3, 2, 14, 14)
    assert "model.visual.downsample.scales" not in shapes
    assert shapes["model.visual.downsample.weight"] == (4096, 1024, 2, 2)
    assert all(
        f"model.visual.blocks.{layer}.attn.qkv.scales" in shapes for layer in range(24)
    )
    validate_converted_vision_weight_layout(shapes, bits=bits, group_size=64)
    broken = dict(shapes)
    broken.pop("model.visual.blocks.23.mlp.down_proj.biases")
    with pytest.raises(ValueError, match="parameter names changed"):
        validate_converted_vision_weight_layout(broken, bits=bits, group_size=64)


@pytest.mark.parametrize("bits", [8, 4])
def test_outer_binding_turns_exact_affine_module_paths_into_converted_names(bits):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten

    class _Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(64, 192, bias=True)
            self.proj = nn.Linear(64, 64, bias=True)

    class _MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(64, 128, bias=True)
            self.up_proj = nn.Linear(64, 128, bias=True)
            self.down_proj = nn.Linear(128, 64, bias=True)

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn, self.mlp = _Attention(), _MLP()

    class _Merger(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 64, bias=False)
            self.gate_proj = nn.Linear(64, 128, bias=False)
            self.up_proj = nn.Linear(64, 128, bias=False)
            self.down_proj = nn.Linear(128, 64, bias=False)

    class _TinyTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks, self.merger = [_Block()], _Merger()

    tower = configure_vision_for_converted_weights(
        _TinyTower(), {"bits": bits, "group_size": 64, "mode": "affine"}
    )
    parameters = dict(tree_flatten(tower.parameters()))
    assert parameters["blocks.0.attn.qkv.weight"].shape == (192, 64 * bits // 32)
    assert parameters["blocks.0.attn.qkv.scales"].shape == (192, 1)
    assert parameters["blocks.0.attn.qkv.biases"].shape == (192, 1)
    assert parameters["blocks.0.attn.qkv.bias"].shape == (192,)
    assert "merger.down_proj.scales" in parameters
    assert isinstance(tower.blocks[0].attn.qkv, nn.QuantizedLinear)
    mx.eval(parameters)


def test_runtime_factory_is_ready_and_pins_vendored_auto_processor_contract():
    code = """
import sys
from omlx.patches.glm5_next.vision import (
    GLM5_NEXT_VISION_RUNTIME_READY, converted_vision_parameter_shapes,
    vision_runtime_factory, vision_runtime_gaps,
)
config = {
    'vision_config': {
        'model_type': 'glm5_next_vision', 'depth': 24, 'hidden_size': 1024,
        'intermediate_size': 4096, 'num_heads': 16, 'image_size': 448,
        'patch_size': 14, 'temporal_patch_size': 2, 'spatial_merge_size': 2,
        'out_hidden_size': 4096, 'projection_intermediate_size': 10240,
        'in_channels': 3, 'attention_bias': True, 'attention_dropout': 0.0,
        'hidden_act': 'silu', 'swiglu_limit': 10.0, 'rms_norm_eps': 1e-5,
    },
    'quantization': {'bits': 4, 'group_size': 64, 'mode': 'affine'},
}
names = converted_vision_parameter_shapes(bits=4)
binding = vision_runtime_factory(config, converted_weights=names)
assert GLM5_NEXT_VISION_RUNTIME_READY is True
assert vision_runtime_gaps() == []
assert binding.processor_class == 'transformers.AutoProcessor'
assert binding.processor_revision == 'eb4d9e2a64a013bec12289288b85d0b1210ba0aa'
assert set(binding.processor_required_outputs) == {'input_ids', 'attention_mask', 'mm_token_type_ids'}
assert 'mlx.core' not in sys.modules
assert 'transformers.models.glm5_next.processing_glm5_next' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)
    assert GLM5_NEXT_VISION_RUNTIME_READY is True
    assert vision_runtime_gaps() == []


def test_official_auto_processor_image_video_output_contract_is_accepted():
    processor_output = {
        "input_ids": np.array(
            [
                [
                    10,
                    IMAGE_TOKEN_ID,
                    VIDEO_START_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    IMAGE_TOKEN_ID,
                    VIDEO_END_TOKEN_ID,
                ]
            ]
        ),
        "attention_mask": np.ones((1, 6), dtype=np.int32),
        "mm_token_type_ids": np.array([[0, 1, 0, 2, 2, 0]], dtype=np.int32),
        "pixel_values": np.zeros((8, 1176), dtype=np.float32),
        "image_grid_thw": np.array([[1, 2, 4]], dtype=np.int64),
        "pixel_values_videos": np.zeros((8, 1176), dtype=np.float32),
        "video_grid_thw": np.array([[2, 2, 2]], dtype=np.int64),
        # The official processor can retain metadata used to render timestamps;
        # it is intentionally not consumed by the vision encoder.
        "video_metadata": [object()],
    }
    binding = vision_runtime_factory(
        {"vision_config": _official_vision_config()},
        quantization={"bits": 8, "group_size": 64, "mode": "affine"},
    )
    prepared = binding.prepare_media(processor_output)
    assert prepared.kind is MediaKind.IMAGE_AND_VIDEO
    assert prepared.image.split_sizes == (2,)
    assert prepared.video.encoder_grid_thw == ((1, 2, 2), (1, 2, 2))
    assert prepared.video.split_sizes == (2,)


def test_processor_contract_routes_image_video_and_flattens_video_frames():
    image_grid = _Array((1, 3), [[1, 2, 4]])
    video_grid = _Array((1, 3), [[3, 2, 2]])
    prepared = prepare_media_inputs(
        pixel_values=_Array((8, 1176)),
        image_grid_thw=image_grid,
        pixel_values_videos=_Array((12, 1176)),
        video_grid_thw=video_grid,
        second_per_grid_ts=[0.5],
    )
    assert prepared.kind is MediaKind.IMAGE_AND_VIDEO
    assert prepared.image.split_sizes == (2,)
    assert prepared.image.encoder_grid_thw == ((1, 2, 4),)
    assert prepared.video.split_sizes == (3,)
    assert prepared.video.encoder_grid_thw == ((1, 2, 2),) * 3
    reject_unsupported_media(pixel_values=_Array((8, 1176)), image_grid_thw=image_grid)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"pixel_values": _Array((8, 1176))}, "requires both"),
        (
            {
                "pixel_values": _Array((8, 1175)),
                "image_grid_thw": _Array((1, 3), [[1, 2, 4]]),
            },
            "processor shape",
        ),
        (
            {
                "pixel_values": _Array((6, 1176)),
                "image_grid_thw": _Array((1, 3), [[1, 2, 4]]),
            },
            "do not match",
        ),
        (
            {
                "pixel_values": _Array((6, 1176)),
                "image_grid_thw": _Array((1, 3), [[1, 2, 3]]),
            },
            "divisible",
        ),
        ({"image": object()}, "raw media aliases"),
    ],
)
def test_unsupported_media_shapes_remain_fail_closed(kwargs, match):
    with pytest.raises(Glm5NextVisionUnsupportedError, match=match):
        prepare_media_inputs(**kwargs)


def test_positions_and_attention_boundaries_match_independent_reference():
    grid = _Array((2, 3), [[2, 2, 4], [1, 4, 2]])
    expected_positions, boundaries = [], [0]
    for temporal, height, width in grid.tolist():
        frame = []
        for block_h in range(height // 2):
            for block_w in range(width // 2):
                for inner_h in range(2):
                    for inner_w in range(2):
                        frame.append((block_h * 2 + inner_h, block_w * 2 + inner_w))
        expected_positions.extend(frame * temporal)
        for _ in range(temporal):
            boundaries.append(boundaries[-1] + height * width)
    assert vision_position_ids(grid) == tuple(expected_positions)
    assert vision_cu_seqlens(grid) == tuple(boundaries)


def _mx_and_components():
    mx = pytest.importorskip("mlx.core")
    return mx, make_vision_component_classes()


def test_patch_embedding_matches_independent_numpy_contraction():
    mx, components = _mx_and_components()
    patch = components["PatchEmbed"](
        embed_dim=3, in_channels=1, temporal_patch_size=2, patch_size=2
    )
    pixels = np.arange(16, dtype=np.float32).reshape(2, 8) / 7
    weight = np.arange(24, dtype=np.float32).reshape(3, 1, 2, 2, 2) / 11
    bias = np.array([0.2, -0.3, 0.7], dtype=np.float32)
    patch.proj.weight, patch.proj.bias = mx.array(weight), mx.array(bias)
    expected = pixels @ weight.reshape(3, 8).T + bias
    np.testing.assert_allclose(
        np.asarray(patch(mx.array(pixels))), expected, rtol=1e-6, atol=1e-6
    )


def _rms(x, weight, eps=1e-5):
    return x * (np.mean(x * x, axis=-1, keepdims=True) + eps) ** -0.5 * weight


def _rotate_half(x):
    half = x.shape[-1] // 2
    return np.concatenate((-x[..., half:], x[..., :half]), axis=-1)


def test_segmented_qk_normalized_rotary_attention_matches_numpy():
    mx, components = _mx_and_components()
    attention = components["Attention"](hidden_size=4, num_heads=2)
    rng = np.random.default_rng(7)
    x = rng.normal(size=(5, 4)).astype(np.float32)
    qkv_w, qkv_b = (
        rng.normal(size=(12, 4)).astype(np.float32) / 3,
        rng.normal(size=12).astype(np.float32) / 5,
    )
    proj_w, proj_b = (
        rng.normal(size=(4, 4)).astype(np.float32) / 3,
        rng.normal(size=4).astype(np.float32) / 5,
    )
    q_norm, k_norm = np.array([1.2, 0.8], np.float32), np.array([0.7, 1.1], np.float32)
    cos = rng.uniform(0.3, 1.0, size=(5, 2)).astype(np.float32)
    sin = rng.uniform(-0.4, 0.4, size=(5, 2)).astype(np.float32)
    attention.qkv.weight, attention.qkv.bias = mx.array(qkv_w), mx.array(qkv_b)
    attention.proj.weight, attention.proj.bias = mx.array(proj_w), mx.array(proj_b)
    attention.q_norm.weight, attention.k_norm.weight = (
        mx.array(q_norm),
        mx.array(k_norm),
    )
    actual = np.asarray(
        attention(mx.array(x), (0, 2, 5), (mx.array(cos), mx.array(sin)))
    )

    qkv = (x @ qkv_w.T + qkv_b).reshape(5, 3, 2, 2)
    q, k, v = _rms(qkv[:, 0], q_norm), _rms(qkv[:, 1], k_norm), qkv[:, 2]
    q = q * cos[:, None] + _rotate_half(q) * sin[:, None]
    k = k * cos[:, None] + _rotate_half(k) * sin[:, None]
    chunks = []
    for start, stop in ((0, 2), (2, 5)):
        scores = np.einsum("ihd,jhd->hij", q[start:stop], k[start:stop]) / math.sqrt(2)
        scores -= scores.max(axis=-1, keepdims=True)
        probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
        chunks.append(np.einsum("hij,jhd->ihd", probs, v[start:stop]))
    expected = np.concatenate(chunks).reshape(5, 4) @ proj_w.T + proj_b
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def _gelu(x):
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return (0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))).astype(np.float32)


def test_downsample_and_projection_merger_match_numpy():
    mx, components = _mx_and_components()
    rng = np.random.default_rng(11)
    downsample = components["Downsample"](2, 3, 2)
    hidden, down_w, down_b = (
        rng.normal(size=(8, 2)).astype(np.float32),
        rng.normal(size=(3, 2, 2, 2)).astype(np.float32),
        rng.normal(size=3).astype(np.float32),
    )
    downsample.weight, downsample.bias = mx.array(down_w), mx.array(down_b)
    expected_down = (
        np.einsum("nhwc,ochw->no", hidden.reshape(2, 2, 2, 2), down_w) + down_b
    )
    np.testing.assert_allclose(
        np.asarray(downsample(mx.array(hidden))), expected_down, rtol=1e-6, atol=1e-6
    )

    merger = components["PatchMerger"](3, 5)
    matrices = {
        name: rng.normal(size=shape).astype(np.float32) / 3
        for name, shape in {
            "proj": (3, 3),
            "gate_proj": (5, 3),
            "up_proj": (5, 3),
            "down_proj": (3, 5),
        }.items()
    }
    for name, value in matrices.items():
        getattr(merger, name).weight = mx.array(value)
    ln_weight, ln_bias = (
        np.array([0.8, 1.1, 1.3], np.float32),
        np.array([-0.2, 0.1, 0.3], np.float32),
    )
    merger.post_projection_norm.weight, merger.post_projection_norm.bias = (
        mx.array(ln_weight),
        mx.array(ln_bias),
    )
    actual = np.asarray(merger(mx.array(expected_down.astype(np.float32))))
    projected = expected_down @ matrices["proj"].T
    normalized = (projected - projected.mean(-1, keepdims=True)) / np.sqrt(
        projected.var(-1, keepdims=True) + 1e-5
    )
    merged = _gelu(normalized * ln_weight + ln_bias)
    gate, up = (
        np.minimum(merged @ matrices["gate_proj"].T, 10),
        np.clip(merged @ matrices["up_proj"].T, -10, 10),
    )
    expected = ((gate / (1 + np.exp(-gate))) * up) @ matrices["down_proj"].T
    np.testing.assert_allclose(actual, expected, rtol=4e-5, atol=4e-5)


def test_placeholder_masks_and_injection_route_shared_token_by_video_span():
    mx, _ = _mx_and_components()
    model_class = make_vision_model_class()
    ids = mx.array(
        [
            [
                10,
                IMAGE_TOKEN_ID,
                VIDEO_START_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                VIDEO_END_TOKEN_ID,
                11,
            ]
        ]
    )
    image_mask, video_mask = model_class.placeholder_masks(ids)
    assert np.asarray(image_mask)[..., 0].tolist() == [
        [False, True, False, False, False, False, False]
    ]
    assert np.asarray(video_mask)[..., 0].tolist() == [
        [False, False, False, True, True, False, False]
    ]
    embeddings = mx.zeros((1, 7, 3))
    result = np.asarray(
        model_class.inject_media_features(
            embeddings,
            ids,
            image_features=mx.array([[1.0, 2.0, 3.0]]),
            video_features=mx.array([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
        )
    )
    np.testing.assert_array_equal(result[0, 1], [1, 2, 3])
    np.testing.assert_array_equal(result[0, 3], [4, 5, 6])
    np.testing.assert_array_equal(result[0, 4], [7, 8, 9])
    with pytest.raises(Glm5NextVisionUnsupportedError, match="do not match"):
        model_class.inject_media_features(
            embeddings, ids, video_features=mx.zeros((1, 3))
        )
