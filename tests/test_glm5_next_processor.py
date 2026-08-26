# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import re

import numpy as np
import pytest

from omlx.patches.glm5_next.processor import (
    TRANSFORMERS_PROCESSOR_REVISION,
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
    VideoMetadata,
    install_glm5_next_processor_namespace,
    smart_resize,
)
from omlx.patches.glm5_next.vision import (
    GLM5_NEXT_VISION_RUNTIME_READY,
    prepare_media_inputs,
    vision_runtime_gaps,
)


class _Tokenizer:
    image_token_id = 154854
    video_token_id = 154855
    model_input_names = ["input_ids", "attention_mask"]
    chat_template = "unused"
    _tokens = {
        "<|image|>": 154854,
        "<|video|>": 154855,
        "<|begin_of_image|>": 154830,
        "<|end_of_image|>": 154831,
        "<|begin_of_video|>": 154832,
        "<|end_of_video|>": 154833,
    }

    def convert_tokens_to_ids(self, token):
        return self._tokens[token]

    def __call__(self, texts, **kwargs):
        rows = []
        pattern = re.compile("(" + "|".join(map(re.escape, self._tokens)) + ")")
        for text in texts:
            ids = []
            for piece in filter(None, pattern.split(text)):
                if piece in self._tokens:
                    ids.append(self._tokens[piece])
                else:
                    ids.extend([7] * len(piece))
            rows.append(ids)
        if len({len(row) for row in rows}) != 1:
            raise ValueError("fixture tokenizer requires equal lengths")
        return {"input_ids": rows, "attention_mask": [[1] * len(row) for row in rows]}

    def apply_chat_template(self, messages, **kwargs):
        prompt = "".join(
            f"<|{message['role']}|>{message['content']}" for message in messages
        )
        if kwargs.get("add_generation_prompt"):
            prompt += "<|assistant|>"
        return prompt

    def batch_decode(self, outputs, **kwargs):
        return ["decoded"] * len(outputs)


def _image_reference(image_chw):
    rows = []
    for block_h in range(1):
        for block_w in range(1):
            for inner_h in range(2):
                for inner_w in range(2):
                    y, x = (block_h * 2 + inner_h) * 14, (block_w * 2 + inner_w) * 14
                    row = []
                    for channel in range(3):
                        patch = image_chw[channel, y : y + 14, x : x + 14].reshape(-1)
                        row.extend(patch)
                        row.extend(patch)
                    rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _video_reference(video_tchw):
    if len(video_tchw) & 1:
        video_tchw = np.concatenate((video_tchw, video_tchw[-1:]))
    rows = []
    for grid_t in range(len(video_tchw) // 2):
        for block_h in range(1):
            for block_w in range(1):
                for inner_h in range(2):
                    for inner_w in range(2):
                        y, x = (
                            (block_h * 2 + inner_h) * 14,
                            (block_w * 2 + inner_w) * 14,
                        )
                        row = []
                        for channel in range(3):
                            for temporal in range(2):
                                row.extend(
                                    video_tchw[
                                        grid_t * 2 + temporal,
                                        channel,
                                        y : y + 14,
                                        x : x + 14,
                                    ].reshape(-1)
                                )
                        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def test_pinned_smart_resize_and_image_patch_order_match_independent_fixture():
    assert TRANSFORMERS_PROCESSOR_REVISION == "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"
    assert smart_resize(2, 56, 56) == (112, 112)
    image = np.arange(28 * 28 * 3, dtype=np.float32).reshape(28, 28, 3)
    processor = Glm5NextImageProcessor(
        do_resize=False, do_rescale=False, do_normalize=False
    )
    output = processor(image)
    assert output["image_grid_thw"].tolist() == [[1, 2, 2]]
    expected = _image_reference(image.transpose(2, 0, 1))
    np.testing.assert_array_equal(output["pixel_values"], expected)


def test_resize_rescale_and_clip_normalization_match_constant_fixture():
    image = np.full((56, 56, 3), 255, dtype=np.uint8)
    output = Glm5NextImageProcessor()(image)
    assert output["image_grid_thw"].tolist() == [[1, 8, 8]]
    expected_channels = (1 - np.array([0.48145466, 0.4578275, 0.40821073])) / np.array(
        [0.26862954, 0.26130258, 0.27577711]
    )
    first = output["pixel_values"][0].reshape(3, 2, 14, 14)
    for channel in range(3):
        np.testing.assert_allclose(
            first[channel], expected_channels[channel], rtol=1e-6, atol=1e-6
        )


def test_video_temporal_pair_padding_patch_order_and_grid_match_fixture():
    video = np.arange(3 * 28 * 28 * 3, dtype=np.float32).reshape(3, 28, 28, 3)
    processor = Glm5NextVideoProcessor(
        do_resize=False, do_rescale=False, do_normalize=False
    )
    output = processor(video, video_metadata=VideoMetadata(4, 3))
    assert output["video_grid_thw"].tolist() == [[2, 2, 2]]
    expected = _video_reference(video.transpose(0, 3, 1, 2))
    np.testing.assert_array_equal(output["pixel_values_videos"], expected)
    assert output["pixel_values_videos"].shape == (8, 1176)


def test_frame_sampling_and_timestamp_prompt_expansion_are_pinned():
    video_processor = Glm5NextVideoProcessor(
        do_resize=False, do_rescale=False, do_normalize=False
    )
    metadata = VideoMetadata(4, 8, duration=2)
    assert video_processor.sample_frames(metadata).tolist() == [0, 2, 4, 6]
    processor = Glm5NextProcessor(
        Glm5NextImageProcessor(do_resize=False, do_rescale=False, do_normalize=False),
        _Tokenizer(),
        video_processor,
    )
    video = np.zeros((4, 28, 28, 3), dtype=np.uint8)
    output = processor(
        "<|begin_of_video|><|video|><|end_of_video|>",
        videos=video,
        video_metadata=VideoMetadata(4, 4),
    )
    ids = output["input_ids"][0]
    assert np.count_nonzero(ids == 154854) == 2
    assert output["mm_token_type_ids"][0][ids == 154854].tolist() == [2, 2]
    # Each temporal pair produces one frame structure and timestamps use [::2].
    rendered = processor.replace_video_token(output, 0)
    assert rendered.count("<|begin_of_image|>") == 2
    assert "0.0 seconds" in rendered and "0.5 seconds" in rendered


def test_combined_processor_outputs_satisfy_native_vision_abi_end_to_end():
    processor = Glm5NextProcessor(
        Glm5NextImageProcessor(do_resize=False, do_rescale=False, do_normalize=False),
        _Tokenizer(),
        Glm5NextVideoProcessor(do_resize=False, do_rescale=False, do_normalize=False),
    )
    image = np.zeros((28, 28, 3), dtype=np.uint8)
    video = np.zeros((4, 28, 28, 3), dtype=np.uint8)
    output = processor(
        "<|image|><|begin_of_video|><|video|><|end_of_video|>",
        images=image,
        videos=video,
        video_metadata=VideoMetadata(4, 4),
    )
    prepared = prepare_media_inputs(output)
    assert prepared.image.split_sizes == (1,)
    assert prepared.video.split_sizes == (2,)
    assert output["pixel_values"].shape == (4, 1176)
    assert output["pixel_values_videos"].shape == (8, 1176)
    assert GLM5_NEXT_VISION_RUNTIME_READY is True
    assert vision_runtime_gaps() == []


def test_chat_template_collects_media_expands_placeholders_and_routes_modalities():
    processor = Glm5NextProcessor(
        Glm5NextImageProcessor(do_resize=False, do_rescale=False, do_normalize=False),
        _Tokenizer(),
        Glm5NextVideoProcessor(do_resize=False, do_rescale=False, do_normalize=False),
    )
    image = np.zeros((28, 28, 3), dtype=np.uint8)
    video = np.zeros((2, 28, 28, 3), dtype=np.uint8)
    output = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "video", "video": video},
                    {"type": "text", "text": "describe"},
                ],
            }
        ],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        video_metadata=VideoMetadata(2, 2),
    )
    image_positions = output["input_ids"] == 154854
    assert image_positions.sum() == 2
    assert sorted(output["mm_token_type_ids"][image_positions].tolist()) == [1, 2]
    assert output["image_grid_thw"].tolist() == [[1, 2, 2]]
    assert output["video_grid_thw"].tolist() == [[1, 2, 2]]
    prepare_media_inputs(output)


def test_namespace_installation_and_auto_processor_local_loading(tmp_path, monkeypatch):
    import transformers

    assert install_glm5_next_processor_namespace() in (True, False)
    from transformers.models.glm5_next.image_processing_glm5_next import (
        Glm5NextImageProcessor as NamespacedImageProcessor,
    )
    from transformers.models.glm5_next.processing_glm5_next import (
        Glm5NextProcessor as NamespacedProcessor,
    )

    assert NamespacedProcessor is Glm5NextProcessor
    assert NamespacedImageProcessor is Glm5NextImageProcessor
    assert (
        NamespacedProcessor.__module__
        == "transformers.models.glm5_next.processing_glm5_next"
    )
    assert (
        NamespacedImageProcessor.__module__
        == "transformers.models.glm5_next.image_processing_glm5_next"
    )
    config = {
        "processor_class": "Glm5NextProcessor",
        "image_processor": {
            "image_processor_type": "Glm5NextImageProcessor",
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "do_resize": False,
            "do_rescale": False,
            "do_normalize": False,
        },
        "video_processor": {
            "video_processor_type": "Glm5NextVideoProcessor",
            "patch_size": 14,
            "temporal_patch_size": 2,
            "merge_size": 2,
            "do_resize": False,
            "do_rescale": False,
            "do_normalize": False,
        },
    }
    (tmp_path / "processor_config.json").write_text(json.dumps(config))
    (tmp_path / "chat_template.jinja").write_text("{{ messages[0].content }}")
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    loaded = transformers.AutoProcessor.from_pretrained(tmp_path)
    assert isinstance(loaded, Glm5NextProcessor)
    result = loaded("<|image|>", images=np.zeros((28, 28, 3), dtype=np.uint8))
    assert result["image_grid_thw"].tolist() == [[1, 2, 2]]


def test_raw_or_mismatched_prompt_media_remains_fail_closed():
    processor = Glm5NextProcessor(
        Glm5NextImageProcessor(do_resize=False),
        _Tokenizer(),
        Glm5NextVideoProcessor(do_resize=False),
    )
    with pytest.raises(ValueError, match=r"no <\|image\|> placeholder"):
        processor("plain text", images=np.zeros((28, 28, 3), dtype=np.uint8))
