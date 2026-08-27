# SPDX-License-Identifier: Apache-2.0
"""Native mlx-vlm outer model for GLM-5.3-Flash (``glm5_next``).

This module deliberately composes the strict GLM5-Next text implementation
with the independently validated vision tower.  It is not an alias for
``glm_moe_dsa`` (GLM-5.2), and media is never discarded on a text fallback.
"""

from __future__ import annotations

import json
import sys
import types
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import BaseModelConfig, InputEmbeddingsFeatures

from omlx.patches.glm5_next.model import Model as TextOuterModel
from omlx.patches.glm5_next.model import TextModel as NativeTextModel
from omlx.patches.glm5_next.model import (
    TextModelArgs,
    native_vision_ready,
    require_runtime_ready,
)
from omlx.patches.glm5_next.nvfp4 import bind_glm5_next_nvfp4
from omlx.patches.glm5_next.processor import (
    Glm5NextImageProcessor,
    Glm5NextProcessor,
    Glm5NextVideoProcessor,
    load_glm5_next_processor,
)
from omlx.patches.glm5_next.vision import (
    IMAGE_TOKEN_ID,
    VISION_PREFIX,
    Glm5NextVisionModel,
    Glm5NextVisionUnsupportedError,
    configure_vision_for_converted_weights,
    prepare_media_inputs,
    sanitize_vision_weights,
    validate_vision_config,
)

GLM5_NEXT_NATIVE_VLM = True


def _known(cls: type, values: Mapping[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    return {key: value for key, value in values.items() if key in names}


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "glm5_next_vision"
    depth: int = 24
    hidden_size: int = 1_024
    intermediate_size: int = 4_096
    num_heads: int = 16
    image_size: int = 448
    patch_size: int = 14
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 4_096
    projection_intermediate_size: int = 10_240
    in_channels: int = 3
    attention_bias: bool = True
    attention_dropout: float = 0.0
    hidden_act: str = "silu"
    swiglu_limit: float = 10.0
    rms_norm_eps: float = 1e-5
    quantization: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | VisionConfig) -> VisionConfig:
        if isinstance(values, cls):
            return values
        if not isinstance(values, Mapping):
            raise ValueError("glm5_next vision_config must be an object")
        validate_vision_config(values)
        return cls(**_known(cls, values))


# The native text config already owns the strict 45-layer architecture
# validation required by the checkpoint contract.
TextConfig = TextModelArgs


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str = "glm5_next"
    architectures: list[str] | None = None
    tie_word_embeddings: bool = False
    image_token_id: int = IMAGE_TOKEN_ID
    image_token_index: int = IMAGE_TOKEN_ID
    video_token_id: int = IMAGE_TOKEN_ID
    video_token_index: int = IMAGE_TOKEN_ID
    eos_token_id: int | list[int] | None = None
    quantization: dict[str, Any] | None = None
    quantization_config: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ModelConfig:
        if not isinstance(values, Mapping) or values.get("model_type") != "glm5_next":
            raise ValueError("model_type must be 'glm5_next'")
        if bool(values.get("tie_word_embeddings", False)):
            raise ValueError("GLM5-Next has an independent final head")
        text = TextConfig.from_dict(values.get("text_config"))
        vision = VisionConfig.from_dict(values.get("vision_config"))
        return cls(
            **_known(
                cls,
                {
                    **values,
                    "model_type": "glm5_next",
                    "text_config": text,
                    "vision_config": vision,
                    "tie_word_embeddings": False,
                },
            )
        )


class LanguageModel(NativeTextModel):
    """Native text graph with the input-embedding spelling used by mlx-vlm."""

    def __init__(self, config: TextConfig | Mapping[str, Any], *_args: Any):
        if not isinstance(config, TextConfig):
            config = TextConfig.from_dict(config)
        super().__init__(config)
        self.config = config

    def __call__(
        self,
        inputs,
        inputs_embeds=None,
        *,
        input_embeddings=None,
        cache=None,
        return_hidden=False,
        mask=None,
        **_kwargs,
    ):
        del mask
        if inputs_embeds is not None and input_embeddings is not None:
            raise ValueError("pass only one of inputs_embeds and input_embeddings")
        embeddings = inputs_embeds if inputs_embeds is not None else input_embeddings
        return super().__call__(
            inputs,
            cache=cache,
            input_embeddings=embeddings,
            return_hidden=return_hidden,
        )


VisionModel = Glm5NextVisionModel


def _concat_features(parts: Any):
    if isinstance(parts, (list, tuple)):
        if not parts:
            return None
        return parts[0] if len(parts) == 1 else mx.concatenate(list(parts), axis=0)
    return parts


class Model(nn.Module):
    """Strict multimodal outer model consumed by the pinned mlx-vlm loader."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if not isinstance(config, ModelConfig):
            config = ModelConfig.from_dict(config)
        # Both checks happen before allocating the 320B text graph or tower.
        require_runtime_ready()
        if not native_vision_ready():
            raise RuntimeError("GLM5-Next native vision runtime is not ready")
        self.config = config
        self.model_type = config.model_type
        self.vision_tower = VisionModel(config.vision_config)
        self.language_model = LanguageModel(config.text_config, config)
        # The multimodal outer model must bind the exact ModelOpt NVFP4
        # carriers through the same shared invariant as the text outer model;
        # the checkpoint supplies 129 routed/MTP ``global_scale`` tensors that
        # ordinary SwitchLinear modules cannot accept.
        quantization = config.quantization or config.quantization_config
        self._nvfp4 = bind_glm5_next_nvfp4(self, quantization)
        self._converted_affine = isinstance(quantization, Mapping) and not self._nvfp4

    @property
    def layers(self):
        return self.language_model.layers

    def make_cache(self):
        return self.language_model.make_cache()

    def prepare_dsa_kv_projections(self) -> int:
        return self.language_model.prepare_dsa_kv_projections()

    def encode_image(self, pixel_values, image_grid_thw):
        """Expose oMLX's cacheable image-feature protocol."""

        dtype = self.vision_tower.dtype
        values = (
            pixel_values
            if isinstance(pixel_values, mx.array)
            else mx.array(pixel_values)
        )
        return _concat_features(
            self.vision_tower.encode_image(values.astype(dtype), image_grid_thw)
        )

    def encode_video(self, pixel_values_videos, video_grid_thw):
        dtype = self.vision_tower.dtype
        values = (
            pixel_values_videos
            if isinstance(pixel_values_videos, mx.array)
            else mx.array(pixel_values_videos)
        )
        return _concat_features(
            self.vision_tower.encode_video(values.astype(dtype), video_grid_thw)
        )

    def get_input_embeddings(
        self,
        input_ids=None,
        pixel_values=None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        if input_ids is None:
            raise Glm5NextVisionUnsupportedError("GLM5-Next requires input_ids")
        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        pixel_values_videos = kwargs.get("pixel_values_videos")
        if pixel_values is None and pixel_values_videos is None:
            return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

        media = prepare_media_inputs(
            pixel_values=pixel_values,
            image_grid_thw=kwargs.get("image_grid_thw"),
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=kwargs.get("video_grid_thw"),
            second_per_grid_ts=kwargs.get("second_per_grid_ts"),
        )

        cached = kwargs.get("cached_image_features")
        image_features = cached
        if media.image is not None and image_features is None:
            image_features = self.encode_image(
                media.image.pixel_values, kwargs.get("image_grid_thw")
            )

        video_features = None
        if media.video is not None:
            video_features = self.encode_video(
                media.video.pixel_values, kwargs.get("video_grid_thw")
            )

        inputs_embeds = self.vision_tower.inject_media_features(
            inputs_embeds,
            input_ids,
            image_features=image_features,
            video_features=video_features,
        )
        return InputEmbeddingsFeatures(inputs_embeds=inputs_embeds)

    def __call__(
        self,
        input_ids,
        pixel_values=None,
        mask=None,
        cache=None,
        **kwargs,
    ):
        features = self.get_input_embeddings(
            input_ids,
            pixel_values,
            mask=mask,
            **kwargs,
        )
        return self.language_model(
            input_ids,
            inputs_embeds=features.inputs_embeds,
            mask=mask,
            cache=cache,
            **kwargs,
        )

    def sanitize(self, weights):
        """Sanitize text weights while retaining the complete vision tree."""

        source = dict(weights)
        # Reuse the exact native MoE/MTP/text rewrites, then add vision under
        # the mlx-vlm-owned tower path.  TextOuterModel intentionally drops
        # vision, so it cannot accidentally leak a second copy.
        shell = TextOuterModel.__new__(TextOuterModel)
        shell._converted_affine = self._converted_affine
        output = TextOuterModel.sanitize(shell, source)

        vision = source
        if not self._converted_affine:
            vision = sanitize_vision_weights(source)
        for key, value in vision.items():
            if key.startswith(VISION_PREFIX):
                output["vision_tower." + key[len(VISION_PREFIX) :]] = value
        return output

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate


def install_mlx_vlm_submodules(package: types.ModuleType | None = None) -> None:
    """Publish the package surface used by pinned and conventional mlx-vlm."""

    package_name = "mlx_vlm.models.glm5_next"
    package = package or sys.modules[package_name]
    package.__path__ = []
    members = {
        "config": {
            "ModelConfig": ModelConfig,
            "TextConfig": TextConfig,
            "VisionConfig": VisionConfig,
        },
        "language": {"LanguageModel": LanguageModel, "TextConfig": TextConfig},
        "vision": {"VisionModel": VisionModel, "VisionConfig": VisionConfig},
        "glm5_next": {"Model": Model},
        "processing_glm5_next": {
            "Glm5NextProcessor": Glm5NextProcessor,
            "Glm5NextImageProcessor": Glm5NextImageProcessor,
            "Glm5NextVideoProcessor": Glm5NextVideoProcessor,
            "load_processor": load_glm5_next_processor,
        },
    }
    for suffix, exports in members.items():
        name = f"{package_name}.{suffix}"
        module = types.ModuleType(name)
        module.__package__ = package_name
        module.__file__ = __file__
        module.__dict__.update(exports)
        module.__all__ = tuple(exports)
        sys.modules[name] = module
        setattr(package, suffix, module)


def install_mlx_format_sanitize_patch() -> bool:
    """Run the GLM sanitizer for converted MLX-format artifacts.

    Pinned mlx-vlm skips ``Model.sanitize`` whenever safetensors metadata says
    ``format=mlx``.  GLM's streaming converter correctly marks its output MLX,
    but deliberately preserves official source roots (``model.visual.*`` and
    ``model.language_model.*``).  Those roots still require this adapter's
    one-time tree remap.  The wrapper is model-type scoped and restores the
    safetensors reader before returning, so other mlx-vlm families retain the
    upstream fast path.
    """

    import mlx_vlm.utils as vlm_utils

    current = vlm_utils.load_model
    if getattr(current, "_glm5_next_mlx_sanitize", False):
        return False

    def load_model(model_path, lazy=False, **kwargs):
        path = Path(model_path)
        try:
            config = json.loads((path / "config.json").read_text())
        except (OSError, ValueError, TypeError):
            return current(model_path, lazy, **kwargs)
        if config.get("model_type") != "glm5_next":
            return current(model_path, lazy, **kwargs)

        original_safe_open = vlm_utils.safetensors.safe_open

        class _SanitizingSafeOpen:
            def __init__(self, wrapped):
                self._wrapped = wrapped

            def __enter__(self):
                entered = self._wrapped.__enter__()
                if entered is not None:
                    self._wrapped = entered
                return self

            def __exit__(self, *args):
                return self._wrapped.__exit__(*args)

            def __getattr__(self, name):
                return getattr(self._wrapped, name)

            def metadata(self):
                metadata = self._wrapped.metadata()
                if not metadata:
                    return metadata
                metadata = dict(metadata)
                metadata.pop("format", None)
                return metadata

        def safe_open(*args, **open_kwargs):
            return _SanitizingSafeOpen(original_safe_open(*args, **open_kwargs))

        vlm_utils.safetensors.safe_open = safe_open
        try:
            model = current(model_path, lazy, **kwargs)
            # Affine Q4 replaces each DSA kv_b_proj during ``current``.  Build
            # the algebraically equivalent per-head K/V split only after the
            # checkpoint triples are loaded, keeping first-token latency out
            # of the serving path.
            prepare = getattr(model, "prepare_dsa_kv_projections", None)
            if callable(prepare):
                prepare()
            return model
        finally:
            vlm_utils.safetensors.safe_open = original_safe_open

    load_model._glm5_next_mlx_sanitize = True
    load_model.__wrapped__ = current
    vlm_utils.load_model = load_model
    return True


def install_prompt_format_patch() -> bool:
    """Teach pinned mlx-vlm to retain GLM image parts during chat formatting."""

    from mlx_vlm.prompt_utils import MODEL_CONFIG, MessageFormat

    expected = MessageFormat.LIST_WITH_IMAGE_FIRST
    if MODEL_CONFIG.get("glm5_next") == expected:
        return False
    MODEL_CONFIG["glm5_next"] = expected
    return True


__all__ = [
    "GLM5_NEXT_NATIVE_VLM",
    "Glm5NextProcessor",
    "LanguageModel",
    "Model",
    "ModelConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
    "configure_vision_for_converted_weights",
    "install_mlx_format_sanitize_patch",
    "install_mlx_vlm_submodules",
    "install_prompt_format_patch",
    "load_glm5_next_processor",
]
