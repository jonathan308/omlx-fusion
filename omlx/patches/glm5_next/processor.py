# SPDX-License-Identifier: Apache-2.0
"""Vendored GLM-5.3 processor stack for the pinned Transformers ABI.

Ported from Hugging Face Transformers commit
``eb4d9e2a64a013bec12289288b85d0b1210ba0aa``:

* ``models/glm5_next/image_processing_glm5_next.py``
* ``models/glm5_next/video_processing_glm5_next.py``
* ``models/glm5_next/processing_glm5_next.py``

The upstream files are Apache-2.0. This bounded NumPy/Pillow port preserves
their model-facing resize, normalization, patch, grid, timestamp, and prompt
contracts while avoiding a mandatory Torch/Torchvision runtime dependency.
"""

from __future__ import annotations

import io
import json
import math
import sys
import types
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image

TRANSFORMERS_PROCESSOR_REVISION: Final = "eb4d9e2a64a013bec12289288b85d0b1210ba0aa"
OPENAI_CLIP_MEAN: Final = (0.48145466, 0.4578275, 0.40821073)
OPENAI_CLIP_STD: Final = (0.26862954, 0.26130258, 0.27577711)


class ProcessorBatch(dict):
    """Small BatchFeature-compatible mapping used by the local runtime."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def to(self, *args: Any, **kwargs: Any) -> ProcessorBatch:
        return ProcessorBatch(
            {
                key: value.to(*args, **kwargs) if hasattr(value, "to") else value
                for key, value in self.items()
            }
        )


@dataclass
class VideoMetadata:
    fps: float | None
    total_num_frames: int
    duration: float | None = None
    timestamps: list[float] | None = None

    def __post_init__(self) -> None:
        if self.timestamps is None:
            fps = self.fps or 24.0
            self.timestamps = [index / fps for index in range(self.total_num_frames)]
        if self.duration is None and self.fps:
            self.duration = self.total_num_frames / self.fps


def smart_resize(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 28,
    min_pixels: int = 16,
    max_pixels: int = 8000,
) -> tuple[int, int]:
    """Exact aligned canvas calculation from the pinned processor."""

    pixels_per_token = temporal_factor * factor**2
    min_pixels *= pixels_per_token
    max_pixels *= pixels_per_token

    def align(value: int | float) -> int:
        return math.ceil(value / factor) * factor

    def fit_within_budget(aligned_frames: int) -> tuple[int, int]:
        minimum_pixels = aligned_frames * factor**2
        if max_pixels < minimum_pixels:
            raise ValueError(
                f"max_pixels={max_pixels} is too small. At least {minimum_pixels} "
                "pixels are required for one aligned patch."
            )
        low, high = 1, height
        best_height, best_width = factor, factor
        while low <= high:
            content_height = (low + high) // 2
            content_width = max(1, math.floor(width * content_height / height))
            candidate_height, candidate_width = (
                align(content_height),
                align(content_width),
            )
            if aligned_frames * candidate_height * candidate_width <= max_pixels:
                best_height, best_width = candidate_height, candidate_width
                low = content_height + 1
            else:
                high = content_height - 1
        return best_height, best_width

    aligned_frames = max(
        temporal_factor, round(num_frames / temporal_factor) * temporal_factor
    )
    aligned_height, aligned_width = align(height), align(width)
    aligned_pixel_budget = aligned_frames * aligned_height * aligned_width
    if aligned_pixel_budget < min_pixels:
        scale = math.sqrt(min_pixels / (num_frames * height * width))
        aligned_height = align(max(1, math.ceil(height * scale)))
        aligned_width = align(max(1, math.ceil(width * scale)))
        aligned_pixel_budget = aligned_frames * aligned_height * aligned_width
    if aligned_pixel_budget > max_pixels:
        aligned_height, aligned_width = fit_within_budget(aligned_frames)
    return aligned_height, aligned_width


def _resolve_file(source: str | Path, filename: str, **kwargs: Any) -> Path | None:
    path = Path(source)
    if path.is_dir():
        candidate = path / filename
        return candidate if candidate.is_file() else None
    try:
        from huggingface_hub import hf_hub_download

        accepted = {
            key: kwargs[key]
            for key in (
                "revision",
                "cache_dir",
                "force_download",
                "local_files_only",
                "token",
            )
            if key in kwargs and kwargs[key] is not None
        }
        return Path(hf_hub_download(str(source), filename=filename, **accepted))
    except Exception:
        return None


def _processor_config(source: str | Path, **kwargs: Any) -> dict[str, Any]:
    path = _resolve_file(source, "processor_config.json", **kwargs)
    if path is None:
        raise FileNotFoundError(f"processor_config.json not found for {source}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("processor_config.json root must be an object")
    return value


def _load_image(value: Any) -> np.ndarray:
    if isinstance(value, Image.Image):
        image = value.convert("RGB")
        return np.asarray(image)
    if isinstance(value, (str, Path)):
        text = str(value)
        if text.startswith(("http://", "https://")):
            with urllib.request.urlopen(text) as response:  # noqa: S310 - explicit user media
                image = Image.open(io.BytesIO(response.read())).convert("RGB")
        else:
            image = Image.open(text).convert("RGB")
        return np.asarray(image)
    array = np.asarray(value)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError(f"image must be rank 2/3; found shape {array.shape}")
    if array.shape[-1] in (1, 3, 4):
        array = array[..., :3]
    elif array.shape[0] in (1, 3, 4):
        array = array[:3].transpose(1, 2, 0)
    else:
        raise ValueError(f"cannot infer image channel dimension from {array.shape}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return array


def _resize_and_pad(
    image: np.ndarray,
    *,
    target_height: int,
    target_width: int,
    content_height: int,
    content_width: int,
) -> np.ndarray:
    if (image.shape[0], image.shape[1]) != (content_height, content_width):
        pil = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), mode="RGB")
        image = np.asarray(
            pil.resize(
                (content_width, content_height), resample=Image.Resampling.BICUBIC
            )
        )
    output = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    output[:content_height, :content_width] = image
    return output


def _normalized_chw(
    image: np.ndarray,
    *,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: tuple[float, ...] | list[float],
    image_std: tuple[float, ...] | list[float],
) -> np.ndarray:
    value = image.astype(np.float32)
    if do_rescale:
        value *= rescale_factor
    if do_normalize:
        value = (value - np.asarray(image_mean, dtype=np.float32)) / np.asarray(
            image_std, dtype=np.float32
        )
    return value.transpose(2, 0, 1)


def _convert_batch(data: dict[str, Any], return_tensors: str | None) -> ProcessorBatch:
    if return_tensors in (None, "np", "numpy"):
        return ProcessorBatch(data)
    if return_tensors in ("pt", "torch"):
        try:
            import torch
        except ImportError as error:
            raise ImportError("return_tensors='pt' requires torch") from error
        return ProcessorBatch(
            {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in data.items()
            }
        )
    if return_tensors == "mlx":
        import mlx.core as mx

        return ProcessorBatch(
            {
                key: mx.array(value) if isinstance(value, np.ndarray) else value
                for key, value in data.items()
            }
        )
    raise ValueError(f"unsupported return_tensors={return_tensors!r}")


class Glm5NextImageProcessor:
    model_input_names = ["pixel_values", "image_grid_thw"]

    def __init__(self, **kwargs: Any):
        self.do_resize = kwargs.get("do_resize", True)
        self.do_rescale = kwargs.get("do_rescale", True)
        self.rescale_factor = kwargs.get("rescale_factor", 1 / 255)
        self.do_normalize = kwargs.get("do_normalize", True)
        self.image_mean = kwargs.get("image_mean", list(OPENAI_CLIP_MEAN))
        self.image_std = kwargs.get("image_std", list(OPENAI_CLIP_STD))
        self.patch_size = kwargs.get("patch_size", 14)
        self.temporal_patch_size = kwargs.get("temporal_patch_size", 2)
        self.merge_size = kwargs.get("merge_size", 2)
        self.patch_expand_factor = kwargs.get("patch_expand_factor", 1)
        self.min_image_tokens = kwargs.get("min_image_tokens", 16)
        self.max_image_tokens = kwargs.get("max_image_tokens", 8000)
        if (self.patch_size, self.temporal_patch_size, self.merge_size) != (14, 2, 2):
            raise ValueError(
                "GLM5-Next processor requires patch=14, temporal=2, merge=2"
            )

    @classmethod
    def from_pretrained(
        cls, source: str | Path, **kwargs: Any
    ) -> Glm5NextImageProcessor:
        config = _processor_config(source, **kwargs).get("image_processor", {})
        return cls(**{**config, **{k: v for k, v in kwargs.items() if k in config}})

    def _resize(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        factor = self.patch_size * self.merge_size * self.patch_expand_factor
        target_height, target_width = smart_resize(
            self.temporal_patch_size,
            height,
            width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
            min_pixels=self.min_image_tokens,
            max_pixels=self.max_image_tokens,
        )
        pixels_per_token = self.temporal_patch_size * factor**2
        scale = min(target_height / height, target_width / width)
        if (
            self.temporal_patch_size * height * width
            >= pixels_per_token * self.min_image_tokens
        ):
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))
        return _resize_and_pad(
            image,
            target_height=target_height,
            target_width=target_width,
            content_height=content_height,
            content_width=content_width,
        )

    def patchify(self, images: np.ndarray) -> tuple[np.ndarray, int, int]:
        batch, channel, height, width = images.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = images.reshape(
            batch,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        ).transpose(0, 2, 5, 3, 6, 1, 4, 7)
        patches = np.repeat(patches[..., None, :, :], self.temporal_patch_size, axis=6)
        return (
            patches.reshape(
                batch,
                grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size**2,
            ),
            grid_h,
            grid_w,
        )

    def preprocess(
        self, images: Any, *, return_tensors: str | None = None, **kwargs: Any
    ) -> ProcessorBatch:
        items = images if isinstance(images, (list, tuple)) else [images]
        flattened, grids = [], []
        for item in items:
            image = _load_image(item)
            if self.do_resize:
                image = self._resize(image)
            chw = _normalized_chw(
                image,
                do_rescale=kwargs.get("do_rescale", self.do_rescale),
                rescale_factor=kwargs.get("rescale_factor", self.rescale_factor),
                do_normalize=kwargs.get("do_normalize", self.do_normalize),
                image_mean=kwargs.get("image_mean", self.image_mean),
                image_std=kwargs.get("image_std", self.image_std),
            )
            patches, grid_h, grid_w = self.patchify(chw[None])
            flattened.append(patches[0])
            grids.append((1, grid_h, grid_w))
        data = {
            "pixel_values": np.concatenate(flattened, axis=0).astype(np.float32),
            "image_grid_thw": np.asarray(grids, dtype=np.int64),
        }
        return _convert_batch(data, return_tensors)

    __call__ = preprocess

    def get_number_of_image_patches(
        self, height: int, width: int, images_kwargs: dict | None = None
    ) -> int:
        values = images_kwargs or {}
        patch, merge = (
            values.get("patch_size", self.patch_size),
            values.get("merge_size", self.merge_size),
        )
        resized_height, resized_width = smart_resize(
            self.temporal_patch_size,
            height,
            width,
            factor=patch * merge,
            temporal_factor=self.temporal_patch_size,
            min_pixels=values.get("min_image_tokens", self.min_image_tokens),
            max_pixels=values.get("max_image_tokens", self.max_image_tokens),
        )
        return (resized_height // patch) * (resized_width // patch)


def _video_array(video: Any) -> np.ndarray:
    if isinstance(video, np.ndarray):
        array = video
    elif isinstance(video, (list, tuple)):
        array = np.stack([_load_image(frame) for frame in video])
    else:
        raise ValueError(
            "video must be a frame array/list; use decode_video for file/URL media"
        )
    if array.ndim != 4:
        raise ValueError(
            f"video must have shape (frames,h,w,c) or (frames,c,h,w); found {array.shape}"
        )
    if array.shape[-1] in (1, 3, 4):
        array = array[..., :3]
    elif array.shape[1] in (1, 3, 4):
        array = array[:, :3].transpose(0, 2, 3, 1)
    else:
        raise ValueError(f"cannot infer video channel dimension from {array.shape}")
    return array


def _decode_video(source: str | Path) -> tuple[np.ndarray, VideoMetadata]:
    """Decode file/URL video media when the optional PyAV backend is present."""

    try:
        import av
    except ImportError as error:
        raise ImportError(
            "video file/URL inputs require the optional PyAV backend"
        ) from error
    text = str(source)
    if text.startswith(("http://", "https://")):
        with urllib.request.urlopen(text) as response:  # noqa: S310 - explicit user media
            container = av.open(io.BytesIO(response.read()))
    else:
        container = av.open(text)
    try:
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else None
        frames, timestamps = [], []
        for index, frame in enumerate(container.decode(stream)):
            frames.append(frame.to_ndarray(format="rgb24"))
            timestamps.append(
                float(frame.time) if frame.time is not None else index / (fps or 24)
            )
    finally:
        container.close()
    if not frames:
        raise ValueError(f"video contains no decodable frames: {source}")
    metadata = VideoMetadata(
        fps,
        len(frames),
        duration=(timestamps[-1] + 1 / (fps or 24)),
        timestamps=timestamps,
    )
    return np.stack(frames), metadata


class Glm5NextVideoProcessor:
    model_input_names = ["pixel_values_videos", "video_grid_thw"]

    def __init__(self, **kwargs: Any):
        self.do_resize = kwargs.get("do_resize", True)
        self.do_rescale = kwargs.get("do_rescale", True)
        self.rescale_factor = kwargs.get("rescale_factor", 1 / 255)
        self.do_normalize = kwargs.get("do_normalize", True)
        self.image_mean = kwargs.get("image_mean", list(OPENAI_CLIP_MEAN))
        self.image_std = kwargs.get("image_std", list(OPENAI_CLIP_STD))
        self.patch_size = kwargs.get("patch_size", 14)
        self.temporal_patch_size = kwargs.get("temporal_patch_size", 2)
        self.merge_size = kwargs.get("merge_size", 2)
        self.patch_expand_factor = kwargs.get("patch_expand_factor", 1)
        self.min_image_tokens = kwargs.get("min_image_tokens", 16)
        self.max_image_tokens = kwargs.get("max_image_tokens", 240000)
        self.max_frames = kwargs.get("max_frames", 2048)
        self.max_duration = kwargs.get("max_duration", 0)
        self.fps = kwargs.get("fps", 2)
        if (self.patch_size, self.temporal_patch_size, self.merge_size) != (14, 2, 2):
            raise ValueError(
                "GLM5-Next video processor requires patch=14, temporal=2, merge=2"
            )

    @classmethod
    def from_pretrained(
        cls, source: str | Path, **kwargs: Any
    ) -> Glm5NextVideoProcessor:
        config = _processor_config(source, **kwargs).get("video_processor", {})
        return cls(**{**config, **{k: v for k, v in kwargs.items() if k in config}})

    def sample_frames(
        self, metadata: VideoMetadata, fps: float | None = None, **kwargs: Any
    ) -> np.ndarray:
        if metadata is None or metadata.fps is None:
            raise ValueError("Glm5Next frame sampling requires video metadata with fps")
        total_frames = metadata.total_num_frames
        duration = metadata.duration or round((total_frames - 1) / metadata.fps) + 1
        max_seconds = int(duration)
        duration = (
            duration if self.max_duration <= 0 else min(duration, self.max_duration)
        )
        target_fps = fps if fps is not None else self.fps
        extract_t = min(int(duration * target_fps), self.max_frames)
        timestamps = [index / metadata.fps for index in range(total_frames)]
        if total_frames < extract_t:
            indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()
        else:
            indices, current_second = [], 0.0
            for index, timestamp in enumerate(timestamps):
                if timestamp >= current_second:
                    current_second += 1 / target_fps
                    indices.append(index)
                    if current_second >= max_seconds:
                        break
        if len(indices) < extract_t:
            start, end = (
                (0, max(total_frames - 1, 0))
                if not indices
                else (indices[0], indices[-1])
            )
            indices = np.linspace(start, end, extract_t, dtype=int).tolist()
        elif len(indices) > extract_t:
            indices = np.linspace(0, total_frames - 1, extract_t, dtype=int).tolist()
        unique = list(dict.fromkeys(indices))
        if len(unique) & 1:
            unique.append(unique[-1])
        return np.asarray(unique, dtype=np.int64)

    def _resize(self, video: np.ndarray) -> np.ndarray:
        frames, height, width = video.shape[:3]
        factor = self.patch_size * self.merge_size * self.patch_expand_factor
        target_height, target_width = smart_resize(
            frames,
            height,
            width,
            temporal_factor=self.temporal_patch_size,
            factor=factor,
            min_pixels=self.min_image_tokens,
            max_pixels=self.max_image_tokens,
        )
        pixels_per_token = self.temporal_patch_size * factor**2
        scale = min(target_height / height, target_width / width)
        if frames * height * width >= pixels_per_token * self.min_image_tokens:
            scale = min(1.0, scale)
        content_height = max(1, min(target_height, math.floor(height * scale)))
        content_width = max(1, min(target_width, math.floor(width * scale)))
        return np.stack(
            [
                _resize_and_pad(
                    frame,
                    target_height=target_height,
                    target_width=target_width,
                    content_height=content_height,
                    content_width=content_width,
                )
                for frame in video
            ]
        )

    def patchify(self, videos: np.ndarray) -> tuple[np.ndarray, int, int, int]:
        batch, frames, channel, height, width = videos.shape
        if pad := -frames % self.temporal_patch_size:
            videos = np.concatenate(
                (videos, np.repeat(videos[:, -1:], pad, axis=1)), axis=1
            )
            frames += pad
        grid_t, grid_h, grid_w = (
            frames // self.temporal_patch_size,
            height // self.patch_size,
            width // self.patch_size,
        )
        patches = videos.reshape(
            batch,
            grid_t,
            self.temporal_patch_size,
            channel,
            grid_h // self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w // self.merge_size,
            self.merge_size,
            self.patch_size,
        ).transpose(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
        return (
            patches.reshape(
                batch,
                grid_t * grid_h * grid_w,
                channel * self.temporal_patch_size * self.patch_size**2,
            ),
            grid_t,
            grid_h,
            grid_w,
        )

    def preprocess(
        self,
        videos: Any,
        *,
        video_metadata: list[VideoMetadata] | VideoMetadata | None = None,
        do_sample_frames: bool = False,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> ProcessorBatch:
        if isinstance(videos, (str, Path)) or (
            isinstance(videos, np.ndarray) and videos.ndim == 4
        ):
            items = [videos]
        elif (
            isinstance(videos, (list, tuple))
            and videos
            and all(
                isinstance(item, Image.Image)
                or (isinstance(item, np.ndarray) and item.ndim == 3)
                for item in videos
            )
        ):
            # One video supplied directly as a list of frames.
            items = [videos]
        else:
            # A list of independent videos: each entry is a path/URL decoded
            # through the pinned backend, a 4-D frame array, or a frame list.
            items = list(videos)
        metadata_items = (
            list(video_metadata)
            if isinstance(video_metadata, (list, tuple))
            else [video_metadata] * len(items)
        )
        flattened, grids, output_metadata = [], [], []
        for item, metadata in zip(items, metadata_items):
            if isinstance(item, (str, Path)):
                video, decoded_metadata = _decode_video(item)
                metadata = metadata or decoded_metadata
            else:
                video = _video_array(item)
            metadata = metadata or VideoMetadata(24.0, len(video))
            if do_sample_frames:
                indices = self.sample_frames(metadata, fps=kwargs.get("fps"))
                video = video[indices]
                source_timestamps = metadata.timestamps or [
                    index / (metadata.fps or 24)
                    for index in range(metadata.total_num_frames)
                ]
                metadata = VideoMetadata(
                    metadata.fps,
                    len(video),
                    timestamps=[source_timestamps[int(index)] for index in indices],
                )
            if self.do_resize:
                video = self._resize(video)
            normalized = np.stack(
                [
                    _normalized_chw(
                        frame,
                        do_rescale=kwargs.get("do_rescale", self.do_rescale),
                        rescale_factor=kwargs.get(
                            "rescale_factor", self.rescale_factor
                        ),
                        do_normalize=kwargs.get("do_normalize", self.do_normalize),
                        image_mean=kwargs.get("image_mean", self.image_mean),
                        image_std=kwargs.get("image_std", self.image_std),
                    )
                    for frame in video
                ]
            )
            patches, grid_t, grid_h, grid_w = self.patchify(normalized[None])
            flattened.append(patches[0])
            grids.append((grid_t, grid_h, grid_w))
            output_metadata.append(metadata)
        data = {
            "pixel_values_videos": np.concatenate(flattened, axis=0).astype(np.float32),
            "video_grid_thw": np.asarray(grids, dtype=np.int64),
            "video_metadata": output_metadata,
        }
        return _convert_batch(data, return_tensors)

    __call__ = preprocess

    def get_number_of_video_patches(
        self, frames: int, height: int, width: int, videos_kwargs: dict | None = None
    ) -> int:
        values = videos_kwargs or {}
        patch, merge = (
            values.get("patch_size", self.patch_size),
            values.get("merge_size", self.merge_size),
        )
        resized_height, resized_width = smart_resize(
            frames,
            height,
            width,
            factor=patch * merge,
            temporal_factor=self.temporal_patch_size,
            min_pixels=values.get("min_image_tokens", self.min_image_tokens),
            max_pixels=values.get("max_image_tokens", self.max_image_tokens),
        )
        grid_t = math.ceil(frames / self.temporal_patch_size)
        return grid_t * (resized_height // patch) * (resized_width // patch)


def _tolist(value: Any) -> list:
    return value.tolist() if hasattr(value, "tolist") else value


class Glm5NextProcessor:
    """Pinned multimodal prompt/token/media coordinator."""

    def __init__(
        self,
        image_processor: Glm5NextImageProcessor | None = None,
        tokenizer: Any = None,
        video_processor: Glm5NextVideoProcessor | None = None,
        chat_template: str | None = None,
        **kwargs: Any,
    ):
        if tokenizer is None:
            raise ValueError("Glm5NextProcessor requires a tokenizer")
        self.image_processor = image_processor or Glm5NextImageProcessor()
        self.video_processor = video_processor or Glm5NextVideoProcessor()
        self.tokenizer = tokenizer
        self.chat_template = chat_template or getattr(tokenizer, "chat_template", None)
        self.image_token = getattr(tokenizer, "image_token", None) or "<|image|>"
        self.video_token = getattr(tokenizer, "video_token", None) or "<|video|>"
        self.image_token_id = getattr(
            tokenizer, "image_token_id", None
        ) or tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = getattr(
            tokenizer, "video_token_id", None
        ) or tokenizer.convert_tokens_to_ids(self.video_token)
        self.video_start_id = tokenizer.convert_tokens_to_ids("<|begin_of_video|>")
        self.video_end_id = tokenizer.convert_tokens_to_ids("<|end_of_video|>")

    @classmethod
    def from_pretrained(cls, source: str | Path, **kwargs: Any) -> Glm5NextProcessor:
        config = _processor_config(source, **kwargs)
        tokenizer = kwargs.pop("tokenizer", None)
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer_kwargs = {
                key: kwargs[key]
                for key in (
                    "revision",
                    "cache_dir",
                    "force_download",
                    "local_files_only",
                    "token",
                    "trust_remote_code",
                )
                if key in kwargs and kwargs[key] is not None
            }
            tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
        template_path = _resolve_file(source, "chat_template.jinja", **kwargs)
        template = (
            template_path.read_text()
            if template_path is not None
            else getattr(tokenizer, "chat_template", None)
        )
        return cls(
            Glm5NextImageProcessor(**config.get("image_processor", {})),
            tokenizer,
            Glm5NextVideoProcessor(**config.get("video_processor", {})),
            chat_template=template,
        )

    @property
    def model_input_names(self) -> list[str]:
        return list(
            dict.fromkeys(
                getattr(
                    self.tokenizer, "model_input_names", ["input_ids", "attention_mask"]
                )
                + self.image_processor.model_input_names
                + self.video_processor.model_input_names
                + ["mm_token_type_ids"]
            )
        )

    def replace_image_token(
        self, image_inputs: dict, image_idx: int, **kwargs: Any
    ) -> str:
        count = (
            int(np.prod(_tolist(image_inputs["image_grid_thw"])[image_idx]))
            // self.image_processor.merge_size**2
        )
        return self.image_token * count

    def replace_frame_token_id(
        self, timestamp_sec: float, num_image_tokens: int = 1
    ) -> str:
        return f"<|begin_of_image|>{self.image_token * num_image_tokens}<|end_of_image|>{timestamp_sec:.1f} seconds"

    def replace_video_token(
        self, video_inputs: dict, video_idx: int, **kwargs: Any
    ) -> str:
        grid = _tolist(video_inputs["video_grid_thw"])[video_idx]
        frames = int(grid[0])
        image_tokens = (
            int(np.prod(grid)) // self.video_processor.merge_size**2 // frames
        )
        metadata = video_inputs["video_metadata"][video_idx]
        fps = metadata.fps or 24
        timestamps = (
            metadata.timestamps or [index / fps for index in range(frames * 2)]
        )[::2]
        selected = list(timestamps[:frames])
        while len(selected) < frames:
            selected.append(selected[-1] if selected else 0)
        return "".join(
            self.replace_frame_token_id(value, image_tokens) for value in selected
        )

    def create_mm_token_type_ids(self, input_ids: list) -> list[list[int]]:
        output = []
        for row in input_ids:
            ids = np.asarray(row)
            in_video = np.cumsum(ids == self.video_start_id) > np.cumsum(
                ids == self.video_end_id
            )
            types = np.zeros_like(ids)
            types[(ids == self.image_token_id) & ~in_video] = 1
            types[(ids == self.image_token_id) & in_video] = 2
            output.append(types.tolist())
        return output

    @staticmethod
    def _replace_exact(
        texts: list[str], token: str, replacements: list[str]
    ) -> list[str]:
        output, index = [], 0
        for text in texts:
            pieces = text.split(token)
            needed = len(pieces) - 1
            if index + needed > len(replacements):
                raise ValueError(f"more {token} placeholders than media inputs")
            rebuilt = pieces[0]
            for piece in pieces[1:]:
                rebuilt += replacements[index] + piece
                index += 1
            output.append(rebuilt)
        if index != len(replacements):
            raise ValueError(
                f"{len(replacements) - index} media inputs have no {token} placeholder"
            )
        return output

    def __call__(
        self,
        text: str | list[str] | None = None,
        images: Any = None,
        videos: Any = None,
        *,
        video_metadata: list[VideoMetadata] | VideoMetadata | None = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> ProcessorBatch:
        texts = [text or ""] if isinstance(text, (str, type(None))) else list(text)
        image_inputs = (
            self.image_processor(images, return_tensors=None)
            if images is not None
            else None
        )
        video_inputs = (
            self.video_processor(
                videos,
                video_metadata=video_metadata,
                do_sample_frames=kwargs.pop("do_sample_frames", False),
                return_tensors=None,
            )
            if videos is not None
            else None
        )
        if image_inputs is not None:
            replacements = [
                self.replace_image_token(image_inputs, index)
                for index in range(len(image_inputs["image_grid_thw"]))
            ]
            texts = self._replace_exact(texts, self.image_token, replacements)
        if video_inputs is not None:
            replacements = [
                self.replace_video_token(video_inputs, index)
                for index in range(len(video_inputs["video_grid_thw"]))
            ]
            texts = self._replace_exact(texts, self.video_token, replacements)
        tokenizer_keys = {"padding", "truncation", "max_length", "add_special_tokens"}
        tokenizer_kwargs = {
            key: value for key, value in kwargs.items() if key in tokenizer_keys
        }
        encoded = self.tokenizer(texts, return_tensors=None, **tokenizer_kwargs)
        data = dict(encoded)
        ids = _tolist(data["input_ids"])
        if ids and isinstance(ids[0], int):
            ids = [ids]
        data["input_ids"] = np.asarray(ids, dtype=np.int64)
        if "attention_mask" in data:
            data["attention_mask"] = np.asarray(
                _tolist(data["attention_mask"]), dtype=np.int64
            )
        else:
            data["attention_mask"] = np.ones_like(data["input_ids"])
        data["mm_token_type_ids"] = np.asarray(
            self.create_mm_token_type_ids(ids), dtype=np.int64
        )
        if image_inputs is not None:
            data.update(image_inputs)
        if video_inputs is not None:
            data.update(video_inputs)
        return _convert_batch(data, return_tensors)

    def apply_chat_template(
        self,
        conversation: list[dict[str, Any]],
        *,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        return_dict: bool = False,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> Any:
        images, videos, rendered_messages = [], [], []
        for message in conversation:
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") in ("image", "image_url"):
                        value = item.get(
                            "image", item.get("url", item.get("image_url"))
                        )
                        if isinstance(value, dict):
                            value = value.get("url")
                        images.append(value)
                        parts.append(
                            f"<|begin_of_image|>{self.image_token}<|end_of_image|>"
                        )
                    elif item.get("type") in ("video", "video_url"):
                        value = item.get(
                            "video", item.get("url", item.get("video_url"))
                        )
                        if isinstance(value, dict):
                            value = value.get("url")
                        videos.append(value)
                        parts.append(
                            f"<|begin_of_video|>{self.video_token}<|end_of_video|>"
                        )
                content = "".join(parts)
            rendered_messages.append({**message, "content": content})
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                rendered_messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                chat_template=self.chat_template,
                **{
                    key: value
                    for key, value in kwargs.items()
                    if key not in {"video_metadata", "do_sample_frames"}
                },
            )
        else:
            prompt = "".join(
                f"<|{item['role']}|>{item['content']}" for item in rendered_messages
            )
            if add_generation_prompt:
                prompt += "<|assistant|>"
        if not tokenize:
            return prompt
        output = self(
            prompt,
            images=images or None,
            videos=videos or None,
            video_metadata=kwargs.get("video_metadata"),
            do_sample_frames=kwargs.get("do_sample_frames", False),
            return_tensors=return_tensors,
        )
        return output if return_dict else output["input_ids"]

    def post_process_image_text_to_text(
        self, generated_outputs: Any, **kwargs: Any
    ) -> Any:
        return self.tokenizer.batch_decode(generated_outputs, **kwargs)


def install_glm5_next_processor_namespace() -> bool:
    """Install exact namespace aliases only when upstream lacks GLM5 support."""

    try:
        __import__("transformers.models.glm5_next.processing_glm5_next")
        return False
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        import transformers
        import transformers.models
    except ImportError:
        return False

    package_name = "transformers.models.glm5_next"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = []
        package.__package__ = package_name
        sys.modules[package_name] = package
        transformers.models.glm5_next = package

    modules = {
        "processing_glm5_next": {"Glm5NextProcessor": Glm5NextProcessor},
        "image_processing_glm5_next": {
            "Glm5NextImageProcessor": Glm5NextImageProcessor
        },
        "video_processing_glm5_next": {
            "Glm5NextVideoProcessor": Glm5NextVideoProcessor
        },
    }
    for suffix, members in modules.items():
        name = package_name + "." + suffix
        module = types.ModuleType(name)
        module.__package__ = package_name
        module.__file__ = __file__
        module.__dict__.update(members)
        module.__all__ = list(members)
        sys.modules[name] = module
        setattr(package, suffix, module)
        for member_name, member in members.items():
            member.__module__ = name
            setattr(package, member_name, member)
            setattr(transformers, member_name, member)

    try:
        from transformers.models.auto.image_processing_auto import (
            IMAGE_PROCESSOR_MAPPING_NAMES,
        )
        from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES
        from transformers.models.auto.video_processing_auto import (
            VIDEO_PROCESSOR_MAPPING_NAMES,
        )

        PROCESSOR_MAPPING_NAMES.setdefault("glm5_next", "Glm5NextProcessor")
        IMAGE_PROCESSOR_MAPPING_NAMES.setdefault(
            "glm5_next",
            {"pil": "Glm5NextImageProcessor", "torchvision": "Glm5NextImageProcessor"},
        )
        VIDEO_PROCESSOR_MAPPING_NAMES.setdefault("glm5_next", "Glm5NextVideoProcessor")
    except (ImportError, AttributeError):
        pass
    return True


def load_glm5_next_processor(source: str | Path, **kwargs: Any) -> Glm5NextProcessor:
    install_glm5_next_processor_namespace()
    return Glm5NextProcessor.from_pretrained(source, **kwargs)


__all__ = [
    "Glm5NextImageProcessor",
    "Glm5NextProcessor",
    "Glm5NextVideoProcessor",
    "ProcessorBatch",
    "TRANSFORMERS_PROCESSOR_REVISION",
    "VideoMetadata",
    "install_glm5_next_processor_namespace",
    "load_glm5_next_processor",
    "smart_resize",
]
