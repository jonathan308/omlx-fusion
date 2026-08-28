# SPDX-License-Identifier: Apache-2.0
"""``mlx_lm.models.glm5_next`` backed by oMLX's native GLM implementation."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.mlx_vlm_glm5_next_compat import (
    apply_mlx_vlm_glm5_next_compat_patch,
)

apply_mlx_vlm_glm5_next_compat_patch()

from omlx.patches.mlx_lm_glm5_next.pipeline_patch import (  # noqa: E402
    apply_glm5_pipeline_patch,
)

apply_glm5_pipeline_patch()

from mlx_vlm.models.glm5_next.config import ModelConfig as _ModelConfig  # noqa: E402
from mlx_vlm.models.glm5_next.language import (  # noqa: E402
    LanguageModel as _LanguageModel,
)
from mlx_vlm.models.glm5_next.linear import linear_forward  # noqa: E402

ModelArgs = _ModelConfig
SUPPORTS_PIPELINE = True
HONORS_PIPELINE_ASSIGNMENT = True


class Model(nn.Module):
    """Text-only GLM-5.3 wrapper with direct checkpoint parameter roots."""

    _omlx_supports_rank_zero_logits = True

    def __init__(self, args: _ModelConfig) -> None:
        super().__init__()
        self.config = args
        self.args = args.text_config
        self.model_type = str(args.model_type or "glm5_next")
        # Direct ``language_model.*`` root matches converted checkpoint indexes,
        # allowing selective stage-file discovery before any tensor is loaded.
        self.language_model = _LanguageModel(args.text_config, args)

    @property
    def model(self) -> Any:
        return self.language_model.model

    @property
    def layers(self) -> Any:
        return self.model.pipeline_layers

    @property
    def _omlx_output_vocab_size(self) -> int:
        return int(self.args.vocab_size)

    @property
    def head_dim(self) -> int:
        return int(getattr(self.args, "qk_head_dim", 0) or 0)

    @property
    def n_kv_heads(self) -> int:
        return int(getattr(self.args, "num_key_value_heads", 1) or 1)

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    def make_cache(self) -> list[Any]:
        return self.language_model.make_cache()

    def __call__(
        self,
        inputs: mx.array,
        cache: Any = None,
        mask: Any = None,
        skip_logits: bool = False,
        **kwargs: Any,
    ) -> mx.array | None:
        del mask
        hidden = self.model(
            inputs,
            cache=cache,
            inputs_embeds=kwargs.get("inputs_embeds"),
        )
        if skip_logits:
            return None
        keep = int(kwargs.get("num_logits_to_keep", 0) or 0)
        if keep:
            hidden = hidden[:, -keep:, :]
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return linear_forward(self.language_model.lm_head, hidden)

    def sanitize(self, weights: dict[str, mx.array]) -> dict[str, mx.array]:
        """Delegate text sanitization while dropping unsupported vision state."""

        language: dict[str, mx.array] = {}
        for key, value in weights.items():
            normalized = key
            if normalized.startswith("model.language_model."):
                normalized = "language_model.model." + normalized[len("model.language_model.") :]
            elif normalized.startswith("lm_head."):
                normalized = "language_model." + normalized
            if normalized.startswith("language_model."):
                language[normalized[len("language_model.") :]] = value
        language = self.language_model.sanitize(language)
        return {f"language_model.{key}": value for key, value in language.items()}
