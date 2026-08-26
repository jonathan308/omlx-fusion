# SPDX-License-Identifier: Apache-2.0
"""omlx speculative-decoding wrappers.

This package collects exact proposal algorithms plus integration code that
bridges omlx scheduling/cache infrastructure with speculative decoding in
mlx-lm and mlx-vlm. Runtime adapters stay isolated so the surface of
internal-API dependencies is easy to audit on each upstream bump.
"""

from .ngram_mod import NGramMod, NGramModSequence

__all__ = ["NGramMod", "NGramModSequence"]
