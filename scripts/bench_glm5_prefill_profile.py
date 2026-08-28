#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cold, zero-cache GLM-5.3 DFlash prefill profiler A/B.

This script is intentionally not part of CI and is not run by the profiler
patch.  Each A/B arm creates and stops a fresh local DFlashEngine; it never
contacts or controls an oMLX server.  The profiled arm enables target-forward
telemetry only.  Compare its ``final_eval_ms`` with benchmark TTFT/prefill wall
to quantify DFlash work outside the target forward (feature concat/projection,
feature-store write, and runtime boundary clear).

Example (M3 only):

    python scripts/bench_glm5_prefill_profile.py \
      --target /path/to/GLM-5.3-Flash-oQ4e \
      --draft /path/to/GLM-5.3-Flash-DFlash2 \
      --i-understand-local-model-load
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="GLM-5.3 target model path")
    parser.add_argument("--draft", required=True, help="GLM DFlash2 draft path")
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[2000, 10000, 20000],
        help="Exact zero-cache prompt lengths (default: 2000 10000 20000)",
    )
    parser.add_argument("--generation-length", type=int, default=8)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--i-understand-local-model-load",
        action="store_true",
        help="Required acknowledgement: both A/B arms load the models locally",
    )
    return parser.parse_args()


def _hardware_guard() -> str:
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit("This physical profile is restricted to Apple Silicon.")
    result = subprocess.run(
        ["/usr/sbin/system_profiler", "SPHardwareDataType"],
        check=True,
        capture_output=True,
        text=True,
    )
    chip = next(
        (
            line.split(":", 1)[1].strip()
            for line in result.stdout.splitlines()
            if line.strip().startswith("Chip:")
        ),
        "unknown",
    )
    if "M5" in chip:
        raise SystemExit(
            "Refusing M5: the qualified MLX base predates the sorted NAX "
            "gather_qmm >32768-row fix required for trustworthy long prefill."
        )
    return chip


def _engine_settings() -> SimpleNamespace:
    # Disable both DFlash cache tiers. _run_single_test additionally requests
    # skip_cache_store and raises if the engine reports cached_tokens != 0.
    return SimpleNamespace(
        dflash_max_ctx=None,
        dflash_in_memory_cache=False,
        dflash_in_memory_cache_max_entries=0,
        dflash_in_memory_cache_max_bytes=0,
        dflash_ssd_cache=False,
        dflash_ssd_cache_max_bytes=0,
        dflash_draft_sink_size=0,
        dflash_block_size=None,
        dflash_verify_mode=None,
    )


async def _run_arm(
    *,
    label: str,
    target: str,
    draft: str,
    lengths: list[int],
    generation_length: int,
    output_dir: Path,
) -> list[dict]:
    from omlx.admin.benchmark import _generate_prompt, _run_single_test
    from omlx.engine.dflash import DFlashEngine

    enabled = label == "profile"
    os.environ["OMLX_GLM5_PREFILL_PROFILE"] = "1" if enabled else "0"
    os.environ["OMLX_GLM5_PREFILL_PROFILE_WARMUP"] = "0"
    os.environ["OMLX_GLM5_PREFILL_PROFILE_INTERVAL"] = "1"

    profile_log = output_dir / f"{label}.glm-target-profile.jsonl"
    handler = logging.FileHandler(profile_log, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    glm_logger = logging.getLogger(
        "mlx_vlm.models.glm5_next.language"
    )
    glm_logger.addHandler(handler)
    glm_logger.setLevel(logging.INFO)

    engine = DFlashEngine(
        target,
        draft,
        model_settings=_engine_settings(),
        fallback_engine_type="vlm",
    )
    results = []
    try:
        await engine.start()
        tokenizer = engine.tokenizer
        prompts = {
            length: _generate_prompt(tokenizer, length)
            for length in lengths
        }
        # Deliberately no prompt warmup: this is the requested cold zero-cache
        # ladder, and each A/B arm starts from a fresh engine/model load.
        for length in lengths:
            result = await _run_single_test(
                engine,
                prompts[length],
                generation_length,
                length,
            )
            result = dict(result)
            result["arm"] = label
            result["requested_pp"] = length
            results.append(result)
            print(
                f"{label:8s} pp={length:6d} "
                f"ttft={result.get('ttft_ms')}ms "
                f"pp={result.get('processing_tps')} tok/s",
                flush=True,
            )
    finally:
        await engine.stop()
        glm_logger.removeHandler(handler)
        handler.close()
    return results


async def _main() -> None:
    args = _arguments()
    if not args.i_understand_local_model_load:
        raise SystemExit("Pass --i-understand-local-model-load to run this script.")
    if any(length <= 1 for length in args.lengths):
        raise SystemExit("Every prefill length must be greater than one token.")
    chip = _hardware_guard()
    target = str(Path(args.target).expanduser().resolve())
    draft = str(Path(args.draft).expanduser().resolve())
    for path in (target, draft):
        if not Path(path).is_dir():
            raise SystemExit(f"Model directory does not exist: {path}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir or Path(f"glm-prefill-profile-{stamp}")
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "chip": chip,
        "target": target,
        "draft": draft,
        "lengths": args.lengths,
        "generation_length": args.generation_length,
        "cache": "disabled; benchmark additionally requires cached_tokens=0",
        "profile_scope": "glm_target_forward",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    all_results = []
    for label in ("baseline", "profile"):
        all_results.extend(
            await _run_arm(
                label=label,
                target=target,
                draft=draft,
                lengths=args.lengths,
                generation_length=args.generation_length,
                output_dir=output_dir,
            )
        )
    (output_dir / "results.json").write_text(
        json.dumps(all_results, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_main())

