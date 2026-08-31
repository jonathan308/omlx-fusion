# SPDX-License-Identifier: Apache-2.0
"""Qwen4 scalar-pair physical harness loader routing tests."""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager
from pathlib import Path


def _benchmark_module():
    path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "bench_qwen4_scalar_pair_wavefront.py"
    )
    spec = importlib.util.spec_from_file_location("qwen4_scalar_pair_bench", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_physical_harness_uses_vlm_mmap_loader_and_never_text_loader(monkeypatch):
    import mlx_vlm.utils as mlx_vlm_utils

    from omlx.engine import vlm as engine_vlm
    from omlx.utils import model_loading

    bench = _benchmark_module()
    events = []
    patched = {}
    loaded_model = object()
    loaded_processor = object()

    def prepatch(path, *, model_settings, for_vlm=False):
        patched.update(
            path=path,
            settings=model_settings,
            for_vlm=for_vlm,
        )

    def vlm_load(path, **kwargs):
        events.append(("vlm-load", path, kwargs))
        return loaded_model, loaded_processor

    def forbidden_text_loader(*_args, **_kwargs):
        raise AssertionError("physical Qwen4 harness must never use load_text_model")

    def fake_context(name):
        @contextmanager
        def manager(path):
            events.append(("enter", name, path))
            try:
                yield
            finally:
                events.append(("exit", name, path))

        return manager

    monkeypatch.setattr(model_loading, "maybe_apply_pre_load_patches", prepatch)
    monkeypatch.setattr(model_loading, "load_text_model", forbidden_text_loader)
    monkeypatch.setattr(mlx_vlm_utils, "load", vlm_load)
    context_names = (
        "_strip_audio_config_if_orphaned",
        "_drop_gemma4_mlx_shared_kv_extras_on_load",
        "_force_minimax_m3_moe_sanitize_on_load",
        "_force_qwen4_exp_sanitize_on_load",
        "_remap_nested_visual_on_load",
        "_transpose_qwen35_mlx_vision_patch_embed_on_load",
        "_load_optiq_vision_sidecar_on_load",
    )
    for name in context_names:
        monkeypatch.setattr(engine_vlm, name, fake_context(name))

    model_path = Path("/tmp/mock-qwen4-vlm")
    actual = bench._load_qwen4_vlm(model_path)
    assert actual == (loaded_model, loaded_processor)
    assert patched["path"] == str(model_path)
    assert patched["for_vlm"] is True
    settings = patched["settings"]
    assert settings.mtp_enabled is False
    assert settings.mtp_num_draft_tokens == 0
    assert settings.qwen4_ple_ssd_offload is True
    assert settings.trust_remote_code is False
    assert (
        "vlm-load",
        str(model_path),
        {"lazy": True, "trust_remote_code": False},
    ) in events
    entered = {event[1] for event in events if event[0] == "enter"}
    assert entered == set(context_names)
