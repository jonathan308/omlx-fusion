# SPDX-License-Identifier: Apache-2.0
"""Single-node MLX command-buffer caps: seeded only when absent, opt-out."""

import os
import subprocess
import sys

from omlx.settings import (
    METAL_COMMAND_BUFFER_CAPS_ENV,
    METAL_COMMAND_BUFFER_DEFAULTS,
    apply_metal_command_buffer_defaults,
)


def test_defaults_match_the_distributed_hostfile_values():
    # cluster/deployment.py injects these for every rank with the
    # kIOGPUCommandBufferCallbackErrorTimeout rationale; the single-node
    # seeding must not drift from them.
    assert METAL_COMMAND_BUFFER_DEFAULTS == {
        "MLX_MAX_OPS_PER_BUFFER": "16",
        "MLX_MAX_MB_PER_BUFFER": "512",
    }


def test_seeded_only_when_absent():
    environ: dict[str, str] = {}
    applied = apply_metal_command_buffer_defaults(environ)
    assert applied == METAL_COMMAND_BUFFER_DEFAULTS
    assert environ["MLX_MAX_OPS_PER_BUFFER"] == "16"
    assert environ["MLX_MAX_MB_PER_BUFFER"] == "512"


def test_operator_values_always_win():
    environ = {
        "MLX_MAX_OPS_PER_BUFFER": "2",
        "MLX_MAX_MB_PER_BUFFER": "128",
    }
    applied = apply_metal_command_buffer_defaults(environ)
    assert applied == {}
    assert environ["MLX_MAX_OPS_PER_BUFFER"] == "2"
    assert environ["MLX_MAX_MB_PER_BUFFER"] == "128"


def test_partial_override_keeps_the_operator_half():
    environ = {"MLX_MAX_OPS_PER_BUFFER": "4"}
    applied = apply_metal_command_buffer_defaults(environ)
    assert applied == {"MLX_MAX_MB_PER_BUFFER": "512"}
    assert environ["MLX_MAX_OPS_PER_BUFFER"] == "4"


def test_killswitch_restores_previous_behavior_exactly():
    environ = {METAL_COMMAND_BUFFER_CAPS_ENV: "0"}
    applied = apply_metal_command_buffer_defaults(environ)
    assert applied == {}
    assert "MLX_MAX_OPS_PER_BUFFER" not in environ
    assert "MLX_MAX_MB_PER_BUFFER" not in environ


def test_killswitch_accepts_other_false_spellings():
    for spelling in ("false", "OFF", " no "):
        environ = {METAL_COMMAND_BUFFER_CAPS_ENV: spelling}
        assert apply_metal_command_buffer_defaults(environ) == {}


def test_killswitch_true_spellings_still_seed():
    for spelling in ("1", "true", "yes", "on"):
        environ = {METAL_COMMAND_BUFFER_CAPS_ENV: spelling}
        assert apply_metal_command_buffer_defaults(environ) == (
            METAL_COMMAND_BUFFER_DEFAULTS
        )


def test_cli_import_has_no_env_side_effect():
    # Importing the CLI (or any omlx module) must not change the process
    # environment; the seeding belongs to the server entry points only.
    code = (
        "import os, sys;"
        "assert 'MLX_MAX_OPS_PER_BUFFER' not in os.environ;"
        "import omlx.cli, omlx.settings, omlx.cluster.host_tune;"
        "assert 'MLX_MAX_OPS_PER_BUFFER' not in os.environ;"
        "assert 'MLX_MAX_MB_PER_BUFFER' not in os.environ"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("MLX_MAX_")
    }
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert result.returncode == 0, result.stderr
