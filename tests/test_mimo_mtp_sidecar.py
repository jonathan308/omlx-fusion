# SPDX-License-Identifier: Apache-2.0
"""A MiMo MTP sidecar under <model>/mtp/ reaches the loader from the batched engine."""

from omlx.engine import batched as batched_mod
from omlx.utils import model_loading


def test_sidecar_config_only_when_the_file_exists(tmp_path):
    assert model_loading.mimo_mtp_sidecar_config(tmp_path) is None
    sidecar = tmp_path / "mtp" / "model_mtp.safetensors"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"")
    assert model_loading.mimo_mtp_sidecar_config(str(tmp_path)) == {
        "omlx_mtp_sidecar": str(sidecar)
    }


def test_batched_engine_load_kwargs_carry_the_sidecar(tmp_path):
    assert batched_mod._mtp_sidecar_load_kwargs(str(tmp_path)) == {}
    sidecar = tmp_path / "mtp" / "model_mtp.safetensors"
    sidecar.parent.mkdir()
    sidecar.write_bytes(b"")
    assert batched_mod._mtp_sidecar_load_kwargs(str(tmp_path)) == {
        "model_config": {"omlx_mtp_sidecar": str(sidecar)}
    }
