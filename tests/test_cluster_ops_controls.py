# SPDX-License-Identifier: Apache-2.0
"""Dashboard cluster operations: stop-generation, warmup, known-answer gate.

Every control rides an existing engine/admin path — the lockstep-cancel-aware
``abort_all_requests``, supervision's sacrificial warmup, and the packaged
GPU-vs-CPU gate — these tests pin the wiring, not the machinery.
"""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import routes
from omlx.cluster.supervision import WarmupFailedError


class _DistributedEngine:
    def __init__(self, *, aborted=0, endpoint="http://127.0.0.1:18099", phase="ready"):
        self.aborted = aborted
        self.abort_calls = 0
        self.endpoint = endpoint
        self.phase = phase

    async def abort_all_requests(self):
        self.abort_calls += 1
        return self.aborted

    def cluster_status(self):
        return {
            "deployment_id": "dep-1",
            "phase": self.phase,
            "endpoint": self.endpoint,
        }


class _SingleNodeEngine:
    """A loaded non-distributed engine: never a cluster stop/warmup target."""

    async def abort_all_requests(self):  # pragma: no cover - must not run
        raise AssertionError("single-node engines are out of scope here")


class _Pool:
    def __init__(self, entries):
        self._entries = entries

    def get_loaded_model_ids(self):
        return list(self._entries)

    def get_entry(self, model_id):
        return self._entries[model_id]


def _client(pool=None):
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def _use_pool(monkeypatch, pool):
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)


def test_the_new_controls_live_on_the_admin_router_not_the_peer_router():
    admin_paths = {route.path for route in routes.router.routes}
    peer_paths = {route.path for route in routes.peer_router.routes}
    for path in (
        "/admin/api/cluster/stop-generation",
        "/admin/api/cluster/warmup",
        "/admin/api/cluster/known-answer",
    ):
        assert path in admin_paths
        assert path not in peer_paths


def test_stop_generation_aborts_every_loaded_distributed_engine(monkeypatch):
    engine = _DistributedEngine(aborted=3)
    pool = _Pool(
        {
            "cluster-model": SimpleNamespace(engine=engine),
            "local-model": SimpleNamespace(engine=_SingleNodeEngine()),
        }
    )
    _use_pool(monkeypatch, pool)

    response = _client().post("/admin/api/cluster/stop-generation")

    assert response.status_code == 200
    payload = response.json()
    assert payload["aborted"] == 3
    assert payload["engines"] == 1
    assert payload["deployment_ids"] == ["dep-1"]
    assert engine.abort_calls == 1


def test_stop_generation_without_a_deployment_is_a_stated_409(monkeypatch):
    _use_pool(monkeypatch, _Pool({}))

    response = _client().post("/admin/api/cluster/stop-generation")

    assert response.status_code == 409
    assert "no distributed deployment" in response.json()["detail"]


def test_warmup_runs_the_supervision_path_against_the_live_endpoint(monkeypatch):
    engine = _DistributedEngine()
    _use_pool(monkeypatch, _Pool({"cluster-model": SimpleNamespace(engine=engine)}))
    calls = {}

    def fake_warmup(endpoint, **kwargs):
        calls["endpoint"] = endpoint
        calls.update(kwargs)
        return {"ok": True, "completion_tokens": 8, "prompt_tokens": 9}

    monkeypatch.setattr(
        "omlx.cluster.supervision.run_startup_warmup", fake_warmup
    )

    response = _client().post(
        "/admin/api/cluster/warmup", params={"max_tokens": 4, "timeout": 30}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["completion_tokens"] == 8
    assert payload["model_id"] == "cluster-model"
    assert payload["deployment_id"] == "dep-1"
    assert calls == {
        "endpoint": "http://127.0.0.1:18099",
        "max_tokens": 4,
        "timeout_s": 30.0,
    }


def test_warmup_refuses_a_deployment_that_is_not_serving(monkeypatch):
    _use_pool(
        monkeypatch,
        _Pool(
            {
                "cluster-model": SimpleNamespace(
                    engine=_DistributedEngine(phase="loading", endpoint=None)
                )
            }
        ),
    )

    response = _client().post("/admin/api/cluster/warmup")

    assert response.status_code == 409
    assert "no serving endpoint" in response.json()["detail"]


def test_warmup_refuses_a_deployment_that_is_not_ready(monkeypatch):
    _use_pool(
        monkeypatch,
        _Pool({"cluster-model": SimpleNamespace(engine=_DistributedEngine(phase="loading"))}),
    )

    response = _client().post("/admin/api/cluster/warmup")

    assert response.status_code == 409
    assert "not ready" in response.json()["detail"]


def test_warmup_failure_is_a_502_with_the_reason(monkeypatch):
    _use_pool(
        monkeypatch,
        _Pool({"cluster-model": SimpleNamespace(engine=_DistributedEngine())}),
    )

    def explode(endpoint, **kwargs):
        raise WarmupFailedError("warmup generation returned no choices")

    monkeypatch.setattr("omlx.cluster.supervision.run_startup_warmup", explode)

    response = _client().post("/admin/api/cluster/warmup")

    assert response.status_code == 502
    assert "no choices" in response.json()["detail"]


def test_warmup_validates_bounds(monkeypatch):
    _use_pool(
        monkeypatch,
        _Pool({"cluster-model": SimpleNamespace(engine=_DistributedEngine())}),
    )
    client = _client()

    assert client.post("/admin/api/cluster/warmup", params={"max_tokens": 0}).status_code == 422
    assert client.post("/admin/api/cluster/warmup", params={"max_tokens": 4096}).status_code == 422
    assert client.post("/admin/api/cluster/warmup", params={"timeout": 0}).status_code == 422


def test_known_answer_returns_the_gate_verdict(monkeypatch):
    monkeypatch.setattr(
        "omlx.custom_kernels.known_answer.run_checks",
        lambda: {
            "ok": True,
            "mlx_version": "0.32.0",
            "report": ["    matmul_fp32: rel=1e-7"],
            "failures": [],
        },
    )

    response = _client().post("/admin/api/cluster/known-answer")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["mlx_version"] == "0.32.0"
    assert payload["failures"] == []


def test_known_answer_reports_divergence_and_crash(monkeypatch):
    monkeypatch.setattr(
        "omlx.custom_kernels.known_answer.run_checks",
        lambda: {
            "ok": False,
            "mlx_version": "0.32.0",
            "report": [],
            "failures": ["quantized_matmul_4bit gpu/cpu divergence"],
        },
    )
    response = _client().post("/admin/api/cluster/known-answer")
    assert response.status_code == 200
    assert response.json()["ok"] is False

    def explode():
        raise RuntimeError("mlx.core is unavailable")

    monkeypatch.setattr("omlx.custom_kernels.known_answer.run_checks", explode)
    response = _client().post("/admin/api/cluster/known-answer")
    assert response.status_code == 500
    assert "could not run" in response.json()["detail"]


def test_controls_503_until_the_server_is_initialized(monkeypatch):
    monkeypatch.setattr(routes, "_get_engine_pool", None)
    client = _client()

    assert client.post("/admin/api/cluster/stop-generation").status_code == 503
    assert client.post("/admin/api/cluster/warmup").status_code == 503
