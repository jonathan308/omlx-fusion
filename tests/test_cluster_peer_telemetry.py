# SPDX-License-Identifier: Apache-2.0
"""Per-host serving telemetry rides the peer-health channel.

The dashboard's cluster tab shows every rank's marker-published state —
memory, throughput, cache — from the one read the heartbeat already does.
"""

import json
import os
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import routes
from omlx.cluster.liveness import (
    PeerHealth,
    check_peers,
    marker_telemetry_digest,
    read_marker,
)

HOSTS = {0: ("test-mbp", "127.0.0.1"), 1: ("mac-studio", "Studio.local")}


def _marker(state_dir, deployment, rank, **extra):
    payload = {
        "schema_version": 1,
        "deployment_id": deployment,
        "pid": os.getpid(),
        "rank": rank,
        "world_size": 2,
        "model": "test-model",
        "backend": "jaccl",
        "plan_hash": "0" * 64,
        "phase": "ready",
        "updated_at": datetime.now(UTC).isoformat(),
    }
    payload.update(extra)
    (state_dir / f"{deployment}-rank-{rank}.json").write_text(json.dumps(payload))


def _full_metrics():
    return {
        "scope": "end_to_end_pipeline",
        "active_requests": 1,
        "requests_completed": 41,
        "requests_failed": 0,
        "requests_cancelled": 2,
        "prompt_tokens_total": 100_000,
        "completion_tokens_total": 5_000,
        "cached_tokens_total": 60_000,
        "aggregate_decode_tps": 23.5,
        "cache": {
            "affinity": "none",
            "lookups": 10,
            "hits": 6,
            "misses": 4,
            "hit_rate": 0.6,
            "tokens_reused": 60_000,
            "entries": 3,
            "bytes": 1 << 30,
        },
        "pipeline": {
            "batch_steps": 500,
            "busy_seconds": 100.0,
            "idle_seconds": 900.0,
            "utilization": 0.1,
            "microbatch_target": 1,
            "async_overlap": False,
            "last_batch": None,
        },
        "last_request": {
            "status": "running",
            "prompt_tokens": 2_048,
            "cached_tokens": 1_024,
            "completion_tokens": 127,
            "elapsed_seconds": 9.5,
            "ttft_seconds": 1.5,
            "prefill_tps": 680.0,
            "decode_tps": 25.0,
            "end_to_end_tps": 13.4,
        },
    }


def test_the_digest_keeps_only_bounded_known_fields():
    digest = marker_telemetry_digest(
        {
            "load_stage": "ready",
            "start_layer": 0,
            "end_layer": 56,
            "measured_weight_bytes": 77 * 1024**3,
            "load_memory_bytes": 90 * 1024**3,
            "wired_limit_bytes": 110 * 1024**3,
            "capacity_bytes": 115 * 1024**3,
            # Not part of the digest contract: dropped, not forwarded.
            "assignments": [{"node_id": "x"}],
            "performance_profiles": [{"collective_latency_seconds": 1}],
            "error": "should not leak",
        }
        | {"metrics": _full_metrics()}
    )

    assert digest["load_stage"] == "ready"
    assert digest["start_layer"] == 0
    assert digest["end_layer"] == 56
    assert digest["measured_weight_bytes"] == 77 * 1024**3
    assert digest["load_memory_bytes"] == 90 * 1024**3
    assert digest["wired_limit_bytes"] == 110 * 1024**3
    assert digest["capacity_bytes"] == 115 * 1024**3
    assert "assignments" not in digest
    assert "performance_profiles" not in digest
    assert "error" not in digest
    metrics = digest["metrics"]
    assert metrics["active_requests"] == 1
    assert metrics["requests_completed"] == 41
    assert metrics["requests_cancelled"] == 2
    assert metrics["aggregate_decode_tps"] == 23.5
    assert metrics["cache"]["hit_rate"] == 0.6
    assert metrics["cache"]["tokens_reused"] == 60_000
    assert metrics["pipeline"]["utilization"] == 0.1
    assert metrics["last_request"]["decode_tps"] == 25.0
    assert metrics["last_request"]["status"] == "running"


def test_the_digest_is_fail_soft_per_field():
    digest = marker_telemetry_digest(
        {
            "load_stage": 42,  # wrong type: dropped
            "measured_weight_bytes": -5,  # negative: dropped
            "capacity_bytes": True,  # bool is not a counter: dropped
            "planned_weight_bytes": 10,
            "metrics": {
                "active_requests": 2,
                "aggregate_decode_tps": float("nan"),  # non-finite: dropped
                "cache": {"hit_rate": 7.5},  # out of range: dropped
                "last_request": {"status": "melted"},  # unknown state: dropped
            },
        }
    )

    assert digest == {
        "planned_weight_bytes": 10,
        "metrics": {"active_requests": 2},
    }


def test_the_digest_tolerates_an_empty_marker():
    assert marker_telemetry_digest({}) == {}
    assert marker_telemetry_digest({"metrics": "not-a-dict"}) == {}


def test_check_peers_attaches_telemetry_only_when_asked(tmp_path):
    _marker(tmp_path, "d", 0, measured_weight_bytes=123, metrics=_full_metrics())
    _marker(tmp_path, "d", 1, measured_weight_bytes=456)

    def remote_reader(_target, path):
        marker = read_marker(tmp_path / os.path.basename(path))
        return marker, True, datetime.now(UTC).timestamp(), ""

    plain = check_peers(
        HOSTS,
        state_dir=str(tmp_path),
        deployment_id="d",
        probe=lambda target: True,
        remote_reader=remote_reader,
        require_heartbeat=True,
    )
    assert all(item.telemetry is None for item in plain)
    assert all("telemetry" not in item.to_dict() for item in plain)

    enriched = check_peers(
        HOSTS,
        state_dir=str(tmp_path),
        deployment_id="d",
        probe=lambda target: True,
        remote_reader=remote_reader,
        require_heartbeat=True,
        include_telemetry=True,
    )
    by_rank = {item.rank: item for item in enriched}
    assert by_rank[0].telemetry["measured_weight_bytes"] == 123
    assert by_rank[0].telemetry["metrics"]["cache"]["hit_rate"] == 0.6
    assert by_rank[1].telemetry["measured_weight_bytes"] == 456
    assert "metrics" not in by_rank[1].telemetry
    assert by_rank[0].to_dict()["telemetry"] == by_rank[0].telemetry


def test_check_peers_telemetry_survives_a_missing_marker(tmp_path):
    _marker(tmp_path, "d", 0, measured_weight_bytes=123)

    health = check_peers(
        {0: ("test-mbp", "127.0.0.1"), 1: ("gone", "192.0.2.1")},
        state_dir=str(tmp_path),
        deployment_id="d",
        probe=lambda target: target == "127.0.0.1",
        require_heartbeat=True,
        include_telemetry=True,
    )
    by_rank = {item.rank: item for item in health}
    assert by_rank[0].telemetry is not None
    assert by_rank[1].telemetry is None
    assert by_rank[1].healthy is False


def test_peer_health_route_forwards_the_telemetry_flag(monkeypatch):
    captured = {}

    def fake_check_peers(hosts_by_rank, **kwargs):
        captured.update(kwargs)
        return (
            PeerHealth(
                node_id="test-mbp",
                rank=0,
                reachable=True,
                seconds_since_heartbeat=1.0,
                phase="ready",
                heartbeat_required=True,
                telemetry={"measured_weight_bytes": 77},
            ),
        )

    monkeypatch.setattr(routes, "check_peers", fake_check_peers)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)

    response = client.get(
        "/admin/api/cluster/peer-health",
        params={
            "hosts": "127.0.0.1,studio.local",
            "deployment_id": "d",
            "include_telemetry": "true",
        },
    )
    assert response.status_code == 200
    assert captured["include_telemetry"] is True
    assert captured["require_heartbeat"] is True
    payload = response.json()
    assert payload["peers"][0]["telemetry"] == {"measured_weight_bytes": 77}

    captured.clear()
    response = client.get(
        "/admin/api/cluster/peer-health",
        params={"hosts": "127.0.0.1,studio.local"},
    )
    assert response.status_code == 200
    assert captured["include_telemetry"] is False
