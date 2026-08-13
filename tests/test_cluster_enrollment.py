# SPDX-License-Identifier: Apache-2.0
"""Tests for guided cluster enrollment: join tokens, redeem, and the CLI."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx import cli
from omlx.cluster import enrollment, routes
from omlx.cluster.enrollment import (
    JoinConnectionError,
    JoinTokenStore,
    decode_join_token,
    encode_join_token,
    join_cluster,
    normalize_coordinator_url,
    redeem_join_token,
    redeem_payload_json,
    runtime_mismatches,
    validate_version_report,
)
from omlx.cluster.ssh_keys import SSHKeyPair, create_key_exchange_token
from omlx.cluster.token_auth import sign_pairing_payload

EXPECTED = {"omlx": "0.5.3", "mlx": "0.32.0", "mlx-lm": "0.31.3"}


def _public_key(seed: bytes = b"\x01") -> str:
    body = base64.b64encode(seed * 32).decode()
    return f"ssh-ed25519 {body} omlx-cluster"


def _key_pair(public_key: str | None = None) -> SSHKeyPair:
    public_key = public_key or _public_key(b"\x02")
    return SSHKeyPair(
        private_key_path=Path("/tmp/omlx-test-key"),
        public_key_path=Path("/tmp/omlx-test-key.pub"),
        public_key=public_key,
        fingerprint="SHA256:coordinator",
        key_type="ed25519",
        created_at=0.0,
    )


def _signed_redeem(secret: str, versions: dict[str, str], node_id: str = "peer.local"):
    """Build the exact payload a peer's join command would send."""

    exchange_token = create_key_exchange_token(
        public_key=_public_key(),
        node_id=node_id,
        shared_secret=secret,
    )
    signature = sign_pairing_payload(
        redeem_payload_json(exchange_token, versions),
        shared_secret=secret,
    )
    return exchange_token, signature


def _redeem_kwargs(store: JoinTokenStore, **overrides):
    kwargs = {
        "store": store,
        "expected_versions": dict(EXPECTED),
        "install_key": lambda **_: True,
        "coordinator_key_provider": _key_pair,
        "hostname_provider": lambda: "coordinator.local",
    }
    kwargs.update(overrides)
    return kwargs


@pytest.fixture
def store():
    enrollment.reset_join_token_store()
    yield enrollment.get_join_token_store()
    enrollment.reset_join_token_store()


@pytest.fixture
def client(store):
    app = FastAPI()
    app.include_router(routes.router)
    app.include_router(routes.peer_router)
    return TestClient(app)


class TestJoinTokenCodec:
    def test_roundtrip(self):
        encoded = encode_join_token("abc12345", "s" * 43)
        assert decode_join_token(encoded) == ("abc12345", "s" * 43)

    def test_rejects_garbage(self):
        assert decode_join_token("not a token") is None
        assert decode_join_token("") is None
        assert decode_join_token(base64.urlsafe_b64encode(b"{}").decode()) is None

    def test_rejects_wrong_version_and_shapes(self):
        for payload in (
            {"v": 2, "id": "abc12345", "secret": "s" * 43},
            {"v": 1, "id": 42, "secret": "s" * 43},
            {"v": 1, "id": "abc12345", "secret": "short"},
            {"v": 1, "id": "abc12345", "secret": " padded " + "s" * 32},
            {"v": 1, "id": "x", "secret": "s" * 43},
            ["v", 1],
        ):
            encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
            assert decode_join_token(encoded) is None


class TestJoinTokenStore:
    def test_mint_returns_decodable_single_use_token(self, store):
        minted = store.mint()
        token_id, secret = decode_join_token(minted["token"])
        assert token_id == minted["token_id"]
        assert len(secret) == 43  # 256 bits, urlsafe
        assert minted["expires_in_seconds"] == enrollment.JOIN_TOKEN_TTL
        assert store.status(token_id)["status"] == "pending"

    def test_redeem_marks_token_used_and_records_peer(self, store):
        minted = store.mint()
        _, secret = decode_join_token(minted["token"])
        exchange_token, signature = _signed_redeem(secret, EXPECTED)

        result = redeem_join_token(
            token=minted["token"],
            exchange_token=exchange_token,
            versions=EXPECTED,
            signature=signature,
            **_redeem_kwargs(store),
        )

        assert result["success"] is True
        assert result["runtime_compatible"] is True
        assert result["runtime_mismatches"] == []
        assert result["peer_key_installed"] is True
        assert result["coordinator"]["node_id"] == "coordinator.local"
        assert result["coordinator"]["public_key"] == _key_pair().public_key
        peer = result["peer"]
        assert peer["node_id"] == "peer.local"
        assert peer["versions"] == EXPECTED

        status = store.status(minted["token_id"])
        assert status["status"] == "redeemed"
        assert status["peer"]["node_id"] == "peer.local"
        assert store.enrolled()[0]["token_id"] == minted["token_id"]

    def test_replay_is_refused(self, store):
        minted = store.mint()
        _, secret = decode_join_token(minted["token"])
        exchange_token, signature = _signed_redeem(secret, EXPECTED)
        first = redeem_join_token(
            token=minted["token"],
            exchange_token=exchange_token,
            versions=EXPECTED,
            signature=signature,
            **_redeem_kwargs(store),
        )
        assert first["success"] is True

        replay = redeem_join_token(
            token=minted["token"],
            exchange_token=exchange_token,
            versions=EXPECTED,
            signature=signature,
            **_redeem_kwargs(store),
        )
        assert replay["success"] is False
        assert replay["status"] == 409
        assert len(store.enrolled()) == 1

    def test_wrong_signature_is_refused_without_consuming(self, store):
        minted = store.mint()
        _, secret = decode_join_token(minted["token"])
        exchange_token, _ = _signed_redeem(secret, EXPECTED)
        # Sign a *different* payload than the one submitted.
        signature = sign_pairing_payload(
            redeem_payload_json(exchange_token, {"omlx": "9.9.9"}),
            shared_secret=secret,
        )

        result = redeem_join_token(
            token=minted["token"],
            exchange_token=exchange_token,
            versions=EXPECTED,
            signature=signature,
            **_redeem_kwargs(store),
        )
        assert result["success"] is False
        assert result["status"] == 403
        assert store.status(minted["token_id"])["status"] == "pending"

    def test_tampered_token_secret_is_refused(self, store):
        minted = store.mint()
        _, secret = decode_join_token(minted["token"])
        exchange_token, signature = _signed_redeem(secret, EXPECTED)
        forged = encode_join_token(minted["token_id"], "a" * 43)

        result = redeem_join_token(
            token=forged,
            exchange_token=exchange_token,
            versions=EXPECTED,
            signature=signature,
            **_redeem_kwargs(store),
        )
        assert result["success"] is False
        assert result["status"] == 403
        assert store.status(minted["token_id"])["status"] == "pending"

    def test_unknown_token_is_refused(self, store):
        forged = encode_join_token("unknown-id", "s" * 43)
        result = redeem_join_token(
            token=forged,
            exchange_token="whatever",
            versions=EXPECTED,
            signature="0" * 64,
            **_redeem_kwargs(store),
        )
        assert result["success"] is False
        assert result["status"] == 403

    def test_expired_token_is_refused(self):
        now = [1_000.0]
        store = JoinTokenStore(clock=lambda: now[0])
        minted = store.mint()
        _, secret = decode_join_token(minted["token"])
        exchange_token, signature = _signed_redeem(secret, EXPECTED)

        now[0] += enrollment.JOIN_TOKEN_TTL + 1
        result = redeem_join_token(
            token=minted["token"],
            exchange_token=exchange_token,
            versions=EXPECTED,
            signature=signature,
            **_redeem_kwargs(store),
        )
        assert result["success"] is False
        assert result["status"] == 410
        assert store.status(minted["token_id"])["status"] == "expired"

    def test_mint_sweeps_expired_tokens(self):
        now = [1_000.0]
        store = JoinTokenStore(clock=lambda: now[0])
        first = store.mint()
        now[0] += enrollment.JOIN_TOKEN_TTL + 1
        second = store.mint()
        # The expired token was swept by the second mint.
        assert store.status(first["token_id"]) is None
        assert store.status(second["token_id"])["status"] == "pending"

    def test_mint_caps_pending_tokens_by_evicting_oldest(self):
        now = [1_000.0]
        store = JoinTokenStore(clock=lambda: now[0])
        minted = []
        for _ in range(enrollment._MAX_PENDING_TOKENS + 2):
            minted.append(store.mint())
            now[0] += 1  # distinct created_at, well within the TTL
        assert store.status(minted[0]["token_id"]) is None
        assert store.status(minted[1]["token_id"]) is None
        assert store.status(minted[-1]["token_id"])["status"] == "pending"

    def test_infrastructure_failure_rolls_token_back(self, store):
        minted = store.mint()
        _, secret = decode_join_token(minted["token"])
        exchange_token, signature = _signed_redeem(secret, EXPECTED)

        def broken_install(**_kwargs):
            raise OSError("disk full")

        result = redeem_join_token(
            token=minted["token"],
            exchange_token=exchange_token,
            versions=EXPECTED,
            signature=signature,
            **_redeem_kwargs(store, install_key=broken_install),
        )
        assert result["success"] is False
        assert result["status"] == 500
        assert store.status(minted["token_id"])["status"] == "pending"

        retry = redeem_join_token(
            token=minted["token"],
            exchange_token=exchange_token,
            versions=EXPECTED,
            signature=signature,
            **_redeem_kwargs(store),
        )
        assert retry["success"] is True

    def test_invalid_exchange_token_rolls_back(self, store):
        minted = store.mint()
        _, secret = decode_join_token(minted["token"])
        exchange_token = base64.urlsafe_b64encode(b"not-a-key-exchange").decode()
        signature = sign_pairing_payload(
            redeem_payload_json(exchange_token, EXPECTED),
            shared_secret=secret,
        )
        result = redeem_join_token(
            token=minted["token"],
            exchange_token=exchange_token,
            versions=EXPECTED,
            signature=signature,
            **_redeem_kwargs(store),
        )
        assert result["success"] is False
        assert result["status"] == 400
        assert store.status(minted["token_id"])["status"] == "pending"


class TestVersionReport:
    def test_mismatch_is_recorded_and_surfaced(self, store):
        minted = store.mint()
        _, secret = decode_join_token(minted["token"])
        reported = {"omlx": "0.5.3", "mlx": "0.31.0", "mlx-lm": "0.31.3"}
        exchange_token, signature = _signed_redeem(secret, reported)

        result = redeem_join_token(
            token=minted["token"],
            exchange_token=exchange_token,
            versions=reported,
            signature=signature,
            **_redeem_kwargs(store),
        )
        assert result["success"] is True
        assert result["runtime_compatible"] is False
        assert result["runtime_mismatches"] == ["mlx coordinator=0.32.0 peer=0.31.0"]
        peer = store.enrolled()[0]
        assert peer["runtime_compatible"] is False
        assert peer["versions"] == reported

    def test_missing_version_reports_as_missing(self):
        mismatches = runtime_mismatches({"omlx": "0.5.3"}, EXPECTED)
        assert "mlx coordinator=0.32.0 peer=missing" in mismatches
        assert "mlx-lm coordinator=0.31.3 peer=missing" in mismatches
        assert not any(item.startswith("omlx ") for item in mismatches)

    def test_malformed_reports_are_rejected(self):
        assert validate_version_report(None) is None
        assert validate_version_report({}) is None
        assert validate_version_report({"omlx": "1.0", "evil": "1.0"}) is None
        assert validate_version_report({"omlx": 10}) is None
        assert validate_version_report({"omlx": " 1.0 "}) is None
        assert validate_version_report({"omlx": "x" * 65}) is None
        assert validate_version_report({"omlx": "1", "mlx": "2", "mlx-lm": "3", "m": "4"}) is None
        assert validate_version_report({"omlx": "1.0"}) == {"omlx": "1.0"}

    def test_invalid_report_does_not_consume_token(self, store):
        minted = store.mint()
        result = redeem_join_token(
            token=minted["token"],
            exchange_token="whatever",
            versions={"omlx": 10},
            signature="0" * 64,
            **_redeem_kwargs(store),
        )
        assert result["success"] is False
        assert result["status"] == 400
        assert store.status(minted["token_id"])["status"] == "pending"


class TestCoordinatorUrl:
    def test_accepts_host_hostport_and_urls(self):
        assert normalize_coordinator_url("studio.local") == "http://studio.local:8000"
        assert normalize_coordinator_url("studio.local:9000") == "http://studio.local:9000"
        assert normalize_coordinator_url("http://studio.local/") == "http://studio.local:8000"
        assert normalize_coordinator_url("https://10.0.0.2:8443") == "https://10.0.0.2:8443"
        assert normalize_coordinator_url("http://[::1]:9000") == "http://[::1]:9000"

    def test_rejects_non_host_values(self):
        for value in ("", "  ", "ftp://studio.local", "http://user@host",
                      "http://host/path", "http://host?q=1", "http://host#frag",
                      "http://host:99999"):
            with pytest.raises(ValueError):
                normalize_coordinator_url(value)


class TestJoinCluster:
    def _join(self, http_post, **overrides):
        kwargs = {
            "http_post": http_post,
            "key_provider": lambda: _key_pair(_public_key(b"\x03")),
            "install_key": lambda **_: True,
            "versions_provider": lambda: dict(EXPECTED),
            "hostname_provider": lambda: "peer.local",
        }
        kwargs.update(overrides)
        store = JoinTokenStore()
        minted = store.mint()
        return join_cluster("studio.local", minted["token"], **kwargs), store, minted

    def test_happy_path_signs_and_installs(self):
        calls = {}

        def http_post(url, body, timeout):
            calls["url"] = url
            calls["body"] = body
            calls["timeout"] = timeout
            return 200, {
                "coordinator": {
                    "node_id": "studio.local",
                    "public_key": _public_key(b"\x09"),
                    "fingerprint": "SHA256:coord",
                },
                "expected_versions": EXPECTED,
                "runtime_compatible": True,
                "runtime_mismatches": [],
            }

        installed = {}
        result, _, minted = self._join(
            http_post,
            install_key=lambda **kw: installed.update(kw) or True,
        )
        assert result["success"] is True
        assert calls["url"] == "http://studio.local:8000/admin/api/cluster/join/redeem"
        body = calls["body"]
        assert body["token"] == minted["token"]
        assert body["versions"] == EXPECTED
        assert len(body["signature"]) == 64
        # The submitted signature verifies against the token secret.
        _, secret = decode_join_token(minted["token"])
        assert body["signature"] == sign_pairing_payload(
            redeem_payload_json(body["exchange_token"], EXPECTED),
            shared_secret=secret,
        )
        # The coordinator's key was installed into this Mac's authorized_keys.
        assert installed["public_key"] == _public_key(b"\x09")
        assert result["coordinator_key_installed"] is True
        assert result["peer"]["node_id"] == "peer.local"

    def test_full_round_trip_through_redeem(self):
        """join_cluster and redeem_join_token agree on the wire contract."""

        join_store = JoinTokenStore()
        minted = join_store.mint()

        def http_post(url, body, timeout):
            assert url.endswith("/admin/api/cluster/join/redeem")
            return 200, redeem_join_token(
                token=body["token"],
                exchange_token=body["exchange_token"],
                versions=body["versions"],
                signature=body["signature"],
                **_redeem_kwargs(join_store),
            )

        result = join_cluster(
            "http://coordinator.local:8000",
            minted["token"],
            http_post=http_post,
            key_provider=lambda: _key_pair(_public_key(b"\x03")),
            install_key=lambda **_: True,
            versions_provider=lambda: dict(EXPECTED),
            hostname_provider=lambda: "peer.local",
        )
        assert result["success"] is True
        assert result["runtime_compatible"] is True
        enrolled = join_store.enrolled()
        assert len(enrolled) == 1
        assert enrolled[0]["node_id"] == "peer.local"

    def test_http_error_mapping(self):
        cases = {
            400: "redeem_rejected",
            403: "invalid_join_token",
            404: "coordinator_not_advertising",
            409: "join_token_already_used",
            410: "join_token_expired",
            500: "unexpected_response",
        }
        for status, code in cases.items():
            result, _, _ = self._join(
                lambda *_, _status=status: (_status, {"detail": "nope"})
            )
            assert result["success"] is False
            assert result["error"] == code
            assert "nope" in result["detail"] or str(status) in result["detail"]

    def test_connection_failure(self):
        def http_post(*_args):
            raise JoinConnectionError("could not reach the coordinator")

        result, _, _ = self._join(http_post)
        assert result["success"] is False
        assert result["error"] == "coordinator_unreachable"

    def test_missing_coordinator_key_in_response(self):
        result, _, _ = self._join(lambda *_: (200, {"coordinator": {}}))
        assert result["success"] is False
        assert result["error"] == "unexpected_response"

    def test_local_key_install_failure(self):
        def broken(**_kwargs):
            raise OSError("permission denied")

        result, _, _ = self._join(lambda *_: (200, {
            "coordinator": {"node_id": "c", "public_key": _public_key(b"\x09")},
        }), install_key=broken)
        assert result["success"] is False
        assert result["error"] == "coordinator_key_install_failed"

    def test_usage_errors(self):
        result = join_cluster("http://host/path", "token")
        assert result["error"] == "invalid_coordinator_url"
        result = join_cluster("studio.local", "not-a-token")
        assert result["error"] == "invalid_join_token"
        store = JoinTokenStore()
        minted = store.mint()
        result = join_cluster(
            "studio.local",
            minted["token"],
            ssh_name="-evil",
            key_provider=lambda: _key_pair(),
        )
        assert result["error"] == "invalid_ssh_name"
        result = join_cluster(
            "studio.local",
            minted["token"],
            timeout=0,
            key_provider=lambda: _key_pair(),
        )
        assert result["error"] == "invalid_timeout"


class TestJoinCli:
    def _args(self, **overrides):
        args = SimpleNamespace(
            cluster_action="join",
            coordinator="studio.local",
            token="token",
            ssh_name=None,
            timeout=10.0,
            json=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_success_output(self, monkeypatch, capsys):
        monkeypatch.setattr(
            enrollment,
            "join_cluster",
            lambda *a, **kw: {
                "success": True,
                "coordinator_url": "http://studio.local:8000",
                "coordinator": {"node_id": "studio.local", "fingerprint": "SHA256:c"},
                "coordinator_key_installed": True,
                "peer": {"node_id": "peer.local", "fingerprint": "SHA256:p"},
                "versions": EXPECTED,
                "expected_versions": EXPECTED,
                "runtime_compatible": True,
                "runtime_mismatches": [],
            },
        )
        assert cli.cluster_command(self._args()) == 0
        out = capsys.readouterr().out
        assert "Joined the cluster at http://studio.local:8000" in out
        assert "Versions:      match omlx 0.5.3, mlx 0.32.0, mlx-lm 0.31.3" in out

    def test_mismatch_prints_pinned_install_line(self, monkeypatch, capsys):
        monkeypatch.setattr(
            enrollment,
            "join_cluster",
            lambda *a, **kw: {
                "success": True,
                "coordinator_url": "http://studio.local:8000",
                "coordinator": {"node_id": "studio.local", "fingerprint": "SHA256:c"},
                "coordinator_key_installed": True,
                "peer": {"node_id": "peer.local", "fingerprint": "SHA256:p"},
                "versions": {**EXPECTED, "mlx": "0.31.0"},
                "expected_versions": EXPECTED,
                "runtime_compatible": False,
                "runtime_mismatches": ["mlx coordinator=0.32.0 peer=0.31.0"],
            },
        )
        assert cli.cluster_command(self._args()) == 0
        out = capsys.readouterr().out
        assert "MISMATCH" in out
        assert "mlx coordinator=0.32.0 peer=0.31.0" in out
        assert 'python3 -m pip install "omlx==0.5.3" "mlx==0.32.0" "mlx-lm==0.31.3"' in out

    def test_failure_exit_codes_and_json(self, monkeypatch, capsys):
        monkeypatch.setattr(
            enrollment,
            "join_cluster",
            lambda *a, **kw: {
                "success": False,
                "error": "invalid_join_token",
                "detail": "the join token is malformed",
            },
        )
        assert cli.cluster_command(self._args()) == 2
        assert "malformed" in capsys.readouterr().err

        monkeypatch.setattr(
            enrollment,
            "join_cluster",
            lambda *a, **kw: {
                "success": False,
                "error": "join_token_expired",
                "detail": "join token expired",
            },
        )
        assert cli.cluster_command(self._args(json=True)) == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["error"] == "join_token_expired"


class TestJoinRoutes:
    def test_mint_and_poll_lifecycle(self, client, monkeypatch):
        monkeypatch.setattr(
            enrollment, "_local_runtime_versions", lambda: dict(EXPECTED)
        )
        monkeypatch.setattr(
            enrollment, "install_authorized_key", lambda **_: True
        )
        monkeypatch.setattr(
            enrollment, "get_or_create_ssh_key", lambda: _key_pair()
        )

        minted = client.post("/admin/api/cluster/join-token").json()
        assert minted["expires_in_seconds"] == enrollment.JOIN_TOKEN_TTL

        pending = client.get(
            "/admin/api/cluster/join-status",
            params={"token_id": minted["token_id"]},
        ).json()
        assert pending["token"]["status"] == "pending"
        assert pending["enrolled"] == []

        _, secret = decode_join_token(minted["token"])
        exchange_token, signature = _signed_redeem(secret, EXPECTED)
        redeemed = client.post(
            "/admin/api/cluster/join/redeem",
            json={
                "token": minted["token"],
                "exchange_token": exchange_token,
                "versions": EXPECTED,
                "signature": signature,
            },
        )
        assert redeemed.status_code == 200
        body = redeemed.json()
        assert body["success"] is True
        assert body["runtime_compatible"] is True

        done = client.get(
            "/admin/api/cluster/join-status",
            params={"token_id": minted["token_id"]},
        ).json()
        assert done["token"]["status"] == "redeemed"
        assert done["token"]["peer"]["node_id"] == "peer.local"
        assert done["enrolled"][0]["runtime_compatible"] is True

    def test_redeem_rejects_wrong_signature_and_replay(self, client, monkeypatch):
        monkeypatch.setattr(
            enrollment, "install_authorized_key", lambda **_: True
        )
        monkeypatch.setattr(
            enrollment, "get_or_create_ssh_key", lambda: _key_pair()
        )
        monkeypatch.setattr(
            enrollment, "_local_runtime_versions", lambda: dict(EXPECTED)
        )
        minted = client.post("/admin/api/cluster/join-token").json()
        _, secret = decode_join_token(minted["token"])
        exchange_token, signature = _signed_redeem(secret, EXPECTED)

        wrong = client.post(
            "/admin/api/cluster/join/redeem",
            json={
                "token": minted["token"],
                "exchange_token": exchange_token,
                "versions": EXPECTED,
                "signature": "0" * 64,
            },
        )
        assert wrong.status_code == 403

        payload = {
            "token": minted["token"],
            "exchange_token": exchange_token,
            "versions": EXPECTED,
            "signature": signature,
        }
        assert client.post("/admin/api/cluster/join/redeem", json=payload).status_code == 200
        replay = client.post("/admin/api/cluster/join/redeem", json=payload)
        assert replay.status_code == 409

    def test_redeem_validates_request_shape(self, client):
        minted = client.post("/admin/api/cluster/join-token").json()
        response = client.post(
            "/admin/api/cluster/join/redeem",
            json={
                "token": minted["token"],
                "exchange_token": "x",
                "versions": EXPECTED,
                "signature": "0" * 64,
                "unexpected": True,
            },
        )
        assert response.status_code == 422

        bad_versions = client.post(
            "/admin/api/cluster/join/redeem",
            json={
                "token": minted["token"],
                "exchange_token": "x",
                "versions": {"omlx": 10},
                "signature": "0" * 64,
            },
        )
        assert bad_versions.status_code == 422

    def test_join_status_unknown_token(self, client):
        payload = client.get(
            "/admin/api/cluster/join-status", params={"token_id": "nope"}
        ).json()
        assert payload["token"] is None
        assert payload["enrolled"] == []


def test_peer_router_registered_with_404_hiding_only():
    source = (Path(__file__).resolve().parents[1] / "omlx/server.py").read_text()

    marker = "app.include_router(\n        cluster_peer_router"
    assert marker in source, "peer router is not registered in server.py"
    block = source.split(marker, 1)[1].split("]")[0]
    assert "Depends(require_distributed_inference_enabled)" in block
    assert "Depends(require_admin)" not in block
