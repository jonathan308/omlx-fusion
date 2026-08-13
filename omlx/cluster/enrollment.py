# SPDX-License-Identifier: Apache-2.0
"""Guided peer enrollment: single-use join tokens and the peer join client.

The manual pairing flow asks the second Mac's owner to install oMLX by hand,
enable Remote Login, and walk a three-step shared-secret key exchange on both
dashboards. Enrollment replaces that cliff with one copyable command:

1. The coordinator's dashboard mints a short-lived, single-use join token.
2. The peer runs ``omlx cluster join <coordinator-url> <token>``.
3. The peer signs its SSH key-exchange token and runtime versions with the
   token secret and redeems them over HTTP; the coordinator installs the
   peer's key, answers with its own public key, and records the versions.

The join token authenticates the redeem call in place of an admin session:
the peer proves possession of the token secret with an HMAC-SHA256 signature
over the exact payload, verified in constant time like the existing pairing
flow. The token secret is 256 bits, so the peer-facing endpoint does not
need rate limiting to resist guessing.

Version compatibility reuses the launch preflight's expectation set
(``launch._local_runtime_versions``: exact omlx/MLX/MLX-LM matches). The
check here is advisory — enrollment succeeds so the dashboard can show the
mismatch — while the preflight remains the hard gate before any launch.
"""

from __future__ import annotations

import base64
import binascii
import hmac
import json
import secrets
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .deployment import validate_ssh_target
from .launch import _local_runtime_versions
from .ssh_keys import (
    SSHKeyPair,
    create_key_exchange_token,
    get_or_create_ssh_key,
    install_authorized_key,
    verify_key_exchange_token,
)
from .token_auth import (
    sign_pairing_payload,
    validate_pairing_secret,
    verify_pairing_signature,
)

# Longer than the pairing token's five minutes: a person must carry the join
# command to another Mac, install oMLX there if needed, and run it.
JOIN_TOKEN_TTL = 600  # 10 minutes
_MAX_PENDING_TOKENS = 16
_MAX_ENROLLED_PEERS = 64
_MAX_RESPONSE_BYTES = 64 * 1024
# The version keys the launch preflight compares; nothing else is accepted.
_VERSION_KEYS = ("omlx", "mlx", "mlx-lm")
_MAX_VERSION_LENGTH = 64

JOIN_REDEEM_PATH = "/admin/api/cluster/join/redeem"
_DEFAULT_COORDINATOR_PORT = 8000


class JoinConnectionError(RuntimeError):
    """The coordinator could not be reached or answered unintelligibly."""


def encode_join_token(token_id: str, secret: str) -> str:
    """Serialize a join token as URL-safe text the CLI can accept."""

    payload = {"v": 1, "id": token_id, "secret": secret}
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return base64.urlsafe_b64encode(raw).decode()


def decode_join_token(encoded: str) -> tuple[str, str] | None:
    """Split a join token into its lookup id and signing secret."""

    try:
        payload = json.loads(base64.urlsafe_b64decode(encoded))
        if not isinstance(payload, dict) or payload.get("v") != 1:
            return None
        token_id = payload["id"]
        secret = payload["secret"]
    except (binascii.Error, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not isinstance(token_id, str) or not isinstance(secret, str):
        return None
    # Bounds mirror token_auth's secret rules so a padded token cannot push
    # an oversized secret into the HMAC verifier.
    if not 8 <= len(token_id) <= 128:
        return None
    try:
        validate_pairing_secret(secret)
    except ValueError:
        return None
    return token_id, secret


def redeem_payload_json(exchange_token: str, versions: dict[str, str]) -> str:
    """Canonical JSON the peer signs and the coordinator verifies.

    Signing the exchange token and the version report together binds the
    peer's key material and its claimed runtime to the join token, so a
    network observer cannot swap either without knowing the token secret.
    """

    return json.dumps(
        {"exchange_token": exchange_token, "versions": versions},
        sort_keys=True,
    )


def validate_version_report(versions: Any) -> dict[str, str] | None:
    """Bound a peer-reported version map, or reject it as malformed."""

    if not isinstance(versions, dict) or not versions:
        return None
    if len(versions) > len(_VERSION_KEYS):
        return None
    cleaned: dict[str, str] = {}
    for key, value in versions.items():
        if key not in _VERSION_KEYS:
            return None
        if (
            not isinstance(value, str)
            or not value.strip()
            or value != value.strip()
            or len(value) > _MAX_VERSION_LENGTH
        ):
            return None
        cleaned[key] = value
    return cleaned


def runtime_mismatches(
    reported: dict[str, str],
    expected: dict[str, str],
) -> list[str]:
    """Diff a peer's version report against the coordinator's runtime.

    Uses the same exact-match semantics and message shape as the launch
    preflight so a join-time warning reads like the launch-time error.
    """

    return [
        f"{name} coordinator={expected[name]} peer={reported.get(name) or 'missing'}"
        for name in expected
        if expected[name] != reported.get(name)
    ]


@dataclass(frozen=True)
class EnrolledPeer:
    """A peer that redeemed a join token, with its reported runtime."""

    token_id: str
    node_id: str
    fingerprint: str
    versions: dict[str, str]
    runtime_compatible: bool
    runtime_mismatches: tuple[str, ...]
    enrolled_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id,
            "node_id": self.node_id,
            "fingerprint": self.fingerprint,
            "versions": dict(self.versions),
            "runtime_compatible": self.runtime_compatible,
            "runtime_mismatches": list(self.runtime_mismatches),
            "enrolled_at": self.enrolled_at,
        }


@dataclass
class _PendingJoin:
    token_id: str
    secret: str
    created_at: float
    expires_at: float
    state: str = "pending"  # pending | redeeming | redeemed
    peer: EnrolledPeer | None = None


def _redeem_error(status: int, detail: str) -> dict[str, Any]:
    return {"success": False, "status": status, "detail": detail}


class JoinTokenStore:
    """Thread-safe in-memory store for pending joins and enrolled peers.

    In-memory is deliberate: a join token outlives neither the process nor
    its ten-minute TTL, and enrolled peers announce themselves via Bonjour
    once they run a server. Nothing here survives a restart, so no restart
    can replay a token.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.RLock()
        self._tokens: dict[str, _PendingJoin] = {}
        self._enrolled_order: list[str] = []

    def _sweep(self, now: float) -> None:
        expired = [
            token_id
            for token_id, record in self._tokens.items()
            if record.state == "pending" and now > record.expires_at
        ]
        for token_id in expired:
            del self._tokens[token_id]

    def mint(self) -> dict[str, Any]:
        """Create a single-use join token for one peer enrollment."""

        with self._lock:
            now = self._clock()
            self._sweep(now)
            while len(self._tokens) >= _MAX_PENDING_TOKENS:
                # Minting is an explicit admin action; evicting the oldest
                # still-pending token keeps the store bounded without
                # failing the request in front of the user.
                oldest = min(
                    self._tokens,
                    key=lambda key: self._tokens[key].created_at,
                )
                del self._tokens[oldest]
            record = _PendingJoin(
                token_id=secrets.token_urlsafe(12),
                secret=secrets.token_urlsafe(32),
                created_at=now,
                expires_at=now + JOIN_TOKEN_TTL,
            )
            self._tokens[record.token_id] = record
            return {
                "token": encode_join_token(record.token_id, record.secret),
                "token_id": record.token_id,
                "expires_at": record.expires_at,
                "expires_in_seconds": int(record.expires_at - now),
            }

    def begin_redeem(
        self,
        encoded_token: str,
        *,
        payload_json: str,
        signature: str,
    ) -> tuple[_PendingJoin | None, dict[str, Any] | None]:
        """Validate a redeem attempt and mark the token as in-flight.

        The signature check happens before the state flip, and both happen
        under the lock, so a replayed or concurrent redeem of the same token
        is refused even while the first attempt is still installing keys.
        """

        decoded = decode_join_token(encoded_token)
        with self._lock:
            if decoded is None:
                return None, _redeem_error(403, "invalid join token")
            token_id, presented_secret = decoded
            record = self._tokens.get(token_id)
            # The id lookup selects the record; the secret itself is then
            # compared in constant time, like any bearer credential.
            if record is None or not hmac.compare_digest(
                presented_secret, record.secret
            ):
                return None, _redeem_error(403, "invalid join token")
            if record.state != "pending":
                return None, _redeem_error(409, "join token was already redeemed")
            if self._clock() > record.expires_at:
                return None, _redeem_error(410, "join token expired")
            if not verify_pairing_signature(
                payload_json,
                signature,
                shared_secret=record.secret,
            ):
                return None, _redeem_error(403, "invalid join token signature")
            record.state = "redeeming"
            return record, None

    def finish_redeem(self, token_id: str, peer: EnrolledPeer) -> EnrolledPeer | None:
        """Record a completed enrollment, stamped with the store's clock."""

        with self._lock:
            record = self._tokens.get(token_id)
            if record is None or record.state != "redeeming":
                return None
            peer = replace(peer, enrolled_at=self._clock())
            record.state = "redeemed"
            record.peer = peer
            self._enrolled_order.append(token_id)
            while len(self._enrolled_order) > _MAX_ENROLLED_PEERS:
                self._enrolled_order.pop(0)
            return peer

    def abort_redeem(self, token_id: str) -> None:
        """Return an in-flight token to pending after an infrastructure error.

        A peer whose redeem failed on the coordinator's side (for example a
        local key-install error) may retry with the same token.
        """

        with self._lock:
            record = self._tokens.get(token_id)
            if record is not None and record.state == "redeeming":
                record.state = "pending"

    def status(self, token_id: str) -> dict[str, Any] | None:
        """Report one token's lifecycle state for dashboard polling."""

        with self._lock:
            record = self._tokens.get(token_id)
            if record is None:
                return None
            state = record.state
            if self._clock() > record.expires_at and state != "redeemed":
                state = "expired"
            elif state == "redeeming":
                state = "pending"
            return {
                "token_id": record.token_id,
                "status": state,
                "expires_at": record.expires_at,
                "peer": record.peer.to_dict() if record.peer else None,
            }

    def enrolled(self) -> list[dict[str, Any]]:
        """List enrolled peers, most recent first."""

        with self._lock:
            return [
                self._tokens[token_id].peer.to_dict()
                for token_id in reversed(self._enrolled_order)
                if token_id in self._tokens and self._tokens[token_id].peer is not None
            ]


_store_lock = threading.Lock()
_join_token_store: JoinTokenStore | None = None


def get_join_token_store() -> JoinTokenStore:
    global _join_token_store
    with _store_lock:
        if _join_token_store is None:
            _join_token_store = JoinTokenStore()
        return _join_token_store


def reset_join_token_store() -> None:
    """Drop all pending tokens and enrollments (tests and shutdown)."""

    global _join_token_store
    with _store_lock:
        _join_token_store = None


def redeem_join_token(
    *,
    token: str,
    exchange_token: str,
    versions: Any,
    signature: str,
    store: JoinTokenStore | None = None,
    expected_versions: dict[str, str] | None = None,
    install_key: Callable[..., bool] | None = None,
    coordinator_key_provider: Callable[[], SSHKeyPair] | None = None,
    hostname_provider: Callable[[], str] = socket.gethostname,
) -> dict[str, Any]:
    """Redeem a join token: exchange SSH keys and record the peer's runtime.

    The token is only consumed once the whole exchange succeeds; a local
    failure rolls the token back to pending so the peer can retry.
    """

    cleaned_versions = validate_version_report(versions)
    if cleaned_versions is None:
        return _redeem_error(400, "invalid version report")
    if store is None:
        store = get_join_token_store()
    if expected_versions is None:
        expected_versions = _local_runtime_versions()
    if install_key is None:
        install_key = install_authorized_key
    if coordinator_key_provider is None:
        coordinator_key_provider = get_or_create_ssh_key

    payload_json = redeem_payload_json(exchange_token, cleaned_versions)
    record, error = store.begin_redeem(
        token,
        payload_json=payload_json,
        signature=signature,
    )
    if error is not None or record is None:
        return error or _redeem_error(403, "invalid join token")

    peer_key = verify_key_exchange_token(
        exchange_token,
        shared_secret=record.secret,
    )
    if peer_key is None:
        store.abort_redeem(record.token_id)
        return _redeem_error(400, "invalid key exchange token")
    try:
        key_installed = install_key(public_key=peer_key.public_key)
        coordinator_key = coordinator_key_provider()
    except Exception as exc:
        store.abort_redeem(record.token_id)
        return _redeem_error(500, f"enrollment failed: {exc}")

    mismatches = runtime_mismatches(cleaned_versions, expected_versions)
    peer = store.finish_redeem(
        record.token_id,
        EnrolledPeer(
            token_id=record.token_id,
            node_id=peer_key.node_id,
            fingerprint=peer_key.fingerprint,
            versions=cleaned_versions,
            runtime_compatible=not mismatches,
            runtime_mismatches=tuple(mismatches),
            # finish_redeem stamps the actual recording time.
            enrolled_at=0.0,
        ),
    )
    if peer is None:  # pragma: no cover - state changed concurrently
        return _redeem_error(409, "join token was already redeemed")
    return {
        "success": True,
        "peer": peer.to_dict(),
        "coordinator": {
            "node_id": hostname_provider(),
            "public_key": coordinator_key.public_key,
            "fingerprint": coordinator_key.fingerprint,
        },
        "peer_key_installed": bool(key_installed),
        "expected_versions": dict(expected_versions),
        "runtime_compatible": not mismatches,
        "runtime_mismatches": mismatches,
    }


def normalize_coordinator_url(value: str) -> str:
    """Accept ``host``, ``host:port``, or a full URL and normalize to a base URL."""

    candidate = value.strip()
    if not candidate:
        raise ValueError("coordinator URL must not be empty")
    if "://" not in candidate:
        candidate = f"http://{candidate}"
    parts = urllib.parse.urlsplit(candidate)
    if parts.scheme not in ("http", "https"):
        raise ValueError("coordinator URL must use http or https")
    if not parts.hostname or parts.username or parts.password:
        raise ValueError("coordinator URL must be a bare host, optionally with a port")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise ValueError("coordinator URL must not include a path, query, or fragment")
    try:
        port = parts.port or _DEFAULT_COORDINATOR_PORT
    except ValueError as exc:
        raise ValueError(f"invalid coordinator port: {parts.netloc}") from exc
    host = parts.hostname
    if ":" in host:  # IPv6 literals need brackets when rebuilt with a port.
        host = f"[{host}]"
    return f"{parts.scheme}://{host}:{port}"


def _http_post_json(url: str, body: dict[str, Any], timeout: float) -> tuple[int, dict]:
    """POST JSON and return ``(status, payload)`` without raising on HTTP errors."""

    # The URL is the operator's own coordinator, taken from the join command
    # they copied out of its dashboard.
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(_MAX_RESPONSE_BYTES)
            return response.status, json.loads(raw.decode())
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read(_MAX_RESPONSE_BYTES).decode())
        except (ValueError, UnicodeDecodeError):
            payload = {}
        return exc.code, payload if isinstance(payload, dict) else {}
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise JoinConnectionError(
            f"could not reach the coordinator at {url}: {exc}"
        ) from exc
    except (ValueError, UnicodeDecodeError) as exc:
        raise JoinConnectionError(
            f"the coordinator at {url} did not answer with enrollment JSON"
        ) from exc


def join_cluster(
    coordinator_url: str,
    join_token: str,
    *,
    ssh_name: str | None = None,
    timeout: float = 10.0,
    http_post: Callable[[str, dict[str, Any], float], tuple[int, dict]] = _http_post_json,
    key_provider: Callable[[], SSHKeyPair] = get_or_create_ssh_key,
    install_key: Callable[..., bool] = install_authorized_key,
    versions_provider: Callable[[], dict[str, str]] = _local_runtime_versions,
    hostname_provider: Callable[[], str] = socket.gethostname,
    authorized_keys_path: Path | None = None,
) -> dict[str, Any]:
    """Enroll this Mac into a coordinator's cluster with a join token.

    Never raises for expected failures: the result carries ``success`` plus a
    machine-readable ``error`` code so the CLI can pick an exit status. All
    network and key operations are injectable for tests.
    """

    def failure(code: str, detail: str) -> dict[str, Any]:
        return {"success": False, "error": code, "detail": detail}

    try:
        base_url = normalize_coordinator_url(coordinator_url)
    except ValueError as exc:
        return failure("invalid_coordinator_url", str(exc))
    decoded = decode_join_token(join_token)
    if decoded is None:
        return failure(
            "invalid_join_token",
            "the join token is malformed; copy the whole command again",
        )
    _, secret = decoded
    try:
        node_id = validate_ssh_target(ssh_name or hostname_provider())
    except ValueError:
        return failure(
            "invalid_ssh_name",
            "pass --ssh-name as the hostname (or user@host) the coordinator "
            "uses to reach this Mac",
        )
    if timeout <= 0:
        return failure("invalid_timeout", "timeout must be positive")

    versions = versions_provider()
    try:
        key_pair = key_provider()
    except RuntimeError as exc:
        return failure("ssh_key_unavailable", str(exc))
    exchange_token = create_key_exchange_token(
        public_key=key_pair.public_key,
        node_id=node_id,
        shared_secret=secret,
    )
    signature = sign_pairing_payload(
        redeem_payload_json(exchange_token, versions),
        shared_secret=secret,
    )

    try:
        status, payload = http_post(
            f"{base_url}{JOIN_REDEEM_PATH}",
            {
                "token": join_token,
                "exchange_token": exchange_token,
                "versions": versions,
                "signature": signature,
            },
            timeout,
        )
    except JoinConnectionError as exc:
        return failure("coordinator_unreachable", str(exc))

    if status == 404:
        return failure(
            "coordinator_not_advertising",
            "the coordinator answered 404 — enable Distributed Inference in "
            "its Settings > Advanced and restart it, then mint a fresh token",
        )
    if status != 200:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        code = {
            400: "redeem_rejected",
            403: "invalid_join_token",
            409: "join_token_already_used",
            410: "join_token_expired",
        }.get(status, "unexpected_response")
        return failure(
            code,
            detail
            if isinstance(detail, str)
            else f"coordinator returned HTTP {status}",
        )

    coordinator = payload.get("coordinator") or {}
    coordinator_key = coordinator.get("public_key")
    if not isinstance(coordinator_key, str) or not coordinator_key.strip():
        return failure(
            "unexpected_response",
            "the coordinator's answer did not include its public key",
        )
    try:
        installed = install_key(
            public_key=coordinator_key,
            authorized_keys_path=authorized_keys_path,
        )
    except OSError as exc:
        return failure("coordinator_key_install_failed", str(exc))

    return {
        "success": True,
        "coordinator_url": base_url,
        "coordinator": {
            "node_id": coordinator.get("node_id", ""),
            "fingerprint": coordinator.get("fingerprint", ""),
        },
        "coordinator_key_installed": bool(installed),
        "peer": {
            "node_id": node_id,
            "fingerprint": key_pair.fingerprint,
        },
        "versions": versions,
        "expected_versions": payload.get("expected_versions") or {},
        "runtime_compatible": bool(payload.get("runtime_compatible")),
        "runtime_mismatches": list(payload.get("runtime_mismatches") or []),
    }
