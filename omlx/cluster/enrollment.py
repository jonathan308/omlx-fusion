# SPDX-License-Identifier: Apache-2.0
"""Guided peer enrollment and one-time headless-worker join state.

Two complementary subsystems live here:

1. Guided peer enrollment (``JoinTokenStore``/``join_cluster``) — the
   dashboard "Add a Mac" flow. The coordinator mints a short-lived,
   single-use join token; the peer runs ``omlx cluster join <url> <token>``,
   signs its SSH key-exchange token and runtime versions with the token
   secret, and redeems them over HTTP; the coordinator installs the peer's
   key, answers with its own public key, and records the versions. The join
   token authenticates the redeem call in place of an admin session: the
   peer proves possession of the token secret with an HMAC-SHA256 signature
   over the exact payload, verified in constant time. The token secret is
   256 bits, so the peer-facing endpoint needs no rate limiting.

   Version compatibility reuses the launch preflight's expectation set
   (``launch._local_runtime_versions``: exact omlx/MLX/MLX-LM matches). The
   check here is advisory — enrollment succeeds so the dashboard can show
   the mismatch — while the preflight remains the hard gate before launch.

2. One-time enrollment state for headless workers
   (``ClusterEnrollmentStore``). Join credentials are deliberately
   memory-only: a server restart invalidates every unclaimed command, while
   completed node records persist without tokens or public-key material.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
import secrets
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextlib import suppress
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

#----------------------------------------------------------------------
# Guided peer enrollment ("Add a Mac")
#----------------------------------------------------------------------

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


#----------------------------------------------------------------------
# One-time headless-worker enrollment state
#----------------------------------------------------------------------

JOIN_KEY_TTL_SECONDS = 10 * 60
JOIN_SESSION_TTL_SECONDS = 20 * 60
MAX_PENDING_JOIN_KEYS = 32
MAX_ENROLLED_NODES = 64


class EnrollmentError(ValueError):
    """A join credential or enrollment payload was refused."""


def _secret_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass
class _JoinKey:
    join_id: str
    digest: str
    controller_url: str
    source_digest: str
    created_at: float
    expires_at: float
    used_at: float | None = None


@dataclass(frozen=True)
class _JoinSession:
    digest: str
    join_id: str
    controller_url: str
    source_digest: str
    node_id: str
    hostname: str
    ssh_user: str
    ssh_port: int
    addresses: tuple[str, ...]
    expires_at: float


@dataclass(frozen=True)
class EnrolledNode:
    node_id: str
    hostname: str
    ssh: str
    ssh_user: str
    ssh_port: int
    addresses: tuple[str, ...]
    accelerator: str
    platform: str
    python_executable: str
    source_digest: str
    ssh_host_fingerprint: str
    joined_at: float
    last_seen_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "ssh": self.ssh,
            "ssh_user": self.ssh_user,
            "ssh_port": self.ssh_port,
            "addresses": list(self.addresses),
            "accelerator": self.accelerator,
            "platform": self.platform,
            "python_executable": self.python_executable,
            "source_digest": self.source_digest,
            "ssh_host_fingerprint": self.ssh_host_fingerprint,
            "joined_at": self.joined_at,
            "last_seen_at": self.last_seen_at,
            "status": "joined",
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EnrolledNode:
        if not isinstance(value, dict):
            raise EnrollmentError("enrolled node record must be an object")
        addresses = value.get("addresses")
        if not isinstance(addresses, list) or not all(
            isinstance(item, str) for item in addresses
        ):
            raise EnrollmentError("enrolled node addresses are malformed")
        return cls(
            node_id=str(value["node_id"]),
            hostname=str(value["hostname"]),
            ssh=str(value["ssh"]),
            ssh_user=str(value["ssh_user"]),
            ssh_port=int(value["ssh_port"]),
            addresses=tuple(addresses),
            accelerator=str(value["accelerator"]),
            platform=str(value["platform"]),
            python_executable=str(value["python_executable"]),
            source_digest=str(value["source_digest"]),
            ssh_host_fingerprint=str(value["ssh_host_fingerprint"]),
            joined_at=float(value["joined_at"]),
            last_seen_at=float(value["last_seen_at"]),
        )


class ClusterEnrollmentStore:
    """Thread-safe one-time join keys plus credential-free node persistence."""

    def __init__(self, base_path: Path, *, clock=time.time) -> None:
        self.base_path = Path(base_path)
        self.path = self.base_path / "cluster" / "enrolled-nodes.json"
        self._clock = clock
        self._lock = threading.RLock()
        self._join_keys: dict[str, _JoinKey] = {}
        self._sessions: dict[str, _JoinSession] = {}
        self._nodes: dict[str, EnrolledNode] = {}
        self.load_error: str | None = None
        try:
            self._load()
        except EnrollmentError as exc:
            self._nodes = {}
            self.load_error = str(exc)

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EnrollmentError(f"could not read enrolled nodes: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise EnrollmentError("unsupported enrolled-node registry schema")
        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list):
            raise EnrollmentError("enrolled-node registry is malformed")
        nodes = [EnrolledNode.from_dict(item) for item in raw_nodes]
        self._nodes = {node.node_id: node for node in nodes}
        if len(self._nodes) != len(nodes):
            raise EnrollmentError("enrolled-node registry has duplicate node IDs")

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "nodes": [node.to_dict() for node in self.list_nodes()],
        }
        descriptor, temporary = tempfile.mkstemp(
            prefix=".enrolled-nodes.", suffix=".tmp", dir=self.path.parent
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            os.chmod(self.path, 0o600)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temporary)

    def _prune(self) -> None:
        now = self._clock()
        self._sessions = {
            digest: session
            for digest, session in self._sessions.items()
            if session.expires_at >= now
        }
        live_session_join_ids = {
            session.join_id for session in self._sessions.values()
        }
        self._join_keys = {
            digest: record
            for digest, record in self._join_keys.items()
            if record.expires_at >= now or record.join_id in live_session_join_ids
        }

    def issue_join_key(
        self,
        *,
        controller_url: str,
        source_digest: str,
        ttl: int = JOIN_KEY_TTL_SECONDS,
    ) -> tuple[str, dict[str, Any]]:
        if not 30 <= int(ttl) <= JOIN_KEY_TTL_SECONDS:
            raise EnrollmentError(
                f"join-key TTL must be between 30 and {JOIN_KEY_TTL_SECONDS} seconds"
            )
        with self._lock:
            self._prune()
            active = [record for record in self._join_keys.values() if record.used_at is None]
            if len(active) >= MAX_PENDING_JOIN_KEYS:
                raise EnrollmentError("too many pending join keys; revoke one and retry")
            raw_key = secrets.token_urlsafe(32)
            now = self._clock()
            record = _JoinKey(
                join_id=secrets.token_hex(8),
                digest=_secret_digest(raw_key),
                controller_url=controller_url,
                source_digest=source_digest,
                created_at=now,
                expires_at=now + int(ttl),
            )
            self._join_keys[record.digest] = record
            return raw_key, self._join_key_dict(record)

    @staticmethod
    def _join_key_dict(record: _JoinKey) -> dict[str, Any]:
        return {
            "join_id": record.join_id,
            "controller_url": record.controller_url,
            "source_digest": record.source_digest,
            "created_at": record.created_at,
            "expires_at": record.expires_at,
            "status": "used" if record.used_at is not None else "pending",
            "used_at": record.used_at,
        }

    def claim(
        self,
        raw_key: str,
        *,
        node_id: str,
        hostname: str,
        ssh_user: str,
        ssh_port: int,
        addresses: tuple[str, ...],
    ) -> tuple[str, _JoinSession]:
        digest = _secret_digest(raw_key)
        with self._lock:
            self._prune()
            record = self._join_keys.get(digest)
            now = self._clock()
            if record is None or record.expires_at < now:
                raise EnrollmentError("invalid or expired join key")
            if record.used_at is not None:
                raise EnrollmentError("join key has already been used")
            record.used_at = now
            raw_session = secrets.token_urlsafe(32)
            session = _JoinSession(
                digest=_secret_digest(raw_session),
                join_id=record.join_id,
                controller_url=record.controller_url,
                source_digest=record.source_digest,
                node_id=node_id,
                hostname=hostname,
                ssh_user=ssh_user,
                ssh_port=ssh_port,
                addresses=addresses,
                expires_at=now + JOIN_SESSION_TTL_SECONDS,
            )
            self._sessions[session.digest] = session
            return raw_session, session

    def authorize_session(self, raw_session: str) -> _JoinSession:
        digest = _secret_digest(raw_session)
        with self._lock:
            self._prune()
            session = self._sessions.get(digest)
            if session is None:
                raise EnrollmentError("invalid or expired join session")
            return session

    def complete(self, raw_session: str, node: EnrolledNode) -> EnrolledNode:
        digest = _secret_digest(raw_session)
        with self._lock:
            session = self.authorize_session(raw_session)
            if node.source_digest != session.source_digest:
                raise EnrollmentError("worker source digest does not match the join command")
            if (
                node.node_id != session.node_id
                or node.hostname != session.hostname
                or node.ssh_user != session.ssh_user
                or node.ssh_port != session.ssh_port
                or node.addresses != session.addresses
            ):
                raise EnrollmentError("worker identity changed after claiming the join key")
            if node.node_id not in self._nodes and len(self._nodes) >= MAX_ENROLLED_NODES:
                raise EnrollmentError("the enrolled-node limit has been reached")
            previous = dict(self._nodes)
            self._nodes[node.node_id] = node
            try:
                self._save()
            except Exception:
                self._nodes = previous
                raise
            self._sessions.pop(digest, None)
            self.load_error = None
            return node

    def list_nodes(self) -> tuple[EnrolledNode, ...]:
        with self._lock:
            return tuple(self._nodes[key] for key in sorted(self._nodes))

    def remove_node(self, node_id: str) -> bool:
        with self._lock:
            if node_id not in self._nodes:
                return False
            previous = dict(self._nodes)
            del self._nodes[node_id]
            try:
                self._save()
            except Exception:
                self._nodes = previous
                raise
            return True

    def revoke_join_key(self, join_id: str) -> bool:
        with self._lock:
            record_digest = next(
                (
                    digest
                    for digest, record in self._join_keys.items()
                    if record.join_id == join_id
                ),
                None,
            )
            if record_digest is None:
                return False
            del self._join_keys[record_digest]
            self._sessions = {
                digest: session
                for digest, session in self._sessions.items()
                if session.join_id != join_id
            }
            return True

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            self._prune()
            return {
                "schema_version": 1,
                "join_keys": [
                    self._join_key_dict(record)
                    for record in sorted(
                        self._join_keys.values(), key=lambda item: item.created_at
                    )
                ],
                "nodes": [node.to_dict() for node in self.list_nodes()],
                "load_error": self.load_error,
            }


_store_lock = threading.Lock()
_configured_store: ClusterEnrollmentStore | None = None


def configure_cluster_enrollment(base_path: Path) -> ClusterEnrollmentStore:
    global _configured_store
    with _store_lock:
        _configured_store = ClusterEnrollmentStore(base_path)
        return _configured_store


def get_cluster_enrollment() -> ClusterEnrollmentStore:
    with _store_lock:
        if _configured_store is None:
            raise RuntimeError("cluster enrollment is not configured")
        return _configured_store
