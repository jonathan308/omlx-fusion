# SPDX-License-Identifier: Apache-2.0
"""The resident wired limit mirrors BatchGenerator's value and never lowers an explicit one."""

import mlx.core as mx

from omlx import process_memory_enforcer as pme


def test_resident_limit_uses_recommended_working_set(monkeypatch):
    calls = []

    def fake_set(value):
        calls.append(value)
        return 0

    monkeypatch.setattr(mx, "device_info", lambda: {"max_recommended_working_set_size": 1234})
    monkeypatch.setattr(mx, "set_wired_limit", fake_set)
    applied, previous = pme._apply_resident_wired_limit()
    assert (applied, previous) == (1234, 0)
    assert calls == [1234]


def test_resident_limit_keeps_a_larger_explicit_limit(monkeypatch):
    calls = []

    def fake_set(value):
        calls.append(value)
        return 5000  # previous limit was already higher

    monkeypatch.setattr(mx, "device_info", lambda: {"max_recommended_working_set_size": 1234})
    monkeypatch.setattr(mx, "set_wired_limit", fake_set)
    applied, previous = pme._apply_resident_wired_limit()
    assert (applied, previous) == (5000, 5000)
    assert calls == [1234, 5000]


def test_resident_limit_without_device_info(monkeypatch):
    monkeypatch.setattr(mx, "device_info", lambda: {})
    monkeypatch.setattr(mx, "set_wired_limit", lambda v: (_ for _ in ()).throw(AssertionError("must not be called")))
    assert pme._apply_resident_wired_limit() == (0, None)


def test_resident_limit_leaves_os_headroom_on_large_machines(monkeypatch):
    gib = 1024**3
    calls = []

    def fake_set(value):
        calls.append(value)
        return 0

    # M5 Ultra 256 GB: Apple recommends 243.2 GiB; keep 10% (25.6 GiB) unwired.
    monkeypatch.setattr(
        mx,
        "device_info",
        lambda: {"max_recommended_working_set_size": int(243.2 * gib), "memory_size": 256 * gib},
    )
    monkeypatch.setattr(mx, "set_wired_limit", fake_set)
    applied, _ = pme._apply_resident_wired_limit()
    assert applied == 256 * gib - int(256 * gib * 0.10)
    assert calls == [applied]


def test_resident_limit_headroom_floor_on_small_machines():
    gib = 1024**3
    # 64 GB: the 16 GiB floor applies; the recommended 48 GiB already fits.
    assert pme._resident_wired_ceiling(48 * gib, 64 * gib) == 48 * gib
    # 32 GB: 16 GiB headroom clamps a 24 GiB request to 16 GiB.
    assert pme._resident_wired_ceiling(24 * gib, 32 * gib) == 16 * gib
    # Unknown memory size leaves the request unchanged.
    assert pme._resident_wired_ceiling(24 * gib, 0) == 24 * gib
