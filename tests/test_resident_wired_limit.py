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
