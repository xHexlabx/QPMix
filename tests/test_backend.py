"""Tests for qpmix._backend."""

from __future__ import annotations

import numpy as np
import pytest

from qpmix import _backend


def test_get_config_reports_expected_keys():
    cfg = _backend.get_config()
    assert set(cfg) == {"jit_enabled", "parallel", "num_threads", "numba_version"}
    assert isinstance(cfg["jit_enabled"], bool)
    assert cfg["num_threads"] >= 1


def test_njit_works_as_bare_decorator():
    @_backend.njit
    def double(x):
        return x * 2

    assert double(3.0) == 6.0


def test_njit_works_with_options():
    @_backend.njit(fastmath=False)
    def add(x, y):
        return x + y

    assert add(1.0, 2.0) == 3.0


def test_njit_kernel_matches_pure_python():
    def kernel(arr):
        total = 0.0
        for i in range(arr.shape[0]):
            total += arr[i] * arr[i]
        return total

    jitted = _backend.njit(kernel)
    arr = np.linspace(0, 1, 101)
    assert jitted(arr) == pytest.approx(kernel(arr))


def test_prange_is_usable_as_range():
    assert list(_backend.prange(3)) == [0, 1, 2]


def test_set_num_threads_rejects_zero():
    with pytest.raises(ValueError, match=">= 1"):
        _backend.set_num_threads(0)


def test_set_num_threads_roundtrip():
    original = _backend.get_num_threads()
    _backend.set_num_threads(1)
    assert _backend.get_num_threads() == 1
    _backend.set_num_threads(original)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("0", False), ("", False), ("no", False)],
)
def test_env_flag_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("QPMIX_TEST_FLAG", value)
    assert _backend._env_flag("QPMIX_TEST_FLAG") is expected


def test_env_flag_default_when_unset(monkeypatch):
    monkeypatch.delenv("QPMIX_TEST_FLAG", raising=False)
    assert _backend._env_flag("QPMIX_TEST_FLAG", default=True) is True
