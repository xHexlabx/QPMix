"""Tests for qpmix.interp."""

from __future__ import annotations

import numpy as np
import pytest

from qpmix.interp import PARALLEL_THRESHOLD, UniformInterpolator, interp_point


@pytest.fixture
def sine():
    v = np.linspace(-4.0, 4.0, 8001)
    return UniformInterpolator.from_samples(v, np.sin(v))


def test_reproduces_nodes_exactly(sine):
    nodes = sine.v0 + sine.dv * np.arange(len(sine))
    assert np.allclose(sine(nodes), sine.values, atol=1e-12)


def test_cubic_accuracy_between_nodes(sine):
    x = np.linspace(-3.9, 3.9, 12345)
    assert np.abs(sine(x) - np.sin(x)).max() < 1e-10


def test_cubic_is_fourth_order():
    """Halving the grid spacing should cut the error by roughly 2**4."""
    x = np.linspace(-2.9, 2.9, 4001)
    errors = []
    for npts in (801, 1601):
        v = np.linspace(-3.0, 3.0, npts)
        f = UniformInterpolator.from_samples(v, np.exp(np.sin(v)))
        errors.append(np.abs(f(x) - np.exp(np.sin(x))).max())
    assert 8 < errors[0] / errors[1] < 32


def test_linear_data_is_exact():
    v = np.linspace(0.0, 10.0, 101)
    f = UniformInterpolator.from_samples(v, 3.0 * v - 1.0)
    x = np.linspace(0.0, 10.0, 997)
    assert np.allclose(f(x), 3.0 * x - 1.0, atol=1e-12)


def test_complex_values_supported():
    v = np.linspace(-2.0, 2.0, 4001)
    f = UniformInterpolator.from_samples(v, np.sin(v) + 1j * np.cos(v))
    x = np.linspace(-1.9, 1.9, 555)
    assert np.abs(f(x) - (np.sin(x) + 1j * np.cos(x))).max() < 1e-10
    assert f(x).dtype == np.complex128


def test_scalar_input_returns_scalar(sine):
    out = sine(0.5)
    assert np.ndim(out) == 0
    assert out == pytest.approx(np.sin(0.5), abs=1e-10)


def test_shape_is_preserved(sine):
    x = np.linspace(-1, 1, 24).reshape(2, 3, 4)
    assert sine(x).shape == (2, 3, 4)


def test_empty_input(sine):
    assert sine(np.array([])).shape == (0,)


def test_extrapolation_is_linear():
    v = np.linspace(0.0, 1.0, 101)
    f = UniformInterpolator.from_samples(v, 2.0 * v)
    assert f(-1.0) == pytest.approx(-2.0, abs=1e-9)
    assert f(3.0) == pytest.approx(6.0, abs=1e-9)


def test_parallel_and_serial_kernels_agree():
    """The two kernels are separate compilations; they must not diverge."""
    v = np.linspace(-3.0, 3.0, 20001)
    f = UniformInterpolator.from_samples(v, np.tanh(v))
    x = np.linspace(-2.5, 2.5, PARALLEL_THRESHOLD + 1000)
    big = f(x)
    small = np.concatenate([f(chunk) for chunk in np.array_split(x, 64)])
    assert np.array_equal(big, small)


def test_interp_point_matches_array_path(sine):
    for x in (-3.3, 0.0, 0.123456, 3.9):
        assert interp_point(x, sine.v0, 1 / sine.dv, sine.values) == pytest.approx(
            float(sine(x)), abs=1e-14
        )


def test_vmin_vmax_and_len(sine):
    assert sine.vmin == pytest.approx(-4.0)
    assert sine.vmax == pytest.approx(4.0)
    assert len(sine) == 8001


def test_eval_into_returns_buffer(sine):
    x = np.linspace(-1, 1, 10)
    out = np.empty(10, dtype=float)
    assert sine.eval_into(x, out) is out


def test_eval_into_rejects_shape_mismatch(sine):
    with pytest.raises(ValueError, match="same shape"):
        sine.eval_into(np.zeros(3), np.zeros(4))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"v0": 0.0, "dv": 0.0, "values": np.zeros(10)}, "dv must be positive"),
        ({"v0": 0.0, "dv": -1.0, "values": np.zeros(10)}, "dv must be positive"),
        ({"v0": 0.0, "dv": 1.0, "values": np.zeros(3)}, "at least 4 points"),
        ({"v0": 0.0, "dv": 1.0, "values": np.zeros((4, 4))}, "must be 1-D"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        UniformInterpolator(**kwargs)


def test_from_samples_rejects_non_uniform():
    with pytest.raises(ValueError, match="uniformly spaced"):
        UniformInterpolator.from_samples(np.array([0.0, 1.0, 3.0, 4.0]), np.zeros(4))
