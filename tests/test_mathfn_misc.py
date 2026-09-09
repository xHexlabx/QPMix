"""Tests for qpmix.mathfn.misc."""

from __future__ import annotations

import numpy as np
import pytest

from qpmix.mathfn.misc import slope, slope_span_n


def test_slope_of_a_line_is_the_gradient():
    x = np.linspace(0, 10, 101)
    assert np.allclose(slope(x, 3 * x + 7), 3.0)


def test_slope_of_a_parabola():
    x = np.linspace(-2, 2, 2001)
    der = slope(x, x**2)
    assert np.abs(der[1:-1] - 2 * x[1:-1]).max() < 1e-10


def test_slope_preserves_length():
    x = np.linspace(0, 1, 37)
    assert slope(x, np.sin(x)).shape == (37,)


def test_slope_handles_two_points():
    assert slope(np.array([0.0, 2.0]), np.array([1.0, 5.0])) == pytest.approx(
        [2.0, 2.0]
    )


def test_slope_rejects_bad_input():
    with pytest.raises(ValueError, match="same shape"):
        slope(np.zeros(3), np.zeros(4))
    with pytest.raises(ValueError, match="at least 2"):
        slope(np.zeros(1), np.zeros(1))


@pytest.mark.parametrize("n", [1, 3, 11, 21])
def test_slope_span_n_of_a_line(n):
    x = np.linspace(0, 10, 201)
    assert np.allclose(slope_span_n(x, -4 * x + 1, n=n), -4.0)


def test_slope_span_n_smooths_noise():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 2001)
    y = 2 * x + rng.normal(0, 0.05, x.size)
    wide = slope_span_n(x, y, n=51)
    narrow = slope_span_n(x, y, n=3)
    assert np.std(wide) < np.std(narrow)


def test_slope_span_n_never_returns_zero_when_asked():
    x = np.linspace(0, 1, 101)
    der = slope_span_n(x, np.zeros_like(x), n=5, nozeros=True)
    assert np.all(der != 0)


def test_slope_span_n_allows_zero_when_asked():
    x = np.linspace(0, 1, 101)
    der = slope_span_n(x, np.zeros_like(x), n=5, nozeros=False)
    assert np.all(der == 0)


def test_slope_span_n_stays_centred_near_the_edges():
    """A centred difference recovers a quadratic's derivative exactly.

    The span shrinks symmetrically towards each edge, so the result stays
    exact right up to the two end points, which have no centred stencil
    left and fall back to a one-sided difference.
    """
    x = np.linspace(-1, 1, 401)
    der = slope_span_n(x, x**2, n=21, nozeros=False)
    assert np.abs(der[1:-1] - 2 * x[1:-1]).max() < 1e-9
    assert der[0] == pytest.approx(x[0] + x[1])
    assert der[-1] == pytest.approx(x[-1] + x[-2])


@pytest.mark.parametrize("n", [0, 2, -1])
def test_slope_span_n_rejects_bad_span(n):
    x = np.linspace(0, 1, 11)
    with pytest.raises(ValueError, match="odd integer"):
        slope_span_n(x, x, n=n)
