"""Tests for qpmix.mathfn.kktrans."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from qpmix.mathfn.ivcurve_models import perfect, perfect_kk, polynomial
from qpmix.mathfn.kktrans import hilbert_transform, kk_trans, kk_trans_trapz


def test_hilbert_transform_matches_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=1024)
    assert np.abs(hilbert_transform(x) - hilbert(x).imag).max() < 1e-12


def test_hilbert_of_cosine_is_sine():
    n = 4096
    t = 2 * np.pi * np.arange(n) / n
    x = np.cos(8 * t)
    assert np.abs(hilbert_transform(x) - np.sin(8 * t)).max() < 1e-10


def test_hilbert_transform_is_linear():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=512), rng.normal(size=512)
    lhs = hilbert_transform(2 * a - 3 * b)
    rhs = 2 * hilbert_transform(a) - 3 * hilbert_transform(b)
    assert np.abs(lhs - rhs).max() < 1e-12


def test_kk_transform_of_perfect_iv_matches_analytic_result():
    v = np.linspace(-35, 35, 70001)
    ikk = kk_trans(v, perfect(v), n=20)
    # Stay away from the logarithmic singularity at v = +/-1.
    mask = (np.abs(v) > 1.1) & (np.abs(v) < 10)
    assert np.abs(ikk[mask] - perfect_kk(v[mask])).max() < 5e-3


def test_kk_transform_agrees_with_direct_quadrature():
    v = np.linspace(-20, 20, 2001)
    i = polynomial(v, 20)
    fast = kk_trans(v, i, n=50)
    slow = kk_trans_trapz(v, i)
    mask = np.abs(v) < 5
    assert np.abs(fast[mask] - slow[mask]).max() < 0.02


def test_kk_transform_is_even():
    """The KK transform of an odd I-V curve is an even function of bias."""
    v = np.linspace(-30, 30, 6001)
    ikk = kk_trans(v, polynomial(v, 30))
    assert np.abs(ikk - ikk[::-1]).max() < 1e-8


def test_padding_converges():
    """More padding must change the answer less and less."""
    v = np.linspace(-35, 35, 14001)
    i = polynomial(v, 50)
    ref = kk_trans(v, i, n=200)
    errors = [np.abs(kk_trans(v, i, n=n) - ref).max() for n in (5, 20, 80)]
    assert errors[0] > errors[1] > errors[2]


def test_output_shape_matches_input():
    v = np.linspace(-10, 10, 1001)
    assert kk_trans(v, polynomial(v, 30)).shape == (1001,)


@pytest.mark.parametrize("fn", [kk_trans, kk_trans_trapz])
def test_rejects_non_uniform_spacing(fn):
    v = np.r_[np.linspace(0, 1, 50), np.linspace(1.5, 5, 50)]
    with pytest.raises(ValueError, match="constant"):
        fn(v, np.zeros_like(v))


def test_kk_trans_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        kk_trans(np.linspace(0, 1, 10), np.zeros(9))
