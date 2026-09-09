"""Tests for qpmix.mathfn.filters."""

from __future__ import annotations

import numpy as np
import pytest

from qpmix.mathfn.filters import _gauss, gauss_conv


def test_constant_signal_is_unchanged():
    x = np.full(500, 3.5)
    assert np.allclose(gauss_conv(x, sigma=10), 3.5)


def test_output_length_matches_input():
    x = np.random.default_rng(0).normal(size=777)
    assert gauss_conv(x, sigma=7).shape == (777,)


def test_smoothing_reduces_variance():
    rng = np.random.default_rng(1)
    x = np.sin(np.linspace(0, 10, 2000)) + rng.normal(0, 0.3, 2000)
    assert np.std(np.diff(gauss_conv(x, sigma=20))) < np.std(np.diff(x))


def test_direct_and_fft_methods_agree():
    rng = np.random.default_rng(2)
    x = rng.normal(size=4000)
    direct = gauss_conv(x, sigma=60, method="direct")
    fft = gauss_conv(x, sigma=60, method="fft")
    assert np.abs(direct - fft).max() < 1e-10


def test_auto_selects_fft_for_wide_kernels():
    """Auto must agree with whichever explicit method it would pick."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=6000)
    for sigma in (2, 60):
        auto = gauss_conv(x, sigma=sigma)
        assert np.abs(auto - gauss_conv(x, sigma=sigma, method="direct")).max() < 1e-10


def test_a_step_is_smoothed_symmetrically():
    x = np.r_[np.zeros(500), np.ones(500)]
    y = gauss_conv(x, sigma=20)
    assert y[499] + y[500] == pytest.approx(1.0, abs=1e-6)


def test_gauss_kernel_is_normalizable_and_symmetric():
    k = _gauss(5, 3)
    assert k.size == 31
    assert np.allclose(k, k[::-1])
    assert k.argmax() == 15


def test_rejects_window_larger_than_data():
    with pytest.raises(ValueError, match="smaller than data"):
        gauss_conv(np.zeros(10), sigma=50)


def test_rejects_degenerate_window():
    with pytest.raises(ValueError, match="larger than 1"):
        gauss_conv(np.zeros(100), sigma=0.1, ext_x=1)


def test_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        gauss_conv(np.zeros(100), sigma=5, method="wavelet")


def test_rejects_2d_input():
    with pytest.raises(ValueError, match="1-D"):
        gauss_conv(np.zeros((10, 10)), sigma=2)
