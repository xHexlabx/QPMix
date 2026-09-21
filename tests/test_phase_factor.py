"""Tests for qpmix.phase_factor."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

from qpmix.phase_factor import (
    _as_nb_tuple,
    calculate_phase_factor_coeff,
    choose_num_theta,
    drive_level,
)


def _vj(num_f, num_p, npts, seed=0, vmax=0.4):
    rng = np.random.default_rng(seed)
    vj = np.zeros((num_f + 1, num_p + 1, npts), dtype=complex)
    for f in range(1, num_f + 1):
        for p in range(1, num_p + 1):
            mag = rng.uniform(0.05, vmax, npts) / p
            vj[f, p] = mag * np.exp(1j * rng.uniform(-np.pi, np.pi, npts))
    return vj


def test_single_harmonic_gives_bessel_functions():
    """With one harmonic, C_k reduces to J_k(alpha) exp(-i k phi)."""
    npts, num_b = 5, 10
    freq = np.array([0.0, 0.4])
    alpha_target, phi = 1.3, 0.7
    vj = np.zeros((2, 2, npts), dtype=complex)
    vj[1, 1] = alpha_target * freq[1] * np.exp(1j * phi)

    ckh = calculate_phase_factor_coeff(vj, freq, 1, 1, num_b)
    for k in range(-num_b, num_b + 1):
        expected = jv(k, alpha_target) * np.exp(-1j * k * phi)
        assert ckh[1, k] == pytest.approx(np.full(npts, expected), abs=1e-12)


def test_coefficients_are_unitary():
    """|C_k|^2 sums to 1: the phase factor has unit modulus."""
    vj = _vj(2, 2, 32, seed=1)
    freq = np.array([0.0, 0.30, 0.32])
    ckh = calculate_phase_factor_coeff(vj, freq, 2, 2, 40)
    power = (np.abs(ckh[1:]) ** 2).sum(axis=1)
    assert np.abs(power - 1.0).max() < 1e-10


def test_zero_drive_gives_a_delta():
    vj = np.zeros((2, 2, 4), dtype=complex)
    freq = np.array([0.0, 0.4])
    ckh = calculate_phase_factor_coeff(vj, freq, 1, 1, 8)
    assert np.allclose(ckh[1, 0], 1.0)
    assert np.abs(ckh[1, 1:]).max() < 1e-12


def test_fft_and_direct_methods_agree():
    vj = _vj(2, 2, 24, seed=2)
    freq = np.array([0.0, 0.30, 0.32])
    fft = calculate_phase_factor_coeff(vj, freq, 2, 2, 20, method="fft")
    direct = calculate_phase_factor_coeff(vj, freq, 2, 2, 20, method="direct")
    assert np.abs(fft - direct).max() < 1e-12


@pytest.mark.parametrize("num_p", [3, 4])
def test_fft_beats_the_truncated_convolution(num_p):
    """The direct recursion truncates every intermediate convolution at
    +/-num_b; the transform never forms them, so it is closer to the
    untruncated answer."""
    vj = _vj(2, num_p, 16, seed=3)
    freq = np.array([0.0, 0.30, 0.32])
    nb = 15

    def crop(c, nb_from):
        keep = np.r_[
            np.arange(0, nb + 1), np.arange(2 * nb_from + 1 - nb, 2 * nb_from + 1)
        ]
        return c[:, keep]

    truth = crop(
        calculate_phase_factor_coeff(vj, freq, 2, num_p, 90, method="direct"), 90
    )
    fft = calculate_phase_factor_coeff(vj, freq, 2, num_p, nb, method="fft")
    direct = calculate_phase_factor_coeff(vj, freq, 2, num_p, nb, method="direct")
    assert np.abs(fft - truth).max() < np.abs(direct - truth).max()
    assert np.abs(fft - truth).max() < 1e-13


def test_result_is_independent_of_the_fft_length_once_large_enough():
    vj = _vj(1, 2, 16, seed=4)
    freq = np.array([0.0, 0.3])
    base = calculate_phase_factor_coeff(vj, freq, 1, 2, 15, num_theta=1024)
    for m in (128, 256, 512):
        got = calculate_phase_factor_coeff(vj, freq, 1, 2, 15, num_theta=m)
        assert np.abs(got - base).max() < 1e-12


def test_output_shape_and_per_tone_num_b():
    vj = _vj(2, 1, 7, seed=5)
    freq = np.array([0.0, 0.30, 0.32])
    ckh = calculate_phase_factor_coeff(vj, freq, 2, 1, (5, 12))
    assert ckh.shape == (3, 25, 7)
    # The array is sized by the largest num_b (12).  Tone 1 is truncated at
    # 5, so its unused middle band must be exactly zero, while tone 2 fills
    # the array completely.
    assert np.abs(ckh[1, 6:-5]).max() == 0.0
    assert np.abs(ckh[2]).min() > 0.0


def test_drive_level_definition():
    """alpha = |vj| / (p * freq), phi = arg(vj)."""
    vj = np.zeros((2, 3, 4), dtype=complex)
    vj[1, 1] = 0.6 * np.exp(1j * 0.25)
    vj[1, 2] = 0.4 * np.exp(-1j * 1.0)
    freq = np.array([0.0, 0.3])
    alpha, phi = drive_level(vj, freq, 1, 2)
    assert alpha[1, 1] == pytest.approx(0.6 / 0.3)
    assert alpha[1, 2] == pytest.approx(0.4 / (2 * 0.3))
    assert phi[1, 1] == pytest.approx(0.25)
    assert phi[1, 2] == pytest.approx(-1.0)


def test_choose_num_theta_grows_with_drive_and_num_b():
    assert choose_num_theta(0.0, 15) < choose_num_theta(50.0, 15)
    assert choose_num_theta(1.0, 5) < choose_num_theta(1.0, 60)


def test_choose_num_theta_is_a_fast_length():
    n = choose_num_theta(3.0, 15)
    remaining = n
    for factor in (2, 3, 5, 7, 11):
        while remaining % factor == 0:
            remaining //= factor
    assert remaining == 1


def test_hard_driven_junction_stays_accurate():
    """A large alpha needs a long transform; choose_num_theta must supply it."""
    npts, num_b = 8, 30
    freq = np.array([0.0, 0.05])
    vj = np.zeros((2, 2, npts), dtype=complex)
    vj[1, 1] = 1.5  # alpha = 30
    ckh = calculate_phase_factor_coeff(vj, freq, 1, 1, num_b)
    for k in range(-num_b, num_b + 1):
        assert ckh[1, k, 0] == pytest.approx(jv(k, 30.0), abs=1e-11)


def test_rejects_unknown_method():
    vj = _vj(1, 1, 4)
    with pytest.raises(ValueError, match="Unknown method"):
        calculate_phase_factor_coeff(vj, np.array([0.0, 0.3]), 1, 1, 5, method="magic")


@pytest.mark.parametrize(
    ("num_b", "num_f", "expected"),
    [(15, 2, (15, 15)), ((4, 5, 6), 2, (4, 5)), ([7, 8], 2, (7, 8))],
)
def test_num_b_normalization(num_b, num_f, expected):
    assert _as_nb_tuple(num_b, num_f) == expected


def test_num_b_tuple_must_cover_every_tone():
    with pytest.raises(ValueError, match="one value of num_b"):
        _as_nb_tuple((5,), 3)


def test_bessel_orders_tracks_the_dropped_weight():
    """sum_n J_n(a)^2 = 1, so the limit returned must be the first one whose
    tail is inside the tolerance, and the one below it must not be."""
    from scipy.special import jv

    from qpmix.phase_factor import bessel_orders

    assert bessel_orders(0.0) == 0
    for alpha, tol in ((1.12, 1e-6), (1.12, 1e-9), (11.0, 1e-9), (40.0, 1e-9)):
        b = bessel_orders(alpha, tol)

        def tail(k, alpha=alpha):
            return 1.0 - np.sum(jv(np.arange(-k, k + 1), alpha) ** 2)

        assert tail(b) <= tol
        assert tail(b - 1) > tol


def test_required_num_b_scales_with_the_harmonic_and_takes_the_worst_bias():
    from qpmix.phase_factor import bessel_orders, required_num_b

    freq = np.array([0.0, 0.3, 0.4])
    vj = np.zeros((3, 3, 4), dtype=complex)
    vj[1, 1, :] = 0.3 * np.array([0.1, 1.0, 2.0, 0.5])  # alpha up to 2, harmonic 1
    vj[2, 2, :] = 1.2  # alpha 1.2 / (2 * 0.4) = 1.5 on harmonic 2 of tone 2
    need = required_num_b(vj, freq, 2, 2, tol=1e-9)
    assert need == (bessel_orders(2.0, 1e-9), 2 * bessel_orders(1.5, 1e-9))
