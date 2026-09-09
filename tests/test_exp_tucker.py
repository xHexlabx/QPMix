"""Tests for qpmix.exp.tucker.

Tucker theory is checked two ways: against a plain loop over the defining
sums, and against the full multi-tone engine in
:func:`qpmix.qtcurrent.qtcurrent`, which derives the same currents by a
completely different route.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

import qpmix
from qpmix.exp.tucker import (
    ac_current,
    bessel_ladder,
    pumped_iv_curve,
    recover_alpha,
)

VPH = 0.3


@pytest.fixture(scope="module")
def resp():
    return qpmix.RespFnPolynomial(50, verbose=False)


@pytest.fixture
def photon_step():
    """Bias voltages on the flat part of the first photon step."""
    return np.linspace(1 - VPH + 0.25 * VPH, 1 - 0.2 * VPH, 121)


# -- Bessel ladder -----------------------------------------------------


@pytest.mark.parametrize("num_b", [1, 5, 20])
def test_bessel_ladder_matches_scipy(num_b):
    alpha = np.array([0.0, 0.4, 1.7, 5.0])
    ladder = bessel_ladder(alpha, num_b)
    top = num_b + 1
    assert ladder.shape == (2 * top + 1, alpha.size)
    for n in range(-top, top + 1):
        assert np.allclose(ladder[n + top], jv(n, alpha), atol=1e-14)


def test_bessel_ladder_obeys_the_reflection_formula():
    alpha = np.linspace(0, 4, 17)
    ladder = bessel_ladder(alpha, 6)
    top = 7
    for n in range(1, top + 1):
        assert np.allclose(ladder[top - n], (-1) ** n * ladder[top + n])


def test_bessel_ladder_at_zero_drive():
    ladder = bessel_ladder(np.zeros(3), 4)
    assert np.allclose(ladder[5], 1.0)
    assert np.abs(np.delete(ladder, 5, axis=0)).max() == 0.0


# -- the defining sums -------------------------------------------------


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0, 2.5])
def test_pumped_iv_matches_a_plain_loop(resp, alpha):
    vb = np.linspace(0.05, 1.5, 201)
    expected = sum(jv(n, alpha) ** 2 * resp.idc(vb + n * VPH) for n in range(-20, 21))
    assert np.abs(pumped_iv_curve(resp, vb, VPH, alpha) - expected).max() < 1e-14


@pytest.mark.parametrize("alpha", [0.0, 0.3, 1.0, 2.5])
def test_ac_current_matches_a_plain_loop(resp, alpha):
    vb = np.linspace(0.05, 1.5, 201)
    expected = np.zeros(vb.size, dtype=complex)
    for n in range(-20, 21):
        jn, jm, jp = jv(n, alpha), jv(n - 1, alpha), jv(n + 1, alpha)
        expected += jn * (jm + jp) * resp.idc(vb + n * VPH)
        expected += 1j * jn * (jm - jp) * resp.ikk(vb + n * VPH)
    assert np.abs(ac_current(resp, vb, VPH, alpha) - expected).max() < 1e-14


@pytest.mark.parametrize("alpha", [0.3, 1.0, 2.5])
def test_tucker_matches_the_full_mtsda_engine(resp, alpha):
    """An independent check: qtcurrent gets here by multi-tone spectral
    domain analysis, not by Bessel sums."""
    cct = qpmix.EmbeddingCircuit(1, 1, vb_npts=201, vb_max=1.5)
    cct.freq[1] = VPH
    vj = cct.initialize_vj()
    vj[1, 1, :] = alpha * VPH

    idc = qpmix.qtcurrent(vj, cct, resp, 0.0, num_b=25, verbose=False)
    iac = qpmix.qtcurrent(vj, cct, resp, VPH, num_b=25, verbose=False)
    assert np.abs(pumped_iv_curve(resp, cct.vb, VPH, alpha, 25) - idc).max() < 1e-13
    assert np.abs(ac_current(resp, cct.vb, VPH, alpha, 25) - iac).max() < 1e-13


def test_undriven_junction_gives_the_dc_iv_curve(resp):
    vb = np.linspace(0.05, 1.5, 101)
    assert np.abs(pumped_iv_curve(resp, vb, VPH, 0.0) - resp.idc(vb)).max() < 1e-14
    assert np.abs(ac_current(resp, vb, VPH, 0.0)).max() < 1e-14


def test_alpha_may_vary_with_bias(resp):
    vb = np.linspace(0.5, 1.2, 51)
    alpha = np.linspace(0.2, 1.2, 51)
    got = pumped_iv_curve(resp, vb, VPH, alpha)
    for i in (0, 25, 50):
        one = pumped_iv_curve(resp, vb[i : i + 1], VPH, alpha[i])
        assert got[i] == pytest.approx(one[0])


# -- drive-level recovery ----------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.4, 1.0, 1.8, 2.4])
def test_recover_alpha_inverts_the_pumped_curve(resp, photon_step, alpha):
    idc = pumped_iv_curve(resp, photon_step, VPH, alpha)
    got = recover_alpha(resp, photon_step, idc, VPH, alpha_max=1.5)
    assert np.abs(got - alpha).max() < 1e-9


def test_recover_alpha_beats_fixed_bisection(resp, photon_step):
    """QMix takes 15 bisection steps, so it cannot do better than
    alpha_max / 2**15; a safeguarded Newton iteration is not so limited."""
    idc = pumped_iv_curve(resp, photon_step, VPH, 0.9)
    got = recover_alpha(resp, photon_step, idc, VPH, alpha_max=1.5)
    assert np.abs(got - 0.9).max() < 1.5 / 2**15 / 1000


def test_recover_alpha_handles_a_varying_drive(resp, photon_step):
    alpha = np.linspace(0.1, 1.4, photon_step.size)
    idc = pumped_iv_curve(resp, photon_step, VPH, alpha)
    got = recover_alpha(resp, photon_step, idc, VPH, alpha_max=1.5)
    assert np.abs(got - alpha).max() < 1e-8


def test_recover_alpha_of_an_unpumped_curve_is_zero(resp, photon_step):
    got = recover_alpha(resp, photon_step, resp.idc(photon_step), VPH)
    assert np.abs(got).max() < 1e-6


def test_recover_alpha_stays_bounded_where_inversion_is_impossible(resp):
    """Above the gap the pumped current falls with drive level, so there is
    no solution; the result must stay in range rather than diverge."""
    vb = np.linspace(0.05, 2.0, 101)
    absurd = np.full(vb.size, 50.0)
    got = recover_alpha(resp, vb, absurd, VPH, alpha_max=1.5)
    assert np.all(np.isfinite(got))
    assert got.min() >= 0.0
    assert got.max() <= 4 * 1.5 + 1e-9


def test_recover_alpha_rejects_mismatched_shapes(resp):
    with pytest.raises(ValueError, match="same shape"):
        recover_alpha(resp, np.zeros(5), np.zeros(6), VPH)


def test_recover_alpha_is_insensitive_to_the_initial_guess(resp, photon_step):
    idc = pumped_iv_curve(resp, photon_step, VPH, 1.1)
    for alpha_max in (0.5, 1.5, 3.0):
        got = recover_alpha(resp, photon_step, idc, VPH, alpha_max=alpha_max)
        assert np.abs(got - 1.1).max() < 1e-8
