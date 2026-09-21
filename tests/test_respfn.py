"""Tests for qpmix.respfn."""

from __future__ import annotations

import numpy as np
import pytest

import qpmix.mathfn.ivcurve_models as iv
from qpmix.respfn import (
    RespFn,
    RespFnExponential,
    RespFnFromIVData,
    RespFnPerfect,
    RespFnPolynomial,
    _perfect_kk_tabulated,
    _uniform_grid,
)

ALL_CLASSES = [RespFnPerfect, RespFnPolynomial, RespFnExponential]


@pytest.fixture(scope="module")
def responses():
    return {cls.__name__: cls(verbose=False) for cls in ALL_CLASSES}


@pytest.mark.parametrize("name", [c.__name__ for c in ALL_CLASSES])
def test_response_is_kk_plus_j_idc(responses, name):
    resp = responses[name]
    v = np.linspace(-4, 4, 501)
    out = resp(v)
    assert np.allclose(out.real, resp.ikk(v))
    assert np.allclose(out.imag, resp.idc(v))


@pytest.mark.parametrize("name", [c.__name__ for c in ALL_CLASSES])
def test_call_and_resp_are_the_same(responses, name):
    v = np.linspace(-2, 2, 101)
    assert np.array_equal(responses[name](v), responses[name].resp(v))


@pytest.mark.parametrize("name", [c.__name__ for c in ALL_CLASSES])
def test_conjugate_and_swap_helpers(responses, name):
    resp = responses[name]
    v = np.linspace(-2, 2, 101)
    assert np.allclose(resp.resp_conj(v), np.conj(resp(v)))
    assert np.allclose(resp.resp_swap(v), 1j * np.conj(resp(v)))


@pytest.mark.parametrize("name", [c.__name__ for c in ALL_CLASSES])
def test_dc_curve_is_odd(responses, name):
    v = np.linspace(0.02, 6, 601)
    assert np.abs(responses[name].idc(v) + responses[name].idc(-v)).max() < 1e-6


@pytest.mark.parametrize("name", [c.__name__ for c in ALL_CLASSES])
def test_kk_curve_is_even(responses, name):
    v = np.linspace(0.02, 6, 601)
    assert np.abs(responses[name].ikk(v) - responses[name].ikk(-v)).max() < 1e-6


@pytest.mark.parametrize("name", [c.__name__ for c in ALL_CLASSES])
def test_ohmic_far_above_the_gap(responses, name):
    v = np.linspace(6, 25, 201)
    assert np.abs(responses[name].idc(v) - v).max() < 0.05


def test_polynomial_matches_its_own_model():
    resp = RespFnPolynomial(50, verbose=False)
    v = np.linspace(-6, 6, 4001)
    assert np.abs(resp.idc(v) - iv.polynomial(v, 50)).max() < 1e-4


def test_exponential_matches_its_own_model():
    resp = RespFnExponential(rsg=1000, verbose=False)
    v = np.linspace(-6, 6, 4001)
    expected = iv.exponential(v, 2.8e-3, 14, 1000, 4e4)
    assert np.abs(resp.idc(v) - expected).max() < 1e-4


def test_perfect_reproduces_the_step():
    resp = RespFnPerfect(verbose=False)
    assert resp.idc(0.5) == pytest.approx(0.0, abs=1e-9)
    assert resp.idc(2.0) == pytest.approx(2.0, abs=1e-9)
    assert resp.idc(-2.0) == pytest.approx(-2.0, abs=1e-9)


def test_perfect_kk_matches_the_analytic_result_away_from_the_gap():
    resp = RespFnPerfect(verbose=False)
    v = np.linspace(1.05, 20, 401)
    assert np.abs(resp.ikk(v) - iv.perfect_kk(v)).max() < 1e-3


def test_perfect_kk_table_is_bounded_at_the_singularity():
    """QMix substitutes 100 at v = +/-1, which would poison the stencil."""
    v = _uniform_grid(35.0, 5e-4)
    kk = _perfect_kk_tabulated(v)
    assert np.all(np.isfinite(kk))
    assert np.abs(kk).max() < 10


def test_higher_resolution_lowers_the_interpolation_error():
    v = np.linspace(-3, 3, 12345)
    coarse = RespFnPolynomial(50, verbose=False, dv=4e-3)
    fine = RespFnPolynomial(50, verbose=False, dv=5e-4)
    truth = iv.polynomial(v, 50)
    assert np.abs(fine.idc(v) - truth).max() < np.abs(coarse.idc(v) - truth).max()


def test_derivatives_match_a_numerical_derivative():
    resp = RespFnPolynomial(50, verbose=False)
    v = np.linspace(1.5, 5, 4001)
    # np.gradient falls back to one-sided differences at the two end points,
    # so compare the interior.
    assert np.abs(resp.didc(v) - np.gradient(resp.idc(v), v))[1:-1].max() < 1e-4
    assert np.abs(resp.dikk(v) - np.gradient(resp.ikk(v), v))[1:-1].max() < 1e-4


def test_didc_of_the_perfect_curve_is_a_step():
    resp = RespFnPerfect(verbose=False)
    assert resp.didc(0.5) == pytest.approx(0.0, abs=1e-6)
    assert resp.didc(2.0) == pytest.approx(1.0, abs=1e-6)


def test_smearing_softens_the_transition():
    sharp = RespFnPerfect(verbose=False)
    smeared = RespFnPerfect(verbose=False, v_smear=0.05)
    v = np.linspace(0.9, 1.1, 201)
    assert (
        np.abs(np.gradient(smeared.idc(v))).max()
        < np.abs(np.gradient(sharp.idc(v))).max()
    )
    # Far from the gap the two must agree.
    far = np.linspace(3, 10, 101)
    assert np.abs(smeared.idc(far) - sharp.idc(far)).max() < 1e-3


def test_from_iv_data_recovers_the_source_curve():
    v = np.linspace(0, 10, 4001)
    i = iv.polynomial(v, 30)
    resp = RespFnFromIVData(v, i, verbose=False)
    check = np.linspace(0.1, 8, 501)
    assert np.abs(resp.idc(check) - iv.polynomial(check, 30)).max() < 1e-3


def test_from_iv_data_handles_unsorted_duplicated_input():
    v = np.linspace(0, 10, 2001)
    i = iv.polynomial(v, 30)
    shuffled = np.random.default_rng(0).permutation(v.size)
    v2 = np.r_[v[shuffled], v[:50]]
    i2 = np.r_[i[shuffled], i[:50]]
    resp = RespFnFromIVData(v2, i2, verbose=False)
    check = np.linspace(0.1, 8, 301)
    assert np.abs(resp.idc(check) - iv.polynomial(check, 30)).max() < 1e-3


def test_respfn_stores_the_tabulated_curves():
    resp = RespFnPolynomial(50, verbose=False)
    assert resp.voltage.shape == resp.current.shape == resp.current_kk.shape
    assert resp.voltage_kk is resp.voltage
    assert resp.voltage[0] == pytest.approx(-35.0)
    assert resp.voltage[-1] == pytest.approx(35.0)


def test_table_grid_places_the_gap_on_a_node():
    v = _uniform_grid(35.0, 5e-4)
    assert np.abs(v - 1.0).min() < 1e-12
    assert np.abs(v - 0.0).min() < 1e-12


def test_respfn_rejects_data_that_does_not_start_at_zero():
    v = np.linspace(0.1, 10, 101)
    with pytest.raises(ValueError, match="must be zero"):
        RespFn(v, iv.polynomial(v, 30), verbose=False)


def test_respfn_rejects_data_that_stops_too_early():
    v = np.linspace(0, 3, 101)
    with pytest.raises(ValueError, match="at least 5"):
        RespFn(v, iv.polynomial(v, 30), verbose=False)


def test_respfn_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        RespFn(np.linspace(0, 10, 101), np.zeros(100), verbose=False)


def test_from_iv_data_rejects_data_that_stops_below_vlimit():
    v = np.linspace(0, 1.5, 101)
    with pytest.raises(ValueError, match="vlimit"):
        RespFnFromIVData(v, iv.polynomial(v, 30), verbose=False)


def test_from_iv_data_extends_ohmically_above_vlimit():
    """Measured curves stop just above the gap; the response function has to
    reach tens of gap voltages, so the tail is continued ohmically."""
    v = np.linspace(0, 2.0, 2001)
    resp = RespFnFromIVData(v, iv.polynomial(v, 30), verbose=False)
    far = np.array([10.0, 20.0, 30.0])
    assert np.abs(resp.idc(far) - far).max() < 1e-3


def test_unknown_keyword_is_rejected():
    with pytest.raises(TypeError, match="Unexpected keyword"):
        RespFnPolynomial(50, verbose=False, max_npts_dc=101)


def test_verbose_output_mentions_the_table(capsys):
    RespFnPolynomial(50, verbose=True)
    assert "Tabulated on" in capsys.readouterr().out


def test_perfect_verbose_reports_the_analytic_path(capsys):
    RespFnPerfect(verbose=True)
    assert "analytic" in capsys.readouterr().out


def test_from_iv_data_uses_the_whole_measured_tail_by_default():
    """The offset i - v of a real junction keeps drifting above the gap, and
    the KK transform weights the tail logarithmically: cutting the data at
    1.8 (QMix's rule, and the old default) biases ikk across the sub-gap
    region.  Using the sweep out to 3.5 gets much closer to the answer from
    the full curve, and the DC curve below the cut is untouched."""

    def drifting(v):
        return iv.polynomial(v, 30) - 0.1 * v**2 / (1 + v**2)

    v_full = np.linspace(0, 35, 7001)
    ref = RespFn(v_full, drifting(v_full), verbose=False)
    v = np.linspace(0, 3.5, 3501)
    full = RespFnFromIVData(v, drifting(v), verbose=False)
    cut = RespFnFromIVData(v, drifting(v), verbose=False, vlimit=1.8)
    check = np.linspace(0.3, 1.5, 121)
    err_full = np.abs(full.ikk(check) - ref.ikk(check)).max()
    err_cut = np.abs(cut.ikk(check) - ref.ikk(check)).max()
    assert err_cut > 3e-3
    assert err_full < err_cut / 2
    assert np.abs(full.idc(check) - cut.idc(check)).max() < 1e-8
