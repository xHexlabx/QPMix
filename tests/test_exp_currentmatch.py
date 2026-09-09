"""Tests for qpmix.exp.currentmatch and the offset verification API.

The recovery tests are round trips: a pumped I-V curve is simulated for a
*known* Thevenin source with the full engine, and the fit is asked to find
that source again from the curve alone.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import qpmix
from qpmix.exp.currentmatch import (
    VOLTAGE_METHODS,
    _admissible,
    _unpack,
    current_residual,
    default_guesses,
    recover_zemb_current_match,
    voltage_windows,
)
from qpmix.exp.iv_data import check_offset, dciv_curve
from qpmix.exp.simulate import simulate_dciv

VPH, VT, ZT = 0.30, 0.55, 0.42 - 0.31j


@pytest.fixture(scope="module")
def resp():
    return qpmix.RespFnPolynomial(50, verbose=False)


@pytest.fixture(scope="module")
def pumped(resp):
    """A pumped I-V curve for a known source, from the full engine."""
    cct = qpmix.EmbeddingCircuit(1, 1, vb_npts=601, vb_max=2.0)
    cct.freq[1] = VPH
    cct.vt[1, 1] = VT
    cct.zt[1, 1] = ZT
    vj = qpmix.harmonic_balance(cct, resp, num_b=15, verbose=False)
    idc = qpmix.qtcurrent(vj, cct, resp, 0.0, num_b=15, verbose=False)
    return cct.vb, idc


# -- bias windows -------------------------------------------------------


def test_first_photon_window_sits_on_the_first_step():
    ((lo, hi),) = voltage_windows("first_photon", 0.3)
    assert 1 - 0.3 < lo < hi < 1.0


@pytest.mark.parametrize("method", VOLTAGE_METHODS)
def test_every_method_gives_increasing_windows(method):
    windows = voltage_windows(method, 0.25)
    assert windows
    for lo, hi in windows:
        assert hi > lo
    lows = [w[0] for w in windows]
    assert lows == sorted(lows)


def test_photon_steps_gives_one_window_per_step():
    windows = voltage_windows("photon_steps", 0.25, n_steps=4)
    assert len(windows) == 4
    # Disjoint: each window ends before the next one begins.
    for (_, hi), (lo, _) in itertools.pairwise(windows):
        assert lo > hi


def test_full_subgap_stays_below_the_gap():
    ((lo, hi),) = voltage_windows("full_subgap", 0.3)
    assert 0 < lo < hi < 1.0


def test_full_range_extends_above_the_gap():
    ((_, hi),) = voltage_windows("full_range", 0.3)
    assert hi > 1.0


def test_fit_range_narrows_the_window():
    wide = voltage_windows("first_photon", 0.3, fit_range=(0.05, 0.95))[0]
    narrow = voltage_windows("first_photon", 0.3, fit_range=(0.35, 0.65))[0]
    assert wide[0] < narrow[0] < narrow[1] < wide[1]


def test_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown method"):
        voltage_windows("wishful", 0.3)


def test_rejects_a_non_positive_photon_voltage():
    with pytest.raises(ValueError, match="must be positive"):
        voltage_windows("first_photon", 0.0)


# -- starting points ----------------------------------------------------


def test_default_grid_has_nine_points():
    assert len(default_guesses(1)) == 9


def test_grid_stays_at_nine_points_for_more_harmonics():
    """Higher harmonics start from the fundamental's impedance, so the grid
    does not grow as 9**harmonics."""
    for harmonics in (1, 2, 3):
        guesses = default_guesses(harmonics)
        assert len(guesses) == 9
        assert len(guesses[0]) == 1 + 2 * harmonics


def test_unpack_round_trip():
    vt, zt = _unpack([0.5, 0.4, -0.3, 0.2, 0.1], 2)
    assert vt == 0.5
    assert zt == (0.4 - 0.3j, 0.2 + 0.1j)


def test_unpack_rejects_the_wrong_length():
    with pytest.raises(ValueError, match="Expected 3 parameters"):
        _unpack([0.5, 0.4], 1)


@pytest.mark.parametrize(
    ("vt", "zt", "ok"),
    [
        (0.5, (0.4 - 0.3j,), True),
        (0.5, (-0.1 - 0.3j,), False),  # negative resistance
        (-0.5, (0.4 - 0.3j,), False),  # negative source voltage
        (0.5, (0.4 - 50j,), False),  # runaway reactance
        (np.nan, (0.4 - 0.3j,), False),
    ],
)
def test_admissibility_rules(vt, zt, ok):
    assert _admissible(vt, zt, max_reactance=5.0) is ok


# -- recovery -----------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("method", VOLTAGE_METHODS)
def test_recovers_a_known_source(resp, pumped, method):
    voltage, current = pumped
    result = recover_zemb_current_match(
        resp, voltage, current, VPH, method=method, num_b=15, guesses="seeded"
    )
    assert result.zt[0] == pytest.approx(ZT, abs=5e-3)
    assert result.vt == pytest.approx(VT, abs=5e-3)
    assert result.err < 1e-3


@pytest.mark.slow
def test_seeding_needs_far_fewer_evaluations(resp, pumped):
    """The voltage-match answer is already close, so it collapses the
    nine-point grid to a handful of runs."""
    voltage, current = pumped
    grid = recover_zemb_current_match(
        resp, voltage, current, VPH, num_b=15, guesses="grid"
    )
    seeded = recover_zemb_current_match(
        resp, voltage, current, VPH, num_b=15, guesses="seeded"
    )
    assert seeded.n_runs < grid.n_runs
    assert seeded.n_calls < grid.n_calls
    assert seeded.zt[0] == pytest.approx(grid.zt[0], abs=5e-3)


@pytest.mark.slow
def test_agrees_with_the_voltage_match_method(resp, pumped):
    """Two independent routes to the same source: a closed-form load-line
    fit, and a full simulation fitted to the measured curve."""
    from qpmix.exp.tucker import ac_current, recover_alpha
    from qpmix.exp.zemb import recover_zemb

    voltage, current = pumped
    lo, hi = voltage_windows("first_photon", VPH)[0]
    mask = (voltage >= lo) & (voltage <= hi)
    alpha = recover_alpha(resp, voltage[mask], current[mask], VPH, num_b=15)
    vj = alpha * VPH
    zj = vj / ac_current(resp, voltage[mask], VPH, alpha, num_b=15)
    voltage_match = recover_zemb(vj, zj)

    current_match = recover_zemb_current_match(
        resp,
        voltage,
        current,
        VPH,
        method="full_subgap",
        num_b=15,
        guesses="seeded",
    )
    assert current_match.zt[0] == pytest.approx(voltage_match.zt, abs=5e-3)
    assert current_match.vt == pytest.approx(voltage_match.vt, abs=5e-3)


@pytest.mark.slow
def test_recovers_a_second_harmonic_impedance(resp):
    cct = qpmix.EmbeddingCircuit(1, 2, vb_npts=601, vb_max=2.0)
    cct.freq[1] = VPH
    cct.vt[1, 1] = VT
    cct.zt[1, 1] = ZT
    cct.zt[1, 2] = 0.25 + 0.40j
    vj = qpmix.harmonic_balance(cct, resp, num_b=15, verbose=False)
    idc = qpmix.qtcurrent(vj, cct, resp, 0.0, num_b=15, verbose=False)

    result = recover_zemb_current_match(
        resp,
        cct.vb,
        idc,
        VPH,
        harmonics=2,
        method="full_subgap",
        num_b=15,
        guesses="seeded",
    )
    assert len(result.zt) == 2
    assert result.zt[0] == pytest.approx(ZT, abs=0.03)
    assert result.zt[1] == pytest.approx(0.25 + 0.40j, abs=0.05)


@pytest.mark.slow
def test_explicit_windows_override_the_method(resp, pumped):
    voltage, current = pumped
    result = recover_zemb_current_match(
        resp,
        voltage,
        current,
        VPH,
        windows=[(0.75, 0.95)],
        num_b=15,
        guesses=[[0.55, 0.42, -0.31]],
    )
    assert result.windows == ((0.75, 0.95),)
    assert result.zt[0] == pytest.approx(ZT, abs=0.05)


@pytest.mark.slow
def test_result_carries_the_simulated_curve(resp, pumped):
    voltage, current = pumped
    result = recover_zemb_current_match(
        resp, voltage, current, VPH, num_b=15, guesses=[[0.55, 0.42, -0.31]]
    )
    assert result.simulated.shape == result.voltage.shape
    assert np.all(np.isfinite(result.simulated))
    assert result.n_calls > 0


def test_reports_a_window_with_no_data(resp, pumped):
    voltage, current = pumped
    with pytest.raises(ValueError, match="measured points fall"):
        recover_zemb_current_match(
            resp, voltage, current, VPH, windows=[(50.0, 60.0)], num_b=9
        )


def test_rejects_mismatched_inputs(resp):
    with pytest.raises(ValueError, match="same shape"):
        recover_zemb_current_match(resp, np.zeros(5), np.zeros(6), VPH)


def test_rejects_zero_harmonics(resp, pumped):
    voltage, current = pumped
    with pytest.raises(ValueError, match="at least one harmonic"):
        recover_zemb_current_match(resp, voltage, current, VPH, harmonics=0)


def test_rejects_unknown_guess_option(resp, pumped):
    voltage, current = pumped
    with pytest.raises(ValueError, match="Unknown guesses"):
        recover_zemb_current_match(
            resp, voltage, current, VPH, guesses="magic", num_b=9
        )


def test_rejects_empty_guess_list(resp, pumped):
    voltage, current = pumped
    with pytest.raises(ValueError, match="No starting points"):
        recover_zemb_current_match(resp, voltage, current, VPH, guesses=[])


@pytest.mark.slow
def test_reports_when_no_solution_is_admissible(resp, pumped):
    """Forcing the search to start deep in unphysical territory and pinning
    it there must be reported, not returned as an answer."""
    voltage, current = pumped
    with pytest.raises(ValueError, match="admissible"):
        recover_zemb_current_match(
            resp,
            voltage,
            current,
            VPH,
            num_b=9,
            guesses=[[-1.0, -1.0, 0.0]],
            maxiter=1,
        )


# -- residual helper ----------------------------------------------------


@pytest.mark.slow
def test_residual_is_smallest_at_the_true_source(resp, pumped):
    voltage, current = pumped
    exact = current_residual(resp, voltage, current, VPH, VT, ZT, num_b=15)
    for wrong in (ZT + 0.2, ZT - 0.3j, ZT * 2):
        assert (
            current_residual(resp, voltage, current, VPH, VT, wrong, num_b=15) > exact
        )


@pytest.mark.slow
def test_residual_matches_what_the_fit_reports(resp, pumped):
    """The same quantity the optimiser minimises, exposed on its own."""
    voltage, current = pumped
    result = recover_zemb_current_match(
        resp, voltage, current, VPH, num_b=15, guesses=[[VT, ZT.real, ZT.imag]]
    )
    direct = current_residual(
        resp,
        result.voltage,
        result.current,
        VPH,
        result.vt,
        result.zt,
        windows=result.windows,
        num_b=15,
    )
    assert direct == pytest.approx(result.err, rel=1e-9)


@pytest.mark.slow
def test_residual_accepts_several_harmonics(resp):
    cct = qpmix.EmbeddingCircuit(1, 2, vb_npts=401, vb_max=1.5)
    cct.freq[1] = VPH
    cct.vt[1, 1] = VT
    cct.zt[1, 1] = ZT
    cct.zt[1, 2] = 0.25 + 0.40j
    vj = qpmix.harmonic_balance(cct, resp, num_b=15, verbose=False)
    idc = qpmix.qtcurrent(vj, cct, resp, 0.0, num_b=15, verbose=False)

    exact = current_residual(resp, cct.vb, idc, VPH, VT, (ZT, 0.25 + 0.40j), num_b=15)
    wrong = current_residual(resp, cct.vb, idc, VPH, VT, (ZT, 0.90 - 0.90j), num_b=15)
    assert exact < wrong


# -- offset verification ------------------------------------------------


def test_offset_check_passes_on_a_corrected_curve():
    raw = simulate_dciv(voffset=-8e-5, ioffset=-2e-6)
    _, _, dc = dciv_curve(raw, verbose=False)
    report = check_offset(dc.vraw, dc.iraw)
    assert report.passed
    assert abs(report.i_at_zero) < 1e-7


def test_offset_check_catches_a_bad_correction():
    raw = simulate_dciv()
    _, _, dc = dciv_curve(raw, voffset=2e-4, ioffset=0.0, verbose=False)
    report = check_offset(dc.vraw, dc.iraw)
    assert not report.passed
    assert report.asym_rms > report.tolerance


def test_offset_asymmetry_grows_with_the_error():
    raw = simulate_dciv()
    errors = []
    for voffset in (0.0, 5e-5, 2e-4):
        _, _, dc = dciv_curve(raw, voffset=voffset, ioffset=0.0, verbose=False)
        errors.append(check_offset(dc.vraw, dc.iraw).asym_rms)
    assert errors == sorted(errors)


def test_offset_check_reports_the_tolerance_it_used():
    raw = simulate_dciv()
    _, _, dc = dciv_curve(raw, verbose=False)
    report = check_offset(dc.vraw, dc.iraw, tolerance=1e-3)
    assert report.tolerance == 1e-3
    assert report.passed


def test_offset_summary_states_the_verdict():
    raw = simulate_dciv()
    _, _, dc = dciv_curve(raw, verbose=False)
    assert "OK" in check_offset(dc.vraw, dc.iraw).summary()


def test_offset_check_needs_data_spanning_the_window():
    v = np.linspace(0.0, 1e-3, 101)
    with pytest.raises(ValueError, match="check window"):
        check_offset(v, v, window=5e-4)
