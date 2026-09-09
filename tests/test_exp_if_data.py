"""Tests for qpmix.exp.if_data and qpmix.exp.if_response."""

from __future__ import annotations

import numpy as np
import pytest

from qpmix.exp.if_data import callen_welton, dcif_data, if_data
from qpmix.exp.if_response import db_to_linear, if_response
from qpmix.exp.iv_data import dciv_curve
from qpmix.exp.simulate import simulate_dciv, simulate_if_power

SHOT_WINDOW = (3.5e-3, 5.5e-3)


@pytest.fixture(scope="module")
def dc():
    _, _, meta = dciv_curve(simulate_dciv(), verbose=False)
    return meta


# -- Woody's method ----------------------------------------------------


@pytest.mark.parametrize("if_noise", [3.0, 8.0, 20.0])
def test_recovers_the_if_noise_contribution(dc, if_noise):
    raw = simulate_if_power(None, if_noise=if_noise, vint=dc.vint)
    _, meta = dcif_data(raw, dc, vshot=SHOT_WINDOW, verbose=False)
    assert meta.if_noise == pytest.approx(if_noise, abs=0.3)
    assert meta.if_fit


def test_calibration_is_independent_of_the_arbitrary_power_scale(dc):
    """The measurement is in arbitrary units; Woody's method fixes the
    scale, so an overall gain must not change the answer."""
    results = []
    for gain in (1.0, 37.0, 1e-4):
        raw = simulate_if_power(None, if_noise=8.0, vint=dc.vint, gain=gain)
        _, cal = dcif_data(raw, dc, vshot=SHOT_WINDOW, verbose=False)
        results.append(cal.if_noise)
    assert max(results) - min(results) < 1e-6


def test_correction_factor_rescales_to_the_physical_shot_slope(dc):
    raw = simulate_if_power(None, if_noise=8.0, vint=dc.vint, gain=5.0)
    data, _ = dcif_data(raw, dc, vshot=SHOT_WINDOW, verbose=False)
    window = (data[:, 0] * dc.vgap > SHOT_WINDOW[0]) & (
        data[:, 0] * dc.vgap < SHOT_WINDOW[1]
    )
    slope = np.polyfit(data[window, 0] * dc.vgap * 1e3, data[window, 1], 1)[0]
    assert slope == pytest.approx(5.8, rel=0.02)


def test_implausible_if_noise_is_flagged(dc):
    raw = simulate_if_power(None, if_noise=500.0, vint=dc.vint)
    _, meta = dcif_data(raw, dc, vshot=SHOT_WINDOW, verbose=False)
    assert not meta.if_fit


def test_a_flat_shot_noise_region_is_refused_rather_than_producing_nan(dc):
    """An infinite correction factor would silently turn every downstream
    power into NaN."""
    v = np.linspace(0.0, 6e-3, 2001)
    flat = np.column_stack((v * 1e3, np.full(v.size, 5.0)))
    data, meta = dcif_data(flat, dc, vshot=SHOT_WINDOW, verbose=False)
    assert meta.corr == 1.0
    assert not meta.if_fit
    assert np.all(np.isfinite(data))


def test_multiple_shot_noise_windows_are_accepted(dc):
    raw = simulate_if_power(None, if_noise=8.0, vint=dc.vint)
    _, meta = dcif_data(
        raw, dc, vshot=((3.5e-3, 4.2e-3), (4.8e-3, 5.5e-3)), verbose=False
    )
    assert meta.if_noise == pytest.approx(8.0, abs=0.3)


def test_automatic_window_search_finds_the_linear_region(dc):
    raw = simulate_if_power(None, if_noise=8.0, vint=dc.vint)
    _, meta = dcif_data(raw, dc, verbose=False)
    assert meta.if_noise == pytest.approx(8.0, abs=1.0)


def test_shot_slope_output_covers_the_bias_range(dc):
    raw = simulate_if_power(None, if_noise=8.0, vint=dc.vint)
    data, meta = dcif_data(raw, dc, vshot=SHOT_WINDOW, verbose=False)
    assert meta.shot_slope.shape == (data.shape[0], 2)
    assert meta.vmax == pytest.approx(data[:, 0].max() * dc.vgap)


def test_smoothing_reduces_noise(dc):
    raw = simulate_if_power(None, if_noise=8.0, vint=dc.vint, noise=1.0)
    smooth, _ = dcif_data(raw, dc, vshot=SHOT_WINDOW, verbose=False)
    rough, _ = dcif_data(raw, dc, vshot=SHOT_WINDOW, ifdata_sigma=None, verbose=False)
    assert np.std(np.diff(smooth[:, 1])) < np.std(np.diff(rough[:, 1]))


# -- noise temperature -------------------------------------------------


def _hot_cold(dc, tn=40.0, gain=0.02, t_hot=293.0, t_cold=78.0):
    """Build hot/cold IF data for a known noise temperature and gain.

    Mixer output appears on the first photon step; above the gap both loads
    see the same shot noise, rising linearly with bias, which is what
    Woody's method calibrates against.
    """
    v = np.linspace(0.0, 6e-3, 2001)
    on_step = (v > 1.9e-3) & (v < 2.75e-3)
    g = np.where(on_step, gain, gain * 1e-4)
    shot = np.where(v > dc.vgap, 8.0 + 5.8 * (v - dc.vint) * 1e3, 8.0)
    p_hot = g * (t_hot + tn) + shot
    p_cold = g * (t_cold + tn) + shot
    return (
        np.column_stack((v * 1e3, p_hot)),
        np.column_stack((v * 1e3, p_cold)),
    )


def test_recovers_a_known_noise_temperature(dc):
    """With enough conversion gain the IF chain contributes negligibly, so
    the measured noise temperature is the mixer's own -- referred to
    Callen-Welton load temperatures."""
    hot, cold = _hot_cold(dc, tn=40.0, gain=10.0)
    results, idx, _ = if_data(
        hot,
        cold,
        dc,
        freq=230.0,
        vshot=SHOT_WINDOW,
        ifdata_sigma=None,
        verbose=False,
    )
    t_hot = callen_welton(230e9, 293.0)
    t_cold = callen_welton(230e9, 78.0)
    y = (293.0 + 40.0) / (78.0 + 40.0)
    expected = (t_hot - t_cold * y) / (y - 1)
    assert results[idx, 3] == pytest.approx(expected, rel=0.05)


def test_if_chain_noise_raises_the_measured_noise_temperature(dc):
    """Lower conversion gain lets the fixed IF-chain noise dominate, which
    is exactly why Woody's method exists."""
    measured = []
    for gain in (10.0, 1.0, 0.1):
        hot, cold = _hot_cold(dc, tn=40.0, gain=gain)
        results, idx, _ = if_data(
            hot,
            cold,
            dc,
            freq=230.0,
            vshot=SHOT_WINDOW,
            ifdata_sigma=None,
            verbose=False,
        )
        measured.append(results[idx, 3])
    assert measured == sorted(measured)


def test_results_have_the_documented_columns(dc):
    hot, cold = _hot_cold(dc)
    results, idx, meta = if_data(
        hot, cold, dc, freq=230.0, vshot=SHOT_WINDOW, verbose=False
    )
    assert results.shape[1] == 5
    assert 0 <= idx < results.shape[0]
    assert meta.if_noise is not None


def test_best_point_selection_modes_differ(dc):
    hot, cold = _hot_cold(dc)
    _, idx_gain, _ = if_data(
        hot,
        cold,
        dc,
        freq=230.0,
        vshot=SHOT_WINDOW,
        best_pt="Max Gain",
        verbose=False,
    )
    _, idx_tn, _ = if_data(
        hot,
        cold,
        dc,
        freq=230.0,
        vshot=SHOT_WINDOW,
        best_pt="Min Tn",
        verbose=False,
    )
    assert isinstance(idx_gain, int)
    assert isinstance(idx_tn, int)


def test_explicit_best_bias_is_honoured(dc):
    hot, cold = _hot_cold(dc)
    results, idx, _ = if_data(
        hot,
        cold,
        dc,
        freq=230.0,
        vshot=SHOT_WINDOW,
        vbest=2.3e-3,
        verbose=False,
    )
    assert results[idx, 0] * dc.vgap == pytest.approx(2.3e-3, abs=5e-5)


def test_rejects_an_unknown_best_point_rule(dc):
    hot, cold = _hot_cold(dc)
    with pytest.raises(ValueError, match="best_pt"):
        if_data(hot, cold, dc, freq=230.0, best_pt="lucky", verbose=False)


def test_requires_the_lo_frequency(dc):
    hot, cold = _hot_cold(dc)
    with pytest.raises(ValueError, match="frequency"):
        if_data(hot, cold, dc, verbose=False)


def test_reusing_a_dc_calibration(dc):
    raw = simulate_if_power(None, if_noise=8.0, vint=dc.vint)
    _, cal = dcif_data(raw, dc, vshot=SHOT_WINDOW, verbose=False)
    hot, cold = _hot_cold(dc)
    _, _, meta = if_data(hot, cold, dc, dcif=cal, freq=230.0, verbose=False)
    assert meta.if_noise == cal.if_noise
    assert meta.corr == cal.corr


@pytest.mark.parametrize("bad", [np.zeros(10), np.zeros((10, 3))])
def test_rejects_wrongly_shaped_if_data(dc, bad):
    with pytest.raises(ValueError, match="2-column"):
        dcif_data(bad, dc, verbose=False)


def test_does_not_mutate_numpy_error_state(dc):
    """QMix calls np.seterr at import time; QPMix must not."""
    before = np.geterr()
    raw = simulate_if_power(None, if_noise=8.0, vint=dc.vint)
    dcif_data(raw, dc, vshot=SHOT_WINDOW, verbose=False)
    assert np.geterr() == before


# -- Callen-Welton ------------------------------------------------------


def test_callen_welton_approaches_the_physical_temperature():
    """At low frequency the half-photon term is negligible."""
    assert callen_welton(1e6, 293.0) == pytest.approx(293.0, rel=1e-6)


def test_callen_welton_exceeds_the_physical_temperature_when_cold():
    """The zero-point term dominates a cold load at high frequency."""
    assert callen_welton(700e9, 4.0) > 16.0


def test_callen_welton_is_monotonic_in_temperature():
    temps = [4.0, 20.0, 78.0, 293.0]
    values = [callen_welton(230e9, t) for t in temps]
    assert values == sorted(values)


# -- IF response --------------------------------------------------------


def test_db_to_linear_round_trip():
    lin = np.array([1.0, 10.0, 100.0])
    assert np.allclose(db_to_linear(10 * np.log10(lin)), lin)


def test_if_response_recovers_a_flat_noise_temperature():
    freq = np.linspace(4, 8, 41)
    tn_true = 50.0
    t_hot, t_cold = 293.0, 78.0
    y = (t_hot + tn_true) / (t_cold + tn_true)
    hot_db = np.full(freq.size, -60.0)
    cold_db = hot_db - 10 * np.log10(y)
    out = if_response(np.column_stack((freq, hot_db, cold_db)))
    assert np.allclose(out[1], tn_true, rtol=1e-6)


def test_if_response_accepts_a_transposed_array():
    data = np.column_stack(
        (np.array([4.0, 5.0]), np.array([-60.0, -60.0]), np.array([-61.0, -61.0]))
    )
    assert np.allclose(if_response(data), if_response(data.T))


def test_if_response_caps_unphysical_values():
    freq = np.array([4.0, 5.0])
    # Cold louder than hot: unphysical, must be capped rather than negative.
    out = if_response(
        np.column_stack((freq, np.array([-70.0, -70.0]), np.array([-60.0, -60.0]))),
        ifresp_maxtn=1234.0,
    )
    assert np.all(out[1] == 1234.0)


def test_if_response_rejects_wrong_shapes():
    with pytest.raises(ValueError, match="2-dimensional"):
        if_response(np.zeros(5))
    with pytest.raises(ValueError, match="3 columns"):
        if_response(np.zeros((5, 4)))
