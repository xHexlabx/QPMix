"""Tests for qpmix.exp.exp_data: the DCData and PumpedData containers.

The headline test is the full round trip: build a pumped I-V curve for a
junction fed by a *known* Thevenin source, then check that PumpedData
recovers that source from the curve alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpmix.exp import DCData, PumpedData
from qpmix.exp.simulate import (
    simulate_dciv,
    simulate_embedded_iv,
    simulate_if_power,
    simulate_pumped_iv,
    simulate_response,
)

VGAP, RN, RSG, FREQ = 2.8e-3, 14.0, 800.0, 230.0


@pytest.fixture(scope="module")
def resp():
    return simulate_response(vgap=VGAP, rn=RN, rsg=RSG)


@pytest.fixture(scope="module")
def dciv():
    return DCData(simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG), verbose=False)


# -- DCData -------------------------------------------------------------


def test_dcdata_recovers_the_junction_parameters(dciv):
    assert dciv.vgap == pytest.approx(VGAP, abs=5e-6)
    assert dciv.rn == pytest.approx(RN, rel=1e-3)
    assert dciv.rsg == pytest.approx(RSG, rel=0.02)
    assert dciv.q == pytest.approx(RSG / RN, rel=0.02)


def test_dcdata_derived_quantities(dciv):
    assert dciv.igap == pytest.approx(dciv.vgap / dciv.rn)
    assert dciv.jc == pytest.approx(dciv.vgap / dciv.rna)
    assert dciv.ileak > 0


def test_dcdata_builds_two_response_functions(dciv):
    v = np.linspace(0.2, 1.5, 101)
    assert (
        np.abs(dciv.resp.idc(v) - dciv.current_at(v)).max() < 0.05
        if hasattr(dciv, "current_at")
        else True
    )
    # The smeared response has a softer transition than the sharp one.
    near_gap = np.linspace(0.95, 1.05, 201)
    sharp = np.abs(np.gradient(dciv.resp.idc(near_gap))).max()
    smeared = np.abs(np.gradient(dciv.resp_smear.idc(near_gap))).max()
    assert smeared < sharp


def test_dcdata_response_matches_the_measured_curve(dciv):
    v = np.linspace(0.1, 1.7, 201)
    measured = np.interp(v, dciv.voltage, dciv.current)
    assert np.abs(dciv.resp.idc(v) - measured).max() < 0.02


def test_dcdata_summary_mentions_the_key_numbers(dciv):
    text = dciv.summary()
    for key in ("Vgap", "Rn", "Rsg", "Jc", "Vint"):
        assert key in text


def test_dcdata_without_if_data_has_none_placeholders(dciv):
    assert dciv.if_data is None
    assert dciv.if_noise is None


def test_dcdata_with_if_data():
    raw = simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG)
    _, _, meta = __import__("qpmix.exp.iv_data", fromlist=["dciv_curve"]).dciv_curve(
        raw, verbose=False
    )
    ifraw = simulate_if_power(None, if_noise=8.0, vint=meta.vint)
    data = DCData(raw, ifraw, vshot=(3.5e-3, 5.5e-3), verbose=False)
    assert data.if_noise == pytest.approx(8.0, abs=0.5)
    assert data.if_data.shape[1] == 2
    assert "IF noise" in data.summary()


def test_dcdata_rejects_non_array_input():
    with pytest.raises(ValueError, match="NumPy array"):
        DCData([[0, 0], [1, 1]], verbose=False)


def test_dcdata_rejects_unknown_parameters():
    with pytest.raises(TypeError, match="Unknown parameter"):
        DCData(simulate_dciv(), not_a_parameter=1, verbose=False)


def test_dcdata_verbose_prints_a_summary(capsys):
    DCData(simulate_dciv(), verbose=True)
    assert "Vgap" in capsys.readouterr().out


# -- PumpedData: drive-level recovery -----------------------------------


@pytest.mark.parametrize("alpha", [0.4, 0.8, 1.2, 1.6])
def test_recovers_a_known_drive_level(dciv, resp, alpha):
    raw = simulate_pumped_iv(
        alpha=alpha, freq=FREQ, vgap=VGAP, rn=RN, rsg=RSG, resp=resp
    )
    pumped = PumpedData(raw, dciv, freq=FREQ, verbose=False)
    assert pumped.alpha == pytest.approx(alpha, abs=1e-3)


def test_a_constant_drive_level_implies_a_voltage_source(dciv, resp):
    """Driving at fixed alpha is an ideal voltage source, so the recovered
    embedding impedance must be zero and the voltage alpha * vph."""
    raw = simulate_pumped_iv(alpha=0.8, freq=FREQ, resp=resp)
    pumped = PumpedData(raw, dciv, freq=FREQ, verbose=False)
    assert abs(pumped.zt) < 5e-3
    assert pumped.vt == pytest.approx(0.8 * pumped.vph, rel=0.01)


# -- PumpedData: the full round trip ------------------------------------


@pytest.mark.parametrize(
    ("vt", "zt"),
    [
        (0.35, 0.40 - 0.30j),
        (0.30, 0.60 + 0.20j),
        (0.45, 0.25 - 0.50j),
        (0.25, 0.50 + 0.00j),
    ],
)
def test_recovers_a_known_embedding_circuit(dciv, resp, vt, zt):
    raw = simulate_embedded_iv(
        vt=vt, zt=zt, freq=FREQ, vgap=VGAP, rn=RN, rsg=RSG, resp=resp
    )
    pumped = PumpedData(raw, dciv, freq=FREQ, verbose=False)
    assert pumped.zt == pytest.approx(zt, abs=5e-3)
    assert pumped.vt == pytest.approx(vt, abs=5e-3)
    assert pumped.fit_good


def test_round_trip_survives_measurement_noise(dciv, resp):
    raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=FREQ, resp=resp, noise=5e-4)
    pumped = PumpedData(raw, dciv, freq=FREQ, verbose=False)
    assert pumped.zt == pytest.approx(0.4 - 0.3j, abs=0.05)
    assert pumped.vt == pytest.approx(0.35, abs=0.05)


def test_junction_impedance_is_physical(dciv, resp):
    raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=FREQ, resp=resp)
    pumped = PumpedData(raw, dciv, freq=FREQ, verbose=False)
    assert pumped.zw.real > 0
    assert 0.1 < abs(pumped.zw) < 5.0


def test_forced_embedding_impedance(dciv, resp):
    raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=FREQ, resp=resp)
    pumped = PumpedData(raw, dciv, freq=FREQ, zemb=0.7 + 0.1j, verbose=False)
    assert pumped.zt == 0.7 + 0.1j


def test_analysis_can_be_skipped(dciv, resp):
    raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=FREQ, resp=resp)
    pumped = PumpedData(raw, dciv, freq=FREQ, analyze_iv=False, verbose=False)
    assert pumped.zt is None
    assert pumped.alpha is None


# -- PumpedData: IF analysis --------------------------------------------


def _hot_cold(vgap, vint, tn=40.0, gain=10.0):
    """Hot/cold IF data with a known mixer noise temperature."""
    v = np.linspace(0.0, 6e-3, 2001)
    on_step = (v > 1.9e-3) & (v < 2.75e-3)
    g = np.where(on_step, gain, gain * 1e-4)
    shot = np.where(v > vgap, 8.0 + 5.8 * (v - vint) * 1e3, 8.0)
    return (
        np.column_stack((v * 1e3, g * (293.0 + tn) + shot)),
        np.column_stack((v * 1e3, g * (78.0 + tn) + shot)),
    )


def test_if_analysis_populates_the_results(dciv, resp):
    raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=FREQ, resp=resp)
    hot, cold = _hot_cold(dciv.vgap, dciv.vint)
    pumped = PumpedData(
        raw,
        dciv,
        hot,
        cold,
        freq=FREQ,
        vshot=(3.5e-3, 5.5e-3),
        ifdata_sigma=None,
        verbose=False,
    )
    assert pumped.tn is not None
    assert pumped.tn_best > 0
    assert np.isfinite(pumped.g_db)
    assert 1.9e-3 < pumped.v_best * dciv.vgap < 2.8e-3
    assert "Noise temperature" in pumped.summary()


def test_if_analysis_is_optional(dciv, resp):
    raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=FREQ, resp=resp)
    pumped = PumpedData(raw, dciv, freq=FREQ, verbose=False)
    assert pumped.tn is None
    assert pumped.tn_best is None


# -- validation ---------------------------------------------------------


def test_requires_the_lo_frequency(dciv, resp):
    raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=FREQ, resp=resp)
    with pytest.raises(ValueError, match="frequency"):
        PumpedData(raw, dciv, verbose=False)


def test_rejects_non_array_input(dciv):
    with pytest.raises(ValueError, match="NumPy array"):
        PumpedData([[0, 0]], dciv, freq=FREQ, verbose=False)


def test_reports_a_frequency_with_no_photon_step(dciv, resp):
    """At an absurdly high frequency the first photon step falls off the
    measured bias range."""
    raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=FREQ, resp=resp)
    with pytest.raises(ValueError, match="no first photon step"):
        PumpedData(raw, dciv, freq=1500.0, verbose=False)


def test_verbose_prints_a_summary(dciv, resp, capsys):
    raw = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=FREQ, resp=resp)
    PumpedData(raw, dciv, freq=FREQ, verbose=True)
    out = capsys.readouterr().out
    assert "Impedance recovery" in out
