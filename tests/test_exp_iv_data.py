"""Tests for qpmix.exp.iv_data.

These are round-trip tests: synthetic I-V curves are built from known
parameters, distorted the way a real measurement is, and the analysis is
asked to recover the numbers that went in.
"""

from __future__ import annotations

import numpy as np
import pytest

from qpmix.exp.iv_data import (
    CURR_UNITS,
    VOLT_UNITS,
    _take_one_pass,
    dciv_curve,
    iv_curve,
)
from qpmix.exp.simulate import simulate_dciv, simulate_pumped_iv

VGAP, RN, RSG = 2.8e-3, 14.0, 800.0


@pytest.fixture
def clean_iv():
    return simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG)


# -- recovering the junction parameters ---------------------------------


def test_recovers_gap_voltage(clean_iv):
    _, _, dc = dciv_curve(clean_iv, verbose=False)
    assert dc.vgap == pytest.approx(VGAP, abs=5e-6)


def test_recovers_normal_resistance(clean_iv):
    _, _, dc = dciv_curve(clean_iv, verbose=False)
    assert dc.rn == pytest.approx(RN, rel=1e-3)


def test_recovers_subgap_resistance(clean_iv):
    _, _, dc = dciv_curve(clean_iv, verbose=False)
    assert dc.rsg == pytest.approx(RSG, rel=0.02)


def test_derived_quantities_are_consistent(clean_iv):
    _, _, dc = dciv_curve(clean_iv, verbose=False)
    import scipy.constants as sc

    assert dc.igap == pytest.approx(dc.vgap / dc.rn)
    assert dc.fgap == pytest.approx(sc.e * dc.vgap / sc.h)


def test_normalized_curve_passes_through_the_gap(clean_iv):
    v, i, _ = dciv_curve(clean_iv, verbose=False)
    assert np.interp(1.0, v, i) == pytest.approx(0.5, abs=0.05)


def test_normalized_curve_is_ohmic_above_the_gap(clean_iv):
    v, i, _ = dciv_curve(clean_iv, verbose=False)
    mask = (v > 1.4) & (v < 2.0)
    assert np.polyfit(v[mask], i[mask], 1)[0] == pytest.approx(1.0, rel=1e-3)


def test_output_is_resampled_onto_a_uniform_grid(clean_iv):
    v, i, _ = dciv_curve(clean_iv, npts=1001, vmax=5e-3, verbose=False)
    assert v.size == i.size == 1001
    assert np.allclose(np.diff(v), np.diff(v)[0])


# -- corrections --------------------------------------------------------


def test_recovers_an_injected_offset():
    raw = simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG, voffset=-8e-5, ioffset=-2e-6)
    _, _, dc = dciv_curve(raw, verbose=False)
    assert dc.offset[0] == pytest.approx(-8e-5, abs=3e-6)
    assert dc.offset[1] == pytest.approx(-2e-6, abs=1e-7)
    assert dc.vgap == pytest.approx(VGAP, abs=1e-5)


def test_offset_recovery_survives_noise():
    """The offset fit smooths first, so it tolerates far more noise than an
    unsmoothed reflection fit would."""
    raw = simulate_dciv(
        vgap=VGAP, rn=RN, rsg=RSG, voffset=-8e-5, ioffset=-2e-6, noise=2e-3
    )
    _, _, dc = dciv_curve(raw, verbose=False)
    assert dc.offset[0] == pytest.approx(-8e-5, abs=5e-6)


def test_explicit_offsets_are_used_verbatim(clean_iv):
    _, _, dc = dciv_curve(clean_iv, voffset=1e-5, ioffset=2e-7, verbose=False)
    assert dc.offset == (1e-5, 2e-7)


def test_unipolar_data_cannot_fit_an_offset():
    """With no negative branch there is nothing to reflect against."""
    raw = simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG)
    raw = raw[raw[:, 0] >= 0]
    _, _, dc = dciv_curve(raw, verbose=False)
    assert dc.offset == (0.0, 0.0)


def test_gain_multipliers_are_applied():
    raw = simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG)
    scaled = raw.copy()
    scaled[:, 0] /= 1.05
    scaled[:, 1] /= 0.97
    _, _, dc = dciv_curve(scaled, v_multiplier=1.05, i_multiplier=0.97, verbose=False)
    assert dc.vgap == pytest.approx(VGAP, abs=1e-5)
    assert dc.rn == pytest.approx(RN, rel=1e-3)


def test_series_resistance_is_removed():
    """Adding a series resistance raises the apparent normal resistance;
    correcting for it must bring it back."""
    raw = simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG)
    rseries = 2.0
    distorted = raw.copy()
    distorted[:, 0] += distorted[:, 1] * rseries * 1e-3 / VOLT_UNITS["mV"]

    _, _, uncorrected = dciv_curve(distorted, verbose=False)
    _, _, corrected = dciv_curve(distorted, rseries=rseries, verbose=False)
    assert uncorrected.rn > RN + 1.0
    assert corrected.rn == pytest.approx(RN, rel=0.05)
    assert corrected.rseries == rseries


def test_double_sweep_is_reduced_to_one_pass():
    raw = simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG, double_sweep=True)
    _, _, dc = dciv_curve(raw, verbose=False)
    assert dc.vgap == pytest.approx(VGAP, abs=1e-5)


@pytest.mark.parametrize("reverse", [False, True])
def test_take_one_pass_leaves_monotonic_data_alone(reverse):
    v = np.linspace(-1, 1, 21)
    i = v * 2
    if reverse:
        v, i = v[::-1], i[::-1]
    gv, gi = _take_one_pass(v, i)
    assert np.array_equal(gv, np.linspace(-1, 1, 21))
    assert np.array_equal(gi, np.linspace(-1, 1, 21) * 2)


def test_take_one_pass_handles_tiny_input():
    v = np.array([1.0])
    gv, _ = _take_one_pass(v, v)
    assert gv.size == 1


def test_filtering_reduces_noise_but_keeps_the_gap():
    raw = simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG, noise=2e-3)
    _, i_filt, dc_filt = dciv_curve(raw, verbose=False)
    _, i_raw, dc_raw = dciv_curve(raw, filter_data=False, verbose=False)
    assert np.std(np.diff(i_filt)) < np.std(np.diff(i_raw))
    assert dc_filt.vgap == pytest.approx(dc_raw.vgap, abs=2e-5)


# -- units --------------------------------------------------------------


@pytest.mark.parametrize("v_fmt", list(VOLT_UNITS))
@pytest.mark.parametrize("i_fmt", list(CURR_UNITS))
def test_all_unit_combinations_give_the_same_answer(v_fmt, i_fmt):
    raw = simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG, v_fmt=v_fmt, i_fmt=i_fmt)
    _, _, dc = dciv_curve(raw, v_fmt=v_fmt, i_fmt=i_fmt, verbose=False)
    assert dc.vgap == pytest.approx(VGAP, abs=5e-6)
    assert dc.rn == pytest.approx(RN, rel=1e-3)


# -- pumped curves ------------------------------------------------------


def test_pumped_curve_lands_on_the_dc_axes(clean_iv):
    _, _, dc = dciv_curve(clean_iv, verbose=False)
    raw = simulate_pumped_iv(alpha=0.8, freq=230.0)
    v, i = iv_curve(raw, dc, verbose=False)
    assert v.size == i.size
    assert np.allclose(np.diff(v), np.diff(v)[0])


def test_pumping_adds_current_below_the_gap(clean_iv):
    v_dc, i_dc, dc = dciv_curve(clean_iv, verbose=False)
    raw = simulate_pumped_iv(alpha=1.0, freq=230.0)
    v, i = iv_curve(raw, dc, verbose=False)
    step = (v > 0.75) & (v < 0.95)
    assert i[step].max() > np.interp(0.95, v_dc, i_dc) * 2


# -- validation ---------------------------------------------------------


@pytest.mark.parametrize("bad", [np.zeros(10), np.zeros((10, 3))])
def test_rejects_wrongly_shaped_input(bad):
    with pytest.raises(ValueError, match="2-column"):
        dciv_curve(bad, verbose=False)


def test_rejects_unknown_units(clean_iv):
    with pytest.raises(ValueError, match="voltage unit"):
        dciv_curve(clean_iv, v_fmt="kV", verbose=False)
    with pytest.raises(ValueError, match="current unit"):
        dciv_curve(clean_iv, i_fmt="kA", verbose=False)


def test_rejects_unknown_parameters(clean_iv):
    with pytest.raises(TypeError, match="Unknown parameter"):
        dciv_curve(clean_iv, not_a_parameter=1, verbose=False)


def test_reports_an_empty_normal_resistance_window(clean_iv):
    with pytest.raises(ValueError, match="normal resistance"):
        dciv_curve(clean_iv, vrn=(50.0, 60.0), verbose=False)


def test_reports_an_out_of_range_subgap_voltage(clean_iv):
    with pytest.raises(ValueError, match="outside the measured range"):
        dciv_curve(clean_iv, vrsg=50.0, verbose=False)


@pytest.mark.parametrize("npts", [1001, 3001, 6001])
def test_results_are_stable_across_resampling_density(clean_iv, npts):
    _, _, dc = dciv_curve(clean_iv, npts=npts, verbose=False)
    assert dc.vgap == pytest.approx(VGAP, abs=2e-5)
    assert dc.rn == pytest.approx(RN, rel=1e-3)
    assert dc.rsg == pytest.approx(RSG, rel=0.05)


def test_local_line_widens_until_it_is_conditioned():
    """A fixed voltage window can collapse to one point on a coarse grid,
    which makes the polynomial fit singular; this one takes a fixed number
    of nearest samples instead."""
    from qpmix.exp.iv_data import _local_line

    v = np.linspace(0.0, 1.0, 11)
    p = _local_line(v, 3.0 * v + 1.0, centre=0.5, min_points=5)
    assert p[0] == pytest.approx(3.0)
    assert p[1] == pytest.approx(1.0)


def test_rejects_an_oversized_filter_window(clean_iv):
    with pytest.raises(ValueError, match="filter_nwind"):
        dciv_curve(clean_iv, npts=11, filter_nwind=51, verbose=False)


def test_warns_about_implausible_normal_resistance(capsys):
    raw = simulate_dciv(vgap=VGAP, rn=RN, rsg=RSG, i_fmt="A")
    dciv_curve(raw, i_fmt="mA", verbose=True)
    assert "normal resistance" in capsys.readouterr().out
