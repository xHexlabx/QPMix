"""Tests for qpmix.exp.zemb (embedding-circuit recovery)."""

from __future__ import annotations

import numpy as np
import pytest

from qpmix.exp.zemb import (
    GOOD_ERROR,
    error_function,
    error_surface,
    recover_zemb,
    source_voltage,
)


def load_line(vt, zt, zj):
    """Junction voltage produced by a Thevenin source across ``zj``."""
    return vt * zj / (zt + zj)


@pytest.fixture
def junction():
    """A spread of junction impedances, as a photon step would give."""
    return np.array([0.35 + 0.30j, 0.45 + 0.18j, 0.55 + 0.05j, 0.68 - 0.10j])


# -- the two published equations ---------------------------------------


def test_error_is_zero_for_a_perfect_load_line(junction):
    zt, vt = 0.3 - 0.4j, 0.5
    vj = load_line(vt, zt, junction)
    assert error_function(vj, junction, zt) == pytest.approx(0.0, abs=1e-20)


def test_error_grows_away_from_the_true_impedance(junction):
    zt, vt = 0.3 - 0.4j, 0.5
    vj = load_line(vt, zt, junction)
    errors = [
        error_function(vj, junction, zt + delta) for delta in (0.0, 0.05, 0.2, 0.6)
    ]
    assert errors == sorted(errors)


def test_source_voltage_is_recovered_exactly(junction):
    zt, vt = 0.3 - 0.4j, 0.5
    vj = load_line(vt, zt, junction)
    assert source_voltage(vj, junction, zt) == pytest.approx(vt, rel=1e-12)


# -- error surface ------------------------------------------------------


def test_surface_matches_the_scalar_function(junction):
    zt, vt = 0.3 - 0.4j, 0.5
    vj = load_line(vt, zt, junction)
    zr = np.linspace(0, 1, 11)
    zi = np.linspace(-1, 1, 21)
    surface = error_surface(vj, junction, zr, zi)
    assert surface.shape == (11, 21)
    for i in (0, 5, 10):
        for j in (0, 10, 20):
            expected = error_function(vj, junction, complex(zr[i], zi[j]))
            assert surface[i, j] == pytest.approx(expected)


def test_surface_is_finite_everywhere(junction):
    """A candidate impedance can cancel a junction impedance exactly, which
    divides by zero; those points must be finite sentinels, not NaN."""
    vj = load_line(0.5, 0.3 - 0.4j, junction)
    zr = np.linspace(0, 1, 51)
    zi = np.linspace(-1, 1, 101)
    surface = error_surface(vj, junction, zr, zi)
    assert not np.isnan(surface).any()


# -- recovery -----------------------------------------------------------


@pytest.mark.parametrize(
    ("vt", "zt"),
    [(0.5, 0.3 - 0.4j), (0.25, 0.7 + 0.2j), (0.8, 0.1 - 0.9j), (0.4, 0.5 + 0.0j)],
)
def test_recovers_a_perfect_load_line(junction, vt, zt):
    vj = load_line(vt, zt, junction)
    result = recover_zemb(vj, junction)
    assert result.zt == pytest.approx(zt, abs=1e-4)
    assert result.vt == pytest.approx(vt, abs=1e-4)
    assert result.fit_good
    assert result.err < GOOD_ERROR


def test_refinement_beats_the_bare_grid(junction):
    """QMix returns the best grid point, so its answer is quantised to the
    grid spacing; refining lifts that limit."""
    vt, zt = 0.5, 0.3137 - 0.4271j  # deliberately off-grid
    vj = load_line(vt, zt, junction)
    coarse = recover_zemb(vj, junction, refine=False)
    fine = recover_zemb(vj, junction, refine=True)
    assert abs(fine.zt - zt) < abs(coarse.zt - zt)
    assert fine.err <= coarse.err


def test_forced_impedance_is_used_verbatim(junction):
    vj = load_line(0.5, 0.3 - 0.4j, junction)
    result = recover_zemb(vj, junction, zemb=0.6 + 0.1j)
    assert result.zt == 0.6 + 0.1j
    assert result.vt == pytest.approx(source_voltage(vj, junction, 0.6 + 0.1j))


def test_recovery_stays_inside_the_search_box(junction):
    """The refinement is bounded, so it cannot wander into unphysical
    territory such as negative resistance."""
    vj = load_line(0.5, 0.3 - 0.4j, junction)
    result = recover_zemb(vj, junction, remb_range=(0.4, 1.0), xemb_range=(0.0, 1.0))
    assert 0.4 <= result.zt.real <= 1.0
    assert 0.0 <= result.zt.imag <= 1.0


def test_noisy_points_still_give_a_close_answer(junction):
    rng = np.random.default_rng(0)
    vt, zt = 0.5, 0.3 - 0.4j
    vj = load_line(vt, zt, junction)
    vj = vj * (1 + rng.normal(0, 1e-3, vj.shape))
    result = recover_zemb(vj, junction)
    assert abs(result.zt - zt) < 0.02
    assert result.vt == pytest.approx(vt, rel=0.02)


def test_result_carries_the_surface_it_searched(junction):
    vj = load_line(0.5, 0.3 - 0.4j, junction)
    result = recover_zemb(vj, junction, nreal=21, nimag=31)
    assert result.err_surf.shape == (21, 31)
    assert result.zt_real.size == 21
    assert result.zt_imag.size == 31


def test_rejects_mismatched_inputs():
    with pytest.raises(ValueError, match="same shape"):
        recover_zemb(np.zeros(3, dtype=complex), np.zeros(4, dtype=complex))
