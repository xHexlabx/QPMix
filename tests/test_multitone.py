"""Tests for qpmix.multitone.

The grid engine is validated three ways: against the multi-dimensional
engine (which it must reproduce exactly when both enumerate the same index
tuples), against Tucker theory for a single tone, and against the
photon-number sum rule ``Idc = sum_K |C_K|^2 Idc0(V0 + K*df)``, which holds
for any number of tones.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

import qpmix
from qpmix._backend import JIT_ENABLED
from qpmix.multitone import (
    MAX_DENOMINATOR,
    ToneGrid,
    interpolate_respfn_grid,
    phase_factor_grid,
    qtcurrent_grid,
)


def _circuit(num_f, num_p=1, npts=41, spacing=0.02, first=0.30):
    cct = qpmix.EmbeddingCircuit(num_f, num_p, vb_npts=npts, vb_max=2)
    for f in range(1, num_f + 1):
        cct.freq[f] = round(first + spacing * (f - 1), 6)
    return cct


def _drive(cct, seed=0, vmax=0.35):
    rng = np.random.default_rng(seed)
    vj = cct.initialize_vj()
    for f in range(1, cct.num_f + 1):
        for p in range(1, cct.num_p + 1):
            mag = rng.uniform(0.05, vmax, cct.vb_npts) / p
            vj[f, p] = mag * np.exp(1j * rng.uniform(-np.pi, np.pi, cct.vb_npts))
    return vj


# -- ToneGrid ----------------------------------------------------------


def test_grid_recovers_exact_multiples():
    grid = ToneGrid.from_frequencies([0.30, 0.32, 0.34], num_b=9)
    assert grid.multipliers == (15, 16, 17)
    assert grid.df == pytest.approx(0.02)
    assert grid.frequency_error < 1e-12
    assert grid.num_k == 9 * (15 + 16 + 17)


def test_grid_frequencies_round_trip():
    freqs = [0.25, 0.5, 0.75]
    grid = ToneGrid.from_frequencies(freqs, num_b=5)
    assert np.allclose(grid.frequencies, freqs)


def test_grid_uses_the_coarsest_spacing():
    """Common factors must be divided out, or num_k is needlessly large."""
    grid = ToneGrid.from_frequencies([0.20, 0.40], num_b=5)
    assert grid.multipliers == (1, 2)
    assert grid.df == pytest.approx(0.20)


def test_num_k_matches_the_multi_dimensional_reach():
    """The largest offset the multi-D method can reach is sum(n_f * num_b_f)."""
    grid = ToneGrid.from_frequencies([0.30, 0.32], num_b=(9, 4))
    assert grid.num_k == 15 * 9 + 16 * 4


def test_explicit_df_is_honoured():
    grid = ToneGrid.from_frequencies([0.30, 0.32], num_b=5, df=0.01)
    assert grid.df == pytest.approx(0.01)
    assert grid.multipliers == (30, 32)


def test_explicit_df_rejects_a_spacing_that_is_too_coarse():
    with pytest.raises(ValueError, match="too coarse"):
        ToneGrid.from_frequencies([0.30], num_b=5, df=1.0)


def test_offset_maps_frequency_to_index():
    grid = ToneGrid.from_frequencies([0.30, 0.32], num_b=5)
    assert grid.offset(0.30) == 15
    assert grid.offset(0.0) == 0
    assert grid.offset(-0.02) == -1


def test_offset_tolerance_is_strict_on_an_exact_grid():
    grid = ToneGrid.from_frequencies([0.30, 0.32], num_b=5)
    assert grid.tolerance == grid.df * 1e-6
    with pytest.raises(ValueError, match="not on the grid"):
        grid.offset(0.30 + 1e-5)
    assert grid.offset(0.30 + 1e-5, tol=1e-4) == grid.multipliers[0]


def test_offset_tolerates_the_grids_own_approximation_error():
    """A budgeted grid represents the third tone at ``120 * df``, about
    1e-3 away from the requested 0.3417.  Asking for the current at the
    requested frequency -- or at an IF formed from it -- must still find
    that grid point rather than rejecting the grid's own tone."""
    grid = ToneGrid.from_frequencies([0.30, 0.32, 0.3417], num_b=5, max_num_k=2000)
    assert grid.frequency_error > grid.df * 1e-6
    assert grid.offset(0.3417) == grid.multipliers[2]
    assert grid.offset(0.3417 - 0.30) == grid.multipliers[2] - grid.multipliers[0]


def test_offset_rejects_an_off_grid_frequency():
    grid = ToneGrid.from_frequencies([0.30, 0.32], num_b=5)
    with pytest.raises(ValueError, match="not on the grid"):
        grid.offset(0.013)


def test_grid_rejects_bad_frequencies():
    with pytest.raises(ValueError, match="at least one"):
        ToneGrid.from_frequencies([], num_b=5)
    with pytest.raises(ValueError, match="must be > 0"):
        ToneGrid.from_frequencies([0.3, 0.0], num_b=5)


def test_closely_spaced_tones_are_refused_rather_than_allocated():
    """An LO and RF 5 MHz apart at 230 GHz need a 1.4e6-point grid."""
    cct = qpmix.EmbeddingCircuit(2, 1, vb_npts=401)
    cct.freq[1], cct.freq[2] = 0.3396, 0.33961
    with pytest.raises(MemoryError, match="closely spaced"):
        ToneGrid.from_circuit(cct, num_b=15)


def test_memory_guard_can_be_disabled():
    cct = qpmix.EmbeddingCircuit(2, 1, vb_npts=401)
    cct.freq[1], cct.freq[2] = 0.3396, 0.33961
    grid = ToneGrid.from_circuit(cct, num_b=15, max_entries=0)
    assert grid.num_k > 10**5


def test_a_larger_denominator_cap_never_fits_worse():
    freqs = [0.3013, 0.3197]
    errors = [
        ToneGrid.from_frequencies(freqs, num_b=5, max_denominator=d).frequency_error
        for d in (20, 50, 200, MAX_DENOMINATOR)
    ]
    assert errors == sorted(errors, reverse=True)


def test_num_k_is_not_monotonic_in_the_denominator_cap():
    """Documented gotcha: a tighter cap can force a *finer* grid, which is
    why ``max_num_k`` exists as the cost knob."""
    freqs = [0.3013, 0.3197]
    tight = ToneGrid.from_frequencies(freqs, num_b=5, max_denominator=200)
    loose = ToneGrid.from_frequencies(freqs, num_b=5, max_denominator=MAX_DENOMINATOR)
    assert tight.num_k > loose.num_k


def test_max_num_k_trades_accuracy_for_cost():
    freqs = [0.3013, 0.3197]
    grids = [
        ToneGrid.from_frequencies(freqs, num_b=5, max_num_k=cap)
        for cap in (200, 1000, 5000)
    ]
    assert [g.num_k for g in grids] == sorted(g.num_k for g in grids)
    errors = [g.frequency_error for g in grids]
    assert errors == sorted(errors, reverse=True)
    for g, cap in zip(grids, (200, 1000, 5000), strict=True):
        assert g.num_k <= cap


def test_max_num_k_finds_the_exact_grid_when_the_budget_allows():
    grid = ToneGrid.from_frequencies([0.30, 0.32], num_b=9, max_num_k=10_000)
    assert grid.frequency_error == 0.0
    assert grid.multipliers == (15, 16)


def test_max_num_k_reports_an_impossible_budget():
    with pytest.raises(ValueError, match="No common grid"):
        ToneGrid.from_frequencies([0.30, 0.32], num_b=9, max_num_k=1)


def test_report_mentions_both_costs():
    grid = ToneGrid.from_frequencies([0.30, 0.32, 0.34], num_b=9)
    text = grid.report(401)
    assert "grid entries" in text
    assert "direct entries" in text
    assert "multipliers" in text


# -- phase factor ------------------------------------------------------


def test_phase_factor_is_unitary_for_many_tones():
    """sum_K |C_K|^2 = 1: the phase factor has unit modulus."""
    cct = _circuit(6, npts=17)
    grid = ToneGrid.from_circuit(cct, num_b=7)
    ck = phase_factor_grid(_drive(cct, seed=1, vmax=0.2), cct, grid)
    power = (np.abs(ck) ** 2).sum(axis=0)
    assert np.abs(power - 1.0).max() < 1e-10


def test_phase_factor_reduces_to_bessel_for_one_tone():
    cct = _circuit(1, npts=5)
    grid = ToneGrid.from_frequencies([cct.freq[1]], num_b=12, df=cct.freq[1])
    vj = cct.initialize_vj()
    vj[1, 1, :] = 1.4 * cct.freq[1]
    ck = phase_factor_grid(vj, cct, grid)
    for k in range(-12, 13):
        assert ck[k, 0] == pytest.approx(jv(k, 1.4), abs=1e-11)


def test_phase_factor_of_an_undriven_junction_is_a_delta():
    cct = _circuit(5, npts=7)
    grid = ToneGrid.from_circuit(cct, num_b=4)
    ck = phase_factor_grid(cct.initialize_vj(), cct, grid)
    assert np.allclose(ck[0], 1.0)
    assert np.abs(ck[1:]).max() < 1e-12


def test_correlation_jit_and_numpy_paths_agree(monkeypatch, resp_poly):
    """The blocked kernel and the vectorised fallback must not diverge."""
    import qpmix.multitone as mt

    cct = _circuit(6, npts=23)
    vj = _drive(cct, seed=12, vmax=0.2)
    fl = [0.0] + [round(float(cct.freq[f]), 4) for f in range(1, 7)]
    jitted = qtcurrent_grid(vj, cct, resp_poly, fl, num_b=5, verbose=False)
    monkeypatch.setattr(mt, "JIT_ENABLED", False)
    fallback = qtcurrent_grid(vj, cct, resp_poly, fl, num_b=5, verbose=False)
    assert np.abs(jitted - fallback).max() < 1e-12


def test_phase_factor_jit_and_numpy_paths_agree(monkeypatch):
    import qpmix.multitone as mt

    cct = _circuit(3, npts=11)
    grid = ToneGrid.from_circuit(cct, num_b=5)
    vj = _drive(cct, seed=2)
    jitted = phase_factor_grid(vj, cct, grid)
    monkeypatch.setattr(mt, "JIT_ENABLED", False)
    fallback = phase_factor_grid(vj, cct, grid)
    assert np.abs(jitted - fallback).max() < 1e-12


# -- physics -----------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.3, 1.0, 2.5])
def test_grid_engine_matches_tucker_theory(resp_poly, alpha):
    cct = _circuit(1, npts=101)
    vph = cct.freq[1]
    vj = cct.initialize_vj()
    vj[1, 1, :] = alpha * vph

    idc = qpmix.qtcurrent(
        vj, cct, resp_poly, 0.0, num_b=25, verbose=False, method="grid"
    )
    expected = sum(
        jv(n, alpha) ** 2 * resp_poly.idc(cct.vb + n * vph) for n in range(-30, 31)
    )
    assert np.abs(idc - expected).max() < 1e-11


@pytest.mark.parametrize("num_f", [1, 2, 3, 5, 8])
def test_dc_current_obeys_the_photon_number_sum_rule(resp_poly, num_f):
    """Idc = sum_K |C_K|^2 Idc0(V0 + K*df), for any number of tones."""
    cct = _circuit(num_f, npts=21)
    grid = ToneGrid.from_circuit(cct, num_b=7)
    vj = _drive(cct, seed=num_f, vmax=0.25)

    idc = qtcurrent_grid(vj, cct, resp_poly, 0.0, num_b=7, verbose=False, grid=grid)
    ck = phase_factor_grid(vj, cct, grid)
    expected = sum(
        np.abs(ck[k]) ** 2 * resp_poly.idc(cct.vb + k * grid.df)
        for k in range(-grid.num_k, grid.num_k + 1)
    )
    assert np.abs(idc - expected).max() < 1e-12


@pytest.mark.parametrize("num_f", [1, 3, 6, 10])
def test_undriven_junction_reproduces_the_dc_iv_curve(resp_poly, num_f):
    cct = _circuit(num_f, npts=31)
    idc = qpmix.qtcurrent(
        cct.initialize_vj(),
        cct,
        resp_poly,
        0.0,
        num_b=5,
        verbose=False,
        method="grid",
    )
    assert np.abs(idc - resp_poly.idc(cct.vb)).max() < 1e-12


def test_adding_an_unexcited_tone_changes_nothing(resp_poly):
    results = []
    for num_f in (2, 4, 6, 8):
        cct = _circuit(num_f, npts=21)
        vj = cct.initialize_vj()
        vj[1, 1, :] = 0.3
        results.append(
            qpmix.qtcurrent(
                vj,
                cct,
                resp_poly,
                [0.0, 0.30],
                num_b=7,
                verbose=False,
                method="grid",
            )
        )
    for other in results[1:]:
        assert np.abs(results[0] - other).max() < 1e-12


# -- equivalence with the multi-dimensional engine ----------------------


def _extra_tuples_exist(grid, freq, num_b):
    """True if some index tuple beyond the obvious one reaches ``freq``."""
    import itertools

    a = grid.offset(freq)
    hits = sum(
        1
        for t in itertools.product(range(-num_b, num_b + 1), repeat=grid.num_f)
        if sum(k * n for k, n in zip(t, grid.multipliers, strict=True)) == a
    )
    return hits > 1


@pytest.mark.parametrize(
    ("num_f", "num_p", "spacing"),
    [(1, 1, 0.02), (1, 3, 0.02), (2, 1, 0.02), (2, 1, 0.03)],
)
def test_grid_matches_the_direct_engine(resp_poly, num_f, num_p, spacing):
    """Where no extra intermodulation tuples exist, the two agree exactly."""
    cct = _circuit(num_f, num_p, npts=31, spacing=spacing)
    grid_obj = ToneGrid.from_circuit(cct, num_b=9)
    assert not _extra_tuples_exist(grid_obj, float(cct.freq[1]), 9), (
        "test case is meant to have a unique index tuple"
    )
    vj = _drive(cct, seed=num_f * 7 + num_p)
    fl = [0.0] + [
        round(float(cct.freq[f] * p), 4)
        for f in range(1, num_f + 1)
        for p in range(1, num_p + 1)
    ]
    direct = qpmix.qtcurrent(
        vj, cct, resp_poly, fl, num_b=9, verbose=False, method="direct"
    )
    grid = qpmix.qtcurrent(
        vj, cct, resp_poly, fl, num_b=9, verbose=False, method="grid"
    )
    assert np.abs(direct - grid).max() < 1e-8


@pytest.mark.skipif(
    not JIT_ENABLED,
    reason="forces the multi-dimensional kernels at 3-4 tones, which are "
    "interpreted loops without numba (this is why 'auto' avoids them)",
)
@pytest.mark.parametrize("num_f", [3, 4])
def test_grid_includes_products_the_direct_engine_truncates(resp_poly, num_f):
    """With commensurate tones, index tuples beyond +/-num_p reach the same
    output frequency.  The grid engine sums them all; the multi-dimensional
    engine only reaches them once num_p is raised, and then the two agree.
    """
    num_b = 9
    cct = _circuit(num_f, 1, npts=31)
    vj = _drive(cct, seed=num_f)
    fl = [0.0] + [round(float(cct.freq[f]), 4) for f in range(1, num_f + 1)]

    grid = qpmix.qtcurrent(
        vj, cct, resp_poly, fl, num_b=num_b, verbose=False, method="grid"
    )
    narrow = qpmix.qtcurrent(
        vj, cct, resp_poly, fl, num_b=num_b, verbose=False, method="direct"
    )

    wide_cct = _circuit(num_f, num_b, npts=31)
    wide_vj = wide_cct.initialize_vj()
    wide_vj[1:, 1] = vj[1:, 1]
    wide = qpmix.qtcurrent(
        wide_vj,
        wide_cct,
        resp_poly,
        fl,
        num_b=num_b,
        verbose=False,
        method="direct",
    )

    assert np.abs(wide - grid).max() < 1e-8
    assert np.abs(narrow - grid).max() > 1e-3


# -- API ---------------------------------------------------------------


def test_qtcurrent_auto_switches_above_four_tones(resp_poly):
    cct = _circuit(5, npts=21)
    vj = _drive(cct, seed=3, vmax=0.2)
    auto = qpmix.qtcurrent(vj, cct, resp_poly, 0.0, num_b=5, verbose=False)
    explicit = qpmix.qtcurrent(
        vj, cct, resp_poly, 0.0, num_b=5, verbose=False, method="grid"
    )
    assert np.array_equal(auto, explicit)


def test_direct_method_refuses_more_than_four_tones(resp_poly):
    cct = _circuit(5, npts=11)
    vj = _drive(cct, seed=4, vmax=0.2)
    with pytest.raises(ValueError, match="at most 4 tones"):
        qpmix.qtcurrent(
            vj, cct, resp_poly, 0.0, num_b=5, verbose=False, method="direct"
        )


def test_grid_kwargs_are_rejected_by_the_direct_engine(resp_poly):
    cct = _circuit(2, npts=11)
    vj = _drive(cct, seed=5)
    with pytest.raises(TypeError, match="Unexpected keyword"):
        qpmix.qtcurrent(
            vj,
            cct,
            resp_poly,
            0.0,
            num_b=5,
            verbose=False,
            method="direct",
            num_theta=256,
        )


def test_scalar_and_list_frequency_conventions_match_the_direct_engine(resp_poly):
    cct = _circuit(5, npts=17)
    vj = _drive(cct, seed=6, vmax=0.2)
    scalar = qpmix.qtcurrent(vj, cct, resp_poly, 0.0, num_b=5, verbose=False)
    listed = qpmix.qtcurrent(vj, cct, resp_poly, [0.0], num_b=5, verbose=False)
    assert scalar.shape == (17,)
    assert scalar.dtype == np.float64
    assert listed.shape == (1, 17)
    assert np.allclose(listed[0].real, scalar)


def test_precomputed_grid_resp_matrix_is_reused(resp_poly):
    cct = _circuit(5, npts=17)
    grid = ToneGrid.from_circuit(cct, num_b=5)
    vj = _drive(cct, seed=7, vmax=0.2)
    rm = interpolate_respfn_grid(cct, resp_poly, grid)
    a = qtcurrent_grid(vj, cct, resp_poly, 0.0, num_b=5, verbose=False, grid=grid)
    b = qtcurrent_grid(
        vj, cct, resp_poly, 0.0, num_b=5, verbose=False, grid=grid, resp_matrix=rm
    )
    assert np.array_equal(a, b)


def test_interpolate_respfn_dispatches_on_tone_count(resp_poly):
    cct = _circuit(5, npts=13)
    rm = qpmix.interpolate_respfn(cct, resp_poly, 5)
    grid = ToneGrid.from_circuit(cct, num_b=5)
    assert rm.shape == (2 * grid.num_k + 1, 13)


def test_num_theta_override_does_not_change_the_answer(resp_poly):
    cct = _circuit(3, npts=17)
    vj = _drive(cct, seed=8)
    grid = ToneGrid.from_circuit(cct, num_b=7)
    base = qtcurrent_grid(vj, cct, resp_poly, 0.0, verbose=False, grid=grid)
    big = qtcurrent_grid(
        vj, cct, resp_poly, 0.0, verbose=False, grid=grid, num_theta=1 << 14
    )
    assert np.abs(base - big).max() < 1e-11


def test_verbose_reports_the_grid(resp_poly, capsys):
    cct = _circuit(5, npts=11)
    vj = _drive(cct, seed=9, vmax=0.2)
    qpmix.qtcurrent(vj, cct, resp_poly, 0.0, num_b=5, verbose=True)
    out = capsys.readouterr().out
    assert "common grid" in out
    assert "num_k" in out


def test_circuit_accepts_many_tones():
    cct = qpmix.EmbeddingCircuit(12, 2, vb_npts=11)
    assert cct.num_f == 12
    assert cct.num_n == 24
    assert cct.vt.shape == (13, 3)


def test_circuit_still_rejects_nonsense_tone_counts():
    with pytest.raises(ValueError, match="int >= 1"):
        qpmix.EmbeddingCircuit(0)
    with pytest.raises(ValueError, match="int >= 1"):
        qpmix.EmbeddingCircuit(2.5)


# -- harmonic balance --------------------------------------------------


def test_harmonic_balance_grid_matches_direct(resp_poly):
    """Two tones give the same tuple set either way, so the two engines must
    agree exactly."""
    cct = _circuit(2, npts=31)
    for f in (1, 2):
        cct.vt[f, 1] = 0.3
        cct.zt[f, 1] = 0.3 - 0.3j
    kwargs = dict(num_b=7, verbose=False, stop_rerror=1e-6)
    direct = qpmix.harmonic_balance(cct, resp_poly, method="direct", **kwargs)
    grid = qpmix.harmonic_balance(cct, resp_poly, method="grid", **kwargs)
    assert np.abs(direct - grid).max() < 1e-9


def test_harmonic_balance_grid_is_what_makes_many_tones_practical(resp_poly):
    """Harmonic balance is dominated by current evaluations, so it has to be
    able to reach the grid engine too -- otherwise the multi-dimensional
    response matrix is rebuilt on every iteration however fast qtcurrent is.
    """
    cct = _circuit(4, npts=21)
    for f in range(1, 5):
        cct.vt[f, 1] = 0.15
        cct.zt[f, 1] = 0.3 - 0.3j
    vj = qpmix.harmonic_balance(
        cct, resp_poly, num_b=5, verbose=False, method="grid", stop_rerror=1e-4
    )
    assert vj.shape == (5, 2, 21)
    assert np.all(np.isfinite(vj))


def test_grid_resp_matrix_must_match_the_grid(resp_poly):
    cct = _circuit(3, npts=11)
    vj = _drive(cct, seed=3)
    grid = ToneGrid.from_circuit(cct, num_b=5)
    other = interpolate_respfn_grid(cct, resp_poly, ToneGrid.from_circuit(cct, num_b=6))
    with pytest.raises(ValueError, match="Pass the grid it was built from"):
        qtcurrent_grid(
            vj,
            cct,
            resp_poly,
            0.0,
            num_b=5,
            verbose=False,
            grid=grid,
            resp_matrix=other,
        )


def test_an_explicit_grid_opts_harmonic_balance_into_the_grid_engine(resp_poly):
    """Below five tones ``method="auto"`` stays multi-dimensional, but a
    grid handed in is an opt-in as clear as ``method="grid"``."""
    cct = _circuit(3, npts=11)
    for f in (1, 2, 3):
        cct.vt[f, 1] = 0.2
        cct.zt[f, 1] = 0.3 - 0.3j
    grid = ToneGrid.from_circuit(cct, num_b=5)
    kwargs = dict(num_b=5, verbose=False, stop_rerror=1e-6)
    with_grid = qpmix.harmonic_balance(cct, resp_poly, grid=grid, **kwargs)
    explicit = qpmix.harmonic_balance(cct, resp_poly, method="grid", **kwargs)
    direct = qpmix.harmonic_balance(cct, resp_poly, method="direct", **kwargs)
    assert np.abs(with_grid - explicit).max() < 1e-12
    assert np.abs(with_grid - direct).max() > 1e-9


def test_harmonic_balance_rejects_a_grid_with_the_direct_engine(resp_poly):
    cct = _circuit(2, npts=11)
    for f in (1, 2):
        cct.vt[f, 1] = 0.1
        cct.zt[f, 1] = 0.3
    grid = ToneGrid.from_circuit(cct, num_b=5)
    with pytest.raises(TypeError, match="does not use one"):
        qpmix.harmonic_balance(
            cct, resp_poly, num_b=5, verbose=False, method="direct", grid=grid
        )
    vj = qpmix.harmonic_balance(cct, resp_poly, num_b=5, verbose=False, grid=grid)
    with pytest.raises(TypeError, match="does not use one"):
        qpmix.check_hb_error(vj, cct, resp_poly, num_b=5, method="fft", grid=grid)


def test_harmonic_balance_takes_an_explicit_grid_for_measured_frequencies(resp_poly):
    """A measured gap frequency makes even a perfect comb incommensurate
    once normalized: the rational fit asks for multipliers near 1e14 and
    refuses.  The comb is still there in hertz, so a grid built from that
    spacing is small -- and harmonic balance has to (a) accept it, (b) use
    it for both the matrix and every current evaluation, and (c) hand the
    grid unrounded frequencies, or ``offset`` rejects every tone."""
    fgap, spacing = 677.037e9, 0.6e9
    cct = qpmix.EmbeddingCircuit(4, 1, vb_npts=21, vb_max=2, fgap=fgap)
    for f, f_hz in enumerate((225e9, 227.4e9, 228e9, 228.6e9), start=1):
        cct.set_freq(f_hz, f=f, units="Hz")
        cct.vt[f, 1] = 0.15
        cct.zt[f, 1] = 0.3 - 0.3j
    with pytest.raises(MemoryError, match="closely spaced"):
        qpmix.harmonic_balance(cct, resp_poly, num_b=4, verbose=False, method="grid")

    grid = ToneGrid.from_circuit(cct, num_b=4, df=spacing / fgap)
    assert grid.multipliers == (375, 379, 380, 381)
    assert grid.num_k == 4 * (375 + 379 + 380 + 381)
    rm = interpolate_respfn_grid(cct, resp_poly, grid)
    kwargs = dict(num_b=4, verbose=False, stop_rerror=1e-5, grid=grid)
    vj = qpmix.harmonic_balance(cct, resp_poly, **kwargs)
    assert vj.shape == (5, 2, 21)
    assert np.all(np.isfinite(vj))
    qpmix.check_hb_error(vj, cct, resp_poly, num_b=4, stop_rerror=1e-3, grid=grid)
    reused = qpmix.harmonic_balance(cct, resp_poly, resp_matrix=rm, **kwargs)
    assert np.abs(reused - vj).max() < 1e-12


def test_harmonic_balance_rejects_too_many_tones_for_the_direct_engine(resp_poly):
    cct = _circuit(5, npts=11)
    for f in range(1, 6):
        cct.vt[f, 1] = 0.1
        cct.zt[f, 1] = 0.3
    with pytest.raises(ValueError, match="at most 4 tones"):
        qpmix.harmonic_balance(cct, resp_poly, num_b=5, verbose=False, method="direct")


@pytest.mark.parametrize("num_f", [5, 8])
def test_harmonic_balance_solves_many_tone_circuits(resp_poly, num_f):
    cct = _circuit(num_f, npts=21)
    for f in range(1, num_f + 1):
        cct.vt[f, 1] = 0.25 / np.sqrt(num_f)
        cct.zt[f, 1] = 0.3 - 0.3j
    vj = qpmix.harmonic_balance(
        cct, resp_poly, num_b=5, verbose=False, stop_rerror=1e-4
    )
    assert vj.shape == (num_f + 1, 2, 21)
    qpmix.check_hb_error(vj, cct, resp_poly, num_b=5, stop_rerror=1e-2)


def test_grid_truncation_warns_when_num_b_is_too_small_for_the_drive(resp_poly):
    """On the grid the truncation is on the total offset num_k, so the test
    is the summed spectral spread of every tone against it."""
    import warnings

    from qpmix.phase_factor import DriveLevelWarning

    cct = _circuit(2, npts=5)
    vj = cct.initialize_vj()
    vj[1, 1] = 0.6  # alpha 2 on tone 1
    vj[2, 1] = 0.1
    with pytest.warns(DriveLevelWarning, match="num_k"):
        qtcurrent_grid(vj, cct, resp_poly, 0.0, num_b=1, verbose=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DriveLevelWarning)
        qtcurrent_grid(vj, cct, resp_poly, 0.0, num_b=12, verbose=False)
