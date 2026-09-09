"""Tests for qpmix.qtcurrent.

The important checks here are against Tucker theory (Tucker and Feldman,
*Rev. Mod. Phys.* **57**, 1055, 1985), which gives closed-form expressions
for the single-tone tunneling currents.  Those are an independent analytic
ground truth, not a comparison against another implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jv

import qpmix
from qpmix._backend import JIT_ENABLED
from qpmix.qtcurrent import (
    _fft_shape,
    _matching_tuples,
    _needed_tuples,
    _select_method,
    qtcurrent,
)

# -- Tucker theory reference ------------------------------------------


def tucker_dc_current(voltage, resp, alpha, vph, num_b=25):
    """Pumped DC I-V curve, Tucker and Feldman Eqn. 3.3."""
    i_dc = np.zeros(voltage.size, dtype=float)
    for n in range(-num_b, num_b + 1):
        i_dc += jv(n, alpha) ** 2 * resp.idc(voltage + n * vph)
    return i_dc


def tucker_ac_current(voltage, resp, alpha, vph, num_b=25):
    """AC tunneling current at the drive frequency, Tucker and Feldman."""
    i_ac = np.zeros(voltage.size, dtype=complex)
    for n in range(-num_b, num_b + 1):
        jn = jv(n, alpha)
        i_ac += jn * (jv(n - 1, alpha) + jv(n + 1, alpha)) * resp.idc(voltage + n * vph)
        i_ac += (
            1j
            * jn
            * (jv(n - 1, alpha) - jv(n + 1, alpha))
            * resp.ikk(voltage + n * vph)
        )
    return i_ac


# -- physics ----------------------------------------------------------


def test_unpumped_junction_reproduces_the_dc_iv_curve(resp_poly, make_circuit):
    cct = make_circuit(npts=101)
    vj = cct.initialize_vj()
    idc = qtcurrent(vj, cct, resp_poly, 0.0, num_b=9, verbose=False)
    assert np.abs(idc - resp_poly.idc(cct.vb)).max() < 1e-12


def test_unpumped_junction_carries_no_ac_current(resp_poly, make_circuit):
    cct = make_circuit(npts=51)
    vj = cct.initialize_vj()
    iac = qtcurrent(vj, cct, resp_poly, cct.freq[1], num_b=9, verbose=False)
    assert np.abs(iac).max() < 1e-12


@pytest.mark.parametrize("alpha", [0.3, 1.0, 2.5])
def test_dc_current_matches_tucker_theory(resp_poly, make_circuit, alpha):
    cct = make_circuit(npts=101)
    vph = cct.freq[1]
    vj = cct.initialize_vj()
    vj[1, 1, :] = alpha * vph

    idc = qtcurrent(vj, cct, resp_poly, 0.0, num_b=25, verbose=False)
    expected = tucker_dc_current(cct.vb, resp_poly, alpha, vph)
    assert np.abs(idc - expected).max() < 1e-11


@pytest.mark.parametrize("alpha", [0.3, 1.0, 2.5])
def test_ac_current_matches_tucker_theory(resp_poly, make_circuit, alpha):
    cct = make_circuit(npts=101)
    vph = cct.freq[1]
    vj = cct.initialize_vj()
    vj[1, 1, :] = alpha * vph

    iac = qtcurrent(vj, cct, resp_poly, vph, num_b=25, verbose=False)
    expected = tucker_ac_current(cct.vb, resp_poly, alpha, vph)
    assert np.abs(iac - expected).max() < 1e-11


def test_pumped_curve_shows_photon_steps(resp_perfect, make_circuit):
    """A pumped junction turns on in steps of one photon voltage.

    The definitive signature is that d(Idc)/dV has a sharp peak at each
    ``V = 1 - n * vph`` and is flat in between.
    """
    cct = make_circuit(npts=3001, vb_max=1.5)
    vph = 0.25
    cct.freq[1] = vph
    vj = cct.initialize_vj()
    vj[1, 1, :] = 1.5 * vph
    idc = qtcurrent(vj, cct, resp_perfect, 0.0, num_b=20, verbose=False)
    didv = np.gradient(idc, cct.vb)

    def peak_near(v0, halfwidth=0.02):
        window = np.abs(cct.vb - v0) < halfwidth
        return didv[window].max()

    for n in (0, 1, 2):
        edge = peak_near(1 - n * vph)
        middle = peak_near(1 - (n + 0.5) * vph)
        assert edge > 5 * middle, f"no photon step at n={n}"

    # Four photon steps below the gap the current is negligible.
    deep = cct.vb < 1 - 3.2 * vph
    assert np.abs(idc[deep]).max() < 1e-3 * idc[-1]


def test_dc_current_is_real(resp_poly, make_circuit, random_vj):
    cct = make_circuit(num_f=2, npts=41)
    vj = random_vj(2, 1, 41)
    idc = qtcurrent(vj, cct, resp_poly, 0.0, num_b=11, verbose=False)
    assert idc.dtype == np.float64


def test_adding_an_unexcited_tone_changes_nothing(resp_poly, make_circuit):
    """Tones with zero drive must not perturb the answer at all."""
    freqs = (0.30, 0.32, 0.35, 0.38)
    results = []
    for num_f in (1, 2, 3, 4):
        cct = make_circuit(num_f=num_f, npts=31, freqs=freqs)
        vj = cct.initialize_vj()
        vj[1, 1, :] = 0.4
        results.append(
            qtcurrent(vj, cct, resp_poly, [0.0, 0.30], num_b=9, verbose=False)
        )
    for other in results[1:]:
        assert np.abs(results[0] - other).max() < 1e-12


def test_adding_an_unexcited_harmonic_changes_nothing(resp_poly, make_circuit):
    results = []
    for num_p in (1, 2, 3):
        cct = make_circuit(num_p=num_p, npts=31)
        vj = cct.initialize_vj()
        vj[1, 1, :] = 0.4
        results.append(
            qtcurrent(vj, cct, resp_poly, [0.0, 0.30], num_b=9, verbose=False)
        )
    for other in results[1:]:
        assert np.abs(results[0] - other).max() < 1e-12


def test_which_tone_is_excited_does_not_matter(resp_poly, make_circuit):
    """Driving tone 1 or tone 2 at the same frequency must give the same
    DC current."""
    out = []
    for tone in (1, 2):
        cct = make_circuit(num_f=2, npts=31, freqs=(0.31, 0.31))
        vj = cct.initialize_vj()
        vj[tone, 1, :] = 0.4
        out.append(qtcurrent(vj, cct, resp_poly, 0.0, num_b=11, verbose=False))
    assert np.abs(out[0] - out[1]).max() < 1e-12


# -- API and dispatch --------------------------------------------------


def test_scalar_frequency_returns_1d(resp_poly, make_circuit):
    cct = make_circuit(npts=17)
    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.3
    assert qtcurrent(vj, cct, resp_poly, 0.0, num_b=7, verbose=False).shape == (17,)
    assert qtcurrent(vj, cct, resp_poly, [0.0], num_b=7, verbose=False).shape == (1, 17)


def test_list_of_frequencies_returns_2d(resp_poly, make_circuit):
    cct = make_circuit(npts=17)
    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.3
    out = qtcurrent(vj, cct, resp_poly, [0.0, 0.30, 0.60], num_b=7, verbose=False)
    assert out.shape == (3, 17)
    assert out.dtype == np.complex128


def test_unreachable_frequency_gives_zero(resp_poly, make_circuit):
    cct = make_circuit(npts=17)
    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.3
    out = qtcurrent(vj, cct, resp_poly, 0.123456, num_b=7, verbose=False)
    assert np.abs(out).max() == 0.0


@pytest.mark.parametrize(("num_f", "num_p"), [(1, 1), (1, 3), (2, 1), (2, 2)])
def test_direct_and_fft_methods_agree(resp_poly, make_circuit, random_vj, num_f, num_p):
    cct = make_circuit(num_f=num_f, num_p=num_p, npts=33)
    vj = random_vj(num_f, num_p, 33, seed=num_f * 10 + num_p)
    fl = [0.0] + [
        round(cct.freq[f] * p, 4)
        for f in range(1, num_f + 1)
        for p in range(1, num_p + 1)
    ]
    direct = qtcurrent(vj, cct, resp_poly, fl, num_b=9, verbose=False, method="direct")
    fft = qtcurrent(vj, cct, resp_poly, fl, num_b=9, verbose=False, method="fft")
    assert np.abs(direct - fft).max() < 1e-12


def test_auto_matches_both_explicit_methods(resp_poly, make_circuit, random_vj):
    cct = make_circuit(num_f=2, npts=25)
    vj = random_vj(2, 1, 25, seed=9)
    fl = [0.0, 0.30, 0.32, 0.02]
    auto = qtcurrent(vj, cct, resp_poly, fl, num_b=9, verbose=False)
    direct = qtcurrent(vj, cct, resp_poly, fl, num_b=9, verbose=False, method="direct")
    assert np.abs(auto - direct).max() < 1e-12


def test_precomputed_resp_matrix_gives_the_same_answer(resp_poly, make_circuit):
    cct = make_circuit(num_f=2, npts=25)
    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.3
    vj[2, 1, :] = 0.1
    rm = qpmix.interpolate_respfn(cct, resp_poly, 9)
    a = qtcurrent(vj, cct, resp_poly, [0.0, 0.30], num_b=9, verbose=False)
    b = qtcurrent(
        vj, cct, resp_poly, [0.0, 0.30], num_b=9, verbose=False, resp_matrix=rm
    )
    assert np.array_equal(a, b)


def test_rejects_a_mismatched_resp_matrix(resp_poly, make_circuit):
    cct = make_circuit(num_f=2, npts=25)
    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.3
    with pytest.raises(ValueError, match="wrong shape"):
        qtcurrent(
            vj,
            cct,
            resp_poly,
            0.0,
            num_b=9,
            verbose=False,
            resp_matrix=np.zeros((19, 19, 24), dtype=complex),
        )


def test_rejects_zero_frequency_tone(resp_poly, make_circuit):
    cct = make_circuit(npts=11)
    cct.freq[1] = 0.0
    with pytest.raises(ValueError, match="must be > 0"):
        qtcurrent(cct.initialize_vj(), cct, resp_poly, 0.0, verbose=False)


def test_rejects_unknown_method(resp_poly, make_circuit):
    cct = make_circuit(npts=11)
    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.2
    with pytest.raises(ValueError, match="Unknown method"):
        qtcurrent(vj, cct, resp_poly, 0.0, verbose=False, method="magic")


def test_verbose_reports_the_chosen_method(resp_poly, make_circuit, capsys):
    cct = make_circuit(npts=11)
    vj = cct.initialize_vj()
    vj[1, 1, :] = 0.2
    qtcurrent(vj, cct, resp_poly, 0.0, num_b=7, verbose=True)
    out = capsys.readouterr().out
    assert "1 tone(s)" in out
    assert "method:" in out


# -- index bookkeeping -------------------------------------------------


def test_matching_tuples_finds_the_right_combinations():
    freq = np.array([0.0, 0.30, 0.32])
    out = _matching_tuples(np.array([0.0, 0.30, 0.02, 0.62]), freq, 2, 1)
    assert set(out[0]) == {(0, 0)}
    assert set(out[1]) == {(1, 0)}
    assert set(out[2]) == {(-1, 1)}
    assert set(out[3]) == {(1, 1)}


def test_needed_tuples_deduplicates():
    assert _needed_tuples([[(1, 0)], [(1, 0), (0, 1)]]) == [(1, 0), (0, 1)]


def test_fft_shape_leaves_room_for_the_offsets():
    for nb, num_p in [(15, 1), (9, 3), (20, 2)]:
        (n,) = _fft_shape((nb,), num_p)
        assert n >= 2 * (nb + num_p) + 1


def test_select_method_honours_an_explicit_choice():
    tuples = [[(0, 0)]]
    assert _select_method("direct", 2, 1, (15, 15), 401, tuples) == "direct"
    assert _select_method("fft", 2, 1, (15, 15), 401, tuples) == "fft"


@pytest.mark.skipif(
    not JIT_ENABLED, reason="without a JIT the vectorised FFT path always wins"
)
def test_select_method_prefers_direct_for_a_single_tuple():
    assert _select_method("auto", 2, 1, (15, 15), 401, [[(0, 0)]]) == "direct"


@pytest.mark.skipif(JIT_ENABLED, reason="only applies to the NumPy fallback")
def test_select_method_prefers_fft_without_a_jit():
    """The direct kernels are interpreted loops without numba; the FFT path
    stays vectorised, so it must be chosen even for a single tuple."""
    assert _select_method("auto", 2, 1, (15, 15), 401, [[(0, 0)]]) == "fft"


def test_select_method_prefers_fft_when_many_tuples_are_wanted():
    """Asking for every intermodulation product at once flips the choice."""
    import itertools

    freq = np.array([0.0, 0.30, 0.32])
    num_p = 4
    span = range(-num_p, num_p + 1)
    freq_out = np.array(
        sorted(
            {
                round(float(np.dot(t, freq[1:3])), 4)
                for t in itertools.product(span, repeat=2)
            }
        )
    )
    tuples = _matching_tuples(freq_out, freq, 2, num_p)
    assert len(_needed_tuples(tuples)) > 40
    assert _select_method("auto", 2, num_p, (11, 11), 401, tuples) == "fft"


def test_select_method_falls_back_when_memory_would_blow_up():
    tuples = [[(0, 0, 0, 0)] * 40]
    assert _select_method("auto", 4, 3, (15,) * 4, 401, tuples) == "direct"


def test_select_method_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown method"):
        _select_method("magic", 1, 1, (5,), 11, [[(0,)]])
