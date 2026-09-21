"""Tests for qpmix.harmonic_balance."""

from __future__ import annotations

import numpy as np
import pytest

import qpmix
from qpmix.harmonic_balance import (
    _broyden_update,
    _hb_freq_list,
    _k_to_fp,
    _to_complex,
    _to_real,
    check_hb_error,
    harmonic_balance,
)
from qpmix.qtcurrent import qtcurrent


def _driven(cct, vt=0.5, zt=0.3 - 0.3j):
    for f in range(1, cct.num_f + 1):
        for p in range(1, cct.num_p + 1):
            cct.vt[f, p] = vt / p
            cct.zt[f, p] = zt
    return cct


# -- the defining property --------------------------------------------


@pytest.mark.parametrize(("num_f", "num_p"), [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_solution_satisfies_the_circuit_equation(resp_poly, make_circuit, num_f, num_p):
    """vj must equal vt - zt * ij(vj), which is the whole point."""
    cct = _driven(make_circuit(num_f=num_f, num_p=num_p, npts=41))
    vj = harmonic_balance(cct, resp_poly, num_b=11, verbose=False, stop_rerror=1e-5)

    freq_list = _hb_freq_list(cct)
    current = qtcurrent(vj, cct, resp_poly, freq_list, num_b=11, verbose=False)
    ij = current.reshape((num_f, num_p, cct.vb_npts))

    for f in range(1, num_f + 1):
        for p in range(1, num_p + 1):
            residual = cct.vt[f, p] - cct.zt[f, p] * ij[f - 1, p - 1] - vj[f, p]
            assert np.max(np.abs(residual) / np.abs(vj[f, p])) < 1e-4


def test_zero_embedding_impedance_is_a_no_op(resp_poly, make_circuit):
    """With zt = 0 the junction simply sees the Thevenin voltage."""
    cct = make_circuit(npts=21)
    cct.vt[1, 1] = 0.4
    vj = harmonic_balance(cct, resp_poly, num_b=9, verbose=False)
    assert np.allclose(vj[1, 1], 0.4)


def test_output_shape_and_zero_padding(resp_poly, make_circuit):
    cct = _driven(make_circuit(num_f=2, num_p=2, npts=21))
    vj = harmonic_balance(cct, resp_poly, num_b=9, verbose=False)
    assert vj.shape == (3, 3, 21)
    assert np.all(vj[0, :] == 0)
    assert np.all(vj[:, 0] == 0)


def test_larger_source_impedance_lowers_the_junction_voltage(resp_poly, make_circuit):
    low = _driven(make_circuit(npts=21), zt=0.1)
    high = _driven(make_circuit(npts=21), zt=2.0)
    vj_low = harmonic_balance(low, resp_poly, num_b=9, verbose=False)
    vj_high = harmonic_balance(high, resp_poly, num_b=9, verbose=False)
    assert np.abs(vj_high[1, 1]).max() < np.abs(vj_low[1, 1]).max()


# -- solver options ----------------------------------------------------


@pytest.mark.parametrize(("num_f", "num_p"), [(1, 1), (2, 1), (2, 2)])
def test_broyden_and_newton_reach_the_same_solution(
    resp_poly, make_circuit, num_f, num_p
):
    kwargs = dict(num_b=9, verbose=False, stop_rerror=1e-6)
    newton = harmonic_balance(
        _driven(make_circuit(num_f=num_f, num_p=num_p, npts=31)),
        resp_poly,
        jacobian="newton",
        **kwargs,
    )
    broyden = harmonic_balance(
        _driven(make_circuit(num_f=num_f, num_p=num_p, npts=31)),
        resp_poly,
        jacobian="broyden",
        **kwargs,
    )
    assert np.abs(newton - broyden).max() < 1e-4


def test_line_search_can_be_disabled(resp_poly, make_circuit):
    a = harmonic_balance(
        _driven(make_circuit(npts=21)),
        resp_poly,
        num_b=9,
        verbose=False,
        line_search=True,
        stop_rerror=1e-6,
    )
    b = harmonic_balance(
        _driven(make_circuit(npts=21)),
        resp_poly,
        num_b=9,
        verbose=False,
        line_search=False,
        stop_rerror=1e-6,
    )
    assert np.abs(a - b).max() < 1e-5


def test_damping_still_converges(resp_poly, make_circuit):
    cct = _driven(make_circuit(npts=21))
    vj = harmonic_balance(
        cct, resp_poly, num_b=9, verbose=False, damp_coeff=0.5, max_it=25
    )
    check_hb_error(vj, cct, resp_poly, num_b=9, stop_rerror=1e-2)


def test_initial_guess_is_honoured(resp_poly, make_circuit):
    cct = _driven(make_circuit(npts=21))
    good = harmonic_balance(cct, resp_poly, num_b=9, verbose=False, max_it=20)
    from_guess = harmonic_balance(
        cct, resp_poly, num_b=9, verbose=False, vj_initial=good, max_it=20
    )
    assert np.abs(good - from_guess).max() < 1e-4


def test_starting_from_the_answer_converges_immediately(resp_poly, make_circuit):
    cct = _driven(make_circuit(npts=21))
    answer = harmonic_balance(
        cct, resp_poly, num_b=9, verbose=False, stop_rerror=1e-8, max_it=30
    )
    _, iterations, converged = harmonic_balance(
        cct,
        resp_poly,
        num_b=9,
        verbose=False,
        vj_initial=answer,
        stop_rerror=1e-6,
        mode="x",
    )
    assert converged
    assert iterations == 0


@pytest.mark.parametrize("mode", ["o", "x", "m"])
def test_output_modes(resp_poly, make_circuit, mode):
    cct = _driven(make_circuit(npts=21))
    out = harmonic_balance(cct, resp_poly, num_b=9, verbose=False, mode=mode)
    if mode == "o":
        assert isinstance(out, np.ndarray)
    else:
        assert len(out) == 3 if mode == "x" else len(out) == 2


def test_mode_x_reports_convergence(resp_poly, make_circuit):
    cct = _driven(make_circuit(npts=21))
    _, _, converged = harmonic_balance(
        cct, resp_poly, num_b=9, verbose=False, mode="x", max_it=20
    )
    assert converged is True


def test_mode_m_reports_a_per_point_mask(resp_poly, make_circuit):
    cct = _driven(make_circuit(npts=21))
    _, mask = harmonic_balance(cct, resp_poly, num_b=9, verbose=False, mode="m")
    assert mask.shape == (21,)
    assert mask.dtype == np.bool_
    assert mask.all()


def test_failure_to_converge_is_reported(resp_poly, make_circuit, capsys):
    """Quiet runs report non-convergence through the return value, not by
    printing: a fitting loop calls this thousands of times and cannot have
    it write to stdout.  QMix prints unconditionally."""
    cct = _driven(make_circuit(npts=21))
    _, _, converged = harmonic_balance(
        cct,
        resp_poly,
        num_b=9,
        verbose=False,
        mode="x",
        max_it=0,
        stop_rerror=1e-12,
    )
    assert converged is False
    assert "DID NOT ACHIEVE" not in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore::qpmix.harmonic_balance.ConvergenceWarning")
def test_failure_to_converge_is_printed_when_verbose(resp_poly, make_circuit, capsys):
    cct = _driven(make_circuit(npts=21))
    harmonic_balance(cct, resp_poly, num_b=9, verbose=True, max_it=0, stop_rerror=1e-12)
    assert "DID NOT ACHIEVE" in capsys.readouterr().out


@pytest.mark.filterwarnings("ignore::qpmix.harmonic_balance.ConvergenceWarning")
def test_verbose_prints_progress(resp_poly, make_circuit, capsys):
    cct = _driven(make_circuit(npts=21))
    harmonic_balance(cct, resp_poly, num_b=9, verbose=True, max_it=3)
    out = capsys.readouterr().out
    assert "Running harmonic balance" in out
    assert "Error after" in out


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"jacobian": "secant"}, "Unknown jacobian"),
        ({"mode": "z"}, "Unknown mode"),
    ],
)
def test_option_validation(resp_poly, make_circuit, kwargs, match):
    cct = _driven(make_circuit(npts=11))
    with pytest.raises(ValueError, match=match):
        harmonic_balance(cct, resp_poly, verbose=False, **kwargs)


def test_check_hb_error_raises_on_a_bad_solution(resp_poly, make_circuit):
    cct = _driven(make_circuit(npts=11))
    bogus = cct.initialize_vj()
    bogus[1, 1, :] = 0.123
    with pytest.raises(AssertionError):
        check_hb_error(bogus, cct, resp_poly, num_b=9, stop_rerror=1e-6)


# -- helpers -----------------------------------------------------------


def test_real_complex_split_round_trips():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(3, 7)) + 1j * rng.normal(size=(3, 7))
    assert np.array_equal(_to_complex(_to_real(z)), z)


def test_real_split_interleaves_components():
    z = np.array([[1 + 2j], [3 + 4j]])
    assert _to_real(z).ravel().tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("k", "num_p", "expected"),
    [(0, 1, (1, 1)), (1, 1, (2, 1)), (0, 2, (1, 1)), (1, 2, (1, 2)), (2, 2, (2, 1))],
)
def test_k_to_fp(k, num_p, expected):
    assert _k_to_fp(k, num_p) == expected


def test_hb_freq_list_covers_every_signal(make_circuit):
    cct = make_circuit(num_f=2, num_p=2)
    assert _hb_freq_list(cct) == [0.3, 0.6, 0.32, 0.64]


def test_hb_freq_list_is_not_rounded(make_circuit):
    """The grid engine needs the exact products: ``ToneGrid.offset`` has to
    recognise ``p * freq[f]`` as a grid point, and a four-decimal rounding
    is off by up to 5e-5 -- far outside the tolerance of a fine grid."""
    cct = make_circuit(num_f=1, num_p=2, freqs=(225e9 / 677.037e9,))
    assert _hb_freq_list(cct) == [cct.freq[1], 2 * cct.freq[1]]


def test_broyden_update_satisfies_the_secant_condition():
    """After the update, J^-1 df must equal dx."""
    rng = np.random.default_rng(1)
    n, npts = 2, 5
    inv_j = np.zeros((2 * n, 2 * n, npts))
    for i in range(npts):
        inv_j[:, :, i] = np.linalg.inv(
            rng.normal(size=(2 * n, 2 * n)) + 4 * np.eye(2 * n)
        )
    dvj = rng.normal(size=(n, npts)) + 1j * rng.normal(size=(n, npts))
    derr = rng.normal(size=(n, npts)) + 1j * rng.normal(size=(n, npts))

    updated = _broyden_update(inv_j, dvj, derr)
    got = np.einsum("pqi,qi->pi", updated, _to_real(derr))
    assert np.abs(got - _to_real(dvj)).max() < 1e-9


def test_broyden_update_is_a_no_op_for_a_null_step():
    inv_j = np.tile(np.eye(2)[:, :, None], (1, 1, 3))
    zeros = np.zeros((1, 3), dtype=complex)
    assert np.array_equal(_broyden_update(inv_j, zeros, zeros), inv_j)


def test_top_level_exports_are_wired_up():
    assert qpmix.harmonic_balance is harmonic_balance
    assert qpmix.check_hb_error is check_hb_error


def test_non_convergence_issues_a_warning_even_when_quiet(resp_poly, make_circuit):
    """Silence was the failure mode: verbose=False in a fitting loop returned
    a half-converged vj with no trace unless the caller asked for mode="x".
    Now it warns regardless, and a converged run stays quiet."""
    import warnings

    from qpmix.harmonic_balance import ConvergenceWarning

    cct = _driven(make_circuit(npts=11))
    with pytest.warns(ConvergenceWarning, match="did not converge"):
        harmonic_balance(
            cct, resp_poly, num_b=9, verbose=False, max_it=0, stop_rerror=1e-12
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        harmonic_balance(cct, resp_poly, num_b=9, verbose=False)
        # Asking for the flag is the programmatic route, so no warning then.
        _, _, converged = harmonic_balance(
            cct, resp_poly, num_b=9, verbose=False, mode="x", max_it=0
        )
        assert converged is False
