"""Cross-validation against the upstream QMix package.

These tests are skipped unless ``qmix`` is installed.  They are the
regression net for the claim that QPMix computes the same physics as QMix:
every difference must be at the level of QMix's own interpolation error
(~1e-4 on the response function), not larger.

Install the reference with::

    pip install QMix
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import qpmix

qmix = pytest.importorskip("qmix", reason="upstream QMix not installed")

pytestmark = pytest.mark.reference


def _circuits(num_f, num_p, npts, freqs=(0.30, 0.32, 0.35, 0.38)):
    out = []
    for pkg in (qmix, qpmix):
        cct = pkg.circuit.EmbeddingCircuit(
            num_f, num_p, vb_npts=npts, vb_max=2, vgap=2.8e-3, rn=14.0
        )
        for f in range(1, num_f + 1):
            cct.freq[f] = freqs[f - 1]
        out.append(cct)
    return out


def _random_vj(num_f, num_p, npts, seed=0):
    rng = np.random.default_rng(seed)
    vj = np.zeros((num_f + 1, num_p + 1, npts), dtype=complex)
    for f in range(1, num_f + 1):
        for p in range(1, num_p + 1):
            mag = rng.uniform(0.05, 0.45, npts) / p
            vj[f, p] = mag * np.exp(1j * rng.uniform(-np.pi, np.pi, npts))
    return vj


@pytest.fixture(scope="module")
def resp_pair():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return (
            qmix.respfn.RespFnPolynomial(50, verbose=False),
            qpmix.RespFnPolynomial(50, verbose=False),
        )


def test_response_functions_agree(resp_pair):
    ref, new = resp_pair
    v = np.linspace(-6, 6, 20001)
    assert np.abs(ref.idc(v) - new.idc(v)).max() < 1e-4
    assert np.abs(ref.ikk(v) - new.ikk(v)).max() < 1e-3


def test_qpmix_is_closer_to_a_high_accuracy_reference(resp_pair):
    """The residual disagreement is QMix's interpolation error, not ours."""
    ref, new = resp_pair
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        truth = qmix.respfn.RespFnPolynomial(
            50, verbose=False, max_npts_dc=6001, max_npts_kk=6001
        )
    v = np.linspace(-4, 4, 20001)
    err_ref = np.abs(ref(v) - truth(v)).max()
    err_new = np.abs(new(v) - truth(v)).max()
    assert err_new < err_ref


def test_kk_transform_agrees():
    v = np.linspace(-35, 35, 7001)
    i = np.where(np.abs(v) < 1, 0.0, v)
    ref = qmix.mathfn.kktrans.kk_trans(v, i, 50)
    new = qpmix.mathfn.kktrans.kk_trans(v, i, 50)
    assert np.abs(ref - new).max() < 1e-9


@pytest.mark.parametrize("num_p", [1, 2])
def test_phase_factor_coefficients_agree(num_p):
    vj = _random_vj(2, num_p, 61, seed=num_p)
    freq = np.array([0.0, 0.30, 0.32])
    ref = qmix.qtcurrent.calculate_phase_factor_coeff(vj, freq, 2, num_p, 15)
    new = qpmix.calculate_phase_factor_coeff(vj, freq, 2, num_p, 15)
    assert np.abs(ref - new).max() < 1e-13


@pytest.mark.parametrize("num_f", [1, 2, 3, 4])
def test_response_matrix_agrees(resp_pair, num_f):
    ref_resp, new_resp = resp_pair
    c_ref, c_new = _circuits(num_f, 1, 21)
    nb = 4
    ref = qmix.qtcurrent.interpolate_respfn(c_ref, ref_resp, nb)
    new = qpmix.interpolate_respfn(c_new, new_resp, nb)
    assert ref.shape == new.shape
    assert np.abs(ref - new).max() < 1e-3


@pytest.mark.parametrize(("num_f", "num_p"), [(1, 1), (1, 2), (2, 1), (2, 2), (3, 1)])
def test_tunneling_currents_agree(resp_pair, num_f, num_p):
    ref_resp, new_resp = resp_pair
    npts = 41
    c_ref, c_new = _circuits(num_f, num_p, npts)
    vj = _random_vj(num_f, num_p, npts, seed=num_f * 10 + num_p)
    fl = [0.0] + [
        round(c_new.freq[f] * p, 4)
        for f in range(1, num_f + 1)
        for p in range(1, num_p + 1)
    ]
    ref = qmix.qtcurrent.qtcurrent(vj, c_ref, ref_resp, fl, num_b=11, verbose=False)
    new = qpmix.qtcurrent(vj, c_new, new_resp, fl, num_b=11, verbose=False)
    assert np.abs(ref - new).max() < 1e-3


@pytest.mark.parametrize(("num_f", "num_p"), [(1, 1), (2, 1), (1, 2)])
def test_harmonic_balance_agrees(resp_pair, num_f, num_p):
    ref_resp, new_resp = resp_pair
    c_ref, c_new = _circuits(num_f, num_p, 61)
    for cct in (c_ref, c_new):
        for f in range(1, num_f + 1):
            for p in range(1, num_p + 1):
                cct.vt[f, p] = 0.5 / p
                cct.zt[f, p] = 0.3 - 0.3j
    ref = qmix.harmonic_balance.harmonic_balance(
        c_ref, ref_resp, num_b=11, verbose=False
    )
    new = qpmix.harmonic_balance(c_new, new_resp, num_b=11, verbose=False)
    assert np.abs(ref - new).max() < 1e-3


def test_ivcurve_models_agree_to_machine_precision():
    v = np.linspace(-35, 35, 7001)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name in ("perfect", "perfect_kk", "polynomial", "exponential", "expanded"):
            ref = getattr(qmix.mathfn.ivcurve_models, name)(v)
            new = getattr(qpmix.mathfn.ivcurve_models, name)(v)
            assert np.nanmax(np.abs(ref - new)) < 1e-12, name
