"""Tests for qpmix._respmat (the fused response-matrix builder)."""

from __future__ import annotations

import numpy as np
import pytest

from qpmix._respmat import _fft_offsets, build_resp_matrix
from qpmix.qtcurrent import interpolate_respfn


def test_fft_offsets_ordering():
    assert _fft_offsets(3).tolist() == [0, 1, 2, 3, -3, -2, -1]


@pytest.mark.parametrize("num_f", [1, 2, 3, 4])
def test_matrix_shape(resp_poly, num_f):
    vb = np.linspace(0, 2, 17)
    freq = np.array([0.0, 0.30, 0.32, 0.35, 0.38])
    nb = (3,) * num_f
    out = build_resp_matrix(resp_poly, vb, freq, nb)
    assert out.shape == (7,) * num_f + (17,)
    assert out.dtype == np.complex128


@pytest.mark.parametrize("num_f", [1, 2, 3, 4])
def test_every_entry_is_the_shifted_response(resp_poly, num_f):
    """R[k, l, ...] must equal resp(vb + k*f1 + l*f2 + ...) exactly."""
    rng = np.random.default_rng(0)
    vb = np.linspace(0, 2, 13)
    freq = np.array([0.0, 0.30, 0.32, 0.35, 0.38])
    nb = (3,) * num_f
    out = build_resp_matrix(resp_poly, vb, freq, nb)

    offsets = [_fft_offsets(n) for n in nb]
    for _ in range(12):
        idx = tuple(int(rng.integers(0, 7)) for _ in range(num_f))
        shift = sum(offsets[a][i] * freq[a + 1] for a, i in enumerate(idx))
        assert np.allclose(out[idx], resp_poly(vb + shift), atol=1e-14)


def test_zero_offset_entry_is_the_unshifted_response(resp_poly):
    vb = np.linspace(0, 2, 21)
    freq = np.array([0.0, 0.30, 0.32])
    out = build_resp_matrix(resp_poly, vb, freq, (5, 5))
    assert np.allclose(out[0, 0], resp_poly(vb))


def test_jit_and_numpy_paths_agree(resp_poly, monkeypatch):
    """The fused kernel and the NumPy fallback must be interchangeable."""
    import qpmix._respmat as rm

    vb = np.linspace(0, 2, 19)
    freq = np.array([0.0, 0.30, 0.32])
    jitted = build_resp_matrix(resp_poly, vb, freq, (4, 4))
    monkeypatch.setattr(rm, "JIT_ENABLED", False)
    fallback = build_resp_matrix(resp_poly, vb, freq, (4, 4))
    assert np.abs(jitted - fallback).max() < 1e-12


def test_asymmetric_num_b_per_tone(resp_poly):
    vb = np.linspace(0, 2, 11)
    freq = np.array([0.0, 0.30, 0.32])
    out = build_resp_matrix(resp_poly, vb, freq, (2, 5))
    assert out.shape == (5, 11, 11)
    assert np.allclose(out[0, 0], resp_poly(vb))


def test_interpolate_respfn_matches_the_builder(resp_poly, make_circuit):
    cct = make_circuit(num_f=2, npts=15)
    direct = build_resp_matrix(resp_poly, cct.vb, cct.freq, (4, 4))
    assert np.array_equal(interpolate_respfn(cct, resp_poly, 4), direct)
    assert np.array_equal(interpolate_respfn(cct, resp_poly, (4, 4)), direct)


def test_rejects_too_many_tones(resp_poly):
    with pytest.raises(ValueError, match="1, 2, 3 or 4"):
        build_resp_matrix(resp_poly, np.linspace(0, 1, 5), np.zeros(6), (2,) * 5)
