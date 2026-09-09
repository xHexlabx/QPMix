"""Shared fixtures for the QPMix test suite."""

from __future__ import annotations

import numpy as np
import pytest

import qpmix


def pytest_configure(config):
    """Register the optional cross-validation marker."""
    config.addinivalue_line(
        "markers", "reference: cross-validation against the upstream QMix package"
    )


@pytest.fixture(scope="session")
def resp_poly():
    """A polynomial response function, shared across the whole session.

    Building a response function is the one genuinely expensive setup step,
    so it is built once.  Nothing in the test suite mutates it.
    """
    return qpmix.RespFnPolynomial(50, verbose=False)


@pytest.fixture(scope="session")
def resp_perfect():
    """The ideal step response function."""
    return qpmix.RespFnPerfect(verbose=False)


@pytest.fixture
def make_circuit():
    """Factory for embedding circuits with sensible defaults."""

    def _make(num_f=1, num_p=1, npts=51, vb_max=2.0, freqs=(0.30, 0.32, 0.35, 0.38)):
        cct = qpmix.EmbeddingCircuit(
            num_f, num_p, vb_npts=npts, vb_max=vb_max, vgap=2.8e-3, rn=14.0
        )
        for f in range(1, num_f + 1):
            cct.freq[f] = freqs[f - 1]
        return cct

    return _make


@pytest.fixture
def random_vj():
    """Factory for a plausible random junction-voltage array."""

    def _make(num_f, num_p, npts, seed=0, vmax=0.4):
        rng = np.random.default_rng(seed)
        vj = np.zeros((num_f + 1, num_p + 1, npts), dtype=complex)
        for f in range(1, num_f + 1):
            for p in range(1, num_p + 1):
                mag = rng.uniform(0.05, vmax, npts) / p
                phase = rng.uniform(-np.pi, np.pi, npts)
                vj[f, p] = mag * np.exp(1j * phase)
        return vj

    return _make
