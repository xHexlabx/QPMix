"""Package-level smoke tests: the public API and a full end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest

import qpmix


def test_version_is_exposed():
    assert isinstance(qpmix.__version__, str)
    assert qpmix.__version__.count(".") >= 1


def test_everything_in_all_is_importable():
    for name in qpmix.__all__:
        assert hasattr(qpmix, name), name


def test_no_import_time_matplotlib(monkeypatch):
    """Importing qpmix must not drag in matplotlib."""
    import subprocess
    import sys

    code = "import qpmix, sys; print('matplotlib' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_end_to_end_230ghz_mixer():
    """A complete single-tone SIS mixer simulation at 230 GHz."""
    cct = qpmix.EmbeddingCircuit(1, 1, vb_npts=201, vb_max=1.5, vgap=2.8e-3, rn=14.0)
    cct.set_freq(230, units="GHz")
    cct.zt[1, 1] = 0.3 - 0.3j
    cct.set_available_power(50, units="nW")

    resp = qpmix.RespFnPolynomial(50, verbose=False)
    vj = qpmix.harmonic_balance(cct, resp, num_b=11, verbose=False)
    idc = qpmix.qtcurrent(vj, cct, resp, 0.0, num_b=11, verbose=False)

    assert idc.shape == (201,)
    assert np.all(np.isfinite(idc))
    # The pumped curve must sit above the unpumped one below the gap...
    unpumped = resp.idc(cct.vb)
    below = cct.vb < 0.95
    assert np.all(idc[below] >= unpumped[below] - 1e-9)
    # ...and carry current on the first photon step.
    step = (cct.vb > 1 - cct.freq[1]) & (cct.vb < 0.99)
    assert idc[step].max() > 0.05
    qpmix.check_hb_error(vj, cct, resp, num_b=11, stop_rerror=1e-2)


def test_end_to_end_two_tone_mixer():
    """LO plus a weak RF signal: the IF current must appear at f_LO - f_RF."""
    cct = qpmix.EmbeddingCircuit(2, 1, vb_npts=101, vb_max=1.5, vgap=2.8e-3, rn=14.0)
    cct.set_freq(230, f=1, units="GHz")
    cct.set_freq(232, f=2, units="GHz")
    cct.zt[1, 1] = 0.3 - 0.3j
    cct.zt[2, 1] = 0.3 - 0.3j
    cct.set_available_power(50, f=1, units="nW")
    cct.set_available_power(0.05, f=2, units="nW")

    resp = qpmix.RespFnPolynomial(50, verbose=False)
    vj = qpmix.harmonic_balance(cct, resp, num_b=11, verbose=False)

    f_if = round(float(cct.freq[2] - cct.freq[1]), 4)
    i_if = qpmix.qtcurrent(vj, cct, resp, f_if, num_b=11, verbose=False)
    assert i_if.shape == (101,)
    # Down-conversion happens on the photon step, not deep in the subgap.
    on_step = (cct.vb > 1 - cct.freq[1]) & (cct.vb < 0.99)
    deep = cct.vb < 0.3
    assert np.abs(i_if[on_step]).max() > 10 * np.abs(i_if[deep]).max()


def test_backend_config_is_reported():
    cfg = qpmix.get_config()
    assert isinstance(cfg["jit_enabled"], bool)


@pytest.mark.parametrize("resp_cls", [qpmix.RespFnPerfect, qpmix.RespFnPolynomial])
def test_simulation_runs_with_every_response_class(resp_cls):
    cct = qpmix.EmbeddingCircuit(1, 1, vb_npts=41, vb_max=1.5)
    cct.freq[1] = 0.3
    cct.vt[1, 1] = 0.4
    cct.zt[1, 1] = 0.3
    resp = resp_cls(verbose=False)
    vj = qpmix.harmonic_balance(cct, resp, num_b=9, verbose=False)
    idc = qpmix.qtcurrent(vj, cct, resp, 0.0, num_b=9, verbose=False)
    assert np.all(np.isfinite(idc))
