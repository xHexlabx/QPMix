"""QPMix: fast multi-tone spectral domain analysis of SIS quasiparticle mixers.

QPMix simulates the quasiparticle tunneling currents in
Superconductor/Insulator/Superconductor (SIS) junctions, which are used as
heterodyne mixers at millimetre and submillimetre wavelengths in radio
astronomy.  Because it works in the spectral domain with multiple tones, it
handles power saturation, higher-order harmonics, sub-harmonic pumping,
harmonic mixing and frequency multiplication.

QPMix is a modernised, re-engineered derivative of **QMix** by John Garrett
and Ghassan Yassin (https://github.com/garrettj403/QMix), released under the
same GPL-3.0 licence.  The physics and the equation numbering follow QMix
and P. Kittara's 2002 DPhil thesis; the numerical methods, the packaging and
the test suite are new.  See ``CITATION.cff`` and the README for the papers
to cite.

Quick start:

    >>> import numpy as np
    >>> import qpmix
    >>> cct = qpmix.EmbeddingCircuit(1, 1, vb_npts=101, vgap=2.8e-3, rn=14.0)
    >>> cct.set_freq(230, units='GHz')
    >>> cct.vt[1, 1] = 0.5
    >>> cct.zt[1, 1] = 0.3 - 0.3j
    >>> resp = qpmix.RespFnPolynomial(50, verbose=False)
    >>> vj = qpmix.harmonic_balance(cct, resp, num_b=9, verbose=False)
    >>> idc = qpmix.qtcurrent(vj, cct, resp, 0.0, num_b=9, verbose=False)
    >>> idc.shape
    (101,)

"""

from qpmix._backend import get_config, set_num_threads
from qpmix.circuit import EmbeddingCircuit, read_circuit
from qpmix.harmonic_balance import check_hb_error, harmonic_balance
from qpmix.interp import UniformInterpolator
from qpmix.phase_factor import calculate_phase_factor_coeff
from qpmix.qtcurrent import interpolate_respfn, qtcurrent
from qpmix.respfn import (
    RespFn,
    RespFnExponential,
    RespFnFromIVData,
    RespFnPerfect,
    RespFnPolynomial,
)

__version__ = "0.1.0"

__all__ = [
    "EmbeddingCircuit",
    "RespFn",
    "RespFnExponential",
    "RespFnFromIVData",
    "RespFnPerfect",
    "RespFnPolynomial",
    "UniformInterpolator",
    "__version__",
    "calculate_phase_factor_coeff",
    "check_hb_error",
    "get_config",
    "harmonic_balance",
    "interpolate_respfn",
    "qtcurrent",
    "read_circuit",
    "set_num_threads",
]
