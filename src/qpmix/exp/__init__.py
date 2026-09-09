"""Importing and analyzing measured SIS mixer data.

This sub-package turns raw laboratory measurements into physics:

===================================  =========================================
:class:`DCData`                      An unpumped DC I-V curve, and everything
                                     measured from it.
:class:`PumpedData`                  A pumped I-V curve plus hot/cold IF data:
                                     embedding circuit, noise temperature,
                                     gain.
:mod:`qpmix.exp.tucker`              Tucker theory, and drive-level recovery.
:mod:`qpmix.exp.zemb`                Fitting the Thevenin equivalent source by
                                     voltage matching (closed form, fast).
:mod:`qpmix.exp.currentmatch`        Fitting it by current matching (full
                                     harmonic balance, any bias window, and
                                     higher harmonics).
:mod:`qpmix.exp.simulate`            Synthetic measurements with known
                                     answers, for testing.
:data:`qpmix.exp.parameters.PARAMS`  Every tunable parameter and its default.
===================================  =========================================

All data is passed in as NumPy arrays; see the individual modules for the
expected column layout.

Examples:

    >>> from qpmix.exp import DCData
    >>> from qpmix.exp.simulate import simulate_dciv
    >>> dciv = DCData(simulate_dciv(), verbose=False)
    >>> round(dciv.rn, 1)
    14.0

"""

from qpmix.exp.currentmatch import (
    CurrentMatchResult,
    recover_zemb_current_match,
    voltage_windows,
)
from qpmix.exp.exp_data import DCData, PumpedData
from qpmix.exp.if_data import DCIFData, dcif_data, if_data
from qpmix.exp.if_response import if_response
from qpmix.exp.iv_data import (
    DCIVData,
    OffsetReport,
    check_offset,
    dciv_curve,
    iv_curve,
)
from qpmix.exp.parameters import PARAMS, merge_params
from qpmix.exp.tucker import ac_current, pumped_iv_curve, recover_alpha
from qpmix.exp.zemb import ZembResult, recover_zemb

__all__ = [
    "PARAMS",
    "CurrentMatchResult",
    "DCData",
    "DCIFData",
    "DCIVData",
    "OffsetReport",
    "PumpedData",
    "ZembResult",
    "ac_current",
    "check_offset",
    "dcif_data",
    "dciv_curve",
    "if_data",
    "if_response",
    "iv_curve",
    "merge_params",
    "pumped_iv_curve",
    "recover_alpha",
    "recover_zemb",
    "recover_zemb_current_match",
    "voltage_windows",
]
