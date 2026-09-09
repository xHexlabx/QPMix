"""Default parameters for importing and analyzing experimental data.

Every parameter below can be overridden by passing it as a keyword argument
to :class:`qpmix.exp.DCData` or :class:`qpmix.exp.PumpedData`, or to any of
the lower-level import functions.

Compared to upstream QMix, this module adds validation: unknown keyword
arguments are rejected rather than silently ignored, so a typo in a
parameter name is caught immediately instead of quietly leaving the default
in place.  :data:`PARAMS` remains a plain dictionary for compatibility.

Examples:

    >>> from qpmix.exp.parameters import PARAMS, merge_params
    >>> PARAMS['t_cold']
    78.0
    >>> merge_params({'t_cold': 20.0})['t_cold']
    20.0
    >>> merge_params({'t_kold': 20.0})
    Traceback (most recent call last):
        ...
    TypeError: Unknown parameter(s): ['t_kold']. Did you mean: t_cold?

"""

from __future__ import annotations

import difflib
from typing import Any

__all__ = ["PARAMS", "merge_params", "params"]

#: Default parameters.  See the module docstring for what each one does.
PARAMS: dict[str, Any] = {
    # -- units of the imported data ---------------------------------
    #: Units of imported voltage data: 'uV', 'mV' or 'V'.
    "v_fmt": "mV",
    #: Units of imported current data: 'uA', 'mA' or 'A'.
    "i_fmt": "mA",
    # -- importing I-V data -----------------------------------------
    #: Maximum bias voltage to import, in volts.
    "vmax": 6e-3,
    #: Number of points in the resampled I-V curve.
    "npts": 6001,
    #: Plot every step of the import for debugging.
    "debug": False,
    # -- correcting offsets -----------------------------------------
    #: Current offset in the raw data, in amps.  None to fit it.
    "ioffset": None,
    #: Voltage offset in the raw data, in volts.  None to fit it.
    "voffset": None,
    #: Voltage range to search for the offset, in volts.
    "voffset_range": (-1e-3, 1e-3),
    # -- correcting the measurement system --------------------------
    #: Series resistance of the DC bias system, in ohms.  None if absent.
    "rseries": None,
    #: Scale the imported current by this factor.
    "i_multiplier": 1.0,
    #: Scale the imported voltage by this factor.
    "v_multiplier": 1.0,
    # -- filtering I-V data -----------------------------------------
    #: Filter the I-V data?
    "filter_data": True,
    #: Angle to rotate the I-V curve by before filtering, in radians.
    "filter_theta": 0.785,
    #: Savitzky-Golay window length.  Must be odd.
    "filter_nwind": 21,
    #: Savitzky-Golay polynomial order.
    "filter_npoly": 3,
    # -- analyzing the DC I-V curve ---------------------------------
    #: Voltage range for the normal-resistance fit, in volts.
    "vrn": (3.5e-3, 4.5e-3),
    #: Voltage at which to measure the subgap resistance, in volts.
    "vrsg": 2e-3,
    #: Voltage at which to measure the leakage current, in volts.
    "vleak": 2e-3,
    #: Gap voltage in volts.  None to find it from the data.
    "vgap": None,
    # -- analyzing pumped I-V data (impedance recovery) -------------
    #: Recover the embedding circuit from the pumped I-V curve?
    "analyze_iv": True,
    #: Fit interval, as a fraction of the first photon step.
    "fit_range": (0.25, 0.8),
    #: Embedding resistances to search, normalized to Rn.
    "remb_range": (0, 1),
    #: Embedding reactances to search, normalized to Rn.
    "xemb_range": (-1, 1),
    #: Force the embedding impedance to this normalized value.
    "zemb": None,
    #: Refine the recovered impedance off the search grid?
    "zemb_refine": True,
    #: Initial guess for the drive level during impedance recovery.
    "alpha_max": 1.5,
    #: Bessel-function summation limit for the Tucker-theory sums.
    "num_b": 20,
    # -- importing IF data ------------------------------------------
    #: Number of points in the resampled IF data.
    "ifdata_npts": 3000,
    #: Gaussian smoothing width for the IF data, in volts.
    "ifdata_sigma": 1e-5,
    # -- analyzing DC IF data ---------------------------------------
    #: Voltage range(s) for the shot-noise fit, in volts.  None to find it.
    "vshot": None,
    # -- analyzing pumped IF data (noise temperature) ---------------
    #: Analyze the IF data?
    "analyze_if": True,
    #: Cold load temperature, in kelvin.
    "t_cold": 78.0,
    #: Hot load temperature, in kelvin.
    "t_hot": 293.0,
    #: Bias voltage for the quoted best result, in volts.  None to choose.
    "vbest": None,
    #: How to choose the best bias: 'Max Gain' or 'Min Tn'.
    "best_pt": "Max Gain",
    # -- IF response ------------------------------------------------
    #: Cap on the reported noise temperature, in kelvin.
    "ifresp_maxtn": 1e6,
    # -- response function ------------------------------------------
    #: Voltage smear of the smeared response function.
    "v_smear": 0.020,
    # -- junction and signal ----------------------------------------
    #: Junction area, in square microns.
    "area": 1.5,
    #: Local-oscillator frequency, in GHz.
    "freq": None,
    # -- plotting ---------------------------------------------------
    #: Maximum bias voltage in plots, in mV.
    "vmax_plot": 4.0,
    # -- miscellaneous ----------------------------------------------
    #: A label for this data set.
    "comment": "",
    #: Print progress to the terminal.
    "verbose": True,
}

#: Backwards-compatible alias for :data:`PARAMS`.
params = PARAMS


def merge_params(kwargs: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
    """Merge user options over :data:`PARAMS`, rejecting unknown names.

    Args:
        kwargs (dict, optional): User-supplied options.  Default is None.
        **extra: Further options, merged after ``kwargs``.

    Returns:
        dict: A fresh dictionary holding every parameter.

    Raises:
        TypeError: If any option is not a known parameter.  The message
            suggests the closest match.

    """
    merged = dict(PARAMS)
    supplied = dict(kwargs or {})
    supplied.update(extra)

    unknown = sorted(set(supplied) - set(PARAMS))
    if unknown:
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name, PARAMS, n=1)
            if close:
                hints.append(close[0])
        suffix = f" Did you mean: {', '.join(hints)}?" if hints else ""
        raise TypeError(f"Unknown parameter(s): {unknown}.{suffix}")

    merged.update(supplied)
    return merged
