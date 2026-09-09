"""Noise temperature versus IF frequency, from spectrum-analyzer data.

Hot and cold blackbody loads are measured with a spectrum analyzer across
the IF band.  The Y-factor at each frequency gives the noise temperature
there -- the "IF response" of the receiver.

Examples:

    >>> import numpy as np
    >>> from qpmix.exp.if_response import if_response
    >>> f = np.array([4.0, 5.0, 6.0])
    >>> hot = np.array([-60.0, -60.0, -60.0])
    >>> cold = np.array([-61.0, -61.0, -61.0])
    >>> data = if_response(np.column_stack((f, hot, cold)))
    >>> data.shape
    (4, 3)

"""

from __future__ import annotations

from typing import Any

import numpy as np

from qpmix.exp.parameters import merge_params

__all__ = ["db_to_linear", "if_response"]


def db_to_linear(db: np.ndarray) -> np.ndarray:
    """Convert a power in decibels to linear units.

    Args:
        db (ndarray): Power in dB.

    Returns:
        ndarray: Power in linear units.

    """
    return 10 ** (np.asarray(db, dtype=float) / 10.0)


def if_response(if_data: np.ndarray, **kwargs: Any) -> np.ndarray:
    """Noise temperature versus IF frequency from hot/cold spectra.

    Args:
        if_data (ndarray): Three columns: IF frequency in GHz, hot-load
            power in dBm, cold-load power in dBm.  A transposed array is
            accepted and corrected.

    Keyword Args:
        **kwargs: Any parameter from :data:`qpmix.exp.parameters.PARAMS`.
            Relevant here: ``t_hot``, ``t_cold`` and ``ifresp_maxtn``.

    Returns:
        ndarray: Four rows: frequency, noise temperature in kelvin, hot
        power in dBm, cold power in dBm.

    Raises:
        ValueError: If the input is not a three-column array.

    """
    kw = merge_params(kwargs)

    data = np.asarray(if_data, dtype=float)
    if data.ndim != 2:
        raise ValueError("IF response data must be 2-dimensional.")
    if data.shape[1] != 3:
        data = data.T
    if data.shape[1] != 3:
        raise ValueError("IF response data must have 3 columns.")

    freq, hot_db, cold_db = data.T

    with np.errstate(divide="ignore", invalid="ignore"):
        y = db_to_linear(hot_db) / db_to_linear(cold_db)
    # A Y-factor at or below 1 is unphysical; clamp it just above.
    y = np.where(np.isfinite(y), y, 1.0)
    y = np.maximum(y, 1.0 + 1e-6)

    tn = (kw["t_hot"] - kw["t_cold"] * y) / (y - 1)
    tn = np.where((tn < 0) | (tn > kw["ifresp_maxtn"]), kw["ifresp_maxtn"], tn)

    return np.vstack((freq, tn, hot_db, cold_db))
