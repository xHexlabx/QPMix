"""Importing and analyzing measured current-voltage (I-V) data.

"I-V data" is the DC tunneling current versus DC bias voltage measured from
an SIS junction.  *DC I-V data* is measured with no local-oscillator
injection; *pumped I-V data* is measured with it.

Input is a two-column NumPy array: bias voltage in the first column,
tunneling current in the second.  Units are set by the ``v_fmt`` and
``i_fmt`` parameters.

The import pipeline is: read and clean, apply gain corrections, correct the
voltage/current offset, filter, remove any series resistance, then measure
the gap voltage, normal resistance and subgap resistance.
:func:`dciv_curve` returns the normalized curve plus a :class:`DCIVData`
record holding everything it measured.

Differences from upstream QMix
------------------------------

* Argument validation raises :class:`ValueError` rather than using
  ``assert``, which vanishes under ``python -O``.
* The default normal-resistance fit range is ``(3.5e-3, 4.5e-3)`` volts, as
  QMix's own documentation states.  The QMix code says ``(3.5e-3, 5e3)`` --
  five *kilovolts* -- so in practice it fits from 3.5 mV to the end of the
  data instead of over the intended 1 mV window.
* :mod:`matplotlib` is imported only when ``debug=True`` actually needs it,
  so importing this module stays cheap.

Examples:

    >>> import numpy as np
    >>> from qpmix.exp.iv_data import dciv_curve
    >>> from qpmix.exp.simulate import simulate_dciv
    >>> raw = simulate_dciv(vgap=2.8e-3, rn=14.0, rsg=800.0)
    >>> v, i, dc = dciv_curve(raw, verbose=False)
    >>> bool(abs(dc.vgap - 2.8e-3) < 5e-5)
    True

"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import scipy.constants as sc
from scipy.optimize import minimize
from scipy.signal import savgol_filter

from qpmix.exp.clean_data import remove_doubles_xy, remove_nans_xy, sort_xy
from qpmix.exp.parameters import merge_params
from qpmix.mathfn.filters import gauss_conv
from qpmix.mathfn.misc import slope

__all__ = ["DCIVData", "dciv_curve", "iv_curve"]

#: Multipliers converting the named voltage unit into volts.
VOLT_UNITS = {"uV": 1e-6, "mV": 1e-3, "V": 1.0}

#: Multipliers converting the named current unit into amps.
CURR_UNITS = {"uA": 1e-6, "mA": 1e-3, "A": 1.0}


class DCIVData(NamedTuple):
    """Everything measured from a DC I-V curve.

    Attributes:
        vraw (ndarray): Bias voltage in volts, filtered and offset-corrected
            but before any series-resistance correction.
        iraw (ndarray): Tunneling current in amps, likewise.
        vnorm (ndarray): Bias voltage, normalized to the gap voltage.
        inorm (ndarray): Tunneling current, normalized to the gap current.
        vgap (float): Gap voltage, in volts.
        igap (float): Gap current ``vgap / rn``, in amps.
        fgap (float): Gap frequency ``e * vgap / h``, in Hz.
        rn (float): Normal-state resistance, in ohms.
        rsg (float): Subgap resistance, in ohms.
        offset (tuple): The ``(voltage, current)`` offset that was removed,
            in volts and amps.
        vint (float): Where a line fitted to the normal-state branch crosses
            zero current, in volts.
        rseries (float or None): The series resistance that was removed, in
            ohms.

    """

    vraw: np.ndarray
    iraw: np.ndarray
    vnorm: np.ndarray
    inorm: np.ndarray
    vgap: float
    igap: float
    fgap: float
    rn: float
    rsg: float
    offset: tuple[float, float]
    vint: float
    rseries: float | None


def _debug_plot(volt_v, curr_a, title):  # pragma: no cover - interactive
    """Plot one step of the import pipeline.

    Args:
        volt_v (ndarray): Bias voltage in volts.
        curr_a (ndarray): Tunneling current in amps.
        title (str): Plot title.

    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(volt_v * 1e3, curr_a * 1e6)
    plt.title(title)
    plt.xlabel("Voltage (mV)")
    plt.ylabel("Current (uA)")
    plt.grid()
    plt.show()


def dciv_curve(ivdata: np.ndarray, **kwargs: Any):
    """Import and analyze an unpumped DC I-V curve.

    Args:
        ivdata (ndarray): Two-column array of voltage and current.

    Keyword Args:
        **kwargs: Any parameter from :data:`qpmix.exp.parameters.PARAMS`;
            see that module for the full list.  The ones that matter here
            are ``v_fmt``, ``i_fmt``, ``vmax``, ``npts``, ``vgap``,
            ``voffset``, ``ioffset``, ``voffset_range``, ``rseries``,
            ``i_multiplier``, ``v_multiplier``, ``filter_data``,
            ``filter_theta``, ``filter_nwind``, ``filter_npoly``, ``vrn``,
            ``vrsg`` and ``debug``.

    Returns:
        tuple: ``(normalized voltage, normalized current, DCIVData)``.

    Raises:
        ValueError: If the input is not a two-column array, or a parameter
            is unknown.

    """
    kw = merge_params(kwargs)
    volt_v, curr_a = _load_iv(ivdata, kw)

    if kw["debug"]:  # pragma: no cover - interactive
        _debug_plot(volt_v, curr_a, "Initial import (raw data)")

    volt_v = volt_v * kw["v_multiplier"]
    curr_a = curr_a * kw["i_multiplier"]

    volt_v, curr_a, offset = _correct_offset(volt_v, curr_a, kw)
    if kw["debug"]:  # pragma: no cover - interactive
        _debug_plot(volt_v, curr_a, "After correcting the I/V offset")

    volt_v, curr_a = _filter_iv_data(volt_v, curr_a, kw)
    if kw["debug"]:  # pragma: no cover - interactive
        _debug_plot(volt_v, curr_a, "After filtering")

    vraw, iraw = volt_v.copy(), curr_a.copy()

    volt_v, curr_a = _correct_series_resistance(volt_v, curr_a, kw)
    if kw["debug"]:  # pragma: no cover - interactive
        _debug_plot(volt_v, curr_a, "After correcting the series resistance")

    vgap = kw["vgap"]
    if vgap is None:
        vgap = _find_gap_voltage(volt_v, curr_a, kw)
    rn, vint = _find_normal_resistance(volt_v, curr_a, kw)
    rsg = _find_subgap_resistance(volt_v, curr_a, kw)
    fgap = sc.e * vgap / sc.h
    igap = vgap / rn

    if kw["verbose"] and not 1.0 <= rn <= 50.0:
        print(
            f"Warning: normal resistance is {rn:.2f} ohms, which is outside "
            f"the usual 1-50 ohm range. Check the current and voltage units."
        )

    # Normalize and resample onto a uniform grid.
    v_out = np.linspace(-kw["vmax"], kw["vmax"], kw["npts"]) / vgap
    i_out = np.interp(v_out, volt_v / vgap, curr_a / igap)

    dc = DCIVData(
        vraw=vraw,
        iraw=iraw,
        vnorm=v_out,
        inorm=i_out,
        vgap=float(vgap),
        igap=float(igap),
        fgap=float(fgap),
        rn=float(rn),
        rsg=float(rsg),
        offset=offset,
        vint=float(vint),
        rseries=kw["rseries"],
    )
    return v_out, i_out, dc


def iv_curve(ivdata: np.ndarray, dc: DCIVData, **kwargs: Any):
    """Import a pumped I-V curve, using metadata from the unpumped curve.

    The offset, gap voltage and normalization all come from ``dc``, so the
    pumped curve lands on the same axes as the DC curve.

    Args:
        ivdata (ndarray): Two-column array of voltage and current.
        dc (DCIVData): Metadata from :func:`dciv_curve`.

    Keyword Args:
        **kwargs: See :func:`dciv_curve`.

    Returns:
        tuple: ``(normalized voltage, normalized current)``.

    Raises:
        ValueError: If the input is not a two-column array.

    """
    kw = merge_params(kwargs)
    volt_v, curr_a = _load_iv(ivdata, kw)

    volt_v = volt_v * kw["v_multiplier"]
    curr_a = curr_a * kw["i_multiplier"]

    if kw["voffset"] is not None and kw["ioffset"] is not None:
        volt_v = volt_v - kw["voffset"]
        curr_a = curr_a - kw["ioffset"]
    else:
        volt_v = volt_v - dc.offset[0]
        curr_a = curr_a - dc.offset[1]

    volt_v, curr_a = _filter_iv_data(volt_v, curr_a, kw)
    volt_v, curr_a = _correct_series_resistance(volt_v, curr_a, kw)

    v_out = np.linspace(-kw["vmax"], kw["vmax"], kw["npts"]) / dc.vgap
    i_out = np.interp(v_out, volt_v / dc.vgap, curr_a / dc.igap, left=0, right=0)
    return v_out, i_out


# -- import and clean --------------------------------------------------


def _load_iv(ivdata: np.ndarray, kw: dict) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column array, convert units, and clean it.

    Args:
        ivdata (ndarray): Two-column array of voltage and current.
        kw (dict): Merged parameters.

    Returns:
        tuple: ``(voltage in volts, current in amps)``.

    Raises:
        ValueError: If the array has the wrong shape or the units are not
            recognised.

    """
    ivdata = np.asarray(ivdata, dtype=float)
    if ivdata.ndim != 2 or ivdata.shape[1] != 2:
        raise ValueError("I-V data must be a 2-column array (voltage, current).")
    if kw["v_fmt"] not in VOLT_UNITS:
        raise ValueError(f"Unknown voltage unit: {kw['v_fmt']!r}")
    if kw["i_fmt"] not in CURR_UNITS:
        raise ValueError(f"Unknown current unit: {kw['i_fmt']!r}")

    volt_v = ivdata[:, 0] * VOLT_UNITS[kw["v_fmt"]]
    curr_a = ivdata[:, 1] * CURR_UNITS[kw["i_fmt"]]

    volt_v, curr_a = remove_nans_xy(volt_v, curr_a)
    volt_v, curr_a = _take_one_pass(volt_v, curr_a)
    volt_v, curr_a = sort_xy(volt_v, curr_a)
    return remove_doubles_xy(volt_v, curr_a)


def _take_one_pass(v: np.ndarray, i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract a single sweep from a hysteretic up-down-up measurement.

    I-V curves are usually swept from zero up, all the way down, then back
    to zero.  Hysteresis shifts the apparent gap voltage between the two
    directions, so only the portion sweeping *away* from zero is kept --
    that is where the largest gap voltage is measured.

    Args:
        v (ndarray): Bias voltage.
        i (ndarray): Tunneling current.

    Returns:
        tuple: One pass of ``(voltage, current)``, not necessarily sorted.

    """
    if v.size < 2:
        return v, i
    idx_min, idx_max = int(v.argmin()), int(v.argmax())

    if idx_min == 0 and idx_max == v.size - 1:
        return v, i
    if idx_max == 0 and idx_min == v.size - 1:
        return v[::-1], i[::-1]

    if idx_max < idx_min:
        start = int(np.abs(v[idx_max : idx_min + 1] - v[0]).argmin()) + idx_max
        vout = np.r_[v[start : idx_min + 1][::-1], v[0 : idx_max + 1]]
        iout = np.r_[i[start : idx_min + 1][::-1], i[0 : idx_max + 1]]
    else:
        start = int(np.abs(v[idx_min : idx_max + 1] - v[0]).argmin()) + idx_min
        vout = np.r_[v[0 : idx_min + 1][::-1], v[start : idx_max + 1]]
        iout = np.r_[i[0 : idx_min + 1][::-1], i[start : idx_max + 1]]
    return vout, iout


# -- corrections -------------------------------------------------------


def _correct_offset(volt_v, curr_a, kw):
    """Find and remove the voltage and current offset.

    An SIS I-V curve is odd about the origin, so the offset is whatever
    shift makes the curve overlap its own point reflection.  That is fitted
    here.

    Note:
        Supplying *both* ``voffset`` and ``ioffset`` skips the fit and uses
        them as given -- which is what the parameter documentation promises.
        QMix instead treats them only as a starting guess and fits anyway,
        so a known offset could still be moved.  Supplying just one is still
        treated as a starting guess.  If the sweep is unipolar there is
        nothing to reflect against and no fit is attempted.

    Args:
        volt_v (ndarray): Bias voltage in volts.
        curr_a (ndarray): Tunneling current in amps.
        kw (dict): Merged parameters.

    Returns:
        tuple: ``(voltage, current, (voffset, ioffset))``.

    """
    vrange = kw["voffset_range"]
    if np.isscalar(vrange):
        vrange = (-abs(vrange), abs(vrange))
    voffset, ioffset = kw["voffset"], kw["ioffset"]

    if voffset is not None and ioffset is not None:
        # Both offsets are known, so there is nothing to fit.
        return volt_v - voffset, curr_a - ioffset, (voffset, ioffset)

    bipolar = (volt_v.min() < vrange[0]) and (vrange[1] < volt_v.max())
    if not bipolar:
        voffset = 0.0 if voffset is None else voffset
        ioffset = 0.0 if ioffset is None else ioffset
        return volt_v - voffset, curr_a - ioffset, (voffset, ioffset)

    mask = (vrange[0] <= volt_v) & (volt_v <= vrange[1])
    v_red, i_red = volt_v[mask], curr_a[mask]
    probe = np.linspace(vrange[0] / 2, vrange[1] / 2, 101)

    # The offset is a bulk property of the curve, so smoothing first costs
    # almost nothing and makes the fit far more tolerant of measurement
    # noise: measured against synthetic data with a known offset, the
    # recovered voltage offset stays within 1 uV up to 0.2% current noise,
    # where the unsmoothed fit used by QMix has already failed completely.
    # Gaussian smoothing is symmetric, so it commutes with the point
    # reflection the fit relies on.
    if i_red.size > 32:
        width = max(3.0, i_red.size / 128.0)
        i_red = gauss_conv(i_red, sigma=width)

    def mismatch(offset):
        """Total disagreement between the curve and its reflection."""
        vv, ii = v_red - offset[0], i_red - offset[1]
        y1 = np.interp(probe, vv, ii)
        y2 = np.interp(probe, -vv[::-1], -ii[::-1])
        return float(np.sum(np.abs(y1 - y2)))

    guess = np.array(
        [
            0.0 if voffset is None else voffset,
            float(np.mean(i_red)) if ioffset is None else ioffset,
        ]
    )
    result = minimize(mismatch, x0=guess)
    voffset, ioffset = float(result.x[0]), float(result.x[1])
    return volt_v - voffset, curr_a - ioffset, (voffset, ioffset)


def _filter_iv_data(volt_v, curr_a, kw):
    """Smooth the I-V curve without smearing the gap transition.

    The curve is rotated by ``filter_theta`` before filtering so that the
    near-vertical transition becomes a well-behaved function of the rotated
    abscissa, filtered with a Savitzky-Golay window, and rotated back.  The
    technique follows Grimes et al. (2004).

    Args:
        volt_v (ndarray): Bias voltage in volts.
        curr_a (ndarray): Tunneling current in amps.
        kw (dict): Merged parameters.

    Returns:
        tuple: The filtered ``(voltage, current)``.

    Raises:
        ValueError: If the Savitzky-Golay window is longer than the data.

    """
    if not kw["filter_data"]:
        return volt_v, curr_a

    npts = kw["npts"]
    nwind, npoly = kw["filter_nwind"], kw["filter_npoly"]
    if nwind > npts:
        raise ValueError("filter_nwind must not exceed npts.")

    vmax, imax = volt_v.max(), curr_a.max()
    x, y = _rotate(volt_v / vmax, curr_a / imax, -kw["filter_theta"])

    x_grid = np.linspace(x.min(), x.max(), npts)
    y_filtered = savgol_filter(np.interp(x_grid, x, y), nwind, npoly)

    x, y = _rotate(x_grid, y_filtered, kw["filter_theta"])
    x_grid = np.linspace(x.min(), x.max(), npts)
    y = np.interp(x_grid, x, y)
    return x_grid * vmax, y * imax


def _rotate(x, y, theta):
    """Rotate a curve about the origin.

    Args:
        x (ndarray): Abscissae.
        y (ndarray): Ordinates.
        theta (float): Rotation angle in radians.

    Returns:
        tuple: The rotated ``(x, y)``.

    """
    cos, sin = np.cos(theta), np.sin(theta)
    return cos * x - sin * y, sin * x + cos * y


def _correct_series_resistance(vmeas, imeas, kw):
    """Remove a series resistance in the DC bias system.

    Args:
        vmeas (ndarray): Measured voltage in volts.
        imeas (ndarray): Measured current in amps.
        kw (dict): Merged parameters.

    Returns:
        tuple: The corrected ``(voltage, current)``.

    """
    rseries = kw["rseries"]
    if rseries is None:
        return vmeas, imeas

    with np.errstate(divide="ignore", invalid="ignore"):
        rstatic = vmeas / imeas
    rstatic = np.where(rstatic < 0, 0.0, rstatic)
    good = np.isfinite(rstatic)
    rj = rstatic[good] - rseries
    return imeas[good] * rj, imeas[good]


# -- measurements ------------------------------------------------------


def _vrn_mask(volt_v, kw):
    """Select the normal-state fitting window, with a clear error if empty.

    Args:
        volt_v (ndarray): Bias voltage in volts.
        kw (dict): Merged parameters.

    Returns:
        ndarray: Boolean mask selecting the window.

    Raises:
        ValueError: If fewer than two points fall inside ``vrn``.

    """
    vmin, vmax = kw["vrn"]
    mask = (vmin < volt_v) & (volt_v < vmax)
    if mask.sum() < 2:
        raise ValueError(
            f"No I-V data between {vmin} V and {vmax} V to fit the normal "
            f"resistance. Adjust the 'vrn' parameter."
        )
    return mask


def _local_line(volt_v, curr_a, centre, min_points=5):
    """Fit a line to the data closest to ``centre``.

    The window widens until it holds at least ``min_points`` samples, so the
    fit stays conditioned however coarsely the curve was sampled.  A fixed
    voltage window cannot do that: QMix uses +/-10 uV, which collapses to
    one or two points on a coarsely resampled curve and makes the polynomial
    fit singular.

    Args:
        volt_v (ndarray): Bias voltage in volts.
        curr_a (ndarray): Tunneling current in amps.
        centre (float): Centre of the window, in volts.
        min_points (int, optional): Smallest usable window.  Default is 5.

    Returns:
        ndarray: Polynomial coefficients ``[slope, intercept]``.

    """
    order = np.argsort(np.abs(volt_v - centre))
    take = max(min_points, 1)
    idx = order[:take]
    return np.polyfit(volt_v[idx], curr_a[idx], 1)


def _find_normal_resistance(volt_v, curr_a, kw):
    """Fit the normal-state branch of the I-V curve.

    Args:
        volt_v (ndarray): Bias voltage in volts.
        curr_a (ndarray): Tunneling current in amps.
        kw (dict): Merged parameters.

    Returns:
        tuple: ``(normal resistance in ohms, intercept voltage in volts)``.

    Raises:
        ValueError: If no data falls inside ``vrn``.

    """
    mask = _vrn_mask(volt_v, kw)
    slope_, intercept = np.polyfit(volt_v[mask], curr_a[mask], 1)
    return 1 / slope_, -intercept / slope_


def _find_gap_voltage(volt_v, curr_a, kw):
    """Locate the gap voltage.

    The gap is taken as the bias voltage at half the current of the
    "corner", where a line through the steepest part of the transition meets
    the normal-state line.

    Args:
        volt_v (ndarray): Bias voltage in volts.
        curr_a (ndarray): Tunneling current in amps.
        kw (dict): Merged parameters.

    Returns:
        float: The gap voltage, in volts.

    """
    mask = _vrn_mask(volt_v, kw)
    prn = np.polyfit(volt_v[mask], curr_a[mask], 1)

    vstep = volt_v[1] - volt_v[0]
    above = volt_v > 2e-3
    der = slope(volt_v[above], curr_a[above])
    der = gauss_conv(der, sigma=max(0.2e-3 / vstep, 1.0))
    vgap = volt_v[above][der.argmax()]

    pgap = _local_line(volt_v, curr_a, vgap)
    vcorner = (pgap[1] - prn[1]) / (prn[0] - pgap[0])
    icorner = pgap[0] * vcorner + pgap[1]
    return (icorner / 2 - pgap[1]) / pgap[0]


def _find_subgap_resistance(volt_v, curr_a, kw):
    """Measure the subgap resistance at ``vrsg``.

    Args:
        volt_v (ndarray): Bias voltage in volts.
        curr_a (ndarray): Tunneling current in amps.
        kw (dict): Merged parameters.

    Returns:
        float: The subgap resistance, in ohms.

    Raises:
        ValueError: If there is no data near ``vrsg``.

    """
    vrsg = kw["vrsg"]
    if not volt_v.min() <= vrsg <= volt_v.max():
        raise ValueError(
            f"vrsg = {vrsg} V is outside the measured range "
            f"({volt_v.min():.3g} to {volt_v.max():.3g} V); cannot measure "
            f"the subgap resistance."
        )
    mask = (vrsg - 1e-4 < volt_v) & (volt_v < vrsg + 1e-4)
    if mask.sum() >= 2:
        return 1 / np.polyfit(volt_v[mask], curr_a[mask], 1)[0]
    return 1 / _local_line(volt_v, curr_a, vrsg)[0]
