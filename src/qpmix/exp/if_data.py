"""Importing and analyzing measured IF output power.

The "IF data" is the intermediate-frequency output power of the mixer
versus DC bias voltage.  *DC IF data* is measured with no LO injection and
is used to calibrate the power scale; *IF data* is measured with the LO on,
for a hot and a cold blackbody load, and gives the noise temperature and
conversion gain.

Two calibrations happen here:

**Woody's method** (Woody 1985).  With no LO, the IF output above the gap is
dominated by shot noise, which rises linearly with bias voltage at a known
rate of 5.8 K/mV.  Fitting that slope converts the measured power from
arbitrary units into kelvin, and extrapolating the fit back to the
intercept voltage of the normal-state branch gives the IF noise
contribution of the receiver chain.

**Y-factor.**  With the LO on, the ratio of hot-load to cold-load output
power gives the noise temperature, using Callen-Welton load temperatures so
that the half-photon zero-point term is included.

Input is a two-column NumPy array: bias voltage, then IF power.

Differences from upstream QMix
------------------------------

* QMix calls ``numpy.seterr(divide='ignore', invalid='ignore')`` at import
  time, which silently disables those warnings for the entire interpreter.
  Here the suppression is scoped to the handful of expressions that need it.
* Argument validation raises :class:`ValueError` instead of using
  ``assert``.
* The automatic shot-noise window search is expressed as an explicit,
  documented sequence of criteria rather than a chain of masks.

Examples:

    >>> import numpy as np
    >>> from qpmix.exp.if_data import dcif_data
    >>> from qpmix.exp.simulate import simulate_dciv, simulate_if_power
    >>> from qpmix.exp.iv_data import dciv_curve
    >>> _, _, dc = dciv_curve(simulate_dciv(), verbose=False)
    >>> raw = simulate_if_power(None, if_noise=8.0, vint=dc.vint)
    >>> _, meta = dcif_data(raw, dc, vshot=(3.5e-3, 5.5e-3))
    >>> bool(abs(meta.if_noise - 8.0) < 0.5)
    True

"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import scipy.constants as sc
from scipy import stats
from scipy.signal import savgol_filter

from qpmix.exp.clean_data import clean_matrix
from qpmix.exp.iv_data import VOLT_UNITS, DCIVData
from qpmix.exp.parameters import merge_params
from qpmix.mathfn import slope_span_n
from qpmix.mathfn.filters import gauss_conv

__all__ = ["DCIFData", "dcif_data", "if_data"]

#: Shot noise raises the IF output by this many kelvin per mV of bias.
SHOT_SLOPE_K_PER_MV = 5.8


class DCIFData(NamedTuple):
    """Calibration derived from IF power measured without an LO.

    Attributes:
        if_noise (float): IF noise contribution of the receiver chain, in
            kelvin.
        corr (float): Factor converting the measured power into kelvin.
        if_fit (bool): Whether the recovered IF noise is physically
            plausible.
        shot_slope (ndarray): Two-column array of normalized bias voltage
            and the fitted shot-noise line, in kelvin.
        vmax (float): Largest bias voltage in the data, in volts.

    """

    if_noise: float
    corr: float
    if_fit: bool
    shot_slope: np.ndarray
    vmax: float


def dcif_data(ifdata: np.ndarray, dc: DCIVData, **kwargs: Any):
    """Calibrate IF power measured with no LO injection.

    Args:
        ifdata (ndarray): Two-column array of voltage and IF power.
        dc (qpmix.exp.iv_data.DCIVData): Metadata from the DC I-V curve.

    Keyword Args:
        **kwargs: Any parameter from :data:`qpmix.exp.parameters.PARAMS`.
            Relevant here: ``v_fmt``, ``v_multiplier``, ``vmax``,
            ``ifdata_npts``, ``ifdata_sigma``, ``rseries`` and ``vshot``.

    Returns:
        tuple: ``(calibrated IF data, DCIFData)``.  The IF data is a
        two-column array of normalized bias voltage and power in kelvin.

    """
    kw = merge_params(kwargs)
    data = _load_if(ifdata, dc, kw)
    if_noise, corr, shot_line, good = _find_if_noise(data, dc, kw)
    data[:, 1] *= corr

    meta = DCIFData(
        if_noise=float(if_noise),
        corr=float(corr),
        if_fit=bool(good),
        shot_slope=np.vstack((data[:, 0], shot_line)).T,
        vmax=float(data[:, 0].max() * dc.vgap),
    )
    return data, meta


def if_data(
    if_hot: np.ndarray,
    if_cold: np.ndarray,
    dc: DCIVData,
    dcif: DCIFData | None = None,
    **kwargs: Any,
):
    """Analyze a hot/cold load measurement.

    Args:
        if_hot (ndarray): Two-column array of voltage and IF power with the
            hot load.
        if_cold (ndarray): The same, with the cold load.
        dc (qpmix.exp.iv_data.DCIVData): Metadata from the DC I-V curve.
        dcif (DCIFData, optional): Calibration from :func:`dcif_data`.  If
            omitted, it is derived from the average of the hot and cold
            measurements.  Default is None.

    Keyword Args:
        **kwargs: Any parameter from :data:`qpmix.exp.parameters.PARAMS`.
            Relevant here: ``t_hot``, ``t_cold``, ``freq``, ``vbest``,
            ``best_pt``, plus everything :func:`dcif_data` uses.

    Returns:
        tuple: ``(results, index of the best bias point, DCIFData)`` where
        ``results`` has columns of normalized bias voltage, hot power, cold
        power, noise temperature and gain.

    Raises:
        ValueError: If the hot and cold measurements are on different bias
            voltages, or ``best_pt`` is not recognised.

    """
    kw = merge_params(kwargs)
    hot = _load_if(if_hot, dc, kw)
    cold = _load_if(if_cold, dc, kw)

    if dcif is None or dcif.corr is None:
        average = (hot + cold) / 2.0
        if_noise, corr, shot_line, good = _find_if_noise(average, dc, kw)
        shot_slope = np.vstack((cold[:, 0], shot_line)).T
    else:
        if_noise, corr = dcif.if_noise, dcif.corr
        shot_slope, good = dcif.shot_slope, dcif.if_fit

    hot[:, 1] *= corr
    cold[:, 1] *= corr

    tn, gain, idx_best = _find_tn_gain(hot, cold, dc, kw)
    results = np.vstack((hot[:, 0], hot[:, 1], cold[:, 1], tn, gain)).T

    meta = DCIFData(
        if_noise=float(if_noise),
        corr=float(corr),
        if_fit=bool(good),
        shot_slope=shot_slope,
        vmax=float(hot[:, 0].max() * dc.vgap),
    )
    if kw["verbose"]:
        print(f"\t- IF noise:\t{if_noise:+6.2f} K")
        print(f"\t- noise temp:\t{tn[idx_best]:6.1f} K")
        with np.errstate(divide="ignore", invalid="ignore"):
            print(f"\t- gain:\t\t{10 * np.log10(gain[idx_best]):+6.2f} dB")
    return results, idx_best, meta


# -- noise temperature -------------------------------------------------


def callen_welton(freq_hz: float, t_phys: float) -> float:
    """Callen-Welton effective temperature of a blackbody load.

    Includes the half-photon zero-point term, which matters at
    submillimetre wavelengths where ``h f`` is comparable to ``k T``.

    Args:
        freq_hz (float): Frequency in Hz.
        t_phys (float): Physical temperature in kelvin.

    Returns:
        float: The effective temperature in kelvin.

    """
    x = sc.h * float(freq_hz) / (2 * sc.k)
    return x / np.tanh(x / float(t_phys))


def _find_tn_gain(hot, cold, dc, kw):
    """Noise temperature and gain from the hot/cold Y-factor.

    Args:
        hot (ndarray): Calibrated hot-load IF data.
        cold (ndarray): Calibrated cold-load IF data.
        dc (DCIVData): DC I-V metadata.
        kw (dict): Merged parameters.

    Returns:
        tuple: ``(noise temperature, gain, index of the best bias point)``.

    Raises:
        ValueError: If the bias voltages do not match, the LO frequency is
            unset, or ``best_pt`` is not recognised.

    """
    vnorm = hot[:, 0]
    if not np.array_equal(vnorm, cold[:, 0]):
        raise ValueError("Hot and cold measurements must be on the same bias grid.")
    if kw["freq"] is None:
        raise ValueError("The LO frequency ('freq', in GHz) must be set.")

    p_hot, p_cold = hot[:, 1], cold[:, 1]
    t_hot = callen_welton(kw["freq"] * 1e9, kw["t_hot"])
    t_cold = callen_welton(kw["freq"] * 1e9, kw["t_cold"])

    with np.errstate(divide="ignore", invalid="ignore"):
        y = p_hot / p_cold
    y = np.where(np.isfinite(y), y, 1.0)
    # A Y-factor at or below 1 is unphysical and would give a negative
    # noise temperature; clamp it just above 1.
    y = np.maximum(y, 1.0 + 1e-10)

    tn = (t_hot - t_cold * y) / (y - 1)
    gain = (p_hot - p_cold) / (t_hot - t_cold)

    if kw["vbest"] is not None:
        idx = int(np.abs(vnorm * dc.vgap - kw["vbest"]).argmin())
    elif kw["best_pt"].lower() == "max gain":
        idx = int(np.nanargmax(gain))
    elif kw["best_pt"].lower() == "min tn":
        idx = int(np.nanargmin(np.where(tn > 0, tn, np.inf)))
    else:
        raise ValueError(f"best_pt not recognized: {kw['best_pt']!r}")

    return tn, gain, idx


# -- Woody's method ----------------------------------------------------


def _shot_noise_window(x, y):
    """Find the bias range where the IF output is pure shot noise.

    The shot-noise region is linear, so it is located by looking for where
    the second derivative is small, the first derivative matches the bulk
    value, and the points form a contiguous run rather than isolated
    coincidences.  Josephson features break all three.

    Args:
        x (ndarray): Normalized bias voltage.
        y (ndarray): IF power.

    Returns:
        ndarray: Boolean mask selecting the shot-noise region.

    """
    y_filt = savgol_filter(y, 21, 3)
    first = savgol_filter(slope_span_n(x, y_filt, 11), 51, 3)
    second = savgol_filter(np.abs(slope_span_n(x, first, 11)), 51, 3)

    # 1. Above the gap, and flat.
    flat = np.abs(second) < np.max(np.abs(second)) * 1e-2
    mask = (x > 1.7) & flat
    # 2. Rising at about the bulk rate.
    if mask.any():
        median = np.median(first[mask])
        mask &= (first > 0.0) & (first < median * 2)
    # 3. At least two adjacent points, so isolated hits are dropped.
    adjacent = np.zeros_like(mask)
    adjacent[:-1] = mask[:-1] & mask[1:]
    return mask & adjacent


def _find_if_noise(data, dc, kw):
    """Calibrate the IF power scale from the shot-noise slope.

    Args:
        data (ndarray): Two-column IF data, normalized voltage and power.
        dc (DCIVData): DC I-V metadata.
        kw (dict): Merged parameters.

    Returns:
        tuple: ``(IF noise in K, correction factor, fitted line, is the fit
        plausible)``.

    """
    x, y = data[:, 0], data[:, 1]
    vshot = kw["vshot"]

    if vshot is None:
        mask = _shot_noise_window(x, y)
    else:
        ranges = vshot if np.ndim(vshot[0]) else (vshot,)
        mask = np.zeros_like(x, dtype=bool)
        for lo, hi in ranges:
            mask |= (lo < x * dc.vgap) & (x * dc.vgap < hi)

    if mask.sum() < 5:
        if kw["verbose"]:
            print("\t\tShot noise fit failed; using all bias above 2*Vgap.")
        mask = x > 2.0

    slope_, intercept, _, _, _ = stats.linregress(x[mask], y[mask])

    if not np.isfinite(slope_) or slope_ <= 0:
        # Shot noise must rise with bias voltage.  A flat or falling fit
        # means the window missed it, and rescaling by 5.8 / slope would
        # produce an infinite correction factor that then silently poisons
        # every downstream power with NaN.
        if kw["verbose"]:
            print(
                "\t\tShot noise slope is not positive; leaving the IF "
                "power in its measured units."
            )
        return 0.0, 1.0, np.zeros_like(x), False

    line = slope_ * x + intercept

    # Rescale so the fitted slope equals the physical 5.8 K/mV.
    corr = SHOT_SLOPE_K_PER_MV / slope_ * dc.vgap * 1e3
    line = line * corr

    if_noise = float(np.interp(dc.vint / dc.vgap, x, line))
    return if_noise, corr, line, 0 < if_noise < 50


# -- import ------------------------------------------------------------


def _load_if(ifdata, dc, kw):
    """Import IF data onto the normalized bias grid.

    Args:
        ifdata (ndarray): Two-column array of voltage and IF power.
        dc (DCIVData): DC I-V metadata.
        kw (dict): Merged parameters.

    Returns:
        ndarray: Two-column array of normalized voltage and IF power.

    Raises:
        ValueError: If the input is not a two-column array or the voltage
            units are unknown.

    """
    ifdata = np.array(ifdata, dtype=float, copy=True)
    if ifdata.ndim != 2 or ifdata.shape[1] != 2:
        raise ValueError("IF data must be a 2-column array (voltage, power).")
    if kw["v_fmt"] not in VOLT_UNITS:
        raise ValueError(f"Unknown voltage unit: {kw['v_fmt']!r}")

    ifdata[:, 0] *= VOLT_UNITS[kw["v_fmt"]]
    ifdata = clean_matrix(ifdata)
    ifdata[:, 0] *= kw["v_multiplier"]
    ifdata[:, 0] -= dc.offset[0]

    if kw["rseries"] is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rstatic = dc.vraw / dc.iraw
        rstatic = np.where(rstatic < 0, 0.0, rstatic)
        v = ifdata[:, 0]
        rj = np.interp(v, dc.vraw, rstatic) - kw["rseries"]
        ifdata[:, 0] = np.interp(v, dc.vraw, dc.iraw) * rj

    ifdata[:, 0] /= dc.vgap

    v_out = np.linspace(0, kw["vmax"] / dc.vgap, kw["ifdata_npts"])
    p_out = np.interp(v_out, ifdata[:, 0], ifdata[:, 1], left=0, right=0)
    out = np.vstack((v_out, p_out)).T

    sigma = kw["ifdata_sigma"]
    if sigma is not None:
        step = (out[1, 0] - out[0, 0]) * dc.vgap
        # Values above 0.5 are already in samples, not volts.
        sigma = sigma * step if sigma > 0.5 else sigma
        out[:, 1] = gauss_conv(out[:, 1], sigma / step)
    return out
