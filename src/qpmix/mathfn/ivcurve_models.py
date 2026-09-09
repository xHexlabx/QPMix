"""DC I-V curve models for SIS junctions.

Models range from the idealised step (:func:`perfect`) through the
polynomial and exponential fits used in the literature, up to the
:func:`expanded` model, which adds leakage current, the proximity effect
and a current offset above the gap.

All models take and return *normalized* quantities: voltage in units of the
gap voltage ``Vgap`` and current in units of the gap current
``Igap = Vgap / Rn``.

Compared to upstream QMix these implementations differ in two ways:

* :func:`polynomial` is algebraically rearranged so it cannot overflow.
  The textbook form ``v**(2n+1) / (1 + v**(2n))`` overflows float64 for
  ``order >= 51`` at the voltages the response function is tabulated over,
  even though the *result* is perfectly well behaved (it tends to ``v``).
* The overflow-prone exponential models are wrapped in
  :func:`numpy.errstate` instead of calling :func:`numpy.seterr`, which
  would silently and permanently change NumPy's error handling for the
  whole interpreter.

Examples:

    >>> import numpy as np
    >>> from qpmix.mathfn.ivcurve_models import perfect, polynomial
    >>> perfect(np.array([0.5, 1.0, 2.0]))
    array([0. , 0.5, 2. ])
    >>> float(np.round(polynomial(2.0, order=50), 6))
    2.0

"""

from __future__ import annotations

import numpy as np

__all__ = [
    "expanded",
    "exponential",
    "perfect",
    "perfect_kk",
    "polynomial",
]


def perfect(voltage: np.ndarray | float) -> np.ndarray | float:
    """The ideal DC I-V curve.

    Zero below the gap, equal to the bias voltage above it, and 0.5 exactly
    at the gap.

    Args:
        voltage (ndarray or float): Normalized bias voltage.

    Returns:
        ndarray or float: Normalized DC tunneling current.

    """
    v = np.asarray(voltage, dtype=float)
    current = np.where(np.abs(v) < 1, 0.0, v)
    current = np.where(v == 1.0, 0.5, current)
    current = np.where(v == -1.0, -0.5, current)
    if np.isscalar(voltage) or np.ndim(voltage) == 0:
        return float(current)
    return current


def perfect_kk(
    voltage: np.ndarray | float, max_kk: float = 100.0
) -> np.ndarray | float:
    """Analytic Kramers-Kronig transform of the perfect I-V curve.

    Args:
        voltage (ndarray or float): Normalized bias voltage.
        max_kk (float, optional): Value to substitute at the logarithmic
            singularities ``v = +/-1``.  Default is 100.

    Returns:
        ndarray or float: The KK transform.

    """
    v = np.asarray(voltage, dtype=float)
    kk = np.full(v.shape, float(max_kk))
    finite = np.abs(v) != 1.0
    vf = v[finite]
    with np.errstate(divide="ignore", invalid="ignore"):
        kk[finite] = -1 / np.pi * (2 + vf * np.log(np.abs((vf - 1) / (vf + 1))))
    if np.isscalar(voltage) or np.ndim(voltage) == 0:
        return float(kk)
    return kk


def polynomial(voltage: np.ndarray | float, order: float = 50) -> np.ndarray | float:
    """The polynomial I-V curve model of Kennedy (1999).

    Higher ``order`` gives a sharper transition at the gap.

    Args:
        voltage (ndarray or float): Normalized bias voltage.
        order (float, optional): Polynomial order, usually 30-50.  Default
            is 50.

    Returns:
        ndarray or float: Normalized DC tunneling current.

    """
    v = np.asarray(voltage, dtype=float)
    absv = np.abs(v)
    p = 2 * order

    # Two algebraically identical branches, each of which keeps the power
    # below 1 so that neither can overflow:
    #   |v| >= 1:  v / (1 + |v|**-p)
    #   |v| <  1:  v * |v|**p / (1 + |v|**p)
    out = np.empty(v.shape, dtype=float)
    big = absv >= 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out[big] = v[big] / (1.0 + absv[big] ** (-p))
    small = ~big
    vs = absv[small] ** p
    out[small] = v[small] * vs / (1.0 + vs)

    if np.isscalar(voltage) or np.ndim(voltage) == 0:
        return float(out)
    return out


def exponential(
    voltage: np.ndarray | float,
    vgap: float = 2.8e-3,
    rn: float = 14.0,
    rsg: float = 300.0,
    agap: float = 4e4,
    model: str = "fixed",
) -> np.ndarray | float:
    """The exponential I-V curve model of Rashid et al. (2016).

    Note:
        The equation as published yields a subgap resistance that is half
        of the intended value, and a slightly low normal resistance.  The
        default ``model="fixed"`` corrects this; ``model="original"``
        reproduces the published form.

    Args:
        voltage (ndarray or float): Normalized bias voltage.
        vgap (float, optional): Gap voltage in volts.  Default is 2.8e-3.
        rn (float, optional): Normal resistance in ohms.  Default is 14.
        rsg (float, optional): Subgap resistance in ohms.  Default is 300.
        agap (float, optional): Gap linearity coefficient.  Default is 4e4.
        model (str, optional): ``"fixed"`` (equivalently ``"corrected"``)
            or ``"original"``.  Default is ``"fixed"``.

    Returns:
        ndarray or float: Normalized DC tunneling current.

    Raises:
        ValueError: If ``model`` is not recognised.

    """
    igap = vgap / rn
    v_v = np.asarray(voltage, dtype=float) * vgap
    kind = model.lower()

    with np.errstate(over="ignore"):
        if kind in ("fixed", "corrected"):
            i_a = (
                # Sub-gap resistance
                v_v / (rsg * 2) * _logistic(-agap * (v_v + vgap))
                - v_v / (rsg * 2) * _logistic(agap * (v_v + vgap))
                - v_v / (rsg * 2) * _logistic(-agap * (v_v - vgap))
                + v_v / (rsg * 2) * _logistic(agap * (v_v - vgap))
                # Normal resistance
                + v_v / rn * _logistic(agap * (v_v + vgap))
                + v_v / rn * _logistic(-agap * (v_v - vgap))
            )
        elif kind == "original":
            i_a = (
                # Sub-gap resistance
                v_v / rsg * _logistic(-agap * (v_v + vgap))
                + v_v / rsg * _logistic(agap * (v_v - vgap))
                # Normal resistance
                + v_v / rn * _logistic(agap * (v_v + vgap))
                + v_v / rn * _logistic(-agap * (v_v - vgap))
            )
        else:
            raise ValueError(f"Model not recognized: {model!r}")

    out = i_a / igap
    if np.isscalar(voltage) or np.ndim(voltage) == 0:
        return float(out)
    return out


def expanded(
    voltage: np.ndarray | float,
    vgap: float = 2.8e-3,
    rn: float = 14.0,
    rsg: float = 5e2,
    agap: float = 4e4,
    a0: float = 1e4,
    ileak: float = 5e-6,
    vnot: float = 2.85e-3,
    inot: float = 1e-5,
    anot: float = 2e4,
    ioff: float = 1e-5,
) -> np.ndarray | float:
    """An expanded I-V curve model with leakage, proximity effect and offset.

    Built on top of the exponential model, with extra terms for leakage
    current, the proximity-effect notch above the gap, the onset of thermal
    tunneling, and the reduced current amplitude often seen above the gap.
    Complex, but it reproduces measured curves closely.

    Args:
        voltage (ndarray or float): Normalized bias voltage.
        vgap (float, optional): Gap voltage in volts.  Default is 2.8e-3.
        rn (float, optional): Normal resistance in ohms.  Default is 14.
        rsg (float, optional): Subgap resistance in ohms.  Default is 5e2.
        agap (float, optional): Gap linearity coefficient.  Default is 4e4.
        a0 (float, optional): Linearity coefficient at the origin.  Default
            is 1e4.
        ileak (float, optional): Leakage current amplitude in amps.
            Default is 5e-6.
        vnot (float, optional): Notch location in volts.  Default is
            2.85e-3.
        inot (float, optional): Notch current amplitude in amps.  Default
            is 1e-5.
        anot (float, optional): Notch linearity.  Default is 2e4.
        ioff (float, optional): Current offset in amps.  Default is 1e-5.

    Returns:
        ndarray or float: Normalized DC tunneling current.

    """
    v = np.asarray(voltage, dtype=float)
    v_v = v * vgap
    igap = vgap / rn

    with np.errstate(over="ignore"):
        i_a = (
            # Leakage current
            ileak * 2 * _logistic(-a0 * v_v)
            - ileak * np.ones_like(v)
            - ileak * _logistic(-agap * (v_v - vgap))
            + ileak * _logistic(agap * (v_v + vgap))
            # Sub-gap resistance
            + v_v / (rsg * 2) * _logistic(-agap * (v_v + vgap))
            - v_v / (rsg * 2) * _logistic(agap * (v_v + vgap))
            - v_v / (rsg * 2) * _logistic(-agap * (v_v - vgap))
            + v_v / (rsg * 2) * _logistic(agap * (v_v - vgap))
            # Transition and normal resistance
            + v_v / rn * _logistic(agap * (v_v + vgap))
            + v_v / rn * _logistic(-agap * (v_v - vgap))
            # Notch above the gap (proximity effect)
            + inot * _logistic(anot * (v_v - vnot))
            + inot * _logistic(anot * (v_v + vnot))
            - inot
            # Current offset seen above the gap
            + ioff * _logistic(agap * (v_v - vgap))
            + ioff * _logistic(agap * (v_v + vgap))
            - ioff
        )

    out = i_a / igap
    if np.isscalar(voltage) or np.ndim(voltage) == 0:
        return float(out)
    return out


def _logistic(z: np.ndarray) -> np.ndarray:
    """Numerically stable ``1 / (1 + exp(z))``.

    Args:
        z (ndarray): Exponent.

    Returns:
        ndarray: The logistic value, with no overflow for large ``|z|``.

    """
    # exp() overflows for z > ~709; there the result underflows to 0 anyway,
    # so evaluate the mirrored form where the exponent is negative.
    z = np.asarray(z, dtype=float)
    out = np.empty(z.shape, dtype=float)
    pos = z >= 0
    ez = np.exp(-z[pos])
    out[pos] = ez / (1.0 + ez)
    out[~pos] = 1.0 / (1.0 + np.exp(z[~pos]))
    return out
