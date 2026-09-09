r"""Recovering the embedding circuit from a pumped I-V curve.

Given a measured pumped I-V curve, Tucker theory gives the drive level
:math:`\alpha` and hence the junction's AC voltage and impedance at every
bias point on the first photon step.  The junction is driven by a Thevenin
source, so those points all lie on one load line, and the source impedance
:math:`Z_T` and voltage :math:`V_T` can be fitted to them.  This is the RF
voltage-match method of Skalare (1989) and Withington *et al.* (1995).

The error to minimise is Eqn. 26 of Withington *et al.*:

.. math::

    \varepsilon(Z_T) = \frac{1}{N}\left[
        \sum_i |V_i|^2
        - \frac{\big(\sum_i |V_i Z_i / (Z_T + Z_i)|\big)^2}
               {\sum_i |Z_i / (Z_T + Z_i)|^2} \right]

and the matching source voltage is Eqn. 27.

What is different here
----------------------

QMix evaluates that error on a 101 x 201 grid with a Python double loop --
20 301 iterations, each summing over the fit region -- and returns the best
*grid point*, so the answer is quantised to the grid spacing.  Here the
whole surface is one broadcast expression, and the grid minimum is then
polished with a local optimiser, so the result is not tied to the grid at
all.

Examples:

    >>> import numpy as np
    >>> from qpmix.exp.zemb import error_surface, recover_zemb
    >>> zj = np.array([0.4 + 0.2j, 0.5 + 0.1j, 0.6 - 0.1j])
    >>> zt = 0.3 - 0.4j
    >>> vt = 0.5
    >>> vj = vt * zj / (zt + zj)                     # a perfect load line
    >>> result = recover_zemb(vj, zj)
    >>> bool(abs(result.zt - zt) < 1e-6 and abs(result.vt - vt) < 1e-6)
    True

"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.optimize import minimize

__all__ = [
    "GOOD_ERROR",
    "ZembResult",
    "error_function",
    "error_surface",
    "recover_zemb",
    "source_voltage",
]

#: A fit whose residual error is below this is reported as a good fit.
GOOD_ERROR = 7e-7


class ZembResult(NamedTuple):
    """The recovered Thevenin equivalent circuit.

    Attributes:
        zt (complex): Embedding impedance, normalized to the normal
            resistance.
        vt (float): Embedding voltage, normalized to the gap voltage.
        err (float): Residual of the fit.
        fit_good (bool): Whether the residual is below :data:`GOOD_ERROR`.
        zt_real (ndarray): Real axis of the searched error surface.
        zt_imag (ndarray): Imaginary axis of the searched error surface.
        err_surf (ndarray): The error surface, shape
            ``(len(zt_real), len(zt_imag))``.

    """

    zt: complex
    vt: float
    err: float
    fit_good: bool
    zt_real: np.ndarray
    zt_imag: np.ndarray
    err_surf: np.ndarray


def error_function(vj: np.ndarray, zj: np.ndarray, zt: complex) -> float:
    """Mismatch between the measured points and one candidate source.

    Eqn. 26 of Withington *et al.* (1995).

    Args:
        vj (ndarray): Junction AC voltage at each bias point, normalized.
        zj (ndarray): Junction AC impedance at each bias point, normalized.
        zt (complex): Candidate embedding impedance.

    Returns:
        float: The mismatch, zero for a perfect fit.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = zj / (zt + zj)
        err1 = np.sum(np.abs(vj) ** 2)
        err2 = np.sum(np.abs(vj * ratio))
        err3 = np.sum(np.abs(ratio) ** 2)
        value = (err1 - err2**2 / err3) / vj.size
    return float(value) if np.isfinite(value) else np.inf


def source_voltage(vj: np.ndarray, zj: np.ndarray, zt: complex) -> float:
    """Source voltage that best matches the measured points.

    Eqn. 27 of Withington *et al.* (1995).

    Args:
        vj (ndarray): Junction AC voltage at each bias point, normalized.
        zj (ndarray): Junction AC impedance at each bias point, normalized.
        zt (complex): Embedding impedance.

    Returns:
        float: The embedding voltage, normalized to the gap voltage.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = zj / (zt + zj)
        value = np.sum(np.abs(vj * ratio)) / np.sum(np.abs(ratio) ** 2)
    return float(value) if np.isfinite(value) else 0.0


def error_surface(
    vj: np.ndarray, zj: np.ndarray, zt_real: np.ndarray, zt_imag: np.ndarray
) -> np.ndarray:
    """Evaluate the error over a grid of candidate impedances.

    One broadcast expression rather than a double loop; for the default
    101 x 201 grid that is roughly a hundredfold saving.

    Args:
        vj (ndarray): Junction AC voltage at each bias point.
        zj (ndarray): Junction AC impedance at each bias point.
        zt_real (ndarray): Candidate resistances.
        zt_imag (ndarray): Candidate reactances.

    Returns:
        ndarray: The error surface, shape ``(len(zt_real), len(zt_imag))``.

    """
    zt = zt_real[:, None, None] + 1j * zt_imag[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = zj[None, None, :] / (zt + zj[None, None, :])
        err1 = np.sum(np.abs(vj) ** 2)
        err2 = np.sum(np.abs(vj[None, None, :] * ratio), axis=-1)
        err3 = np.sum(np.abs(ratio) ** 2, axis=-1)
        surface = (err1 - err2**2 / err3) / vj.size
    return np.where(np.isfinite(surface), surface, np.inf)


def recover_zemb(
    vj: np.ndarray,
    zj: np.ndarray,
    remb_range: tuple[float, float] = (0.0, 1.0),
    xemb_range: tuple[float, float] = (-1.0, 1.0),
    nreal: int = 101,
    nimag: int = 201,
    zemb: complex | None = None,
    refine: bool = True,
) -> ZembResult:
    """Fit the Thevenin equivalent source to measured junction points.

    A grid search locates the basin, then a local optimiser polishes the
    answer off the grid.

    Args:
        vj (ndarray): Junction AC voltage at each bias point, normalized.
        zj (ndarray): Junction AC impedance at each bias point, normalized.
        remb_range (tuple, optional): Resistances to search, normalized to
            ``Rn``.  Default is ``(0, 1)``.
        xemb_range (tuple, optional): Reactances to search.  Default is
            ``(-1, 1)``.
        nreal (int, optional): Grid points along the real axis.  Default is
            101.
        nimag (int, optional): Grid points along the imaginary axis.
            Default is 201.
        zemb (complex, optional): Force this embedding impedance instead of
            fitting one.  Default is None.
        refine (bool, optional): Polish the grid minimum with a local
            optimiser.  Default is True.

    Returns:
        ZembResult: The recovered circuit and the error surface.

    Raises:
        ValueError: If the two input arrays have different shapes.

    """
    vj = np.asarray(vj)
    zj = np.asarray(zj)
    if vj.shape != zj.shape:
        raise ValueError("vj and zj must have the same shape.")

    # Points where the AC current vanishes give an infinite junction
    # impedance and carry no information about the source.
    good = np.isfinite(vj) & np.isfinite(zj)
    if not good.all():
        vj, zj = vj[good], zj[good]
    if vj.size < 2:
        raise ValueError("Need at least 2 usable points to fit the source.")

    zt_real = np.linspace(remb_range[0], remb_range[1], nreal)
    zt_imag = np.linspace(xemb_range[0], xemb_range[1], nimag)
    surface = error_surface(vj, zj, zt_real, zt_imag)

    i, j = np.unravel_index(np.argmin(surface), surface.shape)
    best = complex(zt_real[i], zt_imag[j])
    err = float(surface[i, j])

    if zemb is not None:
        best = complex(zemb)
        err = error_function(vj, zj, best)
    elif refine:
        # The grid only locates the basin; the minimum itself is continuous.
        # Keep the search inside the requested box, or the optimiser can
        # wander into unphysical territory such as negative resistance.
        result = minimize(
            lambda p: error_function(vj, zj, complex(p[0], p[1])),
            x0=np.array([best.real, best.imag]),
            method="Nelder-Mead",
            bounds=(tuple(remb_range), tuple(xemb_range)),
        )
        if result.fun < err:
            best = complex(result.x[0], result.x[1])
            err = float(result.fun)

    return ZembResult(
        zt=best,
        vt=source_voltage(vj, zj, best),
        err=err,
        fit_good=err <= GOOD_ERROR,
        zt_real=zt_real,
        zt_imag=zt_imag,
        err_surf=surface,
    )
