r"""Tucker theory: the small-signal limit used to analyze measured data.

For a junction driven by a *single* tone of drive level
:math:`\alpha = V_\omega / V_{ph}`, Tucker and Feldman (*Rev. Mod. Phys.*
**57**, 1055, 1985) give the tunneling currents in closed form:

.. math::

    I_{dc}(V_0) &= \sum_n J_n^2(\alpha)\, I_{dc}^0(V_0 + n V_{ph}) \\
    I_\omega(V_0) &= \sum_n J_n(\alpha)
        \Big[ \big(J_{n-1} + J_{n+1}\big) I_{dc}^0(V_0 + n V_{ph})
        + i\big(J_{n-1} - J_{n+1}\big) I_{kk}^0(V_0 + n V_{ph}) \Big]

That is enough to analyze measured data — recovering the drive level from a
pumped I-V curve, then the junction impedance — without running a full
harmonic balance.  :mod:`qpmix.exp.exp_data` uses it for exactly that, and
the QPMix test suite uses it as an independent analytic check on
:func:`qpmix.qtcurrent.qtcurrent`.

What is different here
----------------------

*Vectorised Bessel evaluation.*  QMix loops over ``n`` in Python, calling
``scipy.special.jv`` and interpolating the response function once per
order — ``2*num_b + 1`` round trips through both.  Here every order is
evaluated in one call, and the response function is interpolated for all
orders at once, which the O(1) table lookup in :mod:`qpmix.interp` makes
almost free.  The reflection formula :math:`J_{-n} = (-1)^n J_n` halves the
Bessel work again.

*Newton instead of bisection.*  Recovering :math:`\alpha` from a measured
pumped I-V curve means inverting the first equation above.  QMix takes 15
fixed bisection steps, so the answer is only ever good to
``alpha_max / 2**15`` and costs 15 full evaluations.  The derivative is
available analytically,

.. math::

    \frac{\partial I_{dc}}{\partial \alpha}
        = \sum_n J_n(\alpha)\big(J_{n-1}(\alpha) - J_{n+1}(\alpha)\big)
          I_{dc}^0(V_0 + n V_{ph}),

and the neighbouring orders are already in hand, so a safeguarded Newton
iteration converges to machine precision in a handful of steps.

Examples:

    >>> import numpy as np
    >>> from qpmix.respfn import RespFnPolynomial
    >>> from qpmix.exp.tucker import pumped_iv_curve, recover_alpha
    >>> resp = RespFnPolynomial(50, verbose=False)
    >>> vb = np.linspace(0.75, 0.95, 41)          # the first photon step
    >>> idc = pumped_iv_curve(resp, vb, 0.3, 0.8)
    >>> alpha = recover_alpha(resp, vb, idc, 0.3)
    >>> bool(np.abs(alpha - 0.8).max() < 1e-9)
    True

"""

from __future__ import annotations

import numpy as np
from scipy.special import jv

__all__ = [
    "ac_current",
    "bessel_ladder",
    "pumped_iv_curve",
    "recover_alpha",
]


def bessel_ladder(alpha: np.ndarray, num_b: int) -> np.ndarray:
    """Bessel functions ``J_n(alpha)`` for every order at once.

    Only non-negative orders are evaluated; the rest follow from
    ``J_{-n} = (-1)**n J_n``.

    Args:
        alpha (ndarray): Drive levels, any shape.
        num_b (int): Largest order to return.  Orders run from ``-num_b-1``
            to ``num_b+1`` so that neighbouring orders are available for the
            AC current and the derivative.

    Returns:
        ndarray: Shape ``(2*(num_b+1) + 1,) + alpha.shape``, indexed so that
        element ``[n + num_b + 1]`` is ``J_n(alpha)``.

    """
    alpha = np.asarray(alpha, dtype=float)
    top = num_b + 1
    orders = np.arange(0, top + 1)
    shape = (1,) * alpha.ndim
    positive = jv(orders.reshape((-1, *shape)), alpha[None, ...])

    out = np.empty((2 * top + 1, *alpha.shape), dtype=float)
    out[top:] = positive
    signs = (-1.0) ** orders[1:]
    out[:top] = (signs.reshape((-1, *shape)) * positive[1:])[::-1]
    return out


def _response_ladder(resp, vb: np.ndarray, vph: float, num_b: int):
    """Interpolate the response function at every photon-step offset.

    Args:
        resp (qpmix.respfn.RespFn): The response function.
        vb (ndarray): Normalized bias voltages.
        vph (float): Normalized photon voltage.
        num_b (int): Summation limit.

    Returns:
        tuple: ``(idc, ikk)``, each of shape ``(2*num_b + 1,) + vb.shape``.

    """
    vb = np.asarray(vb, dtype=float)
    offsets = np.arange(-num_b, num_b + 1, dtype=float)
    shifted = vb[None, ...] + offsets.reshape((-1, *((1,) * vb.ndim))) * vph
    table = resp(shifted)
    return table.imag, table.real


def pumped_iv_curve(
    resp, vb: np.ndarray, vph: float, alpha, num_b: int = 20
) -> np.ndarray:
    """Pumped DC I-V curve from Tucker theory.

    ``Idc(V0) = sum_n J_n(alpha)**2 * Idc0(V0 + n*vph)``.

    Args:
        resp (qpmix.respfn.RespFn): The response function.
        vb (ndarray): Normalized bias voltages.
        vph (float): Normalized photon voltage.
        alpha (float or ndarray): Drive level, scalar or one per bias point.
        num_b (int, optional): Summation limit.  Default is 20.

    Returns:
        ndarray: The pumped DC tunneling current, shaped like ``vb``.

    """
    vb = np.asarray(vb, dtype=float)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), vb.shape)
    idc, _ = _response_ladder(resp, vb, vph, num_b)
    jn = bessel_ladder(alpha, num_b)[1:-1]
    return np.einsum("n...,n...->...", jn**2, idc)


def ac_current(resp, vb: np.ndarray, vph: float, alpha, num_b: int = 20) -> np.ndarray:
    """AC tunneling current at the drive frequency, from Tucker theory.

    Args:
        resp (qpmix.respfn.RespFn): The response function.
        vb (ndarray): Normalized bias voltages.
        vph (float): Normalized photon voltage.
        alpha (float or ndarray): Drive level, scalar or one per bias point.
        num_b (int, optional): Summation limit.  Default is 20.

    Returns:
        ndarray: The complex AC tunneling current, shaped like ``vb``.

    """
    vb = np.asarray(vb, dtype=float)
    alpha = np.broadcast_to(np.asarray(alpha, dtype=float), vb.shape)
    idc, ikk = _response_ladder(resp, vb, vph, num_b)

    ladder = bessel_ladder(alpha, num_b)
    jn = ladder[1:-1]
    j_minus = ladder[:-2]
    j_plus = ladder[2:]

    real = np.einsum("n...,n...->...", jn * (j_minus + j_plus), idc)
    imag = np.einsum("n...,n...->...", jn * (j_minus - j_plus), ikk)
    return real + 1j * imag


def _current_from_ladder(idc_ladder, alpha, num_b):
    """Pumped current from a pre-interpolated response ladder.

    Args:
        idc_ladder (ndarray): ``Idc0(V0 + n*vph)``, shape
            ``(2*num_b+1,) + vb.shape``.
        alpha (ndarray): Drive level at each bias point.
        num_b (int): Summation limit.

    Returns:
        ndarray: The pumped DC current.

    """
    jn = bessel_ladder(alpha, num_b)[1:-1]
    return np.einsum("n...,n...->...", jn**2, idc_ladder)


def _current_and_derivative(idc_ladder, alpha, num_b):
    """Pumped current and its derivative with respect to ``alpha``.

    Uses ``2 J_n' = J_{n-1} - J_{n+1}``, so
    ``d/dalpha sum_n J_n**2 I_n = sum_n J_n (J_{n-1} - J_{n+1}) I_n``.  The
    neighbouring orders are already in the ladder, so this is nearly free.

    Args:
        idc_ladder (ndarray): The response ladder.
        alpha (ndarray): Drive level at each bias point.
        num_b (int): Summation limit.

    Returns:
        tuple: ``(current, d current / d alpha)``.

    """
    ladder = bessel_ladder(alpha, num_b)
    jn = ladder[1:-1]
    dj = ladder[:-2] - ladder[2:]
    current = np.einsum("n...,n...->...", jn**2, idc_ladder)
    derivative = np.einsum("n...,n...->...", jn * dj, idc_ladder)
    return current, derivative


def recover_alpha(
    resp,
    vb: np.ndarray,
    idc_measured: np.ndarray,
    vph: float,
    alpha_max: float = 1.5,
    num_b: int = 20,
    alpha_cap: float | None = None,
    ngrid: int = 96,
    max_iter: int = 40,
    tol: float = 1e-12,
) -> np.ndarray:
    """Recover the drive level from a measured pumped I-V curve.

    Inverts ``Idc(alpha) = sum_n J_n(alpha)**2 Idc0(V0 + n*vph)`` at every
    bias point.

    Note:
        The pumped I-V curve is only monotonic in ``alpha`` on the rising
        part of a photon step; above the gap, and past the first Bessel
        maximum, it is not.  So a bracket cannot be assumed — this function
        first scans a coarse grid of drive levels to locate the *first*
        upward crossing at each bias point, and only then refines.  Where no
        crossing exists the grid point closest to the measurement is
        returned, which keeps the result bounded instead of letting Newton
        run away.

        The response function is interpolated once, up front; the grid scan
        and the refinement then cost only Bessel evaluations.

    Args:
        resp (qpmix.respfn.RespFn): The response function.
        vb (ndarray): Normalized bias voltages.
        idc_measured (ndarray): Measured pumped DC current, normalized.
        vph (float): Normalized photon voltage.
        alpha_max (float, optional): Expected drive level; sets the default
            search cap.  Default is 1.5.
        num_b (int, optional): Summation limit.  Default is 20.
        alpha_cap (float, optional): Largest drive level to consider.
            Default is ``4 * alpha_max``.
        ngrid (int, optional): Points in the bracketing scan.  Default is
            96.
        max_iter (int, optional): Maximum refinement iterations.  Default is
            40.
        tol (float, optional): Convergence tolerance on the residual.
            Default is 1e-12.

    Returns:
        ndarray: The drive level at each bias point, in ``[0, alpha_cap]``.

    Raises:
        ValueError: If the bias and current arrays have different shapes.

    """
    vb = np.asarray(vb, dtype=float)
    measured = np.asarray(idc_measured, dtype=float)
    if vb.shape != measured.shape:
        raise ValueError("vb and idc_measured must have the same shape.")

    cap = float(alpha_cap) if alpha_cap is not None else 4.0 * max(alpha_max, 1e-6)
    idc_ladder, _ = _response_ladder(resp, vb, vph, num_b)

    # Coarse scan: current at every grid drive level, for every bias point.
    # The grid levels do not vary from bias point to bias point, so the
    # Bessel functions are evaluated once for the whole grid and contracted
    # against the response ladder in a single einsum -- rather than once per
    # grid point per bias point.
    grid = np.linspace(0.0, cap, ngrid)
    jn_grid = bessel_ladder(grid, num_b)[1:-1]  # (2*num_b+1, ngrid)
    flat = idc_ladder.reshape(idc_ladder.shape[0], -1)
    scan = np.einsum("ng,nv->gv", jn_grid**2, flat).reshape((ngrid, *vb.shape))

    residual = scan - measured[None, ...]
    rising = np.diff(scan, axis=0) > 0
    crosses = (residual[:-1] <= 0) & (residual[1:] > 0) & rising
    has_cross = crosses.any(axis=0)
    first = np.argmax(crosses, axis=0)

    step = grid[1] - grid[0]
    lo = np.where(has_cross, grid[first], 0.0)
    hi = np.where(has_cross, grid[first] + step, cap)
    # No upward crossing: fall back to the closest point on the scan.
    closest = grid[np.abs(residual).argmin(axis=0)]

    alpha = 0.5 * (lo + hi)
    for _ in range(max_iter):
        current, derivative = _current_and_derivative(idc_ladder, alpha, num_b)
        err = current - measured
        if np.abs(err[has_cross]).max(initial=0.0) < tol:
            break
        lo = np.where(err < 0, np.maximum(lo, alpha), lo)
        hi = np.where(err > 0, np.minimum(hi, alpha), hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            candidate = alpha - np.where(derivative != 0, err / derivative, np.inf)
        outside = ~np.isfinite(candidate) | (candidate <= lo) | (candidate >= hi)
        alpha = np.where(outside, 0.5 * (lo + hi), candidate)

    alpha = np.where(has_cross, alpha, closest)
    return np.clip(alpha, 0.0, cap)
