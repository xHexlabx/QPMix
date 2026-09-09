"""Numerical differentiation helpers.

Examples:

    >>> import numpy as np
    >>> from qpmix.mathfn.misc import slope
    >>> x = np.linspace(0, 1, 101)
    >>> bool(np.allclose(slope(x, 3 * x + 1), 3.0))
    True

"""

from __future__ import annotations

import numpy as np

__all__ = ["slope", "slope_span_n"]


def slope(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Take a centred derivative, ``dy/dx``.

    The result has the same number of points as the input; the two end
    points use one-sided differences.

    Args:
        x (ndarray): Independent variable.
        y (ndarray): Dependent variable.

    Returns:
        ndarray: The derivative.

    Raises:
        ValueError: If the inputs have fewer than two points or mismatched
            shapes.

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if x.size < 2:
        raise ValueError("Need at least 2 points to take a derivative.")

    der = np.empty(x.size, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if x.size > 2:
            der[1:-1] = (y[2:] - y[:-2]) / (x[2:] - x[:-2])
        der[0] = (y[1] - y[0]) / (x[1] - x[0])
        der[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return der


def slope_span_n(
    x: np.ndarray,
    y: np.ndarray,
    n: int = 11,
    nozeros: bool = True,
) -> np.ndarray:
    """Take a centred derivative over a span of ``n`` points.

    Averaging over a wider span suppresses noise, which matters when the
    derivative of measured I-V data is used as a divisor.

    Near the edges the span shrinks symmetrically until it fits, so the
    derivative stays centred everywhere it can.

    Args:
        x (ndarray): Independent variable.
        y (ndarray): Dependent variable.
        n (int, optional): Span in points.  Must be odd.  Default is 11.
        nozeros (bool, optional): Replace exact zeros in the result with a
            small number, so the output is safe to divide by.  Default is
            True.

    Returns:
        ndarray: The derivative.

    Raises:
        ValueError: If ``n`` is not a positive odd integer.

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    if n < 1 or n % 2 == 0:
        raise ValueError("Span n must be a positive odd integer.")

    npts = x.size
    half = (n - 1) // 2
    half = min(half, (npts - 1) // 2)

    der = np.empty(npts, dtype=float)
    idx = np.arange(npts)
    # Symmetric half-width, shrinking towards the edges: this is a single
    # vectorised gather instead of the per-edge Python loop it replaces.
    width = np.minimum(half, np.minimum(idx, npts - 1 - idx))
    width = np.maximum(width, 1) if npts > 1 else width
    lo = np.maximum(idx - width, 0)
    hi = np.minimum(idx + width, npts - 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        der[:] = (y[hi] - y[lo]) / (x[hi] - x[lo])

    if nozeros:
        der[der == 0] = 1e-10
    return der
