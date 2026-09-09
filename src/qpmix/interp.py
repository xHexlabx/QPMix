"""Fast interpolation on a uniform grid.

Why this module exists
----------------------

Evaluating the SIS response function is the single hottest operation in a
multi-tone simulation: :func:`qpmix.qtcurrent.qtcurrent` needs it at
``(2*num_b+1)**num_f * npts`` voltages, which reaches tens of millions of
points for a three-tone problem.

Upstream QMix evaluates a cubic spline that is fitted to a *non-uniformly*
sampled curve (~100 knots, packed around the gap).  Every single evaluation
therefore costs a binary search through the knot vector — a data-dependent
branch that defeats the branch predictor, cannot be vectorised, and pays
FITPACK call overhead on top.

:class:`UniformInterpolator` instead pre-samples the curve onto a *uniform*
grid once, at build time, and then evaluates it with pure index arithmetic:

    ``i = floor((v - v0) / dv)``

That is O(1) with no branches on the data, it vectorises, and — because
:func:`qpmix.qtcurrent.qtcurrent` sweeps the bias voltage monotonically —
the table is walked sequentially, so the hardware prefetcher does the rest.
The four-tap Catmull-Rom stencil is fourth-order accurate, so a dense table
is *more* accurate than the ~100-knot spline it replaces while also being
much faster.

Examples:

    >>> import numpy as np
    >>> from qpmix.interp import UniformInterpolator
    >>> v = np.linspace(-2.0, 2.0, 4001)
    >>> f = UniformInterpolator.from_samples(v, np.sin(v))
    >>> bool(np.allclose(f(np.array([0.3, -1.1])), np.sin([0.3, -1.1]), atol=1e-9))
    True

"""

from __future__ import annotations

import numpy as np

from qpmix._backend import JIT_ENABLED, njit, prange

__all__ = ["UniformInterpolator", "interp_point"]

#: Arrays smaller than this are evaluated by the serial kernel; spinning up
#: threads is not worth it below roughly this many points.
PARALLEL_THRESHOLD = 32_768

#: Points per chunk in the NumPy fallback.  Keeps the handful of temporaries
#: the vectorised form needs inside cache instead of streaming through RAM.
NUMPY_CHUNK = 1 << 18


def _interp_point(x, v0, inv_dv, y):
    """Interpolate a single point (see :class:`UniformInterpolator`).

    Kept as a standalone scalar function so the JIT can inline it into the
    fused response-matrix kernels in :mod:`qpmix.qtcurrent`, which build
    their sample positions on the fly instead of materialising them.

    Args:
        x (float): Position.
        v0 (float): Position of the first grid node.
        inv_dv (float): Reciprocal grid spacing.
        y (ndarray): The tabulated values.

    Returns:
        The interpolated value.

    """
    n = y.shape[0]
    s = (x - v0) * inv_dv
    i = int(np.floor(s))
    if i < 1:
        return y[0] + s * (y[1] - y[0])
    if i > n - 3:
        return y[n - 1] + (s - (n - 1)) * (y[n - 1] - y[n - 2])
    t = s - i
    y0 = y[i - 1]
    y1 = y[i]
    y2 = y[i + 1]
    y3 = y[i + 2]
    return y1 + 0.5 * t * (
        (y2 - y0)
        + t * ((2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) + t * (3.0 * (y1 - y2) + y3 - y0))
    )


interp_point = njit(_interp_point)


def _eval_body(x, v0, inv_dv, y, out, lo, hi):
    """Shared Catmull-Rom body (see :func:`_eval_serial`)."""
    n = y.shape[0]
    for j in range(lo, hi):
        s = (x[j] - v0) * inv_dv
        i = int(np.floor(s))
        if i < 1:
            # Below the table: the response function is linear out here.
            out[j] = y[0] + s * (y[1] - y[0])
        elif i > n - 3:
            out[j] = y[n - 1] + (s - (n - 1)) * (y[n - 1] - y[n - 2])
        else:
            t = s - i
            y0 = y[i - 1]
            y1 = y[i]
            y2 = y[i + 1]
            y3 = y[i + 2]
            out[j] = y1 + 0.5 * t * (
                (y2 - y0)
                + t
                * (
                    (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3)
                    + t * (3.0 * (y1 - y2) + y3 - y0)
                )
            )


_eval_body_jit = njit(_eval_body)


@njit
def _eval_serial(x, v0, inv_dv, y, out):  # pragma: no cover - trivial wrapper
    _eval_body_jit(x, v0, inv_dv, y, out, 0, x.shape[0])


@njit(parallel=True)
def _eval_parallel(x, v0, inv_dv, y, out):  # pragma: no cover - trivial wrapper
    npts = x.shape[0]
    nblk = 64
    blk = (npts + nblk - 1) // nblk
    for b in prange(nblk):
        lo = b * blk
        hi = min(lo + blk, npts)
        if lo < hi:
            _eval_body_jit(x, v0, inv_dv, y, out, lo, hi)


def _eval_numpy(x, v0, inv_dv, y, out):
    """Vectorised NumPy evaluation, used when numba is unavailable.

    Same Catmull-Rom stencil as the JIT kernel, expressed as array
    operations.  Evaluated in chunks so the several temporaries the
    vectorised form needs stay cache-resident rather than streaming
    through main memory.

    Args:
        x (ndarray): Positions, 1-D.
        v0 (float): Position of the first grid node.
        inv_dv (float): Reciprocal grid spacing.
        y (ndarray): The tabulated values.
        out (ndarray): Destination, same shape as ``x``.

    """
    n = y.shape[0]
    for start in range(0, x.size, NUMPY_CHUNK):
        xs = x[start : start + NUMPY_CHUNK]
        s = (xs - v0) * inv_dv
        i = np.floor(s).astype(np.intp)
        ic = np.clip(i, 1, n - 3)
        t = s - ic
        y0 = y[ic - 1]
        y1 = y[ic]
        y2 = y[ic + 1]
        y3 = y[ic + 2]
        val = y1 + 0.5 * t * (
            (y2 - y0)
            + t
            * ((2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) + t * (3.0 * (y1 - y2) + y3 - y0))
        )
        low = i < 1
        if low.any():
            val[low] = y[0] + s[low] * (y[1] - y[0])
        high = i > n - 3
        if high.any():
            val[high] = y[n - 1] + (s[high] - (n - 1)) * (y[n - 1] - y[n - 2])
        out[start : start + xs.size] = val


class UniformInterpolator:
    """Cubic interpolation of tabulated data on a uniform grid.

    Args:
        v0 (float): Position of the first grid node.
        dv (float): Grid spacing.  Must be positive.
        values (ndarray): Tabulated values, 1-D.  May be real or complex.

    Attributes:
        v0 (float): Position of the first grid node.
        dv (float): Grid spacing.
        values (ndarray): The tabulated values.

    Raises:
        ValueError: If ``dv`` is not positive or ``values`` has fewer than
            four points.

    """

    __slots__ = ("_inv_dv", "dv", "v0", "values")

    def __init__(self, v0: float, dv: float, values: np.ndarray) -> None:
        values = np.ascontiguousarray(values)
        if values.ndim != 1:
            raise ValueError("values must be 1-D.")
        if values.size < 4:
            raise ValueError("Need at least 4 points for cubic interpolation.")
        if not dv > 0:
            raise ValueError("Grid spacing dv must be positive.")
        self.v0 = float(v0)
        self.dv = float(dv)
        self._inv_dv = 1.0 / float(dv)
        self.values = values

    @classmethod
    def from_samples(cls, v: np.ndarray, y: np.ndarray) -> UniformInterpolator:
        """Build an interpolator from data that is already uniformly spaced.

        Args:
            v (ndarray): Uniformly spaced sample positions, increasing.
            y (ndarray): Sample values.

        Returns:
            UniformInterpolator: The interpolator.

        Raises:
            ValueError: If ``v`` is not uniformly spaced.

        """
        v = np.asarray(v, dtype=float)
        step = np.diff(v)
        if step.size == 0 or not np.allclose(step, step[0], rtol=1e-9, atol=0.0):
            raise ValueError("Sample positions must be uniformly spaced.")
        return cls(v[0], float(step[0]), np.asarray(y))

    @property
    def vmin(self) -> float:
        """Lower edge of the tabulated range."""
        return self.v0

    @property
    def vmax(self) -> float:
        """Upper edge of the tabulated range."""
        return self.v0 + self.dv * (self.values.size - 1)

    def __len__(self) -> int:
        return int(self.values.size)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"UniformInterpolator(npts={len(self)}, "
            f"range=[{self.vmin:.3f}, {self.vmax:.3f}], "
            f"dtype={self.values.dtype})"
        )

    def __call__(self, v: np.ndarray | float) -> np.ndarray | complex | float:
        """Interpolate at the given positions.

        Args:
            v (ndarray or float): Position(s) at which to interpolate.  Any
                shape; positions outside the tabulated range are linearly
                extrapolated from the end nodes.

        Returns:
            ndarray or scalar: Interpolated values, shaped like ``v``.

        """
        scalar = np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 0)
        x = np.ascontiguousarray(v, dtype=float).ravel()
        out = np.empty(x.shape, dtype=self.values.dtype)
        self.eval_into(x, out)
        if scalar:
            return out[0]
        return out.reshape(np.shape(v))

    def eval_into(self, x: np.ndarray, out: np.ndarray) -> np.ndarray:
        """Interpolate into a pre-allocated output buffer.

        Avoids an allocation in tight loops.  Both arrays must be 1-D,
        contiguous and the same length.

        Args:
            x (ndarray): Positions, 1-D contiguous float64.
            out (ndarray): Destination, 1-D contiguous, same length as ``x``.

        Returns:
            ndarray: ``out``, for convenience.

        Raises:
            ValueError: If the shapes do not match.

        """
        if x.shape != out.shape:
            raise ValueError("x and out must have the same shape.")
        if x.size == 0:
            return out
        if not JIT_ENABLED:
            _eval_numpy(x, self.v0, self._inv_dv, self.values, out)
            return out
        kernel = _eval_parallel if x.size >= PARALLEL_THRESHOLD else _eval_serial
        kernel(x, self.v0, self._inv_dv, self.values, out)
        return out
