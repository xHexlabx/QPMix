"""Fused construction of the interpolated response-function matrix.

:func:`qpmix.qtcurrent.qtcurrent` needs the response function evaluated at
every photon-step-shifted bias voltage,

    ``R[k, l, ..., i] = resp(vb[i] + k*f1 + l*f2 + ...)``

which is an array of ``(2*num_b+1)**num_f * npts`` complex values.  Upstream
QMix builds this in two passes: first it materialises the full array of
*voltages* (a float array the same size as the result, plus several
broadcasting temporaries), then it interpolates it.  For three tones with
``num_b=15`` that is roughly 240 MB of temporaries written and read back
before any useful work happens.

The kernels here fuse the two passes: the shifted voltage is computed in a
register and immediately interpolated, so nothing but the result is ever
written to memory.  Combined with the O(1) table lookup from
:mod:`qpmix.interp`, this turns the single most expensive step in a
multi-tone simulation into a streaming, cache-friendly, thread-parallel
loop.

The ``k`` index follows NumPy's FFT ordering — ``0, 1, ..., num_b,
-num_b, ..., -1`` — which is what lets the rest of the package index the
arrays with signed offsets and have Python's negative indexing do the
wrap-around for free.
"""

from __future__ import annotations

import numpy as np

from qpmix._backend import JIT_ENABLED, njit, prange
from qpmix.interp import interp_point

__all__ = ["build_resp_matrix"]


@njit(parallel=True)
def _respmat_1(vb, freq, nb1, v0, inv_dv, table, out):  # pragma: no cover
    npts = vb.shape[0]
    n1 = 2 * nb1 + 1
    f1 = freq[1]
    for j in prange(n1):
        k = j if j <= nb1 else j - n1
        shift = k * f1
        for i in range(npts):
            out[j, i] = interp_point(vb[i] + shift, v0, inv_dv, table)


@njit(parallel=True)
def _respmat_2(vb, freq, nb1, nb2, v0, inv_dv, table, out):  # pragma: no cover
    npts = vb.shape[0]
    n1 = 2 * nb1 + 1
    n2 = 2 * nb2 + 1
    f1 = freq[1]
    f2 = freq[2]
    for j in prange(n1):
        k = j if j <= nb1 else j - n1
        s1 = k * f1
        for m in range(n2):
            l = m if m <= nb2 else m - n2
            shift = s1 + l * f2
            for i in range(npts):
                out[j, m, i] = interp_point(vb[i] + shift, v0, inv_dv, table)


@njit(parallel=True)
def _respmat_3(vb, freq, nb1, nb2, nb3, v0, inv_dv, table, out):  # pragma: no cover
    npts = vb.shape[0]
    n1 = 2 * nb1 + 1
    n2 = 2 * nb2 + 1
    n3 = 2 * nb3 + 1
    f1 = freq[1]
    f2 = freq[2]
    f3 = freq[3]
    for j in prange(n1):
        k = j if j <= nb1 else j - n1
        s1 = k * f1
        for m in range(n2):
            l = m if m <= nb2 else m - n2
            s2 = s1 + l * f2
            for q in range(n3):
                mm = q if q <= nb3 else q - n3
                shift = s2 + mm * f3
                for i in range(npts):
                    out[j, m, q, i] = interp_point(vb[i] + shift, v0, inv_dv, table)


@njit(parallel=True)
def _respmat_4(
    vb, freq, nb1, nb2, nb3, nb4, v0, inv_dv, table, out
):  # pragma: no cover
    npts = vb.shape[0]
    n1 = 2 * nb1 + 1
    n2 = 2 * nb2 + 1
    n3 = 2 * nb3 + 1
    n4 = 2 * nb4 + 1
    f1 = freq[1]
    f2 = freq[2]
    f3 = freq[3]
    f4 = freq[4]
    for j in prange(n1):
        k = j if j <= nb1 else j - n1
        s1 = k * f1
        for m in range(n2):
            l = m if m <= nb2 else m - n2
            s2 = s1 + l * f2
            for q in range(n3):
                mm = q if q <= nb3 else q - n3
                s3 = s2 + mm * f3
                for r in range(n4):
                    nn = r if r <= nb4 else r - n4
                    shift = s3 + nn * f4
                    for i in range(npts):
                        out[j, m, q, r, i] = interp_point(
                            vb[i] + shift, v0, inv_dv, table
                        )


_KERNELS = {1: _respmat_1, 2: _respmat_2, 3: _respmat_3, 4: _respmat_4}


def _fft_offsets(nb: int) -> np.ndarray:
    """Signed offsets in NumPy FFT order: ``0, 1, ..., nb, -nb, ..., -1``.

    Args:
        nb (int): Summation limit.

    Returns:
        ndarray: The offsets, length ``2 * nb + 1``.

    """
    return np.r_[np.arange(0, nb + 1), np.arange(-nb, 0)]


def build_resp_matrix(
    resp, vb: np.ndarray, freq: np.ndarray, nb_list: tuple[int, ...]
) -> np.ndarray:
    """Evaluate the response function at every photon-step-shifted voltage.

    Args:
        resp (qpmix.respfn.RespFn): The response function.
        vb (ndarray): Normalized DC bias voltages.
        freq (ndarray): Normalized frequencies, with ``freq[0]`` unused.
        nb_list (tuple): Summation limit for each tone.

    Returns:
        ndarray: Complex array of shape
        ``(2*nb1+1, ..., 2*nbF+1, len(vb))``.

    Raises:
        ValueError: If the number of tones is not between 1 and 4.

    """
    num_f = len(nb_list)
    if not 1 <= num_f <= 4:
        raise ValueError("num_f must be 1, 2, 3 or 4.")

    vb = np.ascontiguousarray(vb, dtype=float)
    freq = np.ascontiguousarray(freq, dtype=float)
    shape = (*tuple(2 * nb + 1 for nb in nb_list), vb.size)
    table = resp.table

    if JIT_ENABLED:
        out = np.empty(shape, dtype=np.complex128)
        _KERNELS[num_f](vb, freq, *nb_list, table.v0, 1.0 / table.dv, table.values, out)
        return out

    # NumPy fallback: same result, but the shifted voltages have to be
    # materialised because there is no way to fuse the loops without a JIT.
    voltage = vb.reshape((1,) * num_f + (-1,)).copy()
    for axis, nb in enumerate(nb_list):
        offsets = _fft_offsets(nb).astype(float) * freq[axis + 1]
        bshape = [1] * (num_f + 1)
        bshape[axis] = offsets.size
        voltage = voltage + offsets.reshape(bshape)
    return np.ascontiguousarray(table(voltage))
