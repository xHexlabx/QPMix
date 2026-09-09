"""Inner kernels for the quasiparticle tunneling current.

These evaluate Eqn. 5.25 of Kittara's thesis, the correlation

    ``RS+/-(a,b,...) = sum_{k,l,...} C1[k] C1*[k+/-a] C2[l] C2*[l+/-b] ...
    R[k,l,...]``

for one index tuple.  There is one kernel per number of tones because the
loop nest has to be static for the JIT to unroll and vectorise it.

Two things differ from the equivalent QMix kernels:

*Cache blocking.*  The bias-voltage axis is walked in blocks of
:data:`BLOCK` points rather than in one sweep.  The coefficient slices
``C[k, block]`` that the inner loops re-read for every ``(k, l, ...)``
combination then stay resident in L2, while the much larger response matrix
streams through the cache exactly once.  The block index is also the unit
of thread-level parallelism, so each thread owns a disjoint slice of the
output and no synchronisation is needed.

*Optional single sum.*  When the requested tuple is its own negation — the
DC term, ``a = b = ... = 0`` — ``RS-`` is identically equal to ``RS+`` and
the kernel is told to skip it.
"""

from __future__ import annotations

import numpy as np

from qpmix._backend import njit, prange

__all__ = ["BLOCK", "coeff_1", "coeff_2", "coeff_3", "coeff_4"]

#: Bias points per cache block.  At 16 bytes per complex value this keeps
#: the re-read coefficient slices inside a few hundred kilobytes.
BLOCK = 128


@njit(parallel=True)
def coeff_1(a, ccc1, resp, nb1, rs_p, rs_m, want_m, block):  # pragma: no cover
    npts = rs_p.shape[0]
    nblk = (npts + block - 1) // block
    for bi in prange(nblk):
        i0 = bi * block
        i1 = min(i0 + block, npts)
        for k in range(-nb1, nb1 + 1):
            kp = k + a
            km = k - a
            do_p = -nb1 <= kp <= nb1
            do_m = want_m and -nb1 <= km <= nb1
            if do_p:
                for i in range(i0, i1):
                    rs_p[i] += ccc1[k, i] * _conj(ccc1[kp, i]) * resp[k, i]
            if do_m:
                for i in range(i0, i1):
                    rs_m[i] += ccc1[k, i] * _conj(ccc1[km, i]) * resp[k, i]


@njit(parallel=True)
def coeff_2(
    a, b, ccc1, ccc2, resp, nb1, nb2, rs_p, rs_m, want_m, block
):  # pragma: no cover
    npts = rs_p.shape[0]
    nblk = (npts + block - 1) // block
    for bi in prange(nblk):
        i0 = bi * block
        i1 = min(i0 + block, npts)
        width = i1 - i0
        t1_p = _cbuf(width)
        t1_m = _cbuf(width)
        for k in range(-nb1, nb1 + 1):
            kp = k + a
            km = k - a
            do_kp = -nb1 <= kp <= nb1
            do_km = want_m and -nb1 <= km <= nb1
            if not (do_kp or do_km):
                continue
            # Hoist the tone-1 product: invariant inside the l loop.
            if do_kp:
                for i in range(width):
                    t1_p[i] = ccc1[k, i0 + i] * _conj(ccc1[kp, i0 + i])
            if do_km:
                for i in range(width):
                    t1_m[i] = ccc1[k, i0 + i] * _conj(ccc1[km, i0 + i])
            for l in range(-nb2, nb2 + 1):
                lp = l + b
                lm = l - b
                if do_kp and -nb2 <= lp <= nb2:
                    for i in range(width):
                        j = i0 + i
                        rs_p[j] += (
                            t1_p[i] * ccc2[l, j] * _conj(ccc2[lp, j]) * resp[k, l, j]
                        )
                if do_km and -nb2 <= lm <= nb2:
                    for i in range(width):
                        j = i0 + i
                        rs_m[j] += (
                            t1_m[i] * ccc2[l, j] * _conj(ccc2[lm, j]) * resp[k, l, j]
                        )


@njit(parallel=True)
def coeff_3(
    a, b, c, ccc1, ccc2, ccc3, resp, nb1, nb2, nb3, rs_p, rs_m, want_m, block
):  # pragma: no cover
    npts = rs_p.shape[0]
    nblk = (npts + block - 1) // block
    for bi in prange(nblk):
        i0 = bi * block
        i1 = min(i0 + block, npts)
        width = i1 - i0
        t1_p = _cbuf(width)
        t1_m = _cbuf(width)
        t2_p = _cbuf(width)
        t2_m = _cbuf(width)
        for k in range(-nb1, nb1 + 1):
            kp = k + a
            km = k - a
            do_kp = -nb1 <= kp <= nb1
            do_km = want_m and -nb1 <= km <= nb1
            if not (do_kp or do_km):
                continue
            if do_kp:
                for i in range(width):
                    t1_p[i] = ccc1[k, i0 + i] * _conj(ccc1[kp, i0 + i])
            if do_km:
                for i in range(width):
                    t1_m[i] = ccc1[k, i0 + i] * _conj(ccc1[km, i0 + i])
            for l in range(-nb2, nb2 + 1):
                lp = l + b
                lm = l - b
                do_lp = do_kp and -nb2 <= lp <= nb2
                do_lm = do_km and -nb2 <= lm <= nb2
                if not (do_lp or do_lm):
                    continue
                if do_lp:
                    for i in range(width):
                        j = i0 + i
                        t2_p[i] = t1_p[i] * ccc2[l, j] * _conj(ccc2[lp, j])
                if do_lm:
                    for i in range(width):
                        j = i0 + i
                        t2_m[i] = t1_m[i] * ccc2[l, j] * _conj(ccc2[lm, j])
                for m in range(-nb3, nb3 + 1):
                    mp = m + c
                    mm = m - c
                    if do_lp and -nb3 <= mp <= nb3:
                        for i in range(width):
                            j = i0 + i
                            rs_p[j] += (
                                t2_p[i]
                                * ccc3[m, j]
                                * _conj(ccc3[mp, j])
                                * resp[k, l, m, j]
                            )
                    if do_lm and -nb3 <= mm <= nb3:
                        for i in range(width):
                            j = i0 + i
                            rs_m[j] += (
                                t2_m[i]
                                * ccc3[m, j]
                                * _conj(ccc3[mm, j])
                                * resp[k, l, m, j]
                            )


@njit(parallel=True)
def coeff_4(
    a,
    b,
    c,
    d,
    ccc1,
    ccc2,
    ccc3,
    ccc4,
    resp,
    nb1,
    nb2,
    nb3,
    nb4,
    rs_p,
    rs_m,
    want_m,
    block,
):  # pragma: no cover
    npts = rs_p.shape[0]
    nblk = (npts + block - 1) // block
    for bi in prange(nblk):
        i0 = bi * block
        i1 = min(i0 + block, npts)
        width = i1 - i0
        t1_p = _cbuf(width)
        t1_m = _cbuf(width)
        t2_p = _cbuf(width)
        t2_m = _cbuf(width)
        t3_p = _cbuf(width)
        t3_m = _cbuf(width)
        for k in range(-nb1, nb1 + 1):
            kp = k + a
            km = k - a
            do_kp = -nb1 <= kp <= nb1
            do_km = want_m and -nb1 <= km <= nb1
            if not (do_kp or do_km):
                continue
            if do_kp:
                for i in range(width):
                    t1_p[i] = ccc1[k, i0 + i] * _conj(ccc1[kp, i0 + i])
            if do_km:
                for i in range(width):
                    t1_m[i] = ccc1[k, i0 + i] * _conj(ccc1[km, i0 + i])
            for l in range(-nb2, nb2 + 1):
                lp = l + b
                lm = l - b
                do_lp = do_kp and -nb2 <= lp <= nb2
                do_lm = do_km and -nb2 <= lm <= nb2
                if not (do_lp or do_lm):
                    continue
                if do_lp:
                    for i in range(width):
                        j = i0 + i
                        t2_p[i] = t1_p[i] * ccc2[l, j] * _conj(ccc2[lp, j])
                if do_lm:
                    for i in range(width):
                        j = i0 + i
                        t2_m[i] = t1_m[i] * ccc2[l, j] * _conj(ccc2[lm, j])
                for m in range(-nb3, nb3 + 1):
                    mp = m + c
                    mm = m - c
                    do_mp = do_lp and -nb3 <= mp <= nb3
                    do_mm = do_lm and -nb3 <= mm <= nb3
                    if not (do_mp or do_mm):
                        continue
                    if do_mp:
                        for i in range(width):
                            j = i0 + i
                            t3_p[i] = t2_p[i] * ccc3[m, j] * _conj(ccc3[mp, j])
                    if do_mm:
                        for i in range(width):
                            j = i0 + i
                            t3_m[i] = t2_m[i] * ccc3[m, j] * _conj(ccc3[mm, j])
                    for n in range(-nb4, nb4 + 1):
                        np_ = n + d
                        nm = n - d
                        if do_mp and -nb4 <= np_ <= nb4:
                            for i in range(width):
                                j = i0 + i
                                rs_p[j] += (
                                    t3_p[i]
                                    * ccc4[n, j]
                                    * _conj(ccc4[np_, j])
                                    * resp[k, l, m, n, j]
                                )
                        if do_mm and -nb4 <= nm <= nb4:
                            for i in range(width):
                                j = i0 + i
                                rs_m[j] += (
                                    t3_m[i]
                                    * ccc4[n, j]
                                    * _conj(ccc4[nm, j])
                                    * resp[k, l, m, n, j]
                                )


@njit
def _conj(z):  # pragma: no cover - inlined by the JIT
    """Complex conjugate of a scalar."""
    return z.real - 1j * z.imag


@njit
def _cbuf(n):  # pragma: no cover - inlined by the JIT
    """Allocate a thread-local complex scratch buffer of length ``n``."""
    return np.empty(n, dtype=np.complex128)
