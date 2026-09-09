"""Quasiparticle tunneling currents through an SIS junction.

Given the voltage applied across the junction, this module calculates the
resulting tunneling current at any requested frequency, using multi-tone
spectral domain analysis (MTSDA).  The formulation follows P. Kittara's 2002
DPhil thesis; inline comments name the specific equations.

Everything here is normalized: voltages to the gap voltage, frequencies to
the gap frequency, currents to the gap current.

The calculation has three stages, and QPMix changes the algorithm used for
each of them:

=========================  ===================  =============================
Stage                      QMix                 QPMix
=========================  ===================  =============================
Phase-factor coefficients  Bessel + convolve    Single FFT
                           O(num_p*num_b^2)     O(M log M)
Response-function matrix   Materialise voltage  Fused, in-register, with an
                           array, then a        O(1) uniform-table lookup
                           spline binary search
Current summation          Nested loops per     Same loops, cache-blocked and
                           output frequency     thread-parallel; optionally
                                                one FFT correlation for all
                                                output frequencies at once
=========================  ===================  =============================

See ``docs/PERFORMANCE.md`` for measured numbers and the complexity
analysis.

Examples:

    >>> import numpy as np
    >>> from qpmix.circuit import EmbeddingCircuit
    >>> from qpmix.respfn import RespFnPolynomial
    >>> cct = EmbeddingCircuit(1, 1, vb_npts=21)
    >>> cct.freq[1] = 0.3
    >>> resp = RespFnPolynomial(50, verbose=False)
    >>> vj = cct.initialize_vj()
    >>> vj[1, 1, :] = 0.3
    >>> idc = qtcurrent(vj, cct, resp, 0.0, num_b=9, verbose=False)
    >>> idc.shape
    (21,)
    >>> bool(np.all(np.isreal(idc)))
    True

"""

from __future__ import annotations

import itertools
from timeit import default_timer as timer

import numpy as np
from scipy.fft import fftn, ifftn, next_fast_len

from qpmix import _kernels
from qpmix._backend import JIT_ENABLED
from qpmix._respmat import build_resp_matrix
from qpmix.phase_factor import _as_nb_tuple, calculate_phase_factor_coeff

__all__ = [
    "ROUND_FREQ",
    "calculate_phase_factor_coeff",
    "interpolate_respfn",
    "qtcurrent",
]

#: The multi-dimensional kernels cover up to this many tones.  Above it the
#: response matrix would not fit in memory, and ``qtcurrent`` switches to the
#: common-grid engine in :mod:`qpmix.multitone`.
MAX_DIRECT_TONES = 4

#: Frequencies are compared after rounding to this many decimal places.
#: Two intermodulation products land on the same output frequency only if
#: they agree to within this tolerance.
ROUND_FREQ = 4

#: Above this many bytes the FFT correlation path is rejected by ``auto``,
#: because its working set would not fit in memory comfortably.
FFT_MEMORY_BUDGET = 2 * 1024**3

#: Cost of the FFT correlation path, per ``V log2(V)`` where ``V`` is the
#: padded transform volume, in units of one direct inner-loop multiply-add.
#: A pure flop count would put this near 10, but the transforms are
#: memory-bandwidth bound rather than flop bound, so the effective figure is
#: much lower.  1.4 is calibrated against ``benchmarks/bench_method.py``:
#: across that sweep the chosen path is always within 15% of the faster one,
#: and it picks the true winner wherever the gap is larger than that.
FFT_COST_FACTOR = 1.4


def qtcurrent(
    vj: np.ndarray,
    cct,
    resp,
    freq_list,
    num_b: int | tuple[int, ...] = 15,
    verbose: bool = True,
    resp_matrix: np.ndarray | None = None,
    method: str = "auto",
    **grid_kwargs,
) -> np.ndarray:
    """Calculate the quasiparticle tunneling current.

    The current is returned for every frequency in ``freq_list``, each
    normalized to the gap frequency.  For example, to get the DC current
    and the AC current at 230 GHz, pass ``[0, 230e9 / fgap]``.

    Note:
        Up to four non-harmonically related tones are supported.

    Args:
        vj (ndarray): Voltage across the SIS junction, shape
            ``(num_f + 1, num_p + 1, npts)``, complex.
        cct (qpmix.circuit.EmbeddingCircuit): The embedding circuit.
        resp (qpmix.respfn.RespFn): The response function.
        freq_list (float or sequence): Frequencies to solve for, normalized
            to the gap frequency.
        num_b (int or tuple, optional): Summation limit for the phase-factor
            coefficients, one value or one per tone.  Default is 15.
        verbose (bool, optional): Print progress to the terminal.  Default
            is True.
        resp_matrix (ndarray, optional): A pre-computed response matrix from
            :func:`interpolate_respfn`.  Pass this when calling
            ``qtcurrent`` repeatedly with the same frequencies, as harmonic
            balance does.  Default is None.
        method (str, optional): ``"auto"``, ``"direct"``, ``"fft"`` or
            ``"grid"``.  The first three evaluate the current summation
            (Eqn. 5.25) with the multi-dimensional kernels and give
            identical answers; ``"grid"`` switches to the common-grid
            engine in :mod:`qpmix.multitone`, which is the only option
            above four tones.  ``"auto"`` uses the grid engine when there
            are more than four tones and picks between ``"direct"`` and
            ``"fft"`` otherwise.  Default is ``"auto"``.
        **grid_kwargs: Forwarded to
            :func:`qpmix.multitone.qtcurrent_grid` when the grid engine is
            used (``grid``, ``num_theta``).

    Returns:
        ndarray: The tunneling current.  Shape ``(len(freq_list), npts)`` if
        ``freq_list`` is a sequence, otherwise ``(npts,)`` — and real rather
        than complex when that single frequency is zero.

    Raises:
        ValueError: If any tone frequency is not positive, or if ``method``
            is not recognised.

    """
    num_f = cct.num_f
    num_p = cct.num_p
    npts = cct.vb_npts
    freq = cct.freq

    if freq[1:].min() <= 0.0:
        raise ValueError("All tone frequencies must be > 0.")

    if method == "grid" or (method == "auto" and num_f > MAX_DIRECT_TONES):
        from qpmix.multitone import qtcurrent_grid

        return qtcurrent_grid(
            vj,
            cct,
            resp,
            freq_list,
            num_b=num_b,
            verbose=verbose,
            resp_matrix=resp_matrix,
            **grid_kwargs,
        )
    if grid_kwargs:
        raise TypeError(
            f"Unexpected keyword argument(s) for method={method!r}: "
            f"{sorted(grid_kwargs)}"
        )
    if num_f > MAX_DIRECT_TONES:
        raise ValueError(
            f"method={method!r} supports at most {MAX_DIRECT_TONES} tones; "
            f"this circuit has {num_f}. Use method='auto' or 'grid' to run "
            f"it through qpmix.multitone."
        )

    freq_is_list = np.ndim(freq_list) > 0
    freq_out = np.atleast_1d(np.asarray(freq_list, dtype=float)).round(ROUND_FREQ)
    nb_list = _as_nb_tuple(num_b, num_f)

    if verbose:
        print("Calculating tunneling current...")
        print(f" - {num_f} tone(s)")
        print(f" - {num_p} harmonic(s)")
        start_time = timer()

    # Stage 1: phase-factor coefficients, C_k (Eqns. 5.7 and 5.12).
    ccc = calculate_phase_factor_coeff(vj, freq, num_f, num_p, nb_list)

    # Stage 2: the interpolated response function.
    if resp_matrix is None:
        resp_matrix = interpolate_respfn(cct, resp, nb_list)
    elif resp_matrix.ndim != num_f + 1 or resp_matrix.shape[-1] != npts:
        raise ValueError("resp_matrix has the wrong shape for this circuit.")

    # Stage 3: the current summation (Eqns. 5.25 and 5.26).
    tuples_for = _matching_tuples(freq_out, freq, num_f, num_p)
    chosen = _select_method(method, num_f, num_p, nb_list, npts, tuples_for)

    if chosen == "fft":
        rs_lookup = _correlate_fft(ccc, resp_matrix, nb_list, num_p, npts)
    else:
        rs_lookup = _correlate_direct(ccc, resp_matrix, nb_list, tuples_for, npts)

    current_out = np.zeros((freq_out.size, npts), dtype=complex)
    for idx, tuples in enumerate(tuples_for):
        at_dc = freq_out[idx] == 0.0
        for t in tuples:
            current_out[idx] += _assemble(t, rs_lookup, at_dc)

    if verbose:
        print(f" - method: {chosen}")
        print("Done.")
        print(f"Time: {timer() - start_time:.4f} s\n")

    if freq_is_list:
        return current_out
    if freq_out[0] == 0.0:
        return current_out[0].real
    return current_out[0]


def interpolate_respfn(
    cct, resp, num_b: int | tuple[int, ...], method: str = "auto", grid=None
) -> np.ndarray:
    """Interpolate the response function at every voltage ``qtcurrent`` needs.

    Call this once and pass the result to :func:`qtcurrent` as
    ``resp_matrix`` when running many simulations with the same input
    frequencies; it is the most expensive part of a single ``qtcurrent``
    call.

    Args:
        cct (qpmix.circuit.EmbeddingCircuit): The embedding circuit.
        resp (qpmix.respfn.RespFn): The response function.
        num_b (int or tuple): Summation limit for the phase-factor
            coefficients.
        method (str, optional): Which engine the matrix is for; must match
            what :func:`qtcurrent` will use.  Default is ``"auto"``.
        grid (qpmix.multitone.ToneGrid, optional): Reuse a grid instead of
            deriving one.  Only used by the grid engine.  Default is None.

    Returns:
        ndarray: The response matrix.  Shape
        ``(2*nb1+1, ..., 2*nbF+1, npts)`` for the multi-dimensional engine,
        or ``(2*num_k+1, npts)`` for the grid engine.

    """
    if method == "grid" or (method == "auto" and cct.num_f > MAX_DIRECT_TONES):
        from qpmix.multitone import ToneGrid, interpolate_respfn_grid

        if grid is None:
            grid = ToneGrid.from_circuit(cct, num_b=num_b)
        return interpolate_respfn_grid(cct, resp, grid)
    nb_list = _as_nb_tuple(num_b, cct.num_f)
    return build_resp_matrix(resp, cct.vb, cct.freq, nb_list)


# -- index bookkeeping -------------------------------------------------


def _matching_tuples(
    freq_out: np.ndarray, freq: np.ndarray, num_f: int, num_p: int
) -> list[list[tuple[int, ...]]]:
    """Find the index tuples that contribute to each output frequency.

    An index tuple ``(a, b, ...)`` contributes to output frequency
    ``f_out`` when ``a*f1 + b*f2 + ... == f_out``.

    Note:
        A tuple and its negation are never both returned for the same
        output frequency.  Eqn. 5.26 combines ``RS+(t)`` and ``RS-(t)``,
        and ``RS-(t) == RS+(-t)``, so the coefficient for ``t`` already
        carries the contribution of ``-t``.  The two can only collide at
        ``f_out == 0``, and only when the tones are commensurate enough
        that some non-zero tuple sums to zero frequency -- but there,
        enumerating both would double-count the DC current.

    Args:
        freq_out (ndarray): Requested output frequencies, already rounded.
        freq (ndarray): Tone frequencies, with ``freq[0]`` unused.
        num_f (int): Number of tones.
        num_p (int): Number of harmonics.

    Returns:
        list: One list of index tuples per output frequency.

    """
    span = range(num_p, -(num_p + 1), -1)
    out: list[list[tuple[int, ...]]] = [[] for _ in freq_out]
    seen: list[set[tuple[int, ...]]] = [set() for _ in freq_out]
    for t in itertools.product(span, repeat=num_f):
        f_t = round(float(np.dot(t, freq[1 : num_f + 1])), ROUND_FREQ)
        for idx, f_o in enumerate(freq_out):
            if f_t != f_o:
                continue
            if tuple(-x for x in t) in seen[idx]:
                continue
            seen[idx].add(t)
            out[idx].append(t)
    return out


def _needed_tuples(
    tuples_for: list[list[tuple[int, ...]]],
) -> list[tuple[int, ...]]:
    """Collect every index tuple whose correlation has to be evaluated.

    Deduplicates across output frequencies: a tuple that contributes to two
    different output frequencies is only computed once.

    Args:
        tuples_for (list): Output of :func:`_matching_tuples`.

    Returns:
        list: The unique index tuples, in a stable order.

    """
    seen: dict[tuple[int, ...], None] = {}
    for tuples in tuples_for:
        for t in tuples:
            seen.setdefault(t, None)
    return list(seen)


def _assemble(
    t: tuple[int, ...],
    rs: dict[tuple[int, ...], np.ndarray],
    at_dc: bool = False,
) -> np.ndarray:
    """Combine the two correlation sums into a current coefficient.

    Eqn. 5.26 in Kittara's thesis.

    Note:
        At zero output frequency the tunneling current is real by
        construction, so the quadrature term is dropped.  For the all-zero
        tuple it vanishes anyway; for a non-zero tuple that happens to sum
        to zero frequency it does not, and keeping it would leave a
        spurious imaginary DC current.  See :func:`_matching_tuples` for
        the companion half of this bookkeeping.

    Args:
        t (tuple): The index tuple.
        rs (dict): Correlation sums keyed by index tuple.
        at_dc (bool, optional): True when this tuple contributes to zero
            output frequency.  Default is False.

    Returns:
        ndarray: The current coefficient at this index tuple.

    """
    rs_p = rs[t]
    if not any(t):
        return rs_p.imag + 0j
    rs_m = rs[tuple(-x for x in t)]
    if at_dc:
        return (rs_p.imag + rs_m.imag) + 0j
    return (rs_p.imag + rs_m.imag) - 1j * (rs_p.real - rs_m.real)


# -- method selection --------------------------------------------------


def _select_method(
    method: str,
    num_f: int,
    num_p: int,
    nb_list: tuple[int, ...],
    npts: int,
    tuples_for: list[list[tuple[int, ...]]],
) -> str:
    """Choose between the direct and FFT correlation paths.

    Both compute Eqn. 5.25 exactly.  The direct path costs
    ``O(n_tuples * prod(2*nb+1) * npts)``; the FFT path costs
    ``O(V log V * npts)`` with ``V = prod(2*(nb+num_p)+1)``, independent of
    how many tuples are wanted.  So the FFT wins once enough output
    frequencies (or harmonics) are requested.

    Args:
        method (str): ``"auto"``, ``"direct"`` or ``"fft"``.
        num_f (int): Number of tones.
        num_p (int): Number of harmonics.
        nb_list (tuple): Summation limit per tone.
        npts (int): Number of bias points.
        tuples_for (list): Output of :func:`_matching_tuples`.

    Returns:
        str: ``"direct"`` or ``"fft"``.

    Raises:
        ValueError: If ``method`` is not recognised.

    """
    if method not in ("auto", "direct", "fft", "grid"):
        raise ValueError(f"Unknown method: {method!r}")
    if method != "auto":
        return method

    n_tuples = len(_needed_tuples(tuples_for))
    if n_tuples == 0:
        return "direct"

    pad_shape = _fft_shape(nb_list, num_p)
    fits = float(np.prod(pad_shape)) * npts * 16 * 3 <= FFT_MEMORY_BUDGET
    if not JIT_ENABLED:
        # Without a JIT the direct kernels run as interpreted Python loops,
        # while the FFT path stays fully vectorised.  Always prefer it.
        return "fft" if fits else "direct"

    direct_ops = 2 * n_tuples * np.prod([2 * nb + 1 for nb in nb_list])
    if not fits:
        return "direct"
    volume = float(np.prod(pad_shape))
    # Two multi-dimensional transforms; the coefficient transform is
    # separable and effectively free.  See FFT_COST_FACTOR.
    fft_ops = FFT_COST_FACTOR * volume * np.log2(volume)
    return "fft" if fft_ops < direct_ops else "direct"


def _fft_shape(nb_list: tuple[int, ...], num_p: int) -> tuple[int, ...]:
    """Zero-padded transform length for each tone axis.

    The correlation is only wanted for offsets up to ``+/-num_p``, so the
    axis has to be long enough that circular wrap-around lands in the zero
    padding: ``2*(num_b + num_p) + 1``, rounded up to an efficient FFT
    length.

    Args:
        nb_list (tuple): Summation limit per tone.
        num_p (int): Number of harmonics.

    Returns:
        tuple: The padded length of each tone axis.

    """
    return tuple(next_fast_len(2 * (nb + num_p) + 1) for nb in nb_list)


# -- the two correlation paths -----------------------------------------


def _correlate_direct(
    ccc: np.ndarray,
    resp_matrix: np.ndarray,
    nb_list: tuple[int, ...],
    tuples_for: list[list[tuple[int, ...]]],
    npts: int,
) -> dict[tuple[int, ...], np.ndarray]:
    """Evaluate Eqn. 5.25 with the blocked, thread-parallel kernels.

    Args:
        ccc (ndarray): Phase-factor coefficients.
        resp_matrix (ndarray): The interpolated response function.
        nb_list (tuple): Summation limit per tone.
        tuples_for (list): Output of :func:`_matching_tuples`.
        npts (int): Number of bias points.

    Returns:
        dict: ``RS+`` keyed by index tuple, including the negated tuples
        needed by :func:`_assemble`.

    """
    num_f = len(nb_list)
    kernel = (_kernels.coeff_1, _kernels.coeff_2, _kernels.coeff_3, _kernels.coeff_4)[
        num_f - 1
    ]
    ccc_tones = [np.ascontiguousarray(ccc[f]) for f in range(1, num_f + 1)]

    rs: dict[tuple[int, ...], np.ndarray] = {}
    for t in _needed_tuples(tuples_for):
        neg = tuple(-x for x in t)
        if t in rs:
            continue
        rs_p = np.zeros(npts, dtype=complex)
        # RS-(t) == RS+(-t), so one kernel call fills both entries.  When
        # the tuple is its own negation (the DC term) the second sum is
        # identical to the first and is skipped entirely.
        want_m = neg != t
        rs_m = np.zeros(npts, dtype=complex) if want_m else rs_p
        kernel(
            *t,
            *ccc_tones,
            resp_matrix,
            *nb_list,
            rs_p,
            rs_m,
            want_m,
            _kernels.BLOCK,
        )
        rs[t] = rs_p
        rs[neg] = rs_m
    return rs


def _correlate_fft(
    ccc: np.ndarray,
    resp_matrix: np.ndarray,
    nb_list: tuple[int, ...],
    num_p: int,
    npts: int,
) -> dict[tuple[int, ...], np.ndarray]:
    r"""Evaluate Eqn. 5.25 for every index tuple at once, with an FFT.

    Writing ``P[k,l,...] = C1[k] C2[l] ... R[k,l,...]`` and
    ``Q[k,l,...] = C1[k] C2[l] ...``, Eqn. 5.25 is exactly a
    cross-correlation,

    .. math::

        RS^+(a,b,\dots) = \sum_{k,l,\dots} P[k,l,\dots]\,
                           \overline{Q[k+a,\,l+b,\dots]},

    which the correlation theorem turns into
    ``ifftn(fftn(P) * conj(fftn(Q)))`` evaluated at ``-t``.  Zero-padding
    each axis to ``2*(num_b + num_p) + 1`` keeps the circular wrap-around
    inside the padding, so the result is the *linear* correlation QMix
    computes, to machine precision.

    One pair of transforms then yields the coefficients for every index
    tuple, instead of one nested loop per tuple.

    Args:
        ccc (ndarray): Phase-factor coefficients.
        resp_matrix (ndarray): The interpolated response function.
        nb_list (tuple): Summation limit per tone.
        num_p (int): Number of harmonics.
        npts (int): Number of bias points.

    Returns:
        dict: ``RS+`` keyed by index tuple, for every tuple in
        ``[-num_p, num_p]**num_f``.

    """
    num_f = len(nb_list)
    pad = _fft_shape(nb_list, num_p)
    axes = tuple(range(num_f))

    # Q is a separable outer product of the per-tone coefficients, so its
    # transform is the outer product of 1-D transforms.
    q = np.ones((1,) * num_f + (npts,), dtype=complex)
    for axis, nb in enumerate(nb_list):
        c = np.zeros((pad[axis], npts), dtype=complex)
        c[: nb + 1] = ccc[axis + 1, : nb + 1]
        if nb:
            c[pad[axis] - nb :] = ccc[axis + 1, -nb:]
        shape = [1] * (num_f + 1)
        shape[axis] = pad[axis]
        shape[-1] = npts
        q = q * c.reshape(shape)

    p = np.zeros((*pad, npts), dtype=complex)
    block = tuple(
        np.r_[np.arange(0, nb + 1), np.arange(pad[i] - nb, pad[i])]
        for i, nb in enumerate(nb_list)
    )
    p[np.ix_(*block, np.arange(npts))] = resp_matrix
    p *= q

    corr = ifftn(
        fftn(p, axes=axes, workers=-1) * np.conj(fftn(q, axes=axes, workers=-1)),
        axes=axes,
        workers=-1,
    )

    span = range(-num_p, num_p + 1)
    return {
        t: corr[tuple((-x) % pad[i] for i, x in enumerate(t))]
        for t in itertools.product(span, repeat=num_f)
    }
