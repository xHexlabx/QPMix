r"""Simulations with arbitrarily many tones.

Why four tones is the wall
--------------------------

Kittara's formulation carries one summation index per tone, so the
interpolated response function is an array of
``(2*num_b + 1)**num_f * npts`` complex values.  That is *exponential* in
the number of tones:

===========  ==========================  ===========
``num_f``    entries (``num_b=15``,      memory
             ``npts=401``)
===========  ==========================  ===========
1            12 431                      0.2 MB
2            385 361                     6 MB
3            11 946 191                  191 MB
4            370 331 921                 5.9 GB
5            11 480 289 551              184 GB
===========  ==========================  ===========

QMix stops at four because five does not fit in memory.  Adding more
hand-written loop nests would not help.

The common-grid formulation
---------------------------

The exponential blow-up is an artefact of treating the tones as mutually
incommensurate.  Put every tone on a shared frequency grid instead --
tone ``f`` at ``n_f * df`` for integer ``n_f`` -- and the whole
multi-dimensional index collapses to a single one.  The phase-factor
generating function becomes

.. math::

    \sum_k C_k e^{ik\theta}
        = \exp\!\Big(i \sum_f \sum_p
              \alpha_{f,p}\, \sin(p\, n_f\, \theta - \phi_{f,p})\Big)

which is still *one* scalar function of :math:`\theta`, and therefore
still *one* FFT, no matter how many tones it contains.  The response
function is then needed only at ``vb + k*df`` for
:math:`|k| \le K = \sum_f n_f\,\mathrm{num\_b}_f`, so memory grows
**linearly** with the number of tones rather than exponentially, and the
current summation reduces to the one-dimensional correlation that
:mod:`qpmix.qtcurrent` already has a kernel for.

The trade-off
-------------

``K`` is set by the *multipliers*, not by the tone count: closely spaced
tones need a fine grid, which makes every ``n_f`` large.  An LO and an RF
signal 5 MHz apart at 230 GHz need ``n = (46000, 46001)``, giving
``K = 1.4e6`` -- far worse than the two-dimensional method.  So:

* **many tones on a coarse grid** (a comb, a band split into channels, a
  continuum approximation) -- use this module;
* **a few closely spaced tones** (LO + RF + IF) -- use
  :func:`qpmix.qtcurrent.qtcurrent` directly;
* **a comb whose spacing is not a simple fraction of the gap frequency**
  (any *measured* ``fgap``) -- the rational fit cannot see the comb once
  the tones are normalized, so give :meth:`ToneGrid.from_circuit` the
  spacing directly: ``df=spacing_hz / cct.fgap``.

:meth:`ToneGrid.report` prints both costs so the choice can be made from
numbers rather than intuition.

Examples:

    >>> import numpy as np
    >>> from qpmix.circuit import EmbeddingCircuit
    >>> cct = EmbeddingCircuit(3, 1, vb_npts=21)
    >>> cct.freq[1:] = [0.30, 0.32, 0.34]
    >>> grid = ToneGrid.from_circuit(cct, num_b=9)
    >>> grid.multipliers
    (15, 16, 17)
    >>> round(grid.df, 4)
    0.02
    >>> grid.num_k
    432

"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import gcd, lcm
from timeit import default_timer as timer

import numpy as np
from scipy.fft import fft, next_fast_len

from qpmix import _kernels
from qpmix._backend import JIT_ENABLED, njit, prange
from qpmix._respmat import build_resp_matrix
from qpmix.phase_factor import _as_nb_tuple, drive_level

__all__ = [
    "ToneGrid",
    "interpolate_respfn_grid",
    "phase_factor_grid",
    "qtcurrent_grid",
]

#: Default cap on the denominator used when fitting tone frequencies to a
#: common grid.  Larger values track the requested frequencies more closely
#: but push up ``num_k`` and therefore the cost.
MAX_DENOMINATOR = 10_000

#: ``ToneGrid.from_circuit`` refuses to build a grid needing more than this
#: many response-function entries, rather than silently allocating it.
DEFAULT_MAX_ENTRIES = 400_000_000


@dataclass(frozen=True)
class ToneGrid:
    """A common frequency grid shared by every tone.

    Tone ``f`` sits at ``multipliers[f] * df``.  Together with ``num_k``
    this fixes the cost of a simulation.

    Args:
        df (float): Grid spacing, normalized to the gap frequency.
        multipliers (tuple): Integer multiplier for each tone.
        num_b (tuple): Summation limit for each tone.
        num_p (int): Number of harmonics.
        requested (tuple): The frequencies that were asked for, so the
            approximation error can be reported.

    Attributes:
        num_k (int): Offset truncation, ``sum(n_f * num_b_f)``.
        frequencies (tuple): The frequencies the grid actually represents.

    """

    df: float
    multipliers: tuple[int, ...]
    num_b: tuple[int, ...]
    num_p: int
    requested: tuple[float, ...]

    def __post_init__(self) -> None:
        """Reject grids that cannot represent the requested tones.

        Raises:
            ValueError: If the spacing is not positive, a multiplier is
                below one (a tone would sit at DC), or two tones share a
                multiplier (they would alias onto the same frequency).

        """
        if not self.df > 0:
            raise ValueError("Grid spacing df must be positive.")
        if len(self.multipliers) != len(self.num_b):
            raise ValueError("Need one num_b per tone.")
        if len(self.multipliers) != len(self.requested):
            raise ValueError("Need one requested frequency per tone.")
        if any(n < 1 for n in self.multipliers):
            raise ValueError(
                f"Grid spacing df={self.df:.6g} is too coarse for "
                f"{self.requested}: multipliers {self.multipliers}."
            )
        if len(set(self.multipliers)) != len(self.multipliers):
            raise ValueError(
                f"Grid spacing df={self.df:.6g} makes distinct tones "
                f"{self.requested} share a multiplier {self.multipliers}."
            )

    @property
    def num_f(self) -> int:
        """Number of tones."""
        return len(self.multipliers)

    @property
    def num_k(self) -> int:
        """Offset truncation: the largest reachable multiple of ``df``."""
        pairs = zip(self.multipliers, self.num_b, strict=True)
        return int(sum(n * b for n, b in pairs))

    @property
    def frequencies(self) -> tuple[float, ...]:
        """The tone frequencies the grid actually represents."""
        return tuple(n * self.df for n in self.multipliers)

    @property
    def frequency_error(self) -> float:
        """Largest absolute difference between requested and gridded tones."""
        return max(
            abs(got - want)
            for got, want in zip(self.frequencies, self.requested, strict=True)
        )

    @property
    def entries(self) -> int:
        """Number of complex values in the gridded response matrix."""
        return 2 * self.num_k + 1

    @property
    def tolerance(self) -> float:
        """How far off a grid point :meth:`offset` lets a frequency sit.

        An exact grid accepts only rounding noise.  An approximate one --
        from ``max_num_k`` or a capped ``max_denominator`` -- represents
        tone ``f`` at ``multipliers[f] * df`` rather than at
        ``requested[f]``, so a frequency formed from the requested tones (a
        harmonic, an IF, an intermodulation product) is off the grid by
        that approximation error times its order.  Allow for the highest
        order the multi-dimensional engine enumerates, ``num_f * num_p``.
        """
        return max(self.df * 1e-6, self.num_f * self.num_p * self.frequency_error)

    def offset(self, frequency: float, tol: float | None = None) -> int:
        """Index offset corresponding to an output frequency.

        Args:
            frequency (float): Output frequency, normalized.  Pass it
                unrounded: on a fine grid, a value rounded to a few
                decimals is no longer on the grid.
            tol (float, optional): How far from a grid point the frequency
                may sit and still be accepted.  Default is None, meaning
                :attr:`tolerance`.

        Returns:
            int: The offset ``a`` such that ``a * df`` is that frequency.

        Raises:
            ValueError: If the frequency is not on the grid.

        """
        a = round(frequency / self.df)
        if tol is None:
            tol = self.tolerance
        if abs(a * self.df - frequency) > tol:
            raise ValueError(
                f"Frequency {frequency} is not on the grid (df = {self.df}, "
                f"nearest point {a * self.df}, tolerance {tol:.3g})."
            )
        return int(a)

    @classmethod
    def from_frequencies(
        cls,
        frequencies,
        num_b: int | tuple[int, ...] = 15,
        num_p: int = 1,
        df: float | None = None,
        max_denominator: int = MAX_DENOMINATOR,
        max_num_k: int | None = None,
    ) -> ToneGrid:
        """Fit a set of tone frequencies onto a common grid.

        Each frequency is approximated by a rational multiple of a single
        spacing, found by simultaneous rational approximation.

        Note:
            ``num_k`` is *not* monotonic in ``max_denominator``: a tighter
            cap can force a rational approximation that happens to need a
            much finer grid.  To control cost directly, pass ``max_num_k``
            and let this method search for the most accurate grid that fits
            the budget.

        Args:
            frequencies (sequence): Normalized tone frequencies, all > 0.
            num_b (int or tuple, optional): Summation limit, one value or
                one per tone.  Default is 15.
            num_p (int, optional): Number of harmonics.  Default is 1.
            df (float, optional): Force this grid spacing instead of
                deriving one.  Default is None.
            max_denominator (int, optional): Cap on the denominator used in
                the rational approximation.  Default is 10 000.
            max_num_k (int, optional): Budget for ``num_k``.  When given,
                search denominators up to ``max_denominator`` and return
                the most accurate grid that stays within the budget.
                Default is None.

        Returns:
            ToneGrid: The grid.

        Raises:
            ValueError: If any frequency is not positive, if a forced
                ``df`` does not divide them, or if no grid fits
                ``max_num_k``.

        """
        freqs = tuple(float(f) for f in frequencies)
        if not freqs:
            raise ValueError("Need at least one frequency.")
        if min(freqs) <= 0:
            raise ValueError("All tone frequencies must be > 0.")
        nb = _as_nb_tuple(num_b, len(freqs))

        if df is None and max_num_k is not None:
            return cls._search(freqs, nb, num_p, max_denominator, max_num_k)
        if df is None:
            df, mult = _common_grid(freqs, max_denominator)
        else:
            df = float(df)
            mult = tuple(round(f / df) for f in freqs)
        return cls(df=df, multipliers=mult, num_b=nb, num_p=num_p, requested=freqs)

    @classmethod
    def _search(
        cls,
        freqs: tuple[float, ...],
        nb: tuple[int, ...],
        num_p: int,
        max_denominator: int,
        max_num_k: int,
    ) -> ToneGrid:
        """Find the most accurate grid whose ``num_k`` fits a budget.

        Args:
            freqs (tuple): The frequencies.
            nb (tuple): Summation limit per tone.
            num_p (int): Number of harmonics.
            max_denominator (int): Largest denominator to try.
            max_num_k (int): Budget for ``num_k``.

        Returns:
            ToneGrid: The best grid within budget.

        Raises:
            ValueError: If no denominator produces a grid within budget.

        """
        best: ToneGrid | None = None
        for denom in range(1, max_denominator + 1):
            df, mult = _common_grid(freqs, denom)
            try:
                candidate = cls(
                    df=df, multipliers=mult, num_b=nb, num_p=num_p, requested=freqs
                )
            except ValueError:
                # This denominator collapses or zeroes a tone; try a finer one.
                continue
            if candidate.num_k > max_num_k:
                continue
            if best is None or candidate.frequency_error < best.frequency_error:
                best = candidate
            if best.frequency_error == 0.0:
                break
        if best is None:
            raise ValueError(
                f"No common grid with num_k <= {max_num_k} exists for "
                f"{freqs} at num_b={nb}. Raise max_num_k or lower num_b."
            )
        return best

    @classmethod
    def from_circuit(
        cls,
        cct,
        num_b: int | tuple[int, ...] = 15,
        df: float | None = None,
        max_denominator: int = MAX_DENOMINATOR,
        max_num_k: int | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> ToneGrid:
        """Fit the tones of an embedding circuit onto a common grid.

        Args:
            cct (qpmix.circuit.EmbeddingCircuit): The circuit.
            num_b (int or tuple, optional): Summation limit.  Default is 15.
            df (float, optional): Force this grid spacing.  Default is None.
            max_denominator (int, optional): Cap on the rational
                approximation denominator.  Default is 10 000.
            max_num_k (int, optional): Budget for ``num_k``; see
                :meth:`from_frequencies`.  Default is None.
            max_entries (int, optional): Refuse to build a grid needing more
                than this many response-function entries.  Pass ``0`` to
                disable the check.  Default is 4e8.

        Returns:
            ToneGrid: The grid.

        Raises:
            MemoryError: If the resulting grid would exceed ``max_entries``.

        """
        grid = cls.from_frequencies(
            cct.freq[1 : cct.num_f + 1],
            num_b=num_b,
            num_p=cct.num_p,
            df=df,
            max_denominator=max_denominator,
            max_num_k=max_num_k,
        )
        needed = grid.entries * cct.vb_npts
        if max_entries and needed > max_entries:
            raise MemoryError(
                f"This grid needs {needed:,} response-function entries "
                f"({needed * 16 / 1024**3:.1f} GB) because the tones are "
                f"closely spaced (multipliers {grid.multipliers}). Widen "
                f"the grid with a smaller max_denominator or an explicit "
                f"df, lower num_b, or use qtcurrent's multi-dimensional "
                f"path instead. Pass max_entries=0 to allocate it anyway."
            )
        return grid

    def report(self, npts: int) -> str:
        """Describe the grid and compare its cost with the direct method.

        Args:
            npts (int): Number of bias voltage points.

        Returns:
            str: A multi-line summary.

        """
        grid_entries = self.entries * npts
        direct_entries = np.prod([2 * b + 1 for b in self.num_b], dtype=float) * npts
        lines = [
            f"ToneGrid: {self.num_f} tone(s), {self.num_p} harmonic(s)",
            f"  spacing df    : {self.df:.6g}",
            f"  multipliers   : {self.multipliers}",
            f"  frequencies   : {tuple(round(f, 6) for f in self.frequencies)}",
            f"  freq. error   : {self.frequency_error:.3e}",
            f"  num_k         : {self.num_k:,}",
            f"  grid entries  : {grid_entries:,} "
            f"({grid_entries * 16 / 1024**2:.1f} MB)",
            f"  direct entries: {direct_entries:,.0f} "
            f"({direct_entries * 16 / 1024**3:.2f} GB)",
            f"  ratio         : grid is {direct_entries / grid_entries:.4g}x smaller",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ToneGrid(num_f={self.num_f}, df={self.df:.6g}, "
            f"multipliers={self.multipliers}, num_k={self.num_k})"
        )


def _common_grid(
    freqs: tuple[float, ...], max_denominator: int
) -> tuple[float, tuple[int, ...]]:
    """Find a grid spacing that every frequency is close to a multiple of.

    Each frequency is written as a fraction with a bounded denominator; the
    spacing is then ``1 / lcm(denominators)``, reduced by the gcd of the
    resulting multipliers so the grid is as coarse as it can be.

    Args:
        freqs (tuple): The frequencies.
        max_denominator (int): Cap on each fraction's denominator.

    Returns:
        tuple: ``(df, multipliers)``.

    """
    fractions = [Fraction(f).limit_denominator(max_denominator) for f in freqs]
    denom = 1
    for fr in fractions:
        denom = lcm(denom, fr.denominator)
    mult = [int(fr.numerator * (denom // fr.denominator)) for fr in fractions]

    common = 0
    for n in mult:
        common = gcd(common, n)
    if common > 1:
        mult = [n // common for n in mult]
        denom //= gcd(denom, common) if denom % common == 0 else 1
        # Recompute the spacing from the reduced multipliers so that
        # multiplier * df still reproduces the frequency.
        return freqs[0] / mult[0], tuple(mult)
    return 1.0 / denom, tuple(mult)


# -- phase factor on the grid -----------------------------------------


def _support(alpha: np.ndarray, multipliers, num_p: int) -> float:
    """Spectral half-width of the phase factor, in grid steps.

    ``|J_n(alpha)|`` is below machine epsilon once ``n`` exceeds roughly
    ``alpha + 4*alpha**(1/3) + 25``; each drive term at frequency
    ``p * n_f * df`` spreads the spectrum by that much times ``p * n_f``.

    Args:
        alpha (ndarray): Drive levels, shape ``(num_f+1, num_p+1, npts)``.
        multipliers (tuple): Grid multiplier of each tone.
        num_p (int): Number of harmonics.

    Returns:
        float: The half-width.

    """
    total = 0.0
    for f, n in enumerate(multipliers, start=1):
        for p in range(1, num_p + 1):
            a = float(alpha[f, p].max())
            total += p * n * (a + 4.0 * a ** (1 / 3) + 25.0)
    return total


@njit(parallel=True)
def _grid_phase(alpha, phi, nu, theta, out):  # pragma: no cover - JIT kernel
    """Sample the combined phase factor on a uniform theta grid.

    Fuses the sum over every tone and harmonic, the sine and the complex
    exponential into one pass, so the ``(num_theta, npts)`` phase array is
    never materialised.

    Args:
        alpha (ndarray): Drive levels, shape ``(num_terms, npts)``.
        phi (ndarray): Phases, same shape.
        nu (ndarray): Integer grid frequency ``p * n_f`` of each term.
        theta (ndarray): Sample positions.
        out (ndarray): Destination, shape ``(num_theta, npts)``, complex.

    """
    num_terms, npts = alpha.shape
    for m in prange(theta.shape[0]):
        th = theta[m]
        for i in range(npts):
            acc = 0.0
            for j in range(num_terms):
                acc += alpha[j, i] * np.sin(nu[j] * th - phi[j, i])
            out[m, i] = np.cos(acc) + 1j * np.sin(acc)


def _grid_phase_numpy(alpha, phi, nu, theta, out):
    """NumPy fallback for :func:`_grid_phase`.

    Args:
        alpha (ndarray): Drive levels, shape ``(num_terms, npts)``.
        phi (ndarray): Phases, same shape.
        nu (ndarray): Integer grid frequency of each term.
        theta (ndarray): Sample positions.
        out (ndarray): Destination, shape ``(num_theta, npts)``, complex.

    """
    phase = np.zeros(out.shape, dtype=float)
    for j in range(alpha.shape[0]):
        phase += alpha[j][None, :] * np.sin(nu[j] * theta[:, None] - phi[j][None, :])
    np.exp(1j * phase, out=out)


def phase_factor_grid(
    vj: np.ndarray, cct, grid: ToneGrid, num_theta: int | None = None
) -> np.ndarray:
    """Phase-factor spectrum coefficients on the common grid.

    One FFT, however many tones there are.

    Args:
        vj (ndarray): Junction voltage, shape
            ``(num_f + 1, num_p + 1, npts)``.
        cct (qpmix.circuit.EmbeddingCircuit): The embedding circuit.
        grid (ToneGrid): The common frequency grid.
        num_theta (int, optional): Override the FFT length.  Default is
            chosen from the spectral support.

    Returns:
        ndarray: ``C_k``, shape ``(2 * num_k + 1, npts)``, complex, with
        ``k`` in NumPy FFT order.

    """
    npts = vj.shape[2]
    num_f, num_p = grid.num_f, grid.num_p
    alpha, phi = drive_level(vj, cct.freq, num_f, num_p)

    # Flatten (tone, harmonic) into a list of sinusoidal drive terms.
    terms_a = np.empty((num_f * num_p, npts), dtype=float)
    terms_p = np.empty((num_f * num_p, npts), dtype=float)
    nu = np.empty(num_f * num_p, dtype=np.int64)
    j = 0
    for f in range(1, num_f + 1):
        for p in range(1, num_p + 1):
            terms_a[j] = alpha[f, p]
            terms_p[j] = phi[f, p]
            nu[j] = p * grid.multipliers[f - 1]
            j += 1

    num_k = grid.num_k
    if num_theta is None:
        support = _support(alpha, grid.multipliers, num_p)
        num_theta = int(next_fast_len(int(num_k + support + 32)))

    theta = 2 * np.pi * np.arange(num_theta) / num_theta
    signal = np.empty((num_theta, npts), dtype=complex)
    sample = _grid_phase if JIT_ENABLED else _grid_phase_numpy
    sample(terms_a, terms_p, nu, theta, signal)

    spec = fft(signal, axis=0, workers=-1)
    spec /= num_theta

    ck = np.zeros((2 * num_k + 1, npts), dtype=complex)
    ck[: num_k + 1] = spec[: num_k + 1]
    if num_k:
        ck[num_k + 1 :] = spec[num_theta - num_k :]
    return ck


def interpolate_respfn_grid(cct, resp, grid: ToneGrid) -> np.ndarray:
    """Interpolate the response function at every grid offset.

    Args:
        cct (qpmix.circuit.EmbeddingCircuit): The embedding circuit.
        resp (qpmix.respfn.RespFn): The response function.
        grid (ToneGrid): The common frequency grid.

    Returns:
        ndarray: Shape ``(2 * num_k + 1, npts)``, complex.

    """
    freq = np.array([0.0, grid.df], dtype=float)
    return build_resp_matrix(resp, cct.vb, freq, (grid.num_k,))


def _correlate(ck, resp_matrix, num_k, offsets, npts):
    """Evaluate ``RS+(a) = sum_k C_k conj(C_{k+a}) R_k`` for each offset.

    Args:
        ck (ndarray): Phase-factor coefficients, FFT order.
        resp_matrix (ndarray): Interpolated response function, FFT order.
        num_k (int): Offset truncation.
        offsets (iterable): The offsets to evaluate, and their negations.
        npts (int): Number of bias points.

    Returns:
        dict: ``RS+`` keyed by offset.

    """
    wanted: list[int] = []
    for a in offsets:
        for want in (a, -a):
            if want not in wanted:
                wanted.append(want)

    if JIT_ENABLED:
        # The blocked kernel streams the response matrix through cache once
        # per offset and needs no temporaries.
        rs = {}
        for a in wanted:
            acc = np.zeros(npts, dtype=complex)
            _kernels.coeff_1(a, ck, resp_matrix, num_k, acc, acc, False, _kernels.BLOCK)
            rs[a] = acc
        return rs

    # NumPy fallback: re-index from FFT order to natural order so the
    # correlation is a plain shifted slice product.  Interpreting the
    # blocked kernel would be hopeless at these truncation limits.
    c_nat = np.roll(ck, num_k, axis=0)
    r_nat = np.roll(resp_matrix, num_k, axis=0)
    span = 2 * num_k
    rs = {}
    for a in wanted:
        lo = max(0, -a)
        hi = min(span, span - a)
        if lo > hi:
            rs[a] = np.zeros(npts, dtype=complex)
            continue
        rs[a] = (
            c_nat[lo : hi + 1]
            * np.conj(c_nat[lo + a : hi + a + 1])
            * r_nat[lo : hi + 1]
        ).sum(axis=0)
    return rs


def qtcurrent_grid(
    vj: np.ndarray,
    cct,
    resp,
    freq_list,
    num_b: int | tuple[int, ...] = 15,
    verbose: bool = True,
    grid: ToneGrid | None = None,
    resp_matrix: np.ndarray | None = None,
    num_theta: int | None = None,
) -> np.ndarray:
    """Quasiparticle tunneling current, for any number of tones.

    Same physics and the same return convention as
    :func:`qpmix.qtcurrent.qtcurrent`, but every tone is placed on a common
    frequency grid so the cost grows linearly rather than exponentially
    with the tone count.  See the module docstring for when this is the
    right trade.

    Args:
        vj (ndarray): Junction voltage, shape
            ``(num_f + 1, num_p + 1, npts)``.
        cct (qpmix.circuit.EmbeddingCircuit): The embedding circuit.
        resp (qpmix.respfn.RespFn): The response function.
        freq_list (float or sequence): Output frequencies, normalized.
            Each must lie on the grid.
        num_b (int or tuple, optional): Summation limit.  Default is 15.
        verbose (bool, optional): Print progress.  Default is True.
        grid (ToneGrid, optional): Reuse a grid instead of deriving one.
            Default is None.
        resp_matrix (ndarray, optional): Reuse a matrix from
            :func:`interpolate_respfn_grid`.  Default is None.
        num_theta (int, optional): Override the FFT length.  Default is
            None.

    Returns:
        ndarray: The tunneling current, shaped as
        :func:`qpmix.qtcurrent.qtcurrent` returns it.

    Raises:
        ValueError: If an output frequency is not on the grid, or if
            ``resp_matrix`` was built for a different grid.

    """
    npts = cct.vb_npts
    if grid is None:
        grid = ToneGrid.from_circuit(cct, num_b=num_b)

    freq_is_list = np.ndim(freq_list) > 0
    freq_out = np.atleast_1d(np.asarray(freq_list, dtype=float))

    if verbose:
        print("Calculating tunneling current (common grid)...")
        print(f" - {grid.num_f} tone(s), {grid.num_p} harmonic(s)")
        print(f" - grid df = {grid.df:.6g}, num_k = {grid.num_k:,}")
        start_time = timer()

    ck = phase_factor_grid(vj, cct, grid, num_theta=num_theta)
    if resp_matrix is None:
        resp_matrix = interpolate_respfn_grid(cct, resp, grid)
    elif resp_matrix.shape != (grid.entries, npts):
        raise ValueError(
            f"resp_matrix has shape {resp_matrix.shape}, but this grid needs "
            f"{(grid.entries, npts)}. Pass the grid it was built from."
        )

    # Every output frequency is a single offset on the grid, so this is the
    # one-dimensional correlation that _kernels.coeff_1 already implements.
    offsets = [grid.offset(float(f)) for f in freq_out]
    rs = _correlate(ck, resp_matrix, grid.num_k, offsets, npts)

    current_out = np.zeros((freq_out.size, npts), dtype=complex)
    for idx, a in enumerate(offsets):
        rs_p = rs[a]
        if a == 0:
            current_out[idx] = rs_p.imag + 0j
        else:
            rs_m = rs[-a]
            current_out[idx] = (rs_p.imag + rs_m.imag) - 1j * (rs_p.real - rs_m.real)

    if verbose:
        print("Done.")
        print(f"Time: {timer() - start_time:.4f} s\n")

    if freq_is_list:
        return current_out
    if freq_out[0] == 0.0:
        return current_out[0].real
    return current_out[0]
