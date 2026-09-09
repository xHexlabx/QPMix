r"""Embedding-circuit recovery by matching the pumped I-V *curve*.

Two different ways to recover the Thevenin equivalent source
-------------------------------------------------------------

:mod:`qpmix.exp.zemb` implements the **voltage-match** method of Skalare
(1989) and Withington *et al.* (1995).  Tucker theory turns the measured
pumped I-V curve into a drive level, hence an AC voltage and impedance, at
every bias point; those points lie on one load line, and the source is
fitted to it in closed form.  It is fast -- milliseconds -- but it inherits
Tucker theory's assumptions: one tone, no harmonics, and a well-behaved
first photon step to invert on.

This module implements **current matching** instead.  A trial source
``(V_T, Z_T)`` is fed through a full harmonic balance to *simulate* a pumped
I-V curve, and the source is varied until the simulated curve matches the
measured one:

.. math::

    \min_{V_T, Z_T} \; \operatorname{RMS}\big[
        I_{dc}^{\rm sim}(V_0; V_T, Z_T) - I_{dc}^{\rm meas}(V_0)\big]

Each objective evaluation costs a harmonic balance, so this is orders of
magnitude slower.  In exchange it makes no small-signal assumption, works on
*any* bias window rather than only the first photon step, and extends
naturally to higher harmonics, where the embedding impedance at ``2*f_LO``
is fitted alongside the fundamental.

Choosing where to fit
---------------------

:func:`voltage_windows` builds the bias windows the residual is evaluated
over:

=================  ============================================================
``first_photon``   The flat middle of the first photon step -- the classical
                   choice, and the only one the voltage-match method can use.
``full_subgap``    Everything below the gap, so every photon step
                   contributes.
``full_range``     The whole measured sweep, including above the gap.
``photon_steps``   One disjoint window per photon step, which weights the
                   steps equally instead of letting the widest dominate.
=================  ============================================================

Picking a solution
------------------

The residual surface has local minima, so a single simplex run is not
trustworthy.  A small grid of starting points (nine by default: three
resistances by three reactances) is optimised independently, and the answer
is the one the runs *agree* on -- the most frequently recovered solution
among those that are physically admissible -- rather than simply the lowest
residual, which is easily a spurious minimum.  Ties are broken by residual.

Seeding from the voltage-match answer (``guesses="seeded"``) usually
collapses this to one or two runs, since that answer is already close.

Examples:

    >>> from qpmix.exp.currentmatch import voltage_windows
    >>> lo, hi = voltage_windows("first_photon", vph=0.3)[0]
    >>> round(lo, 4), round(hi, 4)
    (0.775, 0.94)
    >>> len(voltage_windows("photon_steps", vph=0.3, n_steps=3))
    3

"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any, NamedTuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from qpmix.circuit import EmbeddingCircuit
from qpmix.harmonic_balance import harmonic_balance
from qpmix.qtcurrent import interpolate_respfn, qtcurrent

__all__ = [
    "VOLTAGE_METHODS",
    "CurrentMatchResult",
    "current_residual",
    "default_guesses",
    "recover_zemb_current_match",
    "voltage_windows",
]

#: Bias-window strategies understood by :func:`voltage_windows`.
VOLTAGE_METHODS = ("first_photon", "full_subgap", "full_range", "photon_steps")

#: Default resistances tried as starting points, normalized to ``Rn``.
DEFAULT_REAL_GUESSES = (0.1, 0.5, 1.0)

#: Default reactances tried as starting points, normalized to ``Rn``.
DEFAULT_IMAG_GUESSES = (-0.5, 0.0, 0.5)


class CurrentMatchResult(NamedTuple):
    """Outcome of a current-matching fit.

    Attributes:
        vt (float): Thevenin voltage, normalized to the gap voltage.
        zt (tuple): Embedding impedance of each harmonic, normalized to
            ``Rn``.  Length equals ``harmonics``.
        err (float): RMS current residual of the chosen solution,
            normalized to the gap current.
        occurrences (int): How many independent runs converged on this
            solution.
        n_runs (int): How many starting points were tried.
        n_admissible (int): How many runs produced a physically admissible
            solution.
        selection (str): ``"mode"`` if the answer is the most frequently
            recovered solution, ``"best"`` if no solution recurred and the
            lowest residual was taken instead.
        windows (tuple): The bias windows the residual was evaluated over.
        candidates (tuple): One ``(start, solution, residual, admissible)``
            entry per run, in the order they were tried.
        voltage (ndarray): The bias grid the fit was performed on.
        current (ndarray): The measured current on that grid.
        simulated (ndarray): The simulated current for the chosen solution.
        n_calls (int): How many times the residual was evaluated in total.
        n_not_converged (int): How many of those harmonic balances did not
            reach the error target.  A few is normal -- the relative error
            at near-zero bias is large however good the solution is -- but a
            large fraction means the fit is unreliable.

    """

    vt: float
    zt: tuple[complex, ...]
    err: float
    occurrences: int
    n_runs: int
    n_admissible: int
    selection: str
    windows: tuple[tuple[float, float], ...]
    candidates: tuple
    voltage: np.ndarray
    current: np.ndarray
    simulated: np.ndarray
    n_calls: int
    n_not_converged: int


def voltage_windows(
    method: str,
    vph: float,
    fit_range: tuple[float, float] = (0.25, 0.8),
    v_floor: float = 0.05,
    v_ceiling: float | None = None,
    n_steps: int = 3,
) -> list[tuple[float, float]]:
    """Build the bias windows to evaluate the current residual over.

    Args:
        method (str): One of :data:`VOLTAGE_METHODS`.
        vph (float): Photon voltage, normalized to the gap voltage.
        fit_range (tuple, optional): Fraction of a photon step to keep, as
            ``(from the bottom, up to)``.  ``(0.25, 0.8)`` drops the first
            25% and the last 20% of each step, keeping the flat middle.
            Default is ``(0.25, 0.8)``.
        v_floor (float, optional): Lowest normalized bias voltage to
            include.  Default is 0.05.
        v_ceiling (float, optional): Highest normalized bias voltage to
            include.  Defaults to just below the gap for ``full_subgap`` and
            to ``1 + 2 * vph`` for ``full_range``.
        n_steps (int, optional): How many photon steps ``photon_steps``
            covers.  Default is 3.

    Returns:
        list: One ``(low, high)`` pair per window, in increasing order.

    Raises:
        ValueError: If the method is unknown, the photon voltage is not
            positive, or no usable window results.

    """
    if method not in VOLTAGE_METHODS:
        raise ValueError(f"Unknown method: {method!r}. Choose from {VOLTAGE_METHODS}.")
    if vph <= 0:
        raise ValueError("The photon voltage must be positive.")

    lo_frac, hi_frac = fit_range
    if method == "first_photon":
        windows = [(1 - vph + lo_frac * vph, 1 - (1 - hi_frac) * vph)]
    elif method == "photon_steps":
        windows = [
            (1 - (n + 1) * vph + lo_frac * vph, 1 - n * vph - (1 - hi_frac) * vph)
            for n in range(n_steps)
        ][::-1]
    elif method == "full_subgap":
        ceiling = 1 - (1 - hi_frac) * vph if v_ceiling is None else v_ceiling
        windows = [(v_floor, ceiling)]
    else:  # full_range
        ceiling = 1 + 2 * vph if v_ceiling is None else v_ceiling
        windows = [(v_floor, ceiling)]

    windows = [(lo, hi) for lo, hi in windows if hi > lo]
    if not windows:
        raise ValueError(
            f"Method {method!r} produced no usable bias window at "
            f"vph={vph:.4f}. Check the photon voltage and 'fit_range'."
        )
    return windows


def default_guesses(
    harmonics: int = 1,
    vt: float = 1.0,
    real_values=DEFAULT_REAL_GUESSES,
    imag_values=DEFAULT_IMAG_GUESSES,
) -> list[list[float]]:
    """Build the default grid of starting points.

    Higher harmonics start from the same impedance as the fundamental, so
    the grid stays at nine points however many harmonics are fitted rather
    than growing as ``9**harmonics``.

    Args:
        harmonics (int, optional): Number of harmonics to fit.  Default is
            1.
        vt (float, optional): Starting Thevenin voltage.  Default is 1.
        real_values (sequence, optional): Resistances to try.  Default is
            :data:`DEFAULT_REAL_GUESSES`.
        imag_values (sequence, optional): Reactances to try.  Default is
            :data:`DEFAULT_IMAG_GUESSES`.

    Returns:
        list: Starting parameter vectors, each ``[vt, re1, im1, re2, ...]``.

    """
    return [
        [vt] + [z for _ in range(harmonics) for z in (re, im)]
        for re, im in itertools.product(real_values, imag_values)
    ]


def _unpack(params, harmonics):
    """Split a parameter vector into ``(vt, [zt per harmonic])``.

    Args:
        params (sequence): ``[vt, re1, im1, re2, im2, ...]``.
        harmonics (int): Number of harmonics.

    Returns:
        tuple: ``(vt, tuple of complex impedances)``.

    Raises:
        ValueError: If the vector is the wrong length.

    """
    expected = 1 + 2 * harmonics
    if len(params) != expected:
        raise ValueError(
            f"Expected {expected} parameters for {harmonics} harmonic(s), "
            f"got {len(params)}."
        )
    vt = float(params[0])
    zt = tuple(complex(params[1 + 2 * k], params[2 + 2 * k]) for k in range(harmonics))
    return vt, zt


def _admissible(vt, zt, max_reactance):
    """Is this a physically sensible embedding circuit?

    A passive source has non-negative resistance and a non-negative
    available voltage; a wildly large reactance means the simplex has
    wandered off rather than converged.

    Args:
        vt (float): Thevenin voltage.
        zt (tuple): Embedding impedances.
        max_reactance (float): Largest acceptable ``|Im zt|``.

    Returns:
        bool: True if the solution is admissible.

    """
    if not np.isfinite(vt) or vt <= 0:
        return False
    for z in zt:
        if not np.isfinite(z) or z.real < 0 or abs(z.imag) > max_reactance:
            return False
    return True


class _Objective:
    """The current residual, with everything reusable computed once.

    The bias sweep, the tone frequency and ``num_b`` are all fixed while the
    source is being fitted, and they are what determine the interpolated
    response matrix -- by far the largest array involved.  Building it once
    and handing it to every harmonic balance keeps the inner loop to the
    solve itself.
    """

    def __init__(
        self, resp, voltage, current, vph, harmonics, windows, num_b, hb_kwargs
    ):
        self.resp = resp
        self.voltage = voltage
        self.current = current
        self.harmonics = harmonics
        self.num_b = num_b
        self.hb_kwargs = hb_kwargs
        self.calls = 0
        self.not_converged = 0

        self.cct = EmbeddingCircuit(
            1,
            harmonics,
            vb_npts=voltage.size,
            vb_min=float(voltage[0]),
            vb_max=float(voltage[-1]),
        )
        self.cct.freq[1] = vph
        self.resp_matrix = interpolate_respfn(self.cct, resp, num_b)

        masks = [(voltage >= lo) & (voltage <= hi) for lo, hi in windows]
        self.masks = [m for m in masks if m.sum() > 0]
        if not self.masks:
            raise ValueError(
                "No measured bias points fall inside the requested windows."
            )

    def simulate(self, params):
        """Simulated pumped DC I-V curve for one candidate source.

        Args:
            params (sequence): ``[vt, re1, im1, ...]``.

        Returns:
            ndarray: The simulated DC current on the fit grid.

        """
        vt, zt = _unpack(params, self.harmonics)
        self.cct.vt[:] = 0
        self.cct.zt[:] = 0
        self.cct.vt[1, 1] = vt
        for k, z in enumerate(zt, start=1):
            self.cct.zt[1, k] = z

        vj, _, converged = harmonic_balance(
            self.cct,
            self.resp,
            num_b=self.num_b,
            verbose=False,
            mode="x",
            resp_matrix=self.resp_matrix,
            **self.hb_kwargs,
        )
        if not converged:
            self.not_converged += 1
        return qtcurrent(
            vj,
            self.cct,
            self.resp,
            0.0,
            num_b=self.num_b,
            verbose=False,
            resp_matrix=self.resp_matrix,
        )

    def __call__(self, params):
        """RMS current residual, averaged over the fit windows.

        Args:
            params (sequence): ``[vt, re1, im1, ...]``.

        Returns:
            float: The residual, or ``inf`` if the simulation failed.

        """
        self.calls += 1
        try:
            simulated = self.simulate(params)
        except (ValueError, FloatingPointError):
            return np.inf
        if not np.all(np.isfinite(simulated)):
            return np.inf
        errors = [
            np.sqrt(np.mean((self.current[m] - simulated[m]) ** 2)) for m in self.masks
        ]
        return float(np.mean(errors))


def current_residual(
    resp,
    voltage: np.ndarray,
    current: np.ndarray,
    vph: float,
    vt: float,
    zt,
    windows=None,
    num_b: int = 20,
    hb_kwargs: dict[str, Any] | None = None,
) -> float:
    """RMS current residual of one candidate embedding circuit.

    The same quantity :func:`recover_zemb_current_match` minimises, exposed
    on its own.  Use it to ask how sharply the data actually constrains a
    parameter -- vary one and watch the residual.  If it barely moves, the
    measurement does not determine that parameter, however confidently the
    optimiser reported it.

    It is also the only fair way to compare two fits: each one resamples the
    measured curve its own way, so their reported residuals are not
    comparable until both are re-evaluated here, on one grid.

    Args:
        resp (qpmix.respfn.RespFn): Response function.
        voltage (ndarray): Bias voltage, normalized.
        current (ndarray): Measured pumped current, normalized.
        vph (float): Photon voltage, normalized.
        vt (float): Candidate Thevenin voltage.
        zt (complex or sequence): Candidate embedding impedance, one per
            harmonic.
        windows (sequence, optional): Bias windows to evaluate over.
            Defaults to the whole supplied range.
        num_b (int, optional): Bessel summation limit.  Default is 20.
        hb_kwargs (dict, optional): Extra harmonic-balance options.  Default
            is None.

    Returns:
        float: The RMS residual, or ``inf`` if the simulation failed.

    """
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    zt = np.atleast_1d(np.asarray(zt, dtype=complex))
    if windows is None:
        windows = ((float(voltage.min()), float(voltage.max())),)

    options = {"max_it": 30, "stop_rerror": 1e-4}
    options.update(hb_kwargs or {})
    objective = _Objective(
        resp, voltage, current, vph, zt.size, tuple(windows), num_b, options
    )
    params = [float(vt)]
    for z in zt:
        params += [float(z.real), float(z.imag)]
    return objective(params)


def recover_zemb_current_match(
    resp,
    voltage: np.ndarray,
    current: np.ndarray,
    vph: float,
    harmonics: int = 1,
    method: str = "first_photon",
    fit_range: tuple[float, float] = (0.25, 0.8),
    windows=None,
    guesses="grid",
    num_b: int = 20,
    npts: int | None = None,
    cluster_tol: float = 0.02,
    max_reactance: float = 5.0,
    maxiter: int = 400,
    xatol: float = 1e-6,
    fatol: float = 1e-10,
    hb_kwargs: dict[str, Any] | None = None,
    verbose: bool = False,
    **window_kwargs: Any,
) -> CurrentMatchResult:
    """Recover the embedding circuit by matching the pumped I-V curve.

    Args:
        resp (qpmix.respfn.RespFn): Response function from the unpumped
            curve.
        voltage (ndarray): Measured bias voltage, normalized to the gap
            voltage.
        current (ndarray): Measured pumped current, normalized to the gap
            current.
        vph (float): Photon voltage, normalized to the gap voltage.
        harmonics (int, optional): How many harmonics of the LO to fit an
            embedding impedance for.  Default is 1.
        method (str, optional): Bias-window strategy, one of
            :data:`VOLTAGE_METHODS`.  Default is ``"first_photon"``.
        fit_range (tuple, optional): Fraction of each photon step to keep.
            Default is ``(0.25, 0.8)``.
        windows (sequence, optional): Explicit ``(low, high)`` windows,
            overriding ``method``.  Default is None.
        guesses (str or sequence, optional): ``"grid"`` for the nine-point
            default grid, ``"seeded"`` to start from the voltage-match
            answer plus a few perturbations, or an explicit list of
            parameter vectors.  Default is ``"grid"``.
        num_b (int, optional): Bessel summation limit for the simulation.
            Default is 20.
        npts (int, optional): Points in the uniform grid the fit runs on.
            Defaults to however many measured points fall in the windows,
            capped at 2000.
        cluster_tol (float, optional): Two solutions count as the same when
            every parameter agrees to within this.  Default is 0.02.
        max_reactance (float, optional): Reject solutions whose reactance
            exceeds this.  Default is 5.
        maxiter (int, optional): Simplex iteration limit per run.  Default
            is 400.
        xatol (float, optional): Simplex parameter tolerance.  Default is
            1e-6.
        fatol (float, optional): Simplex residual tolerance.  Default is
            1e-10.
        hb_kwargs (dict, optional): Extra options for
            :func:`qpmix.harmonic_balance.harmonic_balance`.  ``max_it``
            defaults to 30 here rather than 10, because a fit sweeps through
            parameter values far from the solution.  Default is None.
        verbose (bool, optional): Report each run.  Default is False.
        **window_kwargs: Forwarded to :func:`voltage_windows`.

    Returns:
        CurrentMatchResult: The recovered circuit and every candidate.

    Raises:
        ValueError: If the inputs are inconsistent, the harmonic count is
            below one, or no run produced a usable solution.

    """
    if harmonics < 1:
        raise ValueError("Need at least one harmonic.")
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)
    if voltage.shape != current.shape:
        raise ValueError("voltage and current must have the same shape.")

    if windows is None:
        windows = voltage_windows(method, vph, fit_range=fit_range, **window_kwargs)
    windows = tuple((float(lo), float(hi)) for lo, hi in windows)

    # Resample onto a uniform grid spanning the windows: the simulation runs
    # on an EmbeddingCircuit's uniform bias sweep, so the measured curve has
    # to live on the same grid for the residual to mean anything.
    lo = min(w[0] for w in windows)
    hi = max(w[1] for w in windows)
    inside = (voltage >= lo) & (voltage <= hi)
    if inside.sum() < 8:
        raise ValueError(
            f"Only {inside.sum()} measured points fall between {lo:.3f} and "
            f"{hi:.3f} (normalized). Check 'vph' and the window method."
        )
    if npts is None:
        npts = int(min(inside.sum(), 2000))
    grid = np.linspace(lo, hi, npts)
    spline = CubicSpline(voltage, current, extrapolate=False)
    grid_current = spline(grid)

    hb_options = {"max_it": 30, "stop_rerror": 1e-4}
    hb_options.update(hb_kwargs or {})
    objective = _Objective(
        resp, grid, grid_current, vph, harmonics, windows, num_b, hb_options
    )

    starts = _resolve_guesses(
        guesses, harmonics, resp, voltage, current, vph, num_b, verbose
    )

    candidates = []
    for idx, start in enumerate(starts):
        result = minimize(
            objective,
            np.asarray(start, dtype=float),
            method="Nelder-Mead",
            options={"xatol": xatol, "fatol": fatol, "maxiter": maxiter},
        )
        vt, zt = _unpack(result.x, harmonics)
        ok = bool(np.isfinite(result.fun)) and _admissible(vt, zt, max_reactance)
        candidates.append((tuple(start), tuple(result.x), float(result.fun), ok))
        if verbose:
            flag = "ok " if ok else "rej"
            print(
                f"  run {idx + 1}/{len(starts)} [{flag}] "
                f"vt={vt:+.4f} zt={zt[0]:+.4f} err={result.fun:.3e}"
            )

    return _select(candidates, harmonics, cluster_tol, windows, objective)


def _resolve_guesses(guesses, harmonics, resp, voltage, current, vph, num_b, verbose):
    """Turn the ``guesses`` argument into a list of starting vectors.

    Args:
        guesses (str or sequence): See :func:`recover_zemb_current_match`.
        harmonics (int): Number of harmonics.
        resp (qpmix.respfn.RespFn): Response function.
        voltage (ndarray): Measured bias voltage.
        current (ndarray): Measured current.
        vph (float): Photon voltage.
        num_b (int): Bessel summation limit.
        verbose (bool): Report the seed.

    Returns:
        list: Starting parameter vectors.

    Raises:
        ValueError: If ``guesses`` is an unknown string or is empty.

    """
    if isinstance(guesses, str):
        if guesses == "grid":
            return default_guesses(harmonics)
        if guesses == "seeded":
            seed = _voltage_match_seed(resp, voltage, current, vph, num_b)
            if seed is None:
                if verbose:
                    print("  voltage-match seed failed; using the default grid")
                return default_guesses(harmonics)
            vt, zt = seed
            if verbose:
                print(f"  seeded from voltage match: vt={vt:+.4f} zt={zt:+.4f}")

            def build(scale, harmonic_scale=1.0):
                """One start: the seed, with the harmonics scaled."""
                out = [vt, zt.real * scale, zt.imag * scale]
                for _ in range(harmonics - 1):
                    out += [
                        zt.real * scale * harmonic_scale,
                        zt.imag * scale * harmonic_scale,
                    ]
                return out

            starts = [build(1.0), build(0.6), build(1.6)]
            if harmonics > 1:
                # The embedding circuit often presents a very different
                # impedance at the harmonics -- frequently close to a short.
                # Seeding them all at the fundamental's value explores only
                # one corner, and on real data that corner can contain no
                # admissible solution at all.
                starts += [build(1.0, 0.0), build(1.0, 2.0), build(1.0, -1.0)]
            return starts
        raise ValueError(f"Unknown guesses option: {guesses!r}")

    starts = [list(g) for g in guesses]
    if not starts:
        raise ValueError("No starting points were supplied.")
    return starts


def _voltage_match_seed(resp, voltage, current, vph, num_b):
    """Run the fast voltage-match method to seed the optimiser.

    Args:
        resp (qpmix.respfn.RespFn): Response function.
        voltage (ndarray): Measured bias voltage.
        current (ndarray): Measured current.
        vph (float): Photon voltage.
        num_b (int): Bessel summation limit.

    Returns:
        tuple or None: ``(vt, zt)``, or None if the method does not apply.

    """
    from qpmix.exp.tucker import ac_current, recover_alpha
    from qpmix.exp.zemb import recover_zemb

    if vph >= 1.0:
        return None
    lo, hi = voltage_windows("first_photon", vph)[0]
    mask = (voltage >= lo) & (voltage <= hi)
    if mask.sum() < 5:
        return None
    vb, ib = voltage[mask], current[mask]
    alpha = recover_alpha(resp, vb, ib, vph, num_b=num_b)
    vj = alpha * vph
    with np.errstate(divide="ignore", invalid="ignore"):
        zj = vj / ac_current(resp, vb, vph, alpha, num_b=num_b)
    try:
        result = recover_zemb(vj, zj)
    except ValueError:
        return None
    return result.vt, result.zt


def _select(candidates, harmonics, cluster_tol, windows, objective):
    """Choose the solution the independent runs agree on.

    Args:
        candidates (list): ``(start, solution, residual, admissible)``.
        harmonics (int): Number of harmonics.
        cluster_tol (float): Agreement tolerance.
        windows (tuple): The bias windows used.
        objective (_Objective): The objective, for the final simulation.

    Returns:
        CurrentMatchResult: The chosen solution.

    Raises:
        ValueError: If no run produced an admissible solution.

    """
    admissible = [c for c in candidates if c[3]]
    if not admissible:
        raise ValueError(
            f"None of the {len(candidates)} runs produced a physically "
            f"admissible embedding circuit (non-negative resistance, "
            f"positive voltage, bounded reactance). Try a different bias "
            f"window or more starting points."
        )

    keys = [tuple(np.round(np.asarray(c[1]) / cluster_tol)) for c in admissible]
    counts = Counter(keys)
    top_count = max(counts.values())

    if top_count > 1:
        selection = "mode"
        winners = [k for k, n in counts.items() if n == top_count]
        pool = [c for c, k in zip(admissible, keys, strict=True) if k in winners]
    else:
        # Nothing recurred, so agreement cannot guide the choice.
        selection = "best"
        pool = admissible

    best = min(pool, key=lambda c: c[2])
    vt, zt = _unpack(best[1], harmonics)
    occurrences = counts[tuple(np.round(np.asarray(best[1]) / cluster_tol))]

    return CurrentMatchResult(
        vt=vt,
        zt=zt,
        err=best[2],
        occurrences=int(occurrences),
        n_runs=len(candidates),
        n_admissible=len(admissible),
        selection=selection,
        windows=windows,
        candidates=tuple(candidates),
        voltage=objective.voltage,
        current=objective.current,
        simulated=objective.simulate(best[1]),
        n_calls=objective.calls,
        n_not_converged=objective.not_converged,
    )
