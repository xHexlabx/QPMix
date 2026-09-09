"""How many tones before a band behaves like a continuum?

A broadband signal of bandwidth ``W`` centred on ``f0`` is represented by
``N`` equally spaced tones sharing a fixed total drive power.  As ``N``
grows the comb approaches the continuous band, so the simulated pumped
DC I-V curve should stop changing.  This script finds where it stops.

Getting the limit right
-----------------------

Only the *random-phase* comb has a continuum limit.  Each tone carries
amplitude ``alpha/sqrt(N)``, so the waveform's RMS is fixed while, by the
central limit theorem, its amplitude distribution tends to a Gaussian with
that variance -- a band-limited noise process.  A single realization is one
sample path and keeps changing with ``N``; the *ensemble average* is what
converges, so this script averages over ``--realizations`` draws and also
reports the spread between them, which is the statistical floor below which
convergence cannot be measured.

The deterministic combs (``--phase chirp`` and ``--phase flat``) are
offered for comparison and deliberately do **not** converge: they hold the
RMS fixed while the crest factor grows with ``N``, so the waveform keeps
changing character no matter how finely the band is sampled.

Two ceilings
------------

* a **physics ceiling** -- the ``N`` past which the ensemble answer no
  longer moves, so the band can be treated as a continuum;
* a **cost ceiling** -- at fixed bandwidth the tone spacing shrinks like
  ``1/N``, so the common grid must get finer and ``num_k`` grows roughly
  like ``N**2``.  That, not the tone count, is what eventually stops you.
  (The multi-dimensional engine would instead scale as
  ``(2*num_b+1)**N``, which stops you at four.)

Run with::

    python benchmarks/study_continuum.py
    python benchmarks/study_continuum.py --realizations 32 --alpha 1.5
    python benchmarks/study_continuum.py --phase chirp --realizations 1
"""

from __future__ import annotations

import argparse
import sys
import warnings
from timeit import default_timer as timer

import numpy as np

warnings.filterwarnings("ignore")

import qpmix  # noqa: E402
from qpmix.multitone import ToneGrid  # noqa: E402

#: Tone counts for which the comb stays exactly commensurate on a coarse
#: grid: the spacing divides the band evenly *and* divides the band centre.
#: Off-ladder counts still work, they just need a finer grid.
DEFAULT_COUNTS = (1, 3, 5, 7, 11, 13, 21, 31)


def tone_frequencies(f0: float, bandwidth: float, n: int) -> list[float]:
    """``n`` equally spaced tones spanning the band, as exact decimals.

    Args:
        f0 (float): Band centre, normalized.
        bandwidth (float): Band width, normalized.
        n (int): Number of tones.

    Returns:
        list: The tone frequencies.

    """
    if n == 1:
        return [round(f0, 9)]
    lo = f0 - bandwidth / 2
    step = bandwidth / (n - 1)
    return [round(lo + j * step, 9) for j in range(n)]


def tone_phases(freqs, mode: str, f0: float, bandwidth: float, seed: int):
    """Phase of each tone.

    The point of the study is convergence in ``n``, so the phases must
    describe *one* underlying waveform sampled ever more finely.  Drawing
    fresh random phases for every ``n`` would compare different signals and
    the differences would never settle.

    Args:
        freqs (list): Tone frequencies.
        mode (str): ``"chirp"`` for a deterministic quadratic phase law,
            ``"flat"`` for an impulse-like comb, or ``"random"`` for one
            realization of a random-phase signal.
        f0 (float): Band centre.
        bandwidth (float): Band width.
        seed (int): Seed, used by ``"random"``.

    Returns:
        ndarray: One phase per tone, in radians.

    """
    x = (np.asarray(freqs) - (f0 - bandwidth / 2)) / bandwidth
    if mode == "chirp":
        # A smooth function of frequency: refining the comb samples the
        # same continuous waveform more finely.
        return 2 * np.pi * 4.0 * x**2
    if mode == "flat":
        return np.zeros_like(x)
    if mode == "random":
        return np.random.default_rng(seed).uniform(-np.pi, np.pi, x.size)
    raise ValueError(f"Unknown phase mode: {mode!r}")


def ensemble(freqs, args, resp):
    """Ensemble-averaged pumped DC I-V curve for one tone count.

    Args:
        freqs (list): Tone frequencies.
        args (argparse.Namespace): Parsed command-line options.
        resp (qpmix.respfn.RespFn): The response function.

    Returns:
        tuple: ``(mean, spread, grid, seconds)`` where ``spread`` is the
        standard error of the mean over realizations, or all-None if no
        grid fits the budget.

    """
    reps = args.realizations if args.phase == "random" else 1
    curves, grid, total = [], None, 0.0
    for r in range(reps):
        phases = tone_phases(
            freqs, args.phase, args.centre, args.bandwidth, args.seed + r
        )
        idc, grid, secs = simulate(
            freqs,
            phases,
            args.alpha,
            resp,
            args.npts,
            args.num_b,
            args.max_num_k,
        )
        if idc is None:
            return None, None, None, None
        curves.append(idc)
        total += secs
    stack = np.asarray(curves)
    mean = stack.mean(axis=0)
    spread = stack.std(axis=0).max() / np.sqrt(reps) if reps > 1 else 0.0
    return mean, spread, grid, total


def simulate(freqs, phases, alpha_total, resp, npts, num_b, max_num_k):
    """Pumped DC I-V curve for a comb of tones at fixed total drive power.

    Args:
        freqs (list): Tone frequencies, normalized.
        phases (ndarray): Phase of each tone.
        alpha_total (float): Total drive level, ``sqrt(sum alpha_f**2)``.
        resp (qpmix.respfn.RespFn): The response function.
        npts (int): Bias points.
        num_b (int): Summation limit.
        max_num_k (int): Budget for the common grid.

    Returns:
        tuple: ``(idc, grid, seconds)``, or ``(None, None, None)`` if no
        grid fits the budget.

    """
    num_f = len(freqs)
    cct = qpmix.EmbeddingCircuit(num_f, 1, vb_npts=npts, vb_max=2)
    for f in range(1, num_f + 1):
        cct.freq[f] = freqs[f - 1]

    try:
        grid = ToneGrid.from_circuit(
            cct, num_b=num_b, max_num_k=max_num_k, max_entries=0
        )
    except ValueError:
        return None, None, None

    # Fixed total drive power, split equally across the band.
    vj = cct.initialize_vj()
    for f in range(1, num_f + 1):
        alpha_f = alpha_total / np.sqrt(num_f)
        vj[f, 1] = alpha_f * cct.freq[f] * np.exp(1j * phases[f - 1])

    t0 = timer()
    idc = qpmix.qtcurrent(
        vj, cct, resp, 0.0, num_b=num_b, verbose=False, method="grid", grid=grid
    )
    return idc, grid, timer() - t0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--centre", type=float, default=0.30, help="band centre")
    ap.add_argument("--bandwidth", type=float, default=0.06, help="band width")
    ap.add_argument("--alpha", type=float, default=1.0, help="total drive level")
    ap.add_argument("--npts", type=int, default=101, help="bias points")
    ap.add_argument("--num-b", type=int, default=9, help="summation limit")
    ap.add_argument(
        "--phase",
        default="random",
        choices=("chirp", "flat", "random"),
        help="phase law across the band (only 'random' has a continuum limit)",
    )
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    ap.add_argument(
        "--realizations", type=int, default=12, help="ensemble size for 'random'"
    )
    ap.add_argument("--max-num-k", type=int, default=60_000, help="grid cost budget")
    ap.add_argument(
        "--tones", type=int, nargs="*", default=None, help="tone counts to try"
    )
    args = ap.parse_args()

    counts = args.tones or list(DEFAULT_COUNTS)
    resp = qpmix.RespFnPolynomial(50, verbose=False)
    reps = args.realizations if args.phase == "random" else 1

    print("Continuum convergence study")
    print(f"  backend    : {qpmix.get_config()}")
    print(
        f"  band       : {args.centre - args.bandwidth / 2:.4f} .. "
        f"{args.centre + args.bandwidth / 2:.4f} (normalized)"
    )
    print(f"  phase law  : {args.phase} ({reps} realization(s))")
    print(f"  total drive: alpha = {args.alpha}")
    print(f"  bias points: {args.npts}, num_b = {args.num_b}")
    print(f"  grid budget: num_k <= {args.max_num_k:,}\n")

    results = []
    print("-" * 88)
    print(
        f"{'N':>4s} {'num_k':>9s} {'grid MB':>9s} {'time (ms)':>10s} "
        f"{'ens. spread':>12s}  {'change vs previous N':>21s}"
    )
    print("-" * 88)
    previous = None
    for n in counts:
        freqs = tone_frequencies(args.centre, args.bandwidth, n)
        idc, spread, grid, secs = ensemble(freqs, args, resp)
        if idc is None:
            print(f"{n:>4d} {'-':>9s}  (no grid within budget -- cost ceiling)")
            continue
        mb = grid.entries * args.npts * 16 / 1024**2
        change = "-" if previous is None else f"{np.abs(idc - previous).max():.3e}"
        print(
            f"{n:>4d} {grid.num_k:>9,d} {mb:>9.1f} {secs * 1e3:>10.1f} "
            f"{spread:>12.3e}  {change:>21s}"
        )
        results.append((n, idc, grid.num_k, spread))
        previous = idc

    if len(results) < 3:
        print("\nNot enough points to judge convergence.")
        return 1

    n_ref, idc_ref = results[-1][0], results[-1][1]
    scale = np.abs(idc_ref).max()
    floor = max(r[3] for r in results) / scale
    target = max(1e-3, 3 * floor)

    print("\n" + "-" * 88)
    print(f"Convergence against the N = {n_ref} reference (relative, max-norm)")
    print("-" * 88)
    ceiling = None
    for n, idc, _, _ in results[:-1]:
        rel = np.abs(idc - idc_ref).max() / scale
        mark = ""
        if ceiling is None and rel < target:
            ceiling = n
            mark = f"  <-- within {target:.1e} of the continuum"
        print(f"  N = {n:>3d}: {rel:.3e}{mark}")
    print(f"\n  statistical floor (ensemble spread / scale): {floor:.3e}")
    print(f"  convergence target used                    : {target:.3e}")

    ns = np.array([r[0] for r in results if r[0] > 1], dtype=float)
    ks = np.array([r[2] for r in results if r[0] > 1], dtype=float)
    print("\n" + "-" * 88)
    if ns.size >= 3:
        q = np.polyfit(np.log(ns), np.log(ks), 1)[0]
        print(f"Cost scaling: num_k ~ N^{q:.2f} at fixed bandwidth")
    print("(The multi-dimensional engine would instead scale as (2*num_b+1)^N.)")

    print("")
    if ceiling is None:
        print(
            f"No tone count below {n_ref} reached the target: the band is not "
            f"a continuum yet at this bandwidth and drive level. Re-run with "
            f"larger --tones, or more --realizations to lower the "
            f"statistical floor."
        )
    else:
        print(
            f"Physics ceiling: {ceiling} tones already reproduce the N = "
            f"{n_ref} answer to within the target, so beyond that the band "
            f"can be treated as a continuum."
        )
    print(
        "Cost ceiling: at fixed bandwidth the grid spacing shrinks like 1/N, "
        "so num_k keeps growing after the physics has stopped changing -- "
        "that, not the tone count itself, is what eventually stops you."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
