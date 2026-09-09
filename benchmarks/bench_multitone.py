"""Scaling of the common-grid engine with the number of tones.

Two things are measured:

1. **Grid vs multi-dimensional**, for the 1-4 tones both engines can do.
   They compute the same physics, so this is a like-for-like comparison.
2. **Scaling past four tones**, which only the grid engine can reach at
   all.  The multi-dimensional column shows what the direct method *would*
   have needed.

Run with::

    python benchmarks/bench_multitone.py
    python benchmarks/bench_multitone.py --quick
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

#: Tones sit on this grid, so the multipliers stay small.
SPACING = 0.02
FIRST = 0.20


def build(num_f, npts, num_b, seed=0):
    """An ``num_f``-tone circuit with a fixed random drive."""
    cct = qpmix.EmbeddingCircuit(num_f, 1, vb_npts=npts, vb_max=2)
    for f in range(1, num_f + 1):
        cct.freq[f] = round(FIRST + SPACING * (f - 1), 6)
    rng = np.random.default_rng(seed)
    vj = cct.initialize_vj()
    for f in range(1, num_f + 1):
        # Fixed total drive power, spread over the tones.
        mag = 0.5 / np.sqrt(num_f)
        vj[f, 1] = mag * np.exp(1j * rng.uniform(-np.pi, np.pi, npts))
    return cct, vj


def time_it(fn, repeat=3):
    """Best of ``repeat`` runs, after one warm-up call."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = timer()
        fn()
        best = min(best, timer() - t0)
    return best


def bench_head_to_head(npts, num_b):
    """Grid vs multi-dimensional, over the tone counts both support.

    Note that ``method="auto"`` deliberately stays on the multi-dimensional
    engine at four tones and below, even where the grid is faster: the grid
    also sums intermodulation products that the multi-dimensional engine
    truncates at +/-num_p, so switching automatically would change results,
    not just run times.  The grid is an opt-in.
    """
    print("\nGrid vs multi-dimensional engine (grid is opt-in below 5 tones)")
    print("-" * 92)
    print(
        f"{'tones':>5s} {'num_k':>7s} {'grid entries':>14s} "
        f"{'direct entries':>16s} {'grid (ms)':>10s} {'direct (ms)':>12s} "
        f"{'speed-up':>9s}"
    )
    resp = qpmix.RespFnPolynomial(50, verbose=False)
    for num_f in (1, 2, 3, 4):
        cct, vj = build(num_f, npts, num_b)
        grid = ToneGrid.from_circuit(cct, num_b=num_b)
        fl = [0.0] + [round(float(cct.freq[f]), 4) for f in range(1, num_f + 1)]

        t_grid = time_it(
            lambda c=cct, v=vj, f=fl: qpmix.qtcurrent(
                v, c, resp, f, num_b=num_b, verbose=False, method="grid"
            ),
            2,
        )
        t_direct = time_it(
            lambda c=cct, v=vj, f=fl: qpmix.qtcurrent(
                v, c, resp, f, num_b=num_b, verbose=False, method="direct"
            ),
            2,
        )
        grid_entries = grid.entries * npts
        direct_entries = (2 * num_b + 1) ** num_f * npts
        print(
            f"{num_f:>5d} {grid.num_k:>7,d} {grid_entries:>14,d} "
            f"{direct_entries:>16,d} {t_grid * 1e3:>10.2f} "
            f"{t_direct * 1e3:>12.2f} {t_direct / t_grid:>8.1f}x"
        )


def bench_scaling(npts, num_b, tone_counts):
    """How the grid engine scales past the four-tone wall."""
    print("\nScaling of the grid engine (method='grid' forced throughout)")
    print("-" * 92)
    print(
        f"{'tones':>5s} {'num_k':>8s} {'grid entries':>14s} {'grid MB':>9s} "
        f"{'qtcurrent (ms)':>15s} | {'direct would need':>22s}"
    )
    resp = qpmix.RespFnPolynomial(50, verbose=False)
    for num_f in tone_counts:
        cct, vj = build(num_f, npts, num_b)
        grid = ToneGrid.from_circuit(cct, num_b=num_b)
        fl = [0.0] + [round(float(cct.freq[f]), 4) for f in range(1, num_f + 1)]
        t = time_it(
            lambda c=cct, v=vj, f=fl: qpmix.qtcurrent(
                v, c, resp, f, num_b=num_b, verbose=False, method="grid"
            ),
            2,
        )
        entries = grid.entries * npts
        would = float((2 * num_b + 1) ** num_f) * npts
        would_str = (
            f"{would * 16 / 1024**3:,.0f} GB"
            if would * 16 > 1024**3
            else f"{would * 16 / 1024**2:,.0f} MB"
        )
        print(
            f"{num_f:>5d} {grid.num_k:>8,d} {entries:>14,d} "
            f"{entries * 16 / 1024**2:>9.1f} {t * 1e3:>15.2f} | "
            f"{would_str:>22s}"
        )


def bench_harmonic_balance(npts, num_b, tone_counts):
    """A full many-tone simulation, not just one current evaluation."""
    print("\nHarmonic balance with many tones")
    print("-" * 92)
    print(f"{'tones':>5s} {'signals':>8s} {'newton (s)':>12s} {'broyden (s)':>12s}")
    resp = qpmix.RespFnPolynomial(50, verbose=False)
    for num_f in tone_counts:

        def make(num_f=num_f):
            cct = qpmix.EmbeddingCircuit(num_f, 1, vb_npts=npts, vb_max=2)
            for f in range(1, num_f + 1):
                cct.freq[f] = round(FIRST + SPACING * (f - 1), 6)
                cct.vt[f, 1] = 0.3 / np.sqrt(num_f)
                cct.zt[f, 1] = 0.3 - 0.3j
            return cct

        t_n = time_it(
            lambda: qpmix.harmonic_balance(
                make(),
                resp,
                num_b=num_b,
                verbose=False,
                jacobian="newton",
                stop_rerror=1e-4,
            ),
            1,
        )
        t_b = time_it(
            lambda: qpmix.harmonic_balance(
                make(),
                resp,
                num_b=num_b,
                verbose=False,
                jacobian="broyden",
                stop_rerror=1e-4,
            ),
            1,
        )
        print(f"{num_f:>5d} {num_f:>8d} {t_n:>12.2f} {t_b:>12.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="smaller sizes")
    args = ap.parse_args()

    npts = 101 if args.quick else 401
    num_b = 5 if args.quick else 9
    scaling = (1, 2, 4, 6, 8) if args.quick else (1, 2, 3, 4, 6, 8, 12, 16, 24)
    hb = (2, 4, 8) if args.quick else (2, 4, 8, 12, 16)

    print("QPMix multi-tone benchmark")
    print(f"  backend: {qpmix.get_config()}")
    print(f"  npts={npts}, num_b={num_b}, tone spacing={SPACING}")

    bench_head_to_head(npts, num_b)
    bench_scaling(npts, num_b, scaling)
    bench_harmonic_balance(npts, num_b, hb)
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
