"""Calibrate the direct-vs-FFT cost model in qpmix.qtcurrent.

The current summation (Kittara Eqn. 5.25) can be evaluated two ways:

* ``direct`` -- one blocked, thread-parallel loop nest per index tuple, so
  the cost is proportional to how many output frequencies are wanted;
* ``fft``    -- one pair of multi-dimensional transforms that produces the
  coefficients for *every* index tuple at once, so the cost is independent
  of how many are wanted.

``qtcurrent(method="auto")`` picks between them with the model in
:func:`qpmix.qtcurrent._select_method`, tuned by
:data:`qpmix.qtcurrent.FFT_COST_FACTOR`.  This script measures both paths
across a sweep and reports whether the model's prediction matches the
measured winner.

Run with::

    python benchmarks/bench_method.py
"""

from __future__ import annotations

import itertools
import warnings
from timeit import default_timer as timer

import numpy as np

warnings.filterwarnings("ignore")

import qpmix  # noqa: E402
from qpmix.qtcurrent import (  # noqa: E402
    _matching_tuples,
    _needed_tuples,
    _select_method,
)

#: How much slower than optimal the model's choice may be before it counts
#: as a real mismatch rather than a tie.
TOLERANCE = 0.15

RESP = qpmix.RespFnPolynomial(50, verbose=False)


def case(num_f, num_p, num_b, npts, n_out):
    """Time both paths for one configuration and check the model's choice."""
    cct = qpmix.EmbeddingCircuit(
        num_f, num_p, vb_npts=npts, vb_max=2, vgap=2.8e-3, rn=14.0
    )
    for f in range(1, num_f + 1):
        cct.freq[f] = (0.30, 0.32, 0.35, 0.38)[f - 1]

    rng = np.random.default_rng(0)
    vj = cct.initialize_vj()
    for f in range(1, num_f + 1):
        for p in range(1, num_p + 1):
            mag = rng.uniform(0.05, 0.4, npts) / p
            vj[f, p] = mag * np.exp(1j * rng.uniform(-np.pi, np.pi, npts))

    span = range(-num_p, num_p + 1)
    freqs = sorted(
        {
            round(float(np.dot(t, cct.freq[1 : num_f + 1])), 4)
            for t in itertools.product(span, repeat=num_f)
        }
    )
    fl = [f for f in freqs if f >= 0][:n_out]

    resp_matrix = qpmix.interpolate_respfn(cct, RESP, num_b)

    def time(method, repeat=3):
        qpmix.qtcurrent(
            vj,
            cct,
            RESP,
            fl,
            num_b,
            verbose=False,
            method=method,
            resp_matrix=resp_matrix,
        )
        t0 = timer()
        for _ in range(repeat):
            qpmix.qtcurrent(
                vj,
                cct,
                RESP,
                fl,
                num_b,
                verbose=False,
                method=method,
                resp_matrix=resp_matrix,
            )
        return (timer() - t0) / repeat * 1e3

    t_direct, t_fft = time("direct"), time("fft")
    tuples = _matching_tuples(np.round(fl, 4), cct.freq, num_f, num_p)
    predicted = _select_method("auto", num_f, num_p, (num_b,) * num_f, npts, tuples)
    measured = "fft" if t_fft < t_direct else "direct"

    # A "wrong" choice only matters if it actually costs time.
    chosen_time = t_fft if predicted == "fft" else t_direct
    penalty = chosen_time / min(t_direct, t_fft) - 1.0
    ok = penalty <= TOLERANCE
    verdict = "OK" if measured == predicted else f"tie ({penalty:+.0%})"
    if not ok:
        verdict = f"MISMATCH ({penalty:+.0%})"

    print(
        f"f={num_f} p={num_p:2d} nb={num_b:2d} npts={npts} "
        f"tuples={len(_needed_tuples(tuples)):3d} | "
        f"direct={t_direct:8.2f} ms  fft={t_fft:8.2f} ms | "
        f"measured={measured:6s} predicted={predicted:6s} {verdict}"
    )
    return ok


def main() -> int:
    print(f"backend: {qpmix.get_config()}\n")
    cases = [
        # A normal simulation asks for only a handful of output frequencies.
        (1, 1, 15, 401, 2),
        (1, 2, 15, 401, 3),
        (1, 3, 15, 401, 4),
        (1, 4, 15, 401, 5),
        (1, 8, 15, 401, 9),
        (1, 20, 15, 401, 21),
        (2, 1, 15, 401, 3),
        (2, 2, 15, 401, 5),
        (2, 3, 11, 401, 7),
        # ...but a full intermodulation sweep asks for many.
        (2, 2, 11, 401, 12),
        (2, 3, 11, 401, 24),
        (2, 4, 11, 401, 40),
        (2, 5, 11, 401, 60),
        (3, 1, 9, 201, 4),
    ]
    agree = sum(case(*c) for c in cases)
    print(
        f"\nthe model's choice is within {TOLERANCE:.0%} of optimal "
        f"in {agree}/{len(cases)} cases"
    )
    return 0 if agree == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
