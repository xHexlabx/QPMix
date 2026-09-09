"""Benchmark the experimental-data analysis against upstream QMix.

Three parts of :mod:`qpmix.exp` do real numerical work, and all three are
measured here:

* the Tucker-theory sums, evaluated for every Bessel order at once rather
  than in a Python loop;
* drive-level recovery, a safeguarded Newton iteration rather than a fixed
  number of bisection steps -- which also makes it far more accurate;
* the impedance-recovery error surface, one broadcast expression rather
  than a 101 x 201 double loop.

Run with::

    python benchmarks/bench_exp.py
"""

from __future__ import annotations

import sys
import warnings
from timeit import default_timer as timer

import numpy as np

warnings.filterwarnings("ignore")

import qpmix  # noqa: E402
from qpmix.exp.simulate import (  # noqa: E402
    simulate_dciv,
    simulate_embedded_iv,
    simulate_response,
)
from qpmix.exp.tucker import (  # noqa: E402
    ac_current,
    pumped_iv_curve,
    recover_alpha,
)
from qpmix.exp.zemb import error_function, error_surface  # noqa: E402

try:
    import qmix
    import qmix.exp.exp_data as qed
    import qmix.respfn

    HAS_QMIX = True
except ImportError:  # pragma: no cover - optional comparison
    HAS_QMIX = False

VPH = 0.3
NUM_B = 20


def time_it(fn, repeat=5):
    """Best of ``repeat`` runs, after one warm-up call."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = timer()
        fn()
        best = min(best, timer() - t0)
    return best


def row(name, t_ref, t_new, note=""):
    """Print one benchmark row."""
    if t_ref is None:
        print(f"{name:<42s} {'-':>11s} {t_new * 1e3:>11.2f} {'-':>9s}  {note}")
    else:
        print(
            f"{name:<42s} {t_ref * 1e3:>11.2f} {t_new * 1e3:>11.2f} "
            f"{t_ref / t_new:>8.1f}x  {note}"
        )


def header(title):
    """Print a section header."""
    print(f"\n{title}")
    print("-" * 92)
    print(f"{'benchmark':<42s} {'QMix (ms)':>11s} {'QPMix (ms)':>11s} {'speedup':>9s}")


def bench_tucker():
    """The Tucker-theory sums."""
    header("Tucker theory (single-tone currents)")
    resp_new = qpmix.RespFnPolynomial(50, verbose=False)
    resp_ref = qmix.respfn.RespFnPolynomial(50, verbose=False) if HAS_QMIX else None

    for npts in (401, 4001):
        vb = np.linspace(0.05, 1.5, npts)
        alpha = np.full(npts, 1.0)
        t_ref = (
            time_it(
                lambda v=vb, a=alpha: qed._find_pumped_iv_curve(
                    resp_ref, v, VPH, a, num_b=NUM_B
                )
            )
            if HAS_QMIX
            else None
        )
        t_new = time_it(
            lambda v=vb: pumped_iv_curve(resp_new, v, VPH, 1.0, num_b=NUM_B)
        )
        row(f"pumped I-V curve, npts={npts}", t_ref, t_new)

        t_ref = (
            time_it(
                lambda v=vb, a=alpha: qed._find_ac_current(
                    resp_ref, v, VPH, a, num_b=NUM_B
                )
            )
            if HAS_QMIX
            else None
        )
        t_new = time_it(lambda v=vb: ac_current(resp_new, v, VPH, 1.0, num_b=NUM_B))
        row(f"AC current, npts={npts}", t_ref, t_new)


def bench_alpha():
    """Drive-level recovery: speed and accuracy."""
    header("Drive-level recovery (inverting the pumped I-V curve)")
    resp_new = qpmix.RespFnPolynomial(50, verbose=False)
    resp_ref = qmix.respfn.RespFnPolynomial(50, verbose=False) if HAS_QMIX else None

    for npts in (201, 2001):
        step = np.linspace(1 - VPH + 0.25 * VPH, 1 - 0.2 * VPH, npts)
        idc = pumped_iv_curve(resp_new, step, VPH, 1.0, num_b=NUM_B)
        holder = type("D", (), {"resp": resp_ref})() if HAS_QMIX else None
        t_ref = (
            time_it(
                lambda s=step, i=idc, h=holder: qed._find_alpha(
                    h, s, i, VPH, alpha_max=1.5, num_b=NUM_B
                ),
                3,
            )
            if HAS_QMIX
            else None
        )
        t_new = time_it(
            lambda s=step, i=idc: recover_alpha(
                resp_new, s, i, VPH, alpha_max=1.5, num_b=NUM_B
            ),
            3,
        )
        row(f"recover alpha, npts={npts}", t_ref, t_new)

    print("\n  accuracy (true alpha recovered from a noise-free curve):")
    step = np.linspace(1 - VPH + 0.25 * VPH, 1 - 0.2 * VPH, 201)
    for alpha in (0.2, 0.8, 1.5, 2.2):
        idc = pumped_iv_curve(resp_new, step, VPH, alpha, num_b=NUM_B)
        err_new = np.abs(
            recover_alpha(resp_new, step, idc, VPH, alpha_max=1.5, num_b=NUM_B) - alpha
        ).max()
        if HAS_QMIX:
            holder = type("D", (), {"resp": resp_ref})()
            err_ref = np.abs(
                qed._find_alpha(holder, step, idc, VPH, alpha_max=1.5, num_b=NUM_B)
                - alpha
            ).max()
            print(
                f"    alpha={alpha:4.1f}:  QMix {err_ref:.2e}   "
                f"QPMix {err_new:.2e}   ({err_ref / err_new:,.0f}x closer)"
            )
        else:
            print(f"    alpha={alpha:4.1f}:  QPMix {err_new:.2e}")


def bench_zemb():
    """The impedance-recovery error surface."""
    header("Impedance recovery (error surface over 101 x 201 impedances)")
    rng = np.random.default_rng(0)
    zr = np.linspace(0, 1, 101)
    zi = np.linspace(-1, 1, 201)

    def loop(v, z):
        """The double loop QMix uses, for comparison."""
        out = np.empty((zr.size, zi.size))
        for i in range(zr.size):
            for j in range(zi.size):
                out[i, j] = error_function(v, z, complex(zr[i], zi[j]))
        return out

    for n in (50, 500):
        zj = rng.uniform(0.3, 0.8, n) + 1j * rng.uniform(-0.3, 0.3, n)
        vj = 0.5 * zj / (0.3 - 0.4j + zj)
        t_ref = time_it(lambda v=vj, z=zj: loop(v, z), 1)
        t_new = time_it(lambda v=vj, z=zj: error_surface(v, z, zr, zi), 3)
        row(f"error surface, {n} bias points", t_ref, t_new, "(loop vs broadcast)")


def bench_end_to_end():
    """A complete analysis of one pumped measurement."""
    header("End-to-end analysis of one measurement")
    from qpmix.exp import DCData, PumpedData

    resp = simulate_response()
    raw_dc = simulate_dciv()
    raw_pumped = simulate_embedded_iv(vt=0.35, zt=0.4 - 0.3j, freq=230.0, resp=resp)

    t_dc = time_it(lambda: DCData(raw_dc, verbose=False), 2)
    dciv = DCData(raw_dc, verbose=False)
    t_pumped = time_it(
        lambda: PumpedData(raw_pumped, dciv, freq=230.0, verbose=False), 2
    )
    row("DCData (import + response functions)", None, t_dc)
    row("PumpedData (import + impedance recovery)", None, t_pumped)

    pumped = PumpedData(raw_pumped, dciv, freq=230.0, verbose=False)
    print(
        f"\n  recovered zt = {pumped.zt:+.4f} (true +0.4000-0.3000j), "
        f"vt = {pumped.vt:.4f} (true 0.3500)"
    )


def main() -> int:
    print("QPMix experimental-analysis benchmark")
    print(f"  qpmix   {qpmix.__version__}")
    print(f"  qmix    {'available' if HAS_QMIX else 'NOT INSTALLED'}")
    if HAS_QMIX:
        bench_tucker()
        bench_alpha()
    bench_zemb()
    bench_end_to_end()
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
