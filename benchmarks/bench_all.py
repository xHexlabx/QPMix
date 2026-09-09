"""Benchmark QPMix against the upstream QMix package, stage by stage.

Run with::

    python benchmarks/bench_all.py            # full sweep
    python benchmarks/bench_all.py --quick    # small sizes only

QMix must be importable for the comparison columns; if it is not, only the
QPMix timings are reported.

Every QPMix timing excludes JIT compilation: each case is run once to warm
the cache before it is timed.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from timeit import default_timer as timer

import numpy as np

warnings.filterwarnings("ignore")

import qpmix  # noqa: E402
from qpmix.phase_factor import _as_nb_tuple  # noqa: E402
from qpmix.qtcurrent import _matching_tuples, _select_method  # noqa: E402

try:
    import qmix

    HAS_QMIX = True
except ImportError:  # pragma: no cover - optional comparison
    HAS_QMIX = False


def time_it(fn, repeat: int = 3) -> float:
    """Run ``fn`` once to warm up, then return the best of ``repeat`` runs."""
    fn()
    best = float("inf")
    for _ in range(repeat):
        t0 = timer()
        fn()
        best = min(best, timer() - t0)
    return best


def row(name: str, t_ref: float | None, t_new: float, note: str = "") -> None:
    """Print one benchmark row."""
    if t_ref is None:
        print(f"{name:<44s} {'-':>11s} {t_new * 1e3:>11.2f} {'-':>9s}  {note}")
    else:
        print(
            f"{name:<44s} {t_ref * 1e3:>11.2f} {t_new * 1e3:>11.2f} "
            f"{t_ref / t_new:>8.1f}x  {note}"
        )


def header(title: str) -> None:
    print(f"\n{title}")
    print("-" * 92)
    print(f"{'benchmark':<44s} {'QMix (ms)':>11s} {'QPMix (ms)':>11s} {'speedup':>9s}")


def bench_kktrans() -> None:
    header("Kramers-Kronig transform (mathfn.kktrans)")
    for npts in (7001, 14001, 70001):
        v = np.linspace(-35, 35, npts)
        i = np.where(np.abs(v) < 1, 0.0, v)
        t_ref = (
            time_it(lambda v=v, i=i: qmix.mathfn.kktrans.kk_trans(v, i, 50))
            if HAS_QMIX
            else None
        )
        t_new = time_it(lambda v=v, i=i: qpmix.mathfn.kktrans.kk_trans(v, i, 50))
        row(f"kk_trans, npts={npts}", t_ref, t_new)


def bench_respfn() -> None:
    header("Response function construction and evaluation")
    t_ref = (
        time_it(lambda: qmix.respfn.RespFnPolynomial(50, verbose=False), 1)
        if HAS_QMIX
        else None
    )
    t_new = time_it(lambda: qpmix.RespFnPolynomial(50, verbose=False), 1)
    row("build RespFnPolynomial(50)", t_ref, t_new, "one-off cost")

    r_ref = qmix.respfn.RespFnPolynomial(50, verbose=False) if HAS_QMIX else None
    r_new = qpmix.RespFnPolynomial(50, verbose=False)
    for n in (10**5, 10**6, 10**7):
        v = np.linspace(-3, 3, n)
        t_ref = time_it(lambda v=v: r_ref(v)) if HAS_QMIX else None
        t_new = time_it(lambda v=v: r_new(v))
        row(f"evaluate resp(v), {n:,} points", t_ref, t_new)


def bench_phase_factor() -> None:
    header("Phase-factor coefficients (Kittara Eqns. 5.7 + 5.12)")
    rng = np.random.default_rng(0)
    num_f, num_b, npts = 2, 15, 401
    freq = np.array([0.0, 0.30, 0.32])
    for num_p in (1, 2, 3, 4):
        vj = np.zeros((num_f + 1, num_p + 1, npts), dtype=complex)
        for f in range(1, num_f + 1):
            for p in range(1, num_p + 1):
                mag = rng.uniform(0.05, 0.45, npts) / p
                vj[f, p] = mag * np.exp(1j * rng.uniform(-np.pi, np.pi, npts))
        args = (vj, freq, num_f, num_p, num_b)
        t_ref = (
            time_it(lambda a=args: qmix.qtcurrent.calculate_phase_factor_coeff(*a))
            if HAS_QMIX
            else None
        )
        t_new = time_it(lambda a=args: qpmix.calculate_phase_factor_coeff(*a))
        row(f"num_f=2, num_p={num_p}, num_b={num_b}, npts={npts}", t_ref, t_new)


def bench_respmatrix(sizes) -> None:
    header("Response matrix interpolation (qtcurrent.interpolate_respfn)")
    for num_f, num_b, npts in sizes:
        c_ref = _circuit(qmix, num_f, 1, npts) if HAS_QMIX else None
        c_new = _circuit(qpmix, num_f, 1, npts)
        r_ref = qmix.respfn.RespFnPolynomial(50, verbose=False) if HAS_QMIX else None
        r_new = qpmix.RespFnPolynomial(50, verbose=False)
        t_ref = (
            time_it(
                lambda c=c_ref, r=r_ref, nb=num_b: qmix.qtcurrent.interpolate_respfn(
                    c, r, nb
                ),
                2,
            )
            if HAS_QMIX
            else None
        )
        t_new = time_it(
            lambda c=c_new, r=r_new, nb=num_b: qpmix.interpolate_respfn(c, r, nb), 2
        )
        size = (2 * num_b + 1) ** num_f * npts
        row(
            f"num_f={num_f}, num_b={num_b}, npts={npts}",
            t_ref,
            t_new,
            f"{size:,} values",
        )


def _circuit(pkg, num_f, num_p, npts):
    c = pkg.circuit.EmbeddingCircuit(
        num_f, num_p, vb_npts=npts, vb_max=2, vgap=2.8e-3, rn=14.0
    )
    for f in range(1, num_f + 1):
        c.freq[f] = (0.30, 0.32, 0.35, 0.38)[f - 1]
    return c


def bench_qtcurrent(sizes) -> None:
    header("Full qtcurrent call (all stages)")
    rng = np.random.default_rng(1)
    for num_f, num_p, num_b, npts in sizes:
        c_ref = _circuit(qmix, num_f, num_p, npts) if HAS_QMIX else None
        c_new = _circuit(qpmix, num_f, num_p, npts)
        r_ref = qmix.respfn.RespFnPolynomial(50, verbose=False) if HAS_QMIX else None
        r_new = qpmix.RespFnPolynomial(50, verbose=False)
        vj = np.zeros((num_f + 1, num_p + 1, npts), dtype=complex)
        for f in range(1, num_f + 1):
            for p in range(1, num_p + 1):
                mag = rng.uniform(0.05, 0.45, npts) / p
                vj[f, p] = mag * np.exp(1j * rng.uniform(-np.pi, np.pi, npts))
        fl = [0.0] + [
            round(c_new.freq[f] * p, 4)
            for f in range(1, num_f + 1)
            for p in range(1, num_p + 1)
        ]
        ref_args = (vj, c_ref, r_ref, fl, num_b)
        new_args = (vj, c_new, r_new, fl, num_b)
        t_ref = (
            time_it(lambda a=ref_args: qmix.qtcurrent.qtcurrent(*a, verbose=False), 2)
            if HAS_QMIX
            else None
        )
        t_new = time_it(lambda a=new_args: qpmix.qtcurrent(*a, verbose=False), 2)
        chosen = _select_method(
            "auto",
            num_f,
            num_p,
            _as_nb_tuple(num_b, num_f),
            npts,
            _matching_tuples(np.round(fl, 4), c_new.freq, num_f, num_p),
        )
        row(
            f"num_f={num_f}, num_p={num_p}, num_b={num_b}, npts={npts}",
            t_ref,
            t_new,
            f"auto->{chosen}",
        )


def bench_harmonic_balance(sizes) -> None:
    header("Harmonic balance (full simulation)")
    for num_f, num_p, num_b, npts in sizes:

        def make(pkg, num_f=num_f, num_p=num_p, npts=npts):
            c = _circuit(pkg, num_f, num_p, npts)
            for f in range(1, num_f + 1):
                for p in range(1, num_p + 1):
                    c.vt[f, p] = 0.5 / p
                    c.zt[f, p] = 0.3 - 0.3j
            return c

        r_ref = qmix.respfn.RespFnPolynomial(50, verbose=False) if HAS_QMIX else None
        r_new = qpmix.RespFnPolynomial(50, verbose=False)
        t_ref = (
            time_it(
                lambda r=r_ref, nb=num_b: qmix.harmonic_balance.harmonic_balance(
                    make(qmix), r, num_b=nb, verbose=False
                ),
                1,
            )
            if HAS_QMIX
            else None
        )
        t_nwt = time_it(
            lambda r=r_new, nb=num_b: qpmix.harmonic_balance(
                make(qpmix), r, num_b=nb, verbose=False, jacobian="newton"
            ),
            1,
        )
        t_bry = time_it(
            lambda r=r_new, nb=num_b: qpmix.harmonic_balance(
                make(qpmix), r, num_b=nb, verbose=False, jacobian="broyden"
            ),
            1,
        )
        row(f"num_f={num_f}, num_p={num_p}, npts={npts} [newton]", t_ref, t_nwt)
        row(f"num_f={num_f}, num_p={num_p}, npts={npts} [broyden]", t_ref, t_bry)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="small sizes only")
    args = ap.parse_args()

    print("QPMix benchmark suite")
    print(f"  qpmix   {qpmix.__version__}   backend: {qpmix.get_config()}")
    print(f"  qmix    {'available' if HAS_QMIX else 'NOT INSTALLED (no comparison)'}")

    if args.quick:
        rm = [(1, 15, 201), (2, 15, 201)]
        qt = [(1, 1, 15, 201), (2, 1, 15, 201)]
        hb = [(1, 1, 11, 201), (2, 1, 11, 201)]
    else:
        rm = [(1, 15, 401), (2, 15, 401), (3, 9, 401)]
        qt = [
            (1, 1, 15, 401),
            (1, 3, 15, 401),
            (2, 1, 15, 401),
            (2, 2, 15, 401),
            (3, 1, 9, 401),
        ]
        hb = [(1, 1, 15, 401), (2, 1, 15, 401), (2, 2, 11, 201)]

    bench_kktrans()
    bench_respfn()
    bench_phase_factor()
    bench_respmatrix(rm)
    bench_qtcurrent(qt)
    bench_harmonic_balance(hb)
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
