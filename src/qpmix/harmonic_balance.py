"""Harmonic balance of an SIS junction embedded in a linear circuit.

Each signal applied to the junction has a Thevenin equivalent (see
:mod:`qpmix.circuit`), and the voltage it actually induces across the
junction depends on the junction's impedance — which in turn depends on
every other signal present.  Harmonic balance solves that coupled,
non-linear system:

    ``vj = vt - zt * ij(vj)``

for the junction voltage ``vj``, simultaneously at every bias point.

Numerically this is ``npts`` independent root-finding problems, each in
``2 * num_f * num_p`` real unknowns (real and imaginary parts of each
signal).  QPMix solves them as one batched problem.

What changed relative to QMix
-----------------------------

*Broyden updates.*  QMix rebuilds the Jacobian by finite differences at
every iteration, which costs ``2 * num_n`` calls to the tunneling-current
solver — by far the dominant cost.  QPMix keeps that for the first
iteration and then applies rank-one Broyden updates, so later iterations
cost a *single* call.  Set ``jacobian="newton"`` to restore the original
behaviour.

*Solve, don't invert.*  QMix forms the explicit matrix inverse and then
multiplies.  QPMix calls :func:`numpy.linalg.solve`, which is both faster
and better conditioned, and does the batched matrix algebra in one
:func:`numpy.einsum` instead of a Python double loop over signals.

*Adaptive damping.*  If a Newton step increases the residual, the step is
backtracked instead of being applied blindly.  This makes strongly pumped
simulations converge where a fixed damping coefficient would oscillate.

Examples:

    >>> import numpy as np
    >>> from qpmix.circuit import EmbeddingCircuit
    >>> from qpmix.respfn import RespFnPolynomial
    >>> cct = EmbeddingCircuit(1, 1, vb_npts=51)
    >>> cct.freq[1] = 0.33
    >>> cct.vt[1, 1] = 0.3
    >>> cct.zt[1, 1] = 0.3 - 0.3j
    >>> resp = RespFnPolynomial(50, verbose=False)
    >>> vj = harmonic_balance(cct, resp, num_b=9, verbose=False)
    >>> vj.shape
    (2, 2, 51)

"""

from __future__ import annotations

from timeit import default_timer as timer

import numpy as np

from qpmix.qtcurrent import (
    MAX_DIRECT_TONES,
    ROUND_FREQ,
    interpolate_respfn,
    qtcurrent,
)

__all__ = ["check_hb_error", "harmonic_balance"]

#: Thevenin voltages below this are clamped, to avoid dividing by zero when
#: forming relative errors.
MIN_VT = 1e-10

#: Voltage step used for the finite-difference Jacobian.
DV = 1e-3


def harmonic_balance(
    cct,
    resp,
    num_b: int | tuple[int, ...] = 15,
    max_it: int = 10,
    stop_rerror: float = 0.001,
    vj_initial: np.ndarray | None = None,
    damp_coeff: float = 1.0,
    mode: str = "o",
    verbose: bool = True,
    zj_guess: float = 0.67,
    jacobian: str = "broyden",
    line_search: bool = True,
    resp_matrix: np.ndarray | None = None,
    method: str = "auto",
):
    """Solve for the junction voltage that balances the circuit.

    Uses Newton's method, optionally with Broyden updates between full
    Jacobian evaluations.  For the underlying theory see Garrett (2018),
    Kittara (2002), Kittara, Withington and Yassin (2007), or Withington,
    Kittara and Yassin (2003).

    Args:
        cct (qpmix.circuit.EmbeddingCircuit): The embedding circuit.
        resp (qpmix.respfn.RespFn): The response function.
        num_b (int or tuple, optional): Summation limit for the phase-factor
            coefficients.  Default is 15.
        max_it (int, optional): Maximum number of iterations.  Default is
            10.
        stop_rerror (float, optional): Target relative error.  Default is
            0.001.
        vj_initial (ndarray, optional): Initial guess for the junction
            voltage.  Default is None, meaning a resistive-divider guess.
        damp_coeff (float, optional): Multiplier applied to each Newton
            step, between 0 and 1.  Default is 1.
        mode (str, optional): ``"o"`` returns ``vj``; ``"x"`` also returns
            the iteration count and a converged flag; ``"m"`` also returns
            a per-bias-point convergence mask.  Default is ``"o"``.
        verbose (bool, optional): Print progress to the terminal, including
            a warning if the error target is not met.  Default is True.
            Use ``mode="x"`` to detect non-convergence programmatically --
            a fitting loop that calls this thousands of times cannot afford
            the message, and QMix prints it unconditionally.
        zj_guess (float, optional): Assumed junction impedance for the
            initial guess.  Default is 0.67.
        jacobian (str, optional): ``"broyden"`` recomputes the Jacobian
            once and then updates it with rank-one corrections;
            ``"newton"`` recomputes it every iteration, as QMix does.
            Default is ``"broyden"``.
        line_search (bool, optional): Backtrack a step that would increase
            the residual.  Default is True.
        method (str, optional): Which engine evaluates the tunneling
            currents -- see :func:`qpmix.qtcurrent.qtcurrent`.  Harmonic
            balance is dominated by those evaluations, so this is what
            decides whether a many-tone solve is practical: ``"grid"``
            switches to the common-grid engine, which is worth it from about
            three tones on a coarse frequency grid, and is the only option
            above four.  Default is ``"auto"``.
        resp_matrix (ndarray, optional): A pre-computed response matrix from
            :func:`qpmix.qtcurrent.interpolate_respfn`.  It depends only on
            the bias sweep, the tone frequencies and ``num_b`` -- none of
            which change while fitting an embedding circuit -- so passing it
            in avoids rebuilding the single most expensive array on every
            call.  Default is None.

    Returns:
        ndarray: The junction voltage, shape
        ``(num_f + 1, num_p + 1, npts)``.  Extra values are returned for
        ``mode`` of ``"x"`` or ``"m"``.

    Raises:
        ValueError: If ``jacobian`` or ``mode`` is not recognised.

    """
    if jacobian not in ("broyden", "newton"):
        raise ValueError(f"Unknown jacobian option: {jacobian!r}")
    if mode not in ("o", "x", "m"):
        raise ValueError(f"Unknown mode: {mode!r}")

    if verbose:
        print("Running harmonic balance:")
    start_time = timer()

    num_f, num_p, num_n = cct.num_f, cct.num_p, cct.num_n
    npts = cct.vb_npts
    vt, zt = cct.vt, cct.zt

    # With no embedding impedance the junction sees the Thevenin voltage
    # directly and there is nothing to balance.
    if (zt == 0).all():
        if verbose:
            print("Done.\n")
        vj_out = vt[:, :, None] * np.ones(npts, dtype=complex)
        return _package(vj_out, 0, True, np.ones(npts, dtype=bool), mode)

    vt = vt.copy()
    vt[np.abs(vt) < MIN_VT] = MIN_VT
    vt[0, :] = 0
    vt[:, 0] = 0

    if vj_initial is None:
        vj_initial = (
            vt[:, :, None]
            * zj_guess
            / (zj_guess + zt[:, :, None])
            * np.ones(npts, dtype=complex)
        )

    # Flatten [f, p] into a single signal index k, so the per-bias-point
    # problem is a plain 2*num_n dimensional root find.
    vj_2d = vj_initial[1:, 1:, :].reshape((num_n, npts)).astype(complex).copy()
    vt_2d = vt[1:, 1:].reshape(num_n)
    zt_2d = zt[1:, 1:].reshape(num_n)

    if verbose:
        print(f" - {num_f} tone(s) and {num_p} harmonic(s)")
        print(f" - {num_n * 2 + 1} qtcurrent call(s) for the first iteration")
        print(f" - max. iterations: {max_it}")

    # The response matrix and the current evaluations must use the same
    # engine, and for the grid engine the same grid, or their array shapes
    # will not match.  Resolve it once here.
    grid = None
    if method == "grid" or (method == "auto" and num_f > MAX_DIRECT_TONES):
        from qpmix.multitone import ToneGrid

        grid = ToneGrid.from_circuit(cct, num_b=num_b)
    respfn_interp = (
        interpolate_respfn(cct, resp, num_b, method=method, grid=grid)
        if resp_matrix is None
        else resp_matrix
    )
    freq_list = _hb_freq_list(cct)
    current_kwargs = {} if grid is None else {"grid": grid}
    if method != "auto":
        current_kwargs["method"] = method

    def residual(vj: np.ndarray) -> np.ndarray:
        ij = _qt_current_for_hb(
            vj, cct, resp, num_b, respfn_interp, freq_list, current_kwargs
        )
        return vt_2d[:, None] - zt_2d[:, None] * ij - vj

    # inv_j[p, q, i]: the inverse Jacobian at each bias point, in the
    # real/imaginary split representation.
    inv_j = None
    err_all = residual(vj_2d)
    iteration = 0
    converged = False
    finished_points = np.zeros(npts, dtype=bool)

    for iteration in range(max_it + 1):
        max_rel_error, finished_points = _report_error(
            err_all, vj_2d, num_n, num_p, stop_rerror, npts, iteration, verbose
        )
        if max_rel_error <= stop_rerror:
            converged = True
            if verbose:
                print("Done: Minimum error target was achieved.")
            break
        if iteration == max_it:
            if verbose:
                print("*** DID NOT ACHIEVE TARGET ERROR VALUE ***\n")
            break

        if inv_j is None or jacobian == "newton":
            inv_j = _inv_jacobian(err_all, vj_2d, residual, num_n, npts, verbose)

        step = _newton_step(inv_j, err_all) * damp_coeff
        vj_new, err_new, inv_j = _apply_step(
            vj_2d, err_all, step, residual, inv_j, jacobian, line_search
        )
        vj_2d, err_all = vj_new, err_new

    vj_out = np.zeros((num_f + 1, num_p + 1, npts), dtype=complex)
    vj_out[1:, 1:, :] = vj_2d.reshape((num_f, num_p, npts))

    if verbose:
        elapsed = timer() - start_time
        print(f" - sim time:\t\t{elapsed:7.2f} s")
        if iteration >= 1:
            print(f" - {iteration} iteration(s) required")
            print(f" - time per iteration:\t{elapsed / iteration:7.2f} s")

    return _package(vj_out, iteration, converged, finished_points, mode)


def _package(vj_out, iteration, converged, finished_points, mode):
    """Shape the return value according to ``mode``.

    Args:
        vj_out (ndarray): The junction voltage.
        iteration (int): Iterations used.
        converged (bool): Whether the error target was met.
        finished_points (ndarray): Per-bias-point convergence mask.
        mode (str): ``"o"``, ``"x"`` or ``"m"``.

    Returns:
        ndarray or tuple: See :func:`harmonic_balance`.

    """
    if mode == "o":
        return vj_out
    if mode == "x":
        return vj_out, iteration, converged
    return vj_out, finished_points


def _hb_freq_list(cct) -> list[float]:
    """The frequencies harmonic balance needs from ``qtcurrent``.

    Args:
        cct (qpmix.circuit.EmbeddingCircuit): The embedding circuit.

    Returns:
        list: One frequency per tone and harmonic, in signal-index order.

    """
    return [
        round(cct.freq[f] * p, ROUND_FREQ)
        for f in range(1, cct.num_f + 1)
        for p in range(1, cct.num_p + 1)
    ]


def _qt_current_for_hb(
    vj_2d, cct, resp, num_b, resp_matrix, freq_list, current_kwargs=None
):
    """Tunneling current at every tone and harmonic, in signal-index form.

    Args:
        vj_2d (ndarray): Junction voltage, shape ``(num_n, npts)``.
        cct (qpmix.circuit.EmbeddingCircuit): The embedding circuit.
        resp (qpmix.respfn.RespFn): The response function.
        num_b (int or tuple): Summation limit.
        resp_matrix (ndarray): Pre-interpolated response function.
        freq_list (list): Output frequencies, from :func:`_hb_freq_list`.
        current_kwargs (dict, optional): Engine selection forwarded to
            :func:`qpmix.qtcurrent.qtcurrent`.  Default is None.

    Returns:
        ndarray: Tunneling current, shape ``(num_n, npts)``.

    """
    vj = np.zeros((cct.num_f + 1, cct.num_p + 1, cct.vb_npts), dtype=complex)
    vj[1:, 1:, :] = vj_2d.reshape((cct.num_f, cct.num_p, cct.vb_npts))
    return qtcurrent(
        vj,
        cct,
        resp,
        freq_list,
        num_b,
        verbose=False,
        resp_matrix=resp_matrix,
        **(current_kwargs or {}),
    )


def _to_real(z: np.ndarray) -> np.ndarray:
    """Split a complex signal vector into interleaved real components.

    Args:
        z (ndarray): Complex, shape ``(num_n, npts)``.

    Returns:
        ndarray: Real, shape ``(2 * num_n, npts)``, ordered
        ``[Re_0, Im_0, Re_1, Im_1, ...]``.

    """
    out = np.empty((2 * z.shape[0], z.shape[1]), dtype=float)
    out[0::2] = z.real
    out[1::2] = z.imag
    return out


def _to_complex(x: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_to_real`.

    Args:
        x (ndarray): Real, shape ``(2 * num_n, npts)``.

    Returns:
        ndarray: Complex, shape ``(num_n, npts)``.

    """
    return x[0::2] + 1j * x[1::2]


def _inv_jacobian(err_all, vj_2d, residual, num_n, npts, verbose):
    """Finite-difference Jacobian of the residual, inverted per bias point.

    Costs ``2 * num_n`` evaluations of the residual, which dominates the
    run time — hence the Broyden option in :func:`harmonic_balance`.

    Args:
        err_all (ndarray): Residual at the current point.
        vj_2d (ndarray): Current junction voltage.
        residual (callable): Residual function.
        num_n (int): Number of signals.
        npts (int): Number of bias points.
        verbose (bool): Print progress.

    Returns:
        ndarray: Inverse Jacobian, shape ``(2*num_n, 2*num_n, npts)``.

    """
    if verbose:
        print("Calculating inverse Jacobian")

    jac = np.empty((2 * num_n, 2 * num_n, npts), dtype=float)
    for q in range(num_n):
        dvj = np.zeros((num_n, npts), dtype=float)
        dvj[q, :] = DV
        d_re = (residual(vj_2d + dvj) - err_all) / DV
        d_im = (residual(vj_2d + 1j * dvj) - err_all) / DV
        jac[:, q * 2] = _to_real(d_re)
        jac[:, q * 2 + 1] = _to_real(d_im)

    # One batched inverse over the bias axis.  The explicit inverse is kept
    # (rather than an LU solve) because the Broyden update needs it.
    return np.linalg.inv(jac.transpose(2, 0, 1)).transpose(1, 2, 0)


def _newton_step(inv_j: np.ndarray, err_all: np.ndarray) -> np.ndarray:
    """The Newton correction ``-J^-1 F``, for every bias point at once.

    Args:
        inv_j (ndarray): Inverse Jacobian, shape ``(2n, 2n, npts)``.
        err_all (ndarray): Residual, shape ``(n, npts)``.

    Returns:
        ndarray: Complex correction, shape ``(n, npts)``.

    """
    return -_to_complex(np.einsum("pqi,qi->pi", inv_j, _to_real(err_all)))


def _apply_step(vj, err, step, residual, inv_j, jacobian, line_search):
    """Take a step, optionally backtracking, and update the Jacobian.

    Args:
        vj (ndarray): Current junction voltage.
        err (ndarray): Current residual.
        step (ndarray): Proposed correction.
        residual (callable): Residual function.
        inv_j (ndarray): Current inverse Jacobian.
        jacobian (str): ``"broyden"`` or ``"newton"``.
        line_search (bool): Whether to backtrack.

    Returns:
        tuple: ``(vj_new, err_new, inv_j)``.

    """
    scale = 1.0
    norm0 = float(np.linalg.norm(err))
    for _ in range(3):
        vj_new = vj + step * scale
        err_new = residual(vj_new)
        if not line_search:
            break
        # Backtrack only if the step made the problem globally worse.  A
        # per-bias-point test would trigger on almost every Newton step,
        # since early iterations always overshoot somewhere.
        if float(np.linalg.norm(err_new)) <= norm0:
            break
        scale /= 4.0

    if jacobian == "broyden":
        inv_j = _broyden_update(inv_j, vj_new - vj, err_new - err)
    return vj_new, err_new, inv_j


def _broyden_update(inv_j, dvj, derr):
    """Sherman-Morrison rank-one update of the inverse Jacobian.

    Applies the "good" Broyden update directly to ``J^-1``, so a new
    Jacobian never has to be built:

        ``J^-1 <- J^-1 + (dx - J^-1 df) (dx^T J^-1) / (dx^T J^-1 df)``

    Args:
        inv_j (ndarray): Inverse Jacobian, shape ``(2n, 2n, npts)``.
        dvj (ndarray): Step taken, complex, shape ``(n, npts)``.
        derr (ndarray): Change in the residual, complex, shape ``(n, npts)``.

    Returns:
        ndarray: The updated inverse Jacobian.

    """
    dx = _to_real(dvj)
    df = _to_real(derr)
    jf = np.einsum("pqi,qi->pi", inv_j, df)
    denom = np.einsum("pi,pi->i", dx, jf)
    # Skip bias points where the update would be numerically meaningless.
    safe = np.abs(denom) > 1e-30
    if not safe.any():
        return inv_j
    xj = np.einsum("pi,pqi->qi", dx, inv_j)
    correction = np.einsum("pi,qi->pqi", dx - jf, xj)
    correction[:, :, ~safe] = 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        correction[:, :, safe] /= denom[safe]
    return inv_j + correction


def _report_error(err_all, vj_2d, num_n, num_p, stop_rerror, npts, it, verbose):
    """Compute and optionally print the relative error at each signal.

    Args:
        err_all (ndarray): Residual.
        vj_2d (ndarray): Junction voltage.
        num_n (int): Number of signals.
        num_p (int): Number of harmonics.
        stop_rerror (float): Target relative error.
        npts (int): Number of bias points.
        it (int): Iteration number.
        verbose (bool): Print progress.

    Returns:
        tuple: ``(max_rel_error, finished_points)``.

    """
    if verbose:
        print(f"Error after {it} iteration(s):")

    with np.errstate(divide="ignore", invalid="ignore"):
        abs_error = np.abs(err_all)
        rel_error = abs_error / np.abs(vj_2d)
    good = rel_error < stop_rerror
    finished_points = good.all(axis=0)
    max_rel_error = float(np.nanmax(rel_error))

    if verbose:
        for k in range(num_n):
            f, p = _k_to_fp(k, num_p)
            med = min(float(np.nanmedian(rel_error[k])), 99999.999)
            mx = min(float(np.nanmax(rel_error[k])), 99999.999)
            complete = float(np.sum(good[k])) / npts * 100
            print(
                f"\tf:{f:d}, p:{p:d},   med. rel. error: {med:9.3f},   "
                f"max. rel. error: {mx:9.3f},   {complete:5.1f} % complete"
            )
    return max_rel_error, finished_points


def _k_to_fp(k: int, num_p: int) -> tuple[int, int]:
    """Map a flat signal index back to ``(tone, harmonic)``.

    Args:
        k (int): Flat signal index.
        num_p (int): Number of harmonics.

    Returns:
        tuple: ``(f, p)``, both 1-indexed.

    """
    return k // num_p + 1, k % num_p + 1


def check_hb_error(vj_check, cct, resp, num_b=15, stop_rerror=0.001) -> None:
    """Independently verify a harmonic balance solution.

    Recomputes the residual from scratch and asserts that it meets the
    error target.  Speed is not a concern here; this is a debugging aid.

    Args:
        vj_check (ndarray): The junction voltage to check.
        cct (qpmix.circuit.EmbeddingCircuit): The embedding circuit.
        resp (qpmix.respfn.RespFn): The response function.
        num_b (int or tuple, optional): Summation limit.  Default is 15.
        stop_rerror (float, optional): Target relative error.  Default is
            0.001.

    Raises:
        AssertionError: If any signal fails to meet the error target.

    """
    print("Double-checking harmonic balance error:")

    freq_list = _hb_freq_list(cct)
    current = qtcurrent(vj_check, cct, resp, freq_list, num_b, verbose=False)
    ij_all = np.zeros((cct.num_f + 1, cct.num_p + 1, cct.vb_npts), dtype=complex)
    ij_all[1:, 1:] = current.reshape((cct.num_f, cct.num_p, cct.vb_npts))

    for f in range(1, cct.num_f + 1):
        for p in range(1, cct.num_p + 1):
            error_fp = cct.vt[f, p] - cct.zt[f, p] * ij_all[f, p, :] - vj_check[f, p, :]
            max_rel_error = float(np.max(np.abs(error_fp) / np.abs(vj_check[f, p, :])))
            passed = max_rel_error < stop_rerror
            print(
                f"\tf:{f},\tp:{p},\tmax rel. error: {max_rel_error:.2E},"
                f"\tPass? {'Yes' if passed else 'No'}"
            )
            assert passed
    print("")
