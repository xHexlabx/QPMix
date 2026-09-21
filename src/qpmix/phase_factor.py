r"""Phase-factor spectrum coefficients :math:`C_k(H)`.

Background
----------

An SIS junction driven by a voltage
:math:`V(t) = \sum_p V_p \sin(p\omega t + \phi_p)` acquires a phase factor
whose spectrum is the set of coefficients :math:`C_k`.  Kittara's thesis
(Eqns. 5.7 and 5.12) evaluates them in two steps:

1. Expand each harmonic with the Jacobi-Anger identity, which needs Bessel
   functions :math:`J_n(\alpha_p)` for every :math:`n` up to ``num_b``, at
   every bias point;
2. Convolve the harmonics together, an O(``num_p`` * ``num_b``\ :sup:`2`)
   recursion.

The comment in the upstream source says of step 1: *"This chunk of code
dominates the computation time of this function."*

The FFT formulation
-------------------

Both steps collapse into a single FFT.  Substituting the Jacobi-Anger
identity :math:`e^{i\alpha\sin\theta} = \sum_n J_n(\alpha)e^{in\theta}`
into the convolution shows that the generating function of the
:math:`C_k` is simply

.. math::

    \sum_k C_k e^{ik\theta}
        = \prod_p \exp\!\big(i\,\alpha_p \sin(p\theta - \phi_p)\big)
        = \exp\!\Big(i \sum_p \alpha_p \sin(p\theta - \phi_p)\Big)

so the coefficients are the Fourier coefficients of a function that costs
one ``sin``, one ``exp`` and one FFT to evaluate:

.. math::

    C_k = \frac{1}{M}\sum_{m} \exp\!\Big(
        i\sum_p \alpha_p \sin(p\theta_m - \phi_p)\Big) e^{-2\pi i k m / M},
    \qquad \theta_m = 2\pi m / M.

No Bessel functions, no convolution.  The cost drops from
O(``num_p`` * ``num_b``\ :sup:`2`) plus ``num_p * num_b`` Bessel
evaluations to a single O(``M log M``) transform, and NumPy's FFT output
ordering is already the signed-``k`` ordering the rest of QPMix uses.

It is also *more* accurate: the direct recursion truncates the intermediate
convolutions at :math:`\pm` ``num_b`` at every harmonic, whereas the
transform never forms them.  The only error is spectral aliasing,
:math:`C_k^{\rm FFT} = \sum_j C_{k+jM}`, and since
:math:`|C_k| \sim |J_k(\sum_p \alpha_p)|` falls off super-exponentially
past :math:`k \approx \sum_p\alpha_p`, choosing ``M`` with enough headroom
drives that below machine precision.  :func:`choose_num_theta` does so.

Examples:

    >>> import numpy as np
    >>> vj = np.zeros((2, 2, 3), dtype=complex)
    >>> vj[1, 1] = 0.5 + 0.0j
    >>> freq = np.array([0.0, 0.4])
    >>> ckh = calculate_phase_factor_coeff(vj, freq, 1, 1, 5)
    >>> ckh.shape
    (2, 11, 3)
    >>> from scipy.special import jv
    >>> bool(np.allclose(ckh[1, 2].real, jv(2, 0.5 / 0.4)))
    True

"""

from __future__ import annotations

import numpy as np
from scipy.fft import fft, next_fast_len
from scipy.special import j0, j1
from scipy.special import jv as bessel

from qpmix._backend import JIT_ENABLED, njit, prange

__all__ = [
    "DRIVE_LEVEL_TOL",
    "DriveLevelWarning",
    "bessel_orders",
    "calculate_phase_factor_coeff",
    "choose_num_theta",
    "required_num_b",
]

#: Headroom multiplier on the drive level when sizing the FFT.  The
#: aliasing error is set by |J_n(alpha)| at n = M - num_b, and J_n(alpha)
#: is already below 1e-16 for n > alpha + 4*alpha**(1/3) + 25.
_ALPHA_HEADROOM = 3.0
_ALPHA_PAD = 32


def choose_num_theta(alpha_max: float, num_b: int) -> int:
    """Choose the FFT length for :func:`calculate_phase_factor_coeff`.

    The transform must be long enough that the aliased coefficients
    ``C_{k+M}`` are negligible.  Bessel functions decay super-exponentially
    once the order exceeds the argument, so ``M`` needs to clear
    ``num_b + a few times alpha``.

    Args:
        alpha_max (float): Largest total drive level
            ``sum_p alpha_p`` over all bias points.
        num_b (int): Largest summation limit over all tones.

    Returns:
        int: An efficient FFT length.

    """
    need = 2 * num_b + _ALPHA_HEADROOM * alpha_max + _ALPHA_PAD
    return int(next_fast_len(int(np.ceil(need))))


def calculate_phase_factor_coeff(
    vj: np.ndarray,
    freq: np.ndarray,
    num_f: int,
    num_p: int,
    num_b: int | tuple[int, ...],
    method: str = "fft",
    num_theta: int | None = None,
) -> np.ndarray:
    """Calculate the overall phase-factor spectrum coefficients.

    Eqns. 5.7 and 5.12 in Kittara's thesis.

    Args:
        vj (ndarray): Voltage across the junction, shape
            ``(num_f + 1, num_p + 1, npts)``, complex.
        freq (ndarray): Normalized frequencies, shape ``(num_f + 1,)``.
        num_f (int): Number of fundamental tones.
        num_p (int): Number of harmonics.
        num_b (int or tuple): Summation limit, one per tone or one shared.
        method (str, optional): ``"fft"`` (default) or ``"direct"``.  The
            direct method evaluates Bessel functions and convolves the
            harmonics, exactly as QMix does; it is kept as an independent
            check on the transform.
        num_theta (int, optional): Override the FFT length.  Default is
            chosen by :func:`choose_num_theta`.

    Returns:
        ndarray: Coefficients ``C_k``, shape
        ``(num_f + 1, 2 * max(num_b) + 1, npts)``, complex, with ``k``
        in NumPy FFT order.

    Raises:
        ValueError: If ``method`` is not recognised.

    """
    nb_tuple = _as_nb_tuple(num_b, num_f)
    alpha, phi = drive_level(vj, freq, num_f, num_p)

    if method == "fft":
        return _ckh_fft(alpha, phi, num_f, num_p, nb_tuple, num_theta)
    if method == "direct":
        return _ckh_direct(alpha, phi, num_f, num_p, nb_tuple)
    raise ValueError(f"Unknown method: {method!r}")


def drive_level(
    vj: np.ndarray, freq: np.ndarray, num_f: int, num_p: int
) -> tuple[np.ndarray, np.ndarray]:
    """Junction drive level and phase for every tone and harmonic.

    Eqn. 5.5 in Kittara's thesis: ``alpha[f, p] = |vj[f, p]| / (p * freq[f])``.

    Args:
        vj (ndarray): Voltage across the junction.
        freq (ndarray): Normalized frequencies.
        num_f (int): Number of fundamental tones.
        num_p (int): Number of harmonics.

    Returns:
        tuple: ``(alpha, phi)``, both shaped like ``vj`` and real.

    """
    alpha = np.zeros(vj.shape, dtype=float)
    p_idx = np.arange(1, num_p + 1, dtype=float)
    for f in range(1, num_f + 1):
        alpha[f, 1:] = np.abs(vj[f, 1:]) / (p_idx[:, None] * freq[f])
    return alpha, np.angle(vj)


class DriveLevelWarning(UserWarning):
    """The Bessel summation limit does not cover the realised drive level."""


#: Largest fraction of the phase-factor weight that a summation limit may
#: drop before :func:`qpmix.qtcurrent.qtcurrent` and
#: :func:`qpmix.harmonic_balance.harmonic_balance` issue a
#: :class:`DriveLevelWarning`.
DRIVE_LEVEL_TOL = 1e-6


def bessel_orders(alpha: float, tol: float = 1e-9) -> int:
    """Smallest ``b`` such that the weight outside ``|n| <= b`` is below ``tol``.

    ``sum_n J_n(alpha)^2 = 1``, so ``1 - sum_{|n| <= b} J_n(alpha)^2`` is
    exactly the fraction of the phase-factor weight that a summation limit
    of ``b`` throws away for one harmonic at drive level ``alpha``.

    Args:
        alpha (float): Drive level.
        tol (float, optional): Acceptable dropped weight.  Default is 1e-9.

    Returns:
        int: The summation limit.

    """
    alpha = abs(float(alpha))
    if not np.isfinite(alpha):
        raise ValueError("The drive level must be finite.")
    b = int(np.ceil(alpha))  # J_n(alpha) only starts to decay once n > alpha
    while 1.0 - float(np.sum(bessel(np.arange(-b, b + 1), alpha) ** 2)) > tol:
        b += 1
    return b


def required_num_b(
    vj: np.ndarray, freq: np.ndarray, num_f: int, num_p: int, tol: float = 1e-9
) -> tuple[int, ...]:
    """Per-tone summation limit that covers the realised drive levels.

    Harmonic ``p`` of a tone spreads the spectrum by ``p`` steps per Bessel
    order, so its requirement is ``p * bessel_orders(alpha[f, p])``; the
    maximum over harmonics and bias points is returned for each tone.  Use
    it to choose ``num_b`` from a solved junction voltage instead of
    guessing: the IF tone of a mixer has a tiny photon voltage, so a small
    induced IF voltage is already a large drive level.

    Args:
        vj (ndarray): Junction voltage, shape ``(num_f + 1, num_p + 1, npts)``.
        freq (ndarray): Normalized frequencies, shape ``(num_f + 1,)``.
        num_f (int): Number of tones.
        num_p (int): Number of harmonics.
        tol (float, optional): Acceptable dropped weight per harmonic.
            Default is 1e-9.

    Returns:
        tuple: One summation limit per tone.

    """
    alpha, _ = drive_level(vj, freq, num_f, num_p)
    need = []
    for f in range(1, num_f + 1):
        b_f = 0
        for p in range(1, num_p + 1):
            finite = alpha[f, p][np.isfinite(alpha[f, p])]
            if finite.size:
                b_f = max(b_f, p * bessel_orders(float(finite.max()), tol))
        need.append(b_f)
    return tuple(need)


@njit(parallel=True)
def _phase_factor_signal(alpha_f, phi_f, theta, out):  # pragma: no cover
    """Sample ``exp(i * sum_p alpha_p sin(p*theta - phi_p))`` for one tone.

    Fuses the harmonic sum, the sine and the complex exponential into a
    single pass, so the ``(num_theta, npts)`` phase array is never
    materialised.  The transcendentals then vectorise together instead of
    each costing a separate strided pass over memory.

    Args:
        alpha_f (ndarray): Drive levels for this tone, shape
            ``(num_p, npts)``.
        phi_f (ndarray): Phases for this tone, same shape.
        theta (ndarray): Sample positions, shape ``(num_theta,)``.
        out (ndarray): Destination, shape ``(num_theta, npts)``, complex.

    """
    num_p, npts = alpha_f.shape
    for m in prange(theta.shape[0]):
        th = theta[m]
        for i in range(npts):
            acc = 0.0
            for p in range(num_p):
                acc += alpha_f[p, i] * np.sin((p + 1) * th - phi_f[p, i])
            out[m, i] = np.cos(acc) + 1j * np.sin(acc)


def _phase_factor_signal_numpy(alpha_f, phi_f, theta, out):
    """NumPy fallback for :func:`_phase_factor_signal`.

    Args:
        alpha_f (ndarray): Drive levels, shape ``(num_p, npts)``.
        phi_f (ndarray): Phases, same shape.
        theta (ndarray): Sample positions.
        out (ndarray): Destination, shape ``(num_theta, npts)``, complex.

    """
    num_p = alpha_f.shape[0]
    phase = np.zeros(out.shape, dtype=float)
    for p in range(num_p):
        phase += alpha_f[p][None, :] * np.sin(
            (p + 1) * theta[:, None] - phi_f[p][None, :]
        )
    np.exp(1j * phase, out=out)


def _ckh_fft(
    alpha: np.ndarray,
    phi: np.ndarray,
    num_f: int,
    num_p: int,
    nb_tuple: tuple[int, ...],
    num_theta: int | None,
) -> np.ndarray:
    """FFT evaluation of the phase-factor coefficients (see module docs).

    Args:
        alpha (ndarray): Drive levels.
        phi (ndarray): Phases, in radians.
        num_f (int): Number of fundamental tones.
        num_p (int): Number of harmonics.
        nb_tuple (tuple): Summation limit for each tone.
        num_theta (int or None): FFT length, or None to choose one.

    Returns:
        ndarray: The coefficients.

    """
    npts = alpha.shape[2]
    max_nb = max(nb_tuple)

    if num_theta is None:
        alpha_max = float(alpha[1:, 1:].sum(axis=1).max(initial=0.0))
        num_theta = choose_num_theta(alpha_max, max_nb)

    theta = 2 * np.pi * np.arange(num_theta) / num_theta
    ckh = np.zeros((num_f + 1, 2 * max_nb + 1, npts), dtype=complex)
    sample = _phase_factor_signal if JIT_ENABLED else _phase_factor_signal_numpy

    # The transform runs along axis 0, so the wanted low-|k| coefficients
    # come out as contiguous slices and no transpose copy is needed.
    signal = np.empty((num_theta, npts), dtype=complex)
    workers = -1 if npts * num_theta > 1 << 16 else 1
    for f in range(1, num_f + 1):
        nb = nb_tuple[f - 1]
        sample(
            np.ascontiguousarray(alpha[f, 1:]),
            np.ascontiguousarray(phi[f, 1:]),
            theta,
            signal,
        )
        spec = fft(signal, axis=0, workers=workers)
        spec /= num_theta
        ckh[f, : nb + 1] = spec[: nb + 1]
        if nb:
            ckh[f, 2 * max_nb + 1 - nb :] = spec[num_theta - nb :]

    return ckh


def _ckh_direct(
    alpha: np.ndarray,
    phi: np.ndarray,
    num_f: int,
    num_p: int,
    nb_tuple: tuple[int, ...],
) -> np.ndarray:
    """Bessel-and-convolve evaluation, as in QMix.

    Kept as an independent reference implementation for the tests, and as
    a fallback if the FFT length ever needs to be sized manually.

    Args:
        alpha (ndarray): Drive levels.
        phi (ndarray): Phases, in radians.
        num_f (int): Number of fundamental tones.
        num_p (int): Number of harmonics.
        nb_tuple (tuple): Summation limit for each tone.

    Returns:
        ndarray: The coefficients.

    """
    npts = alpha.shape[2]
    max_nb = max(nb_tuple)

    # Jacobi-Anger coefficients, Eqn. 5.7.
    jac = np.zeros((num_f + 1, num_p + 1, 2 * max_nb + 1, npts), dtype=complex)
    for f in range(1, num_f + 1):
        for p in range(1, num_p + 1):
            jac[f, p, 0] = j0(alpha[f, p])
            for n in range(1, nb_tuple[f - 1] + 1):
                jn = j1(alpha[f, p]) if n == 1 else bessel(n, alpha[f, p])
                jac[f, p, n] = jn * np.exp(-1j * n * phi[f, p])
                jac[f, p, -n] = (-1) ** n * np.conj(jac[f, p, n])

    return _convolve_coefficients(jac)


@njit
def _convolve_coefficients(jac):  # pragma: no cover - exercised via _ckh_direct
    """Convolve the per-harmonic spectra together (Eqn. 5.12).

    See Withington and Kollberg (1989).

    Args:
        jac (ndarray): Jacobi-Anger coefficients.

    Returns:
        ndarray: The overall phase-factor coefficients.

    """
    _, num_p, num_b, _ = jac.shape
    num_p -= 1
    num_b = (num_b - 1) // 2

    ckh_last = jac[:, 1, :, :].copy()
    if num_p == 1:
        return ckh_last

    for p in range(2, num_p + 1):
        ckh_next = np.zeros_like(ckh_last)
        for k in range(-num_b, num_b + 1):
            l_min = max(-num_b, int((k - num_b) / p))
            l_max = min(num_b, int((k + num_b) / p))
            for l in range(l_min, l_max + 1):
                ckh_next[1:, k] += ckh_last[1:, k - p * l] * jac[1:, p, l]
        ckh_last = ckh_next

    return ckh_last


def _as_nb_tuple(num_b: int | tuple[int, ...], num_f: int) -> tuple[int, ...]:
    """Normalize ``num_b`` into one summation limit per tone.

    Args:
        num_b (int or tuple): One shared limit, or one per tone.  A tuple is
            0-indexed: ``num_b[0]`` applies to the first tone.
        num_f (int): Number of tones.

    Returns:
        tuple: Exactly ``num_f`` summation limits.

    Raises:
        ValueError: If a tuple is given with fewer entries than tones.

    """
    if isinstance(num_b, (tuple, list, np.ndarray)):
        if len(num_b) < num_f:
            raise ValueError(
                "There must be one value of num_b for each fundamental frequency."
            )
        return tuple(int(v) for v in num_b[:num_f])
    return (int(num_b),) * num_f
