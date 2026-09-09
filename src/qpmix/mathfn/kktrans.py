"""Kramers-Kronig transform of a DC I-V curve.

The KK transform supplies the real part of the SIS response function.  It
is a Hilbert transform, so it is computed with an FFT rather than by
quadrature.

Two things make this faster than the upstream implementation:

1. **Real-input FFT.**  ``scipy.signal.hilbert`` builds the full complex
   analytic signal even though the input is real and only the imaginary
   part of the result is wanted.  Going through ``rfft``/``irfft`` instead
   halves both the arithmetic and the peak memory.

2. **5-smooth transform lengths.**  The transform is zero-padded by a
   factor of ``n`` (50 by default) to suppress circular wrap-around.  The
   padded length ``npts * n`` is usually a terrible FFT size — 6801 * 50
   has a prime factor of 2267.  Rounding up with
   :func:`scipy.fft.next_fast_len` costs a few extra zeros and buys a large
   speed-up.

Examples:

    >>> import numpy as np
    >>> from qpmix.mathfn.kktrans import kk_trans
    >>> v = np.linspace(-5, 5, 2001)
    >>> i = np.where(np.abs(v) < 1, 0.0, v)
    >>> ikk = kk_trans(v, i)
    >>> ikk.shape
    (2001,)

"""

from __future__ import annotations

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft

__all__ = ["hilbert_transform", "kk_trans", "kk_trans_trapz"]


def hilbert_transform(x: np.ndarray, npad: int | None = None) -> np.ndarray:
    """Hilbert transform of a real signal, via a real-input FFT.

    Args:
        x (ndarray): Real input signal, 1-D.
        npad (int, optional): Length to zero-pad to before transforming.
            Rounded up to an efficient FFT length.  Default is ``len(x)``.

    Returns:
        ndarray: The Hilbert transform, truncated back to ``len(x)``.

    """
    x = np.asarray(x, dtype=float)
    npts = x.size
    nfft = next_fast_len(npts if npad is None else max(npad, npts))

    spec = rfft(x, n=nfft, workers=-1)
    # H{x} = F^-1[ -i sgn(f) X(f) ].  For a real signal only f >= 0 is
    # stored, and the DC and Nyquist bins have no imaginary counterpart.
    spec *= -1j
    spec[0] = 0.0
    if nfft % 2 == 0:
        spec[-1] = 0.0

    return irfft(spec, n=nfft, workers=-1)[:npts]


def kk_trans(v: np.ndarray, i: np.ndarray, n: int = 50) -> np.ndarray:
    """Kramers-Kronig transform of DC I-V data.

    Note:
        The voltage spacing must be uniform.

    Args:
        v (ndarray): Normalized bias voltage, uniformly spaced.
        i (ndarray): Normalized DC tunneling current.
        n (int, optional): Zero-padding factor for the Hilbert transform.
            Default is 50.

    Returns:
        ndarray: The KK transform, same length as ``v``.

    Raises:
        ValueError: If the voltage spacing is not uniform.

    """
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    if v.shape != i.shape:
        raise ValueError("v and i must have the same shape.")
    step = np.diff(v)
    if step.size == 0 or np.abs(step - step[0]).max() > 1e-5:
        raise ValueError("Voltage spacing must be constant.")

    # Subtract v so the transformed quantity decays, making the KK integral
    # well defined at v -> infinity.
    return -hilbert_transform(i - v, npad=v.size * n)


def kk_trans_trapz(v: np.ndarray, i: np.ndarray) -> np.ndarray:
    """Kramers-Kronig transform by direct trapezoidal quadrature.

    Kept as an independent, O(N^2) reference implementation to validate
    :func:`kk_trans` against.  It is much slower and less accurate near the
    singularity, so it is not used in simulations.

    Args:
        v (ndarray): Normalized bias voltage, uniformly spaced.
        i (ndarray): Normalized DC tunneling current.

    Returns:
        ndarray: The KK transform.

    Raises:
        ValueError: If the voltage spacing is not uniform.

    """
    v = np.asarray(v, dtype=float)
    i = np.asarray(i, dtype=float)
    step = np.diff(v)
    if step.size == 0 or np.abs(step - step[0]).max() > 1e-5:
        raise ValueError("Voltage spacing must be constant.")

    ikk = np.empty(v.size, dtype=float)
    for a in range(v.size):
        v_prime = np.delete(v, a)
        i_prime = np.delete(i, a)
        ikk[a] = np.trapezoid((i_prime - v_prime) / (v_prime - v[a]), x=v_prime)
    return ikk / np.pi
