"""Smoothing filters for noisy I-V data.

The Gaussian smoother here automatically switches between direct and
FFT-based convolution.  Upstream QMix always calls :func:`numpy.convolve`,
which is O(N*M); the response-function setup smooths with windows over a
thousand points wide, where an FFT convolution (O(N log N)) is far cheaper.
The crossover is measured, not guessed — see ``benchmarks/bench_filters.py``.

Examples:

    >>> import numpy as np
    >>> from qpmix.mathfn.filters import gauss_conv
    >>> x = np.ones(500)
    >>> bool(np.allclose(gauss_conv(x, sigma=5), 1.0))
    True

"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve

__all__ = ["gauss_conv"]

#: Window lengths at or above this use the FFT path.
FFT_WINDOW_THRESHOLD = 128


def gauss_conv(
    x: np.ndarray,
    sigma: float = 10,
    ext_x: float = 3,
    method: str = "auto",
) -> np.ndarray:
    """Smooth data by convolving with a Gaussian kernel.

    The signal is reflected at both ends before convolving, so the output
    has the same length as the input and does not droop towards zero at the
    edges.

    Args:
        x (ndarray): Noisy data, 1-D.
        sigma (float, optional): Standard deviation of the Gaussian, in
            samples.  Default is 10.
        ext_x (float, optional): The kernel extends ``ext_x * sigma`` in
            each direction.  Default is 3.
        method (str, optional): ``"auto"``, ``"direct"`` or ``"fft"``.
            ``"auto"`` picks the FFT path for wide kernels.  Default is
            ``"auto"``.

    Returns:
        ndarray: The filtered data, same length as ``x``.

    Raises:
        ValueError: If the window would be longer than the data, if
            ``sigma * ext_x < 1``, or if ``method`` is unknown.

    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("gauss_conv expects 1-D data.")
    if sigma * ext_x < 1:
        raise ValueError("Window size must be larger than 1. Increase ext_x.")

    window = _gauss(sigma, ext_x)
    wlen = window.size
    if wlen > x.size:
        raise ValueError("Window size must be smaller than data size.")
    window = window / window.sum()

    # Reflect-pad so the convolution stays valid across the whole signal.
    padded = np.r_[x[wlen - 1 : 0 : -1], x, x[-2 : -wlen - 1 : -1]]

    if method == "auto":
        method = "fft" if wlen >= FFT_WINDOW_THRESHOLD else "direct"
    if method == "direct":
        y = np.convolve(window, padded, mode="valid")
    elif method == "fft":
        y = fftconvolve(padded, window, mode="valid")
    else:
        raise ValueError(f"Unknown method: {method!r}")

    return y[wlen // 2 : -wlen // 2 + 1]


def _gauss(sigma: float, n_sigma: float = 3) -> np.ndarray:
    """Build a discrete Gaussian kernel centred on zero.

    Args:
        sigma (float): Standard deviation, in samples.
        n_sigma (float, optional): Half-extent of the kernel, in units of
            ``sigma``.  Default is 3.

    Returns:
        ndarray: The kernel, on integer sample offsets.

    """
    x_range = n_sigma * sigma
    x = np.arange(-x_range, x_range + 1e-5, 1, dtype=float)
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x / sigma) ** 2)
