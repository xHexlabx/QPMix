"""Compute backend selection and JIT compilation for QPMix.

QPMix runs on plain NumPy/SciPy.  When :mod:`numba` is installed the hot
kernels in :mod:`qpmix.qtcurrent` and :mod:`qpmix.interp` are JIT-compiled
instead, which is typically one to two orders of magnitude faster.

Design notes
------------

*Lazy compilation.*  Upstream QMix attaches explicit eager signatures to
every ``@njit`` kernel, so importing the package pays the full LLVM
compilation cost before a single line of user code runs.  QPMix compiles
lazily and caches the result on disk (``cache=True``), so ``import qpmix``
stays fast and each kernel is only ever built for the dtypes it actually
sees.

*Graceful degradation.*  If numba is missing (or explicitly disabled via
``QPMIX_DISABLE_JIT=1``), :func:`njit` becomes a no-op decorator and
:func:`prange` becomes :func:`range`.  Every kernel in QPMix is written so
that it produces identical results either way; only the speed changes.

Examples:

    >>> from qpmix._backend import get_config
    >>> cfg = get_config()
    >>> sorted(cfg)
    ['jit_enabled', 'num_threads', 'numba_version', 'parallel']

"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

__all__ = [
    "HAS_NUMBA",
    "JIT_ENABLED",
    "get_config",
    "njit",
    "prange",
    "set_num_threads",
]


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


try:  # pragma: no cover - depends on the installed environment
    import numba as _numba

    HAS_NUMBA = True
    _NUMBA_VERSION: str | None = _numba.__version__
except ImportError:  # pragma: no cover - depends on the installed environment
    _numba = None
    HAS_NUMBA = False
    _NUMBA_VERSION = None

#: True when kernels will actually be JIT-compiled.
JIT_ENABLED: bool = HAS_NUMBA and not _env_flag("QPMIX_DISABLE_JIT")

#: True when parallel kernels may use ``numba.prange`` over threads.
PARALLEL_ENABLED: bool = JIT_ENABLED and not _env_flag("QPMIX_DISABLE_PARALLEL")


def njit(*args: Any, **kwargs: Any) -> Callable[..., Any]:
    """Decorate a kernel for JIT compilation, or leave it as pure Python.

    Accepts the same keyword arguments as :func:`numba.njit`.  ``cache`` and
    ``fastmath`` default to True, and ``parallel`` is silently downgraded to
    False when parallel execution is disabled.

    Args:
        *args: An optional bare function, so the decorator works both as
            ``@njit`` and as ``@njit(parallel=True)``.
        **kwargs: Options forwarded to :func:`numba.njit`.

    Returns:
        The decorated function, or a decorator.

    """
    kwargs.setdefault("cache", True)
    kwargs.setdefault("fastmath", True)
    if kwargs.get("parallel") and not PARALLEL_ENABLED:
        kwargs["parallel"] = False
    # ``cache`` and ``parallel`` are mutually exclusive in older numba builds.
    if kwargs.get("parallel"):
        kwargs["cache"] = False

    def wrap(func: Callable[..., Any]) -> Callable[..., Any]:
        if not JIT_ENABLED:
            return func
        try:
            return _numba.njit(**kwargs)(func)
        except RuntimeError:  # pragma: no cover - e.g. no on-disk cache location
            return _numba.njit(**{**kwargs, "cache": False})(func)

    if args and callable(args[0]):
        return wrap(args[0])
    return wrap


if PARALLEL_ENABLED:  # pragma: no cover - depends on the installed environment
    prange = _numba.prange
else:  # pragma: no cover - depends on the installed environment
    prange = range


def set_num_threads(n: int) -> None:
    """Set the number of threads used by parallel kernels.

    Args:
        n (int): Number of threads.  Must be at least 1.

    Raises:
        ValueError: If ``n`` is less than 1.

    """
    if n < 1:
        raise ValueError("Number of threads must be >= 1.")
    if PARALLEL_ENABLED:  # pragma: no cover - depends on the environment
        _numba.set_num_threads(n)


def get_num_threads() -> int:
    """Return the number of threads available to parallel kernels."""
    if PARALLEL_ENABLED:  # pragma: no cover - depends on the environment
        return int(_numba.get_num_threads())
    return 1


def get_config() -> dict[str, Any]:
    """Report which compute backend QPMix is using.

    Returns:
        dict: Backend description with the keys ``jit_enabled``,
        ``parallel``, ``num_threads`` and ``numba_version``.

    """
    return {
        "jit_enabled": JIT_ENABLED,
        "parallel": PARALLEL_ENABLED,
        "num_threads": get_num_threads(),
        "numba_version": _NUMBA_VERSION,
    }
