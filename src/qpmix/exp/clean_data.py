"""Cleaning experimental data: NaNs, sorting and repeated abscissae.

Data can be handled either as a pair of equal-length arrays (``x``/``y``) or
as a two-column matrix.  Both are supported; the matrix helpers are thin
wrappers over the x/y ones, so the two can never drift apart.

Examples:

    >>> import numpy as np
    >>> from qpmix.exp.clean_data import clean_xy
    >>> x = np.array([2.0, 1.0, 1.0, np.nan, 3.0])
    >>> y = np.array([20.0, 10.0, 99.0, 5.0, 30.0])
    >>> clean_xy(x, y)
    (array([1., 2., 3.]), array([10., 20., 30.]))

"""

from __future__ import annotations

import numpy as np

__all__ = [
    "clean_matrix",
    "clean_xy",
    "matrix_to_xy",
    "remove_doubles_matrix",
    "remove_doubles_xy",
    "remove_nans_matrix",
    "remove_nans_xy",
    "sort_matrix",
    "sort_xy",
    "xy_to_matrix",
]


def remove_nans_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop points where either coordinate is NaN.

    Args:
        x (ndarray): Abscissae.
        y (ndarray): Ordinates.

    Returns:
        tuple: ``(x, y)`` with NaN rows removed.

    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    return x[mask], y[mask]


def sort_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort both arrays by the abscissa.

    Args:
        x (ndarray): Abscissae.
        y (ndarray): Ordinates.

    Returns:
        tuple: ``(x, y)`` sorted by ``x``.

    """
    idx = np.argsort(x, kind="stable")
    return x[idx], y[idx]


def remove_doubles_xy(
    x: np.ndarray, y: np.ndarray, check: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Drop points that repeat the previous abscissa.

    Args:
        x (ndarray): Abscissae, already sorted.
        y (ndarray): Ordinates.
        check (bool, optional): Verify that ``x`` is sorted.  Default is
            True.

    Returns:
        tuple: ``(x, y)`` with repeated abscissae removed.

    Raises:
        ValueError: If ``check`` is True and ``x`` is not sorted.

    """
    if check and x.size > 1 and np.diff(x).min() < 0:
        raise ValueError("x must be sorted before removing doubles.")
    mask = np.ones(x.size, dtype=bool)
    mask[1:] = x[1:] != x[:-1]
    return x[mask], y[mask]


def clean_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove NaNs, sort by ``x``, then drop repeated abscissae.

    Args:
        x (ndarray): Abscissae.
        y (ndarray): Ordinates.

    Returns:
        tuple: The cleaned ``(x, y)``.

    Raises:
        ValueError: If the two arrays have different lengths.

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")
    x, y = remove_nans_xy(x, y)
    x, y = sort_xy(x, y)
    return remove_doubles_xy(x, y)


def xy_to_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Stack two arrays into a two-column matrix.

    Args:
        x (ndarray): First column.
        y (ndarray): Second column.

    Returns:
        ndarray: The matrix, shape ``(n, 2)``.

    """
    return np.vstack((x, y)).T


def matrix_to_xy(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a two-column matrix into two arrays.

    Args:
        matrix (ndarray): A two-column matrix.

    Returns:
        tuple: ``(first column, second column)``.

    """
    return matrix[:, 0], matrix[:, 1]


def _check_matrix(matrix: np.ndarray) -> np.ndarray:
    """Validate that the input is a two-column matrix.

    Args:
        matrix (ndarray): Candidate matrix.

    Returns:
        ndarray: The matrix as a float array.

    Raises:
        ValueError: If it is not two-dimensional with two columns.

    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != 2:
        raise ValueError("Matrix should have exactly 2 columns.")
    return matrix


def remove_nans_matrix(matrix: np.ndarray) -> np.ndarray:
    """Drop rows containing a NaN.

    Args:
        matrix (ndarray): A two-column matrix.

    Returns:
        ndarray: The matrix with NaN rows removed.

    """
    matrix = _check_matrix(matrix)
    return xy_to_matrix(*remove_nans_xy(*matrix_to_xy(matrix)))


def sort_matrix(matrix: np.ndarray, col: int = 0) -> np.ndarray:
    """Sort the rows of a matrix by one column.

    Args:
        matrix (ndarray): A two-column matrix.
        col (int, optional): Column to sort by.  Default is 0.

    Returns:
        ndarray: The sorted matrix.

    """
    matrix = _check_matrix(matrix)
    return matrix[np.argsort(matrix[:, col], kind="stable")]


def remove_doubles_matrix(
    matrix: np.ndarray, col: int = 0, check: bool = True
) -> np.ndarray:
    """Drop rows that repeat the previous value in one column.

    Args:
        matrix (ndarray): A two-column matrix, sorted by ``col``.
        col (int, optional): Column to deduplicate.  Default is 0.
        check (bool, optional): Verify that the column is sorted.  Default
            is True.

    Returns:
        ndarray: The matrix with repeated values removed.

    Raises:
        ValueError: If ``check`` is True and the column is not sorted.

    """
    matrix = _check_matrix(matrix)
    column = matrix[:, col]
    if check and column.size > 1 and np.diff(column).min() < 0:
        raise ValueError("Matrix must be sorted before removing doubles.")
    mask = np.ones(column.size, dtype=bool)
    mask[1:] = column[1:] != column[:-1]
    return matrix[mask]


def clean_matrix(matrix: np.ndarray) -> np.ndarray:
    """Remove NaNs, sort by the first column, drop repeated values.

    Args:
        matrix (ndarray): A two-column matrix.

    Returns:
        ndarray: The cleaned matrix.

    Raises:
        ValueError: If the input is not a two-column matrix.

    """
    matrix = _check_matrix(matrix)
    return xy_to_matrix(*clean_xy(*matrix_to_xy(matrix)))
