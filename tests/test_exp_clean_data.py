"""Tests for qpmix.exp.clean_data."""

from __future__ import annotations

import numpy as np
import pytest

from qpmix.exp import clean_data as cd


@pytest.fixture
def messy():
    x = np.array([2.0, 1.0, 1.0, np.nan, 3.0, 2.0])
    y = np.array([20.0, 10.0, 99.0, 5.0, 30.0, 88.0])
    return x, y


def test_remove_nans_xy():
    x = np.array([1.0, np.nan, 3.0])
    y = np.array([1.0, 2.0, np.nan])
    gx, gy = cd.remove_nans_xy(x, y)
    assert gx.tolist() == [1.0]
    assert gy.tolist() == [1.0]


def test_sort_xy_is_stable():
    x = np.array([2.0, 1.0, 2.0])
    y = np.array([20.0, 10.0, 21.0])
    gx, gy = cd.sort_xy(x, y)
    assert gx.tolist() == [1.0, 2.0, 2.0]
    assert gy.tolist() == [10.0, 20.0, 21.0]


def test_remove_doubles_keeps_the_first_of_each_run():
    x = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0])
    y = np.arange(6.0)
    gx, gy = cd.remove_doubles_xy(x, y)
    assert gx.tolist() == [1.0, 2.0, 3.0]
    assert gy.tolist() == [0.0, 2.0, 5.0]


def test_remove_doubles_rejects_unsorted_input():
    with pytest.raises(ValueError, match="sorted"):
        cd.remove_doubles_xy(np.array([3.0, 1.0, 2.0]), np.zeros(3))


def test_remove_doubles_can_skip_the_check():
    x = np.array([3.0, 1.0, 2.0])
    assert cd.remove_doubles_xy(x, np.zeros(3), check=False)[0].size == 3


def test_clean_xy_does_all_three(messy):
    x, y = cd.clean_xy(*messy)
    assert x.tolist() == [1.0, 2.0, 3.0]
    assert np.all(np.diff(x) > 0)
    assert not np.isnan(y).any()


def test_clean_xy_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same shape"):
        cd.clean_xy(np.zeros(3), np.zeros(4))


def test_clean_xy_of_already_clean_data_is_a_no_op():
    x = np.linspace(0, 1, 11)
    y = x**2
    gx, gy = cd.clean_xy(x, y)
    assert np.array_equal(gx, x)
    assert np.array_equal(gy, y)


def test_xy_matrix_round_trip():
    x = np.linspace(0, 1, 7)
    y = np.sin(x)
    mx, my = cd.matrix_to_xy(cd.xy_to_matrix(x, y))
    assert np.array_equal(mx, x)
    assert np.array_equal(my, y)


def test_matrix_helpers_match_the_xy_helpers(messy):
    x, y = messy
    from_xy = cd.xy_to_matrix(*cd.clean_xy(x, y))
    from_matrix = cd.clean_matrix(cd.xy_to_matrix(x, y))
    assert np.array_equal(from_xy, from_matrix)


def test_sort_matrix_by_second_column():
    mat = np.array([[1.0, 30.0], [2.0, 10.0], [3.0, 20.0]])
    assert cd.sort_matrix(mat, col=1)[:, 1].tolist() == [10.0, 20.0, 30.0]


def test_remove_doubles_matrix_rejects_unsorted():
    mat = np.array([[3.0, 1.0], [1.0, 2.0], [2.0, 3.0]])
    with pytest.raises(ValueError, match="sorted"):
        cd.remove_doubles_matrix(mat)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((4, 3)), np.zeros(4), np.zeros((2, 2, 2))],
)
def test_matrix_helpers_reject_wrong_shapes(bad):
    with pytest.raises(ValueError, match="2 columns"):
        cd.clean_matrix(bad)
