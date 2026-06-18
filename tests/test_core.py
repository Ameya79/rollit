import numpy as np
import pytest
from rollit.core import _make_windows

def test_output_shape(small_arr):
    result = _make_windows(small_arr, window=3)
    assert result.shape == (3, 3)  # 5 - 3 + 1 = 3 rows

def test_window_values_correct(small_arr):
    result = _make_windows(small_arr, window=3)
    expected = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
        [3.0, 4.0, 5.0]
    ])
    np.testing.assert_array_equal(result, expected)

def test_is_view_not_copy(small_arr):
    result = _make_windows(small_arr, window=3)
    assert np.shares_memory(result, small_arr)

def test_window_equals_length(small_arr):
    result = _make_windows(small_arr, window=5)
    assert result.shape == (1, 5)
    np.testing.assert_array_equal(result[0], small_arr)

def test_window_larger_than_arr(small_arr):
    with pytest.raises(ValueError, match="Window size 6 exceeds array length 5"):
        _make_windows(small_arr, window=6)

def test_window_of_one(small_arr):
    result = _make_windows(small_arr, window=1)
    assert result.shape == (5, 1)
    expected = small_arr.reshape(-1, 1)
    np.testing.assert_array_equal(result, expected)
