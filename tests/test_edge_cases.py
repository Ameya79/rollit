import numpy as np
import pytest
import rollit

def test_input_is_list_raises():
    with pytest.raises(TypeError, match="Expected np.ndarray"):
        rollit.mean([1, 2, 3], window=2)

def test_input_is_2d_raises():
    arr = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="Expected 1D array"):
        rollit.mean(arr, window=2)

def test_empty_array_raises():
    arr = np.array([])
    with pytest.raises(ValueError, match="Input array must not be empty"):
        rollit.mean(arr, window=1)

def test_window_zero_raises(small_arr):
    with pytest.raises(ValueError, match="window must be a positive integer"):
        rollit.mean(small_arr, window=0)

def test_window_negative_raises(small_arr):
    with pytest.raises(ValueError, match="window must be a positive integer"):
        rollit.mean(small_arr, window=-1)

def test_window_float_raises(small_arr):
    with pytest.raises(ValueError, match="window must be a positive integer"):
        rollit.mean(small_arr, window=3.0)

def test_window_bool_raises(small_arr):
    # Booleans are subclasses of int in Python, so isinstance(True, int) is True.
    # We explicitly check for bool in _validate.
    with pytest.raises(ValueError, match="window must be a positive integer"):
        rollit.mean(small_arr, window=True)

def test_window_larger_than_arr_raises(small_arr):
    with pytest.raises(ValueError, match="exceeds array length"):
        rollit.mean(small_arr, window=10)

def test_single_element_array():
    arr = np.array([42.0])
    result = rollit.mean(arr, window=1)
    assert len(result) == 1
    assert result[0] == 42.0

def test_all_nan_input():
    arr = np.array([np.nan, np.nan, np.nan])
    result = rollit.mean(arr, window=2)
    assert len(result) == 2
    assert np.all(np.isnan(result))

def test_integer_array():
    arr = np.array([1, 2, 3, 4, 5])
    result = rollit.sum(arr, window=3)
    # Output length should be 3
    assert len(result) == 3
    np.testing.assert_array_almost_equal(result, [6.0, 9.0, 12.0])

def test_output_is_always_float():
    arr = np.array([1, 2, 3, 4, 5])
    
    res_mean = rollit.mean(arr, window=3)
    res_std = rollit.std(arr, window=3)
    res_sum = rollit.sum(arr, window=3)
    res_min = rollit.min(arr, window=3)
    res_max = rollit.max(arr, window=3)
    res_zscore = rollit.zscore(arr, window=3)
    res_normalize = rollit.normalize(arr, window=3)
    res_apply = rollit.apply(arr, window=3, fn=np.sum)
    
    assert res_mean.dtype == np.float64
    assert res_std.dtype == np.float64
    assert res_sum.dtype == np.float64
    assert res_min.dtype == np.float64
    assert res_max.dtype == np.float64
    assert res_zscore.dtype == np.float64
    assert res_normalize.dtype == np.float64
    assert res_apply.dtype == np.float64

def test_min_periods_validation(small_arr):
    with pytest.raises(ValueError, match="min_periods must be a positive integer"):
        rollit.mean(small_arr, window=3, min_periods=0)
    with pytest.raises(ValueError, match="min_periods must be a positive integer"):
        rollit.mean(small_arr, window=3, min_periods=-2)
    with pytest.raises(ValueError, match="min_periods must be a positive integer"):
        rollit.mean(small_arr, window=3, min_periods=2.5)
    with pytest.raises(ValueError, match="min_periods must be a positive integer"):
        rollit.mean(small_arr, window=3, min_periods=True)
    with pytest.raises(ValueError, match="cannot be greater than window"):
        rollit.mean(small_arr, window=3, min_periods=4)
