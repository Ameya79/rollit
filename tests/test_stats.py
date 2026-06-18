import numpy as np
import pytest
import rollit

# --- mean() ---
def test_mean_values(small_arr):
    result = rollit.mean(small_arr, window=3)
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_mean_output_length(small_arr):
    result = rollit.mean(small_arr, window=3)
    assert len(result) == len(small_arr) - 3 + 1

def test_mean_output_is_ndarray(small_arr):
    result = rollit.mean(small_arr, window=3)
    assert isinstance(result, np.ndarray)

def test_mean_float_input():
    arr = np.array([1.5, 2.5, 3.5, 4.5])
    result = rollit.mean(arr, window=2)
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)

# --- std() ---
def test_std_ddof1(small_arr):
    result = rollit.std(small_arr, window=3)
    expected = np.array([
        np.std([1, 2, 3], ddof=1),
        np.std([2, 3, 4], ddof=1),
        np.std([3, 4, 5], ddof=1)
    ])
    np.testing.assert_array_almost_equal(result, expected)

def test_std_constant_array(constant_arr):
    result = rollit.std(constant_arr, window=3)
    expected = np.zeros(8)
    np.testing.assert_array_almost_equal(result, expected)

# --- sum() ---
def test_sum_values(small_arr):
    result = rollit.sum(small_arr, window=3)
    expected = np.array([6.0, 9.0, 12.0])
    np.testing.assert_array_almost_equal(result, expected)

# --- min() and max() ---
def test_min_values(small_arr):
    result = rollit.min(small_arr, window=3)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_max_values(small_arr):
    result = rollit.max(small_arr, window=3)
    expected = np.array([3.0, 4.0, 5.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_min_max_relationship(small_arr):
    arr = np.array([5.0, 2.0, 9.0, 1.0, 7.0])
    mins = rollit.min(arr, window=3)
    maxs = rollit.max(arr, window=3)
    assert np.all(mins <= maxs)

# --- zscore() ---
def test_zscore_outlier_detected(arr_with_outlier):
    result = rollit.zscore(arr_with_outlier, window=5)
    # The spike is at index 10. The window ending at index 10 has outlier at its last index.
    # So the value at index 10 (which is 100.0) should have a huge zscore compared to
    # the window [1.0, 1.0, 1.0, 1.0, 100.0]
    # mean = 20.8, std = np.std([1,1,1,1,100], ddof=1) = ~44.27
    # zscore = (100 - 20.8) / 44.27 = ~1.78. Wait, is it > 1.5? Yes.
    # Let's verify exact index of the zscore. The window [1, 1, 1, 1, 100] ends at index 10,
    # which maps to output index 10 - 5 + 1 = 6.
    assert result[6] > 1.5

def test_zscore_constant_array(constant_arr):
    result = rollit.zscore(constant_arr, window=3)
    # Standard deviation is 0, so division by zero returns NaN
    assert np.all(np.isnan(result))

def test_zscore_output_length(small_arr):
    result = rollit.zscore(small_arr, window=3)
    assert len(result) == len(small_arr) - 3 + 1

# --- normalize() ---
def test_normalize_range():
    arr = np.array([1.0, 5.0, 2.0, 9.0, 1.0])
    result = rollit.normalize(arr, window=3)
    # For each window, the value is normalized relative to min and max of that window
    # Window 0: [1, 5, 2] -> min=1, max=5. last_val=2 -> (2-1)/(5-1) = 0.25
    # Window 1: [5, 2, 9] -> min=2, max=9. last_val=9 -> (9-2)/(9-2) = 1.0
    # Window 2: [2, 9, 1] -> min=1, max=9. last_val=1 -> (1-1)/(9-1) = 0.0
    expected = np.array([0.25, 1.0, 0.0])
    np.testing.assert_array_almost_equal(result, expected)
    assert np.all((result >= 0.0) & (result <= 1.0))

def test_normalize_constant_array(constant_arr):
    result = rollit.normalize(constant_arr, window=3)
    assert np.all(np.isnan(result))

# --- apply() ---
def test_apply_median(small_arr):
    result = rollit.apply(small_arr, window=3, fn=np.median)
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_apply_custom_fn(small_arr):
    result = rollit.apply(small_arr, window=3, fn=lambda x: x[0] + x[-1])
    # Window 1: [1, 2, 3] -> 1 + 3 = 4
    # Window 2: [2, 3, 4] -> 2 + 4 = 6
    # Window 3: [3, 4, 5] -> 3 + 5 = 8
    expected = np.array([4.0, 6.0, 8.0])
    np.testing.assert_array_almost_equal(result, expected)

def test_apply_output_length(small_arr):
    result = rollit.apply(small_arr, window=3, fn=np.mean)
    assert len(result) == len(small_arr) - 3 + 1

# --- min_periods ---
def test_min_periods_handling():
    arr = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
    # Windows of size 3:
    # 0: [1, nan, 3] -> 2 valid
    # 1: [nan, 3, 4] -> 2 valid
    # 2: [3, 4, 5]   -> 3 valid
    
    # If min_periods=3: window 0 and 1 should be NaN
    res_mean_3 = rollit.mean(arr, window=3, min_periods=3)
    assert np.isnan(res_mean_3[0])
    assert np.isnan(res_mean_3[1])
    assert res_mean_3[2] == 4.0
    
    # If min_periods=2: window 0 and 1 should NOT be NaN (using np.nanmean behavior? No, np.mean with NaNs is NaN unless we use min_periods to mask it. Wait, np.mean([1.0, np.nan, 3.0]) is np.nan.
    # Ah, the PRD says: "All functions return np.nan for windows with fewer valid values than min_periods".
    # It does not mean they ignore NaNs if they have enough valid values, unless they are nan-aware (which version plan says is v1.1 scope: "No NaN-ignoring variants like nanmean (add in v1.1)").
    # So np.mean([1.0, np.nan, 3.0]) is naturally np.nan. But with min_periods=3, it is also NaN because counts < 3.
    # Let's verify with an array without NaN, but we test min_periods validation and behavior.
    # Wait, if min_periods=None, it behaves standard.
    # Let's test standard min_periods check with clean inputs:
    arr_clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # With min_periods=2, all windows of size 3 have 3 valid values >= 2, so no extra masking.
    res_mean_2 = rollit.mean(arr_clean, window=3, min_periods=2)
    np.testing.assert_array_almost_equal(res_mean_2, [2.0, 3.0, 4.0])
