import numpy as np
import rollit

print(f"Using rollit version: {rollit.__version__}")
print("-" * 50)

# 1. Create sample 1D array
arr = np.array([10.0, 20.0, 15.0, 30.0, 25.0])
print(f"Input Array: {arr}")
window = 3

# 2. Rolling Mean
rolling_mean = rollit.mean(arr, window=window)
print(f"Rolling Mean (window={window}): {rolling_mean}")

# 3. Rolling Sum
rolling_sum = rollit.sum(arr, window=window)
print(f"Rolling Sum (window={window}): {rolling_sum}")

# 4. Rolling Std
rolling_std = rollit.std(arr, window=window)
print(f"Rolling Std (window={window}): {rolling_std}")

# 5. Rolling Z-Score (calculates z-score of the last value in each window)
# Outlier detection showcase
signal = np.array([1.0, 1.1, 1.0, 1.2, 100.0])
zscores = rollit.zscore(signal, window=5)
print(f"\nSignal with outlier: {signal}")
print(f"Rolling Z-Score (window=5): {zscores}")

# 6. Rolling Min-Max Normalization (scales the last value to [0, 1] relative to the window)
norm_val = rollit.normalize(arr, window=window)
print(f"\nRolling Normalization (window={window}): {norm_val}")

# 7. Custom apply function (e.g., computing window range: max - min)
range_fn = lambda w: w.max() - w.min()
rolling_range = rollit.apply(arr, window=window, fn=range_fn)
print(f"Custom Rolling Range (window={window}): {rolling_range}")

# 8. Handling min_periods
arr_with_nans = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
mean_min_periods = rollit.mean(arr_with_nans, window=3, min_periods=2)
print(f"\nArray with NaNs: {arr_with_nans}")
# Note: NP functions naturally propagate NaN, but min_periods handles the masking
print(f"Rolling Mean (window=3, min_periods=2): {mean_min_periods}")