import numpy as np
from numpy.lib.stride_tricks import as_strided

def _make_windows(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Converts 1D array into 2D array of sliding windows.
    Input:  [1, 2, 3, 4, 5], window=3
    Output: [[1, 2, 3],
             [2, 3, 4],
             [3, 4, 5]]
    """
    if window > len(arr):
        raise ValueError(f"Window size {window} exceeds array length {len(arr)}")
    
    shape = (len(arr) - window + 1, window)
    strides = (arr.strides[0], arr.strides[0])
    result = as_strided(arr, shape=shape, strides=strides)
    result.flags.writeable = False
    return result
