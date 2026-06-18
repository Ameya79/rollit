import numpy as np
from .core import _make_windows

def _validate(arr, window, min_periods=None):
    """Check that inputs to rolling functions are valid.

    Raises TypeError if *arr* is not an ndarray, and ValueError for
    non-1D arrays, empty arrays, bad *window* sizes, or invalid
    *min_periods* values.
    """
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(arr).__name__}")
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got {arr.ndim}D")
    if len(arr) == 0:
        raise ValueError("Input array must not be empty")
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window}")
    if window > len(arr):
        raise ValueError(f"window ({window}) exceeds array length ({len(arr)})")
    if min_periods is not None:
        if isinstance(min_periods, bool) or not isinstance(min_periods, int) or min_periods < 1:
            raise ValueError(f"min_periods must be a positive integer, got {min_periods}")
        if min_periods > window:
            raise ValueError(f"min_periods ({min_periods}) cannot be greater than window ({window})")

def _apply_min_periods(windows: np.ndarray, result: np.ndarray, min_periods: int) -> np.ndarray:
    """Mask positions where the window has fewer than *min_periods* valid values.

    Converts *result* to float64 and sets entries to NaN wherever the
    corresponding window contains fewer than *min_periods* non-NaN
    observations.  If *min_periods* is None the result is returned
    unchanged (aside from the dtype cast).
    """
    # Ensure result is float64 as per requirements
    result = result.astype(np.float64)
    if min_periods is None:
        return result
    # Count non-nan values in each window
    valid_counts = np.sum(~np.isnan(windows), axis=1)
    result[valid_counts < min_periods] = np.nan
    return result

def mean(arr: np.ndarray, window: int, min_periods: int = None) -> np.ndarray:
    """Compute the rolling mean over a 1D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array of numeric values.
    window : int
        Size of the moving window. Must be a positive integer.
    min_periods : int, optional
        Minimum number of non-NaN observations in a window required to
        produce a value. If a window has fewer valid values, NaN is
        returned for that position. Defaults to None (no minimum).

    Returns
    -------
    np.ndarray
        1D array of length ``len(arr) - window + 1`` containing the
        rolling mean for each window position.

    Examples
    --------
    >>> import numpy as np
    >>> import rollit
    >>> rollit.mean(np.array([1., 2., 3., 4., 5.]), window=3)
    array([2., 3., 4.])
    """
    _validate(arr, window, min_periods)
    windows = _make_windows(arr, window)
    result = np.mean(windows, axis=1)
    return _apply_min_periods(windows, result, min_periods)

def std(arr: np.ndarray, window: int, min_periods: int = None) -> np.ndarray:
    """Compute the rolling sample standard deviation over a 1D numpy array.

    Uses ``ddof=1`` (Bessel's correction) so the result is the sample
    standard deviation rather than the population standard deviation.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array of numeric values.
    window : int
        Size of the moving window. Must be a positive integer.
    min_periods : int, optional
        Minimum number of non-NaN observations in a window required to
        produce a value. If a window has fewer valid values, NaN is
        returned for that position. Defaults to None (no minimum).

    Returns
    -------
    np.ndarray
        1D array of length ``len(arr) - window + 1`` containing the
        rolling sample standard deviation for each window position.

    Examples
    --------
    >>> import numpy as np
    >>> import rollit
    >>> rollit.std(np.array([1., 2., 3., 4., 5.]), window=3)
    array([1., 1., 1.])
    """
    _validate(arr, window, min_periods)
    windows = _make_windows(arr, window)
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.std(windows, axis=1, ddof=1)
    return _apply_min_periods(windows, result, min_periods)

def sum(arr: np.ndarray, window: int, min_periods: int = None) -> np.ndarray:
    """Compute the rolling sum over a 1D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array of numeric values.
    window : int
        Size of the moving window. Must be a positive integer.
    min_periods : int, optional
        Minimum number of non-NaN observations in a window required to
        produce a value. If a window has fewer valid values, NaN is
        returned for that position. Defaults to None (no minimum).

    Returns
    -------
    np.ndarray
        1D array of length ``len(arr) - window + 1`` containing the
        rolling sum for each window position.

    Examples
    --------
    >>> import numpy as np
    >>> import rollit
    >>> rollit.sum(np.array([1., 2., 3., 4., 5.]), window=3)
    array([ 6.,  9., 12.])
    """
    _validate(arr, window, min_periods)
    windows = _make_windows(arr, window)
    result = np.sum(windows, axis=1)
    return _apply_min_periods(windows, result, min_periods)

def min(arr: np.ndarray, window: int, min_periods: int = None) -> np.ndarray:
    """Compute the rolling minimum over a 1D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array of numeric values.
    window : int
        Size of the moving window. Must be a positive integer.
    min_periods : int, optional
        Minimum number of non-NaN observations in a window required to
        produce a value. If a window has fewer valid values, NaN is
        returned for that position. Defaults to None (no minimum).

    Returns
    -------
    np.ndarray
        1D array of length ``len(arr) - window + 1`` containing the
        rolling minimum for each window position.

    Examples
    --------
    >>> import numpy as np
    >>> import rollit
    >>> rollit.min(np.array([3., 1., 4., 1., 5.]), window=3)
    array([1., 1., 1.])
    """
    _validate(arr, window, min_periods)
    windows = _make_windows(arr, window)
    result = np.min(windows, axis=1)
    return _apply_min_periods(windows, result, min_periods)

def max(arr: np.ndarray, window: int, min_periods: int = None) -> np.ndarray:
    """Compute the rolling maximum over a 1D numpy array.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array of numeric values.
    window : int
        Size of the moving window. Must be a positive integer.
    min_periods : int, optional
        Minimum number of non-NaN observations in a window required to
        produce a value. If a window has fewer valid values, NaN is
        returned for that position. Defaults to None (no minimum).

    Returns
    -------
    np.ndarray
        1D array of length ``len(arr) - window + 1`` containing the
        rolling maximum for each window position.

    Examples
    --------
    >>> import numpy as np
    >>> import rollit
    >>> rollit.max(np.array([3., 1., 4., 1., 5.]), window=3)
    array([4., 4., 5.])
    """
    _validate(arr, window, min_periods)
    windows = _make_windows(arr, window)
    result = np.max(windows, axis=1)
    return _apply_min_periods(windows, result, min_periods)

def zscore(arr: np.ndarray, window: int, min_periods: int = None) -> np.ndarray:
    """Compute the rolling z-score of the last value in each window.

    For each window the z-score is calculated as
    ``(last - mean) / std`` where *last* is the final element of the
    window, *mean* is the window mean, and *std* is the sample standard
    deviation (``ddof=1``).  Returns NaN for positions where the
    standard deviation is zero.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array of numeric values.
    window : int
        Size of the moving window. Must be a positive integer.
    min_periods : int, optional
        Minimum number of non-NaN observations in a window required to
        produce a value. If a window has fewer valid values, NaN is
        returned for that position. Defaults to None (no minimum).

    Returns
    -------
    np.ndarray
        1D array of length ``len(arr) - window + 1`` containing the
        rolling z-score for each window position.

    Examples
    --------
    >>> import numpy as np
    >>> import rollit
    >>> rollit.zscore(np.array([1., 2., 3., 4., 5.]), window=3)
    array([1., 1., 1.])
    """
    _validate(arr, window, min_periods)
    windows = _make_windows(arr, window)
    with np.errstate(divide='ignore', invalid='ignore'):
        mu = np.mean(windows, axis=1)
        sigma = np.std(windows, axis=1, ddof=1)
        last_values = arr[window - 1:]
        result = (last_values - mu) / np.where(sigma == 0, np.nan, sigma)
    return _apply_min_periods(windows, result, min_periods)

def normalize(arr: np.ndarray, window: int, min_periods: int = None) -> np.ndarray:
    """Compute rolling min-max normalization of the last value in each window.

    For each window the normalized value is calculated as
    ``(last - min) / (max - min)`` where *last* is the final element of
    the window.  Returns NaN for positions where ``max == min``.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array of numeric values.
    window : int
        Size of the moving window. Must be a positive integer.
    min_periods : int, optional
        Minimum number of non-NaN observations in a window required to
        produce a value. If a window has fewer valid values, NaN is
        returned for that position. Defaults to None (no minimum).

    Returns
    -------
    np.ndarray
        1D array of length ``len(arr) - window + 1`` containing the
        rolling min-max normalized value for each window position.

    Examples
    --------
    >>> import numpy as np
    >>> import rollit
    >>> rollit.normalize(np.array([1., 2., 3., 4., 5.]), window=3)
    array([1., 1., 1.])
    """
    _validate(arr, window, min_periods)
    windows = _make_windows(arr, window)
    with np.errstate(divide='ignore', invalid='ignore'):
        w_min = np.min(windows, axis=1)
        w_max = np.max(windows, axis=1)
        last_values = arr[window - 1:]
        denom = np.where((w_max - w_min) == 0, np.nan, w_max - w_min)
        result = (last_values - w_min) / denom
    return _apply_min_periods(windows, result, min_periods)

def apply(arr: np.ndarray, window: int, fn, min_periods: int = None) -> np.ndarray:
    """Apply a custom function to each rolling window.

    Iterates over windows using a Python loop and calls *fn* on each
    one, so it is significantly slower than the vectorized functions in
    this module.  Use it only when the desired operation cannot be
    expressed with the built-in rolling functions.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D array of numeric values.
    window : int
        Size of the moving window. Must be a positive integer.
    fn : callable
        A function that takes a 1D numpy array (the window) and returns
        a scalar value.
    min_periods : int, optional
        Minimum number of non-NaN observations in a window required to
        produce a value. If a window has fewer valid values, NaN is
        returned for that position. Defaults to None (no minimum).

    Returns
    -------
    np.ndarray
        1D array of length ``len(arr) - window + 1`` containing the
        result of applying *fn* to each window.

    Examples
    --------
    >>> import numpy as np
    >>> import rollit
    >>> rollit.apply(np.array([1., 2., 3., 4., 5.]), window=3, fn=np.median)
    array([2., 3., 4.])
    """
    _validate(arr, window, min_periods)
    windows = _make_windows(arr, window)
    result = np.array([fn(w) for w in windows])
    return _apply_min_periods(windows, result, min_periods)
