import pytest
import numpy as np

@pytest.fixture
def small_arr():
    # simple known array — easy to manually verify results
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

@pytest.fixture
def large_arr():
    # 1 million elements for performance tests
    np.random.seed(42)
    return np.random.randn(1_000_000)

@pytest.fixture
def arr_with_outlier():
    # flat signal with one obvious spike — for zscore tests
    arr = np.ones(20)
    arr[10] = 100.0
    return arr

@pytest.fixture
def constant_arr():
    # all same value — tests zero-division handling in std, zscore, normalize
    return np.ones(10)
