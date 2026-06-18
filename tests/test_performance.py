import pytest
import time
import numpy as np
import rollit

@pytest.mark.slow
def test_mean_1m_elements(large_arr):
    start = time.perf_counter()
    rollit.mean(large_arr, window=30)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.15  # must complete in under 150ms


@pytest.mark.slow
def test_apply_is_slower_than_mean(large_arr):
    t1 = time.perf_counter()
    rollit.mean(large_arr[:10_000], window=30)
    mean_time = time.perf_counter() - t1

    t2 = time.perf_counter()
    rollit.apply(large_arr[:10_000], window=30, fn=np.mean)
    apply_time = time.perf_counter() - t2

    assert apply_time > mean_time  # apply is always slower — document this fact
