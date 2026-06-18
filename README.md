<p align="center">
  <img src="rollit_logo.png" alt="rollit logo" width="200">
</p>

<h1 align="center">rollit</h1>

<p align="center">
  <strong>Rolling window statistics for numpy arrays. No pandas needed.</strong>
</p>

<p align="center">
  <a href="https://github.com/Ameya79/rollit/actions/workflows/ci.yml"><img src="https://github.com/Ameya79/rollit/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/rollit/"><img src="https://img.shields.io/pypi/v/rollit" alt="PyPI"></a>
  <a href="https://pypi.org/project/rollit/"><img src="https://img.shields.io/pypi/dm/rollit" alt="Downloads"></a>
  <a href="https://github.com/Ameya79/rollit/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Ameya79/rollit" alt="License"></a>
  <a href="https://pypi.org/project/rollit/"><img src="https://img.shields.io/pypi/pyversions/rollit" alt="Python"></a>
</p>

---

## The Problem

When working with raw numpy arrays — sensor data, stock prices, fitness metrics, ML features — developers constantly need rolling window statistics. There is no built-in numpy function for this. Every developer either:

- 🐢 Writes a **slow Python for-loop** every time
- 📦 Converts to a **pandas DataFrame** just to call `.rolling()`, adding a **35 MB dependency** for a single operation
- 🤷 Copies a **hacky `stride_tricks` snippet** from StackOverflow that they don't fully understand

**`rollit` solves this.** Clean API. Fast. Zero dependencies beyond numpy.

---

## Installation

```bash
pip install rollit
```

---

## Quick Start

```python
import numpy as np
import rollit

arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

rollit.mean(arr, window=3)   # array([2., 3., 4.])
rollit.sum(arr, window=3)    # array([6., 9., 12.])
rollit.std(arr, window=3)    # array([1., 1., 1.])
rollit.min(arr, window=3)    # array([1., 2., 3.])
rollit.max(arr, window=3)    # array([3., 4., 5.])
```

---

## Before & After

### Before (pandas — 35 MB dependency)

```python
import numpy as np
import pandas as pd

arr = np.random.randn(1_000_000)
df = pd.DataFrame(arr)
rolling_mean = df.rolling(window=30).mean().values.flatten()[29:]
```

### Before (manual loop — ~500 ms)

```python
import numpy as np

arr = np.random.randn(1_000_000)
rolling_mean = []
for i in range(len(arr) - 29):
    rolling_mean.append(np.mean(arr[i:i+30]))
rolling_mean = np.array(rolling_mean)
```

### After (rollit — ~5 ms, zero extra dependencies)

```python
import numpy as np
import rollit

arr = np.random.randn(1_000_000)
rolling_mean = rollit.mean(arr, window=30)
```

---

## API Reference

All functions share a unified signature:

```python
rollit.<function>(arr, window, min_periods=None)
```

| Parameter | Type | Description |
|---|---|---|
| `arr` | `np.ndarray` | Input 1D numpy array of numeric values |
| `window` | `int` | Size of the moving window (positive integer) |
| `min_periods` | `int`, optional | Minimum number of non-NaN observations required to produce a value. Defaults to `None` |

### Core Functions

| Function | Description | Example Output |
|---|---|---|
| `rollit.mean(arr, 3)` | Rolling average | `array([2., 3., 4.])` |
| `rollit.std(arr, 3)` | Rolling sample standard deviation (ddof=1) | `array([1., 1., 1.])` |
| `rollit.sum(arr, 3)` | Rolling sum | `array([6., 9., 12.])` |
| `rollit.min(arr, 3)` | Rolling minimum | `array([1., 2., 3.])` |
| `rollit.max(arr, 3)` | Rolling maximum | `array([3., 4., 5.])` |

*For `arr = [1.0, 2.0, 3.0, 4.0, 5.0]`*

### Anomaly Detection — `zscore()`

Computes the z-score of the **last value** in each window: `(last - mean) / std`.
Values above **+2** or below **−2** are statistical outliers.

```python
# Flat signal with one spike
signal = np.ones(20)
signal[10] = 100.0

scores = rollit.zscore(signal, window=5)
# The window containing the spike will have a very high z-score
```

> **Note:** Returns `NaN` when the standard deviation of a window is 0 (constant values).

### Normalization — `normalize()`

Computes rolling min-max normalization of the **last value** in each window: `(last - min) / (max - min)`.
Scales each value to `[0, 1]` relative to its local window.

```python
arr = np.array([1.0, 5.0, 2.0, 9.0, 1.0])
rollit.normalize(arr, window=3)
# array([0.25, 1.  , 0.  ])
```

> **Note:** Returns `NaN` when all values in a window are equal (`max == min`).

### Custom Function — `apply()`

Escape hatch for any function that can't be vectorized. Runs a Python loop internally, so it's **slower** than the built-in functions.

```python
# Rolling median
rollit.apply(arr, window=3, fn=np.median)
# array([2., 3., 4.])

# Rolling range
rollit.apply(arr, window=3, fn=lambda w: w.max() - w.min())
# array([2., 3., 3.])
```

---

## Output Length

`rollit` only returns values for **complete windows** — no NaN padding:

```
input length  = L
window size   = W
output length = L - W + 1
```

For an array of length 10 with window 3:
- Output length = **8**
- Output index `0` ← `arr[0:3]`
- Output index `7` ← `arr[7:10]`

---

## Performance

All functions (except `apply`) use `numpy.lib.stride_tricks.as_strided` to create read-only memory views — no data is copied.

| Array Size | Window | Function | Time |
|---|---|---|---|
| 10,000 | 7 | `mean` | < 1 ms |
| 1,000,000 | 30 | `mean` | < 10 ms |
| 1,000,000 | 30 | `apply` (custom fn) | ~800 ms |

> `apply()` uses a Python loop by design. Use vectorized functions whenever possible.

---

## Important Notes

- **Input must be a 1D numpy array.** Passing a Python list raises `TypeError`. Passing a 2D array raises `ValueError`.
- **`std()` uses `ddof=1`** (sample standard deviation), matching the pandas default.
- **Avoid `from rollit import *`** — `sum`, `min`, and `max` would shadow Python builtins. Use `import rollit` and call `rollit.sum()`, `rollit.min()`, `rollit.max()` instead.
- **`min_periods` in v1.0** counts non-NaN values per window and masks positions below the threshold with NaN. However, the underlying numpy operations (e.g., `np.mean`) are **not** NaN-aware — a window containing any NaN will naturally produce NaN regardless of `min_periods`. Full NaN-aware support (`nanmean`, etc.) is planned for v2.0.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, guidelines, and how to submit a PR.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

## License

MIT — see [LICENSE](LICENSE) for details.
