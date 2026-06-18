"""rollit — Rolling window statistics for numpy arrays.

Provides fast, vectorized rolling window operations on 1D numpy arrays
using stride tricks. Zero dependencies beyond numpy.
"""

from .stats import mean, std, sum, min, max, zscore, normalize, apply

__version__ = "1.0.1"
__all__ = ["mean", "std", "sum", "min", "max", "zscore", "normalize", "apply"]
