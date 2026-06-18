# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-06-18

### Added
- `mean()`, `std()`, `sum()`, `min()`, `max()`, `zscore()`, `normalize()`, `apply()` rolling stats operations.
- Stride-tricks based vectorized window view implementation in `_make_windows` for optimal performance.
- Complete input validation handling input types, shapes, dimensions, and window constraints.
- Integrated `min_periods` support across all statistical functions.
- Full test suite spanning unit, edge case, and performance benchmark tests.
- GitHub Actions CI/CD workflows.
