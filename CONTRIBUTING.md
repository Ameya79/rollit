# Contributing to rollit

First off, thank you for considering contributing to `rollit`! It's people like you that make the open-source community such a great place.

## Setup

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rollit.git
   cd rollit
   ```
3. Create a virtual environment and install the development dependencies:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   pip install numpy pytest pytest-cov build twine
   ```
4. Run the test suite to verify everything works:
   ```bash
   pytest tests/ -v
   ```

## Making Changes

- Create a descriptive branch for your work: `git checkout -b feature/my-cool-feature` or `git checkout -b bugfix/fix-some-bug`.
- Write your code changes.
- Add unit tests for your changes in the appropriate file in the `tests/` directory.
- Ensure all tests pass:
   ```bash
   pytest tests/ -v
   ```
- Ensure code coverage remains above 90%:
   ```bash
   pytest tests/ --cov=rollit --cov-fail-under=90
   ```

## Adding a New Function

1. Implement the logic in [stats.py](rollit/stats.py).
2. Export the function in [__init__.py](rollit/__init__.py).
3. Add tests in [test_stats.py](tests/test_stats.py).
4. Update `CHANGELOG.md` with your additions.

## Submitting a Pull Request

1. Push your branch to GitHub:
   ```bash
   git push origin feature/my-cool-feature
   ```
2. Open a Pull Request against the `main` branch.
3. Provide a clear description of the problem solved or the feature added.
4. Verify that all status checks and tests pass in the GitHub Actions tab.
