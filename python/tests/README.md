# Running Tests

## Installation

Install test dependencies:

```bash
pip install -r requirements-dev.txt
```

Or install both runtime and test dependencies:

```bash
pip install -e ".[dev]"
```

## Running Tests

Run all tests:

```bash
pytest
```

Run with verbose output:

```bash
pytest -v
```

Run a specific test file:

```bash
pytest tests/test_domain_calibrator.py
```

Run a specific test:

```bash
pytest tests/test_domain_calibrator.py::TestDomainCalibratorPy::test_init_success
```

## Test Coverage

Generate coverage report:

```bash
pytest --cov=domain_calibrator --cov-report=html
```

View coverage report:

```bash
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

## Test Structure

- `test_domain_calibrator.py` - Tests for DomainCalibratorPy class
- `test_domain.py` - Tests for Domain class and authentication
- `test_marker_calibrator.py` - Tests for QR code detection and pose estimation
- `test_auth.py` - Tests for domain authentication
- `conftest.py` - Shared fixtures and test configuration

All tests use mocking to avoid actual API calls, making them fast and reliable.

Note: `test_auth.py` may make real API calls to test authentication.
