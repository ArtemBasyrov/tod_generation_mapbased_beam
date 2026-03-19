"""
Main entry point to run the full TOD test suite.

Usage:
    python tests/run_all_tests.py           # run all tests
    python tests/run_all_tests.py -v        # verbose
    python tests/run_all_tests.py -k "Fmt"  # filter by name

Individual modules can also be run independently:
    pytest tests/test_tod_utils.py -v
    pytest tests/test_tod_calibrate.py -v
    pytest tests/test_tod_core.py -v
    pytest tests/test_numba_healpy.py -v
"""

import sys
import os
import pytest

if __name__ == "__main__":
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    # Pass any extra CLI arguments (e.g. -v, -k "...", -x) straight through.
    extra_args = sys.argv[1:]
    sys.exit(pytest.main([tests_dir] + extra_args))
