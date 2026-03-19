"""
Tests for the tod_calibrate module.

Covers: _memory_cap, _candidate_batch_sizes.

Can be run independently:
    pytest tests/test_tod_calibrate.py -v
    python tests/test_tod_calibrate.py
"""

import os
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Ensure project root and stubs are available when run as a standalone file.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

for _mod_name in ["pixell", "pixell.enmap"]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = MagicMock()

if "tod_io" not in sys.modules:
    sys.modules["tod_io"] = MagicMock()

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from tod_calibrate import _memory_cap, _candidate_batch_sizes


# ===========================================================================
# TestMemoryCap
# ===========================================================================

class TestMemoryCap:
    """Tests for tod_calibrate._memory_cap."""

    def test_basic_arithmetic(self):
        """Computes max(1, int(gb*1e9/1.5 / (100*sel))) correctly."""
        gb  = 4.0
        sel = 1000
        expected = max(1, int(gb * 1e9 / 1.5 / (100 * sel)))
        assert _memory_cap(gb, sel) == expected

    def test_returns_at_least_one_with_tiny_budget(self):
        """Returns at least 1 even when memory budget is extremely small."""
        assert _memory_cap(1e-9, 10**9) >= 1

    def test_scales_linearly_with_memory(self):
        """Result doubles when max_memory_gb doubles."""
        sel = 500
        r1  = _memory_cap(2.0, sel)
        r2  = _memory_cap(4.0, sel)
        assert abs(r2 - 2 * r1) <= 1

    def test_scales_inversely_with_beam_sel(self):
        """Result halves (approx) when max_beam_sel doubles."""
        gb = 8.0
        r1 = _memory_cap(gb, 100)
        r2 = _memory_cap(gb, 200)
        assert abs(r1 - 2 * r2) <= 1


# ===========================================================================
# TestCandidateBatchSizes
# ===========================================================================

class TestCandidateBatchSizes:
    """Tests for tod_calibrate._candidate_batch_sizes."""

    def test_mem_cap_is_last_element(self):
        """mem_cap is always the last element in the returned list."""
        for mem_cap in [1, 7, 16, 100, 1000]:
            result = _candidate_batch_sizes(mem_cap)
            assert result[-1] == mem_cap

    def test_all_values_le_mem_cap(self):
        """All returned values are <= mem_cap."""
        for mem_cap in [1, 4, 64, 500]:
            result = _candidate_batch_sizes(mem_cap)
            assert all(c <= mem_cap for c in result)

    def test_strictly_sorted_ascending(self):
        """The returned list is strictly sorted in ascending order."""
        for mem_cap in [1, 10, 128, 1024]:
            result = _candidate_batch_sizes(mem_cap)
            assert result == sorted(set(result))
            assert len(result) == len(set(result))

    def test_powers_of_two_included(self):
        """All powers of two from 2 up to the first power >= mem_cap are included."""
        mem_cap = 100
        result  = _candidate_batch_sizes(mem_cap)
        k = 1
        while True:
            p = 2 ** k
            if p > mem_cap:
                break
            assert p in result, f"Power of two {p} missing from {result}"
            k += 1

    def test_mem_cap_one(self):
        """Works correctly for mem_cap=1, returning a single-element list [1]."""
        result = _candidate_batch_sizes(1)
        assert result == [1]


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
