"""
Tests for the tod_calibrate module.

Covers: _memory_cap, _candidate_batch_sizes, _calibrate_n_processes.

Can be run independently:
    pytest tests/test_tod_calibrate.py -v
    python tests/test_tod_calibrate.py
"""

import os
import sys
from unittest.mock import MagicMock, patch

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
from tod_calibrate import _memory_cap, _candidate_batch_sizes, _calibrate_n_processes


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


# ===========================================================================
# TestCalibrateNProcesses
# ===========================================================================

class TestCalibrateNProcesses:
    """
    Tests for tod_calibrate._calibrate_n_processes.

    _calibrate_batch_size and _get_memory_per_process are mocked so that the
    logic under test is the n-process selection algorithm only.
    """

    # Minimal beam_data: n_sel drives max_beam_sel = 1_000_000.
    _BEAM_DATA = {"beam_A": {"n_sel": 1_000_000}}

    def test_returns_n_within_ceiling(self):
        """n_optimal is always in [1, n_cpu_ceiling]."""
        results = [(1, 100), (2, 300), (3, 350), (4, 400), (6, 380)]
        with patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)), \
             patch("tod_calibrate._get_memory_per_process", return_value=1.0):
            n_opt, _ = _calibrate_n_processes(
                self._BEAM_DATA, ".", 0, [None], n_cpu_ceiling=4,
            )
        assert 1 <= n_opt <= 4

    def test_returns_at_least_one(self):
        """n_optimal is always >= 1."""
        results = [(1, 50), (2, 100)]
        with patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)), \
             patch("tod_calibrate._get_memory_per_process", return_value=1.0):
            n_opt, _ = _calibrate_n_processes(
                self._BEAM_DATA, ".", 0, [None], n_cpu_ceiling=1,
            )
        assert n_opt >= 1

    def test_ceiling_one_returns_one(self):
        """With n_cpu_ceiling=1, n_optimal is always 1."""
        results = [(1, 1000), (2, 2000), (4, 3000)]
        with patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)), \
             patch("tod_calibrate._get_memory_per_process", return_value=8.0):
            n_opt, _ = _calibrate_n_processes(
                self._BEAM_DATA, ".", 0, [None], n_cpu_ceiling=1,
            )
        assert n_opt == 1

    def test_maximizes_total_throughput(self):
        """
        The selected n maximizes n × per-process throughput.

        Setup (total_memory_gb=1.0, max_beam_sel=1_000_000):
          _memory_cap(1.0 / n, 1e6)  →  affordable batch sizes:
            n=1 → cap=6,  bs=4  (tp=400), total=  400
            n=2 → cap=3,  bs=3  (tp=350), total=  700
            n=3 → cap=2,  bs=2  (tp=300), total=  900  ← winner
            n=4 → cap=1,  bs=1  (tp=100), total=  400
        """
        results = [(1, 100), (2, 300), (3, 350), (4, 400), (6, 380)]
        with patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)), \
             patch("tod_calibrate._get_memory_per_process", return_value=1.0):
            n_opt, bs_opt = _calibrate_n_processes(
                self._BEAM_DATA, ".", 0, [None], n_cpu_ceiling=4,
            )
        assert n_opt  == 3
        assert bs_opt == 2

    def test_batch_size_is_affordable_for_optimal_n(self):
        """
        The returned batch_size never exceeds the memory cap for the optimal n.
        """
        results = [(1, 100), (2, 300), (3, 350), (4, 400), (6, 380)]
        total_gb = 1.0
        with patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)), \
             patch("tod_calibrate._get_memory_per_process", return_value=total_gb):
            n_opt, bs_opt = _calibrate_n_processes(
                self._BEAM_DATA, ".", 0, [None], n_cpu_ceiling=4,
            )
        max_beam_sel = 1_000_000
        cap = _memory_cap(total_gb / n_opt, max_beam_sel)
        assert bs_opt <= cap

    def test_all_affordable_skipped_when_memory_too_small(self):
        """
        If memory per process is so small that no batch size is affordable for
        some n values, those n are skipped and the result is still valid.
        """
        # Use a very large max_beam_sel so only n=1 has an affordable batch size.
        beam_data = {"beam_A": {"n_sel": 100_000_000}}
        # _memory_cap(1.0, 1e8) = max(1, int(1e9/1.5/1e10)) = max(1, 0) = 1
        # _memory_cap(0.5, 1e8) = max(1, int(0.5e9/1.5/1e10)) = max(1, 0) = 1
        # So cap=1 for all n; results must contain bs=1 to avoid "no affordable" skips.
        results = [(1, 500)]
        with patch("tod_calibrate._calibrate_batch_size", return_value=(None, results)), \
             patch("tod_calibrate._get_memory_per_process", return_value=1.0):
            n_opt, bs_opt = _calibrate_n_processes(
                beam_data, ".", 0, [None], n_cpu_ceiling=4,
            )
        # Any n in [1,4] is valid; what matters is the result is sane.
        assert 1 <= n_opt <= 4
        assert bs_opt == 1


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
