"""
Tests for the tod_utils module.

Covers: _fmt_time, _should_print_batch, _is_cluster,
        _get_memory_per_process, _get_ncpus.

Can be run independently:
    pytest tests/test_tod_utils.py -v
    python tests/test_tod_utils.py
"""

import os
import sys
import importlib
from unittest.mock import MagicMock, patch

import numpy.testing as npt
import pytest

# ---------------------------------------------------------------------------
# Ensure project root and stubs are available when run as a standalone file
# (conftest.py handles this automatically under pytest).
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
from tod_utils import (
    _fmt_time,
    _should_print_batch,
    _is_cluster,
    _get_memory_per_process,
    _get_ncpus,
)
import tod_config as config


# ===========================================================================
# TestFmtTime
# ===========================================================================


class TestFmtTime:
    """Tests for tod_utils._fmt_time."""

    def test_below_60_seconds(self):
        """Values under 60 seconds render as '{x:.2f}s'."""
        assert _fmt_time(0.0) == "0.00s"
        assert _fmt_time(1.5) == "1.50s"
        assert _fmt_time(59.99) == "59.99s"

    def test_minutes_range(self):
        """Values in [60, 3600) render as '{x/60:.2f}m'."""
        assert _fmt_time(60.0) == f"{60.0 / 60:.2f}m"
        assert _fmt_time(120.0) == f"{120.0 / 60:.2f}m"
        assert _fmt_time(3599.9) == f"{3599.9 / 60:.2f}m"

    def test_hours_range(self):
        """Values >= 3600 render as '{x/3600:.2f}h'."""
        assert _fmt_time(3600.0) == f"{3600.0 / 3600:.2f}h"
        assert _fmt_time(7200.0) == f"{7200.0 / 3600:.2f}h"
        assert _fmt_time(36000.0) == f"{36000.0 / 3600:.2f}h"

    def test_boundary_60_is_minutes(self):
        """Exact value 60 is formatted as minutes, not seconds."""
        result = _fmt_time(60.0)
        assert result.endswith("m"), f"Expected minutes suffix, got {result!r}"

    def test_boundary_3600_is_hours(self):
        """Exact value 3600 is formatted as hours, not minutes."""
        result = _fmt_time(3600.0)
        assert result.endswith("h"), f"Expected hours suffix, got {result!r}"


# ===========================================================================
# TestShouldPrintBatch
# ===========================================================================


class TestShouldPrintBatch:
    """Tests for tod_utils._should_print_batch."""

    def test_always_true_when_few_batches(self):
        """Returns True for all indices when n_batches <= max_prints."""
        for idx in range(10):
            assert _should_print_batch(idx, 10, max_prints=100) is True

    def test_always_true_for_first_batch(self):
        """Returns True for batch_idx=0 regardless of n_batches."""
        assert _should_print_batch(0, 10000, max_prints=100) is True

    def test_always_true_for_last_batch(self):
        """Returns True for batch_idx == n_batches - 1."""
        assert _should_print_batch(9999, 10000, max_prints=100) is True

    def test_true_at_step_multiples(self):
        """Returns True at multiples of step = n_batches // max_prints."""
        n_batches = 1000
        max_prints = 100
        step = n_batches // max_prints  # == 10
        assert _should_print_batch(step, n_batches, max_prints=max_prints) is True
        assert _should_print_batch(2 * step, n_batches, max_prints=max_prints) is True

    def test_false_at_non_step_non_boundary(self):
        """Returns False at indices that are not step multiples or boundaries."""
        n_batches = 1000
        max_prints = 100
        step = n_batches // max_prints  # == 10
        assert _should_print_batch(1, n_batches, max_prints=max_prints) is False
        assert _should_print_batch(step - 1, n_batches, max_prints=max_prints) is False


# ===========================================================================
# TestIsCluster
# ===========================================================================


class TestIsCluster:
    """Tests for tod_utils._is_cluster."""

    _HPC_VARS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "SGE_TASK_ID")

    def test_returns_false_when_no_hpc_vars(self):
        """Returns False when all four HPC environment variables are absent/empty."""
        env_patch = {v: "" for v in self._HPC_VARS}
        with patch.dict(os.environ, env_patch, clear=False):
            for v in self._HPC_VARS:
                os.environ.pop(v, None)
            assert _is_cluster() is False

    @pytest.mark.parametrize("var", _HPC_VARS)
    def test_returns_true_for_each_hpc_var(self, var):
        """Returns True when a single HPC environment variable is set to a non-empty value."""
        env_patch = {v: "" for v in self._HPC_VARS}
        env_patch[var] = "12345"
        with patch.dict(os.environ, env_patch, clear=False):
            for v in self._HPC_VARS:
                if v != var:
                    os.environ.pop(v, None)
            os.environ[var] = "12345"
            assert _is_cluster() is True


# ===========================================================================
# TestGetMemoryPerProcess
# ===========================================================================


class TestGetMemoryPerProcess:
    """Tests for tod_utils._get_memory_per_process."""

    _HPC_VARS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "SGE_TASK_ID")

    def _clear_hpc_env(self):
        return {v: "" for v in self._HPC_VARS}

    def test_local_path_no_hpc(self):
        """Local (non-cluster) path returns available_gb * 0.75 / n_processes."""
        available_gb = 16.0
        n_processes = 4

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.available = available_gb * 1e9

        env_patch = self._clear_hpc_env()
        with patch.dict(os.environ, env_patch, clear=False):
            for v in self._HPC_VARS:
                os.environ.pop(v, None)
            with patch.dict(sys.modules, {"psutil": mock_psutil}):
                import importlib
                import tod_utils as _tu

                importlib.reload(_tu)
                result = _tu._get_memory_per_process(n_processes)

        expected = available_gb * 0.75 / n_processes
        npt.assert_allclose(result, expected, rtol=1e-6)

    def test_cluster_path(self):
        """Cluster path (SLURM_JOB_ID set) returns available_gb * 1.0 / n_processes."""
        available_gb = 16.0
        n_processes = 4

        mock_psutil = MagicMock()
        mock_psutil.virtual_memory.return_value.available = available_gb * 1e9

        env_patch = self._clear_hpc_env()
        env_patch["SLURM_JOB_ID"] = "99999"
        with patch.dict(os.environ, env_patch, clear=False):
            os.environ["SLURM_JOB_ID"] = "99999"
            with patch.dict(sys.modules, {"psutil": mock_psutil}):
                import importlib
                import tod_utils as _tu

                importlib.reload(_tu)
                result = _tu._get_memory_per_process(n_processes)

        expected = available_gb * 1.0 / n_processes
        npt.assert_allclose(result, expected, rtol=1e-6)

    def test_fallback_when_psutil_unavailable(self):
        """Falls back to config.max_memory_per_process when psutil is unavailable."""
        n_processes = 4
        with patch.dict(sys.modules, {"psutil": None}):
            import importlib
            import tod_utils as _tu

            importlib.reload(_tu)
            result = _tu._get_memory_per_process(n_processes)

        assert result == config.max_memory_per_process


# ===========================================================================
# TestGetNcpus
# ===========================================================================


class TestGetNcpus:
    """Tests for tod_utils._get_ncpus."""

    _HPC_VARS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "SGE_TASK_ID")

    def _clear_hpc_env(self):
        return {v: "" for v in self._HPC_VARS}

    def test_psutil_path(self):
        """psutil path returns min(affinity_cores // hyperthreads, config.n_processes)."""
        logical_cpus = 16
        physical_cpus = 8
        affinity_set = set(range(12))

        mock_psutil = MagicMock()
        mock_psutil.cpu_count.side_effect = lambda logical: (
            logical_cpus if logical else physical_cpus
        )

        env_patch = self._clear_hpc_env()
        with patch.dict(os.environ, env_patch, clear=False):
            for v in self._HPC_VARS:
                os.environ.pop(v, None)
            with patch.dict(sys.modules, {"psutil": mock_psutil}):
                import importlib
                import tod_utils as _tu

                importlib.reload(_tu)
                with patch.object(
                    _tu.os, "sched_getaffinity", return_value=affinity_set, create=True
                ):
                    result = _tu._get_ncpus()

        nthreads_per_core = logical_cpus // physical_cpus  # 2
        ncores_available = len(affinity_set) // nthreads_per_core  # 6
        expected = min(ncores_available, config.n_processes)
        assert result == expected

    def test_slurm_path_no_psutil(self):
        """SLURM_CPUS_PER_TASK path (no psutil) returns the env var value as int."""
        slurm_cpus = 8
        env_patch = self._clear_hpc_env()
        env_patch["SLURM_CPUS_PER_TASK"] = str(slurm_cpus)

        with patch.dict(os.environ, env_patch, clear=False):
            os.environ["SLURM_CPUS_PER_TASK"] = str(slurm_cpus)
            for v in self._HPC_VARS:
                os.environ.pop(v, None)
            with patch.dict(sys.modules, {"psutil": None}):
                import importlib
                import tod_utils as _tu

                importlib.reload(_tu)
                result = _tu._get_ncpus()

        assert result == slurm_cpus

    def test_config_fallback(self):
        """Falls back to config.n_processes when psutil is absent and no SLURM var is set."""
        env_patch = self._clear_hpc_env()
        env_patch["SLURM_CPUS_PER_TASK"] = ""

        with patch.dict(os.environ, env_patch, clear=False):
            os.environ.pop("SLURM_CPUS_PER_TASK", None)
            for v in self._HPC_VARS:
                os.environ.pop(v, None)
            with patch.dict(sys.modules, {"psutil": None}):
                import importlib
                import tod_utils as _tu

                importlib.reload(_tu)
                result = _tu._get_ncpus()

        assert result == config.n_processes


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
