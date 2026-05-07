"""
CPU / process count, memory detection, and progress utilities for HPC and
local environments.

Falls back to ``n_processes`` from config if SLURM is unavailable. Requires
psutil.
"""

import os
import tod_config as config


# Fraction of available RAM reserved for the OS and other user processes
# when running locally. On a cluster the job owns the node, so we use 1.0.
_LOCAL_RAM_FRACTION = 0.75

# HPC scheduler env vars whose presence indicates a batch/cluster job.
_CLUSTER_ENV_VARS = (
    "SLURM_JOB_ID",
    "PBS_JOBID",
    "LSB_JOBID",
    "SGE_TASK_ID",
    "SLURM_CPUS_PER_TASK",
)


def _cpu_ceiling():
    """
    Return the number of physical cores available to this process.

    Priority
    --------
    1. SLURM_CPUS_PER_TASK (scheduler knows exactly what was allocated)
    2. psutil affinity-aware core count (respects taskset / cgroup limits)
    3. os.cpu_count() (total logical CPUs — last resort)
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm is not None:
        try:
            return int(slurm)
        except ValueError:
            pass

    try:
        import psutil

        nthreads_per_core = psutil.cpu_count(logical=True) // psutil.cpu_count(
            logical=False
        )
        return len(os.sched_getaffinity(0)) // nthreads_per_core
    except Exception:
        pass

    return os.cpu_count() or config.n_processes


def _is_cluster():
    """Return True when running inside a recognised HPC batch scheduler."""
    return any(os.environ.get(v) for v in _CLUSTER_ENV_VARS)


def _get_ncpus():
    """Return the CPU ceiling for this job.

    On a cluster (SLURM / PBS / LSF / SGE detected) this is the number of
    cores allocated by the scheduler. On a local workstation the result is
    capped at ``config.n_processes`` to keep the machine responsive.

    The actual number of worker processes to launch is determined later by
    :func:`~tod_calibrate.calibrate_runtime`, which balances CPU count
    against per-process memory to maximise total throughput.

    Returns:
        int: Maximum number of CPUs available to this job.
    """
    cluster = _is_cluster()
    n_cpu = _cpu_ceiling()

    if cluster:
        print(
            f"[cpu] Cluster — {n_cpu} CPUs available (scheduler/psutil); "
            f"optimal worker count determined by calibration"
        )
    else:
        n_cpu = min(n_cpu, config.n_processes)
        print(
            f"[cpu] Local — {n_cpu} CPUs (capped at config.n_processes={config.n_processes})"
        )

    return n_cpu


def _get_memory_per_process(n_processes):
    """Determine the per-process memory budget in GB.

    On a cluster (SLURM / PBS / LSF / SGE detected) the job owns the node and
    the full available RAM is divided among the worker processes. Locally only
    ``_LOCAL_RAM_FRACTION`` (75 %) of available RAM is used so the rest of the
    system stays responsive.

    Args:
        n_processes (int): Number of worker processes to budget memory for.

    Returns:
        float: Per-process memory budget in GB.
    """
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / 1e9
        cluster = _is_cluster()
        fraction = 1.0 if cluster else _LOCAL_RAM_FRACTION
        env_label = "cluster" if cluster else f"local (×{fraction})"
        memory_gb = available_gb * fraction / n_processes
        print(
            f"[mem] {available_gb:.1f} GB available  fraction={fraction}  "
            f"{n_processes} processes  → {memory_gb:.2f} GB/process  ({env_label})"
        )
        return memory_gb
    except Exception as exc:
        raise RuntimeError("psutil is required to determine available memory") from exc


def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.2f}m"
    else:
        return f"{seconds / 3600:.2f}h"


def _should_print_batch(batch_idx, n_batches, max_prints=100):
    """
    Returns True if this batch index should print a progress message.
    Always prints first and last batch. For everything in between,
    prints at most max_prints evenly-spaced updates.
    """
    if n_batches <= max_prints:
        return True
    if batch_idx == 0 or batch_idx == n_batches - 1:
        return True
    step = n_batches // max_prints
    return batch_idx % step == 0
