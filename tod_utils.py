"""
CPU / process count and memory detection for HPC and local environments.

Falls back to `n_processes` / `max_memory_per_process` from config if SLURM
or psutil are unavailable.
"""
import os
import multiprocessing
import tod_config as config


def get_ncpus():
    """
    Determine the number of worker processes to use.

    Priority
    --------
    1. psutil affinity-aware core count (local multi-core)
    2. SLURM_CPUS_PER_TASK environment variable (HPC cluster)
    3. config.n_processes (explicit fallback)

    Returns
    -------
    ncpus : int
    """

    # 1. psutil (affinity-aware — respects taskset / cgroup limits)
    try:
        import psutil
        nthreads_per_core = (psutil.cpu_count(logical=True)
                             // psutil.cpu_count(logical=False))
        ncores_available  = len(os.sched_getaffinity(0)) // nthreads_per_core
        ncpus = min(ncores_available, config.n_processes)
        print(f"[cpu] Using {ncpus}/{ncores_available} available cores (psutil)")
        return ncpus
    except Exception:
        pass

    # 2. SLURM
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm is not None:
        try:
            ncpus = int(slurm)
            print(f"[cpu] Using {ncpus} CPUs from SLURM_CPUS_PER_TASK")
            return ncpus
        except ValueError:
            pass

    # 3. Config fallback
    ncpus = config.n_processes
    print(f"[cpu] Using {ncpus} CPUs from config (fallback)")
    return ncpus


# Fraction of available RAM reserved for the OS and other user processes
# when running locally. On a cluster the job owns the node, so we use 1.0.
_LOCAL_RAM_FRACTION = 0.75

# HPC scheduler env vars whose presence indicates a batch/cluster job.
_CLUSTER_ENV_VARS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "SGE_TASK_ID")


def _is_cluster():
    """Return True when running inside a recognised HPC batch scheduler."""
    return any(os.environ.get(v) for v in _CLUSTER_ENV_VARS)


def get_memory_per_process(n_processes):
    """
    Determine the per-process memory budget in GB.

    On a cluster (SLURM / PBS / LSF / SGE detected) the full available RAM is
    shared among the worker processes — the job owns the node.
    Locally, only _LOCAL_RAM_FRACTION of available RAM is used so the rest of
    the system stays responsive.

    Priority
    --------
    1. psutil available memory * fraction / n_processes (auto-detected)
    2. config.max_memory_per_process (explicit fallback)

    Returns
    -------
    memory_gb : float
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / 1e9
        cluster      = _is_cluster()
        fraction     = 1.0 if cluster else _LOCAL_RAM_FRACTION
        env_label    = "cluster" if cluster else f"local (×{fraction})"
        memory_gb    = available_gb * fraction / n_processes
        print(f"[mem] {available_gb:.1f} GB available  fraction={fraction}  "
              f"{n_processes} processes  → {memory_gb:.2f} GB/process  ({env_label})")
        return memory_gb
    except Exception:
        pass

    memory_gb = config.max_memory_per_process
    print(f"[mem] Using {memory_gb} GB/process from config (fallback)")
    return memory_gb


def _fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"
    

def should_print_batch(batch_idx, n_batches, max_prints=100):
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