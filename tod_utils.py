"""
CPU / process count and memory detection for HPC and local environments.

Falls back to `n_processes` / `max_memory_per_process` from config if SLURM
or psutil are unavailable.
"""
import os
import multiprocessing
import tod_config as config
import numpy as np


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
        nthreads_per_core = (psutil.cpu_count(logical=True)
                             // psutil.cpu_count(logical=False))
        return len(os.sched_getaffinity(0)) // nthreads_per_core
    except Exception:
        pass

    return os.cpu_count() or config.n_processes


def _get_ncpus():
    """Return the CPU ceiling for this job.

    On a cluster (SLURM / PBS / LSF / SGE detected) this is the number of
    cores allocated by the scheduler. On a local workstation the result is
    capped at ``config.n_processes`` to keep the machine responsive.

    The actual number of worker processes to launch is determined later by
    :func:`~tod_calibrate._calibrate_n_processes`, which balances CPU count
    against per-process memory to maximise total throughput.

    Returns:
        int: Maximum number of CPUs available to this job.
    """
    cluster = _is_cluster()
    n_cpu   = _cpu_ceiling()

    if cluster:
        print(f"[cpu] Cluster — {n_cpu} CPUs available (scheduler/psutil); "
              f"optimal worker count determined by calibration")
    else:
        n_cpu = min(n_cpu, config.n_processes)
        print(f"[cpu] Local — {n_cpu} CPUs (capped at config.n_processes={config.n_processes})")

    return n_cpu


# Fraction of available RAM reserved for the OS and other user processes
# when running locally. On a cluster the job owns the node, so we use 1.0.
_LOCAL_RAM_FRACTION = 0.75

# HPC scheduler env vars whose presence indicates a batch/cluster job.
_CLUSTER_ENV_VARS = ("SLURM_JOB_ID", "PBS_JOBID", "LSB_JOBID", "SGE_TASK_ID",
                     "SLURM_CPUS_PER_TASK")


def _is_cluster():
    """Return True when running inside a recognised HPC batch scheduler."""
    return any(os.environ.get(v) for v in _CLUSTER_ENV_VARS)


def _get_memory_per_process(n_processes):
    """Determine the per-process memory budget in GB.

    On a cluster (SLURM / PBS / LSF / SGE detected) the job owns the node and
    the full available RAM is divided among the worker processes. Locally only
    ``_LOCAL_RAM_FRACTION`` (75 %) of available RAM is used so the rest of the
    system stays responsive.

    Detection priority:

    1. ``psutil`` available memory × fraction ÷ ``n_processes`` (auto-detected).
    2. ``config.max_memory_per_process`` (explicit fallback when ``psutil`` is
       unavailable).

    Args:
        n_processes (int): Number of worker processes to budget memory for.

    Returns:
        float: Per-process memory budget in GB.
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


def compute_bell(ra, dec, pixel_map,
                 lmax=1000,
                 power_cut=0.99,
                 normalise=True,
                 verbose=True):
    """Compute the effective beam transfer function B_ell from a pixelised beam.

    Evaluates

        B_ell = sum_i [ w_i * P_ell(cos θ_i) ]

    where ``w_i`` are the normalised beam weights, ``θ_i`` is the angular
    distance of pixel *i* from the beam centre, and ``P_ell`` are Legendre
    polynomials computed via the three-term recurrence (O(N) memory).

    The beam centre is taken to be the point with zero offset, i.e.
    ``ra_offset = 0, dec_offset = 0``, and the angular distance to each
    pixel is

        cos θ_i = cos(dec_i) * cos(ra_i)

    which is the standard haversine approximation for small-angle offsets.

    Args:
        ra (array-like): RA offsets from beam centre [rad].  Can be any
            shape; flattened internally.
        dec (array-like): Dec offsets from beam centre [rad].  Same shape
            as ``ra``.
        pixel_map (array-like): Beam power values (linear, **not** dB).
            Same shape as ``ra`` and ``dec``.
        lmax (int): Maximum multipole ℓ (inclusive).
        power_cut (float): Fraction of total power for hard pixel selection.
            ``1.0`` selects all pixels (fast path, no sorting needed).
        normalise (bool): If ``True`` (default) normalise so that B_0 = 1.
        verbose (bool): Print selection and pixel-count diagnostics.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - **ell**  – integer array ``[0, 1, …, lmax]``
            - **bell** – float64 array of B_ell values, length ``lmax + 1``
    """
    pixel_map = np.asarray(pixel_map, dtype=np.float64)
    ra        = np.asarray(ra,  dtype=np.float64).ravel()
    dec       = np.asarray(dec, dtype=np.float64).ravel()
    flat      = pixel_map.ravel()

    # ── 1. Pixel selection ────────────────────────────────────────────────────
    if power_cut >= 1.0:
        # Fast path: include all pixels, skip the O(N log N) threshold sort.
        sel = np.ones(flat.shape, dtype=bool)
        if verbose:
            print(f"  power_cut=1.0: selecting all {len(flat)} pixels")
    else:
        dB_cut  = _compute_dB_threshold_from_power(flat, power_cut)
        log_map = 10.0 * np.log10(np.abs(flat) + 1e-30)
        sel     = (log_map > dB_cut)
        if verbose:
            print(f"  power_cut={power_cut}: "
                  f"{np.sum(sel)}/{len(flat)} pixels selected "
                  f"(dB_cut={dB_cut:.2f})")

    if not np.any(sel):
        raise ValueError("No pixels survive the power-cut selection. "
                         "Check that pixel_map is in linear (not dB) units.")

    beam_vals = flat[sel]
    ra_sel    = ra[sel]
    dec_sel   = dec[sel]

    # ── 2. Normalise ──────────────────────────────────────────────────────────
    norm = beam_vals.sum()
    if norm <= 0:
        raise ValueError("Sum of beam values is non-positive after selection.")
    beam_vals = beam_vals / norm

    # ── 3. Angular distance cos θ from beam centre ────────────────────────────
    cos_theta = np.cos(dec_sel) * np.cos(ra_sel)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # ── 4. Legendre recurrence ────────────────────────────────────────────────
    N    = len(cos_theta)
    bell = np.empty(lmax + 1, dtype=np.float64)

    P_prev2 = np.ones(N,  dtype=np.float64)   # P_0 = 1
    P_prev1 = cos_theta.copy()                 # P_1 = x

    bell[0] = np.dot(beam_vals, P_prev2)
    if lmax >= 1:
        bell[1] = np.dot(beam_vals, P_prev1)

    for ell_idx in range(1, lmax):
        l       = float(ell_idx)
        P_curr  = ((2.0*l + 1.0) * cos_theta * P_prev1 - l * P_prev2) / (l + 1.0)
        bell[ell_idx + 1] = np.dot(beam_vals, P_curr)
        P_prev2 = P_prev1
        P_prev1 = P_curr

    # ── 5. Normalise B_0 = 1 ─────────────────────────────────────────────────
    if normalise and bell[0] != 0.0:
        bell /= bell[0]

    ell = np.arange(lmax + 1, dtype=np.int64)
    return ell, np.abs(bell)


def _compute_dB_threshold_from_power(beam_vals, power_cut):
    """Compute the dB threshold that retains a given fraction of total beam power.

    Finds the dB level such that the sum of all pixel amplitudes whose dB value
    exceeds that level equals ``power_cut × total_power``. Used to select the
    smallest set of beam pixels that accounts for the requested power fraction.

    Args:
        beam_vals (numpy.ndarray): Beam pixel amplitude values (linear, not dB).
            Can be any shape; flattened internally.
        power_cut (float): Fraction of total power to retain, e.g. ``0.99`` to
            keep 99 % of the beam power.

    Returns:
        float: dB threshold. Pixels whose ``10 log10(|val|)`` exceeds this
            value collectively contribute ``≈ power_cut × total_power``.
    """
    # Flatten and ensure array
    prof = np.asarray(beam_vals).flatten()
    
    # Calculate target power
    target_power = np.sum(prof) * power_cut
    
    # Convert to dB
    prof_dB = 10 * np.log10(prof)
    
    # Sort by dB values and get corresponding linear values
    sort_idx = np.argsort(prof_dB)
    sorted_dB = prof_dB[sort_idx]
    sorted_prof = prof[sort_idx]
    
    # Calculate cumulative sum from highest to lowest (reverse order)
    # This gives the sum of all pixels with dB >= threshold
    cumulative_sums = np.cumsum(sorted_prof[::-1])[::-1]
    
    # Find index where cumulative sum is closest to target
    idx = np.argmin(np.abs(cumulative_sums - target_power))
    
    # Return the threshold value
    return sorted_dB[idx]