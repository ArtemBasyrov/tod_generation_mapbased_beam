"""
Empirical batch-size calibration for tod_exact_gen_batched.

Key design decisions
--------------------
* probe_n = mem_cap * _MIN_BATCHES_PER_PROBE  — every candidate runs the same
  total number of samples so small and large candidates are measured on equal
  footing.

* Interleaved repeats — one full pass over all candidates per repeat round,
  rather than n_repeats consecutive runs per candidate.  Consecutive runs of
  the same candidate warm the L3 cache for that candidate's working set and
  bias `min` heavily toward small batch sizes (small working set stays warm).
  Interleaving gives every candidate a comparable cache state.

* mean timing — `min` amplifies the warm-cache best case; `mean` over
  interleaved rounds is more representative of sustained throughput.
"""
import gc
import time
import numpy as np
import healpy as hp

from tod_io   import load_scan_data_batch
from tod_core import precompute_rotation_vector_batch, beam_tod_batch
from tod_utils import _fmt_time, get_memory_per_process

_BYTES_PER_SAMPLE_PER_BEAM = 100   # memory model ceiling constant
_MEMORY_SAFETY_FACTOR       = 1.5
_MIN_BATCHES_PER_PROBE      = 6   # each candidate must fill this many batches


def _memory_cap(max_memory_gb, max_beam_sel):
    budget = max_memory_gb * 1e9 / _MEMORY_SAFETY_FACTOR
    return max(1, int(budget // (_BYTES_PER_SAMPLE_PER_BEAM * max_beam_sel)))


def _candidate_batch_sizes(mem_cap):
    log_max    = int(np.log2(mem_cap)) + 1
    powers     = [int(2**k) for k in range(1, log_max + 1)]
    extras     = [mem_cap, mem_cap // 2, mem_cap // 4]
    candidates = sorted(set(powers + [c for c in extras if c >= 1]))
    return [c for c in candidates if c <= mem_cap]


def calibrate_batch_size(beam_data, folder_scan, probe_day, mp, n_processes,
                         n_repeats=3, prefix=""):
    """Find the batch size that maximises sustained throughput on this hardware.

    Runs once before the day loop. Generates candidate batch sizes from a power-
    of-two sequence capped by an empirical memory model, then measures sustained
    throughput using interleaved timing repeats (one full pass over all
    candidates per repeat round) to avoid L3 cache warm-up bias.

    Args:
        beam_data (dict): Loaded beam data from :func:`prepare_beam_data`.
        folder_scan (str): Path to the scan data directory.
        probe_day (int): Any valid day index; used to load the timing probe
            data.
        mp (list[numpy.ndarray]): Sky map components ``[I, Q, U]``.
        n_processes (int): Number of worker processes. Used to derive the
            per-process memory budget via :func:`~tod_utils.get_memory_per_process`.
        n_repeats (int): Number of interleaved timing rounds per candidate.
            Defaults to ``3``.
        prefix (str): Log-message prefix (e.g. process name). Defaults to ``""``.

    Returns:
        tuple:
            - **best_batch_size** (*int*) – Batch size with highest mean
              throughput.
            - **results** (*list[tuple[int, float]]*) – Per-candidate list of
              ``(batch_size, throughput_samples_per_second)``.
    """
    nside        = hp.get_nside(mp[0])
    max_beam_sel = max(d['n_sel'] for d in beam_data.values())
    first_bf     = next(iter(beam_data))
    ra0, dec0    = beam_data[first_bf]['ra'], beam_data[first_bf]['dec']

    max_memory_gb = get_memory_per_process(n_processes)
    mem_cap       = _memory_cap(max_memory_gb, max_beam_sel)
    candidates = _candidate_batch_sizes(mem_cap)

    # Every candidate must run at least _MIN_BATCHES_PER_PROBE full batches.
    # The largest candidate (mem_cap) is the binding constraint.
    probe_n = mem_cap * _MIN_BATCHES_PER_PROBE
    theta_p, phi_p, psi_p = load_scan_data_batch(folder_scan, probe_day, 0, probe_n)
    probe_n = min(probe_n, len(phi_p))

    print(prefix + f"[calibrate] probe_n={probe_n}, mem_cap={mem_cap}, candidates={candidates}")

    def _run_one(bs):
        n_batches = (probe_n + bs - 1) // bs
        t0 = time.perf_counter()
        for b in range(n_batches):
            s, e = b * bs, min((b + 1) * bs, probe_n)
            phi_b, theta_b, psi_b = phi_p[s:e], theta_p[s:e], psi_p[s:e]
            rot_vecs, betas = precompute_rotation_vector_batch(ra0, dec0, phi_b, theta_b)
            psis_b = -betas + psi_b
            for data in beam_data.values():
                beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b)
        return time.perf_counter() - t0

    _run_one(candidates[len(candidates) // 2])  # warm-up (Python/numpy JIT)
    gc.collect()

    # Interleaved measurement: one pass over all candidates per round.
    # This gives each candidate a comparable cache state instead of letting
    # consecutive runs of the same candidate keep its working set warm.
    times = {bs: [] for bs in candidates}
    for _ in range(n_repeats):
        for bs in candidates:
            times[bs].append(_run_one(bs))
        gc.collect()

    results = []
    for bs in candidates:
        mean_t     = np.mean(times[bs])
        throughput = probe_n / mean_t
        results.append((bs, throughput))
        print(prefix + f"[calibrate]   batch_size={bs:8d}  time={_fmt_time(mean_t)}  throughput={throughput:,.0f} samp/s")

    best_bs, best_tp = max(results, key=lambda x: x[1])
    print(prefix + f"[calibrate] -> optimal batch_size={best_bs}  ({best_tp:,.0f} samp/s)  [mem_cap={mem_cap}]")
    return best_bs, results


def calibrate_n_processes(beam_data, folder_scan, probe_day, mp, n_cpu_ceiling,
                           n_repeats=3, prefix=""):
    """Find the optimal number of worker processes for maximum total throughput.

    Estimates total throughput as ``throughput_per_process(n) × n`` and returns
    the ``n`` that maximises it. This correctly handles the HPC pattern where
    using all allocated cores gives each process too little RAM, leading to tiny
    batches with high Python overhead that make fewer-but-larger workers faster
    end-to-end.

    Strategy:

    1. Run :func:`calibrate_batch_size` once with the full available memory
       (as if ``n = 1``) to obtain a throughput curve over all candidate batch
       sizes.
    2. For each candidate ``n`` in ``[1, n_cpu_ceiling]``, find the largest
       batch size that fits in ``total_memory / n`` and read off its per-process
       throughput from the curve.
    3. Return the ``n`` with the highest ``n × per_process_throughput``.

    Args:
        beam_data (dict): Loaded beam data from :func:`prepare_beam_data`.
        folder_scan (str): Path to the scan data directory.
        probe_day (int): Any valid day index for the timing probe.
        mp (list[numpy.ndarray]): Sky map components ``[I, Q, U]``.
        n_cpu_ceiling (int): Maximum number of worker processes allowed
            (from the scheduler allocation or ``config.n_processes``).
        n_repeats (int): Timing rounds for the calibration run. Defaults to
            ``3``.
        prefix (str): Log-message prefix. Defaults to ``""``.

    Returns:
        tuple:
            - **n_optimal** (*int*) – Optimal number of worker processes.
            - **batch_size** (*int*) – Optimal batch size for that process
              count.
    """
    max_beam_sel = max(d['n_sel'] for d in beam_data.values())

    # Total usable memory: get_memory_per_process(1) = available × fraction / 1
    total_memory_gb = get_memory_per_process(1)

    # Run calibration with full memory to get throughput at all batch sizes.
    print(prefix + f"[n_proc] Calibrating throughput curve "
          f"(total_memory={total_memory_gb:.1f} GB, cpu_ceiling={n_cpu_ceiling})...")
    _, results = calibrate_batch_size(
        beam_data, folder_scan, probe_day, mp,
        n_processes=1,
        n_repeats=n_repeats,
        prefix=prefix,
    )
    # results: list of (batch_size, throughput_samp_per_s), ordered by batch_size
    results_dict = dict(results)
    sorted_bs    = sorted(results_dict)

    # For each candidate n compute estimated total throughput.
    print(prefix + f"[n_proc] Scanning n_processes 1 … {n_cpu_ceiling}:")
    best_n, best_total_tp, best_bs = 1, 0.0, sorted_bs[-1]

    for n in range(1, n_cpu_ceiling + 1):
        mem_per_proc = total_memory_gb / n
        cap          = _memory_cap(mem_per_proc, max_beam_sel)
        affordable   = [bs for bs in sorted_bs if bs <= cap]
        if not affordable:
            print(prefix + f"[n_proc]   n={n:3d}: mem/proc={mem_per_proc:.2f} GB  "
                  f"→ no affordable batch size, skipping")
            continue
        bs_n       = max(affordable, key=lambda bs: results_dict[bs])
        tp_per_proc = results_dict[bs_n]
        total_tp   = tp_per_proc * n
        marker     = "  ←" if total_tp > best_total_tp else ""
        print(prefix + f"[n_proc]   n={n:3d}: mem/proc={mem_per_proc:.2f} GB  "
              f"cap={cap}  batch_size={bs_n}  "
              f"tp/proc={tp_per_proc:,.0f}  total={total_tp:,.0f}{marker}")
        if total_tp > best_total_tp:
            best_total_tp = total_tp
            best_n        = n
            best_bs       = bs_n

    print(prefix + f"[n_proc] → n_optimal={best_n}  batch_size={best_bs}  "
          f"(est. total throughput={best_total_tp:,.0f} samp/s)")
    return best_n, best_bs
