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

from tod_io   import _load_scan_data_batch, open_scan_day
from tod_core import precompute_rotation_vector_batch, beam_tod_batch
from tod_utils import _fmt_time, _get_memory_per_process

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


def _calibrate_batch_size(beam_data, folder_scan, probe_day, mp, n_processes,
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
            per-process memory budget via :func:`~tod_utils._get_memory_per_process`.
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

    max_memory_gb = _get_memory_per_process(n_processes)
    mem_cap       = _memory_cap(max_memory_gb, max_beam_sel)
    candidates = _candidate_batch_sizes(mem_cap)

    # Every candidate must run at least _MIN_BATCHES_PER_PROBE full batches.
    # The largest candidate (mem_cap) is the binding constraint.
    probe_n = mem_cap * _MIN_BATCHES_PER_PROBE
    theta_p, phi_p, psi_p = _load_scan_data_batch(folder_scan, probe_day, 0, probe_n)
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


def _calibrate_n_processes(beam_data, folder_scan, probe_day, mp, n_cpu_ceiling,
                           n_repeats=3, prefix=""):
    """Find the optimal number of worker processes for maximum total throughput.

    Estimates total throughput as ``throughput_per_process(n) × n`` and returns
    the ``n`` that maximises it. This correctly handles the HPC pattern where
    using all allocated cores gives each process too little RAM, leading to tiny
    batches with high Python overhead that make fewer-but-larger workers faster
    end-to-end.

    Strategy:

    1. Run :func:`_calibrate_batch_size` once with the full available memory
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

    # Total usable memory: _get_memory_per_process(1) = available × fraction / 1
    total_memory_gb = _get_memory_per_process(1)

    # Run calibration with full memory to get throughput at all batch sizes.
    print(prefix + f"[n_proc] Calibrating throughput curve "
          f"(total_memory={total_memory_gb:.1f} GB, cpu_ceiling={n_cpu_ceiling})...")
    _, results = _calibrate_batch_size(
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


def _run_clustering_probe(nside, mp, beam_entries, rot_vecs, phi_b, theta_b, psis_b,
                           interp_mode, sigma_deg, radius_deg):
    """Run beam_tod_batch for all entries and accumulate into a (3, B) array."""
    B = len(phi_b)
    tod = np.zeros((3, B), dtype=np.float64)
    for data in beam_entries:
        contrib = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b,
                                 interp_mode=interp_mode,
                                 sigma_deg=sigma_deg,
                                 radius_deg=radius_deg)
        for comp, vals in contrib.items():
            tod[comp] += vals
    return tod


def calibrate_beam_clustering(
        beam_data,
        folder_scan,
        probe_day,
        mp,
        error_threshold      = 1e-3,
        interp_mode          = 'bilinear',
        interp_sigma_deg     = None,
        interp_radius_deg    = None,
):
    """Find the (tail_fraction, n_clusters) pair that maximises speedup while
    keeping the relative RMS TOD error below ``error_threshold``.

    Loads a short probe batch, computes an exact reference TOD, then sweeps
    a fixed (tail_fraction × n_clusters) grid.  For each pair, beam pixels are
    clustered on a copy of beam_data (original is not modified), a clustered
    TOD is computed, and the relative RMS error vs. the exact reference is
    measured.  The pair that maximises the average per-beam speedup subject to
    rel_rms <= error_threshold is returned.  If no pair qualifies, the
    minimum-error pair is returned with a warning.

    The probe uses the double-Rodrigues path (no vec_rolled / dtheta / dphi)
    so that only clustering error — not cache approximation error — is measured.

    Args:
        beam_data (dict): Exact (unclustered) beam data from prepare_beam_data.
            Not modified.
        folder_scan (str): Path to the scan data directory.
        probe_day (int): Day index used to load the probe scan samples.
        mp (list[numpy.ndarray]): Sky map components [I, Q, U].
        error_threshold (float): Maximum tolerated relative RMS TOD error.
        interp_mode (str): Beam interpolation mode passed to beam_tod_batch.
        interp_sigma_deg (float | None): Gaussian sigma; None → pixel size.
        interp_radius_deg (float | None): Gaussian radius; None → 3 × sigma.

    Returns:
        tuple[float, int]: (best_tail_fraction, best_n_clusters)
    """
    # Search grid — fixed internally, same philosophy as batch-size calibration.
    # Tail fractions: fine steps at the low end where accuracy matters most,
    # coarser at higher fractions where the beam fringe is already well-sampled.
    tail_fractions  = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    # Cluster counts: covers from very aggressive (10) to nearly-exact (2000).
    n_clusters_list = (10, 20, 50, 100, 200, 500, 1000, 2000)
    # Probe length: enough samples to average over varied scan angles.
    n_probe_samples = 1000
    from beam_cluster import cluster_beam_pixels

    nside   = hp.get_nside(mp[0])
    first_bf = next(iter(beam_data))
    ra0, dec0 = beam_data[first_bf]['ra'], beam_data[first_bf]['dec']

    # ── Load probe batch — strided across the full day ────────────────────────
    # Taking the first N consecutive samples would bias the error estimate
    # toward one sky patch and one scan angle.  Instead, open the memory-mapped
    # scan files and stride uniformly across the entire day so the probe covers
    # the full range of boresight directions and polarisation angles.
    theta_mmap, phi_mmap, psi_mmap = open_scan_day(folder_scan, probe_day)
    N_day  = len(phi_mmap)
    stride = max(1, N_day // n_probe_samples)
    idx    = slice(0, N_day, stride)
    phi_b   = np.array(phi_mmap[idx],   dtype=np.float32)[:n_probe_samples]
    theta_b = np.array(theta_mmap[idx], dtype=np.float32)[:n_probe_samples]
    psi_b   = np.array(psi_mmap[idx],   dtype=np.float32)[:n_probe_samples]
    B = len(phi_b)
    print(f"[clust_calib] Probe: {B} samples strided across {N_day} "
          f"(stride={stride}, day={probe_day})")

    rot_vecs, betas = precompute_rotation_vector_batch(ra0, dec0, phi_b, theta_b)
    psis_b = -betas + psi_b

    # ── Build stripped beam entries (exact, double-Rodrigues path) ────────────
    # Only beam_vals, vec_orig, n_sel, comp_indices, mp_stacked — no cache keys.
    def _make_entry(data):
        entry = {
            'beam_vals':   data['beam_vals'],
            'vec_orig':    data['vec_orig'],
            'n_sel':       data['n_sel'],
            'comp_indices': data['comp_indices'],
            'mp_stacked':  np.ascontiguousarray(
                               np.stack([mp[c] for c in data['comp_indices']])),
            'ra':          data['ra'],
            'dec':         data['dec'],
        }
        return entry

    exact_entries = [_make_entry(data) for data in beam_data.values()]

    # ── Warm-up: compile Numba JIT kernels ────────────────────────────────────
    print("[clust_calib] Warming up Numba kernels …")
    _run_clustering_probe(nside, mp, exact_entries, rot_vecs, phi_b, theta_b, psis_b,
                          interp_mode, interp_sigma_deg, interp_radius_deg)

    # ── Exact reference TOD ───────────────────────────────────────────────────
    print("[clust_calib] Computing exact reference TOD …")
    tod_exact = _run_clustering_probe(nside, mp, exact_entries, rot_vecs,
                                       phi_b, theta_b, psis_b,
                                       interp_mode, interp_sigma_deg, interp_radius_deg)
    exact_rms = float(np.sqrt(np.mean(tod_exact ** 2)))
    if exact_rms == 0.0:
        exact_rms = 1.0   # guard against all-zero map

    # Pre-compute S_bf (unclustered pixel counts) for speedup calculation
    S_bf = {bf: data['n_sel'] for bf, data in beam_data.items()}

    # ── Grid sweep ────────────────────────────────────────────────────────────
    print("[clust_calib] Sweeping clustering parameters …")
    results = []   # (tf, K_req, K_out, speedup, rel_rms)

    for tf in tail_fractions:
        for K_req in n_clusters_list:
            # Build clustered entries using copies of beam_vals / vec_orig
            clust_entries = []
            K_out_per_bf  = {}
            for bf, data in beam_data.items():
                bv_copy  = data['beam_vals'].copy()
                vo_copy  = data['vec_orig'].copy()
                vec_c, bv_c, _ = cluster_beam_pixels(
                    vo_copy, bv_copy,
                    n_clusters=K_req,
                    tail_fraction=tf,
                    verbose=False,
                )
                K_out = len(bv_c)
                K_out_per_bf[bf] = K_out

                entry = {
                    'beam_vals':   bv_c,
                    'vec_orig':    vec_c,
                    'n_sel':       K_out,
                    'comp_indices': data['comp_indices'],
                    'mp_stacked':  np.ascontiguousarray(
                                       np.stack([mp[c] for c in data['comp_indices']])),
                    'ra':          data['ra'],
                    'dec':         data['dec'],
                }
                clust_entries.append(entry)

            tod_clust = _run_clustering_probe(
                nside, mp, clust_entries, rot_vecs, phi_b, theta_b, psis_b,
                interp_mode, interp_sigma_deg, interp_radius_deg,
            )

            diff    = tod_exact - tod_clust
            rel_rms = float(np.sqrt(np.mean(diff ** 2))) / exact_rms

            # Average speedup over all beams
            speedup = float(np.mean([S_bf[bf] / K_out_per_bf[bf]
                                     for bf in beam_data]))
            K_out_repr = int(np.mean(list(K_out_per_bf.values())))

            results.append((tf, K_req, K_out_repr, speedup, rel_rms))

    # ── Print ASCII table ─────────────────────────────────────────────────────
    print()
    print(f"[clust_calib] error_threshold={error_threshold:.1e}")
    print(f"{'tail%':>6s}  {'K':>5s}  {'K_out':>6s}  {'speedup':>8s}  {'rel.RMS':>9s}  {'status'}")
    print("-" * 52)
    prev_tf = None
    for tf, K_req, K_out, speedup, rel_rms in results:
        if prev_tf is not None and tf != prev_tf:
            print("-" * 52)
        status = "✓" if rel_rms <= error_threshold else "✗"
        print(f"{tf*100:>5.1f}%  {K_req:>5d}  {K_out:>6d}  {speedup:>8.2f}x  "
              f"{rel_rms:>9.2e}  {status}")
        prev_tf = tf
    print("-" * 52)

    # ── Select best ───────────────────────────────────────────────────────────
    passing = [(tf, K_req, K_out, speedup, rel_rms)
               for tf, K_req, K_out, speedup, rel_rms in results
               if rel_rms <= error_threshold]

    if passing:
        best = max(passing, key=lambda x: x[3])   # max speedup
        best_tf, best_K_req = best[0], best[1]
        print(f"\n[clust_calib] Recommendation: tail_fraction={best_tf}, "
              f"n_clusters={best_K_req}  "
              f"(speedup={best[3]:.2f}x, rel_rms={best[4]:.2e})")
    else:
        # Nothing qualifies — return minimum-error pair
        best = min(results, key=lambda x: x[4])
        best_tf, best_K_req = best[0], best[1]
        print(f"\n[clust_calib] WARNING: no (tf, K) pair achieved rel_rms <= "
              f"{error_threshold:.1e}.")
        print(f"[clust_calib] Returning minimum-error pair: "
              f"tail_fraction={best_tf}, n_clusters={best_K_req}  "
              f"(rel_rms={best[4]:.2e})")

    return best_tf, best_K_req
