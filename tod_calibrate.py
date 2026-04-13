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

from tod_io import _load_scan_data_batch, open_scan_day
from tod_core import precompute_rotation_vector_batch, beam_tod_batch
from tod_utils import _fmt_time, _get_memory_per_process, compute_bell

# Per-method memory ceiling (bytes per sample per selected beam pixel).
# Accounts for: rotation vector buffer (B×S×3×float32 = 12 B/sample/pixel),
# interpolation neighbor indices+weights, and tod accumulation buffers.
# Empirically measured; bicubic/gaussian query ~40 neighbors vs 4 for bilinear.
_BYTES_PER_SAMPLE_PER_BEAM = {
    "nearest": 25,  # 1 neighbor; dominated by rotation buffer
    "bilinear": 64,  # 4 neighbors (4×index + 4×weight + rotation buffer)
    "bicubic": 400,  # ~40 gnomonic neighbors
    "gaussian": 400,  # disc query, comparable to bicubic
}
_BYTES_PER_SAMPLE_PER_BEAM_DEFAULT = 100  # fallback for unknown methods
_MEMORY_SAFETY_FACTOR = 1.5
_MIN_BATCHES_PER_PROBE = 6  # each candidate must fill this many batches
_MAX_PROBE_SAMPLES = 100_000  # cap probe at 100K samples for fast calibration


def _memory_cap(max_memory_gb, max_beam_sel, interp_mode="bilinear"):
    bpspb = _BYTES_PER_SAMPLE_PER_BEAM.get(
        interp_mode, _BYTES_PER_SAMPLE_PER_BEAM_DEFAULT
    )
    budget = max_memory_gb * 1e9 / _MEMORY_SAFETY_FACTOR
    return max(1, int(budget // (bpspb * max_beam_sel)))


def _candidate_batch_sizes(mem_cap, max_memory_per_process_gb=None):
    """Generate candidate batch sizes, with adaptive minimum based on system size.

    For small systems (< 10 GB/proc, e.g., laptops), the optimal batch size is
    often 256–512. For large systems (> 100 GB/proc, e.g., HPC clusters),
    it's almost always 4K+. This scales the floor accordingly.

    Args:
        mem_cap (int): Maximum batch size (samples) that fit in memory.
        max_memory_per_process_gb (float | None): Available memory per process in GB.
            If None, no filtering applied (backward compatible with tests).
            If provided, adaptive minimum scales with system size.

    Returns:
        list[int]: Sorted candidate batch sizes.
    """
    # Adaptive minimum: scale with available memory per process
    # Heuristic: min_bs ≈ 128 + (memory_gb * 64)
    # - 1.5 GB/proc  → ~128–192
    # - 10 GB/proc   → ~1.25K
    # - 100+ GB/proc → ~12.5K+
    # When None (tests/legacy code), use no filtering for backward compatibility
    if max_memory_per_process_gb is None:
        min_bs = 1  # backward compatible: include all powers of 2
    else:
        min_bs = max(256, int(128 + max_memory_per_process_gb * 64))

    log_max = int(np.log2(mem_cap)) + 1
    powers = [int(2**k) for k in range(1, log_max + 1)]
    extras = [mem_cap, mem_cap // 2, mem_cap // 4]
    candidates = sorted(set(powers + [c for c in extras if c >= 1]))
    return [c for c in candidates if min_bs <= c <= mem_cap]


def _calibrate_batch_size(
    beam_data,
    folder_scan,
    probe_day,
    mp,
    n_processes,
    n_repeats=3,
    prefix="",
    interp_mode="bilinear",
):
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
    nside = hp.get_nside(mp[0])
    max_beam_sel = max(d["n_sel"] for d in beam_data.values())
    first_bf = next(iter(beam_data))
    ra0, dec0 = beam_data[first_bf]["ra"], beam_data[first_bf]["dec"]

    max_memory_gb = _get_memory_per_process(n_processes)
    mem_cap = _memory_cap(max_memory_gb, max_beam_sel, interp_mode=interp_mode)
    candidates = _candidate_batch_sizes(
        mem_cap, max_memory_per_process_gb=max_memory_gb
    )

    # Every candidate must run at least _MIN_BATCHES_PER_PROBE full batches.
    # The largest candidate (mem_cap) is the binding constraint.
    # Cap at _MAX_PROBE_SAMPLES to keep calibration fast (~5 min even on small systems).
    probe_n = min(mem_cap * _MIN_BATCHES_PER_PROBE, _MAX_PROBE_SAMPLES)
    theta_p, phi_p, psi_p = _load_scan_data_batch(folder_scan, probe_day, 0, probe_n)
    probe_n = min(probe_n, len(phi_p))

    print(
        prefix
        + f"[calibrate] probe_n={probe_n}, mem_cap={mem_cap}, candidates={candidates}"
    )

    def _run_one(bs):
        n_batches = (probe_n + bs - 1) // bs
        t0 = time.perf_counter()
        for b in range(n_batches):
            s, e = b * bs, min((b + 1) * bs, probe_n)
            phi_b, theta_b, psi_b = phi_p[s:e], theta_p[s:e], psi_p[s:e]
            rot_vecs, betas = precompute_rotation_vector_batch(
                ra0, dec0, phi_b, theta_b
            )
            psis_b = -betas + psi_b
            for data in beam_data.values():
                beam_tod_batch(
                    nside,
                    mp,
                    data,
                    rot_vecs,
                    phi_b,
                    theta_b,
                    psis_b,
                    interp_mode=interp_mode,
                    spin2_corr=0,
                )
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
        mean_t = np.mean(times[bs])
        throughput = probe_n / mean_t
        results.append((bs, throughput))
        print(
            prefix
            + f"[calibrate]   batch_size={bs:8d}  time={_fmt_time(mean_t)}  throughput={throughput:,.0f} samp/s"
        )

    best_bs, best_tp = max(results, key=lambda x: x[1])
    print(
        prefix
        + f"[calibrate] -> optimal batch_size={best_bs}  ({best_tp:,.0f} samp/s)  [mem_cap={mem_cap}]"
    )
    return best_bs, results


def _calibrate_n_processes(
    beam_data,
    folder_scan,
    probe_day,
    mp,
    n_cpu_ceiling,
    n_repeats=3,
    prefix="",
    interp_mode="bilinear",
):
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
    max_beam_sel = max(d["n_sel"] for d in beam_data.values())

    # Total usable memory: _get_memory_per_process(1) = available × fraction / 1
    total_memory_gb = _get_memory_per_process(1)

    # Run calibration with full memory to get throughput at all batch sizes.
    print(
        prefix + f"[n_proc] Calibrating throughput curve "
        f"(total_memory={total_memory_gb:.1f} GB, cpu_ceiling={n_cpu_ceiling})..."
    )
    _, results = _calibrate_batch_size(
        beam_data,
        folder_scan,
        probe_day,
        mp,
        n_processes=1,
        n_repeats=n_repeats,
        prefix=prefix,
        interp_mode=interp_mode,
    )
    # results: list of (batch_size, throughput_samp_per_s), ordered by batch_size
    results_dict = dict(results)
    sorted_bs = sorted(results_dict)

    # For each candidate n compute estimated total throughput.
    print(prefix + f"[n_proc] Scanning n_processes 1 … {n_cpu_ceiling}:")
    best_n, best_total_tp, best_bs = 1, 0.0, sorted_bs[-1]

    for n in range(1, n_cpu_ceiling + 1):
        mem_per_proc = total_memory_gb / n
        cap = _memory_cap(mem_per_proc, max_beam_sel, interp_mode=interp_mode)
        affordable = [bs for bs in sorted_bs if bs <= cap]
        if not affordable:
            print(
                prefix + f"[n_proc]   n={n:3d}: mem/proc={mem_per_proc:.2f} GB  "
                f"→ no affordable batch size, skipping"
            )
            continue
        bs_n = max(affordable, key=lambda bs: results_dict[bs])
        tp_per_proc = results_dict[bs_n]
        total_tp = tp_per_proc * n
        marker = "  ←" if total_tp > best_total_tp else ""
        print(
            prefix + f"[n_proc]   n={n:3d}: mem/proc={mem_per_proc:.2f} GB  "
            f"cap={cap}  batch_size={bs_n}  "
            f"tp/proc={tp_per_proc:,.0f}  total={total_tp:,.0f}{marker}"
        )
        if total_tp > best_total_tp:
            best_total_tp = total_tp
            best_n = n
            best_bs = bs_n

    print(
        prefix + f"[n_proc] → n_optimal={best_n}  batch_size={best_bs}  "
        f"(est. total throughput={best_total_tp:,.0f} samp/s)"
    )
    return best_n, best_bs


def _run_clustering_probe(
    nside,
    mp,
    beam_entries,
    rot_vecs,
    phi_b,
    theta_b,
    psis_b,
    interp_mode,
):
    """Run beam_tod_batch for all entries and accumulate into a (3, B) array."""
    B = len(phi_b)
    tod = np.zeros((3, B), dtype=np.float64)
    for data in beam_entries:
        contrib = beam_tod_batch(
            nside,
            mp,
            data,
            rot_vecs,
            phi_b,
            theta_b,
            psis_b,
            interp_mode=interp_mode,
            spin2_corr=0,
        )
        for comp, vals in contrib.items():
            tod[comp] += vals
    return tod


def calibrate_beam_clustering(
    beam_data,
    folder_scan=None,
    probe_day=None,
    mp=None,
    error_threshold=1e-3,
    bell_lmax=None,
    interp_mode="bilinear",
    interp_sigma_deg=None,
    interp_radius_deg=None,
):
    """Find the (tail_fraction, n_clusters) pair that maximises speedup while
    keeping the B_ell divergence from the analytical beam below
    ``error_threshold``.

    Computes the reference B_ell (power_cut=1.0) from the unclustered beam
    geometry, then sweeps a fixed (tail_fraction × n_clusters) grid.  For
    each pair, beam pixels are clustered on a copy of beam_data (original is
    not modified), B_ell is recomputed and compared to the reference.  The
    pair that maximises speedup subject to relative-RMS B_ell divergence <=
    ``error_threshold`` is returned.  If no pair qualifies, the minimum-
    divergence pair is returned with a warning.

    This approach is significantly faster than TOD-based calibration because
    it requires only beam geometry—no scan data loading and no Numba kernel
    calls per grid point.  The B_ell divergence is also a more direct quality
    metric: clustering that faithfully reproduces B_ell at full power will
    also reproduce the TOD accurately.

    Args:
        beam_data (dict): Exact (unclustered) beam data from prepare_beam_data.
            Not modified.
        folder_scan (str | None): Unused. Kept for backward compatibility.
        probe_day (int | None): Unused. Kept for backward compatibility.
        mp (list | None): Unused. Kept for backward compatibility.
        error_threshold (float): Maximum tolerated relative RMS B_ell
            divergence between clustered and reference beam.
        bell_lmax (int | None): Maximum multipole for B_ell comparison.
            When ``None`` (default) it is set automatically to
            ``2 × nside`` of the sky map ``mp``.  If ``mp`` is also
            ``None``, falls back to 500.
        interp_mode (str): Unused. Kept for backward compatibility.
        interp_sigma_deg (float | None): Unused. Kept for backward compatibility.
        interp_radius_deg (float | None): Unused. Kept for backward compatibility.

    Returns:
        tuple[float, int]: (best_tail_fraction, best_n_clusters)
    """
    # Search grid — same as before.
    tail_fractions = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    n_clusters_list = (10, 20, 50, 100, 200, 500, 1000, 2000)

    from beam_cluster import cluster_beam_pixels

    # ── Auto-determine bell_lmax from map nside ───────────────────────────────
    if bell_lmax is None:
        if mp is not None:
            bell_lmax = 2 * hp.get_nside(mp[0])
        else:
            bell_lmax = 500

    # ── Helper: B_ell from vec_orig + beam_vals (power_cut=1.0) ──────────────
    def _bell_from_vecs(vec, bvals):
        theta_pix, phi_pix = hp.vec2ang(vec)
        dec_offset = np.pi / 2.0 - theta_pix
        _, bell = compute_bell(
            phi_pix,
            dec_offset,
            bvals.astype(np.float64),
            lmax=bell_lmax,
            power_cut=1.0,
            verbose=False,
        )
        return bell

    # ── Reference B_ell from unclustered beam ────────────────────────────────
    print(
        f"[clust_calib] Computing reference B_ell (power_cut=1.0, lmax={bell_lmax}) …"
    )
    ref_bells = {}
    for bf, data in beam_data.items():
        ref_bells[bf] = _bell_from_vecs(data["vec_orig"], data["beam_vals"])

    S_bf = {bf: data["n_sel"] for bf, data in beam_data.items()}

    # ── Grid sweep using B_ell divergence ────────────────────────────────────
    print("[clust_calib] Sweeping clustering parameters …")
    results = []  # (tf, K_req, K_out_repr, speedup, bell_div)

    for tf in tail_fractions:
        for K_req in n_clusters_list:
            K_out_per_bf = {}
            bell_divs = []

            for bf, data in beam_data.items():
                bv_copy = data["beam_vals"].copy()
                vo_copy = data["vec_orig"].copy()
                vec_c, bv_c, _ = cluster_beam_pixels(
                    vo_copy,
                    bv_copy,
                    n_clusters=K_req,
                    tail_fraction=tf,
                    verbose=False,
                )
                K_out = len(bv_c)
                K_out_per_bf[bf] = K_out

                bell_clust = _bell_from_vecs(vec_c, bv_c)
                bell_ref = ref_bells[bf]
                ref_rms = float(np.sqrt(np.mean(bell_ref**2)))
                bell_div = float(np.sqrt(np.mean((bell_clust - bell_ref) ** 2))) / (
                    ref_rms + 1e-30
                )
                bell_divs.append(bell_div)

            mean_bell_div = float(np.mean(bell_divs))
            speedup = float(np.mean([S_bf[bf] / K_out_per_bf[bf] for bf in beam_data]))
            K_out_repr = int(np.mean(list(K_out_per_bf.values())))
            results.append((tf, K_req, K_out_repr, speedup, mean_bell_div))

    # ── Print ASCII table ─────────────────────────────────────────────────────
    print()
    print(f"[clust_calib] error_threshold={error_threshold:.1e}")
    print(
        f"{'tail%':>6s}  {'K':>5s}  {'K_out':>6s}  {'speedup':>8s}  "
        f"{'B_ell div':>10s}  {'status'}"
    )
    print("-" * 56)
    prev_tf = None
    for tf, K_req, K_out, speedup, bell_div in results:
        if prev_tf is not None and tf != prev_tf:
            print("-" * 56)
        status = "✓" if bell_div <= error_threshold else "✗"
        print(
            f"{tf * 100:>5.1f}%  {K_req:>5d}  {K_out:>6d}  {speedup:>8.2f}x  "
            f"{bell_div:>10.2e}  {status}"
        )
        prev_tf = tf
    print("-" * 56)

    # ── Select best ───────────────────────────────────────────────────────────
    passing = [
        (tf, K_req, K_out, speedup, bell_div)
        for tf, K_req, K_out, speedup, bell_div in results
        if bell_div <= error_threshold
    ]

    if passing:
        best = max(passing, key=lambda x: x[3])  # max speedup
        best_tf, best_K_req = best[0], best[1]
        print(
            f"\n[clust_calib] Recommendation: tail_fraction={best_tf}, "
            f"n_clusters={best_K_req}  "
            f"(speedup={best[3]:.2f}x, B_ell div={best[4]:.2e})"
        )
    else:
        best = min(results, key=lambda x: x[4])  # min B_ell divergence
        best_tf, best_K_req = best[0], best[1]
        print(
            f"\n[clust_calib] WARNING: no (tf, K) pair achieved B_ell div <= "
            f"{error_threshold:.1e}."
        )
        print(
            f"[clust_calib] Returning minimum-divergence pair: "
            f"tail_fraction={best_tf}, n_clusters={best_K_req}  "
            f"(B_ell div={best[4]:.2e})"
        )

    return float(best_tf), int(best_K_req)
