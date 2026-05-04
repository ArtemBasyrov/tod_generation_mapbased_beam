"""
Runtime calibration for tod_exact_gen_batched.

Calibrates three knobs jointly:
  * n_processes (P)        — worker processes (parallel over days)
  * numba_threads (T)      — threads per worker (parallel over batch via prange)
  * batch_size (B)         — samples per fused-kernel invocation

The fused kernel (a1d5d36) parallelises the entire Rodrigues+gather over
prange(B). Spawning P workers each using T = NUMBA_NUM_THREADS_DEFAULT (=all
cores) oversubscribes by P×; the new search enforces P*T ≤ N_cores.

Strategy (~30s wall time):
  Phase A — single-process throughput vs T at a fixed B.
  Phase B — for best T, sweep B around 16×T to land on the throughput plateau.
  Phase C — enumerate (P, T) with P*T ≤ N_cores; pick max  P × tp(T).
            Memory budget per process must accommodate mp_stacked (already in
            shared memory, but counted defensively) plus transient per-batch
            buffers.

Beam clustering calibration is unchanged (driven by science accuracy, not speed).
"""

import gc
import time
import numpy as np
import healpy as hp

import numba

from tod_io import _load_scan_data_batch
from tod_core import precompute_rotation_vector_batch, beam_tod_batch
from tod_utils import _fmt_time, _get_memory_per_process, compute_bell

# Per-batch transient memory (bytes per sample). The fused kernel no longer
# materialises a (B, S, 3) Rodrigues buffer (commit a1d5d36); only small
# B-scaled arrays remain (pointings, weights, output).
_BYTES_PER_SAMPLE_TRANSIENT = {
    "nearest": 80,
    "bilinear": 120,
    "bicubic": 400,
    "gaussian": 400,
}
_BYTES_PER_SAMPLE_TRANSIENT_DEFAULT = 200
_MEMORY_SAFETY_FACTOR = 1.5

# Each thread needs at least this many samples to amortise prange overhead.
_MIN_SAMPLES_PER_THREAD = 256
# Above this we hit a plateau; further B growth wastes memory.
_TARGET_SAMPLES_PER_THREAD = 1024

# Probe sizing — keep total wall time near 30s.
_PROBE_TARGET_SECONDS = 1.5  # per measurement cell
_PROBE_MIN_SAMPLES = 20_000
_PROBE_MAX_SAMPLES = 400_000


# ── Memory model ─────────────────────────────────────────────────────────────


def _per_proc_static_bytes(beam_data, nside):
    """Per-worker static memory: mp_stacked for every beam file."""
    npix = 12 * nside * nside
    return sum(
        d["mp_stacked"].nbytes
        if "mp_stacked" in d
        else len(d["comp_indices"]) * npix * 4
        for d in beam_data.values()
    )


def _max_batch_for_memory(mem_per_proc_gb, beam_data, nside, interp_mode):
    """Largest B that fits the per-batch transient buffers in budget."""
    bps = _BYTES_PER_SAMPLE_TRANSIENT.get(
        interp_mode, _BYTES_PER_SAMPLE_TRANSIENT_DEFAULT
    )
    static_gb = _per_proc_static_bytes(beam_data, nside) / 1e9
    transient_budget_gb = mem_per_proc_gb / _MEMORY_SAFETY_FACTOR - static_gb
    if transient_budget_gb <= 0.05:
        return 0
    return max(1, int(transient_budget_gb * 1e9 // bps))


# ── Probe runner ─────────────────────────────────────────────────────────────


def _make_probe_data(beam_data, folder_scan, probe_day, n_samples):
    theta_p, phi_p, psi_p = _load_scan_data_batch(folder_scan, probe_day, 0, n_samples)
    n = min(n_samples, len(phi_p))
    return phi_p[:n], theta_p[:n], psi_p[:n]


def _run_one(nside, mp, beam_data, ra0, dec0, phi_p, theta_p, psi_p, bs, interp_mode):
    """Run probe at given batch size. Returns wall time."""
    n = len(phi_p)
    n_batches = (n + bs - 1) // bs
    t0 = time.perf_counter()
    for b in range(n_batches):
        s, e = b * bs, min((b + 1) * bs, n)
        phi_b, theta_b, psi_b = phi_p[s:e], theta_p[s:e], psi_p[s:e]
        rot_vecs, betas = precompute_rotation_vector_batch(ra0, dec0, phi_b, theta_b)
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
            )
    return time.perf_counter() - t0


def _measure_throughput(
    nside,
    mp,
    beam_data,
    ra0,
    dec0,
    phi_full,
    theta_full,
    psi_full,
    bs,
    n_threads,
    interp_mode,
    prefix="",
):
    """Set thread count, measure throughput at given batch size.

    phi_full/theta_full/psi_full must hold _PROBE_MAX_SAMPLES samples
    (loaded once by the caller); this function slices them.
    """
    numba.set_num_threads(max(1, n_threads))

    # Adapt probe size to target ~_PROBE_TARGET_SECONDS.
    # Use a small pilot to estimate samples/sec, then size the real probe.
    pilot_n = max(bs * 2, 4_000)
    pilot_n = min(pilot_n, len(phi_full))
    phi_p, theta_p, psi_p = phi_full[:pilot_n], theta_full[:pilot_n], psi_full[:pilot_n]
    pilot_t = _run_one(
        nside, mp, beam_data, ra0, dec0, phi_p, theta_p, psi_p, bs, interp_mode
    )
    rate = len(phi_p) / max(pilot_t, 1e-6)
    target_n = int(rate * _PROBE_TARGET_SECONDS)
    target_n = max(_PROBE_MIN_SAMPLES, min(target_n, len(phi_full)))
    target_n = max(target_n, bs * 4)  # at least 4 batches

    if target_n > len(phi_p):
        phi_p = phi_full[:target_n]
        theta_p = theta_full[:target_n]
        psi_p = psi_full[:target_n]

    # Take the best of 2 short runs to suppress noise.
    best_t = float("inf")
    for _ in range(2):
        t = _run_one(
            nside, mp, beam_data, ra0, dec0, phi_p, theta_p, psi_p, bs, interp_mode
        )
        best_t = min(best_t, t)
    gc.collect()
    tp = len(phi_p) / best_t
    print(
        prefix + f"  T={n_threads:>3d}  B={bs:>6d}  "
        f"n={len(phi_p):>7d}  t={_fmt_time(best_t):>7s}  "
        f"tp={tp:>12,.0f} samp/s"
    )
    return tp


def _thread_candidates(n_cores):
    """Powers of 2 up to n_cores, plus n_cores itself."""
    cands = set()
    t = 1
    while t <= n_cores:
        cands.add(t)
        t *= 2
    cands.add(n_cores)
    return sorted(cands)


def _process_thread_pairs(n_cores, max_processes):
    """All (P, T) with P*T ≤ n_cores, P ≤ max_processes, T ≥ 1."""
    pairs = []
    for t in _thread_candidates(n_cores):
        max_p = min(n_cores // t, max_processes)
        for p in range(1, max_p + 1):
            if p * t <= n_cores:
                pairs.append((p, t))
    return pairs


# ── Public entry point ──────────────────────────────────────────────────────


def calibrate_runtime(
    beam_data,
    folder_scan,
    probe_day,
    mp,
    n_cpu_ceiling,
    max_processes_user,
    interp_mode="bilinear",
    prefix="",
):
    """Joint (n_processes, numba_threads, batch_size) calibration.

    Args:
        beam_data: from prepare_beam_data (after clustering, with mp_stacked).
        folder_scan: scan directory.
        probe_day: any valid day index for probe data.
        mp: list of sky-map components.
        n_cpu_ceiling: hard ceiling from scheduler/affinity (_get_ncpus()).
        max_processes_user: user-configured n_processes (laptop cap, etc).
            Acts as an upper bound on P.
        interp_mode: 'nearest' or 'bilinear'.
        prefix: log prefix.

    Returns:
        (n_processes, n_threads, batch_size)
    """
    nside = hp.get_nside(mp[0])
    first_bf = next(iter(beam_data))
    ra0, dec0 = beam_data[first_bf]["ra"], beam_data[first_bf]["dec"]

    n_cores = max(1, n_cpu_ceiling)
    max_p = max(1, min(max_processes_user, n_cores))
    total_mem_gb = _get_memory_per_process(1)

    print(
        prefix + f"[calibrate] n_cores={n_cores}  max_processes={max_p}  "
        f"total_mem={total_mem_gb:.1f} GB  interp={interp_mode}"
    )

    # ── Phase A: throughput vs threads at a reference batch size ────────────
    # Use a B large enough to feed the maximum thread count (T=n_cores).
    ref_bs = max(_TARGET_SAMPLES_PER_THREAD * n_cores, 4096)
    # Cap by memory at P=1.
    bs_cap_p1 = _max_batch_for_memory(total_mem_gb, beam_data, nside, interp_mode)
    if bs_cap_p1 < 256:
        raise RuntimeError(
            f"[calibrate] memory too tight: per-process budget "
            f"{total_mem_gb:.2f} GB cannot fit static beam arrays "
            f"({_per_proc_static_bytes(beam_data, nside) / 1e9:.2f} GB) plus "
            f"any reasonable batch."
        )
    ref_bs = min(ref_bs, bs_cap_p1)

    # Load probe scan data once — _measure_throughput slices as needed.
    phi_full, theta_full, psi_full = _make_probe_data(
        beam_data, folder_scan, probe_day, _PROBE_MAX_SAMPLES
    )

    print(prefix + f"[calibrate] Phase A — sweep threads at B={ref_bs}")
    tp_by_threads = {}
    for t in _thread_candidates(n_cores):
        tp = _measure_throughput(
            nside,
            mp,
            beam_data,
            ra0,
            dec0,
            phi_full,
            theta_full,
            psi_full,
            bs=ref_bs,
            n_threads=t,
            interp_mode=interp_mode,
            prefix=prefix,
        )
        tp_by_threads[t] = tp

    # ── Phase B: at each candidate T, find a good B (only for non-trivial T) ─
    # We sweep B at the top few T candidates because B-scaling can differ.
    print(prefix + "[calibrate] Phase B — sweep batch size at top thread counts")
    top_threads = sorted(tp_by_threads, key=lambda t: -tp_by_threads[t])[:3]
    bs_by_threads = {}
    tp_at_bs = {}  # (T, B_chosen) -> throughput
    for t in top_threads:
        # Candidate batch sizes around the target sweet spot for this T.
        target = _TARGET_SAMPLES_PER_THREAD * t
        cands = sorted(
            {
                max(_MIN_SAMPLES_PER_THREAD * t, 256),
                max(target // 2, 512),
                target,
                target * 2,
            }
        )
        cands = [b for b in cands if 256 <= b <= bs_cap_p1]
        if not cands:
            cands = [min(ref_bs, bs_cap_p1)]
        # Prepend the already-measured ref point if not already in the list
        best_b, best_tp = ref_bs, tp_by_threads[t]
        for b in cands:
            if b == ref_bs:
                tp = tp_by_threads[t]
            else:
                tp = _measure_throughput(
                    nside,
                    mp,
                    beam_data,
                    ra0,
                    dec0,
                    phi_full,
                    theta_full,
                    psi_full,
                    bs=b,
                    n_threads=t,
                    interp_mode=interp_mode,
                    prefix=prefix,
                )
            tp_at_bs[(t, b)] = tp
            if tp > best_tp:
                best_tp, best_b = tp, b
        bs_by_threads[t] = best_b
        tp_by_threads[t] = best_tp  # update with best B for this T

    # ── Phase C: enumerate (P, T) with P*T ≤ n_cores; pick best ─────────────
    print(prefix + "[calibrate] Phase C — score (P, T) combinations")
    print(
        prefix + f"  {'P':>3s}  {'T':>3s}  {'B':>6s}  "
        f"{'tp/proc':>14s}  {'est total':>14s}  status"
    )
    print(prefix + "  " + "-" * 60)

    best = None  # (score, P, T, B)
    for p, t in _process_thread_pairs(n_cores, max_p):
        if t not in tp_by_threads:
            continue  # only score Ts we measured
        mem_per_proc = total_mem_gb / p
        bs_cap = _max_batch_for_memory(mem_per_proc, beam_data, nside, interp_mode)
        if bs_cap < _MIN_SAMPLES_PER_THREAD * t:
            print(
                prefix + f"  {p:>3d}  {t:>3d}  {'-':>6s}  {'-':>14s}  {'-':>14s}  oom"
            )
            continue
        # Use the B chosen in phase B for this T, capped by per-proc memory.
        b_choice = min(bs_by_threads.get(t, ref_bs), bs_cap)
        # Look up throughput at (T, b_choice) if measured, else use phase A ref.
        tp_per_proc = tp_at_bs.get((t, b_choice), tp_by_threads[t])
        score = p * tp_per_proc
        marker = ""
        if best is None or score > best[0]:
            best = (score, p, t, b_choice)
            marker = "  ←"
        print(
            prefix + f"  {p:>3d}  {t:>3d}  {b_choice:>6d}  "
            f"{tp_per_proc:>14,.0f}  {score:>14,.0f}{marker}"
        )

    if best is None:
        raise RuntimeError(
            "[calibrate] no viable (P, T) combination — "
            "memory too tight for any batch size."
        )
    _, P, T, B = best
    print(
        prefix + f"[calibrate] → n_processes={P}  numba_threads={T}  "
        f"batch_size={B}  (est total tp={best[0]:,.0f} samp/s)"
    )
    return P, T, B


# ── Beam clustering calibration (unchanged) ──────────────────────────────────


def _run_clustering_probe(
    nside, mp, beam_entries, rot_vecs, phi_b, theta_b, psis_b, interp_mode
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
):
    """Find (tail_fraction, n_clusters) maximising speedup s.t. B_ell error
    ≤ error_threshold.

    Computes reference B_ell (power_cut=1.0) from unclustered beam, then
    sweeps a fixed (tail_fraction × n_clusters) grid. The pair maximising
    speedup with relative-RMS B_ell divergence ≤ error_threshold wins; if
    no pair qualifies, the minimum-divergence pair is returned with a warning.
    """
    tail_fractions = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    n_clusters_list = (10, 20, 50, 100, 200, 500, 1000, 2000)

    from beam_cluster import cluster_beam_pixels

    if bell_lmax is None:
        if mp is not None:
            bell_lmax = 2 * hp.get_nside(mp[0])
        else:
            bell_lmax = 500

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

    print(
        f"[clust_calib] Computing reference B_ell (power_cut=1.0, lmax={bell_lmax}) …"
    )
    ref_bells = {}
    for bf, data in beam_data.items():
        ref_bells[bf] = _bell_from_vecs(data["vec_orig"], data["beam_vals"])

    S_bf = {bf: data["n_sel"] for bf, data in beam_data.items()}

    # Pre-compute n_tail for each (beam_file, tail_fraction) to enable
    # short-circuiting when K_req >= n_tail (no clustering occurs).
    n_tail_per_bf_tf = {}
    for bf, data in beam_data.items():
        bv = data["beam_vals"]
        n_tail_per_bf_tf[bf] = {}
        sort_idx = np.argsort(bv)
        cumsum = np.cumsum(bv[sort_idx])
        S = len(bv)
        for tf in tail_fractions:
            n_tail = int(np.searchsorted(cumsum, tf, side="right"))
            n_tail_per_bf_tf[bf][tf] = max(1, min(n_tail, S - 1))

    print("[clust_calib] Sweeping clustering parameters …")
    results = []
    for tf in tail_fractions:
        for K_req in n_clusters_list:
            K_out_per_bf = {}
            bell_divs = []
            for bf, data in beam_data.items():
                n_tail = n_tail_per_bf_tf[bf][tf]
                if K_req >= n_tail:
                    # Tail already fits in K_req clusters — no reduction possible.
                    K_out_per_bf[bf] = S_bf[bf]
                    bell_divs.append(0.0)
                    continue
                vec_c, bv_c, _ = cluster_beam_pixels(
                    data["vec_orig"],
                    data["beam_vals"],
                    n_clusters=K_req,
                    tail_fraction=tf,
                    verbose=False,
                )
                K_out_per_bf[bf] = len(bv_c)
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

    passing = [r for r in results if r[4] <= error_threshold]
    if passing:
        best = max(passing, key=lambda x: x[3])
        best_tf, best_K_req = best[0], best[1]
        print(
            f"\n[clust_calib] Recommendation: tail_fraction={best_tf}, "
            f"n_clusters={best_K_req}  "
            f"(speedup={best[3]:.2f}x, B_ell div={best[4]:.2e})"
        )
    else:
        best = min(results, key=lambda x: x[4])
        best_tf, best_K_req = best[0], best[1]
        print(
            f"\n[clust_calib] WARNING: no (tf, K) pair achieved B_ell div "
            f"<= {error_threshold:.1e}."
        )
        print(
            f"[clust_calib] Returning minimum-divergence pair: "
            f"tail_fraction={best_tf}, n_clusters={best_K_req}  "
            f"(B_ell div={best[4]:.2e})"
        )

    return float(best_tf), int(best_K_req)
