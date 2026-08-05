"""
Runtime calibration for tod_exact_gen_batched.

Calibrates three knobs jointly:
  * n_processes (P)        — worker processes (parallel over days)
  * numba_threads (T)      — threads per worker (parallel over batch via prange)
  * batch_size (B)         — samples per fused-kernel invocation

The fused kernel parallelises Rodrigues+gather over prange(B), so P workers
each using T = NUMBA_NUM_THREADS_DEFAULT oversubscribes by P×; the search
enforces P*T ≤ N_cores.

Strategy (~30s wall time):
  Phase A — single-process throughput vs T at a fixed B.
  Phase B — for best T, sweep B around 16×T to land on the throughput plateau.
  Phase C — enumerate (P, T) with P*T ≤ N_cores; pick max  P × tp(T).
            Memory budget per process must accommodate mp_stacked (already in
            shared memory, but counted defensively) plus transient per-batch
            buffers.
"""

import gc
import time
import numpy as np
import healpy as hp

import numba

import tod_config as config
from tod_io import load_scan_data_batch
from tod_core import precompute_rotation_vector_batch, beam_tod_batch
from tod_utils import _fmt_time, _get_memory_per_process
from tod_beam_math import compute_bell
from beam_cluster import (
    cluster_beam_pixels,
    _tangent_frame,
    _to_unit,
    _whitener,
)


def _k_crossover(beam_vals, n_tail):
    """Cluster count at which the optimal quantizer cell shrinks to one node.

    Clustering has two regimes, and which one a configuration sits in decides
    both how fast the error falls with the cluster budget and whether whitened
    assignment helps at all.

    Below the crossover the cells are large compared with the node grid, the
    continuum (Zador) optimum applies, and the within-cluster scatter falls as
    ``1/K``.  Above it the optimum asks for cells smaller than one grid cell,
    which cannot be delivered: the inner tail degenerates into one-node cells
    of zero scatter and each extra centroid instead pushes that singleton
    boundary outward into steeply falling beam power, so the error falls
    exponentially.  Cells above the crossover are pinned to the node lattice
    rather than free to take the shape an anisotropic distortion measure asks
    for, which is why whitened assignment stops paying there.

    The crossover is where the singleton region closes.  Equating the centroid
    budget at that point gives ``K_x = S(B_cut) / (a^2 sqrt(B_cut))`` with
    ``S(B) = int sqrt(B) dOmega`` over the clustered region.  Written in the
    normalised node weights ``w_s`` the grid spacing cancels, which is what
    makes this evaluable for an arbitrary beam:

        K_x = sum_{s in tail} sqrt(w_s) / sqrt(max_{s in tail} w_s).

    A Gaussian beam collapses this to ``2 * P / a^2`` with
    ``P = 2 * pi * sigma_x * sigma_y``, independent of the tail fraction.  Real
    beams are not Gaussian and their crossover does move with the tail
    fraction, so it is measured here rather than taken from a fitted width.

    Args:
        beam_vals: (S,) beam weights, any normalisation.
        n_tail: number of lowest-weight nodes forming the clustered tail.

    Returns:
        float: the crossover cluster count, or 0.0 for an empty tail.
    """
    w = np.asarray(beam_vals, dtype=np.float64)
    if n_tail <= 0 or w.size == 0:
        return 0.0
    tail = np.sort(w)[:n_tail]
    w_cut = tail[-1]
    if w_cut <= 0:
        return 0.0
    return float(np.sqrt(np.clip(tail, 0.0, None)).sum() / np.sqrt(w_cut))


def _whitening_is_inert(beam_data):
    """True when whitened assignment reproduces the plain partition exactly.

    ``_whitener`` declines to build a transform for a beam already round to
    within its guard, so both settings of ``whiten`` then cluster the same
    vectors and return the same labels.  Sweeping both would pay twice for one
    partition.
    """
    for data in beam_data.values():
        v = _to_unit(data["vec_orig"])
        w = np.asarray(data["beam_vals"], dtype=np.float64)
        frame = _tangent_frame(v, w)
        if frame is not None and _whitener(v, w, frame) is not None:
            return False
    return True


def _beam_frame_rotation(vec, weights):
    """Rotation taking the power-weighted beam centre onto the north pole."""
    v = np.asarray(vec, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    c = (w[:, None] * v).sum(axis=0)
    n = np.linalg.norm(c)
    if n <= 0:
        return np.eye(3)
    c = c / n
    axis = np.cross(c, np.array([0.0, 0.0, 1.0]))
    s, co = np.linalg.norm(axis), float(c[2])
    if s < 1e-15:
        return np.eye(3) if co > 0 else -np.eye(3)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return np.eye(3) + k + k @ k * ((1.0 - co) / s**2)


def beam_m2_multipoles(vec, weights, ells, rotation=None):
    """The beam's ``m = ±2`` multipoles ``b_{l,2}``, exactly on the sphere.

    This is the quantity that leaks temperature into polarisation: the spin-2
    response couples to the beam's ``m = ±2`` harmonic, and a clustering that
    changes it has changed the effective beam's ellipticity.

    Evaluated as ``b_{l,2} = sum_i w_i Pbar_l^2(cos theta_i) exp(-2 i phi_i)``
    with ``(theta, phi)`` in the beam-centred frame and ``Pbar`` the normalised
    associated Legendre function, by the standard stable upward recursion.

    Why not the second-moment deficit.  ``Delta M = -Sigma_bar`` is only the
    ``k^2`` Taylor coefficient of the transfer.  Above ``l ~ 100`` the centroid
    phases ``exp(-i k . c_k)`` decohere, so the error is a phase-incoherent sum
    of per-cell covariances and no longer follows its own low-order expansion:
    a reweighting can drive ``Delta M`` to machine precision while leaving the
    real ``m = ±2`` untouched.  A criterion built on ``Delta M`` can therefore
    certify a configuration that leaks.

    Args:
        vec: (S, 3) beam-node unit vectors.
        weights: (S,) node weights.
        ells: multipoles to evaluate.
        rotation: (3, 3) beam-frame rotation. ``None`` derives it from this
            node set; pass a shared one to compare two sets in one frame.

    Returns:
        np.ndarray: complex ``b_{l,2}`` for each requested multipole.
    """
    v = _to_unit(vec)
    w = np.asarray(weights, dtype=np.float64)
    R = _beam_frame_rotation(v, w) if rotation is None else rotation
    n = v @ R.T
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    x = np.clip(n[:, 2], -1.0, 1.0)
    phase = w * np.exp(-2j * np.arctan2(n[:, 1], n[:, 0]))

    ells = np.asarray(sorted(int(e) for e in ells), dtype=np.int64)
    lmax, m = int(ells.max()), 2
    st = np.sqrt(np.clip(1.0 - x * x, 0.0, None))
    p = np.full_like(x, np.sqrt(1.0 / (4.0 * np.pi)))
    for mm in (1, 2):
        p = -np.sqrt((2.0 * mm + 1.0) / (2.0 * mm)) * st * p

    want = {2: p} if 2 in ells else {}
    prev, cur, lcur = p, np.sqrt(2.0 * m + 3.0) * x * p, 3
    target = set(int(e) for e in ells)
    if 3 in target:
        want[3] = cur
    while lcur < lmax:
        ln = lcur + 1
        a = np.sqrt((4.0 * ln * ln - 1.0) / (ln * ln - m * m))
        b = np.sqrt(((ln - 1.0) ** 2 - m * m) / (4.0 * (ln - 1.0) ** 2 - 1.0))
        prev, cur, lcur = cur, a * (x * cur - b * prev), ln
        if lcur in target:
            want[lcur] = cur
    return np.array([(phase * want[int(e)]).sum() for e in ells])


def _m2_probe_ells(lmax, n=8):
    """Multipoles the m = ±2 error is scored on: log-spaced over the sky band."""
    lo = max(2, int(lmax) // 32)
    return np.unique(np.round(np.geomspace(lo, max(lo + 1, int(lmax)), n)).astype(int))


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


def _make_probe_data(beam_data, folder_scan, probe_day, n_samples):
    theta_p, phi_p, psi_p = load_scan_data_batch(
        folder_scan, probe_day, 0, n_samples, dtype=config.precision_dtype
    )
    n = min(n_samples, len(phi_p))
    return phi_p[:n], theta_p[:n], psi_p[:n]


def _run_one(
    nside,
    mp,
    beam_data,
    ra0,
    dec0,
    phi_p,
    theta_p,
    psi_p,
    bs,
    interp_mode,
    center_idx=None,
    z_skip_threshold=-1.0,
):
    """Run probe at given batch size. Returns wall time."""
    n = len(phi_p)
    n_batches = (n + bs - 1) // bs
    t0 = time.perf_counter()
    for b in range(n_batches):
        s, e = b * bs, min((b + 1) * bs, n)
        phi_b, theta_b, psi_b = phi_p[s:e], theta_p[s:e], psi_p[s:e]
        rot_vecs, betas = precompute_rotation_vector_batch(
            ra0, dec0, phi_b, theta_b, center_idx=center_idx
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
                z_skip_threshold=z_skip_threshold,
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
    center_idx=None,
    z_skip_threshold=-1.0,
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
        nside,
        mp,
        beam_data,
        ra0,
        dec0,
        phi_p,
        theta_p,
        psi_p,
        bs,
        interp_mode,
        center_idx=center_idx,
        z_skip_threshold=z_skip_threshold,
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
            nside,
            mp,
            beam_data,
            ra0,
            dec0,
            phi_p,
            theta_p,
            psi_p,
            bs,
            interp_mode,
            center_idx=center_idx,
            z_skip_threshold=z_skip_threshold,
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


def calibrate_runtime(
    beam_data,
    folder_scan,
    probe_day,
    mp,
    n_cpu_ceiling,
    max_processes_user,
    interp_mode="bilinear",
    prefix="",
    center_idx=None,
    z_skip_threshold=-1.0,
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
    nside = hp.get_nside(next(iter(mp.values())) if isinstance(mp, dict) else mp[0])
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
            center_idx=center_idx,
            z_skip_threshold=z_skip_threshold,
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
                    center_idx=center_idx,
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
    ellipticity_tolerance=2.0,
    bell_lmax=None,
    interp_mode="bilinear",
    tail_fractions=None,
    n_clusters_list=None,
    whiten_options=(False, True),
):
    """Find (tail_fraction, n_clusters, whiten) maximising speedup within the
        error budget.

        Two bounds are enforced, because they are sensitive to different harmonics
        and neither implies the other.

        ``error_threshold`` bounds the relative-RMS divergence of ``B_ell``, which
        depends on each node only through its distance from the beam centre and is
        therefore the ``m = 0`` harmonic alone: it constrains the beam *width*.

    ``ellipticity_tolerance`` bounds the ellipticity the clustering *adds* to
        the beam, measured as the RMS over the sky band of
        ``|b_{l,2}[clustered] - b_{l,2}[exact]|`` — the beam's own ``m = ±2``
        multipoles, computed exactly on the sphere by
        :func:`beam_m2_multipoles`.  ``B_ell`` cannot see this at all: no
        rearrangement of nodes at fixed radius changes it, so a clustering that
        reshapes the beam scores identically to one that does not.  It is the
        harmful term, since an ``m = ±2`` error is what sources T->P leakage
        and, unlike the interpolation kernel, is not in the mapmaker's forward
        model to be deconvolved.

        This is deliberately *not* the second-moment deficit ``Delta M``.  That
        is only the ``k^2`` Taylor coefficient of the transfer, and above
        ``l ~ 100`` the centroid phases decohere so the error stops following
        it: a reweighting can drive ``Delta M`` to machine precision while the
        real ``m = ±2`` is untouched, which would let the gate certify a
        configuration that leaks.

        The bound is relative — a factor over the smallest ``q2`` any point in the
        sweep achieves for this beam — because what is achievable depends on the
        beam, the grid spacing and the cluster budget together and cannot be known
        before running.  1.0 keeps only the least-ellipticity configuration, larger
        values trade ellipticity for speed, ``None`` reports ``q2`` without gating.
        The table also carries ``q2`` as a fraction of the beam's own quadrupole,
        which is the number to compare against a leakage budget.

        ``whiten_options`` is the third axis.  Whitened assignment is
        shape-preserving in the continuum-quantizer regime below the crossover
        cluster count ``K_x`` (:func:`_k_crossover`).  Above ``K_x`` it may
        lose on both harmonics, but *whether* it does is a property of the
        beam rather than of ``K_x``: the residual anisotropy the whitening
        leaves behind saturates near 0.6–0.8, so a beam whose own ellipticity
        exceeds that keeps winning at every cluster count.  Measured on
        elliptical Gaussians at ``a = 2'``, ``f = 0.005``: axis ratio 1.5 and 2
        lose above ``K_x``, while 3 and 4 still win by 1.6–2.9x at
        ``K/K_x = 4``.  The axis is therefore swept rather than predicted.
        Pass a single-element tuple to pin the choice and halve the sweep.

        Computes reference B_ell (power_cut=1.0) from unclustered beam, then
        sweeps a (tail_fraction × n_clusters × whiten) grid. The point maximising
        speedup subject to both bounds wins; if none qualifies, the point with
        the smallest added ellipticity is returned with a warning.

        ``tail_fractions`` overrides the default tail-fraction grid — e.g. to
        enforce a lower bound derived from the noise floor of a measured beam,
        so that the exactly-kept main-lobe pixels stay signal-dominated.
        ``n_clusters_list`` overrides the cluster-count grid.  Left at ``None``
        the grid is built per tail fraction around that beam's own ``K_x``,
        spanning ``0.1`` to ``4`` times it, so both regimes are sampled and the
        exponential leg above ``K_x`` is reachable.  A fixed grid cannot do
        this: ``K_x`` varies by a factor of four across tail fractions on a real
        beam, and the cluster count a tight error budget needs sits above the
        crossover, where a grid capped below it would never look.

    Returns:
        tuple[float, int, bool]: the chosen ``(tail_fraction, n_clusters,
        whiten)``.
    """
    if tail_fractions is None:
        tail_fractions = (0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
    else:
        tail_fractions = tuple(sorted(float(tf) for tf in tail_fractions))
    if n_clusters_list is not None:
        n_clusters_list = tuple(sorted(int(k) for k in n_clusters_list))
    whiten_options = tuple(dict.fromkeys(bool(w) for w in whiten_options))
    if not whiten_options:
        raise ValueError("whiten_options must contain at least one value")
    if len(whiten_options) > 1 and any("c4" in d for d in beam_data.values()):
        # The quadrant path ignores whitening, so the axis would measure nothing.
        print(
            "[clust_calib] Quadrant (C4) clustering is active — whitening does "
            "not apply; sweeping one."
        )
        whiten_options = (False,)
    elif len(whiten_options) > 1 and _whitening_is_inert(beam_data):
        print(
            "[clust_calib] Beam is round to within the whitening guard, so the "
            "whitened and plain assignments are identical — sweeping one."
        )
        whiten_options = (False,)

    if bell_lmax is None:
        if mp is not None:
            _ref = next(iter(mp.values())) if isinstance(mp, dict) else mp[0]
            bell_lmax = 2 * hp.get_nside(_ref)
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
            # bvals are finished quadrature weights (clustered or not) and
            # already carry the cos(dec) Jacobian.
            apply_jacobian=False,
            verbose=False,
        )
        return bell

    print(
        f"[clust_calib] Computing reference B_ell (power_cut=1.0, lmax={bell_lmax}) …"
    )
    ref_bells = {}
    for bf, data in beam_data.items():
        ref_bells[bf] = _bell_from_vecs(data["vec_orig"], data["beam_vals"])

    # The m = +-2 reference: the unclustered beam's own multipoles, and the
    # frame they were measured in, so every candidate is scored against the
    # same beam axes rather than its own.
    m2_ells = _m2_probe_ells(bell_lmax)
    m2_frame, m2_ref = {}, {}
    for bf, data in beam_data.items():
        R = _beam_frame_rotation(_to_unit(data["vec_orig"]), data["beam_vals"])
        m2_frame[bf] = R
        m2_ref[bf] = beam_m2_multipoles(
            data["vec_orig"], data["beam_vals"], m2_ells, rotation=R
        )

    S_bf = {bf: data["n_sel"] for bf, data in beam_data.items()}
    # The beam's own m = +-2, the scale added ellipticity is read against. For a
    # symmetric beam this is a noise floor rather than a signal, so the ratio
    # column against it is meaningless there.
    q2_beam = max(
        (float(np.sqrt(np.mean(np.abs(b) ** 2))) for b in m2_ref.values()),
        default=0.0,
    )

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

    # The crossover cluster count separates the two error regimes and decides
    # whether whitening can work, so it sets where the sweep looks.  It is the
    # most constraining beam's value, and it moves with the tail fraction.
    k_cross_per_tf = {
        tf: max(
            (
                _k_crossover(data["beam_vals"], n_tail_per_bf_tf[bf][tf])
                for bf, data in beam_data.items()
            ),
            default=0.0,
        )
        for tf in tail_fractions
    }

    def _k_grid(tf):
        """Cluster counts to try at this tail fraction."""
        if n_clusters_list is not None:
            return n_clusters_list
        n_tail_min = min(n_tail_per_bf_tf[bf][tf] for bf in beam_data)
        kx = k_cross_per_tf[tf]
        if kx <= 0:
            return (max(1, n_tail_min - 1),)
        grid = np.unique(np.round(kx * np.geomspace(0.1, 4.0, 8)).astype(int))
        grid = grid[(grid >= 1) & (grid < n_tail_min)]
        return tuple(int(k) for k in grid) or (max(1, n_tail_min - 1),)

    print("[clust_calib] Sweeping clustering parameters …")
    print(
        "[clust_calib] K_x (crossover) per tail fraction: "
        + ", ".join(f"{tf:g}→{k_cross_per_tf[tf]:.0f}" for tf in tail_fractions)
    )
    results = []
    for tf in tail_fractions:
        # Above K_x the whitened/plain shape ratio has saturated, because the
        # residual anisotropy the whitening leaves behind has stopped growing.
        # One loss there is therefore predictive of every larger K, so the
        # whitened leg is dropped for the rest of this tail fraction and the
        # most expensive cells are never run twice.  Whether it ever loses is
        # a property of the beam and not of K_x: a beam elliptical enough that
        # its own ellipticity exceeds that residual keeps winning at every K,
        # and for such a beam this exit simply never fires.  Below K_x nothing
        # is skipped, since the ratio is still moving fast there and a single
        # Lloyd local minimum could misread it.
        whiten_beaten = False
        for K_req in _k_grid(tf):
            active = (
                tuple(w for w in whiten_options if not w)
                if whiten_beaten
                else whiten_options
            )
            q2_by_whiten = {}
            for whiten in active:
                K_out_per_bf = {}
                bell_divs = []
                q2s = []
                reduced = True
                for bf, data in beam_data.items():
                    n_tail = n_tail_per_bf_tf[bf][tf]
                    if K_req >= n_tail:
                        # Tail already fits in K_req clusters — no reduction
                        # possible, so this row measures nothing.
                        K_out_per_bf[bf] = S_bf[bf]
                        bell_divs.append(0.0)
                        q2s.append(0.0)
                        reduced = False
                        continue
                    vec_c, bv_c, _ = cluster_beam_pixels(
                        data["vec_orig"],
                        data["beam_vals"],
                        n_clusters=K_req,
                        tail_fraction=tf,
                        verbose=False,
                        whiten=whiten,
                        c4=data.get("c4"),
                    )
                    K_out_per_bf[bf] = len(bv_c)
                    bell_clust = _bell_from_vecs(vec_c, bv_c)
                    bell_ref = ref_bells[bf]
                    ref_rms = float(np.sqrt(np.mean(bell_ref**2)))
                    bell_div = float(np.sqrt(np.mean((bell_clust - bell_ref) ** 2))) / (
                        ref_rms + 1e-30
                    )
                    bell_divs.append(bell_div)
                    b2 = beam_m2_multipoles(vec_c, bv_c, m2_ells, rotation=m2_frame[bf])
                    q2s.append(float(np.sqrt(np.mean(np.abs(b2 - m2_ref[bf]) ** 2))))

                mean_bell_div = float(np.mean(bell_divs))
                max_q2 = float(np.max(q2s))
                speedup = float(
                    np.mean([S_bf[bf] / K_out_per_bf[bf] for bf in beam_data])
                )
                K_out_repr = int(np.mean(list(K_out_per_bf.values())))
                k_ratio = K_req / k_cross_per_tf[tf] if k_cross_per_tf[tf] > 0 else 0.0
                q2_by_whiten[whiten] = max_q2
                results.append(
                    (
                        tf,
                        K_req,
                        whiten,
                        K_out_repr,
                        speedup,
                        mean_bell_div,
                        max_q2,
                        reduced,
                        k_ratio,
                    )
                )

            if (
                not whiten_beaten
                and k_ratio > 1.0
                and reduced
                and len(q2_by_whiten) == 2
                and q2_by_whiten[True] > q2_by_whiten[False]
            ):
                whiten_beaten = True
                print(
                    f"[clust_calib]   tail={tf:g}: whitening lost at K={K_req} "
                    f"(K/K_x={k_ratio:.2f}); dropping it for larger K here."
                )

    # The achievable floor is measured, not assumed: it is the least added
    # ellipticity anywhere in the sweep that already meets the m = 0 bound.
    # Rows that clustered nothing are excluded: they score q2 = 0 because no
    # nodes were merged, and a single one would drag the floor to zero and
    # switch the m = ±2 gate off without saying so.
    ok_m0 = [r for r in results if r[7] and r[5] <= error_threshold]
    q2_floor = min((r[6] for r in ok_m0), default=0.0)
    q2_cap = (
        ellipticity_tolerance * q2_floor
        if (ellipticity_tolerance and q2_floor > 0)
        else None
    )

    print()
    print(f"[clust_calib] B_ell (m=0) <= {error_threshold:.1e}")
    if q2_cap:
        print(
            f"[clust_calib] added ellipticity q2(l={bell_lmax}) <= "
            f"{ellipticity_tolerance:g} x {q2_floor:.2e} = {q2_cap:.2e}   "
            f"(beam's own quadrupole q2_beam = {q2_beam:.2e})"
        )
    else:
        print("[clust_calib] added ellipticity: reported only")
    print(
        f"{'tail%':>6s}  {'K':>6s}  {'K/K_x':>6s}  {'whiten':>6s}  {'K_out':>6s}  "
        f"{'speedup':>8s}  {'B_ell div':>10s}  {'q2 added':>10s}  "
        f"{'/q2_beam':>9s}  {'status'}"
    )
    print("-" * 100)
    prev_tf = None
    for tf, K_req, whiten, K_out, speedup, bell_div, q2, reduced, k_ratio in results:
        if prev_tf is not None and tf != prev_tf:
            print("-" * 100)
        good_m0 = bell_div <= error_threshold
        good_m2 = (q2_cap is None) or q2 <= q2_cap
        if not reduced:
            status = "— no reduction"
        elif good_m0 and good_m2:
            status = "✓"
        else:
            status = "✗ m=0" if not good_m0 else "✗ m=±2"
        rel = q2 / q2_beam if q2_beam > 0 else float("inf")
        print(
            f"{tf * 100:>5.1f}%  {K_req:>6d}  {k_ratio:>6.2f}  "
            f"{'yes' if whiten else 'no':>6s}  {K_out:>6d}  {speedup:>8.2f}x  "
            f"{bell_div:>10.2e}  {q2:>10.2e}  {rel:>9.2e}  {status}"
        )
        prev_tf = tf
    print("-" * 100)

    passing = [r for r in ok_m0 if (q2_cap is None) or r[6] <= q2_cap]
    if passing:
        # Rank on speedup, break ties on the least added ellipticity. A whitened
        # row and its plain twin cluster the same tail to the same K, so their
        # speedups are identical and the whiten axis presents nothing but ties;
        # without the second key every one of them resolves to whichever leg was
        # swept first, independently of how much shape error it commits.
        best = max(passing, key=lambda x: (x[4], -x[6]))
        print(
            f"\n[clust_calib] Recommendation: tail_fraction={best[0]}, "
            f"n_clusters={best[1]}, whiten={best[2]}  "
            f"(speedup={best[4]:.2f}x, B_ell div={best[5]:.2e}, "
            f"q2 added={best[6]:.2e}, K/K_x={best[8]:.2f})"
        )
    else:
        # Fall back on the harmful term: an m=+-2 error reshapes the beam and
        # sources T->P leakage, where the m=0 term only rescales its width.
        # Rows that clustered nothing are excluded, or the fallback would
        # "win" by recommending a configuration that does no clustering.
        candidates = [r for r in results if r[7]] or results
        best = min(candidates, key=lambda x: x[6])
        print(
            f"\n[clust_calib] WARNING: no (tf, K, whiten) point met both bounds "
            f"(B_ell <= {error_threshold:.1e}"
            + (f", q2 <= {q2_cap:.1e}" if q2_cap else "")
            + ")."
        )
        print(
            f"[clust_calib] Returning the point with the least added "
            f"ellipticity: tail_fraction={best[0]}, n_clusters={best[1]}, "
            f"whiten={best[2]}  "
            f"(B_ell div={best[5]:.2e}, q2 added={best[6]:.2e})"
        )

    return float(best[0]), int(best[1]), bool(best[2])
