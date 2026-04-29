import os
import time
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from functools import partial

import numpy as np
import healpy as hp

import tod_config as config
from tod_io import load_beam, load_scan_information, open_scan_day
from tod_core import precompute_rotation_vector_batch, beam_tod_batch
from tod_calibrate import calibrate_runtime, calibrate_beam_clustering
from tod_utils import (
    _get_ncpus,
    _fmt_time,
    _should_print_batch,
    _compute_dB_threshold_from_power,
)
from beam_cluster import cluster_beam_pixels, cluster_cached_arrays

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ── Config ────────────────────────────────────────────────────────────────────
folder_beam = config.FOLDER_BEAM
folder_scan = config.FOLDER_SCAN
folder_tod_output = config.FOLDER_TOD_OUTPUT
beam_files = [config.beam_file_I, config.beam_file_Q, config.beam_file_U]
start_day = config.start_day
end_day = config.end_day

interp_mode = config.beam_interp_method

# ── Worker-global state (populated by _worker_init in each spawned process) ───
# MP is loaded in the parent, placed in shared memory, and attached here
# read-only by each worker — no copies, no re-loading the FITS file.
_g_mp = None  # list of 3 float32 arrays (views into shared memory)
_g_beam_data = None  # beam_data dict with mp_stacked from shared memory
_g_shm_handles = []  # SharedMemory handles kept alive for worker lifetime


# ── Pool initialiser ─────────────────────────────────────────────────────────


def _worker_init(beam_data_static, mp_desc, beam_shm_descs, n_threads):
    """
    Called once in each spawned worker process.

    Attaches to the SharedMemory blocks created by the parent and builds
    zero-copy numpy views that the worker uses for the lifetime of the process.
    The SharedMemory handles are stored in _g_shm_handles so they are not
    garbage-collected (which would invalidate the buffer).

    Parameters
    ----------
    beam_data_static : dict  — beam_data without large arrays (small scalars /
                               tiny arrays safe to pickle: beam_vals, vec_orig,
                               psi_grid, sel, ra, dec, …)
    mp_desc          : dict  — {'name', 'shape', 'dtype'} for the stacked MP block
    beam_shm_descs   : dict  — {beam_filename: {'name', 'shape', 'dtype'}}
                               for mp_stacked
    """
    global _g_mp, _g_beam_data, _g_shm_handles

    if n_threads is not None and n_threads > 0:
        import numba

        numba.set_num_threads(int(n_threads))

    # Attach to the stacked (3, npix) sky-map block
    shm_mp = SharedMemory(name=mp_desc["name"])
    _g_shm_handles.append(shm_mp)  # keep alive
    mp_full = np.ndarray(mp_desc["shape"], dtype=mp_desc["dtype"], buffer=shm_mp.buf)
    _g_mp = [mp_full[i] for i in range(mp_desc["shape"][0])]  # list of 3 views

    # Attach to each beam entry's mp_stacked block
    _g_beam_data = {}
    for bf, static in beam_data_static.items():
        desc = beam_shm_descs[bf]
        shm = SharedMemory(name=desc["name"])
        _g_shm_handles.append(shm)
        ms = np.ndarray(desc["shape"], dtype=desc["dtype"], buffer=shm.buf)
        entry = dict(static)
        entry["mp_stacked"] = ms
        _g_beam_data[bf] = entry


# ── Beam preparation ──────────────────────────────────────────────────────────


def prepare_beam_data(beam_filenames):
    """Load and preprocess all unique beam files into a beam-data dictionary.

    For each unique beam filename, loads the FITS map, selects pixels by power
    threshold, normalises beam weights, and precomputes unit vectors.

    Args:
        beam_filenames (list[str]): List of beam filenames (one per Stokes
            component, in the order ``[I, Q, U]``). Duplicate filenames are
            de-duplicated; the corresponding ``comp_indices`` lists which Stokes
            components share a given beam file.

    Returns:
        dict[str, dict]: Beam-data dictionary keyed by beam filename. Each
            value is a dict with the following entries:

            - ``'ra'`` – RA offset grid [rad]
            - ``'dec'`` – Dec offset grid [rad]
            - ``'beam_vals'`` – Normalised beam weights, shape ``(S,)``
            - ``'sel'`` – Boolean selection mask over the full beam map
            - ``'comp_indices'`` – List of Stokes component indices using this
              beam (e.g. ``[0]`` for I-only, ``[1, 2]`` if Q and U share a map)
            - ``'n_sel'`` – Number of selected pixels ``S``
            - ``'vec_orig'`` – Beam-pixel unit vectors, shape ``(S, 3)``
    """
    beam_groups = {}
    for i, bf in enumerate(beam_filenames):
        beam_groups.setdefault(bf, []).append(i)

    beam_threshold_map = {
        config.beam_file_I: config.power_threshold_I,
        config.beam_file_Q: config.power_threshold_Q,
        config.beam_file_U: config.power_threshold_U,
    }

    beam_data = {}
    for bf, comp_indices in beam_groups.items():
        ra, dec, pixel_map = load_beam(folder_beam, bf)

        db_cut = _compute_dB_threshold_from_power(pixel_map, beam_threshold_map[bf])
        sel = 10 * np.log10(np.abs(pixel_map) + 1e-30) > db_cut
        beam_vals = pixel_map[sel].astype(np.float32)
        norm = beam_vals.sum()
        if norm != 0:
            beam_vals /= norm

        theta_orig = np.pi / 2 - dec
        vec_orig = np.stack(
            [
                np.sin(theta_orig) * np.cos(ra),
                np.sin(theta_orig) * np.sin(ra),
                np.cos(theta_orig),
            ],
            axis=-1,
        )[sel].astype(np.float32)

        beam_data[bf] = {
            "ra": ra,
            "dec": dec,
            "beam_vals": beam_vals,
            "sel": sel,
            "comp_indices": comp_indices,
            "n_sel": int(sel.sum()),
            "vec_orig": vec_orig,
        }
        print(f"  Beam {bf}: {sel.sum()} selected pixels")

    return beam_data


def apply_beam_clustering(beam_data, n_clusters, tail_fraction=None):
    """Apply weighted spherical k-means clustering to beam_data in-place.

    Called after prepare_beam_data() and, if used, after calibrate_beam_clustering().
    The pre-clustering beam_vals serve as pixel weights for both the k-means
    and the subsequent cache-array reduction.

    Args:
        beam_data (dict): Beam data from prepare_beam_data (exact, unclustered).
            Modified in-place. Cache arrays (vec_rolled, dtheta, dphi) are
            reduced from (N_psi, S, *) to (N_psi, K_out, *) if present.
        n_clusters (int): Max clusters for the tail (or all pixels in full mode).
        tail_fraction (float | None): Fraction of power to treat as tail.
            None → full mode (cluster all pixels).
    """
    _CACHE_KEYS = ("vec_rolled", "dtheta", "dphi")
    for bf, data in beam_data.items():
        bv_pre = data["beam_vals"]  # (S,) — needed as weights before overwrite
        vo_pre = data["vec_orig"]  # (S, 3)
        S = data["n_sel"]

        vec_out, bv_out, labels = cluster_beam_pixels(
            vo_pre,
            bv_pre,
            n_clusters=n_clusters,
            tail_fraction=tail_fraction,
        )
        K = len(bv_out)

        # Cluster cache arrays with pre-clustering weights before overwriting beam_vals
        cache_sub = {k: data[k] for k in _CACHE_KEYS if k in data}
        if cache_sub:
            print(f"    [{bf}] Clustering cache arrays …")
            clustered = cluster_cached_arrays(cache_sub, labels, bv_pre, K)
            for k, arr in clustered.items():
                data[k] = arr

        data["beam_vals"] = bv_out
        data["vec_orig"] = vec_out
        data["n_sel"] = K
        print(f"  [{bf}] Beam clustered: {S} → {K} pixels")


# ── TOD generation ────────────────────────────────────────────────────────────


def tod_exact_gen_batched(
    beam_data,
    day_index,
    mp,
    batch_size,
    process_name=None,
):
    """Generate TOD for a single observation day using batched processing.

    Opens the scan files as persistent memory-maps (avoiding repeated
    ``open``/``mmap`` syscalls per batch), then processes the day in
    ``ceil(n_samples / batch_size)`` batches. Each batch computes Rodrigues
    rotation vectors, calls :func:`~tod_core.beam_tod_batch` for every beam
    entry, and accumulates the results.

    Args:
        beam_data (dict): Pre-loaded beam data from :func:`prepare_beam_data`.
            Must include ``'mp_stacked'`` for the Numba gather path.
        day_index (int): Zero-based index of the observation day.
        mp (list[numpy.ndarray]): Sky map components ``[I, Q, U]``. Used on
            the fallback (non-stacked) path only.
        batch_size (int): Number of detector samples per processing batch. Use
            the value returned by :func:`~tod_calibrate._calibrate_n_processes`.
        process_name (str | None): Label for log messages (e.g. the
            ``multiprocessing.Process`` name). Defaults to ``None``.

    Returns:
        numpy.ndarray: TOD array of shape ``(3, n_samples)``, dtype
            ``float64``. Axis 0 is the Stokes component index ``[I, Q, U]``.
    """
    prefix = f"[{process_name}] " if process_name else ""
    nside = hp.get_nside(mp[0])

    # Open mmaps once for the whole day — avoids re-opening 3 files per batch,
    # which at batch_size=8 would otherwise dominate I/O overhead.
    theta_mmap, phi_mmap, psi_mmap = open_scan_day(folder_scan, day_index)
    n_samples = len(phi_mmap)

    first_bf = next(iter(beam_data))
    ra0, dec0 = beam_data[first_bf]["ra"], beam_data[first_bf]["dec"]

    batch_size = max(1, min(batch_size, n_samples))
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(
        prefix
        + f"Day {day_index} — {n_samples} samples, batch_size={batch_size}, "
        + f"n_batches={n_batches}"
    )

    tod_day = np.zeros((3, n_samples))
    start_time = time.time()

    for batch_idx in range(n_batches):
        bs = batch_idx * batch_size
        be = min(bs + batch_size, n_samples)

        # ETA
        if _should_print_batch(batch_idx, n_batches):
            elapsed = time.time() - start_time
            if batch_idx > 0:
                eta = elapsed / batch_idx * (n_batches - batch_idx)
                eta_str = _fmt_time(eta)
            else:
                eta_str = "..."
            print(
                prefix
                + f"Batch {batch_idx + 1}/{n_batches}  samples {bs}-{be - 1}  ETA {eta_str}"
            )

        theta_b = np.array(theta_mmap[bs:be], dtype=np.float32)
        phi_b = np.array(phi_mmap[bs:be], dtype=np.float32)
        psi_b = np.array(psi_mmap[bs:be], dtype=np.float32)
        rot_vecs, betas = precompute_rotation_vector_batch(ra0, dec0, phi_b, theta_b)
        psis_b = -betas + psi_b

        tod_batch = np.zeros((3, be - bs))
        for data in beam_data.values():
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
                tod_batch[comp] += vals

        tod_day[:, bs:be] = tod_batch

    total = time.time() - start_time
    print(
        prefix
        + f"Done — {n_samples} samples in {_fmt_time(total)} ({total / n_batches:.2f}s/batch)"
    )
    return tod_day


# ── Per-day worker (used by multiprocessing pool) ─────────────────────────────


def _process_day(day_index, batch_size, Nb):
    """
    Worker entry point.  beam_data and mp are *not* passed as arguments —
    they live in the module-level globals populated by _worker_init, so no
    pickling / copying of the large sky-map arrays occurs per task.
    """
    process_name = multiprocessing.current_process().name
    print(f"[{process_name}] Processing day {day_index + 1}/{Nb}")
    try:
        tod_day = tod_exact_gen_batched(
            _g_beam_data,
            day_index,
            _g_mp,
            batch_size,
            process_name=process_name,
        )
        output_file = f"{folder_tod_output}tod_day_{day_index}.npy"
        np.save(output_file, tod_day)
        print(f"[{process_name}] Saved {output_file}")
        return day_index, True, None
    except Exception as e:
        print(f"[{process_name}] Error on day {day_index}: {e}")
        return day_index, False, str(e)


# ── Calibration cache ─────────────────────────────────────────────────────────


def _save_calibration(n_processes, n_threads, batch_size):
    """Write calibration results back to the active config file."""
    import yaml

    with open(config.CONFIG_FILE) as f:
        raw = yaml.safe_load(f)
    raw["calibration_n_processes"] = int(n_processes)
    raw["calibration_numba_threads"] = int(n_threads)
    raw["calibration_batch_size"] = int(batch_size)
    raw["calibration_enabled"] = False
    with open(config.CONFIG_FILE, "w") as f:
        yaml.dump(
            raw,
            f,
            default_flow_style=False,
            allow_unicode=True,
            explicit_start=True,
            sort_keys=False,
        )
    print(
        f"Calibration saved to {config.CONFIG_FILE} "
        f"(n_processes={n_processes}, numba_threads={n_threads}, "
        f"batch_size={batch_size})"
    )


def _save_clustering_calibration(tail_fraction, n_clusters):
    """Write clustering calibration results to the active config YAML file."""
    import yaml

    with open(config.CONFIG_FILE) as f:
        raw = yaml.safe_load(f)
    raw["n_beam_clusters"] = int(n_clusters)
    raw["beam_cluster_tail_fraction"] = float(tail_fraction)
    raw["clustering_calibration_enabled"] = False  # disable after save
    with open(config.CONFIG_FILE, "w") as f:
        yaml.dump(
            raw,
            f,
            default_flow_style=False,
            allow_unicode=True,
            explicit_start=True,
            sort_keys=False,
        )
    print(
        f"Clustering calibration saved: tail_fraction={tail_fraction:.4f}, "
        f"n_clusters={n_clusters}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main(n_cpu_ceiling):
    t0 = time.time()
    Nb, _ = load_scan_information(folder_scan)

    start = max(start_day or 0, 0)
    end = min(end_day or Nb, Nb)
    days = range(start, end)

    os.makedirs(folder_tod_output, exist_ok=True)

    # Load the sky map here (inside main / under __name__ guard) so that
    # spawned worker processes — which re-import this module — never execute
    # this line themselves.
    print("Loading sky map...")
    MP = [
        m.astype(np.float32) for m in hp.read_map(config.path_to_map, field=(0, 1, 2))
    ]

    # ── Load exact beam data (clustering applied separately below) ─────────────
    print("Loading beam data...")
    beam_data = prepare_beam_data(beam_files)

    # ── Beam pixel clustering ──────────────────────────────────────────────────
    # Calibration: sweep (tail_fraction, K) grid on a probe batch to find the
    # best setting within the configured error tolerance.  Runs only when
    # clustering_calibration_enabled=True; disabled automatically after first run.
    if config.clustering_calibration_enabled:
        print("Running beam clustering calibration …")
        best_tf, best_K = calibrate_beam_clustering(
            beam_data,
            folder_scan=folder_scan,
            probe_day=start,
            mp=MP,
            error_threshold=config.clustering_error_threshold,
            interp_mode=interp_mode,
        )
        _save_clustering_calibration(best_tf, best_K)
        # Update in-memory config so clustering is applied this run too
        config.n_beam_clusters = best_K
        config.beam_cluster_tail_fraction = best_tf

    if config.n_beam_clusters is not None:
        print(
            f"Applying beam clustering "
            f"(tail_fraction={config.beam_cluster_tail_fraction}, "
            f"n_clusters={config.n_beam_clusters}) …"
        )
        apply_beam_clustering(
            beam_data,
            n_clusters=config.n_beam_clusters,
            tail_fraction=config.beam_cluster_tail_fraction,
        )

    # Stack sky-map components per beam entry into a contiguous (C, N) float32
    # array.  The Numba gather kernel requires this layout.
    for data in beam_data.values():
        data["mp_stacked"] = np.ascontiguousarray(
            np.stack([MP[c] for c in data["comp_indices"]])  # (C, N_hp)
        )

    use_cached = not config.calibration_enabled or (
        config.calibration_n_processes is not None
        and config.calibration_numba_threads is not None
        and config.calibration_batch_size is not None
    )
    if use_cached:
        ncpus = config.calibration_n_processes
        n_threads = config.calibration_numba_threads
        batch_size = config.calibration_batch_size
        print(
            f"Using cached calibration: n_processes={ncpus}, "
            f"numba_threads={n_threads}, batch_size={batch_size}"
        )
    else:
        print("Calibrating runtime (n_processes × numba_threads × batch_size)...")
        ncpus, n_threads, batch_size = calibrate_runtime(
            beam_data,
            folder_scan,
            probe_day=start,
            mp=MP,
            n_cpu_ceiling=n_cpu_ceiling,
            max_processes_user=config.n_processes,
            interp_mode=interp_mode,
        )
        _save_calibration(ncpus, n_threads, batch_size)
    print(
        f"Processing days {start}–{end - 1}  ({len(days)} days,  "
        f"{ncpus} workers × {n_threads} threads)"
    )

    if ncpus > 1:
        # ── Allocate shared memory ─────────────────────────────────────────
        # Pack all three MP components into one contiguous (3, npix) block so
        # that a single SharedMemory allocation covers everything.
        mp_arr = np.ascontiguousarray(np.stack(MP))  # (3, npix) float32
        shm_mp = SharedMemory(create=True, size=mp_arr.nbytes)
        np.ndarray(mp_arr.shape, dtype=mp_arr.dtype, buffer=shm_mp.buf)[:] = mp_arr
        mp_desc = {"name": shm_mp.name, "shape": mp_arr.shape, "dtype": mp_arr.dtype}

        # One block per unique beam file for mp_stacked (C, npix) float32.
        beam_shms = {}
        beam_shm_descs = {}
        for bf, data in beam_data.items():
            ms = data["mp_stacked"]
            shm = SharedMemory(create=True, size=ms.nbytes)
            np.ndarray(ms.shape, dtype=ms.dtype, buffer=shm.buf)[:] = ms
            beam_shms[bf] = shm
            beam_shm_descs[bf] = {
                "name": shm.name,
                "shape": ms.shape,
                "dtype": ms.dtype,
            }

        # Only small arrays remain in the pickle payload: beam_vals, vec_orig,
        # psi_grid, sel, ra, dec, comp_indices, n_sel.
        _SHARED_KEYS = {"mp_stacked"}
        beam_data_static = {
            bf: {k: v for k, v in data.items() if k not in _SHARED_KEYS}
            for bf, data in beam_data.items()
        }

        worker = partial(_process_day, batch_size=batch_size, Nb=Nb)
        try:
            with multiprocessing.Pool(
                processes=ncpus,
                initializer=_worker_init,
                initargs=(beam_data_static, mp_desc, beam_shm_descs, n_threads),
            ) as pool:
                results = pool.map(worker, days)
        finally:
            # Release shared memory only after all workers have finished.
            shm_mp.close()
            shm_mp.unlink()
            for shm in beam_shms.values():
                shm.close()
                shm.unlink()

        failed = [r for r in results if not r[1]]
        print(f"\nDone — {len(results) - len(failed)}/{len(results)} days OK")
        for day, _, err in failed:
            print(f"  Day {day} failed: {err}")
    else:
        if n_threads is not None and n_threads > 0:
            import numba

            numba.set_num_threads(int(n_threads))
        for day_index in days:
            tod_day = tod_exact_gen_batched(
                beam_data,
                day_index,
                MP,
                batch_size,
                process_name="main",
            )
            output_file = f"{folder_tod_output}/tod_day_{day_index}.npy"
            np.save(output_file, tod_day)

    print(f"\nTotal run time: {(time.time() - t0) / 60:.2f}m")


if __name__ == "__main__":
    multiprocessing.set_start_method(config.mp_start_method)
    main(_get_ncpus())
