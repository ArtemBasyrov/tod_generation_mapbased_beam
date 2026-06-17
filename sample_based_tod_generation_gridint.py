import os
import time
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import healpy as hp

import tod_config as config
from tod_io import load_scan_information, open_scan_day
from tod_core import precompute_rotation_vector_batch, beam_tod_batch
from tod_calibrate import calibrate_runtime, calibrate_beam_clustering
from tod_utils import _get_ncpus, _fmt_time, _should_print_batch
from tod_runcontext import build_run_context
from tod_focalplane import (
    build_beam_specs,
    combine_detector_signal,
    detector_pointing_batch,
    load_detectors,
    tod_output_path,
)
from tod_pipeline_helpers import (
    prepare_beam_sets,
    apply_beam_clustering,
    merge_beam_entries,
    apply_hwp_modulation,
    resolve_spin2_skip_threshold,
    save_runtime_calibration,
    save_clustering_calibration,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Run-scoped parameters (folder paths, interp mode, precision, HWP, …) are
# carried in a RunContext (see tod_runcontext.py), built once in main() and
# passed to each worker at pool start-up.

# ── Worker-global state (populated by _worker_init in each spawned process) ───
# Beam data lives in shared memory; the run context (small scalars) is pickled
# to the worker once at init.
_g_ctx = None  # RunContext — run-scoped parameters
_g_beam_sets = None  # {beam_key: merged beam-data dict} with shared mp_stacked
_g_shm_handles = []  # SharedMemory handles kept alive for worker lifetime


def _worker_init(beam_sets_static, ctx, mp_shm_descs, n_threads):
    """
    Called once in each spawned worker process.

    Attaches to the SharedMemory blocks created by the parent and builds
    zero-copy numpy views that the worker uses for the lifetime of the process.
    The SharedMemory handles are stored in _g_shm_handles so they are not
    garbage-collected (which would invalidate the buffer).

    The sky-map block ``mp_stacked`` is shared per Stokes-component signature
    ``tuple(comp_indices)`` — detectors whose beams cover the same components
    attach the *same* block, so the sky map is stored once regardless of how
    many beam sets (detectors) reference it.

    Parameters
    ----------
    beam_sets_static : dict  — {beam_key: {merged_key: static entry}}; static
                               entries drop mp_stacked but keep the small arrays
                               (beam_vals, vec_orig, sel, ra, dec, comp_indices)
    ctx              : RunContext — run-scoped parameters (paths, interp mode,
                               nside, batch size, precision, HWP, …)
    mp_shm_descs     : dict  — {comp_signature: {'name', 'shape', 'dtype'}}
                               for the deduped mp_stacked sky-map blocks
    """
    global _g_ctx, _g_beam_sets, _g_shm_handles

    if n_threads is not None and n_threads > 0:
        import numba

        numba.set_num_threads(int(n_threads))

    _g_ctx = ctx

    mp_views = {}
    for sig, desc in mp_shm_descs.items():
        shm = SharedMemory(name=desc["name"])
        _g_shm_handles.append(shm)
        mp_views[sig] = np.ndarray(desc["shape"], dtype=desc["dtype"], buffer=shm.buf)

    _g_beam_sets = {}
    for beam_key, bd_static in beam_sets_static.items():
        beam_data = {}
        for mk, static in bd_static.items():
            entry = dict(static)
            entry["mp_stacked"] = mp_views[tuple(static["comp_indices"])]
            beam_data[mk] = entry
        _g_beam_sets[beam_key] = beam_data


def tod_exact_gen_batched(
    ctx,
    beam_data,
    day_index,
    process_name=None,
    q_det=None,
):
    """Generate TOD for a single observation day using batched processing.

    Opens the scan files as persistent memory-maps (avoiding repeated
    ``open``/``mmap`` syscalls per batch), then processes the day in
    ``ceil(n_samples / batch_size)`` batches. Each batch computes Rodrigues
    rotation vectors, calls :func:`~tod_core.beam_tod_batch` for every beam
    entry, and accumulates the results.

    When ``q_det`` is given, the boresight pointing batch is first transformed
    into the detector's frame via :func:`tod_focalplane.detector_pointing_batch`
    (float64, then cast to ``ctx.precision_dtype``) before entering the kernel.
    When ``q_det`` is ``None`` the boresight pointing is used verbatim, leaving
    single-detector output bit-identical to the legacy path.

    Args:
        ctx (RunContext): Run-scoped parameters (scan folder, interp mode,
            nside, batch size, precision, spin-2 skip threshold, HWP, …).
        beam_data (dict): One detector's merged beam set (from
            :func:`prepare_beam_sets` → :func:`merge_beam_entries`). Must
            include ``'mp_stacked'`` for the Numba gather path.
        day_index (int): Zero-based index of the observation day.
        process_name (str | None): Label for log messages (e.g. the
            ``multiprocessing.Process`` name). Defaults to ``None``.
        q_det (numpy.ndarray | None): Detector offset quaternion ``(4,)``, or
            ``None`` for the boresight detector (no pointing transform).

    Returns:
        numpy.ndarray: TOD array of shape ``(3, n_samples)``, dtype matching
            ``ctx.precision_dtype``. Axis 0 is the Stokes component
            index ``[I, Q, U]``.
    """
    prefix = f"[{process_name}] " if process_name else ""
    _dt = ctx.precision_dtype

    # Open mmaps once for the whole day — avoids re-opening 3 files per batch,
    # which at batch_size=8 would otherwise dominate I/O overhead.
    theta_mmap, phi_mmap, psi_mmap = open_scan_day(ctx.folder_scan, day_index)
    n_samples = len(phi_mmap)

    first_bf = next(iter(beam_data))
    ra0, dec0 = beam_data[first_bf]["ra"], beam_data[first_bf]["dec"]

    beam_center_idx = ctx.beam_center_idx

    batch_size = max(1, min(ctx.batch_size, n_samples))
    n_batches = (n_samples + batch_size - 1) // batch_size
    print(
        prefix
        + f"Day {day_index} — {n_samples} samples, batch_size={batch_size}, "
        + f"n_batches={n_batches}"
    )

    tod_day = np.zeros((3, n_samples), dtype=_dt)
    start_time = time.time()

    for batch_idx in range(n_batches):
        bs = batch_idx * batch_size
        be = min(bs + batch_size, n_samples)

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

        if q_det is None:
            theta_b = np.array(theta_mmap[bs:be], dtype=_dt)
            phi_b = np.array(phi_mmap[bs:be], dtype=_dt)
            psi_b = np.array(psi_mmap[bs:be], dtype=_dt)
        else:
            # Compose per-detector pointing in float64, then cast to the run
            # precision. Overhead is O(B) against the kernel's O(B·S).
            theta_d, phi_d, psi_d = detector_pointing_batch(
                np.asarray(theta_mmap[bs:be], dtype=np.float64),
                np.asarray(phi_mmap[bs:be], dtype=np.float64),
                np.asarray(psi_mmap[bs:be], dtype=np.float64),
                q_det,
            )
            theta_b = theta_d.astype(_dt)
            phi_b = phi_d.astype(_dt)
            psi_b = psi_d.astype(_dt)
        rot_vecs, betas = precompute_rotation_vector_batch(
            ra0, dec0, phi_b, theta_b, center_idx=beam_center_idx
        )
        psis_b = -betas + psi_b

        tod_batch = np.zeros((3, be - bs), dtype=_dt)
        for data in beam_data.values():
            contrib = beam_tod_batch(
                ctx.nside,
                None,
                data,
                rot_vecs,
                phi_b,
                theta_b,
                psis_b,
                interp_mode=ctx.interp_mode,
                z_skip_threshold=ctx.z_skip_threshold,
            )
            for comp, vals in contrib.items():
                tod_batch[comp] += vals

        if ctx.hwp_enabled:
            apply_hwp_modulation(
                tod_batch,
                day_index=day_index,
                sample_start=bs,
                fsamp=ctx.fsamp,
                f_hwp=ctx.hwp_freq_hz,
                phi0=ctx.hwp_phi0_rad,
            )

        tod_day[:, bs:be] = tod_batch

    total = time.time() - start_time
    print(
        prefix
        + f"Done — {n_samples} samples in {_fmt_time(total)} ({total / n_batches:.2f}s/batch)"
    )
    return tod_day


def _process_task(task):
    """
    Worker entry point for one ``(day_index, det_index)`` task.

    ``ctx`` and the beam sets are *not* passed as arguments — they live in the
    module-level globals populated by :func:`_worker_init`, so no pickling /
    copying of the large sky-map arrays (or re-pickling of the context) occurs
    per task.  The detector is looked up from ``ctx.detectors`` by index, and
    its beam set from ``_g_beam_sets`` by ``detector.beam_key``.

    Returns a 4-tuple ``(task, ok, err, payload)``. In furax-export mode the
    detector's scalar timestream ``(n_samples,)`` is returned as ``payload`` —
    the I/Q/U → scalar combination is done here so only one row (not three) is
    held over IPC and buffered by the main process until the day is whole.
    Otherwise the full ``(3, n_samples)`` TOD is written to a ``tod_day_*.npy``
    file and ``payload`` is ``None``.
    """
    day_index, det_index = task
    detector = _g_ctx.detectors[det_index]
    process_name = multiprocessing.current_process().name
    print(f"[{process_name}] Processing day {day_index}, detector {detector.name}")
    try:
        tod_day = tod_exact_gen_batched(
            _g_ctx,
            _g_beam_sets[detector.beam_key],
            day_index,
            process_name=process_name,
            q_det=detector.quat,
        )
        if _g_ctx.furax_export:
            # Combine to the detector's scalar signal here so the main process
            # buffers a single (n,) row per detector instead of a (3, n) TOD.
            theta_m, phi_m, psi_m = open_scan_day(_g_ctx.folder_scan, day_index)
            signal = combine_detector_signal(
                tod_day,
                np.asarray(theta_m, dtype=np.float64),
                np.asarray(phi_m, dtype=np.float64),
                np.asarray(psi_m, dtype=np.float64),
                detector,
            )
            return task, True, None, signal
        output_file = tod_output_path(_g_ctx.folder_tod_output, day_index, detector)
        np.save(output_file, tod_day)
        print(f"[{process_name}] Saved {output_file}")
        return task, True, None, None
    except Exception as e:
        print(
            f"[{process_name}] Error on day {day_index}, detector {detector.name}: {e}"
        )
        return task, False, str(e), None


def _load_converter():
    """Lazily import the furax export module (pulls in toast at import time).

    Imported on first use rather than at module load so the heavy toast import
    only happens when ``furax_export`` is enabled.
    """
    import tod_to_furax

    return tod_to_furax


def _expected_obs_paths(folder_out, day_index, n_splits):
    """Output HDF5 paths the converter would write for one day."""
    if n_splits == 1:
        names = [f"obs_day_{day_index}"]
    else:
        names = [f"obs_day_{day_index}_p{p}" for p in range(n_splits)]
    return [os.path.join(folder_out, f"{n}.h5") for n in names]


def _assemble_and_write_day(day_index, det_tods, ctx, fsamp, export):
    """Assemble a day's in-memory per-detector TODs and write one HDF5 obs.

    Called from ``main()`` once all of a day's detector tasks have returned
    their scalar ``(n_samples,)`` timestreams (already combined at each
    detector's polarization angle in the worker). The boresight pointing is read
    once here for the observation, the per-detector rows are stacked, and a
    single ``obs_day_{N}.h5`` is written into ``ctx.folder_tod_output`` using the
    pipeline precision. No ``.npy`` files are involved on this path.

    Args:
        day_index (int): Observation-day index whose detectors are all done.
        det_tods (dict[int, numpy.ndarray]): ``{det_index: (n,) scalar signal}``.
        ctx (RunContext): Run context (scan/output folders, detectors, HWP,
            precision).
        fsamp (float): Sample rate [Hz].
        export (dict): Resolved export parameters (``t0_unix``).
    """
    tod_to_furax = _load_converter()

    theta_m, phi_m, psi_m = open_scan_day(ctx.folder_scan, day_index)
    theta = np.asarray(theta_m, dtype=np.float64)
    phi = np.asarray(phi_m, dtype=np.float64)
    psi = np.asarray(psi_m, dtype=np.float64)

    detectors = ctx.detectors
    n_samples = theta.shape[0]
    signal = np.empty((len(detectors), n_samples), dtype=np.float64)
    for di in range(len(detectors)):
        signal[di] = det_tods[di]

    tod_to_furax.write_day_observation(
        day_index=day_index,
        signal=signal,
        theta=theta,
        phi=phi,
        psi=psi,
        detectors=list(detectors),
        folder_out=ctx.folder_tod_output,
        fsamp=fsamp,
        t0_unix=export["t0_unix"],
        hwp_enabled=ctx.hwp_enabled,
        f_hwp=ctx.hwp_freq_hz,
        phi0_hwp=ctx.hwp_phi0_rad,
        n_splits=1,
        signal_dtype=ctx.precision_dtype,
    )
    print(f"[export] Wrote obs_day_{day_index}.h5")


def main(n_cpu_ceiling):
    """Generate TODs for the day range configured in ``tod_config``.

    Loads the sky map and beams, optionally runs clustering and runtime
    calibration, then processes each day either in-process (``ncpus == 1``)
    or via a multiprocessing pool with the sky-map components stored in
    POSIX shared memory. Per-day TOD arrays are written to
    ``config.FOLDER_TOD_OUTPUT`` as ``tod_day_{i}.npy``.

    Args:
        n_cpu_ceiling (int): Hard upper bound on worker processes, typically
            from :func:`tod_utils._get_ncpus`.
    """
    t0 = time.time()
    Nb, fsamp = load_scan_information(config.FOLDER_SCAN)

    start = max(config.start_day or 0, 0)
    end = min(config.end_day or Nb, Nb)
    days = range(start, end)

    # Resolve the focal plane and the beam set each detector convolves with.
    # Detectors that share a beam file (e.g. an A/B polarization pair) share one
    # beam_key, so each unique beam set is loaded and clustered only once.
    detectors = load_detectors()
    beam_specs = build_beam_specs(detectors)

    os.makedirs(config.FOLDER_TOD_OUTPUT, exist_ok=True)

    # Load the sky map here (inside main / under __name__ guard) so that
    # spawned worker processes — which re-import this module — never execute
    # this line themselves.
    print(
        f"Loading sky map (precision={config.precision}, "
        f"fields={list(config.map_fields)})..."
    )
    _raw = hp.read_map(config.path_to_map, field=tuple(config.map_fields))
    if len(config.map_fields) == 1:
        _raw = (_raw,)
    MP = {
        c: np.asarray(m).astype(config.precision_dtype)
        for c, m in zip(config.map_fields, _raw)
    }

    print(f"Loading beam data for {len(beam_specs)} beam set(s)...")
    beam_sets = prepare_beam_sets(beam_specs)

    # Calibration and (for non-default focal planes) a representative throughput
    # probe use the first beam set; clustering parameters are global, so they
    # apply uniformly to every set.
    first_key = next(iter(beam_sets))

    if config.clustering_calibration_enabled:
        print("Running beam clustering calibration …")
        best_tf, best_K = calibrate_beam_clustering(
            beam_sets[first_key],
            folder_scan=config.FOLDER_SCAN,
            probe_day=start,
            mp=MP,
            error_threshold=config.clustering_error_threshold,
            interp_mode=config.beam_interp_method,
        )
        save_clustering_calibration(best_tf, best_K)
        # Update in-memory config so clustering is applied this run too
        config.n_beam_clusters = best_K
        config.beam_cluster_tail_fraction = best_tf

    # Clustering parameters resolve per beam set: a detector's own
    # n_beam_clusters / beam_cluster_tail_fraction override, or the global
    # config value (possibly just set by calibration) when it does not override.
    for key, beam_data in beam_sets.items():
        spec = beam_specs[key]
        n_clusters = (
            spec.n_clusters if spec.n_clusters is not None else config.n_beam_clusters
        )
        tail_fraction = (
            spec.tail_fraction
            if spec.tail_fraction is not None
            else config.beam_cluster_tail_fraction
        )
        if n_clusters is None:
            continue
        print(
            f"Applying beam clustering to set '{key}' "
            f"(tail_fraction={tail_fraction}, n_clusters={n_clusters}) …"
        )
        apply_beam_clustering(
            beam_data, n_clusters=n_clusters, tail_fraction=tail_fraction
        )

    # Spin-2 skip threshold is derived from beam geometry per source, before the
    # entries are merged (the threshold depends only on vec_orig / beam_vals).
    # Take the most conservative (largest beam radius) over every detector's
    # beam sources so the equatorial skip band is safe for all of them.
    all_sources = {
        f"{key}::{mk}": entry
        for key, bd in beam_sets.items()
        for mk, entry in bd.items()
    }
    z_skip_threshold = resolve_spin2_skip_threshold(
        all_sources, config.spin2_skip_tolerance
    )

    # Collapse each beam set's per-source entries into one unified
    # multi-component entry so every Stokes component is gathered in a single
    # kernel call and the Q/U spin-2 frame rotation is always applied together.
    beam_sets = {key: merge_beam_entries(bd) for key, bd in beam_sets.items()}

    # Stack sky-map components into a contiguous (C, N) array in the active
    # precision, deduped by Stokes-component signature: detectors whose beams
    # cover the same components share one block (and one shared-memory copy),
    # so the sky map is held once no matter how many beam sets reference it.
    mp_blocks = {}
    for beam_data in beam_sets.values():
        for data in beam_data.values():
            sig = tuple(data["comp_indices"])
            if sig not in mp_blocks:
                mp_blocks[sig] = np.ascontiguousarray(
                    np.stack([MP[c] for c in sig])  # (C, N_hp)
                )
            data["mp_stacked"] = mp_blocks[sig]

    use_cached = not config.calibration_enabled
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
        _cx, _cy = config.beam_center_x, config.beam_center_y
        ncpus, n_threads, batch_size = calibrate_runtime(
            beam_sets[first_key],
            config.FOLDER_SCAN,
            probe_day=start,
            mp=MP,
            n_cpu_ceiling=n_cpu_ceiling,
            max_processes_user=config.n_processes,
            interp_mode=config.beam_interp_method,
            center_idx=(_cx, _cy) if (_cx is not None and _cy is not None) else None,
            z_skip_threshold=z_skip_threshold,
        )
        save_runtime_calibration(ncpus, n_threads, batch_size)
    print(
        f"Processing days {start}–{end - 1}  ({len(days)} days,  "
        f"{ncpus} workers × {n_threads} threads)"
    )

    nside = hp.get_nside(next(iter(MP.values())))

    # Build the run context once; threaded through the worker pool (pickled
    # once at init) or used directly on the single-process path.
    ctx = build_run_context(
        nside=nside,
        batch_size=batch_size,
        z_skip_threshold=z_skip_threshold,
        fsamp=fsamp,
        detectors=detectors,
    )

    detectors = ctx.detectors
    n_det = len(detectors)
    print(f"Focal plane: {n_det} detector(s) — {[d.name for d in detectors]}")

    # Resolve the integrated-export time origin once (when enabled). One
    # observation is written into FOLDER_TOD_OUTPUT per day, assembled in memory
    # from the workers' TODs — no .npy is written on this path.
    export = None
    if config.furax_export:
        from datetime import datetime

        export = {
            "t0_unix": datetime.fromisoformat(config.furax_export_t0).timestamp(),
        }
        print(f"Integrated furax export → {ctx.folder_tod_output} (.h5 only)")

    # Task grid: one task per (day, detector). Resume skips a whole day once its
    # obs_day_{N}.h5 exists (export mode) or each detector whose tod_day_*.npy
    # exists (non-export mode).
    def _obs_complete(day):
        return export is not None and os.path.exists(
            _expected_obs_paths(ctx.folder_tod_output, day, 1)[0]
        )

    all_tasks = [(day, di) for day in days for di in range(n_det)]
    tasks = []
    n_skipped = 0
    for day in days:
        if _obs_complete(day):
            n_skipped += n_det
            continue
        for di in range(n_det):
            if export is None and os.path.exists(
                tod_output_path(ctx.folder_tod_output, day, detectors[di])
            ):
                n_skipped += 1
            else:
                tasks.append((day, di))
    if n_skipped:
        print(
            f"Resume: skipping {n_skipped}/{len(all_tasks)} task(s) with existing output"
        )

    # Per-day completion tracking for the in-memory export finalizer. Each
    # worker hands back its (3, n) TOD; the main process buffers them per day
    # and writes the observation once all of a day's detectors have arrived.
    day_remaining = {}
    day_failed = set()
    day_tods = {}  # day -> {det_index: (3, n) TOD}, held until the day is whole
    pending_set = set(tasks)
    for day in days:
        rem = sum(1 for di in range(n_det) if (day, di) in pending_set)
        if rem:
            day_remaining[day] = rem

    def _maybe_export(day):
        if export is None or day in day_failed:
            return
        if day_remaining.get(day) == 0 and day in day_tods:
            _assemble_and_write_day(day, day_tods.pop(day), ctx, fsamp, export)
            day_remaining.pop(day, None)

    if not tasks:
        print("Nothing to generate — all outputs already present.")
        print(f"\nTotal run time: {(time.time() - t0) / 60:.2f}m")
        return

    if ncpus > 1:
        # One shared-memory block per Stokes-component signature (deduped sky
        # map), reused across every beam set that covers the same components.
        beam_shms = {}
        mp_shm_descs = {}
        for sig, ms in mp_blocks.items():
            shm = SharedMemory(create=True, size=ms.nbytes)
            np.ndarray(ms.shape, dtype=ms.dtype, buffer=shm.buf)[:] = ms
            beam_shms[sig] = shm
            mp_shm_descs[sig] = {
                "name": shm.name,
                "shape": ms.shape,
                "dtype": ms.dtype,
            }

        # Only small arrays remain in the pickle payload: beam_vals, vec_orig,
        # sel, ra, dec, comp_indices, n_sel — for every beam set.
        _SHARED_KEYS = {"mp_stacked"}
        beam_sets_static = {
            beam_key: {
                mk: {k: v for k, v in data.items() if k not in _SHARED_KEYS}
                for mk, data in beam_data.items()
            }
            for beam_key, beam_data in beam_sets.items()
        }

        results = []
        try:
            with multiprocessing.Pool(
                processes=ncpus,
                initializer=_worker_init,
                initargs=(beam_sets_static, ctx, mp_shm_descs, n_threads),
            ) as pool:
                # Results land as they finish (better tail load-balancing).
                for res in pool.imap_unordered(_process_task, tasks, chunksize=1):
                    (rday, rdi), rok, rerr, payload = res
                    results.append(((rday, rdi), rok, rerr))
                    if rok:
                        if payload is not None:
                            day_tods.setdefault(rday, {})[rdi] = payload
                        if rday in day_remaining:
                            day_remaining[rday] -= 1
                            _maybe_export(rday)
                    else:
                        day_failed.add(rday)
        finally:
            # Release shared memory only after all workers have finished.
            for shm in beam_shms.values():
                shm.close()
                shm.unlink()

        failed = [r for r in results if not r[1]]
        print(f"\nDone — {len(results) - len(failed)}/{len(results)} task(s) OK")
        for (day, di), _, err in failed:
            print(f"  Day {day}, detector {detectors[di].name} failed: {err}")
    else:
        if n_threads is not None and n_threads > 0:
            import numba

            numba.set_num_threads(int(n_threads))
        for day_index, det_index in tasks:
            detector = detectors[det_index]
            tod_day = tod_exact_gen_batched(
                ctx,
                beam_sets[detector.beam_key],
                day_index,
                process_name="main",
                q_det=detector.quat,
            )
            if export is not None:
                day_tods.setdefault(day_index, {})[det_index] = tod_day
            else:
                output_file = tod_output_path(
                    ctx.folder_tod_output, day_index, detector
                )
                np.save(output_file, tod_day)
            if day_index in day_remaining:
                day_remaining[day_index] -= 1
                _maybe_export(day_index)

    print(f"\nTotal run time: {(time.time() - t0) / 60:.2f}m")


if __name__ == "__main__":
    multiprocessing.set_start_method(config.mp_start_method)
    main(_get_ncpus())
