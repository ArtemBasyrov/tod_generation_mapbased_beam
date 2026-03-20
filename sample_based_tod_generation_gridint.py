import os
import time
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from functools import partial

import numpy as np
import healpy as hp

import sample_based_tod_generation_config as config
from tod_io        import load_beam, load_scan_information, load_scan_data_batch, open_scan_day
from tod_core      import precompute_rotation_vector_batch, beam_tod_batch
from tod_calibrate import calibrate_batch_size
from tod_utils     import get_ncpus, _fmt_time, should_print_batch

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ── Config ────────────────────────────────────────────────────────────────────
folder_beam          = config.FOLDER_BEAM
folder_scan          = config.FOLDER_SCAN
folder_tod_output    = config.FOLDER_TOD_OUTPUT
beam_files           = [config.beam_file_I, config.beam_file_Q, config.beam_file_U]
start_day = config.start_day
end_day   = config.end_day

# ── Worker-global state (populated by _worker_init in each spawned process) ───
# MP is loaded in the parent, placed in shared memory, and attached here
# read-only by each worker — no copies, no re-loading the FITS file.
_g_mp          = None   # list of 3 float32 arrays (views into shared memory)
_g_beam_data   = None   # beam_data dict with mp_stacked from shared memory
_g_shm_handles = []     # SharedMemory handles kept alive for worker lifetime


# ── Pool initialiser ─────────────────────────────────────────────────────────

def _worker_init(beam_data_static, mp_desc, beam_shm_descs):
    """
    Called once in each spawned worker process.

    Attaches to the SharedMemory blocks created by the parent and builds
    zero-copy numpy views that the worker uses for the lifetime of the process.
    The SharedMemory handles are stored in _g_shm_handles so they are not
    garbage-collected (which would invalidate the buffer).

    Parameters
    ----------
    beam_data_static : dict  — beam_data without 'mp_stacked' (small arrays,
                               safe to pickle)
    mp_desc          : dict  — {'name', 'shape', 'dtype'} for the stacked MP block
    beam_shm_descs   : dict  — {beam_filename: {'name', 'shape', 'dtype'}}
    """
    global _g_mp, _g_beam_data, _g_shm_handles

    # Attach to the stacked (3, npix) sky-map block
    shm_mp = SharedMemory(name=mp_desc['name'])
    _g_shm_handles.append(shm_mp)                      # keep alive
    mp_full = np.ndarray(mp_desc['shape'], dtype=mp_desc['dtype'], buffer=shm_mp.buf)
    _g_mp = [mp_full[i] for i in range(mp_desc['shape'][0])]   # list of 3 views

    # Attach to each beam entry's mp_stacked block
    _g_beam_data = {}
    for bf, static in beam_data_static.items():
        desc = beam_shm_descs[bf]
        shm  = SharedMemory(name=desc['name'])
        _g_shm_handles.append(shm)
        ms = np.ndarray(desc['shape'], dtype=desc['dtype'], buffer=shm.buf)
        entry = dict(static)
        entry['mp_stacked'] = ms
        _g_beam_data[bf] = entry


# ── Beam preparation ──────────────────────────────────────────────────────────

def prepare_beam_data(beam_filenames):
    """Load and preprocess all unique beam files. Returns beam_data dict."""
    beam_groups = {}
    for i, bf in enumerate(beam_filenames):
        beam_groups.setdefault(bf, []).append(i)

    beam_data = {}
    for bf, comp_indices in beam_groups.items():
        ra, dec, pixel_map = load_beam(folder_beam, bf)

        sel       = (10 * np.log10(np.abs(pixel_map) + 1e-30) > -25)
        beam_vals = pixel_map[sel].astype(np.float32)
        norm      = beam_vals.sum()
        if norm != 0:
            beam_vals /= norm

        theta_orig = np.pi/2 - dec
        vec_orig   = np.stack([np.sin(theta_orig) * np.cos(ra),
                               np.sin(theta_orig) * np.sin(ra),
                               np.cos(theta_orig)], axis=-1)[sel].astype(np.float32)

        beam_data[bf] = {
            'ra': ra, 'dec': dec,
            'beam_vals': beam_vals,
            'sel': sel,
            'comp_indices': comp_indices,
            'n_sel': int(sel.sum()),
            'vec_orig': vec_orig,
        }
        print(f"  Beam {bf}: {sel.sum()} selected pixels")

    return beam_data


# ── TOD generation ────────────────────────────────────────────────────────────

def tod_exact_gen_batched(beam_data, day_index, mp, batch_size, process_name=None):
    """
    Generate TOD for one day using batched memory-mapped loading and
    bilinear HEALPix interpolation.

    Parameters
    ----------
    beam_data  : dict  — pre-loaded beam data from prepare_beam_data()
    batch_size : int   — pre-calibrated batch size, uniform across all days
    """
    prefix    = f"[{process_name}] " if process_name else ""
    nside     = hp.get_nside(mp[0])

    # Open mmaps once for the whole day — avoids re-opening 3 files per batch,
    # which at batch_size=8 would otherwise dominate I/O overhead.
    theta_mmap, phi_mmap, psi_mmap = open_scan_day(folder_scan, day_index)
    n_samples = len(phi_mmap)

    first_bf  = next(iter(beam_data))
    ra0, dec0 = beam_data[first_bf]['ra'], beam_data[first_bf]['dec']

    batch_size = max(1, min(batch_size, n_samples))
    n_batches  = (n_samples + batch_size - 1) // batch_size
    print(prefix + f"Day {day_index} — {n_samples} samples, batch_size={batch_size}, n_batches={n_batches}")

    tod_day    = np.zeros((3, n_samples))
    start_time = time.time()

    for batch_idx in range(n_batches):
        bs = batch_idx * batch_size
        be = min(bs + batch_size, n_samples)

        #ETA
        if should_print_batch(batch_idx, n_batches):
            elapsed = time.time() - start_time
            if batch_idx > 0:
                eta = elapsed / batch_idx * (n_batches - batch_idx)
                eta_str = _fmt_time(eta)
            else:
                eta_str = "..."
            print(prefix + f"Batch {batch_idx+1}/{n_batches}  samples {bs}-{be-1}  ETA {eta_str}")

        theta_b = np.array(theta_mmap[bs:be], dtype=np.float32)
        phi_b   = np.array(phi_mmap[bs:be],   dtype=np.float32)
        psi_b   = np.array(psi_mmap[bs:be],   dtype=np.float32)
        rot_vecs, betas        = precompute_rotation_vector_batch(ra0, dec0, phi_b, theta_b)
        psis_b                 = -betas + psi_b

        tod_batch = np.zeros((3, be - bs))
        for data in beam_data.values():
            contrib = beam_tod_batch(nside, mp, data, rot_vecs, phi_b, theta_b, psis_b)
            for comp, vals in contrib.items():
                tod_batch[comp] += vals

        tod_day[:, bs:be] = tod_batch

    total = time.time() - start_time
    print(prefix + f"Done — {n_samples} samples in {_fmt_time(total)} ({total/n_batches:.2f}s/batch)")
    return tod_day


# ── Per-day worker (used by multiprocessing pool) ─────────────────────────────

def process_day(day_index, batch_size, Nb):
    """
    Worker entry point.  beam_data and mp are *not* passed as arguments —
    they live in the module-level globals populated by _worker_init, so no
    pickling / copying of the large sky-map arrays occurs per task.
    """
    process_name = multiprocessing.current_process().name
    print(f"[{process_name}] Processing day {day_index+1}/{Nb}")
    try:
        tod_day     = tod_exact_gen_batched(_g_beam_data, day_index, _g_mp, batch_size, process_name=process_name)
        output_file = f'{folder_tod_output}tod_day_{day_index}.npy'
        np.save(output_file, tod_day)
        print(f"[{process_name}] Saved {output_file}")
        return day_index, True, None
    except Exception as e:
        print(f"[{process_name}] Error on day {day_index}: {e}")
        return day_index, False, str(e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(ncpus):
    t0 = time.time()
    Nb, _ = load_scan_information(folder_scan)

    start = max(start_day or 0, 0)
    end   = min(end_day   or Nb, Nb)
    days  = range(start, end)
    print(f"Processing days {start}–{end-1}  ({len(days)} days,  {ncpus} workers)")

    os.makedirs(folder_tod_output, exist_ok=True)

    # Load the sky map here (inside main / under __name__ guard) so that
    # spawned worker processes — which re-import this module — never execute
    # this line themselves.
    print("Loading sky map...")
    MP = [m.astype(np.float32) for m in hp.read_map(config.path_to_map, field=(0, 1, 2))]

    # Load beams and calibrate once — result is shared by all days and workers.
    print("Loading beam data...")
    beam_data = prepare_beam_data(beam_files)

    # Stack sky-map components per beam entry into a contiguous (C, N) float32
    # array.  The Numba gather kernel requires this layout.
    for data in beam_data.values():
        data['mp_stacked'] = np.ascontiguousarray(
            np.stack([MP[c] for c in data['comp_indices']])  # (C, N_hp)
        )

    print("Calibrating batch size...")
    batch_size, _ = calibrate_batch_size(
        beam_data, folder_scan, probe_day=start,
        mp=MP, n_processes=ncpus,
    )

    if ncpus > 1:
        # ── Allocate shared memory ─────────────────────────────────────────
        # Pack all three MP components into one contiguous (3, npix) block so
        # that a single SharedMemory allocation covers everything.
        mp_arr = np.ascontiguousarray(np.stack(MP))           # (3, npix) float32
        shm_mp = SharedMemory(create=True, size=mp_arr.nbytes)
        np.ndarray(mp_arr.shape, dtype=mp_arr.dtype, buffer=shm_mp.buf)[:] = mp_arr
        mp_desc = {'name': shm_mp.name, 'shape': mp_arr.shape, 'dtype': mp_arr.dtype}

        # One block per unique beam file (in practice often just one entry).
        beam_shms     = {}
        beam_shm_descs = {}
        for bf, data in beam_data.items():
            ms  = data['mp_stacked']                           # (C, npix) float32
            shm = SharedMemory(create=True, size=ms.nbytes)
            np.ndarray(ms.shape, dtype=ms.dtype, buffer=shm.buf)[:] = ms
            beam_shms[bf]      = shm
            beam_shm_descs[bf] = {'name': shm.name, 'shape': ms.shape, 'dtype': ms.dtype}

        # Small arrays (beam_vals, vec_orig, …) — safe to pickle per worker.
        beam_data_static = {
            bf: {k: v for k, v in data.items() if k != 'mp_stacked'}
            for bf, data in beam_data.items()
        }

        worker = partial(process_day, batch_size=batch_size, Nb=Nb)
        try:
            with multiprocessing.Pool(
                    processes=ncpus,
                    initializer=_worker_init,
                    initargs=(beam_data_static, mp_desc, beam_shm_descs)) as pool:
                results = pool.map(worker, days)
        finally:
            # Release shared memory only after all workers have finished.
            shm_mp.close(); shm_mp.unlink()
            for shm in beam_shms.values():
                shm.close(); shm.unlink()

        failed = [r for r in results if not r[1]]
        print(f"\nDone — {len(results)-len(failed)}/{len(results)} days OK")
        for day, _, err in failed:
            print(f"  Day {day} failed: {err}")
    else:
        for day_index in days:
            tod_day     = tod_exact_gen_batched(beam_data, day_index, MP, batch_size, process_name="main")
            output_file = f'{folder_tod_output}/tod_day_{day_index}.npy'
            np.save(output_file, tod_day)

    print(f"\nTotal run time: {(time.time()-t0)/60:.2f}m")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main(get_ncpus())
