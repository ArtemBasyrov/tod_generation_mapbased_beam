import os
import time
import multiprocessing
from functools import partial

import numpy as np
import healpy as hp

import sample_based_tod_generation_config as config
from tod_io        import load_beam, load_scan_information, load_scan_data_batch, open_scan_day
from tod_core      import precompute_rotation_vector_batch, beam_tod_batch
from tod_calibrate import calibrate_batch_size
from tod_utils     import get_ncpus, _fmt_time, should_print_batch

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# ── Config ────────────────────────────────────────────────────────────────────
folder_beam          = config.FOLDER_BEAM
folder_scan          = config.FOLDER_SCAN
folder_tod_output    = config.FOLDER_TOD_OUTPUT
beam_files           = [config.beam_file_I, config.beam_file_Q, config.beam_file_U]
start_day = config.start_day
end_day   = config.end_day

MP = [m.astype(np.float32) for m in hp.read_map(config.path_to_map, field=(0, 1, 2))]


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

def process_day(day_index, beam_data, mp, batch_size, Nb):
    process_name = multiprocessing.current_process().name
    print(f"[{process_name}] Processing day {day_index+1}/{Nb}")
    try:
        tod_day     = tod_exact_gen_batched(beam_data, day_index, mp, batch_size, process_name=process_name)
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

    # Load beams and calibrate once — result is shared by all days and workers.
    print("Loading beam data...")
    beam_data = prepare_beam_data(beam_files)

    # Stack sky-map components per beam entry into a contiguous (C, N) float32
    # array.  The Numba gather kernel requires this layout.  With fork COW the
    # array is shared across all worker processes at no extra RAM cost.
    for data in beam_data.values():
        data['mp_stacked'] = np.ascontiguousarray(
            np.stack([MP[c] for c in data['comp_indices']])  # (C, N)
        )

    print("Calibrating batch size...")
    batch_size, _ = calibrate_batch_size(
        beam_data, folder_scan, probe_day=start,
        mp=MP, n_processes=ncpus,
    )

    if ncpus > 1:
        worker = partial(process_day, beam_data=beam_data, mp=MP, batch_size=batch_size, Nb=Nb)
        with multiprocessing.Pool(processes=ncpus) as pool:
            results = pool.map(worker, days)
            failed  = [r for r in results if not r[1]]
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
    multiprocessing.set_start_method('fork')
    main(get_ncpus())
