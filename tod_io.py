import os
import numpy as np
from pixell import enmap


def load_beam(folder_beam, filename):
    """Load a beam map and return RA/Dec offsets and pixel amplitudes.

    Reads a pixell/enmap FITS beam map, extracts the WCS-based sky coordinates,
    and returns them as offsets relative to the beam centre pixel (the grid
    centre, computed as ``(H // 2, W // 2)`` from the map shape ``(H, W)``).

    Args:
        folder_beam (str): Path to the directory containing beam FITS files.
            Must end with a path separator.
        filename (str): Filename of the beam FITS file relative to
            ``folder_beam``.

    Returns:
        tuple:
            - **ra** (*numpy.ndarray*) – RA offsets from beam centre [rad],
              same shape as the beam map.
            - **dec** (*numpy.ndarray*) – Dec offsets from beam centre [rad],
              same shape as the beam map.
            - **pixel_map** (*numpy.ndarray*) – Beam amplitude values (linear,
              not dB), same shape as the beam map.
    """
    beam_map = enmap.read_map(folder_beam + filename)
    ra, dec = beam_map.posmap()
    ra = np.array(ra)
    dec = np.array(dec)
    center_idx = (ra.shape[0] // 2, ra.shape[1] // 2)
    ra = ra - ra[center_idx]
    dec = dec - dec[center_idx]
    pixel_map = np.array(beam_map[0])
    return ra, dec, pixel_map


def load_scan_information(folder):
    """Discover the number of observation days and the sampling rate.

    Scans ``folder`` for files named ``psi_N.npy`` and infers the total day
    count from the highest index found. The sample rate is estimated from the
    length of the first psi file divided by 86 400 (seconds per day).

    Args:
        folder (str): Path to the scan data directory.

    Returns:
        tuple:
            - **nb_of_days** (*int*) – Total number of observation days found.
            - **fsamp** (*float*) – Estimated sample rate [samples / second].
    """
    files = [f for f in os.listdir(folder) if f.startswith("psi")]
    if not files:
        raise FileNotFoundError(f"No 'psi_*.npy' files found in scan folder {folder!r}")
    try:
        max_index = max(int(f.split("_")[1].split(".")[0]) for f in files)
    except (IndexError, ValueError) as exc:
        raise ValueError(
            f"Could not parse scan-file index from filenames in {folder!r}. "
            f"Expected names like 'psi_<N>.npy', got: {files[:5]!r}"
        ) from exc
    nb_of_days = max_index + 1
    first_file = os.path.join(folder, files[0])
    fsamp = np.load(first_file).shape[0] / 86400
    return nb_of_days, fsamp


def open_scan_day(folder_scan, day_index):
    """Open the three scan-data files for one day as persistent memory-maps.

    The caller is responsible for holding the returned objects alive for as
    long as slices from them are needed.  If the returned memmap references
    are reassigned or garbage-collected, subsequent array slices will silently
    read stale or zeroed memory — keep at least one live reference per day
    for the entire processing window of that day.  Keeping the mmaps open
    across all batches avoids the repeated open/header-parse/mmap syscalls
    that :func:`_load_scan_data_batch` would otherwise incur on every batch call.

    Args:
        folder_scan (str): Path to the scan data directory. Must end with a
            path separator.
        day_index (int): Zero-based index of the observation day.

    Returns:
        tuple:
            - **theta_mmap** (*numpy.memmap*) – Boresight colatitude [rad].
            - **phi_mmap** (*numpy.memmap*) – Boresight longitude [rad].
            - **psi_mmap** (*numpy.memmap*) – Polarisation roll angle [rad].
    """
    theta_mmap = np.load(folder_scan + f"theta_{day_index}.npy", mmap_mode="r")
    phi_mmap = np.load(folder_scan + f"phi_{day_index}.npy", mmap_mode="r")
    psi_mmap = np.load(folder_scan + f"psi_{day_index}.npy", mmap_mode="r")
    return theta_mmap, phi_mmap, psi_mmap


def _load_scan_data_batch(folder_scan, day_index, start_idx, end_idx):
    """Load a contiguous batch of scan samples for one day into RAM.

    Opens the three scan files for ``day_index`` as memory-maps, slices the
    requested sample range, and returns them as contiguous ``float32`` arrays.
    Prefer :func:`open_scan_day` when processing many batches from the same
    day to avoid redundant file opens.

    Args:
        folder_scan (str): Path to the scan data directory. Must end with a
            path separator.
        day_index (int): Zero-based index of the observation day.
        start_idx (int): First sample index (inclusive).
        end_idx (int): Last sample index (exclusive).

    Returns:
        tuple:
            - **theta** (*numpy.ndarray*) – Boresight colatitude [rad],
              shape ``(end_idx - start_idx,)``, dtype ``float32``.
            - **phi** (*numpy.ndarray*) – Boresight longitude [rad],
              same shape.
            - **psi** (*numpy.ndarray*) – Polarisation roll angle [rad],
              same shape.
    """
    phi_mmap = np.load(folder_scan + f"phi_{day_index}.npy", mmap_mode="r")
    theta_mmap = np.load(folder_scan + f"theta_{day_index}.npy", mmap_mode="r")
    psi_mmap = np.load(folder_scan + f"psi_{day_index}.npy", mmap_mode="r")
    theta = np.array(theta_mmap[start_idx:end_idx], dtype=np.float32)
    phi = np.array(phi_mmap[start_idx:end_idx], dtype=np.float32)
    psi = np.array(psi_mmap[start_idx:end_idx], dtype=np.float32)
    return theta, phi, psi


def _count_scan_samples(folder_scan, day_index):
    """Return the total number of detector samples for one observation day.

    Args:
        folder_scan (str): Path to the scan data directory. Must end with a
            path separator.
        day_index (int): Zero-based index of the observation day.

    Returns:
        int: Number of samples in ``phi_{day_index}.npy``.
    """
    phi_mmap = np.load(folder_scan + f"phi_{day_index}.npy", mmap_mode="r")
    return len(phi_mmap)
