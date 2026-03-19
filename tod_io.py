import os
import numpy as np
from pixell import enmap


def load_beam(folder_beam, filename):
    beam_map = enmap.read_map(folder_beam + filename)
    ra, dec = beam_map.posmap()
    ra  = np.array(ra)  - ra[100, 100]
    dec = np.array(dec) - dec[100, 100]
    pixel_map = np.array(beam_map[0])
    return ra, dec, pixel_map


def load_scan_information(folder):
    files = [f for f in os.listdir(folder) if f.startswith('psi')]
    max_index = max(int(f.split('_')[1].split('.')[0]) for f in files)
    nb_of_days = max_index + 1
    first_file = os.path.join(folder, files[0])
    fsamp = np.load(first_file).shape[0] / 86400
    return nb_of_days, fsamp


def open_scan_day(folder_scan, day_index):
    """
    Open the three scan-data files for one day as memory-maps and return them.

    The caller is responsible for holding the returned objects alive for as
    long as slices from them are needed.  Keeping the mmaps open across all
    batches avoids the repeated open/header-parse/mmap syscalls that
    load_scan_data_batch would otherwise incur on every batch call.

    Returns
    -------
    theta_mmap, phi_mmap, psi_mmap : numpy memmap arrays
    """
    theta_mmap = np.load(folder_scan + f'theta_{day_index}.npy', mmap_mode='r')
    phi_mmap   = np.load(folder_scan + f'phi_{day_index}.npy',   mmap_mode='r')
    psi_mmap   = np.load(folder_scan + f'psi_{day_index}.npy',   mmap_mode='r')
    return theta_mmap, phi_mmap, psi_mmap


def load_scan_data_batch(folder_scan, day_index, start_idx, end_idx):
    phi_mmap   = np.load(folder_scan + f'phi_{day_index}.npy',   mmap_mode='r')
    theta_mmap = np.load(folder_scan + f'theta_{day_index}.npy', mmap_mode='r')
    psi_mmap   = np.load(folder_scan + f'psi_{day_index}.npy',   mmap_mode='r')
    theta = np.array(theta_mmap[start_idx:end_idx], dtype=np.float32)
    phi   = np.array(phi_mmap[start_idx:end_idx],   dtype=np.float32)
    psi   = np.array(psi_mmap[start_idx:end_idx],   dtype=np.float32)
    return theta, phi, psi


def count_scan_samples(folder_scan, day_index):
    phi_mmap = np.load(folder_scan + f'phi_{day_index}.npy', mmap_mode='r')
    return len(phi_mmap)
