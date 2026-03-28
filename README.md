# TOD Generation from Beam Convolution

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://tod-generation-mapbased-beam.readthedocs.io/en/latest/index.html)

Sample-based Time-Ordered Data (TOD) generation for CMB experiments. Convolves
polarised sky maps (I, Q, U) with pixelated beam patterns over a boresight
scan trajectory, producing one TOD file per processing batch (days are used as
the default batching unit, but any convenient grouping can be used).

---

## Overview

The pipeline projects a HEALPix sky map through an instrumental beam for each
pointing sample in the scan. The core operation is:

```
tod[t] = Σ_s  beam_val[s] × skymap[ R(theta_t, phi_t, psi_t) · beam_pixel[s] ]
```

where `R` is a compound rotation that recenters the beam on the boresight
direction and applies the polarisation roll angle. Three execution paths are
available, trading accuracy for speed:

| Mode | Rotations at runtime | Speed |
|---|---|---|
| Flat-sky (cached) | 0 | Fastest |
| Single-Rodrigues (cached) | 1 | Fast |
| Full double-Rodrigues | 2 | Exact |

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy healpy pixell numba pyyaml psutil

# 2. Copy and edit the config
cp config.yaml config_local.yaml
$EDITOR config_local.yaml   # set FOLDER_SCAN, FOLDER_TOD_OUTPUT, path_to_map, etc.

# 3. (Optional) pre-compute beam rotation cache (~25% speed-up, but reduced accuracy)
python precompute_beam_cache.py --n_psi 720

# 4. Run the pipeline
python sample_based_tod_generation_gridint.py
```

The script auto-detects available CPUs and memory, calibrates the optimal batch
size and process count on first run, saves the result to `config_local.yaml`,
and skips calibration on subsequent runs.

---

## Configuration

All settings live in `config.yaml` (or `config_local.yaml`, which takes
precedence when present). Both files use YAML syntax.

### Paths

| Key | Type | Description |
|---|---|---|
| `FOLDER_SCAN` | `str` | Directory containing scan files (`theta_N.npy`, `phi_N.npy`, `psi_N.npy`). Must end with `/`. |
| `FOLDER_TOD_OUTPUT` | `str` | Output directory for TOD files (`tod_day_N.npy`). Created automatically if absent. |
| `path_to_map` | `str` | Path to the HEALPix sky map FITS file containing I, Q, U fields. |
| `FOLDER_BEAM` | `str` | Directory containing beam FITS files. |
| `beam_file_I` | `str` | Filename of the intensity (I) beam map inside `FOLDER_BEAM`. |
| `beam_file_Q` | `str` | Filename of the Q-polarisation beam map inside `FOLDER_BEAM`. |
| `beam_file_U` | `str` | Filename of the U-polarisation beam map inside `FOLDER_BEAM`. |

### Beam pixel selection

| Key | Type | Default | Description |
|---|---|---|---|
| `power_fraction_threshold_I` | `float` | `0.99` | Fraction of total beam power to retain for the I beam. Pixels below the implied dB cut are discarded, reducing computation while keeping 99 % of the signal. |
| `power_fraction_threshold_Q` | `float` | `0.99` | Same for the Q beam. |
| `power_fraction_threshold_U` | `float` | `0.99` | Same for the U beam. |

Increase toward `1.0` for higher fidelity (more beam pixels, slower). Decrease
toward `0.9` to aggressively prune faint sidelobes.

### Batch range

The pipeline uses the term *day* for the scan file index suffix, but this can
represent any batching unit you choose — an observation session, a CES, an
hour of data, etc.

| Key | Type | Default | Description |
|---|---|---|---|
| `start_day` | `int` | `0` | First batch index to process (inclusive). |
| `end_day` | `int` | total batches | Last batch index to process (exclusive). Set to `null` to process all batches. |

### Multiprocessing

| Key | Type | Default | Description |
|---|---|---|---|
| `n_processes` | `int` | — | Maximum worker processes on a local machine. On a cluster the scheduler allocation takes precedence and this value is used only as a cap. Required on local machines. |
| `max_memory_per_process` | `float` | — | Per-process memory budget in GB. Used as a fallback when `psutil` is unavailable. |

### Calibration cache

The first run measures sustained throughput at several batch sizes and process
counts, then writes the optimal values back into the active config file. Set
`calibration_skip: true` to reuse those values without re-measuring.

| Key | Type | Default | Description |
|---|---|---|---|
| `calibration_skip` | `bool` | `false` | Skip calibration and use cached values below. |
| `calibration_n_processes` | `int \| null` | `null` | Cached optimal process count (written automatically). |
| `calibration_batch_size` | `int \| null` | `null` | Cached optimal batch size (written automatically). |

### Beam rotation cache

Pre-computing the psi-roll rotation eliminates one of the two Rodrigues
rotations per sample at runtime. Enable by pointing `beam_cache_dir` at the
directory produced by `precompute_beam_cache.py`.

| Key | Type | Default | Description |
|---|---|---|---|
| `beam_cache_dir` | `str \| null` | `null` | Path to the cache directory. `null` disables caching (full double-Rodrigues path). |
| `beam_cache_n_psi` | `int` | `720` | Number of psi bins in the cache. **Must match** the `--n_psi` value used when generating the cache. |

### Beam interpolation

| Key | Type | Default | Description |
|---|---|---|---|
| `beam_interp_method` | `str` | `'bilinear'` | Interpolation strategy. See table below. |
| `beam_interp_sigma_deg` | `float \| null` | `null` | Gaussian kernel width in degrees (`'gaussian'` only). Defaults to one HEALPix pixel resolution. |
| `beam_interp_radius_deg` | `float \| null` | `null` | Neighbour search radius in degrees (`'gaussian'` only). Defaults to `3 × sigma`. |

**Interpolation methods:**

| Value | Description | Speed |
|---|---|---|
| `'nearest'` | Single nearest-pixel lookup. No blending between pixels. | Fastest |
| `'bilinear'` | 4-pixel bilinear HEALPix interpolation via a fused Numba kernel. Best balance of speed and accuracy for most beams. **This is the recommended method.** | Fast |
| `'gaussian'` | Isotropic Gaussian kernel over all pixels within `radius_deg`. Avoids grid-aligned interpolation artefacts; best for wide or asymmetric beams. | Slow |

### Example `config.yaml`

```yaml
---
  FOLDER_SCAN:       "/data/scan/"
  FOLDER_TOD_OUTPUT: "/data/tod/"
  path_to_map:       "/data/maps/cmb_IQU.fits"

  FOLDER_BEAM:  "/data/beams/"
  beam_file_I:  "beam_I.fits"
  beam_file_Q:  "beam_Q.fits"
  beam_file_U:  "beam_U.fits"
  power_fraction_threshold_I: 0.99
  power_fraction_threshold_Q: 0.99
  power_fraction_threshold_U: 0.99

  start_day: 0
  end_day: 366

  n_processes: 8
  max_memory_per_process: 2.0   # GB

  calibration_skip: false
  calibration_n_processes: null
  calibration_batch_size: null

  beam_cache_dir: "/data/beam_cache/"
  beam_cache_n_psi: 720

  beam_interp_method: 'bilinear'
  beam_interp_sigma_deg: null
  beam_interp_radius_deg: null
```

---

## Data Formats

### Inputs

#### Sky map (`path_to_map`)

A FITS file readable by `healpy.read_map`. Must contain three fields:

- Field 0: Stokes I (intensity)
- Field 1: Stokes Q (linear polarisation)
- Field 2: Stokes U (linear polarisation)

All three fields must share the same HEALPix `nside`. Values are loaded as
`float32`.

#### Beam files (`FOLDER_BEAM / beam_file_{I,Q,U}`)

[pixell / enmap](https://pixell.readthedocs.io/en/latest/usage.html#usagepage) FITS format (2D map). The map must be centred on the beam axis;
the pixel at index `(100, 100)` is taken as the beam centre. RA and Dec
coordinates are read from the map's WCS header and converted to offsets
relative to the beam centre. Values represent beam amplitude (linear, not dB).

#### Scan files (`FOLDER_SCAN`)

One triplet of `.npy` files per processing batch (referred to as a *day* in
the filenames, but this can represent any convenient grouping — an observation
session, a CES, an hour of data, etc.):

```
theta_{day_index}.npy   # boresight colatitude  [rad], float32 or float64
phi_{day_index}.npy     # boresight longitude   [rad], float32 or float64
psi_{day_index}.npy     # polarisation roll     [rad], float32 or float64
```

Each file is a 1-D array with one element per detector sample. The files are
opened as memory-maps, so only the batch currently being processed is resident
in RAM.

The total number of batches is inferred from the highest-indexed `psi_*.npy`
file in the scan folder. The sample rate is estimated as
`len(psi_0.npy) / 86400` (samples per second).

#### Beam rotation cache (`beam_cache_dir`)

Optional `.npz` files produced by `precompute_beam_cache.py`. Each file
stores:

| Array | Shape | dtype | Description |
|---|---|---|---|
| `psi_grid` | `(N_psi,)` | float32 | psi bin centres [rad] covering `[0, 2π)` |
| `vec_rolled` | `(N_psi, S, 3)` | float32 | Beam-pixel unit vectors after psi-roll for each bin |
| `beam_vals` | `(S,)` | float32 | Normalised beam weights for the selected `S` pixels |
| `beam_ctr` | `(3,)` | float32 | Beam-centre unit vector (always `[1, 0, 0]`) |
| `dtheta` | `(N_psi, S)` | float32 | Flat-sky colatitude offsets [rad] (present when `--flat_sky` used) |
| `dphi` | `(N_psi, S)` | float32 | Flat-sky phi offsets [rad] (raw; divide by `sin(theta_b)` at runtime) |

Cache filenames follow the pattern:
```
{beam_stem}_cache_npsi{N_psi}.npz
```

### Outputs

#### TOD files (`FOLDER_TOD_OUTPUT`)

One `.npy` file per processing batch:

```
tod_day_{day_index}.npy   # shape (3, n_samples), dtype float32
```

Axis 0 indexes the Stokes component: `[I, Q, U]`.
Axis 1 indexes the detector sample.

---

## Pre-computing the Beam Cache

The beam cache eliminates the psi-roll Rodrigues rotation at runtime, yielding
roughly a **25% speed-up**. For very narrow beams the flat-sky approximation
additionally eliminates the recentering rotation and the `vec2ang` call,
leaving only a table lookup and HEALPix interpolation.

> **Note:** Because the psi-roll is evaluated on a discrete grid rather than
> continuously, using the cache introduces a small interpolation error.
> **Not recommended for experiments requiring high precision.**

```bash
python precompute_beam_cache.py [--n_psi 720] [--config config.yaml]
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--n_psi` | `720` | Number of psi bins. 720 gives 0.5° resolution; increase for wider beams. |
| `--config` | `config.yaml` | Path to config file. |

After generation, set `beam_cache_dir` in your config to the output directory.

---

## Running on HPC / SLURM

The pipeline is SLURM-aware. When `SLURM_CPUS_PER_TASK` is set the calibration
uses the full node memory and all allocated CPUs. No special flags are needed:

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

python sample_based_tod_generation_gridint.py
```

The calibration step will find the process count that maximises
`throughput_per_process × n_processes` given the available memory, which on
memory-constrained nodes is often fewer than the total allocated CPUs.

---

## Repository Structure

```
.
├── sample_based_tod_generation_gridint.py  # Main entry point
├── tod_core.py                             # Core Numba JIT kernels
├── tod_io.py                               # File I/O (beam, scan, output)
├── tod_config.py                           # Config loader
├── tod_calibrate.py                        # Batch-size / process-count calibration
├── tod_utils.py                            # CPU/memory detection and utilities
├── numba_healpy.py                         # Numba re-implementation of HEALPix helpers
├── precompute_beam_cache.py                # One-time beam cache generation
├── config.yaml                             # Default configuration template
└── tests/                                  # pytest test suite
    ├── test_tod_core.py
    ├── test_numba_healpy.py
    ├── test_gaussian_interp.py
    ├── test_precompute_beam_cache.py
    ├── test_tod_calibrate.py
    ├── test_tod_utils.py
    └── run_all_tests.py
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `healpy` | HEALPix map I/O and pixel utilities |
| `numba` | JIT compilation of rotation and interpolation kernels |
| `pixell` | enmap beam file loading |
| `pyyaml` | Config file parsing |
| `psutil` | CPU/memory auto-detection (optional but recommended) |

Install with:

```bash
pip install numpy healpy pixell numba pyyaml psutil
```

---

## Tests

```bash
cd tests
python run_all_tests.py
# or
pytest tests/
```
