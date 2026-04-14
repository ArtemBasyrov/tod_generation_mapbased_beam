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
direction and applies the polarisation roll angle. 

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy healpy pixell numba pyyaml psutil

# 2. Copy and edit the config
cp config.yaml config_local.yaml
$EDITOR config_local.yaml   # set FOLDER_SCAN, FOLDER_TOD_OUTPUT, path_to_map, etc.

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
counts, then writes the optimal values back into the active config file.
`calibration_enabled` is automatically reset to `false` after calibration
completes so subsequent runs reuse the cached values without re-measuring.

| Key | Type | Default | Description |
|---|---|---|---|
| `calibration_enabled` | `bool` | `true` | Run calibration on this invocation. Automatically set to `false` after calibration completes. |
| `calibration_n_processes` | `int \| null` | `null` | Cached optimal process count (written automatically). |
| `calibration_batch_size` | `int \| null` | `null` | Cached optimal batch size (written automatically). |


### Beam interpolation

| Key | Type | Default | Description |
|---|---|---|---|
| `beam_interp_method` | `str` | `'bilinear'` | Interpolation strategy. Available values depend on the active git branch (see table below). |
| `beam_interp_sigma_deg` | `float \| null` | `null` | **`gaussian` branch only.** Gaussian kernel width in degrees. Defaults to one HEALPix pixel resolution. |
| `beam_interp_radius_deg` | `float \| null` | `null` | **`gaussian` branch only.** Neighbour search radius in degrees. Defaults to `3 × sigma`. |

**Interpolation methods:**

| Value | Branch | Description | Speed |
|---|---|---|---|
| `'nearest'` | `main` | Single nearest-pixel lookup. No blending between pixels. Rotationally unstable — not recommended for polarisation analysis. | Fastest |
| `'bilinear'` | `main` | 4-pixel bilinear HEALPix interpolation via a fused Numba kernel. Best balance of speed and accuracy. **Recommended default.** | Fast |
| `'bicubic'` | `bicubic` | Keys/Catmull-Rom kernel via gnomonic projection (~30–50 pixels). ~10× more rotationally stable than bilinear. | Slower |
| `'gaussian'` | `gaussian` | Isotropic Gaussian kernel over all pixels within `radius_deg`. Avoids grid artefacts; requires `beam_interp_sigma_deg`. | Slow |

### Beam pixel clustering

Spatial k-means clustering on the unit sphere reduces the number of effective
beam pixels at runtime. Only the low-power *tail* of the beam is clustered;
the bright main-lobe pixels are kept pixel-exact. For a typical 30′ Gaussian
beam with 3 % tail power this gives a 3–5× speed-up in TOD generation.

#### Cluster calibration

The calibration sweeps a `(tail_fraction × n_clusters)` grid and, for each
pair, clusters the beam pixels and recomputes the beam transfer function B_ℓ
from the clustered geometry. It selects the pair that maximises the pixel-count
speedup while keeping the B_ℓ divergence from the reference beam below
`clustering_error_threshold`. No scan data or TOD generation is required.

**Quickstart:**

1. Set `clustering_calibration_enabled: true` and run the pipeline once. The
   calibration writes the optimal `(tail_fraction, n_clusters)` pair back to
   the config and resets the flag automatically.
2. On all subsequent runs the saved values are used directly.

**Manual override:** set `clustering_calibration_enabled: false` and fill in
`n_beam_clusters` and `beam_cluster_tail_fraction` by hand.

**Disable entirely:** set `n_beam_clusters: null`.

| Key | Type | Default | Description |
|---|---|---|---|
| `n_beam_clusters` | `int \| null` | `null` | Maximum clusters for the tail. `null` disables clustering entirely. Written automatically by calibration. |
| `beam_cluster_tail_fraction` | `float \| null` | `null` | Fraction of total beam power treated as the "tail" to be clustered. The remaining `(1 − fraction)` of power pixels are kept pixel-exact. Written automatically by calibration. |
| `clustering_calibration_enabled` | `bool` | `false` | Run the clustering calibration sweep on this invocation. Automatically reset to `false` after completion. |
| `clustering_error_threshold` | `float` | `1.0e-5` | Maximum tolerated B_ℓ divergence (see below). The calibration selects the fastest pair that stays within this bound. |

#### Clustering error metric: B_ℓ divergence

The quality of a given `(tail_fraction, n_clusters)` pair is measured by the
**relative RMS divergence in the beam transfer function B_ℓ**:

```
ε = RMS( B_ℓ^{clustered} − B_ℓ^{ref} ) / RMS( B_ℓ^{ref} )
```

where B_ℓ^{ref} is computed from the full unclustered beam pixel set
(power_cut = 1.0) and B_ℓ^{clustered} is computed from the centroid pixels
produced by that pair. Both curves are evaluated at multipoles up to
`bell_lmax` (default: `2 × nside` of the sky map).

Using B_ℓ divergence as the threshold metric has two advantages over TOD-based
error measurement:

- **No scan data needed.** The metric is computed purely from beam geometry,
  so the calibration sweep is fast.
- **Direct beam fidelity.** A clustering that reproduces B_ℓ faithfully will
  also reproduce the TOD accurately, because B_ℓ controls how the beam couples
  to each angular scale of the sky.

> **Note:** Clustering is applied only to the TOD-generation path. B_ℓ
> computation itself must always use the full unclustered beam pixel set, since
> the Legendre polynomial oscillations that define B_ℓ are destroyed by pixel
> merging.

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

  calibration_enabled: true
  calibration_n_processes: null
  calibration_batch_size: null

  beam_interp_method: 'bilinear'   # or 'nearest'; use 'bicubic'/'gaussian' on those branches

  n_beam_clusters: null
  beam_cluster_tail_fraction: null
  clustering_calibration_enabled: false
  clustering_error_threshold: 1.0e-5
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
for a map of shape `(H, W)` the pixel at index `(H // 2, W // 2)` is taken as
the beam centre. RA and Dec coordinates are read from the map's WCS header and
converted to offsets relative to that centre pixel. Values represent beam
amplitude (linear, not dB).

Normalisation of the beam file is **not required**. The pipeline selects pixels
that together carry a fraction `power_fraction_threshold` of the total beam
power and re-normalises those weights to sum to one, so the absolute scale of
the beam file does not affect the signal amplitude in the output TOD.

See the [beam creation example](https://tod-generation-mapbased-beam.readthedocs.io/en/latest/beam_creation.html)
in the documentation for a step-by-step guide to generating a synthetic beam
file with pixell.

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

### Outputs

#### TOD files (`FOLDER_TOD_OUTPUT`)

One `.npy` file per processing batch:

```
tod_day_{day_index}.npy   # shape (3, n_samples), dtype float32
```

Axis 0 indexes the Stokes component: `[I, Q, U]`.
Axis 1 indexes the detector sample.

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

## Branch Structure

Experimental interpolation methods live on dedicated branches to keep `main`
clean. Switch branches to enable a different interpolation kernel:

| Branch | Extra module | Adds |
|--------|-------------|------|
| `main` | — | `nearest` + `bilinear` (production) |
| `gaussian` | `tod_gaussian.py` | `gaussian` kernel |
| `bicubic` | `tod_bicubic.py` | `bicubic` kernel |

```bash
git checkout gaussian   # enables beam_interp_method: gaussian
git checkout bicubic    # enables beam_interp_method: bicubic
git checkout main       # production default
```

Passing an unsupported `beam_interp_method` on any branch raises a `ValueError`
pointing to the correct branch.

## Repository Structure

```
.
├── sample_based_tod_generation_gridint.py  # Main entry point
├── tod_core.py                             # Core Numba JIT kernels + interpolation dispatch
├── tod_rotations.py                        # Rodrigues rotations kernels
├── tod_gaussian.py                         # [gaussian branch] Gaussian kernel
├── tod_bicubic.py                          # [bicubic branch]  Keys/Catmull-Rom kernel
├── tod_bilinear.py                         # Bilinear interpolation
├── tod_nearest.py                          # Nearest pixel interpolation
├── tod_io.py                               # File I/O (beam, scan, output)
├── tod_config.py                           # Config loader
├── tod_calibrate.py                        # Batch-size / process-count / clustering calibration
├── tod_utils.py                            # CPU/memory detection and utilities
├── numba_healpy.py                         # Numba re-implementation of HEALPix helpers
├── beam_cluster.py                         # Spherical k-means beam-pixel clustering
├── config.yaml                             # Default configuration template
└── tests/                                  # pytest test suite
    ├── test_tod_core.py
    ├── test_numba_healpy.py
    ├── test_tod_rotations.py
    ├── test_gaussian_interp.py             # [gaussian branch only]
    ├── test_bicubic_interp.py              # [bicubic branch only]
    ├── test_tod_bilinear.py
    ├── test_tod_calibrate.py
    ├── test_beam_cluster.py
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
