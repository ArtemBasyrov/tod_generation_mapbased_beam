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
# 1. Install dependencies (core + furax HDF5 export)
pip install numpy healpy pixell numba pyyaml psutil   # core TOD generation
pip install toast astropy h5py                         # furax HDF5 export (on by default)

# 2. Copy and edit the config
cp config.yaml config_local.yaml
$EDITOR config_local.yaml   # set FOLDER_SCAN, FOLDER_TOD_OUTPUT, path_to_map, etc.

# 3. Run the pipeline
python sample_based_tod_generation_gridint.py
```

The script auto-detects available CPUs and memory, calibrates the optimal batch
size and process count on first run, saves the result to `config_local.yaml`,
and skips calibration on subsequent runs.

By default the pipeline writes one furax-compatible TOAST HDF5 observation per
day (`obs_day_N.h5`). Set `furax_export: false` to write raw NumPy
`tod_day_N.npy` files instead. If the `toast` stack is not installed, use
`furax_export: false`.

---

## Configuration

All settings live in `config.yaml` (or `config_local.yaml`, which takes
precedence when present). Both files use YAML syntax.

### Paths

| Key | Type | Description |
|---|---|---|
| `FOLDER_SCAN` | `str` | Directory containing scan files (`theta_N.npy`, `phi_N.npy`, `psi_N.npy`). Must end with `/`. |
| `FOLDER_TOD_OUTPUT` | `str` | Output directory for TOD files (`obs_day_N.h5` by default, or `tod_day_N.npy` when `furax_export: false`). Created automatically if absent. |
| `path_to_map` | `str` | Path to the HEALPix sky map FITS file containing I, Q, U fields. |
| `FOLDER_BEAM` | `str` | Directory containing beam FITS files. |
| `beam_file_I` | `str` | Filename of the intensity (I) beam map inside `FOLDER_BEAM`. |
| `beam_file_Q` | `str` | Filename of the Q-polarisation beam map inside `FOLDER_BEAM`. |
| `beam_file_U` | `str` | Filename of the U-polarisation beam map inside `FOLDER_BEAM`. |

Only the `beam_file_*` / `power_fraction_threshold_*` entries whose Stokes index
appears in `map_fields` are required (see below); entries for inactive
components may be omitted or set to `null`.

### Stokes component selection

| Key | Type | Default | Description |
|---|---|---|---|
| `map_fields` | `list[int]` | `[0, 1, 2]` | Which Stokes components to read from the input FITS map. A non-empty subset of `[0, 1, 2]` = `[T, Q, U]`. Use `[0]` for a temperature-only map. |

The spin-2 Q/U frame rotation runs only when **both** Q (`1`) and U (`2`) are
active. Any other subset (`[0]`, `[1]`, `[0, 1]`, …) takes the scalar gather
path. The output TOD shape is always `(3, n_samples)`; rows for inactive
components are written as zeros.

### Beam centre

| Key | Type | Default | Description |
|---|---|---|---|
| `beam_center_x` | `int \| null` | `null` | Row index of the beam-centre pixel in the beam array. `null` → `H // 2`. |
| `beam_center_y` | `int \| null` | `null` | Column index of the beam-centre pixel. `null` → `W // 2`. |

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
| `mp_start_method` | `str` | `'spawn'` | Multiprocessing start method. `'spawn'` is safe everywhere; `'fork'` is faster on Linux (avoids re-running Numba JIT in each worker) but may deadlock on macOS. |

### Calibration cache

The first run measures sustained throughput at several batch sizes and process
counts, then writes the optimal values back into the active config file.
`calibration_enabled` is automatically reset to `false` after calibration
completes so subsequent runs reuse the cached values without re-measuring.

| Key | Type | Default | Description |
|---|---|---|---|
| `calibration_enabled` | `bool` | `true` | Run calibration on this invocation. Automatically set to `false` after calibration completes. |
| `calibration_n_processes` | `int \| null` | `null` | Cached optimal process count (written automatically). |
| `calibration_numba_threads` | `int \| null` | `null` | Cached optimal Numba `prange` thread count per worker (written automatically). |
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
| `'nearest'` | `main` | Single nearest-pixel lookup. No blending between pixels. Exhibits discrete boundary-jump artefacts — not recommended for polarisation analysis. | Fastest |
| `'bilinear'` | `main` | 4-pixel bilinear HEALPix interpolation via a fused Numba kernel. Best balance of speed and accuracy. **Recommended default.** | Fast |
| `'bicubic'` | `bicubic` | Keys/Catmull-Rom kernel via gnomonic projection (~30–50 pixels). Regular-grid method on an irregular pixelization — not science-ready. | Slower |
| `'gaussian'` | `gaussian` | Isotropic Gaussian kernel over all pixels within `radius_deg`. Avoids grid artefacts; requires `beam_interp_sigma_deg`. | Slow |
| `'totalconvolve'` | `totalconvolve` | ducc0 NUFFT synthesis (`synthesis_general`, spin-0 for T, spin-2 for Q/U). Accuracy set by `totalconvolve_epsilon`; flat noise floor eliminated. Requires `pip install ducc0`. | 5–15× slower |

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

### Spin-2 skip optimisation

The spin-2 Q/U frame correction is negligible near the equator and grows toward
the poles. Skipping it inside an equatorial band saves work at a bounded cost.

| Key | Type | Default | Description |
|---|---|---|---|
| `spin2_skip_tolerance` | `float \| null` | `null` | Maximum per-sample fractional Q/U error tolerated by skipping the correction in the equatorial band. `null` or `0` always applies the correction. Recommended when used: `0.01` (1 %). |

### Working precision

| Key | Type | Default | Description |
|---|---|---|---|
| `precision` | `str` | `'float32'` | Working precision for the float32-side of the pipeline (sky map, pointings, beam values, Rodrigues rotation, saved TOD). `'float64'` is a validation knob — slower, more memory, isolates whether high-ℓ residuals are precision-limited. Float64 surfaces (bilinear weights, spin-2 cache, accumulators, B_ℓ) are unchanged either way. |

### Half-wave plate (HWP)

When enabled, the generator rotates the per-sample `(Q, U)` output by
`4·φ_HWP(t)`, with `φ_HWP(t) = 2π·f_HWP·t + φ₀` and `t` continuous across days.
The rotation is applied **after** beam convolution and does not alter the beam
shape or the Rodrigues rotation.

| Key | Type | Default | Description |
|---|---|---|---|
| `hwp_enabled` | `bool` | `false` | Enable HWP modulation of the output Q/U. |
| `hwp_rotation_frequency_hz` | `float` | `0.0` | Physical HWP rotation rate `f_HWP` [Hz]. |
| `hwp_initial_phase_rad` | `float` | `0.0` | Phase `φ₀` at `t = 0` (start of day 0) [rad]. |

### Focal-plane detectors

By default the boresight is a single implicit detector. Define a `detectors`
list to simulate a focal plane: each detector is offset from the boresight in
the TOAST xi-eta-gamma convention (degrees) and its per-detector pointing is
composed on the fly via quaternions that mirror `furax.math.quaternion` exactly,
so the generator and furax compose identical pointing.

| Key | Type | Default | Description |
|---|---|---|---|
| `detectors` | `list \| null` | `null` | Focal-plane detector list. Each entry is `{name, xi_deg, eta_deg, gamma_deg}`. `gamma_deg` is the polarisation-angle offset (an A/B pair shares xi/eta and differs by `gamma_deg = 90`). `null` → single boresight detector. |
| `detector_subset` | `list \| null` | `null` | Run only part of the focal plane (detector names or 0-based indices). Used to shard a focal plane across HPC nodes. `null` → all detectors. |

**Per-detector beams.** A detector entry may additionally carry
`beam_file_{I,Q,U}`, `power_fraction_threshold_{I,Q,U}`, `n_beam_clusters`
and/or `beam_cluster_tail_fraction`. Any omitted key falls back to the global
value (an omitted clustering key inherits the global / calibrated clustering).
Detectors whose resolved beam files **and** clustering coincide share one beam
set in memory, so an A/B pair on the same beams is loaded and clustered once.
This is how genuinely asymmetric per-detector beams are run.

```yaml
  detectors:
    - {name: det_000A, xi_deg: 0.0, eta_deg: 0.0, gamma_deg: 0.0}
    - {name: det_000B, xi_deg: 0.0, eta_deg: 0.0, gamma_deg: 90.0}
    - {name: det_001A, xi_deg: 0.3, eta_deg: 0.1, gamma_deg: 0.0,
       beam_file_I: beam_001_I.fits, beam_file_Q: beam_001_Q.fits,
       beam_file_U: beam_001_U.fits,
       n_beam_clusters: 100, beam_cluster_tail_fraction: 0.03}
```

### furax HDF5 export

By default the generator writes one furax-compatible TOAST HDF5 observation per
day instead of `.npy` files. Each observation holds a `(n_det, n_samples)`
detdata block, assembled in memory from the workers' TODs. The standalone
`tod_to_furax.py` (project root) re-exports existing `.npy` files and keeps its
own `--output` / `--precision` / `--split-per-day` flags.

| Key | Type | Default | Description |
|---|---|---|---|
| `furax_export` | `bool` | `true` | `true` → write `obs_day_N.h5` per day (no `.npy`). `false` → write raw per-detector `tod_day_N.npy`. Requires the `toast` stack (imported lazily). |
| `furax_export_t0` | `str` | `'2030-01-01T00:00:00+00:00'` | UTC time of sample 0 of day 0 (timestream time origin). |

### Example `config.yaml`

```yaml
---
  FOLDER_SCAN:       "/data/scan/"
  FOLDER_TOD_OUTPUT: "/data/tod/"
  path_to_map:       "/data/maps/cmb_IQU.fits"
  map_fields: [0, 1, 2]

  FOLDER_BEAM:  "/data/beams/"
  beam_file_I:  "beam_I.fits"
  beam_file_Q:  "beam_Q.fits"
  beam_file_U:  "beam_U.fits"
  power_fraction_threshold_I: 0.99
  power_fraction_threshold_Q: 0.99
  power_fraction_threshold_U: 0.99
  beam_center_x: null
  beam_center_y: null

  start_day: 0
  end_day: 366

  n_processes: 8
  max_memory_per_process: 2.0   # GB
  mp_start_method: 'spawn'

  calibration_enabled: true
  calibration_n_processes: null
  calibration_numba_threads: null
  calibration_batch_size: null

  beam_interp_method: 'bilinear'   # or 'nearest'; use 'bicubic'/'gaussian'/'totalconvolve' on those branches
  spin2_skip_tolerance: null
  precision: 'float32'

  n_beam_clusters: null
  beam_cluster_tail_fraction: null
  clustering_calibration_enabled: false
  clustering_error_threshold: 1.0e-5

  hwp_enabled: false
  hwp_rotation_frequency_hz: 0.0
  hwp_initial_phase_rad: 0.0

  detectors: null
  detector_subset: null

  furax_export: true
  furax_export_t0: '2030-01-01T00:00:00+00:00'
```

---

## Data Formats

### Inputs

#### Sky map (`path_to_map`)

A FITS file readable by `healpy.read_map`. Must contain three fields:

- Field 0: Stokes I (intensity)
- Field 1: Stokes Q (linear polarisation)
- Field 2: Stokes U (linear polarisation)

Only the fields listed in `map_fields` are read and allocated; the others are
never loaded. All read fields must share the same HEALPix `nside`. Values are
loaded at the configured `precision` (`float32` by default).

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

The output format depends on `furax_export`.

#### furax HDF5 observations (`furax_export: true`, default)

One TOAST HDF5 observation per day:

```
obs_day_{day_index}.h5    # (n_det, n_samples) detdata block
```

Each observation bundles every configured detector (the focal plane from
`detectors`) with timestamps anchored at `furax_export_t0`. The stored signal
dtype follows `precision`. Loadable directly by furax's `MultiObservationMapMaker`.

#### NumPy TOD files (`furax_export: false`)

One `.npy` file per processing batch and detector:

```
tod_day_{day_index}.npy            # implicit boresight detector
tod_day_{day_index}_{name}.npy     # per configured focal-plane detector
```

Each file has shape `(3, n_samples)` and dtype matching `precision`.
Axis 0 indexes the Stokes component `[I, Q, U]` (rows for components absent from
`map_fields` are zero); axis 1 indexes the detector sample.

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
| `totalconvolve` | `tod_totalconvolve.py` | `totalconvolve` (ducc0 NUFFT) kernel |

```bash
git checkout gaussian        # enables beam_interp_method: gaussian
git checkout bicubic         # enables beam_interp_method: bicubic
git checkout totalconvolve   # enables beam_interp_method: totalconvolve
git checkout main            # production default
```

Passing an unsupported `beam_interp_method` on any branch raises a `ValueError`
pointing to the correct branch.

## Repository Structure

```
.
├── sample_based_tod_generation_gridint.py  # Main entry point
├── run_calibration.py                      # Standalone calibration entry
├── tod_core.py                             # Core Numba JIT kernels + interpolation dispatch
├── tod_rotations.py                        # Rodrigues rotation kernels
├── tod_spin2.py                            # Spin-2 frame-rotation primitives (every Q/U path)
├── tod_bilinear.py                         # Bilinear interpolation
├── tod_nearest.py                          # Nearest-pixel interpolation
├── tod_focalplane.py                       # Focal-plane detectors + on-the-fly quaternion pointing
├── tod_to_furax.py                         # Multi-detector TOAST HDF5 export (furax-compatible)
├── tod_io.py                               # File I/O (beam, scan, output)
├── tod_config.py                           # Config loader
├── tod_runcontext.py                       # Frozen run-scoped parameters (incl. detectors)
├── tod_calibrate.py                        # Batch-size / process-count / clustering calibration
├── tod_beam_math.py                        # B_ell computation, beam-power dB threshold
├── tod_pipeline_helpers.py                 # Shared boilerplate for the entry scripts
├── tod_utils.py                            # CPU/memory detection and utilities
├── numba_healpy.py                         # Numba re-implementation of HEALPix helpers
├── beam_cluster.py                         # Spherical k-means beam-pixel clustering
├── config.yaml                             # Default configuration template
└── tests/                                  # pytest test suite
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
| `toast` | furax HDF5 export (default-on) |
| `astropy` | Time handling for the HDF5 export |
| `h5py` | HDF5 observation I/O |

Install with:

```bash
pip install numpy healpy pixell numba pyyaml psutil   # core TOD generation
pip install toast astropy h5py                         # furax HDF5 export (default-on)
```

The downstream furax GPU mapmaker (`furax_mapmaker/`) has its own JAX/furax
stack, separate from the export dependencies above.

---

## Tests

```bash
cd tests
python run_all_tests.py
# or
pytest tests/
```
