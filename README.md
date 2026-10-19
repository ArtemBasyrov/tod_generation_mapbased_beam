# TOD Generation from Beam Convolution

[![Documentation](https://img.shields.io/badge/docs-readthedocs-blue)](https://tod-generation-mapbased-beam.readthedocs.io/en/latest/index.html)

Sample-based Time-Ordered Data (TOD) generation for CMB experiments.
The pipeline convolves polarised HEALPix sky maps (I, Q, U) with pixelated beam patterns along a boresight scan trajectory.
It writes one output file per processing batch.
The batch is called a *day* throughout, but any convenient grouping works.

By default each day is written as a TOAST HDF5 observation, ready for the [furax](https://github.com/CMBSciPol/furax) mapmaker.

---

## Overview

For each pointing sample the pipeline evaluates

```text
tod[t] = Σ_s  beam_val[s] × skymap[ R(theta_t, phi_t, psi_t) · beam_pixel[s] ]
```

`R` is a compound rotation.
It recentres the beam on the boresight direction and applies the polarisation roll angle.
The sky map is read at each rotated beam node by bilinear (default) or nearest-pixel HEALPix interpolation.

`beam_val[s]` is the beam amplitude times the `cos(dec)` solid-angle Jacobian of its CAR cell, normalised to sum to one.
The Q/U contributions are parallel-transported into the boresight frame (spin-2 transport) before they are summed.

---

## Quick Start

```bash
# 1. Install dependencies
pip install numpy healpy pixell numba pyyaml psutil toast astropy h5py

# 2. Copy and edit the config
cp config.yaml config_local.yaml
$EDITOR config_local.yaml   # set FOLDER_SCAN, FOLDER_TOD_OUTPUT, path_to_map, FOLDER_BEAM, beam_file_*

# 3. Run the pipeline
python sample_based_tod_generation_gridint.py
```

The first run detects the available CPUs and memory.
It then calibrates the process count, Numba threads per worker and batch size, and writes them to the active config.
Later runs reuse those values and skip the calibration.

`python run_calibration.py` runs the calibration on its own, without generating TODs.

---

## Configuration

All settings live in `config.yaml`.
If `config_local.yaml` exists it is read instead, so machine-specific paths stay out of the tracked file.
The calibration steps write their results back into whichever file is active.

### Paths and input maps

| Key | Type | Default | Description |
|---|---|---|---|
| `FOLDER_SCAN` | `str` | required | Directory with the scan files `theta_N.npy`, `phi_N.npy`, `psi_N.npy`. Must end with `/`. |
| `FOLDER_TOD_OUTPUT` | `str` | required | Output directory. Created if absent. |
| `path_to_map` | `str` | required | HEALPix FITS sky map (RING ordering). |
| `map_fields` | `list[int]` | `[0, 1, 2]` | Stokes components to read from the map: `0` = T, `1` = Q, `2` = U. Any non-empty subset is valid. See [Partial-Stokes maps](#partial-stokes-maps). |
| `FOLDER_BEAM` | `str` | required | Directory with the beam FITS files. Must end with `/`. |
| `beam_file_I` | `str` | — | I beam file inside `FOLDER_BEAM`. Required when `0` is in `map_fields`. |
| `beam_file_Q` | `str` | — | Q beam file. Required when `1` is in `map_fields`. |
| `beam_file_U` | `str` | — | U beam file. Required when `2` is in `map_fields`. |
| `beam_center_x` | `int \| null` | `null` | Row (Dec-axis) index of the beam-centre pixel. `null` uses `H // 2`. |
| `beam_center_y` | `int \| null` | `null` | Column (RA-axis) index of the beam-centre pixel. `null` uses `W // 2`. |

Components may share one beam file.
Q and U may also use different beams.
A Q/U weight difference is applied in the detector polarisation basis, so the TOD stays independent of ψ and of the sky coordinate system.

### Beam pixel selection

| Key | Type | Default | Description |
|---|---|---|---|
| `power_fraction_threshold_I` | `float` | — | Fraction of the integrated I beam power to keep. Required when `0` is in `map_fields`. |
| `power_fraction_threshold_Q` | `float` | — | Same for the Q beam. |
| `power_fraction_threshold_U` | `float` | — | Same for the U beam. |

Pixels are ranked by `B cos(dec)`, so the threshold is a fraction of integrated power rather than of peak amplitude.
The template uses `1.00`, which keeps every non-zero pixel.
Lower values drop the faintest pixels and run faster.
With clustering enabled, `1.00` is affordable; see [Beam pixel clustering](#beam-pixel-clustering).

### Batch range

| Key | Type | Default | Description |
|---|---|---|---|
| `start_day` | `int` | required | First day to process (inclusive). |
| `end_day` | `int \| null` | required | Last day to process (exclusive). `null` processes every day found in `FOLDER_SCAN`. |

### Multiprocessing and runtime calibration

| Key | Type | Default | Description |
|---|---|---|---|
| `n_processes` | `int` | required | Upper bound on worker processes. On a cluster the scheduler allocation takes precedence and this is only a cap. |
| `mp_start_method` | `str` | `'spawn'` | `'spawn'` is safe everywhere. `'fork'` starts workers faster on Linux but may deadlock on macOS. |
| `calibration_enabled` | `bool` | `true` | Run the runtime calibration on this invocation. Reset to `false` once it finishes. |
| `calibration_n_processes` | `int \| null` | `null` | Calibrated process count. Written automatically. |
| `calibration_numba_threads` | `int \| null` | `null` | Calibrated Numba threads per worker. Written automatically. |
| `calibration_batch_size` | `int \| null` | `null` | Calibrated samples per batch. Written automatically. |

To force a re-calibration, set `calibration_enabled: true`.
To override the result, edit the three cached values and keep `calibration_enabled: false`.

The per-process memory budget comes from `psutil`, which is required.
Locally the pipeline budgets 75% of available RAM.
On a detected cluster (SLURM, PBS, LSF, SGE) it budgets all of it.

### Numerics

| Key | Type | Default | Description |
|---|---|---|---|
| `beam_interp_method` | `str` | `'bilinear'` | `'bilinear'` or `'nearest'`. See the table below. |
| `precision` | `str` | `'float32'` | `'float32'` or `'float64'`. Sets the dtype of the sky map, beam weights and node vectors, and the stored TOD. |
| `spin2_skip_tolerance` | `float \| null` | `null` | Skip the spin-2 transport for boresights in an equatorial band, where the rotation it applies stays below this tolerance in `\|2δ\|` [rad]. `null` or `0` always applies it. |

| Value | Description |
|---|---|
| `'bilinear'` | 4-pixel HEALPix bilinear interpolation. Smooth under rotation. **Recommended.** |
| `'nearest'` | Uses the HEALPix pixel that contains each beam node. Faster (7,684 against 4,272 samples/s/core on an M4 Pro), but jumps when nodes cross pixel boundaries. Do not use it for polarisation analysis. |

`precision` does not affect the pointing, which is always float64.
It also leaves the bilinear weights, the spin-2 cache, the accumulators and `B_ℓ` at float64.
With float32, the remaining cost is the beam node vectors, at 3.8e-6 of the TOD rms.

`spin2_skip_tolerance: 0.01` is a reasonable speed setting.
Leave it `null` for validation runs.

### Half-wave plate

| Key | Type | Default | Description |
|---|---|---|---|
| `hwp_enabled` | `bool` | `false` | Modulate Q/U by an ideal rotating HWP. Requires both Q and U in `map_fields`. |
| `hwp_rotation_frequency_hz` | `float` | `0.0` | Physical rotation rate `f_HWP` [Hz]. Must be positive when the HWP is enabled. |
| `hwp_initial_phase_rad` | `float` | `0.0` | HWP angle at the start of day 0 [rad]. |

The HWP angle is `φ(t) = 2π f_HWP t + φ₀`, wrapped to `[0, 2π)`.
Time `t` runs continuously from the start of day 0.
The generator rotates the beam-convolved Q/U by `4φ`, so a mapmaker still forms `d = I + Q′ cos 2ψ + U′ sin 2ψ`.
The HDF5 export stores the same wrapped angle, bit for bit, so furax models exactly the modulation in the signal.

### Output format

| Key | Type | Default | Description |
|---|---|---|---|
| `furax_export` | `bool` | `true` | `true` writes one TOAST HDF5 observation per day. `false` writes the raw `(3, n_samples)` `.npy` instead. |
| `furax_export_t0` | `str` | `'2030-01-01T00:00:00+00:00'` | ISO-8601 UTC time of sample 0 of day 0. |

See [Outputs](#outputs) for both formats.

### Beam pixel clustering

Spherical k-means merges the low-power *tail* of the beam into centroids.
The bright main lobe stays pixel-exact.
This is what makes a 100%-power beam affordable: on the 30′ reference beam it reduces 40,401 grid nodes to 6,397 (6.3×).

| Key | Type | Default | Description |
|---|---|---|---|
| `n_beam_clusters` | `int \| null` | `null` | Number of tail clusters. `null` disables clustering. Written by the clustering calibration. |
| `beam_cluster_tail_fraction` | `float \| null` | `null` | Fraction of integrated beam power treated as the tail. Written by the clustering calibration. |
| `beam_cluster_whiten` | `bool` | `true` | Assign pixels to clusters in the frame where the beam's second moment is isotropic. Whether this helps depends on the beam. Written by the clustering calibration. |
| `beam_symmetric` | `bool` | `false` | Cluster one quadrant and copy the partition onto the other three by 90° rotation. Cancels the clustering's manufactured `m = ±2` exactly. Only for a beam that is genuinely round. Overrides `beam_cluster_whiten`. |
| `clustering_calibration_enabled` | `bool` | `false` | Run the clustering calibration on this invocation. Reset to `false` once it finishes. |
| `clustering_error_threshold` | `float` | `1.0e-3` | **`m = 0` bound.** Maximum relative RMS divergence of `B_ℓ`, which bounds the beam *width*. The template sets `1.0e-5`. |
| `clustering_ellipticity_tolerance` | `float \| null` | `1.0` | **`m = ±2` bound.** Maximum error in the beam's own `b_{ℓ,2}` multipoles, as a factor over the least the sweep achieves for this beam. Bounds the beam *shape*. Must be `≥ 1.0`. `null` reports without gating. |

The two bounds constrain different harmonics, and neither implies the other.
`B_ℓ` depends on each node only through its distance from the beam centre, so it cannot see the ellipticity that clustering adds.
That added ellipticity is the term that sources T→P leakage.

**Calibration.**
Set `clustering_calibration_enabled: true` and run once.
The calibrator sweeps a `(tail_fraction × n_clusters × whiten)` grid using beam geometry alone, with no scan data.
It picks the fastest point inside both bounds and breaks speed ties on the least added ellipticity.
It writes `n_beam_clusters`, `beam_cluster_tail_fraction` and `beam_cluster_whiten` back to the config.
If no point meets both bounds, it falls back to the point with the least added ellipticity.
The `n_clusters` grid spans 0.1 to 4 times each tail fraction's crossover cluster count `K_×`, which the table prints as `K/K_x`.

**Manual setting.**
Keep `clustering_calibration_enabled: false` and set `n_beam_clusters`, `beam_cluster_tail_fraction` and `beam_cluster_whiten` by hand.

**`beam_symmetric`.**
The beam's measured 90° asymmetry is printed when it loads.
The value is reported, not enforced.
The flag needs an odd-sized beam grid with equal RA and Dec spacing, centred on the beam-centre pixel.

The [clustering documentation](https://tod-generation-mapbased-beam.readthedocs.io/en/latest/beam_cluster_calibration.html) defines both metrics and gives guidance on choosing the bounds.

### Partial-Stokes maps

`map_fields` selects which components are loaded.
Inactive components are never allocated, and their beam files are never opened.
The spin-2 transport runs only when both Q and U are active.
The output keeps all three I/Q/U rows, and inactive rows are zero.

`tod_config.py` raises a `ValueError` at import that lists any missing `beam_file_*` or `power_fraction_threshold_*` entry for an active component.

### Example `config_local.yaml`

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
  power_fraction_threshold_I: 1.00
  power_fraction_threshold_Q: 1.00
  power_fraction_threshold_U: 1.00
  beam_center_x: null
  beam_center_y: null

  start_day: 0
  end_day: 366

  n_processes: 8
  mp_start_method: 'spawn'
  calibration_enabled: true
  calibration_n_processes: null
  calibration_numba_threads: null
  calibration_batch_size: null

  beam_interp_method: 'bilinear'
  precision: 'float32'
  spin2_skip_tolerance: null

  n_beam_clusters: null
  beam_cluster_tail_fraction: null
  beam_cluster_whiten: true
  beam_symmetric: false
  clustering_calibration_enabled: true
  clustering_error_threshold: 1.0e-5
  clustering_ellipticity_tolerance: 1.0

  hwp_enabled: false
  hwp_rotation_frequency_hz: 0.0
  hwp_initial_phase_rad: 0.0

  furax_export: true
  furax_export_t0: '2030-01-01T00:00:00+00:00'
```

---

## Data Formats

### Inputs

#### Sky map (`path_to_map`)

A HEALPix FITS file in RING ordering, readable by `healpy.read_map`.
Fields 0, 1 and 2 hold Stokes I, Q and U.
Only the fields in `map_fields` are read, and they must share one `nside`.
The maps are cast to `precision` when loaded.

Q and U must follow the HEALPix/COSMO polarisation convention, as healpy and PySM write them.
An IAU-convention map has the opposite sign of U, and the spin-2 transport would then rotate it the wrong way without any error.

#### Beam files (`FOLDER_BEAM` / `beam_file_{I,Q,U}`)

A [pixell / enmap](https://pixell.readthedocs.io/en/latest/usage.html#usagepage) FITS map on a CAR grid.
Only the first map component is read, so each Stokes beam that differs needs its own file.
RA and Dec come from the WCS header and are expressed as offsets from the beam-centre pixel.
That pixel is `(H // 2, W // 2)` for a map of shape `(H, W)`, unless `beam_center_x` / `beam_center_y` say otherwise.
Values are linear beam amplitude, not dB.

Use an odd-sized grid, such as 201 × 201, so the centre pixel is unambiguous.
`beam_symmetric: true` requires it.

The beam does not need to be normalised.
The pipeline multiplies each pixel by `cos(dec)`, keeps the requested power fraction, and normalises the weights to sum to one.
Zero-weight pixels are dropped.

The [beam creation example](https://tod-generation-mapbased-beam.readthedocs.io/en/latest/beam_creation.html) generates a synthetic beam file with pixell.

#### Scan files (`FOLDER_SCAN`)

One triplet of `.npy` files per day:

```text
theta_{day_index}.npy   # boresight colatitude  [rad]
phi_{day_index}.npy     # boresight longitude   [rad]
psi_{day_index}.npy     # polarisation angle    [rad]
```

Each file is a 1-D array with one element per sample, in float32 or float64.
The files are memory-mapped, and the pointing is converted to float64 batch by batch.
The number of days is inferred from the highest-indexed `psi_*.npy`.
The sample rate is `len(psi_0.npy) / 86400` Hz.

### Outputs

#### `furax_export: true` (default)

```text
obs_day_{day_index}.h5    # one TOAST observation, signal shape (1, n_samples)
index.sqlite              # observation index written by TOAST's SaveHDF5
```

The stored detector signal is `d = I + Q cos 2ψ + U sin 2ψ`, after any HWP modulation, at the pipeline `precision`.
That dtype must match furax's `double_precision`.
No `.npy` file is written.
The export needs `toast`, which is imported at start-up so a missing install fails before any generation.

#### `furax_export: false`

```text
tod_day_{day_index}.npy   # shape (3, n_samples), dtype = precision
```

Axis 0 indexes the Stokes component `[I, Q, U]`.
Axis 1 indexes the sample.

`python tod_to_furax.py` re-exports these files to HDF5 afterwards.
It writes to `<FOLDER_TOD_OUTPUT>/furax_h5/` unless `--output` is given.
It also takes `--start-day`, `--end-day`, `--t0`, `--split-per-day` and `--precision`.
`--precision` defaults to `float64`, unlike the integrated export.

#### Mapmaking

`furax_mapmaker/run_furax_mapmaker.py` runs furax's `MultiObservationMapMaker` over the exported observations and writes IQU maps.
`furax_mapmaker/submit_furax_mapmaker.sh` is the Slurm script for the NERSC Perlmutter GPU partition.

---

## Running on HPC / SLURM

The pipeline detects SLURM, PBS, LSF and SGE allocations.
On a cluster it uses all allocated CPUs, capped by `n_processes`, and the full node memory.
No extra flags are needed:

```bash
#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

python sample_based_tod_generation_gridint.py
```

The runtime calibration searches process count, Numba threads per worker and batch size together, within the memory budget.
On memory-limited nodes it often picks fewer processes than allocated CPUs.

---

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations |
| `healpy` | HEALPix map I/O and pixel utilities |
| `numba` | JIT compilation of the rotation and interpolation kernels |
| `pixell` | enmap beam file loading |
| `pyyaml` | Config file parsing |
| `psutil` | CPU and memory detection (required) |
| `toast`, `astropy`, `h5py` | TOAST HDF5 export for furax (`furax_export: true`, the default) |

```bash
pip install -r requirements.txt
```

The furax GPU mapmaker in `furax_mapmaker/` has its own JAX/furax stack, which is not listed here.

---

## Tests

```bash
pytest tests/
```
