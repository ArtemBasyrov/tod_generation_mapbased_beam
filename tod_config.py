import os
import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))

_local = os.path.join(_HERE, "config_local.yaml")
_default = os.path.join(_HERE, "config.yaml")

_cfg_file = _local if os.path.exists(_local) else _default
CONFIG_FILE = _cfg_file

with open(_cfg_file) as _f:
    _cfg = yaml.safe_load(_f)

FOLDER_BEAM = _cfg["FOLDER_BEAM"]
FOLDER_SCAN = _cfg["FOLDER_SCAN"]
FOLDER_TOD_OUTPUT = _cfg["FOLDER_TOD_OUTPUT"]
path_to_map = _cfg["path_to_map"]

# Which Stokes components to read from the input FITS map.
# Subset of [0, 1, 2] (= [T, Q, U]). Defaults to all three for backward
# compatibility. With map_fields = [0], a temperature-only FITS file is
# supported: Q/U TOD rows are filled with zeros and the spin-2 frame rotation
# is skipped entirely. The output TOD shape is always (3, n_samples)
# regardless of which fields are loaded.
_map_fields_raw = _cfg.get("map_fields", [0, 1, 2])
if not isinstance(_map_fields_raw, (list, tuple)) or not _map_fields_raw:
    raise ValueError(
        f"map_fields must be a non-empty list/tuple, got {_map_fields_raw!r}"
    )
map_fields = tuple(sorted({int(x) for x in _map_fields_raw}))
if any(c not in (0, 1, 2) for c in map_fields):
    raise ValueError(
        f"map_fields entries must be in {{0, 1, 2}} (T, Q, U); got {map_fields!r}"
    )

# Per-component beam files and power thresholds are required only for the
# Stokes components listed in map_fields. Entries for inactive components
# may be omitted from the YAML or left as null.
beam_file_I = _cfg.get("beam_file_I")
beam_file_Q = _cfg.get("beam_file_Q")
beam_file_U = _cfg.get("beam_file_U")
power_threshold_I = _cfg.get("power_fraction_threshold_I")
power_threshold_Q = _cfg.get("power_fraction_threshold_Q")
power_threshold_U = _cfg.get("power_fraction_threshold_U")

_COMP_LABELS = {0: "I", 1: "Q", 2: "U"}
_beam_files_by_idx = {0: beam_file_I, 1: beam_file_Q, 2: beam_file_U}
_thresholds_by_idx = {
    0: power_threshold_I,
    1: power_threshold_Q,
    2: power_threshold_U,
}
_missing = []
for _c in map_fields:
    _lbl = _COMP_LABELS[_c]
    if _beam_files_by_idx[_c] is None:
        _missing.append(f"beam_file_{_lbl}")
    if _thresholds_by_idx[_c] is None:
        _missing.append(f"power_fraction_threshold_{_lbl}")
if _missing:
    raise ValueError(
        f"map_fields={list(map_fields)} requires config entries: " + ", ".join(_missing)
    )
start_day = _cfg["start_day"]
end_day = _cfg["end_day"]
n_processes = _cfg["n_processes"]
# Optional calibration cache. All three must be set (or calibration_enabled=true)
# for the run to skip calibration. Users can hand-edit any of these to override
# the calibrated value.
calibration_enabled = _cfg.get("calibration_enabled", True)
calibration_n_processes = _cfg.get("calibration_n_processes", None)
calibration_numba_threads = _cfg.get("calibration_numba_threads", None)
calibration_batch_size = _cfg.get("calibration_batch_size", None)

# Working precision for the float32-side of the pipeline.
# Governs sky map, pointings, beam values, Rodrigues rotation, and the saved
# TOD output. Float64 surfaces that exist for accumulation precision (bilinear
# weights, spin-2 cache, TOD accumulator, B_ell) are NOT affected.
# Use 'float64' as a precision-validation knob; 'float32' is the default and
# matches the legacy behaviour.
_precision_raw = _cfg.get("precision", "float32")
_VALID_PRECISION = {"float32": np.float32, "float64": np.float64}
if _precision_raw not in _VALID_PRECISION:
    raise ValueError(
        f"precision must be one of {sorted(_VALID_PRECISION)!r}, got {_precision_raw!r}"
    )
precision = _precision_raw
precision_dtype = _VALID_PRECISION[_precision_raw]

# Additional methods live on feature branches: 'bicubic' (Keys/Catmull-Rom)
# and 'gaussian' (isotropic Gaussian kernel).
_interp_method_raw = _cfg.get("beam_interp_method", "bilinear")
_VALID_INTERP = {"nearest", "bilinear"}
if _interp_method_raw not in _VALID_INTERP:
    raise ValueError(
        f"beam_interp_method must be one of {sorted(_VALID_INTERP)!r}, "
        f"got {_interp_method_raw!r}"
    )
beam_interp_method = _interp_method_raw

# Spin-2 Q/U correction skip optimisation.
# When > 0, the spin-2 frame rotation is bypassed for boresight samples in the
# equatorial band, where its angular magnitude is small.  The tolerance is the
# maximum |2δ| (≈ fractional Q/U error) the optimisation is allowed to introduce.
# null or 0 → disabled.
spin2_skip_tolerance = _cfg.get("spin2_skip_tolerance", None)

# Beam pixel clustering (k-means on the unit sphere before TOD generation).
#
# n_beam_clusters    : int | None — max clusters for the tail (or all pixels
#                      when tail mode is disabled).  None disables clustering.
# beam_cluster_tail_fraction : float | None — fraction of total beam power
#                      that is treated as the "tail" to be clustered; the
#                      remaining (1 - fraction) of power pixels are kept
#                      exactly as-is.  None → cluster all selected pixels
#                      (full mode, higher error).
#
# Recommended: set both together, e.g.
#   n_beam_clusters: 100
#   beam_cluster_tail_fraction: 0.03   # cluster only the faint 3% fringe
n_beam_clusters = _cfg.get("n_beam_clusters", None)
beam_cluster_tail_fraction = _cfg.get("beam_cluster_tail_fraction", None)

clustering_calibration_enabled = _cfg.get("clustering_calibration_enabled", False)
# B_ell divergence bound. B_ell depends on each beam node only through its
# angular distance from the centre, so it is the m = 0 harmonic alone and no
# rearrangement of nodes at fixed radius changes it: this bounds the width
# error and is blind to any ellipticity the clustering manufactures.
clustering_error_threshold = _cfg.get("clustering_error_threshold", 1e-3)
# Bound on the ellipticity clustering ADDS to the beam, which is what the
# m = 0 statistic cannot see.  Clustering's whole leading error is a beam
# second-moment deficit; the part proportional to the beam's own second moment
# narrows the beam without reshaping it and is harmless, and the rest is added
# ellipticity that sources T->P leakage.
#
# Expressed as a factor over the least added ellipticity the (tail_fraction,
# n_clusters) sweep can reach for *this* beam, not as an absolute bound: what
# is achievable depends on the beam, the grid spacing and the cluster budget
# together, so an absolute number cannot be chosen before running. 1.0 accepts
# only the least-ellipticity configuration; larger values trade ellipticity
# for speed; None ignores the term and reports it.
clustering_ellipticity_tolerance = _cfg.get("clustering_ellipticity_tolerance", 1.0)
if clustering_ellipticity_tolerance is not None:
    clustering_ellipticity_tolerance = float(clustering_ellipticity_tolerance)
    if clustering_ellipticity_tolerance < 1.0:
        raise ValueError(
            "clustering_ellipticity_tolerance is a factor over the least added "
            "ellipticity the (tail_fraction, n_clusters) sweep can reach for "
            "this beam, so a value below 1.0 rejects every candidate including "
            f"the best one. Got {clustering_ellipticity_tolerance!r}; use 1.0 to "
            "keep only the least-ellipticity configuration, or null to report "
            "the term without gating on it."
        )

# Half-wave plate (HWP) modulation.
# When hwp_enabled is True, the TOD generator rotates the per-sample (Q, U)
# output by 4·φ_HWP(t) with φ_HWP(t) = 2π·hwp_rotation_frequency_hz·t +
# hwp_initial_phase_rad. Time t is seconds since the start of day 0
# (continuous across days). The HWP is applied AFTER beam convolution and
# affects polarization angle only — it does not modify the beam shape or
# the Rodrigues rotation.
hwp_enabled = bool(_cfg.get("hwp_enabled", False))
hwp_rotation_frequency_hz = float(_cfg.get("hwp_rotation_frequency_hz", 0.0))
hwp_initial_phase_rad = float(_cfg.get("hwp_initial_phase_rad", 0.0))
if hwp_enabled and not (hwp_rotation_frequency_hz > 0.0):
    raise ValueError(
        "hwp_enabled is True but hwp_rotation_frequency_hz must be > 0; "
        f"got {hwp_rotation_frequency_hz!r}"
    )
if hwp_enabled and not (1 in map_fields and 2 in map_fields):
    raise ValueError(
        "hwp_enabled is True but map_fields does not contain both Q (1) and "
        f"U (2); got map_fields={list(map_fields)}. HWP modulation rotates "
        "Q and U into each other and is meaningless without both."
    )

# Multiprocessing start method ('spawn' or 'fork').
# 'spawn' (default): safe on all platforms; re-triggers Numba JIT in each worker.
# 'fork': faster worker startup on Linux (Numba cache already compiled); may cause
#         deadlocks on macOS with some system libraries.
mp_start_method = _cfg.get("mp_start_method", "spawn")

# Beam centre coordinates (row index, column index) in the beam matrix.
# None → use H // 2 and W // 2 (centre of the array).
beam_center_x = _cfg.get("beam_center_x", None)
beam_center_y = _cfg.get("beam_center_y", None)
