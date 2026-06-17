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
clustering_error_threshold = _cfg.get("clustering_error_threshold", 1e-3)

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

# Focal-plane detectors.
# When absent (or null/empty), the pipeline runs a single implicit boresight
# detector and writes legacy-named tod_day_{N}.npy files. When present, each
# entry describes a detector offset from the boresight in the TOAST
# xi-eta-gamma convention (degrees); per-detector pointing is composed on the
# fly (see tod_focalplane.py). Schema per entry:
#   name: str (unique), xi_deg: float, eta_deg: float, gamma_deg: float
# detector_subset (list of names or 0-based indices, or null) optionally runs
# only part of the focal plane — used to shard a focal plane across HPC nodes.
_detectors_raw = _cfg.get("detectors", None)
if _detectors_raw is not None and not isinstance(_detectors_raw, (list, tuple)):
    raise ValueError(
        f"detectors must be a list of detector entries or null, "
        f"got {type(_detectors_raw).__name__}"
    )
_REQUIRED_DET_KEYS = ("name", "xi_deg", "eta_deg", "gamma_deg")
detectors = None
if _detectors_raw:
    detectors = []
    _seen_names = set()
    for _i, _entry in enumerate(_detectors_raw):
        if not isinstance(_entry, dict):
            raise ValueError(
                f"detectors[{_i}] must be a mapping with keys "
                f"{list(_REQUIRED_DET_KEYS)}, got {type(_entry).__name__}"
            )
        _missing_keys = [k for k in _REQUIRED_DET_KEYS if k not in _entry]
        if _missing_keys:
            raise ValueError(
                f"detectors[{_i}] is missing required keys: {_missing_keys}"
            )
        _name = _entry["name"]
        if not isinstance(_name, str) or not _name:
            raise ValueError(
                f"detectors[{_i}].name must be a non-empty string, got {_name!r}"
            )
        if _name in _seen_names:
            raise ValueError(f"duplicate detector name {_name!r} in detectors")
        _seen_names.add(_name)
        _clean = {"name": _name}
        for _k in ("xi_deg", "eta_deg", "gamma_deg"):
            _v = float(_entry[_k])
            if not np.isfinite(_v):
                raise ValueError(f"detectors[{_i}].{_k} must be finite, got {_v!r}")
            _clean[_k] = _v
        # Optional per-detector beam overrides (Phase 2). When a key is absent
        # or null, the detector falls back to the global beam_file_* /
        # power_fraction_threshold_* for that Stokes component. Detectors whose
        # resolved beam files coincide share a single beam set at run time.
        for _bk in ("beam_file_I", "beam_file_Q", "beam_file_U"):
            _bv = _entry.get(_bk)
            if _bv is not None:
                _clean[_bk] = str(_bv)
        for _tk in (
            "power_fraction_threshold_I",
            "power_fraction_threshold_Q",
            "power_fraction_threshold_U",
        ):
            _tv = _entry.get(_tk)
            if _tv is not None:
                _tv = float(_tv)
                if not np.isfinite(_tv):
                    raise ValueError(
                        f"detectors[{_i}].{_tk} must be finite, got {_tv!r}"
                    )
                _clean[_tk] = _tv
        # Optional per-detector beam-clustering overrides. When absent (or null)
        # the detector inherits the global n_beam_clusters / beam_cluster_tail_
        # fraction (themselves possibly set by clustering calibration).
        _ncl = _entry.get("n_beam_clusters")
        if _ncl is not None:
            _ncl = int(_ncl)
            if _ncl <= 0:
                raise ValueError(
                    f"detectors[{_i}].n_beam_clusters must be a positive int, "
                    f"got {_ncl!r}"
                )
            _clean["n_beam_clusters"] = _ncl
        _tf = _entry.get("beam_cluster_tail_fraction")
        if _tf is not None:
            _tf = float(_tf)
            if not (0.0 < _tf <= 1.0):
                raise ValueError(
                    f"detectors[{_i}].beam_cluster_tail_fraction must be in "
                    f"(0, 1], got {_tf!r}"
                )
            _clean["beam_cluster_tail_fraction"] = _tf
        detectors.append(_clean)

detector_subset = _cfg.get("detector_subset", None)
if detector_subset is not None:
    if not isinstance(detector_subset, (list, tuple)) or not detector_subset:
        raise ValueError(
            f"detector_subset must be a non-empty list of detector names or "
            f"indices, or null; got {detector_subset!r}"
        )
    if detectors is None:
        raise ValueError(
            "detector_subset is set but no detectors: section is configured"
        )

# Integrated furax export.
# Enabled by default. When true, the generator writes one TOAST HDF5
# observation per day (obs_day_{N}.h5) into FOLDER_TOD_OUTPUT and removes the
# intermediate per-detector tod_day_*.npy files — the output directory then
# holds only .h5 files. When false, only the raw tod_day_*.npy are written.
# The stored detector signal uses the pipeline `precision`; one observation is
# emitted per day (control granularity via start_day/end_day). Requires toast
# in the active environment (imported lazily, only when enabled). The standalone
# tod_to_furax.py keeps its own --output / --precision / --split-per-day knobs.
furax_export = bool(_cfg.get("furax_export", True))
# ISO-8601 UTC timestamp of sample 0 of day 0 (the timestream time origin).
furax_export_t0 = _cfg.get("furax_export_t0", "2030-01-01T00:00:00+00:00")
