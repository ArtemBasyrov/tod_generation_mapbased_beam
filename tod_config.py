import os
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
beam_file_I = _cfg["beam_file_I"]
beam_file_Q = _cfg["beam_file_Q"]
beam_file_U = _cfg["beam_file_U"]
power_threshold_I = _cfg["power_fraction_threshold_I"]
power_threshold_Q = _cfg["power_fraction_threshold_Q"]
power_threshold_U = _cfg["power_fraction_threshold_U"]
start_day = _cfg["start_day"]
end_day = _cfg["end_day"]
n_processes = _cfg["n_processes"]
max_memory_per_process = _cfg["max_memory_per_process"]

# Optional calibration cache. All three must be set (or calibration_enabled=true)
# for the run to skip calibration. Users can hand-edit any of these to override
# the calibrated value.
calibration_enabled = _cfg.get("calibration_enabled", True)
calibration_n_processes = _cfg.get("calibration_n_processes", None)
calibration_numba_threads = _cfg.get("calibration_numba_threads", None)
calibration_batch_size = _cfg.get("calibration_batch_size", None)

# Beam grid interpolation method
# beam_interp_method: 'nearest'  → single nearest-pixel lookup (fastest)
#                     'bilinear' → 4-pixel bilinear HEALPix (default, fast Numba kernel)
# ('bicubic' branch adds Keys/Catmull-Rom; 'gaussian' branch adds isotropic Gaussian kernel)
_interp_method_raw = _cfg.get("beam_interp_method", "bilinear")
_VALID_INTERP = {"nearest", "bilinear"}
if _interp_method_raw not in _VALID_INTERP:
    raise ValueError(
        f"beam_interp_method must be one of {sorted(_VALID_INTERP)!r}, "
        f"got {_interp_method_raw!r}"
    )
beam_interp_method = _interp_method_raw

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

# Clustering calibration
# enable_clustering_calibration : set True to trigger a calibration sweep on
#   the next run; automatically reset to False after calibration completes.
# clustering_error_threshold     : maximum tolerated relative RMS TOD error.
# clustering_tail_fractions      : list of tail fractions to sweep.
# clustering_n_clusters_list     : list of K values to sweep.
# clustering_n_probe_samples     : number of scan samples used for the probe.
clustering_calibration_enabled = _cfg.get("clustering_calibration_enabled", False)
clustering_error_threshold = _cfg.get("clustering_error_threshold", 1e-3)

# Multiprocessing start method ('spawn' or 'fork').
# 'spawn' (default): safe on all platforms; re-triggers Numba JIT in each worker.
# 'fork': faster worker startup on Linux (Numba cache already compiled); may cause
#         deadlocks on macOS with some system libraries.
mp_start_method = _cfg.get("mp_start_method", "spawn")
