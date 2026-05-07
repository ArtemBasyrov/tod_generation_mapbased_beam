"""
Standalone calibration script.

Runs runtime calibration (n_processes × numba_threads × batch_size) and/or
beam clustering calibration, then writes the results back to the active config
file (config_local.yaml if it exists, otherwise config.yaml).

Usage
-----
    python run_calibration.py            # runtime calibration only
    python run_calibration.py --runtime  # same
    python run_calibration.py --clustering
    python run_calibration.py --runtime --clustering  # both
"""

import argparse
import os

import numpy as np
import healpy as hp

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import tod_config as config
from tod_io import load_scan_information
from tod_calibrate import calibrate_runtime, calibrate_beam_clustering
from tod_utils import _get_ncpus
from tod_pipeline_helpers import (
    prepare_beam_data,
    apply_beam_clustering,
    save_runtime_calibration,
    save_clustering_calibration,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run TOD generation calibration and update the config file."
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="Run runtime calibration (n_processes × numba_threads × batch_size)",
    )
    parser.add_argument(
        "--clustering", action="store_true", help="Run beam clustering calibration"
    )
    args = parser.parse_args()

    # Default: runtime only
    run_runtime = args.runtime or (not args.runtime and not args.clustering)
    run_clustering = args.clustering

    n_cpu_ceiling = _get_ncpus()
    Nb, _ = load_scan_information(config.FOLDER_SCAN)
    probe_day = max(config.start_day or 0, 0)

    print("Loading sky map...")
    MP = [
        m.astype(np.float32) for m in hp.read_map(config.path_to_map, field=(0, 1, 2))
    ]

    print("Loading beam data...")
    beam_files = [config.beam_file_I, config.beam_file_Q, config.beam_file_U]
    beam_data = prepare_beam_data(beam_files)

    if run_clustering:
        print("Running beam clustering calibration...")
        best_tf, best_K = calibrate_beam_clustering(
            beam_data,
            folder_scan=config.FOLDER_SCAN,
            probe_day=probe_day,
            mp=MP,
            error_threshold=config.clustering_error_threshold,
            interp_mode=config.beam_interp_method,
        )
        save_clustering_calibration(best_tf, best_K)
        n_clusters, tail_fraction = best_K, best_tf
    else:
        n_clusters = config.n_beam_clusters
        tail_fraction = config.beam_cluster_tail_fraction

    if n_clusters is not None:
        print(
            f"Applying beam clustering "
            f"(tail_fraction={tail_fraction}, n_clusters={n_clusters}) ..."
        )
        apply_beam_clustering(
            beam_data, n_clusters=n_clusters, tail_fraction=tail_fraction
        )

    for data in beam_data.values():
        data["mp_stacked"] = np.ascontiguousarray(
            np.stack([MP[c] for c in data["comp_indices"]])
        )

    if run_runtime:
        print(
            "Running runtime calibration (n_processes × numba_threads × batch_size)..."
        )
        _cx, _cy = config.beam_center_x, config.beam_center_y
        n_processes, n_threads, batch_size = calibrate_runtime(
            beam_data,
            config.FOLDER_SCAN,
            probe_day=probe_day,
            mp=MP,
            n_cpu_ceiling=n_cpu_ceiling,
            max_processes_user=config.n_processes,
            interp_mode=config.beam_interp_method,
            center_idx=(_cx, _cy) if (_cx is not None and _cy is not None) else None,
        )
        save_runtime_calibration(n_processes, n_threads, batch_size)


if __name__ == "__main__":
    main()
