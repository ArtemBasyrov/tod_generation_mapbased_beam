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
import yaml

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import tod_config as config
from tod_io import load_beam, load_scan_information
from tod_calibrate import calibrate_runtime, calibrate_beam_clustering
from tod_utils import _get_ncpus, _compute_dB_threshold_from_power
from beam_cluster import cluster_beam_pixels, cluster_cached_arrays


# ── Beam helpers (mirrors main script) ───────────────────────────────────────


def _prepare_beam_data():
    beam_files = [config.beam_file_I, config.beam_file_Q, config.beam_file_U]
    beam_threshold_map = {
        config.beam_file_I: config.power_threshold_I,
        config.beam_file_Q: config.power_threshold_Q,
        config.beam_file_U: config.power_threshold_U,
    }
    beam_groups = {}
    for i, bf in enumerate(beam_files):
        beam_groups.setdefault(bf, []).append(i)

    beam_data = {}
    for bf, comp_indices in beam_groups.items():
        ra, dec, pixel_map = load_beam(config.FOLDER_BEAM, bf)
        db_cut = _compute_dB_threshold_from_power(pixel_map, beam_threshold_map[bf])
        sel = 10 * np.log10(np.abs(pixel_map) + 1e-30) > db_cut
        beam_vals = pixel_map[sel].astype(np.float32)
        norm = beam_vals.sum()
        if norm != 0:
            beam_vals /= norm
        theta_orig = np.pi / 2 - dec
        vec_orig = np.stack(
            [
                np.sin(theta_orig) * np.cos(ra),
                np.sin(theta_orig) * np.sin(ra),
                np.cos(theta_orig),
            ],
            axis=-1,
        )[sel].astype(np.float32)
        beam_data[bf] = {
            "ra": ra,
            "dec": dec,
            "beam_vals": beam_vals,
            "sel": sel,
            "comp_indices": comp_indices,
            "n_sel": int(sel.sum()),
            "vec_orig": vec_orig,
        }
        print(f"  Beam {bf}: {sel.sum()} selected pixels")
    return beam_data


def _apply_clustering(beam_data, n_clusters, tail_fraction):
    _CACHE_KEYS = ("vec_rolled", "dtheta", "dphi")
    for bf, data in beam_data.items():
        bv_pre = data["beam_vals"]
        vo_pre = data["vec_orig"]
        S = data["n_sel"]
        vec_out, bv_out, labels = cluster_beam_pixels(
            vo_pre, bv_pre, n_clusters=n_clusters, tail_fraction=tail_fraction
        )
        K = len(bv_out)
        cache_sub = {k: data[k] for k in _CACHE_KEYS if k in data}
        if cache_sub:
            clustered = cluster_cached_arrays(cache_sub, labels, bv_pre, K)
            for k, arr in clustered.items():
                data[k] = arr
        data["beam_vals"] = bv_out
        data["vec_orig"] = vec_out
        data["n_sel"] = K
        print(f"  [{bf}] Beam clustered: {S} → {K} pixels")


# ── Config writers ────────────────────────────────────────────────────────────


def _save_runtime_calibration(n_processes, n_threads, batch_size):
    with open(config.CONFIG_FILE) as f:
        raw = yaml.safe_load(f)
    raw["calibration_n_processes"] = int(n_processes)
    raw["calibration_numba_threads"] = int(n_threads)
    raw["calibration_batch_size"] = int(batch_size)
    raw["calibration_enabled"] = False
    with open(config.CONFIG_FILE, "w") as f:
        yaml.dump(
            raw,
            f,
            default_flow_style=False,
            allow_unicode=True,
            explicit_start=True,
            sort_keys=False,
        )
    print(
        f"Saved to {config.CONFIG_FILE}: "
        f"n_processes={n_processes}, numba_threads={n_threads}, batch_size={batch_size}"
    )


def _save_clustering_calibration(tail_fraction, n_clusters):
    with open(config.CONFIG_FILE) as f:
        raw = yaml.safe_load(f)
    raw["n_beam_clusters"] = int(n_clusters)
    raw["beam_cluster_tail_fraction"] = float(tail_fraction)
    raw["clustering_calibration_enabled"] = False
    with open(config.CONFIG_FILE, "w") as f:
        yaml.dump(
            raw,
            f,
            default_flow_style=False,
            allow_unicode=True,
            explicit_start=True,
            sort_keys=False,
        )
    print(
        f"Saved to {config.CONFIG_FILE}: "
        f"tail_fraction={tail_fraction:.4f}, n_clusters={n_clusters}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────


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
    beam_data = _prepare_beam_data()

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
        _save_clustering_calibration(best_tf, best_K)
        n_clusters, tail_fraction = best_K, best_tf
    else:
        n_clusters = config.n_beam_clusters
        tail_fraction = config.beam_cluster_tail_fraction

    if n_clusters is not None:
        print(
            f"Applying beam clustering "
            f"(tail_fraction={tail_fraction}, n_clusters={n_clusters}) ..."
        )
        _apply_clustering(beam_data, n_clusters=n_clusters, tail_fraction=tail_fraction)

    for data in beam_data.values():
        data["mp_stacked"] = np.ascontiguousarray(
            np.stack([MP[c] for c in data["comp_indices"]])
        )

    if run_runtime:
        print(
            "Running runtime calibration (n_processes × numba_threads × batch_size)..."
        )
        n_processes, n_threads, batch_size = calibrate_runtime(
            beam_data,
            config.FOLDER_SCAN,
            probe_day=probe_day,
            mp=MP,
            n_cpu_ceiling=n_cpu_ceiling,
            max_processes_user=config.n_processes,
            interp_mode=config.beam_interp_method,
        )
        _save_runtime_calibration(n_processes, n_threads, batch_size)


if __name__ == "__main__":
    main()
