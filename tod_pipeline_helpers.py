"""
Pipeline helpers shared between the two entry scripts
(sample_based_tod_generation_gridint.py and run_calibration.py).

prepare_beam_data            — load beams from disk, dB-threshold + normalise,
                               build (S, 3) beam-pixel unit vectors.
apply_beam_clustering        — in-place spherical k-means reduction of every
                               beam entry, plus reduction of any precomputed
                               cache arrays attached to the entry.
resolve_spin2_skip_threshold — derive the equatorial-band cos(θ) cutoff for
                               the spin-2 Q/U rotation skip optimisation.
save_runtime_calibration     — persist (n_processes, numba_threads, batch_size)
                               back into the active config YAML.
save_clustering_calibration  — persist (n_clusters, tail_fraction) back into
                               the active config YAML.
"""

import numpy as np
import yaml

import tod_config as config
from tod_io import load_beam
from tod_beam_math import _compute_dB_threshold_from_power
from tod_spin2 import compute_spin2_skip_z_threshold
from beam_cluster import cluster_beam_pixels, cluster_cached_arrays


def prepare_beam_data(beam_filenames):
    """Load and preprocess all unique beam files into a beam-data dictionary.

    For each unique beam filename, loads the FITS map, selects pixels by power
    threshold (read from ``config.power_threshold_{I,Q,U}``), normalises beam
    weights, and precomputes unit vectors.

    Args:
        beam_filenames (list[str]): List of beam filenames (one per Stokes
            component, in the order ``[I, Q, U]``). Duplicate filenames are
            de-duplicated; the corresponding ``comp_indices`` lists which Stokes
            components share a given beam file.

    Returns:
        dict[str, dict]: Beam-data dictionary keyed by beam filename. Each
            value is a dict with the following entries:

            - ``'ra'`` – RA offset grid [rad]
            - ``'dec'`` – Dec offset grid [rad]
            - ``'beam_vals'`` – Normalised beam weights, shape ``(S,)``
            - ``'sel'`` – Boolean selection mask over the full beam map
            - ``'comp_indices'`` – List of Stokes component indices using this
              beam (e.g. ``[0]`` for I-only, ``[1, 2]`` if Q and U share a map)
            - ``'n_sel'`` – Number of selected pixels ``S``
            - ``'vec_orig'`` – Beam-pixel unit vectors, shape ``(S, 3)``
    """
    beam_threshold_map = {
        config.beam_file_I: config.power_threshold_I,
        config.beam_file_Q: config.power_threshold_Q,
        config.beam_file_U: config.power_threshold_U,
    }

    beam_groups = {}
    for i, bf in enumerate(beam_filenames):
        beam_groups.setdefault(bf, []).append(i)

    beam_data = {}
    for bf, comp_indices in beam_groups.items():
        ra, dec, pixel_map = load_beam(
            config.FOLDER_BEAM,
            bf,
            center_x=config.beam_center_x,
            center_y=config.beam_center_y,
        )

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


_CLUSTER_CACHE_KEYS = ("vec_rolled", "dtheta", "dphi")


def apply_beam_clustering(beam_data, n_clusters, tail_fraction=None):
    """Apply weighted spherical k-means clustering to ``beam_data`` in-place.

    Pre-clustering ``beam_vals`` serve as pixel weights for both the k-means
    and the subsequent reduction of any precomputed cache arrays attached to
    the beam entry.

    Args:
        beam_data (dict): Beam data from :func:`prepare_beam_data` (exact,
            unclustered). Modified in-place. Cache arrays
            (``vec_rolled``, ``dtheta``, ``dphi``) are reduced from
            ``(N_psi, S, *)`` to ``(N_psi, K_out, *)`` if present.
        n_clusters (int): Max clusters for the tail (or all pixels in full
            mode).
        tail_fraction (float | None): Fraction of power to treat as tail.
            ``None`` → full mode (cluster all pixels).
    """
    for bf, data in beam_data.items():
        bv_pre = data["beam_vals"]  # (S,) — needed as weights before overwrite
        vo_pre = data["vec_orig"]  # (S, 3)
        S = data["n_sel"]

        vec_out, bv_out, labels = cluster_beam_pixels(
            vo_pre,
            bv_pre,
            n_clusters=n_clusters,
            tail_fraction=tail_fraction,
        )
        K = len(bv_out)

        cache_sub = {k: data[k] for k in _CLUSTER_CACHE_KEYS if k in data}
        if cache_sub:
            print(f"    [{bf}] Clustering cache arrays …")
            clustered = cluster_cached_arrays(cache_sub, labels, bv_pre, K)
            for k, arr in clustered.items():
                data[k] = arr

        data["beam_vals"] = bv_out
        data["vec_orig"] = vec_out
        data["n_sel"] = K
        print(f"  [{bf}] Beam clustered: {S} → {K} pixels")


def resolve_spin2_skip_threshold(beam_data, tolerance, beam_radius_quantile=0.999):
    """Derive the spin-2 Q/U rotation-skip cos(θ) cutoff for the equatorial band.

    When ``tolerance`` is set, finds the smallest |cos(θ_pts)| such that
    boresights in the equatorial band (|bz| ≤ cutoff) bypass the spin-2 Q/U
    rotation, with a worst-case |2δ| bounded by ``tolerance`` over all
    beam-pixel positions within the effective beam radius. Returns -1.0 to
    disable the optimisation (tolerance unset, or too tight for the beam size).

    Args:
        beam_data (dict): Beam data from :func:`prepare_beam_data` (post-clustering).
            Each entry must provide ``vec_orig`` (S, 3) and ``beam_vals`` (S,).
        tolerance (float | None): Spin-2 skip tolerance in radians. ``None`` or
            non-positive disables the optimisation.
        beam_radius_quantile (float): Quantile q for the beam-power-weighted
            enclosed radius — the smallest R with Σ_{r_i ≤ R} |b_i| ≥ q · Σ |b_i|.
            Default 0.999 drops the lowest-contribution 0.1% of beam power.

    Returns:
        float: ``z_skip_threshold`` for the gather kernels, or -1.0 if disabled.
    """
    if not tolerance or tolerance <= 0:
        return -1.0

    # Beam radius: beam-power-weighted enclosed radius.  The unweighted
    # max overstates the relevant scale because tail pixels contribute
    # to TOD error proportionally to their beam value.  The centre direction
    # is the beam-weighted mean of vec_orig (convention-independent under
    # any future change of beam frame).  Max across beam entries is kept as
    # a conservative aggregation.
    beam_radius = 0.0
    for _data in beam_data.values():
        vo = _data["vec_orig"].astype(np.float64)
        bv = _data["beam_vals"].astype(np.float64)
        v_centre = (vo * bv[:, None]).sum(axis=0)
        n = float(np.linalg.norm(v_centre))
        if n < 1e-12:
            continue
        v_centre /= n
        cos_off = np.clip(vo @ v_centre, -1.0, 1.0)
        r_pix = np.arccos(cos_off)
        w = np.abs(bv)
        w_total = float(w.sum())
        if w_total <= 0.0:
            continue
        order = np.argsort(r_pix)
        w_cum = np.cumsum(w[order]) / w_total
        idx = int(np.searchsorted(w_cum, beam_radius_quantile))
        if idx >= r_pix.size:
            idx = r_pix.size - 1
        r_enc = float(r_pix[order[idx]])
        if r_enc > beam_radius:
            beam_radius = r_enc

    z_skip_threshold = compute_spin2_skip_z_threshold(beam_radius, float(tolerance))
    if z_skip_threshold < 0.0:
        print(
            f"Spin-2 skip: tolerance={tolerance} too tight "
            f"for beam_radius_eff={np.degrees(beam_radius):.3f}° "
            f"(q={beam_radius_quantile}) — optimisation effectively "
            f"disabled (no equatorial band)."
        )
    else:
        theta_band_deg = np.degrees(np.arccos(z_skip_threshold))
        print(
            f"Spin-2 skip enabled: tol={tolerance}, "
            f"beam_radius_eff={np.degrees(beam_radius):.3f}° "
            f"(q={beam_radius_quantile}), "
            f"z_threshold={z_skip_threshold:.6f} "
            f"(boresight band θ ∈ [{theta_band_deg:.2f}°, "
            f"{180 - theta_band_deg:.2f}°] bypasses correction)"
        )
    return z_skip_threshold


def _write_config(updates):
    """Read the active config YAML, merge ``updates`` into it, and write back."""
    with open(config.CONFIG_FILE) as f:
        raw = yaml.safe_load(f)
    raw.update(updates)
    with open(config.CONFIG_FILE, "w") as f:
        yaml.dump(
            raw,
            f,
            default_flow_style=False,
            allow_unicode=True,
            explicit_start=True,
            sort_keys=False,
        )


def save_runtime_calibration(n_processes, n_threads, batch_size):
    """Write runtime calibration results back to the active config YAML."""
    _write_config(
        {
            "calibration_n_processes": int(n_processes),
            "calibration_numba_threads": int(n_threads),
            "calibration_batch_size": int(batch_size),
            "calibration_enabled": False,
        }
    )
    print(
        f"Calibration saved to {config.CONFIG_FILE} "
        f"(n_processes={n_processes}, numba_threads={n_threads}, "
        f"batch_size={batch_size})"
    )


def save_clustering_calibration(tail_fraction, n_clusters):
    """Write clustering calibration results back to the active config YAML."""
    _write_config(
        {
            "n_beam_clusters": int(n_clusters),
            "beam_cluster_tail_fraction": float(tail_fraction),
            "clustering_calibration_enabled": False,
        }
    )
    print(
        f"Clustering calibration saved: tail_fraction={tail_fraction:.4f}, "
        f"n_clusters={n_clusters}"
    )
