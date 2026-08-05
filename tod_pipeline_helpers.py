"""
Pipeline helpers shared by the entry scripts.

prepare_beam_data            — load beams from disk, dB-threshold + normalise,
                               build (S, 3) beam-pixel unit vectors.
apply_beam_clustering        — in-place spherical k-means reduction of every
                               beam entry, plus reduction of any precomputed
                               cache arrays attached to the entry.
resolve_spin2_skip_threshold — derive the equatorial-band cos(θ) cutoff for
                               the spin-2 Q/U rotation skip optimisation.
apply_hwp_modulation         — rotate Q/U rows of a TOD batch in place to
                               model an ideal continuously rotating HWP.
save_runtime_calibration     — persist (n_processes, numba_threads, batch_size)
                               back into the active config YAML.
save_clustering_calibration  — persist (n_clusters, tail_fraction, whiten)
                               back into the active config YAML.
"""

import numpy as np
import yaml

import tod_config as config
from tod_io import load_beam
from tod_beam_math import _compute_dB_threshold_from_power
from tod_spin2 import compute_spin2_skip_z_threshold
from beam_cluster import (
    c4_asymmetry,
    c4_orbits,
    cluster_beam_pixels,
    cluster_cached_arrays,
)


_GRID_SQUARENESS_TOL = 1e-6


def _check_square_beam_grid(filename, ra, dec):
    """Verify the beam grid has equal RA and Dec spacings.

    Quadrant clustering rotates *grid indices*, ``(di, dj) -> (-dj, di)``, which
    is a 90-degree rotation on the sky only when the two spacings match. On a
    rectangular grid it is a rotation composed with an anisotropic scaling, and
    the ``m = +-2`` cancellation the path exists for does not hold at all — so
    this is a precondition, not a tolerance to be traded.

    Args:
        filename (str): Beam filename, for the error message.
        ra: (H, W) RA offset grid [rad]; the column index runs with RA.
        dec: (H, W) Dec offset grid [rad]; the row index runs with Dec.

    Raises:
        ValueError: If the two spacings differ by more than a relative 1e-6.
    """
    d_dec = float(np.abs(np.diff(dec, axis=0)).mean())
    d_ra = float(np.abs(np.diff(ra, axis=1)).mean())
    if abs(d_ra - d_dec) > _GRID_SQUARENESS_TOL * max(d_ra, d_dec):
        raise ValueError(
            f"beam_symmetric requires a square beam grid, but {filename} is "
            f"gridded at {np.degrees(d_ra) * 60:.6f}' in RA and "
            f"{np.degrees(d_dec) * 60:.6f}' in Dec. Quadrant clustering would "
            f"rotate the grid into an anisotropic scaling of itself and would "
            f"not cancel the m = +-2 artifact. Regrid the beam, or set "
            f"beam_symmetric: false."
        )


def prepare_beam_data(beam_filenames, active_fields=None):
    """Load and preprocess all unique beam files into a beam-data dictionary.

    For each unique beam filename, loads the FITS map, applies the ``cos(dec)``
    solid-angle Jacobian, selects pixels by power threshold (read from
    ``config.power_threshold_{I,Q,U}``), normalises beam weights, and
    precomputes unit vectors.

    Args:
        beam_filenames (list[str]): List of beam filenames (one per Stokes
            component, in the order ``[I, Q, U]``). Duplicate filenames are
            de-duplicated; the corresponding ``comp_indices`` lists which Stokes
            components share a given beam file.
        active_fields (tuple[int, ...] | None): Which Stokes component indices
            (0=T, 1=Q, 2=U) are present in the loaded sky map. Beam entries
            whose comp_indices fall entirely outside this set are dropped, and
            within each kept entry comp_indices is filtered to the active set.
            ``None`` → use ``config.map_fields``.

    Returns:
        dict[str, dict]: Beam-data dictionary keyed by beam filename. Each
            value is a dict with the following entries:

            - ``'ra'`` – RA offset grid [rad]
            - ``'dec'`` – Dec offset grid [rad]
            - ``'beam_vals'`` – Normalised beam weights ``B cos(dec)`` summing
              to 1, shape ``(S,)``
            - ``'sel'`` – Boolean selection mask over the full beam map
            - ``'comp_indices'`` – List of Stokes component indices using this
              beam (e.g. ``[0]`` for I-only, ``[1, 2]`` if Q and U share a map)
            - ``'n_sel'`` – Number of selected pixels ``S``
            - ``'vec_orig'`` – Beam-pixel unit vectors, shape ``(S, 3)``
    """
    if active_fields is None:
        active_fields = config.map_fields
    active_set = set(int(c) for c in active_fields)

    # Inactive entries in beam_filenames may legitimately be None.
    _per_idx_threshold = (
        config.power_threshold_I,
        config.power_threshold_Q,
        config.power_threshold_U,
    )
    beam_threshold_map = {}
    beam_groups = {}
    for i, bf in enumerate(beam_filenames):
        if i not in active_set:
            continue
        if bf is None:
            raise ValueError(
                f"beam_filenames[{i}] is None but component {i} is active "
                f"(map_fields={sorted(active_set)})"
            )
        beam_groups.setdefault(bf, []).append(i)
        beam_threshold_map[bf] = _per_idx_threshold[i]

    beam_data = {}
    for bf, comp_indices in beam_groups.items():
        ra, dec, pixel_map = load_beam(
            config.FOLDER_BEAM,
            bf,
            center_x=config.beam_center_x,
            center_y=config.beam_center_y,
        )

        # The beam file stores point samples, while the convolution integrates
        # against dOmega = cos(dec) dRA dDec; cos(dec) is what turns the samples
        # into a quadrature rule for that integral. Without it the effective
        # beam is B / cos(dec), which manufactures an m = +-2 ellipticity of
        # relative amplitude r^2 / 8 out of a perfectly symmetric beam — the
        # very asymmetry this pipeline exists to measure. The constant part of
        # the cell area cancels in the normalisation below.
        weighted_map = pixel_map * np.cos(dec)

        threshold = beam_threshold_map[bf]
        if threshold >= 1.0:
            # Every pixel contributes; skip the O(N log N) ranking entirely.
            sel = np.ones(weighted_map.shape, dtype=bool)
        else:
            # Ranking on the weighted values makes the retained fraction a
            # fraction of integrated power over the sphere.
            db_cut = _compute_dB_threshold_from_power(weighted_map, threshold)
            sel = 10 * np.log10(np.abs(weighted_map) + 1e-30) >= db_cut

        # A pixel of weight zero contributes beam_val * sky = 0 to every
        # sample, so dropping it changes no output bit. Keeping it costs:
        # it consumes beam-clustering budget and a slot in the gather loop.
        # Tested on the working dtype, which also catches weights that
        # underflow to zero when precision is float32.
        sel &= weighted_map.astype(config.precision_dtype) != 0

        beam_vals = weighted_map[sel].astype(config.precision_dtype)
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
        )[sel].astype(config.precision_dtype)

        beam_data[bf] = {
            "ra": ra,
            "dec": dec,
            "beam_vals": beam_vals,
            "sel": sel,
            "comp_indices": comp_indices,
            "n_sel": int(sel.sum()),
            "vec_orig": vec_orig,
        }

        if config.beam_symmetric:
            _check_square_beam_grid(bf, ra, dec)
            # Grid offsets from the beam-centre pixel, in the same convention
            # load_beam used to centre the RA/Dec offsets.
            H, W = weighted_map.shape
            ci = config.beam_center_x if config.beam_center_x is not None else H // 2
            cj = config.beam_center_y if config.beam_center_y is not None else W // 2
            gi, gj = np.divmod(np.arange(weighted_map.size)[sel.ravel()], W)
            orbit, rot = c4_orbits(gi - ci, gj - cj)
            beam_data[bf]["c4"] = (orbit, rot)
            # Reported, not enforced. The cancellation degrades smoothly as the
            # beam departs from 90-degree symmetry, so this is the number that
            # says whether the path was worth taking: ~1e-4 for a symmetric
            # beam, ~1e-2 for a genuinely asymmetric one, where the
            # construction also costs node reduction.
            print(
                f"  Beam {bf}: quadrant clustering on, C4 asymmetry "
                f"{c4_asymmetry(beam_vals, orbit, rot):.2e}"
            )
        print(f"  Beam {bf}: {sel.sum()} selected pixels")

    return beam_data


_CLUSTER_CACHE_KEYS = ("vec_rolled", "dtheta", "dphi")


def apply_beam_clustering(beam_data, n_clusters, tail_fraction=None, whiten=None):
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
        whiten (bool | None): Assign in the frame where the beam's second
            moment is isotropic. ``None`` → ``config.beam_cluster_whiten``.
    """
    if whiten is None:
        whiten = config.beam_cluster_whiten

    for bf, data in beam_data.items():
        bv_pre = data["beam_vals"]  # (S,) — needed as weights before overwrite
        vo_pre = data["vec_orig"]  # (S, 3)
        S = data["n_sel"]

        vec_out, bv_out, labels = cluster_beam_pixels(
            vo_pre,
            bv_pre,
            n_clusters=n_clusters,
            tail_fraction=tail_fraction,
            whiten=whiten,
            c4=data.get("c4"),
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


def apply_hwp_modulation(tod_batch, day_index, sample_start, fsamp, f_hwp, phi0):
    """Rotate the Q/U rows of a TOD batch in place to model an ideal HWP.

    Models a continuously rotating half-wave plate by applying
        Q' =  Q·cos(4φ) + U·sin(4φ)
        U' = −Q·sin(4φ) + U·cos(4φ)
    where φ(t) = 2π·f_hwp·t + phi0 and t is seconds since the start of day 0
    (continuous across days). Only the polarization rows are touched; T is
    unchanged.

    Args:
        tod_batch (np.ndarray): Shape ``(3, B)``; modified in place.
        day_index (int): Zero-based observation-day index.
        sample_start (int): Index of the first sample of this batch within the
            day (i.e. ``bs`` in the caller).
        fsamp (float): Sample rate [samples/s].
        f_hwp (float): HWP physical rotation frequency [Hz].
        phi0 (float): Initial HWP phase at t=0 [rad].
    """
    B = tod_batch.shape[1]
    if B == 0:
        return
    dt = 1.0 / float(fsamp)
    t0 = day_index * 86400.0 + sample_start * dt
    t = t0 + np.arange(B, dtype=np.float64) * dt
    phi = 2.0 * np.pi * float(f_hwp) * t + float(phi0)
    c = np.cos(4.0 * phi).astype(tod_batch.dtype, copy=False)
    s = np.sin(4.0 * phi).astype(tod_batch.dtype, copy=False)
    Q = tod_batch[1].copy()
    U = tod_batch[2].copy()
    tod_batch[1] = Q * c + U * s
    tod_batch[2] = -Q * s + U * c


def save_clustering_calibration(tail_fraction, n_clusters, whiten=None):
    """Write clustering calibration results back to the active config YAML."""
    written = {
        "n_beam_clusters": int(n_clusters),
        "beam_cluster_tail_fraction": float(tail_fraction),
        "clustering_calibration_enabled": False,
    }
    if whiten is not None:
        written["beam_cluster_whiten"] = bool(whiten)
    _write_config(written)
    print(
        f"Clustering calibration saved: tail_fraction={tail_fraction:.4f}, "
        f"n_clusters={n_clusters}"
        + ("" if whiten is None else f", whiten={bool(whiten)}")
    )
