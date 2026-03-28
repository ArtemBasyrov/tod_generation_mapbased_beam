"""
precompute_beam_cache.py
========================
Standalone pre-computation script to be run ONCE before the main TOD pipeline.

What it does
------------
The main loop in `beam_tod_batch` (tod_core.py) applies a *double* Rodrigues
rotation to every (sample b, beam-pixel s) pair at runtime:

    Rodrigues 1  –  recenter beam to pointing direction (theta_b, phi_b)
    Rodrigues 2  –  roll beam by polarisation angle psi_b  (axis = pointing dir)

Due to the rotation-group identity, the psi-roll commutes through Rodrigues 1:

    R2(psi, ax_pts) · R1(theta,phi) · v  =  R1(theta,phi) · R2(psi, beam_ctr) · v

where beam_ctr = [1, 0, 0] (the beam-centre direction in beam-frame coordinates).

This means R2 can be precomputed entirely in the beam frame for a discrete grid
of psi values.  At runtime only Rodrigues 1 is needed, cutting rotation work
roughly in half.

Output (per unique beam file)
-----------------------------
A .npz file containing:
    psi_grid   : (N_psi,)      float32  – psi bin centres [rad]
    vec_rolled : (N_psi, S, 3) float32  – beam-pixel unit vectors after psi roll
    beam_vals  : (S,)          float32  – normalised beam weights (same as main)
    beam_ctr   : (3,)          float32  – beam-centre unit vector [1, 0, 0]

Usage
-----
    python precompute_beam_cache.py [--n_psi 720] [--config config.yaml]

    # then point the main script at the cache directory via config:
    #   beam_cache_dir: /path/to/cache/
"""

import os
import argparse
import time

import numpy as np
import numba

import tod_config as config
from tod_io import load_beam
from tod_utils import _compute_dB_threshold_from_power


# ── Parameters ────────────────────────────────────────────────────────────────

DEFAULT_N_PSI = 720    # 0.5-degree bins; increase for wider beams


# ── Rodrigues rotation (beam-frame, around beam_ctr) ──────────────────────────

@numba.jit(nopython=True, cache=True, parallel=True)
def _roll_vectors_jit(vec_orig, beam_ctr, psi_grid, out):
    """
    Apply a pure psi-roll (Rodrigues rotation around beam_ctr) to every
    beam pixel for every psi bin.

    Parameters
    ----------
    vec_orig  : (S, 3)          float32  – original beam-pixel unit vectors
    beam_ctr  : (3,)            float32  – beam-centre unit vector (rotation axis)
    psi_grid  : (N_psi,)        float32  – psi angles to evaluate [rad]
    out       : (N_psi, S, 3)   float32  – written in place
    """
    N_psi = psi_grid.shape[0]
    S     = vec_orig.shape[0]
    kx = beam_ctr[0]; ky = beam_ctr[1]; kz = beam_ctr[2]

    for k in numba.prange(N_psi):      # parallel over psi bins
        cp = np.cos(psi_grid[k])
        sp = np.sin(psi_grid[k])
        om = 1.0 - cp
        for s in range(S):
            vx = vec_orig[s, 0]
            vy = vec_orig[s, 1]
            vz = vec_orig[s, 2]
            dkv = kx*vx + ky*vy + kz*vz
            out[k, s, 0] = vx*cp + (ky*vz - kz*vy)*sp + kx*dkv*om
            out[k, s, 1] = vy*cp + (kz*vx - kx*vz)*sp + ky*dkv*om
            out[k, s, 2] = vz*cp + (kx*vy - ky*vx)*sp + kz*dkv*om


# ── Angular-offset precomputation (optional Phase 2) ──────────────────────────

def _compute_angular_offsets(vec_rolled, beam_ctr):
    """
    Convert rolled beam-pixel unit vectors to flat-sky angular offsets from
    the beam centre.

    In the local tangent plane at beam_ctr = [1, 0, 0]:
        e_theta  (north direction) = [0, 0, 1]
        e_phi    (east direction)  = [0, 1, 0]

    For a unit vector v close to beam_ctr, the small-angle offsets are:
        dtheta ≈ v[2]   (projection onto north)
        dphi   ≈ v[1]   (projection onto east, un-divided by sin(theta))

    These offsets are used at runtime with the flat-sky approximation:
        theta_s(b) ≈ theta_b + dtheta[k, s]
        phi_s  (b) ≈ phi_b   + dphi  [k, s] / sin(theta_b)

    Parameters
    ----------
    vec_rolled : (N_psi, S, 3) float32
    beam_ctr   : (3,)          float32

    Returns
    -------
    dtheta : (N_psi, S) float32 – colatitude offset [rad]
    dphi   : (N_psi, S) float32 – raw phi offset (divide by sin(theta_b) at runtime)

    Notes
    -----
    This is only valid when the beam is narrow (< ~5 deg).  For wider beams
    use vec_rolled directly and the exact Rodrigues 1 + HEALPix path.
    The bilinear weights still require per-sample recomputation at runtime
    (they depend on the fractional pixel offset which varies with pointing).
    """
    # Local tangent plane at beam_ctr = [1, 0, 0]:
    # e_theta = [0, 0, 1], e_phi = [0, 1, 0]
    # For a vector v near beam_ctr, parallel-transport gives:
    #   component along e_theta → v[2]
    #   component along e_phi   → v[1]
    dtheta = vec_rolled[:, :, 2].copy()   # (N_psi, S)
    dphi   = vec_rolled[:, :, 1].copy()   # (N_psi, S)
    return dtheta.astype(np.float32), dphi.astype(np.float32)


# ── Per-beam precomputation ───────────────────────────────────────────────────

def precompute_beam(bf, folder_beam, n_psi, power_threshold, compute_offsets=True):
    """Load one beam file, select pixels by power, and precompute psi-rolled vectors.

    Mirrors the pixel-selection logic in
    :func:`~sample_based_tod_generation_gridint.prepare_beam_data` so that the
    same ``S`` pixels and normalisation are used at cache-generation time and at
    runtime.

    Args:
        bf (str): Beam filename relative to ``folder_beam``.
        folder_beam (str): Path to the beam data directory.
        n_psi (int): Number of psi bins. 720 gives 0.5° resolution. Increase
            for beams wider than ~5°.
        power_threshold (float): Fraction of total beam power to retain for
            pixel selection (e.g. ``0.99`` keeps 99 % of power).
        compute_offsets (bool): If ``True`` (default), also compute flat-sky
            angular offsets ``dtheta`` and ``dphi`` for the fastest runtime
            path. Set to ``False`` for beams wider than ~5° where the
            flat-sky approximation is invalid.

    Returns:
        dict: Cache dictionary with the following keys:

            - ``'psi_grid'`` (*numpy.ndarray*, ``(N_psi,)``) – psi bin centres
              [rad].
            - ``'vec_rolled'`` (*numpy.ndarray*, ``(N_psi, S, 3)``) – beam-pixel
              unit vectors after psi-roll for each bin.
            - ``'beam_vals'`` (*numpy.ndarray*, ``(S,)``) – normalised beam
              weights.
            - ``'beam_ctr'`` (*numpy.ndarray*, ``(3,)``) – beam-centre unit
              vector (always ``[1, 0, 0]``).
            - ``'dtheta'`` (*numpy.ndarray*, ``(N_psi, S)``) – flat-sky
              colatitude offsets [rad]. Present only when
              ``compute_offsets=True``.
            - ``'dphi'`` (*numpy.ndarray*, ``(N_psi, S)``) – flat-sky phi
              offsets [rad] (divide by ``sin(theta_b)`` at runtime). Present
              only when ``compute_offsets=True``.
    """
    print(f"  Loading {bf} ...", flush=True)
    ra, dec, pixel_map = load_beam(folder_beam, bf)

    # ── pixel selection (mirrors prepare_beam_data) ────────────────────────
    dB_cut = _compute_dB_threshold_from_power(pixel_map, power_threshold)
    sel       = (10 * np.log10(np.abs(pixel_map) + 1e-30) > dB_cut)
    beam_vals = pixel_map[sel].astype(np.float32)
    norm      = beam_vals.sum()
    if norm != 0:
        beam_vals /= norm

    theta_orig = np.pi / 2 - dec
    vec_orig   = np.stack([
        np.sin(theta_orig) * np.cos(ra),
        np.sin(theta_orig) * np.sin(ra),
        np.cos(theta_orig),
    ], axis=-1)[sel].astype(np.float32)   # (S, 3)

    S = vec_orig.shape[0]
    print(f"    {S} selected pixels", flush=True)

    # ── beam-centre unit vector ────────────────────────────────────────────
    # Centre pixel is computed from the grid shape, matching load_beam() and
    # precompute_rotation_vector_batch().
    center_idx = (ra.shape[0] // 2, ra.shape[1] // 2)
    phi_c   = float(ra [center_idx])
    th_c    = float(np.pi / 2 - dec[center_idx])
    beam_ctr = np.array([
        np.sin(th_c) * np.cos(phi_c),
        np.sin(th_c) * np.sin(phi_c),
        np.cos(th_c),
    ], dtype=np.float32)
    # ra/dec offsets are already centred, so beam_ctr = [1, 0, 0].
    assert np.allclose(beam_ctr, [1., 0., 0.], atol=1e-4), (
        f"Unexpected beam_ctr {beam_ctr} — check that load_beam() centres the grid correctly")

    # ── psi grid ──────────────────────────────────────────────────────────
    # Cover the full circle; psi outside [0, 2π] wraps via modulo at runtime.
    psi_grid = np.linspace(0.0, 2 * np.pi, n_psi, endpoint=False, dtype=np.float32)

    # ── JIT warm-up (first call compiles; subsequent calls use cache) ──────
    _roll_vectors_jit(
        vec_orig[:1].copy(),
        beam_ctr,
        psi_grid[:1].copy(),
        np.empty((1, 1, 3), dtype=np.float32),
    )

    # ── main precomputation ───────────────────────────────────────────────
    vec_rolled = np.empty((n_psi, S, 3), dtype=np.float32)
    t0 = time.perf_counter()
    _roll_vectors_jit(vec_orig, beam_ctr, psi_grid, vec_rolled)
    dt = time.perf_counter() - t0

    mem_mb = vec_rolled.nbytes / 1e6
    print(f"    vec_rolled: {n_psi} × {S} × 3  =  {mem_mb:.1f} MB  ({dt:.2f}s)", flush=True)

    result = dict(
        psi_grid   = psi_grid,
        vec_rolled = vec_rolled,
        beam_vals  = beam_vals,
        beam_ctr   = beam_ctr,
    )

    if compute_offsets:
        dtheta, dphi = _compute_angular_offsets(vec_rolled, beam_ctr)
        result['dtheta'] = dtheta   # (N_psi, S)  flat-sky colatitude offsets
        result['dphi']   = dphi     # (N_psi, S)  flat-sky phi offsets (raw)
        print(f"    angular offsets: max |dtheta|={np.max(np.abs(dtheta)):.4f} rad", flush=True)

    return result


# ── I/O ───────────────────────────────────────────────────────────────────────

def _cache_filename(bf, output_dir, n_psi):
    """Return the .npz cache path for a given beam filename.

    The cache file is named ``{beam_stem}_cache_npsi{n_psi}.npz`` inside
    ``output_dir``. This naming scheme allows multiple psi-bin resolutions to
    coexist in the same directory.

    Args:
        bf (str): Beam filename (basename or full path; only the stem is used).
        output_dir (str): Directory where cache files are stored.
        n_psi (int): Number of psi bins; embedded in the filename so the main
            script can verify it matches ``config.beam_cache_n_psi``.

    Returns:
        str: Absolute path to the ``.npz`` cache file.
    """
    stem = os.path.splitext(os.path.basename(bf))[0]
    return os.path.join(output_dir, f"{stem}_cache_npsi{n_psi}.npz")


def _save_cache(cache, path):
    np.savez_compressed(path, **cache)
    size_mb = os.path.getsize(path) / 1e6
    print(f"    Saved → {path}  ({size_mb:.1f} MB compressed)", flush=True)


def _load_cache(path):
    """Load a precomputed beam cache file.

    Args:
        path (str): Path to the ``.npz`` cache file produced by
            :func:`precompute_beam` / :func:`_save_cache`.

    Returns:
        dict[str, numpy.ndarray]: Dictionary of arrays. Expected keys:
            ``'psi_grid'``, ``'vec_rolled'``, ``'beam_vals'``, ``'beam_ctr'``,
            and optionally ``'dtheta'`` and ``'dphi'``.
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ── Runtime helper (used by main TOD script) ──────────────────────────────────

def _lookup_psi_bin(psi_values, psi_grid):
    """Map psi angles to the nearest bin index in an evenly-spaced psi grid.

    Wraps ``psi_values`` to ``[0, 2π)`` before binning, so the input may span
    any range.

    Args:
        psi_values (numpy.ndarray): Psi angles [rad], shape ``(B,)``.
            Any dtype is accepted.
        psi_grid (numpy.ndarray): Evenly-spaced psi bin centres [rad] covering
            ``[0, 2π)``, shape ``(N_psi,)``.

    Returns:
        numpy.ndarray: Nearest-bin indices, shape ``(B,)``, dtype ``int64``.
            Values are in ``[0, N_psi)``.
    """
    n_psi = len(psi_grid)
    dpsi  = 2 * np.pi / n_psi
    # Wrap to [0, 2π) then divide by bin width
    psi_wrapped = np.mod(psi_values, 2 * np.pi)
    return np.round(psi_wrapped / dpsi).astype(np.int64) % n_psi


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Precompute beam psi-roll cache")
    parser.add_argument("--n_psi",     type=int,  default=DEFAULT_N_PSI,
                        help="Number of psi bins (default: %(default)s)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write .npz files (default: FOLDER_BEAM)")
    parser.add_argument("--no_offsets", action="store_true",
                        help="Skip flat-sky angular-offset precomputation (Phase 2)")
    args = parser.parse_args()

    output_dir = args.output_dir or config.FOLDER_BEAM
    os.makedirs(output_dir, exist_ok=True)

    beam_threshold_map = {
        config.beam_file_I: config.power_threshold_I,
        config.beam_file_Q: config.power_threshold_Q,
        config.beam_file_U: config.power_threshold_U,
    }
    beam_files = list({config.beam_file_I, config.beam_file_Q, config.beam_file_U})

    print(f"Precomputing beam cache")
    print(f"  n_psi      = {args.n_psi}")
    print(f"  output_dir = {output_dir}")
    print(f"  beams      = {beam_files}")
    print()

    for bf in beam_files:
        out_path = _cache_filename(bf, output_dir, args.n_psi)
        if os.path.exists(out_path):
            print(f"  [skip] {bf} — cache exists at {out_path}")
            continue

        print(f"  Processing {bf}")
        cache = precompute_beam(
            bf,
            config.FOLDER_BEAM,
            n_psi             = args.n_psi,
            power_threshold   = beam_threshold_map[bf],
            compute_offsets   = not args.no_offsets,
        )
        _save_cache(cache, out_path)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
