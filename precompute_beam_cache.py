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


# ── Parameters ────────────────────────────────────────────────────────────────

DEFAULT_N_PSI   = 720          # 0.5-degree bins; increase for wider beams
BEAM_CENTER_IDX = (100, 100)   # must match precompute_rotation_vector_batch()
DB_THRESHOLD_DB = -35          # pixel selection threshold (same as prepare_beam_data)


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

def precompute_beam(bf, folder_beam, n_psi, compute_offsets=True):
    """
    Load one beam file, apply pixel selection, and precompute vec_rolled.

    Parameters
    ----------
    bf              : str   – beam filename (relative to folder_beam)
    folder_beam     : str   – beam folder path
    n_psi           : int   – number of psi bins
    compute_offsets : bool  – also compute flat-sky angular offsets (Phase 2)

    Returns
    -------
    dict with keys:
        psi_grid, vec_rolled, beam_vals, beam_ctr
        [dtheta, dphi]  if compute_offsets=True
    """
    print(f"  Loading {bf} ...", flush=True)
    ra, dec, pixel_map = load_beam(folder_beam, bf)

    # ── pixel selection (mirrors prepare_beam_data) ────────────────────────
    sel       = (10 * np.log10(np.abs(pixel_map) + 1e-30) > DB_THRESHOLD_DB)
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
    # Must match BEAM_CENTER_IDX used in precompute_rotation_vector_batch().
    phi_c   = float(ra [BEAM_CENTER_IDX])
    th_c    = float(np.pi / 2 - dec[BEAM_CENTER_IDX])
    beam_ctr = np.array([
        np.sin(th_c) * np.cos(phi_c),
        np.sin(th_c) * np.sin(phi_c),
        np.cos(th_c),
    ], dtype=np.float32)
    # ra/dec offsets are already centred, so beam_ctr = [1, 0, 0].
    # This assertion guards against BEAM_CENTER_IDX mismatches.
    assert np.allclose(beam_ctr, [1., 0., 0.], atol=1e-4), (
        f"Unexpected beam_ctr {beam_ctr} — check BEAM_CENTER_IDX")

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

def cache_filename(bf, output_dir, n_psi):
    """Return the .npz path for a given beam file."""
    stem = os.path.splitext(os.path.basename(bf))[0]
    return os.path.join(output_dir, f"{stem}_cache_npsi{n_psi}.npz")


def save_cache(cache, path):
    np.savez_compressed(path, **cache)
    size_mb = os.path.getsize(path) / 1e6
    print(f"    Saved → {path}  ({size_mb:.1f} MB compressed)", flush=True)


def load_cache(path):
    """Load a precomputed cache file. Returns dict of arrays."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


# ── Runtime helper (used by main TOD script) ──────────────────────────────────

def lookup_psi_bin(psi_values, psi_grid):
    """
    Map an array of psi angles (radians, arbitrary range) to the nearest bin
    index in psi_grid.

    Parameters
    ----------
    psi_values : (B,) float32 or float64
    psi_grid   : (N_psi,) float32   evenly spaced on [0, 2π)

    Returns
    -------
    indices : (B,) int64
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

    beam_files = list({config.beam_file_I, config.beam_file_Q, config.beam_file_U})

    print(f"Precomputing beam cache")
    print(f"  n_psi      = {args.n_psi}")
    print(f"  output_dir = {output_dir}")
    print(f"  beams      = {beam_files}")
    print()

    for bf in beam_files:
        out_path = cache_filename(bf, output_dir, args.n_psi)
        if os.path.exists(out_path):
            print(f"  [skip] {bf} — cache exists at {out_path}")
            continue

        print(f"  Processing {bf}")
        cache = precompute_beam(
            bf,
            config.FOLDER_BEAM,
            n_psi         = args.n_psi,
            compute_offsets = not args.no_offsets,
        )
        save_cache(cache, out_path)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
