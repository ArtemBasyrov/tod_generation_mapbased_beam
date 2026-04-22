"""
Core numerical routines for sample-based TOD generation.

All functions are stateless and take only arrays as arguments.

Rotation kernels (in tod_rotations.py)
---------------------------------------
_rodrigues_jit                — fused double Rodrigues rotation (recenter + pol. roll).
_rodrigues1_from_rolled_jit   — single Rodrigues rotation (recenter only) for cached
                                pre-rolled beam vectors.
_rotation_params              — per-sample scalars needed by _rodrigues_jit.
_recenter_and_rotate          — fused recenter + pol-roll wrapper.
precompute_rotation_vector_batch — Rodrigues vectors and pol. angle offsets for a batch.

Gather/accumulate kernels (in tod_bilinear.py)
--------------------------
_gather_accum_jit        — fused HEALPix bilinear gather + beam-weighted accumulation.
_gather_accum_dedup_jit  — bilinear + per-boresight pixel dedup + trig-free spin-2.

HEALPix RING helpers (in numba_healpy.py)
-----------------------------------------
_ring_above_jit, _ring_info_jit, _ring_z_jit,
_get_interp_weights_jit, get_interp_weights_numba
"""

import numpy as np
import healpy as hp

from numba_healpy import get_interp_weights_numba
from tod_rotations import (
    _rodrigues_jit,
    _rotation_params,
    precompute_rotation_vector_batch,
)
from tod_nearest import (
    _gather_accum_nearest_jit,
)
from tod_bilinear import (
    _gather_accum_jit,
    _gather_accum_dedup_jit,
)

# Target working-set size for the (B × Sc × 3 × float32) vec_rot intermediate.
# Sized to stay within a typical L2 cache (2 MB).
_S_TILE_TARGET_BYTES = 2 * 1024 * 1024

# Maximum number of S-tiles per beam entry.  Each tile makes one call into the
# HEALPix interpolation logic.  Capping at _MAX_TILES ensures Sc is always at
# least S/_MAX_TILES, keeping per-tile overhead bounded while still preventing
# out-of-memory.
_MAX_TILES = 8


def beam_tod_batch(
    nside,
    mp,
    data,
    rot_vecs,
    phi_b,
    theta_b,
    psis_b,
    n_target=None,
    interp_mode="bilinear",
):
    """Accumulate the TOD contribution of one beam entry for a batch of samples.

    Tiles over the ``S`` selected beam pixels so that the
    ``(B × Sc × 3 × float32)`` intermediate vector buffer stays within the L2
    cache target. Uses Numba JIT kernels for both the rotation and the
    gather + accumulation steps.

    The execution path is selected automatically based on ``data``:

    * **Full double-Rodrigues** — used when no cache arrays are
      present. Applies both Rodrigues rotations per ``(B, S)`` element at
      runtime.

    Args:
        nside (int): HEALPix ``nside`` of the sky map.
        mp (list[numpy.ndarray]): Sky map components ``[I, Q, U]``. Each
            element is a 1-D ``float32`` array of length ``12 * nside**2``.
            Used only on the full double-Rodrigues fallback path.
        data (dict): Beam data entry as returned by :func:`prepare_beam_data`.
            Required keys: ``'vec_orig'``, ``'beam_vals'``, ``'comp_indices'``.
            Optional keys for cached paths: ``'mp_stacked'``, ``'vec_rolled'``,
            ``'psi_grid'``, ``'dtheta'``, ``'dphi'``.
        rot_vecs (numpy.ndarray): Rodrigues rotation vectors from
            :func:`precompute_rotation_vector_batch`, shape ``(B, 3)``.
        phi_b (numpy.ndarray): Boresight longitude [rad], shape ``(B,)``.
        theta_b (numpy.ndarray): Boresight colatitude [rad], shape ``(B,)``.
        psis_b (numpy.ndarray): Combined rotation angle ``psi_b - beta`` [rad],
            shape ``(B,)``.
        n_target (numpy.ndarray | None): Unused; kept for API compatibility.
        interp_mode (str): Sky-map interpolation strategy. One of:

            * ``'bilinear'`` *(default)* — 4-pixel bilinear HEALPix
              interpolation via the dedup Numba kernel.
            * ``'nearest'`` — single nearest-pixel lookup; fastest, no pixel
              mixing.
            (``'gaussian'`` and ``'bicubic'`` are available on their respective branches.)

    Returns:
        dict[int, numpy.ndarray]: Mapping from Stokes component index to a
            ``(B,)`` ``float32`` array containing the beam-weighted sky-map
            accumulation for that component over the batch.
    """
    B = phi_b.shape[0]
    vec_orig = data["vec_orig"]  # (S, 3)
    beam_vals = data["beam_vals"]  # (S,)
    S = vec_orig.shape[0]
    comp_indices = data["comp_indices"]
    C = len(comp_indices)
    mp_stacked = data.get("mp_stacked")  # (C, N) float32, or None

    # Q and U channel positions within the C-dim of mp_stacked
    c_q = -1
    c_u = -1
    for _ci, _comp in enumerate(comp_indices):
        if _comp == 1:
            c_q = _ci
        elif _comp == 2:
            c_u = _ci

    use_nearest = interp_mode == "nearest"
    if interp_mode not in ("nearest", "bilinear"):
        raise ValueError(
            f"interp_mode {interp_mode!r} not available on main branch; "
            "switch to the 'gaussian' or 'bicubic' branch"
        )

    # Lower bound from L2 target; upper bound from _MAX_TILES cap.
    # The max() ensures we never produce more than _MAX_TILES tiles even when
    # the memory-based Sc is tiny (e.g. Sc=79 at B=2212 → 64 tiles → 64 C calls).
    Sc = max(1, _S_TILE_TARGET_BYTES // (B * 3 * 4))  # memory target
    Sc = max(Sc, -(-S // _MAX_TILES))  # tile-count cap (ceiling div)
    Sc = min(Sc, S)

    axes, cos_a, sin_a, ax_pts, cos_p, sin_p = _rotation_params(
        rot_vecs, phi_b, theta_b, psis_b
    )

    tod = {comp: np.zeros(B, dtype=np.float32) for comp in comp_indices}

    for s0 in range(0, S, Sc):
        s1 = min(s0 + Sc, S)
        bv_chunk = beam_vals[s0:s1]  # (Sc,)

        vec_chunk = np.asarray(vec_orig[s0:s1], dtype=np.float32)  # (Sc, 3)
        vec_rot = np.empty((B, s1 - s0, 3), dtype=np.float32)
        _rodrigues_jit(vec_chunk, axes, cos_a, sin_a, ax_pts, cos_p, sin_p, vec_rot)

        if mp_stacked is not None:
            tod_arr = np.zeros((C, B), dtype=np.float64)
            if use_nearest:
                _gather_accum_nearest_jit(
                    vec_rot,
                    nside,
                    mp_stacked,
                    bv_chunk,
                    B,
                    s1 - s0,
                    tod_arr,
                    ax_pts,
                    c_q,
                    c_u,
                )
            else:
                _gather_accum_dedup_jit(
                    vec_rot,
                    nside,
                    mp_stacked,
                    bv_chunk,
                    B,
                    s1 - s0,
                    tod_arr,
                    ax_pts,
                    c_q,
                    c_u,
                )
            for i, comp in enumerate(comp_indices):
                tod[comp] += tod_arr[i].astype(np.float32)
        else:
            theta_flat, phi_flat = hp.vec2ang(vec_rot.reshape(-1, 3).astype(np.float64))
            pixels, weights = get_interp_weights_numba(nside, theta_flat, phi_flat)
            mp_gathered = np.stack([mp[c][pixels] for c in comp_indices])
            mp_flat = np.einsum("ckn,kn->cn", mp_gathered, weights)
            tod_chunk = mp_flat.reshape(C, B, s1 - s0) @ bv_chunk
            for i, comp in enumerate(comp_indices):
                tod[comp] += tod_chunk[i].astype(np.float32)

    return tod
